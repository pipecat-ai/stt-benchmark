"""Tests for the Oruk Fourier-2B STT service registration and HTTP contract."""

import asyncio
import email
import time

import httpx
import pytest
from openai import AsyncOpenAI
from pipecat.audio.utils import pcm_to_wav
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.workers.runner import WorkerRunner

from stt_benchmark.config import reset_config
from stt_benchmark.evals.manifest import write_suite
from stt_benchmark.models import AudioSample, ServiceName
from stt_benchmark.observers.metrics_collector import MetricsCollectorObserver
from stt_benchmark.observers.transcription_collector import TranscriptionCollectorObserver
from stt_benchmark.reporting.readme_table import (
    README_TABLE_END,
    README_TABLE_START,
    parse_table_rows,
    upsert_readme_rows,
)
from stt_benchmark.services import (
    STT_SERVICES,
    create_oruk,
    create_stt_service,
    get_available_services,
    get_display_names,
    get_service_definition,
    is_service_available,
    parse_service_name,
    parse_services_arg,
)

ORUK_TRANSCRIPTION_RESPONSE = {
    "id": "tr_test123456",
    "object": "audio.transcription",
    "model": "oruk-fourier-2b",
    "task": "transcription",
    "text": "  I am genuinely excited that this is finally working.  ",
    "language": "en",
    "duration": 1.5,
    "audio_seconds": 1.5,
    "segments": [],
    "usage": {
        "audio_seconds": 1.5,
        "billable_seconds": 1.5,
        "billing_unit": "audio_second",
        "billing_basis": "subscription",
        "rate_per_minute_usd": "0.0080",
        "estimated_cost_usd": "0.000200",
        "cost_microusd": 200,
        "pricing_version": "2026-07-11",
    },
    "request_id": "req_test123456",
    "model_revision": "8c2a8871f1304aa2f634da4fb95e24438c4945b9",
}


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Isolate environment variables and .env lookup for each test."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ORUK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_config()
    yield
    reset_config()


def _parse_multipart(request: httpx.Request) -> list[tuple[str, str | None, str, bytes]]:
    """Parse a multipart/form-data request into (name, filename, content_type, payload) tuples."""
    content_type = request.headers["content-type"]
    raw = b"Content-Type: " + content_type.encode("ascii") + b"\r\n\r\n" + request.content
    msg = email.message_from_bytes(raw)
    assert msg.is_multipart()
    parts = []
    for part in msg.get_payload():
        name = part.get_param("name", header="content-disposition")
        parts.append(
            (
                str(name),
                part.get_filename(),
                part.get_content_type(),
                part.get_payload(decode=True) or b"",
            )
        )
    return parts


def _attach_mock_transport(service: OpenAISTTService, handler) -> httpx.AsyncClient:
    """Replace the underlying OpenAI client's HTTP transport while keeping base_url and api_key."""
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service._client = AsyncOpenAI(
        api_key=service._client.api_key,
        base_url=str(service._client.base_url),
        http_client=http_client,
        max_retries=0,
    )
    return http_client


def test_oruk_registry_and_cli_parsing():
    assert ServiceName.ORUK == "oruk"
    assert "oruk" in STT_SERVICES

    defn = get_service_definition("oruk")
    assert defn.factory is create_oruk
    assert defn.vendor == "Oruk"
    assert defn.model_label == "oruk-fourier-2b"
    assert defn.required_env_vars == ["ORUK_API_KEY"]
    assert defn.needs_aiohttp is False
    assert defn.is_current is True

    assert parse_service_name("oruk") is ServiceName.ORUK
    assert parse_service_name(" ORUK ") is ServiceName.ORUK
    assert parse_services_arg("oruk") == [ServiceName.ORUK]
    assert get_display_names([ServiceName.ORUK]) == {"oruk": "Oruk"}


def test_oruk_availability_from_env_and_dotenv(monkeypatch, tmp_path):
    assert is_service_available("oruk") is False
    assert ServiceName.ORUK not in get_available_services()

    monkeypatch.setenv("ORUK_API_KEY", "test-env-api-key")
    reset_config()
    assert is_service_available("oruk") is True
    assert ServiceName.ORUK in get_available_services()

    # Verify .env loading via BenchmarkConfig.oruk_api_key
    monkeypatch.delenv("ORUK_API_KEY")
    (tmp_path / ".env").write_text("ORUK_API_KEY=test-dotenv-api-key\n", encoding="utf-8")
    reset_config()
    assert is_service_available("oruk") is True


def test_create_oruk_requires_oruk_api_key_even_when_openai_key_is_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-unused-openai-key")
    reset_config()

    with pytest.raises(ValueError, match="ORUK_API_KEY"):
        create_stt_service(ServiceName.ORUK)


def test_create_oruk_configures_openai_segmented_stt(monkeypatch):
    monkeypatch.setenv("ORUK_API_KEY", "test-oruk-api-key")
    reset_config()

    service = create_stt_service(ServiceName.ORUK)
    assert isinstance(service, OpenAISTTService)
    assert isinstance(service, SegmentedSTTService)
    assert str(service._client.base_url).rstrip("/") == "https://speech-api.oruk.ai/v1"
    assert service._client.api_key == "test-oruk-api-key"
    assert service._settings.model == "oruk-fourier-2b"
    assert service._settings.language == "en"
    assert service._settings.prompt is None
    assert service._settings.temperature is None
    assert service._settings.keywords is None


async def test_oruk_run_stt_wire_contract_and_transcription_frame(monkeypatch):
    monkeypatch.setenv("ORUK_API_KEY", "test-oruk-api-key")
    reset_config()

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["parts"] = _parse_multipart(request)
        return httpx.Response(200, json=ORUK_TRANSCRIPTION_RESPONSE)

    service = create_oruk()
    assert isinstance(service, OpenAISTTService)
    http_client = _attach_mock_transport(service, handler)
    try:
        wav = pcm_to_wav(b"\x00\x00" * 1600, 16000)
        frames = [frame async for frame in service.run_stt(wav)]
    finally:
        await http_client.aclose()

    assert captured["method"] == "POST"
    assert captured["url"] == "https://speech-api.oruk.ai/v1/audio/transcriptions"
    assert captured["authorization"] == "Bearer test-oruk-api-key"

    # Strict contract check: only file, model, and language (additionalProperties: false).
    part_names = [name for name, _, _, _ in captured["parts"]]
    assert part_names == ["model", "language", "file"]
    parts_by_name = {
        name: (filename, ctype, data) for name, filename, ctype, data in captured["parts"]
    }
    assert parts_by_name["model"][2] == b"oruk-fourier-2b"
    assert parts_by_name["language"][2] == b"en"
    assert parts_by_name["file"][0] == "audio.wav"
    assert parts_by_name["file"][1] == "audio/wav"
    assert parts_by_name["file"][2].startswith(b"RIFF")

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "I am genuinely excited that this is finally working."


async def test_oruk_run_stt_yields_error_frame_on_http_failure(monkeypatch):
    monkeypatch.setenv("ORUK_API_KEY", "test-invalid-api-key")
    reset_config()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                    "message": "Invalid or revoked API key.",
                }
            },
        )

    service = create_oruk()
    assert isinstance(service, OpenAISTTService)
    http_client = _attach_mock_transport(service, handler)
    try:
        wav = pcm_to_wav(b"\x00\x00" * 1600, 16000)
        frames = [frame async for frame in service.run_stt(wav)]
    finally:
        await http_client.aclose()

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert "401" in frames[0].error


class _NullSink(FrameProcessor):
    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


async def test_oruk_segmented_pipeline_emits_transcript_and_ttfb(monkeypatch):
    """Verify Oruk works end-to-end as a SegmentedSTTService driven by VAD frames."""
    monkeypatch.setenv("ORUK_API_KEY", "test-oruk-api-key")
    reset_config()

    requests_seen = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests_seen
        requests_seen += 1
        return httpx.Response(200, json=ORUK_TRANSCRIPTION_RESPONSE)

    service = create_oruk()
    assert isinstance(service, OpenAISTTService)
    http_client = _attach_mock_transport(service, handler)

    metrics = MetricsCollectorObserver()
    transcripts = TranscriptionCollectorObserver()
    metrics.set_current_sample("sample-oruk")
    transcripts.set_current_sample("sample-oruk")

    worker = PipelineWorker(
        Pipeline([service, _NullSink()]),
        params=PipelineParams(audio_in_sample_rate=16000, audio_in_channels=1, enable_metrics=True),
        observers=[metrics, transcripts],
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        await asyncio.sleep(0.05)
        await worker.queue_frame(VADUserStartedSpeakingFrame())
        # 200ms of 16 kHz 16-bit mono PCM (padded with 0.5s trailing silence by SegmentedSTTService)
        await worker.queue_frame(
            InputAudioRawFrame(audio=b"\x01\x00" * 3200, sample_rate=16000, num_channels=1)
        )
        await worker.queue_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=time.time()))
        ttfb = await metrics.wait_for_ttfb(timeout=5.0)
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(running, timeout=5.0)
    finally:
        if not running.done():
            await runner.cancel()
            await asyncio.gather(running, return_exceptions=True)
        await http_client.aclose()

    assert requests_seen == 1
    assert (
        transcripts.get_transcription_for_sample("sample-oruk")
        == "I am genuinely excited that this is finally working."
    )
    assert ttfb is not None
    assert ttfb > 0.0


def test_oruk_manifest_and_readme_upsert(tmp_path):
    pcm_path = tmp_path / "sample.pcm"
    pcm_path.write_bytes(b"\x00\x00" * 16000)
    sample = AudioSample(
        sample_id="sample-0",
        audio_path=str(pcm_path),
        duration_seconds=1.0,
        dataset_index=0,
    )

    manifest_path = write_suite(tmp_path / "run", {ServiceName.ORUK: [sample]}, concurrency=1)
    assert manifest_path.exists()
    assert "service: oruk" in manifest_path.read_text(encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text(
        f"{README_TABLE_START}\n"
        "| Vendor | Model | Transcripts | Perfect | WER Mean | Pooled WER "
        "| TTFS Median | TTFS P95 | TTFS P99 |\n"
        "|--------|-------|-------------|---------|----------|------------"
        "|-------------|----------|----------|\n"
        "| OpenAI | gpt-4o-transcribe | 99.3% | 75.7% | 3.10% | 2.95% | 637ms | 965ms | 1655ms |\n"
        "| Rime | N/A | 100.0% | 82.5% | 1.25% | 1.21% | 323ms | 381ms | 415ms |\n"
        f"{README_TABLE_END}\n",
        encoding="utf-8",
    )
    metrics = {
        "success_rate": 100.0,
        "perfect_pct": 85.0,
        "wer_mean": 1.20,
        "pooled_wer": 1.15,
        "ttfb_median": 350.0,
        "ttfb_p95": 420.0,
        "ttfb_p99": 490.0,
    }
    written = upsert_readme_rows(readme, [(get_service_definition("oruk"), metrics)])
    assert written == ["Oruk — oruk-fourier-2b"]
    rows = parse_table_rows(readme.read_text(encoding="utf-8"))
    assert [(r["vendor"], r["model"]) for r in rows] == [
        ("OpenAI", "gpt-4o-transcribe"),
        ("Oruk", "oruk-fourier-2b"),
        ("Rime", "N/A"),
    ]
