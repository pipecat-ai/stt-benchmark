"""Writing the eval suite a benchmark run executes.

A run is one manifest: an entry per STT service, each playing its pending
samples to the benchmark bot. A sample is one scenario with a single turn, so
the suite spawns a fresh bot, and the bot opens a fresh connection to the STT
service, for every sample.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from stt_benchmark.config import get_config
from stt_benchmark.evals.collector import RESULT_MESSAGE_TYPE
from stt_benchmark.models import AudioSample, ServiceName

BOT_PATH = Path(__file__).with_name("bot.py")

# What the turn's budget allows on top of the bot's own limits: the wait for a
# late TTFB, and slack for the VAD's stop and the message's way back.
TTFB_WAIT_SECS = 2.5
BUDGET_MARGIN_SECS = 10.0


def ensure_wav(sample: AudioSample) -> Path:
    """The sample's audio as a WAV file next to its PCM, written on first use.

    The harness plays recordings libsndfile can read; the dataset is stored as
    headerless PCM.

    Args:
        sample: The sample to play.

    Returns:
        The WAV file's absolute path.
    """
    pcm_path = Path(sample.audio_path).resolve()
    wav_path = pcm_path.with_suffix(".wav")
    if not wav_path.exists() or wav_path.stat().st_mtime < pcm_path.stat().st_mtime:
        audio = np.fromfile(pcm_path, dtype=np.int16)
        sf.write(wav_path, audio, get_config().sample_rate, subtype="PCM_16")
    return wav_path


def sample_budget_ms(sample: AudioSample) -> int:
    """How long the harness waits for the bot's result: the audio, then the bot's limits, then slack."""
    config = get_config()
    seconds = (
        sample.duration_seconds
        + config.max_silence_timeout_secs
        + config.transcription_timeout_secs
        + TTFB_WAIT_SECS
        + BUDGET_MARGIN_SECS
    )
    return int(seconds * 1000)


def _scenario(sample: AudioSample) -> dict:
    return {
        "name": sample.sample_id,
        "turns": [
            {
                # The harness wants the words of a recording for its judge; nothing
                # here is judged, so the turn only names what it plays.
                "user": f"(sample {sample.sample_id})",
                "audio": str(ensure_wav(sample)),
                "expect": [
                    {
                        "event": "server_message",
                        "text_contains": RESULT_MESSAGE_TYPE,
                        "within_ms": sample_budget_ms(sample),
                    }
                ],
            }
        ],
    }


def write_suite(
    run_dir: Path,
    samples_by_service: dict[ServiceName, list[AudioSample]],
    *,
    concurrency: int,
    vad_stop_secs: float | None = None,
    service_concurrency: dict[ServiceName, int] | None = None,
) -> Path:
    """Write a run's manifest and scenario files.

    Args:
        run_dir: The run's directory; the manifest, the scenarios and the
            suite's output all go under it.
        samples_by_service: The samples each service is to transcribe. A
            service with none is left out.
        concurrency: How many samples run at once across all services.
        vad_stop_secs: The VAD's stop silence, or None for the configured default.
        service_concurrency: A cap on how many of one service's samples run at
            once, for a provider that limits concurrent streams.

    Returns:
        The manifest's path.
    """
    scenarios_dir = run_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    stop_secs = get_config().vad_stop_secs if vad_stop_secs is None else vad_stop_secs

    entries = []
    for service, samples in samples_by_service.items():
        if not samples:
            continue
        scenario_file = {
            "name": service.value,
            "user": {"modality": "audio"},
            "scenarios": [_scenario(sample) for sample in samples],
        }
        (scenarios_dir / f"{service.value}.yaml").write_text(
            yaml.safe_dump(scenario_file, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        entry: dict = {
            "bot": BOT_PATH.name,
            "name": service.value,
            "runner_body": {"data": {"service": service.value, "vad_stop_secs": stop_secs}},
            "scenarios": [service.value],
        }
        cap = (service_concurrency or {}).get(service)
        if cap is not None:
            entry["concurrency"] = cap
        entries.append(entry)

    manifest = {
        "bots_dir": str(BOT_PATH.parent),
        "scenarios_dir": "scenarios",
        "concurrency": concurrency,
        "record": False,
        "spawn": "{python} {bot} --port {port}",
        "suite": entries,
    }
    manifest_path = run_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path
