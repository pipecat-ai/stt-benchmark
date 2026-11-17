"""The suite a benchmark run writes is one the eval harness can load and run."""

import numpy as np
import pytest
import soundfile as sf
from pipecat.evals.scenario import EvalScenarioFile
from pipecat.evals.suite import EvalManifest

from stt_benchmark.evals.manifest import BOT_PATH, ensure_wav, sample_budget_ms, write_suite
from stt_benchmark.models import AudioSample, ServiceName


@pytest.fixture
def samples(tmp_path):
    out = []
    for i, seconds in enumerate((1.0, 2.5)):
        pcm = (np.sin(np.arange(int(16000 * seconds)) / 20) * 8000).astype(np.int16)
        path = tmp_path / "audio" / f"sample-{i}.pcm"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(pcm.tobytes())
        out.append(
            AudioSample(
                sample_id=f"sample-{i}",
                audio_path=str(path),
                duration_seconds=seconds,
                dataset_index=i,
            )
        )
    return out


def test_the_wav_holds_the_samples_audio_unchanged(samples):
    wav = ensure_wav(samples[0])
    audio, rate = sf.read(wav, dtype="int16")
    assert rate == 16000
    assert audio.tobytes() == open(samples[0].audio_path, "rb").read()
    # Written once: a second call leaves the file alone.
    written = wav.stat().st_mtime_ns
    assert ensure_wav(samples[0]).stat().st_mtime_ns == written


def test_the_manifest_runs_every_sample_of_every_service(tmp_path, samples):
    run_dir = tmp_path / "run"
    manifest_path = write_suite(
        run_dir,
        {
            ServiceName.DEEPGRAM: samples,
            ServiceName.SONIOX: samples[:1],
            ServiceName.CARTESIA: [],
        },
        concurrency=8,
        vad_stop_secs=0.3,
        service_concurrency={ServiceName.SONIOX: 2},
    )

    manifest = EvalManifest.load(manifest_path)

    assert manifest.concurrency == 8
    assert [(r.label, r.scenario) for r in manifest.runs] == [
        ("deepgram", "deepgram/sample-0"),
        ("deepgram", "deepgram/sample-1"),
        ("soniox", "soniox/sample-0"),
    ]
    deepgram, _, soniox = manifest.runs
    assert deepgram.bot_path == BOT_PATH
    assert deepgram.runner_body == {"service": "deepgram", "vad_stop_secs": 0.3}
    assert (deepgram.concurrency, soniox.concurrency) == (None, 2)


def test_a_sample_is_one_turn_that_waits_for_the_bots_result(tmp_path, samples):
    write_suite(tmp_path / "run", {ServiceName.DEEPGRAM: samples}, concurrency=1)

    scenario = EvalScenarioFile.load(tmp_path / "run" / "scenarios" / "deepgram.yaml")[
        "deepgram/sample-1"
    ]

    assert scenario.user_audio
    (turn,) = scenario.turns
    assert turn.audio == str(ensure_wav(samples[1]))
    (expectation,) = turn.expect
    assert expectation.event == "server_message"
    assert expectation.text_contains == "stt-benchmark-result"
    assert expectation.within_ms == sample_budget_ms(samples[1])
    # The audio, the bot's 10 s + 10 s limits, the TTFB wait and the margin.
    assert expectation.within_ms == int((2.5 + 10 + 10 + 2.5 + 10) * 1000)
