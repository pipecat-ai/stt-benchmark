"""A suite run stores each sample's result as the suite reports it."""

import json

import numpy as np
import pytest

from stt_benchmark.evals import runner
from stt_benchmark.models import AudioSample, ServiceName
from stt_benchmark.storage.database import Database


class _FakeSuite:
    """Stands in for the eval suite: writes one record per run, as the real one does."""

    def __init__(self, manifest):
        self.manifest = manifest

    async def run(self, logs_dir, *, results_path):
        with results_path.open("a") as f:
            for i, run in enumerate(self.manifest.runs):
                message = {
                    "type": "stt-benchmark-result",
                    "transcription": f"transcript {i}",
                    "ttfb_seconds": 0.3 + i / 100,
                    "capped": False,
                }
                record = {
                    "name": run.label,
                    "scenario": run.scenario,
                    "passed": True,
                    "error": None,
                    "failures": [],
                    "turns": [{"expectations": [{"matched": json.dumps(message)}]}],
                }
                f.write(json.dumps(record) + "\n")


@pytest.fixture
def samples(tmp_path):
    out = []
    for i in range(2):
        path = tmp_path / f"sample-{i}.pcm"
        path.write_bytes(np.zeros(16000, dtype=np.int16).tobytes())
        out.append(
            AudioSample(
                sample_id=f"sample-{i}", audio_path=str(path), duration_seconds=1.0, dataset_index=i
            )
        )
    return out


async def test_every_result_is_stored_and_reported(tmp_path, samples, monkeypatch):
    monkeypatch.setattr(runner, "EvalSuite", _FakeSuite)
    db = Database(db_path=tmp_path / "results.db")
    await db.initialize()
    for sample in samples:
        await db.insert_sample(sample)
    seen = []

    results = await runner.run_suite(
        tmp_path / "run",
        {ServiceName.DEEPGRAM: samples},
        db,
        concurrency=2,
        model="nova-3",
        on_result=seen.append,
    )

    assert [r.sample_id for r in results] == ["sample-0", "sample-1"]
    assert seen == results
    stored = await db.get_results_for_service(ServiceName.DEEPGRAM, "nova-3")
    assert {r.sample_id: (r.transcription, r.ttfb_seconds) for r in stored} == {
        "sample-0": ("transcript 0", 0.3),
        "sample-1": ("transcript 1", 0.31),
    }
    assert await db.get_samples_without_results(ServiceName.DEEPGRAM, "nova-3") == []
    await db.close()
