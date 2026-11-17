"""Suite records read back as the benchmark results the rest of the tool stores."""

import json

from stt_benchmark.evals.results import ResultsReader, benchmark_result
from stt_benchmark.models import AudioSample, ServiceName

SAMPLES = {
    "sample-0": AudioSample(
        sample_id="sample-0", audio_path="a.pcm", duration_seconds=3.5, dataset_index=0
    )
}


def _record(matched: str = "", **fields) -> dict:
    record = {
        "name": "deepgram",
        "scenario": "deepgram/sample-0",
        "passed": True,
        "error": None,
        "skipped": None,
        "failures": [],
        "turns": [
            {
                "turn_index": 0,
                "status": "passed",
                "expectations": [
                    {
                        "expectation_index": 0,
                        "event_name": "server_message",
                        "passed": True,
                        "matched": matched,
                    }
                ],
            }
        ],
    }
    return {**record, **fields}


def _message(**fields) -> str:
    message = {"type": "stt-benchmark-result", "transcription": "hello there", "ttfb_seconds": 0.31}
    return json.dumps({**message, "capped": False, **fields})


def test_a_passed_record_gives_the_bots_transcription_and_ttfs():
    result = benchmark_result(_record(_message()), SAMPLES, model="nova-3")

    assert result.sample_id == "sample-0"
    assert result.service_name == ServiceName.DEEPGRAM
    assert result.model_name == "nova-3"
    assert result.transcription == "hello there"
    assert result.ttfb_seconds == 0.31
    assert result.audio_duration_seconds == 3.5
    assert result.error is None


def test_a_sample_the_stt_returned_nothing_for_is_a_result_without_an_error():
    result = benchmark_result(_record(_message(transcription=None, ttfb_seconds=None)), SAMPLES)

    assert (result.transcription, result.ttfb_seconds, result.error) == (None, None, None)


def test_a_failed_record_gives_the_harnesss_failure_as_the_error():
    record = _record(
        passed=False,
        failures=[
            {
                "turn_index": 0,
                "expectation_index": 0,
                "event_name": "server_message",
                "kind": "timeout",
                "reason": "no matching 'server_message' event arrived within 36000ms",
            }
        ],
    )

    result = benchmark_result(record, SAMPLES)

    assert result.transcription is None
    assert result.error == "timeout: no matching 'server_message' event arrived within 36000ms"


def test_a_run_that_never_started_gives_the_suites_error():
    result = benchmark_result(
        _record(passed=False, error="bot exited with code 1", turns=[]), SAMPLES
    )

    assert result.error == "bot exited with code 1"


def test_a_passed_record_without_the_bots_message_is_an_error():
    result = benchmark_result(_record("some other message"), SAMPLES)

    assert result.error == "the bot reported no result"


def test_the_reader_returns_each_record_once_as_the_file_grows(tmp_path):
    path = tmp_path / "results.jsonl"
    reader = ResultsReader(path)
    assert list(reader.read_new()) == []

    path.write_text('{"n": 1}\n{"n": 2}\n')
    assert list(reader.read_new()) == [{"n": 1}, {"n": 2}]
    assert list(reader.read_new()) == []

    # A line the suite is still writing waits for its newline.
    with path.open("a") as f:
        f.write('{"n": 3}\n{"n":')
    assert list(reader.read_new()) == [{"n": 3}]
    with path.open("a") as f:
        f.write(" 4}\n")
    assert list(reader.read_new()) == [{"n": 4}]
