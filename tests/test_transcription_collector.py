"""Tests for collecting segmented STT transcriptions."""

from pipecat.frames.frames import TranscriptionFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService

from stt_benchmark.models import AudioSample, BenchmarkResult, ServiceName
from stt_benchmark.observers.transcription_collector import TranscriptionCollectorObserver
from stt_benchmark.storage.database import Database


class StubSTTService(STTService):
    """Minimal STT service used as the source of observer events."""

    async def run_stt(self, audio):
        if False:
            yield audio


async def test_multiple_transcription_frames_are_aggregated_and_persisted(tmp_path):
    """All final segments for one input become one stored benchmark transcript."""
    sample_id = "segmented-transcription"
    observer = TranscriptionCollectorObserver()
    observer.set_current_sample(sample_id)
    stt_service = StubSTTService()

    for text in ("hello", "from", "pipecat"):
        await observer.on_push_frame(
            FramePushed(
                source=stt_service,
                destination=stt_service,
                frame=TranscriptionFrame(text=text, user_id="user", timestamp="timestamp"),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )

    transcription = observer.get_transcription_for_sample(sample_id)
    assert transcription == "hello from pipecat"

    # Use an isolated test database, matching the CLI's --test database name.
    db = Database(db_path=tmp_path / "test_results.db")
    await db.initialize()
    try:
        await db.insert_sample(
            AudioSample(
                sample_id=sample_id,
                audio_path="unused.pcm",
                duration_seconds=1.0,
                dataset_index=0,
            )
        )
        await db.insert_result(
            BenchmarkResult(
                sample_id=sample_id,
                service_name=ServiceName.DEEPGRAM,
                transcription=transcription,
                audio_duration_seconds=1.0,
            )
        )

        stored_results = await db.get_results_for_service(ServiceName.DEEPGRAM)
        assert len(stored_results) == 1
        assert stored_results[0].transcription == "hello from pipecat"
    finally:
        await db.close()
