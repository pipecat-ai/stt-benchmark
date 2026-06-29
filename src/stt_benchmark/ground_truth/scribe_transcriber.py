"""ElevenLabs Scribe v2 transcription for ground truth generation.

Uses the ElevenLabs Speech-to-Text HTTP API (batch mode) to transcribe audio samples.
"""

import asyncio
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from loguru import logger

from stt_benchmark.config import BenchmarkConfig, get_config
from stt_benchmark.ground_truth.gemini_transcriber import pcm_to_wav
from stt_benchmark.models import AudioSample, GroundTruth
from stt_benchmark.storage.database import Database

SCRIBE_API_URL = "https://api.elevenlabs.io/v1/speech-to-text"


class ScribeTranscriber:
    """Generates ground truth transcriptions using ElevenLabs Scribe v2."""

    def __init__(
        self,
        model_name: str = "scribe_v2",
        config: BenchmarkConfig | None = None,
    ):
        self.config = config or get_config()
        self.model_name = model_name
        self.db = Database()

        if not self.config.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY not set in environment")

        self.api_key = self.config.elevenlabs_api_key

        # Rate limiting
        self.requests_per_minute = 60
        self.request_times: list[float] = []

    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]

        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)

        self.request_times.append(time.time())

    async def _transcribe(self, wav_bytes: bytes, session: aiohttp.ClientSession) -> str | None:
        """Send audio to Scribe API and return transcription text."""
        data = aiohttp.FormData()
        data.add_field(
            "file",
            wav_bytes,
            filename="audio.wav",
            content_type="audio/wav",
        )
        data.add_field("model_id", self.model_name)
        data.add_field("language_code", "eng")
        data.add_field("tag_audio_events", "false")
        data.add_field("timestamps_granularity", "none")

        headers = {"xi-api-key": self.api_key}

        async with session.post(SCRIBE_API_URL, data=data, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"status={response.status} {error_text}")

            result = await response.json()
            return result.get("text", "").strip() or None

    async def transcribe_sample(
        self, sample: AudioSample, session: aiohttp.ClientSession | None = None
    ) -> GroundTruth | None:
        """Transcribe a single audio sample.

        Args:
            sample: AudioSample to transcribe
            session: Optional aiohttp session to reuse

        Returns:
            GroundTruth if successful, None if failed
        """
        await self._rate_limit()

        try:
            audio_path = Path(sample.audio_path)
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_path}")
                return None

            pcm_bytes = audio_path.read_bytes()
            wav_bytes = pcm_to_wav(pcm_bytes, sample_rate=16000, channels=1)

            if session:
                transcription = await self._transcribe(wav_bytes, session)
            else:
                async with aiohttp.ClientSession() as new_session:
                    transcription = await self._transcribe(wav_bytes, new_session)

            if not transcription:
                logger.warning(f"Empty response for sample {sample.sample_id}")
                return None

            return GroundTruth(
                sample_id=sample.sample_id,
                text=transcription,
                model_used=self.model_name,
                generated_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Error transcribing sample {sample.sample_id}: {e}")
            return None

    async def transcribe_batch(
        self,
        samples: list[AudioSample],
        progress_callback: Callable | None = None,
        save_incrementally: bool = True,
        force: bool = False,
    ) -> list[GroundTruth]:
        """Transcribe a batch of samples.

        Args:
            samples: List of AudioSample to transcribe
            progress_callback: Optional callback(current, total, sample_id)
            save_incrementally: Save each result to DB as it completes
            force: Re-transcribe even if ground truth already exists

        Returns:
            List of GroundTruth objects
        """
        await self.db.initialize()
        results = []

        async with aiohttp.ClientSession() as session:
            for i, sample in enumerate(samples):
                if progress_callback:
                    progress_callback(i, len(samples), sample.sample_id)

                if not force:
                    existing = await self.db.get_ground_truth(
                        sample.sample_id, model_used=self.model_name
                    )
                    if existing:
                        logger.debug(f"Sample {sample.sample_id} already transcribed, skipping")
                        results.append(existing)
                        continue

                gt = await self.transcribe_sample(sample, session=session)
                if gt:
                    results.append(gt)
                    if save_incrementally:
                        await self.db.insert_ground_truth(gt)
                    text_preview = gt.text[:50] + "..." if len(gt.text) > 50 else gt.text
                    logger.info(f"[{i+1}/{len(samples)}] Transcribed: {text_preview}")
                else:
                    logger.warning(
                        f"[{i+1}/{len(samples)}] Failed to transcribe sample {sample.sample_id}"
                    )

        return results

    async def generate_all_ground_truth(
        self,
        progress_callback: Callable | None = None,
    ) -> list[GroundTruth]:
        """Generate ground truth for all samples that don't have it yet.

        Args:
            progress_callback: Optional progress callback

        Returns:
            List of newly generated GroundTruth objects
        """
        await self.db.initialize()

        samples = await self.db.get_samples_without_ground_truth(model_used=self.model_name)
        if not samples:
            logger.info("All samples already have ground truth")
            return []

        logger.info(f"Generating ground truth for {len(samples)} samples")
        return await self.transcribe_batch(samples, progress_callback=progress_callback)
