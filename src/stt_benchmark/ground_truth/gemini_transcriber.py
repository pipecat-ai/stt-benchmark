"""Gemini transcription for ground truth generation.

Uses the google-genai library to transcribe audio samples.
"""

import asyncio
import io
import time
import wave
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types
from loguru import logger

from stt_benchmark.config import BenchmarkConfig, get_config
from stt_benchmark.models import AudioSample, GroundTruth
from stt_benchmark.storage.database import Database


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Convert raw PCM audio to WAV format."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    wav_buffer.seek(0)
    return wav_buffer.read()


# Transcription prompt for literal transcription
TRANSCRIPTION_PROMPT = """Transcribe the following audio EXACTLY and LITERALLY as spoken.

CRITICAL - ANTI-HALLUCINATION RULES:
1. Transcribe ONLY words that are FULLY and COMPLETELY audible from start to finish
2. Do NOT complete partial or cut-off words - if a word is truncated, omit it entirely
3. Do NOT add words based on context, even if they would "make sense" to complete a sentence
4. Audio may be truncated mid-word or mid-sentence - this is expected and normal
5. If audio ends abruptly, end your transcription at the last COMPLETE word
6. NEVER guess what word "should" come next based on the phrase or context

CRITICAL - PHONETIC ACCURACY RULES:
1. Transcribe the EXACT sounds you hear, not what would make semantic sense
2. Do NOT substitute similar-sounding words based on context (e.g., if you hear "doctor's" do NOT write "daughter's" just because it fits the context better)
3. If a word sounds unusual or doesn't fit the context, transcribe what you HEAR, not what you EXPECT
4. Trust your ears over your language understanding - write the phonemes, not the "logical" word

VERIFICATION before finalizing:
- Check that every word at the END was fully audible, not inferred from context
- Check that no words were substituted for "more sensible" alternatives
- The transcription may sound grammatically odd or contextually strange - that's OK

Other instructions:
- Include filler words like "um", "uh", "like" if spoken
- Use standard English spelling for the sounds you hear
- Include contractions as spoken (e.g., "don't" not "do not")
- If audio is unclear or inaudible, indicate with [inaudible]
- Do not include any commentary or explanation

Output only the transcription text, nothing else."""


class GeminiTranscriber:
    """Generates ground truth transcriptions using Gemini."""

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        config: BenchmarkConfig | None = None,
    ):
        self.config = config or get_config()
        self.model_name = model_name
        self.db = Database()

        # Configure Gemini client
        if not self.config.google_api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")

        self.client = genai.Client(api_key=self.config.google_api_key)

        # Rate limiting
        self.requests_per_minute = self.config.gemini_requests_per_minute
        self.request_times: list[float] = []

    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]

        if len(self.request_times) >= self.requests_per_minute:
            # Wait until oldest request is 1 minute old
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)

        self.request_times.append(time.time())

    async def transcribe_sample(self, sample: AudioSample) -> GroundTruth | None:
        """Transcribe a single audio sample.

        Args:
            sample: AudioSample to transcribe

        Returns:
            GroundTruth if successful, None if failed
        """
        await self._rate_limit()

        try:
            # Load audio file
            audio_path = Path(sample.audio_path)
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_path}")
                return None

            pcm_bytes = audio_path.read_bytes()

            # Convert PCM to WAV format for Gemini
            wav_bytes = pcm_to_wav(pcm_bytes, sample_rate=16000, channels=1)

            # Create audio part for Gemini with WAV format
            audio_part = types.Part.from_bytes(
                data=wav_bytes,
                mime_type="audio/wav",
            )

            # Generate transcription
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=[TRANSCRIPTION_PROMPT, audio_part],
                config=types.GenerateContentConfig(
                    temperature=0.0,  # Deterministic for consistency
                    max_output_tokens=1024,
                ),
            )

            if not response.text:
                logger.warning(f"Empty response for sample {sample.sample_id}")
                return None

            transcription = response.text.strip()

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

        for i, sample in enumerate(samples):
            if progress_callback:
                progress_callback(i, len(samples), sample.sample_id)

            # Check if already transcribed (unless force is True)
            if not force:
                existing = await self.db.get_ground_truth(sample.sample_id)
                if existing:
                    logger.debug(f"Sample {sample.sample_id} already transcribed, skipping")
                    results.append(existing)
                    continue

            # Transcribe
            gt = await self.transcribe_sample(sample)
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

        # Get samples without ground truth
        samples = await self.db.get_samples_without_ground_truth()
        if not samples:
            logger.info("All samples already have ground truth")
            return []

        logger.info(f"Generating ground truth for {len(samples)} samples")
        return await self.transcribe_batch(samples, progress_callback=progress_callback)
