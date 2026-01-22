"""HuggingFace dataset download and preparation for STT benchmarking."""

import hashlib
import uuid
from collections.abc import Callable

from datasets import load_dataset
from loguru import logger

from stt_benchmark.config import BenchmarkConfig, get_config
from stt_benchmark.models import AudioSample
from stt_benchmark.storage.database import Database


def generate_sample_id(dataset_index: int, audio_id: str) -> str:
    """Generate a deterministic sample ID."""
    hash_input = f"{dataset_index}:{audio_id}".encode()
    hash_bytes = hashlib.md5(hash_input).digest()
    return str(uuid.UUID(bytes=hash_bytes))


def audio_array_to_pcm(audio_array: list, sample_rate: int) -> bytes:
    """Convert audio array to 16-bit PCM bytes at 16kHz.

    Args:
        audio_array: Audio samples as float array (-1 to 1 range)
        sample_rate: Original sample rate of the audio

    Returns:
        16-bit PCM audio bytes at 16kHz
    """
    import numpy as np
    from scipy import signal

    # Convert to numpy array
    audio = np.array(audio_array, dtype=np.float32)

    # Resample to 16kHz if necessary
    target_rate = 16000
    if sample_rate != target_rate:
        num_samples = int(len(audio) * target_rate / sample_rate)
        audio = signal.resample(audio, num_samples)

    # Normalize and convert to 16-bit PCM
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    return audio_int16.tobytes()


class DatasetDownloader:
    """Downloads and processes audio samples for STT benchmarking."""

    def __init__(
        self,
        num_samples: int = 100,
        seed: int = 42,
        offset: int = 0,
        config: BenchmarkConfig | None = None,
    ):
        self.config = config or get_config()
        self.num_samples = num_samples
        self.seed = seed
        self.offset = offset
        self.db = Database()

    async def download_and_prepare(
        self,
        progress_callback: Callable | None = None,
    ) -> list[AudioSample]:
        """Download dataset, filter, and prepare audio files.

        Args:
            progress_callback: Optional callback(current, total, message) for progress

        Returns:
            List of AudioSample objects
        """
        self.config.ensure_dirs()
        await self.db.initialize()

        # Check if we already have samples
        existing_count = await self.db.get_sample_count()
        target_total = self.offset + self.num_samples
        if self.offset == 0 and existing_count >= self.num_samples:
            logger.info(f"Already have {existing_count} samples in database")
            return await self.db.get_all_samples()

        logger.info(
            f"Existing samples: {existing_count}, downloading indices {self.offset}-{target_total - 1}"
        )

        if progress_callback:
            progress_callback(0, 4, "Loading dataset from HuggingFace...")

        # Load dataset with streaming
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        dataset = load_dataset(
            self.config.dataset_name,
            split="train",
            streaming=True,
        )

        if progress_callback:
            progress_callback(1, 4, "Filtering for English, non-synthetic samples...")

        # Filter for English, non-synthetic samples
        filtered = dataset.filter(lambda x: x["language"] == "eng" and x["synthetic"] is False)

        if progress_callback:
            progress_callback(2, 4, "Shuffling and selecting samples...")

        # Shuffle with seed for reproducibility
        shuffled = filtered.shuffle(seed=self.seed)

        # Calculate the end index
        end_index = self.offset + self.num_samples

        if progress_callback:
            progress_callback(
                3,
                4,
                f"Processing {self.num_samples} samples (indices {self.offset}-{end_index - 1})...",
            )

        # Process samples
        samples = []
        processed_count = 0
        for i, item in enumerate(shuffled):
            # Skip items before offset
            if i < self.offset:
                continue

            if i >= end_index:
                break

            if progress_callback and processed_count % 10 == 0:
                progress_callback(
                    3,
                    4,
                    f"Processing sample {processed_count + 1}/{self.num_samples} (index {i})...",
                )

            sample = await self._process_sample(item, i)
            if sample:
                samples.append(sample)
                processed_count += 1

            # Log progress periodically
            if processed_count % 25 == 0:
                logger.info(
                    f"Processed {processed_count}/{self.num_samples} samples (at index {i})"
                )

        # Batch insert samples
        await self.db.insert_samples_batch(samples)
        logger.info(f"Saved {len(samples)} samples to database")

        if progress_callback:
            progress_callback(4, 4, f"Complete! {len(samples)} samples prepared.")

        return samples

    async def _process_sample(self, item: dict, dataset_index: int) -> AudioSample | None:
        """Process a single dataset item.

        Args:
            item: Dataset item with audio and metadata
            dataset_index: Index in the filtered/shuffled dataset

        Returns:
            AudioSample if successful, None if failed
        """
        try:
            # Extract audio data
            audio_data = item["audio"]
            audio_array = audio_data["array"]
            sample_rate = audio_data["sampling_rate"]

            # Generate sample ID
            sample_id = generate_sample_id(dataset_index, item.get("id", str(dataset_index)))

            # Convert to 16kHz 16-bit PCM
            pcm_bytes = audio_array_to_pcm(audio_array, sample_rate)

            # Calculate duration
            duration_seconds = len(pcm_bytes) / 2 / 16000  # 16-bit = 2 bytes, 16kHz

            # Save to file
            audio_path = self.config.audio_dir / f"{sample_id}.pcm"
            audio_path.write_bytes(pcm_bytes)

            return AudioSample(
                sample_id=sample_id,
                audio_path=str(audio_path),
                duration_seconds=duration_seconds,
                language=item.get("language", "eng"),
                dataset_index=dataset_index,
            )

        except Exception as e:
            logger.error(f"Error processing sample {dataset_index}: {e}")
            return None


async def download_dataset(
    num_samples: int = 100,
    seed: int = 42,
    offset: int = 0,
    progress_callback: Callable | None = None,
) -> list[AudioSample]:
    """Convenience function to download and prepare the dataset.

    Args:
        num_samples: Number of samples to download
        seed: Random seed for reproducibility
        offset: Number of samples to skip (for incremental downloads)
        progress_callback: Optional progress callback

    Returns:
        List of AudioSample objects
    """
    downloader = DatasetDownloader(num_samples=num_samples, seed=seed, offset=offset)
    return await downloader.download_and_prepare(progress_callback=progress_callback)
