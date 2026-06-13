"""Audio source resolution for debug runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydub import AudioSegment
from pydub.exceptions import PydubException
from rich.console import Console
from rich.table import Table

from stt_benchmark.config import get_config
from stt_benchmark.models import AudioSample
from stt_benchmark.storage.database import Database

Channel = Literal["left", "right"]
SUPPORTED_MEDIA_SUFFIXES = frozenset({".wav", ".mp3"})
BYTE_PER_SAMPLE = 2

console = Console()


async def open_database(test: bool) -> Database:
    config = get_config()
    db_path = config.data_dir / "test_results.db" if test else config.results_db
    db = Database(db_path=db_path)
    await db.initialize()
    if test:
        copied = await db.copy_samples_from(config.results_db)
        if copied > 0:
            console.print(f"[dim]Copied {copied} sample(s) from main database[/dim]")
    return db


async def list_samples(test: bool) -> None:
    db = await open_database(test)
    try:
        samples = await db.get_all_samples()
        if not samples:
            console.print(
                "[yellow]No samples found. Run 'stt-benchmark download' first.[/yellow]"
            )
            return

        table = Table(title=f"Samples ({'test' if test else 'main'} database)")
        table.add_column("Index", justify="right")
        table.add_column("Sample ID")
        table.add_column("Duration (s)", justify="right")
        table.add_column("Language")
        table.add_column("Audio path")

        for index, sample in enumerate(samples):
            table.add_row(
                str(index),
                sample.sample_id,
                f"{sample.duration_seconds:.2f}",
                sample.language,
                sample.audio_path,
            )
        console.print(table)
    finally:
        await db.close()


async def resolve_sample(
    *,
    sample_id: str | None,
    sample_index: int | None,
    test: bool,
) -> AudioSample:
    db = await open_database(test)
    try:
        if sample_id is not None:
            sample = await db.get_sample(sample_id)
            if sample is None:
                raise ValueError(
                    f"Sample {sample_id!r} not found in "
                    f"{'test' if test else 'main'} database."
                )
            return sample

        samples = await db.get_all_samples()
        if not samples:
            raise ValueError("No samples found. Run 'stt-benchmark download' first.")
        if sample_index is None or sample_index < 0 or sample_index >= len(samples):
            raise ValueError(
                f"Sample index must be between 0 and {len(samples) - 1}."
            )
        return samples[sample_index]
    finally:
        await db.close()


def _resample_pcm(
    pcm: bytes,
    *,
    source_rate: int,
    target_rate: int,
) -> bytes:
    if source_rate == target_rate:
        return pcm

    audio = AudioSegment(
        data=pcm,
        sample_width=BYTE_PER_SAMPLE,
        frame_rate=source_rate,
        channels=1,
    )
    return audio.set_frame_rate(target_rate).raw_data


def load_audio_file(
    path: Path,
    *,
    sample_rate: int = 16000,
    source_sample_rate: int | None = None,
    channel: Channel = "left",
) -> tuple[bytes, float]:
    """Load audio as 16-bit mono PCM at ``sample_rate`` and return duration in seconds."""
    suffix = path.suffix.lower()
    if suffix == ".pcm":
        source_rate = source_sample_rate or get_config().sample_rate
        pcm = _resample_pcm(
            path.read_bytes(),
            source_rate=source_rate,
            target_rate=sample_rate,
        )
    elif suffix in SUPPORTED_MEDIA_SUFFIXES:
        pcm = _decode_media_file(path, channel=channel, sample_rate=sample_rate)
    else:
        raise ValueError(f"Unsupported audio type {suffix!r}; use .pcm, .wav, or .mp3.")

    if not pcm:
        raise ValueError(f"Audio file is empty: {path}")

    duration_seconds = len(pcm) / (sample_rate * BYTE_PER_SAMPLE)
    return pcm, duration_seconds


def _decode_media_file(path: Path, *, channel: Channel, sample_rate: int) -> bytes:
    audio_format = path.suffix.lower().removeprefix(".")
    try:
        audio = AudioSegment.from_file(path, format=audio_format)
    except (PydubException, ValueError, OSError) as exc:
        raise ValueError(f"Couldn't decode {path}") from exc

    if audio.channels == 1:
        selected = audio
    elif channel == "left":
        selected = audio.split_to_mono()[0]
    else:
        selected = audio.split_to_mono()[1]

    normalized = (
        selected.set_frame_rate(sample_rate)
        .set_channels(1)
        .set_sample_width(BYTE_PER_SAMPLE)
    )
    return normalized.raw_data
