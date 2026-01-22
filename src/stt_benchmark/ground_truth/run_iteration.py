"""Run a ground truth transcription iteration.

Generates transcriptions for a repeatable set of samples and saves to JSONL
for longitudinal comparison as we iterate on the prompt.

Usage:
    uv run stt-benchmark ground-truth iterate --samples 100
"""

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from stt_benchmark.config import get_config
from stt_benchmark.ground_truth.gemini_transcriber import (
    TRANSCRIPTION_PROMPT,
    GeminiTranscriber,
)
from stt_benchmark.models import AudioSample
from stt_benchmark.storage.database import Database


def get_prompt_hash(prompt: str) -> str:
    """Generate a short hash of the prompt for identification."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


async def get_first_n_samples(db: Database, n: int) -> list[AudioSample]:
    """Get first N samples ordered by dataset_index for repeatability."""
    cursor = await db._conn.execute(
        """
        SELECT * FROM samples
        ORDER BY dataset_index
        LIMIT ?
        """,
        (n,),
    )
    rows = await cursor.fetchall()
    return [
        AudioSample(
            sample_id=row["sample_id"],
            audio_path=row["audio_path"],
            duration_seconds=row["duration_seconds"],
            language=row["language"],
            dataset_index=row["dataset_index"],
        )
        for row in rows
    ]


async def run_iteration(
    num_samples: int = 100,
    progress_callback: Callable | None = None,
) -> Path:
    """Run a transcription iteration and save to JSONL.

    Args:
        num_samples: Number of samples to transcribe
        progress_callback: Optional callback(current, total, sample_id)

    Returns:
        Path to the output JSONL file
    """
    config = get_config()
    db = Database()
    await db.initialize()

    # Get samples
    samples = await get_first_n_samples(db, num_samples)
    if not samples:
        logger.error("No samples found in database")
        raise ValueError("No samples found")

    logger.info(f"Selected {len(samples)} samples for transcription")

    # Create output directory
    runs_dir = config.data_dir / "ground_truth_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Generate run ID from timestamp
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = runs_dir / f"{run_id}.jsonl"

    # Create transcriber
    transcriber = GeminiTranscriber()

    # Write header
    header = {
        "type": "header",
        "run_id": run_id,
        "model": transcriber.model_name,
        "prompt_hash": get_prompt_hash(TRANSCRIPTION_PROMPT),
        "prompt_text": TRANSCRIPTION_PROMPT,
        "num_samples": len(samples),
        "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    with open(output_path, "w") as f:
        f.write(json.dumps(header) + "\n")

    # Transcribe samples and append to file
    completed = 0
    errors = 0

    for i, sample in enumerate(samples):
        if progress_callback:
            progress_callback(i, len(samples), sample.sample_id)

        logger.info(f"[{i + 1}/{len(samples)}] Transcribing {sample.sample_id}...")

        try:
            gt = await transcriber.transcribe_sample(sample)

            if gt:
                record = {
                    "type": "sample",
                    "sample_id": sample.sample_id,
                    "audio_path": sample.audio_path,
                    "duration_seconds": sample.duration_seconds,
                    "transcription": gt.text,
                    "generated_at": gt.generated_at.isoformat() + "Z",
                }
                completed += 1
            else:
                record = {
                    "type": "sample",
                    "sample_id": sample.sample_id,
                    "audio_path": sample.audio_path,
                    "duration_seconds": sample.duration_seconds,
                    "transcription": None,
                    "error": "Empty response from Gemini",
                    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
                errors += 1

        except Exception as e:
            logger.error(f"Error transcribing {sample.sample_id}: {e}")
            record = {
                "type": "sample",
                "sample_id": sample.sample_id,
                "audio_path": sample.audio_path,
                "duration_seconds": sample.duration_seconds,
                "transcription": None,
                "error": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            errors += 1

        # Append to file (streaming writes)
        with open(output_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # Write footer with summary
    footer = {
        "type": "footer",
        "run_id": run_id,
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_samples": len(samples),
        "successful": completed,
        "errors": errors,
    }

    with open(output_path, "a") as f:
        f.write(json.dumps(footer) + "\n")

    logger.info(f"Run complete: {completed} successful, {errors} errors")
    logger.info(f"Output saved to: {output_path}")

    await db.close()
    return output_path


def list_runs() -> list[dict]:
    """List all available runs.

    Returns:
        List of run info dicts with run_id, model, num_samples, reviewed_count
    """
    config = get_config()
    runs_dir = config.data_dir / "ground_truth_runs"

    if not runs_dir.exists():
        return []

    jsonl_files = sorted(runs_dir.glob("*.jsonl"))
    run_files = [f for f in jsonl_files if not f.stem.endswith("_notes")]

    runs = []
    for run_path in run_files:
        try:
            header, samples, footer = load_run(run_path)
            run_id = header["run_id"]
            model = header.get("model", "?")
            num_samples = len(samples)

            # Check for notes
            notes_path = run_path.parent / f"{run_id}_notes.jsonl"
            reviewed_count = 0
            if notes_path.exists():
                notes = load_existing_notes(notes_path)
                reviewed_count = len(notes)

            runs.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "num_samples": num_samples,
                    "reviewed_count": reviewed_count,
                    "path": str(run_path),
                }
            )

        except Exception as e:
            logger.warning(f"Error loading run {run_path.name}: {e}")

    return runs


def load_run(run_path: Path) -> tuple[dict, list[dict], dict | None]:
    """Load a run JSONL file.

    Returns:
        (header, samples, footer)
    """
    header = None
    samples = []
    footer = None

    with open(run_path) as f:
        for line in f:
            record = json.loads(line.strip())
            if record["type"] == "header":
                header = record
            elif record["type"] == "sample":
                samples.append(record)
            elif record["type"] == "footer":
                footer = record

    if not header:
        raise ValueError(f"No header found in {run_path}")

    return header, samples, footer


def load_existing_notes(notes_path: Path) -> dict[str, dict]:
    """Load existing notes if any, returning a dict keyed by sample_id."""
    notes = {}
    if notes_path.exists():
        with open(notes_path) as f:
            for line in f:
                record = json.loads(line.strip())
                if record["type"] == "review":
                    notes[record["sample_id"]] = record
    return notes


def load_existing_edits(notes_path: Path) -> dict[str, dict]:
    """Load existing edits if any, returning a dict keyed by sample_id.

    If multiple edits exist for a sample, returns the most recent one.
    """
    edits = {}
    if notes_path.exists():
        with open(notes_path) as f:
            for line in f:
                record = json.loads(line.strip())
                if record["type"] == "edit":
                    # Later edits override earlier ones
                    edits[record["sample_id"]] = record
    return edits
