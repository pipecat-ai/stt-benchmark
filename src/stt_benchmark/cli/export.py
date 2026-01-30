"""CLI command for exporting benchmark data for a specific service."""

import asyncio
import csv
import json
from pathlib import Path

import typer
from rich.console import Console

from stt_benchmark.models import ServiceName
from stt_benchmark.services import parse_service_name
from stt_benchmark.storage.database import Database

app = typer.Typer()
console = Console()


def parse_service(service: str) -> ServiceName:
    """Parse a single service name."""
    try:
        return parse_service_name(service)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None


@app.callback(invoke_without_command=True)
def export_service_data(
    service: str = typer.Argument(..., help="Service name to export data for"),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: ./export_<service>)",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name filter",
    ),
    format: str = typer.Option(
        "all",
        "--format",
        "-f",
        help="Export format: csv, json, or all",
    ),
):
    """Export benchmark data for a specific service.

    Exports all data needed for a provider to verify WER results:
    - Sample metadata (ID, duration, dataset index)
    - Ground truth transcriptions
    - Service transcriptions
    - WER metrics and error details

    This allows providers to independently verify benchmark results.
    """
    console.print("\n[bold blue]STT Benchmark - Export Service Data[/bold blue]\n")

    service_name = parse_service(service)
    output_dir = Path(output) if output else Path(f"./export_{service_name.value}")

    console.print(f"Service: {service_name.value}")
    if model:
        console.print(f"Model: {model}")
    console.print(f"Output: {output_dir}")

    asyncio.run(_export_data(service_name, model, output_dir, format))


async def _export_data(
    service_name: ServiceName,
    model_name: str | None,
    output_dir: Path,
    export_format: str,
):
    """Export all data for a service."""
    db = Database()
    await db.initialize()

    # Get all data for this service
    report_data = await db.get_report_data(service_name, model_name)

    if not report_data:
        console.print(f"[yellow]No data found for {service_name.value}[/yellow]")
        await db.close()
        return

    # Get additional sample info
    samples = await db.get_all_samples()
    sample_map = {s.sample_id: s for s in samples}

    # Build export records
    records = []
    for r in report_data:
        sample = sample_map.get(r["sample_id"])
        record = {
            "sample_id": r["sample_id"],
            "dataset_index": sample.dataset_index if sample else None,
            "audio_duration_seconds": r["duration"],
            "ground_truth": r["ground_truth"],
            "transcription": r["transcription"],
            "normalized_reference": r.get("normalized_reference", ""),
            "normalized_hypothesis": r.get("normalized_hypothesis", ""),
            "wer": r["wer"],
            "substitutions": r["substitutions"],
            "deletions": r["deletions"],
            "insertions": r["insertions"],
            "reference_words": r["ref_words"],
            "ttfb_seconds": r["ttfb"],
        }
        records.append(record)

    await db.close()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export based on format
    if export_format in ("csv", "all"):
        csv_path = output_dir / f"{service_name.value}_results.csv"
        _export_csv(records, csv_path)
        console.print(f"[green]✓[/green] Exported CSV: {csv_path}")

    if export_format in ("json", "all"):
        json_path = output_dir / f"{service_name.value}_results.json"
        _export_json(records, service_name, model_name, json_path)
        console.print(f"[green]✓[/green] Exported JSON: {json_path}")

    # Export README with verification instructions
    if export_format == "all":
        readme_path = output_dir / "README.md"
        _export_readme(service_name, model_name, len(records), readme_path)
        console.print(f"[green]✓[/green] Exported README: {readme_path}")

    console.print(f"\n[bold]Exported {len(records)} samples[/bold]")
    console.print("\nProviders can use this data to:")
    console.print("  1. Verify transcriptions match what their service returned")
    console.print("  2. Verify ground truth is reasonable")
    console.print("  3. Recalculate WER using their own methodology")
    console.print("  4. Identify specific samples they want to discuss")


def _export_csv(records: list[dict], path: Path):
    """Export records to CSV."""
    if not records:
        return

    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _export_json(
    records: list[dict],
    service_name: ServiceName,
    model_name: str | None,
    path: Path,
):
    """Export records to JSON with metadata."""
    export_data = {
        "metadata": {
            "service": service_name.value,
            "model": model_name,
            "sample_count": len(records),
            "dataset": "pipecat-ai/smart-turn-data-v3.1-train",
            "notes": "WER is semantic WER - only errors that affect LLM understanding",
        },
        "results": records,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)


def _export_readme(
    service_name: ServiceName,
    model_name: str | None,
    sample_count: int,
    path: Path,
):
    """Export README with verification instructions."""
    model_str = f" ({model_name})" if model_name else ""
    content = f"""# Benchmark Export: {service_name.value}{model_str}

This export contains all benchmark data for {service_name.value} to enable independent verification.

## Contents

- `{service_name.value}_results.csv` - All results in CSV format
- `{service_name.value}_results.json` - All results in JSON format with metadata

## Data Fields

| Field | Description |
|-------|-------------|
| `sample_id` | Unique identifier for the audio sample |
| `dataset_index` | Index in the source dataset (pipecat-ai/smart-turn-data-v3.1-train) |
| `audio_duration_seconds` | Duration of the audio sample |
| `ground_truth` | Ground truth transcription (generated via Gemini with human review) |
| `transcription` | Transcription returned by {service_name.value} |
| `normalized_reference` | Normalized ground truth (for WER calculation) |
| `normalized_hypothesis` | Normalized transcription (for WER calculation) |
| `wer` | Semantic Word Error Rate (0.0 = perfect, 1.0 = 100% errors) |
| `substitutions` | Number of word substitutions |
| `deletions` | Number of word deletions |
| `insertions` | Number of word insertions |
| `reference_words` | Total words in normalized reference |
| `ttfb_seconds` | Time to first byte (latency) |

## Semantic WER Methodology

We use **Semantic WER**, which only counts errors that would impact how an LLM agent understands the user's intent.

**Counted as errors:**
- Word substitutions that change meaning
- Nonsense/hallucinated words
- Missing words that change intent
- Wrong names, numbers, negations

**NOT counted as errors:**
- Punctuation and capitalization differences
- Contractions ("don't" → "do not")
- Singular/plural variations
- Filler words ("um", "uh")
- Number format differences ("5" vs "five")

## Verification Steps

1. **Verify transcriptions**: Compare the `transcription` field against your service's logs
2. **Verify ground truth**: Listen to samples and verify `ground_truth` is accurate
3. **Recalculate WER**: Use your own WER calculation on `normalized_reference` vs `normalized_hypothesis`
4. **Identify disputes**: Note any `sample_id` values you want to discuss

## Audio Access

Audio samples are from the public dataset: `pipecat-ai/smart-turn-data-v3.1-train`

You can access them via HuggingFace:
```python
from datasets import load_dataset
ds = load_dataset("pipecat-ai/smart-turn-data-v3.1-train")
# Use dataset_index to find specific samples
```

## Sample Count

This export contains **{sample_count}** samples.

## Questions?

If you have questions about specific samples or methodology, please reference the `sample_id` when discussing.
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
