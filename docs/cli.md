# CLI Reference

Complete documentation for the `stt-benchmark` command-line interface.

## Overview

```bash
uv run stt-benchmark [COMMAND] [OPTIONS]
```

## Commands

| Command | Description |
|---------|-------------|
| `download` | Download audio samples from People's Speech dataset |
| `run` | Run STT benchmarks on audio samples |
| `ground-truth` | Generate/manage ground truth transcriptions |
| `wer` | Calculate Semantic WER metrics |
| `report` | Generate and view benchmark reports |
| `export` | Export data for a specific service (for provider verification) |

---

## download

Download audio samples from the People's Speech dataset.

```bash
uv run stt-benchmark download [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-n, --num-samples` | INT | 100 | Number of samples to download |
| `-s, --seed` | INT | 42 | Random seed for reproducibility |
| `-o, --offset` | INT | 0 | Skip N samples (for incremental downloads) |
| `--min-duration` | FLOAT | - | Minimum audio duration in seconds |
| `--max-duration` | FLOAT | - | Maximum audio duration in seconds |

### Examples

```bash
# Download 100 samples (default)
uv run stt-benchmark download

# Download 500 samples with specific duration range
uv run stt-benchmark download --num-samples 500 --min-duration 5 --max-duration 15

# Incremental download (skip first 100)
uv run stt-benchmark download --num-samples 100 --offset 100
```

---

## run

Run STT benchmarks on downloaded audio samples.

```bash
uv run stt-benchmark run [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-s, --services` | TEXT | all | Services to benchmark (comma-separated or 'all') |
| `-n, --limit` | INT | - | Limit number of samples to benchmark |
| `-m, --model` | TEXT | - | Model name override |
| `--skip-existing/--no-skip-existing` | BOOL | True | Skip already benchmarked samples |
| `-v, --vad-stop-secs` | FLOAT | 0.2 | VAD silence duration to trigger stop |
| `-t, --test` | BOOL | False | Use separate test database to avoid affecting real data |

### Available Services

```
assemblyai, aws, azure, cartesia, deepgram, elevenlabs, fal,
gladia, google, gradium, groq, nvidia, openai, sambanova,
sarvam, soniox, speechmatics, whisper
```

### Examples

```bash
# Benchmark specific services
uv run stt-benchmark run --services deepgram,openai

# Benchmark all configured services
uv run stt-benchmark run --services all

# Quick test with 10 samples
uv run stt-benchmark run --services deepgram --limit 10

# Override model
uv run stt-benchmark run --services deepgram --model nova-3

# Re-benchmark existing samples
uv run stt-benchmark run --services deepgram --no-skip-existing

# Adjust VAD sensitivity (shorter = faster, longer = more accurate)
uv run stt-benchmark run --services deepgram --vad-stop-secs 0.3

# Test service configurations without affecting real data
uv run stt-benchmark run --test --limit 1 --no-skip-existing

# Test specific services
uv run stt-benchmark run --test -s deepgram,gladia,gradium --limit 1
```

---

## ground-truth

Generate and manage ground truth transcriptions using Gemini.

```bash
uv run stt-benchmark ground-truth [OPTIONS] [SUBCOMMAND]
```

### Options (main command)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-n, --limit` | INT | - | Limit number of samples to transcribe |
| `-m, --model` | TEXT | gemini-3-flash-preview | Gemini model to use |
| `-f, --force` | BOOL | False | Re-transcribe samples that already have ground truth |

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `iterate` | Run a repeatable transcription iteration (saves to JSONL) |
| `list` | List available transcription runs |
| `review` | Interactive review with audio playback and editing |
| `edit` | Edit ground truth for a specific sample in the database |
| `import` | Import ground truth from a JSONL file (with human corrections) |

### Examples

```bash
# Generate ground truth for all samples
uv run stt-benchmark ground-truth

# Generate for 50 samples with specific model
uv run stt-benchmark ground-truth --limit 50 --model gemini-3-flash-preview

# Force regenerate existing ground truth
uv run stt-benchmark ground-truth --force
```

### iterate

Run a transcription iteration for comparison/review.

```bash
uv run stt-benchmark ground-truth iterate [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--samples` | INT | 100 | Number of samples to transcribe |

```bash
# Run an iteration with 100 samples
uv run stt-benchmark ground-truth iterate --samples 100
```

### import

Import ground truth transcriptions from a JSONL file into the database.

```bash
uv run stt-benchmark ground-truth import <JSONL_FILE> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--force, -f` | BOOL | False | Overwrite existing ground truth entries |
| `--test, -t` | BOOL | False | Import to test database (test_results.db) instead of main database |

**Important:** You import the **main JSONL file** (not the `_notes.jsonl` file). Include the full path:

```bash
# Import from the ground_truth_runs directory
uv run stt-benchmark ground-truth import stt_benchmark_data/ground_truth_runs/2026-01-03_17-00-06.jsonl

# Force overwrite existing entries
uv run stt-benchmark ground-truth import stt_benchmark_data/ground_truth_runs/2026-01-03_17-00-06.jsonl --force

# Import to test database (for test runs)
uv run stt-benchmark ground-truth import stt_benchmark_data/ground_truth_runs/2026-01-03_17-00-06.jsonl --test
```

**How it works with human corrections:**

The import command automatically looks for a `_notes.jsonl` sidecar file in the same directory:

```
stt_benchmark_data/ground_truth_runs/
├── 2026-01-03_17-00-06.jsonl        ← Import THIS file (AI transcriptions)
└── 2026-01-03_17-00-06_notes.jsonl  ← Auto-detected (your edits from review)
```

If the notes file exists, your human corrections are merged automatically:

```bash
$ uv run stt-benchmark ground-truth import stt_benchmark_data/ground_truth_runs/2026-01-03_17-00-06.jsonl
Found 12 human corrections in notes file
✓ Imported 100 ground truth transcriptions
  (12 with human corrections)
```

The import stores:
- The corrected text as the ground truth
- The original AI text for reference
- Verification metadata (`verified_by: human`, timestamp)

Only samples that exist in your local database will be imported.
Samples not yet downloaded will be skipped.

### list

List available transcription runs.

```bash
uv run stt-benchmark ground-truth list
```

Output:
```
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Run ID              ┃ Model                 ┃ Samples ┃ Reviewed ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ 2026-01-20_14-30-00 │ gemini-3-flash-preview│     100 │     45   │
│ 2026-01-19_10-15-00 │ gemini-3-flash-preview│      50 │     50   │
└─────────────────────┴───────────────────────┴─────────┴──────────┘
```

### review

Interactive review of transcription runs with audio playback and editing.

```bash
uv run stt-benchmark ground-truth review <run_id>
```

**Requirements:** `ffplay` (from ffmpeg) for audio playback.

**Controls:**

| Key | Action |
|-----|--------|
| `p` / `r` | Play / Replay audio |
| `e` | Edit transcription (correct errors) |
| `a` | Approve transcription |
| `n` | Add note (flag for review) |
| `Enter` | Skip to next |
| `q` | Quit |

```bash
# Review a specific run
uv run stt-benchmark ground-truth review 2026-01-20_14-30-00
```

**Batch Review & Resume:**

You can review samples in batches - progress is saved automatically:

```bash
# Session 1: Review some samples, then press 'q' to quit
uv run stt-benchmark ground-truth review 2026-01-20_14-30-00
# ... review samples 1-50, then quit

# Session 2: Automatically resumes where you left off
uv run stt-benchmark ground-truth review 2026-01-20_14-30-00
# Found 50 existing reviews
# Found 3 existing edits
# Starting from sample 51
```

**How Edits Work:**

1. Edits are saved to a separate `<run_id>_notes.jsonl` file
2. Original AI transcriptions are preserved
3. When you run `import`, human corrections are automatically applied
4. You can re-import with `--force` to apply additional corrections

### edit

Edit ground truth for a specific sample directly in the database.

```bash
uv run stt-benchmark ground-truth edit <SAMPLE_ID> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--text, -t` | TEXT | - | New transcription (interactive if not provided) |

Use this for samples already imported into the database.

```bash
# Interactive mode - will prompt for new text
uv run stt-benchmark ground-truth edit f3464b75

# Direct mode - provide text on command line
uv run stt-benchmark ground-truth edit f3464b75 --text "Resume playing"

# Partial sample IDs work (matches prefix)
uv run stt-benchmark ground-truth edit f346 --text "Resume playing"
```

The original AI-generated text is preserved in the database for reference.

---

## wer

Calculate Semantic WER metrics using Claude.

```bash
uv run stt-benchmark wer [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-s, --services` | TEXT | all | Services to evaluate (comma-separated or 'all') |
| `-m, --model` | TEXT | - | Model name filter |
| `-f, --force-recalculate` | BOOL | False | Delete existing metrics and recalculate |
| `-t, --test` | BOOL | False | Use test database (test_results.db) instead of main database |

### Examples

```bash
# Calculate WER for all services with results
uv run stt-benchmark wer

# Calculate for specific services
uv run stt-benchmark wer --services deepgram,openai

# Force recalculate (clears existing metrics)
uv run stt-benchmark wer --services deepgram --force-recalculate

# Calculate WER on test database
uv run stt-benchmark wer --test
```

### What Semantic WER Measures

**Counted as errors:**
- Word substitutions that change meaning
- Nonsense/hallucinated words
- Missing words that change intent
- Wrong names, numbers, negations

**NOT counted as errors:**
- Punctuation and capitalization
- Contractions ("don't" → "do not")
- Singular/plural variations
- Filler words ("um", "uh")
- Number format differences

---

## report

Generate and view benchmark reports.

```bash
uv run stt-benchmark report [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-s, --service` | TEXT | - | Service for detailed report (omit for comparison table) |
| `-o, --output` | TEXT | stt_benchmark_data | Output directory for report files |
| `-m, --model` | TEXT | - | Model name filter |
| `-e, --errors` | INT | - | Show N worst samples (requires --service) |
| `-t, --test` | BOOL | False | Use test database (test_results.db) instead of main database |

### Examples

```bash
# Show comparison table of all services
uv run stt-benchmark report

# Detailed report for a specific service
uv run stt-benchmark report --service deepgram

# Show 10 worst samples
uv run stt-benchmark report --service deepgram --errors 10

# Custom output directory
uv run stt-benchmark report --service deepgram --output ./reports

# View results from test database
uv run stt-benchmark report --test
```

### Output Formats

**Comparison table (default):**
```
                                        Service Comparison
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Service    ┃ Transcripts       ┃ Perfect ┃ WER Mean ┃ Pooled WER ┃ TTFS Median ┃ TTFS P95 ┃ TTFS P99 ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ deepgram   │ 998/1000 (99.8%)  │   75.3% │    1.91% │      1.85% │       247ms │    299ms │    316ms │
│ assemblyai │ 998/1000 (99.8%)  │   65.2% │    3.86% │      3.72% │       326ms │   1476ms │   1936ms │
└────────────┴───────────────────┴─────────┴──────────┴────────────┴─────────────┴──────────┴──────────┘
```

**Detailed report (with --service):**
- `validation_summary.txt` - Statistics, distribution, outlier analysis
- `validation_full.csv` - Complete per-sample data

---

## export

Export benchmark data for a specific service, enabling providers to independently verify WER results.

```bash
uv run stt-benchmark export [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-s, --service` | TEXT | (required) | Service name to export data for |
| `-o, --output` | TEXT | ./export_{service} | Output directory |
| `-m, --model` | TEXT | - | Model name filter |
| `-f, --format` | TEXT | all | Export format: csv, json, or all |
| `-t, --test` | BOOL | False | Export from test database (test_results.db) instead of main database |

### Examples

```bash
# Export all data for deepgram
uv run stt-benchmark export -s deepgram

# Export to a specific directory
uv run stt-benchmark export -s deepgram -o ./deepgram_verification

# Export only CSV format
uv run stt-benchmark export -s deepgram --format csv

# Export for a specific model
uv run stt-benchmark export -s deepgram --model nova-3

# Export from test database
uv run stt-benchmark export -s deepgram --test
```

### Output Files

When using `--format all` (default), the export creates:

| File | Description |
|------|-------------|
| `{service}_results.csv` | All results in CSV format |
| `{service}_results.json` | All results in JSON format with metadata |
| `README.md` | Verification instructions for the provider |

### Exported Data Fields

| Field | Description |
|-------|-------------|
| `sample_id` | Unique identifier for the audio sample |
| `dataset_index` | Index in the source dataset |
| `audio_duration_seconds` | Duration of the audio sample |
| `ground_truth` | Ground truth transcription |
| `transcription` | Transcription returned by the service |
| `normalized_reference` | Normalized ground truth (for WER calculation) |
| `normalized_hypothesis` | Normalized transcription (for WER calculation) |
| `wer` | Semantic Word Error Rate |
| `substitutions` | Number of word substitutions |
| `deletions` | Number of word deletions |
| `insertions` | Number of word insertions |
| `reference_words` | Total words in normalized reference |
| `ttfb_seconds` | Time to first byte (latency) |

### Use Case: Provider Verification

When a provider disputes WER results, export their data so they can:

1. **Verify transcriptions** - Compare against their service logs
2. **Verify ground truth** - Listen to samples and check accuracy
3. **Recalculate WER** - Use their own methodology on normalized text
4. **Identify disputes** - Reference specific `sample_id` values

This approach is transparent (providers see exactly what was measured) while being fair (no competitor data is shared).

---

## Environment Variables

| Variable | Required For |
|----------|--------------|
| `ANTHROPIC_API_KEY` | `wer` command (Claude) |
| `GOOGLE_API_KEY` | `ground-truth` command (Gemini) |
| `DEEPGRAM_API_KEY` | `run --services deepgram` |
| `OPENAI_API_KEY` | `run --services openai` |
| ... | See README for full list |

Variables can be set in a `.env` file in the project root.

---

## Test vs Main Database

The benchmark uses two separate SQLite databases:

| Database | File | Purpose |
|----------|------|---------|
| **Main** | `stt_benchmark_data/benchmark.db` | Production benchmark data |
| **Test** | `stt_benchmark_data/test_results.db` | Testing without affecting real data |

**Important:** Ground truth must be imported separately to each database. The test database does not share ground truth with the main database.

---

## Testing Service Configurations

Use the `--test` flag to validate service configurations without affecting your real benchmark data.

### Quick Test (Transcription Only)

```bash
# Test all available services with 1 sample each
uv run stt-benchmark run --test --limit 1 --no-skip-existing

# Or test specific services
uv run stt-benchmark run --test -s deepgram,assemblyai --limit 1

# View test results (no WER without ground truth)
uv run stt-benchmark report --test

# Clean up when done (optional)
rm ./stt_benchmark_data/test_results.db
```

### Full Test Workflow with WER

To calculate semantic WER on test runs, you must import ground truth to the test database:

```bash
# 1. Run test benchmark
uv run stt-benchmark run --test -s deepgram,assemblyai --limit 100

# 2. Import ground truth to test database (use your actual run_id)
uv run stt-benchmark ground-truth import stt_benchmark_data/ground_truth_runs/2026-01-03_17-00-06.jsonl --test

# 3. Calculate WER on test data
uv run stt-benchmark wer --test

# 4. View test results with WER
uv run stt-benchmark report --test

# 5. Force recalculate WER if ground truth changed
uv run stt-benchmark wer --test --force-recalculate
```

This is useful when:
- Setting up new service configurations
- Verifying API keys and endpoints work
- Testing changes to `services.py` factory functions
- Running full end-to-end tests with WER calculation
- Testing with updated Pipecat versions

---

## Typical Workflow (Main Database)

This is the standard workflow for production benchmarking:

```bash
# 1. Download samples from HuggingFace dataset
uv run stt-benchmark download --num-samples 1000

# 2. Run benchmarks against STT services
uv run stt-benchmark run --services deepgram,assemblyai,openai

# 3. Generate ground truth (AI transcription with Gemini)
uv run stt-benchmark ground-truth iterate --samples 1000
# Creates: stt_benchmark_data/ground_truth_runs/<timestamp>.jsonl

# 4. Review and correct ground truth (can be done in batches)
uv run stt-benchmark ground-truth list
uv run stt-benchmark ground-truth review <run_id>
# Controls: [p] play, [e] edit, [a] approve, [n] note, [q] quit
# Progress is saved automatically - resume anytime

# 5. Import ground truth with human corrections (use full path)
uv run stt-benchmark ground-truth import stt_benchmark_data/ground_truth_runs/<run_id>.jsonl

# 6. Calculate semantic WER
uv run stt-benchmark wer

# 7. View results
uv run stt-benchmark report
uv run stt-benchmark report --service deepgram --errors 10

# 8. (Optional) Export data for provider verification
uv run stt-benchmark export -s deepgram -o ./deepgram_verification
```

### Re-running with Updated Ground Truth

After making corrections to ground truth:

```bash
# Re-import with --force to overwrite existing entries
uv run stt-benchmark ground-truth import stt_benchmark_data/ground_truth_runs/<run_id>.jsonl --force

# Recalculate WER with updated ground truth
uv run stt-benchmark wer --force-recalculate

# View updated results
uv run stt-benchmark report
```

### Correcting Ground Truth After Import

If you find errors in already-imported ground truth:

```bash
# Option 1: Edit a specific sample directly in the database
uv run stt-benchmark ground-truth edit <sample_id> --text "Correct transcription"

# Option 2: Re-review in the review tool and re-import
uv run stt-benchmark ground-truth review <run_id>
uv run stt-benchmark ground-truth import stt_benchmark_data/ground_truth_runs/<run_id>.jsonl --force
```
