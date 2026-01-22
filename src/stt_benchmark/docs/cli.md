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

### Available Services

```
deepgram, assemblyai, openai, groq, gladia, elevenlabs, 
cartesia, google, azure, aws, speechmatics, soniox, whisper
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
| `review` | Interactive review with audio playback |

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

Interactive review of transcription runs with audio playback.

```bash
uv run stt-benchmark ground-truth review <run_id>
```

**Requirements:** `ffplay` (from ffmpeg) for audio playback.

**Controls:**

| Key | Action |
|-----|--------|
| `p` / `r` | Play / Replay audio |
| `a` | Approve transcription |
| `n` | Add note (flag for review) |
| `Enter` | Skip to next |
| `q` | Quit |

```bash
# Review a specific run
uv run stt-benchmark ground-truth review 2026-01-20_14-30-00
```

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

### Examples

```bash
# Calculate WER for all services with results
uv run stt-benchmark wer

# Calculate for specific services
uv run stt-benchmark wer --services deepgram,openai

# Force recalculate (clears existing metrics)
uv run stt-benchmark wer --services deepgram --force-recalculate
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
```

### Output Formats

**Comparison table (default):**
```
                          Service Comparison (Semantic WER)                          
┏━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Service    ┃ Model ┃ Samples ┃ WER Mean ┃ WER Median ┃ TTFB Mean ┃ TTFB Median ┃
┡━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ deepgram   │ -     │     500 │     3.2% │       0.0% │     245ms │       220ms │
│ assemblyai │ -     │     500 │     4.1% │       0.0% │     380ms │       324ms │
└────────────┴───────┴─────────┴──────────┴────────────┴───────────┴─────────────┘
```

**Detailed report (with --service):**
- `validation_summary.txt` - Statistics, distribution, outlier analysis
- `validation_full.csv` - Complete per-sample data

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

## Typical Workflow

```bash
# 1. Download samples
uv run stt-benchmark download --num-samples 100

# 2. Run benchmarks
uv run stt-benchmark run --services deepgram,openai,groq

# 3. Generate ground truth
uv run stt-benchmark ground-truth

# 4. (Optional) Review and correct ground truth
uv run stt-benchmark ground-truth list
uv run stt-benchmark ground-truth review <run_id>

# 5. Calculate semantic WER
uv run stt-benchmark wer

# 6. View results
uv run stt-benchmark report
uv run stt-benchmark report --service deepgram --errors 5
```
