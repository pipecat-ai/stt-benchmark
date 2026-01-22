# Running Analysis Guide

This guide walks through a complete STT benchmarking analysis from start to finish.

## Overview

A complete analysis involves:
1. Downloading audio samples
2. Running benchmarks across STT services
3. Generating ground truth transcriptions
4. Calculating semantic WER
5. Reviewing results and identifying issues

## Prerequisites

```bash
# Install dependencies
cd stt-benchmark
uv sync

# Set up API keys
cp env.example .env
# Edit .env with your keys
```

**Required API keys:**
- `ANTHROPIC_API_KEY` - For semantic WER calculation
- `GOOGLE_API_KEY` - For ground truth generation
- STT service keys for services you want to benchmark

---

## Step 1: Download Audio Samples

Download samples from the People's Speech dataset:

```bash
uv run stt-benchmark download --num-samples 100
```

**Recommendations:**
- Start with 50-100 samples for initial testing
- Use 500+ samples for statistically meaningful results
- Samples are ~5-15 seconds of conversational speech

**Verify download:**
```bash
ls -la stt_benchmark_data/audio/ | head -10
```

---

## Step 2: Run Benchmarks

### Quick Test (1 service, few samples)

```bash
uv run stt-benchmark run --services deepgram --limit 10
```

### Full Benchmark (multiple services)

```bash
uv run stt-benchmark run --services deepgram,openai,groq,assemblyai
```

### All Configured Services

```bash
uv run stt-benchmark run --services all
```

**What's measured:**
- **TTFB** - Time from user stops speaking to first transcription byte
- **Transcription** - Full text output for WER calculation

**Typical runtime:** ~1-2 minutes per 100 samples per service (varies by service latency).

### Monitoring Progress

The CLI shows a progress bar. For more detail:

```bash
# Check database for results
sqlite3 stt_benchmark_data/results.db "SELECT service_name, COUNT(*) FROM benchmark_results GROUP BY service_name;"
```

### Handling Errors

Some samples may fail (network issues, service errors). Check error counts:

```bash
sqlite3 stt_benchmark_data/results.db "SELECT service_name, COUNT(*) as errors FROM benchmark_results WHERE error IS NOT NULL GROUP BY service_name;"
```

Re-run to fill in gaps:

```bash
# Skip existing will only process samples without results
uv run stt-benchmark run --services deepgram
```

---

## Step 3: Generate Ground Truth

Ground truth is the reference transcription we compare STT results against.

### Basic Generation

```bash
uv run stt-benchmark ground-truth
```

This uses Gemini to transcribe all samples. Results are saved to the database.

### Verify Ground Truth Coverage

```bash
sqlite3 stt_benchmark_data/results.db "SELECT COUNT(*) as samples_with_gt FROM ground_truths;"
```

---

## Step 4: Calculate Semantic WER

### Calculate for All Services

```bash
uv run stt-benchmark wer
```

### Calculate for Specific Services

```bash
uv run stt-benchmark wer --services deepgram,openai
```

### Force Recalculation

If you've updated ground truth or want fresh results:

```bash
uv run stt-benchmark wer --services deepgram --force-recalculate
```

**What happens:**
- Claude compares each transcription to ground truth
- Only semantic errors are counted (not punctuation, contractions, etc.)
- Full reasoning traces are saved for debugging

**Typical runtime:** ~30-60 seconds per 100 samples per service.

---

## Step 5: View Results

### Comparison Table

```bash
uv run stt-benchmark report
```

Output:
```
                          Service Comparison (Semantic WER)                          
┏━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Service    ┃ Model ┃ Samples ┃ WER Mean ┃ WER Median ┃ TTFB Mean ┃ TTFB Median ┃
┡━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ deepgram   │ -     │     100 │     3.2% │       0.0% │     245ms │       220ms │
│ groq       │ -     │     100 │     4.8% │       0.0% │     180ms │       165ms │
│ openai     │ -     │     100 │     5.8% │       2.1% │     520ms │       485ms │
└────────────┴───────┴─────────┴──────────┴────────────┴───────────┴─────────────┘
```

### Detailed Service Report

```bash
uv run stt-benchmark report --service deepgram
```

Creates:
- `stt_benchmark_data/validation_summary.txt` - Statistics and outliers
- `stt_benchmark_data/validation_full.csv` - Per-sample data

### Worst Samples Analysis

Find samples with highest error rates:

```bash
uv run stt-benchmark report --service deepgram --errors 10
```

This helps identify:
- Audio quality issues
- Accents or speech patterns that cause problems
- Potential ground truth errors

---

## Step 6: Ground Truth Quality (Optional)

For high-quality benchmarks, review and correct ground truth.

### Run a Review Iteration

```bash
# Generate iteration for review
uv run stt-benchmark ground-truth iterate --samples 50

# List available runs
uv run stt-benchmark ground-truth list

# Interactive review
uv run stt-benchmark ground-truth review 2026-01-20_14-30-00
```

### Review Controls

| Key | Action |
|-----|--------|
| `p` | Play audio |
| `a` | Approve (transcription is correct) |
| `e` | Edit (fix transcription, saves to database) |
| `n` | Note (flag for later) |
| `Enter` | Skip |
| `q` | Quit |

### Direct Edits

If you know a specific sample needs correction:

```bash
uv run stt-benchmark ground-truth edit <sample_id> --text "corrected text"
```

### After Corrections

Recalculate WER with the updated ground truth:

```bash
uv run stt-benchmark wer --force-recalculate
```

---

## Interpreting Results

### TTFB

| TTFB Range | Assessment |
|------------|------------|
| < 300ms | Excellent - suitable for real-time voice agents |
| 300-500ms | Good - acceptable for most applications |
| 500-800ms | Fair - noticeable latency |
| > 800ms | Poor - may cause conversation flow issues |

### Semantic WER

| WER Range | Assessment |
|-----------|------------|
| < 3% | Excellent - minimal errors |
| 3-5% | Good - occasional errors |
| 5-10% | Fair - some accuracy issues |
| > 10% | Poor - significant errors |

### Key Metrics

- **Mean vs Median WER**: High mean with low median indicates outliers (some very bad samples)
- **P95 TTFB**: Worst-case latency (important for user experience)
- **Sample count**: Ensure sufficient samples for statistical significance (100+ recommended)

---

## Common Issues

### "No ground truth for sample"

Run ground truth generation:
```bash
uv run stt-benchmark ground-truth
```

### "API key not set"

Check your `.env` file and ensure the key is set:
```bash
grep DEEPGRAM_API_KEY .env
```

### High Error Rates

1. Check if it's a service issue or audio issue:
   ```bash
   # Same samples failing across services = audio issue
   uv run stt-benchmark report --service deepgram --errors 5
   uv run stt-benchmark report --service openai --errors 5
   ```

2. Review problematic samples:
   ```bash
   # Play the audio and check ground truth
   uv run stt-benchmark ground-truth review <run_id>
   ```

### Timeout Errors

Some services may time out on long audio. The default timeout is 10 seconds after audio completes. Check logs for timeout messages.

---

## Database Queries

### Sample Statistics

```sql
-- Samples per service
SELECT service_name, COUNT(*) as count, 
       AVG(ttfb_seconds) as avg_ttfb
FROM benchmark_results 
WHERE error IS NULL
GROUP BY service_name;
```

### Error Analysis

```sql
-- Samples with errors
SELECT service_name, sample_id, error
FROM benchmark_results
WHERE error IS NOT NULL
LIMIT 20;
```

### WER Distribution

```sql
-- WER by service
SELECT service_name, 
       AVG(wer) as mean_wer,
       MIN(wer) as min_wer,
       MAX(wer) as max_wer
FROM wer_metrics
GROUP BY service_name;
```

Run queries with:
```bash
sqlite3 stt_benchmark_data/results.db "YOUR QUERY HERE"
```

---

## Exporting Data

### CSV Export

The `--service` report automatically creates CSV:
```bash
uv run stt-benchmark report --service deepgram
# Creates: stt_benchmark_data/validation_full.csv
```

### Manual Export

```bash
sqlite3 -header -csv stt_benchmark_data/results.db \
  "SELECT * FROM benchmark_results WHERE service_name='deepgram'" \
  > deepgram_results.csv
```

---

## Batch Analysis Script

For running comprehensive benchmarks:

```bash
#!/bin/bash

SERVICES="deepgram,openai,groq,assemblyai"
SAMPLES=500

echo "Downloading samples..."
uv run stt-benchmark download --num-samples $SAMPLES

echo "Running benchmarks..."
uv run stt-benchmark run --services $SERVICES

echo "Generating ground truth..."
uv run stt-benchmark ground-truth

echo "Calculating semantic WER..."
uv run stt-benchmark wer

echo "Generating reports..."
uv run stt-benchmark report

for service in ${SERVICES//,/ }; do
    uv run stt-benchmark report --service $service
done

echo "Done! Results in stt_benchmark_data/"
```
