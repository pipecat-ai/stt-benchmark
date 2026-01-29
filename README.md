# STT Benchmark

A framework for benchmarking Speech-to-Text services with TTFS (Time To Final Segment) latency and Semantic WER (Word Error Rate) accuracy measurement.

## Results Summary

Benchmark results on 1000 samples from the `pipecat-ai/smart-turn-data-v3.1-train` dataset.

| Service | Transcripts | Perfect | WER Mean | TTFS Median | TTFS P95 | TTFS P99 |
|---------|-------------|------------|----------|-------------|----------|----------|
| assemblyai | 99.8% | 65.2% | 3.86% | 326ms | 1476ms | 1936ms |
| aws | 99.9% | 76.1% | 1.81% | 1105ms | 1513ms | 1924ms |
| azure | 99.9% | 80.2% | 1.44% | 1006ms | 1339ms | 1743ms |
| cartesia | 99.9% | 59.8% | 4.14% | 257ms | 279ms | 526ms |
| deepgram | 99.8% | 75.3% | 1.91% | 247ms | 299ms | 316ms |
| elevenlabs | 99.7% | 79.8% | 3.26% | 295ms | 380ms | 465ms |
| gladia | 99.6% | 68.6% | 5.08% | 882ms | 1733ms | 1948ms |
| google | 100.0% | 67.8% | 3.14% | 863ms | 1132ms | 1522ms |
| openai | 100.0% | 74.8% | 3.70% | 852ms | 1413ms | 1806ms |
| soniox | 100.0% | 78.8% | 2.06% | 454ms | 853ms | 1166ms |
| speechmatics | 99.9% | 80.8% | 1.81% | 543ms | 743ms | 813ms |

### Latency vs Accuracy Trade-off

![STT Service Pareto Frontier](assets/stt_pareto_frontier.png)

The Pareto frontier shows services that offer the best trade-off between latency and accuracy—no other service is better on both metrics. Services on the frontier represent efficient choices depending on your priorities.

### Metrics Glossary

| Metric | Description |
|--------|-------------|
| **Transcripts** | Percentage of samples where STT successfully returned a transcription |
| **Perfect** | Perfect transcriptions (0% semantic WER) out of total benchmark runs |
| **WER Mean** | Average semantic word error rate across all samples |
| **TTFS Median** | Median time from user stops speaking to final transcription segment |
| **TTFS P95** | 95th percentile TTFS - worst 5% of samples have latency above this |
| **TTFS P99** | 99th percentile TTFS - worst 1% of samples have latency above this |

> **Semantic WER** measures only transcription errors that would impact an LLM agent's understanding. Punctuation, contractions, filler words, and equivalent phrasings are ignored.

> **TTFS (Time To Final Segment)** is measured from when the user stops speaking to when the final transcription segment is received. For streaming voice agents, lower TTFS means faster response times.

## Quick Start

```bash
# Install dependencies
uv sync

# Download audio samples
uv run stt-benchmark download --num-samples 100

# Run benchmarks
uv run stt-benchmark run --services deepgram,openai

# Generate ground truth (Gemini)
uv run stt-benchmark ground-truth

# Calculate semantic WER (Claude)
uv run stt-benchmark wer

# View results
uv run stt-benchmark report
```

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd stt-benchmark
uv sync
```

## Environment Variables

Copy `env.example` to `.env` and set your API keys:

```bash
cp env.example .env
```

```bash
# Required for evaluation
ANTHROPIC_API_KEY=sk-ant-...    # Semantic WER calculation (Claude)
GOOGLE_API_KEY=...               # Ground truth generation (Gemini)

# STT Services (alphabetized - set the ones you want to benchmark)
ASSEMBLYAI_API_KEY=...
CARTESIA_API_KEY=...
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...
FAL_KEY=...
GLADIA_API_KEY=...
GRADIUM_API_KEY=...
GROQ_API_KEY=...
NVIDIA_API_KEY=...
OPENAI_API_KEY=...
SAMBANOVA_API_KEY=...
SARVAM_API_KEY=...
SONIOX_API_KEY=...
SPEECHMATICS_API_KEY=...

# Services with special requirements
AWS_ACCESS_KEY_ID=...                                      # AWS Transcribe
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
AZURE_SPEECH_API_KEY=...                                   # Azure Speech
AZURE_SPEECH_REGION=eastus
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json  # Google Cloud STT
```

## How It Works

### TTFB Measurement

**TTFB for STT is different from typical request/response TTFB.** Since STT services receive continuous audio input, there's no discrete request to measure from. Instead, we measure from when the user **stops speaking** to when the **final transcription** arrives.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ VADUserStartedSpeaking              Actual speech    VADUserStopped         │
│        t=0                            ends           SpeakingFrame          │
│         │                              │                  │                 │
│         ▼                              ▼                  ▼                 │
│  ═══════╪══════════════════════════════╪══════════════════╪════             │
│         │      Audio streaming to STT  │   VAD stop_secs  │                 │
│         │                              │◄────────────────►│                 │
│         │                              │                  │                 │
│         │                              └──── TTFB ────────┼────────►        │
│         │                           speech_end_time       │     T3          │
│         │                                                 │  (final         │
│         │     T1              T2                          │ transcript)     │
│         │      │               │                          │                 │
│         │      ▼               ▼                          │                 │
│         │  transcript      transcript                     │                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key points:**
- `speech_end_time` = `VADUserStoppedSpeakingFrame` timestamp − VAD `stop_secs`
- TTFB = final `TranscriptionFrame` receipt time − `speech_end_time`
- Streaming services emit multiple partial transcripts; we use the **final** one

**Why the final transcript?** For LLM/TTS, there's a discrete input→output making TTFB simple. For streaming STT, audio flows continuously and generates multiple `TranscriptionFrame`s. We can't know when the STT service finalized audio for intermediate transcripts, so we measure from the final one and use the VAD signal to determine when the user actually stopped spekaing.

### Semantic WER

Traditional WER penalizes every word difference equally. "gonna" vs "going to" counts as 2 errors.

**Semantic WER** uses Claude to evaluate whether differences actually matter:

| Ignored (not errors) | Counted (errors) |
|---------------------|------------------|
| Punctuation, capitalization | Word substitutions that change meaning |
| Contractions ("don't" → "do not") | Nonsense/hallucinated words |
| Singular/plural ("license" → "licenses") | Missing words that change intent |
| Filler words ("um", "uh") | Wrong names, numbers, negations |
| Number formats ("3" → "three") | Factual errors |

This gives accuracy metrics that reflect real-world impact on downstream LLM applications.

## Supported Services

| Service | Default Model | API Key |
|---------|---------------|---------|
| `assemblyai` | - | `ASSEMBLYAI_API_KEY` |
| `aws` | - | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_REGION` |
| `azure` | - | `AZURE_SPEECH_API_KEY` + `AZURE_SPEECH_REGION` |
| `cartesia` | ink-whisper | `CARTESIA_API_KEY` |
| `deepgram` | nova-3-general | `DEEPGRAM_API_KEY` |
| `deepgram_flux` | flux-general-en | `DEEPGRAM_API_KEY` |
| `elevenlabs` | scribe_v2_realtime | `ELEVENLABS_API_KEY` |
| `fal` | - | `FAL_KEY` |
| `gladia` | solaria-1 | `GLADIA_API_KEY` |
| `google` | latest_long | `GOOGLE_APPLICATION_CREDENTIALS` |
| `gradium` | - | `GRADIUM_API_KEY` |
| `groq` | whisper-large-v3-turbo | `GROQ_API_KEY` |
| `nvidia` | - | `NVIDIA_API_KEY` |
| `openai` | gpt-4o-transcribe | `OPENAI_API_KEY` |
| `sambanova` | Whisper-Large-v3 | `SAMBANOVA_API_KEY` |
| `sarvam` | saarika:v2.5 | `SARVAM_API_KEY` |
| `soniox` | stt-rt-preview | `SONIOX_API_KEY` |
| `speechmatics` | - | `SPEECHMATICS_API_KEY` |
| `whisper` | faster-distil-whisper-medium.en | None (local) |

## CLI Commands

### Running Benchmarks

```bash
# Benchmark specific services
uv run stt-benchmark run --services deepgram,openai

# Benchmark all configured services
uv run stt-benchmark run --services all

# Limit samples and adjust VAD
uv run stt-benchmark run --services deepgram --limit 50 --vad-stop-secs 0.3
```

### Generating Ground Truth

```bash
# Generate ground truth for all samples
uv run stt-benchmark ground-truth

# Interactive review with audio playback
uv run stt-benchmark ground-truth review <run_id>
```

### Calculating Semantic WER

```bash
# Calculate for all services
uv run stt-benchmark wer

# Force recalculate
uv run stt-benchmark wer --services deepgram --force-recalculate
```

### Viewing Reports

```bash
# Compare all services
uv run stt-benchmark report

# Detailed report for one service
uv run stt-benchmark report --service deepgram

# Show worst samples
uv run stt-benchmark report --service deepgram --errors 10
```

See [docs/cli.md](docs/cli.md) for complete CLI reference.

## Output Structure

```
stt_benchmark_data/
├── audio/                    # Downloaded audio files
├── results.db                # SQLite database
├── ground_truth_runs/        # Iteration JSONL files
├── validation_summary.txt    # Generated reports
└── validation_full.csv
```

### Database Tables

| Table | Description |
|-------|-------------|
| `samples` | Audio sample metadata |
| `benchmark_results` | TTFB and transcription results |
| `ground_truths` | Reference transcriptions (Gemini) |
| `wer_metrics` | Semantic WER calculations |
| `semantic_wer_traces` | Full Claude reasoning traces |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        PipelineTask                      │
│  observers=[MetricsCollector, TranscriptionCollector]    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│       ┌──────────────────┐    ┌───────────────┐          │
│       │ SyntheticInput   │───▶│  STTService   │          │
│       │ Transport        │    │               │          │
│       │                  │    │ Emits:        │          │
│       │ - Plays audio    │    │ - Transcript  │          │
│       │ - Silero VAD     │    │ - MetricsFrame│          │
│       │ - Real-time pace │    │   (TTFB)      │          │
│       └──────────────────┘    └───────────────┘          │
│                                     │                    │
│                           Observers capture frames       │
└──────────────────────────────────────────────────────────┘
```

## Documentation

- [CLI Reference](docs/cli.md) - Complete command documentation
- [Running Analysis](docs/analysis.md) - Step-by-step analysis guide

## License

MIT
