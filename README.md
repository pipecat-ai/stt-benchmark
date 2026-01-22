# STT Benchmark

A framework for benchmarking Speech-to-Text services with TTFB (Time To First Byte) latency and Semantic WER (Word Error Rate) accuracy measurement.

## Results Summary

> TODO: This is a PLACEHOLDER. Replace me!

| Service | Semantic WER | TTFB Median | TTFB P95 | Samples |
|---------|--------------|-------------|----------|---------|
| deepgram | 3.2% | 220ms | 412ms | 500 |
| assemblyai | 4.1% | 324ms | 890ms | 500 |
| groq | 5.1% | 180ms | 340ms | 500 |
| openai | 5.8% | 485ms | 1120ms | 500 |

> **Semantic WER** measures only transcription errors that would impact an LLM agent's understanding. Punctuation, contractions, filler words, and equivalent phrasings are ignored.

> **TTFB** is measured from when the user stops speaking to when the first transcription byte is received. For streaming voice agents, lower TTFB means faster response times.

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
HATHORA_API_KEY=...
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
│ VADUserStartedSpeaking              Actual speech    VADUserStopped        │
│        t=0                            ends           SpeakingFrame         │
│         │                              │                  │                │
│         ▼                              ▼                  ▼                │
│  ═══════╪══════════════════════════════╪══════════════════╪════            │
│         │      Audio streaming to STT  │   VAD stop_secs  │                │
│         │                              │◄────────────────►│                │
│         │                              │                  │                │
│         │                              └──── TTFB ────────┼────────►       │
│         │                           speech_end_time       │     T3         │
│         │                                                 │  (final        │
│         │     T1              T2                          │ transcript)    │
│         │      │               │                          │                │
│         │      ▼               ▼                          │                │
│         │  transcript      transcript                     │                │
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
| `hathora` | nvidia-parakeet-tdt-0.6b-v3 | `HATHORA_API_KEY` |
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

# Edit a specific sample's ground truth
uv run stt-benchmark ground-truth edit <sample_id> --text "corrected text"
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
