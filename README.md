# Measuring our (Aiphoria) STT — in-house quickstart

> Local fork of pipecat-ai/stt-benchmark, extended with our deploy stack. This
> top section is the in-house guide for benchmarking **our** ASR; the upstream
> docs follow below.

## What "ours" means here

"Ours" covers **two families** of registered services — both defined in
`src/stt_benchmark/services.py` and registered in `src/stt_benchmark/models.py`.
All report TTFS on the same shared clock, so they're directly comparable.

**A. Direct asr-backend path** — our own ASR stack (Triton `asr-en-ensemble` +
`asr-backend-service` gRPC v2, model `hf/aiphoria_english_asr_v2_160ms` CTC):

| service name | setup | what it measures |
|---|---|---|
| `aiphoria` | 1a | our ASR + our **native in-Triton VAD-EOU** (`max_silence_ms=640`); final on backend `is_final` with `eou_reason` organic/repeating. **True product latency.** |
| `aiphoria_exteou` | 1b | our ASR + the **shared external `eou.py`** (640 ms rule), finalization decided client-side. Read the **1a − 1b gap** as our in-Triton/gRPC EOU overhead. |

**B. Production speech-proxy path** — Deepgram reached **through our** staging
speech-proxy (`speech-proxy.main.stage.aiphoria.pro:443`, same platform_proto v2
gRPC API, TLS). This is the real production integration path and where the
**Deepgram VAD-fix recognizers** are measured. All three use the same
`DeepgramProxySTTService` client (`services_aiphoria/deepgram_proxy.py`); only the
server-side recognizer differs. The proxy finalizes on the **first** `is_final`
with non-empty text (`eou_reason`-agnostic):

| service name | setup | recognizer / what it measures |
|---|---|---|
| `deepgram_proxy` | 4 | `asr_deepgram_en_nova3` — proxy-native Deepgram UtteranceEnd (`utterance_end_ms=1000`). Baseline production path (2026-05-19). |
| `deepgram_proxy_v2` | 4b | same recognizer, re-measured after a reported server-side proxy fix (2026-05-21). Separate name preserves the 4 rows for before/after. |
| `deepgram_proxy_vad_v2` | 4c→refix | `asr_deepgram_en_nova3_vad_v2` — the **faster-VAD-endpointing recognizer** (the fix we track). ⚠️ premature endpointing truncates long turns / fires before the anchor → those turns become *unmeasurable* (see caveat below). |

(For contrast the harness also has the **non-ours** direct-Deepgram-cloud setups
`deepgram` (2), `deepgram_native` (2b), `deepgram_exteou` (3) — those need
`DEEPGRAM_API_KEY`.)

Metric = TTFS (pipecat TTFB instrumentation): final `TranscriptionFrame` receipt
− (Silero VAD stop − `--vad-stop-secs`, default 0.2 s). A final that fires
*before* that anchor is unmeasurable, so premature endpointing shows up as a
*missing* sample, not a fast one. Results go to `stt_benchmark_data/results.db`
(or `test.db` with `--test`). Sample selection (dataset
`pipecat-ai/smart-turn-data-v3.1-train`, seed 42) lives in
`src/stt_benchmark/config.py` (env prefix `STT_BENCHMARK_`, `.env`), **not** as
`run` flags; concurrency is 1.

## Prerequisites

1. **Audio present** — download once into `stt_benchmark_data/audio/`:
   ```bash
   uv run stt-benchmark download --num-samples 100
   ```
2. **Target reachable** — the benchmark is only a gRPC **client**; the target is
   **hardcoded per service**, no env var or CLI flag:
   - *aiphoria / aiphoria_exteou* → `localhost:50052`, `grpc.aio.insecure_channel`,
     `language_id="en"` (`AiphoriaSTTService.__init__`). The `asr-backend-service`
     (fronting Triton) must already be running. **No API key.**
   - *deepgram_proxy\** → `speech-proxy.main.stage.aiphoria.pro:443`,
     `grpc.aio.secure_channel` (TLS), recognizer passed as `language_id`
     (`PROXY_TARGET` / `PROXY_RECOGNIZER` in `deepgram_proxy.py`). The proxy talks
     to Deepgram server-side, so the **client needs no `DEEPGRAM_API_KEY`** — only
     network reach to the proxy.

   To point at a different backend/proxy, recognizer, or toggle TLS, edit the
   `create_*` factory in `src/stt_benchmark/services.py` (or the `PROXY_*`
   constants), not a flag.

## Concrete commands

```bash
# Always run from the harness dir, via its uv env:
cd /home/mle/asr-junk/danil-andreev/tasks/deepgram_vs_ours_latency/stt-benchmark

# --- A. direct asr-backend path (our ASR) ---
# 1. Smoke-test: 2 samples into the throwaway --test DB
uv run stt-benchmark run --services aiphoria --limit 2 --test
# 2. Full product-latency run (Setup 1a)
uv run stt-benchmark run --services aiphoria --limit 100 --no-skip-existing
# 3. Both of ours in one run (comma-separated) -> gives the 1a vs 1b gap
uv run stt-benchmark run --services aiphoria,aiphoria_exteou --limit 100 --no-skip-existing

# --- B. production speech-proxy path (Deepgram + our proxy / VAD-fix recognizers) ---
# 4. Smoke the proxy + the vad_v2 fix recognizer first (cheap, throwaway DB)
uv run stt-benchmark run --services deepgram_proxy_vad_v2 --limit 2 --test
# 5. Full re-measure of the vad_v2 fix (the recognizer we track)
uv run stt-benchmark run --services deepgram_proxy_vad_v2 --limit 100 --no-skip-existing
# 6. Before/after across the proxy recognizers in one shot
uv run stt-benchmark run --services deepgram_proxy,deepgram_proxy_v2,deepgram_proxy_vad_v2 \
    --limit 100 --no-skip-existing

# --- tuning + reporting (apply to either path) ---
# 7. Tighten / loosen the speech-end anchor
uv run stt-benchmark run --services aiphoria --limit 100 --vad-stop-secs 0.3 --no-skip-existing
# 8. Aggregate -> report
uv run stt-benchmark report               # built-in reporter (all services in the DB)
uv run python ../make_report.py           # task's curated TTFS table+json into ../results/
# For the proxy/vad_v2 path also run the validity-aware analysis (kept% + early-final gate):
uv run python ../analyze_vad_v2_refix_20260530.py
```

`run` flags (`src/stt_benchmark/cli/benchmark.py`, registered as `run` in
`cli/main.py`): `--services/-s` (comma-separated names, or `all`), `--limit/-n N`,
`--model/-m`, `--skip-existing/--no-skip-existing` (default **skip**; pass
`--no-skip-existing` to re-measure rows already in the DB), `--vad-stop-secs/-v`
(default 0.2), `--test/-t` (use the separate test DB so you don't touch the real
`results.db`). Those are the **only** `run` flags — seed/dataset come from config.

> ⚠️ **Validity caveat for `deepgram_proxy_vad_v2`.** Its aggressive VAD often
> endpoints before the Silero speech-end anchor and/or truncates long turns, so a
> raw TTFS percentile over the DB is misleading. Gate on transcript completeness
> (kept% vs the full-turn `deepgram_proxy` reference) and drop early/no-TTFB rows
> before reading latency — that's exactly what `../analyze_vad_v2_refix_20260530.py`
> does. See `../README.md` (task root) for the full 2026-05-30 result.

> Task-dir convention: give each real run its own launcher script + log/marker
> (see `../run_*_*.sh`), and for a before/after on the same path, register a **new
> service name** in `models.py` + `services.py` rather than overwriting existing
> rows. Available commands: `run`, `download`, `report`, `ground-truth`, `wer`,
> `export` (the registered service names live in `models.py`).

---

# STT Benchmark

A framework for benchmarking Speech-to-Text services with TTFS (Time To Final Segment) latency and Semantic WER (Word Error Rate) accuracy measurement.

## Results Summary

Benchmark results on 1000 samples from the `pipecat-ai/smart-turn-data-v3.1-train` dataset.

| Service | Transcripts | Perfect | WER Mean | Pooled WER | TTFS Median | TTFS P95 | TTFS P99 |
|---------|-------------|---------|----------|------------|-------------|----------|----------|
| AssemblyAI | 99.8% | 66.8% | 3.49% | 3.02% | 256ms | 362ms | 417ms |
| AWS | 100.0% | 77.4% | 1.68% | 1.75% | 1136ms | 1527ms | 1897ms |
| Azure | 100.0% | 82.9% | 1.21% | 1.18% | 1016ms | 1345ms | 1791ms |
| Cartesia | 99.9% | 60.5% | 3.92% | 4.36% | 266ms | 364ms | 898ms |
| Deepgram | 99.8% | 76.5% | 1.71% | 1.62% | 247ms | 298ms | 326ms |
| Elevenlabs | 99.7% | 81.3% | 3.16% | 3.12% | 281ms | 348ms | 407ms |
| Google | 100.0% | 69.0% | 2.84% | 2.85% | 878ms | 1155ms | 1570ms |
| Mistral | 99.3% | 68.8% | 4.44% | 4.97% | 525ms | 973ms | 1913ms |
| OpenAI | 99.3% | 75.9% | 3.24% | 3.06% | 637ms | 965ms | 1655ms |
| Smallest AI | 100.0% | 72.4% | 2.30% | 2.37% | 398ms | 533ms | 1593ms |
| Soniox | 99.8% | 84.1% | 1.25% | 1.29% | 249ms | 281ms | 310ms |
| Speechmatics | 99.7% | 83.2% | 1.40% | 1.07% | 495ms | 676ms | 736ms |

### Latency vs Accuracy Trade-off

**Typical Latency (Median)**

![STT Service Pareto Frontier - Median](assets/stt_pareto_frontier.png)

**Worst-Case Latency (P95)**

![STT Service Pareto Frontier - P95](assets/stt_pareto_frontier_p95.png)

The Pareto frontier shows services that offer the best trade-off between latency and accuracy—no other service is better on both metrics. Services on the frontier represent efficient choices depending on your priorities.

For production voice agents, **P95 latency matters more than median**. Even occasional high latency (5% of interactions) can break the conversational flow. A service with great median but poor P95 indicates inconsistent performance.

### Metrics Glossary

| Metric | Description |
|--------|-------------|
| **Transcripts** | Percentage of samples where STT successfully returned a transcription |
| **Perfect** | Perfect transcriptions (0% semantic WER) out of total benchmark runs |
| **WER Mean** | Average semantic word error rate across all samples |
| **Pooled WER** | Weighted WER (total errors / total reference words) |
| **TTFS Median** | Median time from user stops speaking to final transcription segment |
| **TTFS P95** | 95th percentile TTFS - worst 5% of samples have latency above this |
| **TTFS P99** | 99th percentile TTFS - worst 1% of samples have latency above this |

> **Semantic WER** measures only transcription errors that would impact an LLM agent's understanding. Punctuation, contractions, filler words, and equivalent phrasings are ignored.

> **TTFS (Time To Final Segment)** is measured from when the user stops speaking to when the final transcription segment is received. For streaming voice agents, lower TTFS means faster response times.

## Measure TTFS for Your Service

If you're using Pipecat and want TTFS latency numbers for your STT service and configuration, see **[Measuring TTFS](docs/measuring-ttfs.md)** for a quick start guide. The P95/P99 values from this tool can be used directly in Pipecat's `ttfs_p99_latency` service configuration (Pipecat 0.0.102+).

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

## How It Works

### TTFS Measurement

**TTFS for STT is different from typical request/response latency.** Since STT services receive continuous audio input, there's no discrete request to measure from. Instead, we measure from when the user **stops speaking** to when the **final transcription** arrives.

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
│         │                              └──── TTFS ────────┼────────►        │
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
- TTFS = final `TranscriptionFrame` receipt time − `speech_end_time`
- Streaming services emit multiple partial transcripts; we use the **final** one

**Why the final transcript?** For LLM/TTS, there's a discrete input→output making latency measurement simple. For streaming STT, audio flows continuously and generates multiple `TranscriptionFrame`s. We can't know when the STT service finalized audio for intermediate transcripts, so we measure from the final one and use the VAD signal to determine when the user actually stopped speaking.

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

`assemblyai`, `aws`, `azure`, `cartesia`, `deepgram`, `deepgram_flux`, `elevenlabs`, `fal`, `gladia`, `google`, `gradium`, `groq`, `nvidia`, `nvidia_sagemaker`, `openai`, `sarvam`, `smallest`, `soniox`, `speechmatics`, `whisper`

See `env.example` for required API keys.

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
| `benchmark_results` | TTFS and transcription results |
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
│       ┌──────────────────┐                               │
│       │ SyntheticInput   │  Plays audio at real-time pace│
│       │ Transport        │                               │
│       └────────┬─────────┘                               │
│                ▼                                         │
│       ┌──────────────────┐                               │
│       │ VADProcessor     │  Emits VAD frames via Silero  │
│       └────────┬─────────┘                               │
│                ▼                                         │
│       ┌──────────────────┐                               │
│       │ STTService       │  Emits transcript + metrics   │
│       └──────────────────┘                               │
│                           Observers capture frames       │
└──────────────────────────────────────────────────────────┘
```

## Dataset

The benchmark dataset (audio samples and ground truth transcriptions) is publicly available on Hugging Face:

**[pipecat-ai/stt-benchmark-data](https://huggingface.co/datasets/pipecat-ai/stt-benchmark-data)**

Audio samples are sourced from the `pipecat-ai/smart-turn-data-v3.1-train` dataset. Ground truth transcriptions are generated with Gemini and human-reviewed.

## Documentation

- [CLI Reference](docs/cli.md) - Complete command documentation
- [Running Analysis](docs/analysis.md) - Step-by-step analysis guide

## License

MIT
