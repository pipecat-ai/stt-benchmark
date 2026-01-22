# Measuring TTFS for Your STT Service

TTFS (Time To Final Segment) measures the time from when the user stops speaking to when the final transcription segment arrives. This is the latency that directly impacts how fast your voice agent can respond.

This guide covers running the TTFS benchmark for your own STT service, configuration, and network location. No ground truth or WER evaluation needed, just the latency measurement.

## Quick Start

```bash
# Install
git clone https://github.com/pipecat-ai/stt-benchmark
cd stt-benchmark
uv sync

# Set your API key
cp env.example .env
# Edit .env and add your STT service API key

# Download audio samples
uv run stt-benchmark download --num-samples 100

# Run TTFS benchmark
uv run stt-benchmark run --services deepgram

# View results
uv run stt-benchmark report
```

The report will show TTFS Median, P95, and P99 for your service.

## Using Results with Pipecat

Starting in Pipecat 0.0.102, you can set the `ttfs_p99_latency` argument on your STT service to tell the context aggregator how long to wait for final transcripts. This lets your agent make better decisions about when to proceed to LLM inference versus waiting for additional transcript segments.

```python
stt = DeepgramSTTService(
    api_key=os.getenv("DEEPGRAM_API_KEY"),
    ttfs_p99_latency=0.4,
)
```

## How TTFS Is Measured

For LLM and TTS services, latency is straightforward: discrete input in, output out. Streaming STT works differently, audio flows continuously to the provider, which generates multiple transcription segments over time.

We can't measure when the STT service finalized audio for intermediate transcripts, so we measure from the **last** transcription segment received. The two reference points are:

1. **Speech end time**: The VAD stop-speaking event minus the VAD stop delay — the moment the user actually stopped speaking
2. **Final transcript receipt time**: When the last transcription segment arrives

TTFS = final transcript receipt time − speech end time

See the [README](../README.md#ttfs-measurement) for a detailed diagram.

## Options

### Number of samples

More samples give more stable percentile estimates. For P95/P99 values you'll use in production, we recommend at least 100 samples:

```bash
uv run stt-benchmark download --num-samples 100
uv run stt-benchmark run --services deepgram
```

### VAD stop threshold

The default VAD stop threshold is 0.2 seconds, which works very well in all of our testing and we recommend not modifying the value. Though, if you require a change, you can adjust this to match your production configuration:

```bash
uv run stt-benchmark run --services deepgram --vad-stop-secs 0.3
```

### Multiple services

Compare services side by side:

```bash
uv run stt-benchmark run --services deepgram,soniox,speechmatics
uv run stt-benchmark report
```

### Benchmarking all configured services

Any service with its API key set in `.env` will be included:

```bash
uv run stt-benchmark run --services all
```

## Interpreting Results

The report shows three latency percentiles:

| Metric | What it tells you |
|--------|-------------------|
| **TTFS Median** | Typical latency — what most interactions will experience |
| **TTFS P95** | Worst 5% of interactions have latency above this value |
| **TTFS P99** | Worst 1% of interactions have latency above this value |

For production voice agents, **P95 is the most important number**. It characterizes the worst-case latency your users will regularly encounter. Median describes the average experience, but P95 tells you how long your agent needs to wait to be confident the final transcript has arrived.

Services that support **finalization** (confirming receipt of the final transcript via metadata) can reduce effective latency, since your application gets an explicit signal rather than waiting out a timeout window. See the main README for more on how TTFS measurement works.

## Supported Services

See [`env.example`](../env.example) for the full list of supported services and their required environment variables. Service configurations are defined in [`src/stt_benchmark/services.py`](../src/stt_benchmark/services.py).
