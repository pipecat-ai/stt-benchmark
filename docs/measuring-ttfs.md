# Measuring TTFS for Your STT Service

TTFS (Time To Final Segment) measures the time from when the user stops speaking to when the last transcription segment for that turn arrives: the moment your agent has the complete transcript. This is the latency that directly impacts how fast your voice agent can respond.

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

An agent that replied after an earlier segment would be answering part of what the user said, so we measure to the **last** transcription segment received for the turn. The two reference points are:

1. **Speech end time**: The VAD stop-speaking event minus the VAD stop delay — the moment the user actually stopped speaking
2. **Final transcript receipt time**: When the last transcription segment arrives

TTFS = final transcript receipt time − speech end time

If the service signals that the transcript is complete, the last segment is the one it marks as final. If it doesn't, the last segment is the latest one to arrive within 2 seconds of the VAD reporting the end of speech. See the [README](../README.md#ttfs-measurement) for a detailed diagram.

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

### Running faster

Samples run one at a time by default, so that nothing else on the machine affects the latency being measured. To get through a large run faster, run several at once:

```bash
uv run stt-benchmark run --services deepgram --concurrency 8
```

What your machine can take depends on its CPU, so if you plan to use the numbers, compare a small run at `--concurrency 1` with one at your chosen level first. See [Concurrency](cli.md#concurrency) for details.

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

Which number matters most depends on whether your service signals that a transcript is complete:

- **With a completion signal** (finalization), your agent moves on as soon as the finalized transcript arrives. Users experience the TTFS of each turn: typically the median, occasionally the P95 or P99.
- **Without one**, your agent can't tell whether more text is coming, so it waits a fixed time on every turn. That time is the `ttfs_p99_latency` you configure, so users experience roughly the P99 on every turn, whatever the median is.

Either way, set `ttfs_p99_latency` from the P99 this tool reports: it is how long the agent waits when no finalized transcript arrives. See the main README for more on [how TTFS measurement works](../README.md#ttfs-measurement).

## Supported Services

See [`env.example`](../env.example) for the full list of supported services and their required environment variables. Service configurations are defined in [`src/stt_benchmark/services.py`](../src/stt_benchmark/services.py).
