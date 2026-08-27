# Importing Ground Truth

Ground truth transcriptions ship with this repo in [`ground_truth/`](ground_truth/), so
importing them takes one command after cloning. (The generated data under
`stt_benchmark_data/` — audio, databases — is gitignored and stays local.)

The checked-in run covers 5000 samples:

| File | Contents |
|------|----------|
| `2026-01-03_17-00-06.jsonl` | Header + 5000 Gemini transcriptions (`gemini-3-flash-preview`) |
| `2026-01-03_17-00-06_notes.jsonl` | Human review of the first 1000 samples, including 171 corrections |

Keep the two files side by side and keep their names — the importer locates the
`_notes.jsonl` file by name and applies those corrections automatically.

Benchmark runs use 1000 samples, which is also exactly the reviewed portion of this run —
so at the standard size every imported transcription has been human-checked. The remaining
4000 are raw Gemini output, available if you download a larger set.

## Prerequisite: matching sample IDs

Import matches records to your local database by `sample_id`, which is derived from the
sample's index in the shuffled HuggingFace dataset (see `generate_sample_id()` in
`src/stt_benchmark/dataset/downloader.py`). Any ID not already in your database is skipped.

To line up with this run, download with the defaults — dataset
`pipecat-ai/smart-turn-data-v3.1-train`, seed `42`, offset `0`. Samples are drawn in a fixed
shuffled order, so 1000 samples gives you the first 1000 records of the run and the rest are
skipped on import.

## Steps

```bash
# 1. Install dependencies
uv sync

# 2. Download the standard 1000 samples (seed 42 is the default)
uv run stt-benchmark download --num-samples 1000

# 3. Import — no copying needed, point straight at the checked-in run
uv run stt-benchmark ground-truth import ground_truth/2026-01-03_17-00-06.jsonl
```

Import prints a summary like:

```
Found 171 human corrections in notes file
Model: gemini-3-flash-preview
Total samples in file: 5000
✓ Imported 1000 ground truth transcriptions
  (171 with human corrections)
Skipped 4000 (sample not in database)

Ground truth coverage: 1000/1000 samples
```

The 4000 skipped records are the samples you didn't download — expected at this size. But if
coverage comes back well below 1000, the local download didn't line up: check the sample
count, seed, and offset used in step 2.

### Options

| Flag | Effect |
|------|--------|
| `--force` | Overwrite ground truth entries that already exist |
| `--test` | Import into `test_results.db` instead of the main database |

## Reviewing or correcting the transcriptions

The `list` and `review` commands only scan `stt_benchmark_data/ground_truth_runs/`, so copy
the pair there first if you want to listen to audio and edit transcriptions:

```bash
mkdir -p stt_benchmark_data/ground_truth_runs
cp ground_truth/2026-01-03_17-00-06*.jsonl stt_benchmark_data/ground_truth_runs/

uv run stt-benchmark ground-truth list
uv run stt-benchmark ground-truth review 2026-01-03_17-00-06
```

Review appends your edits to the notes file in that directory. Re-import with `--force` to
apply them, and copy the updated notes file back into `ground_truth/` if you want to share
the corrections.

## Next steps

```bash
# Run benchmarks against the STT services you have keys for
uv run stt-benchmark run --services deepgram,assemblyai

# Score them against the imported ground truth
uv run stt-benchmark wer

# View results
uv run stt-benchmark report
```

See [docs/cli.md](docs/cli.md) for the full command reference.
