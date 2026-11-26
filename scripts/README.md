# Scripts

## Pareto Frontier Plot

`pareto-frontier-plot.py` generates the README's latency/accuracy charts: scatter plots of TTFS latency vs Semantic WER on log axes with a Pareto frontier overlay, and a per-service latency distribution (median/P95/P99) chart.

### Usage

```bash
# Plot all services using defaults
python scripts/pareto-frontier-plot.py

# Plot specific services
python scripts/pareto-frontier-plot.py -s deepgram assemblyai soniox

# Use a config file
python scripts/pareto-frontier-plot.py -c scripts/plot-config.json

# Config file with CLI overrides
python scripts/pareto-frontier-plot.py -c scripts/plot-config.json --latency p95
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o`, `--output` | Output file path or directory | `assets/` |
| `--charts` | Chart types to generate (space-separated): `pareto`, `range` | `pareto range` |
| `-l`, `--latency` | Latency metrics for the Pareto charts (space-separated): `median`, `p95`, `p99` | `median` |
| `-s`, `--services` | Services to include (space-separated) | all available |
| `-c`, `--config` | Path to a JSON config file | none |
| `--show` | Display the plot interactively | off |

CLI arguments always take precedence over config file values.

### Config File

A JSON file that stores plot settings for repeatable generation. See `plot-config.json` for a working example.

```json
{
  "services": ["deepgram", "assemblyai", "soniox"],
  "display_names": {
    "deepgram": "Deepgram",
    "assemblyai": "AssemblyAI",
    "soniox": "Soniox"
  },
  "latency": ["median", "p95"],
  "output": "assets/",
  "show": false
}
```

| Key | Type | Description |
|-----|------|-------------|
| `services` | list of strings | Which services to include in the plot |
| `display_names` | dict | Maps service keys to display labels on the plot |
| `charts` | list of strings | Chart types to generate: `["pareto", "range"]` |
| `latency` | string or list | Latency metrics for the Pareto charts: `"p95"` or `["median", "p95"]` |
| `output` | string | Output file path or directory |
| `show` | boolean | Display the plot interactively |
| `label_offsets` | dict | Per-metric hand-placed label positions for dense regions, e.g. `{"median": {"Deepgram": [8, 2, "left"]}}` — `[dx, dy, alignment]` with offsets in points from the dot. Overrides the script's built-in defaults for that metric; labels not listed are placed automatically. |

All keys are optional. Omitted keys fall back to defaults.

## Results Snapshot

`snapshot_results.py` records the README's current results as a dated snapshot under `results/`, so results can be compared over time. Run it after `stt-benchmark update-readme` and `pareto-frontier-plot.py`. It writes `results/<label>/summary.csv`, `run.json` and copies of the two charts, and rebuilds `results/history.csv` from every snapshot. Running it again with the same label replaces that snapshot.

```bash
uv run python scripts/snapshot_results.py 2026-11 \
    --measured-from 2026-11-23 --measured-to 2026-11-26 \
    --region us-west-2 --instance-type c7i.4xlarge
```

| Flag | Description | Default |
|------|-------------|---------|
| `label` | Snapshot name, such as `2026-11` | required |
| `--measured-from`, `--measured-to` | First and last day of measurement (UTC) | required |
| `--region` | Where the benchmark ran from | required |
| `--instance-type` | Machine the benchmark ran on | none |
| `--pipecat-version` | Pipecat version measured with | the one installed |
| `--samples` | Samples per service | `1000` |
| `--notes` | A caveat to record in `run.json`; repeatable | none |

## Judge Experiments

Two scripts measure a semantic WER judge before the benchmark is re-scored with it. Neither writes to the results database: each logs every judgment to `runs.jsonl` in its output directory (under `stt_benchmark_data/judge_experiments/` by default), and re-running the same command resumes from the log. Judges are given as `model:effort`, where an effort of `none` runs without thinking at temperature 0, the setup of the Claude Sonnet 4.5 judge.

### Consistency

`judge_consistency.py` judges a set of transcriptions several times with each judge. Half are transcriptions the stored judge found errors in. It reports how often repeated runs agree, how much judge noise that adds to a 1,000-sample mean WER, and how closely each judge matches the stored scores.

```bash
# 100 transcriptions, 3 runs each: Sonnet 4.5 (the stored judge), Sonnet 5.5 at medium and high
uv run python scripts/judge_consistency.py

uv run python scripts/judge_consistency.py --pairs 200 --repeats 5 --judges claude-sonnet-5-5:medium
```

| Flag | Description | Default |
|------|-------------|---------|
| `--pairs` | Transcriptions to judge | `100` |
| `--repeats` | Runs per transcription | `3` |
| `--judges` | Judges to compare (space-separated `model:effort`) | Sonnet 4.5, Sonnet 5.5 `medium` and `high` |
| `--seed` | Seed for picking transcriptions | `0` |
| `--test` | Use `test_results.db` | off |
| `--output-dir` | Where to write runs and the summary | `judge_experiments/consistency_*` |

### Impact

`judge_impact.py` re-judges every transcription for the chosen services and compares the result with the stored scores: mean and pooled WER per service with a 95% confidence interval on the change, the ranking, and the gaps between neighbouring services. It writes the largest disagreements to `disagreements.md` for manual review.

```bash
uv run python scripts/judge_impact.py --services meta,azure,soniox,deepgram,mistral

# Trial run on the first 10 transcriptions per service
uv run python scripts/judge_impact.py --services meta,azure --limit 10

# After a re-score: compare a copy of the database saved beforehand with the current scores (no API calls)
uv run python scripts/judge_impact.py --services meta,azure --baseline stt_benchmark_data/results_backup.db
```

| Flag | Description | Default |
|------|-------------|---------|
| `--services` | Comma-separated services to re-judge | required |
| `--judge` | New judge as `model:effort` | `claude-sonnet-5-5:medium` |
| `--review` | Disagreements to write out | `40` |
| `--limit` | Judge only the first N transcriptions per service | all |
| `--baseline` | Compare this database copy's scores with the current ones instead of judging | none |
| `--test` | Use `test_results.db` | off |
| `--output-dir` | Where to write runs and the report | `judge_experiments/impact_*` |
