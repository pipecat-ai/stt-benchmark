# Scripts

## Pareto Frontier Plot

`pareto-frontier-plot.py` generates scatter plots of TTFS latency vs Semantic WER with a Pareto frontier overlay.

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
| `-l`, `--latency` | Latency metrics to plot (space-separated): `median`, `p95`, `p99` | `median p95` |
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
| `latency` | string or list | Latency metrics to plot: `"p95"` or `["median", "p95"]` |
| `output` | string | Output file path or directory |
| `show` | boolean | Display the plot interactively |

All keys are optional. Omitted keys fall back to defaults.
