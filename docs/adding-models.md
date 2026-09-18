# Adding Models

This benchmark treats **one (vendor, model) pair as one "service" entry**. A vendor
with several models has one entry per model — for example `cartesia` (Ink-Whisper)
and `cartesia_ink2` (Ink-2), or `assemblyai` (Universal-Streaming) and
`assemblyai_universal_3_6_pro` (Universal-3.6 Pro).

## Why we keep old models instead of overwriting

When a vendor ships a new model, **add a new entry — do not edit the existing one
in place.** Editing in place silently swaps the number under a single label and
throws away a result you can't reproduce later. Keeping both means:

- Every published number traces to a specific model.
- Readers can see how a vendor changed from one release to the next.
- The old config stays runnable, so its numbers remain reproducible.

This is the opposite of a "best model per vendor" table, where only the top number
survives and you can't tell which model produced it.

## Naming convention

- The vendor's **first/original** model keeps the bare vendor key: `cartesia`,
  `assemblyai`. These are never renamed.
- Every **new** model (going forward) uses a full `vendor_model` key derived from
  the model string, so the key is unambiguous on its own — e.g. model
  `universal-3-6-pro` → `assemblyai_universal_3_6_pro`. Replace characters that
  aren't valid in identifiers (`-`, `.`, `:`) with `_`.
- Existing shorter keys from before this convention (e.g. `cartesia_ink2`) are
  left as-is rather than backdated.
- Mark the superseded entry `is_current=False`. The newest stays `is_current=True`
  (the default).

## Choosing `model_label`

`model_label` records exactly what was benchmarked, so it's the literal model
string — not a marketing name. Pick it in this order:

1. **The factory pins a model string** → use that string verbatim: Deepgram
   `nova-3-general`, Cartesia `ink-2`, AssemblyAI `universal-3-6-pro`. This
   keeps the label reproducible and unambiguous (it's the value you'd pass to
   re-run it).
2. **The vendor exposes no selectable model** (AWS, Azure, fal, Speechmatics, …)
   → use `N/A`. These vendors serve one engine with no model string to pin; you'd
   only add a second entry once they ship a *named/versioned* model (e.g.
   `aws_<newmodel>`), leaving the bare entry as the `N/A` baseline.

`model_label` is always a non-empty string — use the model string or `N/A`, never
`None` or an empty value. It's never a problem in the plots: `get_display_names`
only appends the model label when a vendor has two or more *plotted* models, so a
single-model vendor renders as just "AWS" / "Azure" (the `N/A` is never shown).
The label's job is the generated README Model column and in-code self-description.

## Checklist: add a new model for an existing vendor

1. **`src/stt_benchmark/services.py`** — add a factory function returning the
   configured service, e.g. `create_assemblyai_universal_3_6_pro()`. Add a short
   comment noting which model it is and that it supersedes the previous one.
2. **`src/stt_benchmark/services.py`** — add an entry to `STT_SERVICES` with
   `vendor`, `model_label`, `required_env_vars`, and `is_current`. Set the
   *previous* model's entry to `is_current=False`.
3. **`src/stt_benchmark/models.py`** — add a `ServiceName` enum value matching the
   new registry key
   (e.g. `ASSEMBLYAI_UNIVERSAL_3_6_PRO = "assemblyai_universal_3_6_pro"`).
4. **`README.md`** — nothing to hand-edit. Do **not** hand-type the Results Summary row
   — step 6 writes it from your benchmark database between the `RESULTS_TABLE` markers.
5. **Run the benchmark** for the new entry:
   ```bash
   uv run stt-benchmark run --services assemblyai_universal_3_6_pro
   uv run stt-benchmark ground-truth   # if not already generated for these samples
   uv run stt-benchmark wer --services assemblyai_universal_3_6_pro
   ```
6. **Add your row to the README table** — a *targeted* upsert that reads only this
   service's metrics from your local database and inserts/replaces just its row,
   leaving every other vendor's row untouched (Vendor/Model and number formatting
   come from the registry):
   ```bash
   uv run stt-benchmark update-readme --services assemblyai_universal_3_6_pro
   ```
7. **Regenerate the plots from the README** — the README table is the single
   source of truth; this reads it and never rewrites it:
   ```bash
   uv run python scripts/pareto-frontier-plot.py
   ```
8. **Commit** the changed `README.md` and `assets/*.png` and open a PR. The diff
   is your one new row plus the regenerated charts.

## Checklist: add a brand-new vendor

Same as above, but the key is just the vendor name (no suffix) and
`is_current=True`.

## Why the README table is the single source of truth

More than one person contributes results, and each contributor only has raw
benchmark runs for the vendors *they* ran locally. So the published numbers live
in the **README Results Summary table** (between the `<!-- RESULTS_TABLE:START -->`
/ `<!-- RESULTS_TABLE:END -->` markers), and everyone updates it the same way:

- `stt-benchmark update-readme --services <key>` upserts **one row at a time**
  from your local database — a targeted insert/replace keyed by
  `(vendor, model_label)`. It never rebuilds the whole table, so it can't wipe a
  row you don't have locally. Rows for services with no registry entry are
  preserved too.
- `scripts/pareto-frontier-plot.py` **reads** the table and renders every row
  into the Pareto charts. It is read-only with respect to the README, so the
  charts always match the published numbers and regenerating them is safe.

## Where the metadata lives

- `vendor` / `model_label` / `is_current` live on `ServiceDefinition` in
  `services.py`, so each entry is self-describing. `update-readme` uses them for
  the Vendor/Model columns and the table sort order (vendor, current model first).
  Plot labels are derived from `vendor` / `model_label` for the chart legend.
- `scripts/plot-config.json` holds optional plot tunables (`latency`, `output`,
  and per-label `display_names` overrides). It no longer selects which models
  appear — the plot renders whatever rows are in the README table. Use
  `pareto-frontier-plot.py -s <label> ...` for an ad-hoc subset.
