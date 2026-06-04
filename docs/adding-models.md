# Adding Models

This benchmark treats **one (vendor, model) pair as one "service" entry**. A vendor
with several models has one entry per model — for example `cartesia` (Ink-Whisper)
and `cartesia_ink2` (Ink-2), or `assemblyai` (Universal-Streaming) and
`assemblyai_u3_rt_pro` (Universal-3 RT Pro).

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
  the model string, so the key is unambiguous on its own — e.g. model `u3-rt-pro`
  → `assemblyai_u3_rt_pro`. Replace characters that aren't valid in identifiers
  (`-`, `.`, `:`) with `_`.
- Existing shorter keys from before this convention (e.g. `cartesia_ink2`) are
  left as-is rather than backdated.
- Mark the superseded entry `is_current=False`. The newest stays `is_current=True`
  (the default).

## Choosing `model_label`

`model_label` records exactly what was benchmarked, so it's the literal model
string — not a marketing name. Pick it in this order:

1. **The factory pins a model string** → use that string verbatim: Deepgram
   `nova-3-general`, Cartesia `ink-2`, AssemblyAI `u3-rt-pro`. This keeps the
   label reproducible and unambiguous (it's the value you'd pass to re-run it).
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
   configured service, e.g. `create_assemblyai_u3_rt_pro()`. Add a short comment noting
   which model it is and that it supersedes the previous one.
2. **`src/stt_benchmark/services.py`** — add an entry to `STT_SERVICES` with
   `vendor`, `model_label`, `required_env_vars`, and `is_current`. Set the
   *previous* model's entry to `is_current=False`.
3. **`src/stt_benchmark/models.py`** — add a `ServiceName` enum value matching the
   new registry key (e.g. `ASSEMBLYAI_U3_RT_PRO = "assemblyai_u3_rt_pro"`).
4. **`scripts/plot-config.json`** — add the new key to the `services` list to
   feature it in the published plots and the generated README table. Labels are
   derived automatically from `vendor` / `model_label`, so there is no name map to
   update (add an optional `display_names` override only to override a derived
   label). Keep the superseded key in the list too if you want both in the plot.
5. **`README.md`** — add the new key to the Supported Services list. You do **not**
   edit the Results Summary table by hand — it's generated between the
   `RESULTS_TABLE` markers in step 7.
6. **Run the benchmark** for the new entry:
   ```bash
   uv run stt-benchmark run --services assemblyai_u3_rt_pro
   uv run stt-benchmark ground-truth   # if not already generated for these samples
   uv run stt-benchmark wer --services assemblyai_u3_rt_pro
   ```
7. **Regenerate plots + README table** (one command does both — the table is
   rebuilt from the registry + database for the plot-config services, and written
   between the `RESULTS_TABLE` markers in `README.md`):
   ```bash
   uv run python scripts/pareto-frontier-plot.py --config scripts/plot-config.json
   ```

## Checklist: add a brand-new vendor

Same as above, but the key is just the vendor name (no suffix), `is_current=True`,
and you add it to `plot-config.json` rather than swapping an existing key.

## Where the metadata lives

- `vendor` / `model_label` / `is_current` live on `ServiceDefinition` in
  `services.py`, so each entry is self-describing. Plot/report labels are derived
  from these via `get_display_names()` — there is no separate name map to maintain.
- `plot-config.json` decides **which** models appear in the published plots (its
  `services` list), plus optional per-service `display_names` overrides. Keeping a
  model out of the plots does not remove it from the registry — it stays runnable.
- The README **Results Summary table** is generated, not hand-edited:
  `scripts/pareto-frontier-plot.py` rebuilds it from the registry (Vendor =
  `vendor`, Model = `model_label`) and the results database for the same
  plot-config services, and writes it between the `<!-- RESULTS_TABLE:START -->` /
  `<!-- RESULTS_TABLE:END -->` markers in `README.md`. If those markers are
  missing it falls back to writing `assets/results-table.md`.
