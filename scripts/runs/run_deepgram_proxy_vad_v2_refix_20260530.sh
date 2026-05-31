#!/usr/bin/env bash
# 2026-05-30 re-measure -- run name: dgvs_proxy_vad_v2_refix_100
# Full seed-42 100-sample TTFS re-measure of the CURRENT recognizer
# asr_deepgram_en_nova3_vad_v2 after the 2026-05-30 server-side fix.
# Re-uses the deepgram_proxy_vad_v2 benchmark service (the current service) so no
# harness changes are needed; --no-skip-existing overwrites the 2026-05-21 4c rows,
# so we snapshot results.db first to preserve the 4c raw rows for before/after.
# Same fixed seed-42 100-sample table -> directly comparable to setups 2/2b/1a/1b/3/4/4b.
set -u
TASK=/home/mle/asr-junk/danil-andreev/tasks/deepgram_vs_ours_latency
cd "$TASK/stt-benchmark"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

BK="stt_benchmark_data/results.db.bak_4c_20260521"
if [ ! -f "$BK" ]; then
  cp stt_benchmark_data/results.db "$BK"
  echo "[$(ts)] backed up results.db -> $BK (preserves 4c rows)"
fi

echo "[$(ts)] START dgvs_proxy_vad_v2_refix_100 (deepgram_proxy_vad_v2, 100 samples)"
uv run stt-benchmark run --services deepgram_proxy_vad_v2 --limit 100 --no-skip-existing \
  > "$TASK/run_deepgram_proxy_vad_v2_refix_100.log" 2>&1
rc=$?
echo "[$(ts)] DONE dgvs_proxy_vad_v2_refix_100 rc=$rc"
touch "$TASK/.proxy_vad_v2_refix_run_done"
