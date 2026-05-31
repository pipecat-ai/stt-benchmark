#!/usr/bin/env bash
# 2026-05-30 rerun -- run name: dgvs_proxy_vad_v2_rerun_20260530_142747_100
# Full seed-42 100-sample TTFS run of the current speech-proxy recognizer
# asr_deepgram_en_nova3_vad_v2 via the deepgram_proxy_vad_v2 benchmark service.
# This intentionally reuses the existing service name, so --no-skip-existing
# overwrites current deepgram_proxy_vad_v2 rows in results.db. The script snapshots
# results.db first, then runs the validity-aware vad_v2 analysis.
set -u

TASK=/home/mle/asr-junk/danil-andreev/tasks/deepgram_vs_ours_latency
HARNESS="$TASK/stt-benchmark"
RUN_NAME=dgvs_proxy_vad_v2_rerun_20260530_142747_100
LOG="$TASK/run_deepgram_proxy_vad_v2_rerun_20260530_142747_100.log"
ANALYSIS_LOG="$TASK/analyze_vad_v2_rerun_20260530_142747.log"
MARKER="$TASK/.proxy_vad_v2_rerun_20260530_142747_done"
BK="stt_benchmark_data/results.db.bak_before_${RUN_NAME}"

cd "$HARNESS"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

if [ ! -f "$BK" ]; then
  cp stt_benchmark_data/results.db "$BK"
  echo "[$(ts)] backed up results.db -> $BK"
fi

echo "[$(ts)] START $RUN_NAME (deepgram_proxy_vad_v2, 100 samples)"
uv run stt-benchmark run --services deepgram_proxy_vad_v2 --limit 100 --no-skip-existing \
  > "$LOG" 2>&1
rc=$?
echo "[$(ts)] DONE $RUN_NAME rc=$rc"

if [ "$rc" -eq 0 ]; then
  echo "[$(ts)] START analyze_vad_v2_refix_20260530.py for $RUN_NAME"
  uv run python ../analyze_vad_v2_refix_20260530.py > "$ANALYSIS_LOG" 2>&1
  analysis_rc=$?
  echo "[$(ts)] DONE analysis rc=$analysis_rc log=$ANALYSIS_LOG"
else
  analysis_rc=1
  echo "[$(ts)] SKIP analysis because benchmark failed"
fi

touch "$MARKER"
exit "$rc"
