#!/usr/bin/env bash
# Setup 4b launcher -- run name: dgvs_proxy_v2_100
# Re-measure of the production speech-proxy path (gRPC v2, TLS, recognizer
# asr_deepgram_en_nova3) AFTER a reported server-side fix on the proxy. Same
# DeepgramProxySTTService client (change is server-side), same fixed seed-42
# 100-sample table as setups 2/2b/1a/1b/3/4 -> directly comparable.
# Service name deepgram_proxy_v2 so the pre-fix (2026-05-19) deepgram_proxy rows
# are PRESERVED for a before/after diff. NO wipe; --no-skip-existing only touches
# deepgram_proxy_v2 rows.
set -u
cd /home/mle/asr-junk/danil-andreev/tasks/deepgram_vs_ours_latency/stt-benchmark
LOG=/home/mle/asr-junk/danil-andreev/tasks/deepgram_vs_ours_latency
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] START dgvs_proxy_v2_100 (deepgram_proxy_v2, 100 samples)"
uv run stt-benchmark run --services deepgram_proxy_v2 --limit 100 --no-skip-existing \
  > "$LOG/run_deepgram_proxy_v2_100.log" 2>&1
rc=$?
echo "[$(ts)] DONE dgvs_proxy_v2_100 rc=$rc"
touch "$LOG/.proxy_v2_run_done"
