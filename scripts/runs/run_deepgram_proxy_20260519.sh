#!/usr/bin/env bash
# Setup 4 launcher -- run name: dgvs_proxy_100
# Deepgram via the production speech-proxy (gRPC v2, TLS,
# recognizer asr_deepgram_en_nova3, UtteranceEnd 1000ms). 100 samples, same
# fixed seed-42 sample table as setups 2/2b/1a/1b/3 -> directly comparable.
# NO wipe: only re-runs the deepgram_proxy rows, every other setup is preserved.
set -u
cd /home/mle/asr-junk/danil-andreev/tasks/deepgram_vs_ours_latency/stt-benchmark
LOG=/home/mle/asr-junk/danil-andreev/tasks/deepgram_vs_ours_latency
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] START dgvs_proxy_100 (deepgram_proxy, 100 samples)"
uv run stt-benchmark run --services deepgram_proxy --limit 100 --no-skip-existing \
  > "$LOG/run_deepgram_proxy_100.log" 2>&1
rc=$?
echo "[$(ts)] DONE dgvs_proxy_100 rc=$rc"
touch "$LOG/.proxy_run_done"
