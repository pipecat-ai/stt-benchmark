#!/usr/bin/env bash
# Setup 4c launcher -- run name: dgvs_proxy_vad_v2_100
# Speech-proxy path with the NEW recognizer asr_deepgram_en_nova3_vad_v2 -- the
# actual server-side fix (faster VAD endpointing; smoke: final ~70ms after
# speech end vs ~1.37s on the old recognizer). Same DeepgramProxySTTService
# client, same fixed seed-42 100-sample table -> directly comparable to
# setups 2/2b/1a/1b/3/4/4b. Distinct service name preserves all prior rows.
# NO wipe; --no-skip-existing only touches deepgram_proxy_vad_v2 rows.
set -u
cd /home/mle/asr-junk/danil-andreev/tasks/deepgram_vs_ours_latency/stt-benchmark
LOG=/home/mle/asr-junk/danil-andreev/tasks/deepgram_vs_ours_latency
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] START dgvs_proxy_vad_v2_100 (deepgram_proxy_vad_v2, 100 samples)"
uv run stt-benchmark run --services deepgram_proxy_vad_v2 --limit 100 --no-skip-existing \
  > "$LOG/run_deepgram_proxy_vad_v2_100.log" 2>&1
rc=$?
echo "[$(ts)] DONE dgvs_proxy_vad_v2_100 rc=$rc"
touch "$LOG/.proxy_vad_v2_run_done"
