uv run stt-benchmark run --services asr_backend --asr-backend-url asr.prod.overtime.ai:443 --asr-backend-use-ssl --language en --chunk-ms 260 --test --limit 100 --no-skip-existing --model prod_overtime_asr_backend_en_2026_06_13

uv run stt-benchmark run --services speech_proxy --speech-proxy-url speech-proxy.prod.overtime.ai:443 --speech-proxy-use-ssl --recognizer aiphoria__en --chunk-ms 260 --test --limit 100 --no-skip-existing --model prod_overtime_speech_proxy_aiphoria_en_2026_06_13

uv run stt-benchmark run --services speech_proxy --speech-proxy-url speech-proxy.main.stage.aiphoria.pro:443 --speech-proxy-use-ssl --recognizer asr_deepgram_en_nova3_vad_v2 --chunk-ms 260 --test --limit 100 --no-skip-existing --model main_stage_speech_proxy_deepgram_en_nova3_vad_v2_2026_06_13

uv run stt-benchmark run --services speech_proxy --speech-proxy-url speech-proxy.main.stage.aiphoria.pro:443 --speech-proxy-use-ssl --recognizer asr_deepgram_flux_en --chunk-ms 260 --test --limit 100 --no-skip-existing --model main_stage_speech_proxy_deepgram_flux_en_2026_06_13