uv run stt-benchmark run --services deepgram_native --chunk-ms 260 --test --limit 100 --sample-rate 16000 --model direct_deepgram_nova3_2026_06_13

uv run stt-benchmark run --services deepgram_flux --chunk-ms 260 --test --limit 100 --sample-rate 16000 --model direct_deepgram_flux_2026_06_13

uv run stt-benchmark run --services deepgram_flux --flux-aggressive-eou --chunk-ms 260 --test --limit 100 --sample-rate 16000 --model direct_deepgram_flux_eager_2026_06_14

uv run stt-benchmark run --services deepgram_native --chunk-ms 260 --test --limit 100 --sample-rate 8000 --model direct_deepgram_nova3_8khz_2026_06_13

uv run stt-benchmark run --services deepgram_flux --chunk-ms 260 --test --limit 100 --sample-rate 8000 --model direct_deepgram_flux_8khz_2026_06_13

uv run stt-benchmark run --services deepgram_flux --flux-aggressive-eou --chunk-ms 260 --test --limit 100 --sample-rate 8000 --model direct_deepgram_flux_8khz_eager_2026_06_14

uv run stt-benchmark run --services deepgram_native --chunk-ms 80 --test --limit 100 --sample-rate 8000 --model direct_deepgram_nova3_8khz_80ms2026_06_13

uv run stt-benchmark run --services deepgram_flux --chunk-ms 80 --test --limit 100 --sample-rate 8000 --model direct_deepgram_flux_8khz_80ms_2026_06_13

uv run stt-benchmark run --services deepgram_flux --flux-aggressive-eou --chunk-ms 80 --test --limit 100 --sample-rate 8000 --model direct_deepgram_flux_8khz_80ms_eager_2026_06_14
