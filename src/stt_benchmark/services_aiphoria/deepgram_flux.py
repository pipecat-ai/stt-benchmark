"""Pipecat STT service for Deepgram Flux via the production speech-proxy.

Deepgram Flux is Deepgram's conversational turn-detection model, reached through
the SAME staging speech-proxy + platform_proto v2 gRPC API as the other proxy
setups (asr_deepgram_en_nova3 / asr_deepgram_en_nova3_vad_v2). Recognizer id:
`asr_deepgram_flux_en`.

Contract (probed 2026-06-09): growing (cumulative) interims, then a SINGLE
complete `is_final` per turn that self-endpoints ~1 s after speech end. This is the
same single-final contract as nova3, so the parent DeepgramProxySTTService handles
it UNCHANGED (finalize on the first is_final with non-empty text); only the
recognizer differs. Unlike asr_deepgram_en_nova3_vad_v2 (unstable, multi-segment
finals, ~50% no-final/timeout), Flux finalized 99/100 with median 100% transcript
coverage; full-pipeline TTFS p50 ~1079 ms against the shared Silero speech-end
anchor (see the task README "asr_deepgram_flux_en evaluation").
"""

from __future__ import annotations

from stt_benchmark.services_aiphoria.deepgram_proxy import DeepgramProxySTTService

FLUX_RECOGNIZER = "asr_deepgram_flux_en"


class DeepgramFluxSTTService(DeepgramProxySTTService):
    """Deepgram Flux via the production speech-proxy (gRPC v2, TLS).

    Identical transport and single-final client contract as
    DeepgramProxySTTService; only the recognizer is the Flux model.
    """

    def __init__(self, *, recognizer: str = FLUX_RECOGNIZER, **kwargs) -> None:
        super().__init__(recognizer=recognizer, **kwargs)
