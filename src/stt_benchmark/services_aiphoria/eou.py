"""Shared external Silero-VAD end-of-utterance detector.

Faithful re-implementation of our in-Triton VAD-EOU
(triton-setups templates/ctc-conformer-streaming/v4: silero-vad/1/model.py +
postprocessing/1/parts/state.py VadState + model.py _check_eou VAD path) so that
benchmark setups 1b (our ASR + external EOU) and 3 (Deepgram ASR + external EOU)
use byte-identical EOU logic, differing only by the ASR engine.

Replicated exactly:
  * Silero v5 ONNX, 16 kHz, 512-sample windows, 64-sample context, state (2,1,128).
  * VadState: max_silence_frames = max(1, round(max_silence_ms / 32)),
    activation/deactivation threshold 0.5, rolling cache pre-filled with ones,
    speech latch; EOU = speech-started AND last `max_silence_frames` window
    probabilities all < 0.5.
Deliberate, semantics-preserving adaptation: VadState is updated one Silero
window at a time (continuous-time realization) instead of once per ASR chunk.
This is exactly equivalent to the product behaviour (EOU the first instant the
trailing 640 ms are all silence after speech) and avoids the chunk-batching
artifact; it is identical for 1b and 3 because the audio is identical.
The non-empty-hypothesis gate and the once-per-utterance latch are applied by
the calling STT service (mirroring postprocessing model.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime

SILERO_PATH = str(Path(__file__).with_name("silero_vad.onnx"))

SAMPLE_RATE = 16000
WINDOW = 512  # silero v5 @ 16 kHz
CONTEXT = 64
VAD_FRAME_MS = 32  # 512 samples @ 16 kHz (matches Triton model.py comment)
THRESHOLD = 0.5  # activation == deactivation == 0.5 (state.py VadState defaults)


class _VadState:
    """Verbatim port of triton-setups v4 postprocessing parts/state.py VadState."""

    def __init__(self, max_silence_frames: int) -> None:
        self._max_size = max_silence_frames
        self._cache = np.ones([self._max_size], dtype=np.float32)
        self._speech = False

    def update(self, new_probs: np.ndarray) -> bool:
        new_frames = new_probs.shape[-1]
        preserve = self._max_size - new_frames
        if preserve > 0:
            self._cache[:preserve] = self._cache[-preserve:]
        self._cache[-new_frames:] = new_probs[0]

        if not self._speech and (new_probs > THRESHOLD).any():
            self._speech = True
            return False
        if self._speech and (self._cache < THRESHOLD).all():
            self._speech = False
            return True
        return False


class ExternalVadEou:
    """Streaming external VAD-EOU. Feed PCM16 bytes; returns True once when the
    VAD silence criterion fires (caller applies the non-empty-text gate)."""

    def __init__(self, max_silence_ms: int = 640) -> None:
        max_silence_frames = max(1, round(max_silence_ms / VAD_FRAME_MS))  # 640 -> 20
        self._vad = _VadState(max_silence_frames)
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._sess = onnxruntime.InferenceSession(
            SILERO_PATH, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)
        self._buf = np.zeros((0,), dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self._fired = False

    def feed(self, pcm_s16: bytes) -> bool:
        """Process audio; return True the first time VAD-EOU fires."""
        if self._fired or not pcm_s16:
            return False
        samples = np.frombuffer(pcm_s16, dtype=np.int16).astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, samples])
        while self._buf.shape[0] >= WINDOW:
            sub = self._buf[:WINDOW]
            self._buf = self._buf[WINDOW:]
            inp = np.concatenate([self._context, sub[None, :]], axis=1)
            prob, state_n = self._sess.run(
                None, {"input": inp, "state": self._state, "sr": self._sr}
            )
            self._state = state_n
            self._context = sub[None, -CONTEXT:]
            p = np.asarray(prob, dtype=np.float32).reshape(1, 1)
            if self._vad.update(p):
                self._fired = True
                return True
        return False

    @property
    def fired(self) -> bool:
        return self._fired
