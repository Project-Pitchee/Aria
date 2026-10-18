"""Silero VAD wrapper (SSOT §3.1.1 Step 2, Q11-D1).

Only the ONNX detector is supported here; the WebRTC comparison arm and the
Sohn energy fallback belong to the evaluation harness, not this pipeline.

Two deliberate simplifications relative to the upstream Silero demo:

* One threshold (0.5) for both speech start and end, because the SSOT writes a
  single ``speech_prob >= 0.5``.
* No ``speech_pad_ms`` and no ``min_speech_duration`` — the SSOT's only
  boundary rules are the 250 ms whole-recording trim and the 25 ms per-segment
  trim. Adding padding here would silently change T_speech.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

try:
    from . import constants as C
except ImportError:
    import constants as C

DEFAULT_MODEL_PATH = Path(
    "/Users/dannyfeng/Documents/Pitchee/Dependencies/PitcheeCore/models/SileroVAD.onnx")


class ProbabilityModel(Protocol):
    """Minimal contract needed from a detector; makes the merge logic testable."""

    def probabilities(self, pcm: np.ndarray) -> np.ndarray:
        """Per-chunk speech probability for ``len(pcm) / VAD_CHUNK_SAMPLES`` chunks."""


@dataclass(frozen=True)
class VadSegment:
    """Half-open sample interval [start, end) on the VAD input."""
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def merge_segments(segments: Sequence[tuple[int, int]],
                   min_silence_samples: int) -> list[tuple[int, int]]:
    """Merge runs separated by less than ``min_silence_samples``.

    The SSOT merges pauses shorter than 100 ms; a gap exactly equal to the
    threshold is kept as a real pause (strict inequality).
    """
    merged: list[list[int]] = []
    for start, end in segments:
        if end <= start:
            continue
        if merged and start - merged[-1][1] < min_silence_samples:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([int(start), int(end)])
    return [(a, b) for a, b in merged]


def segments_from_probabilities(probabilities: np.ndarray,
                                n_samples: int,
                                chunk: int = C.VAD_CHUNK_SAMPLES,
                                threshold: float = C.VAD_THRESHOLD,
                                min_silence_s: float = C.VAD_MIN_SILENCE_S,
                                sample_rate: int = C.TARGET_SR) -> list[VadSegment]:
    """Turn per-chunk probabilities into sample intervals, then merge pauses."""
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    active = probs >= threshold
    runs: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start * chunk, min((index) * chunk, n_samples)))
            start = None
    if start is not None:
        runs.append((start * chunk, min(len(probs) * chunk, n_samples)))
    merged = merge_segments(runs, int(round(min_silence_s * sample_rate)))
    return [VadSegment(a, b) for a, b in merged if b > a]


class SileroVad:
    """Silero v5 ONNX detector driven one 512-sample chunk at a time."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH,
                 session=None, threads: int = 1):
        self.model_path = Path(model_path)
        if session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"])
        self.session = session
        self.model_sha256 = (hashlib.sha256(self.model_path.read_bytes()).hexdigest()
                             if self.model_path.exists() else "")

    def probabilities(self, pcm: np.ndarray, reset: bool = True) -> np.ndarray:
        """Per-chunk speech probability.

        Silero v5 ONNX expects each 512-sample chunk prefixed by the previous
        64 samples (``VAD_CONTEXT_SAMPLES``), exactly as the upstream
        ``OnnxWrapper`` does. Feeding bare 512-sample chunks runs without error
        but collapses probabilities towards zero on clean speech.
        """
        pcm = np.asarray(pcm, dtype=np.float32)
        chunk = C.VAD_CHUNK_SAMPLES
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros(C.VAD_CONTEXT_SAMPLES, dtype=np.float32)
        sr = np.array(C.TARGET_SR, dtype=np.int64)
        probs = []
        for start in range(0, len(pcm), chunk):
            block = pcm[start:start + chunk]
            if len(block) < chunk:
                block = np.pad(block, (0, chunk - len(block)))
            framed = np.concatenate([context, block]).reshape(1, -1)
            out, state = self.session.run(
                None, {"input": framed, "state": state, "sr": sr})
            probs.append(float(out.reshape(-1)[0]))
            context = block[-C.VAD_CONTEXT_SAMPLES:]
        return np.asarray(probs, dtype=np.float64)

    def segments(self, pcm: np.ndarray) -> list[VadSegment]:
        return segments_from_probabilities(self.probabilities(pcm), len(pcm))
