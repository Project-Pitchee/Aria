"""Two-level boundary trim, periodic voiced confirmation and segment A.

SSOT §3.1.1, implemented in the order the spec writes it:

  Step 1  trim 250 ms from the head and tail of the whole recording
  Step 2  Silero VAD on the trimmed signal, merge pauses < 100 ms
  Step 3  trim 25 ms from the head and tail of each surviving VAD sub-segment
  Step 4  per-frame periodicity (autocorrelation peak > 0.45) -> concatenate
          the periodic regions into A

The periodicity check runs inside the already-trimmed sub-segments, so T_speech
and T_voiced are both measured on the same step-3 output. Sub-segments shorter
than 2 x 25 ms vanish at step 3; that is a consequence of the spec, not an
extra gate, and ``dropped_short`` records how many.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from . import constants as C
    from .vad import VadSegment
except ImportError:
    import constants as C
    from vad import VadSegment


@dataclass
class Periodicity:
    """Frame-level periodicity for one signal.

    ``voiced_mask`` is a per-sample boolean: a sample is periodic when any
    voiced frame covers it. That keeps trailing partial frames represented
    instead of silently truncating the tail of a segment.
    """

    frame_rms: np.ndarray
    peak: np.ndarray
    voiced_frame: np.ndarray
    voiced_mask: np.ndarray
    frame_bounds: list[tuple[int, int]] = field(default_factory=list)

    @property
    def voiced_s(self) -> float:
        return float(self.voiced_mask.sum()) / C.TARGET_SR


def trim_edges(samples: np.ndarray, trim_s: float,
               sample_rate: int = C.TARGET_SR) -> np.ndarray:
    """Drop ``trim_s`` from both ends; return an empty array if too short."""
    cut = int(round(trim_s * sample_rate))
    if len(samples) <= 2 * cut:
        return samples[:0]
    return samples[cut:len(samples) - cut]


def frame_bounds(n_samples: int, frame: int, hop: int) -> list[tuple[int, int]]:
    if n_samples < frame:
        return [(0, n_samples)] if n_samples else []
    bounds = []
    start = 0
    while start + frame <= n_samples:
        bounds.append((start, start + frame))
        start += hop
    if bounds and bounds[-1][1] < n_samples:
        bounds.append((n_samples - frame, n_samples))
    return bounds


def periodicity(samples: np.ndarray,
                sample_rate: int = C.TARGET_SR,
                frame_s: float = C.PERIOD_FRAME_S,
                hop_s: float = C.PERIOD_HOP_S,
                f0_min: float = C.F0_MIN_HZ,
                f0_max: float = C.F0_MAX_HZ,
                peak_min: float = C.PERIODICITY_PEAK_MIN) -> Periodicity:
    """Normalized autocorrelation peak per frame (SSOT Step 4).

    The normalization is the unbiased-overlap form ``r(l) = sum(x[n] x[n+l]) /
    sqrt(E0 * El)`` restricted to overlapping samples, so a decaying tail cannot
    inflate the peak. Lags span the full F0 search range and the maximum over
    that range is the frame's periodicity evidence.
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    frame = int(round(frame_s * sample_rate))
    hop = int(round(hop_s * sample_rate))
    lag_min = max(1, int(np.floor(sample_rate / f0_max)))
    lag_max = int(np.ceil(sample_rate / f0_min))
    bounds = frame_bounds(n, frame, hop)
    if not bounds:
        empty = np.zeros(0)
        return Periodicity(empty, empty, empty, np.zeros(n, dtype=bool), [])

    rms = np.asarray([np.sqrt(np.mean(samples[a:b] ** 2)) for a, b in bounds])
    loud_floor = C.PERIOD_RMS_FLOOR_RATIO * (rms.max() if rms.size else 0.0)
    peaks = np.zeros(len(bounds))
    voiced = np.zeros(len(bounds), dtype=bool)
    voiced_mask = np.zeros(n, dtype=bool)
    for index, (a, b) in enumerate(bounds):
        block = samples[a:b]
        if rms[index] <= max(loud_floor, C.RMS_EPSILON):
            continue
        block = block - block.mean()
        energy = float(np.dot(block, block))
        if energy <= 0.0:
            continue
        best = 0.0
        for lag in range(lag_min, min(lag_max, len(block) - 1) + 1):
            left = block[:-lag]
            right = block[lag:]
            denom = np.sqrt(float(np.dot(left, left)) * float(np.dot(right, right)))
            if denom <= C.RMS_EPSILON:
                continue
            value = float(np.dot(left, right)) / denom
            if value > best:
                best = value
        peaks[index] = best
        if best > peak_min:
            voiced[index] = True
            voiced_mask[a:b] = True
    return Periodicity(rms, peaks, voiced, voiced_mask, bounds)


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of True."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


@dataclass
class SegmentationResult:
    """Everything stage 0 derives from the waveform before quality scoring."""

    x_pcm: np.ndarray            # original mono 16 kHz PCM (for clipping)
    x_trim: np.ndarray           # step 1 output
    subsegments: list[tuple[int, int]]   # step 3 intervals on x_trim
    dropped_short: int
    speech_mask: np.ndarray      # per-sample bool on x_trim, step 3 output
    voiced_mask: np.ndarray      # per-sample bool on x_trim, step 4 output
    segment_a: np.ndarray        # concatenated periodic voiced audio
    periodicity: Periodicity

    @property
    def t_raw_s(self) -> float:
        return len(self.x_pcm) / C.TARGET_SR

    @property
    def t_speech_s(self) -> float:
        return float(self.speech_mask.sum()) / C.TARGET_SR

    @property
    def t_voiced_s(self) -> float:
        return float(self.voiced_mask.sum()) / C.TARGET_SR

    @property
    def voiced_ratio(self) -> float:
        return self.t_voiced_s / self.t_speech_s if self.t_speech_s > 0 else 0.0

    @property
    def n_blocks(self) -> int:
        """Count of 500 ms analysis blocks on A (SSOT W_窗数)."""
        return int(np.floor(self.t_voiced_s / C.CVA_BLOCK_S))


def segment(x_pcm: np.ndarray, vad_segments: list[VadSegment],
            sample_rate: int = C.TARGET_SR) -> SegmentationResult:
    """Run the four SSOT steps on one recording."""
    x_pcm = np.asarray(x_pcm, dtype=np.float64)
    x_trim = trim_edges(x_pcm, C.EDGE_TRIM_S, sample_rate)

    # Step 3: per-sub-segment 25 ms trim, measured on x_trim.
    kept: list[tuple[int, int]] = []
    dropped = 0
    for seg in vad_segments:
        start = np.clip(seg.start, 0, len(x_trim))
        end = np.clip(seg.end, 0, len(x_trim))
        cut = int(round(C.SUBBOUNDARY_TRIM_S * sample_rate))
        if end - start <= 2 * cut:
            dropped += 1
            continue
        kept.append((int(start + cut), int(end - cut)))

    speech_mask = np.zeros(len(x_trim), dtype=bool)
    voiced_mask = np.zeros(len(x_trim), dtype=bool)
    pieces: list[np.ndarray] = []
    peak_parts, rms_parts, frame_parts, bounds_parts = [], [], [], []
    for a, b in kept:
        speech_mask[a:b] = True
        per = periodicity(x_trim[a:b], sample_rate=sample_rate)
        voiced_mask[a:b] = per.voiced_mask
        for s, e in contiguous_runs(per.voiced_mask):
            pieces.append(x_trim[a + s:a + e])
        peak_parts.append(per.peak)
        rms_parts.append(per.frame_rms)
        frame_parts.append(per.voiced_frame.astype(bool))
        bounds_parts.extend((a + s, a + e) for s, e in per.frame_bounds)

    if pieces:
        segment_a = np.concatenate(pieces)
    else:
        segment_a = np.zeros(0, dtype=np.float64)

    combined = Periodicity(
        frame_rms=np.concatenate(rms_parts) if rms_parts else np.zeros(0),
        peak=np.concatenate(peak_parts) if peak_parts else np.zeros(0),
        voiced_frame=np.concatenate(frame_parts) if frame_parts else np.zeros(0, dtype=bool),
        voiced_mask=voiced_mask,
        frame_bounds=bounds_parts,
    )
    return SegmentationResult(
        x_pcm=x_pcm,
        x_trim=x_trim,
        subsegments=kept,
        dropped_short=dropped,
        speech_mask=speech_mask,
        voiced_mask=voiced_mask,
        segment_a=segment_a,
        periodicity=combined,
    )
