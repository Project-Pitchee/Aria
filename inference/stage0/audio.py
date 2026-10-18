"""Audio loading and resampling for stage 0.

Stage 0 is the only place that touches the raw recording (SSOT §2). Every
downstream stage must consume ``Stage0Audio.data`` (the concatenated periodic
voiced segment A) and the quality report, never the original file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from . import constants as C
except ImportError:  # script-style import
    import constants as C


@dataclass(frozen=True)
class LoadedAudio:
    """Raw PCM plus the provenance the report's ``input_spec`` needs."""

    pcm: np.ndarray          # float64, mono, shape (n,), range [-1, 1]
    sample_rate: int
    bit_depth: int           # 16 for PCM_16; 0 when unknown
    channels: int            # channels in the source file before downmix
    resampled: bool
    source_sha256: str

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / float(self.sample_rate)

    def input_spec(self) -> dict:
        return {
            "sample_rate_hz": int(self.sample_rate),
            "bit_depth": int(self.bit_depth),
            "channels": int(self.channels),
            "resampled": bool(self.resampled),
            "source_sha256": self.source_sha256,
        }


_SUBTYPE_BITS = {
    "PCM_S8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
    "FLOAT": 32, "DOUBLE": 64,
}


def _mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.astype(np.float64, copy=False)
    return samples.astype(np.float64, copy=False).mean(axis=1)


def resample_to(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Band-limited rational resampling.

    ``scipy.signal.resample_poly`` uses a 20-tap-per-phase Kaiser filter by
    default; that is the ENGINEERING choice this pipeline pins. Pure up/down
    ratios fall back to interpolation only when the rate already matches.
    """
    if src_sr == dst_sr:
        return samples.astype(np.float64, copy=False)
    from math import gcd

    from scipy.signal import resample_poly

    factor = gcd(int(src_sr), int(dst_sr))
    up, down = int(dst_sr) // factor, int(src_sr) // factor
    return resample_poly(samples, up, down).astype(np.float64, copy=False)


def load_wav(path: str | Path) -> LoadedAudio:
    """Load any soundfile-readable recording as mono float64 at 16 kHz."""
    import soundfile as sf

    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    info = sf.info(str(path))
    raw, sr = sf.read(str(path), dtype="float64", always_2d=True)
    channels = int(raw.shape[1])
    mono = raw.mean(axis=1)
    resampled = int(sr) != C.TARGET_SR
    pcm = resample_to(mono, int(sr), C.TARGET_SR) if resampled else mono
    return LoadedAudio(
        pcm=np.asarray(pcm, dtype=np.float64),
        sample_rate=C.TARGET_SR,
        bit_depth=_SUBTYPE_BITS.get(info.subtype, 0),
        channels=channels,
        resampled=resampled,
        source_sha256=digest,
    )
