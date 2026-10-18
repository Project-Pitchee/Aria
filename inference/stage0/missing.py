"""Feature-availability interface and the 3-bit missingness audit.

R5 / M_T5 / miss_rate consume the 14-feature availability mask that the
feature layer (M2) will produce. M1 does not yet compute those features, so it
takes availability from a provider; the default provider derived here answers
only the questions stage 0 can answer on its own and is PLACEHOLDER-scoped.

The interface exists so the M2 feature layer drops in without touching the
schema: any object with ``availability(x_trim, segment_a) -> Sequence[bool]``
of length 14 can be passed to the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

try:
    from . import constants as C
except ImportError:
    import constants as C

try:
    from .acoustic_missingness import ACOUSTIC_FEATURES
except ImportError:  # stage0 used standalone
    ACOUSTIC_FEATURES = (
        "hnr_db", "cpps_db", "jitter_local", "jitter_rap", "jitter_ppq5",
        "shimmer_local", "shimmer_apq3", "shimmer_apq5", "f0_median",
        "f0_iqr", "f0_cv", "voiced_fraction", "intensity_median", "intensity_iqr",
    )


class AvailabilityProvider(Protocol):
    def availability(self, segment_a: np.ndarray,
                     voiced_ratio: float) -> Sequence[bool]:
        """Length-14 boolean availability for the acoustic feature layout."""


@dataclass
class Stage0Availability:
    """PLACEHOLDER provider: derives what stage 0 evidence can support.

    Rules (all provisional, replaced by the M2 feature extractor):

    * recorded audio below 60 s cannot feed a long-window cepstral estimate;
    * F0 family (jitter, f0 statistics, voiced_fraction) needs at least 2 s of
      periodic voiced audio;
    * HNR / intensity need at least 1 s of periodic voiced audio;
    * spectral family (CPPs, shimmer) needs at least 3 s of periodic voiced
      audio and a non-silent segment.

    Duration-only rules are exactly the kind of proxy the red-line "gates must
    not target the voice itself" rule forbids if left in place, so this
    provider is marked provisional in every report it feeds.
    """

    name: str = "stage0-availability-provisional"

    def availability(self, segment_a: np.ndarray, voiced_ratio: float,
                     t_voiced_s: float | None = None) -> np.ndarray:
        t_voiced = (float(t_voiced_s) if t_voiced_s is not None
                    else len(segment_a) / C.TARGET_SR)
        silent = segment_a.size == 0 or float(np.max(np.abs(segment_a))) < 1e-6
        f0_ok = (not silent) and t_voiced >= 2.0
        energy_ok = (not silent) and t_voiced >= 1.0
        spectral_ok = (not silent) and t_voiced >= 3.0
        groups = {
            "f0": f0_ok,
            "energy": energy_ok,
            "spectral": spectral_ok,
        }
        mapping = {}
        for name in C.MISS_GROUP_F0:
            mapping[name] = groups["f0"]
        for name in C.MISS_GROUP_ENERGY:
            mapping[name] = groups["energy"]
        for name in C.MISS_GROUP_SPECTRAL:
            mapping[name] = groups["spectral"]
        mask = np.array([bool(mapping.get(name, energy_ok))
                         for name in ACOUSTIC_FEATURES], dtype=bool)
        return mask


def miss_rate(mask: Sequence[bool]) -> float:
    mask = np.asarray(mask, dtype=bool)
    if mask.size != C.N_ACOUSTIC_FEATURES:
        raise ValueError(
            f"availability mask must have {C.N_ACOUSTIC_FEATURES} entries, got {mask.size}")
    return float(1.0 - mask.mean())


def miss_code(mask: Sequence[bool]) -> int:
    """3-bit coarse pattern: [F0 family | energy/HNR family | spectral family].

    A bit is set when *every* feature in that family is missing, mirroring the
    SSOT wording "#{...}全缺".
    """
    mask = np.asarray(mask, dtype=bool)
    index = {name: i for i, name in enumerate(ACOUSTIC_FEATURES)}
    groups = (C.MISS_GROUP_F0, C.MISS_GROUP_ENERGY, C.MISS_GROUP_SPECTRAL)
    code = 0
    for bit, names in enumerate(groups):
        members = [index[n] for n in names if n in index]
        if members and not mask[members].any():
            code |= (1 << bit)
    return code


def key3_missing(mask: Sequence[bool]) -> bool:
    """CPPs / HNR / F0-or-GNE: ALL three unavailable -> hard reject M_T5."""
    mask = np.asarray(mask, dtype=bool)
    index = {name: i for i, name in enumerate(ACOUSTIC_FEATURES)}
    cpps = index.get("cpps_db")
    hnr = index.get("hnr_db")
    f0_members = [index[n] for n in ("f0_median", "f0_iqr", "f0_cv", "voiced_fraction")
                  if n in index]
    cpps_missing = cpps is not None and not mask[cpps]
    hnr_missing = hnr is not None and not mask[hnr]
    f0_missing = bool(f0_members) and not mask[f0_members].any()
    return bool(cpps_missing and hnr_missing and f0_missing)
