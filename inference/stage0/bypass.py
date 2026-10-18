"""Self-contamination-resistant noise floor and the severe bypass (SSOT §3.1.2).

The bypass exists because heavily disordered voices can fail every periodicity
test while still carrying sustained energy in the voice band. The v0.2 form
compared full-band energy to the median of VAD-silent frames, which silently
broke when breathiness filled every frame the VAD called silent. v0.2.1 pins
the noise floor to the P10 of all 50 ms frame RMS values and demands three
consecutive voice-band frames before any frame counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from . import constants as C
except ImportError:
    import constants as C


@dataclass(frozen=True)
class BypassEvidence:
    frame_rms: np.ndarray
    e_noise: float
    active_mask: np.ndarray        # frames satisfying (a), (b) and run length (c)
    t_active_s: float
    e_active: float
    snr_active_db: float
    trigger: bool


def frame_rms(samples: np.ndarray, frame_samples: int) -> np.ndarray:
    """Non-overlapping frame RMS over complete frames only."""
    n_frames = len(samples) // frame_samples
    if n_frames == 0:
        return np.zeros(0)
    usable = samples[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    return np.sqrt(np.mean(usable.astype(np.float64) ** 2, axis=1))


def band_energy_ratio(samples: np.ndarray, frame_samples: int,
                      band: tuple[float, float],
                      sample_rate: int = C.TARGET_SR) -> np.ndarray:
    """Fraction of each frame's spectral energy inside ``band``."""
    n_frames = len(samples) // frame_samples
    if n_frames == 0:
        return np.zeros(0)
    usable = samples[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    window = np.hanning(frame_samples)
    spectrum = np.abs(np.fft.rfft(usable * window, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame_samples, d=1.0 / sample_rate)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    total = spectrum.sum(axis=1)
    banded = spectrum[:, in_band].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total > C.RMS_EPSILON, banded / total, 0.0)
    return ratio


def qualify_run(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Keep only True frames that belong to a run of length >= ``min_run``."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    index = 0
    n = len(mask)
    while index < n:
        if not mask[index]:
            index += 1
            continue
        end = index
        while end < n and mask[end]:
            end += 1
        if end - index >= min_run:
            out[index:end] = True
        index = end
    return out


def bypass_evidence(x_trim: np.ndarray, t_voiced_s: float,
                    sample_rate: int = C.TARGET_SR) -> BypassEvidence:
    """Compute T_active, the active-frame SNR and the trigger flag."""
    frame = int(round(C.BYPASS_FRAME_S * sample_rate))
    rms = frame_rms(x_trim, frame)
    if rms.size == 0:
        return BypassEvidence(rms, 0.0, np.zeros(0, dtype=bool), 0.0,
                              float("nan"), float("nan"), False)
    e_noise_raw = float(np.percentile(rms, C.BYPASS_NOISE_PERCENTILE))
    # Digital silence has a zero noise floor; without a floor every frame would
    # count as "+6 dB above noise" on pure rounding error. The floor keeps the
    # ratio test meaningful and leaves the reported E_noise unchanged.
    e_noise = max(e_noise_raw, C.DIGITAL_SILENCE_RMS)
    loud = rms > C.BYPASS_ACTIVE_OVER_NOISE * e_noise
    banded = band_energy_ratio(x_trim, frame, C.BYPASS_BAND_HZ, sample_rate)
    in_band = banded > C.BYPASS_BAND_RATIO_MIN
    active = qualify_run(loud & in_band, C.BYPASS_MIN_CONSECUTIVE)

    t_active = float(active.sum()) * C.BYPASS_FRAME_S
    if active.any():
        e_active = float(np.median(rms[active]))
        if e_noise > C.RMS_EPSILON:
            snr_active = 20.0 * np.log10(e_active / e_noise)
        else:
            snr_active = float("inf")
    else:
        e_active = float("nan")
        snr_active = float("-inf")

    trigger = (
        t_voiced_s < C.BYPASS_VOICED_MAX_S
        and t_active >= C.BYPASS_ACTIVE_MIN_S
        and snr_active > C.BYPASS_SNR_MIN_DB
    )
    return BypassEvidence(rms, e_noise_raw, active, t_active, e_active,
                          float(snr_active), bool(trigger))
