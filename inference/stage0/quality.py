"""q_raw (8 physical dimensions), q_dmg (4 normalized damage dimensions), d.

SSOT §3.1.3. Two of the four damage detectors have no formula anywhere in the
SSOT (``T60_est`` and ``codec_artifact_score``); they are implemented here as
explicit PLACEHOLDER estimators and repeated verbatim in the milestone report.
``vowel_coverage`` likewise has only a ratio in the SSOT — the detector below
follows the De Jong & Wempe (2009) syllable-nucleus idea the older scheme
document cites, with every free parameter marked PLACEHOLDER.

``snr_est_db`` per SSOT is ``min(20 log10(signal_rms / E_noise), WADA_SNR(A))``.
The paper's lookup table is not in the repository, so WADA-SNR here is the
paper's statistic G = log E|x| - E log|x| inverted through a table rebuilt by
simulation from the paper's model (Gamma(0.4) speech amplitudes, Gaussian
noise, fixed seed). That is a deviation from "use the published table".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from . import constants as C
except ImportError:
    import constants as C


# --------------------------------------------------------------------------
# WADA-SNR (Kim & Stern, Interspeech 2008)
# --------------------------------------------------------------------------
_WADA_TABLE: tuple[np.ndarray, np.ndarray] | None = None


def _wada_statistic(signal: np.ndarray) -> float:
    """Kim & Stern G = log(E|x|) - E(log|x|) over the whole waveform."""
    magnitude = np.maximum(np.abs(np.asarray(signal, dtype=np.float64)),
                           C.WADA_AMPLITUDE_FLOOR)
    if magnitude.size < 2:
        return float("nan")
    return float(np.log(magnitude.mean()) - np.log(magnitude).mean())


def wada_lookup_table():
    """Rebuild the paper's G -> SNR table by simulation (deterministic seed).

    DEVIATION from "use the published table": the Kim & Stern (2008) table is
    not in the repository. It is regenerated with the paper's own model: clean
    speech amplitudes ~ Gamma(shape 0.4) with random sign, unit power; additive
    unit Gaussian noise scaled to each grid SNR. G rises monotonically with
    SNR; ``np.maximum.accumulate`` removes Monte-Carlo wiggle at the noisy end.
    """
    global _WADA_TABLE
    if _WADA_TABLE is not None:
        return _WADA_TABLE
    rng = np.random.default_rng(C.WADA_SIM_SEED)
    n = C.WADA_SIM_SAMPLES
    speech = rng.gamma(C.WADA_GAMMA_SHAPE, 1.0, n) * rng.choice([-1.0, 1.0], n)
    speech /= np.sqrt(np.mean(speech ** 2))
    noise = rng.standard_normal(n)
    grid = np.arange(C.WADA_GRID_DB[0], C.WADA_GRID_DB[1] + 0.5, 1.0)
    stats = np.array([_wada_statistic(speech + noise * 10.0 ** (-snr / 20.0))
                      for snr in grid])
    stats = np.maximum.accumulate(stats)
    _WADA_TABLE = (stats, grid)
    return _WADA_TABLE


def wada_snr_db(signal: np.ndarray, sample_rate: int = C.TARGET_SR) -> float:
    """Invert the G -> SNR table; values outside clamp to the grid ends."""
    value = _wada_statistic(signal)
    if not np.isfinite(value):
        return float("nan")
    stats, grid = wada_lookup_table()
    estimate = float(np.interp(value, stats, grid))
    return float(np.clip(estimate, C.SNR_REPORT_MIN_DB, C.SNR_REPORT_MAX_DB))


# --------------------------------------------------------------------------
# Damage detectors written as explicit placeholders
# --------------------------------------------------------------------------
def t60_estimate(x_trim: np.ndarray, sample_rate: int = C.TARGET_SR) -> float:
    """PLACEHOLDER T60 from a Schroeder backward integral on the quietest tail.

    The SSOT never defines how T60_est is measured. This estimator takes the
    last ``T60_WINDOW_S`` of the recording, integrates energy backwards, fits
    a line to the -5..-25 dB part of the decay, and returns ``3 * 60 / slope``.
    Poor fits (R^2 < T60_MIN_R2) return 0.0 so q_reverb stays silent rather
    than inventing reverb. Replace with a calibrated detector before release.
    """
    window = int(round(C.T60_WINDOW_S * sample_rate))
    if len(x_trim) < window or window <= 0:
        return 0.0
    tail = np.asarray(x_trim[-window:], dtype=np.float64)
    energy = np.cumsum((tail ** 2)[::-1])[::-1]
    if energy[0] <= C.RMS_EPSILON:
        return 0.0
    edc_db = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-20))
    times = np.arange(len(edc_db)) / sample_rate
    mask = (edc_db <= -5.0) & (edc_db >= -25.0)
    if mask.sum() < 10:
        return 0.0
    slope, intercept = np.polyfit(times[mask], edc_db[mask], 1)
    fitted = slope * times[mask] + intercept
    residual = edc_db[mask] - fitted
    ss_tot = float(np.sum((edc_db[mask] - edc_db[mask].mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / ss_tot if ss_tot > 0 else 0.0
    if slope >= 0 or r2 < C.T60_MIN_R2:
        return 0.0
    return float(np.clip(-60.0 / slope, 0.0, 2.0))


def codec_artifact_score(spectrum: np.ndarray, freqs: np.ndarray) -> float:
    """PLACEHOLDER codec score: high-band energy relative to the speech band.

    Low-bitrate codecs and aggressive noise suppression cut the 6.5-7.8 kHz
    shelf while leaving 300-3400 Hz mostly intact. The SSOT gives no formula,
    so this is a placeholder ratio mapped by ``CODEC_RATIO_REF``; 0 = clean.
    """
    total = float(spectrum.sum())
    if total <= C.RMS_EPSILON:
        return 0.0
    high = float(spectrum[(freqs >= C.CODEC_HIGH_BAND_HZ[0]) &
                          (freqs <= C.CODEC_HIGH_BAND_HZ[1])].sum())
    mid = float(spectrum[(freqs >= C.CODEC_MID_BAND_HZ[0]) &
                         (freqs <= C.CODEC_MID_BAND_HZ[1])].sum())
    if mid <= C.RMS_EPSILON:
        return 1.0
    ratio = high / mid
    return float(np.clip(1.0 - ratio / C.CODEC_RATIO_REF, 0.0, 1.0))


# --------------------------------------------------------------------------
# Clipping (original PCM, absolute full-scale, plus flat-top detection)
# --------------------------------------------------------------------------
def clipping_ratio(x_pcm: np.ndarray) -> float:
    """SSOT clipping definition on the original, un-resampled-scale PCM.

    ``|x| >= 0.99`` (hard limit) OR ``|x| >= 0.50`` with two consecutive
    near-zero sample differences (flat rail). Never relative to max|x|.
    """
    x = np.asarray(x_pcm, dtype=np.float64)
    if x.size == 0:
        return 0.0
    hard = np.abs(x) >= C.CLIP_HARD_LIMIT
    if x.size <= C.CLIP_FLAT_LOOKBACK:
        return float(hard.mean())
    diffs = [np.abs(x[C.CLIP_FLAT_LOOKBACK:] - x[C.CLIP_FLAT_LOOKBACK - k:x.size - k])
             for k in range(1, C.CLIP_FLAT_LOOKBACK + 1)]
    flat = np.maximum.reduce(diffs) < C.CLIP_FLAT_DELTA
    loud = np.abs(x[C.CLIP_FLAT_LOOKBACK:]) >= C.CLIP_FLAT_LIMIT
    flat_top = np.concatenate((np.zeros(C.CLIP_FLAT_LOOKBACK, dtype=bool), flat & loud))
    return float(np.mean(hard | flat_top))


# --------------------------------------------------------------------------
# Vowel-like coverage
# --------------------------------------------------------------------------
def _intensity_envelope(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = int(round(0.010 * sample_rate))
    hop = frame // 2
    n_frames = max(1, 1 + (len(signal) - frame) // hop) if len(signal) >= frame else 1
    env = np.empty(n_frames)
    for index in range(n_frames):
        a = index * hop
        env[index] = np.sqrt(np.mean(signal[a:a + frame] ** 2))
    return env


def vowel_like_mask(segment_a: np.ndarray, voiced_mask: np.ndarray,
                    sample_rate: int = C.TARGET_SR) -> np.ndarray:
    """PLACEHOLDER vowel-like regions on A (De Jong & Wempe syllable nuclei).

    Peaks in the 10 ms intensity envelope with >= 2 dB prominence and >= 150 ms
    spacing are expanded to 250 ms regions clipped to A, required to overlap a
    periodic frame, and merged. Every constant is a placeholder; the SSOT only
    fixes the ratio ``vowel_coverage = vowel-like duration / T_voiced``.
    """
    n = len(segment_a)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    env = _intensity_envelope(segment_a, sample_rate)
    if len(env) < 3:
        return out
    env_db = 20.0 * np.log10(np.maximum(env, C.RMS_EPSILON))
    frame = int(round(0.010 * sample_rate))
    hop = frame // 2
    min_gap = int(round(C.VOWEL_PEAK_MIN_GAP_S / (hop / sample_rate)))
    half = int(round(C.VOWEL_REGION_S / 2.0 * sample_rate))
    last_peak = -min_gap
    for index in range(1, len(env_db) - 1):
        if not (env_db[index] > env_db[index - 1] and env_db[index] >= env_db[index + 1]):
            continue
        neighbourhood = env_db[max(0, index - 15):index + 16]
        if env_db[index] - np.median(neighbourhood) < C.VOWEL_PEAK_PROMINENCE_DB:
            continue
        if index - last_peak < min_gap:
            continue
        center = index * hop
        a = max(0, center - half)
        b = min(n, center + half)
        if b <= a:
            continue
        region = np.zeros(n, dtype=bool)
        region[a:b] = True
        if (region & voiced_mask).sum() < (b - a) // 2:
            continue
        out |= region
        last_peak = index
    return out


# --------------------------------------------------------------------------
# Quality vector assembly
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class QualityVectors:
    q_raw: np.ndarray            # 8 physical dimensions, SSOT order
    q_dmg: np.ndarray            # [q_clip, q_reverb, q_snr, q_codec]
    d: float
    snr_est_db: float
    diagnostics: dict


def quality_vectors(x_pcm: np.ndarray, x_trim: np.ndarray, segment_a: np.ndarray,
                    voiced_mask: np.ndarray, voiced_ratio: float,
                    e_noise: float, n_missing: int,
                    signal_rms: float | None = None,
                    sample_rate: int = C.TARGET_SR) -> QualityVectors:
    """Assemble q_raw / q_dmg / d exactly in SSOT §3.1.3 order.

    ``voiced_mask`` is the per-sample periodic mask on the same time base as
    ``x_trim``; ``signal_rms`` is the median RMS over periodic frames and is
    passed in by the caller that already computed frame RMS.
    """
    if signal_rms is None or not np.isfinite(signal_rms):
        signal_rms = float(np.sqrt(np.mean(segment_a ** 2))) if segment_a.size else 0.0

    from scipy.signal import welch

    if segment_a.size >= C.SPECTRAL_WELCH_SEGMENT:
        freqs, spectrum = welch(segment_a, fs=sample_rate,
                                nperseg=C.SPECTRAL_WELCH_SEGMENT)
    elif segment_a.size:
        freqs, spectrum = welch(segment_a, fs=sample_rate,
                                nperseg=max(2, segment_a.size))
    else:
        freqs, spectrum = np.zeros(1), np.zeros(1)
    total_energy = float(spectrum.sum())
    if total_energy > C.RMS_EPSILON:
        flatness = float(np.exp(np.mean(np.log(np.maximum(spectrum, 1e-20)))) /
                         np.mean(spectrum))
        lf_ratio = float(spectrum[freqs < C.LF_RUMBLE_HZ].sum() / total_energy)
        tilt = 10.0 * np.log10(
            max(float(spectrum[(freqs >= C.TILT_HIGH_HZ[0]) &
                               (freqs <= C.TILT_HIGH_HZ[1])].sum()), 1e-20) /
            max(float(spectrum[(freqs >= C.TILT_LOW_HZ[0]) &
                               (freqs <= C.TILT_LOW_HZ[1])].sum()), 1e-20))
    else:
        flatness, lf_ratio, tilt = float("nan"), float("nan"), float("nan")

    # snr_est_db = min(energy ratio, WADA-SNR) — the stricter of the two.
    if e_noise > C.RMS_EPSILON and signal_rms > 0:
        energy_snr = 20.0 * np.log10(signal_rms / e_noise)
    else:
        energy_snr = float("nan")
    wada = wada_snr_db(segment_a, sample_rate) if segment_a.size else float("nan")
    candidates = [v for v in (energy_snr, wada) if np.isfinite(v)]
    snr_est = min(candidates) if candidates else float("nan")

    clip_ratio = clipping_ratio(x_pcm)
    recording_level = (20.0 * np.log10(np.sqrt(np.mean(x_trim ** 2)))
                       if x_trim.size else float("-inf"))

    q_raw = np.array([
        snr_est,
        flatness,
        lf_ratio,
        clip_ratio,
        recording_level,
        tilt,
        voiced_ratio,
        float(n_missing),
    ], dtype=np.float64)

    q_clip = float(np.clip(clip_ratio / C.Q_CLIP_SCALE, 0.0, 1.0))
    t60 = t60_estimate(x_trim, sample_rate)
    q_reverb = float(np.clip((t60 - C.Q_REVERB_T60_ONSET_S) / C.Q_REVERB_T60_SCALE_S,
                             0.0, 1.0))
    q_snr = (float(np.clip((C.Q_SNR_CLEAN_DB - snr_est) / C.Q_SNR_SCALE_DB, 0.0, 1.0))
             if np.isfinite(snr_est) else 1.0)
    q_codec = codec_artifact_score(spectrum, freqs)
    q_dmg = np.array([q_clip, q_reverb, q_snr, q_codec], dtype=np.float64)
    d = float(1.0 - np.prod(1.0 - q_dmg))
    return QualityVectors(
        q_raw=q_raw,
        q_dmg=q_dmg,
        d=d,
        snr_est_db=float(snr_est) if np.isfinite(snr_est) else float("nan"),
        diagnostics={"energy_snr_db": energy_snr, "wada_snr_db": wada,
                     "t60_est_s": t60, "clipping_ratio": clip_ratio},
    )
