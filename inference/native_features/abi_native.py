"""Native (numpy) ABI nine-feature engine on stage-0 segment A.

Reimplements PART 2 of the original ABI Praat script (see
``training/abi_praat.py``) on top of ``praat_port`` without calling Praat.
The input contract matches ``abi_praat.segment_features``: segment A at any
rate, resampled to 44.1 kHz with Praat's sinc resampler (precision 50)
before the measures run.
"""

from __future__ import annotations

import math

import numpy as np

from . import praat_port as P

FEATURE_NAMES = ("cpps", "jitter_local", "gne_max", "hfno6000", "hnrd",
                 "h1h2", "shimmer_local_db", "shimmer_local", "psd")
SCRIPT_RATE = 44_100
_NAN = float("nan")


def _ltas_1to1_db(spec: P.Spectrum) -> np.ndarray:
    """Spectrum_to_Ltas_1to1: dB/Hz value of every bin."""
    p = 2.0 * (spec.re ** 2 + spec.im ** 2) * spec.dx
    with np.errstate(divide="ignore"):
        return np.where(p == 0.0, -300.0, 10.0 * np.log10(p / 4.0e-10))


def _ltas_bands_db(spec: P.Spectrum, bandwidth: float):
    """Spectrum_to_Ltas(bandwidth) -> (values, x1, dx, xmax)."""
    nb = math.ceil(spec.xmax / bandwidth)
    if bandwidth <= spec.dx:
        raise P.PraatError(f"Bandwidth must be greater than {spec.dx}.")
    energy = 2.0 * (spec.re ** 2 + spec.im ** 2)
    fmin = np.arange(nb) * bandwidth
    mean_e = P.mean_nointerp(energy, 0.0, spec.dx, 0.0, spec.xmax, fmin, fmin + bandwidth)
    p = mean_e * spec.dx
    with np.errstate(divide="ignore"):
        vals = np.where(p == 0.0, -300.0, 10.0 * np.log10(p / 4.0e-10))
    return vals, 0.5 * bandwidth, bandwidth, spec.xmax


def _ltas_mean_energy_db(v, x1, dx, xmax, fmin, fmax) -> float:
    m = P.mean_nointerp(10.0 ** (0.1 * v), x1, dx, 0.0, xmax, fmin, fmax)[0]
    return 10.0 * math.log10(m)


def window_measures(win: P.Sound):
    """One 0.1 s window of PART 2 -> (hfno6000, hnrd, h1h2)."""
    spec = P.to_spectrum(win, fast=True)
    ltas = _ltas_1to1_db(spec)
    lx1, ldx, lxmax = 0.0, spec.dx, spec.xmax
    lmin = P.vector_window_extremum(ltas, lx1, ldx, 0.0, lxmax, 0.0, 10000.0, False)
    ltas = ltas - lmin                              # Formula self+|min| / self-min
    e_lo = _ltas_mean_energy_db(ltas, lx1, ldx, lxmax, 0.0, 6000.0)
    e_hi = _ltas_mean_energy_db(ltas, lx1, ldx, lxmax, 6000.0, 10000.0)
    hfno = e_lo / e_hi

    mid = win.xmin + 0.05
    cg = P.to_power_cepstrogram(win, 60.0, 0.002, 5000.0, 50.0)
    nt = cg.z.shape[1]
    iframe = min(max(P.x_to_nearest_index(cg.t1, cg.dt, mid), 1), nt)
    nq = cg.z.shape[0]
    dqc = ((nq - 1) * cg.dq) / (nq - 1)
    col = P.subtract_trend_robust(cg.z[:, iframe - 1:iframe], dqc)[:, 0]
    _, qpeak = P.cepstrum_peak(col, dqc, 60.0, 400.0, P.PEAK_PARABOLIC)
    f_ceps = 1.0 / qpeak

    l6, l6x1, l6dx, l6xmax = _ltas_bands_db(spec, 6.0)
    _, f_peak = P.vector_extremum_and_x(l6, l6x1, l6dx, 0.0, l6xmax, f_ceps - 20, f_ceps + 20,
                                        P.PEAK_NONE, True)

    def ltas_max(lo, hi):
        return P.vector_window_extremum(ltas, lx1, ldx, 0.0, lxmax, lo, hi, True)

    def ltas_min(lo, hi):
        return P.vector_window_extremum(ltas, lx1, ldx, 0.0, lxmax, lo, hi, False)

    first = math.ceil(500 / f_peak)
    last = math.floor(1500 / f_peak)
    peaks = [ltas_max(p * f_peak - 20, p * f_peak + 20) for p in range(first, last + 1)]
    last_v = math.floor(1500 / f_ceps) - 1
    valleys = [ltas_min(v * f_peak + 20, (v + 1) * f_peak - 20) for v in range(first, last_v + 1)]
    if not peaks or not valleys:
        hnrd = _NAN                                 # Table "Get mean" of an empty table
    else:
        hnrd = float(np.mean(peaks)) - float(np.mean(valleys))

    h1 = P.vector_window_extremum(l6, l6x1, l6dx, 0.0, l6xmax, f_peak - 20, f_peak + 20, True)
    h2 = P.vector_window_extremum(l6, l6x1, l6dx, 0.0, l6xmax, 2 * f_peak - 20, 2 * f_peak + 20, True)
    return hfno, hnrd, h1 - h2


def spectral_features(abi: P.Sound) -> dict:
    """PART 2 window loop + global CPPS on the 44.1 kHz Sound ``abi``."""
    start, duration = abi.xmin, abi.xmax - abi.xmin
    n_windows = math.floor(duration / 0.1)
    rows = []
    for n in range(1, n_windows + 1):
        end_w = start + n * 0.1
        win = P.extract_part(abi, end_w - 0.1, end_w, preserve_times=True)
        rows.append(window_measures(win))
    rows = np.array(rows, dtype=np.float64).reshape(-1, 3)
    out = {}
    for k, name in enumerate(("hfno6000", "hnrd", "h1h2")):
        col = rows[:, k]
        # Table "Get mean" throws on an undefined cell (whole script fails) and
        # returns undefined for zero rows; both end as NaN here.
        out[name] = float(col.mean()) if col.size and np.all(np.isfinite(col)) else _NAN
    cg = P.to_power_cepstrogram(abi, 60.0, 0.002, 5000.0, 50.0)
    out["cpps"] = P.cpps(cg, abi.xmin, abi.xmax, 0.01, 0.001, 60.0, 330.0)
    return out


def to_script_rate(x: np.ndarray, sample_rate: int) -> P.Sound:
    s = P.Sound.from_samples(x, sample_rate)
    if sample_rate == SCRIPT_RATE:
        return s
    return P.resample(s, SCRIPT_RATE, 50)


def segment_features(segment_a: np.ndarray, sample_rate: int = 16_000) -> dict:
    """Nine ABI variables from segment A; NaN where Praat would be undefined."""
    from .gne_port import gne_maximum
    from .voice_port import voice_report

    abi = to_script_rate(segment_a, sample_rate)
    values = spectral_features(abi)
    values.update(voice_report(abi))
    values["gne_max"] = gne_maximum(abi)
    return {k: values[k] for k in FEATURE_NAMES}
