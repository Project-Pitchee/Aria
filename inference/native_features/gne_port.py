"""GNE maximum: ``To Harmonicity (gne)... 500 4500 1000 80`` then ``Get maximum``.

Port of Sound_to_Harmonicity_GNE (Praat 6.1.38, fon/Sound_to_Harmonicity_GNE.cpp)
with Sound_to_LPC_autocorrelation and LPC_Sound_filterInverse (LPC/Sound_and_LPC.cpp).
Inner products and the inverse-filter accumulation use numpy's summation
order rather than Praat's, so results agree to rounding, not bit for bit.
"""

from __future__ import annotations

import math

import numpy as np

from . import praat_port as P
from .praat_port import index_to_x


def _lpc_autocorrelation(sound: P.Sound, order=13, analysis_width=30e-3, dt=10e-3):
    """Sound_to_LPC_autocorrelation, pre-emphasis off (1e9 Hz). Returns (t1, dt, A)."""
    fs = 1.0 / sound.dx
    window_duration = 2.0 * analysis_width
    if math.floor(window_duration / sound.dx) <= order:
        raise P.PraatError("Analysis window duration too short.")
    if window_duration > sound.dx * sound.nx:
        window_duration = sound.dx * sound.nx
    n_frames, t1 = P.short_term_analysis(sound.x1, sound.dx, sound.nx, window_duration, dt)
    nsf = int(P.c_round(window_duration * fs))
    window = P.gaussian_window(nsf)
    A = np.zeros((n_frames, order))
    for k in range(n_frames):
        t = index_to_x(t1, dt, k + 1)
        idx = P.into_sound_indices(sound, t - 0.5 * window_duration, nsf)
        x = P.gather(sound.z, idx)
        x = (x - x.mean()) * window
        r = np.array([np.dot(x[:nsf - i], x[i:]) for i in range(order + 1)])
        if r[0] == 0.0:
            continue                                   # zero coefficients
        a = np.zeros(order + 2)                        # a[1..order+1], 1-based
        a[1] = 1.0
        rc1 = -r[1] / r[0]
        a[2] = rc1
        gain = r[0] + r[1] * rc1
        ncoef = order
        for i in range(2, order + 1):
            s = 0.0
            for j in range(1, i + 1):
                s += r[i - j + 1] * a[j]
            rci = -s / gain
            for j in range(2, i // 2 + 2):
                at = a[j] + rci * a[i - j + 2]
                a[i - j + 2] += rci * a[j]
                a[j] = at
            a[i + 1] = rci
            gain += rci * s
            if gain <= 0.0:
                ncoef = i - 1
                break
        A[k, :ncoef] = a[2:2 + ncoef]
    return t1, dt, A


def _filter_inverse(sound: P.Sound, t1, dt, A):
    n = sound.nx
    order = A.shape[1]
    times = np.array([index_to_x(sound.x1, sound.dx, i) for i in range(1, n + 1)])
    frame = np.floor((times - t1) / dt + 1.0 + 0.5).astype(np.int64)     # Sampled_xToNearestIndex
    valid = (frame >= 1) & (frame <= A.shape[0])
    coef = np.zeros((n, order))
    coef[valid] = A[frame[valid] - 1]
    src = sound.z.copy()
    for j in range(1, order + 1):
        prev = np.zeros(n)
        prev[j:] = sound.z[:-j]
        src += coef[:, j - 1] * prev                   # usableDepth: zeros before sample 1
    src[~valid] = 0.0
    return P.Sound(src, sound.xmin, sound.xmax, sound.x1, sound.dx)


def _band_filter(re, im, dx, fmid, bandwidth):
    x = np.arange(re.size) * dx
    fmin, fmax = fmid - bandwidth / 2.0, fmid + bandwidth / 2.0
    out = (x < fmin) | (x > fmax)
    factor = 0.5 + 0.5 * np.cos((2.0 * math.pi / bandwidth) * (x - fmid))
    factor[out] = 0.0
    return re * factor, im * factor


def _lagged(E, n, d):
    """C[row, col] = sum_i row[i] * col[i + d]."""
    if d >= 0:
        return E[:, :n - d] @ E[:, d:].T
    return E[:, -d:] @ E[:, :n + d].T


def gne_maximum(me: P.Sound, fmin=500.0, fmax=4500.0, bandwidth=1000.0, step=80.0) -> float:
    original = P.resample(me, 10000, 500)
    original.z -= original.z.mean()
    duration = me.xmax - me.xmin
    t1, dt, A = _lpc_autocorrelation(original)
    flat = _filter_inverse(original, t1, dt, A)
    spec = P.to_spectrum(flat, fast=True)
    h_re, h_im = spec.im.copy(), -spec.re
    envelopes = []
    fmid = fmin
    while fmid <= fmax:
        b_re, b_im = _band_filter(spec.re, spec.im, spec.dx, fmid, bandwidth)
        hb_re, hb_im = _band_filter(h_re, h_im, spec.dx, fmid, bandwidth)
        band = P.spectrum_to_sound(P.Spectrum(b_re, b_im, spec.xmax, spec.dx))
        hband = P.spectrum_to_sound(P.Spectrum(hb_re, hb_im, spec.xmax, spec.dx))
        env = P.extract_part(band, 0.0, duration, preserve_times=True)
        m = env.nx
        e = np.sqrt(env.z ** 2 + hband.z[:m] ** 2)
        envelopes.append(e - e.mean())
        fmid += step
    E = np.array(envelopes)
    nenv, n = E.shape
    power = np.einsum("ij,ij->i", E, E)
    norm = np.sqrt(power)
    cc = np.full((nenv, nenv), -np.inf)
    # lags -3..3 samples (Sounds_crossCorrelate_short, -3.1e-4..3.1e-4 at 10 kHz)
    lag_lo = math.ceil(-3.1e-4 / original.dx)
    lag_hi = math.floor(3.1e-4 / original.dx)
    for d in range(lag_lo, lag_hi + 1):
        with np.errstate(all="ignore"):   # Accelerate BLAS raises spurious FP flags
            C = _lagged(E, n, d)
        cc = np.maximum(cc, C)
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = cc / np.outer(norm, norm)
    rows, cols = np.tril_indices(nenv, -1)
    keep = np.abs(rows - cols) >= bandwidth / 2.0 / step
    vals = cc[rows[keep], cols[keep]]
    vals = vals[np.isfinite(vals)]
    return float(max(0.0, vals.max())) if vals.size else 0.0
