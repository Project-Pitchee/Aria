"""Pitch (ac), PointProcess (cc) and Voice report, ported from Praat 6.1.38.

Script calls reproduced (PART 2 of the ABI script):

    To Pitch... 0 70 600
    To PointProcess (cc)
    Voice report... 0 0 70 600 1.3 1.6 0.03 0.45

then ``extractNumber`` on the report text, including its rounding to the
printed digits (Melder_percent / Melder_fixed / Melder_fixedExponent).
"""

from __future__ import annotations

import math

import numpy as np

from . import praat_port as P
from .praat_port import fma, index_to_x

_NAN = float("nan")
_LOG2E = 1.4426950408889634073599246810018921374266


def _log2(x):
    return math.log(x) * _LOG2E


# --------------------------------------------------------------------------
# Sound_to_Pitch_ac (Sound_to_Pitch.cpp) + Pitch_pathFinder (Pitch.cpp)
# --------------------------------------------------------------------------

class Pitch:
    def __init__(self, xmin, xmax, nx, dx, x1, ceiling, freq, strength):
        self.xmin, self.xmax, self.nx, self.dx, self.x1 = xmin, xmax, nx, dx, x1
        self.ceiling = ceiling
        self.freq = freq            # frequency of candidate 1 per frame
        self.strength = strength

    def voiced(self, i):            # 1-based
        f = self.freq[i - 1]
        return f > 0.0 and f < self.ceiling

    def value_at_time(self, x):
        """Pitch_getValueAtTime(HERTZ, linear)."""
        if x < self.xmin or x > self.xmax:
            return _NAN
        index_real = (x - self.x1) / self.dx + 1.0
        left = math.floor(index_real)
        phase = index_real - left
        if phase < 0.5:
            near, far = left, left + 1
        else:
            far, near = left, left + 1
            phase = 1.0 - phase
        if near < 1 or near > self.nx or not self.voiced(near):
            return _NAN
        nv = self.freq[near - 1]
        if far < 1 or far > self.nx or not self.voiced(far):
            return nv
        fv = self.freq[far - 1]
        return fma(phase, fv - nv, nv)


def short_term_analysis_fused(x1, dx, nx, window_duration, time_step):
    """Sampled_shortTermAnalysis with Clang FMA contraction of both sums."""
    my_duration = dx * nx
    if window_duration > my_duration:
        raise P.PraatError("shorter than window length.")
    n = math.floor((my_duration - window_duration) / time_step) + 1
    our_mid = fma(0.5, my_duration, fma(-0.5, dx, x1))
    thy_duration = n * time_step
    return n, fma(0.5, time_step, fma(-0.5, thy_duration, our_mid))


def sound_to_pitch_ac(sound: P.Sound, time_step=0.0, minimum_pitch=70.0, ceiling=600.0,
                      periods_per_window=3.0, max_candidates=15, silence=0.03, voicing=0.45,
                      octave_cost=0.01, octave_jump=0.35, vuv_cost=0.14) -> Pitch:
    z, dx, x1 = sound.z, sound.dx, sound.x1
    nx = z.size
    if max_candidates < ceiling / minimum_pitch:
        max_candidates = math.floor(ceiling / minimum_pitch)
    dt = time_step if time_step > 0.0 else periods_per_window / minimum_pitch / 4.0
    duration = dx * nx
    if minimum_pitch < periods_per_window / duration:
        raise P.PraatError("minimum pitch too low for this Sound")
    nsamp_period = math.floor(1.0 / dx / minimum_pitch)
    halfnsamp_period = nsamp_period // 2 + 1
    ceiling = min(ceiling, 0.5 / dx)
    dt_window = periods_per_window / minimum_pitch
    nsamp_window = math.floor(dt_window / dx)
    halfnsamp_window = nsamp_window // 2 - 1
    if halfnsamp_window < 2:
        raise P.PraatError("Analysis window too short.")
    nsamp_window = halfnsamp_window * 2
    maximum_lag = min(math.floor(nsamp_window / periods_per_window) + 2, nsamp_window)
    n_frames, t1 = short_term_analysis_fused(x1, dx, nx, dt_window, dt)

    freq = np.zeros(n_frames)
    strength = np.zeros(n_frames)
    pitch = Pitch(sound.xmin, sound.xmax, n_frames, dt, t1, ceiling, freq, strength)
    global_peak = float(np.max(np.abs(z - np.mean(z)))) if nx else 0.0
    if global_peak == 0.0:
        return pitch

    nfft = 1
    while nfft < nsamp_window * 1.5:
        nfft *= 2
    brent_ixmax = math.floor(nsamp_window * 0.5)
    i = np.arange(1, nsamp_window + 1, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(i * 2 * math.pi / (nsamp_window + 1))
    wr = np.fft.irfft(np.abs(np.fft.rfft(window, nfft)) ** 2, nfft)
    window_r = wr / wr[0]

    times = np.array([index_to_x(t1, dt, k) for k in range(1, n_frames + 1)])
    left = np.floor((times - x1) / dx + 1.0).astype(np.int64)
    right = left + 1
    csum = np.concatenate(([0.0], np.cumsum(z)))
    local_mean = (csum[left + nsamp_period] - csum[right - nsamp_period - 1]) / (2 * nsamp_period)
    start = right - halfnsamp_window
    idx = start[:, None] + np.arange(nsamp_window)[None, :]
    frames = (z[idx - 1] - local_mean[:, None]) * window[None, :]
    lp_lo = max(halfnsamp_window + 1 - halfnsamp_period, 1)
    lp_hi = min(halfnsamp_window + halfnsamp_period, nsamp_window)
    local_peak = np.max(np.abs(frames[:, lp_lo - 1:lp_hi]), axis=1)
    intensity = np.where(local_peak > global_peak, 1.0, local_peak / global_peak)
    ac = np.fft.irfft(np.abs(np.fft.rfft(frames, nfft, axis=1)) ** 2, nfft, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rpos = ac[:, 1:brent_ixmax + 1] / (ac[:, :1] * window_r[None, 1:brent_ixmax + 1])
    B = brent_ixmax
    R = np.concatenate((rpos[:, ::-1], np.ones((n_frames, 1)), rpos), axis=1)   # column = lag + B

    # First pass: local maxima of r, parabolic frequency, sinc-30 strength.
    lag_hi = min(maximum_lag, brent_ixmax) - 1          # i < maximumLag && i < brent_ixmax
    lags = np.arange(2, lag_hi + 1)
    rc, rl, rr = R[:, lags + B], R[:, lags - 1 + B], R[:, lags + 1 + B]
    is_peak = (rc > 0.5 * voicing) & (rc > rl) & (rc >= rr)
    is_peak &= (local_peak > 0.0)[:, None]
    fi, li = np.nonzero(is_peak)
    lag = lags[li]
    r0, rm, rp = R[fi, lag + B], R[fi, lag - 1 + B], R[fi, lag + 1 + B]
    dr = 0.5 * (rp - rm)
    d2r = 2.0 * r0 - rm - rp
    f_max = 1.0 / dx / (lag + dr / d2r)
    s_max = P.interpolate_sinc_rows(R, fi, 1.0 / dx / f_max + B + 1, 30)
    s_max = np.where(s_max > 1.0, 1.0 / s_max, s_max)

    cand_f = [[0.0] for _ in range(n_frames)]
    cand_s = [[0.0] for _ in range(n_frames)]
    cand_i = [[0] for _ in range(n_frames)]
    for k in range(fi.size):
        f = int(fi[k])
        cf, cs, ci = cand_f[f], cand_s[f], cand_i[f]
        fr, st = float(f_max[k]), float(s_max[k])
        if len(cf) < max_candidates:
            cf.append(fr), cs.append(st), ci.append(int(lag[k]))
            continue
        weakest, place = 2.0, 0
        for iw in range(1, max_candidates):
            ls = cs[iw] - octave_cost * _log2(minimum_pitch / cf[iw])
            if ls < weakest:
                weakest, place = ls, iw
        if st - octave_cost * _log2(minimum_pitch / fr) <= weakest:
            place = 0
        if place:
            cf[place], cs[place], ci[place] = fr, st, int(lag[k])

    # Second pass: Brent on sinc-70 interpolation around each candidate lag.
    pf = np.array([f for f in range(n_frames) for _ in cand_i[f][1:]], dtype=np.int64)
    pl = np.array([v for f in range(n_frames) for v in cand_i[f][1:]], dtype=np.float64)
    if pf.size:
        ixmid = pl + B + 1

        def neg_sinc(xx, sel):
            return -P.interpolate_sinc_rows(R, pf[sel], xx, 70)

        xr, fr = P.minimize_brent_many(neg_sinc, ixmid - 1, ixmid + 1, 1e-10)
        ymid = -fr
        xmid = xr - B - 1
        new_f = 1.0 / dx / xmid
        new_s = np.where(ymid > 1.0, 1.0 / ymid, ymid)
        k = 0
        for f in range(n_frames):
            for c in range(1, len(cand_f[f])):
                cand_f[f][c], cand_s[f][c] = float(new_f[k]), float(new_s[k])
                k += 1

    _path_finder(pitch, cand_f, cand_s, intensity, silence, voicing, octave_cost,
                 octave_jump, vuv_cost, ceiling)
    return pitch


def _path_finder(pitch, cand_f, cand_s, intensity, silence, voicing, octave_cost,
                 octave_jump, vuv_cost, ceiling):
    n = pitch.nx
    maxn = max(len(c) for c in cand_f)
    corr = 0.01 / pitch.dx
    octave_jump *= corr
    vuv_cost *= corr
    F = np.zeros((n, maxn))
    D = np.full((n, maxn), -np.inf)
    nc = np.array([len(c) for c in cand_f])
    for f in range(n):
        F[f, :nc[f]] = cand_f[f]
        unv = (2.0 - intensity[f] / (silence / (1.0 + voicing))) if silence > 0 else 0.0
        unv = voicing + max(0.0, unv)
        for c in range(nc[f]):
            fr = cand_f[f][c]
            voiceless = not (fr > 0.0 and fr < ceiling)
            D[f, c] = unv if voiceless else cand_s[f][c] - octave_cost * _log2(ceiling / fr)
    V = (F > 0.0) & (F < ceiling)
    psi = np.zeros((n, maxn), dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logF = np.where(V, np.log(np.where(V, F, 1.0)) * _LOG2E, 0.0)
    for f in range(1, n):
        prev = D[f - 1]
        m1, m2 = nc[f - 1], nc[f]
        v1, v2 = V[f - 1, :m1], V[f, :m2]
        # transition cost [icand2, icand1]
        both = v2[:, None] & v1[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            jump = np.abs((np.log(F[f - 1, :m1][None, :] / np.where(v2, F[f, :m2], 1.0)[:, None])) * _LOG2E)
        cost = np.where(both, octave_jump * jump, np.where(v2[:, None] ^ v1[None, :], vuv_cost, 0.0))
        val = (prev[None, :m1] - cost) + D[f, :m2][:, None]
        place = np.argmax(val, axis=1)
        D[f, :m2] = val[np.arange(m2), place]
        psi[f, :m2] = place
    place = 0
    best = D[n - 1, 0]
    for c in range(1, nc[n - 1]):
        if D[n - 1, c] > best:
            place, best = c, D[n - 1, c]
    for f in range(n - 1, -1, -1):
        pitch.freq[f] = cand_f[f][place]
        pitch.strength[f] = cand_s[f][place]
        cand_f[f][0], cand_f[f][place] = cand_f[f][place], cand_f[f][0]
        cand_s[f][0], cand_s[f][place] = cand_s[f][place], cand_s[f][0]
        place = psi[f, place]


# --------------------------------------------------------------------------
# Sound_Pitch_to_PointProcess_cc (Pitch_to_PointProcess.cpp)
# --------------------------------------------------------------------------

def _voiced_interval_after(pitch: Pitch, after):
    ileft = math.ceil((after - pitch.x1) / pitch.dx + 1.0)
    if ileft > pitch.nx:
        return None
    ileft = max(ileft, 1)
    while ileft <= pitch.nx and not pitch.voiced(ileft):
        ileft += 1
    if ileft > pitch.nx:
        return None
    iright = ileft
    while iright <= pitch.nx and pitch.voiced(iright):
        iright += 1
    iright -= 1
    tleft = index_to_x(pitch.x1, pitch.dx, ileft) - 0.5 * pitch.dx
    tright = index_to_x(pitch.x1, pitch.dx, iright) + 0.5 * pitch.dx
    if tleft >= pitch.xmax - 0.5 * pitch.dx:
        return None
    tleft = max(tleft, pitch.xmin)
    tright = min(tright, pitch.xmax)
    if tright <= after:
        return None
    return tleft, tright


def _find_extremum_abs(s: P.Sound, tmin, tmax):
    """Sound_findExtremum(includeMaxima = includeMinima = true)."""
    imin = math.floor((tmin - s.x1) / s.dx + 1.0)
    imax = math.ceil((tmax - s.x1) / s.dx + 1.0)
    imin, imax = max(imin, 1), min(imax, s.nx)
    n = imax - imin + 1
    y = s.z[imin - 1:imax] if n > 0 else np.empty(0)
    if n <= 0:
        iext = 0.0
    elif n == 1:
        iext = 1.0
    elif n == 2:
        a, b = abs(y[0]), abs(y[1])
        iext = 1.0 if a > b else 2.0 if a < b else 1.5
    else:
        mn, mx = float(y.min()), float(y.max())
        if mn == mx:
            iext = 0.5 * (n + 1.0)
        else:
            ie = (int(np.argmin(y)) if abs(mn) > abs(mx) else int(np.argmax(y))) + 1
            if ie == 1:
                iext = 1.0
            elif ie == n:
                iext = float(n)
            else:
                vm, vl, vr = y[ie - 1], y[ie - 2], y[ie]
                iext = ie + 0.5 * (vr - vl) / (2 * vm - vl - vr)
    if iext != 0.0:
        return fma(imin - 1 + iext - 1, s.dx, s.x1)
    return (tmin + tmax) / 2


def _find_max_correlation(s: P.Sound, t1, window_length, tmin2, tmax2):
    """Sound_findMaximumCorrelation -> (correlation, tout or None, peak)."""
    z, x1, dx, nx = s.z, s.x1, s.dx, s.nx
    half = 0.5 * window_length
    ileft1 = P.iround((t1 - half - x1) / dx + 1.0)
    iright1 = P.iround((t1 + half - x1) / dx + 1.0)
    l2min = math.floor((tmin2 - half - x1) / dx + 1.0)
    l2max = math.ceil((tmax2 - half - x1) / dx + 1.0)
    i1 = np.arange(ileft1, iright1 + 1)
    shifts = np.arange(l2min, l2max + 1) - ileft1
    i2 = i1[None, :] + shifts[:, None]
    ok = (i1[None, :] >= 1) & (i1[None, :] <= nx) & (i2 >= 1) & (i2 <= nx)
    a1 = np.where(ok, z[np.clip(i1, 1, nx)[None, :] - 1], 0.0)
    a2 = np.where(ok, z[np.clip(i2, 1, nx) - 1], 0.0)
    norm1 = (a1 * a1).sum(axis=1)
    norm2 = (a2 * a2).sum(axis=1)
    prod = (a1 * a2).sum(axis=1)
    peaks = np.abs(a2).max(axis=1) if a2.shape[1] else np.zeros(shifts.size)
    best, r1b, r3b, ir, peak = -1.0, _NAN, _NAN, _NAN, 0.0
    r1 = r2 = r3 = 0.0
    for k in range(shifts.size):
        r1, r2 = r2, r3
        r3 = prod[k] / math.sqrt(norm1[k] * norm2[k]) if prod[k] != 0.0 else 0.0
        if r2 > best and r2 >= r1 and r2 >= r3:
            r1b, best, r3b = r1, r2, r3
            ir = (l2min + k) - 1
            peak = float(peaks[k])
    tout = None
    if best > -1.0:
        d2r = 2 * best - r1b - r3b
        if d2r != 0.0:
            drr = 0.5 * (r3b - r1b)
            best += 0.5 * drr * drr / d2r
            ir += drr / d2r
        tout = fma(ir - ileft1, dx, t1)
    return best, tout, peak


def _add_point(points: list, t):
    """PointProcess_addPoint (sorted, no duplicates)."""
    import bisect

    if not points or t >= points[-1]:
        if not points or t != points[-1] or True:
            # Praat appends even when equal to the last point
            points.append(t)
        return
    k = bisect.bisect_right(points, t)          # number of points <= t
    if k == 0 or points[k - 1] != t:
        points.insert(k, t)


def sound_pitch_to_point_process_cc(s: P.Sound, pitch: Pitch) -> np.ndarray:
    points: list[float] = []
    t = pitch.xmin
    added_right = -1e308
    gp = max(abs(float(s.z.min())), abs(float(s.z.max())))
    while True:
        iv = _voiced_interval_after(pitch, t)
        if iv is None:
            break
        tleft, tright = iv
        tmiddle = (tleft + tright) / 2.0
        f0m = pitch.value_at_time(tmiddle)
        tmax = _find_extremum_abs(s, tmiddle - 0.5 / f0m, tmiddle + 0.5 / f0m)
        _add_point(points, tmax)
        tsave = tmax
        while True:
            f0 = pitch.value_at_time(tmax)
            if math.isnan(f0):
                break
            corr, tout, peak = _find_max_correlation(s, tmax, 1.0 / f0, tmax - 1.25 / f0, tmax - 0.8 / f0)
            if tout is not None:
                tmax = tout
            if corr == -1.0:
                tmax -= 1.0 / f0
            if tmax < tleft:
                if corr > 0.7 and peak > 0.023333 * gp and tmax - added_right > 0.8 / f0:
                    _add_point(points, tmax)
                break
            if corr > 0.3 and (peak == 0.0 or peak > 0.01 * gp):
                if tmax - added_right > 0.8 / f0:
                    _add_point(points, tmax)
        tmax = tsave
        while True:
            f0 = pitch.value_at_time(tmax)
            if math.isnan(f0):
                break
            corr, tout, peak = _find_max_correlation(s, tmax, 1.0 / f0, tmax + 0.8 / f0, tmax + 1.25 / f0)
            if tout is not None:
                tmax = tout
            if corr == -1.0:
                tmax += 1.0 / f0
            if tmax > tright:
                if corr > 0.7 and peak > 0.023333 * gp:
                    _add_point(points, tmax)
                    added_right = tmax
                break
            if corr > 0.3 and (peak == 0.0 or peak > 0.01 * gp):
                _add_point(points, tmax)
                added_right = tmax
        t = tright
    return np.array(points)


# --------------------------------------------------------------------------
# Voice report quantities (VoiceAnalysis.cpp, PointProcess.cpp, AmplitudeTier.cpp)
# --------------------------------------------------------------------------

def _is_period(t, i, pmin, pmax, mpf):
    """PointProcess_isPeriod, ``i`` 0-based left point."""
    nt = t.size
    if i < 0 or i + 1 >= nt:
        return False
    if pmin == pmax:
        return True
    iv = t[i + 1] - t[i]
    if iv <= 0.0 or iv < pmin or iv > pmax:
        return False
    prev = t[i] - t[i - 1] if i >= 1 else None
    nxt = t[i + 2] - t[i + 1] if i + 2 < nt else None
    pf = iv / prev if prev is not None and prev > 0.0 else None
    nf = iv / nxt if nxt is not None and nxt > 0.0 else None
    if pf is None and nf is None:
        return True
    if pf is not None and 0.0 < pf < 1.0:
        pf = 1.0 / pf
    if nf is not None and 0.0 < nf < 1.0:
        nf = 1.0 / nf
    if pf is not None and pf > mpf and nf is not None and nf > mpf:
        return False
    return True


def _period_stats(t, pmin, pmax, mpf):
    periods = [t[i + 1] - t[i] for i in range(t.size - 1) if _is_period(t, i, pmin, pmax, mpf)]
    n = len(periods)
    mean = sum(periods) / n if n > 0 else _NAN
    if n < 2:
        return mean, _NAN
    m = sum(periods) / n
    return mean, math.sqrt(sum((p - m) * (p - m) for p in periods) / (n - 1))


def _jitter_local(t, pmin, pmax, mpf, mean_period):
    n_periods = t.size - 1
    if n_periods < 2:
        return _NAN
    total = 0.0
    for i in range(1, t.size - 1):
        p1, p2 = t[i] - t[i - 1], t[i + 1] - t[i]
        f = p1 / p2 if p1 > p2 else p2 / p1
        if pmin == pmax or (pmin <= p1 <= pmax and pmin <= p2 <= pmax and f <= mpf):
            total += abs(p1 - p2)
        else:
            n_periods -= 1
    if n_periods < 2:
        return _NAN
    return (total / (n_periods - 1)) / mean_period


def _hann_rms(s: P.Sound, tmid, wl, wr):
    imin, imax, count = P.window_samples(s.x1, s.dx, s.nx, tmid - wl, tmid + wr)
    if count < 3:
        return _NAN
    i = np.arange(imin, imax + 1)
    tt = np.array([fma(k - 1, s.dx, s.x1) for k in i])
    width = np.where(tt < tmid, wl, wr)
    w = 0.5 + 0.5 * np.cos(math.pi * ((tt - tmid) / width))
    v = s.z[i - 1] * w
    return math.sqrt(float(np.sum(v * v)) / float(np.sum(w * w)))


def _shimmer(t, s, pmin, pmax, mpf, maf):
    if t.size < 3:
        return _NAN, _NAN
    pts = []
    for i in range(1, t.size - 1):
        p1, p2 = t[i] - t[i - 1], t[i + 1] - t[i]
        f = p1 / p2 if p1 > p2 else p2 / p1
        if pmin == pmax or (pmin <= p1 <= pmax and pmin <= p2 <= pmax and f <= mpf):
            pk = _hann_rms(s, t[i], 0.2 * p1, 0.2 * p2)
            if not math.isnan(pk) and pk > 0.0:
                pts.append((t[i], pk))
    num, cnt, db = 0.0, 0, 0.0
    for k in range(1, len(pts)):
        p = pts[k][0] - pts[k - 1][0]
        if pmin == pmax or (pmin <= p <= pmax):
            a1, a2 = pts[k - 1][1], pts[k][1]
            af = a1 / a2 if a1 > a2 else a2 / a1
            if af <= maf:
                num += abs(a1 - a2)
                db += abs(math.log10(a1 / a2))
                cnt += 1
    if cnt < 1:
        return _NAN, _NAN
    num /= cnt
    den = sum(v for _, v in pts[:-1]) / (len(pts) - 1)
    local = num / den if den != 0.0 else _NAN
    return local, 20.0 * (db / cnt)


def _c_printf_round(value: float, digits: int) -> float:
    """``%.*f`` then strtod: Python's formatting is correctly rounded like macOS libc."""
    return float(f"{value:.{digits}f}")


def _min_prec(v):
    return -math.floor(math.log10(abs(v)))


def report_percent(value):
    """Melder_percent(value, 3) -> extractNumber -> value in percent units."""
    if math.isnan(value):
        return _NAN
    if value == 0.0:
        return 0.0
    v = value * 100.0
    s = f"{v:.{max(_min_prec(v), 3)}f}"
    return (0.01 * float(s)) * 100.0          # script: extractNumber (gives fraction) * 100


def report_fixed(value, precision=3):
    if math.isnan(value):
        return _NAN
    if value == 0.0:
        return 0.0
    return float(f"{value:.{max(_min_prec(value), precision)}f}")


def report_fixed_exponent(value, exponent=-3, precision=6):
    if math.isnan(value):
        return _NAN
    if value == 0.0:
        return 0.0
    v = value / (10.0 ** exponent)
    return float(f"{v:.{max(_min_prec(v), precision)}f}E{exponent}")


def voice_report(abi: P.Sound, raw: bool = False) -> dict:
    """Jitter/shimmer/PSD as the script extracts them from ``Voice report``."""
    pitch = sound_to_pitch_ac(abi, 0.0, 70.0, 600.0)
    t = sound_pitch_to_point_process_cc(abi, pitch)
    floor_, ceiling = 70.0, 600.0
    pmin, pmax = 0.8 / ceiling, 1.25 / floor_
    mean_period, sd_period = _period_stats(t, pmin, pmax, 1.3)
    jit = _jitter_local(t, pmin, pmax, 1.3, mean_period)
    shim, shim_db = _shimmer(t, abi, pmin, pmax, 1.3, 1.6)
    if raw:
        return {"jitter_local": jit * 100, "shimmer_local": shim * 100,
                "shimmer_local_db": shim_db, "psd": sd_period, "_points": t, "_pitch": pitch}
    return {"jitter_local": report_percent(jit), "shimmer_local": report_percent(shim),
            "shimmer_local_db": report_fixed(shim_db, 3), "psd": report_fixed_exponent(sd_period, -3, 6)}
