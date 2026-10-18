"""Numpy port of the Praat 6.1.38 primitives the ABI script calls.

Every function names the Praat routine it follows (source tag v6.1.38,
the version bundled with parselmouth 0.4.7). Index conventions are kept
1-based in comments and converted at the array boundary; floating-point
expressions keep Praat's operation order where it affects branching
(window edges, sample-index rounding).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

UNDEF = float("nan")
_EPS = 2.0 ** -53                      # LAPACK dlamch("Epsilon"), round-to-nearest
_SQRT_EPS = math.sqrt(_EPS)
_GOLDEN = 1.0 - 0.6180339887498948482045868343656381177203


class PraatError(RuntimeError):
    pass


def iround(x: float) -> int:
    """Melder_iround: tie up."""
    return int(math.floor(x + 0.5))


def fma(a: float, b: float, c: float) -> float:
    """Correctly rounded a*b + c.

    The parselmouth 0.4.7 arm64 build contracts Praat's ``x1 + (i - 1) * dx``
    (Sampled_indexToX and friends) into a fused multiply-add; the single
    rounding decides ties in later Melder_iround calls, so it is reproduced
    exactly here.
    """
    a, b, c = float(a), float(b), float(c)    # numpy ints would overflow inside Fraction
    if math.isfinite(a) and math.isfinite(b) and math.isfinite(c):
        return float(Fraction(a) * Fraction(b) + Fraction(c))
    return a * b + c


def index_to_x(x1: float, dx: float, index) -> float:
    """Sampled_indexToX with FMA contraction: x1 + (index - 1) * dx."""
    return fma(float(index - 1), dx, x1)


def index_to_x_array(x1: float, dx: float, count: int, first: int = 1) -> np.ndarray:
    return np.array([index_to_x(x1, dx, first + k) for k in range(count)])


def c_round(x: float) -> float:
    """C ``round``: ties away from zero (Sound_createSimple)."""
    return math.copysign(math.floor(abs(x) + 0.5), x)


# --------------------------------------------------------------------------
# Sampled objects
# --------------------------------------------------------------------------

@dataclass
class Sound:
    z: np.ndarray        # samples, 0-based array of Praat z[1][1..nx]
    xmin: float
    xmax: float
    x1: float
    dx: float

    @property
    def nx(self) -> int:
        return int(self.z.size)

    @classmethod
    def from_samples(cls, z, sample_rate: float) -> "Sound":
        z = np.asarray(z, dtype=np.float64).copy()
        dx = 1.0 / sample_rate
        return cls(z, 0.0, z.size / sample_rate, 0.5 * dx, dx)

    def copy(self) -> "Sound":
        return Sound(self.z.copy(), self.xmin, self.xmax, self.x1, self.dx)


def x_to_low_index(x1, dx, x):
    return math.floor((x - x1) / dx + 1.0)


def x_to_high_index(x1, dx, x):
    return math.ceil((x - x1) / dx + 1.0)


def x_to_nearest_index(x1, dx, x):
    return iround((x - x1) / dx + 1.0)


def window_samples(x1, dx, nx, xmin, xmax):
    """Sampled_getWindowSamples -> (imin, imax, count), 1-based."""
    imin = 1 + math.ceil((xmin - x1) / dx)
    imax = 1 + math.floor((xmax - x1) / dx)
    imin = max(imin, 1)
    imax = min(imax, nx)
    return imin, imax, (imax - imin + 1 if imin <= imax else 0)


def short_term_analysis(x1, dx, nx, window_duration, time_step):
    """Sampled_shortTermAnalysis -> (numberOfFrames, firstTime)."""
    my_duration = dx * nx
    if window_duration > my_duration:
        raise PraatError("shorter than window length.")
    n = math.floor((my_duration - window_duration) / time_step) + 1
    our_mid = x1 - 0.5 * dx + 0.5 * my_duration
    return n, our_mid - 0.5 * (n * time_step) + 0.5 * time_step


def _autowindow(dom_lo, dom_hi, xmin, xmax):
    if xmin >= xmax:
        return dom_lo, dom_hi
    return xmin, xmax


def _intersect(dom_lo, dom_hi, xmin, xmax):
    """Function_intersectRangeWithDomain for xmin < xmax."""
    if xmin == xmax:
        return None
    xmin = max(xmin, dom_lo)
    xmax = min(xmax, dom_hi)
    if xmax <= xmin:
        return None
    return xmin, xmax


# --------------------------------------------------------------------------
# Interpolation (NUMinterpol.cpp)
# --------------------------------------------------------------------------

def _sinc_group(y, x, ml, depth):
    """Sinc branch of NUM_interpolate_sinc for one fixed depth >= 3."""
    j = np.arange(depth, dtype=np.float64)
    sign = np.where(np.arange(depth) % 2 == 0, 1.0, -1.0)
    out = np.empty(x.size)
    step = max(1, 2_000_000 // (2 * depth))
    for s in range(0, x.size, step):
        xs, mls = x[s:s + step], ml[s:s + step]
        frac = xs - mls
        a0 = math.pi * frac
        aL = a0[:, None] + math.pi * j
        wL = (0.5 * np.sin(a0))[:, None] * sign / aL * (1.0 + np.cos(aL / (frac + depth)[:, None]))
        idxL = mls[:, None] - np.arange(depth)          # 1-based
        a0r = math.pi * (mls + 1 - xs)
        aR = a0r[:, None] + math.pi * j
        wR = (0.5 * np.sin(a0r))[:, None] * sign / aR * (1.0 + np.cos(aR / (depth + 1.0 - frac)[:, None]))
        idxR = mls[:, None] + 1 + np.arange(depth)
        out[s:s + step] = (y[idxL - 1] * wL).sum(axis=1) + (y[idxR - 1] * wR).sum(axis=1)
    return out


def interpolate_sinc(y: np.ndarray, x, max_depth: int):
    """NUM_interpolate_sinc for an array of 1-based real indices ``x``."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.empty(x.size)
    if n < 1:
        out[:] = UNDEF
        return out
    ml = np.floor(x).astype(np.int64)
    lo, hi = x < 1, x > n
    out[lo] = y[0]
    out[hi] = y[-1]
    exact = ~lo & ~hi & (x == ml)
    out[exact] = y[ml[exact] - 1]
    rest = ~(lo | hi | exact)
    depth = np.minimum(np.minimum(max_depth, ml), n - ml)   # midright-1 == ml
    for d in np.unique(depth[rest]):
        sel = rest & (depth == d)
        xs, mls = x[sel], ml[sel]
        if d <= 0:
            out[sel] = y[np.floor(xs + 0.5).astype(np.int64) - 1]
        elif d == 1:
            out[sel] = y[mls - 1] + (xs - mls) * (y[mls] - y[mls - 1])
        elif d == 2:
            yl, yr = y[mls - 1], y[mls]
            dyl = 0.5 * (yr - y[mls - 2])
            dyr = 0.5 * (y[mls + 1] - yl)
            fil, fir = xs - mls, mls + 1 - xs
            out[sel] = yl * fir + yr * fil - fil * fir * (0.5 * (dyr - dyl) + (fil - 0.5) * (dyl + dyr - 2 * (yr - yl)))
        else:
            out[sel] = _sinc_group(y, xs, mls, int(d))
    return out


def interpolate_sinc_scalar(y, x, max_depth):
    return float(interpolate_sinc(y, np.array([x]), max_depth)[0])


def minimize_brent(f, a, b, tol):
    """NUMminimize_brent -> (x, fx)."""
    v = a + _GOLDEN * (b - a)
    fv = f(v)
    x = w = v
    fx = fw = fv
    for _ in range(60):
        middle = (a + b) / 2.0
        tol_act = _SQRT_EPS * abs(x) + tol / 3.0
        rng = b - a
        if abs(x - middle) + rng / 2.0 <= 2.0 * tol_act:
            return x, fx
        new_step = _GOLDEN * ((b - x) if x < middle else (a - x))
        if abs(x - w) >= tol_act:
            t = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * t
            q = 2.0 * (q - t)
            if q > 0.0:
                p = -p
            else:
                q = -q
            if abs(p) < abs(new_step * q) and p > q * (a - x + 2.0 * tol_act) and p < q * (b - x - 2.0 * tol_act):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0.0 else -tol_act
        t = x + new_step
        ft = f(t)
        if ft <= fx:
            if t < x:
                b = x
            else:
                a = x
            v, w, x = w, x, t
            fv, fw, fx = fw, fx, ft
        else:
            if t < x:
                a = t
            else:
                b = t
            if ft <= fw or w == x:
                v, w = w, t
                fv, fw = fw, ft
            elif ft <= fv or v == x or v == w:
                v, fv = t, ft
    return x, fx


PEAK_NONE, PEAK_PARABOLIC, PEAK_CUBIC, PEAK_SINC70, PEAK_SINC700 = 0, 1, 2, 3, 4
_PEAK_TO_VALUE_DEPTH = {PEAK_CUBIC: 2, PEAK_SINC70: 70, PEAK_SINC700: 700}


def improve_extremum(y, ixmid, depth, is_maximum):
    """NUMimproveExtremum -> (value, ixmid_real); ``ixmid`` 1-based."""
    n = y.size
    if ixmid <= 1:
        return float(y[0]), 1.0
    if ixmid >= n:
        return float(y[-1]), float(n)
    if depth <= PEAK_NONE:
        return float(y[ixmid - 1]), float(ixmid)
    if depth == PEAK_PARABOLIC:
        dy = 0.5 * (y[ixmid] - y[ixmid - 2])
        d2y = 2 * y[ixmid - 1] - y[ixmid - 2] - y[ixmid]
        return float(y[ixmid - 1] + 0.5 * dy * dy / d2y), ixmid + dy / d2y
    vdepth = _PEAK_TO_VALUE_DEPTH[depth]
    sgn = -1.0 if is_maximum else 1.0
    xr, fr = minimize_brent(lambda t: sgn * interpolate_sinc_scalar(y, t, vdepth),
                            ixmid - 1, ixmid + 1, 1e-10)
    return (-fr if is_maximum else fr), xr


def vector_extremum_and_x(y, x1, dx, dom_lo, dom_hi, xmin, xmax, interp, is_maximum):
    """Vector_getMaximumAndX / Vector_getMinimumAndX on a single channel."""
    nx = y.size
    xmin, xmax = _autowindow(dom_lo, dom_hi, xmin, xmax)
    imin, imax, count = window_samples(x1, dx, nx, xmin, xmax)
    better = (lambda a, b: a > b) if is_maximum else (lambda a, b: a < b)
    if count == 0:
        depth = 1 if interp > PEAK_NONE else 0
        yl = interpolate_sinc_scalar(y, (xmin - x1) / dx + 1.0, depth)
        yr = interpolate_sinc_scalar(y, (xmax - x1) / dx + 1.0, depth)
        val = max(yl, yr) if is_maximum else min(yl, yr)
        xx = (xmin + xmax) / 2.0 if yl == yr else (xmin if better(yl, yr) else xmax)
        return val, xx
    ext, xi = float(y[imin - 1]), float(imin)
    if better(y[imax - 1], ext):
        ext, xi = float(y[imax - 1]), float(imax)
    lo, hi = (imin + 1 if imin == 1 else imin), (imax - 1 if imax == nx else imax)
    seg = y[lo - 2:hi + 1]          # y[lo-1 .. hi+1] (1-based)
    if seg.size >= 3:
        c, l, r = seg[1:-1], seg[:-2], seg[2:]
        cand = np.nonzero((c > l) & (c >= r) if is_maximum else (c < l) & (c <= r))[0] + lo
        for i in cand:
            val, ir = improve_extremum(y, int(i), interp, is_maximum)
            if better(val, ext):
                ext, xi = val, ir
    xx = fma(xi - 1, dx, x1)
    return ext, min(max(xx, xmin), xmax)


def vector_window_extremum(y, x1, dx, dom_lo, dom_hi, xmin, xmax, is_maximum):
    """Value-only Vector_getMaximum/Minimum with interpolation NONE."""
    xmin, xmax = _autowindow(dom_lo, dom_hi, xmin, xmax)
    imin, imax, count = window_samples(x1, dx, y.size, xmin, xmax)
    if count == 0:
        return vector_extremum_and_x(y, x1, dx, dom_lo, dom_hi, xmin, xmax, PEAK_NONE, is_maximum)[0]
    seg = y[imin - 1:imax]
    return float(seg.max() if is_maximum else seg.min())


# --------------------------------------------------------------------------
# Sampled_getMean (Sampled.cpp)
# --------------------------------------------------------------------------

def mean_nointerp(v, x1, dx, dom_lo, dom_hi, xmin, xmax):
    """Sampled_getMean(..., interpolate=false) vectorised over window arrays.

    ``v`` holds the (all defined) values in the requested unit.
    """
    v = np.asarray(v, dtype=np.float64)
    nx = v.size
    xmin = np.atleast_1d(np.asarray(xmin, dtype=np.float64)).copy()
    xmax = np.atleast_1d(np.asarray(xmax, dtype=np.float64)).copy()
    auto = xmin >= xmax
    xmin[auto], xmax[auto] = dom_lo, dom_hi
    ok = xmin != xmax
    xmin = np.maximum(xmin, dom_lo)
    xmax = np.minimum(xmax, dom_hi)
    ok &= xmax > xmin
    rimin = (xmin - x1) / dx + 1.0
    rimax = (xmax - x1) / dx + 1.0
    ok &= (rimax >= 0.5) & (rimin < nx + 0.5)
    imin = np.where(rimin < 0.5, 0, np.floor(rimin + 0.5)).astype(np.int64)
    imax = np.where(rimax >= nx + 0.5, nx + 1, np.floor(rimax + 0.5)).astype(np.int64)
    csum = np.concatenate(([0.0], np.cumsum(v)))           # csum[k] = sum v[1..k]
    lo = np.clip(imin + 1, 1, nx + 1)
    hi = np.clip(imax - 1, 0, nx)
    inner = np.where(hi >= lo, csum[np.maximum(hi, 0)] - csum[np.maximum(lo - 1, 0)], 0.0)
    rng = np.where(hi >= lo, (hi - lo + 1).astype(np.float64), 0.0)
    total, width = inner.copy(), rng.copy()
    same = imin == imax
    vi = v[np.clip(imin, 1, nx) - 1]
    va = v[np.clip(imax, 1, nx) - 1]
    ph = rimax - rimin
    total = np.where(same, ph * vi, total)
    width = np.where(same, ph, width)
    left = ~same & (imin >= 1)
    phl = imin - rimin + 0.5
    total = np.where(left, total + phl * vi, total)
    width = np.where(left, width + phl, width)
    right = ~same & (imax <= nx)
    phr = rimax - imax + 0.5
    total = np.where(right, total + phr * va, total)
    width = np.where(right, width + phr, width)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(ok & (width > 0.0), total / width, UNDEF)
    return out


def interp_mean_weights(x1, dx, nx, dom_lo, dom_hi, xmin, xmax):
    """Sampled_getMean(..., interpolate=true) as a linear functional.

    Returns ``(indices, weights)`` (0-based) with weights already divided by
    the definition range, for a sampled function whose values are all
    defined. ``None`` when the mean is undefined.
    """
    xmin, xmax = _autowindow(dom_lo, dom_hi, xmin, xmax)
    r = _intersect(dom_lo, dom_hi, xmin, xmax)
    if r is None:
        return None
    xmin, xmax = r
    w: dict[int, float] = {}
    rng = 0.0

    def add(i, c):
        w[i] = w.get(i, 0.0) + c

    imin, imax, count = window_samples(x1, dx, nx, xmin, xmax)
    if count > 0:
        left_edge = x1 - 0.5 * dx
        right_edge = left_edge + nx * dx
        for i in range(imin, imax + 1):
            add(i, 1.0)
        rng += count
        if xmin > left_edge:
            phase = (fma(imin - 1, dx, x1) - xmin) / dx
            rng -= 0.5
            add(imin, -0.5)
            if imin - 1 >= 1:
                rng += phase
                add(imin, phase * (1.0 - 0.5 * phase))
                add(imin - 1, phase * 0.5 * phase)
            else:
                phase = min(phase, 0.5)
                rng += phase
                add(imin, phase)
        if xmax < right_edge:
            phase = (xmax - index_to_x(x1, dx, imax)) / dx
            rng -= 0.5
            add(imax, -0.5)
            if imax + 1 <= nx:
                rng += phase
                add(imax, phase * (1.0 - 0.5 * phase))
                add(imax + 1, phase * 0.5 * phase)
            else:
                phase = min(phase, 0.5)
                rng += phase
                add(imax, phase)
    else:
        # no sample centres between xmin and xmax: imin == imax + 1 here
        lv_i, rv_i = imax, imin
        p1 = (xmin - index_to_x(x1, dx, imax)) / dx
        p2 = (xmax - index_to_x(x1, dx, imax)) / dx
        if imin == imax + 1:
            ldef, rdef = 1 <= lv_i <= nx, 1 <= rv_i <= nx
            if ldef and rdef:
                rng += p2 - p1
                c = 0.5 * (p1 + p2)
                add(lv_i, (p2 - p1) * (1.0 - c))
                add(rv_i, (p2 - p1) * c)
            elif ldef and p1 < 0.5:
                p2 = min(p2, 0.5)
                rng += p2 - p1
                add(lv_i, p2 - p1)
            elif rdef and p2 > 0.5:
                p1 = max(p1, 0.5)
                rng += p2 - p1
                add(rv_i, p2 - p1)
    if rng <= 0.0:
        return None
    idx = np.fromiter(w.keys(), dtype=np.int64) - 1
    wt = np.fromiter(w.values(), dtype=np.float64) / rng
    return idx, wt


# --------------------------------------------------------------------------
# Sound operations
# --------------------------------------------------------------------------

def _to_old_order(spec: np.ndarray, n: int) -> np.ndarray:
    """numpy rfft -> Praat NUMrealft order [DC, Nyq, re1, im1, ...] (n even)."""
    p = np.empty(n)
    p[0] = spec[0].real
    p[1] = spec[n // 2].real
    p[2::2] = spec[1:n // 2].real
    p[3::2] = spec[1:n // 2].imag
    return p


def _from_old_order(p: np.ndarray, n: int) -> np.ndarray:
    spec = np.empty(n // 2 + 1, dtype=np.complex128)
    spec[0] = p[0]
    spec[n // 2] = p[1]
    spec[1:n // 2] = p[2::2] + 1j * p[3::2]
    return spec


def resample(me: Sound, fs: float, precision: int) -> Sound:
    """Sound_resample."""
    upfactor = fs * me.dx
    if abs(upfactor - 2.0) < 1e-6:
        raise NotImplementedError("Sound_upsample path (exact 2x) not ported")
    if abs(upfactor - 1.0) < 1e-6:
        return me.copy()
    n = iround((me.xmax - me.xmin) * fs)
    if n < 1:
        raise PraatError("The resampled Sound would have no samples.")
    src = me
    if upfactor < 1.0:
        anti, nfft = 1000, 1
        while nfft < me.nx + anti * 2:
            nfft *= 2
        data = np.zeros(nfft)
        data[anti:anti + me.nx] = me.z
        p = _to_old_order(np.fft.rfft(data), nfft)
        p[math.floor(upfactor * nfft) - 1:] = 0.0
        p[1] = 0.0
        back = np.fft.irfft(_from_old_order(p, nfft), nfft)
        src = Sound(back[anti:anti + me.nx].copy(), me.xmin, me.xmax, me.x1, me.dx)
    dxn = 1.0 / fs
    x1n = 0.5 * (me.xmin + me.xmax - (n - 1) / fs)
    x = x1n + np.arange(n) * dxn
    index = (x - src.x1) / src.dx + 1.0
    if precision <= 1:
        left = np.floor(index).astype(np.int64)
        frac = index - left
        ok = (left >= 1) & (left < src.nx)
        z = np.zeros(n)
        li = left[ok]
        z[ok] = (1 - frac[ok]) * src.z[li - 1] + frac[ok] * src.z[li]
    else:
        z = interpolate_sinc(src.z, index, precision)
    return Sound(z, me.xmin, me.xmax, x1n, dxn)


def extract_part(me: Sound, tmin, tmax, preserve_times: bool) -> Sound:
    """Sound_extractPart, rectangular window, relative width 1."""
    tmin, tmax = _autowindow(me.xmin, me.xmax, tmin, tmax)
    itmin = 1 + math.ceil((tmin - me.x1) / me.dx)
    itmax = 1 + math.floor((tmax - me.x1) / me.dx)
    if itmax < itmin:
        raise PraatError("Extracted Sound would contain no samples.")
    n = itmax - itmin + 1
    z = np.zeros(n)
    lo, hi = max(1, itmin), min(itmax, me.nx)
    if hi >= lo:
        z[lo - itmin:hi - itmin + 1] = me.z[lo - 1:hi]
    x1 = index_to_x(me.x1, me.dx, itmin)
    xmin, xmax = tmin, tmax
    if not preserve_times:
        xmin, xmax, x1 = 0.0, tmax - tmin, x1 - tmin
    return Sound(z, xmin, xmax, x1, me.dx)


def pre_emphasis(me: Sound, frequency: float) -> None:
    """Sound_preEmphasis (dwtools version), in place."""
    if frequency >= 0.5 / me.dx:
        return
    c = math.exp(-2.0 * math.pi * frequency * me.dx)
    me.z[1:] = me.z[1:] - c * me.z[:-1]


def gaussian_window(n: int) -> np.ndarray:
    """Sound_createGaussian samples for ``n`` points."""
    i = np.arange(1, n + 1, dtype=np.float64)
    imid = 0.5 * (n + 1)
    edge = math.exp(-12.0)
    return (np.exp(-48.0 * (i - imid) * (i - imid) / (n + 1) / (n + 1)) - edge) / (1 - edge)


def into_sound_indices(me: Sound, start_time: float, n: int):
    """Sound_into_Sound: 1-based source indices for ``n`` target samples."""
    index = x_to_nearest_index(me.x1, me.dx, start_time)
    return index + np.arange(n)


def gather(z: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """z[idx] with 1-based ``idx``; zero outside 1..nx."""
    ok = (idx >= 1) & (idx <= z.size)
    out = np.zeros(idx.shape)
    out[ok] = z[idx[ok] - 1]
    return out


@dataclass
class Spectrum:
    re: np.ndarray
    im: np.ndarray
    xmax: float
    dx: float          # x1 = xmin = 0

    @property
    def nx(self):
        return self.re.size


def to_spectrum(me: Sound, fast: bool = True) -> Spectrum:
    """Sound_to_Spectrum (mono)."""
    n = me.nx
    if fast:
        n = 2
        while n < me.nx:
            n *= 2
    data = np.zeros(n)
    data[:me.nx] = me.z
    spec = np.fft.rfft(data)
    return Spectrum(spec.real * me.dx, spec.imag * me.dx, 0.5 / me.dx, 1.0 / (me.dx * n))


def spectrum_to_sound(me: Spectrum) -> Sound:
    """Spectrum_to_Sound."""
    last = (me.nx - 1) * me.dx
    odd = me.im[-1] != 0.0 or me.xmax - last > 0.25 * me.dx
    n = 2 * me.nx - (1 if odd else 2)
    spec = me.re + 1j * me.im
    amp = me.dx * n * np.fft.irfft(spec, n)
    fs = n * me.dx
    duration = 1.0 / me.dx
    return Sound(amp, 0.0, duration, 0.5 / fs, 1.0 / fs)


# --------------------------------------------------------------------------
# PowerCepstrogram / PowerCepstrum (LPC/PowerCepstrogram.cpp, PowerCepstrum.cpp)
# --------------------------------------------------------------------------

@dataclass
class Cepstrogram:
    z: np.ndarray        # (nq, nFrames), power values
    t1: float
    dt: float
    dq: float            # q1 = 0, qmax = (nq - 1) * dq


def to_power_cepstrogram(me: Sound, pitch_floor=60.0, dt=0.002, fmax=5000.0, preemph=50.0) -> Cepstrogram:
    """Sound_to_PowerCepstrogram."""
    window_duration = 2.0 * (3.0 / pitch_floor)
    if window_duration > me.dx * me.nx:
        window_duration = me.dx * me.nx
    fs = 2 * fmax
    sound = resample(me, fs, 50)
    pre_emphasis(sound, preemph)
    n_frames, t1 = short_term_analysis(me.x1, me.dx, me.nx, window_duration, dt)
    nsf = int(c_round(window_duration * fs))
    window = gaussian_window(nsf)
    nfft = 2
    while nfft < nsf:
        nfft *= 2
    nq = nfft // 2 + 1
    qmax = 0.5 * nfft / fs
    times = index_to_x_array(t1, dt, n_frames)
    starts = np.array([x_to_nearest_index(sound.x1, sound.dx, t - window_duration / 2) for t in times])
    idx = starts[:, None] + np.arange(nsf)[None, :]
    frames = gather(sound.z, idx)
    frames -= frames.mean(axis=1, keepdims=True)
    frames *= window
    spec = np.fft.rfft(frames, nfft, axis=1) * (1.0 / fs)        # Sound_to_Spectrum scaling dx
    logspec = np.log(spec.real ** 2 + spec.imag ** 2 + 1e-300)
    dx_spec = fs / nfft
    ceps = dx_spec * nfft * np.fft.irfft(logspec, nfft, axis=1)   # Spectrum_to_Sound
    z = (ceps[:, :nq] ** 2).T.copy()
    return Cepstrogram(z, t1, dt, qmax / (nq - 1))


def _db(z):
    return 10.0 * np.log10(z + 1e-30)


def theil_fit(x: np.ndarray, y: np.ndarray):
    """NUMlineFit_theil, incomplete method; ``y`` may be (n,) or (n, k)."""
    n = x.size
    if n == 1:
        return np.zeros(y.shape[1:]), y[0]
    if n == 2:
        m = (y[1] - y[0]) / (x[1] - x[0])
        return m, y[0] - m * x[0]
    half = n // 2
    n2 = half + 1 if n % 2 == 1 else half
    yy = y if y.ndim > 1 else y[:, None]
    mbs = (yy[n2:n2 + half] - yy[:half]) / (x[n2:n2 + half] - x[:half])[:, None]
    m = _quantile_half(np.sort(mbs, axis=0))
    b = _quantile_half(np.sort(yy - m[None, :] * x[:, None], axis=0))
    if y.ndim == 1:
        return float(m[0]), float(b[0])
    return m, b


def _quantile_half(a: np.ndarray) -> np.ndarray:
    """NUMquantile(sorted a, 0.5) along axis 0."""
    n = a.shape[0]
    if n == 1:
        return a[0]
    place = 0.5 * n + 0.5
    left = min(max(int(math.floor(place)), 1), n - 1)
    lo, hi = a[left - 1], a[left]
    return np.where(hi == lo, lo, lo + (place - left) * (hi - lo))


def _q_axis(nq, dq):
    return np.arange(nq) * dq


def subtract_trend_robust(z: np.ndarray, dq: float) -> np.ndarray:
    """PowerCepstrum_subtractTrend(0.001, 0, LINEAR, ROBUST_FAST) per column.

    ``qendFit <= qstartFit`` makes Praat fit over the whole quefrency domain.
    """
    nq = z.shape[0]
    q = _q_axis(nq, dq)
    y = _db(z)
    m, b = theil_fit(q, y)
    qq = q.copy()
    qq[0] = 0.5 * dq
    diff = y - (np.multiply.outer(qq, m) + b)
    diff = np.maximum(diff, 0.0)
    return np.exp(diff * math.log(10.0) / 10.0) - 1e-30


def cepstrum_peak(zcol: np.ndarray, dq: float, pitch_floor: float, pitch_ceiling: float, interp=PEAK_PARABOLIC):
    """PowerCepstrum_getMaximumAndQuefrency -> (peak dB, quefrency)."""
    y = _db(zcol)
    qmax = (zcol.size - 1) * dq
    return vector_extremum_and_x(y, 0.0, dq, 0.0, qmax, 1.0 / pitch_ceiling, 1.0 / pitch_floor, interp, True)


def _smooth_matrix(n, x1, dx, dom_lo, dom_hi, half):
    """Dense matrix W with (W @ v)[i] = Sampled_getMean(v, x_i - half, x_i + half, interpolate)."""
    from scipy import sparse

    rows, cols, vals = [], [], []
    for i in range(n):
        xm = index_to_x(x1, dx, i + 1)
        r = interp_mean_weights(x1, dx, n, dom_lo, dom_hi, xm - half, xm + half)
        if r is None:
            raise PraatError("undefined smoothing mean")
        rows.append(np.full(r[0].size, i))
        cols.append(r[0])
        vals.append(r[1])
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def cpps(cg: Cepstrogram, xmin: float, xmax: float, time_window=0.01, quef_window=0.001,
         pitch_floor=60.0, pitch_ceiling=330.0) -> float:
    """PowerCepstrogram_getCPPS(no tilt before smoothing, parabolic, 0.001-0 Straight Robust)."""
    z = cg.z
    nq, nt = z.shape
    ymax = (nq - 1) * cg.dq
    if math.floor(time_window / cg.dt) > 1:
        Wt = _smooth_matrix(nt, cg.t1, cg.dt, xmin, xmax, 0.5 * time_window)
        z = (Wt @ z.T).T
    if math.floor(quef_window / cg.dq) > 1:
        dqc = ymax / (nq - 1)                     # PowerCepstrum_create(ymax, ny)
        if quef_window / dqc > 1.0:
            Wq = _smooth_matrix(nq, 0.0, dqc, 0.0, ymax, 0.5 * quef_window)
            z = np.asarray(Wq @ z)
    dqc = ymax / (nq - 1)
    q = _q_axis(nq, dqc)
    m, b = theil_fit(q, _db(z))
    vals = np.empty(nt)
    for j in range(nt):
        peak, qpeak = cepstrum_peak(z[:, j], dqc, pitch_floor, pitch_ceiling)
        vals[j] = peak - (m[j] * qpeak + b[j])
    return float(np.mean(vals))


# --------------------------------------------------------------------------
# Row-wise sinc interpolation and lock-step Brent (vectorised over problems)
# --------------------------------------------------------------------------

def interpolate_sinc_rows(Y: np.ndarray, rows: np.ndarray, x: np.ndarray, max_depth: int) -> np.ndarray:
    """NUM_interpolate_sinc of row ``rows[k]`` of ``Y`` at 1-based ``x[k]``."""
    n = Y.shape[1]
    x = np.asarray(x, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.int64)
    out = np.empty(x.size)
    ml = np.floor(x).astype(np.int64)
    lo, hi = x < 1, x > n
    out[lo] = Y[rows[lo], 0]
    out[hi] = Y[rows[hi], -1]
    exact = ~lo & ~hi & (x == ml)
    out[exact] = Y[rows[exact], ml[exact] - 1]
    rest = ~(lo | hi | exact)
    depth = np.minimum(np.minimum(max_depth, ml), n - ml)
    for d in np.unique(depth[rest]):
        sel = np.nonzero(rest & (depth == d))[0]
        xs, mls, rs = x[sel], ml[sel], rows[sel]
        if d <= 0:
            out[sel] = Y[rs, np.floor(xs + 0.5).astype(np.int64) - 1]
            continue
        if d == 1:
            out[sel] = Y[rs, mls - 1] + (xs - mls) * (Y[rs, mls] - Y[rs, mls - 1])
            continue
        if d == 2:
            yl, yr = Y[rs, mls - 1], Y[rs, mls]
            dyl = 0.5 * (yr - Y[rs, mls - 2])
            dyr = 0.5 * (Y[rs, mls + 1] - yl)
            fil, fir = xs - mls, mls + 1 - xs
            out[sel] = yl * fir + yr * fil - fil * fir * (0.5 * (dyr - dyl) + (fil - 0.5) * (dyl + dyr - 2 * (yr - yl)))
            continue
        d = int(d)
        j = np.arange(d, dtype=np.float64)
        sign = np.where(np.arange(d) % 2 == 0, 1.0, -1.0)
        frac = xs - mls
        a0 = math.pi * frac
        aL = a0[:, None] + math.pi * j
        wL = (0.5 * np.sin(a0))[:, None] * sign / aL * (1.0 + np.cos(aL / (frac + d)[:, None]))
        a0r = math.pi * (mls + 1 - xs)
        aR = a0r[:, None] + math.pi * j
        wR = (0.5 * np.sin(a0r))[:, None] * sign / aR * (1.0 + np.cos(aR / (d + 1.0 - frac)[:, None]))
        idxL = mls[:, None] - np.arange(d)
        idxR = mls[:, None] + 1 + np.arange(d)
        out[sel] = (Y[rs[:, None], idxL - 1] * wL).sum(axis=1) + (Y[rs[:, None], idxR - 1] * wR).sum(axis=1)
    return out


def minimize_brent_many(f, a: np.ndarray, b: np.ndarray, tol: float):
    """NUMminimize_brent run in lock-step for many independent problems.

    ``f(x, idx)`` evaluates problems ``idx`` at points ``x``.
    Returns ``(x, fx)`` arrays.
    """
    a = np.asarray(a, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    m = a.size
    allidx = np.arange(m)
    v = a + _GOLDEN * (b - a)
    fv = f(v, allidx)
    x, w = v.copy(), v.copy()
    fx, fw = fv.copy(), fv.copy()
    active = np.ones(m, dtype=bool)
    for _ in range(60):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        ai, bi, xi, wi, vi = a[idx], b[idx], x[idx], w[idx], v[idx]
        fxi, fwi, fvi = fx[idx], fw[idx], fv[idx]
        middle = (ai + bi) / 2.0
        tol_act = _SQRT_EPS * np.abs(xi) + tol / 3.0
        done = np.abs(xi - middle) + (bi - ai) / 2.0 <= 2.0 * tol_act
        active[idx[done]] = False
        keep = ~done
        idx = idx[keep]
        if idx.size == 0:
            break
        ai, bi, xi, wi, vi = ai[keep], bi[keep], xi[keep], wi[keep], vi[keep]
        fxi, fwi, fvi = fxi[keep], fwi[keep], fvi[keep]
        middle, tol_act = middle[keep], tol_act[keep]
        new_step = _GOLDEN * np.where(xi < middle, bi - xi, ai - xi)
        tryp = np.abs(xi - wi) >= tol_act
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (xi - wi) * (fxi - fvi)
            q = (xi - vi) * (fxi - fwi)
            p = (xi - vi) * q - (xi - wi) * t
            q = 2.0 * (q - t)
            p = np.where(q > 0.0, -p, p)
            q = np.where(q > 0.0, q, -q)
            acc = tryp & (np.abs(p) < np.abs(new_step * q)) & (p > q * (ai - xi + 2.0 * tol_act)) & (p < q * (bi - xi - 2.0 * tol_act))
            new_step = np.where(acc, p / np.where(acc, q, 1.0), new_step)
        small = np.abs(new_step) < tol_act
        new_step = np.where(small, np.where(new_step > 0.0, tol_act, -tol_act), new_step)
        tt = xi + new_step
        ft = f(tt, idx)
        better = ft <= fxi
        # better: shrink around t
        na = np.where(better, np.where(tt < xi, ai, xi), np.where(tt < xi, tt, ai))
        nb = np.where(better, np.where(tt < xi, xi, bi), np.where(tt < xi, bi, tt))
        c1 = ~better & ((ft <= fwi) | (wi == xi))
        c2 = ~better & ~c1 & ((ft <= fvi) | (vi == xi) | (vi == wi))
        nv = np.where(better, wi, np.where(c1, wi, np.where(c2, tt, vi)))
        nfv = np.where(better, fwi, np.where(c1, fwi, np.where(c2, ft, fvi)))
        nw = np.where(better, xi, np.where(c1, tt, wi))
        nfw = np.where(better, fxi, np.where(c1, ft, fwi))
        nx_ = np.where(better, tt, xi)
        nfx = np.where(better, ft, fxi)
        a[idx], b[idx], v[idx], fv[idx], w[idx], fw[idx], x[idx], fx[idx] = na, nb, nv, nfv, nw, nfw, nx_, nfx
    return x, fx
