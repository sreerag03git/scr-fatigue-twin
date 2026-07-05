"""Spectral analysis: Welch PSD, spectral moments, JONSWAP / Pierson-Moskowitz.

Everything is expressed in ordinary frequency ``f`` [Hz] (not angular frequency)
so that MRU sample rates and wave periods read naturally. Spectral moments are

    m_n = integral f^n S(f) df

evaluated by trapezoidal integration over the supplied one-sided frequency grid.

References
----------
- P.D. Welch (1967), IEEE Trans. Audio Electroacoust. 15(2) (PSD estimation).
- Hasselmann et al. (1973), JONSWAP spectrum; DNV-RP-C205 Sec. 3.5.5.
- Pierson & Moskowitz (1964), J. Geophys. Res. 69(24).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, signal

from .constants import G


@dataclass(frozen=True)
class SeaState:
    """Summary wave/motion parameters recovered from a spectrum."""

    hs: float  # significant height/amplitude = 4*sqrt(m0)
    tp: float  # peak period = 1/f_peak
    tz: float  # mean zero-up-crossing period = sqrt(m0/m2)
    gamma: float  # fitted JONSWAP peak-enhancement (1.0 => Pierson-Moskowitz)


def welch_psd(
    x: ArrayLike,
    fs: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    detrend: str | bool = "constant",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One-sided power spectral density via Welch's method.

    Thin, validated wrapper over :func:`scipy.signal.welch` returning
    ``(f, Pxx)`` on a one-sided grid. ``nperseg`` defaults to a segmenting that
    yields ~8 averages (good bias/variance trade-off) but never exceeds the
    series length.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2:
        raise ValueError("Need at least two samples for a PSD")
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    if nperseg is None:
        nperseg = int(min(x.size, max(256, x.size // 8)))
    nperseg = min(nperseg, x.size)
    f, pxx = signal.welch(
        x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=detrend, scaling="density"
    )
    return f.astype(np.float64), pxx.astype(np.float64)


def spectral_moments(
    f: ArrayLike, s: ArrayLike, orders: tuple[int, ...] = (0, 1, 2, 4)
) -> dict[int, float]:
    """Spectral moments ``m_n = integral f^n S(f) df`` by trapezoidal rule.

    Only the supplied (assumed one-sided, non-negative) frequency grid is used;
    with the physical ``f^-5`` spectral tail ``m4`` converges only because the
    grid is band-limited, which is the intended behaviour for response spectra.
    """
    f = np.asarray(f, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if f.shape != s.shape:
        raise ValueError("f and s must have the same shape")
    out: dict[int, float] = {}
    for n in orders:
        out[n] = float(np.trapezoid(f**n * s, f))
    return out


def significant_from_m0(m0: float) -> float:
    """Significant amplitude ``Hs = 4*sqrt(m0)`` (Hs = 4 sigma)."""
    return 4.0 * np.sqrt(max(m0, 0.0))


def jonswap_alpha(hs: float, tp: float, gamma: float) -> float:
    """Phillips constant ``alpha`` for the closed-form JONSWAP (DNV-RP-C205).

    ``alpha = 5.061 (Hs^2 / Tp^4) (1 - 0.287 ln gamma)`` with Hs in m, Tp in s.
    Used when an *un-normalised* physical spectrum is requested.
    """
    return 5.061 * hs**2 / tp**4 * (1.0 - 0.287 * np.log(gamma))


def jonswap(
    f: ArrayLike,
    hs: float,
    tp: float,
    gamma: float = 3.3,
    *,
    g: float = G,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """JONSWAP spectral density ``S(f)`` [unit^2/Hz].

    Shape (DNV-RP-C205, frequency form)::

        S(f) = alpha g^2 (2 pi)^-4 f^-5 exp[-1.25 (fp/f)^4] * gamma^r
        r    = exp[-(f - fp)^2 / (2 sigma^2 fp^2)],  sigma = 0.07 (f<=fp) else 0.09

    Parameters
    ----------
    normalize:
        When ``True`` (default) the spectrum is scaled so that ``4 sqrt(m0)``
        equals ``hs`` exactly over the supplied grid (the reproducible, grid-
        consistent choice). When ``False`` the physical Phillips ``alpha`` from
        :func:`jonswap_alpha` is used and ``Hs`` is only recovered
        approximately.
    """
    if tp <= 0.0:
        raise ValueError("tp must be positive")
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1 (gamma = 1 is Pierson-Moskowitz)")
    f = np.asarray(f, dtype=np.float64)
    fp = 1.0 / tp
    s = np.zeros_like(f)
    pos = f > 0.0
    fpos = f[pos]
    sigma = np.where(fpos <= fp, 0.07, 0.09)
    r = np.exp(-((fpos - fp) ** 2) / (2.0 * sigma**2 * fp**2))
    alpha = jonswap_alpha(hs, tp, gamma) if not normalize else 1.0
    base = alpha * g**2 * (2.0 * np.pi) ** -4 * fpos**-5 * np.exp(-1.25 * (fp / fpos) ** 4)
    s[pos] = base * gamma**r
    if normalize:
        m0 = float(np.trapezoid(s, f))
        if m0 > 0.0:
            s *= (hs / 4.0) ** 2 / m0
    return s


def pierson_moskowitz(
    f: ArrayLike, hs: float, tp: float, *, g: float = G, normalize: bool = True
) -> NDArray[np.float64]:
    """Pierson-Moskowitz spectrum: the JONSWAP limit ``gamma = 1``."""
    return jonswap(f, hs, tp, gamma=1.0, g=g, normalize=normalize)


def fit_jonswap(
    f: ArrayLike, s: ArrayLike, *, gamma_bounds: tuple[float, float] = (1.0, 7.0)
) -> SeaState:
    """Identify (Hs, Tp, gamma) from a measured/estimated spectrum.

    Hs comes from ``4 sqrt(m0)``; Tp from the spectral peak; gamma from a 1-D
    least-squares fit of the normalised JONSWAP shape to ``s``. Robust to noisy
    peaks (guards empty/degenerate input).
    """
    f = np.asarray(f, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    moments = spectral_moments(f, s, (0, 2))
    m0, m2 = moments[0], moments[2]
    hs = significant_from_m0(m0)
    tz = float(np.sqrt(m0 / m2)) if m2 > 0.0 else float("nan")

    # Peak period from the spectral maximum (ignore the f=0 bin).
    valid = f > 0.0
    if not np.any(valid) or m0 <= 0.0:
        return SeaState(hs=hs, tp=float("nan"), tz=tz, gamma=1.0)
    fpk = f[valid][int(np.argmax(s[valid]))]
    tp = float(1.0 / fpk) if fpk > 0.0 else float("nan")

    def residual(gamma: float) -> float:
        model = jonswap(f, hs, tp, gamma=float(gamma), normalize=True)
        return float(np.sum((model - s) ** 2))

    res = optimize.minimize_scalar(residual, bounds=gamma_bounds, method="bounded")
    gamma = float(res.x) if res.success else 3.3
    return SeaState(hs=hs, tp=tp, tz=tz, gamma=gamma)
