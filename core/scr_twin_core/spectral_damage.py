"""Frequency-domain fatigue damage: narrow-band, Dirlik, Tovo-Benasciutti.

These estimate fatigue damage directly from the stress-response spectral moments
without reconstructing a time history. They are used both as a fast fatigue
pathway and, by mutual comparison with time-domain rainflow, as a validation
signal (methods must agree within a stated band on the same spectrum).

Conventions
-----------
Damage uses a **single-slope** S-N law ``N = a_bar * S^-m`` with the stress
*range* ``S`` in MPa (this is the form in which spectral methods are usually
written and lets all methods be compared on equal footing). Moments are computed
from a one-sided stress PSD; ``stress_to_mpa`` converts the PSD's stress unit to
MPa (1e-6 for a Pa-based PSD). The peak/upcrossing rates are stress-unit-free.

References
----------
- Bendat (1964) narrow-band approximation.
- T. Dirlik (1985), PhD thesis, Univ. of Warwick (rainflow range PDF).
- Benasciutti & Tovo (2006), Int. J. Fatigue 28 (Tovo-Benasciutti method).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma as gamma_fn

if TYPE_CHECKING:
    from .sn import SNCurve


@dataclass(frozen=True)
class SpectralParams:
    """Rate and bandwidth parameters derived from spectral moments."""

    nu0: float  # mean zero-up-crossing rate sqrt(m2/m0) [Hz]
    nup: float  # peak rate sqrt(m4/m2) [Hz]
    alpha1: float  # m1 / sqrt(m0 m4)   ... bandwidth parameter
    alpha2: float  # m2 / sqrt(m0 m4)   (irregularity factor)


def spectral_params(moments: dict[int, float]) -> SpectralParams:
    """Compute upcrossing/peak rates and bandwidth parameters from moments."""
    m0, m1, m2, m4 = moments[0], moments[1], moments[2], moments[4]
    if m0 <= 0.0 or m2 <= 0.0 or m4 <= 0.0:
        raise ValueError("moments m0, m2, m4 must be positive")
    nu0 = math.sqrt(m2 / m0)
    nup = math.sqrt(m4 / m2)
    denom = math.sqrt(m0 * m4)
    alpha1 = m1 / denom
    alpha2 = m2 / denom
    return SpectralParams(nu0=nu0, nup=nup, alpha1=alpha1, alpha2=alpha2)


def _sigma_mpa(m0: float, stress_to_mpa: float) -> float:
    return math.sqrt(m0) * stress_to_mpa


def narrowband_damage_rate(
    moments: dict[int, float],
    *,
    sn_m: float,
    sn_log_a: float,
    stress_to_mpa: float = 1e-6,
) -> float:
    """Narrow-band damage rate [1/s].

    ``D = nu0 / a_bar * E[S^m]`` with Rayleigh ranges giving
    ``E[S^m] = (2 sqrt(2) sigma)^m Gamma(1 + m/2)``.
    """
    p = spectral_params(moments)
    sigma = _sigma_mpa(moments[0], stress_to_mpa)
    a_bar = 10.0**sn_log_a
    e_sm = (2.0 * math.sqrt(2.0) * sigma) ** sn_m * gamma_fn(1.0 + sn_m / 2.0)
    return p.nu0 / a_bar * e_sm


def dirlik_range_pdf(
    s_mpa: NDArray[np.float64], moments: dict[int, float], *, stress_to_mpa: float = 1e-6
) -> NDArray[np.float64]:
    """Dirlik empirical PDF of rainflow stress *ranges* (S in MPa)."""
    m0, m1, m2, m4 = moments[0], moments[1], moments[2], moments[4]
    sigma = _sigma_mpa(m0, stress_to_mpa)
    alpha2 = m2 / math.sqrt(m0 * m4)
    x_m = (m1 / m0) * math.sqrt(m2 / m4)

    d1 = 2.0 * (x_m - alpha2**2) / (1.0 + alpha2**2)
    r = (alpha2 - x_m - d1**2) / (1.0 - alpha2 - d1 + d1**2)
    d2 = (1.0 - alpha2 - d1 + d1**2) / (1.0 - r)
    d3 = 1.0 - d1 - d2
    q = 1.25 * (alpha2 - d3 - d2 * r) / d1

    z = np.asarray(s_mpa, dtype=np.float64) / (2.0 * sigma)
    pdf = (
        (d1 / q) * np.exp(-z / q)
        + (d2 * z / r**2) * np.exp(-(z**2) / (2.0 * r**2))
        + d3 * z * np.exp(-(z**2) / 2.0)
    ) / (2.0 * sigma)
    return pdf.astype(np.float64)


def dirlik_damage_rate(
    moments: dict[int, float],
    *,
    sn_m: float,
    sn_log_a: float,
    stress_to_mpa: float = 1e-6,
) -> float:
    """Dirlik damage rate [1/s] using the closed-form ``E[S^m]``.

    With ``Z = S/(2 sigma)`` and Dirlik coefficients (D1,D2,D3,Q,R),
    ``E[S^m] = (2 sigma)^m [ D1 Q^m Gamma(m+1) + (D2 R^m + D3) 2^{m/2}
    Gamma(1+m/2) ]``. Counting rate is the peak rate ``nu_p``.
    """
    m0, m1, m2, m4 = moments[0], moments[1], moments[2], moments[4]
    sigma = _sigma_mpa(m0, stress_to_mpa)
    p = spectral_params(moments)
    alpha2 = p.alpha2
    x_m = (m1 / m0) * math.sqrt(m2 / m4)

    d1 = 2.0 * (x_m - alpha2**2) / (1.0 + alpha2**2)
    r = (alpha2 - x_m - d1**2) / (1.0 - alpha2 - d1 + d1**2)
    d2 = (1.0 - alpha2 - d1 + d1**2) / (1.0 - r)
    d3 = 1.0 - d1 - d2
    q = 1.25 * (alpha2 - d3 - d2 * r) / d1

    m_ = sn_m
    e_sm = (2.0 * sigma) ** m_ * (
        d1 * q**m_ * gamma_fn(m_ + 1.0)
        + (d2 * r**m_ + d3) * 2.0 ** (m_ / 2.0) * gamma_fn(1.0 + m_ / 2.0)
    )
    a_bar = 10.0**sn_log_a
    return p.nup / a_bar * e_sm


def dirlik_damage_rate_curve(
    moments: dict[int, float],
    curve: SNCurve,
    *,
    stress_to_mpa: float = 1e-6,
    thickness_m: float | None = None,
    n_grid: int = 4000,
    s_max_sigma: float = 12.0,
) -> float:
    """Dirlik damage rate [1/s] against a full (two-slope) DNV S-N curve.

    Numerically integrates the Dirlik rainflow-range PDF against the actual
    cycles-to-failure ``N(S)`` so the spectral pathway uses the *same* S-N law as
    the time-domain pathway (removing the single-slope approximation of
    :func:`dirlik_damage_rate`)::

        D = nu_p * integral p(S) / N(S) dS

    ``s_max_sigma`` bounds the range grid at that many RMS stresses (2 sqrt(2)
    sigma is the RMS range); the Dirlik tail beyond is negligible.
    """
    from .sn import cycles_to_failure  # local import avoids a module cycle

    m0 = moments[0]
    sigma = _sigma_mpa(m0, stress_to_mpa)
    if sigma <= 0.0:
        return 0.0
    p = spectral_params(moments)
    s_max = s_max_sigma * 2.0 * math.sqrt(2.0) * sigma
    s_mpa = np.linspace(s_max / n_grid, s_max, n_grid)
    pdf = dirlik_range_pdf(s_mpa, moments, stress_to_mpa=stress_to_mpa)
    # The Dirlik range S is always in MPa here; N(S) expects Pa (fixed x1e6).
    n_fail = cycles_to_failure(s_mpa * 1.0e6, curve, thickness_m=thickness_m)
    integrand = np.where(np.isfinite(n_fail), pdf / n_fail, 0.0)
    return float(p.nup * np.trapezoid(integrand, s_mpa))


def tovo_benasciutti_damage_rate(
    moments: dict[int, float],
    *,
    sn_m: float,
    sn_log_a: float,
    stress_to_mpa: float = 1e-6,
) -> float:
    """Tovo-Benasciutti damage rate [1/s].

    ``D_TB = [b + (1-b) alpha2^{m-1}] * D_NB`` with the empirical weight (Benasciutti
    & Tovo, 2006)::

        b = (a1 - a2)[1.112(1 + a1 a2 - (a1 + a2)) e^{2.11 a2} + (a1 - a2)] / (a2 - 1)^2

    where ``a1 = alpha1``, ``a2 = alpha2``.
    """
    p = spectral_params(moments)
    a1, a2 = p.alpha1, p.alpha2
    d_nb = narrowband_damage_rate(
        moments, sn_m=sn_m, sn_log_a=sn_log_a, stress_to_mpa=stress_to_mpa
    )
    if abs(a2 - 1.0) < 1e-9:
        return d_nb  # perfectly narrow-band limit
    b = (
        (a1 - a2)
        * (1.112 * (1.0 + a1 * a2 - (a1 + a2)) * math.exp(2.11 * a2) + (a1 - a2))
        / (a2 - 1.0) ** 2
    )
    b = min(max(b, 0.0), 1.0)
    factor = b + (1.0 - b) * a2 ** (sn_m - 1.0)
    return factor * d_nb
