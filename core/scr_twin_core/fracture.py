"""Fracture mechanics: Paris-law fatigue crack growth (BS 7910 / DNV-RP-C210).

A parallel pathway to the S-N life. A surface flaw at the TDP girth weld is grown
under the same rainflow stress ranges via the Paris law

    da/dN = C (dK)^m,     dK = Y * dsigma * sqrt(pi * a),

from an initial depth ``a0`` to a critical depth ``a_c``. This yields a
crack-based fatigue life to cross-check the S-N life, a crack-depth-vs-time curve
that (with the POD curves) drives inspection planning, and a Monte-Carlo
time-to-critical distribution from an initial-flaw population.

Reduced-order screening: a single membrane geometry factor ``Y`` (flat-plate
approximation - badged), an equivalent constant-amplitude range from the rainflow
histogram, and no crack aspect-ratio evolution. A design ECA needs the full
BS 7910 2-D (a/c) integration with the Newman-Raju solutions.

Units: ``a`` in metres, ``dK`` in MPa*sqrt(m), ``dsigma`` in MPa,
``da/dN`` in m/cycle.

Reference
---------
Paris & Erdogan (1963); BS 7910:2019 Sec. 8 & Annex M (crack-growth law and
stress-intensity solutions); DNV-RP-C210 (probabilistic crack growth / RBI).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ParisMaterial:
    """Paris-law crack-growth parameters (SI: m/cycle, MPa*sqrt(m))."""

    C: float
    m: float
    delta_k_th: float  # threshold stress-intensity range [MPa*sqrt(m)]
    name: str = "custom"

    def __post_init__(self) -> None:
        if self.C <= 0.0 or self.m <= 0.0:
            raise ValueError("C and m must be positive")
        if self.delta_k_th < 0.0:
            raise ValueError("delta_k_th must be non-negative")

    @classmethod
    def bs7910_air_mean(cls) -> ParisMaterial:
        """BS 7910 simplified mean crack-growth law, ferritic steel in air."""
        return cls(C=1.65e-11, m=3.0, delta_k_th=2.0, name="BS7910 air (mean)")

    @classmethod
    def bs7910_marine_cp(cls) -> ParisMaterial:
        """Seawater with cathodic protection (faster growth, lower threshold)."""
        return cls(C=4.8e-11, m=3.0, delta_k_th=1.5, name="BS7910 seawater+CP")


def stress_intensity_range(
    delta_sigma_mpa: ArrayLike, a_m: ArrayLike, *, geometry_factor: float = 1.12
) -> NDArray[np.float64]:
    """Stress-intensity range ``dK = Y * dsigma * sqrt(pi * a)`` [MPa*sqrt(m)]."""
    ds = np.asarray(delta_sigma_mpa, dtype=np.float64)
    a = np.asarray(a_m, dtype=np.float64)
    return (geometry_factor * ds * np.sqrt(np.pi * np.clip(a, 0.0, None))).astype(np.float64)


def equivalent_stress_range_mpa(
    edges_pa: ArrayLike, counts: ArrayLike, m: float
) -> float:
    """Paris-equivalent constant-amplitude range from a rainflow histogram.

    ``dsigma_eq = (sum n_i dsigma_i^m / sum n_i)^(1/m)`` [MPa] - the range that
    grows the crack at the same mean rate as the variable-amplitude spectrum.
    """
    edges = np.asarray(edges_pa, dtype=np.float64)
    n = np.asarray(counts, dtype=np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:]) / 1.0e6  # -> MPa
    total = float(n.sum())
    if total <= 0.0:
        return 0.0
    return float((np.sum(n * centres**m) / total) ** (1.0 / m))


def cycles_to_grow(
    a0: float, a_c: float, material: ParisMaterial, delta_sigma_mpa: float,
    *, geometry_factor: float = 1.12,
) -> float:
    """Cycles to grow a crack from ``a0`` to ``a_c`` (closed form for m != 2).

    Returns ``inf`` when the driving ``dK`` at ``a0`` is below the threshold
    (no propagation) or the stress range is non-positive.
    """
    if not (0.0 < a0 < a_c):
        raise ValueError("require 0 < a0 < a_c")
    if delta_sigma_mpa <= 0.0:
        return float("inf")
    y, m, c = geometry_factor, material.m, material.C
    dk0 = stress_intensity_range(delta_sigma_mpa, a0, geometry_factor=y)
    if float(dk0) < material.delta_k_th:
        return float("inf")
    b = c * (y * delta_sigma_mpa * np.sqrt(np.pi)) ** m
    if abs(m - 2.0) < 1e-9:
        return float(np.log(a_c / a0) / b)
    p = 1.0 - m / 2.0
    return float((a_c**p - a0**p) / (b * p))


def crack_life_years(
    a0: float, a_c: float, material: ParisMaterial, delta_sigma_mpa: float,
    cycles_per_year: float, *, geometry_factor: float = 1.12,
) -> float:
    """Crack-based fatigue life [yr] = cycles-to-critical / annual cycle count."""
    if cycles_per_year <= 0.0:
        return float("inf")
    return cycles_to_grow(a0, a_c, material, delta_sigma_mpa, geometry_factor=geometry_factor) / cycles_per_year


def crack_growth_curve(
    a0: float, material: ParisMaterial, delta_sigma_mpa: float, cycles_per_year: float,
    *, a_c: float, years: float = 40.0, n_steps: int = 200, geometry_factor: float = 1.12,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Crack depth ``a(t)`` over ``years`` (RK4 in time on da/dt = nu * da/dN).

    Integration stops (clamps) at ``a_c``. Returns ``(t_years, a_metres)``.
    """
    t = np.linspace(0.0, years, n_steps)
    dt = t[1] - t[0] if n_steps > 1 else years
    y, m, c = geometry_factor, material.m, material.C

    def dadt(a: float) -> float:
        if a >= a_c:
            return 0.0
        dk = float(stress_intensity_range(delta_sigma_mpa, a, geometry_factor=y))
        if dk < material.delta_k_th:
            return 0.0
        return cycles_per_year * c * dk**m  # da/dN * cycles/yr

    a = np.empty(n_steps)
    a[0] = a0
    for i in range(1, n_steps):
        ai = a[i - 1]
        k1 = dadt(ai)
        k2 = dadt(ai + 0.5 * dt * k1)
        k3 = dadt(ai + 0.5 * dt * k2)
        k4 = dadt(ai + dt * k3)
        a[i] = min(ai + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), a_c)
    return t, a


@dataclass(frozen=True)
class InitialFlawDistribution:
    """Initial crack-depth population (exponential, DNV-RP-C210 style)."""

    mean_depth_m: float = 0.2e-3  # 0.2 mm mean initial defect
    min_depth_m: float = 0.05e-3

    def sample(self, rng: np.random.Generator, n: int) -> NDArray[np.float64]:
        a = self.min_depth_m + rng.exponential(self.mean_depth_m, n)
        return a.astype(np.float64)


def crack_growth_mc(
    dist: InitialFlawDistribution, material: ParisMaterial, delta_sigma_mpa: float,
    cycles_per_year: float, *, a_c: float, n_members: int = 5000, seed: int = 0,
    geometry_factor: float = 1.12,
) -> NDArray[np.float64]:
    """Monte-Carlo time-to-critical [yr] from the initial-flaw population."""
    rng = np.random.default_rng(seed)
    a0 = np.clip(dist.sample(rng, n_members), 1e-6, a_c * 0.99)
    out = np.array([
        crack_life_years(float(a), a_c, material, delta_sigma_mpa, cycles_per_year,
                         geometry_factor=geometry_factor)
        for a in a0
    ])
    return out.astype(np.float64)
