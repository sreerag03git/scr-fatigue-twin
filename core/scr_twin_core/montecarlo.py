"""Layer 3a: Monte Carlo propagation to a remaining-fatigue-life distribution.

A vectorised 10,000-member Monte Carlo propagates documented uncertainty sources
onto the annual damage rate and hence the fatigue-life posterior (PDF/CDF,
P10/P50/P90). A separate routine simulates the year-by-year accumulated-damage
divergence between the design prediction and the actual (random wave-climate)
history, with AR(1) interannual persistence.

Uncertainty sources (all editable and reported; project spec 4.6):
  - S-N scatter        : N ~ N_design * 10^(sigma_logN * Z)         (DNV std)
  - environment factor : F ~ Normal(mean, cov)     -> damage * mean/F
  - transfer gain      : g ~ LogNormal(median 1, cov) -> damage * g^m
  - SCF                : h ~ LogNormal(median 1, cov) -> damage * h^m
  - wave climate       : per-year AR(1) LogNormal multiplier (accumulation only)

Determinism: identical seed + model -> identical output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class UncertaintyModel:
    """Distributions for each uncertainty source (multiplicative on damage).

    ``sn_slope_m`` is the S-N exponent used to convert stress-scale
    uncertainties (transfer gain, SCF) into damage-scale multipliers
    (``damage ~ stress^m``).
    """

    sn_logN_std: float = 0.20
    env_factor_mean: float = 0.656
    env_factor_cov: float = 0.05
    tf_gain_cov: float = 0.10
    scf_cov: float = 0.10
    sn_slope_m: float = 3.0
    # Interannual wave climate (calibrated to the paper's divergence fan: at
    # year 15 the actual/design accumulated-damage divergence spans ~5% (P10) to
    # ~28% (P90); see test_montecarlo.py). median>1 encodes an actual climate
    # slightly harsher than the design assumption.
    wave_climate_median: float = 1.165
    wave_climate_logstd: float = 0.21
    wave_climate_ar1: float = 0.35

    def describe(self) -> dict[str, float]:
        """Return the parameter set for provenance / UI display."""
        return {
            "sn_logN_std": self.sn_logN_std,
            "env_factor_mean": self.env_factor_mean,
            "env_factor_cov": self.env_factor_cov,
            "tf_gain_cov": self.tf_gain_cov,
            "scf_cov": self.scf_cov,
            "sn_slope_m": self.sn_slope_m,
            "wave_climate_median": self.wave_climate_median,
            "wave_climate_logstd": self.wave_climate_logstd,
            "wave_climate_ar1": self.wave_climate_ar1,
        }


@dataclass(frozen=True)
class MonteCarloResult:
    """Remaining-life posterior from the Monte Carlo."""

    life_years: NDArray[np.float64]
    damage_rate_per_year: NDArray[np.float64]
    p10: float
    p50: float
    p90: float
    seed: int
    n_members: int
    parameters: dict[str, float] = field(default_factory=dict)

    def histogram(self, bins: int = 50) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Life histogram ``(counts, edges)`` for plotting the PDF."""
        counts, edges = np.histogram(self.life_years, bins=bins)
        return counts.astype(np.float64), edges.astype(np.float64)

    def cdf(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Empirical CDF ``(sorted_life, probability)``."""
        x = np.sort(self.life_years)
        p = np.arange(1, x.size + 1) / x.size
        return x, p.astype(np.float64)


def _lognormal_unit_median(rng: np.random.Generator, cov: float, size: int) -> NDArray[np.float64]:
    """LogNormal draws with median 1 and the given coefficient of variation."""
    if cov <= 0.0:
        return np.ones(size)
    sigma = np.sqrt(np.log(1.0 + cov**2))
    return np.exp(sigma * rng.standard_normal(size))


def run_monte_carlo(
    nominal_damage_rate_per_year: float,
    model: UncertaintyModel,
    *,
    n_members: int = 10_000,
    seed: int = 0,
) -> MonteCarloResult:
    """Propagate uncertainty to the fatigue-life posterior (vectorised).

    Parameters
    ----------
    nominal_damage_rate_per_year:
        Deterministic (design) annual damage rate ``D0`` [1/yr].
    """
    if nominal_damage_rate_per_year <= 0.0:
        raise ValueError("nominal_damage_rate_per_year must be positive")
    if n_members < 1:
        raise ValueError("n_members must be >= 1")

    rng = np.random.default_rng(seed)
    m = model.sn_slope_m

    z_sn = rng.standard_normal(n_members)
    mult_sn = 10.0 ** (model.sn_logN_std * z_sn)

    f_env = rng.normal(model.env_factor_mean, model.env_factor_cov * model.env_factor_mean, n_members)
    f_env = np.clip(f_env, 1e-3, None)
    mult_env = model.env_factor_mean / f_env

    mult_tf = _lognormal_unit_median(rng, model.tf_gain_cov, n_members) ** m
    mult_scf = _lognormal_unit_median(rng, model.scf_cov, n_members) ** m

    rate = nominal_damage_rate_per_year * mult_sn * mult_env * mult_tf * mult_scf
    life = 1.0 / rate

    p10, p50, p90 = np.percentile(life, [10, 50, 90])
    return MonteCarloResult(
        life_years=life.astype(np.float64),
        damage_rate_per_year=rate.astype(np.float64),
        p10=float(p10),
        p50=float(p50),
        p90=float(p90),
        seed=seed,
        n_members=n_members,
        parameters=model.describe(),
    )


def simulate_wave_climate_multipliers(
    model: UncertaintyModel,
    years: int,
    *,
    n_members: int,
    seed: int,
) -> NDArray[np.float64]:
    """AR(1) LogNormal annual wave-climate damage multipliers, shape (members, years).

    ``xi_t`` is a stationary AR(1) Gaussian with lag-1 correlation ``phi`` and
    stationary std ``logstd``; ``W_t = median * exp(xi_t - logstd^2/2)`` so the
    *median* multiplier is ``wave_climate_median``.
    """
    if years < 1:
        raise ValueError("years must be >= 1")
    rng = np.random.default_rng(seed)
    phi = model.wave_climate_ar1
    s = model.wave_climate_logstd

    xi = np.empty((n_members, years))
    xi[:, 0] = s * rng.standard_normal(n_members)
    innov_scale = s * np.sqrt(1.0 - phi**2)
    for t in range(1, years):
        xi[:, t] = phi * xi[:, t - 1] + innov_scale * rng.standard_normal(n_members)
    return model.wave_climate_median * np.exp(xi - 0.5 * s**2)


def accumulated_damage_divergence(
    model: UncertaintyModel,
    years: int,
    *,
    n_members: int = 10_000,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Signed divergence ``actual/design - 1`` of accumulated damage at ``years``.

    Design accumulates the nominal rate (multiplier 1); actual accumulates the
    random AR(1) wave-climate multipliers. Returns one divergence per member.
    """
    w = simulate_wave_climate_multipliers(model, years, n_members=n_members, seed=seed)
    actual = w.sum(axis=1)
    design = float(years)
    return (actual / design - 1.0).astype(np.float64)
