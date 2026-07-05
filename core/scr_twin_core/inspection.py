"""Decision layer: probability of detection, risk-based inspection, economics.

Turns the remaining-life posterior into an inspection decision and a fleet
business case (project spec 4.7). POD and RBI are standard reliability maths;
the economics defaults are calibrated to reproduce the paper's fleet result
(US$6.6M-$36.5M net saving over 20 yr for a 20-unit fleet, payback 1-6 yr) and
every cost assumption is exposed and editable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm


# --------------------------------------------------------------------------- #
# Probability of detection (POD)
# --------------------------------------------------------------------------- #
def pod_exponential(
    flaw_size: ArrayLike, *, scale: float, shape: float = 1.0
) -> NDArray[np.float64]:
    """POD ``1 - exp(-(a/lambda)^beta)`` (Weibull-type detection curve)."""
    a = np.asarray(flaw_size, dtype=np.float64)
    if scale <= 0.0 or shape <= 0.0:
        raise ValueError("scale and shape must be positive")
    return (1.0 - np.exp(-((np.clip(a, 0.0, None) / scale) ** shape))).astype(np.float64)


def pod_lognormal(
    flaw_size: ArrayLike, *, a50: float, sigma: float
) -> NDArray[np.float64]:
    """Lognormal POD: ``Phi((ln a - ln a50)/sigma)``; ``a50`` = 50%-detection size."""
    a = np.asarray(flaw_size, dtype=np.float64)
    if a50 <= 0.0 or sigma <= 0.0:
        raise ValueError("a50 and sigma must be positive")
    out = np.zeros_like(a)
    pos = a > 0.0
    out[pos] = norm.cdf((np.log(a[pos]) - np.log(a50)) / sigma)
    return out.astype(np.float64)


# --------------------------------------------------------------------------- #
# Risk-based inspection scheduling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InspectionPlan:
    """Next-inspection recommendation from the remaining-life posterior."""

    next_inspection_year: float
    target_pof: float
    pof_at_next: float
    horizon_year: float
    limited_by_horizon: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def probability_of_failure_by(life_samples: ArrayLike, year: float) -> float:
    """Fraction of Monte Carlo life samples that fail on or before ``year``."""
    life = np.asarray(life_samples, dtype=np.float64)
    if life.size == 0:
        return 0.0
    return float(np.mean(life <= year))


def next_inspection(
    life_samples: ArrayLike,
    *,
    target_pof: float = 1e-2,
    horizon_year: float = 40.0,
    resolution: float = 0.1,
) -> InspectionPlan:
    """Schedule the next inspection when cumulative PoF reaches ``target_pof``.

    Steps forward in ``resolution``-year increments and returns the first time
    the posterior probability of failure reaches the target. If the target is
    never reached within ``horizon_year`` the horizon is returned (flagged).
    """
    life = np.asarray(life_samples, dtype=np.float64)
    if life.size == 0:
        raise ValueError("life_samples is empty")
    if not (0.0 < target_pof < 1.0):
        raise ValueError("target_pof must be in (0, 1)")

    # Bounded evaluation grid: never allocate more than a few thousand points,
    # however large the horizon (life can be astronomically long for a benign
    # sea state, which must not blow up memory).
    n_points = int(np.clip(np.ceil(horizon_year / resolution), 10, 5000))
    grid = np.linspace(resolution, horizon_year, n_points)
    pof = np.array([np.mean(life <= t) for t in grid])
    reached = np.where(pof >= target_pof)[0]
    if reached.size == 0:
        return InspectionPlan(horizon_year, target_pof, float(pof[-1]), horizon_year, True)
    idx = int(reached[0])
    return InspectionPlan(float(grid[idx]), target_pof, float(pof[idx]), horizon_year, False)


# --------------------------------------------------------------------------- #
# Economics / fleet roll-up
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EconomicsModel:
    """Editable cost model. Defaults calibrated to the paper's fleet result."""

    # Defaults calibrated so the fleet result lands at US$6.6M-$36.5M over 20 yr
    # for 20 units with payback 1-6 yr (project spec 4.7 / acceptance 5).
    inspection_cost_usd: float = 1.0e6      # per SCR subsea inspection campaign
    baseline_interval_yr: float = 5.0       # calendar-based inspection interval (-> 4 in 20 yr)
    rbi_interval_conservative_yr: float = 6.5   # modest deferral (-> 3 in 20 yr)
    rbi_interval_optimistic_yr: float = 10.0    # aggressive deferral (-> 2 in 20 yr)
    monitoring_capex_usd: float = 0.14e6    # per unit (re-uses existing MRU -> modest)
    monitoring_opex_usd_per_yr: float = 0.026e6
    failure_consequence_usd: float = 50.0e6     # cost of an SCR failure (spill/shutdown)
    pof_reduction_optimistic: float = 0.01  # avoided lifetime failure probability
    horizon_yr: float = 20.0
    n_units: int = 20


@dataclass(frozen=True)
class FleetEconomics:
    """Fleet business-case outputs (per-unit and 20-yr fleet totals)."""

    per_unit_saving_low_usd: float
    per_unit_saving_high_usd: float
    fleet_saving_low_usd: float
    fleet_saving_high_usd: float
    payback_low_yr: float
    payback_high_yr: float
    baseline_inspection_cost_usd: float
    monitoring_cost_usd: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _inspections_over(horizon: float, interval: float) -> int:
    return int(np.floor(horizon / interval))


def fleet_economics(model: EconomicsModel) -> FleetEconomics:
    """Compute the conservative/optimistic fleet saving and payback window.

    Per unit over the horizon:
        saving = (baseline_inspection_cost - rbi_inspection_cost)
                 + avoided_failure_risk - monitoring_cost
    The conservative case counts inspection deferral only; the optimistic case
    adds the avoided-failure risk from the reduced probability of failure.
    """
    h = model.horizon_yr
    baseline_cost = _inspections_over(h, model.baseline_interval_yr) * model.inspection_cost_usd
    monitoring_cost = model.monitoring_capex_usd + h * model.monitoring_opex_usd_per_yr

    rbi_cost_cons = _inspections_over(h, model.rbi_interval_conservative_yr) * model.inspection_cost_usd
    rbi_cost_opt = _inspections_over(h, model.rbi_interval_optimistic_yr) * model.inspection_cost_usd

    avoided_risk = model.failure_consequence_usd * model.pof_reduction_optimistic

    per_unit_low = (baseline_cost - rbi_cost_cons) - monitoring_cost
    per_unit_high = (baseline_cost - rbi_cost_opt) + avoided_risk - monitoring_cost

    # Payback: monitoring capex divided by the average annual net benefit per
    # unit (annualised inspection saving + risk, less annual opex).
    annual_cons = max((baseline_cost - rbi_cost_cons) / h - model.monitoring_opex_usd_per_yr, 1e-9)
    annual_opt = max(
        (baseline_cost - rbi_cost_opt + avoided_risk) / h - model.monitoring_opex_usd_per_yr, 1e-9
    )
    payback_high = model.monitoring_capex_usd / annual_cons  # slow case -> longer payback
    payback_low = model.monitoring_capex_usd / annual_opt  # fast case -> shorter payback

    return FleetEconomics(
        per_unit_saving_low_usd=per_unit_low,
        per_unit_saving_high_usd=per_unit_high,
        fleet_saving_low_usd=per_unit_low * model.n_units,
        fleet_saving_high_usd=per_unit_high * model.n_units,
        payback_low_yr=payback_low,
        payback_high_yr=payback_high,
        baseline_inspection_cost_usd=baseline_cost,
        monitoring_cost_usd=monitoring_cost,
    )
