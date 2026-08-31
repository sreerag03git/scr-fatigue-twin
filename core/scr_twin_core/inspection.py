"""Decision layer: probability of detection, risk-based inspection, economics.

Turns the remaining-life posterior into an inspection decision and a *conditional*
fleet business case (project spec 4.7). POD and RBI are standard reliability maths.

The economics deliberately does NOT quote a flat headline saving: the value of
condition-based monitoring (CBM) is conditional on ``phi``, the probability the
asset is actually ageing *slower* than its design assumption. If it ages slower
you may defer inspections and save; if it ages faster you must inspect *more* and
the sensor is a net cost. The discounted decision metric (paper Eq. 11) is::

    dC = sum_t [C_base(t) - C_cbm(t)] / (1+r)^t  -  C_sensor

with C_cbm the phi-weighted mixture of the "ages slower" (defer) and "ages faster"
(inspect more) inspection schedules. ``phi`` is estimated *endogenously* from the
fleet's own remaining-life posterior (``phi = P(life > design_life)``), and the
break-even ``phi*`` at which ``dC = 0`` is reported so the reader sees exactly how
confident they must be before the sensor pays. Every cost/rate assumption is
exposed and editable.
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
# Conditional economics (paper Eq. 11): discounted value of monitoring vs phi
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConditionalEconomicsModel:
    """Editable discounted cost model for the CBM business case (Eq. 11).

    The value of the sensor is *conditional* on ``phi`` = P(asset ages slower than
    design). Two adaptive schedules bracket the calendar baseline: if the twin
    shows the asset ageing slower it defers to ``cbm_interval_slow_yr``; if faster
    it tightens to ``cbm_interval_fast_yr`` (inspect more). All cash flows are
    discounted at ``discount_rate``. Defaults are representative offshore values
    (a subsea SCR inspection campaign ~US$1M; a 5-yr calendar interval; a sensor
    that re-uses the existing MRU so capex is integration-only).
    """

    discount_rate: float = 0.08              # r [1/yr]
    inspection_cost_usd: float = 1.0e6       # per SCR subsea inspection campaign
    baseline_interval_yr: float = 5.0        # calendar-based inspection interval
    cbm_interval_slow_yr: float = 8.0        # defer when the twin shows slower ageing
    cbm_interval_fast_yr: float = 4.0        # tighten when the twin shows faster ageing
    sensor_capex_usd: float = 0.14e6         # per unit (re-uses existing MRU -> modest)
    sensor_opex_usd_per_yr: float = 0.026e6  # analytics / upkeep
    horizon_yr: float = 20.0
    n_units: int = 20
    cost_cov: float = 0.25                   # lognormal COV on costs for the P5-P95 band

    def __post_init__(self) -> None:
        if not (-0.99 < self.discount_rate < 1.0):
            raise ValueError("discount_rate must be in (-0.99, 1.0)")
        for name in (
            "inspection_cost_usd", "sensor_capex_usd", "sensor_opex_usd_per_yr",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("baseline_interval_yr", "cbm_interval_slow_yr", "cbm_interval_fast_yr"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.horizon_yr <= 0.0:
            raise ValueError("horizon_yr must be positive")
        if self.n_units < 1:
            raise ValueError("n_units must be >= 1")
        if self.cost_cov < 0.0:
            raise ValueError("cost_cov must be non-negative")


@dataclass(frozen=True)
class ConditionalEconomics:
    """Conditional (phi-dependent), discounted CBM business-case outputs."""

    discount_rate: float
    horizon_yr: float
    n_units: int
    breakeven_phi: float               # phi at which discounted dC = 0
    phi: float                         # fleet operating point (endogenous or break-even)
    phi_is_endogenous: bool            # True when phi came from the life posterior
    per_unit_delta_c_usd: float        # discounted dC per unit at phi (Eq. 11)
    fleet_delta_c_usd: float           # per-unit dC x n_units
    net_positive: bool                 # sensor pays for the fleet at this phi
    pv_baseline_usd: float             # PV of the calendar inspection schedule / unit
    pv_sensor_usd: float               # PV of sensor capex + opex / unit
    pv_cbm_slow_usd: float             # PV of CBM inspections if certainly slower / unit
    pv_cbm_fast_usd: float             # PV of CBM inspections if certainly faster / unit
    phi_grid: list[float]              # Fig 11(b) x-axis
    fleet_delta_c_p50_usd: list[float]  # median fleet dC(phi)
    fleet_delta_c_p5_usd: list[float]   # P5 cost-uncertainty band
    fleet_delta_c_p95_usd: list[float]  # P95 cost-uncertainty band

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _pv_inspection_schedule(
    interval_yr: float, horizon_yr: float, rate: float, unit_cost: float
) -> float:
    """Present value of a fixed-interval inspection schedule over the horizon.

    Inspections fall at ``interval, 2*interval, ...`` up to (and including) the
    horizon; each is discounted at ``rate`` to its real inspection time.
    """
    if interval_yr <= 0.0:
        raise ValueError("interval_yr must be positive")
    times = np.arange(interval_yr, horizon_yr + 1e-9, interval_yr)
    if times.size == 0:
        return 0.0
    return float(unit_cost * np.sum((1.0 + rate) ** (-times)))


def _pv_annuity(rate: float, horizon_yr: float) -> float:
    """PV of US$1/yr paid at the end of each of ``floor(horizon)`` years."""
    years = np.arange(1, int(np.floor(horizon_yr)) + 1)
    if years.size == 0:
        return 0.0
    return float(np.sum((1.0 + rate) ** (-years.astype(np.float64))))


def phi_ageing_slower(life_samples: ArrayLike, design_life_years: float) -> float:
    """Endogenous ``phi = P(asset ages slower than design) = mean(life > design)``.

    Estimated from the fleet's own remaining-life posterior. Returns ``nan`` when
    the design life is not a finite positive number (e.g. a benign sea state gives
    an infinite deterministic life), so the caller can fall back to the break-even.
    """
    life = np.asarray(life_samples, dtype=np.float64)
    if life.size == 0 or not np.isfinite(design_life_years) or design_life_years <= 0.0:
        return float("nan")
    return float(np.mean(life > design_life_years))


def _present_values(model: ConditionalEconomicsModel, *, c_insp: float, capex: float, opex: float) -> tuple[float, float, float, float]:
    """(pv_baseline, pv_cbm_slow, pv_cbm_fast, pv_sensor) per unit for given costs."""
    r, h = model.discount_rate, model.horizon_yr
    pv_base = _pv_inspection_schedule(model.baseline_interval_yr, h, r, c_insp)
    pv_slow = _pv_inspection_schedule(model.cbm_interval_slow_yr, h, r, c_insp)
    pv_fast = _pv_inspection_schedule(model.cbm_interval_fast_yr, h, r, c_insp)
    pv_sensor = capex + opex * _pv_annuity(r, h)
    return pv_base, pv_slow, pv_fast, pv_sensor


def _delta_c_per_unit(pv_base: float, pv_slow: float, pv_fast: float, pv_sensor: float, phi: float) -> float:
    """Discounted dC per unit (Eq. 11): base PV minus phi-mixed CBM PV minus sensor."""
    pv_cbm = phi * pv_slow + (1.0 - phi) * pv_fast
    return pv_base - pv_cbm - pv_sensor


def breakeven_phi(model: ConditionalEconomicsModel) -> float:
    """phi at which the discounted dC crosses zero (closed form; dC is linear in phi)."""
    pv_base, pv_slow, pv_fast, pv_sensor = _present_values(
        model,
        c_insp=model.inspection_cost_usd,
        capex=model.sensor_capex_usd,
        opex=model.sensor_opex_usd_per_yr,
    )
    denom = pv_fast - pv_slow
    if abs(denom) < 1e-30:
        return float("nan")
    return float((pv_sensor - pv_base + pv_fast) / denom)


def fleet_economics_conditional(
    model: ConditionalEconomicsModel,
    *,
    phi: float | None = None,
    life_samples: ArrayLike | None = None,
    design_life_years: float | None = None,
    n_grid: int = 41,
    n_cost_mc: int = 1500,
    seed: int = 0,
) -> ConditionalEconomics:
    """Conditional CBM business case with the Fig-11(b) phi-vs-dC band.

    ``phi`` (or ``life_samples`` + ``design_life_years`` to derive it endogenously)
    fixes the fleet's operating point; when neither is usable the report falls back
    to the break-even phi (dC = 0). The P5-P95 band sweeps the cost inputs
    lognormally (COV ``model.cost_cov``) at each phi to show cost sensitivity.
    """
    pv_base, pv_slow, pv_fast, pv_sensor = _present_values(
        model,
        c_insp=model.inspection_cost_usd,
        capex=model.sensor_capex_usd,
        opex=model.sensor_opex_usd_per_yr,
    )
    phi_star = breakeven_phi(model)

    # Resolve the operating phi: explicit > endogenous-from-posterior > break-even.
    phi_is_endogenous = False
    if phi is None and life_samples is not None and design_life_years is not None:
        est = phi_ageing_slower(life_samples, design_life_years)
        if np.isfinite(est):
            phi, phi_is_endogenous = est, True
    if phi is None or not np.isfinite(phi):
        phi = phi_star if np.isfinite(phi_star) else 0.0
    phi = float(np.clip(phi, 0.0, 1.0))

    per_unit = _delta_c_per_unit(pv_base, pv_slow, pv_fast, pv_sensor, phi)
    fleet = per_unit * model.n_units

    # Fig 11(b): fleet dC vs phi, with a P5-P95 band from cost uncertainty.
    grid = np.linspace(0.0, 1.0, n_grid)
    if model.cost_cov > 0.0 and n_cost_mc > 0:
        rng = np.random.default_rng(seed)
        sigma = np.sqrt(np.log(1.0 + model.cost_cov**2))
        draw = lambda base: base * np.exp(sigma * rng.standard_normal(n_cost_mc))  # noqa: E731
        c_insp_s, capex_s, opex_s = (
            draw(model.inspection_cost_usd), draw(model.sensor_capex_usd), draw(model.sensor_opex_usd_per_yr),
        )
        ann = _pv_annuity(model.discount_rate, model.horizon_yr)
        r, h = model.discount_rate, model.horizon_yr
        u_base = _pv_inspection_schedule(model.baseline_interval_yr, h, r, 1.0)
        u_slow = _pv_inspection_schedule(model.cbm_interval_slow_yr, h, r, 1.0)
        u_fast = _pv_inspection_schedule(model.cbm_interval_fast_yr, h, r, 1.0)
        pvb = c_insp_s * u_base            # (n_cost_mc,)
        pvs = c_insp_s * u_slow
        pvf = c_insp_s * u_fast
        pvsen = capex_s + opex_s * ann
        # dC(phi) per draw: (grid, n_cost_mc) -> percentiles over draws at each phi.
        cbm = grid[:, None] * pvs[None, :] + (1.0 - grid[:, None]) * pvf[None, :]
        dc = (pvb[None, :] - cbm - pvsen[None, :]) * model.n_units
        p5, p50, p95 = np.percentile(dc, [5, 50, 95], axis=1)
    else:
        p50 = np.array([_delta_c_per_unit(pv_base, pv_slow, pv_fast, pv_sensor, g) * model.n_units for g in grid])
        p5 = p95 = p50

    return ConditionalEconomics(
        discount_rate=model.discount_rate,
        horizon_yr=model.horizon_yr,
        n_units=model.n_units,
        breakeven_phi=phi_star,
        phi=phi,
        phi_is_endogenous=phi_is_endogenous,
        per_unit_delta_c_usd=per_unit,
        fleet_delta_c_usd=fleet,
        net_positive=bool(fleet > 0.0),
        pv_baseline_usd=pv_base,
        pv_sensor_usd=pv_sensor,
        pv_cbm_slow_usd=pv_slow,
        pv_cbm_fast_usd=pv_fast,
        phi_grid=[float(x) for x in grid],
        fleet_delta_c_p50_usd=[float(x) for x in p50],
        fleet_delta_c_p5_usd=[float(x) for x in p5],
        fleet_delta_c_p95_usd=[float(x) for x in p95],
    )
