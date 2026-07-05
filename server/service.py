"""Service layer: assemble complete, chart-ready payloads from scr_twin_core.

The API stays thin — this module does all the sequencing and array decimation so
the frontend receives one coherent, JSON-safe object per request. No physics
lives here; it delegates to the tested core.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scr_twin_core.bayesian import BayesianRateEstimator
from scr_twin_core.config import AnalysisConfig
from scr_twin_core.inspection import EconomicsModel, fleet_economics, next_inspection
from scr_twin_core.miner import SECONDS_PER_YEAR
from scr_twin_core.pipeline import FullResult, run_full_analysis
from scr_twin_core.synthetic import synthetic_mru_motion

MAX_POINTS = 280  # cap transported array length for smooth, light charts


def to_native(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def decimate(x: np.ndarray, n: int = MAX_POINTS) -> list[float]:
    """Uniformly subsample an array to at most ``n`` points (JSON-safe floats)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size <= n:
        return [float(v) for v in x]
    idx = np.linspace(0, x.size - 1, n).round().astype(int)
    return [float(v) for v in x[idx]]


def bayesian_life_fan(
    nominal_rate: float,
    rate_std: float,
    *,
    years: int = 20,
    blocks_per_year: int = 12,
    life_cap: float = 1500.0,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Remaining-life credible-interval fan that contracts with monitoring time.

    The signature visual: a proper Bayesian posterior on the damage rate, updated
    with monthly stationary-window observations, mapped to remaining life
    ``(1 - accumulated_damage)/rate``. Year 0 is the Monte Carlo prior; the band
    narrows as data accrues (90% CI ~ 1/sqrt(T)).
    """
    rate_std = max(rate_std, nominal_rate * 1e-3)
    est = BayesianRateEstimator(
        prior_mean=nominal_rate, block_obs_std=rate_std, prior_std=rate_std
    )
    yrs: list[float] = [0.0]
    low: list[float] = []
    med: list[float] = []
    high: list[float] = []

    def life_ci(post: Any, t: float) -> tuple[float, float, float]:
        acc = min(nominal_rate * t, 0.999)
        lo, m, hi = post.remaining_life(acc)
        # Clamp the (initially very wide) band to a finite display ceiling.
        return min(lo, life_cap), min(m, life_cap), min(hi, life_cap)

    lo, m, hi = life_ci(est.posterior(), 0.0)
    low.append(lo)
    med.append(m)
    high.append(hi)
    for year in range(1, years + 1):
        for _ in range(blocks_per_year):
            est.update_block(nominal_rate)
        lo, m, hi = life_ci(est.posterior(), float(year))
        yrs.append(float(year))
        low.append(lo)
        med.append(m)
        high.append(hi)
    return {"years": yrs, "low": low, "median": med, "high": high}


def _spectrum_payload(result: FullResult) -> dict[str, list[float]]:
    f = np.asarray(result.motion_freqs)
    band = f <= 0.6  # wave-frequency band of interest
    return {
        "freq": decimate(f[band]),
        "motion_psd": decimate(np.asarray(result.motion_psd)[band]),
        "stress_psd": decimate(np.asarray(result.stress_psd)[band]),
    }


def _posterior_payload(result: FullResult) -> dict[str, Any]:
    mc = result.monte_carlo
    counts, edges = mc.histogram(48)
    cdf_x, cdf_p = mc.cdf()
    return {
        "p10": mc.p10, "p50": mc.p50, "p90": mc.p90,
        "n_members": mc.n_members,
        "hist_counts": [float(c) for c in counts],
        "hist_edges": [float(e) for e in edges],
        "cdf_x": decimate(cdf_x), "cdf_p": decimate(cdf_p),
    }


def analyze(
    config: AnalysisConfig,
    heave: np.ndarray,
    fs: float,
    *,
    is_synthetic: bool,
    data_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full chain and assemble the complete dashboard payload."""
    result = run_full_analysis(config, heave, fs, motion_is_synthetic=is_synthetic)
    mc = result.monte_carlo

    plan = next_inspection(mc.life_years, target_pof=1e-2, horizon_year=float(max(60.0, mc.p90)))
    horizon = max(plan.next_inspection_year * 1.5, mc.p50, 30.0)
    pof_years = np.linspace(0.5, horizon, 60)
    pof_vals = [float(np.mean(mc.life_years <= t)) for t in pof_years]

    econ = fleet_economics(EconomicsModel())
    rate_std = float(np.std(mc.damage_rate_per_year))
    fan = bayesian_life_fan(result.annual_damage_rate_time, rate_std)

    return to_native({
        "sea_state": {
            "hs": result.sea_state.hs, "tp": result.sea_state.tp,
            "tz": result.sea_state.tz, "gamma": result.sea_state.gamma,
        },
        "spectrum": _spectrum_payload(result),
        "damage": {
            "annual_rate_time": result.annual_damage_rate_time,
            "annual_rate_spectral": result.annual_damage_rate_spectral,
            "deterministic_life_years": result.deterministic_life_years,
            "block_damage": result.time_domain_block.damage,
            "block_seconds": result.time_domain_block.block_seconds,
        },
        "environment": {
            "enabled": result.environment is not None,
            "factor": result.environment_factor,
            "temperature_factor": result.environment.temperature_factor if result.environment else 1.0,
            "salinity_factor": result.environment.salinity_factor if result.environment else 1.0,
        },
        "posterior": _posterior_payload(result),
        "bayesian_fan": fan,
        "inspection": {
            "next_inspection_year": plan.next_inspection_year,
            "target_pof": plan.target_pof,
            "pof_at_next": plan.pof_at_next,
            "limited_by_horizon": plan.limited_by_horizon,
            "pof_years": [float(t) for t in pof_years],
            "pof_vals": pof_vals,
        },
        "economics": econ.as_dict(),
        "provenance": result.provenance.model_dump(),
        "data_health": data_health,
        "trace": {
            "time": decimate(np.arange(heave.size) / fs, 600),
            "heave": decimate(heave, 600),
        },
    })


def make_synthetic(hs: float, tp: float, gamma: float, duration: float, fs: float, seed: int) -> tuple[np.ndarray, float]:
    """Generate a deterministic synthetic heave record (badged synthetic upstream)."""
    m = synthetic_mru_motion(duration=duration, fs=fs, hs=hs, tp=tp, gamma=gamma, seed=seed)
    return m.heave, m.fs


def stream_seconds_per_year() -> float:
    return SECONDS_PER_YEAR
