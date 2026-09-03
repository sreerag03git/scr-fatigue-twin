"""Service layer: assemble complete, chart-ready payloads from scr_twin_core.

The API stays thin — this module does all the sequencing and array decimation so
the frontend receives one coherent, JSON-safe object per request. No physics
lives here; it delegates to the tested core.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scr_twin_core.bayesian import BayesianRateEstimator
from scr_twin_core.montecarlo import (
    UncertaintyModel,
    simulate_block_rate_observations,
    simulate_wave_climate_multipliers,
)
from scr_twin_core.config import AnalysisConfig
from scr_twin_core.fracture import (
    InitialFlawDistribution,
    ParisMaterial,
    crack_growth_curve,
    crack_growth_mc,
    crack_life_years,
    equivalent_stress_range_mpa,
    stress_intensity_range,
)
from scr_twin_core.inspection import (
    ConditionalEconomicsModel,
    fleet_economics_conditional,
    next_inspection,
    pod_lognormal,
)
from scr_twin_core.miner import SECONDS_PER_YEAR
from scr_twin_core.pipeline import FullResult, long_term_fatigue, run_full_analysis
from scr_twin_core.reliability import form_fatigue_reliability
from scr_twin_core.sn import get_curve
from scr_twin_core.scatter import ScatterDiagram, example_scatter_diagram, load_scatter_csv
from scr_twin_core.viv import CurrentProfile, viv_screening
from scr_twin_core.synthetic import synthetic_mru_6dof, synthetic_mru_motion
from scr_twin_core.transfer import InterpolatedTransferFunction, load_transfer_csv

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

    A proper Bayesian posterior on the damage rate, updated with **genuine
    AR(1)-correlated** per-block observations (not a constant), mapped to
    remaining life ``(1 - accumulated_damage)/rate``. Because the observations are
    autocorrelated the updater uses the AR(1) effective sample size, so the band
    contracts at the honest rate: still ~ ``1/sqrt(T)`` (90% CI halves by year 4)
    but wider in absolute terms than an i.i.d. shrink would give. Year 0 is the
    Monte Carlo prior.
    """
    model = UncertaintyModel()
    phi = model.wave_climate_ar1
    # Per-block observation std anchored to the generator's LogNormal CoV so the
    # data scatter and the assumed obs noise are consistent.
    log_cov = float(np.sqrt(np.exp(model.wave_climate_logstd**2) - 1.0))
    block_std = max(nominal_rate * log_cov, nominal_rate * 1e-3, rate_std * 1e-6)
    obs = simulate_block_rate_observations(
        nominal_rate, model, years=years, blocks_per_year=blocks_per_year, seed=seed, phi=phi,
    )
    est = BayesianRateEstimator(
        prior_mean=nominal_rate, block_obs_std=block_std, prior_std=block_std, obs_ar1=phi,
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
        for b in range(blocks_per_year):
            est.update_block(float(obs[(year - 1) * blocks_per_year + b]))
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


def _transfer_payload(result: FullResult) -> dict[str, Any]:
    """The Layer-1 |H(f)| curve (Fig 4) plus route + validated/illustrative flag."""
    return {
        "freq": decimate(np.asarray(result.hf_freqs), 200),
        "stress_mag": decimate(np.asarray(result.hf_stress_mag), 200),
        "moment_mag": decimate(np.asarray(result.hf_moment_mag), 200),
        "phase": decimate(np.asarray(result.hf_phase), 200),
        "route": result.transfer_route,
        "is_validated": bool(result.transfer_provenance.get("is_validated", False)),
        "provenance": result.transfer_provenance,
    }


def load_transfer(data: bytes | str, **overrides: Any) -> InterpolatedTransferFunction:
    """Parse an imported complex H(f) CSV into a transfer function (raises on bad input)."""
    return load_transfer_csv(data, **overrides)


def _catenary_payload(result: FullResult) -> dict[str, Any]:
    """Static catenary profile (riser shape from TDP at origin to hang-off)."""
    c = result.catenary_profile
    return {
        "x": decimate(np.asarray(c["x"]), 200),
        "y": decimate(np.asarray(c["y"]), 200),
        "catenary_parameter": c["catenary_parameter"],
        "horizontal_span": c["horizontal_span"],
        "arc_length": c["arc_length"],
        "water_depth": c["water_depth"],
        "tdp_curvature": c["tdp_curvature"],
        "top_angle_deg": c["top_angle_deg"],
    }


def _verification_payload(result: FullResult) -> dict[str, Any]:
    """Data for the Dirlik/Bendat-vs-rainflow range-distribution verification plot."""
    h = result.rainflow_hist
    m = result.spectral_moments
    stress_to_mpa = float(h.get("stress_to_mpa", 1.0e-6))
    edges = np.asarray(h["edges"], dtype=np.float64) * stress_to_mpa  # -> MPa
    return {
        "hist_edges_mpa": [float(e) for e in edges],
        "hist_counts": [float(c) for c in np.asarray(h["counts"])],
        "moments": {str(k): float(v) for k, v in m.items()},
        "stress_to_mpa": stress_to_mpa,
        "sigma_mpa": float(np.sqrt(max(m.get(0, 0.0), 0.0)) * stress_to_mpa),
    }


def divergence_fan(
    parameters: dict[str, float], *, years: int = 20, n_members: int = 15000, seed: int = 0,
) -> dict[str, list[float]]:
    """Accumulated-damage divergence fan (actual/design - 1) vs year.

    Uses the run's own uncertainty parameters; a fixed n_members/seed keeps the
    figure deterministic and consistent with the spec-Sec.5 gate regardless of
    the user's Monte Carlo slider.
    """
    model = UncertaintyModel(**parameters)
    w = simulate_wave_climate_multipliers(model, years, n_members=n_members, seed=seed)
    cum = np.cumsum(w, axis=1)
    design = np.arange(1, years + 1, dtype=np.float64)
    div = cum / design - 1.0  # (members, years)
    p10, p50, p90 = np.percentile(div, [10, 50, 90], axis=0)
    return {
        "years": [0.0, *[float(y) for y in design]],
        "p10": [0.0, *[float(v) for v in p10]],
        "p50": [0.0, *[float(v) for v in p50]],
        "p90": [0.0, *[float(v) for v in p90]],
    }


def _long_term_payload(
    config: AnalysisConfig,
    diagram: ScatterDiagram,
    imported_tf: InterpolatedTransferFunction | None,
) -> dict[str, Any]:
    """Long-term scatter-diagram fatigue (DNV-RP-C203 Sec. 5) + driver heat map."""
    lt = long_term_fatigue(config, diagram, imported_tf=imported_tf)
    return {
        "source": diagram.source,
        "annual_damage_rate": lt.annual_damage_rate,
        "life_years": lt.life_years,
        "n_cells": lt.n_cells,
        "hs_values": diagram.hs_values(),
        "tp_values": diagram.tp_values(),
        "contributions": lt.as_dict()["contributions"],
    }


def load_scatter(data: bytes | str) -> ScatterDiagram:
    """Parse an uploaded scatter-diagram CSV (raises loudly on bad input)."""
    return load_scatter_csv(data)


def _viv_payload(config: AnalysisConfig) -> dict[str, Any]:
    """Cross-flow VIV screening (DNV-RP-F204) + dominant mode shape + current profile."""
    vc = config.viv
    if vc.surface_current <= 0.0:
        return {"enabled": False}
    riser = config.riser
    section = riser.pipe_section()
    catenary = riser.catenary()
    correction = config.environment.correction()
    base = get_curve(riser.sn_class, riser.sn_environment)
    curve = correction.apply_to_curve(base) if correction is not None else base
    current = CurrentProfile(vc.surface_current, riser.water_depth, vc.profile_exponent)
    sc = viv_screening(
        catenary, section, curve, current,
        contents_density=riser.contents_density,
        added_mass_coefficient=vc.added_mass_coefficient, strouhal=vc.strouhal,
        damping_ratio=vc.damping_ratio, n_modes=vc.n_modes,
        thickness_m=riser.thickness_for_correction,
    )
    rm = sc.riser_modes
    dom_idx = max(sc.dominant_mode - 1, 0)
    dom_shape = rm.shapes[dom_idx] if rm.shapes.shape[0] > dom_idx else rm.shapes[0]
    # Current profile vs height above the TDP for the schematic.
    heights = np.linspace(0.0, riser.water_depth, 40)
    payload = to_native(sc.as_dict())
    payload.update({
        "enabled": True,
        "span_length": rm.span_length,
        "dominant_shape": {
            "arc": decimate(np.asarray(rm.arc), 200),
            "disp": decimate(np.asarray(dom_shape), 200),
        },
        "current_profile": {
            "height": [float(h) for h in heights],
            "speed": [float(v) for v in current.speed_at_height(heights)],
        },
    })
    return payload


def _reliability_payload(config: AnalysisConfig, result: FullResult) -> dict[str, Any]:
    """FORM fatigue reliability: beta, annual Pf vs the DNV safety-class target."""
    try:
        r = form_fatigue_reliability(
            result.monte_carlo.life_years, config.riser.design_service_life_years,
            result.parameters, safety_class=config.riser.safety_class,
        )
    except ValueError:
        return {"enabled": False}
    return {"enabled": True, **r.as_dict()}


def _crack_payload(config: AnalysisConfig, result: FullResult) -> dict[str, Any]:
    """Paris-law crack-growth pathway (BS 7910) + POD-driven crack-based inspection."""
    riser = config.riser
    hist = result.rainflow_hist
    edges = np.asarray(hist["edges"], dtype=float)
    counts = np.asarray(hist["counts"], dtype=float)
    block_s = float(result.time_domain_block.block_seconds)
    total_cycles = float(counts.sum())
    if total_cycles <= 0.0 or block_s <= 0.0:
        return {"enabled": False}
    m_paris = 3.0
    cycles_per_year = total_cycles / block_s * SECONDS_PER_YEAR
    dsig_eq = equivalent_stress_range_mpa(edges, counts, m_paris)  # hot-spot MPa
    a0 = 1.0e-3                              # postulated initial defect (ECA), 1 mm
    a_crit = float(riser.wall_thickness)     # through-wall breach
    seawater = str(riser.sn_environment.value) == "seawater_cp"
    _base = ParisMaterial.bs7910_marine_cp() if seawater else ParisMaterial.bs7910_air_mean()
    # Conservative ECA: high riser tension -> high R-ratio -> take no threshold benefit.
    mat = ParisMaterial(C=_base.C, m=_base.m, delta_k_th=0.0, name=_base.name + ", no-threshold ECA")
    dk0 = float(stress_intensity_range(dsig_eq, a0))
    propagates = dk0 >= mat.delta_k_th
    life = crack_life_years(a0, a_crit, mat, dsig_eq, cycles_per_year) if propagates else float("inf")

    horizon = 100.0 if not np.isfinite(life) else float(min(max(life * 1.3, 30.0), 400.0))
    t, a = crack_growth_curve(a0, mat, dsig_eq, cycles_per_year, a_c=a_crit,
                              years=horizon, n_steps=120)
    a_grid = np.linspace(1e-4, a_crit, 60)
    pod = pod_lognormal(a_grid, a50=2.0e-3, sigma=0.6)  # subsea MPI/ACFM-class
    mc = crack_growth_mc(InitialFlawDistribution(), mat, dsig_eq, cycles_per_year,
                         a_c=a_crit, n_members=4000, seed=config.seed)
    finite = mc[np.isfinite(mc)]
    frac_prop = float(np.mean(np.isfinite(mc)))
    crack_insp: float | None = None
    if finite.size >= 20:
        plan = next_inspection(finite, target_pof=1e-2,
                               horizon_year=float(max(60.0, np.percentile(finite, 90))))
        crack_insp = plan.next_inspection_year

    return {
        "enabled": True,
        "material": mat.name,
        "equivalent_stress_range_mpa": dsig_eq,
        "cycles_per_year": cycles_per_year,
        "initial_flaw_mm": a0 * 1e3,
        "critical_depth_mm": a_crit * 1e3,
        "delta_k0": dk0,
        "delta_k_threshold": mat.delta_k_th,
        "propagates": propagates,
        "crack_life_years": life,
        "fraction_propagating": frac_prop,
        "crack_inspection_year": crack_insp,
        "a_of_t": {"years": [float(v) for v in t], "depth_mm": [float(v * 1e3) for v in a]},
        "pod": {"size_mm": [float(v * 1e3) for v in a_grid], "prob": [float(v) for v in pod]},
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
    imported_tf: InterpolatedTransferFunction | None = None,
    channels: dict[str, np.ndarray] | None = None,
    scatter_diagram: ScatterDiagram | None = None,
) -> dict[str, Any]:
    """Run the full chain and assemble the complete dashboard payload.

    When ``channels`` (6-DOF) is supplied the hang-off motion is resolved via
    Eq. 6; otherwise ``heave`` alone drives the chain. ``scatter_diagram`` (or the
    illustrative default) drives the long-term (DNV-RP-C203 Sec. 5) fatigue block.
    """
    result = run_full_analysis(
        config, heave, fs, motion_is_synthetic=is_synthetic,
        imported_tf=imported_tf, motion_channels=channels,
    )
    mc = result.monte_carlo
    diagram = scatter_diagram if scatter_diagram is not None else example_scatter_diagram()

    plan = next_inspection(mc.life_years, target_pof=1e-2, horizon_year=float(max(60.0, mc.p90)))
    horizon = max(plan.next_inspection_year * 1.5, mc.p50, 30.0)
    pof_years = np.linspace(0.5, horizon, 60)
    pof_vals = [float(np.mean(mc.life_years <= t)) for t in pof_years]

    # Conditional CBM economics (Eq. 11): phi is estimated endogenously from this
    # run's own remaining-life posterior (P(life > design life)), not assumed.
    econ = fleet_economics_conditional(
        ConditionalEconomicsModel(),
        life_samples=mc.life_years,
        design_life_years=result.deterministic_life_years,
    )
    rate_std = float(np.std(mc.damage_rate_per_year))
    fan = bayesian_life_fan(result.annual_damage_rate_time, rate_std)

    # Cross-flow VIV screening + combined (wave + VIV) fatigue by Miner summation.
    viv_block = _viv_payload(config)
    wave_rate = float(result.annual_damage_rate_time)
    viv_rate = float(viv_block.get("annual_damage_rate", 0.0)) if viv_block.get("enabled") else 0.0
    combined_rate = wave_rate + viv_rate

    return to_native({
        "sea_state": {
            "hs": result.sea_state.hs, "tp": result.sea_state.tp,
            "tz": result.sea_state.tz, "gamma": result.sea_state.gamma,
        },
        "spectrum": _spectrum_payload(result),
        "transfer": _transfer_payload(result),
        "catenary": _catenary_payload(result),
        "verification": _verification_payload(result),
        "divergence_fan": divergence_fan(result.parameters, seed=config.seed),
        "long_term": _long_term_payload(config, diagram, imported_tf),
        "crack": _crack_payload(config, result),
        "reliability": _reliability_payload(config, result),
        "viv": viv_block,
        "combined": {
            "wave_rate": wave_rate,
            "viv_rate": viv_rate,
            "annual_rate": combined_rate,
            "life_years": (1.0 / combined_rate) if combined_rate > 0.0 else float("inf"),
        },
        "dof_contributions": result.dof_contributions,
        "damage": {
            "annual_rate_time": result.annual_damage_rate_time,
            "annual_rate_spectral": result.annual_damage_rate_spectral,
            "deterministic_life_years": result.deterministic_life_years,
            "block_damage": result.time_domain_block.damage,
            "block_seconds": result.time_domain_block.block_seconds,
            "sn_environment": str(config.riser.sn_environment.value),
            "acceptance": result.fatigue_acceptance.as_dict(),
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


def make_synthetic_6dof(
    hs: float, tp: float, gamma: float, duration: float, fs: float, seed: int, heading_deg: float,
) -> tuple[dict[str, np.ndarray], float]:
    """Generate a deterministic synthetic 6-DOF MRU record (channels dict + fs)."""
    m = synthetic_mru_6dof(
        duration=duration, fs=fs, hs=hs, tp=tp, gamma=gamma, seed=seed, heading_deg=heading_deg
    )
    return m.channels, m.fs


def stream_seconds_per_year() -> float:
    return SECONDS_PER_YEAR
