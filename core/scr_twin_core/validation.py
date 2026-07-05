"""Programmatic acceptance-gate checks (project spec 5).

Runs each validation gate that the physics core is responsible for and returns
structured :class:`GateResult` records. This is the data source for the in-app
Validation view and the provenance/PDF export, and it doubles as a dependency-
free self-check (``python -m scr_twin_core.validation``). It intentionally
re-derives results independently of the pytest suite so a reviewer can run it on
a packaged build with no test tooling installed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from .bayesian import BayesianRateEstimator
from .catenary import solve_plain_catenary
from .config import AnalysisConfig, RiserConfig
from .environment import EnvironmentCorrection
from .inspection import EconomicsModel, fleet_economics
from .montecarlo import UncertaintyModel, accumulated_damage_divergence
from .pipeline import run_full_analysis
from .rainflow import count_cycles
from .sn import get_curve
from .spectral import jonswap, significant_from_m0, spectral_moments
from .spectral_damage import dirlik_damage_rate
from .stress import random_phase_timeseries, rfft_frequencies
from .synthetic import synthetic_mru_motion


@dataclass(frozen=True)
class GateResult:
    """Outcome of one acceptance gate."""

    name: str
    category: str  # "physics" | "calibration" | "reliability"
    passed: bool
    target: str
    actual: str
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["passed"] = bool(self.passed)  # coerce numpy.bool_ -> JSON-safe bool
        return d


def _gate_jonswap_hs() -> GateResult:
    f = np.linspace(0.0, 0.6, 6000)
    hs = 3.5
    s = jonswap(f, hs, 10.0, gamma=3.3, normalize=True)
    rec = significant_from_m0(spectral_moments(f, s, (0,))[0])
    err = abs(rec - hs) / hs
    return GateResult(
        "JONSWAP Hs recovery", "physics", err < 1e-3,
        "Hs from 4 sqrt(m0) within 0.1%", f"error={err:.2e}",
    )


def _gate_catenary() -> GateResult:
    cat = solve_plain_catenary(1000.0, 70.0, 1000.0)
    a = cat.catenary_parameter
    depth = a * (np.cosh(cat.horizontal_span / a) - 1.0)
    ok_depth = abs(depth - cat.water_depth) / cat.water_depth < 1e-9
    ok_kappa = abs(cat.tdp_curvature - 1.0 / a) < 1e-12
    return GateResult(
        "Catenary closed-form", "physics", ok_depth and ok_kappa,
        "shape/curvature match closed form", f"depth_err<1e-9={ok_depth}, kappa=1/a={ok_kappa}",
    )


def _gate_sn_table() -> GateResult:
    table = {"D": 52.63, "E": 46.78, "F": 41.52, "F1": 36.84, "F3": 32.75}
    worst = 0.0
    for name, limit in table.items():
        got = get_curve(name).fatigue_limit_mpa
        worst = max(worst, abs(got - limit) / limit)
    return GateResult(
        "S-N DNV-RP-C203 points", "physics", worst < 2e-3,
        "fatigue limits match Table 2-1 <0.2%", f"max_error={worst:.2e}",
    )


def _gate_cross_method(seed: int) -> GateResult:
    fs, n = 2.0, 2**18
    freqs = rfft_frequencies(n, fs)
    shape = np.exp(-((freqs - 0.1) ** 2) / (2.0 * 0.02**2))
    psd = shape * (20.0**2) / np.trapezoid(shape, freqs)
    moments = spectral_moments(freqs, psd, (0, 1, 2, 4))
    x = random_phase_timeseries(freqs, psd, fs=fs, n_samples=n, seed=seed)
    cy = count_cycles(x)
    m, log_a = 3.0, 12.164
    d_time = float(np.sum(cy.counts * cy.ranges**m) / 10.0**log_a) / (n / fs)
    d_dk = dirlik_damage_rate(moments, sn_m=m, sn_log_a=log_a, stress_to_mpa=1.0)
    err = abs(d_dk - d_time) / d_time
    return GateResult(
        "Dirlik vs rainflow", "physics", err < 0.15,
        "spectral & time-domain agree <15%", f"error={err:.1%}",
    )


def _gate_environment() -> GateResult:
    ec = EnvironmentCorrection()
    red = ec.life_reduction
    return GateResult(
        "Arabian Gulf correction", "calibration", 0.30 <= red <= 0.39,
        "life reduction 30-39% (F 0.61-0.70)", f"reduction={red:.1%}, F={ec.combined_factor:.3f}",
    )


def _gate_divergence(seed: int) -> GateResult:
    div = accumulated_damage_divergence(UncertaintyModel(), 15, n_members=30_000, seed=seed)
    p10, p90 = np.percentile(div, [10, 90])
    ok = (0.035 <= p10 <= 0.07) and (0.25 <= p90 <= 0.31)
    return GateResult(
        "Year-15 divergence fan", "calibration", ok,
        "P10~5%, P90~28%", f"P10={p10:.1%}, P90={p90:.1%}",
    )


def _gate_bayesian() -> GateResult:
    bpy, true_rate, obs_std = 100, 0.025, 0.02

    def width_after(years: float) -> float:
        est = BayesianRateEstimator(prior_mean=true_rate, block_obs_std=obs_std)
        for _ in range(int(years * bpy)):
            est.update_block(true_rate)
        return est.posterior().ci90_width

    ratio = width_after(4) / width_after(1)
    return GateResult(
        "Bayesian 1/sqrt(T)", "calibration", abs(ratio - 0.5) < 0.04,
        "90% CI halves by year 4", f"CI(4)/CI(1)={ratio:.3f}",
    )


def _gate_fleet_economics() -> GateResult:
    e = fleet_economics(EconomicsModel())
    low_m, high_m = e.fleet_saving_low_usd / 1e6, e.fleet_saving_high_usd / 1e6
    ok = (
        5.5 <= low_m <= 8.0
        and 33.0 <= high_m <= 40.0
        and 1.0 <= e.payback_low_yr <= 6.0
        and 1.0 <= e.payback_high_yr <= 6.5
    )
    return GateResult(
        "Fleet economics", "calibration", ok,
        "$6.6-36.5M / 20yr / 20 units, payback 1-6yr",
        f"${low_m:.1f}M-${high_m:.1f}M, payback {e.payback_low_yr:.1f}-{e.payback_high_yr:.1f}yr",
    )


def _gate_determinism() -> GateResult:
    cfg = AnalysisConfig(riser=RiserConfig.reference_scr(), seed=11, n_monte_carlo=1000)
    m = synthetic_mru_motion(duration=600.0, fs=4.0, hs=3.0, tp=10.0, seed=1)
    a = run_full_analysis(cfg, m.heave, m.fs)
    b = run_full_analysis(cfg, m.heave, m.fs)
    ok = (
        a.annual_damage_rate_time == b.annual_damage_rate_time
        and np.array_equal(a.monte_carlo.life_years, b.monte_carlo.life_years)
    )
    return GateResult(
        "Determinism", "reliability", ok,
        "identical seed -> identical output", f"byte_identical={ok}",
    )


def run_all_gates(seed: int = 0) -> list[GateResult]:
    """Run every core acceptance gate and return structured results."""
    checks: list[Callable[[], GateResult]] = [
        _gate_jonswap_hs,
        _gate_catenary,
        _gate_sn_table,
        lambda: _gate_cross_method(seed),
        _gate_environment,
        lambda: _gate_divergence(seed),
        _gate_bayesian,
        _gate_fleet_economics,
        _gate_determinism,
    ]
    return [c() for c in checks]


def format_report(results: list[GateResult]) -> str:
    """Render gate results as a fixed-width text table."""
    lines = [
        f"{'GATE':<28} {'CATEGORY':<12} {'RESULT':<6} ACTUAL",
        "-" * 78,
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"{r.name:<28} {r.category:<12} {status:<6} {r.actual}")
    n_pass = sum(r.passed for r in results)
    lines.append("-" * 78)
    lines.append(f"{n_pass}/{len(results)} gates passed")
    return "\n".join(lines)


def main() -> int:
    """CLI entry point: run gates, print the report, exit non-zero on failure."""
    results = run_all_gates()
    print(format_report(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
