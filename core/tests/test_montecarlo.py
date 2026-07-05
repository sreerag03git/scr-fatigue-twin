"""Monte Carlo tests: determinism, percentiles, performance, divergence gate."""

from __future__ import annotations

import time

import numpy as np
import pytest

from scr_twin_core.montecarlo import (
    UncertaintyModel,
    accumulated_damage_divergence,
    run_monte_carlo,
    simulate_wave_climate_multipliers,
)

NOMINAL_RATE = 1.0 / 40.0  # design life 40 yr


def test_deterministic_same_seed():
    a = run_monte_carlo(NOMINAL_RATE, UncertaintyModel(), n_members=10_000, seed=42)
    b = run_monte_carlo(NOMINAL_RATE, UncertaintyModel(), n_members=10_000, seed=42)
    np.testing.assert_array_equal(a.life_years, b.life_years)
    assert a.p50 == b.p50


def test_percentiles_ordered_and_positive():
    r = run_monte_carlo(NOMINAL_RATE, UncertaintyModel(), n_members=10_000, seed=1)
    assert 0.0 < r.p10 < r.p50 < r.p90
    assert r.life_years.size == 10_000
    counts, edges = r.histogram(40)
    assert counts.sum() == 10_000
    x, p = r.cdf()
    assert p[0] > 0.0 and p[-1] == pytest.approx(1.0)


def test_performance_under_budget():
    # Warm up (import/JIT of numpy paths), then time a full 10k update.
    run_monte_carlo(NOMINAL_RATE, UncertaintyModel(), n_members=10_000, seed=0)
    t0 = time.perf_counter()
    run_monte_carlo(NOMINAL_RATE, UncertaintyModel(), n_members=10_000, seed=0)
    elapsed = time.perf_counter() - t0
    # Spec target < 250 ms; assert a generous ceiling to stay robust on slow CI.
    assert elapsed < 0.5, f"10k MC took {elapsed*1000:.1f} ms"


def test_divergence_gate_year15():
    # Acceptance (spec 5): design-vs-actual accumulated-damage divergence at
    # year 15 spans ~5% (P10) to ~28% (P90) across wave-climate realizations.
    div = accumulated_damage_divergence(UncertaintyModel(), 15, n_members=30_000, seed=0)
    p10, p50, p90 = np.percentile(div, [10, 50, 90])
    assert 0.035 <= p10 <= 0.07, f"P10={p10:.3f}"
    assert 0.25 <= p90 <= 0.31, f"P90={p90:.3f}"


def test_wave_climate_multiplier_properties():
    m = UncertaintyModel()
    w = simulate_wave_climate_multipliers(m, 20, n_members=20_000, seed=3)
    assert w.shape == (20_000, 20)
    # median multiplier matches the configured value
    assert np.median(w) == pytest.approx(m.wave_climate_median, rel=0.03)
    # AR(1) persistence: positive lag-1 correlation close to phi
    xi = np.log(w / m.wave_climate_median)
    c0 = xi[:, :-1].ravel()
    c1 = xi[:, 1:].ravel()
    corr = np.corrcoef(c0, c1)[0, 1]
    assert corr == pytest.approx(m.wave_climate_ar1, abs=0.05)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        run_monte_carlo(-1.0, UncertaintyModel())
    with pytest.raises(ValueError):
        accumulated_damage_divergence(UncertaintyModel(), 0)
