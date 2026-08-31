"""Conditional CBM economics tests (paper Eq. 11).

These replace the retracted flat "$6.6-36.5M net saving" story. The honest claim
is a *break-even* on phi = P(asset ages slower than design): the sensor is a net
loss for a fast-ageing fleet and a net gain only for a confidently-slower one, and
phi is estimated endogenously from the remaining-life posterior.
"""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.inspection import (
    ConditionalEconomicsModel,
    breakeven_phi,
    fleet_economics_conditional,
    phi_ageing_slower,
)


def test_breakeven_phi_in_confident_band():
    # With defensible offshore defaults the sensor only pays once you are clearly
    # (but not near-certainly) confident the asset ages slower than design.
    phi_star = breakeven_phi(ConditionalEconomicsModel())
    assert 0.50 <= phi_star <= 0.75


def test_delta_c_rises_with_phi_and_brackets_zero():
    m = ConditionalEconomicsModel()
    e = fleet_economics_conditional(m, n_cost_mc=0)  # deterministic curve
    p50 = np.asarray(e.fleet_delta_c_p50_usd)
    assert p50[0] < 0.0 < p50[-1]                 # loss at phi=0, gain at phi=1
    assert np.all(np.diff(p50) > 0)               # monotonically increasing in phi
    # The break-even crossing sits at phi*.
    grid = np.asarray(e.phi_grid)
    crossing = grid[np.argmin(np.abs(p50))]
    assert abs(crossing - e.breakeven_phi) < 0.05


def test_delta_c_zero_at_breakeven():
    m = ConditionalEconomicsModel()
    e = fleet_economics_conditional(m, phi=breakeven_phi(m), n_cost_mc=0)
    assert abs(e.fleet_delta_c_usd) < 1.0  # essentially zero USD at break-even


def test_phi_endogenous_from_posterior():
    # A posterior centred below the design life -> most members age *faster* ->
    # low phi -> net cost; the flag marks phi as data-derived.
    rng = np.random.default_rng(0)
    life = rng.lognormal(mean=np.log(20.0), sigma=0.3, size=20_000)
    design = 25.0
    phi = phi_ageing_slower(life, design)
    assert 0.0 < phi < 0.5
    e = fleet_economics_conditional(
        ConditionalEconomicsModel(), life_samples=life, design_life_years=design
    )
    assert e.phi_is_endogenous
    assert e.phi == pytest.approx(phi, abs=1e-9)
    assert e.phi < e.breakeven_phi          # below break-even ...
    assert not e.net_positive               # ... so the sensor is a net cost
    assert e.fleet_delta_c_usd < 0.0


def test_confident_slow_fleet_is_net_positive():
    # A posterior well above design life -> high phi -> the sensor pays.
    rng = np.random.default_rng(1)
    life = rng.lognormal(mean=np.log(60.0), sigma=0.2, size=20_000)
    e = fleet_economics_conditional(
        ConditionalEconomicsModel(), life_samples=life, design_life_years=25.0
    )
    assert e.phi > e.breakeven_phi
    assert e.net_positive
    assert e.fleet_delta_c_usd > 0.0


def test_no_posterior_falls_back_to_breakeven():
    m = ConditionalEconomicsModel()
    e = fleet_economics_conditional(m)
    assert not e.phi_is_endogenous
    assert e.phi == pytest.approx(e.breakeven_phi, abs=1e-9)
    assert abs(e.fleet_delta_c_usd) < 1.0


def test_infinite_design_life_falls_back_gracefully():
    # A benign sea state can give an infinite deterministic life; phi is undefined
    # and must not raise or poison the report.
    life = np.full(1000, 500.0)
    assert np.isnan(phi_ageing_slower(life, float("inf")))
    e = fleet_economics_conditional(
        ConditionalEconomicsModel(), life_samples=life, design_life_years=float("inf")
    )
    assert not e.phi_is_endogenous
    assert np.isfinite(e.fleet_delta_c_usd)


def test_cost_band_is_ordered_and_brackets_median():
    e = fleet_economics_conditional(ConditionalEconomicsModel(), seed=0)
    p5 = np.asarray(e.fleet_delta_c_p5_usd)
    p50 = np.asarray(e.fleet_delta_c_p50_usd)
    p95 = np.asarray(e.fleet_delta_c_p95_usd)
    assert np.all(p5 <= p50) and np.all(p50 <= p95)
    assert np.any(p95 - p5 > 0.0)  # a genuine (non-degenerate) band


def test_deterministic_for_fixed_seed():
    a = fleet_economics_conditional(ConditionalEconomicsModel(), seed=7)
    b = fleet_economics_conditional(ConditionalEconomicsModel(), seed=7)
    assert a.fleet_delta_c_p50_usd == b.fleet_delta_c_p50_usd
    assert a.fleet_delta_c_p5_usd == b.fleet_delta_c_p5_usd


def test_dearer_inspection_raises_the_value_of_deferral():
    # If each campaign costs more, deferring (high phi) is worth more, so the
    # gain at phi=1 grows.
    base = fleet_economics_conditional(ConditionalEconomicsModel(), phi=1.0, n_cost_mc=0)
    dearer = fleet_economics_conditional(
        ConditionalEconomicsModel(inspection_cost_usd=2.0e6), phi=1.0, n_cost_mc=0
    )
    assert dearer.fleet_delta_c_usd > base.fleet_delta_c_usd


def test_model_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ConditionalEconomicsModel(discount_rate=2.0)
    with pytest.raises(ValueError):
        ConditionalEconomicsModel(cbm_interval_slow_yr=0.0)
    with pytest.raises(ValueError):
        ConditionalEconomicsModel(n_units=0)
