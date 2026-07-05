"""Decision-layer tests: POD, RBI scheduling, fleet economics (spec 4.7 / 5)."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.inspection import (
    EconomicsModel,
    fleet_economics,
    next_inspection,
    pod_exponential,
    pod_lognormal,
    probability_of_failure_by,
)


def test_pod_exponential_monotone_and_bounded():
    a = np.linspace(0.0, 20.0, 200)
    pod = pod_exponential(a, scale=5.0, shape=1.5)
    assert pod[0] == pytest.approx(0.0)
    assert np.all(np.diff(pod) >= 0)
    assert pod[-1] < 1.0 and pod[-1] > 0.9


def test_pod_lognormal_half_at_a50():
    assert pod_lognormal(np.array([2.0]), a50=2.0, sigma=0.5)[0] == pytest.approx(0.5)


def test_pod_invalid_params_raise():
    with pytest.raises(ValueError):
        pod_exponential(np.array([1.0]), scale=-1.0)
    with pytest.raises(ValueError):
        pod_lognormal(np.array([1.0]), a50=1.0, sigma=0.0)


def test_probability_of_failure_increases_with_time():
    life = np.random.default_rng(0).lognormal(mean=np.log(30), sigma=0.3, size=10000)
    assert probability_of_failure_by(life, 10) < probability_of_failure_by(life, 30)
    assert probability_of_failure_by(life, 1000) == pytest.approx(1.0, abs=1e-6)


def test_next_inspection_reaches_target():
    life = np.random.default_rng(1).lognormal(mean=np.log(25), sigma=0.25, size=20000)
    plan = next_inspection(life, target_pof=1e-2, horizon_year=40)
    assert not plan.limited_by_horizon
    assert plan.pof_at_next >= 0.01
    assert 0.0 < plan.next_inspection_year < 25.0  # well before the P50 life


def test_next_inspection_horizon_limited():
    life = np.full(1000, 100.0)  # all survive far beyond horizon
    plan = next_inspection(life, target_pof=1e-2, horizon_year=20)
    assert plan.limited_by_horizon
    assert plan.next_inspection_year == 20.0


def test_fleet_economics_reproduces_paper():
    e = fleet_economics(EconomicsModel())
    assert 5.5e6 <= e.fleet_saving_low_usd <= 8.0e6
    assert 33e6 <= e.fleet_saving_high_usd <= 40e6
    assert 1.0 <= e.payback_low_yr <= 6.0
    assert 1.0 <= e.payback_high_yr <= 6.5


def test_economics_editable_parameters_flow_through():
    base = fleet_economics(EconomicsModel())
    dearer = fleet_economics(EconomicsModel(inspection_cost_usd=2.0e6))
    assert dearer.fleet_saving_low_usd > base.fleet_saving_low_usd


def test_next_inspection_huge_horizon_no_memory_blowup():
    # A benign sea state can give astronomically long life; the bounded grid
    # must not try to allocate a giant array.
    life = np.full(1000, 1e13)  # life beyond the horizon
    plan = next_inspection(life, target_pof=1e-2, horizon_year=1e12)
    assert plan.limited_by_horizon  # target never reached within horizon
    assert plan.next_inspection_year == pytest.approx(1e12)


def test_next_inspection_rejects_empty():
    with pytest.raises(ValueError):
        next_inspection(np.array([]))
