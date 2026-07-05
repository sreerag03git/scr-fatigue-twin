"""Palmgren-Miner accumulation tests."""

from __future__ import annotations

import numpy as np

from scr_twin_core.miner import SECONDS_PER_YEAR, block_damage, miner_damage
from scr_twin_core.rainflow import CycleCount
from scr_twin_core.sn import cycles_to_failure, get_curve


def _one_cycle(range_pa: float, count: float = 1.0) -> CycleCount:
    return CycleCount(
        np.array([range_pa]), np.array([0.0]), np.array([count])
    )


def test_single_cycle_damage_is_reciprocal_of_N():
    curve = get_curve("D")
    dsig = 100e6
    n = cycles_to_failure(np.array([dsig]), curve)[0]
    d = miner_damage(_one_cycle(dsig), curve)
    assert d == 1.0 / n


def test_half_cycle_is_half_damage():
    curve = get_curve("E")
    dsig = 80e6
    full = miner_damage(_one_cycle(dsig, 1.0), curve)
    half = miner_damage(_one_cycle(dsig, 0.5), curve)
    assert half == 0.5 * full


def test_empty_is_zero_damage():
    curve = get_curve("F")
    empty = CycleCount(np.array([]), np.array([]), np.array([]))
    assert miner_damage(empty, curve) == 0.0


def test_block_annualisation_arithmetic():
    curve = get_curve("D")
    cy = _one_cycle(120e6, count=10.0)
    block_seconds = 1800.0  # 30 min
    res = block_damage(cy, curve, block_seconds)
    expected_rate = res.damage / block_seconds * SECONDS_PER_YEAR
    assert res.damage_rate_per_year == expected_rate
    assert res.life_years == 1.0 / expected_rate


def test_low_stress_block_has_very_long_life():
    curve = get_curve("D")
    cy = _one_cycle(1e6)  # ~1 MPa, far below the fatigue limit
    res = block_damage(cy, curve, 1800.0)
    assert res.damage < 1e-12
    assert res.life_years > 1e9


def test_zero_range_block_has_infinite_life():
    curve = get_curve("D")
    cy = _one_cycle(0.0)  # non-damaging by construction
    res = block_damage(cy, curve, 1800.0)
    assert res.damage == 0.0
    assert np.isinf(res.life_years)
