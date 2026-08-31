"""Palmgren-Miner accumulation tests."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.miner import SECONDS_PER_YEAR, block_damage, miner_damage
from scr_twin_core.rainflow import CycleCount
from scr_twin_core.sn import MeanStressModel, cycles_to_failure, get_curve


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


# --- mean-stress correction wiring ---------------------------------------- #
def _cycle_with_mean(range_pa: float, mean_pa: float) -> CycleCount:
    return CycleCount(np.array([range_pa]), np.array([mean_pa]), np.array([1.0]))


def test_mean_stress_none_is_bit_identical():
    curve = get_curve("D")
    cy = _cycle_with_mean(120e6, 150e6)  # a non-zero mean present
    plain = miner_damage(cy, curve)
    with_none = miner_damage(cy, curve, mean_stress_model=MeanStressModel.NONE,
                             ultimate_strength_pa=500e6, static_mean_pa=200e6)
    assert with_none == plain  # NONE ignores the mean entirely (regression guard)


def test_tensile_cycle_mean_shortens_life():
    curve = get_curve("D")
    ranges = 100e6
    d_none = miner_damage(_cycle_with_mean(ranges, 0.0), curve)
    d_goodman = miner_damage(
        _cycle_with_mean(ranges, 180e6), curve,
        mean_stress_model=MeanStressModel.GOODMAN, ultimate_strength_pa=500e6,
    )
    assert d_goodman > d_none  # tensile mean -> larger equivalent range -> more damage


def test_static_mean_offset_moves_damage_like_a_cycle_mean():
    curve = get_curve("D")
    # A static offset should act the same as an equal per-cycle mean.
    d_cycle_mean = miner_damage(
        _cycle_with_mean(100e6, 200e6), curve,
        mean_stress_model=MeanStressModel.GOODMAN, ultimate_strength_pa=500e6,
    )
    d_static = miner_damage(
        _cycle_with_mean(100e6, 0.0), curve,
        mean_stress_model=MeanStressModel.GOODMAN, ultimate_strength_pa=500e6,
        static_mean_pa=200e6,
    )
    assert d_static == pytest.approx(d_cycle_mean, rel=1e-12)


def test_mean_stress_requires_uts():
    curve = get_curve("D")
    with pytest.raises(ValueError):
        miner_damage(_cycle_with_mean(100e6, 100e6), curve,
                     mean_stress_model=MeanStressModel.GOODMAN)
