"""Arabian Gulf environmental-correction tests (acceptance gate: 30-39%)."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.environment import EnvironmentCorrection
from scr_twin_core.sn import cycles_to_failure, get_curve


def test_default_combined_factor_in_band():
    ec = EnvironmentCorrection()
    assert 0.61 <= ec.combined_factor <= 0.70
    # Acceptance gate (project spec 5): 30-39% life reduction.
    assert 0.30 <= ec.life_reduction <= 0.39


def test_correct_life_scales_by_factor():
    ec = EnvironmentCorrection()
    assert ec.correct_life(100.0) == pytest.approx(100.0 * ec.combined_factor)
    assert ec.correct_damage_rate(1.0) == pytest.approx(1.0 / ec.combined_factor)


def test_apply_to_curve_scales_N_by_factor():
    ec = EnvironmentCorrection()
    base = get_curve("D")
    corrected = ec.apply_to_curve(base)
    dsig = np.array([80e6, 120e6, 200e6])
    n_base = cycles_to_failure(dsig, base)
    n_corr = cycles_to_failure(dsig, corrected)
    np.testing.assert_allclose(n_corr / n_base, ec.combined_factor, rtol=1e-9)


def test_apply_to_curve_preserves_crossing_stress():
    ec = EnvironmentCorrection()
    base = get_curve("E")
    corrected = ec.apply_to_curve(base)
    assert corrected.fatigue_limit_mpa == pytest.approx(base.fatigue_limit_mpa, rel=1e-9)


def test_validate_accepts_defaults_and_rejects_out_of_band():
    EnvironmentCorrection().validate()  # defaults valid
    with pytest.raises(ValueError):
        EnvironmentCorrection(temperature_factor=0.50).validate()
    with pytest.raises(ValueError):
        EnvironmentCorrection(salinity_factor=0.99).validate()
