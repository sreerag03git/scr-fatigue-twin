"""Seabed elastic-foundation TDP-correction tests (Pesce / Lenci)."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.seabed import (
    boundary_layer_length,
    seabed_correction,
    seabed_sensitivity,
    soil_length,
)

EI, H = 4.69e7, 8.60e5  # reference SCR bending stiffness and TDP tension


def test_length_scales():
    assert boundary_layer_length(EI, H) == pytest.approx(np.sqrt(EI / H))
    assert soil_length(EI, 200e3) == pytest.approx((4 * EI / 200e3) ** 0.25)


def test_correction_in_unit_interval_and_rigid_limit():
    cs = seabed_correction(EI, H, 200e3)
    assert 0.0 < cs < 1.0
    # k_v -> inf gives Cs -> 1 (rigid seabed).
    assert seabed_correction(EI, H, 1e20) == pytest.approx(1.0, abs=1e-3)


def test_stiffer_soil_gives_larger_correction():
    assert seabed_correction(EI, H, 50e3) < seabed_correction(EI, H, 5000e3)


def test_sensitivity_band_brackets_conservative_base():
    s = seabed_sensitivity(EI, H, 33.0, sn_slope_m=3.0)
    # Cs < 1 everywhere -> compliant seabed lives EXCEED the rigid base.
    assert s.life_soft > s.life_stiff > s.base_life_years
    # life factor is Cs^-m
    assert np.allclose(s.life_factor, s.correction ** (-3.0))
    assert np.all(np.diff(s.correction) > 0)  # Cs increases with k_v


def test_sensitivity_is_ordered():
    s = seabed_sensitivity(EI, H, 33.0)
    assert s.life_years[0] > s.life_years[-1]  # soft (index 0) -> longer life


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        boundary_layer_length(-1.0, H)
    with pytest.raises(ValueError):
        soil_length(EI, 0.0)
