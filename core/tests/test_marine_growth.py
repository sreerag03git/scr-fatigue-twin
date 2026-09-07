"""Marine-growth (biofouling) tests (DNV-RP-C205 Sec. 4.8)."""

from __future__ import annotations

import math

import pytest

from scr_twin_core.constants import RHO_SEAWATER, G
from scr_twin_core.marine_growth import MarineGrowth


def test_zero_thickness_is_inert():
    mg = MarineGrowth(thickness_m=0.0)
    assert not mg.enabled
    assert mg.effective_diameter(0.32) == 0.32
    assert mg.mass_per_length(0.32) == 0.0
    assert mg.submerged_weight_per_length(0.32) == 0.0


def test_effective_diameter():
    mg = MarineGrowth(thickness_m=0.05)
    assert mg.effective_diameter(0.32) == pytest.approx(0.42)  # +2*50mm


def test_mass_per_length_reference():
    mg = MarineGrowth(thickness_m=0.05, density_kg_m3=1300.0)
    d, d_eff = 0.32, 0.42
    annulus = math.pi / 4.0 * (d_eff**2 - d**2)
    assert mg.mass_per_length(d) == pytest.approx(1300.0 * annulus, rel=1e-12)


def test_submerged_weight_is_net_of_buoyancy():
    mg = MarineGrowth(thickness_m=0.05, density_kg_m3=1300.0)
    d, d_eff = 0.32, 0.42
    annulus = math.pi / 4.0 * (d_eff**2 - d**2)
    expected = (1300.0 - RHO_SEAWATER) * annulus * G
    assert mg.submerged_weight_per_length(d) == pytest.approx(expected, rel=1e-12)
    assert mg.submerged_weight_per_length(d) > 0.0  # denser than seawater


def test_thicker_growth_adds_more_mass():
    d = 0.32
    assert MarineGrowth(0.1).mass_per_length(d) > MarineGrowth(0.05).mass_per_length(d)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        MarineGrowth(thickness_m=-0.01)
    with pytest.raises(ValueError):
        MarineGrowth(thickness_m=0.05, density_kg_m3=0.0)
