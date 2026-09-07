"""Tests for the marine-growth VIV knock-down sweep (Detection-tab live figure)."""

from __future__ import annotations

from scr_twin_core.config import AnalysisConfig, RiserConfig, VivConfig
from server.service import viv_life_sweep


def _cfg(current: float) -> AnalysisConfig:
    return AnalysisConfig(riser=RiserConfig.reference_scr(),
                          viv=VivConfig(surface_current=current))


def test_sweep_is_monotonic_knockdown():
    grid = [t / 1000.0 for t in range(0, 151, 30)]
    sw = viv_life_sweep(_cfg(0.6), grid)
    assert sw["enabled"] is True
    assert sw["thickness_mm"] == [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    life = sw["life_years"]
    # thicker growth -> strictly shorter VIV life (biofouling worsens VIV)
    assert all(life[i] > life[i + 1] for i in range(len(life) - 1))
    assert life[0] > 100.0  # clean riser is long-lived at this current
    assert life[-1] < life[0] / 5.0  # heavy growth is a big knock-down


def test_sweep_effective_diameter_grows():
    grid = [0.0, 0.05, 0.10]
    sw = viv_life_sweep(_cfg(0.6), grid)
    de = sw["effective_diameter_mm"]
    assert de[0] < de[1] < de[2]
    # D_eff = D + 2 t : +50 mm growth adds 100 mm of hydro diameter
    assert round(de[1] - de[0], 3) == 100.0
    assert sw["design_life_years"] == 25.0


def test_sweep_disabled_without_current():
    sw = viv_life_sweep(_cfg(0.0), [0.0, 0.05])
    assert sw["enabled"] is False
    assert sw["life_years"] == []
