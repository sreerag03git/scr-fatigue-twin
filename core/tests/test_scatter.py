"""Long-term scatter-diagram fatigue summation tests (DNV-RP-C203 Sec. 5)."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.config import AnalysisConfig, RiserConfig
from scr_twin_core.pipeline import long_term_fatigue, sea_state_annual_damage_rate
from scr_twin_core.scatter import (
    ScatterCell,
    ScatterDiagram,
    aggregate_long_term,
    example_scatter_diagram,
    load_scatter_csv,
)


def test_two_cell_hand_summation():
    d = ScatterDiagram([ScatterCell(2.0, 8.0, 0.7), ScatterCell(5.0, 12.0, 0.3)])
    lt = aggregate_long_term(d, lambda hs, tp: 0.01 if hs < 3.0 else 0.05)
    assert lt.annual_damage_rate == pytest.approx(0.7 * 0.01 + 0.3 * 0.05)  # 0.022
    assert lt.life_years == pytest.approx(1.0 / 0.022)


def test_single_cell_reproduces_rate():
    d = ScatterDiagram([ScatterCell(3.0, 9.0, 1.0)])
    lt = aggregate_long_term(d, lambda hs, tp: 0.03)
    assert lt.annual_damage_rate == pytest.approx(0.03, rel=1e-12)


def test_aggregate_is_convex_combination_of_cell_rates():
    d = example_scatter_diagram()
    rates = {(c.hs, c.tp): 1e-3 * (1.0 + c.hs) for c in d.cells}
    lt = aggregate_long_term(d, lambda hs, tp: rates[(hs, tp)])
    assert min(rates.values()) <= lt.annual_damage_rate <= max(rates.values())


def test_contributions_sum_to_one_and_are_sorted():
    d = example_scatter_diagram()
    lt = aggregate_long_term(d, lambda hs, tp: hs**3)  # storm-weighted
    total = sum(c.damage_fraction for c in lt.contributions)
    assert total == pytest.approx(1.0, abs=1e-9)
    fracs = [c.damage_fraction for c in lt.contributions]
    assert fracs == sorted(fracs, reverse=True)


def test_probabilities_are_renormalised():
    d = ScatterDiagram([ScatterCell(2.0, 8.0, 7.0), ScatterCell(5.0, 12.0, 3.0)])  # sum 10
    assert sum(c.probability for c in d.cells) == pytest.approx(1.0)


def test_rejects_malformed_diagrams():
    with pytest.raises(ValueError):
        ScatterDiagram([])
    with pytest.raises(ValueError):
        ScatterDiagram([ScatterCell(2.0, 8.0, -0.1), ScatterCell(2.0, 8.0, 1.1)])  # negative prob
    with pytest.raises(ValueError):
        ScatterDiagram([ScatterCell(2.0, 8.0, 0.0)])  # zero total
    with pytest.raises(ValueError):
        ScatterDiagram([ScatterCell(-1.0, 8.0, 1.0)])  # non-positive Hs


def test_load_scatter_csv_roundtrip():
    csv = "# source: unit-test\nHs,Tp,prob\n1.5,8,0.6\n3.5,10,0.4\n"
    d = load_scatter_csv(csv)
    assert d.source == "unit-test"
    assert {round(c.hs, 1) for c in d.cells} == {1.5, 3.5}
    assert sum(c.probability for c in d.cells) == pytest.approx(1.0)


def test_load_scatter_csv_rejects_missing_columns():
    with pytest.raises(ValueError):
        load_scatter_csv("a,b\n1,2\n")


# --- end-to-end physics ---------------------------------------------------- #
def _config() -> AnalysisConfig:
    return AnalysisConfig(riser=RiserConfig.reference_scr(), seed=0, n_monte_carlo=1000)


def test_storm_cell_has_higher_rate_than_calm_cell():
    cfg = _config()
    calm = sea_state_annual_damage_rate(cfg, 1.0, 8.0)
    storm = sea_state_annual_damage_rate(cfg, 6.0, 12.0)
    assert storm > calm > 0.0


def test_long_term_life_is_finite_and_positive():
    cfg = _config()
    lt = long_term_fatigue(cfg, example_scatter_diagram())
    assert 0.0 < lt.annual_damage_rate < np.inf
    assert lt.life_years > 0.0
    assert lt.n_cells == 19
    # the rare severe cells should carry a disproportionate share of the damage
    top = lt.contributions[0]
    assert top.hs >= 3.5  # a storm cell dominates, not the most-probable calm cell


def test_long_term_is_deterministic():
    cfg = _config()
    a = long_term_fatigue(cfg, example_scatter_diagram()).annual_damage_rate
    b = long_term_fatigue(cfg, example_scatter_diagram()).annual_damage_rate
    assert a == b
