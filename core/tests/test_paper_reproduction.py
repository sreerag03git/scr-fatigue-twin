"""Reproduction checks against SPE-235285-MS (the paper this twin implements)."""

from __future__ import annotations

import pytest

from scr_twin_core.config import RiserConfig
from scr_twin_core.montecarlo import (
    convexity_divergence_curves,
    convexity_median_divergence,
)
from scr_twin_core.sn import SNEnvironment
from scr_twin_core.validation import narrowband_bendat_vs_rainflow


def test_paper_case_study_matches_table2():
    p = RiserConfig.paper_case_study()
    assert p.outer_diameter == pytest.approx(0.324)
    assert p.wall_thickness == pytest.approx(0.028)          # 28 mm
    assert p.water_depth == pytest.approx(1200.0)
    assert p.hang_off_angle_deg == pytest.approx(12.0)       # from vertical
    assert p.sn_class == "D"
    assert p.sn_environment is SNEnvironment.SEAWATER_CP
    assert p.design_fatigue_factor == pytest.approx(10.0)
    assert p.design_service_life_years == pytest.approx(25.0)
    # wall exceeds the 25 mm reference, so the thickness correction is exercised
    assert p.thickness_for_correction > 0.025
    # the catenary solves for a representative configuration
    assert p.catenary().catenary_parameter > 0.0


def test_convexity_relation():
    # (1 + c^2)^(m(m-1)/2) - 1 ; zero at c=0, and the paper operating point
    assert convexity_median_divergence(0.0, 4.0) == pytest.approx(0.0)
    assert convexity_median_divergence(0.16, 4.0) == pytest.approx(0.164, abs=0.01)  # ~15% in paper
    # monotone increasing in both c and m
    assert convexity_median_divergence(0.20, 4.0) > convexity_median_divergence(0.10, 4.0)
    assert convexity_median_divergence(0.16, 5.0) > convexity_median_divergence(0.16, 3.0)


def test_convexity_curves_shape():
    curves = convexity_divergence_curves([0.0, 0.1, 0.2, 0.3], slopes=(3.0, 4.0, 5.0))
    assert set(curves) == {3.0, 4.0, 5.0}
    for vals in curves.values():
        assert vals[0] == pytest.approx(0.0)
        assert vals == sorted(vals)  # non-decreasing in c


def test_narrowband_bendat_agreement():
    rows = narrowband_bendat_vs_rainflow(seed=0)
    assert [r["m"] for r in rows] == [3.0, 3.5, 4.0, 4.5, 5.0]
    # rainflow+Miner matches the Bendat (1964) closed form within a few percent
    assert max(r["rel_error"] for r in rows) < 0.03
