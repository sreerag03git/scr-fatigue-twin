"""Catenary tests vs. closed-form solutions and internal identities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scr_twin_core.catenary import (
    solve_plain_catenary,
    solve_plain_catenary_fixed_span,
)


@pytest.fixture
def cat():
    # d = 1000 m, top angle 70 deg from horizontal, w = 1000 N/m
    return solve_plain_catenary(water_depth=1000.0, top_angle_deg=70.0, submerged_weight=1000.0)


def test_catenary_parameter_closed_form(cat):
    theta = math.radians(70.0)
    a_expected = 1000.0 * math.cos(theta) / (1.0 - math.cos(theta))
    assert cat.catenary_parameter == pytest.approx(a_expected, rel=1e-12)
    assert cat.horizontal_tension == pytest.approx(1000.0 * a_expected)
    assert cat.tdp_curvature == pytest.approx(1.0 / a_expected)


def test_shape_endpoints_and_tangent(cat):
    a = cat.catenary_parameter
    # TDP at origin, tangent horizontal, curvature 1/a
    assert cat.shape(np.array([0.0]))[0] == pytest.approx(0.0)
    assert cat.curvature(np.array([0.0]))[0] == pytest.approx(1.0 / a)
    # top of the line reaches the water depth
    assert cat.shape(np.array([cat.horizontal_span]))[0] == pytest.approx(cat.water_depth, rel=1e-9)
    # tangent angle at top equals the hang-off angle
    slope = math.sinh(cat.horizontal_span / a)
    assert math.atan(slope) == pytest.approx(cat.top_angle, rel=1e-9)


def test_tension_force_balance(cat):
    # T_top^2 = H^2 + V_top^2, with V_top = w * arc_length
    v_top = cat.submerged_weight * cat.arc_length
    assert cat.top_tension**2 == pytest.approx(
        cat.horizontal_tension**2 + v_top**2, rel=1e-9
    )
    # T(x) = H + w*y(x)
    x = np.array([cat.horizontal_span])
    assert cat.tension(x)[0] == pytest.approx(
        cat.horizontal_tension + cat.submerged_weight * cat.shape(x)[0], rel=1e-9
    )


def test_fixed_span_solver_round_trip(cat):
    # Re-solving for the same span/depth recovers the same catenary parameter.
    again = solve_plain_catenary_fixed_span(
        cat.horizontal_span, cat.water_depth, cat.submerged_weight
    )
    assert again.catenary_parameter == pytest.approx(cat.catenary_parameter, rel=1e-6)
    assert again.top_angle == pytest.approx(cat.top_angle, rel=1e-6)


def test_deeper_water_fixed_span_increases_tdp_curvature(cat):
    shallow = solve_plain_catenary_fixed_span(cat.horizontal_span, 900.0, cat.submerged_weight)
    deep = solve_plain_catenary_fixed_span(cat.horizontal_span, 1100.0, cat.submerged_weight)
    assert deep.tdp_curvature > shallow.tdp_curvature


@pytest.mark.parametrize(
    "d,ang,w",
    [(-1.0, 70.0, 1000.0), (1000.0, 0.0, 1000.0), (1000.0, 95.0, 1000.0), (1000.0, 70.0, -5.0)],
)
def test_invalid_inputs_raise(d, ang, w):
    with pytest.raises(ValueError):
        solve_plain_catenary(water_depth=d, top_angle_deg=ang, submerged_weight=w)
