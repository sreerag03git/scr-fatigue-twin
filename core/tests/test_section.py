"""Pipe section geometry and submerged-weight tests."""

from __future__ import annotations

import math

import pytest

from scr_twin_core.constants import RHO_SEAWATER, RHO_STEEL, G
from scr_twin_core.section import PipeSection, submerged_weight


@pytest.fixture
def pipe() -> PipeSection:
    return PipeSection(outer_diameter=0.3, wall_thickness=0.03)


def test_geometry_closed_form(pipe):
    assert pipe.inner_diameter == pytest.approx(0.24)
    assert pipe.steel_area == pytest.approx(math.pi / 4 * (0.3**2 - 0.24**2))
    assert pipe.second_moment_area == pytest.approx(math.pi / 64 * (0.3**4 - 0.24**4))
    assert pipe.section_modulus == pytest.approx(pipe.second_moment_area / 0.15)


def test_moment_curvature_stress_consistency(pipe):
    kappa = 1e-3
    m = pipe.moment_from_curvature(kappa)
    # sigma via moment == sigma via curvature == E (D/2) kappa
    assert pipe.stress_from_moment(m) == pytest.approx(pipe.stress_from_curvature(kappa))
    assert pipe.stress_from_curvature(kappa) == pytest.approx(pipe.youngs_modulus * 0.15 * kappa)


def test_submerged_weight_empty_pipe(pipe):
    w = submerged_weight(pipe)  # empty, no coating
    expected = pipe.steel_area * RHO_STEEL * G - (math.pi / 4 * 0.3**2) * RHO_SEAWATER * G
    assert w == pytest.approx(expected)
    assert w > 0.0  # steel-heavy empty pipe is net-submerged


def test_flooded_pipe_heavier_than_empty(pipe):
    empty = submerged_weight(pipe)
    flooded = submerged_weight(pipe, contents_density=1025.0)
    assert flooded > empty


@pytest.mark.parametrize("od,t", [(0.0, 0.01), (0.3, 0.0), (0.3, 0.2)])
def test_invalid_section_raises(od, t):
    with pytest.raises(ValueError):
        PipeSection(outer_diameter=od, wall_thickness=t)
