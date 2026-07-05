"""Transfer-function (Layer 1) tests: quasi-static gain, DAF, interpolation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scr_twin_core.catenary import solve_plain_catenary
from scr_twin_core.constants import RHO_SEAWATER
from scr_twin_core.section import PipeSection
from scr_twin_core.transfer import (
    REFERENCE_HF_PEAK_HZ,
    InterpolatedTransferFunction,
    analytic_transfer_function,
    apply_transfer_to_spectrum,
    linearized_drag_damping,
    quasi_static_moment_gain,
    reference_transfer_function,
)


@pytest.fixture
def rig():
    section = PipeSection(outer_diameter=0.3239, wall_thickness=0.0206)
    cat = solve_plain_catenary(water_depth=1500.0, top_angle_deg=72.0, submerged_weight=1500.0)
    return section, cat


def test_quasi_static_gain_finite_and_step_independent(rig):
    section, cat = rig
    g1 = quasi_static_moment_gain(cat, section, rel_step=1e-3)
    g2 = quasi_static_moment_gain(cat, section, rel_step=1e-4)
    assert math.isfinite(g1) and g1 != 0.0
    # central difference is stable across step sizes
    assert g1 == pytest.approx(g2, rel=1e-3)


def test_linearized_drag_formula():
    section = PipeSection(outer_diameter=0.5, wall_thickness=0.03)
    sigma_u, cd = 0.4, 1.2
    c_eq = linearized_drag_damping(section, sigma_velocity=sigma_u, drag_coefficient=cd)
    expected = math.sqrt(8 / math.pi) * sigma_u * 0.5 * RHO_SEAWATER * cd * 0.5
    assert c_eq == pytest.approx(expected)


def test_daf_static_resonance_rolloff(rig):
    section, cat = rig
    fn = 0.12
    freqs = np.array([1e-4, fn, 10 * fn])
    tf = analytic_transfer_function(
        freqs, cat, section, natural_frequency=fn, sigma_velocity=0.3
    )
    g_qs = quasi_static_moment_gain(cat, section)
    mag = tf.magnitude
    # near-static gain ~ |g_qs|
    assert mag[0] == pytest.approx(abs(g_qs), rel=1e-2)
    # resonance amplifies above the static gain
    assert mag[1] > mag[0]
    # far above resonance the response rolls off well below static
    assert mag[2] < 0.1 * mag[0]
    assert tf.is_reduced_order is True


def test_interpolated_transfer_function():
    ftab = np.array([0.05, 0.1, 0.2, 0.3])
    magtab = np.array([1.0e6, 2.0e6, 1.5e6, 0.5e6])
    phtab = np.array([0.0, -0.5, -1.0, -1.5])
    itf = InterpolatedTransferFunction(ftab, magtab, phtab)
    out = itf.evaluate(np.array([0.1, 0.15]))
    assert out.magnitude[0] == pytest.approx(2.0e6)
    assert out.magnitude[1] == pytest.approx(1.75e6)  # linear midpoint
    assert out.is_reduced_order is False


def test_apply_transfer_to_spectrum(rig):
    section, cat = rig
    freqs = np.linspace(0.01, 0.4, 50)
    tf = analytic_transfer_function(freqs, cat, section, natural_frequency=0.12, sigma_velocity=0.3)
    motion_psd = np.ones_like(freqs)
    out = apply_transfer_to_spectrum(motion_psd, tf)
    np.testing.assert_allclose(out, tf.magnitude**2)


def test_reference_transfer_function():
    f = np.linspace(0.0, 0.6, 400)
    tf = reference_transfer_function(f)
    assert tf.is_reduced_order is False  # stands in for an imported H(f)
    assert tf.magnitude.max() > 0.0
    # magnitude peaks in the wave band near the documented peak frequency
    assert abs(f[np.argmax(tf.magnitude)] - REFERENCE_HF_PEAK_HZ) < 0.03
    # no response below the wave band
    assert np.all(tf.magnitude[f < 0.02] == 0.0)


def test_interpolated_transfer_shape_mismatch_raises():
    with pytest.raises(ValueError):
        InterpolatedTransferFunction([0.1, 0.2], [1.0], [0.0, 0.0])
