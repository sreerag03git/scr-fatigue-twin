"""6-DOF hang-off kinematics (paper Eq. 6) tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scr_twin_core.hang_off_kinematics import (
    PorchGeometry,
    resolve_hang_off,
)


def test_pure_heave_passthrough():
    heave = np.sin(np.linspace(0.0, 10.0, 100))
    r = resolve_hang_off({"heave": heave}, PorchGeometry())
    np.testing.assert_allclose(r.vertical, heave)
    np.testing.assert_allclose(r.inplane, 0.0, atol=1e-12)


def test_pure_pitch_lever_arm():
    pitch = np.full(50, 1.0)  # 1 deg constant
    r = resolve_hang_off({"pitch": pitch}, PorchGeometry(x_p=10.0))
    np.testing.assert_allclose(r.vertical, -10.0 * math.radians(1.0))  # z_ho = -x_p*pitch


def test_pure_roll_lever_arm():
    roll = np.full(50, 2.0)
    r = resolve_hang_off({"roll": roll}, PorchGeometry(y_p=5.0))
    np.testing.assert_allclose(r.vertical, 5.0 * math.radians(2.0))  # z_ho = +y_p*roll


def test_eq6_full_combination():
    rng = np.random.default_rng(0)
    heave, pitch, roll = rng.standard_normal(64), rng.standard_normal(64), rng.standard_normal(64)
    geom = PorchGeometry(x_p=8.0, y_p=3.0)
    r = resolve_hang_off({"heave": heave, "pitch": pitch, "roll": roll}, geom)
    expected = heave - 8.0 * np.deg2rad(pitch) + 3.0 * np.deg2rad(roll)
    np.testing.assert_allclose(r.vertical, expected)


def test_inplane_depends_on_azimuth():
    surge = np.full(40, 1.0)
    along = resolve_hang_off({"surge": surge}, PorchGeometry(riser_azimuth_deg=0.0))
    np.testing.assert_allclose(along.inplane, 1.0)  # surge is along the riser
    across = resolve_hang_off({"surge": surge}, PorchGeometry(riser_azimuth_deg=90.0))
    np.testing.assert_allclose(across.inplane, 0.0, atol=1e-12)  # riser perpendicular


def test_contributions_sum_to_variance():
    rng = np.random.default_rng(3)
    ch = {k: rng.standard_normal(2000) for k in ("heave", "pitch", "roll")}
    r = resolve_hang_off(ch, PorchGeometry(x_p=10.0, y_p=4.0))
    assert sum(r.contributions.values()) == pytest.approx(r.vertical_variance, rel=1e-6)
    assert sum(r.contribution_fractions().values()) == pytest.approx(1.0, abs=1e-6)


def test_pitch_dominates_when_large_on_a_long_lever_arm():
    # Small heave, large pitch on a big porch offset -> pitch drives the fatigue.
    rng = np.random.default_rng(7)
    ch = {
        "heave": 0.1 * rng.standard_normal(4000),
        "pitch": 5.0 * rng.standard_normal(4000),
        "roll": 0.5 * rng.standard_normal(4000),
    }
    r = resolve_hang_off(ch, PorchGeometry(x_p=20.0, y_p=4.0))
    assert r.contribution_fractions()["pitch"] > 0.7


def test_exact_matches_small_angle_for_small_angles():
    rng = np.random.default_rng(4)
    ch = {
        "heave": rng.standard_normal(500), "surge": rng.standard_normal(500),
        "sway": rng.standard_normal(500),
        "pitch": 0.3 * rng.standard_normal(500), "roll": 0.3 * rng.standard_normal(500),
        "yaw": 0.3 * rng.standard_normal(500),
    }
    geom = PorchGeometry(x_p=8.0, y_p=3.0, z_p=15.0, riser_azimuth_deg=30.0)
    small = resolve_hang_off(ch, geom, exact=False)
    exact = resolve_hang_off(ch, geom, exact=True)
    assert np.std(small.vertical - exact.vertical) < 0.02 * np.std(small.vertical)
    assert np.std(small.inplane - exact.inplane) < 0.02 * np.std(small.inplane)


def test_length_mismatch_and_empty_raise():
    with pytest.raises(ValueError):
        resolve_hang_off({"heave": np.zeros(10), "pitch": np.zeros(9)}, PorchGeometry())
    with pytest.raises(ValueError):
        resolve_hang_off({}, PorchGeometry())


def test_pipeline_6dof_uses_resolved_motion_and_reports_contributions():
    from scr_twin_core.config import AnalysisConfig, HangOffConfig, RiserConfig
    from scr_twin_core.pipeline import run_full_analysis
    from scr_twin_core.synthetic import synthetic_mru_6dof

    m = synthetic_mru_6dof(duration=900.0, fs=4.0, hs=4.0, tp=11.0, seed=3, heading_deg=25.0)
    cfg = AnalysisConfig(
        riser=RiserConfig.reference_scr(),
        hang_off=HangOffConfig(porch_x=20.0, porch_z=25.0),
        n_monte_carlo=500,
    )
    res = run_full_analysis(cfg, m.heave, m.fs, motion_channels=m.channels)
    assert set(res.dof_contributions) >= {"heave", "pitch", "roll"}
    assert sum(res.dof_contributions.values()) == pytest.approx(1.0, abs=1e-6)
    assert res.dof_contributions["pitch"] > 0.03  # a 20 m porch makes pitch matter
    assert res.deterministic_life_years > 0.0


def test_pipeline_heave_only_backward_compatible():
    from scr_twin_core.config import AnalysisConfig, RiserConfig
    from scr_twin_core.pipeline import run_full_analysis
    from scr_twin_core.synthetic import synthetic_mru_motion

    m = synthetic_mru_motion(duration=600.0, fs=4.0, hs=4.0, tp=11.0, seed=3)
    res = run_full_analysis(AnalysisConfig(riser=RiserConfig.reference_scr(), n_monte_carlo=500),
                            m.heave, m.fs)
    assert res.dof_contributions == {"heave": 1.0}
