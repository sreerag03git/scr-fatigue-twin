"""Cross-flow VIV screening tests (DNV-RP-F204 / RP-F105 reduced-order)."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.config import RiserConfig
from scr_twin_core.sn import get_curve
from scr_twin_core.viv import (
    CurrentProfile,
    effective_mass_per_length,
    griffin_amplitude,
    riser_modes,
    stability_parameter,
    viv_screening,
)


def _rig():
    r = RiserConfig.reference_scr()
    return r, r.pipe_section(), r.catenary()


def test_current_profile_shear():
    cp = CurrentProfile(surface_velocity=1.0, water_depth=1500.0)
    assert cp.speed_at_height(np.array([0.0]))[0] == pytest.approx(0.0)  # seabed
    assert cp.speed_at_height(np.array([1500.0]))[0] == pytest.approx(1.0)  # surface
    # monotonically increasing toward the surface
    h = np.linspace(0, 1500, 50)
    assert np.all(np.diff(cp.speed_at_height(h)) >= 0.0)


def test_modes_are_ascending_and_match_taut_string():
    r, sec, cat = _rig()
    modes = riser_modes(cat, sec, contents_density=r.contents_density, n_modes=6, n_nodes=600)
    assert np.all(np.diff(modes.frequencies_hz) > 0.0)  # ascending
    # cross-check the fundamental against the taut-string closed form with the
    # mean effective tension: fn ~ (n/2L) sqrt(T/m). Bending only stiffens it, so
    # the beam frequency is within a few % above the string for low modes.
    m = effective_mass_per_length(sec, contents_density=r.contents_density)
    t_eff = float(np.mean(cat.tension(np.linspace(0.0, cat.horizontal_span, 200))))
    fn1_string = (1.0 / (2.0 * modes.span_length)) * np.sqrt(t_eff / m)
    assert modes.frequencies_hz[0] == pytest.approx(fn1_string, rel=0.08)


def test_griffin_amplitude_decreases_with_stability():
    assert griffin_amplitude(0.0) == pytest.approx(1.29, abs=0.01)
    assert griffin_amplitude(2.0) < griffin_amplitude(0.5) < griffin_amplitude(0.0)
    assert griffin_amplitude(50.0) < 0.05  # heavily damped -> negligible VIV


def test_stability_parameter_scales():
    ks_light = stability_parameter(100.0, 0.02, 0.3)
    ks_heavy = stability_parameter(400.0, 0.02, 0.3)
    assert ks_heavy == pytest.approx(4.0 * ks_light)  # linear in mass


def test_no_current_gives_no_viv():
    r, sec, cat = _rig()
    sc = viv_screening(cat, sec, get_curve(r.sn_class),
                       CurrentProfile(0.0, r.water_depth), contents_density=r.contents_density)
    assert sc.annual_damage_rate == 0.0
    assert np.isinf(sc.life_years)
    assert sc.dominant_mode == 0


def test_moderate_current_excites_a_mode_and_gives_finite_life():
    r, sec, cat = _rig()
    sc = viv_screening(cat, sec, get_curve(r.sn_class),
                       CurrentProfile(0.9, r.water_depth), contents_density=r.contents_density,
                       thickness_m=r.thickness_for_correction)
    assert sc.annual_damage_rate > 0.0
    assert 0.0 < sc.life_years < np.inf
    assert sc.dominant_mode >= 1
    assert any(m.excited for m in sc.modes)
    # excited modes sit in the reduced-velocity lock-in band
    for m in sc.modes:
        if m.excited:
            assert 3.0 <= m.reduced_velocity <= 9.0


def test_stronger_current_shortens_viv_life():
    r, sec, cat = _rig()

    def life(u):
        return viv_screening(cat, sec, get_curve(r.sn_class),
                             CurrentProfile(u, r.water_depth), contents_density=r.contents_density,
                             thickness_m=r.thickness_for_correction).life_years

    assert life(1.2) < life(0.7)  # more current -> more VIV damage -> shorter life


def test_viv_is_deterministic():
    r, sec, cat = _rig()
    kw = dict(contents_density=r.contents_density, thickness_m=r.thickness_for_correction)
    a = viv_screening(cat, sec, get_curve(r.sn_class), CurrentProfile(0.9, r.water_depth), **kw)
    b = viv_screening(cat, sec, get_curve(r.sn_class), CurrentProfile(0.9, r.water_depth), **kw)
    assert a.annual_damage_rate == b.annual_damage_rate
