"""S-N curve tests vs. DNV-RP-C203 Table 2-1 tabulated points."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.sn import (
    DNV_C203_IN_AIR,
    DNV_C203_SEAWATER_CP,
    MeanStressModel,
    SNEnvironment,
    apply_mean_stress,
    cycles_to_failure,
    get_curve,
    thickness_factor,
)

# DNV-RP-C203 Table 2-1: stress range (MPa) at the N = 1e7 slope change.
FATIGUE_LIMIT_1E7 = {
    "B1": 106.97,
    "B2": 93.59,
    "C": 73.10,
    "C1": 65.50,
    "C2": 58.48,
    "D": 52.63,
    "E": 46.78,
    "F": 41.52,
    "F1": 36.84,
    "F3": 32.75,
    "G": 29.24,
}


@pytest.mark.parametrize("name,expected", FATIGUE_LIMIT_1E7.items())
def test_fatigue_limit_matches_dnv_table(name, expected):
    curve = get_curve(name)
    assert curve.fatigue_limit_mpa == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize("name", list(DNV_C203_IN_AIR))
def test_two_branches_meet_at_transition(name):
    curve = get_curve(name)
    dsig_pa = curve.fatigue_limit_mpa * 1e6
    n = cycles_to_failure(np.array([dsig_pa]), curve)[0]
    assert n == pytest.approx(curve.n_transition, rel=1e-2)


def test_class_d_reference_point():
    # Class D at 100 MPa: logN = 12.164 - 3*log10(100) = 6.164
    n = cycles_to_failure(np.array([100e6]), get_curve("D"))[0]
    assert np.log10(n) == pytest.approx(6.164, abs=1e-3)


def test_class_d_low_stress_uses_shallow_branch():
    # 30 MPa < 52.63 MPa limit -> m2=5 branch: logN = 15.606 - 5*log10(30)
    n = cycles_to_failure(np.array([30e6]), get_curve("D"))[0]
    expected = 15.606 - 5.0 * np.log10(30.0)
    assert np.log10(n) == pytest.approx(expected, abs=1e-3)


def test_monotonic_decreasing_life_with_stress():
    curve = get_curve("F")
    dsig = np.array([20e6, 40e6, 80e6, 160e6])
    n = cycles_to_failure(dsig, curve)
    assert np.all(np.diff(n) < 0.0)


def test_zero_stress_is_non_damaging():
    n = cycles_to_failure(np.array([0.0, -5e6]), get_curve("D"))
    assert np.all(np.isinf(n))


def test_thickness_correction_reduces_life():
    curve = get_curve("F1")  # k = 0.25
    thin = cycles_to_failure(np.array([100e6]), curve, thickness_m=0.025)[0]
    thick = cycles_to_failure(np.array([100e6]), curve, thickness_m=0.050)[0]
    assert thick < thin
    # factor (50/25)^0.25 on stress
    assert thickness_factor(0.050, 0.25) == pytest.approx(2.0**0.25, rel=1e-9)


def test_thickness_below_reference_is_unity():
    assert thickness_factor(0.010, 0.25) == pytest.approx(1.0)


def test_goodman_increases_equivalent_range():
    dsig = np.array([100e6])
    mean = np.array([200e6])
    eq = apply_mean_stress(dsig, mean, MeanStressModel.GOODMAN, ultimate_strength_pa=500e6)
    assert eq[0] > dsig[0]


def test_goodman_reference_value():
    # dS=100, sig_m=200, sig_u=500 -> S_eq = 100/(1-0.4) = 166.667 MPa
    eq = apply_mean_stress(np.array([100e6]), np.array([200e6]),
                           MeanStressModel.GOODMAN, ultimate_strength_pa=500e6)
    assert eq[0] == pytest.approx(166.6667e6, rel=1e-5)


def test_gerber_reference_value_and_less_conservative_than_goodman():
    # dS=100, sig_m=200, sig_u=500 -> S_eq = 100/(1-0.16) = 119.048 MPa
    gerber = apply_mean_stress(np.array([100e6]), np.array([200e6]),
                               MeanStressModel.GERBER, ultimate_strength_pa=500e6)
    goodman = apply_mean_stress(np.array([100e6]), np.array([200e6]),
                                MeanStressModel.GOODMAN, ultimate_strength_pa=500e6)
    assert gerber[0] == pytest.approx(119.0476e6, rel=1e-5)
    assert gerber[0] < goodman[0]  # Gerber parabola is less conservative


def test_compressive_mean_takes_no_credit():
    # A compressive (negative) mean is clipped to zero -> range unchanged.
    for model in (MeanStressModel.GOODMAN, MeanStressModel.GERBER):
        eq = apply_mean_stress(np.array([100e6]), np.array([-200e6]),
                               model, ultimate_strength_pa=500e6)
        assert eq[0] == pytest.approx(100e6, rel=1e-12)


def test_zero_mean_is_inert_for_every_model():
    dsig = np.array([120e6])
    for model in (MeanStressModel.GOODMAN, MeanStressModel.GERBER):
        eq = apply_mean_stress(dsig, np.array([0.0]), model, ultimate_strength_pa=500e6)
        assert eq[0] == pytest.approx(dsig[0], rel=1e-12)


def test_unknown_class_raises():
    with pytest.raises(KeyError):
        get_curve("Z9")


# --- seawater-with-cathodic-protection S-N family (DNV Table 2-2) ---------- #
def test_seawater_cp_reference_intercepts():
    # Published DNV-RP-C203 Table 2-2 log_a1 values.
    assert DNV_C203_SEAWATER_CP["D"].log_a1 == pytest.approx(11.764, abs=1e-3)
    assert DNV_C203_SEAWATER_CP["B1"].log_a1 == pytest.approx(14.917, abs=1e-3)


def test_seawater_cp_knee_moved_to_1e6():
    cp = get_curve("D", SNEnvironment.SEAWATER_CP)
    assert cp.n_transition == 1.0e6
    # Knee stress for D in seawater-CP ~ 83.4 MPa at 1e6 cycles.
    assert cp.fatigue_limit_mpa == pytest.approx(83.4, rel=2e-3)
    n = cycles_to_failure(np.array([cp.fatigue_limit_mpa * 1e6]), cp)[0]
    assert n == pytest.approx(1.0e6, rel=1e-2)


def test_seawater_cp_shallow_branch_matches_air():
    for name, air in DNV_C203_IN_AIR.items():
        cp = DNV_C203_SEAWATER_CP[name]
        assert cp.m2 == air.m2
        assert cp.log_a2 == pytest.approx(air.log_a2, rel=1e-12)  # coincide for N > 1e7


def test_seawater_cp_more_severe_than_air_in_wave_band():
    # At a typical wave-band range (100 MPa) seawater-CP gives fewer cycles.
    dsig = np.array([100e6])
    n_air = cycles_to_failure(dsig, get_curve("D", SNEnvironment.IN_AIR))[0]
    n_cp = cycles_to_failure(dsig, get_curve("D", SNEnvironment.SEAWATER_CP))[0]
    assert n_cp < n_air


def test_get_curve_defaults_to_air():
    assert get_curve("F1").log_a1 == get_curve("F1", SNEnvironment.IN_AIR).log_a1
    assert get_curve("F1").log_a1 == DNV_C203_IN_AIR["F1"].log_a1
