"""S-N curve tests vs. DNV-RP-C203 Table 2-1 tabulated points."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.sn import (
    DNV_C203_IN_AIR,
    MeanStressModel,
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


def test_unknown_class_raises():
    with pytest.raises(KeyError):
        get_curve("Z9")
