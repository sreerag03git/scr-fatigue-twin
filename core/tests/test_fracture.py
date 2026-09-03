"""Paris-law fracture-mechanics crack-growth tests (BS 7910)."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.fracture import (
    InitialFlawDistribution,
    ParisMaterial,
    crack_growth_curve,
    crack_growth_mc,
    crack_life_years,
    cycles_to_grow,
    equivalent_stress_range_mpa,
    stress_intensity_range,
)

MAT = ParisMaterial.bs7910_air_mean()


def test_stress_intensity_reference():
    # Y=1.12, dsig=100 MPa, a=1 mm -> dK = 1.12*100*sqrt(pi*0.001)
    dk = float(stress_intensity_range(100.0, 1e-3))
    assert dk == pytest.approx(1.12 * 100 * np.sqrt(np.pi * 1e-3), rel=1e-9)


def test_closed_form_matches_rk4():
    a0, ac, ds, cpy = 0.5e-3, 8e-3, 90.0, 4e6
    n_closed = cycles_to_grow(a0, ac, MAT, ds)
    t, a = crack_growth_curve(a0, MAT, ds, cpy, a_c=ac, years=50, n_steps=6000)
    idx = np.argmax(a >= ac * 0.999)
    t_reach = t[idx] if a[idx] >= ac * 0.999 else np.inf
    assert t_reach == pytest.approx(n_closed / cpy, rel=0.03)


def test_closed_form_analytic_value_m3():
    # For m=3: N = 2 (a0^-0.5 - ac^-0.5) / [C (Y dsig sqrt(pi))^3]
    a0, ac, ds = 1e-3, 5e-3, 100.0
    b = MAT.C * (1.12 * ds * np.sqrt(np.pi)) ** 3
    n_hand = 2.0 * (a0**-0.5 - ac**-0.5) / b
    assert cycles_to_grow(a0, ac, MAT, ds) == pytest.approx(n_hand, rel=1e-9)


def test_threshold_gate_stops_growth():
    # A low stress range keeps dK below the threshold -> no propagation.
    assert cycles_to_grow(0.5e-3, 8e-3, MAT, 2.0) == float("inf")
    assert crack_life_years(0.5e-3, 8e-3, MAT, 2.0, 1e6) == float("inf")


def test_crack_depth_monotonic_and_clamped():
    _, a = crack_growth_curve(0.5e-3, MAT, 90.0, 4e6, a_c=8e-3, years=80, n_steps=2000)
    assert np.all(np.diff(a) >= -1e-15)          # non-decreasing
    assert a[-1] <= 8e-3 + 1e-12                  # clamped at a_c


def test_life_scales_inversely_with_cycle_rate():
    l1 = crack_life_years(0.5e-3, 8e-3, MAT, 90.0, 2e6)
    l2 = crack_life_years(0.5e-3, 8e-3, MAT, 90.0, 4e6)
    assert l1 == pytest.approx(2.0 * l2, rel=1e-9)


def test_higher_stress_shortens_life():
    assert (crack_life_years(0.5e-3, 8e-3, MAT, 120.0, 4e6)
            < crack_life_years(0.5e-3, 8e-3, MAT, 80.0, 4e6))


def test_equivalent_range_power_mean():
    edges = np.array([0.0, 40e6, 120e6])   # bins 20 and 80 MPa centres
    counts = np.array([100.0, 10.0])
    eq = equivalent_stress_range_mpa(edges, counts, 3.0)
    hand = ((100 * 20.0**3 + 10 * 80.0**3) / 110) ** (1 / 3)
    assert eq == pytest.approx(hand, rel=1e-9)


def test_marine_grows_faster_than_air():
    air = crack_life_years(0.5e-3, 8e-3, ParisMaterial.bs7910_air_mean(), 90.0, 4e6)
    cp = crack_life_years(0.5e-3, 8e-3, ParisMaterial.bs7910_marine_cp(), 90.0, 4e6)
    assert cp < air


def test_mc_is_deterministic_and_ordered():
    d = InitialFlawDistribution()
    a = crack_growth_mc(d, MAT, 90.0, 4e6, a_c=8e-3, n_members=1500, seed=3)
    b = crack_growth_mc(d, MAT, 90.0, 4e6, a_c=8e-3, n_members=1500, seed=3)
    np.testing.assert_array_equal(a, b)
    finite = a[np.isfinite(a)]
    assert finite.size > 0  # at least some flaws propagate at 90 MPa


def test_material_rejects_bad_params():
    with pytest.raises(ValueError):
        ParisMaterial(C=-1.0, m=3.0, delta_k_th=2.0)
    with pytest.raises(ValueError):
        cycles_to_grow(5e-3, 1e-3, MAT, 90.0)  # a0 >= a_c
