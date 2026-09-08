"""Circumferential (clock-position) TDP girth-weld fatigue map tests."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.circumferential import (
    SECONDS_PER_YEAR,
    _annual_damage_from_ranges,
    circumferential_damage_map,
)
from scr_twin_core.sn import MeanStressModel, cycles_to_failure, get_curve

CURVE = get_curve("D")


def _wave_hist():
    # A simple hot-spot rainflow block: 1e5 cycles at 60 MPa over one hour.
    ranges = np.array([60.0e6])
    counts = np.array([1.0e5])
    block_s = 3600.0
    return ranges, counts, block_s


def test_in_plane_only_peaks_at_crown_and_keel():
    ranges, counts, block_s = _wave_hist()
    r = circumferential_damage_map(
        wave_ranges_pa=ranges, wave_counts=counts, block_seconds=block_s,
        curve=CURVE, heading_deg=0.0, n_positions=72,
    )
    # Damage is maximal in the catenary plane (phi = 0, 180) and ~zero at the
    # neutral axis (phi = 90, 270) for pure in-plane bending.
    ang = r.angles_deg
    d = r.damage_rate
    i0 = int(np.argmin(np.abs(ang - 0.0)))
    i90 = int(np.argmin(np.abs(ang - 90.0)))
    i180 = int(np.argmin(np.abs(ang - 180.0)))
    assert d[i0] == pytest.approx(d[i180], rel=1e-9)
    assert d[i90] < d[i0] * 1e-6
    assert r.worst_angle_deg in (0.0, 180.0)


def test_heading_rotates_the_wave_hot_spot():
    ranges, counts, block_s = _wave_hist()
    r = circumferential_damage_map(
        wave_ranges_pa=ranges, wave_counts=counts, block_seconds=block_s,
        curve=CURVE, heading_deg=30.0, n_positions=360,
    )
    # An oblique heading moves the peak to phi = beta (and beta + 180).
    peak = r.angles_deg[int(np.argmax(r.damage_rate))]
    assert min(abs(peak - 30.0), abs(peak - 210.0)) <= 1.0


def test_crown_reproduces_the_direct_miner_life():
    ranges, counts, block_s = _wave_hist()
    r = circumferential_damage_map(
        wave_ranges_pa=ranges, wave_counts=counts, block_seconds=block_s,
        curve=CURVE, heading_deg=0.0, thickness_m=None, n_positions=72,
    )
    # At phi = 0 with heading 0 the geometric factor is 1, so the crown damage
    # must equal the plain block Miner damage rate independently computed.
    direct = _annual_damage_from_ranges(ranges, counts, block_s, CURVE, None)
    assert r.wave_rate[0] == pytest.approx(direct, rel=1e-9)
    assert r.crown_life_years == pytest.approx(1.0 / direct, rel=1e-9)


def test_viv_adds_an_out_of_plane_lobe():
    ranges, counts, block_s = _wave_hist()
    # A modest wave block plus a cross-flow VIV mode at 40 MPa, 0.5 Hz.
    r = circumferential_damage_map(
        wave_ranges_pa=ranges, wave_counts=counts, block_seconds=block_s,
        curve=CURVE, viv_mode_ranges_pa=[40.0e6], viv_mode_freqs_hz=[0.5],
        heading_deg=0.0, n_positions=72,
    )
    ang = r.angles_deg
    i90 = int(np.argmin(np.abs(ang - 90.0)))
    # VIV contributes purely out-of-plane: non-zero at phi = 90, zero-ish in-plane.
    assert r.viv_rate[i90] > 0.0
    i0 = int(np.argmin(np.abs(ang - 0.0)))
    assert r.viv_rate[i0] < r.viv_rate[i90] * 1e-6
    # The VIV narrow-band rate matches fn * yr / N(range) at its peak.
    n_fail = float(cycles_to_failure(np.array([40.0e6]), CURVE)[0])
    assert r.viv_rate[i90] == pytest.approx(0.5 * SECONDS_PER_YEAR / n_fail, rel=1e-9)


def test_worst_life_is_the_minimum_around_the_weld():
    ranges, counts, block_s = _wave_hist()
    r = circumferential_damage_map(
        wave_ranges_pa=ranges, wave_counts=counts, block_seconds=block_s,
        curve=CURVE, viv_mode_ranges_pa=[35.0e6], viv_mode_freqs_hz=[0.6],
        heading_deg=20.0, n_positions=180,
    )
    assert r.worst_life_years == pytest.approx(float(np.min(r.life_years)), rel=1e-12)
    assert r.best_life_years == pytest.approx(float(np.max(r.life_years)), rel=1e-12)
    assert r.worst_life_years <= r.crown_life_years + 1e-9


def test_colocated_life_is_the_conservative_bound():
    # The co-located combined life (wave & VIV peaks assumed at one point) must be
    # the shortest life anywhere on the weld - the conservative governing figure.
    ranges, counts, block_s = _wave_hist()
    r = circumferential_damage_map(
        wave_ranges_pa=ranges, wave_counts=counts, block_seconds=block_s,
        curve=CURVE, viv_mode_ranges_pa=[45.0e6], viv_mode_freqs_hz=[0.55],
        heading_deg=15.0, n_positions=180,
    )
    assert r.combined_colocated_life_years <= r.worst_life_years + 1e-9
    assert r.combined_colocated_life_years <= float(np.min(r.life_years)) + 1e-9


def test_mean_stress_correction_shortens_life():
    ranges, counts, block_s = _wave_hist()
    base = circumferential_damage_map(
        wave_ranges_pa=ranges, wave_counts=counts, block_seconds=block_s,
        curve=CURVE, heading_deg=0.0,
    )
    corrected = circumferential_damage_map(
        wave_ranges_pa=ranges, wave_counts=counts, block_seconds=block_s,
        curve=CURVE, heading_deg=0.0,
        mean_stress_model=MeanStressModel.GOODMAN, static_mean_pa=200.0e6,
        ultimate_strength_pa=531.0e6,
    )
    # A tensile static mean with Goodman raises the effective range -> shorter life.
    assert corrected.crown_life_years < base.crown_life_years


def test_mismatched_viv_lengths_raise():
    ranges, counts, block_s = _wave_hist()
    with pytest.raises(ValueError, match="same length"):
        circumferential_damage_map(
            wave_ranges_pa=ranges, wave_counts=counts, block_seconds=block_s,
            curve=CURVE, viv_mode_ranges_pa=[40.0e6, 30.0e6], viv_mode_freqs_hz=[0.5],
        )


def test_as_dict_is_json_shaped():
    ranges, counts, block_s = _wave_hist()
    d = circumferential_damage_map(
        wave_ranges_pa=ranges, wave_counts=counts, block_seconds=block_s,
        curve=CURVE, n_positions=36,
    ).as_dict()
    assert set(d) >= {"angles_deg", "damage_rate", "wave_rate", "viv_rate",
                      "life_years", "worst_angle_deg", "worst_life_years",
                      "crown_life_years", "best_life_years",
                      "combined_colocated_life_years", "heading_deg"}
    assert len(d["angles_deg"]) == 36
    assert all(isinstance(v, float) for v in d["damage_rate"])
