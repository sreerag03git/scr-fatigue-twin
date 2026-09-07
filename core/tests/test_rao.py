"""Vessel-RAO import + spectrum-driven motion tests."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.rao import RAOProvenance, VesselRAO, load_rao_csv


def _rao():
    f = np.linspace(0.02, 0.5, 40)
    heave = (1.0 / np.sqrt((1 - (f * 12) ** 2) ** 2 + (2 * 0.2 * f * 12) ** 2)).astype(complex)
    pitch = (3.0 * (f * 10) ** 2).astype(complex)
    return f, {"heave": heave, "pitch": pitch}


def test_build_and_dof_order():
    f, ch = _rao()
    r = VesselRAO(f, ch, provenance=RAOProvenance(source_tool="AQWA", heading_deg=180))
    assert r.dof == ["heave", "pitch"]  # canonical order
    assert r.provenance.is_validated
    assert r.provenance.n_points == 40


def test_evaluate_interpolates_and_absent_is_zero():
    f, ch = _rao()
    r = VesselRAO(f, ch)
    v = r.evaluate(np.array([0.1, 0.2]), "heave")
    assert v.shape == (2,) and np.all(np.isfinite(v))
    assert np.all(r.evaluate(f, "roll") == 0.0)  # DOF not in table


def test_synthesize_shape_and_determinism():
    f, ch = _rao()
    r = VesselRAO(f, ch)
    a = r.synthesize(hs=4.0, tp=11.0, duration=600, fs=4.0, seed=7)
    b = r.synthesize(hs=4.0, tp=11.0, duration=600, fs=4.0, seed=7)
    assert set(a) == {"heave", "pitch"}
    assert a["heave"].shape == a["pitch"].shape
    np.testing.assert_array_equal(a["heave"], b["heave"])
    assert np.std(a["heave"]) > 0.0


def test_bigger_sea_state_gives_more_motion():
    f, ch = _rao()
    r = VesselRAO(f, ch)
    calm = np.std(r.synthesize(hs=2.0, tp=11.0, duration=1200, fs=4.0, seed=1)["heave"])
    rough = np.std(r.synthesize(hs=6.0, tp=11.0, duration=1200, fs=4.0, seed=1)["heave"])
    assert rough > calm


def test_load_csv_roundtrip_and_provenance():
    csv = ("# source_tool: OrcaFlex\n# heading_deg: 180\n"
           "freq_hz,heave_mag,heave_phase_deg,pitch_mag,pitch_phase_deg\n"
           "0.05,1.0,0,0.1,10\n0.10,0.9,5,0.5,20\n0.20,0.4,15,0.8,40\n")
    r = load_rao_csv(csv)
    assert r.dof == ["heave", "pitch"]
    assert r.provenance.source_tool == "OrcaFlex"
    assert r.provenance.heading_deg == 180.0
    assert r.provenance.n_points == 3


def test_load_csv_rejects_malformed():
    with pytest.raises(ValueError):
        load_rao_csv("a,b\n1,2\n")                       # no freq column
    with pytest.raises(ValueError):
        load_rao_csv("freq_hz,foo\n0.1,1.0\n")           # no <dof>_mag column


def test_build_rejects_bad_channels():
    f = np.linspace(0.02, 0.5, 10)
    with pytest.raises(ValueError):
        VesselRAO(f, {"bogus": np.ones(10, dtype=complex)})
    with pytest.raises(ValueError):
        VesselRAO(f, {"heave": np.ones(5, dtype=complex)})  # wrong length
