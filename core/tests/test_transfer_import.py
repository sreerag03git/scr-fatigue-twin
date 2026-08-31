"""Validated H(f) import path: CSV loader + imported route through the pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.config import AnalysisConfig, RiserConfig, TransferConfig
from scr_twin_core.pipeline import run_full_analysis
from scr_twin_core.synthetic import synthetic_mru_motion
from scr_twin_core.transfer import (
    InterpolatedTransferFunction,
    load_transfer_csv,
)

MAG_PHASE_CSV = (
    "# source_tool: OrcaFlex\n"
    "# tool_version: 11.4\n"
    "# load_case: Hs=6.8m Tp=11s heading=180 draft=survival\n"
    "freq_hz,magnitude,phase_rad\n"
    "0.02,1.0e5,0.0\n0.05,1.5e6,-0.3\n0.10,4.0e6,-0.9\n0.20,2.0e6,-1.6\n0.30,0.5e6,-2.4\n0.40,1.0e5,-3.0\n"
)


def test_load_magnitude_phase_with_provenance():
    tf = load_transfer_csv(MAG_PHASE_CSV)
    assert isinstance(tf, InterpolatedTransferFunction)
    assert tf.provenance.source_tool == "OrcaFlex"
    assert tf.provenance.tool_version == "11.4"
    assert tf.provenance.load_case.startswith("Hs=6.8m")
    assert tf.provenance.is_validated is True
    assert tf.provenance.n_points == 6
    out = tf.evaluate(np.array([0.10]))
    assert out.magnitude[0] == pytest.approx(4.0e6, rel=1e-6)
    assert out.is_reduced_order is False


def test_load_real_imag():
    csv = "freq,re,im\n0.05,1.0e6,0.0\n0.10,0.0,-3.0e6\n0.20,-1.0e6,0.0\n"
    tf = load_transfer_csv(csv, source_tool="RIFLEX")
    out = tf.evaluate(np.array([0.10]))
    assert out.magnitude[0] == pytest.approx(3.0e6, rel=1e-6)
    assert np.angle(out.value[0]) == pytest.approx(-np.pi / 2, abs=1e-6)
    assert tf.provenance.source_tool == "RIFLEX"


def test_phase_in_degrees_detected():
    csv = "freq_hz,magnitude,phase_deg\n0.05,1e6,0\n0.10,2e6,-90\n0.20,1e6,-180\n"
    tf = load_transfer_csv(csv)
    out = tf.evaluate(np.array([0.10]))
    assert np.angle(out.value[0]) == pytest.approx(-np.pi / 2, abs=1e-6)


def test_overrides_beat_header():
    tf = load_transfer_csv(MAG_PHASE_CSV, load_case="override case")
    assert tf.provenance.load_case == "override case"
    assert tf.provenance.source_tool == "OrcaFlex"  # header kept where not overridden


@pytest.mark.parametrize("bad", ["", "a,b,c\n1,2,3\n", "freq_hz,magnitude\n0.1,1e6\n"])
def test_malformed_csv_rejected_loudly(bad):
    with pytest.raises(ValueError):
        load_transfer_csv(bad)


def test_negative_magnitude_rejected():
    csv = "freq_hz,magnitude,phase_rad\n0.1,-1e6,0\n0.2,1e6,0\n"
    with pytest.raises(ValueError):
        load_transfer_csv(csv)


def _cfg(route: str) -> AnalysisConfig:
    return AnalysisConfig(
        riser=RiserConfig.reference_scr(),
        transfer=TransferConfig(route=route),
        n_monte_carlo=500,
    )


def test_imported_route_uses_table_and_is_validated():
    tf = load_transfer_csv(MAG_PHASE_CSV)
    m = synthetic_mru_motion(duration=600.0, fs=4.0, hs=3.0, tp=10.0, seed=1)
    res = run_full_analysis(_cfg("imported"), m.heave, m.fs, imported_tf=tf)
    assert res.transfer_route == "imported"
    assert res.transfer_provenance["is_validated"] is True
    assert res.transfer_provenance["source_tool"] == "OrcaFlex"
    assert res.provenance.transfer_is_validated is True
    assert res.deterministic_life_years > 0.0
    # |H(f)| curve exposed for the Fig-4 view
    assert res.hf_freqs.size == res.hf_stress_mag.size == res.hf_moment_mag.size > 0
    assert np.all(res.hf_stress_mag >= 0.0)


def test_imported_route_requires_a_table():
    m = synthetic_mru_motion(duration=600.0, fs=4.0, hs=3.0, tp=10.0, seed=1)
    with pytest.raises(ValueError, match="imported"):
        run_full_analysis(_cfg("imported"), m.heave, m.fs)  # no imported_tf


def test_reference_and_analytic_routes_flagged_not_validated():
    m = synthetic_mru_motion(duration=600.0, fs=4.0, hs=3.0, tp=10.0, seed=1)
    for route in ("reference", "analytic"):
        res = run_full_analysis(_cfg(route), m.heave, m.fs)
        assert res.provenance.transfer_is_validated is False
        assert res.transfer_provenance["is_validated"] is False
