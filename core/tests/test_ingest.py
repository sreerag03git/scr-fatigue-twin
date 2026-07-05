"""Ingestion fuzz/robustness tests (project spec 5): never crash, always report."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from scr_twin_core.ingest import ingest_mru, load_mru_csv

SAMPLE_CSV = Path(__file__).resolve().parents[2] / "data" / "samples" / "synthetic_mru_reference_scr.csv"


def _clean(n=4000, fs=4.0):
    t = np.arange(n) / fs
    heave = 1.5 * np.sin(2 * np.pi * 0.1 * t)
    pitch = 2.0 * np.cos(2 * np.pi * 0.09 * t)
    return t, {"heave": heave, "pitch": pitch}


def test_clean_record_ingests():
    t, ch = _clean()
    r = ingest_mru(t, ch)
    assert r.health.ok
    assert r.fs == pytest.approx(4.0, rel=1e-6)
    assert set(r.channels) == {"heave", "pitch"}
    assert len(r.blocks) >= 1
    assert np.mean(r.channels["heave"]) == pytest.approx(0.0, abs=1e-9)  # de-meaned


def test_gappy_time_is_filled_and_flagged():
    t, ch = _clean()
    keep = np.ones(t.size, bool)
    keep[1000:1200] = False  # excise a chunk -> a gap
    r = ingest_mru(t[keep], {k: v[keep] for k, v in ch.items()})
    assert r.health.ok
    assert r.health.n_gaps >= 1
    assert any("gap" in f for f in r.health.flags)


def test_nans_interpolated_and_flagged():
    t, ch = _clean()
    ch["heave"][500:520] = np.nan
    r = ingest_mru(t, ch)
    assert r.health.ok
    assert r.health.nan_count >= 20
    assert np.isfinite(r.channels["heave"]).all()


def test_non_monotonic_time_sorted():
    t, ch = _clean()
    t[2000:2010] = t[2000:2010][::-1]  # local reversal
    r = ingest_mru(t, ch)
    assert r.health.non_monotonic
    assert np.all(np.diff(r.time) > 0)


def test_empty_does_not_crash():
    r = ingest_mru(np.array([]), {"heave": np.array([])})
    assert not r.health.ok
    assert r.channels == {}


def test_wrong_units_flagged():
    t, ch = _clean()
    ch["heave"] = ch["heave"] * 1000.0  # metres -> millimetres by mistake
    r = ingest_mru(t, ch)
    assert any("units" in f.lower() or "plausible" in f.lower() for f in r.health.flags)


def test_unknown_columns_only():
    t = np.arange(100) / 4.0
    r = ingest_mru(t, {"temperature": np.ones(100), "pressure": np.ones(100)})
    assert not r.health.ok
    assert any("no recognised" in f for f in r.health.flags)


def test_clipping_detected():
    t, ch = _clean()
    ch["heave"] = np.clip(ch["heave"], -0.5, 0.5)  # rail the signal
    r = ingest_mru(t, ch)
    assert r.health.clipped_fraction > 0.01
    assert any("clip" in f.lower() for f in r.health.flags)


def test_missing_time_uses_assumed_fs():
    _, ch = _clean()
    r = ingest_mru(None, ch, assumed_fs=4.0)
    assert r.health.ok
    assert r.fs == pytest.approx(4.0)
    assert any("assuming fs" in f for f in r.health.flags)


@pytest.mark.parametrize(
    "bad_text",
    ["", "not,a,valid\nfile\n", "heave\n\n\n", ",,,\n1,2,3\n"],
)
def test_malformed_csv_never_crashes(bad_text):
    r = load_mru_csv(io.StringIO(bad_text))
    assert isinstance(r.health.flags, list)  # returns a report, no exception
    assert not r.health.ok


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample dataset not generated")
def test_sample_csv_loads():
    r = load_mru_csv(str(SAMPLE_CSV))
    assert r.health.ok
    assert "heave" in r.channels
    assert r.fs == pytest.approx(4.0, rel=0.01)
