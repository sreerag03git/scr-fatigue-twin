"""API contract + reliability tests (the backend must never 500 on bad input)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.main import app

client = TestClient(app)
SAMPLE = Path(__file__).resolve().parents[2] / "data" / "samples" / "synthetic_mru_reference_scr.csv"


@pytest.fixture(scope="module")
def config():
    return client.get("/api/reference-config").json()


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_reference_config_shape(config):
    assert config["riser"]["sn_class"] in {"D", "E", "F", "F1", "F3", "G", "B1", "B2", "C", "C1", "C2"}
    assert config["transfer"]["route"] == "reference"


def test_validation_all_gates_pass():
    r = client.get("/api/validation").json()
    assert r["passed"] == r["total"]
    assert r["total"] >= 9


def test_economics_in_paper_range():
    e = client.get("/api/economics").json()
    assert 5.5e6 <= e["fleet_saving_low_usd"] <= 8.0e6
    assert 33e6 <= e["fleet_saving_high_usd"] <= 40e6


def test_analyze_synthetic_full_payload(config):
    body = {"config": config, "synthetic": {"hs": 4.0, "tp": 11.0, "gamma": 2.5, "duration": 1200, "fs": 4.0, "seed": 7}}
    r = client.post("/api/analyze/synthetic", json=body)
    assert r.status_code == 200
    p = r.json()
    for key in ["sea_state", "spectrum", "damage", "posterior", "bayesian_fan", "inspection", "economics", "provenance"]:
        assert key in p
    assert p["posterior"]["p10"] < p["posterior"]["p50"] < p["posterior"]["p90"]
    assert p["provenance"]["motion_is_synthetic"] is True


def test_storm_shortens_life(config):
    def life(hs):
        body = {"config": config, "synthetic": {"hs": hs, "tp": 11.0, "gamma": 2.5, "duration": 1200, "fs": 4.0, "seed": 7}}
        return client.post("/api/analyze/synthetic", json=body).json()["damage"]["deterministic_life_years"]

    assert life(7.0) < life(3.0)  # rougher sea -> shorter fatigue life


def test_deterministic_repeat(config):
    body = {"config": config, "synthetic": {"hs": 4.0, "tp": 11.0, "gamma": 2.5, "duration": 1200, "fs": 4.0, "seed": 7}}
    a = client.post("/api/analyze/synthetic", json=body).json()
    b = client.post("/api/analyze/synthetic", json=body).json()
    assert a["damage"]["annual_rate_time"] == b["damage"]["annual_rate_time"]
    assert a["posterior"]["p50"] == b["posterior"]["p50"]


def test_ingest_malformed_never_500():
    r = client.post("/api/ingest", files={"file": ("bad.csv", io.BytesIO(b"@@@\ngarbage\n"), "text/csv")})
    assert r.status_code == 200
    assert r.json()["health"]["ok"] is False


def test_ingest_empty_never_500():
    r = client.post("/api/ingest", files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")})
    assert r.status_code == 200
    assert r.json()["health"]["ok"] is False


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample dataset not generated")
def test_ingest_and_analyze_sample(config):
    with open(SAMPLE, "rb") as f:
        r = client.post("/api/ingest", files={"file": ("sample.csv", f, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["health"]["ok"] is True
    token = body["token"]

    a = client.post("/api/analyze/upload", json={"config": config, "token": token})
    assert a.status_code == 200
    assert a.json()["damage"]["deterministic_life_years"] > 0
    assert a.json()["provenance"]["motion_is_synthetic"] is False


def test_analyze_upload_unknown_token(config):
    r = client.post("/api/analyze/upload", json={"config": config, "token": "deadbeef"})
    assert r.status_code == 404


def test_invalid_config_rejected(config):
    bad = {**config, "riser": {**config["riser"], "wall_thickness": 5.0}}  # t > OD/2
    r = client.post("/api/analyze/synthetic", json={"config": bad, "synthetic": {}})
    assert r.status_code == 422  # pydantic validation error
