"""API: validated H(f) upload + imported transfer-function route."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient

from server.main import app

client = TestClient(app)

HF_CSV = (
    "# source_tool: OrcaFlex\n# load_case: Hs=6.8 Tp=11 heading=180\n"
    "freq_hz,magnitude,phase_rad\n"
    "0.02,1e5,0\n0.10,4e6,-0.9\n0.20,2e6,-1.6\n0.40,1e5,-3.0\n"
)


def _cfg(route: str) -> dict:
    c = client.get("/api/reference-config").json()
    c["transfer"]["route"] = route
    return c


def _upload_hf() -> str:
    r = client.post("/api/transfer", files={"file": ("hf.csv", io.BytesIO(HF_CSV.encode()), "text/csv")})
    assert r.status_code == 200
    return r.json()["token"]


def test_transfer_upload_returns_token_and_provenance():
    r = client.post("/api/transfer", files={"file": ("hf.csv", io.BytesIO(HF_CSV.encode()), "text/csv")})
    assert r.status_code == 200
    b = r.json()
    assert b["token"]
    assert b["provenance"]["source_tool"] == "OrcaFlex"
    assert b["provenance"]["is_validated"] is True
    assert b["provenance"]["n_points"] == 4


def test_bad_transfer_csv_is_400():
    r = client.post("/api/transfer", files={"file": ("bad.csv", io.BytesIO(b"nope\n1\n"), "text/csv")})
    assert r.status_code == 400


def test_imported_route_analyze_uses_table():
    tok = _upload_hf()
    body = {"config": _cfg("imported"),
            "synthetic": {"hs": 4.0, "tp": 11.0, "gamma": 2.5, "duration": 1200, "fs": 4.0, "seed": 7},
            "transfer_token": tok}
    r = client.post("/api/analyze/synthetic", json=body)
    assert r.status_code == 200
    p = r.json()
    assert p["transfer"]["route"] == "imported"
    assert p["transfer"]["is_validated"] is True
    assert p["transfer"]["provenance"]["source_tool"] == "OrcaFlex"
    assert p["provenance"]["transfer_is_validated"] is True
    assert len(p["transfer"]["freq"]) > 0
    assert p["damage"]["deterministic_life_years"] > 0


def test_imported_route_without_token_is_400():
    body = {"config": _cfg("imported"), "synthetic": {}}
    r = client.post("/api/analyze/synthetic", json=body)
    assert r.status_code == 400


def test_reference_route_flagged_illustrative():
    body = {"config": _cfg("reference"),
            "synthetic": {"hs": 4.0, "tp": 11.0, "gamma": 2.5, "duration": 1200, "fs": 4.0, "seed": 7}}
    p = client.post("/api/analyze/synthetic", json=body).json()
    assert p["transfer"]["is_validated"] is False
    assert p["transfer"]["route"] == "reference"
