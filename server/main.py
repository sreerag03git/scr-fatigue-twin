"""FastAPI application exposing scr_twin_core to the console frontend.

Design goals: never return a 500 white-screen (every handler validates and
degrades to a 400 with a clear message), fully offline (serves the built
frontend itself), and typed request/response contracts via Pydantic.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from scr_twin_core import __version__ as core_version
from scr_twin_core.config import AnalysisConfig, RiserConfig
from scr_twin_core.ingest import IngestedMRU, load_mru_csv, load_mru_parquet
from scr_twin_core.inspection import EconomicsModel, fleet_economics
from scr_twin_core.sn import DNV_C203_IN_AIR
from scr_twin_core.validation import run_all_gates

from . import service
from .schemas import (
    AnalyzeSyntheticRequest,
    AnalyzeUploadRequest,
    EconomicsParams,
    HealthResponse,
    SyntheticParams,
)
from .store import RunStore


def _resource_base() -> Path:
    """Root for bundled resources (repo root normally; _MEIPASS when frozen)."""
    if getattr(sys, "frozen", False):  # PyInstaller bundle
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    """Writable location for the results DB (user-data dir when frozen)."""
    if getattr(sys, "frozen", False):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "SCR-Twin"
        base.mkdir(parents=True, exist_ok=True)
        return base
    return _resource_base() / "data"


app = FastAPI(title="SCR Fatigue Digital Twin", version=core_version)

# Local results store (spec §3). Path overridable via env for tests/packaging.
_STORE = RunStore(os.environ.get("SCR_TWIN_DB", str(_data_dir() / "runs.sqlite")))


def _persist(source: dict[str, Any], config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Attach source metadata, save the run, and return the payload with run_id."""
    payload["source"] = source
    try:
        payload["run_id"] = _STORE.save(source=str(source.get("kind", "?")), config=config, payload=payload)
    except Exception:  # noqa: BLE001 - persistence must never break the response
        payload["run_id"] = None
    return payload

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store of ingested uploads (local, no persistence, private by design).
_UPLOADS: dict[str, IngestedMRU] = {}


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok", core_version=core_version,
        numpy_version=np.__version__, scipy_version=scipy.__version__,
    )


@app.get("/api/reference-config")
def reference_config() -> dict[str, Any]:
    return AnalysisConfig(riser=RiserConfig.reference_scr()).model_dump()


@app.get("/api/sn-classes")
def sn_classes() -> list[dict[str, Any]]:
    return [
        {
            "name": c.name, "m1": c.m1, "log_a1": c.log_a1, "m2": c.m2,
            "log_a2": c.log_a2, "thickness_exponent": c.thickness_exponent,
            "fatigue_limit_mpa": c.fatigue_limit_mpa,
        }
        for c in DNV_C203_IN_AIR.values()
    ]


@app.get("/api/validation")
def validation() -> dict[str, Any]:
    results = run_all_gates(seed=0)
    return {
        "gates": [r.as_dict() for r in results],
        "passed": int(sum(bool(r.passed) for r in results)),
        "total": len(results),
    }


@app.get("/api/economics")
def economics_default() -> dict[str, Any]:
    return service.to_native(fleet_economics(EconomicsModel()).as_dict())


@app.post("/api/economics")
def economics(params: EconomicsParams) -> dict[str, Any]:
    model = EconomicsModel(**params.model_dump())
    return service.to_native(fleet_economics(model).as_dict())


@app.post("/api/analyze/synthetic")
def analyze_synthetic(req: AnalyzeSyntheticRequest) -> dict[str, Any]:
    try:
        s: SyntheticParams = req.synthetic
        heave, fs = service.make_synthetic(s.hs, s.tp, s.gamma, s.duration, s.fs, s.seed)
        payload = service.analyze(req.config, heave, fs, is_synthetic=True)
        return _persist({"kind": "synthetic", **s.model_dump()}, req.config.model_dump(), payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    name = (file.filename or "").lower()
    try:
        if name.endswith(".parquet"):
            rec = load_mru_parquet(io.BytesIO(raw))
        else:
            rec = load_mru_csv(io.StringIO(raw.decode("utf-8", errors="replace")))
    except Exception as exc:  # noqa: BLE001 - ingestion must never 500
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}") from exc

    token = uuid.uuid4().hex
    _UPLOADS[token] = rec
    preview = {
        "time": service.decimate(rec.time, 600) if rec.time.size else [],
        "heave": service.decimate(rec.channels.get("heave", np.array([])), 600),
    }
    return {"token": token, "health": rec.health.as_dict(), "preview": preview}


@app.post("/api/analyze/upload")
def analyze_upload(req: AnalyzeUploadRequest) -> dict[str, Any]:
    rec = _UPLOADS.get(req.token)
    if rec is None:
        raise HTTPException(status_code=404, detail="Unknown upload token (re-upload the file)")
    if not rec.health.ok or "heave" not in rec.channels:
        raise HTTPException(
            status_code=400,
            detail="Ingested record is not analysable: " + "; ".join(rec.health.flags),
        )
    try:
        payload = service.analyze(
            req.config, rec.channels["heave"], rec.fs,
            is_synthetic=rec.is_synthetic, data_health=rec.health.as_dict(),
        )
        return _persist({"kind": "upload", "channels": rec.health.channels}, req.config.model_dump(), payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    return {"runs": _STORE.list(limit=50), "count": _STORE.count()}


@app.get("/api/runs/{run_id}")
def get_run(run_id: int) -> dict[str, Any]:
    run = _STORE.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run


@app.get("/api/runs/{run_id}/export")
def export_run(run_id: int) -> Response:
    """Download a self-contained, reproducible provenance bundle for a run."""
    run = _STORE.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    bundle = {
        "run_id": run["id"],
        "created_at": run["created_at"],
        "source": run["payload"].get("source"),
        "config": run["config"],
        "provenance": run["payload"].get("provenance"),
        "summary": {
            "sea_state": run["payload"].get("sea_state"),
            "damage": run["payload"].get("damage"),
            "posterior": {k: run["payload"]["posterior"].get(k) for k in ("p10", "p50", "p90", "n_members")},
            "inspection": run["payload"].get("inspection", {}).get("next_inspection_year"),
            "economics": run["payload"].get("economics"),
        },
        "note": "Reproduce with identical config + seed; see provenance for library versions.",
    }
    body = json.dumps(bundle, indent=2)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="scr-twin-run-{run_id}-provenance.json"'},
    )


@app.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    """Stream a synthetic motion record in chunks, then push the analysis.

    A lightweight 'live MRU' feel: frames arrive over ~6 s, then one posterior
    update lands. Guarded so a client disconnect never crashes the server.
    """
    await ws.accept()
    try:
        params = await ws.receive_json()
        s = SyntheticParams(**params.get("synthetic", {}))
        config = AnalysisConfig(**params["config"]) if "config" in params else \
            AnalysisConfig(riser=RiserConfig.reference_scr())
        heave, fs = service.make_synthetic(s.hs, s.tp, s.gamma, s.duration, s.fs, s.seed)
        t = np.arange(heave.size) / fs

        n_chunks = 30
        bounds = np.linspace(0, heave.size, n_chunks + 1).astype(int)
        for i in range(n_chunks):
            a, b = bounds[i], bounds[i + 1]
            await ws.send_json({
                "type": "frame",
                "t": [float(v) for v in t[a:b:4]],
                "heave": [float(v) for v in heave[a:b:4]],
                "progress": (i + 1) / n_chunks,
            })
            await asyncio.sleep(0.12)

        payload = service.analyze(config, heave, fs, is_synthetic=True)
        await ws.send_json({"type": "analysis", "payload": payload})
        await ws.send_json({"type": "done"})
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001 - report to client, keep server alive
        try:
            await ws.send_json({"type": "error", "detail": str(exc)})
        except Exception:  # noqa: BLE001
            pass


@app.exception_handler(Exception)
async def _unhandled(_: Any, exc: Exception) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})


# Serve the built frontend (offline, single process) when present.
_DIST = _resource_base() / "app" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="app")
