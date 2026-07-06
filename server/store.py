"""SQLite results store — run history, posteriors, and provenance.

Local-only persistence (no network, no telemetry). Each analysis run is stored
with its config, seed, provenance and full result payload so any result can be
listed, re-opened, and reproduced from its exported provenance (spec §3, §7).

A fresh connection is opened per call so the store is safe to use from FastAPI's
threadpool without shared-connection hazards.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    source       TEXT NOT NULL,
    is_synthetic INTEGER NOT NULL,
    config_sha   TEXT NOT NULL,
    det_life     REAL,
    life_p10     REAL,
    life_p50     REAL,
    life_p90     REAL,
    next_insp    REAL,
    config_json  TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
"""


class RunStore:
    """Append-only SQLite store of analysis runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # Keep an in-memory connection alive; file DBs open per-call.
        self._mem = sqlite3.connect(self.path) if self.path == ":memory:" else None
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        if self._mem is not None:
            return self._mem
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, *, source: str, config: dict[str, Any], payload: dict[str, Any]) -> int:
        """Persist a run and return its id."""
        prov = payload.get("provenance", {})
        post = payload.get("posterior", {})
        dmg = payload.get("damage", {})
        insp = payload.get("inspection", {})
        row = (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source,
            1 if prov.get("motion_is_synthetic") else 0,
            str(prov.get("config_sha256", "")),
            dmg.get("deterministic_life_years"),
            post.get("p10"), post.get("p50"), post.get("p90"),
            insp.get("next_inspection_year"),
            json.dumps(config),
            json.dumps(payload),
        )
        conn = self._conn()
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "INSERT INTO runs (created_at, source, is_synthetic, config_sha, det_life,"
                " life_p10, life_p50, life_p90, next_insp, config_json, payload_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                row,
            )
            conn.commit()
            return int(cur.lastrowid or 0)
        finally:
            if self._mem is None:
                conn.close()

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recent runs (summary rows, newest first)."""
        conn = self._conn()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, created_at, source, is_synthetic, config_sha, det_life,"
                " life_p10, life_p50, life_p90, next_insp FROM runs"
                " ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            if self._mem is None:
                conn.close()

    def get(self, run_id: int) -> dict[str, Any] | None:
        """Full stored run (config + payload) for re-open / reproduction."""
        conn = self._conn()
        conn.row_factory = sqlite3.Row
        try:
            r = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if r is None:
                return None
            d = dict(r)
            d["config"] = json.loads(d.pop("config_json"))
            d["payload"] = json.loads(d.pop("payload_json"))
            return d
        finally:
            if self._mem is None:
                conn.close()

    def count(self) -> int:
        conn = self._conn()
        try:
            n = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            return int(n)
        finally:
            if self._mem is None:
                conn.close()
