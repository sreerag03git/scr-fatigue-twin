"""RunStore unit tests (SQLite results store)."""

from __future__ import annotations

from server.store import RunStore

PAYLOAD = {
    "provenance": {"motion_is_synthetic": True, "config_sha256": "abc123"},
    "posterior": {"p10": 50.0, "p50": 110.0, "p90": 250.0, "n_members": 10000},
    "damage": {"deterministic_life_years": 113.0},
    "inspection": {"next_inspection_year": 26.0},
}


def test_save_and_get(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    rid = store.save(source="synthetic", config={"seed": 0}, payload=PAYLOAD)
    assert rid >= 1
    got = store.get(rid)
    assert got is not None
    assert got["config"]["seed"] == 0
    assert got["payload"]["posterior"]["p50"] == 110.0
    assert got["life_p50"] == 110.0
    assert got["is_synthetic"] == 1


def test_list_newest_first_and_count(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    ids = [store.save(source="synthetic", config={"seed": i}, payload=PAYLOAD) for i in range(3)]
    rows = store.list()
    assert [r["id"] for r in rows] == list(reversed(ids))
    assert store.count() == 3


def test_get_missing_returns_none(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    assert store.get(999) is None


def test_persists_across_instances(tmp_path):
    path = tmp_path / "runs.sqlite"
    RunStore(path).save(source="upload", config={}, payload=PAYLOAD)
    assert RunStore(path).count() == 1  # a new instance sees the persisted row
