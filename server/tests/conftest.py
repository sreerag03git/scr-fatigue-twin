import os
import pathlib
import sys
import tempfile

# Ensure the repo root is importable so `import server.main` works under pytest.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# Point the results store at a throwaway DB before server.main is imported, so
# tests never touch the real data/runs.sqlite. A file (not :memory:) is used
# because the TestClient runs endpoints across threads.
_db = pathlib.Path(tempfile.gettempdir()) / "scr_twin_test_runs.sqlite"
if _db.exists():
    _db.unlink()
os.environ["SCR_TWIN_DB"] = str(_db)
