import pathlib
import sys

# Ensure the repo root is importable so `import server.main` works under pytest.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
