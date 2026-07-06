"""Desktop entry point for the SCR-Twin console.

Starts the FastAPI backend (which also serves the built frontend) on a local
port and opens the default browser to it. Packaged with PyInstaller into a
self-contained, offline, double-clickable application that needs no separate
Python, Node, or network access. See packaging/README.md.
"""

from __future__ import annotations

import socket
import threading
import time
import webbrowser


def _pick_port(preferred: int = 8000) -> int:
    """Use the preferred port if free, otherwise an OS-assigned free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])


def main() -> None:
    import uvicorn

    from server.main import app

    port = _pick_port(8000)
    url = f"http://127.0.0.1:{port}"

    def _open() -> None:
        time.sleep(1.5)
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - opening a browser is best-effort
            pass

    threading.Thread(target=_open, daemon=True).start()
    print("\n  SCR-Twin - TDP Fatigue Integrity Console")
    print(f"  Running at {url}")
    print("  Close this window to quit.\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
