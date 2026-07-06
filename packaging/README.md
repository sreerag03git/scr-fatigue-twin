# Packaging — offline desktop build

The SCR-Twin console ships as a **standalone, offline desktop application** built
with PyInstaller. The single bundle contains Python, NumPy/SciPy/pandas, FastAPI,
the `scr_twin_core` physics library and the compiled React console. Double-click
`SCR-Twin.exe` and it starts a local server and opens the console in the default
browser — **no Python, Node, or network required**.

## Build it

```powershell
./packaging/build_desktop.ps1
# or, to avoid OneDrive sync locks, output outside the synced folder:
./packaging/build_desktop.ps1 -DistPath D:\builds\dist -WorkPath D:\builds\build
```

Output: `dist\SCR-Twin\SCR-Twin.exe` (a one-dir bundle; ship the whole
`SCR-Twin\` folder, or wrap it in an installer — see below).

Verified on this machine: the packaged exe boots, passes **9/9 acceptance gates**,
runs a full synthetic analysis, serves the console at `http://127.0.0.1:8000`,
and persists runs to `%LOCALAPPDATA%\SCR-Twin\runs.sqlite` — all offline.

Notes:
- **Cold start** is ~30–45 s on first launch (SciPy unpacks); later starts are quicker.
- Building inside OneDrive can intermittently fail with "Access is denied" as
  OneDrive locks files mid-build; use a non-synced `-DistPath` or pause sync.
- The results DB lives in `%LOCALAPPDATA%\SCR-Twin\` when frozen (writable,
  per-user), and in `data/` during development.

## Windows installer (Inno Setup)

The one-dir bundle is wrapped into a proper `Setup.exe` (per-user, no admin) with
Start-menu + optional desktop shortcuts and an uninstaller:

```powershell
winget install --id JRSoftware.InnoSetup      # one-time
./packaging/build_desktop.ps1                  # produce dist/SCR-Twin/
./packaging/build_installer.ps1                # -> dist/SCR-Twin-Setup-0.1.0.exe
```

`packaging/installer.iss` is the Inno Setup script; `build_installer.ps1` finds
`ISCC.exe` and passes the bundle path/version as `/D` defines.

Verified on this machine: the installer compiles (~87 MB Setup.exe from the 303 MB
bundle), **silent-installs (exit 0)** all files + frontend + Start-menu shortcut +
uninstaller (~310 MB installed to `%LOCALAPPDATA%\Programs\SCR-Twin`), and
**uninstalls cleanly (exit 0)** leaving no trace. Sign the Setup.exe with a code-
signing certificate (`signtool`) before public distribution.

## Native Tauri shell — status

The spec's preferred shell is Tauri (Rust). It is **not built here** because the
Windows MSVC C++ build tools (the linker Tauri/cargo require) are not installed
on this machine, and the automated install was blocked by an interactive
elevation prompt (VS Build Tools installer exit 1602). WebView2 and Rust are
present; only the MSVC toolchain is missing.

To enable the native Tauri build later:

1. Install **Visual Studio 2022 Build Tools** with the *Desktop development with
   C++* workload (provides `link.exe` + Windows SDK), approving the UAC prompt.
2. `rustup default stable-msvc`.
3. Add a Tauri shell over the Vite app (`npm create tauri-app` / `@tauri-apps/cli`),
   configured to launch this PyInstaller bundle as a **sidecar** (Tauri handles
   the window + installer; the bundle keeps providing the physics/API).

Per the spec's "reliability wins" guidance, the PyInstaller build is the shipped,
tested installable; Tauri is an additive native shell for when the toolchain is
provisioned. The `core`/`server`/`app` separation means adding it requires no
changes to the physics.
