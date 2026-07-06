# Build the standalone, offline SCR-Twin desktop application (PyInstaller one-dir).
#
# Produces <DistPath>\SCR-Twin\SCR-Twin.exe — a double-clickable app that bundles
# Python, NumPy/SciPy/pandas, FastAPI, the physics core and the built React
# console. It runs the whole thing on a local port with no separate Python, Node,
# or network access, and opens the default browser to it.
#
# Usage (from anywhere):
#   ./packaging/build_desktop.ps1
#   ./packaging/build_desktop.ps1 -DistPath D:\builds\dist   # avoid OneDrive locks
#
# NOTE: Building inside a OneDrive-synced folder can intermittently fail with
# "Access is denied" as OneDrive locks files mid-build. If that happens, pass a
# -DistPath/-WorkPath outside OneDrive, or pause OneDrive sync during the build.

param(
  [string]$DistPath = "dist",
  [string]$WorkPath = "build"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv python not found at $py — create the venv and install core first." }

Write-Host "==> Building frontend (npm run build)" -ForegroundColor Cyan
$env:Path = "C:\Program Files\nodejs;$env:Path"
Push-Location (Join-Path $root "app")
npm install --no-audit --no-fund
npm run build
Pop-Location

Write-Host "==> Ensuring PyInstaller is installed" -ForegroundColor Cyan
& $py -m pip install pyinstaller --quiet

Write-Host "==> Packaging with PyInstaller (this takes a few minutes)" -ForegroundColor Cyan
& $py -m PyInstaller --noconfirm --name SCR-Twin `
  --distpath $DistPath --workpath $WorkPath `
  --paths core `
  --add-data "app/dist;app/dist" `
  --collect-all scipy --collect-all pandas --collect-all pyarrow `
  --collect-submodules uvicorn --collect-submodules scr_twin_core `
  --hidden-import server.main `
  desktop.py

Write-Host ""
Write-Host "==> Done. Launch: $DistPath\SCR-Twin\SCR-Twin.exe" -ForegroundColor Green
Write-Host "    First cold start takes ~30-45 s (unpacking SciPy); subsequent starts are faster." -ForegroundColor DarkGray
