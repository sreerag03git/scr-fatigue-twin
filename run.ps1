# One-command launcher (Windows): build the console and serve it offline at :8000.
#
#   ./run.ps1            build frontend + start the backend (production, one port)
#   ./run.ps1 -Dev       start backend (:8000) + Vite dev server (:5173) with HMR
#
# The backend serves the built frontend from app/dist, so the production mode is a
# single process with no network dependency.

param([switch]$Dev)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$venvPy = Join-Path $root ".venv\Scripts\python.exe"

# Node is often installed outside the interactive PATH — add it explicitly.
$nodeDir = "C:\Program Files\nodejs"
if (Test-Path $nodeDir) { $env:Path = "$nodeDir;" + $env:Path }
$npm = if (Test-Path "$nodeDir\npm.cmd") { "$nodeDir\npm.cmd" } else { "npm" }

Write-Host "==> Python environment" -ForegroundColor Cyan
if (-not (Test-Path $venvPy)) {
  Write-Host "    creating .venv" -ForegroundColor DarkGray
  python -m venv (Join-Path $root ".venv")
}
& $venvPy -m pip install -e (Join-Path $root "core") --quiet
& $venvPy -m pip install -r (Join-Path $root "server\requirements.txt") --quiet

Write-Host "==> Frontend" -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $root "app\node_modules"))) {
  & $npm install --prefix (Join-Path $root "app")
}

if ($Dev) {
  Write-Host "==> Dev mode: backend :8000 + Vite :5173" -ForegroundColor Cyan
  Start-Process -FilePath $venvPy -ArgumentList "-m", "uvicorn", "server.main:app", "--port", "8000", "--reload" -WorkingDirectory $root
  & $npm run dev --prefix (Join-Path $root "app")
} else {
  Write-Host "==> Building frontend" -ForegroundColor Cyan
  & $npm run build --prefix (Join-Path $root "app")
  Write-Host "==> Console at http://127.0.0.1:8000  (Ctrl+C to stop)" -ForegroundColor Green
  Push-Location $root
  & $venvPy -m uvicorn server.main:app --host 127.0.0.1 --port 8000
  Pop-Location
}
