# Compile the SCR-Twin Windows installer (Inno Setup) from the PyInstaller bundle.
#
# Prerequisite: the one-dir bundle exists (run build_desktop.ps1 first) and Inno
# Setup 6 is installed (winget install JRSoftware.InnoSetup).
#
# Usage:
#   ./packaging/build_installer.ps1
#   ./packaging/build_installer.ps1 -SourceDir D:\builds\dist\SCR-Twin -OutputDir D:\builds\dist

param(
  [string]$SourceDir = "",
  [string]$OutputDir = "",
  [string]$AppVersion = "0.1.0"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $SourceDir) { $SourceDir = Join-Path $root "dist\SCR-Twin" }
if (-not $OutputDir) { $OutputDir = Join-Path $root "dist" }

if (-not (Test-Path (Join-Path $SourceDir "SCR-Twin.exe"))) {
  throw "Bundle not found at $SourceDir. Run ./packaging/build_desktop.ps1 first (or pass -SourceDir)."
}

$iscc = Get-ChildItem `
  "$env:LOCALAPPDATA\Programs\Inno Setup*", "C:\Program Files (x86)\Inno Setup*", "C:\Program Files\Inno Setup*" `
  -Filter ISCC.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $iscc) {
  throw "ISCC.exe not found. Install Inno Setup: winget install --id JRSoftware.InnoSetup"
}

$iss = Join-Path $PSScriptRoot "installer.iss"
Write-Host "==> Compiling installer from $SourceDir" -ForegroundColor Cyan
& $iscc.FullName "/DSourceDir=$SourceDir" "/DOutputDir=$OutputDir" "/DAppVersion=$AppVersion" $iss
if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit $LASTEXITCODE" }

Write-Host ""
Write-Host "==> Installer: $OutputDir\SCR-Twin-Setup-$AppVersion.exe" -ForegroundColor Green
