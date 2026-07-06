; Inno Setup script for the SCR-Twin desktop console.
;
; Wraps the PyInstaller one-dir bundle into a per-user Setup.exe with Start-menu
; and optional desktop shortcuts, plus an uninstaller. Installs per-user (no admin
; required), matching the app's offline, self-contained design.
;
; Build:
;   ISCC.exe packaging\installer.iss
; Override bundle/output/version with /D defines, e.g.:
;   ISCC.exe /DSourceDir="D:\builds\dist\SCR-Twin" /DOutputDir="D:\builds\dist" /DAppVersion=0.1.0 packaging\installer.iss

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\SCR-Twin"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif
#define AppName "SCR-Twin"
#define AppExe "SCR-Twin.exe"

[Setup]
AppId={{9E9C2B1A-7C3E-4E2A-9E0D-5C2A1B3D4E5F}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=SCR-Twin project
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=SCR-Twin-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} - TDP Fatigue Integrity Console

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\SCR-Twin Console"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall SCR-Twin"; Filename: "{uninstallexe}"
Name: "{autodesktop}\SCR-Twin Console"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch SCR-Twin now"; Flags: nowait postinstall skipifsilent
