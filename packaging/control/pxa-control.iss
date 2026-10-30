; PXA Control - Windows installer (Inno Setup 6). Per-user install, no admin prompt, Start-menu + optional desktop
; shortcut, uninstaller, "Launch PXA Control" on the last page. Built by .github/workflows/control-build.yml:
;   iscc /DAppVersion=3.1 /DSourceExe=..\..\out\win\pxa-control.exe packaging\control\pxa-control.iss
; Signing: the workflow signs pxa-control.exe before this runs and PXA-Control-Setup.exe after (signtool, sha256, RFC3161
; timestamp) when the certificate secret exists. An unsigned installer triggers SmartScreen: ship signed builds only.
#ifndef AppVersion
  #define AppVersion "3.1"
#endif
#ifndef SourceExe
  #define SourceExe "..\..\out\win\pxa-control.exe"
#endif
[Setup]
AppId={{6A1F3E52-7C8D-4E0B-9D6E-50A5C0B1F7A1}
AppName=PXA Control
AppVersion={#AppVersion}
AppPublisher=PXA Network
AppPublisherURL=https://discord.pxanetwork.com
DefaultDirName={localappdata}\Programs\PXA Control
DefaultGroupName=PXA Control
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=Output
OutputBaseFilename=PXA-Control-Setup
SetupIconFile=pxa-control.ico
UninstallDisplayIcon={app}\pxa-control.exe
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes

[Files]
Source: "{#SourceExe}"; DestDir: "{app}"; Flags: ignoreversion
; an engine build for Windows (llama-server.exe + CUDA DLLs) goes in {app}\engine\bin when one exists; PXA Control
; finds it there by itself (pxa_control.desktop.find_engine: "beside this app")

[Icons]
Name: "{group}\PXA Control"; Filename: "{app}\pxa-control.exe"
Name: "{userdesktop}\PXA Control"; Filename: "{app}\pxa-control.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\pxa-control.exe"; Description: "Launch PXA Control"; Flags: nowait postinstall skipifsilent
