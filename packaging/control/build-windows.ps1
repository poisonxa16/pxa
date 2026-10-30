# Build the compiled PXA Control for Windows x64 (BUILD-CONTROL.md, "Windows x64"). Run from the repo root on a Windows
# host with Python 3.12 x64:
#   python -m pip install pyinstaller==6.22.3 nuitka==4.2.2 zstandard==0.25.0 pillow==12.3.0 pystray==0.19.5
#   powershell -NoProfile -ExecutionPolicy Bypass -File packaging\control\build-windows.ps1 pyinstaller|nuitka [OUT] [TAG]
# TAG (or $env:TAG): the release tag, e.g. v4.0.0-beta.3: VERSION and the exe's file version name it, and the store checks
# a plugin's min_host against it. Nuitka needs MSVC (VS 2022 Build Tools, --msvc=latest). Output:
# <OUT>\<tool>\pxa-control.exe + build-<tool>.json.
param([string]$Tool = "pyinstaller", [string]$Out = "out\win", [string]$Tag = $env:TAG)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
if (-not [IO.Path]::IsPathRooted($Out)) { $Out = Join-Path $Root $Out }
$Out = [IO.Path]::GetFullPath($Out)
$W = Join-Path $Out "stage"
New-Item -ItemType Directory -Force $Out | Out-Null
# the stage also carries the plugin kit and python.org's embeddable Python (pinned in stage.py, downloaded once into
# <OUT>\cache): a compiled Control runs plugins and the speech service on them
$StageArgs = @($W, "--python-embed", (Join-Path $Out "cache"))
if ($Tag) { $StageArgs += @("--tag", $Tag) } else { Write-Warning "no TAG: VERSION is the repo's own (plugins with a newer min_host will not install)" }
python "$Root\packaging\control\stage.py" @StageArgs
if ($LASTEXITCODE -ne 0) { throw "stage.py rc=$LASTEXITCODE" }
# first token after the 'tag:' label (the label is not part of the version)
$Ver = ((((Get-Content "$W\VERSION" -TotalCount 1) -replace '^tag:\s*', '') -split '\s+')[0]) -replace '[^A-Za-z0-9._-]', ''
# Windows wants four numbers: v4.0.0-beta.3 -> 4.0.0.0
$FileVer = "0.0.0.0"
if ($Ver -match '^v?(\d+)\.(\d+)(?:\.(\d+))?') {
  $Patch = "0"
  if ($Matches[3]) { $Patch = $Matches[3] }
  $FileVer = "{0}.{1}.{2}.0" -f $Matches[1], $Matches[2], $Patch
}
$Ico = "$Root\packaging\control\pxa-control.ico"
Push-Location $W
$t0 = Get-Date
switch ($Tool) {
  "pyinstaller" {
    $D = "$Out\pyinstaller"
    python -m PyInstaller --onefile --noconfirm --clean --windowed --name pxa-control --icon $Ico --distpath $D `
      --workpath "$Out\pyi-build" --specpath "$Out\pyi-build" --paths $W --hidden-import pxa_launch `
      --collect-submodules pxa_control --collect-submodules pxa_chat --collect-submodules pxa_ctl `
      --collect-data pxa_control --exclude-module pystray --exclude-module Xlib `
      --add-data "$W\third_party.zip;." --add-data "$W\pxa_control_ui;pxa_control_ui" --add-data "$W\pxa_chat\ui;pxa_chat\ui" `
      --add-data "$W\pxa_thinking_profiles.json;." --add-data "$W\pxa_encode_cells.json;." --add-data "$W\pxa-bench.py;." `
      --add-data "$W\VERSION;." --add-data "$W\plugin-sdk.zip;." --add-data "$W\python-embed.zip;." pxa_control_entry.py
  }
  "nuitka" {
    $D = "$Out\nuitka"
    python -m nuitka --standalone --onefile --assume-yes-for-downloads --msvc=latest --lto=no `
      --windows-console-mode=disable --windows-icon-from-ico=$Ico `
      --company-name="PXA Network" --product-name="PXA Control" --file-version=$FileVer --product-version=$FileVer `
      --file-description="PXA Control" --output-filename=pxa-control.exe --output-dir=$D `
      --include-package=pxa_control --include-package=pxa_chat --include-package=pxa_ctl --include-module=pxa_launch `
      --include-package-data=pxa_control `
      --include-package=PIL --nofollow-import-to=pystray --nofollow-import-to=six --nofollow-import-to=Xlib `
      --nofollow-import-to=*.tests `
      --include-data-dir=pxa_control_ui=pxa_control_ui --include-data-dir=pxa_chat/ui=pxa_chat/ui `
      --include-data-files=pxa_thinking_profiles.json=pxa_thinking_profiles.json `
      --include-data-files=pxa_encode_cells.json=pxa_encode_cells.json --include-data-files=pxa-bench.py=pxa-bench.py `
      --include-data-files=VERSION=VERSION --include-data-files=third_party.zip=third_party.zip `
      --include-data-files=plugin-sdk.zip=plugin-sdk.zip --include-data-files=python-embed.zip=python-embed.zip `
      --onefile-tempdir-spec="{CACHE_DIR}/PXA/control-$Ver" pxa_control_entry.py
  }
  default { Pop-Location; throw "usage: build-windows.ps1 pyinstaller|nuitka [OUT]" }
}
$rc = $LASTEXITCODE
Pop-Location
if ($rc -ne 0) { throw "$Tool rc=$rc" }
$s = [int]((Get-Date) - $t0).TotalSeconds
$b = (Get-Item "$D\pxa-control.exe").Length
"{`"tool`": `"$Tool`", `"build_s`": $s, `"bytes`": $b, `"version`": `"$Ver`"}" | Tee-Object -FilePath "$D\build-$Tool.json"
