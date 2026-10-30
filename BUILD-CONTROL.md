# Building PXA Control as one download per platform

PXA Control (`tools/pxa_control/`, the package; `tools/pxa_control.py` is its compatibility shim) ships as a compiled
program: the user downloads one file, double-clicks it, and the browser opens signed in. No Python, no pip, no CUDA
toolkit, no terminal, no flag for the common case. Measured on the build box 2026-10-14.

**What 4.0 ships.** The Linux tarball (Control plus the engine, `./pxa`), a Windows `pxa-control-win64.exe` with an
engine zip, and a macOS Control client as one zip for Intel and Apple silicon. The Windows and Mac downloads are not
code-signed and the Mac one is not notarized; the install steps are in [`docs/install.md`](docs/install.md). The
`.run`, the Windows installer and the Mac `.dmg` described below are build recipes; they are not 4.0 downloads. This page
is for people who build Control themselves.

## What a user does

| | the tarball / install.sh | the compiled `.run` build |
|---|---|---|
| download | `pxa-v4.0-linux-x86_64-cuda12.8-...tar.gz`, or `curl ... install.sh \| bash` | `pxa-<version>-linux-x86_64.run`, one file (size below) |
| needs on the machine | bash, curl, tar, **python3**, glibc 2.35, NVIDIA driver | glibc 2.35 and the NVIDIA driver (nothing else) |
| steps | unpack (or pipe a script into bash), `cd`, `./pxa` in a terminal | double-click it (file manager: "Run"), or `./pxa-....run` |
| starts | `exec python3 tools/pxa-launch.py` from the source files | a compiled binary; the engine is found by itself |
| browser | opened on 127.0.0.1 (no token on loopback before 2026-10-14) | opened on the signed-in address (`?token=` once, then a cookie) |
| next time | `cd` + `./pxa` again | the applications menu entry "PXA Control" (made on first run), or the .run again (no unpack) |
| quitting | Ctrl-C in the terminal | tray / menu-bar icon: Quit (when the desktop has a tray), or close the terminal |
| models folder | `--models-dir` or the Models tab | remembered; first run with none uses `~/PXA/models` |
| what users get of Control | plain `.py` source (readable, editable) | machine code (Nuitka); the page's HTML/JS stay readable, as any web page's |
| Windows / macOS | see [`docs/install.md`](docs/install.md): `pxa-control-win64.exe` plus an engine zip (unsigned), and a Mac client zip (unsigned, no engine) | recipes below for a signed installer / `PXA Control.app` in a DMG when you have the signing identities |

## The three tools, measured (Linux x86_64, Ubuntu 22.04 build image, 18 cores, same source)

| tool | what it produces | build | size | start to `/api/health` 200 (cold / warm) | what it actually protects |
|---|---|---|---|---|---|
| **Nuitka 4.2.2** `--standalone --onefile` (chosen) | one ELF; Python modules translated to C and compiled; CPython runtime + stdlib inside; unpacks once into `~/.cache/PXA/control-<ver>` and reuses it | 595 s | 19.8 MB | 0.52 s / 0.49 s | no `.py`/`.pyc` of Control in the payload: getting the logic back means reverse-engineering compiled C (string constants, names and the call graph are still visible). Raises the bar a lot; does not make copying impossible (the binary itself can be copied and run). |
| PyInstaller 6.22.3 `--onefile` | one ELF; bootloader + **bytecode** archive; unpacks to `/tmp/_MEI*` on every start | 28 s | 16.6 MB | 0.80 s / 0.79 s (unpacks every time) | almost nothing: `pyinstxtractor` + a decompiler give back near-original source in minutes. A bundler, not protection. |
| Cython 3.3.0 (`cythonize -3`) + a Python launcher | one `.so` per module (core 17 MB) beside an interpreter you still ship (PyInstaller or embedded CPython) | 183 s (compile only) | 24.4 MB of `.so` + the runtime | 0.33 s / 0.21 s (needs Python 3.10 present) | like Nuitka for the modules it compiles, but the shipping (interpreter, the rest of the bytecode, onefile) is still PyInstaller's; two tools, two failure modes. |
| source (today) | `python3 tools/pxa-launch.py` | - | 2.5 MB of .py | 0.64 s / 0.19 s (needs python3) | none |

Why Nuitka: one tool gives the compiled modules AND the single file on all three platforms (`--onefile`, `.exe` with icon
and version info, `--macos-create-app-bundle`), a cached unpack (start-up does not pay the payload twice), and the
fastest cold start measured. Its cost is build time (10 min; ccache makes rebuilds faster) and a C compiler on the
build host (the box has none outside containers: the build runs in `packaging/control/Dockerfile.build-linux`).

Honest limits: compiled code raises the cost of reading, patching and re-branding PXA Control; it does not stop
anyone from running or redistributing a copy of the binary, and the web page (HTML/JS/CSS) is readable like any web
page. Earlier releases (v3.0/v3.1 tarballs and images) already shipped `tools/pxa_control.py` as source; compiling
protects the versions after this one. What is worth enforcing lives server-side (licd grants, signed releases:
PLUGINS-STORE-DESIGN.md), not in the client.

## Linux x86_64 (built and measured here)

```bash
# 1. the build image (pinned tools; Ubuntu 22.04 = the engine release's glibc floor, 2.35)
docker build -t pxa-control-build:u2204 -f packaging/control/Dockerfile.build-linux packaging/control
# 2. the compiled Control (one file, out/nuitka/pxa-control)
docker run --rm --cpuset-cpus=18-35 -v "$PWD":/src:ro -v "$PWD/out":/out -v "$PWD/out/cache":/root/.cache \
    pxa-control-build:u2204 bash /src/packaging/control/build-linux.sh nuitka
# 3. the ONE download: compiled Control + the release's engine (bin/ lib/ lib-compat/) + icon, as a self-checking .run
bash packaging/control/make-linux-run.sh out/nuitka/pxa-control <unpacked release dir> packaging/control/pxa-control.png \
    out/pxa-v3.1-linux-x86_64.run
# 4. smoke: start, health, signed-in page (cold + warm)
python3 packaging/control/measure-start.py out/nuitka/pxa-control 3
```

`build-linux.sh pyinstaller|cython` builds the two alternatives for comparison. `stage.py` stages tools/ the same way
on every OS (the launcher becomes the importable `pxa_launch`, the lever catalog goes inside the page folder, LGPL
libraries go into `third_party.zip`, see Licences).

Pass the release tag (`TAG=v4.0.0 bash build-linux.sh ...`, `build-windows.ps1 nuitka out\win v4.0.0`): `stage.py --tag`
writes it into VERSION, and the store refuses a plugin whose `min_host` is newer than the Control's VERSION. Without a
tag the stage keeps the repo's VERSION file.

Plugins and the speech service run as separate Python processes, which a compiled Control cannot start from itself (its
`sys.executable` is the Control binary). So the stage also writes:

- `plugin-sdk.zip`: the plain-source part of PXA Control that plugins and speech import
  (`pxa_control/plugins/sdk.py`; the same bytes on every build). Control unpacks it once into its config folder
  (`plugins/sdk/<sha16>`) and checks every file against the kit's manifest.
- `python-embed.zip` (Windows builds, `--python-embed <cache dir>`): python.org's embeddable Python 3.13 for x64, pinned
  by version and sha256 in `stage.py` and downloaded once into the cache folder; a download that does not match the pin
  stops the stage. Beside `python.exe` it carries Microsoft's C++ runtime (`msvcp140.dll`, `msvcp140_1.dll`, from the
  msvc-runtime wheel on PyPI, also pinned by sha256 in `stage.py`): the speech engines import them (onnxruntime,
  espeak-ng), python.org's package has only `vcruntime140`, and a Windows without the Visual C++ Redistributable has
  neither. Control unpacks it once into `python/<sha16>` in its config folder (`pxa_control/pyfind.py`).
  Linux and macOS builds use a Python 3.10 - 3.14 found on the computer; without one, a plugin or speech start says what
  to install.

Every build script bundles both files next to VERSION (`--include-data-files` / `--add-data`).

The `.run` (make-linux-run.sh): a 2 KB sh header + one tar.gz. Run it and it checks glibc, verifies the payload's
sha256, unpacks ONCE into `~/.local/share/pxa/<ver>` (install.sh's layout, with the `current` symlink), writes
`~/.local/share/applications/pxa-control.desktop` (menu entry with icon) and starts PXA Control; the next run skips
straight to the binary. PXA Control finds the engine beside itself (`engine/bin/llama-server`), so no `PXA_ENGINE_DIR`,
no flag. Measured: see "One download, measured" below.

## Windows x64 (recipe; .github/workflows/control-build.yml, job windows-x64)

On the self-hosted Windows Server 2022 runner the vigil gate already uses (`[self-hosted, Windows, vigil]`):

1. Python 3.12 x64, `pip install nuitka==4.2.2 zstandard==0.25.0 pillow==12.3.0 pystray==0.19.5`; MSVC Build Tools
   (`--msvc=latest`; Nuitka can also download MinGW64 with `--assume-yes-for-downloads`).
2. `python packaging/control/stage.py out/stage --python-embed out/cache --tag <tag>` (or `build-windows.ps1`, which
   does the stage, the build and the file version from the tag)
3. Nuitka `--standalone --onefile --windows-console-mode=disable --windows-icon-from-ico=packaging/control/pxa-control.ico
   --company-name="PXA Network" --product-name="PXA Control" --file-version/--product-version --onefile-cache-mode=cached`
   (the full command is in the workflow) -> `pxa-control.exe`, no console window.
4. Sign the exe: `signtool sign /fd sha256 /tr http://timestamp.digicert.com /td sha256 /f <pfx> ...` (secret
   `WINDOWS_SIGN_PFX_B64` + `_PASS`; the step is skipped without it and the build is then labelled unsigned).
5. Installer: `iscc packaging/control/pxa-control.iss` -> `PXA-Control-Setup.exe` (per-user, no admin prompt, Start-menu
   and optional desktop shortcut, uninstaller, "Launch PXA Control" at the end); sign it the same way.

User steps of an installer build: download `PXA-Control-Setup.exe`, double-click, Next, Finish (PXA Control opens in the browser). Later: Start
menu "PXA Control"; quit from the tray icon. Engine: the installer has the slot (`{app}\engine\bin\llama-server.exe` + CUDA DLLs) and PXA Control finds an engine there by itself;
the 4.0 engine zip (`pxa-engine-win64-cuda12.zip`) unzipped as `engine\` next to the exe fills the same slot. The 4.0 download is the unsigned exe, not this installer.

## macOS arm64 (recipe; workflow job macos-arm64)

Hosted `macos-14` (Apple silicon), tags and dispatch only (10x minutes):

1. Python 3.12, `pip install nuitka==4.2.2 zstandard==0.25.0 pillow==12.3.0 pystray==0.19.5 pyobjc-framework-Cocoa`.
2. `python packaging/control/stage.py out/stage`; the `.icns` is made from `packaging/control/pxa-control.png` with
   `sips` + `iconutil`.
3. Nuitka `--standalone --macos-create-app-bundle --macos-app-name="PXA Control" --macos-signed-app-name=net.pxanetwork.control
   --macos-app-mode=ui-element` (menu-bar app, no Dock icon) -> `PXA Control.app`.
4. `packaging/control/macos-sign.sh`: Developer ID codesign with hardened runtime, `notarytool submit --wait`, `stapler
   staple` (secrets `MAC_DEVID_P12_B64`, `MAC_NOTARY_KEY_*`; skipped without them).
5. `hdiutil create ... -format UDZO` -> `PXA-Control-macos-arm64.dmg`.

User steps of a signed build: open the DMG, drag PXA Control to Applications, double-click it (first time: the notarized app opens without
the "unidentified developer" prompt). The 4.0 Mac download is not signed or notarized: see [`docs/install.md`](docs/install.md#macos) for the first start. Menu-bar icon: Open / Quit. There is no CUDA on macOS: the Mac build is PXA
Control for a rig on the network (open the rig's Control with `--lan` / an SSH tunnel) until a Metal engine build ships.

## Licences of what the binary carries

- Nuitka's output carries no licence obligation from Nuitka (Apache-2.0 runtime parts). CPython: PSF licence (notice).
- Pillow: MIT-CMU (HPND), notice. PIL is compiled in.
- Microsoft C++ runtime (`msvcp140.dll`, `msvcp140_1.dll` in `python-embed.zip`): unmodified, under Visual Studio's
  distributable-code terms (the REDIST list's Visual C++ runtime files).
- **pystray (LGPL-3.0) and python-xlib (LGPL-2.1) are never compiled in**: `stage.py` puts them, as plain `.py`, in
  `third_party.zip` next to the modules, and `pxa_control.desktop.third_party_dir()` puts that zip on `sys.path`; a user
  can replace it (LGPL's relinking condition). Without them the app runs without a tray icon.
- A third-party notices file belongs in the release beside the binary (owed: generate it from the stage).

## One download, measured (box, 2026-10-14; Nuitka binary 18.8 MB + the v3 Ubuntu 22.04 release engine)

| | number |
|---|---|
| `make-linux-run.sh` (tar + gzip of 1.9 GB, one core) | 147 s |
| `pxa-v3.1-linux-x86_64.run` | **1.54 GB** (the v3 tarball it replaces: 1.48 GB) |
| first run: glibc check + sha256 of the payload + unpack 1.9 GB into `~/.local/share/pxa/v3` + menu entry, to `/api/health` 200 | 55.2 s (busy shared cores, pool disk; single-threaded gunzip is most of it) |
| every later start (the .run again, or the menu entry), to `/api/health` 200 / signed-in page | 0.47-0.55 s / 0.48-0.56 s |
| the compiled Control alone (no engine), cold / warm | 0.52 s / 0.49 s |
| engine found by itself | `engine/bin/llama-server` beside the binary; its bundled libs resolve (in the GPU-less build container it stops only at `libcuda.so.1`, the driver's library, as it must) |

What makes the payload big is the CUDA runtime the engine release already carries (libcublasLt 752 MB, libnccl 384 MB,
two libggml 285 MB each, libcublas 116 MB); Control itself is 1 % of it. A faster first run: a zstd payload (`zstd -T0`,
5-10x faster to unpack) when the target has zstd, else gzip as now; dropping `lib-compat/` (the no-AVX2 libggml,
-272 MB) from a variant for current CPUs.
