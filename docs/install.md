# Install PXA

PXA 4.0 ships as a Linux tarball, a one-line installer, a container image, a Windows package and a
Mac client. The tarball, the installer and the container carry the same engine binaries; models are
separate and are never inside them. Check every download against the sha256 on the release page
before you use it (`sha256sum -c <file>.sha256` on Linux, `shasum -a 256 <file>` on a Mac,
`Get-FileHash <file>` in Windows PowerShell).

| Platform | What you download | Steps |
|---|---|---|
| Linux x86_64 | the tarball | [Tarball](#tarball-recommended) |
| Linux, any distribution with Docker and the NVIDIA runtime | `ghcr.io/poisonxa16/pxa:v4.0` | [Container](#container) |
| Windows x64 | PXA Control and an engine zip (unsigned) | [Windows](#windows) |
| macOS, Intel and Apple silicon | PXA Control (unsigned, a client only) | [macOS](#macos) |
| Android | in testing, not a 4.0 download | [Android](#android) |

## Tarball (recommended)

Needs Linux x86-64, glibc 2.35 or newer (Ubuntu 22.04 and newer) and an NVIDIA driver 570 or newer.
The package carries its own CUDA 12.8 runtime, so the CUDA toolkit is not needed. Download
`pxa-v4.0-linux-x86_64-cuda12.8-sm60_61_70.tar.gz` from the
[release page](https://github.com/poisonxa16/pxa/releases/latest). Then:

```bash
tar xzf pxa-v4.0-linux-x86_64-cuda12.8-sm60_61_70.tar.gz
cd pxa-v4.0
./pxa --doctor                 # checks your cards, driver and CPU; starts nothing
./pxa                          # opens PXA Control in your browser
```

`START-HERE.md` in the tarball goes through the first fifteen minutes, with what a good start looks
like. One line starts a server without the launcher:

```bash
./run-server.sh -m /path/to/model.gguf          # serves on http://127.0.0.1:8080
curl -s http://127.0.0.1:8080/health            # {"status":"ok"}
```

### On a cluster (Environment Modules / Lmod)

The package carries a modulefile. Unpack it anywhere (a shared software path), then:

```bash
module use /path/to/pxa-<version>/modulefiles
module load pxa/<version>      # e.g. pxa/4.0
llama-server --version         # bin/ is on PATH; pxa, pxa-launch and run-server.sh too
```

It sets `PATH`, `LD_LIBRARY_PATH` and `PXA_ENGINE_DIR`, and picks the compatibility library on a CPU
without AVX2 (set `PXA_CPU_LIB=fast` or `compat` before loading to force the choice; the choice is made
on the node that loads the module). It leaves `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` to your
scheduler. To keep your modulefiles elsewhere, symlink `modulefiles/pxa/<version>` into that tree, or
set `PXA_ROOT` to the package folder before loading.

## One command

The installer checks your CPU, glibc and cards, downloads the build that matches, verifies its
checksum and unpacks it:

```bash
curl -fsSL https://raw.githubusercontent.com/poisonxa16/pxa/main/install.sh | bash
```

Add `-s -- --docker` after `bash` to pull the container image instead.

## Container

`ghcr.io/poisonxa16/pxa:v4.0` carries the same binaries. Models are not in the image; mount a folder
at `/models`. The tag `latest` moves only with the public release; `v4.0` stays fixed. The image runs
as a normal user and answers a health check on port 8080.

```bash
docker run -d --name pxa --gpus '"device=0,1"' --shm-size=1g -p 8080:8080 \
    -v /path/to/models:/models:ro \
    -v pxa-cache:/work/.cache/pxa -e PXA_CACHE_DIR=/work/.cache/pxa \
    ghcr.io/poisonxa16/pxa:v4.0
```

The `pxa-cache` volume is where a session's learned expert counts are kept. Without that mount they
are deleted when the container is removed. More than one card needs `--shm-size=1g`.

If `--gpus` fails with `nvidia-container-cli: ldcache error` (some hosts, Unraid among them), use
`--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1` instead.

PXA Control in the container is opt-in: add `-e PXA_CONTROL=1 -p 7777:7777`. Container images and
Compose: `docker/COMPOSE.md`.

## Windows

Needs Windows 10, Windows 11, or Windows Server 2019 or newer, 64-bit, and an NVIDIA driver for CUDA
12. The package does not include a driver. Two files, plus one optional library:

- `pxa-control-win64.exe`: PXA Control. Put it in a folder of its own.
- `pxa-engine-win64-cuda12.zip`: the engine with the CUDA 12 runtime (sm_60, sm_61 and sm_70 cards).
  Unzip it next to `pxa-control-win64.exe` and rename the unzipped folder to `engine`, so
  `engine\bin\llama-server.exe` exists. Control finds that folder by itself.
- `ggml-pxqn.dll`: the PXQN library, published beside the zip, not inside it. Copy it into
  `engine\bin` next to `ggml.dll` to run a PXQN model. Without it, PXQ and other GGUF models still
  run, and a PXQN model is refused with one message. This Windows build of the library has no fast
  CPU math, no locked models and no expert-cache CPU pool; the Linux build has them.

**Windows SmartScreen.** These files are not code-signed. The first time you open one, Windows may
show "Windows protected your PC". Choose **More info**, then **Run anyway**. It asks once per file.
Compare each file with `SHA256SUMS` before you run it.

To run the engine on its own: `engine\bin\llama-server.exe -m model.gguf -ngl 99`, then open
http://127.0.0.1:8080. A check on more Windows machines than the one we tried is still open.

## macOS

The Mac download is PXA Control only. There is no PXA engine on macOS (no NVIDIA GPU): Control manages
your models folder, Hugging Face downloads, the PXA Store, settings and chat, and it opens a GPU
machine's Control over your network. On the GPU machine run `./pxa --lan` (Control then also listens on
your network, behind its access token), then open that machine's address on the Mac.

It runs natively on Intel Macs and on Apple silicon Macs (M1 and later); Rosetta is not needed. Tried
so far: Intel, macOS 13 (Ventura). The Apple silicon half has been built and checked but not yet run
on a real Apple silicon Mac. Written for macOS 11 (Big Sur) or newer.

**First start (Gatekeeper).** macOS asks because the download is not signed with an Apple Developer ID
and is not notarized. That is expected.

1. Unzip it and open the folder.
2. Double-click **Start PXA Control.command**. macOS says it cannot verify the developer.
3. Allow it once, either way:
   - In Terminal, in the unzipped folder, run `xattr -dr com.apple.quarantine .` and double-click
     the file again; or
   - open System Settings, **Privacy & Security**, scroll to Security, press **Open Anyway** next to
     the message about the file, and confirm with your password. (On macOS 13 and 14,
     Control-click, **Open**, **Open** also works.)
4. A Terminal window opens, PXA Control starts and your browser opens signed in. Close the window to
   stop it.

The first launch unpacks the app (about 30 seconds on a slow Mac); later launches take about a second.

## Android

An Android client app (a WebView wrapper that connects to your Control host) is in testing and is not
part of the 4.0 downloads. It is a remote client only: there is no on-device engine. Until it ships,
open your Control's address in the phone's browser and bookmark it.

## From source

`BUILD-FROM-SOURCE.md`. A source build runs classic PXQ files and standard quants. For PXQN files,
add the compiled PXQN library from the release page (section 4b of that guide says how).

## Updating

PXA Control can install a newer Linux release, and from a terminal the same tool is `pxa-update`
(Windows and Mac: download the new package from the release page):

```bash
pxa-update check
pxa-update apply
pxa-update rollback
```

Stop a server from this install before `apply` or `rollback`. Neither touches your models, your
settings, or a model's `.expert-counts.csv`.

Valued Supporters can opt in to the beta channel from the update window and go back to stable at any
time; see [Beta channel](user/BETA-4.0.md). `pxa-update` refuses a download whose name is not exactly
the expected tarball, stages it in the install folder and rolls back on a failure.
