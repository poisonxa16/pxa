# PXA Control: the compiled app (4.0)

PXA Control is the browser interface to your cards and servers. In 4.0 it comes inside the Linux
tarball (`./pxa`), as a compiled Windows exe with an engine zip, and as a Mac client; the build recipe
is in `BUILD-CONTROL.md`. Every page is described in [PXA Control 4.0](user/CONTROL-GUIDE-4.0.md);
the install steps for each platform are in [Install PXA](install.md). This page is what a user needs
to know about the compiled builds.

## Install and start

| platform | download | then |
|---|---|---|
| Linux x86_64 (glibc 2.35+, NVIDIA driver) | the tarball | unpack it, `cd` into the folder, run `./pxa` |
| Linux x86_64, if you were handed one | `pxa-<version>-linux-x86_64.run` | double-click it, or `chmod +x` and run it |
| Windows, macOS, Android | see **Platforms** below | |

The `.run` checks glibc, verifies its own sha256, unpacks **once** into `~/.local/share/pxa/<version>`,
adds a "PXA Control" entry to your applications menu, and opens your browser already signed in.
Later starts skip the unpack (about half a second to a working page). Python is not needed.

**The engine sits next to Control.** The download carries the release engine (`bin/`, `lib/`) and
Control finds it by itself (`engine/bin/llama-server` beside the program). Set `PXA_ENGINE_DIR` only
to use a different engine.

## Flags

None are needed for the common case. All are optional.

| flag | what it does |
|---|---|
| `--port N` | listen on port N (default: the first free port from 7777) |
| `--lan` | also listen on your network, behind the access token (see below) |
| `--no-browser` | do not open a browser |
| `--no-tray` | no tray or menu-bar icon |
| `--models-dir DIR` | a folder with `.gguf` models; may be given more than once; remembered |
| `--version` | print the version and exit |

With no models folder configured, the first run uses `~/PXA/models` (created). Settings are kept in
`~/.config/pxa/control.json`. One Control runs per user: starting it again reopens the running one.

## The access token

Control binds to `127.0.0.1` unless you pass `--lan`, and **a token guards the API in both
modes**.

- **On this machine** (default) the token is fresh for every session. It rides in the address
  Control opens for you (`?token=` once) and then becomes an `HttpOnly`, `SameSite=Strict` cookie,
  named per port. A copy lives in the session's run record (`control-<pid>.json`, mode 0600, in a
  0700 directory of your own config area), which is how running the app a second time opens the
  same session signed in. Only `GET /api/health` and `GET /metrics` answer without the token on loopback.
- **With `--lan`** the token is the persistent one in `~/.config/pxa/token` (mode 0600, created
  on first use). `PXA_CONTROL_TOKEN` overrides it (16 or more of `A-Z a-z 0-9 _ -`). Nothing is
  open without it on `--lan`. Open `http://<host>:<port>/?token=<token>` once from the other device.
- **Ways to present it:** the cookie (a browser), the `X-PXA-Token` header, `Authorization:
  Bearer <token>` (for example a Prometheus scrape job), or `?token=` on a **GET** only. A POST or
  DELETE never takes the token from the query string.

Do not paste the token into chat or a bug report; Control scrubs it from its own logs and reports.

## The Store

The Store tab talks to the licence server. The default is `https://lic.pxanetwork.com`.

| setting | effect |
|---|---|
| nothing set | the live store, `https://lic.pxanetwork.com` |
| `PXA_STORE_URL=<url>` | use that store instead (the environment wins over the config file) |
| `store_url` in `control.json` | same, persisted |
| `PXA_STORE_URL=""` (empty) | **offline**: the built-in fixture catalogue, nothing leaves the machine |

Store calls are signed by a key that lives only on this machine. The compiled build finds the
system CA bundle itself on distributions where the bundled OpenSSL's default path is missing.

## Platforms

PXA Control is a Python server with a web page; the engine is a separate program. What each platform gets:

| platform | download | Control | engine on that machine | notes |
|---|---|---|---|---|
| Linux x86_64 | the tarball (`./pxa`); a `pxa-control-<ver>-linux-x86_64.run` is built too but is not one of the 4.0 downloads | yes | yes, CUDA 12, PXQN included | the reference build |
| Windows x64 | `pxa-control-win64.exe` + `pxa-engine-win64-cuda12.zip` (+ `ggml-pxqn.dll`) | yes (PyInstaller onefile) | yes, CUDA 12 (sm 60/61/70, newer cards by PTX JIT) | see below |
| macOS, Intel and Apple silicon | one universal zip (unsigned) | yes | **no** (no CUDA on macOS) | drives a rig on the network; the Apple silicon half is built and checked but not yet run on a real Apple silicon Mac |
| Android 7+ | in testing, not a 4.0 download | client only | **no** | will open a rig's Control on the LAN |
| any browser | the PWA ("Install app") | client | - | needs `https` or `localhost` |

**Windows.** Put `pxa-control-win64.exe` in a folder, unzip the engine next to it as `engine\` (so
`engine\bin\llama-server.exe` exists); Control finds it by itself. PXQN/PXQN4 models need the licensed
`ggml-pxqn.dll` copied into `engine\bin\` next to `ggml.dll`; without it PXQN files are refused with one message and
PXQ / k-quant files run. Not in the Windows PXQN library (Linux has them): the fast CPU PXQN matmul, locked
(licence-encrypted) models, the expert-cache CPU pool. Nothing is code-signed yet: SmartScreen asks once
("More info" -> "Run anyway"). The CUDA 12.8 runtime and the MSVC/OpenMP runtime are inside the zip; an NVIDIA
driver r525+ is needed.

**macOS.** Control only. The Mac binary runs the Control page (models folder, Hugging Face downloads, Store,
settings) but cannot start a local server: no Metal engine ships.
To drive a GPU rig, run `./pxa --lan` on the rig and open it from the Mac's browser. Unsigned and not notarized: the first start needs
`xattr -dr com.apple.quarantine .` in the unzipped folder, or **Open Anyway** in System Settings, Privacy & Security (the steps are
in [Install PXA](install.md#macos)). One download runs on Intel Macs and on Apple silicon Macs.

**Android.** In testing, not part of the 4.0 downloads. The app is a WebView wrapper (not a Trusted Web Activity: a TWA needs an `https` origin with Digital Asset Links, and
Control on a LAN is plain `http`). It opens on a rig picker (remembered rigs, address + token), then shows that rig's
Control page inside the app. No engine and no inference on the phone; no microphone for voice input; signed with a
throwaway debug key (install from unknown sources).

**PWA.** Control serves `/manifest.webmanifest`, `/sw.js` (network first; the cache stands in only when the network
fails; never `/api/`), `/pwa.js` and `/hosts.html` (the rig picker, also the offline page). Browsers register a service
worker only in a secure context, so "Install app" works for Control on `localhost` (desktop Chrome/Edge) or behind
`https`; on a plain-`http` LAN address use the Android app or a bookmark.

## What is not covered here

Code signing (Windows Authenticode, Apple notarization, a Play Store key) is not done for any of the packages above.
