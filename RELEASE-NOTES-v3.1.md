<p align="center"><img src="docs/assets/pxa-network-banner.png" alt="PXA Network" width="760"></p>

<p align="center">
<a href="https://benchmarks.pxanetwork.com"><img alt="Live leaderboard" src="https://img.shields.io/badge/Leaderboard-benchmarks.pxanetwork.com-E69F00?style=for-the-badge"></a>
<a href="https://discord.gg/EqazvV9tf"><img alt="Discord" src="https://img.shields.io/badge/Discord-PXA%20Network-5865F2?logo=discord&logoColor=white&style=for-the-badge"></a>
<a href="https://ko-fi.com/shatteredrealms1"><img alt="Ko-fi" src="https://img.shields.io/badge/Ko--fi-Support%20PXA-FF5E5B?logo=ko-fi&logoColor=white&style=for-the-badge"></a>
</p>

# PXA v3.1

**Install in one line:** `curl -fsSL https://raw.githubusercontent.com/poisonxa16/pxa/main/install.sh | bash` · Docker: `docker pull ghcr.io/poisonxa16/pxa:v3.1`

v3.1 is the current release. It uses the same model files as v3, so there is nothing to re-quantize. Upgrade if you run a Flash-Next model on one card, use PXA Control, or want the updater. What changed since v3 is below, and the measured numbers are below that.

## The numbers

Everything here was measured on this build on one Tesla P100 16 GB, with greedy output and 512 tokens unless a row says otherwise, on an otherwise idle machine.

**Writing: Overdrive off and on** (Flash-Next PXQN2, three runs each, one window, both arms alternating in that window):

| | Prose | Code |
|---|---:|---:|
| Overdrive off | 26.91 tok/s | 32.28 tok/s |
| Overdrive on | **30.70 tok/s** | **37.96 tok/s** |

About +14% prose and +18% code. Overdrive is four switches that live in the closed library.

**Expert map: the curated map against the learned one** (Flash-Next PXQN 32 GB, three learning sessions, the learned number pooled over them):

| | Prose | Code |
|---|---:|---:|
| Map that ships with the model | 28.50 tok/s | 31.59 tok/s |
| Map the server learned | **32.59 tok/s** | **35.07 tok/s** |

**The expert map against no map at all** (same card, same window, quiet machine). With the map off (`PXA_XCACHE_BOOTSTRAP=0`) decode is 26.12 tok/s prose and 27.95 code; with the map it is 30.31 and 34.95. About +16% prose and +25% code. The no-map path still boots and serves — it took 76 seconds to load and listen. It is slower, not broken.

**Reading the prompt** (six runs each, a different prompt every run, both engines in the same window and each started fresh). Ours is the Flash-Next PXQN 32 GB file and theirs is their Q2_0 file.

| Prompt | PXA | Leading Competitor |
|---|---:|---:|
| 4,096 tokens | **321.4 tok/s** | 167.5 tok/s |
| 8,000 tokens | **358.3 tok/s** | 217.0 tok/s |

Long prompts are where the gap is widest. PXA is steady from run to run; the competitor's long-prompt figure depends on how warm its file cache is and moved between 85 and 264 tok/s across our sessions, so read that column as a range rather than a fixed number. Measured 9 October 2026.

Decode at the end of a session, prose 30.31 tok/s for PXA against 25.17 for the competitor; code 34.95 against 36.27. See Known issues.

## What is new

**PXA Control**
- GPU Profiles: per-card power and clocks. Anything that changes a card stays off until you turn it on.
- A chat assistant on the page. It picks a running server, can use tools with your approval, and keeps the session.
- Built-in web search that works with no setup, no account and no key. You can switch to SearXNG, Brave Search or Tavily in the chat's search settings; Brave and Tavily need your own key.
- Memory across chats: remember, forget, a Memory panel, and a screen for secrets. On by default.
- Thinking controls per model: on or off, an effort level, and a lock so the server setting wins over the client's.
- Model hot swap: register a second model on the same server and the request's model name picks it. Supported for Qwen-family models on Volta (V100) and newer. Sliding-window models, including Gemma 4, are refused. Cards below compute 7.0 stay off as well.
- Gemma drafter: the Launch page offers the Gemma 4 drafter file when it sits next to the model.
- Sleep when idle: a server can free its cards after a set number of quiet seconds, and the next request loads the model back.
- Host access for the chat assistant. Off by default. When you turn it on, the assistant can only use an allowlist you set, for that session only, and its actions are logged.
- Supporter library updates: check, apply and roll back the licensed library from PXA Control.
- Live graphs keep up to 90 days. A bug report names the engine build.
- Stop, then Start on the same port, works. A server Control attached to no longer shows as stopped.
- Encode checks a mixture-of-experts file before the download starts and refuses it if the encoder cannot handle it. A bad licence key now says what is wrong — expired, suspended, disabled or revoked each get their own message.

**Expert map**
- Ships with the model as a curated file (`<model>.expert-counts.csv`). For new mixture-of-experts models the encoder writes one.
- The server learns a second map from your use, saves it after a session, keeps it across restarts, and uses it only when it measures faster than the curated map.
- The Expert map card carries the server's own words about what the map is and is not, so you can see the limits where the buttons are.
- Docker users need the cache volume for the learned map to survive a container removal: `-v pxa-cache:/work/.cache/pxa -e PXA_CACHE_DIR=/work/.cache/pxa`.

**Speed**
- Overdrive is a per-model preset for Flash-Next on one card plus system RAM. Turn it on by loading `presets/pxa-overdrive-flashnext-pxqn2-1xp100.env` as the server's environment; with Docker, pass it with `--env-file`. The full server line is in the cookbook. The switches live in the closed library: without it the preset does nothing, and every switch is off unless the preset turns it on.
- The expert cache can fetch cold experts ahead of the GPU, and the drafter can account for that cost.
- Speculative depth can stop early when the draft is unsure.
- PXA4 ships in this release but does nothing on its own. It only runs on a file encoded for it.

**Updates**
- `pxa-update check`, `pxa-update apply` and `pxa-update rollback`. Downloads are checked against their checksum and staged in the install folder before they are switched to. It refuses to replace an install while a server from that install is running, it does not restart anything by itself, and it does not touch your models or settings.

**API**
- `/v1/responses` and `/v1/messages` are covered by live checks on every release.

**What ships**
- Two engine tarballs, as in v3: one for glibc 2.38 (Ubuntu 24.04) and one for glibc 2.35 (Ubuntu 22.04). The installer picks from your glibc. Both carry sm_60, sm_61 and sm_70 code — the GTX 1080 Ti (sm_61) is now really built, not only named.
- The closed PXQN library is inside both, and is also on the release page on its own for a source build.
- The binary prints its own build number and commit (`llama-server --version`), and the package build refuses to finish if the banner cannot name the commit the tarball was built from.
- The image is `ghcr.io/poisonxa16/pxa:v3.1`.

## Fixed

- A rare hang in the expert cache's CPU worker pool under load: a decode could stall waiting on a job that never started.
- The expert counts file is saved without touching the curated file.

## Known issues

- Overdrive is one card only. Splitting Flash-Next across cards did not go faster.
- The competitor warms up within a session, so its decode numbers change a lot between the start and the end of a session. We report the end, where it is about 4% faster than PXA on code decode.
- Chat memory catches a fact repeated in nearly the same words. A paraphrase can be stored twice.
- The built-in web search is best-effort. Niche queries can come back thin; use one of the providers for better results.
- Hot swap is refused on Pascal cards by the launcher (see PXA Control above).
- The six container test failures from v3 are still environment, not the engine. On the host, PXA Control's own tests pass.
- The v3 known issues still apply where this note does not replace them. See [docs/KNOWN-ISSUES.md](docs/KNOWN-ISSUES.md).

## Upgrade

Unpack into a new folder (`install.sh` does this). Same model files.

- If an old post told you to export a `PXA_*` variable to get speed, unset it. A value you set wins over the default, including a slower one.
- Containers that run Flash-Next with the expert cache on a machine with more than one CPU socket need `--cap-add SYS_NICE`.
- Older clients stop at load with an error on a locked Pro file. Locked files load in v3 or newer.
- The container image is `ghcr.io/poisonxa16/pxa:v3.1`.

## Downloads

Each file has a `.sha256` next to it. The installer checks it.

| File | For |
|---|---|
| `pxa-v3.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz` | Ubuntu 24.04 and newer (glibc 2.38) |
| `pxa-v3.1-linux-x86_64-cuda12.8-sm60_61_70-ubuntu22.04.tar.gz` | Ubuntu 22.04 and newer (glibc 2.35) |

Both need an NVIDIA driver 570 or newer. They bring their own CUDA 12.8 runtime. Cards: Tesla P100, V100, GTX 1080 Ti and other Pascal cards.

## Credits

mistrjirka, quenthalion and thisistimow. Thanks to everyone who tests and reports on the Discord.

## Community

- Live leaderboard: https://benchmarks.pxanetwork.com
- Discord: https://discord.gg/EqazvV9tf
- Support on Ko-fi: https://ko-fi.com/shatteredrealms1
