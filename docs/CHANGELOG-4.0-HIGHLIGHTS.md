# PXA 4.0 highlights

Short form of `docs/CHANGELOG.md`: the 4.0.0 block with the same numbers.

## 4.0.0 (date TBD)

> ## PXA 4.0
>
> - **+42% over a popular community fork on a 27B, same P100 pair:** 26.67 vs 18.78 tok/s tg128.
> - **35B-A3B MoE with MTP on two P100: 138.1 tok/s greedy,** +4.5% over the fork in the same bracket and +38.6% over our previous MTP build.
> - **16k-prompt prefill on the MoE: 1803 tok/s with the opt-in long-prompt preset,** +23% over the fork (NOT-QUIET, fork in the earlier bracket).
> - **28k prompts on one V100 (Flash-Next): 324 -> 597 tok/s,** 1.84x, from an automatic ubatch choice.
> - **27B with MTP on one V100: +14.8% prose / +38.4% code over 3.1** (stock file, greedy, same box).
> - **Flash-Next presets +7.2% prose / +5.9% code on 2x V100,** bit-identical.
> - **One V100 on Flash-Next Overdrive: +19.5% prose / +31.5% code** (38.8 / 47.5 vs 32.4 / 36.1 tok/s; quiet rerun owed).
> - **PXA Control now ships a Windows exe, a macOS Intel build and an Android client,** with an agentic Assistant chat covered by a 152-case test suite.

Numbers are same-box measurements. Most arms ran NOT-QUIET (box load 10-50) and are lower bounds.

### What's new

#### Pascal (sm_60)

- **35B-A3B decode on two P100 goes from 75.9 to 133.4 tok/s** (tg128, +76%), on by default (75.9 is our earlier build; v3.1 cannot run this model).
- **PXQN4 files get the same P100 decode speedup** for MoE, gated DeltaNet and attention layers.
- **MTP verify on P100 is 38.6% faster** (99.6 -> 138.1 tok/s greedy). On by default.
- **A further opt-in verify speedup adds +4.4% / +3.6%** over the fork on the MTP board cell.
- **One-card 27B gains +5.3% prose / +5.2% code over 3.1** (25.18 -> 26.53 and 25.08 -> 26.40 tok/s, same box, bracket and prompts), identical output.
- **Single-token PXQN4 decode on P100 reads 17% more bytes per second** (ffn_down 0.1181 -> 0.1008 ms), byte-identical; on by default for one P100 with a dense 27B PXQN4 file (see Engine kernels, Pascal).
- **MoE prompt processing on P100 is faster:** pp512 440 -> 1028 on the fork's own file in the reproduction. Opt-in.
- **Same-box yardstick:** a popular community fork's P100 patch set was reproduced on our cards (tg128 65.8 -> 119.6) so every comparison above is same-bracket.

#### Volta (sm_70)

- **27B MTP verify is up to 1.46x at width 8** (pp8 155 -> 226 tok/s in the faster opt-in mode; the bit-identical opt-in mode gives 164). Opt-in, widths >= 4.
- **Plain 27B decode on one V100 is +3.0%** (40.3 -> 40.9 tok/s, identical output); the card is GPU-bound now.
- **MTP on V100 is +1.2% faster** from a lower per-token host cost.
- **Single-token decode on V100 is +0.8% faster (tg128)** by default, bit-identical.
- **One V100 decodes +1.5% faster** now that an old per-card default that turned a decode speedup off is gone.
- **Overdrive on V100: prose +20.5% in the close bracket** with an opt-in setting (loaded box).
- **A Flash-Next speedup is in every preset:** 2x V100 layer split +7.2% prose / +5.9% code, bit-identical.

#### Speculation / MTP

- **Depth presets per context** raise 27B 1x V100 8k prose from 57.5 to 62.6 tok/s (+9.1%): `mtp:n_max=4,p_min=0.5`. 16k uses a ramp; 32k n3 is gated-owed.
- **Short-context 27B on a V100 reaches 62.4 prose / 84.4 code** (+5% / +29% vs depth 1) at depth 4, p_min .5.
- **Flash-Next prose stops locking at a shallow depth** with a confidence floor: `mtp:n_max=3,p_min=0.75`.
- **Identical text across restarts is possible on 1x P100:** 24/24 identical over two boots with the determinism setup.
- **35B-A3B MTP acceptance is .774** with det 12/12 and needle 16k + 32k passing.
- **An opt-in Overdrive mode trades output for speed;** it is off by default.

#### Overdrive / Flash-Next

- **A learned expert map makes the cache faster from the start:** 28.50 -> 32.59 prose, 31.59 -> 35.07 code on one P100; the curated map beside the model is worth +16% / +25%.
- **Deterministic Overdrive works:** with the determinism setup, 12/12 and 24/24 identical texts on V100 with no speed cost.
- **The 32 GB file gets smaller and faster:** the per-layer table shrinks by 19 / 29 GB (KLD +0.0043), and 2x V100 is +20.1% faster.
- **28k prompts on one V100 prefill 1.84x faster** (324 -> 597 tok/s): the server picks `-ub 4096` when one card streams experts; default on.
- **Two Overdrive speedups add +6.6% code and +6.0% code.**
- **The one-P100 preset is confirmed by A/B:** 29.52 / 37.81 vs 26.29 / 32.88 tok/s on card 0.
- **Control picks the measured Overdrive setup** by reading the Flash-Next header.
- **Less RAM pressure:** an opt-in direct-read mode for the per-layer table, and the launcher guards against a rotational disk for it.
- **An opt-in thread-placement setting** pins the decode and cold-path threads.
- **16 GB cards boot reliably:** q8_0 KV, no `-b 8192`, and expert maps are found under every file name.

#### Tensor split

- **Flash-Next runs under `-sm tensor` on two 16 GB cards** (it ran out of memory at boot before): 2x P100 MTP prose 36.85 vs 33.98 (+8.4%), code +5.3%.
- **The launcher picks the split for you:** a P100 pair gets `-sm tensor` with its split settings; a V100 pair stays on layer, where it ties.
- **A Flash-Next speedup on both cards** adds +12.7% prose / +18.4% code over before; bit-identical.
- **Planner tests cover uneven pairs and 4-way splits.**

#### Long prefill

- **35B-A3B prefill on two P100 reaches 1803 tok/s at 16k** (+23% over the fork; the arm without the lean attention gave 1730 in a closer bracket). NOT-QUIET.
- **Long-prompt prefill is faster on a P100 pair:** pp4096 1595 -> 1953, pp16384 1257 -> 1730 (+22% / +38%); on by default for Qwen3.5-MoE on all-P100 cards.
- **Lean long-prompt attention adds +5.2% at 16k** (29.7 vs 36.1 ms per attention call); engages from 8192 tokens of context. Opt-in.
- **A second opt-in attention setting adds +4.4% at 4k, +11.7% at 16k.**
- **Flash-Next 28k prefill is 1.84x** with decode unchanged and needle 28k passing.

#### Quantization / PXQN

- **PXQN3 and PXQN2 are 2.5x and 3.0x faster than a community quant at the same bytes** (29.39 vs 11.59, 34.26 vs 11.54 tok/s) with lower KLD at 3 bits (0.0364 vs 0.0631).
- **PXQN4 beats uniform PXQ4 at equal bytes:** 25.5 vs 20.1 tok/s on one P100.
- **PXQN4 files for the 27B and the 35B-A3B MTP file use the newer encode path by default** (19.52 GB, boots 42/42 on two P100).
- **gguf-py reads every PXQN type again** (830 failures -> 0 on two real files).
- **The encoder writes routing counts beside every MoE file.**
- **PXQN revision 2 files** are supported by the closed library.

#### Models

- **Qwen3.6-35B-A3B (qwen35moe) with its MTP block is supported** end to end: encode, boot, 138 tok/s on two P100.
- **Flash-Next fits more hardware:** one V100 or P100 with RAM expert streaming, two 16 GB cards under tensor split.
- **Nemotron, Nemotron-H and Nemotron-H MoE load** on CPU and CUDA (ported from mainline llama.cpp).
- **GLM-5.3-Flash has an N2 file;** no 4.0 speed number yet.
- **Ornith 1.5 35B-A3B uses the same MoE paths** as the Qwen3.6 MoE.

#### PXA Control

- **One download runs everything:** a compiled single file (18.8 MB, 0.5 s start) and a Linux `.run` that carries the engine.
- **The UI is rebuilt:** Home for first run / ready / serving, grouped navigation, Plugins and Store pages, tier badges.
- **Start tells you why it refuses,** using the launcher's own reason.
- **The launcher reads the box:** sockets, NUMA, each card's node, the CPU pool; every decision is named in the plan.
- **The Assistant is an agent:** tools, sub-agents (`spawn_agent`), memory across chats with incognito, recall, `@` references, auto-compaction on exact token counts.
- **A 152-case suite over Control's HTTP API tests the agent loop,** found three fixes, plus 28 unit tests.
- **Chat renders math (KaTeX), switches between edited message versions,** and fits a phone.
- **The licensed library updates by itself** as its own signed, rollback-able artifact, shown beside the encoder's.
- **History is kept on disk** and Home Assistant MQTT is opt-in.
- **Hardening H1-H10 ships** (loopback session protection, CSRF and rebinding guards, request limits, content-security policy).
- **One-click model downloads, the live header names every server,** and the log shows its backlog.

#### Platforms

- **Windows x64 Control exe and engine build:** the engine is `llama-server.exe`, the PXQN library is `ggml-pxqn.dll`. Real-GPU run owed.
- **macOS Intel Control build script,** with docs that state only what was verified.
- **Android client APK** (a WebView wrapper that remotes to your Control host; reproducible command-line build; API 34 emulator smoke test).
- **Installable web app (PWA)** with manifest, shell service worker, host picker, icons.
- **Container fixes:** the cache volume is owned by the `pxq` user and the server runs as pid 1.

#### Licensing / store / sidecar

- **The PXA Store runs in licd:** credits, vouchers, signed catalogue, machine-bound grants, grant-bound downloads. Live since 10-14.
- **A Ko-fi donation becomes a voucher (Store credits)** end to end in the local test (store 21/21, base licd 100/100 tests).
- **Control talks to the real store:** catalogue, link, voucher, redeem, refresh, unlink, supporter-model downloads.
- **The vLLM sidecar ships in the release tarball** and runs without Docker.
- **The sidecar serves PXQN4/PXQN5/PXQN3:** V100 TP2 PXQN4 48.0 tok/s; P100 TP2 +8.6% over vLLM PXQ4; PXQN5 top-1 99.84% vs the engine.
- **`pxa-update` fails closed,** needs the exact tarball name, stages in the install directory and rolls back.
- **The release package carries** the library updater and its verifier, the expert-map tool, the Grafana dashboard and the Overdrive preset.

### Upgrade notes

**Defaults that changed**
- One card that streams MoE experts now picks `-ub 4096`.
- One V100 keeps a decode speedup that an earlier per-card default turned off.
- Flash-Next on a P100 pair starts with `-sm tensor`; a V100 pair stays on layer.
- P100 MoE decode and MTP verify settings are on by default.
- Every Flash-Next preset includes a bit-identical speedup.
- A bit-identical single-token decode speedup is on by default. Every default above can be turned off; docs/LEVERS.md lists the master switches and the documented off switches.

**Levers that need a switch (all off by default)**
- Volta: two faster MTP verify modes (one exact, one faster) and the Overdrive verify graph.
- Prefill: long-prompt attention and prefill settings.
- Speculation: rollback, depth cutoff and confidence settings, and a determinism setup for identical text.
- Overdrive: a direct-read mode, thread placement and a skip mode that changes output.

**Behaviour changes**
- Plain-greedy device-side argmax is opt-in (off by default).
- Texts can differ from 3.1 at near-ties because accumulation order changed; identical text across restarts needs the determinism setup.
- Two unused Overdrive cache settings are removed.

**Known issues**
- Flash-Next is not supported on the vLLM sidecar.
- Windows: the PXQN DLL builds but a real-GPU run is owed; no code signing.
- macOS: Intel only; no Apple Silicon engine or Control build.
- Android: remote client only; no on-device engine.
- Gates owed: the 32k n3 speculation gate; 35B-A3B v3.1 cell and KLD; quiet rerun of the V100 1x Overdrive headline; the same-bracket 3.1 pair for 27B plain decode on one V100.
- Four-P100 Flash-Next greedy text is not stable across boots with speculation on (use `--spec-type none`); V100 logits depend slightly on the KV cell offset.
- GLM-5.3-Flash: no 4.0 speed number yet; a short-prompt NaN on multi-backend execution is narrowed but open; speculative decoding at a draft depth of 4 can still crash with a heap corruption (open).
