# What ENHANCE picks by itself

ENHANCE is the engine's default level. With it, a plain command such as

```bash
llama-server -m Qwen3.8-27B-PXQN4.gguf
```

gets the same settings the launcher (`tools/pxa-launch.py`) would pick for the same cards and the
same file. You do not need `-ngl`, `-sm`, `-ts`, `-b`, `-ub`, `-fa`, `-c` or any `PXA_*`
variable. The engine reads the cards and the file header, then fills every flag you left out.
At boot it prints each decision as one `PXA_REGISTRY:` line, with the reason and the ledger row
behind it. To see the plan without loading anything:

```bash
PXA_EXPLAIN=1 llama-server -m model.gguf                         # the cards in this box
PXA_EXPLAIN=1 PXA_TOPOLOGY=2x600 llama-server -m model.gguf      # describe cards instead (600 = P100, 700 = V100, 610 = 1080 Ti)
```

**A flag you type always wins.** When the value you typed is one that was measured slower than
the ENHANCE pick, the engine still uses your value. It also prints one warning line (`PXA_REGISTRY: COST`, or the reduce's own line)
that says what the value costs and what the better value is (see [Cost lines](#cost-lines)).

`PXA_REFERENCE=1` is the exact-reference switch: it stands every PXA default down. `PXA_ENHANCE=0`
goes back to the older DEFAULT level.

The source of truth is `common/pxa-registry.cpp` (flags and environment) and
`examples/server/server.cpp` (speculation). Kernel levers are listed in [`LEVERS.md`](LEVERS.md).
Ledger ids point to rows in the implementations ledger.

## Per card set

"dense 27B" means the Qwen3.8-27B family (`qwen35`). Tiers are the PXQ file tiers.

| Card set | What ENHANCE picks | Why (ledger) |
|---|---|---|
| **1x P100** (sm_60) | `-ngl 999`, `-fa on`, `-c` = np x 4096. Dense files: `-ub` from the VRAM ladder at load. PXQU MoE: `-b 2048 -ub 2048`. | `-ngl`/`-c`: the launcher's anchor. PXQU cell: `prefill-auto-ubatch-model-aware`. |
| | Speculation: the n-gram stage on its own (`ngram:n_max=64,n_min=2,ngram_size_n=24`). With `--spec-type mtp`, the draft depth follows context depth (`0:1`). | `spec-auto-default-ngram-alone-volta` (fleet-wide since 2026-09-14), `spec-depth-nmax-ramp` |
| | Kernels: quantized-KV narrow attention (`PXA_FA_QKV_DIRECT`), the PXQN decode levers of the release build (`PXA_PXQN_RB`, `PXA_PXQN_RB_RHT`, `PXA_PXQN_RHT_NODE_NY`), DeltaNet conv cluster and in-place carry (`PXA_DN_CONVFUSE`, `PXA_DN_INPLACE`, `PXA_DN_CONVFUSE_NY`). | `ctx16k-fa-qkv-split-pv8`, `gemv-bw-pxqn4-rb`, `pxqn-rb-rht-unfuse`, `pxqn-rht-node-ny`, `dn-convfuse`, `dn-inplace-carry`, `ctx16k-dn-convfuse-depth`, `dnny-convfuse-ny-fire` |
| **2x P100** | `-sm tensor` with an even `-ts` for `qwen35` files in PXQ4, PXQN4, PXQN4S8 or PXQN5, when every layer is on the cards and `-fa` is on. Other files, or `-ts` given, get `-sm layer`. | `launcher-auto-tensor-split`, `auto-sm-pxqn-tiers`: decode +17 to +27%, prefill at 22.6k +44% over layer |
| | Reduce: the fused all-reduce (`PXA_TSPLIT_REDUCE=fused`) for decode. Prefill-width reduces take the two-device route (`PXA_TSPLIT_PF`). `PXA_TSPLIT_FALLBACK=1`, so the engine falls back to layer instead of stopping if its capability check refuses the file. A pair that does not peer takes the staged route by itself. | `pxa-tsplit-fused-reduce`, `tsplit-prefill-pf-supersedes`, `p2p-startup-selftest` |
| | Dense: `-b 8192 -ub 256 -c 32768`. Expert PXQ4/PXQ6 file: `-b 8192 -ub 2048 -c 8192`. | `prefill-auto-ubatch-model-aware` (P100 pair: 340 t/s at 3.1k with `-ub 256`, against 231 with 2048) |
| | With `--spec-type mtp` under the split: depth 2. | `PXA_MTP_TSPLIT_NMAX` row in LEVERS; P56 re-measure 2026-09-28: 43.4 / 44.9 / 53.0 against plain 38.2 / 37.5 / 36.4 (control / code / repetition) |
| | MTP file, bare server, multi-card `-sm tensor` -> auto MTP, depth 3 with top-1 floor 0.8 (P100) / 0.9 (V100); opt out `PXA_SPEC_AUTO_MTP=0` or `--spec-type none`. | ledger `mtpround-auto-mtp`, `mtpround-conf-gate` |
| **4x P100** | `-sm tensor` on four identical P100s (same file rules as the pair). Dense: `-b 2048 -ub 256`. Expert: `-b 2048 -ub 2048`. Device-side expert map and GQA packing (`PXA_MOE_DEVICE_MAP`, `PXA_FA_GQA_PACK=4`). | `tsplit-4card-default` (decode +5 to +41%, prefill at 22.6k about 2x), `prefill-auto-ubatch-model-aware`, `moe-device-map-enhance-default` |
| | Speculation: n-gram alone. `qwen4exp` (Flash-Next) gets `ngram-mod:n_max=4,n_min=2`. | `spec-auto-default-pascal` |
| **1x V100** (sm_70) | `-ngl 999`, `-fa on`. With a q8_0 K and V cache at `-c` 49152 or more, `-ub 1024`. PXQU MoE: `-b 2048 -ub 2048`. | bug #207 (642 against 388 t/s at pp65536), `prefill-auto-ubatch-model-aware` |
| | Speculation: n-gram alone. With `--spec-type mtp`, depth follows context (`0:1, 8192:2, 16384:3`). | `spec-auto-default-ngram-alone-volta`, `spec-depth-nmax-ramp` |
| | Kernels: narrow attention over a quantized cache read in place (`PXA_FA_QKV_DIRECT_VOLTA`), head-256 tile attention (`PXA_FA_D256_VOLTA_TILE=2`), MMA prefill attention including a q8_0 cache (`PXA_FA_MMA_VOLTA`, `_Q8`). | `v100-depth-qkv-direct-volta`, `vwtile-volta-d256-verify-tile`, `onecard-volta-mma-q4kv` |
| **2x V100** | `-sm tensor`, same file rules as the P100 pair. Fused reduce plus the prefill route, as above. `-b 8192 -ub 2048 -c 32768`. | `launcher-auto-tensor-split` (decode +12 to +28%, prefill +3 to +4%), `pxa-tsplit-fused-reduce` (+13% decode) |
| | A PXQN V100 decode lever of the release build (`PXA_PXQN_RR_Q8`). The fused epilogue `PXA_TSPLIT_EPI_Q8` stays off. | `v100-plain-rr-q8` (+3.0% tg128), `v100-plain-epi-q8` (measured loss) |
| | With `--spec-type mtp` under the split: depth 2, the only depth above plain on every class. | mail #10817: 58.9 / 71.8 / 68.1 against plain 53.6 / 53.9 / 53.7 |
| **P100 + V100 mixed** | `-sm layer`. An even tensor split runs every step at the slower card's pace. `-ts` is llama.cpp's free-memory split. `-ub` from the ladder. No measured batch cell. | rule, no measurement (see [Unmeasured](#unmeasured)) |
| **1x 1080 Ti** (sm_61) | `-fa on`, `-b 2048 -ub 768 -c 8192` (a `-ub` of 2048 does not fit in 11 GB). The int8 prefill tile (`PXA_PXQ_INT8_PREFILL` mode 1). Auto-speculation is declined: a draft context runs out of memory mid-prefill on this card. | RELEASE-NOTES-2026-09-07 1080 Ti row (1,306 t/s cold prefill), launcher recipe `1x1080ti-pxq2` |

On any card set, a PCIe link narrower than x4 keeps `-sm layer` and turns pipeline parallelism
off (bug #280). Gemma 4 keeps `-sm layer`; the split there is opt-in with `PXA_TSPLIT_GEMMA4=1`.

## Cost lines

These values are kept as typed, and one line is printed at boot:

| You typed | What it costs | Better value | Ledger |
|---|---|---|---|
| `-sm layer` where ENHANCE picks tensor | P100 pair: decode 17 to 27% slower, prefill at 22.6k 44% slower. V100 pair: decode 12 to 28% slower. 4x P100: decode 5 to 41% slower, prefill about half. | leave `-sm` out | `launcher-auto-tensor-split`, `tsplit-4card-default` |
| `PXA_TSPLIT_REDUCE=off` under a tensor split on a pair whose cards peer | about 7% slower decode (P100 pair, 27B PXQ4: 24.7 against 26.5/26.9 t/s). This one is printed by the reduce itself at its first reduce, because only it knows whether the cards peer. | unset it | `tsplit-reduce-off-warn`, `pxa-tsplit-fused-reduce` |
| `PXA_TSPLIT_REDUCE_PREFILL=1` under a two-card tensor split | about 10% slower prefill (V100 pair, 14.8k prompt: 739.5/745.5 against 820.0/824.7 t/s). Decode and output are unchanged. | unset it | `tsplit-prefill-pf-supersedes` |

Example (2x P100, `-sm layer` typed on the dense 27B):

```
PXA_REGISTRY: COST -sm layer [launcher-auto-tensor-split] -sm layer on 2x P100 (sm_60) (arch qwen35, tier PXQN4): the tensor split measured faster here (...); drop -sm to let ENHANCE pick it
```

`PXA_EXPLAIN=1` lists the `COST` lines under `"costs"` in its JSON.

## Unmeasured

Nothing on this list is changed by ENHANCE until a measurement says so:

- **The speculation shape for a file with an MTP head.** The default is n-gram alone
  (measured 2026-09-13/14, V100 pair and 4x P100). Today's MTP numbers (depth 2 above plain on
  both pairs) compare MTP with plain decoding, not with n-gram alone. On a bare server with a multi-card
  tensor split the MTP head is now used automatically (row above); single-card and layer-split
  runs keep n-gram alone, and `--spec-type mtp` is still available there.
- **The KV cache type.** The default is f16, as in the launcher. q8_0 and q4_0 change the output,
  so they are not a speed-only default.
- **`-b`/`-ub` under the tensor split.** The pair cells were measured on the layer split. No cost
  line is printed for a hand-typed `-b`/`-ub`.
- **Mixed P100 + V100.** There is no batch cell and no split measurement. The launcher's
  capacity `-ts` and the engine's free-memory `-ts` are close, but they have not been compared.
- **4x V100, 3 cards, 5+ cards.** The tensor split has no measurement on these sets, so they keep
  layer.
- **MTP depth under the split on 4x P100.** The depth-2 rule is applied to any multi-card sm_60
  set, but it was measured on a pair.
