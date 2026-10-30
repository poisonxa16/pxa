# What ENHANCE picks by itself

ENHANCE is the engine's default level. With it, a plain command such as

```bash
llama-server -m Qwen3.8-27B-PXQN4.gguf
```

gets the same settings the launcher (`tools/pxa-launch.py`) would pick for the same cards and the
same file. You do not need `-ngl`, `-sm`, `-ts`, `-b`, `-ub`, `-fa`, `-c` or any `PXA_*`
variable. The engine reads the cards and the file header, then fills every flag you left out.
At boot it prints each decision as one `PXA_REGISTRY:` line, with the reason and the measurement
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

The same picks apply in every tool: llama-server, llama-cli, llama-bench and the other common tools resolve the
registry at the same point (after the GGUF header is read, before the model loads), and a lever you name in the
environment is kept. llama-bench takes your own -sm / -ngl / -fa / -b / -ub / KV values and only gets the lever
environment of the cell. `PXA_AUTOCONFIG=0` turns the whole registry off in every tool (no picks, no lever
environment, and in llama-server no automatic -b / -ub either: stock values); the kernels keep their own level.

A few rules keep the other tools honest:

- Only the first model of a process sets the lever environment (the -m model; in llama-server also when hot
  models are registered). A later model whose picks differ gets a WARNING line naming each one; it runs under
  the first model's levers. `PXA_HS_ENV_UNION=1` restores the old behaviour (every model's picks, first wins).
- An empty value (`PXA_X=`) counts as unset, so a blank line in a compose file does not turn a pick off.
- llama-perplexity (PPL and KLD) and llama-imatrix are measuring instruments: they get the lever environment
  only, never a -c, KV cache, batch, split or -ngl pick.
- llama-cli and the other common tools take a -c pick only when you gave no -c and it would not shrink the
  model's trained context window.
- llama-bench resolves every test (each -m / -sm / -ts / KV combination) and applies only the picks they all
  share; the others are named in a warning. Run one -sm or model per invocation to get its own picks.

The source of truth is `common/pxa-registry.cpp` (flags and environment) and
`examples/server/server.cpp` (speculation). Kernel levers are listed in [`LEVERS.md`](LEVERS.md).
Every number in the tables below was measured on the card class named in its row.

## Per card set

"dense 27B" means the Qwen3.8-27B family (`qwen35`). Tiers are the PXQ file tiers.

| Card set | What ENHANCE picks | Why (measured) |
|---|---|---|
| **1x P100** (sm_60) | `-ngl 999`, `-fa on`, `-c` = np x 4096. Dense files: `-ub` from the VRAM ladder at load. PXQU MoE: `-b 2048 -ub 2048`. | `-ngl`/`-c`: the launcher's anchor. the PXQU batch cell was measured per model. |
| | Speculation: the n-gram stage on its own (`ngram:n_max=64,n_min=2,ngram_size_n=24`). With `--spec-type mtp`, the draft depth follows context depth (`0:1`). | n-gram alone has been the default on every card since 2026-09-14; the MTP draft depth ramps with the context depth. |
| | Kernels: quantized-KV narrow attention (`PXA_FA_QKV_DIRECT`), PXQN4 row-block decode GEMV (`PXA_PXQN_RB`, `PXA_PXQN_RB_RHT`), the same form for PXQN4S8 and PXQN5 with its shape rule (`PXA_PXQN_RB_S8`, `PXA_PXQN_RB_N5`, `PXA_PXQN_SM60_RBSHAPE`), rotated sites at verify widths (`PXA_PXQN_RHT_NODE_NY`), DeltaNet conv cluster and in-place carry (`PXA_DN_CONVFUSE`, `PXA_DN_INPLACE`, `PXA_DN_CONVFUSE_NY`). | each of these is the measured default on this card class. |
| | PXQN files: the fused norm + RHT site on 8 blocks (`PXA_PXQN_RMS_RHT_GB=1`) and the many-block greedy argmax (`PXA_ARGMAX_VAL_MB=1`), both bit-identical; `=0` puts either back. | 27B PXQN4 decode +0.6 to +1.4%, every response identical (2026-10-14). |
| | Dense 27B PXQN4 file: the persistent row-block one-column GEMV (`PXA_PXQN_RBP=1`), bit-identical; `=0` puts it back. | 27B PXQN4 server decode +4.6% prose, +4.8% code, every response identical (2026-10-16). |
| **2x P100** | `-sm tensor` with an even `-ts` for `qwen35` files in PXQ4, PXQN4, PXQN4S8 or PXQN5, when every layer is on the cards and `-fa` is on. Other files, or `-ts` given, get `-sm layer`. | decode +17 to +27%, prefill at 22.6k +44% over the layer split. |
| | Reduce: the fused all-reduce (`PXA_TSPLIT_REDUCE=fused`) for decode. Prefill-width reduces take the two-device route (`PXA_TSPLIT_PF`). `PXA_TSPLIT_FALLBACK=1`, so the engine falls back to layer instead of stopping if its capability check refuses the file. A pair that does not peer takes the staged route by itself. | +13% decode over the un-fused reduce; the prefill-width reduce takes the two-device route. |
| | Dense: `-b 8192 -ub 256 -c 32768`. Expert PXQ4/PXQ6 file: `-b 8192 -ub 2048 -c 8192`. | P100 pair: 340 t/s at 3.1k with `-ub 256`, against 231 with 2048. |
| | With `--spec-type mtp` under the split: depth 2. | measured 2026-09-28 on a P100 pair: 43.4 / 44.9 / 53.0 against plain 38.2 / 37.5 / 36.4 (control / code / repetition). |
| | MTP file, bare server, multi-card `-sm tensor` -> auto MTP, depth 3 with top-1 floor 0.8 (P100) / 0.9 (V100); opt out `PXA_SPEC_AUTO_MTP=0` or `--spec-type none`. | the top-1 confidence floor is 0.8 on P100 and 0.9 on V100. |
| | MoE files (`qwen35moe`) on P100s, any card count: one graph shape for every prompt batch (`PXA_PF_STABLE_SHAPE=1`); `=0` puts the fused gate back. | 35B-A3B PXQN4 on a pair, 16k prompt: 1,150.6 -> 1,569.7 t/s, output identical (2026-10-16). |
| | MoE MTP files (`qwen35moe` PXQN4) on a P100 pair with `-sm layer`: the decode stack (`PXA_AUTO_P60`) is on in every tool. It is correct but not bit-identical to the stack-off path (its one-token MoE half moves the greedy text; every other member is identical). `PXA_P60_MEGA_EXACT=1` makes it bit-identical at about 9% decode cost; `PXA_AUTO_P60=0` or `PXA_AUTOCONFIG=0` turns the stack off. | 35B-A3B PXQN4, P01, llama-cli greedy 512 after 6k: default 125.5 t/s, `PXA_P60_MEGA_EXACT=1` 114.8 (sha equal to off), off 74.0 (2026-10-16, NOT-QUIET). |
| | Qwen3.5/3.6 MoE PXQN4 (35B-A3B class): long prefill with a stable prompt-batch shape and the P100 head-256 prompt attention (`PXA_FA_TILE_H2CHUNK=1`, `PXA_FA_LONGPREFILL=3` on top of `PXA_PF_STABLE_SHAPE=1`); `PXA_AUTO_PF16K=0` turns it off. Correct, not bit-identical (KLD within the engine's own noise). | pp16384 +43.5%, pp4096 +22.1% (2026-10-15). |
| | Dense PXQN files on a P100 pair with `-sm tensor`: the prompt pass overlaps its card-to-card traffic with compute (`PXA_PF_SLICE`, `=0` off). Output identical. | 27B PXQN4, `-fa 1 -ub 2048`: pp4096 351.4 -> 399.2 t/s, pp512 344.4 -> 371.2 (2026-10-17, quick probe). |
| | Qwen3.5/3.6 dense PXQN4 (27B class) on one P100 or a P100 pair: the same P100 head-256 prompt attention as the MoE row above (`PXA_FA_TILE_H2CHUNK=1`, `PXA_FA_LONGPREFILL=3`); `PXA_AUTO_PF16K=0` turns it off. Correct, not bit-identical. | 27B PXQN4 pp16384 353.3 -> 372.7 t/s (+5.5%), pp4096 399.2 -> 403.6 (2026-10-17, quick probe); one P100 pp16384 211.7 -> 224.2 (+5.9%), pp4096 244.5 -> 248.3 (2026-10-18). |
| | Qwen3.5/3.6 dense PXQN4 (27B class) on one V100 or one P100, f16 KV, `-ub`/`-b` unset: `-ub` walks from 2048 down to the stock 512 by what fits at the requested context, `-b` raised to match (llama-bench, and the server when the KV cache is f16); `PXA_AUTO_UB_DENSE_2048=0` keeps 512. Correct, not bit-identical to `-ub 512` (larger prompt batches change the GEMM shapes). | llama-bench vs v3.1: pp4096 954.0 -> 1091.9 t/s (-ub 2048), pp16384 -ub 1024 874.8 -> 982.6 (2026-10-18); 1x P100: pp4096 248.0 -> 253.4, pp16384 224.1 -> 230.8 (v3.1 244.7 / 210.9, #23355). |
| | Qwen3.5/3.6 MoE PXQN4 (35B-A3B class) on ONE V100 when the file is larger than the card by up to half again: the cold experts run on a pinned CPU pool without stopping the GPU stream, spread over more, thinner layers (`PXA_AUTO_XCACHE_SMALLTAIL`, `=0` off). Correct, not bit-identical to the old placement (same text as the old CPU path at the same placement). | 35B-A3B greedy prose/code/chat 66.9/67.3/67.8 -> 77.7/78.0/74.7 t/s (v3.1 59-65, NOT-QUIET, 2026-10-19). |
| | Every model whose cold experts run on the CPU pool (`PXA_XCACHE_CPUPOOL`): a pool phase ends when its last task finishes instead of when every pinned worker has arrived, so a worker that another process preempts no longer stalls every cold layer (`PXA_XCACHE_CPUPOOL_TASK_BARRIER`, `=0` = the old barrier). Bit-identical. | 35B-A3B PXQN4 one V100, box load 67-74: 33.8/26.2/31.1 -> 83.4/77.9/75.5 t/s prose/code/chat (2026-10-20, same bracket). |
| **4x P100** | `-sm tensor` on four identical P100s (same file rules as the pair). Dense: `-b 2048 -ub 256`. Expert: `-b 2048 -ub 2048`. Device-side expert map and GQA packing (`PXA_MOE_DEVICE_MAP`, `PXA_FA_GQA_PACK=4`). | decode +5 to +41%, prefill at 22.6k about 2x. |
| | Speculation: n-gram alone. `qwen4exp` (Flash-Next) gets `ngram-mod:n_max=4,n_min=2`. | n-gram alone by default on this class too. |
| **1x V100** (sm_70) | `-ngl 999`, `-fa on`. With a q8_0 K and V cache at `-c` 49152 or more, `-ub 1024`. PXQU MoE: `-b 2048 -ub 2048`. | 642 against 388 t/s at pp65536 for the wider `-ub`. |
| | Speculation: n-gram alone. With `--spec-type mtp`, depth follows context (`0:1, 8192:2, 16384:3`). | n-gram alone by default; the MTP depth follows the context depth. |
| | Kernels: narrow attention over a quantized cache read in place (`PXA_FA_QKV_DIRECT_VOLTA`), head-256 tile attention (`PXA_FA_D256_VOLTA_TILE=2`), MMA prefill attention including a q8_0 cache (`PXA_FA_MMA_VOLTA`, `_Q8`). | each is the measured default on this card class. |
| **2x V100** | `-sm tensor`, same file rules as the P100 pair. Fused reduce plus the prefill route, as above. `-b 8192 -ub 2048 -c 32768`. | decode +12 to +28%, prefill +3 to +4%; the fused reduce adds +13% decode. |
| | ssm_out RHT ranges, concat and q8_1 in one kernel (`PXA_PXQN_RR_Q8`). The fused epilogue `PXA_TSPLIT_EPI_Q8` stays off. | the q8_1 fused form measured +3.0% tg128; the fused epilogue was a measured loss and stays off. |
| | With `--spec-type mtp` under the split: depth 2, the only depth above plain on every class. | 58.9 / 71.8 / 68.1 against plain 53.6 / 53.9 / 53.7 (control / code / repetition). |
| **P100 + V100 mixed** | `-sm layer`. An even tensor split runs every step at the slower card's pace. `-ts` is llama.cpp's free-memory split. `-ub` from the ladder. No measured batch cell. | rule, no measurement (see [Unmeasured](#unmeasured)) |
| **1x 1080 Ti** (sm_61) | `-fa on`, `-b 2048 -ub 768 -c 8192` (a `-ub` of 2048 does not fit in 11 GB). The int8 prefill tile (`PXA_PXQ_INT8_PREFILL` mode 1). Auto-speculation is declined: a draft context runs out of memory mid-prefill on this card. | 1,306 t/s cold prefill measured on this card. |

On any card set, Flash-Next (`qwen4exp`) runs its hyper-connection mixer as two fused kernels (`PXA_QWEN4EXP_HC_FUSED`, engine
default since 2026-10-16, output identical, `=0` off).

On any card set, a PCIe link narrower than x4 keeps `-sm layer` and turns pipeline parallelism
off. Gemma 4 keeps `-sm layer`; the split there is opt-in with `PXA_TSPLIT_GEMMA4=1`.

## Cost lines

These values are kept as typed, and one line is printed at boot:

| You typed | What it costs | Better value |
|---|---|---|---|
| `-sm layer` where ENHANCE picks tensor | P100 pair: decode 17 to 27% slower, prefill at 22.6k 44% slower. V100 pair: decode 12 to 28% slower. 4x P100: decode 5 to 41% slower, prefill about half. | leave `-sm` out |
| `PXA_TSPLIT_REDUCE=off` under a tensor split on a pair whose cards peer | about 7% slower decode (P100 pair, 27B PXQ4: 24.7 against 26.5/26.9 t/s). This one is printed by the reduce itself at its first reduce, because only it knows whether the cards peer. | unset it |
| `PXA_TSPLIT_REDUCE_PREFILL=1` under a two-card tensor split | about 10% slower prefill (V100 pair, 14.8k prompt: 739.5/745.5 against 820.0/824.7 t/s). Decode and output are unchanged. | unset it |

Example (2x P100, `-sm layer` typed on the dense 27B):

```
PXA_REGISTRY: COST -sm layer -sm layer on 2x P100 (sm_60) (arch qwen35, tier PXQN4): the tensor split measured faster here (...); drop -sm to let ENHANCE pick it
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
