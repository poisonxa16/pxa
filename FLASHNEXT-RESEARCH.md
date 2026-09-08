# Flash-Next research record — 2026-08-27/28

Everything established in this session, including the claims that were overturned
and why. Written so none of it has to be re-derived. Evidence tags:
**[M]** measured on hardware · **[S]** read from source or a log · **[C]** computed
here · **[I]** inferred.

---

## 0. The single most expensive mistake: benchmarking the wrong artifact

`run-flashnext.sh:25` defaults to the **public unsloth download**, not our codec:

    MODEL=${MODEL:-/path/to/models/qwen4exp-testfile/UD-IQ1_S/...}

That file has **zero PXQ tensors** (`grep -ci pxq` on its tensor directory = 0).
Census: F32 557 / Q8_0 244 / Q5_K 212 / IQ1_S 68 / IQ4_NL 49 / Q6_K 40 /
IQ2_XXS 28 / BF16 24 / Q4_K 2. **[M]**

Consequence: every PXQ kernel and every `PXA_PXQ*` env knob is **inert** on it.
A PXQ lever measured against that file is guaranteed to show nothing, and will
read as "the lever does not work" rather than "the lever was never engaged".

**Always pass `-m` or `MODEL=` explicitly. Always grep your own run's log for
`llama_model_loader: - type` and confirm `pxq4` tensors are present before
trusting any number.**

Our artifact: `/path/to/models/qwen4exp/Qwen3.8-Flash-Next-PXQ4.gguf`, 114.67 GiB,
384 pxq4 tensors, ~66 GiB across six cards with `per_layer_token_embd` on CPU.

## 1. Codec head-to-head — our PXQ4 vs the public quant

Identical engine, flags, context and token count; no profiler; interleaved arms;
each arm's loaded quant types read out of its own log.
`-c 2048 -n 128 -ts 7,16,16,16,16,16`, six cards.

| arm | samples | median | verdict |
|---|---|---|---|
| ours (PXQ4) | 28.80, 29.26 — **both 127 tokens** | **29.03 tok/s** | valid **[M]** |
| public IQ1_S | 25.52 (57 tok), 25.18 (56 tok) | 25.35 | **VOID** — token counts differ **[M]** |

**Our codec measures 29.03 tok/s (34.45 ms/token).** Logs: `/path/to/models/codec-h2h/`.

The public arm voided itself twice by hitting EOS at ~56-57 tokens against our
127. **Always pass `--ignore-eos`** for any cross-model comparison; without it
the two arms generate different token counts and the comparison is worthless.
Note a short arm is *flattered*, not penalised, because warmup amortises over
fewer tokens — so the public quant is no better than 25.35 and ours is clearly
ahead.

**Incidental but useful: a bracketed control spread of 1.6%.** The two `ours`
samples were separated by a pub run, making them a CTL-a / ... / CTL-b pair.
1.6% is under the 2% threshold, so wall clock may be usable on this six-card
cell for effects above ~3% — pending a proper back-to-back pair. **[M]**

**A number that caused a false alarm and must not be requoted as a codec result:
15.50 tok/s.** That is our PXQ4 *with `PXA_PROFILE=1`*, which syncs before and
after every node. It is instrumentation overhead, not codec speed. **[S/M]**

## 2. Where decode time actually goes

Profile: `/path/to/models/fn-profile.log`, taken on our PXQ4 file at `-c 2048`,
95 tokens. Op shares (op-computes=260000, total_gpu_us=7465342):

| bucket | share |
|---|---|
| MUL_MAT | 52.4% (54748 calls, avg 71.4 us) |
| MOE_FUSED_UP_GATE | 16.7% (4311 calls, avg 289 us) |
| elementwise (UNARY/ADD/MUL/SCALE/CONT/RMS_NORM/REPEAT) | 23.8% (~140k calls, avg 9-12 us) |

Named MUL_MAT buckets sum to only ~25.5%: qkv_mixed 7.4 / linear_attn_out 4.3 /
ffn_moe_logits 3.0 / result_output 2.8 / z 1.9 / shared_expert_gate 1.9 /
Qaux 1.4 / beta 1.3 / kq 0.8 / attn_block_out 0.7.

### PROFILER DISTORTION — mandatory caveat on every number above
`PXA_PROFILE=1` does a `cudaStreamSynchronize` **before and after every node**
(`ggml-cuda.cu:7641-7647`). The profiled run is 15.50 tok/s against ~28.8 clean,
i.e. **~1.79x inflation**, and its own per-op sum (83.9 ms/token) exceeds its own
wall clock. The cost is a **fixed per-node tax**, so it systematically inflates
the share of small/frequent kernels relative to large ones. **[S/C]**
Treat these shares as directional. **Never quote a profiled run as a speed result.**

## 3. THE FINDING: the unnamed 27% is the hyper-connection mixer

Two independent agent fleets converged on this from different directions.

- The `node_#` bucket is **22.96% of profiled GPU time**, 17,632 calls,
  **196.3 matmuls per decoded token** against 196 predicted from the graph
  builder — a 0.2% match. **[M/C]**
- 194 of the 196 are the two **un-`cb`'d** projections inside
  `build_qwen4exp_hc_mix` (`build_qwen4exp.cpp:40` and `:42`): `w_down`
  [10240,320] and `w_up` [320,10240]. `lo` is never named, and `cb` names the
  *sigmoid* of the `w_up` product rather than the product — which is exactly why
  they vanished into the anonymous bucket. Called **97x per token**
  (48 attn + 48 ffn + 1 head mixer). **[S]**
- The other 2 are the PLE key/value projections on layer 1
  (`build_qwen4exp.cpp:114-115`), also un-`cb`'d. **[S]**
- **It is ~60x off the memory roofline** — 675 MB/token of weight traffic is
  ~0.32 ms at aggregate bandwidth versus ~19.1 ms measured. It is
  launch/occupancy bound, not bandwidth bound. Each node pays two kernel
  launches plus a pool allocation, because `hc_down` and `hc_up` take different
  `src1` tensors and so cannot be merged by the `mul_mat_q` chaining loop. **[C]**
- **Mechanism (second fleet, independently):** with R = 10240 the `hc_*_up`
  matmuls exceed the `src0->ne[1] > 512` cap at `ggml-cuda.cu:3321` and fall to
  `ggml_cuda_op_mul_mat_cublas`: `to_fp16(src1)` + GemmEx `CUBLAS_COMPUTE_16F` +
  `to_fp32(dst)` — **three launches**. **[S]**

**Expected gain from a dedicated small-K/large-R f16 GEMV: 1.9-2.9 ms/token
(5-8%).** The de-biased budget for *both* hc matmuls together is only
6.2-7.2 ms/token, so earlier "4-8 ms" claims were too high. **[C]**

**Gate it on small-K/large-R. Do NOT raise the R cap** — the cap's documented
job (`ggml-cuda.cu:3282`) is to never intercept a real dense projection. **[S]**

Correctness risk is real: it changes numerics on the gate that scales the wide
residual at all 97 mix points (fp16 accumulate, tighter but not bit-exact).
Behaviour-gauntlet class, not sha-gateable.

## 4. Second target: 216 F32 GEMVs on a bare cuBLAS SGEMM

`ffn_gate_inp` [2560,512], `ffn_gate_inp_shexp` [2560], `hc_*_inject` [10240,4],
`ssm_beta` and `ssm_alpha` [2560,48] — **216 GEMVs/token costing ~7.93% of GPU
time while carrying under 0.3% of the model's weights.** **[M/C]**

A ready-made lever exists and is **default OFF**: `ggml_cuda_router_gemv_f32`
(`ggml-cuda.cu:3374-3424`). Its shape gate (src0 F32, `src1->ne[1]==1`,
`2 <= src0->ne[1] <= 4096`) admits all 216. **[S]**

**Two cautions before enabling.** The historical −3.1% verdict that keeps it off
was measured on **mode 1**, before mode 3 existed. And the mode-3 guard at
`ggml-cuda.cu:3412` checks only `ne00 % 4` and `nb[1] % 16` and **never the base
pointer**, while the kernel casts both operands to `float4*` — a
contiguous-but-offset `src1` view is a hard misaligned-address fault. **Fix the
guard before enabling.** Realistic gain 0.3-0.5 ms (0.8-1.4%). **[S/C]**

## 5. Dispatch facts worth not re-deriving

- **Nothing hot falls back to dequantize-then-GEMM.** The early return at
  `ggml-cuda.cu:3543` routes every mmvq-capable quantized GEMV to MMVQ on all
  six cards. Do not propose "fix the dequant fallback" — there isn't one. **[S]**
- **sm_70 tensor cores are used nowhere in this workload.** The only PXQ WMMA
  path is `cc==700` + prefill-only + PXQ-only + default OFF. **[S]**
- **PXQ-via-MMVQ cannot run on the four P100s**: mode 1 needs `cc >= 700`,
  mode 2 needs `cc >= 610`; P100 is cc 600. Any PXQ kernel lever helps at most
  the two V100s. **[S]**
- `hc_*_down` [10240,320] has ne1=320 and **passes** `ggml_cuda_small_gemv_f16`
  (`ggml-cuda.cu:3310-3330`, gates on F16 and `ne1 in [2,512]`, default ON) — so
  it runs on a hand-written half2/fp32-accum kernel today. **[S]**

## 6. DEAD LEVERS — do not respend

| lever | verdict |
|---|---|
| CUDA-graph capture / replay | **NEGATIVE, twice, captures verified firing.** P100 −3.9% on a 27B (replays 396/400, byte-identical); **−2.8% on Flash-Next itself** (26.43 → 25.68, 127 tokens both sides, replays=882). The `cc < CC_AMPERE` gate is a CONCLUSION, not an oversight. **[M]** |
| `-sm graph` decode | −17% **[M]** |
| graph-split reduction | 13 → 11 → 9 splits bought ~1% total. Splits are not the bottleneck. **[M]** |
| `PXQ4_MMV_SLICE_MAX` | null at serving level **[M]** |
| `PXA_PXQ_REDUCE_BLK`, `only_active_experts` | no-op; all experts device-resident **[S]** |
| `PXQ6_CANON_CMAX=4` | **NEGATIVE on Flash-Next.** The "driver never selects S > 4" note is scoped to a *122B PXQU48 4xP100* capture. Here `fn-splitmap.log` shows `DENSE_GATEUP dev0/1/4/5 FIRING (S=8)`; CMAX=4 caps that to S=4, 80 → 40 blocks on a 56-SM P100, 1.4 → 0.7 waves. In-source measurement for this node class is −8.6% when under-split. **[S]** |

### A claim of mine that was wrong — recorded so it is not resurrected
I claimed **"only ~2.3 of 6 cards are busy"**, implying large recoverable idle.
**The arithmetic was wrong.** `total_gpu_us` is a *host wall-clock accumulator*
over a serialized single-threaded node loop and cannot exceed its own run's wall
clock; I produced 2.3 by dividing one run's number by a **different** run's wall
clock (89.81 passes x 36.0 ms = 3.233 s, borrowed from an unprofiled,
different-quant run). **[S/C]**

Separately and correctly: the cards **do** serialize into a strict linear
pipeline — CUDA0 layers 0-3 → CUDA1 4-12 → CUDA2 13-21 → CUDA3 22-30 →
CUDA4 31-39 → CUDA5 40-47, each split consuming the previous split's `l_out-N`
— capping aggregate occupancy at 1/6 (measured 15.3%). **But that idle is a data
dependency for a single sequence and is NOT recoverable as single-stream decode
latency.** Pipeline parallelism is additionally disabled by two independent
conditions (`llama.cpp:8791-8809`): the `-ot` tensor override, and the arch being
recurrent/hybrid. Both hold here. **[S]**

## 7. THE MEASUREMENT PROBLEM — read before quoting any speed number

- Run-to-run variance is **+/-4% on an identical binary** (25.80 and 27.88 tok/s
  observed on the same build). **[M]**
- **Every lever identified is worth under 8%.** Two wall-clock samples therefore
  resolve nothing. A tok/s A/B here returns noise that looks like a result.
- A bracketed CTL-a / ARM / CTL-b at fixed fill, n=3, reached a **0.9% control
  spread** — but that was a *homogeneous 4xP100* cell. **The six-card
  heterogeneous control spread has never been measured.** Establish it with a
  CTL-a/CTL-b pair and no arm between them before trusting any bracket. **[S]**
- Instruments, in order of preference:
  1. **nsys per-kernel capture** — true per-kernel time and inter-kernel gaps, no
     sync tax. Prices one kernel family directly instead of hunting a 2%
     whole-model delta. NOT installed as of this session; being added as
     `pxa-sm60-dev:nsys`.
  2. **per-device `PXA_GRAPH` sums** — ~0.1 ms sensitivity, the right instrument
     for placement levers such as a `-ts` rebalance.
  3. **`PXA_PROFILE` name buckets** — per-node attribution to ~1 us.
     **Attribution only. Never a speed result.**
- Two arms of one sweep this session stopped at **53 and 57 tokens** instead of
  127. Their tok/s is not comparable to a full-length control and was discarded.
  **Always state token count beside every number and void any mismatched arm.**

## 8. Speed levers, ranked, with honest magnitudes

| # | lever | gain | effort | risk |
|---|---|---|---|---|
| 1 | dedicated small-K/large-R f16 GEMV for the 97 `hc_*_up` matmuls | **1.9-2.9 ms (5-8%)** | medium: one kernel + dispatch predicate | medium: numerics on the residual gate |
| 2 | **`-ts` rebalance, 4 layers P100 → V100** | **~0.98 ms (2.7%)** | trivial, one flag | **none** — placement only, loud OOM is the only failure |
| 3 | `PXA_ROUTER_FUSE=3` | 0.3-0.5 ms (0.8-1.4%) | env var, no rebuild | medium — fix the base-pointer guard first |
| 4 | `CANON_V2=1` + `PXQ_GU_MINBLK=4` (**not** CMAX=4) | ~1.0% | rebuild + full re-baseline | high process risk: CANON_V2 changes the canonical fp32 fold, invalidating every control on the box |

**Stacked, assuming additivity: 3.3-4.8 ms. 36 ms → 31.2-32.7 ms
(30.6-32.1 tok/s). The 28-31 ms target is NOT reachable with anything currently
identified. Plan for 32 ms.** **[C]**

Lever 2's headroom is 2 layers per V100 (3402 MiB post-load vs 1341 MiB/layer),
so 4 layers is the whole prize. Measure it with per-device `PXA_GRAPH` sums
against the 58,589 us baseline, **never** a wall-clock A/B. **[C]**

## 9. PXQU on 4x P100 — budget facts

**The constraint is expert weight, not context.** Only 12 of 48 layers hold KV
(`full_attention_interval=4`); the other 36 are linear-attention with
context-independent recurrent state (0.110 GiB total). KV is 12 x 2 kv heads x
256 head_dim x 2 (K,V) x 2 B = **24,576 B/token**, cross-checked against the log
(`KV self size = 48.00 MiB` at c=2048). 150,016 tokens = **3.4336 GiB**. **[M/C]**

Meanwhile **120.796 B of the 125.1 B on-GPU params (96.1%) are the 144 routed
expert tensors**, and a straight PXQ4 of this model is 114.67 GiB. **[C]**

### The per-card trap that invalidated an earlier ledger of mine
I budgeted in the **aggregate** — total GiB across four cards minus total
overhead — which hid a **per-card** binding constraint. The logits buffer is
`n_vocab * n_ubatch * 4` (`llama.cpp:3782-3784`, `worst_case_tokens` defaults to
0 at `:7746`) and lands **entirely on whichever card holds the output head**,
which **`-ts` cannot move**. At ub2048 that is 248320 x 2048 x 4 =
**1.8945 GiB on one card**. Confirmed in the log: CUDA5 compute buffer
**490.00 MiB** against 207.62-208.99 on the other five, and
248320 x 512 x 4 = 485.0 MiB. **[S/M/C]**

**Compute buffers are NOT uniform across cards, and the whole error falls on the
one card with no room for it.**

| `-ub` | best layer split | max `-c` | fits 150,016? |
|---|---|---|---|
| 2048 | 13,13,12,10 | 76,143 | **no** |
| 1024 | 13,13,13,9 | 153,035 | technically — 0.023 GiB spare |
| 512 | 12,12,12,12 | 223,540 | yes, comfortably |

`-b 2048 -ub N` keeps prefill batching at 2048 and only chunks the micro-batch.

### Recipe correction: keep `hc_*` at F16
An earlier draft took `hc_*_down/up` F16 → q8_0 for 0.549 GiB. **Do not.** That
evicts 96 calls/token and 629 MB from `ggml_cuda_small_gemv_f16` onto `mul_mat`
plus a separate q8_1 quantize launch, on **emulated dp4a** on P100 — a pure
speed regression, and it perturbs the gate scaling the wide residual at 96 mix
points across 48 layers. **[S]**

### Other recipe facts
- `output.weight` → q6_K (497.31 MiB): ne0=2560, 2560 % 256 == 0, legal.
- `per_layer_token_embd` stays q8_0 CPU-pinned: block-32 and GET_ROWS-legal.
  Cannot be PXQ4/MXFP4 (panel codecs gather nonsense) nor Q4_K (ne0 % 256).
- `ssm_conv1d`/`ple_conv1d` are [4,10240]: ne0=4, so `quantize &= (ne[0] % 32 == 0)`
  copies them as f32. Naming a codec there is inert at best — **every block-32
  codec asserts and ABORTS**. Leave them ruleless by design.
- `ffn_gate_inp` (512-way router, 240 MiB f32) is excluded by name and **cannot**
  be retiered from a `.tiers` file at all.

## 10. Owner decisions recorded

- **No imatrix.** Owner: it is a net negative on our quants. Tier assignment
  therefore stays on the depth x kind sensitivity proxy. Task #40 is closed by
  this decision, not by being done.
- **`-ub` must be at least 1024.** The ub512 configuration is rejected even
  though it has the most headroom.
- Push only to `private`, never `origin`. No model identifier in any artifact.

## 11. Open / next

1. `pxa-sm60-dev:nsys` image, then the prerequisite-0 capture: an nsys decode
   capture on the current binary plus a CTL-a/CTL-b control pair, both at
   `-m <PXQ4> -ts 7,16,16,16,16,16 -c 2048 -fa off`.
   `PXA_GEMV_DBG=1` on the same command line gives the authoritative list of every
   decode GEMV landing on the cuBLAS fallback — it has **zero occurrences in the
   current profile log, so nobody has ever enumerated them**. **[S]**
2. Bank the `-ts` rebalance (per-device `PXA_GRAPH` sums, not wall clock).
3. Re-solve the PXQU tier map for `-ub 1024` with real headroom at >=150k KV.
4. `hc_*_up` GEMV kernel, priced by nsys before it is written.
5. Fix the `PXA_ROUTER_FUSE=3` base-pointer guard, then measure mode 3 isolated.
