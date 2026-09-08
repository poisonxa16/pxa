# 10 — Decode kernel speed: diagnosis, K-chunk-split fix, measurements

Date: 2026-08-18. Hardware: DGX-1V, V100-SXM2-32GB (sm_70, 80 SMs, ~900 GB/s HBM2).
Artifact for serving numbers: `Qwen3.8-27B-PXQ4-vllm-p2a-nf` (norm-fixed, policy p2a,
4.018 GiB/GPU/token at TP=4). Kernel change: commit `9916ea69`.

## The question

TP=4 serving decoded at ~44 tok/s (~200 GiB/s of weight traffic per GPU, ~22% of peak)
vs AWQ/TurboMind's ~80 tok/s at ~31% of peak. Candidate explanations: (a) the unfused
dequant+GEMM path fires during decode; (b) the mmv kernel converts bytes to work badly.

## Diagnosis 1 — dispatch counting kills hypothesis (a)

`site-instr` (instrumented `linear.py`) serving TP=1, enforce-eager, single stream,
after warmup + 3x(1+512)-token runs + one 128-token run:

    mmv=398080  gemm=1920
    m_hist: M=1 x397840   (decode  -> mmv, 100%)
            M=8 x240      (warmup  -> mmv)
            M=11 x1440    (prefill -> dequant+GEMM)
            M=256/2048 x240 each (profile runs -> dequant+GEMM)

Decode-time linear calls are 100% mmv (240 PXQ4 modules per rank = 60 layers x 4).
The 340 MiB dequant workspace is touched only at prefill. Hypothesis (a) is dead.

## Diagnosis 2 — the monolithic mmv is grid-starved (hypothesis (b) confirmed)

`k_pxq4_mmv` launches one 256-thread block per (panel, token). At M=1 on this model's
TP=4 shapes that is 64-136 blocks for 80 SMs: <= 2 blocks/SM, 256-512 threads/SM of a
2048-thread budget, 12.5-25% occupancy — far too few outstanding loads to hide HBM2
latency. Weight-byte throughput, measured with CUDA events (300 iters, M=1):

    shape            slab bytes   mono GB/s   cuBLAS fp16 GEMV GB/s (same card)
    tp4 gate_up      23.7 MB      276         690
    tp4 down         11.8 MB      180         663
    tp4 o_proj        4.2 MB      200         453
    tp4 qkvz         11.1 MB      145         631
    tp2 gate_up      47.4 MB      460         567
    tp2 down         23.7 MB      218         714
    tp2 o_proj        8.4 MB      175         670
    tp2 qkvz         22.3 MB      291         704
    tp1 ffn_down     47.4 MB      253         732

The old kernel runs at 16-32% of peak while dense cuBLAS GEMV reaches 50-81% — so the
4x byte advantage of PXQ4 bought only ~1.2x in time (TP4 layer set: 0.249 ms PXQ4 vs
0.298 ms fp16). Note the shape trend: 272 panels (tp2 gate_up) already reaches 460 GB/s;
64-80 panels sit at 145-200. It is a parallelism problem, not a coalescing problem —
all weight reads are coalesced by construction (64 consecutive sub-scale bytes/warp-pair,
16 B/thread contiguous code rows = 512 B/warp, 128 B contiguous anchor header).
Consistent with this, cutting bandwidth 11.4% (p2a vs p1 at TP=2) gave zero speedup:
decode is latency-bound, so only more parallelism (or fewer launches) helps.

## The fix — K-chunk-split mmv (`k_pxq4_mmv_part` + `k_pxq4_mmv_reduce`)

The canonical-chunk fold already partitions K into nfix (8-16) per-lane partial sums.
The split kernel gives each chunk its own block (grid = panels x nfix x M), stores each
per-lane chunk partial t_c unsummed (fp32, one coalesced 1024 B store per block), and a
64-thread reduce kernel replays the monolithic kernel's exact fold: chunk order, then
kseg order, one final `__float2half_rn`. No addition is reassociated and no rounding
point moves, so the result is bit-identical — not approximately, but by construction,
and gated:

  * hostsim (REAL kernel source on CPU): 10/10 configs bit-exact (vecx 0/1, nfix 8/16,
    M 1-8) — `test_pxq4_mmv_split.py`, now run by `build_hostsim.sh`;
  * GPU: 27/27 shape/M combos bit-exact vs `mmv_out_mono`;
  * the 17 pre-existing hostsim gates still pass.

Cost: an fp32 partials tensor, M * panels * nfix * 256 floats (1.3-2.2 MB at M=1 —
mostly L2-resident between the two kernels; worst case +22% traffic on tp4 down, in
exchange for 8-16x the blocks). Partials come from `at::empty`, the same
graph-pool-backed allocator as the `out = torch.empty(...)` the caller already does
per `apply()`, so CUDA-graph capture safety is unchanged.

Measured (same protocol), mono -> split:

    tp4 gate_up   0.086 -> 0.056 ms   276 -> 422 GB/s
    tp4 down      0.066 -> 0.034 ms   180 -> 346 GB/s
    tp4 o_proj    0.021 -> 0.032 ms   REGRESSION -> stays mono (threshold below)
                                       [DOES NOT REPRODUCE -- see the v5 section; split
                                        wins 1.7-2.2x on this shape in every L2 regime]
    tp4 qkvz      0.077 -> 0.034 ms   145 -> 329 GB/s
    tp2 gate_up   0.103 -> 0.104 ms   tie (was already 272 panels)
    tp2 down      0.109 -> 0.051 ms   218 -> 460 GB/s
    tp2 o_proj    0.048 -> 0.032 ms   175 -> 263 GB/s
    tp2 qkvz      0.077 -> 0.052 ms   291 -> 428 GB/s
    tp1 ffn_down  0.187 -> 0.085 ms   253 -> 555 GB/s

    TP4 per-layer linear time: 0.249 ms -> 0.145 ms (best-of dispatch), 1.72x.

## The dispatch threshold, and why 8 MB  [SUPERSEDED -- see the v5 section below]

**This section's conclusion is wrong and the measurement it rests on does not reproduce.**
The 8 MB byte threshold was replaced in v5 by an occupancy rule (`panels*M <=
2*multiProcessorCount`), because what decides the winner is whether the monolithic grid can
fill the SMs, not how many bytes the tensor holds. Re-measured, TP4 o_proj on the split
path beats mono by 1.7-2.2x in BOTH L2 regimes; there is no regime in which split costs
32 us. The byte rule was costing ~0.79 ms per decode token. Kept below as the record of
what was believed.


`mmv_out` takes the split path when `nfix > 1 && slab_bytes >= PXQ4_MMV_SPLIT_MIN_BYTES`
(default 8 MB, env-overridable). Mechanism: the split pays a fixed floor — two launches
plus the partials round-trip — of roughly 15-25 us; the monolithic kernel, even starved,
moves B bytes at >= ~200 GB/s when panels >= 64. Crossover where
B/450GB/s + floor < B/200GB/s gives B ~ 7 MB, matching the two measured sides of the
boundary (tp4 o_proj 4.2 MB: mono 21 us vs split 32 us; tp2 o_proj 8.4 MB: mono 48 us
vs split 32 us). A panels-aware predicate was considered and rejected: above ~160
panels mono and split converge anyway (tp2 gate_up ties), so bytes capture the only
regime that matters. Known conservatism: the floor amortises over M, so at M=8 the
split would likely win even at 4 MB; M is not in the predicate because decode is M=1
and the difference is ~10 us on the smallest tensor.

## End-to-end

  * TP=1, enforce-eager (NOT the shipping config; dominated by per-op launch/Python
    overhead — 240 pxq4 calls + attention/mamba ops per token, no graphs):
    old 5.23/5.23/5.28 -> new 5.34 tok/s. Confirms eager TP1 is overhead-bound, not
    kernel-bound; useless as a kernel vehicle, recorded for completeness.
  * TP=4, CUDA graphs, p2a-nf (the shipping config), identical flags both arms:
      OLD kernel (v1): 43.40 / 41.57 / 41.70 / 41.36 / 43.31 tok/s
      NEW kernel (v3): 53.38 / 53.61 / 58.23 / 54.46 / 60.01 (mean 55.9, +32% over the 42.3 mean) tok/s
    Predicted from the per-layer delta was mid-50s (0.104 ms x 60 layers = 6.2 ms
    off a ~23 ms step).
  * Capture cost: v1 ~2:41 total; v2 (per-call in-capture at::empty) 209 s/it
    PIECEWISE + 252 s/it FULL and blew the startup deadline; v3 (persistent arena,
    zero in-capture allocations) 2:36 total PIECEWISE + 0:02 total FULL.
  * Eager A/B is not a kernel vehicle on this stack and was retired: TP=4 eager
    v1 measured 4.49-5.26 tok/s (~8x overhead-bound; kernel time ~3% of step).

## Remaining headroom, and what is NOT worth doing

  * SUPERSEDED BY v5 -- see "Remaining headroom, honestly costed" at the end of this
    file. Kept for the record, with the two things it got wrong marked.
    The split kernel sits at 330-460 GB/s on TP4 shapes vs cuBLAS GEMV's 630-730.
    Occupancy is no longer grid-limited (1280-2176 blocks). The next suspects are the
    per-byte ALU cost of MODE_TAB (two smem table lookups + shifts per weight pair;
    ~~random-nibble smem reads carry bank conflicts~~ WRONG -- measured 0.77-0.87% of
    shared wavefronts, not a factor) and the fixed two-launch floor.
    Candidate next steps, in order of expected value: ~~warp-shuffle table lookup~~
    RETIRED, measured a wash at 48 registers instead of 32; fusing the reduce into the
    part kernel via a cooperative last-block-reduces scheme (one launch) DONE in v5,
    worth 5 points of per-layer time; and only then ILP restructuring (changes the fold
    -> re-baselining event) -- still the only large item left.
  * Bandwidth-side work (smaller formats) is currently worthless for speed: measured
    11.4% fewer bytes -> 0% faster. Fix parallelism first; bytes may matter again
    after the kernel reaches ~600 GB/s.
  * The fp16-vs-PXQ4 serving control adds nothing the GB/s table does not already say.

## site-v2 promotion gate

`site-v2` (op version 2) becomes the default `site/` when: (1) a CUDA-graphs TP=4 boot
serves sane text (proves capture-time `at::empty` in the op), and (2) the A/B shows no
regression. Bit-exactness is already proven, so no quality gate is needed — outputs are
byte-identical by construction. Until promotion, the old lib remains the default and v2
is opt-in via PYTHONPATH.

---

# v5 — chunk-major grid, single-launch fused split, occupancy dispatch

Date: 2026-08-18. Same card, same artifact, same flags. v5 is the integration of three
independently-gated changes plus one measured retirement. Every number below was measured
on card 4 (parity, microbench, ncu) or on cards 4-7 (serving); nothing here is projected
unless the line says so.

## What v5 is

1. **Chunk-major grid order.** `k_pxq4_mmv_part` takes `grid = (nfix, panels, M)` instead
   of `(panels, nfix, M)`, so the canonical chunk is the fastest-varying dimension and a
   resident wave covers all chunks of a few CONSECUTIVE panels rather than one chunk of
   every panel. Pure addressing change: `part[]` keeps the identical `(iy, p, c, tid)`
   layout, every lane visits the same `kb` in the same order, `k_pxq4_mmv_reduce` is
   untouched. `nfix` and `panels` now arrive as EXPLICIT ARGUMENTS rather than off
   `gridDim`, which also closes a latent hazard: the pre-swap kernel read `nfix` from
   `gridDim.y` while the reduce took it as an argument and nothing asserted they agreed.
   `pxq4_launch_mmv_split_f16` now passes one value to both and asserts it equals
   `pxq4_canon_nfix(kslabs, CMAX)`.

2. **Single-launch fused split** (`k_pxq4_mmv_fused`). The reduce runs in whichever block
   of a `(panel, token)` arrives last, so the device drain + refill between the two
   kernels disappears and ~240 kernel nodes leave the decode graph. The atomic is an
   ARRIVAL COUNTER, never an accumulator — no floating-point value is ever atomically
   combined, so the expression tree is untouched and the winning block replays the
   reduce's fold verbatim. No block ever spins: every block increments and exits, only
   the one observing `old == nfix-1` continues, so deadlock is impossible regardless of
   scheduling.

3. **Occupancy-based split/mono dispatch.** `panels*M <= 2*multiProcessorCount` replaces
   the 8 MB byte threshold. What decides the winner is whether the MONOLITHIC grid —
   `panels*M` blocks — can fill the SMs; it does not depend on tensor size. The byte
   threshold got the out_proj/o_proj class wrong: 64 of the 240 PXQ4 modules per token
   sat just under 8 MB and shipped on mono at 12.5% occupancy.

4. **The MODE_TAB table-delivery angle is retired**, and `stage_tabs` now takes
   `float (&)[16]` references with a `static_assert`, so widening the table is a compile
   error rather than a silent regression. Pair-staging measured 0.73-0.87x, warp-shuffle
   0.99-1.03x (a wash, at 48 registers instead of 32), and a perfect-1-instruction ALU
   ceiling probe only 1.06-1.11x. There is nothing here worth taking.

## The stale threshold comment was wrong in both directions

The source recorded "TP4 o_proj, 4.2 MB, mono 21 us vs split 32 us". That does not
reproduce in any L2 regime. o_proj's slab is 3.98 MB against a 6 MB L2, so a single-copy
benchmark reads ~30% fast and flatters mono. Re-measured:

    regime                          mono      v3 split   v5 fused
    L2-warm (one copy)              17.7 us   10.4 us    —
    L2-defeated (48 MB rotating)    24.5 us   13.1 us    11.3 us

Split wins by 1.7-2.2x in BOTH regimes; there is no regime in which split costs 32 us.
The 8 MB threshold was costing ~0.79 ms per decode token across the 64 out_proj/o_proj
modules.

## Per-shape, M=1, L2-defeated (48 MB rotating weight set, 300 iters, 5 reps, min)

Card 4, no co-tenant. Two independent whole-program runs agree to <=0.5% on every cell.
`v3 split` is reconstructed in-process as the pre-swap part kernel + the shipping reduce,
and asserted bit-identical to `mmv_out_mono` before any timing is believed.

    shape          slabMB pan nfix | mono ms  GB/s | v3split ms GB/s | v5 grid ms GB/s | v5 fused ms GB/s | fused/v3
    tp4_gate_up     22.58 136   16 | 0.07671  309  | 0.05588   424   | 0.05138   461   | 0.04819   491    | 1.160
    tp4_down        11.29  80   16 | 0.05825  203  | 0.03172   373   | 0.03134   378   | 0.02892   409    | 1.097
    tp4_qkvz        10.62  64   16 | 0.06810  164  | 0.03112   358   | 0.02882   387   | 0.02723   409    | 1.142
    tp4_o_proj       3.98  80    8 | 0.02445  171  | 0.01310   319   | 0.01204   347   | 0.01125   371    | 1.164
    tp2_gate_up     45.16 272   16 | 0.09729  487  | 0.10195   464   | 0.09304   509   | 0.08760   541    | 1.164
    tp2_down        22.58  80   16 | 0.10663  222  | 0.05144   460   | 0.05445   435   | 0.05224   453    | 0.985
    tp2_qkvz        21.25 128   16 | 0.07608  293  | 0.05252   424   | 0.04896   455   | 0.04592   485    | 1.144
    tp2_o_proj       7.97  80   16 | 0.04707  178  | 0.02571   325   | 0.02376   352   | 0.02314   361    | 1.111
    tp1_gate_up     90.31 544   16 | 0.15137  626  | 0.19957   475   | 0.17698   535   | 0.16770   565    | 1.190
    tp1_down        45.16  80   16 | 0.18776  252  | 0.08234   575   | 0.08345   567   | 0.08204   577    | 1.004
    tp1_qkvz        42.50 256   16 | 0.09535  467  | 0.09640   462   | 0.08833   505   | 0.08312   536    | 1.160
    tp1_o_proj      15.94  80   16 | 0.06989  239  | 0.03895   429   | 0.03813   438   | 0.03557   470    | 1.095

TP4 per-layer linear time at M=1, each arm using the dispatch it actually ships with
(v3: split for gate_up/down/qkvz, mono for o_proj; v5: fused split for all four):

    v3  0.14317 ms   ->   v5  0.11559 ms     -19.3%, 1.239x

Decomposition, each step added to the one above it:

    grid order alone        0.14317 -> 0.13599    -5.0 points
    + fusion                0.13599 -> 0.12879    -5.0 points
    + occupancy dispatch    0.12879 -> 0.11559    -9.2 points

The dispatch change is the largest single contributor, and it is the one that costs
nothing in the kernel at all.

## Measured negatives — record these so nobody re-derives them

  * **rb2 register blocking does NOT survive the combination.** Two output rows per lane
    plus half2 activation staging, applied to the FUSED kernel: bit-exact (120/120 device
    combos), 40 registers, 0 spill — and SLOWER on the shipping config. TP4 M=1
    fused+rb2/fused: gate_up 0.999, down 0.976, qkvz 0.944, o_proj 0.978; per-layer
    0.11559 -> 0.11815 ms, i.e. 2.2% worse. Measured standalone against the v3 two-launch
    part kernel rb2 was worth +2.1%, and that result reproduces — but it was an
    instruction-latency saving bought with occupancy (32 -> 40 registers, warps active
    84.6% -> 62.9%). Once the chunk-major order fixed delivery and the fusion removed the
    launch gap, the occupancy is worth more than the instructions. NOT SHIPPED.
  * **tp2_down regresses on the grid swap.** panels=80, kslabs=272, nfix=16: v5 grid is
    0.945x of v3 and the fusion only claws it back to 0.985x, i.e. ~1.5% slower than v3.
    Reproduced across runs. The pattern across panels=80 shapes is non-monotone in kslabs
    (48 -> 1.09, 96 -> 1.08, 136 -> 1.00, 192 -> 1.02, 272 -> 0.95), so this is a
    measurement, not a mechanism. TP4 is the shipping layout and is positive on all four
    of its shapes; a TP2 deployment should keep an eye on this one shape.
  * **The monolithic kernel wins at M >= 2 on the wide shapes**, which is exactly what the
    new dispatch rule now exploits. The rule's decision matches the measured winner on
    every TP4 (shape, M) point at M in {1, 2, 4}.

## Exactness evidence for v5

Nothing about the fold order changed, and it is gated four ways:

  * hostsim (compiles the REAL kernel source for CPU): 17/17 legacy gates PASS; 10/10
    split AND fused configs report `split=True fused=True part_fp32=True` — the fp32
    `part[]` buffer is compared as uint32, which is ~400x more sensitive than the fp16
    output, and every buffer is re-poisoned before EVERY call.
  * cross-`.so` differential against a pristine v3 build: 22/22 cases BIT-EXACT for the
    split path and 22/22 for the fused path (fp32 partials AND fp16 out, poisoned).
  * NEGATIVE CONTROL: rebuilding the identical v5 source with `-march=x86-64-v3` lets g++
    contract the inner accumulation into FMAs; the same gate then reports NOT BIT-EXACT on
    22/22. The gate has power; it is not passing by construction.
  * SASS census, nvcc 12.8.93 `-O3 -arch=sm_70 --expt-relaxed-constexpr` (shipping flags,
    fmad default):

        kernel                 v3 base                  v5
        k_pxq4_mmv        n=360/368 17/19/22      n=360/368 17/19/22   IDENTICAL
        k_pxq4_mmv_part   n=328     17/19/17      n=336     17/19/17   +8 integer prologue
        k_pxq4_mmv_reduce n=128      0/ 0/24      n=128      0/ 0/24   IDENTICAL
        k_pxq4_mmv_fused          —               n=512     17/19/50, ATOM=1

    FFMA/FMUL are untouched in every kernel. The `k_pxq4_mmv_part` growth is entirely
    integer prologue from the explicit arguments — the dot32 loop body is unchanged. The
    fused kernel's FADD 17 -> 50 is the tail: ptxas unrolls the runtime `for cc < nfix`
    chunk fold, which keeps the strictly left-associated chain intact. Exactly one ATOM,
    and it is the arrival counter.
  * device, card 4: **120/120 shape/M/vecx combos bit-exact** (12 real TP4/TP2/TP1 shapes
    x M in {1,2,3,4,8} x vecx in {1,0}) — a superset of the standing 27. Each combo
    compares the fp32 `part[]` of the v5 part kernel against the v3 part kernel as
    uint32, and the fp16 output of the fused path against BOTH `mmv_out_mono` and the
    two-launch split, with every buffer poisoned before every call and the mono output
    asserted non-poison so a kernel that writes nothing cannot pass.
  * device barrier stress — the hostsim runs blocks sequentially, so it is structurally
    incapable of observing the race: 8 groups x 400 launches = **3,200 fused launches**,
    `part[]` memset to 0xFF and `out` to a sentinel before every one: 0 mismatching
    launches, counter read back all-zero every time.

`ptxas -v`: `k_pxq4_mmv_fused` uses **32 registers, 0 spill**, the same as
`k_pxq4_mmv_part`, so achieved occupancy is unchanged by the fusion. Static smem
1152 -> 1168 B for the `last` flag; `pxq4_mmv_supported`'s budget was raised by 16 B to
match, so no shape can be admitted that the fused path cannot run.

## Two hazards that are now correctness dependencies

Both are documented at the arena in `pxq4_kernel_torch.cpp`, not assumed:

  * **Exactly one PXQ4 mmv in flight per device.** Not a new constraint — the shared
    partials arena already required it — but with the barrier it is a CORRECTNESS
    dependency, not only a scratch-aliasing one.
  * **A launch torn down mid-flight leaves a counter non-zero**, after which no block
    observes `old == nfix-1`, `out[]` is never written, and the caller silently consumes
    stale fp16. The failure is SILENT. Any error path that abandons a launch must drop the
    arena tensor so the next call reallocates and re-zeroes.

`k_pxq4_mmv_part` / `k_pxq4_mmv_reduce` are kept compiled and reachable through the
`mmv_out_split2` op precisely so the device gate can keep asserting
fused == split == mono. Do not delete them in the same commit that lands the fusion.

## Boot and capture — the startup gate

Checked because a capture regression once blew the startup deadline (the v2 per-call
in-capture `at::empty` measured 209 s/it PIECEWISE). v5 adds a second persistent arena, so
this is a real risk, not a formality.

    arm                     boot->healthy   graph capture   FULL decode capture
    v5 (cold compile cache)      896 s          463 s            4:12
    v3 (warm cache)              443 s          138 s            0:02
    v5 (warm cache)              403 s          136 s            0:01

**No capture regression.** The first v5 row is a measurement artifact and I nearly reported
it as a regression: v5 was the FIRST arm to boot into a fresh `HOME`, so it paid for the
whole cold torch.compile/inductor cache, and v3 then inherited the warm cache. Re-run with
the cache warm for both, v5 captures in 136 s against v3's 138 s and boots to healthy 40 s
FASTER. The A/B has to be run in both orders, or with per-arm caches, or this artifact will
be read as a kernel regression — it looks exactly like one, right down to reproducing v2's
~207 s/it PIECEWISE signature.

The counter arena is allocated and zeroed exactly once, outside capture, guarded by the
same `cudaStreamIsCapturing` check as the partials arena, and every completed launch
rearms its own slots — so steady state and in-capture are allocation-free and memset-free.
`Graph capturing finished in 136 secs, took 0.12 GiB` vs v3's `0.11 GiB`.

## End-to-end TP=4, CUDA graphs — and why it cannot settle a 10% question on this box

Config exactly as banked: `Qwen3.8-27B-PXQ4-vllm-p2a-nf`, `--quantization pxq4
--attention-backend FLASH_ATTN_V100 --tensor-parallel-size 4 --dtype float16
--gpu-memory-utilization 0.85 --max-model-len 32768 --no-enable-prefix-caching
--trust-remote-code`, no `enforce-eager`, no `PYTORCH_ALLOC_CONF`. Cards 4-7. Decode
measured as 512 `ignore_eos` tokens minus a 1-token prefill call, the same script that
produced the banked v1 and v3 numbers.

### The vehicle that does resolve it: a 240-module CUDA-graph token chain

`src/device_gates/pxq4_v5_graph.cu` builds one decode token's worth of PXQ4 linear work —
240 modules per rank at TP=4, 3.07 GB of DISTINCT weight tensors, ONE shared partials arena
and ONE shared counter arena, exactly as the engine has — captures it in a CUDA graph and
replays it. Two independent runs, agreeing to 0.01%:

    policy                                     nodes   eager ms   graph ms   g/e     per-layer
    v3 SHIPPING (ship grid + 8 MB byte rule)     416     8.6194     8.5323   0.990   0.14220 ms
    + chunk-major grid only                      416     8.3246     8.2359   0.989   0.13727 ms
    + grid + fusion + occupancy dispatch (v5)    240     7.1030     6.9651   0.981   0.11608 ms

    v3 -> v5 in-graph: 8.5512 -> 6.9648 ms/token, -18.55%, 1.228x. 1.586 ms/token removed.

This is the number to trust for "what the kernel change is worth in the engine". It agrees
with the per-shape microbench per-layer figure (0.14317 -> 0.11559 ms, -19.3%) to within
0.8%, which is the cross-check that makes both credible. The fusion removes 176 of the 416
graph nodes. `graph/eager` stays below 1.0 for every policy, so graph replay does not eat
the win — it slightly increases it (0.990 -> 0.981).

### Served tok/s — a null result, reported as one

Two rounds, v5 and v3 booted back to back within each round, 6 then 8 samples per arm:

    round 1   v5  53.63 53.71 53.39 55.08 54.61 66.98          mean 56.23  sd 4.84
              v3  51.68 52.43 60.03 59.04 53.78 37.57          mean 52.42  sd 7.36   (v5 +7.3%)
    round 2   v5  51.83 51.07 49.55 50.46 56.42 43.99 54.93 46.39   mean 50.58 sd 3.82
              v3  63.34 47.08 55.90 54.50 55.99 58.22 58.39 52.31   mean 55.72 sd 4.46   (v5 -9.2%)
    pooled    v5  n=14 mean 53.00 sd 5.12 (44.0-67.0)
              v3  n=14 mean 54.30 sd 6.10 (37.6-63.3)
              difference -1.30 tok/s, SE 2.21, Welch t = -0.59 -> NOT SIGNIFICANT
              95% CI on the difference: -5.63 .. +3.03 tok/s, i.e. -10.4% .. +5.6%

**The two rounds disagree in sign.** Per-sample scatter is 8-14% CV WITHIN a single boot
(63.34 then 47.08 tok/s from the same engine, consecutive requests), which is larger than
the effect being measured. This experiment cannot resolve a 10% change; it can only say the
change is somewhere between -10% and +6%. Both arms serve identical, sane text.

That is not a contradiction of the kernel measurements — it is a statement about this
vehicle. The banked v1 -> v3 comparison (+32%) was big enough to survive this noise; v3 ->
v5 is not. Anyone who wants a served number for a change of this size needs many more
samples, a quiet box, or an in-process A/B that does not pay for a fresh 7-minute boot per
arm. The honest summary is: **-18.6% of PXQ4 linear time, measured in-graph and twice
cross-checked; no detectable change in served tok/s at n=14 per arm.**

### ncu on the combined kernel — the new limiter

tp4 gate_up, `--clock-control none`, 4-6 profiled launches each, v3 part kernel vs the v5
fused kernel:

    metric                                     v3 k_part      v5 k_pxq4_mmv_fused
    sm__warps_active                           84.0-84.8%     91.2-91.5%
    dram__throughput (% of peak)               59.8-60.8%     58.0-59.4%
    smsp__issue_active                         53.7-55.0%     56.1-56.4%
    stall long_scoreboard (per issue-active)   16.1-17.4      12.3-12.5
    stall barrier                               2.2-2.3        6.2-6.4
    stall short_scoreboard                      1.4            2.2-2.3
    stall membar                                  -            0.07-0.08
    shared bank conflicts                      17.4-17.6 k    14.0-14.6 k
    thread_inst_executed                       321,490,944    334,251,552  (+4.0%)
    dram bytes read                            23.73 MB       23.73 MB     (unchanged)

**The new limiter is still global-memory latency, but occupancy is now genuinely spent.**
Warps active is 91.5% — there is no meaningful occupancy left to buy. `long_scoreboard` is
still the dominant stall by 2x over anything else, at 12.3 instructions per issue-active,
and DRAM sits at 58-59% of peak against cuBLAS fp16 GEMV's 70-81% on the same card. That
combination — nearly full occupancy, half the memory pipe, and a memory-latency stall on
top — says the kernel does not have enough INDEPENDENT loads in flight per warp, not that
it has too few warps. The remaining gap is memory-level parallelism inside the dot loop,
which is what an ILP restructuring would buy, and that is precisely the change that breaks
bit-exactness.

Two costs of the fusion are visible and worth stating: `barrier` stalls tripled (2.2 ->
6.3), which is the arrival barrier plus its two `__syncthreads`, and instructions rose 4.0%
for the fold tail. Both are already paid for in the wall-clock numbers. `membar` is 0.07,
so `atom.release.gpu` was the right call over a separate `__threadfence()`.

The lower DRAM percentage for the fused kernel is an artifact of the metric, not a
regression: it is per-ELAPSED, and the fused kernel's elapsed time now includes the reduce
tail, during which the partials come from L2 and DRAM is quiet. Bytes read are identical to
the byte.

## Remaining headroom, honestly costed

  * **Memory-level parallelism in the dot loop** is the only large item left, and it is the
    expensive one. The 58-59%-of-peak / 91.5%-occupancy / long-scoreboard-dominated
    signature says more independent loads per warp, which means restructuring the kb loop
    — and that CHANGES THE FOLD ORDER. It is a re-baselining event: it would need its own
    quality gate, not just a bit-exactness gate, and by the exactness contract it must ship
    as an explicit opt-in if it ships at all. Ceiling if it worked perfectly: 58% -> ~75%
    of peak would be worth roughly another 20% of linear time, ~1.4 ms/token.
  * **The M-aware dispatch is now nearly exhausted.** The occupancy rule already matches the
    measured winner on every TP4 (shape, M) point at M in {1,2,4}. The one known gap is
    tp8_qkvz at M=8 (panels*M = 256, sent to mono, split measures 1.07x faster) — worth
    ~7% on one shape of a layout we do not ship. Not worth a constant change on that
    evidence.
  * **tp2_down's grid-order regression** (~1.5% slower than v3) is the one place a
    conditional grid order would pay. It needs a mechanism first; the pattern across
    panels=80 shapes is non-monotone in kslabs, so there is nothing to condition on yet.
  * **Bandwidth-side work (smaller formats) is still worthless for speed.** Bytes read did
    not move and are not the constraint at 58% of peak.
  * **rb2 and the whole MODE_TAB table-delivery family are closed.** Both are measured
    negatives on the v5 kernel; see above. Do not spend card time on them again.
