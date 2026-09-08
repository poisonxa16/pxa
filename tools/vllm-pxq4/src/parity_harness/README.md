# PXQ4-in-vLLM parity harness (agent D, plan §9)

The correctness gate for the port. Everything here is either **runnable right now on this
machine with numpy alone**, or clearly marked as needing a GPU.

**Status as of writing: 31 gates PASS, 0 fail, ~20 s, no GPU.** Real tensors pulled from
`/path/to/models/pxa-models/Qwen3.8-27B-PXQ4.gguf`; agent A's converter reference, agent C's
numpy twin, AND agent C's real CUDA kernel (host-simulated) all bit-exact against the
production `ggml/src/pxq-cpu.c`. No container was touched. `/path/to/engine-repo` was
read, never written. No GPU was run, and no number in this harness is a throughput claim.

```
$ python3 -m parity_harness.run_gates --real-dir fixtures_real
real fixtures: ['attn_gate','attn_output','attn_q','attn_qkv','ffn_down','ffn_gate']
  pxa.pxq6.book / pxa.pxq6.sub match ggml-pxq6-tables.h exactly
  backbone_rev=2 map=attn_q,attn_qkv,attn_output,attn_gate_ch,shexp,ffn_dense=tier+1;...
  ...
31 passed, 0 failed, 10 skipped   (all 10 skips are the GPU-only gates)
```

The 10 skipped gates are the ones that genuinely need sm_70 silicon: CUDA-graph capture,
the dynamic-smem opt-in, the op schema/meta registration, and the allocation check. They
are written and ready; they need a card, not more code.

---

## Why four implementations and not two

A harness that compares agent A's reference against agent C's kernel proves they agree,
not that either is right. This one closes the loop to the shipping engine:

```
 ggml/src/pxq-cpu.c            cref/       the ACTUAL production dequant, compiled here
        |  G1a  (bit-exact)               <- CPU, runs today
 oracle.py                                 independent numpy transcription (this harness)
        |  G1b  (bit-exact)               <- CPU, runs today, checks EVERY sibling ref
 gguf_to_vllm.reference        agent A     what the converter calls
 pxq4_kernel_ref               agent C     the kernel's numpy twin
        |  H1-H7 (bit-exact)              <- CPU, runs today, via libpxq4_hostsim.so
 k_pxq4_dequant_matrix / k_pxq4_mmv        agent C's REAL device code, host-simulated
        |  G6/G8 (bit-exact, fp16)        <- needs a GPU
 torch.ops.pxq4.*              agent C     the same code, on sm_70
```

**The hostsim leg is the reason most of this runs without a lease.**
`pxq4_kernel_hostsim.cpp` compiles the real `k_pxq4_dequant_matrix` / `k_pxq4_mmv`
against a host shim for `blockIdx`/`threadIdx`/`__syncthreads`, and `hostsim_bridge.py`
drives it through ctypes. So gates H1-H7 test agent C's actual kernel source -- layout
addressing, table values, accumulation order, the fp16 store, and the whole shard
invariant -- on this machine, in a second.

A hostsim PASS is **necessary, not sufficient**. Still owed to real sm_70 hardware:
the launch configuration, the dynamic-smem opt-in (G8h), CUDA-graph capture (G8g), any
MODE using warp primitives (`__shfl_sync`/`prmt`, which the host shim cannot execute),
and nvcc's fp32 contraction choices.

`cref/vendor/` is a verbatim read-only copy of the nine files needed to compile
`pxq-cpu.c` standalone — see `cref/VENDOR_PROVENANCE.txt` for the exact list and the
one-liner that refreshes it. It builds with `cc` and `-lm`; no CUDA, no ggml build system,
about a second.

## Run it

```bash
cd .../impl                                     # the dir containing parity_harness/

# CPU gates, synthetic fixtures only, no external data of any kind
python3 -m parity_harness.run_gates

# CPU gates against real tensors from the artifact
python3 -m parity_harness.run_gates --real-dir fixtures_real

# add the CUDA gates (needs a GPU + agent C's .so)
PXQ4_LIB=/path/to/models/pxa-vllm-pxq4/build/libpxq4_sm70.so \
  python3 -m parity_harness.run_gates --real-dir fixtures_real --gpu

# a single gate, with the skip reasons spelled out
python3 -m parity_harness.run_gates --only G3b -v

# pytest also collects everything, if it happens to be installed
pytest parity_harness --real-dir=fixtures_real
```

Exit code is 0 only if every non-skipped gate passed. **A skip is never a pass** and the
summary lists every skipped gate with its reason.

## Getting real fixtures

The 14.64 GiB artifact lives on the DGX, which has no numpy. `extract_raw.py` is
stdlib-only and self-contained — scp it anywhere:

```bash
# on the DGX (writes only under /path/to/models)
python3 extract_raw.py --gguf /path/to/models/pxa-models/Qwen3.8-27B-PXQ4.gguf \
                       --out  /path/to/models/pxa-parity/raw --panels 4 --panel0 1
# ~6 MB total; copy the directory back and point --real-dir at it
```

`--panels 4` is not a weakening of the test: a panel subrange of a PXQ4 tensor *is* a
valid PXQ4 tensor, which is the same fact the column-shard argument rests on. `--panel0 1`
starts at the second panel so the extractor cannot accidentally be reading the start of
the data section and calling it a tensor.

The extractor hard-fails if any tensor's on-disk byte count disagrees with
`(rows/64) * (128 + (K/32)*1088)`. All six real shapes matched to the byte.

`extract.py` is the numpy/npz equivalent for machines that have numpy; `--real` loads it.

## The gates

| id | what it proves | needs |
|---|---|---|
| **G1a** | `oracle.dequant` == production `pxa_deq_row_pxq6`, **bitwise** | cc |
| G1b | agent A's `reference.dequant` == oracle, bitwise; and A's BOOK/SUB == `ggml-pxq6-tables.h` | agent A |
| G2a | split→join round-trips to the original bytes | — |
| G2b | the geometry gate refuses what vLLM would truncate | — |
| G2c | all six real shapes reproduce their on-disk sizes and 4.25+16/K bpw | — |
| **G3a** | column (row-count) split is bit-exact at TP 1/2/4 | — |
| **G3b** | row (K) split is bit-exact at TP 1/2/4, header duplicated verbatim | — |
| G3c | blob-space byte gather == emitted-space `narrow()`, both axes | — |
| G3d | merged-column (gate_up / in_proj_qkvz) assembly across ranks | — |
| G3e | qkv shard arithmetic; `attn_q` shards never cut a (q,gate) head pair | — |
| G3f | every real PXQ4 module aligns at TP 1/2/4 on its own axis | — |
| G3g | the fused-GDN-`ba` layout is *detected* as unshardable | — |
| G3h | misalignment raises here, and `packed_shard_indices` shows vLLM truncating | — |
| G3i | header-duplication overhead, quantified | — |
| G3j | row-parallel all-reduce stays accurate, and is *not* bit-equal (with the reason) | — |
| N1–N6 | **negative controls**: the gates above actually reject injected bugs | — |
| Gb1–3 | mmv fold vs an exact GEMM; `canon_nfix` pinned; fp16 output headroom | — |
| G6a/b | CUDA `dequant_out` == oracle bitwise; out-variant ABI honoured | GPU |
| G8a | CUDA `mmv_out` == the bit-exact CPU fold model | GPU |
| G8b | `mmv_out` == `dequant`+`mm` within tolerance (the M crossover) | GPU |
| G8c | `apply()` shape contract incl. 3-D activations | GPU |
| G8d | schema annotates `Tensor(a!)` output mutation | GPU |
| G8e | `register_fake` / meta kernels present | GPU |
| G8f | the ops allocate nothing | GPU |
| G8g | **CUDA-graph capture + replay** — plan §10 risk 3, the one UNVERIFIED assumption | GPU |
| G8h | wide-K dynamic smem opt-in (`cudaFuncSetAttribute`) — only bites at TP≤2 | GPU |
| **H1** | the kernel TU's compiled-in tables == `ggml-pxq6-tables.h` | hostsim |
| **H2** | **the real kernel's fp32 dequant == oracle, bitwise** | hostsim |
| H3 | the kernel's fp16 dequant is exactly one RNE of its fp32 | hostsim |
| **H4** | **the real kernel's shard invariant, both axes, TP 1/2/4** | hostsim |
| H5 | the kernel's mmv == the bit-exact fold model | hostsim |
| H6 | `k_pxq4_mmv<VECX=0>` == `<VECX=1>` bitwise (only one ships) | hostsim |
| H7 | the kernel's `canon_nfix` == the model's, over every real kslabs | hostsim |
| G7 | logprob parity vs llama-server at temp 0 | 2 servers |

`logprob_parity.py` is (d): stdlib-only, drives two OpenAI-compatible endpoints, and is
documented to run later. It never imports vllm and so cannot perturb the production
container.

## Runtime, and why the hostsim gates use trimmed fixtures

The pure-numpy gates (G1-G3, N1-N6, Gb1-3) finish in about **5 seconds**, on real
artifact tensors, untrimmed. Run those constantly.

The hostsim gates (H1-H7) are slower for a structural reason: the host shim launches one
**OS thread per CUDA thread**, serially over blocks (`pxq4_kernel_hostsim.cpp` `launch`),
measured here at ~6.7 ms per 64-thread block. A full real `ffn_down` is 4 panels x 544
slabs = 2176 blocks *per dequant*, and H4 does several — minutes each. So the **dequant**
hostsim gates (H2/H3/H4) trim each fixture to 4 panels x 4 slabs, and H4 checks ranks 0
and tp-1 rather than all of them.

**For the dequant gates, trimming does not weaken the test, and it does not weaken it for
the same reason the port works**: a panel subrange and a slab subrange are each themselves
a valid PXQ4 tensor (that is exactly what G3a and G3b prove), so a trimmed real fixture is
still real artifact bytes with the real anchors and the real code distribution — just
fewer of them. The dequant kernel has no fold: one block per slab, decoded identically
however many there are. The untrimmed real tensors remain covered bitwise by G1a/G3a/G3b
through the numpy oracle, which G1a pins to the production C. If you want the dequant
hostsim gates on full tensors, raise `HOSTSIM_MAX_PANELS`/`HOSTSIM_MAX_SLABS` in
`test_f_hostsim.py` and expect minutes; the fast way to get that coverage is the real GPU
gates (G6/G8).

**The mmv gates (H5/H6) must NOT use that trim, and no longer do.** `k_pxq4_mmv` is
*entirely* a fold: `nfix = pxq4_canon_nfix(kslabs)` chunks, chunk `c` spanning
`[(kslabs*c)/nfix, (kslabs*(c+1))/nfix)`, with agent C's EDIT 3 staging only that chunk's
activations into smem and re-basing the read as `pxq4_xs + (kb - b0)*PXQ4_QK`
(`pxq4_kernel.cuh:315`). At `kslabs = 4`, `lim = 4/PXQ4_MMV_KSEG = 1`, so `canon_nfix == 1`:
one chunk, `b0 == 0`, and the re-basing is the identity. A 4-slab trim therefore made EDIT
3 — the one substantive deviation from the shipping engine, and the reason `K = 17408`
works without a capture-hostile `cudaMalloc` workspace — completely untested. Verified by
mutation: deleting the re-basing left H1–H7 at 7 passed, 0 failed.

H5/H6 now draw from `MMV_SHAPES` in `test_f_hostsim.py` — kslabs 4, 16, 18, 34, 64, i.e.
nfix 1, 4, 4, 8, 16, two of them with unequal chunk lengths — and H5 asserts that budget
(nfix > 1 present, CANON_CMAX saturated, at least one ragged case) so a future cap change
cannot silently reopen the hole. Real fixtures in the mmv gates use a separate, larger cap
(`HOSTSIM_MMV_MAX_SLABS = 16`). Cost: the mmv grid is `(N/64, M)`, so runtime scales with
panels x M and only weakly with kslabs; H5+H6 went from ~5 s to ~12 s, and the whole CPU
suite from ~9 s to ~16 s.

## Design decisions worth knowing about

**Bitwise, not `np.array_equal`.** `array_equal` reports `0.0 == -0.0` as equal. PXQ4
emits signed zeros constantly — `BOOK[7]` is exactly 0, so every code-7 element is ±0
depending on its anchor's sign — so a sign-of-zero bug (computing `eff` from `|anchor|`,
say) would slip past on exactly the elements where it is easiest to introduce.
`compare.bitwise_equal` compares raw bit patterns; `compare.bit_diff_report` prints the
patterns, because a 1-ULP fp32 difference prints identically in decimal on both sides.

**Random slab bytes are a stronger fixture than real weights.** Every byte of a slab is a
valid PXQ4 byte (there are no reserved bits), so uniform random bytes are in-format by
construction *and* hit all 16 book entries and all 16 sub levels in every row. Real
quantized weights cluster hard around `BOOK[7] = 0` and would leave most of the table
untested. Real tensors are still run, to prove the generator is not itself what is being
tested and to check the file's own `pxa.pxq6.book`/`sub` KVs.

**The mmv model reproduces the kernel's accumulation ORDER, not just its value.**
`k_pxq6_mmv` commits to a specific fold: `nfix` chunks, KSEG=4 lanes walking
`kb = b0+kseg, +4, …` ascending, per-chunk partials, then an ascending cross-lane
reduction. `oracle.mmv` reproduces all of it. N6 proves this matters — a naive
left-to-right fp32 dot over the same weights gives different bits. Modelling the order is
what lets a G8a failure be attributed to the port rather than dismissed as float noise.

**The FMA-contraction ambiguity is searched, not guessed.** `pxq6_acc2` is
`acc + (a0*x0 + a1*x1)` in source, but nvcc's default `-fmad=true` fuses one of the two
multiplies into the inner add, and which one is a codegen fact not readable from the
source. G8a tries all nine (acc, tail) variants and **reports which matched**. Failing to
match *any* is the real signal, because no contraction choice can rescue a wrong layout.
`oracle._fma32` emulates fp32 FMA in fp64, which is exact here (53 ≥ 2·24+2).

## Two corrections to the plan, both established here

**1. Plan §6.3 is wrong that multiply order is load-bearing.** It says
"The multiply order `(anchor * sub) * book` is load-bearing for bit-exactness — do not
reassociate." For PXQ4 it is not. All three factors are fp16-snapped, so every pairwise
product needs ≤22 significand bits and is *exact* in fp32; both associations are a single
correctly-rounded rounding of the same exact triple product. Gate **N2** verifies this by
exhaustion — 40 009 anchors (including every fp16 edge case) × all 256 (sub, book) pairs,
**zero** bit mismatches — and additionally checks that `sub[i]*book[j]` is exact in fp32
for all 256 pairs. Agent C may fold `anchor*sub` into `eff` with no bit-exactness risk,
which is what `pxq6_pol_p6::row_effs` (pxq6.cuh:337-341) already does. The test is kept
because the property depends on the tables staying fp16-snapped; if a future regeneration
breaks that, N2 fires and the plan's warning becomes live again.

**2. A K-shard is not bit-identical to the unsharded computation, and that is correct.**
Gate G3j records why, with the numbers, so nobody spends a day on it: besides the
all-reduce reordering, `canon_nfix(kslabs)` *depends on K*, so a K/4 shard folds with a
different chunk count than the full tensor (measured: K=5120 → nfix 16, K=1280 → nfix 8).
The **weights** are bit-identical (G3b); the **sum** is not, by construction. Only G3b's
claim is a bit-exactness claim.

## Assumptions flagged in the code

Search for `ASSUMPTION:`. There are four:

1. `oracle.py` — the FMA-contraction search assumes nvcc does not *reassociate* across
   the parentheses in `pxq6_acc2` (contraction alone does not permit it). If it does, G8a
   will match no variant and the message says so rather than blaming the layout.
2. `logprob_parity.py` — the NOISE/SUSPICIOUS/BUG thresholds (1e-3 and 0.1 nats of top-1
   margin) are judgement calls, not measurements. The raw margins are always printed.
3. `test_d_ops_abi.py` — `assert_no_sm70_fastpath` assumes agent B's class is named
   `PXQ4LinearMethod` (plan §6.6). It matches on the type name, so a rename makes it
   silently vacuous; `assert_pxq4_module_coverage` is the paired positive check.
4. `hostsim_bridge.py` — assumes the hostsim TU is built from the same headers as the
   CUDA TU, so a divergence is a build problem rather than a semantic one. H1 checks the
   table half of that, which catches a stale object file.

### A third finding, from running H5

At **fp16 output the nine FMA-contraction variants of the mmv model are usually
indistinguishable** — the final round-to-nearest-even absorbs a sub-ULP fp32 difference.
H5 reports how many variants matched (9 = the test could not discriminate). This is not a
failure, but it does mean **G8a's discrimination power is weaker than the plan implies**:
matching the fold confirms its STRUCTURE (nfix chunking, KSEG lane assignment, ascending
reduction, `eff` applied once per 32-element block) but does not pin the contraction. The
fp32 dequant gates — G1a, H2 — are what carry the bit-exactness weight, and they are
exact because the dequant path contains no fused multiply-add at all.

**And "STRUCTURE" is now measured, not assumed.** With the `MMV_SHAPES` case set, mutants
of `pxq4_kernel.cuh` rebuilt and run through H5 give: re-basing deleted (`:315`) — caught;
staging base wrong (`:308 xt[idx]`) — caught; `nfix` pinned to 1 — caught. One mutant is
**not** caught and is documented rather than hidden: changing the chunk bounds to
`c*(kslabs/nfix)` keeps every activation paired with its own weight and only moves slabs
between KSEG lanes, so it perturbs the fp32 reduction order alone and the fp16 store
absorbs it in ~17 of 18 seed/M combinations tried. That one is owed to G8a on device,
where an fp32 comparison is available. Over-staging (a fixed `n` per chunk) is also not
caught, correctly: the surplus floats are never read, and its real defect is an
out-of-bounds read of `x`, a compute-sanitizer question rather than a numerical one.

## Files

```
oracle.py            independent numpy model: layout, dequant, mmv fold, sharding,
                     vLLM loader arithmetic, table-vs-GGUF check.  numpy only.
compare.py           bitwise comparison + ULP-aware diff reports + error stats
fixtures.py          synthetic (extreme / realistic anchor profiles) + real loaders
gguf_raw.py          struct+mmap GGUF reader that does not care about type ids
extract.py           GGUF -> .npz fixtures (needs numpy)
extract_raw.py       GGUF -> .bin + manifest.json (stdlib only; runs on the DGX)
cref/                the production pxq-cpu.c, vendored read-only, + a CLI + build.sh
cref_bridge.py       builds and drives cref/pxq4_cref
adapters.py          runtime discovery of agents A/B/C; absence downgrades to SKIP
test_a_dequant.py    G1, G2, G6
test_b_linear.py     (b) single-linear parity, Gb1-3, G8a-c
test_c_shard.py      (c) G3a-j  <- the important one
test_d_ops_abi.py    G8d-h + two in-engine helpers for after model load
test_e_mutation.py   N1-N6 negative controls + the §6.3 reassociation proof
logprob_parity.py    (d) G7, stdlib-only, runs later
run_gates.py         the driver; no pytest required
conftest.py          pytest glue, if pytest is present
```

`oracle.py` and `compare.py` import nothing but numpy and are safe to vendor anywhere.
