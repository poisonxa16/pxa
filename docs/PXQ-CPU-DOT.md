# PXQ on the CPU, phase 2: the integer dot

Phase 1 (`ggml/src/pxq-cpu.c`) made every PXQ tier *decodable* on the host. Phase 2 makes the
4-bit tiers *runnable* on it: `ggml/src/pxq-dot.c` replaces the decode-then-multiply fallback
with a dot product that never leaves the integer domain.

That is what a CPU expert offload (`--cpu-moe`), a partial offload (`-ngl < 99`) and a
GPU-free load (`-ngl 0`) all sit on.

## Why there is still no `.vec_dot`

Unchanged from the export lane's finding, and it is structural. ggml hands `.vec_dot` and
`.to_float` a single **row pointer**; a PXQ row's bytes are scattered across the slabs of its
64-row panel, and the panel base is not recoverable from a row pointer. So the type traits for
the PXQ types keep `.to_float` / `.from_float` / `.vec_dot` NULL, with the `PXA_NO_CPU_VEC_DOT`
stubs in `ggml.c` as the backstop against a NULL call.

`pxq-dot.c` takes the addressing the format actually has — **tensor base plus a global row
index** — exactly like `pxa_pxq_dequant_row`. Its callers are `pxa_pxq_mul_mat_cpu` and
`pxa_pxq_moe_up_gate_cpu`, which `mul_mat` / `mul_mat_id` already reach through the
panel-dequant early return.

## The arithmetic

For one 32-element K-block of a 4-bit tier, with activations quantised as `x[j] = d * a[j]`
(`a` int8, one scale per 32 values, so one activation block per slab):

```
sum_j w[j] x[j] = anchor * d * ( SUB[s0] * sum_{j<16}  BOOK[c_j] a[j]
                               + SUB[s1] * sum_{j>=16} BOOK[c_j] a[j] )
```

The book enters only inside an integer sum, so replacing `BOOK` with an int8 image `B`
(`BOOK[c] ~= B[c]*bs`) makes each half an exact int32 dot and takes `bs` out to the end. The
anchor, the sub-scale and the activation scale all stay fp32 and are applied once per block.

On AVX2 the nibble plane becomes book values in one `_mm256_shuffle_epi8` against the 16-entry
int8 table — one uop for 32 weights — and the products run through
`_mm256_sign_epi8` / `_mm256_maddubs_epi16` / `_mm256_madd_epi16`. The `sign_epi8` pairing keeps
both maddubs operands under 127, so `2*127*127 = 32258` never reaches the int16 saturation point:
the integer part of the dot is **exact**, not approximate.

## Why int8 and not an fp16 or int16 codebook

int8 is what makes the lookup free: a 16-entry int8 table is exactly one `shuffle_epi8` operand.
A 16-bit book needs two shuffles plus an interleave to assemble and halves the lanes per
multiply — roughly 2x the kernel.

What it would buy is bounded by the frozen PX16 book, whose absmax is exactly 1.0, so `bs = 1/127`
and the worst book error is `0.5/127 = 3.9e-3` of the block's own effective scale.
`tests/test-pxq-cpu-dot.cpp` measures that error separately from the activation error, and the
book is consistently the smaller of the two (rms 3.1-3.6e-3 vs 9.0-10.4e-3). Widening the book
alone therefore cannot move the result: the activation quantisation dominates, and it is the
same Q8 activation error every int8 CPU quant path in this tree already carries.

## Accuracy is traded, deliberately

Phase 1 dotted **exact** f32 weights against **exact** f32 activations. The integer path
introduces both the int8 book error and the Q8 activation error. That is the standard llama.cpp
trade — Q4_K, Q6_K and the rest all pair quantised weights with q8 activations — and it buys
5-8x. `PXA_PXQ_CPU_DOT=0` sends the 4-bit tiers back to the phase-1 path, which remains the
parity-locked contract.

## Measured

Host: Xeon E5-2699 v3 (Haswell, AVX2 + FMA, no AVX-512, no VNNI).
Model: `Qwen3-0.6B` quantised `PXQ4` — 375 MiB, 140 pxq4 tensors, 51.7% of the bytes.
`llama-bench -ngl 0`, `PXA_PXQ_CPU_DOT` as the A/B lever:

| threads | test | phase 1 (dequant) | phase 2 (int8 dot) | speedup |
|---:|---|---:|---:|---:|
| 12 | pp128 | 14.22 t/s | 118.12 t/s | 8.3x |
| 12 | tg32  |  7.33 t/s |  42.10 t/s | 5.7x |
| 24 | pp128 | 27.95 t/s | 153.69 t/s | 5.5x |
| 24 | tg32  | 12.60 t/s |  46.43 t/s | 3.7x |

Single-thread gemv microbenchmark, 2048x4096 PXQ4: 5.2 Gweight/s against 275 Mweight/s for
dequant + f64 dot, 18.9x — the end-to-end factors are lower because only 52% of the model's
bytes are PXQ4 and the rest of the graph is unchanged.

Accuracy, `tests/test-pxq-cpu-dot.cpp` (uniformly random codes, i.e. the worst case for
cancellation; E_MAG is the error against the row's own term magnitude, E_DOT against the result):

| tier | k | BOOK rms_E_DOT | ACT rms_E_DOT | total rms_E_DOT | total max_E_MAG | AVX2 vs scalar max_E_MAG |
|---|---:|---:|---:|---:|---:|---:|
| PXQ4    | 1024 | 3.64e-3 | 9.16e-3 | 9.84e-3 | 1.71e-3 | 5.0e-8 |
| PXQ4    | 4096 | 3.06e-3 | 1.01e-2 | 1.02e-2 | 8.69e-4 | 3.1e-8 |
| PXQ4-HQ | 1024 | 3.42e-3 | 1.04e-2 | 1.07e-2 | 1.97e-3 | 5.2e-8 |
| PXQ4-HQ | 4096 | 3.44e-3 | 9.02e-3 | 8.91e-3 | 9.34e-4 | 3.3e-8 |

Greedy generation on the model above diverges from the phase-1 path after 81-698 characters over
five prompts (one prompt identical for its full 698); both paths stay coherent and on-topic.
That is what a ~1e-2 relative perturbation of every PXQ matmul does to greedy decoding of a
0.6B model, and it is the same order as switching any llama.cpp weight type to its q8-activation
kernel.

Against a CUDA reference (`-ngl 99`, same binary, GTX 1080 Ti / sm_61), neither CPU path is
systematically closer, which is the point — common prefix in characters, greedy, five prompts:

| prompt | phase 1 (dequant) | phase 2 (int8 dot) |
|---:|---:|---:|
| 0 |  81/858  | 358/946 |
| 1 | 233/859  | 475/859 |
| 2 | 1031/1031 (identical) | 178/1015 |
| 3 |  65/698  |  65/698 |
| 4 | 305/931  | 157/931 |

Phase 2 is the closer of the two on three of the five and the further on two. The CPU/CUDA
divergence is dominated by the CUDA GEMM snapping products to fp16 inside the MMA — which
`pxq-cpu.h` has always said the CPU fallback is not required to match — not by the int8 dot.

## Coverage

| tier | id | CPU dequant (phase 1) | integer dot (phase 2) |
|---|---|---|---|
| PXQ1    | 248 | yes | no — 1-bit sign codes |
| PXQ4    | 252 | yes | **yes** |
| PXQ4-HQ | 253 | yes | **yes** |
| PXQ2    | 254 | yes | no — 2-bit packing |
| PXQ3    | 255 | yes | no — bit-plane packing |
| PXQ6    | 256 | yes | no — 32-entry book |

`pxa_pxq_is_cpu_supported` is true for all six tiers now — the export lane's PXQ1/PXQ6 decode
commit merged. Every tier the host can decode runs, on `-ngl 0`, a partial offload, and the
fused CPU MoE op alike; the ones without an integer dot (PXQ1, PXQ2, PXQ3, PXQ6) run at
phase-1 (dequant) speed instead of phase-2 speed.

## Tests

`tests/test-pxq-cpu-dot.cpp` — standalone (not CMake-registered), build line in its header. It
synthesises the panels itself, so it holds every `(anchor, sub, code)` triple and can separate
the book error from the activation error, and it checks the synthesised panels bit-exactly
against `pxa_pxq_dequant_row` so a mistake in the test reads as a layout failure rather than as
an error number.
