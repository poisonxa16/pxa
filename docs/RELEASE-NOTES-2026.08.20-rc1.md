# PXQ llama.cpp — v2026.08.20-rc1

Release candidate. Supersedes **v2026.08.11**; **249 commits**.
Engine build **396**, unified line `pxa/unified-safe`.

---

## Read this first

**The fork now builds without AVX2.** If you tried a previous release on a machine that builds
stock llama.cpp fine and it failed anyway, that was our bug, not your toolchain — and it is
fixed. See *Portability* below for what was wrong.

**`-DGGML_IQK_MUL_MAT=OFF` now does something.** It previously named a configuration that could
not build. It now produces a working engine with none of the ik GEMM/flash-attention kernels
compiled.

Nothing in this release changes numerics on an AVX2 + CUDA build. Every portability change is a
new fallback branch selected only when the old one could not compile.

---

## Portability — the AVX2 fix

Symptom: stock llama.cpp builds, this fork and ik_llama do not, on the same machine.

`iqk_config.h` defines `IQK_IMPLEMENT` only for `__AVX2__` / `__ARM_FEATURE_DOTPROD`, but
`iqk_quantize.cpp` and `iqk_cpu_ops.cpp` were compiled unconditionally and carry code outside
that guard. On a target without AVX2 they lost `<cstdint>`, `ggml-impl.h` and `ggml-common.h`
while still compiling their bodies, and two quantizers fell into AVX2 intrinsics through an
`#else` on `__aarch64__`.

| fix | effect |
|---|---|
| `<cstdint>` hoisted out of the `IQK_IMPLEMENT` guard in `iqk_common.h` | without the fixed-width types all four `popcount` overloads collapse to one signature — 49 errors, one per includer |
| `iqk_cpu_ops.cpp` includes `ggml-impl.h` / `ggml-common.h` directly | its own `#define IQK_IMPLEMENT` never survived — `iqk_config.h` `#undef`s it first |
| `quantize_row_q8_0_x4`, `quantize_row_q8_1_x4_T` | AVX2 arm moved behind `#elif defined(__AVX2__)` with scalar fallbacks derived from the NEON arms |
| `repack_q8_KV` | scalar branch had never compiled; rewritten and verified byte-identical to the AVX2 path |
| `ggml-impl.h` | reconciles `GGML_USE_IQK_MULMAT` with `IQK_IMPLEMENT` in one place, so the option cannot claim kernels the target cannot build |

⚠ **This also fixes a latent runtime hazard.** AVX2 intrinsics were being compiled into
non-AVX2 binaries; only 6 errors surfaced because a two-phase lookup failure on `hsum_i32_8`
stopped template instantiation before the intrinsic bodies were checked. Fixing that one symbol
alone would have produced a binary that built cleanly and executed an illegal instruction on the
target machine.

### Where the ik seam actually is

- **ik GEMM / flash-attention kernels — optional.** Every call site is
  `if (iqk_x(...)) { fast path }` over a stock ggml path. `-DGGML_IQK_MUL_MAT=OFF` disables them.
- **`iqk_quantize.cpp` / `iqk_cpu_ops.cpp` — not optional.** They implement ~40 quantization
  types (the `_r4` / `_ks` / `_bn` family: `iq4_ks`, `q5_k_r4`, `iq2_bn_r4`, `q8_0_r8`,
  `q1_0_g128`, …) that ggml's `type_traits` table points at. Removing them is ~40 undefined
  references, not a fallback.

Removing ik outright therefore means removing those types from the table first. Tracked as its
own campaign; not a build flag.

**PXQ never depended on ik.** `PXQ1/2/3/4/4HQ/6`, their CUDA kernels, and the `pxq-cpu.h` CPU
fallback are ours and are unaffected by any ik setting.

---

## New architectures

| arch | notes |
|---|---|
| `deepseek4` | DeepSeek-V4 / DS4-Flash. DSV4 memory module (raw SWA ring + three compressed block streams), DSA sparse attention as head-512 flash attention on sm_70. Hyper-connection fusion: **32,077 → 7,639 graph nodes per forward** |
| `muse-glimmer` | dense 30B — SWA 3:1 + NoPE globals, full-width sigmoid attention gate, sandwich norms, softcapped head; vision projector ported |
| `hy_v3` | Hunyuan V3, 295B-A21B |
| `dspark` | DeepSeek-Spark — dedicated graph builder + scheduler |
| `cohere2moe` | North Mini Code — arch, chat parser, template |
| `qwen35` (dense) | accepted by the quantizer's `n_attention_wv` check |

## PXQ codec

- **PXQ2 v3 book** (`PXA_PXQ2_V3`, default OFF) — model-family Lloyd refit. Held-out **−7.9% raw
  / −9.3% population-weighted** error, uniform across weight bands. The pinned-exact-zero variant
  was fit, evaluated and **failed** its pre-registered gate — recorded, not shipped.
- **PXQ2 v2 book** (`PXA_PXQ_CEIL_V2`) — ceiling fix; restores the representable ceiling from
  `0.697×anchor` to `0.987793×anchor`. v1 remains default so every shipped PXQ2 file still
  decodes byte-identically.
- **`pxq_ceiling_check`** — refuses a book/SUB16 composition that cannot reach the row anchor, at
  build time rather than after an 83 GB quantize. The frozen v1 table warns and names the lever;
  any other sub-bar book still aborts.
- `PXQ_CANON_v2` chained-FFMA accumulation: **+5.7%** drafted decode (122B / 4×P100).
- `PXA_PXQ6_SHFL` **+4.1%**, `PXA_PXQ3_PAIRLUT` **+3.7%**, both bit-exact.
- `FUGSPLIT` dense PXQ up/gate as two plain MMVQ launches: **sm_70 +14.7%**, bit-identical.
- `ggml_validate_row_data` accepts the PXA slab types.

## Performance

- **122B / P100 decode 27.69 → 33.20 tok/s**, with a correctness gate.
- `PXA_FA_TILE_VOLTA` — **+6.9% prefill** on sm_70 (default off).
- sm_60: GQA head-packed D=256 vec flash-attention, shared-memory query staging, 4-way ILP.
- `PXA_AUTO_SPEC` — arms the measured-best self-speculation drafter per family.

**Recorded negative results** (kept so they are not re-attempted): FUSERED (loss on both arches),
`PXA_MTP_PREFETCH` (null), ROWX2 (null), `PXA_PXQ_REDUCE_BLK` (null), sm_60 fast-fp16 carve-out
(**−49%/−58% prefill**), and the K9/K10 sm_70 decode probes (**−7.2%** / **−3.4%**) — the last
two with the reusable finding that the PXQ decode dot is *not* shared-memory-bandwidth limited.

## Correctness fixes

- **WMMA half-accumulate mask query-row stride** (`ne11` → `nb31/sizeof(half)`) — live
  garbage-output bug on sm_70 SWA models at depth.
- **Sliding-window KV rewritten** — sliding layers get their own window-sized cache, ring scan
  steps over live cells, eviction against the earliest still-reachable query position.
- `dsa_attn` binds the cuBLAS handle to our stream (upstream race).
- deepseek4 flipped RoPE must not be in-place on sm_70 (`-fa on` aborted).
- 16 upstream ports, including deepstack image-embedding stride OOB, perplexity int overflows,
  uninitialized MROPE/IMROPE sections, CPU-only load crash on a CUDA build.

## vLLM PXQ4 sidecar — what ships, and what it needs

**Included in this release: `tools/vllm-pxq4/`** — 160 files. The PXQ4 CUDA kernels, the vLLM
linear method and quant config, the GGUF→vLLM converter, launch/deploy scripts, a benchmark
harness, a parity harness with a C reference, and 13 design documents covering the format, the
kernels, the plugin surface and the sharding/MTP verdicts.

**No vLLM source is shipped or patched.** It attaches through vLLM's documented
`register_quantization_config` plugin hook — zero lines of vLLM modified — so it survives vLLM
upgrades instead of fighting them. Apache 2.0, matching vLLM.

⚠ **It requires [1Cat-vLLM](https://github.com/KewaiiGamer/1Cat-vLLM), not upstream vLLM.**
Upstream does not target compute capability 7.0. That fork carries the sm_70 work this backend
depends on entirely: the TurboMind sm_70 W4A16 GEMM, the `FLASH_ATTN_V100` attention backend,
and the Qwen Gated-DeltaNet kernels. PXA contributed to getting that V100 support working; this
is a continuation of it, not a fork of it. On sm_80+ hardware you do not need any of that, but
the sm_70 path is what this sidecar was built and measured against.

⚠ **One optional PyTorch patch, for Pascal only.** `tools/patch_sm60_compile.py` lifts a
hardcoded `device major >= 7` gate in `torch/utils/_triton.py::has_triton()`. That gate is not a
real capability limit — Triton 3.3 runs pointwise/reduction kernels on sm_60 fine; only `tl.dot`
is unavailable. It is opt-in, needed only for torch.compile / CUDA graphs on P100, and PXQ4
itself does not depend on it. Disclosed in `LICENSE-NOTICE.md`.

### Build

`src/build.sh` compiles the kernels inside the same image the server runs, so torch/nvcc/gcc are
byte-identical to what will load them. Point it at your own setup:

```bash
PXQ4_IMAGE=vllm/vllm-openai:v0.x PXQ4_SRC=/path/csrc PXQ4_OUT=/path/site/pxq4_vllm/_lib PXQ4_BUILD=/path/build   bash src/build.sh
```

`PXQ4_REF_CONTAINER=<name>` derives the image from a running container instead. The script
mounts the common ancestor of SRC/OUT/BUILD; override with `PXQ4_MOUNT_ROOT`.

### Measured

New subsystem since v2026.08.11. sm_70 validated on real hardware, **sm_60 (P100) support
added**, **CUDA graphs on sm_60: 3.9 → 14.9 tok/s decode**, TP=4 with v3 split mmv **42.3 → 55.9
tok/s**, K-chunk-split decode mmv **1.7–2.3×** on starved shapes. `gguf_to_vllm` restores the
gemma-norm −1 offset for `qwen3_5` layernorms — without it the converted artifact is silently
wrong.

## Serving stack under version control

The serving stack — the orchestration daemon, instance keeper, affinity/cache proxy, launch
scripts, tools, tests and templates — is now tracked. Previously loose files on one box:
untracked, and therefore unmergeable. The stack is operational infrastructure and is not part
of this public release.

---

## Build

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;61;70"
cmake --build build -j
```

Without AVX2, or to build without the ik accelerators:

```bash
cmake -B build -DGGML_IQK_MUL_MAT=OFF
cmake --build build -j
```

Both configurations are gated in CI-equivalent form by the no-ik and AVX-default build gates.

## Not in this RC

- A real imatrix for the shipped PXQ4 (current file has none).
- The GET_ROWS quantizer guard.
- `P100_FP16_GEMM` default-on — gains ~3–4% but fails the quality gate at 94.09%
  same-top-token. Left OFF pending a decision.
- Full ik removal (see *Where the ik seam actually is*).
