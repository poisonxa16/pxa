# vllm-pxq4 — PXQ quantization backend for vLLM (PXQ2 / PXQ3 / PXQ4)

Serve PXQ-quantized models on vLLM, with tensor parallelism, CUDA-graph capture and
paged KV — on V100-class hardware.

This ships **alongside** the `pxa` engine, not inside it. Two runtimes, one
quant family:

| runtime | hardware | why |
|---|---|---|
| `pxa` (this repo) | sm_60 Pascal, sm_61, sm_70 Volta, newer | GGUF-native, runs everywhere, the universal engine |
| `vllm-pxq4` (this package) | sm_70 Volta, and sm_60 Pascal through the sm60 sidecar | tensor parallelism + CUDA graphs, which llama.cpp does not have on Volta |

**Which tier runs where** is one table, in
[`docs/VLLM.md`](../../docs/VLLM.md#2-quant-tier-support--the-truth-table) — PXQ2, PXQ3 and
PXQ4 convert and serve here; PXQ1, PXQ4-HQ and PXQ6 are llama.cpp-only and the converter
refuses them by name rather than skipping their tensors.

**Pascal needs the sidecar, and it is not the primary path.** Stock vLLM's compiled kernels
target compute capability 7.0 and up (`CUDA_SUPPORTED_ARCHS = "7.0;7.5;8.0;..."`), so P100
(sm_60) reaches this backend only through the sm60 sidecar tree and the sm60 image, and
1080 Ti (sm_61) is not served here at all. On those cards `pxa` is the engine to use, which
is the reason it stays the primary one.

## Standing on other people's work

This package exists because two pieces of work happened first, and it would be dishonest
to present it without them:

- **vLLM** (Apache 2.0) — the serving engine, tensor parallelism, paged attention,
  continuous batching, CUDA-graph capture. I patch **zero lines** of it; this plugs in
  through the documented `register_quantization_config` hook.
- **A community Volta port of vLLM** — it carries the TurboMind sm_70 W4A16 GEMM, the
  `FLASH_ATTN_V100` backend and the Gated-DeltaNet sm_70 kernels.
  Upstream vLLM dropped sm_70; that fork carries the TurboMind sm_70 W4A16 GEMM, the
  `FLASH_ATTN_V100` attention backend, and the Qwen Gated-DeltaNet kernels
  (`FlashQLA-SM70`) without which none of this runs on a V100. **PXA Network contributed
  to getting that V100 support working.** This package is the continuation of that effort,
  not a fork of it.
- **PXQ4** — the quantization format, its CUDA kernels, and this backend: PXA Network.

If you only want faster inference on Volta and do not need PXQ, use the Volta vLLM port directly.
This package is for people who have PXQ artifacts.

## Honest performance

Do not read a headline number off this without reading the paragraph under it.

Measured on 4x V100-SXM2-32GB (TP=4), Qwen3.8-27B:

| | decode tok/s |
|---|---|
| `pxa`, `-sm layer` + `ngram_mod` | 47.96 prose / 63.76 code (**measured**) |
| vLLM + AWQ W4A16 (incumbent) | 92.8 peak / 57.4 median (**measured**) |
| vLLM + PXQ4, this package | **projected ~+9% over AWQ** — *not yet measured* |

**PXQ4 is not a smaller format than AWQ.** Measured like-for-like on the language-model
body: AWQ g128 asym = 4.156 bpw; PXQ4 = 4.254 bpw (`4.25 + 16/K`). PXQ4 is ~2.3% *larger*
per tensor. Since decode is bytes-read-per-GPU-per-token bound, a naive port that leaves
the non-PXQ4 tensor classes in fp16 is a **~23% regression**, not a win. Reaching +9%
requires re-encoding `lm_head`, `attn_k`/`attn_v` and `ssm_out`.

**The reason to use PXQ4 here is quality per bit, not size.** The format carries an fp16
row anchor, a per-16-element sub-scale and a non-uniform 16-entry codebook fit against an
importance matrix — a better-conditioned 4-bit than uniform group quantization at
essentially the same footprint.

### A free win for any vLLM deployment, PXQ or not

Profiling the incumbent turned up something unrelated to my format: `lm_head` is served
**BF16 (2.37 GiB)** and sits in the 311-entry `ignore` list. It is read on every decode
step on every rank — roughly 12% of all decode traffic. Quantizing it is likely the
cheapest speedup available on that deployment and needs nothing from this package.
Caveat: the output head is the layer most sensitive to quantization error; `q8_0` captures
about half the win at much lower risk, which is why the PXQ backbone table keeps `output`
at `q8_0` rather than 4-bit.

## Status

Validated on real hardware. All five gates below pass on both architectures — see
`docs/VLLM.md` section 10 for the runs, and `docs/PXA-SM60-SERVING.md` and
`docs/PXA-SM70-SERVING.md` for the measured sweeps behind the numbers. The gates, which is what
each one proves:

1. bit-exact dequant parity against a CPU reference
2. single-linear-layer GEMM parity
3. **sharded parity** — per-rank slices must dequantize to exactly the unsharded result
4. logprob parity vs `pxa` on the same prompts at temp 0
5. an end-to-end throughput measurement to replace the projection above

## Licence

Apache 2.0, matching vLLM. See `LICENSE-NOTICE.md` for the full attribution chain.

---

## Build and test

Everything under `src/` is a FLAT directory on purpose: `build_hostsim.sh`, `setup.py`,
`CMakeLists.txt` and the test imports all resolve relative to it. An earlier tidy-up into
`csrc/ tests/ vllm_pxq4/` broke every entry point, so the layout that works is the one
that ships.

### GPU-free gates (run these first - they need no CUDA, no GPU)

```bash
cd src
bash build_hostsim.sh          # compiles the CPU simulator, then runs the kernel suite
python3 test_pxq4_config.py    # quant config / plugin registration
python3 gguf_to_vllm_test.py   # converter, incl. a bit-exact gate against a C oracle
python3 test_pxq4_linear.py    # linear method (skips cleanly if vLLM is absent)
```

`build_hostsim.sh` compiles `pxq4_kernel_hostsim.cpp`, which includes the REAL
`pxq4_kernel.cuh` unmodified against a stub `cuda_fp16.h` and emulates a CUDA launch.
So the kernel suite exercises the shipping kernel source, not a reimplementation that
could drift from it.

**A compiler is required.** Without `libpxq4_hostsim.so` the 8 simulator-backed tests
FAIL rather than skip, and the failure text tells you to build it. On a box with no
`g++` (my host has none) you will see `9/17` — that is a missing toolchain,
not a kernel defect. Build on a dev host or inside the CUDA container.

### CUDA extension (needs the CUDA toolkit; no GPU needed to compile)

```bash
bash src/build.sh              # sm_70; expect a "prior to sm_75" deprecation warning
```

### Measured status

| suite | result | where |
|---|---|---|
| kernel parity (hostsim) | 17/17 | any host with g++ |
| quant config | 27/27 | anywhere |
| config integration | 6/6 | against the real vLLM fork |
| converter | 79/79 + correct parse of the real 14.64 GiB artifact | anywhere |
| CUDA build | clean sm_70, ops register | CUDA container |
| end-to-end on GPU | **NOT DONE** | needs a V100 |

The real artifact parses as 325 pxq4 + 132 q8_0 + 1 q6_K + 360 f32 + 48 mxfp4, and
`data_start + sum(nbytes) == file size` exactly. Any design assuming uniform PXQ4 is wrong.
