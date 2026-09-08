# PXQ4 sm_70 kernel — validation on real V100 hardware

Everything before this ran on the CPU host simulator. This is the first time the
kernel executed on a GPU.

Date: 2026-08-18. Hardware: Tesla V100-SXM2-32GB, cc 7.0, DGX card 7 only.
Container: `kewaii/vllm:latest`, CUDA 12.8 (nvcc V12.8.93), torch 2.10.0+cu128,
built with `cmake -DCMAKE_CUDA_ARCHITECTURES=70`.

## Result

GPU output is **bitwise identical** to the CPU host-simulator oracle and to the
numpy reference on every case tested. Not "within fp16 tolerance" — exact bit
equality, max abs error 0.0, zero mismatches.

The oracle is legitimate: `pxq4_kernel_hostsim.cpp` compiles the *same kernel
source* for the host, so CPU and GPU are the same arithmetic on two backends.

Source provenance was checked rather than assumed — all six kernel files on the
DGX were md5-identical to branch `pxa/vllm-pxq4-sidecar`.

### Numbers

| path | geometries | elements | mismatches | max abs err |
|---|---|---|---|---|
| `dequant_out` | N=128/K=96, N=256/K=64, N=64/K=4352 | 12288 / 16384 / 278528 | 0 | 0.0 |
| `mmv_out` (vec) | 3 geometries x M in {1,3,8} | up to 2048 out/case | 0 | 0.0 |
| `mmv_out_scalar` | same | same | 0 | 0.0 |

Vector and scalar paths also agree with each other bitwise. The constant-memory
table upload works on hardware: the live device readback (`cudaMemcpyFromSymbol`)
matches the builtin literals in the `.so`, the hostsim's, and numpy's.

### Geometry gate refuses correctly

C++ `TORCH_CHECK` rejects: slab stride != 1088, anchor row != 64, out rows not a
panel multiple (124), out cols not a slab multiple (112), mmv x.K mismatch.
`mmv_supported`: K=100 and K=33 -> False; K=96/4352/5120/6144/17408 -> True.
At the vLLM layer, `PXQ4LinearMethod.create_weights` refuses N=100 (not %64) and
K=48 (not %32) with the intended messages.

### vLLM layer path (partial end-to-end)

With the fork's real parameter classes and a single-rank gloo TP group:
`create_weights` -> `weight_loader_v2` -> `process_weights_after_loading`
(use_mmv=True, mmv_max_m=8, workspace materialized) -> `apply`. M=1 and M=8 take
the mmv path (bitwise == hostsim); M=9 and M=64 take the dequant+GEMM path
(bitwise == cuBLAS on the hostsim-dequantized weight). Bias path correct.

## What was NOT tested — read this before trusting it further

1. **No full vLLM engine run.** Card 7 had ~2.6 GiB free beside the live serving
   shard. Booting an engine on the real 27B PXQ4 checkpoint does not fit in that
   budget and was not worth risking the serving model for.
2. **Synthetic random weights only.** The GGUF->vLLM converter output and real
   checkpoint tensors were not exercised on GPU.
3. **TP=1 only.** Multi-rank sharding of the panel layout is untested on hardware,
   even though the layout was shown to shard cleanly on paper at TP=2 and TP=4.
4. **No CUDA-graph capture, no torch.compile/inductor tracing, no multi-stream.**
   The capture-safety claims in the code remain unverified; the code's own "gate
   G8" caveat still stands.
5. **No perf measurement at all.** This was a correctness pass. Nothing here says
   the kernel is fast.
6. One seed per geometry, three geometries, M <= 64. Edge cases (M near 65535,
   denormal or NaN anchors) were not probed.

Bitwise identity across ~300k dequant elements and ~9k mmv outputs is strong
evidence the arithmetic is exact. It is not evidence about performance, graph
capture, or multi-GPU.

## Constraints observed

The DGX is lent, not ours. GPUs 0-3 (the owner's) were never touched; no container
was stopped, killed or removed; only `docker run --rm` with
`NVIDIA_VISIBLE_DEVICES=7`; free memory was checked before and after every run
(minimum seen 2601 MiB against a 1200 MiB abort threshold); nothing under
`/path/to/models/pxa*` or `/path/to/models/hf` was deleted.
