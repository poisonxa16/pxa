# Attribution

This package is licensed Apache 2.0, matching vLLM.

## vLLM
Copyright the vLLM contributors. Apache License 2.0.
The PXQ4 backend itself registers through the documented
`register_quantization_config` plugin hook and changes no vLLM file.

### Modified vLLM files (Apache License 2.0, section 4)
`patches/*.patch` modify Apache-2.0 vLLM source files. `deploy/Dockerfile.field` applies them
inside the field image. The files they change are vLLM's, under vLLM's Apache License 2.0;
the changes are by PXA Network (2026) and are released under the same licence:
  - `solo-prefill-scheduler.patch` changes `vllm/v1/core/sched/scheduler.py`
  - `swa-keep-hashed-block-pool.patch` changes `vllm/v1/core/block_pool.py`
  - `lenient-acceptance.patch` changes `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`
  - `dflash-pxqn-load.patch` changes `vllm/model_executor/models/qwen3_dflash.py`
Each patch file starts with a notice that says so (section 4(b)). The vLLM copyright and
licence headers in those files are kept unchanged (section 4(c)).

## PyTorch — one optional patch, and it is not vLLM

Separately from the vLLM patches above, `tools/patch_sm60_compile.py`
does edit **PyTorch** (`torch/utils/_triton.py`) to lift a hard-coded `device major >= 7` gate
in `has_triton()`. That gate is not a real capability limit — Triton 3.3 compiles and runs
ordinary pointwise/reduction kernels on sm_60; only `tl.dot` (MMA) is genuinely unavailable,
and inductor does not lower matmuls to Triton unless max_autotune is on.

It is OPT-IN and only needed to get torch.compile / CUDA graphs working on Pascal (P100).
Nothing on sm_70 or newer requires it, and PXQ4 itself does not depend on it.

## The Volta vLLM port (Volta / sm_70 support)

Upstream vLLM does not target compute capability 7.0. That fork carries the
sm_70 work this package depends on entirely:
  - TurboMind sm_70 W4A16 GEMM (csrc/sm70_turbomind/)
  - the FLASH_ATTN_V100 attention backend
  - the Qwen Gated-DeltaNet kernels (FlashQLA-SM70)
PXA Network contributed to getting that V100 support working. This package is a
continuation of that effort, not a fork of it, and contains no code from that port;
the port's own licence terms apply to the port.

## PXQ4
The PXQ quantization format, its CUDA kernels, and this backend: PXA Network.
