# Attribution

This package is licensed Apache 2.0, matching vLLM.

## vLLM
Copyright the vLLM contributors. Apache License 2.0.
This package patches no vLLM source; it registers through the documented
`register_quantization_config` plugin hook.

## PyTorch — one optional patch, and it is not vLLM

Be precise about the claim above: we patch no *vLLM* line, but `tools/patch_sm60_compile.py`
does edit **PyTorch** (`torch/utils/_triton.py`) to lift a hard-coded `device major >= 7` gate
in `has_triton()`. That gate is not a real capability limit — Triton 3.3 compiles and runs
ordinary pointwise/reduction kernels on sm_60; only `tl.dot` (MMA) is genuinely unavailable,
and inductor does not lower matmuls to Triton unless max_autotune is on.

It is OPT-IN and only needed to get torch.compile / CUDA graphs working on Pascal (P100).
Nothing on sm_70 or newer requires it, and PXQ4 itself does not depend on it.

## 1Cat-vLLM (Volta / sm_70 support)
https://github.com/KewaiiGamer/1Cat-vLLM

Upstream vLLM does not target compute capability 7.0. That fork carries the
sm_70 work this package depends on entirely:
  - TurboMind sm_70 W4A16 GEMM (csrc/sm70_turbomind/)
  - the FLASH_ATTN_V100 attention backend
  - the Qwen Gated-DeltaNet kernels (FlashQLA-SM70)
PXA Network contributed to getting that V100 support working. This package is a
continuation of that effort, not a fork of it.

## PXQ4
The PXQ quantization format, its CUDA kernels, and this backend: PXA Network.
