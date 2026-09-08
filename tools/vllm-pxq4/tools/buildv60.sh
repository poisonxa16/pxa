#!/bin/bash
# buildv60.sh — sm_60 (Tesla P100) build of the PXQ4 torch extension.
#
# The kewaii/vllm image CANNOT be used here: its torch 2.10+cu128 ships no sm_60 kernel
# image (arch list sm_70..sm_120), so nothing built against it can even allocate on a P100.
# The last official torch with Pascal support is 2.7.1+cu126; a venv holding it lives at
# /path/to/build-sm60/venv (rebuild: python3 -m venv + pip install torch==2.7.1
# --index-url https://download.pytorch.org/whl/cu126, plus numpy). The pxa-sm60-dev image is
# nvidia/cuda:12.8.1-devel-ubuntu24.04 + python3/venv/cmake/ninja (Dockerfile in
# /path/to/build-sm60/dockerctx).
#
# TORCH_CUDA_ARCH_LIST=6.0 is REQUIRED: torch cmake overrides CMAKE_CUDA_ARCHITECTURES with
# its no-GPU default list (5.0;8.0;8.6;...) when the env var is absent, silently producing a
# library with no sm_60 cubin at all. Verify with cuobjdump --list-elf: exactly one member,
# pxq4_kernel.sm_60.cubin.
#
# Gates (all must pass on a P100 before any number is quoted):
#   gpu_gate_v6.py <lib> --stress     on-device parity: fused-mt == monolithic, 400-launch stress
#   gate_linear.py <lib>              run with DEFAULT env: PXQ4_MMV_SLICE_MAX=8 reroutes
#                                     M=9..16 onto dequant+cuBLAS, which is not bit-exact to the
#                                     gate's sliced-mmv reference and FAILS it by design
#   xarch_dump.py on P100 (sm_60 lib) and V100 (shipping sm_70 lib) with CPU-seeded inputs,
#                                     then bitwise-compare the two dumps (2026-08-19: 72/72 equal)
#
# P100 dispatch note: the split-vs-mono crossover is 2*SM = 112 on a 56-SM P100, which puts
# tp4_gate_up (136 panels) on mono at M=1. PXQ4_MMV_SPLIT_MAX_BLOCKS=300 forces it back onto
# split: measured 116.9us -> 89.9us on P100. Harmless elsewhere; consider it for any P100 serve.
set -euo pipefail
SRC=/path/to/engine-repo/tools/vllm-pxq4/src
OUT=/path/to/build-sm60
BUILD=/path/to/build-sm60/build60
mkdir -p "$BUILD"
docker run --rm -v /path/to/data:/path/to/data -w "$BUILD" pxa-sm60-dev bash -lc '
  set -euo pipefail
  PY=/path/to/build-sm60/venv/bin/python
  PREFIX=$($PY -c "import torch;print(torch.utils.cmake_prefix_path)")
  ABI=$($PY -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
  export TORCH_CUDA_ARCH_LIST=6.0
  cmake -S '"$SRC"' -B . -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=$ABI" > cm.log 2>&1
  cmake --build . -j $(nproc) 2>&1 | tail -3
  cp libpxq4_sm70.so '"$OUT"'/libpxq4_sm60.so
  cuobjdump --list-elf '"$OUT"'/libpxq4_sm60.so
'
echo BUILD_DONE
