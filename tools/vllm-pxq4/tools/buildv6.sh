#!/bin/bash
set -euo pipefail
SRC=/mnt/models/pxa-int-v6/src
OUT=/mnt/models/pxa-int-v6/site/pxq4_vllm/_lib
BUILD=/mnt/models/pxa-int-v6/build
mkdir -p "$BUILD" "$OUT"
docker run --rm \
  -v /mnt/models:/mnt/models \
  -w "$BUILD" \
  -e SRC="$SRC" -e OUT="$OUT" \
  --entrypoint /bin/bash \
  kewaii/vllm:latest -lc '
    set -euo pipefail
    PY=/opt/vllm-venv/bin/python
    PREFIX="$($PY -c "import torch;print(torch.utils.cmake_prefix_path)")"
    ABI="$($PY -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")"
    cmake -S "$SRC" -B . \
      -DCMAKE_PREFIX_PATH="$PREFIX" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=$ABI" > cm.log 2>&1
    cmake --build . -j "$(nproc)" -- CUDA_FLAGS_EXTRA=1 2>&1 | tail -5
    cp -v libpxq4_sm70.so "$OUT/"
  '
echo BUILD_DONE
