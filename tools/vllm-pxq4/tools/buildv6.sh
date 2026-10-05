#!/bin/bash
set -euo pipefail
# PXA_MODELS_DIR should be an absolute path here: it is bind-mounted into the container
# below at the same path (the ./models default is meant for non-docker use).
MODELS_DIR="${PXA_MODELS_DIR:-./models}"
SRC=$MODELS_DIR/pxa-int-v6/src
OUT=$MODELS_DIR/pxa-int-v6/site/pxq4_vllm/_lib
BUILD=$MODELS_DIR/pxa-int-v6/build
mkdir -p "$BUILD" "$OUT"
docker run --rm \
  -v "$MODELS_DIR":"$MODELS_DIR" \
  -w "$BUILD" \
  -e SRC="$SRC" -e OUT="$OUT" \
  --entrypoint /bin/bash \
  ${VLLM_IMAGE:-vllm/vllm-openai:latest} -lc '
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
