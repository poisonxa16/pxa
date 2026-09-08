#!/usr/bin/env bash
# build.sh — build libpxq4_sm70.so against the production container's torch, WITHOUT touching
# the production container.
#
# CONSTRAINTS THIS SCRIPT EXISTS TO RESPECT:
#   * the running vLLM container (vllm-qwen38-27b-cyber-1) is someone else's production
#     service. It is never stopped, restarted, or written to. We only read its image name.
#   * that container's overlay / is 100% full (207 G used, 0 avail), so nothing may be
#     installed into the image and no build tree may live on /.
#   * the DGX host / is also full. Everything lives under /mnt/models.
#
# Usage, from the DGX:  bash build.sh
set -euo pipefail

SRC="${PXQ4_SRC:-/mnt/models/pxa-vllm-pxq4/csrc}"
OUT="${PXQ4_OUT:-/mnt/models/pxa-vllm-pxq4/site/pxq4_vllm/_lib}"
BUILD="${PXQ4_BUILD:-/mnt/models/pxa-vllm-pxq4/build}"

# Same image as the running service, so the torch/nvcc/gcc it links against are byte-identical
# to the ones the server will load it into. --rm, no GPU request, no lease: nvcc does not need
# a device to compile for sm_70.
# Build against the SAME image the server runs, so torch/nvcc/gcc are byte-identical to what
# will dlopen the result. PXQ4_IMAGE names it directly; PXQ4_REF_CONTAINER derives it from a
# running container. The container name below is OUR box's -- it is a fallback, not a
# requirement, and anyone else must set one of the two variables.
IMAGE="${PXQ4_IMAGE:-}"
if [ -z "$IMAGE" ]; then
  REF="${PXQ4_REF_CONTAINER:-vllm-qwen38-27b-cyber-1}"
  IMAGE="$(docker inspect -f '{{.Config.Image}}' "$REF" 2>/dev/null || true)"
fi
if [ -z "$IMAGE" ]; then
  echo "ERROR: could not resolve the vLLM image." >&2
  echo "  set PXQ4_IMAGE=<image>            (e.g. PXQ4_IMAGE=vllm/vllm-openai:v0.x)" >&2
  echo "  or  PXQ4_REF_CONTAINER=<name>     (a RUNNING vLLM container to copy the image from)" >&2
  exit 1
fi
echo "building against image: $IMAGE"

mkdir -p "$BUILD" "$OUT"

# Mount the common ancestor of SRC/OUT/BUILD rather than a hardcoded /mnt/models, so the
# script works wherever the tree actually lives.
MOUNT_ROOT="${PXQ4_MOUNT_ROOT:-$(printf '%s\n%s\n%s\n' "$SRC" "$OUT" "$BUILD" \
  | sed 's|/[^/]*$||' | sort | awk 'NR==1{p=$0} {while (index($0,p)!=1) sub(/\/[^/]*$/,"",p)} END{print p}')}"
[ -z "$MOUNT_ROOT" ] && MOUNT_ROOT=/
echo "mounting: $MOUNT_ROOT"

docker run --rm \
  -v "$MOUNT_ROOT":"$MOUNT_ROOT" \
  -w "$BUILD" \
  -e SRC="$SRC" -e OUT="$OUT" \
  --entrypoint /bin/bash \
  "$IMAGE" -lc '
    set -euo pipefail
    PY=/opt/vllm-venv/bin/python
    PREFIX="$($PY -c "import torch;print(torch.utils.cmake_prefix_path)")"
    ABI="$($PY -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")"
    echo "torch $($PY -c "import torch;print(torch.__version__)")  cxx11abi=$ABI"
    cmake -S "$SRC" -B . \
      -DCMAKE_PREFIX_PATH="$PREFIX" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=$ABI"
    cmake --build . -j "$(nproc)"
    cp -v libpxq4_sm70.so "$OUT/"
  '

echo
echo "smoke test (still no GPU work — just proves the ops register):"
docker run --rm -v /mnt/models:/mnt/models --entrypoint /bin/bash "$IMAGE" -lc "
  /opt/vllm-venv/bin/python - <<'PY'
import torch
torch.ops.load_library('$OUT/libpxq4_sm70.so')
print('pxq4 version        :', torch.ops.pxq4.version())
print('pxq4 mmv_max_m      :', torch.ops.pxq4.mmv_max_m())
print('pxq4 mmv_supported  :', {K: torch.ops.pxq4.mmv_supported(K) for K in (4352, 5120, 6144, 17408)})
print('pxq4 builtin tables :')
print(torch.ops.pxq4.builtin_tables())
PY
"
