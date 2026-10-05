set -e
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SRC_DIR"
OUT_DIR="${PXQ4_BUILD_OUT:-$SRC_DIR/../kernels}"
BUILD_DIR="${PXQ4_BUILD_DIR:-$SRC_DIR/../build-v12b}"
TORCH=$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')
ABI=$(python -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
NVCC="nvcc -O3 -std=c++17 -arch=sm_70 -Xcompiler -fPIC -lineinfo -Wno-deprecated-gpu-targets
      --expt-relaxed-constexpr -D_GLIBCXX_USE_CXX11_ABI=$ABI
      -I. -I$TORCH/include -I$TORCH/include/torch/csrc/api/include"
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
rm -f k.o m.o t.o
$NVCC -c "$SRC_DIR/pxq4_kernel.cu" -o k.o
$NVCC -c "$SRC_DIR/pxq4_mma.cu"    -o m.o -Xptxas -v 2>&1 | grep -E "k_pxq4_mma|registers|spill" || true
g++ -O3 -std=c++17 -fPIC -Wall -Wextra -Wno-unused-parameter -c \
    -D_GLIBCXX_USE_CXX11_ABI=$ABI -I"$SRC_DIR" -I$TORCH/include \
    -I$TORCH/include/torch/csrc/api/include -I/usr/local/cuda/include \
    "$SRC_DIR/pxq4_kernel_torch.cpp" -o t.o
mkdir -p "$OUT_DIR"
g++ -shared -o "$OUT_DIR/libpxq4_sm70_v12b.so" k.o m.o t.o \
    -L$TORCH/lib -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -L/usr/local/cuda/lib64 -lcudart
echo BUILD_OK
ls -la "$OUT_DIR/libpxq4_sm70_v12b.so"
md5sum "$OUT_DIR/libpxq4_sm70_v12b.so"
strings -a "$OUT_DIR/libpxq4_sm70_v12b.so" | grep -E '^(mmv_out|mma_out|f16_mmv_out|gemm2d_out|moe_mmv_out)$' | sort
