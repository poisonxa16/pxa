#!/usr/bin/env bash
# build_pxq_v18.sh -- build libpxq_<arch>_v18.so: everything v14 carries (the frozen v12b PXQ4
# kernels, the PXQ2/PXQ3 tiers, the q8 head and the pascal-ops pack) PLUS the PXQ4HQ tier.
#
# WHY v18 AND NOT v15. Library revisions are a FLAT namespace across this whole package and two
# of the next numbers are already spoken for: libpxq_sm70_v15.so is the library the shipped
# sm_70 image bakes and labels by sha256, and v16 is the concurrent PXQ2/PXQ3 tensor-core work.
# A revision number that means two different sets of object code is the failure the kernels
# README already warns about for libpxq4_sm70_v14.so, so this one takes the next free number
# rather than the next obvious one. When the two lines merge, they merge into ONE build script
# and ONE number; until then neither may reuse the other's.
#
# WHAT IS NEW IN v18, and nothing else is: three translation units (pxq4hq_kernel.cu,
# pxq4hq_mma.cu, pxq4hq_torch.cpp) carrying the tier-253 decode, its sm_70 tensor-core arena
# path and their op registrations. No v14 source file is edited, so a v18 library serves every
# checkpoint a v14 one serves, byte for byte, plus PXQ4HQ.
# Run INSIDE a container from the matching serving image, so the torch ABI
# the .so is linked against is the one that will load it (sm60 image = torch 2.7.1+cu126,
# sm70 image = torch 2.10). Mixing them gives
# `undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv` part-way through model load.
#
#   ARCH=sm60 ./build_pxq_v18.sh          -> ../kernels/libpxq_sm60_v18.so
#   ARCH=sm70 ./build_pxq_v18.sh          -> ../kernels/libpxq_sm70_v18.so
#
# NOTE ON NAMING: libpxq4_sm70_v14.so already exists in kernels/ and is an unrelated PXQ4-only
# build. This artifact is libpxq_*_v14.so -- no "4" -- because it is the first library that is
# not PXQ4-only. Do not rename either one.
#
# -use_fast_math IS FORBIDDEN here for the same reason it is forbidden in CMakeLists.txt: it
# enables contraction and reassociation that would silently change the fp32 fold order, and
# the whole correctness argument of this package is that the arithmetic is bit-identical to
# the shipping llama.cpp kernels.
set -e
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SRC_DIR"

ARCH="${ARCH:-sm70}"
case "$ARCH" in
  sm60)
    # The serving image's own TORCH_CUDA_ARCH_LIST is "6.0;7.0" for this variant, so the
    # library carries both cubins: a Pascal box may host a Volta card and vice versa.
    GEN="-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70" ;;
  sm70)
    GEN="-gencode arch=compute_70,code=sm_70" ;;
  *) echo "ARCH must be sm60 or sm70, got '$ARCH'" >&2; exit 2 ;;
esac

OUT_DIR="${PXQ_BUILD_OUT:-$SRC_DIR/../kernels}"
BUILD_DIR="${PXQ_BUILD_DIR:-$SRC_DIR/../build-pxq-v18-$ARCH}"
JOBS="${MAX_JOBS:-6}"
TORCH=$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')
ABI=$(python -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
NVCC_COMMON="-O3 -std=c++17 -Xcompiler -fPIC -lineinfo -Wno-deprecated-gpu-targets
      --expt-relaxed-constexpr -D_GLIBCXX_USE_CXX11_ABI=$ABI
      -I$SRC_DIR -I$TORCH/include -I$TORCH/include/torch/csrc/api/include"

mkdir -p "$BUILD_DIR" "$OUT_DIR"
cd "$BUILD_DIR"
rm -f k.o m.o t.o k23.o t23.o q8.o tq8.o pop.o tpop.o tmf.o k4hq.o m4hq.o t4hq.o

# The tensor-core TU is sm_70 ONLY: it uses wmma m16n16k16, which does not exist on sm_60 and
# will not compile for it. pxq4_mma_supported() gates the launch on major >= 7, so the sm60
# library simply never reaches this code on a Pascal card -- but the SYMBOL must be present
# because pxq4_kernel_torch.cpp references it unconditionally.
(
  set -x
  nvcc $NVCC_COMMON $GEN                                     -c "$SRC_DIR/pxq4_kernel.cu"  -o k.o &
  nvcc $NVCC_COMMON -gencode arch=compute_70,code=sm_70      -c "$SRC_DIR/pxq4_mma.cu"     -o m.o &
  # -ffp-contract=off applies to the HOST code of this TU only (-Xcompiler), which is the
  # self-test's oracle. It pins the replay to a fixed sequence of separately-rounded
  # operations instead of whatever the host compiler chose to fuse, so a self-test failure is
  # reproducible and means something. Device code is untouched: device FMA contraction is
  # --fmad, still at its default, so the shipped fold is the engine's fold.
  nvcc $NVCC_COMMON $GEN -Xcompiler -ffp-contract=off        -c "$SRC_DIR/pxq23_kernel.cu" -o k23.o &
  # Same host-contraction pin, same reason: this TU also carries a self-test oracle.
  nvcc $NVCC_COMMON $GEN -Xcompiler -ffp-contract=off        -c "$SRC_DIR/pxq_q8.cu"      -o q8.o &
  # pascal-ops: the hand-fused small-op pack (GemmaRMSNorm, RMSNormGated, SwiGLU). Built for
  # BOTH arches like the other portable TUs -- it has no tensor-core code, so unlike
  # pxq4_mma.cu it is not sm_70-only. No -ffp-contract=off pin here: this TU carries no host
  # oracle, and its DEVICE arithmetic is pinned per-operation with __fadd_rn / __fmul_rn
  # rather than by a compiler flag, which is what the byte gate needs.
  nvcc $NVCC_COMMON $GEN                                     -c "$SRC_DIR/pxa_pascal_ops.cu" -o pop.o &
  # PXQ4HQ (tier 253). Same host-contraction pin as the other TUs that carry a self-test
  # oracle, and for the same reason: it pins the replay to a fixed sequence of separately
  # rounded operations so a self-test failure is reproducible and means something.
  nvcc $NVCC_COMMON $GEN -Xcompiler -ffp-contract=off        -c "$SRC_DIR/pxq4hq_kernel.cu" -o k4hq.o &
  # The HQ tensor-core TU, sm_70 ONLY for the same reason pxq4_mma.cu is: wmma m16n16k16 does
  # not exist on sm_60 and will not compile for it. pxq4hq_mma_supported() gates the launch on
  # major >= 7, so an sm60 library never reaches this code on a Pascal card -- but the SYMBOL
  # must be present, because pxq4hq_kernel.cu and pxq4hq_torch.cpp reference it unconditionally.
  nvcc $NVCC_COMMON -gencode arch=compute_70,code=sm_70      -c "$SRC_DIR/pxq4hq_mma.cu"    -o m4hq.o &
  wait
)

g++ -O3 -std=c++17 -fPIC -Wall -Wextra -Wno-unused-parameter -c \
    -D_GLIBCXX_USE_CXX11_ABI=$ABI -I"$SRC_DIR" -I$TORCH/include \
    -I$TORCH/include/torch/csrc/api/include -I/usr/local/cuda/include \
    "$SRC_DIR/pxq4_kernel_torch.cpp" -o t.o
g++ -O3 -std=c++17 -fPIC -Wall -Wextra -Wno-unused-parameter -c \
    -D_GLIBCXX_USE_CXX11_ABI=$ABI -I"$SRC_DIR" -I$TORCH/include \
    -I$TORCH/include/torch/csrc/api/include -I/usr/local/cuda/include \
    "$SRC_DIR/pxq23_torch.cpp" -o t23.o
g++ -O3 -std=c++17 -fPIC -Wall -Wextra -Wno-unused-parameter -c \
    -D_GLIBCXX_USE_CXX11_ABI=$ABI -I"$SRC_DIR" -I$TORCH/include \
    -I$TORCH/include/torch/csrc/api/include -I/usr/local/cuda/include \
    "$SRC_DIR/pxq4hq_torch.cpp" -o t4hq.o
# The fused MoE decode block's binding TU. Its KERNELS are not compiled here: they are
# instantiated inside pxq4_kernel.cu and pxq23_kernel.cu, because pxq4_book_g / pxq2_book_g /
# pxq3_book_g are `static __device__` and therefore TU-local -- a fused kernel compiled in its
# own TU would bind to its own never-uploaded copy of the tables. This file is host-only.
g++ -O3 -std=c++17 -fPIC -Wall -Wextra -Wno-unused-parameter -c \
    -D_GLIBCXX_USE_CXX11_ABI=$ABI -I"$SRC_DIR" -I$TORCH/include \
    -I$TORCH/include/torch/csrc/api/include -I/usr/local/cuda/include \
    "$SRC_DIR/pxq_moe_fused_torch.cpp" -o tmf.o
g++ -O3 -std=c++17 -fPIC -Wall -Wextra -Wno-unused-parameter -c \
    -D_GLIBCXX_USE_CXX11_ABI=$ABI -I"$SRC_DIR" -I$TORCH/include \
    -I$TORCH/include/torch/csrc/api/include -I/usr/local/cuda/include \
    "$SRC_DIR/pxq_q8_torch.cpp" -o tq8.o
g++ -O3 -std=c++17 -fPIC -Wall -Wextra -Wno-unused-parameter -c \
    -D_GLIBCXX_USE_CXX11_ABI=$ABI -I"$SRC_DIR" -I$TORCH/include \
    -I$TORCH/include/torch/csrc/api/include -I/usr/local/cuda/include \
    "$SRC_DIR/pxa_pascal_ops_torch.cpp" -o tpop.o

g++ -shared -o "$OUT_DIR/libpxq_${ARCH}_v18.so" k.o m.o k23.o q8.o pop.o k4hq.o m4hq.o t.o t23.o tq8.o tpop.o tmf.o t4hq.o \
    -L$TORCH/lib -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -L/usr/local/cuda/lib64 -lcudart
echo BUILD_OK
ls -la "$OUT_DIR/libpxq_${ARCH}_v18.so"
md5sum "$OUT_DIR/libpxq_${ARCH}_v18.so"
# Op inventory. The ops are registered through TORCH_LIBRARY string schemas, not exported C
# symbols, so `nm -D` finds nothing -- scan printable strings instead (see kernels README).
strings -a "$OUT_DIR/libpxq_${ARCH}_v18.so" \
  | grep -E '^(mmv_out|mma_out|moe_mmv_out|f16_mmv_out|gemm2d_out|pxq2_mmv_out|pxq3_mmv_out|pxq2_moe_mmv_out|pxq3_moe_mmv_out|pxq2_dequant_out|pxq3_dequant_out|pxq2_linear_out|pxq3_linear_out|pxq_selftest|pxq_set_book|pxq_version|q8_mmv_out|q8_dequant_out|q8_linear_out|q8_selftest|gemma_add_rms_norm_out|gemma_rms_norm_out|rms_norm_gated_out|rms_norm_out|fused_add_rms_norm_out|silu_and_mul_out|pascal_ops_version|moe_gateup_glu_out|pxq2_moe_gateup_glu_out|pxq3_moe_gateup_glu_out|moe_down_fold_out|pxq2_moe_down_fold_out|pxq3_moe_down_fold_out|moe_down_part_out|pxq2_moe_down_part_out|pxq3_moe_down_part_out|moe_slot_fold_out|moe_fused_version|pxq4hq_mmv_out|pxq4hq_mma_out|pxq4hq_dequant_out|pxq4hq_linear_out|pxq4hq_set_tables|pxq4hq_selftest|pxq4hq_version)$' \
  | sort -u
# NOTE ON WHICH NAMES MAY APPEAR HERE. The linker merges string literals that are SUFFIXES
# of longer ones, so "mmv_out" has no storage of its own once "f16_mmv_out" exists, and
# neither does "rms_norm_out" once "gemma_rms_norm_out" does. Listing such a name here
# produces a FALSE BUILD FAILURE -- it did, on the first run of this guard. Every entry
# below is a name that is not a suffix of any other registered op, which is what makes its
# absence meaningful. The definitive check is still a load probe (import torch, load the
# .so, hasattr each op); this is the cheap one that runs on every build.
PXQ_EXPECT_OPS="${PXQ_EXPECT_OPS:-f16_mmv_out gemm2d_out pxq2_mmv_out pxq3_mmv_out pxq_version \
  gemma_add_rms_norm_out gemma_rms_norm_out rms_norm_gated_out \
  silu_and_mul_out pascal_ops_version \
  pxq4hq_mmv_out pxq4hq_mma_out pxq4hq_dequant_out pxq4hq_linear_out pxq4hq_set_tables \
  pxq4hq_selftest pxq4hq_version}"

# HARD CHECK, not a printout. The inventory above is the only thing that can catch a
# link-stage omission: a .cu that compiles and is linked gives a bigger library and a
# BUILD_OK even when the host TU carrying its TORCH_LIBRARY_FRAGMENT was left out of the
# link, in which case the kernels are present and NONE of the ops are registered. That
# happened once (pascal-ops pack, tpop.o missing from the g++ -shared line) and it was
# found by probing the loaded library, not by reading this build's output -- because the
# output was a list nobody had an expected value for. So the expected names live here now
# and a missing one fails the build.
MISSING=""
for op in $PXQ_EXPECT_OPS; do
  strings -a "$OUT_DIR/libpxq_${ARCH}_v18.so" | grep -qx "$op" || MISSING="$MISSING $op"
done
if [ -n "$MISSING" ]; then
  echo "BUILD FAILED: these ops are absent from the library:$MISSING" >&2
  echo "  (kernels can link while their host TU does not -- check every *_torch.cpp" >&2
  echo "   object is on the g++ -shared line, not just the .cu objects)" >&2
  exit 3
fi
echo "OP INVENTORY OK: every expected op is registered."

# Arch inventory: prove the cubins the image needs are actually in there.
cuobjdump --list-elf "$OUT_DIR/libpxq_${ARCH}_v18.so" | sed -n 's/.*\.\(sm_[0-9]*\)\..*/\1/p' | sort -u
