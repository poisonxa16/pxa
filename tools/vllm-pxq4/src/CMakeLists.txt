# CMakeLists.txt — builds libpxq4_sm70.so, the standalone PXQ4 torch extension.
#
# It links against libtorch ONLY. It does not include a single vLLM header, does not link
# against vLLM's _C.abi3.so, and does not require rebuilding vLLM: the operators are published
# through TORCH_LIBRARY(pxq4, ...) and picked up at runtime with torch.ops.load_library().
#
# Build (see build.sh, which does this inside a throwaway container from the production image):
#   cmake -B build -S . -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
#   cmake --build build -j
#
# The production container's / is 100% full and must never be written to. build.sh keeps every
# byte of the build tree under /mnt/models.

cmake_minimum_required(VERSION 3.18)
project(pxq4_sm70 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# V100 only. sm_70 has fp16 HMMA tensor cores and dp4a but NO bf16 and NO fp8, which is why the
# whole design is fp16-in / fp16-out with fp32 accumulation.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

find_package(Torch REQUIRED)
# pxq4_kernel_torch.cpp is compiled by the HOST compiler but includes cuda_runtime_api.h
# (through pxq4_kernel_launch.h, for cudaStream_t). Without CUDA::cudart on the target the
# CXX compile does not get the CUDA include directory and fails to find that header.
find_package(CUDAToolkit REQUIRED)

# The ABI must match the torch wheel exactly or every std::string across the boundary is
# silently mismatched at link time. TorchConfig sets this; build.sh also passes it explicitly
# from torch._C._GLIBCXX_USE_CXX11_ABI so a stale CMake cache cannot get it wrong.
add_library(pxq4_sm70 SHARED
  pxq4_kernel.cu
  pxq4_kernel_torch.cpp
)

target_include_directories(pxq4_sm70 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pxq4_sm70 PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)

# -use_fast_math is FORBIDDEN. It enables contraction and reassociation that would silently
# change the fp32 fold order, and the entire correctness argument for this port is that the
# arithmetic is bit-identical to the shipping llama.cpp kernels. Same reason -ffast-math is
# absent from the host flags.
target_compile_options(pxq4_sm70 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr -lineinfo -Xcompiler=-fPIC>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -Wno-unused-parameter>
)

# PXQ4_CANON_V2 selects the inner accumulation shape and is a re-baselining event, never a
# runtime flag. It must match the engine build that produced the artifact (default 0).
if(PXQ4_CANON_V2)
  target_compile_definitions(pxq4_sm70 PRIVATE PXQ4_CANON_V2=1)
endif()

# The runtime package loads exactly this name: src/pxq4_vllm/_lib/libpxq4_sm70.so (plan §4).
set_target_properties(pxq4_sm70 PROPERTIES OUTPUT_NAME "pxq4_sm70" PREFIX "lib")
