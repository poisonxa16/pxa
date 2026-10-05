// pxqn-host-isa.h -- CLOSED (libggml-pxqn). Pre-included (nvcc -include) into every CUDA translation unit of the
// baseline-x86-64 build of libggml-pxqn (PXA_PXQN_BASELINE_X86, ggml/src/CMakeLists.txt).
//
// It exists for ccache: ccache's key for an nvcc compile is the preprocessed text, and it does NOT see the host
// compiler flags that arrive through -Xcompiler. Two builds that differ only in those flags (baseline x86-64 vs the
// engine's -mavx -mavx2 -mfma -mf16c) therefore share cache entries, and the shared cache hands the baseline build
// AVX objects (and the reverse). One declaration that only the baseline build pre-includes makes the preprocessed
// text, and so the key, differ. Nothing here may change code generation.
#pragma once
typedef int pxa_pxqn_host_isa_baseline_x86_t;
