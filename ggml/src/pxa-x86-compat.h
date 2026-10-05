// pxa-x86-compat.h -- pre-included (nvcc -include) into every CUDA translation unit of the lib-compat build (-DPXA_X86_COMPAT=ON,
// ggml/src/CMakeLists.txt).
//
// It exists for ccache: ccache's key for an nvcc compile is the preprocessed text, and it does NOT see the host compiler flags that
// arrive through -Xcompiler. The lib-compat build (baseline x86-64) and the fast build (-mavx -mavx2 -mfma -mf16c) differ only in
// those flags, so they would share cache entries and the shared cache would hand the compat build AVX2 objects (and the reverse).
// One declaration that only the compat build pre-includes makes the preprocessed text, and so the key, differ.
// Nothing here may change code generation.
#pragma once
typedef int pxa_x86_compat_build_t;
