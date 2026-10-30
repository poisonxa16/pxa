// win-compat/pxa-msvc-attr.h -- included by ggml-cuda.cu for its provenance canary: under MSVC (no GNU attributes) the
// __attribute__ spelling becomes empty for the rest of that translation unit. Everywhere else this header is empty.
#pragma once
#if defined(_MSC_VER) && !defined(__GNUC__) && !defined(__clang__)
#define __attribute__(x)
#endif
