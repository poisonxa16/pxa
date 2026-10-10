// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
// pxa-cold-dev.cuh -- the expert-cache cold path's device view (shared by pxa-xcache.cuh and the PXQN library table).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

struct pxa_cold_dev {
    volatile uint64_t * req; volatile uint32_t * done;
    int32_t * ids; float * cur; float * out;
    uint32_t * seq; unsigned long long * stat;
};

// an optional cold-wait route supplied by libggml-pxqn (pxqn-api.cuh); returns false when it is not taken (no library,
// or the library's switch is off), and the caller then runs its own wait
bool ggml_cuda_pxqn_cold_wait(const pxa_cold_dev & s, const char * ids, size_t inb0, size_t inb1, float * dst, size_t dnb1, size_t dnb2,
                              int n_embd, int n_used, int n_tok, long long timeout_clk, const int32_t * ticket, cudaStream_t stream);
