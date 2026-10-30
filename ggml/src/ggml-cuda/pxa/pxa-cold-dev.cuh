// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
// pxa-cold-dev.cuh -- the expert-cache cold path's device view (shared by pxa-xcache.cuh and the PXQN library table).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
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

// The cold wait never gives up with a result (2026-10-26): a wait longer than PXA_XCACHE_ASYNC_TIMEOUT_S (timeout_clk) is
// counted as slow while the card keeps waiting; one PXQN_COLD_HANG_MULT times that long (default 20 x 30 s = 10 min) is a hung host
// worker, not a slow one: the kernel traps, the next CUDA call reports the failure and the process stops, instead of hanging the card
// forever or handing on rows nobody computed (until 2026-10-26 those rows were written as zero: silent wrong output under host load).
// The hang verdict also needs PXQN_COLD_HANG_FLOOR_CLK SM clocks (4e11: ~5 min at 1.38 GHz, longer at lower clocks), so a short slow-wait
// line (a test's 1 us, an operator's 1 s) never turns a merely slow step into a stop.
#define PXQN_COLD_HANG_MULT 20
#define PXQN_COLD_HANG_FLOOR_CLK 400000000000LL
#ifdef __CUDACC__
static __device__ __forceinline__ bool pxa_cold_hung(long long waited_clk, long long timeout_clk) {
    return waited_clk > PXQN_COLD_HANG_FLOOR_CLK && waited_clk / PXQN_COLD_HANG_MULT > timeout_clk;   // no overflow for any timeout
}
static __device__ __noinline__ void pxa_cold_hang_trap(void) {
    printf("PXA_XCACHE_ASYNC: a cold-expert wait ran %d x PXA_XCACHE_ASYNC_TIMEOUT_S: the host worker is hung; stopping (no output is produced from it)\n",
           PXQN_COLD_HANG_MULT);
    __trap();
}
#endif
