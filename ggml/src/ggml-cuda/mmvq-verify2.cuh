// mmvq-verify2.cuh -- PXA_MOE_VERIFY2: expert-grouped MoE GEMV for 2..4-token verify batches.
//
// At a speculative/MTP verify width of Ny tokens (Ny = 2 for MTP n_max=1, 3 for n_max=2) the
// routed-expert decode path (fast-TG in ggml_cuda_moe_up_gate_unary, and the standalone
// MUL_MAT_ID) runs one id-GEMV launch PER TOKEN, so an expert routed to by two tokens of the
// same verify batch is streamed from DRAM twice, and every layer pays 2*Ny GEMV launches
// instead of 2. This path launches ONCE per projection over all (token, slot) pairs: the block
// for the first occurrence of an expert gathers every later (token, slot) routed to the same
// expert and computes them as extra columns of one GEMV, reading each weight block once; the
// blocks of the later occurrences exit at once.
//
// Numerics: one warp per row (nwarps = 1, one row per block) -- exactly the reduction shape the
// per-token id-GEMV uses at n_ids >= 2 -- and each column keeps its own accumulator fed in the
// same k-block order with the same vec_dot, so every output is meant to be bit-identical to the
// per-token path (verify with test-backend-ops and the in-engine A/B). Default OFF.
//
//   PXA_MOE_VERIFY2=1            enable (unset / 0 = off, the shipped behaviour)
//   PXA_MOE_VERIFY2_MAX_NY=N     widest verify batch served, 2..4 (default 4)
//   PXA_MOE_VERIFY2_LOG=1        one stderr line per call site the first time it engages
#pragma once

#include "common.cuh"

#define PXA_MV2_MAXC 4   // widest column group one block computes (= widest verify batch served)

struct pxa_mv2_args {
    const void * vx_u;        // weights (plain GEMV) or the UP half (fused)
    const void * vx_g;        // GATE half (fused) or nullptr (plain)
    const char * vy;          // q8_1 activations
    int64_t      y_st;        // bytes between tokens
    int64_t      y_sk;        // bytes between slots of one token (0 = all slots share the token's row)
    char       * dst;
    int64_t      d_st;        // bytes between tokens
    int64_t      d_sk;        // bytes between slots
    const char * ids;
    int64_t      ids_nb0;
    int64_t      ids_nb1;
    const char * bias;        // plain only: per-expert bias rows (ADD_ID), or nullptr
    int64_t      bias_nb1;
    uint64_t     nb02;        // bytes between experts in the weight stack
    int          ncols_x;     // K
    int          nrows_x;     // rows computed
    int          nrows_y;     // padded K of the activation rows
    int          ny;          // tokens in the verify batch (2..PXA_MV2_MAXC)
    int          n_ids;       // routed slots per token
    int          unary_op;    // ggml_unary_op for the fused epilogue
    float        limit;
};

bool pxa_moe_verify2_enabled();
int  pxa_moe_verify2_max_ny();
// PXA_MOE_VERIFY2_TILE_W1=1 (needs PXA_MOE_VERIFY2=1 and a PXA_MOE_VERIFY2_TILE): the one-token decode MoE
// up/gate + down use the tiled kernel as well (same per-row arithmetic, so bit-identical)
bool pxa_moe_verify2_tile_w1();
bool pxa_moe_verify2_type_ok(ggml_type type);
void pxa_moe_verify2_log(const char * site, ggml_type type, int ny, int n_ids, int nrows);
// true = launched; false = declined (type/shape not served), caller runs its own path
bool pxa_moe_verify2_launch(ggml_type type, const pxa_mv2_args & a, cudaStream_t stream);
