#pragma once

#include "../../common.cuh"

// ---------------------------------------------------------------------------------------------
// The PXA core: the flash-attention route planner.
//
// WHY THIS FILE EXISTS. Choosing which flash-attention kernel runs a node used to be written out
// twice: once in ggml_cuda_flash_attn_ext() (the dispatcher) and once in
// ggml_cuda_fattn_is_supported() (the answer the scheduler gets from ggml_backend_supports_op).
// The two texts were kept in step by hand, with "must mirror this exactly" comments standing in
// for a compiler check, and they drifted: a lever that rewrites the node was applied by one and
// invisible to the other, and the dispatcher read the compute capability of whichever device
// happened to be current rather than of the device being asked about.
//
// There is now one text. Both callers ask pxa_fa_plan_node(); the graph builder's probe
// (pxa_fa_node_runs_on_gpu() in src/llama-build-context.cpp) reaches the same text through the
// existing ggml_backend_supports_op() path, so no new cross-library entry point is needed.
//
// The two callers do not ask quite the same question, and the places where they genuinely differ
// are marked in fa-route.cu with `q.kind == PXA_FA_QUERY_DISPATCH` and a note each. Those are
// routes the dispatcher SUBSTITUTES for one the support query already accepted (a kernel that
// declines a shape and falls through to its neighbour), so the support answer is unchanged by
// them. Everything else is shared, and drift is no longer possible without an edit you can see.
// ---------------------------------------------------------------------------------------------

// One id per implementation a FLASH_ATTN_EXT node can end up in. Ids are stable: the route census
// prints them and a run's routes are meant to be comparable with another run's.
enum pxa_fa_route {
    PXA_FA_ROUTE_NONE = 0,     // no kernel serves this node (the graph builder then builds the chain)
    PXA_FA_ROUTE_DSA,          // sparse attention over an index list in src[5]
    PXA_FA_ROUTE_DSA_UNSERVED, // carries an index list no dense kernel honours -- a hard error
    PXA_FA_ROUTE_VEC_F16,
    PXA_FA_ROUTE_VEC_F32,
    PXA_FA_ROUTE_TILE_F16,
    PXA_FA_ROUTE_TILE_F32,
    PXA_FA_ROUTE_TILE_V2,      // PXA_FA_TILE_V2 -- same arithmetic, a different tile schedule
    PXA_FA_ROUTE_TILE_BIG,     // PXA_FA_TILE_512 -- 512/512 and 576/512 on sm_60 / sm_70
    PXA_FA_ROUTE_D512_MMA,     // PXA_FA_D512_VOLTA=1
    PXA_FA_ROUTE_D512_TILE,    // PXA_FA_D512_VOLTA=2 (the shipping default under ENHANCE)
    PXA_FA_ROUTE_D256_TILE,    // PXA_FA_D256_VOLTA_TILE -- the same vendored tile kernel at head 256
    PXA_FA_ROUTE_VOLTA_MMA,    // PXA_FA_MMA_VOLTA -- the vendored m8n8k4 MMA kernel
    PXA_FA_ROUTE_WMMA_F16,
    PXA_FA_ROUTE_MMA_F16,
    PXA_FA_ROUTE_MMA_NEW,
    PXA_FA_ROUTE_COUNT
};

enum pxa_fa_query_kind {
    PXA_FA_QUERY_SUPPORT  = 0, // "would this backend run this node?" -- ggml_backend_supports_op
    PXA_FA_QUERY_DISPATCH = 1  // "run it" -- the node is about to be handed to a kernel
};

struct pxa_fa_query_t {
    const ggml_tensor * node;  // the node the kernel will see: on the dispatch side, AFTER
                               // pxa_fa_prepare_node()
    int32_t             n_swa; // the sliding window as the ROUTER sees it. Kept separate because
                               // the SWA rewrite below zeroes op_params[4] on the node it hands
                               // the kernel while routing has always used the original value.
    pxa_fa_query_kind   kind;
};

struct pxa_fa_plan_t {
    pxa_fa_route route;     // which implementation takes the node
    bool         supported; // ... and does that implementation accept this shape/dtype?
    const char * why;       // one short phrase, for the census and for a future --why-route dump
};

// The whole decision, for one node, on one device (ctx.device -- never "the current device").
pxa_fa_plan_t pxa_fa_plan_node(ggml_backend_cuda_context & ctx, const pxa_fa_query_t & q);

const char * pxa_fa_route_name(pxa_fa_route route);

// PXA_FA_SWA_SLICE / PXA_FA_SWA_KEEP rewrite the node the kernel sees. Dispatch side only; the
// scratch must outlive the kernel call. Returns dst itself when no rewrite applies.
struct pxa_fa_scratch_t {
    ggml_tensor dst;
    ggml_tensor K;
    ggml_tensor V;
    ggml_tensor M;
};
ggml_tensor * pxa_fa_prepare_node(ggml_tensor * dst, pxa_fa_scratch_t & scratch);

// PXA_CORE_ROUTES=1: count the route every dispatched node took and print the table at exit.
// Unset: the call is a predictable branch and nothing is printed.
void pxa_fa_route_census(int device, pxa_fa_route route);
