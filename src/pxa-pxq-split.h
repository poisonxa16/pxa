#pragma once

// PXQ panel geometry as the split policy sees it.
//
// This header is the ONE place that answers "may this quant type be cut on this axis", so that the
// loader (which refuses a bad cut), the admission check (which refuses a file BEFORE it is opened
// for tensor creation) and anything else asking the question cannot drift apart. The CUDA uploader
// keeps its own table, but that table is a statement about GEOMETRY (which types are
// panel-interleaved, and how many rows a panel has) -- it is not a policy list and does not need to
// be kept in step with this one.
//
// The geometry, from the codec: a PXQ tensor is addressed as
//   experts outermost -> 64-row panels, row-major -> K/32 slabs,
// and every panel opens with a 128-byte header of 64 fp16 per-row anchors that span ALL of K. So:
//   * dim 2 (expert id) is a flat memcpy, always safe;
//   * dim 1 (output rows) is a pure byte-range copy AT 64-ROW GRANULARITY, free and exact;
//   * dim 0 (K) is exact at 32-element granularity, but the 128-byte anchor header has to be
//     DUPLICATED into every shard. The CUDA uploader does that, type-generically (it reads
//     blck_size / type_size / row_meta_size from the type traits at runtime).
//
// PXQ1/PXQ2/PXQ3/PXQ4/PXQ4HQ/PXQ6 all share that geometry: blck_size 32, row_meta_size 2, 64 rows
// per panel, 128-byte header, SLAB = 64 * type_size. The ladder is therefore K-splittable as a
// matter of geometry. What used to gate it to PXQ4 alone was a VERIFICATION record, not a codec
// limit -- and that verification now exists for every tier (tests/test-pxq-ksplit.cpp produces a
// shard through the shipped slicer, dequantises it, and compares it bit-for-bit against the same
// columns of the unsplit tensor).

#include <cstdlib>
#include "ggml.h"

// Rows per physical PXQ panel. Must match PXQ4_BM / PXQ6_BM in the CUDA codec and the k_map entries
// in ggml/src/ggml-cuda.cu.
#define PXA_PXQ_PANEL_ROWS 64

// The panel-addressed tiers.
static inline bool pxa_pxq_type_is_panel(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_PXQ1:
        case GGML_TYPE_PXQ2:
        case GGML_TYPE_PXQ3:
        case GGML_TYPE_PXQ4:
        case GGML_TYPE_PXQ4HQ:
        case GGML_TYPE_PXQ6:
            return true;
        default:
            return false;
    }
}

// PXA_TSPLIT_KSPLIT_LADDER -- default ON. Set it to 0 to go back to the pre-ladder behaviour, where
// PXQ4 was the only tier allowed a dim-0 (K) cut and every other tier refused at load. Kept as a
// lever because a K-slice that is wrong is wrong WITHOUT being detectably wrong, so there has to be
// a way back that does not need a rebuild.
static inline bool pxa_pxq_k_split_ladder(void) {
    const char * v = getenv("PXA_TSPLIT_KSPLIT_LADDER");
    return v == nullptr || atoi(v) != 0;
}

// Types the split uploader may K-slice byte-exactly.
static inline bool pxa_pxq_k_split_ok(enum ggml_type t) {
    if (!pxa_pxq_type_is_panel(t)) {
        return false;
    }
    return pxa_pxq_k_split_ladder() || t == GGML_TYPE_PXQ4;
}

// The K granularity a panel type admits: one scale byte packs two 4-bit sub-scales covering
// elements 0-15 and 16-31, so a 16-element cut would split a scale byte. 32 is a hard floor.
static inline int pxa_pxq_k_granularity(enum ggml_type t) {
    return pxa_pxq_type_is_panel(t) ? ggml_blck_size(t) : 1;
}
