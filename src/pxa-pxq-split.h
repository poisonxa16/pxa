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
        case GGML_TYPE_PXQN3:     // PXQN rev 1: blck_size 128 -> K granularity 128 via ggml_blck_size
        case GGML_TYPE_PXQN3S8:
        case GGML_TYPE_PXQN4:     // blck_size 32 (a ROTATED PXQN4/q8_0 tensor still cuts at 128: pxa_pxqn_*)
        case GGML_TYPE_PXQN2:     // ladder: blck_size 128 (N2, N1, N5) / 32 (N4S8), same panel geometry
        case GGML_TYPE_PXQN1:
        case GGML_TYPE_PXQN4S8:
        case GGML_TYPE_PXQN5:
        case GGML_TYPE_PXA4:      // blck_size 128
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

// PXA_TSPLIT_SSM_OUT_PANEL -- default ON. Whether a PXQ panel tensor may be cut on dim 0 through
// the uploader's EXPLICIT RANGES arm (today: the DeltaNet ssm_out, whose per-device K is gqa_ratio
// separated head groups). Set it to 0 to restore the previous behaviour exactly -- a file with a
// PXQ-coded ssm_out is refused under a tensor split, whatever the ladder lever says -- without a
// rebuild. The arithmetic is pxa_pxq_k_slice_ranges_2d (ggml/src/pxa-pxq-slice.h), proven per tier
// by tests/test-pxq-ksplit.cpp.
static inline bool pxa_pxq_ssm_out_panel_lever(void) {
    const char * v = getenv("PXA_TSPLIT_SSM_OUT_PANEL");
    return v == nullptr || atoi(v) != 0;
}

// May this panel type take a dim-0 cut through explicit ranges?
static inline bool pxa_pxq_k_split_ranges_ok(enum ggml_type t) {
    return pxa_pxq_k_split_ok(t) && pxa_pxq_ssm_out_panel_lever();
}

// The smallest K a panel shard may have: the panel mat-vec kernels start 4 K-segment lanes on slabs
// 0..3 before testing the bound (pxa/pxq6.cuh k_pxq6_mmv_qp), so a shard needs at least 4 slabs.
// Mirrors PXA_PXQ_MIN_SHARD_K in ggml/src/pxa-pxq-slice.h.
#define PXA_PXQ_MIN_SHARD_K_LLAMA 128

// The K granularity a panel type admits: one scale byte packs two 4-bit sub-scales covering
// elements 0-15 and 16-31, so a 16-element cut would split a scale byte. 32 is a hard floor.
static inline int pxa_pxq_k_granularity(enum ggml_type t) {
    return pxa_pxq_type_is_panel(t) ? ggml_blck_size(t) : 1;
}

// ------------------------------------------------------------------------------------------------
// K-STRADDLE (2026-09-26). A routed-expert FFN is cut on n_ff: up/gate by rows (dim 1, a
// whole 64-row panel is enough) and the down by K (dim 0, a whole K block: 128 for the PXQN 128-K slab tiers).
// When the row cut at panel granularity falls INSIDE a down K block, the block that straddles the cut is
// placed on BOTH devices and each device computes only its own K window of it: the device shard of the down
// holds whole blocks [s0, s1) and its activation covers [s0 + lo, s0 + hi). The window rides the per-device
// shard (a weight LEAF, op NONE, whose op_params nothing else reads; slots 10-12, clear of BitNet's slot 0 and
// the graph builder's slot 15 marker) so the graph can zero-pad the activation to the shard's K
// (ggml_pad_ext: the generic paths stay exact) and the PXQN kernels can skip the padding.
// PXA_TSPLIT_KSTRADDLE=0 restores the block-granular cut (the pre-lane uneven split) without a rebuild.
static inline bool pxa_kstraddle_enabled(void) {
    const char * v = getenv("PXA_TSPLIT_KSTRADDLE");
    return v == nullptr || atoi(v) != 0;
}
#define PXA_KWIN_TAG 0x4E49574B   // "KWIN"
static inline void pxa_kwin_set(struct ggml_tensor * t, int lo, int hi) {
    t->op_params[10] = PXA_KWIN_TAG; t->op_params[11] = lo; t->op_params[12] = hi;
}
static inline bool pxa_kwin_get(const struct ggml_tensor * t, int * lo, int * hi) {
    if (!t || t->op != GGML_OP_NONE || t->op_params[10] != PXA_KWIN_TAG) return false;
    *lo = t->op_params[11]; *hi = t->op_params[12];
    return *lo >= 0 && *hi > *lo && *hi <= t->ne[0];
}
static inline bool pxa_type_is_pxqn(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_PXQN3: case GGML_TYPE_PXQN3S8: case GGML_TYPE_PXQN4: case GGML_TYPE_PXQN2:
        case GGML_TYPE_PXQN1: case GGML_TYPE_PXQN4S8: case GGML_TYPE_PXQN5: case GGML_TYPE_PXA4:
            return true;
        default:
            return false;
    }
}
