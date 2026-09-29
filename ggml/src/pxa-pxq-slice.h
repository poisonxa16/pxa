// pxa-pxq-slice.h -- the panel-aware K-slicer for the PXQ tiers, as one function.
//
// A PXQ tensor is addressed as: experts outermost -> 64-row panels, row-major -> K/32 slabs.
// Every panel opens with a 128-byte header of 64 fp16 per-row anchors that span ALL of K, and a
// consumer kernel reaches an element at
//     panel_base + HDR + kb*SLAB + <slab-local offsets>,   SLAB = 64 * type_size.
//
// So cutting on K is a pure byte permutation: each shard gets the panel header VERBATIM (the
// anchors are per-row and span all of K, so every shard needs the same ones) followed by its own
// whole slabs. Nothing is recomputed and nothing is re-encoded, which is why a shard dequantises
// bit-for-bit to the same values as the corresponding columns of the unsplit tensor.
//
// This lives in its own header because two places must run EXACTLY this arithmetic: the CUDA split
// uploader, which is the shipping path, and tests/test-pxq-ksplit.cpp, which proves per tier that
// the shipping path is byte-exact. A test that re-implements the thing it is testing proves
// nothing, so they call the same function.

#pragma once

#include <stdint.h>
#include <string.h>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Copy one K-shard of a panel-addressed 2D slice.
//
//   src          base of the source slice ([k_full x nrows], panel-interleaved)
//   dst          base of the destination shard ([k_shard x nrows], panel-interleaved)
//   nrows        rows in the slice (ne[1]); must be a multiple of panel_rows
//   k_full       K of the source
//   k_off        first K element of this shard; must be a multiple of the type's block size
//   k_shard      K of this shard; must be a multiple of the type's block size
//   panel_rows   rows per physical panel (64 for every PXQ tier)
//
// The caller has already applied any expert offset to src/dst, exactly as the CUDA kernels expect
// (the expert stride of a shard is recomputed from the shard's own row count, so a shard is a
// first-class tensor with nothing else to fix up).
static inline void pxa_pxq_k_slice_2d(
        enum ggml_type type,
        const void *   src,
        void *         dst,
        int64_t        nrows,
        int64_t        k_full,
        int64_t        k_off,
        int64_t        k_shard,
        int            panel_rows) {
    const int64_t bs = ggml_blck_size(type);
    const int64_t ts = ggml_type_size(type);

    // row_meta_size, from the public row-size contract:
    //   ggml_row_size(t, ne) == row_meta_size + type_size * ne / blck_size
    const int64_t row_size_blk = (int64_t) ggml_row_size(type, bs);
    const int64_t row_meta     = row_size_blk > ts ? row_size_blk - ts : 0;

    const int64_t src_row_size = (int64_t) ggml_row_size(type, k_full);
    const int64_t dst_row_size = (int64_t) ggml_row_size(type, k_shard);

    // Where this shard's first slab starts inside a source panel: past the header, then whole slabs.
    const int64_t source_offset = panel_rows * (row_meta + (k_off / bs) * ts);
    const int64_t body_bytes    = panel_rows * (dst_row_size - row_meta);
    const int64_t header_bytes  = panel_rows * row_meta;

    for (int64_t i01 = 0; i01 < nrows; i01 += panel_rows) {
        const char * s = (const char *) src + i01 * src_row_size;
        char       * d = (char *)       dst + i01 * dst_row_size;
        if (header_bytes > 0) {
            memcpy(d, s, (size_t) header_bytes);
        }
        memcpy(d + header_bytes, s + source_offset, (size_t) body_bytes);
    }
}

// The panel-addressed PXQ tiers, as the ggml side sees them (the llama side has the same list as
// pxa_pxq_type_is_panel in src/pxa-pxq-split.h; the uploader's k_map is the geometry table).
static inline int pxa_pxq_type_is_panel_ggml(enum ggml_type t) {
    return t == GGML_TYPE_PXQ1 || t == GGML_TYPE_PXQ2 || t == GGML_TYPE_PXQ3 ||
           t == GGML_TYPE_PXQ4 || t == GGML_TYPE_PXQ4HQ || t == GGML_TYPE_PXQ6;
}

// Error codes of the range slicers below (negative returns). Each names a DIFFERENT mistake, so
// the caller's abort says which one happened instead of a bare "-1 vs N".
#define PXA_PXQ_SLICE_ERR_RANGE    (-1)   // a range is misaligned to the block or out of bounds
#define PXA_PXQ_SLICE_ERR_PANEL    (-2)   // nrows is not a whole number of panels (or panel_rows < 1,
                                          // or a row-addressed call on a type with row metadata)
#define PXA_PXQ_SLICE_ERR_CAPACITY (-3)   // the shard the ranges describe does not fit dst_size

static inline const char * pxa_pxq_slice_strerror(int64_t rc) {
    switch (rc) {
        case PXA_PXQ_SLICE_ERR_RANGE:    return "a range is not aligned to the block size or runs past K";
        case PXA_PXQ_SLICE_ERR_PANEL:    return "the row count is not a whole number of panels";
        case PXA_PXQ_SLICE_ERR_CAPACITY: return "the shard described by the ranges does not fit the destination";
        default:                         return "ok";
    }
}

// The smallest K a PXQ panel shard may have. The panel mat-vec kernels (pxa/pxq6.cuh
// k_pxq6_mmv_qp and its split twin) start PXQ4_MMV_KSEG = 4 K-segment lanes on slabs 0..3 before
// they test the bound, so a panel tensor with fewer than 4 slabs (128 elements) would read past
// its own panel. Every K-cut of a panel type must leave each shard at least this wide.
#define PXA_PXQ_MIN_SHARD_K 128

// Copy one K-shard of a panel-addressed 2D slice whose K columns are NOT one contiguous block but
// an ordered list of ranges -- the shape the DeltaNet armer hands ssm_out (and any other tensor
// cut on dim 0 through explicit row ranges): with repeat_type 1 a device owns gqa_ratio
// separated head groups of K, not one span.
//
// The same byte permutation as pxa_pxq_k_slice_2d, generalised: each shard panel gets the
// 128-byte header VERBATIM (the per-row anchors span all of K, so every shard needs all of them)
// followed by the slabs of range 0, then range 1, ... in the order given. The shard's K is the sum
// of the range lengths and its slab order is the range order, which is exactly the column order the
// row-addressed arm produces for a non-panel type -- so the consumer sees the same logical tensor
// whatever the codec.
//
//   ranges       n_ranges (first, count) pairs in K elements; first and count must be multiples of
//                the type's block size (32 for every PXQ tier) and first+count <= k_full
//   panel_rows   64 for a PXQ tier, 1 for a row-addressed type (then row_meta must be 0 and this is
//                the plain per-row range copy the uploader always did)
//   dst_size     capacity of dst in bytes; checked BEFORE anything is written
//
// Returns the shard's K (sum of counts), or a PXA_PXQ_SLICE_ERR_* code, in which case nothing has
// been written -- the caller aborts naming the tensor, because a mis-cut panel is wrong without
// being detectably wrong.
static inline int64_t pxa_pxq_k_slice_ranges_2d(
        enum ggml_type  type,
        const void *    src,
        void *          dst,
        size_t          dst_size,
        int64_t         nrows,
        int64_t         k_full,
        const int *     ranges,     // 2*n_ranges ints: first0, count0, first1, count1, ...
        int             n_ranges,
        int             panel_rows) {
    const int64_t bs = ggml_blck_size(type);
    const int64_t ts = ggml_type_size(type);

    const int64_t row_size_blk = (int64_t) ggml_row_size(type, bs);
    const int64_t row_meta     = row_size_blk > ts ? row_size_blk - ts : 0;

    if (panel_rows < 1 || nrows % panel_rows != 0) return PXA_PXQ_SLICE_ERR_PANEL;
    if (panel_rows == 1 && row_meta != 0)          return PXA_PXQ_SLICE_ERR_PANEL;

    int64_t k_shard = 0;
    for (int r = 0; r < n_ranges; ++r) {
        const int64_t first = ranges[2*r + 0];
        const int64_t count = ranges[2*r + 1];
        if (first < 0 || count < 0 || first % bs != 0 || count % bs != 0 || first + count > k_full) {
            return PXA_PXQ_SLICE_ERR_RANGE;
        }
        k_shard += count;
    }

    const int64_t src_row_size = (int64_t) ggml_row_size(type, k_full);
    const int64_t dst_row_size = (int64_t) ggml_row_size(type, k_shard);
    const int64_t header_bytes = panel_rows * row_meta;

    if ((uint64_t) (nrows * dst_row_size) > (uint64_t) dst_size) return PXA_PXQ_SLICE_ERR_CAPACITY;

    for (int64_t i01 = 0; i01 < nrows; i01 += panel_rows) {
        const char * s = (const char *) src + i01 * src_row_size;
        char       * d = (char *)       dst + i01 * dst_row_size;
        if (header_bytes > 0) {
            memcpy(d, s, (size_t) header_bytes);
        }
        d += header_bytes;
        for (int r = 0; r < n_ranges; ++r) {
            const int64_t first = ranges[2*r + 0];
            const int64_t count = ranges[2*r + 1];
            const int64_t bytes = panel_rows * (count / bs) * ts;
            memcpy(d, s + panel_rows * (row_meta + (first / bs) * ts), (size_t) bytes);
            d += bytes;
        }
    }
    return k_shard;
}

// Validate a dim-1 (output-row) cut given as row ranges. A row cut of a panel type is a plain byte
// range copy ONLY while every range starts and ends on a panel boundary (64 consecutive rows ARE
// one panel); a range that halves a panel produces a shard whose headers are some other rows'
// anchors -- wrong without being detectably wrong. Returns the shard's row count, or
// PXA_PXQ_SLICE_ERR_RANGE / PXA_PXQ_SLICE_ERR_PANEL.
static inline int64_t pxa_pxq_row_ranges_check(
        int64_t      nrows_full,
        const int *  ranges,     // 2*n_ranges ints: first0, count0, ...
        int          n_ranges,
        int          panel_rows) {
    if (panel_rows < 1) return PXA_PXQ_SLICE_ERR_PANEL;
    int64_t n = 0;
    for (int r = 0; r < n_ranges; ++r) {
        const int64_t first = ranges[2*r + 0];
        const int64_t count = ranges[2*r + 1];
        if (first < 0 || count < 0 || first + count > nrows_full) return PXA_PXQ_SLICE_ERR_RANGE;
        if (first % panel_rows != 0 || count % panel_rows != 0)  return PXA_PXQ_SLICE_ERR_PANEL;
        n += count;
    }
    return n;
}

#ifdef __cplusplus
}
#endif
