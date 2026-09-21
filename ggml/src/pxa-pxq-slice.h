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

#ifdef __cplusplus
}
#endif
