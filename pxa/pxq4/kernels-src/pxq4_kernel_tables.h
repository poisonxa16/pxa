// pxq4_kernel_tables.h — PXQ4 (GGML_TYPE_PXQ4 = 252, "core" tier) frozen numeric tables and
// geometry constants, vendored VERBATIM from the pxq_llama engine tree.
//
// SOURCE OF TRUTH (read-only, do not edit either copy by hand):
//   /path/to/engine-repo/ggml/include/ggml-pxq6-tables.h:21-44
//     PXQ6_QK / PXQ6_TYPE_SIZE / PXQ6_BM / PXQ6_SLAB_BYTES / PXQ6_HDR_BYTES / PXQ6_ROW_META
//     PXQ6_BOOK_INIT (frozen PX16 16-entry book)
//     PXQ6_SUB16_INIT (E16-row 4-bit energy-weighted sublevels, core tier)
//   /path/to/engine-repo/ggml/src/ggml-cuda/pxq4.cuh:114   PXQ4_MMV_KSEG
//
// FILE-NAME TRAP, restated here because it has already misled readers once:
// the kernels that serve ggml type id 252 live in ggml/src/ggml-cuda/**pxq6.cuh**, not in
// pxq4.cuh. pxq4.cuh documents the RETIRED id-250 MXFP4-repack format (removed 2026-07-21)
// and today contributes only shared macros. Everything vendored here comes from pxq6.cuh.
//
// The literals are written as C99 hexadecimal float constants exactly as in the engine header
// so that transcription is verifiable by textual diff and cannot drift through decimal
// rounding. All 32 values are fp16-exact by construction (the engine asserts this at startup,
// pxq6.cuh:98-108); pxq4_tables_self_check() below replicates that assertion.

#pragma once

#include <stdint.h>

// ---------------------------------------------------------------------------------------------
// geometry — see ggml/src/pxq-cpu.h:1-17 for the authoritative prose description
//
//   panel (64 weight rows) = 128 B anchor header (64 x fp16, anchor[r] at byte 2*r)
//                          + (K/32) slabs of 1088 B, K-major
//   slab  (32 columns)     = 64 B sub-scale SoA (scale byte for row r at slab[r];
//                              low nibble  -> elements  0..15 of this 32-col block,
//                              high nibble -> elements 16..31)
//                          + 64 x 16 B nibble code rows (row r at slab[64 + 16*r];
//                              byte b = code(2b) | code(2b+1) << 4)
//
// Consequence that drives the whole vLLM port: a single weight ROW is not a contiguous byte
// range — its bytes are scattered across every slab of its panel. Any consumer that slices
// rows as contiguous bytes (vLLM's generic GGUF sharder does exactly that) is wrong. Slicing
// is only legal in whole panels (dim 0, 64 rows) and whole slabs (dim 1, 32 columns).
// ---------------------------------------------------------------------------------------------
#define PXQ4_QK          32     // elements per slab column-block
#define PXQ4_TYPE_SIZE   17     // bytes per 32 elements per row (1 scale nibble-pair-half + 16 B)
#define PXQ4_BM          64     // rows per panel
#define PXQ4_SLAB_BYTES  1088   // 64 B scale SoA + 64 * 16 B code rows
#define PXQ4_HDR_BYTES   128    // 64 fp16 row anchors
#define PXQ4_CODE_OFF    64     // byte offset of the code rows inside a slab
#define PXQ4_CODE_BYTES  16     // bytes per row per slab
#define PXQ4_NEFF        2      // distinct effective scales per 32-element block (lo/hi nibble)
#define PXQ4_TYPE_ID     252    // GGML_TYPE_PXQ4

// k-segment count of the decode mmv: block = PXQ4_MMV_KSEG * 64 threads, one 64-thread group
// per k-segment. VENDORED VALUE — changing it changes the fp32 fold order and breaks
// bit-exactness against the shipping engine. (pxq4.cuh:114)
#define PXQ4_MMV_KSEG    4

// Canonical chunk cap. PXQ_CANON_v1/v2 make the mmv fold order a function of SHAPE ONLY, so
// that a K-split kernel and an unsplit kernel produce bit-identical output. We do not ship the
// split kernels, but we DO reproduce the chunked fold so our output matches the shipping
// engine's byte-for-byte. (pxq6.cuh:800-822, PXQ6_CANON_CMAX)
#define PXQ4_CANON_CMAX  16

// PXQ_CANON_V2 selects the inner accumulation SHAPE (pxq6.cuh:575-608). Default 0 in the
// engine, so the shipped artifact was produced and is served with the unchained form.
// Keep 0 unless the engine is rebaselined; the two forms are NOT bit-identical.
#ifndef PXQ4_CANON_V2
#define PXQ4_CANON_V2 0
#endif

// ---------------------------------------------------------------------------------------------
// frozen PX16 book — BIT-IDENTICAL to PXQ6_BOOK_INIT (ggml-pxq6-tables.h:33-37).
// Invariants the engine self-checks: sorted ascending, book[7] == 0, book[15] == 1.
// ---------------------------------------------------------------------------------------------
#define PXQ4_BOOK_INIT { \
    -0x1.f9c0000000000p-1f, -0x1.7880000000000p-1f, -0x1.1e00000000000p-1f, -0x1.adc0000000000p-2f, \
    -0x1.3440000000000p-2f, -0x1.8e40000000000p-3f, -0x1.8740000000000p-4f, 0x0.0p+0f, \
    0x1.5b00000000000p-4f, 0x1.5ec0000000000p-3f, 0x1.0c40000000000p-2f, 0x1.7140000000000p-2f, \
    0x1.e280000000000p-2f, 0x1.3380000000000p-1f, 0x1.8800000000000p-1f, 0x1.0000000000000p+0f }

// E16-row 4-bit energy-weighted sublevels, core tier (ggml-pxq6-tables.h:40-44).
// Invariants: ascending, SUB16[0] > 0, every entry fp16-exact.
#define PXQ4_SUB16_INIT { \
    0x1.b7c0000000000p-3f, 0x1.36c0000000000p-2f, 0x1.72c0000000000p-2f, 0x1.a2c0000000000p-2f, \
    0x1.ccc0000000000p-2f, 0x1.f300000000000p-2f, 0x1.0bc0000000000p-1f, 0x1.1e00000000000p-1f, \
    0x1.3040000000000p-1f, 0x1.4380000000000p-1f, 0x1.5800000000000p-1f, 0x1.6ec0000000000p-1f, \
    0x1.8880000000000p-1f, 0x1.a640000000000p-1f, 0x1.cac0000000000p-1f, 0x1.f9c0000000000p-1f }
