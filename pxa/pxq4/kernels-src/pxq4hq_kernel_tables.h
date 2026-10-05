// pxq4hq_kernel_tables.h -- PXQ4HQ (ggml type id 253) frozen geometry and numeric tables,
// vendored VERBATIM from the pxa engine tree.
//
// SOURCE OF TRUTH (read-only; do not edit either copy by hand):
//   ggml/include/ggml-pxq6-tables.h:22-51  PXQ6HQ_TYPE_SIZE / PXQ6HQ_SLAB_BYTES,
//                                          PXQ6_BOOK_INIT (the frozen PX16 book, SHARED
//                                          with PXQ4), PXQ6_SUB8_INIT (the bs8 sublevels)
//   ggml/src/pxq-cpu.c:7,167-186           the layout row and the reference row dequant
//
// WHAT PXQ4HQ IS. One tier above PXQ4 on the same ladder, and the difference is a single
// axis: the sub-scale BLOCK SIZE. PXQ4 spends one 4-bit sub index per 16 elements; PXQ4HQ
// spends one per 8. Same 64-row panel, same 128 B fp16 anchor header, same 32-column slab,
// same 16 B nibble code rows, same PX16 book, same parity-locked reconstruction contract
//     eff = fp32(anchor_fp16) * SUB[s4] ;  w = eff * fp32(book[c])
// -- everything except which SUB table is indexed and how often. That is why this header
// redefines exactly three things (SLAB_BYTES, CODE_OFF, the SUB table) and inherits the rest
// from pxq4_kernel_tables.h rather than restating it.
//
//   tier     scale B/row   CODE_OFF   SLAB_BYTES   sub granularity   bpw (excl. row meta)
//   PXQ4          1           64         1088       per 16 elems           4.25
//   PXQ4HQ        2          128         1152       per  8 elems           4.50
//
// SCALE SoA, the one addressing difference and the one that silently produces a well-formed
// wrong tensor if it is got wrong. A PXQ4 slab holds ONE scale byte per row at slab[r], whose
// low nibble covers elements 0..15 and whose high nibble covers 16..31. A PXQ4HQ slab holds
// TWO bytes per row, at slab[2*r] and slab[2*r + 1]:
//     slab[2*r    ] low nibble -> elements  0.. 7   high nibble -> elements  8..15
//     slab[2*r + 1] low nibble -> elements 16..23   high nibble -> elements 24..31
// so the four effective scales of a 32-element block are indexed by (element >> 3), which is
// exactly the (b * NEFF) >> 4 pair index the shared kernel bodies already use with NEFF = 4.
//
// THE SUB TABLE IS NOT SHARED. PXQ2/PXQ3/PXQ4/PXQ6 all index the SAME 16-entry SUB16 LUT --
// the engine says so in every tier header and it is why one PXA_PXQ6_SUB override moves all
// of them at once. PXQ4HQ does NOT: bs8 blocks have a different energy distribution, so the
// tier carries its own 16-entry SUB8 fit. A PXQ4HQ tensor decoded against SUB16 loads,
// shards, passes every shape assertion and is uniformly wrong by up to 30% per block, so the
// two LUTs are kept in separate device symbols and are uploaded separately.
#pragma once

#include <stdint.h>

#include "pxq4_kernel_tables.h"   // PXQ4_QK / PXQ4_BM / PXQ4_HDR_BYTES / PXQ4_CODE_BYTES /
                                  // PXQ4_BOOK_INIT (the shared PX16 book) / PXQ4_MMV_KSEG /
                                  // PXQ4_CANON_CMAX / PXQ4_CANON_V2

// ------------------------------------------------------------------------------ tier id
// The ggml type id, used verbatim as the `tier` argument of every op in this package so a
// tier can never be confused with an array index.
#define PXQ_TIER_PXQ4HQ 253

// ------------------------------------------------------------------------------ geometry
#define PXQ4HQ_QK          32
#define PXQ4HQ_TYPE_SIZE   18     // 2 scale bytes + 16 code bytes per 32-element row-block
#define PXQ4HQ_BM          64
#define PXQ4HQ_SLAB_BYTES  1152   // 128 B scale SoA + 64 rows x 16 B codes
#define PXQ4HQ_HDR_BYTES   128    // 64 fp16 row anchors
#define PXQ4HQ_ROW_META    2
#define PXQ4HQ_CODE_OFF    128    // byte offset of the code rows inside a slab
#define PXQ4HQ_CODE_BYTES  16
#define PXQ4HQ_CODE_WORDS  4
#define PXQ4HQ_SCALE_BYTES 2      // scale bytes per row per slab (PXQ4 has 1)
#define PXQ4HQ_NEFF        4      // distinct effective scales per 32-element block
#define PXQ4HQ_BOOK_N      16

// The book is the frozen PX16 book, BIT-IDENTICAL to PXQ4's (ggml-pxq6-tables.h:33-37 is the
// single definition both tiers read). It is aliased rather than re-typed so a future edit to
// the engine table cannot leave the two copies disagreeing.
#define PXQ4HQ_BOOK_INIT PXQ4_BOOK_INIT

// E8-row-4bit-EW sublevels (bs8 HQ tier), fp16-snapped, ascending.
// VERBATIM from ggml/include/ggml-pxq6-tables.h:47-51 (PXQ6_SUB8_INIT). Written as C99
// hexadecimal float constants exactly as in the engine header so that transcription is
// verifiable by textual diff and cannot drift through decimal rounding.
#define PXQ4HQ_SUB8_INIT { \
    0x1.58c0000000000p-3f, 0x1.e440000000000p-3f, 0x1.2640000000000p-2f, 0x1.5280000000000p-2f, \
    0x1.7a80000000000p-2f, 0x1.a040000000000p-2f, 0x1.c4c0000000000p-2f, 0x1.e900000000000p-2f, \
    0x1.07c0000000000p-1f, 0x1.1c80000000000p-1f, 0x1.32c0000000000p-1f, 0x1.4bc0000000000p-1f, \
    0x1.68c0000000000p-1f, 0x1.8b40000000000p-1f, 0x1.b700000000000p-1f, 0x1.f380000000000p-1f }

// Geometry helpers, host-side, so the torch TU can assert a parameter's last dimension
// against the tier rather than inferring it (tiers.py makes the same check in Python).
static inline int pxq4hq_tier_slab_bytes(int tier) {
    return tier == PXQ_TIER_PXQ4HQ ? PXQ4HQ_SLAB_BYTES : 0;
}
static inline int pxq4hq_tier_book_n(int tier) {
    return tier == PXQ_TIER_PXQ4HQ ? PXQ4HQ_BOOK_N : 0;
}
