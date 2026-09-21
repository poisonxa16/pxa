// pxq23_kernel_tables.h -- PXQ2 (ggml type id 254) and PXQ3 (id 255) frozen geometry and
// numeric tables, vendored VERBATIM from the pxa engine tree.
//
// SOURCE OF TRUTH (read-only; do not edit either copy by hand):
//   ggml/include/ggml-pxq2-tables.h   PXQ2_QK / TYPE_SIZE / BM / SLAB_BYTES / HDR_BYTES /
//                                     ROW_META / ZIDX / PXQ2_BOOK_INIT (+ V2, V3 books)
//   ggml/include/ggml-pxq3-tables.h   the PXQ3 twins
//   ggml/src/ggml-cuda/pxa/pxq23.cuh  pxq6_pol_p2 / pxq6_pol_p3 (the decode policies)
//
// The sub-scale LUT is NOT redefined here. PXQ2 and PXQ3 reuse the PXQ6 SUB16 table VERBATIM
// (both engine headers say so in as many words), and PXQ4_SUB16_INIT in pxq4_kernel_tables.h
// is itself a verbatim copy of PXQ6_SUB16_INIT. One table, three code widths -- which is also
// why the engine's PXA_PXQ6_SUB override moves all three at once.
//
// GEOMETRY, the only thing that differs between the three tiers:
//
//   panel (64 weight rows) = 128 B anchor header (64 x fp16, anchor[r] at byte 2*r)
//                          + (K/32) slabs, K-major
//   slab  (32 columns)     = 64 B sub-scale SoA (byte r = row r's scale byte for this
//                              32-column block; low nibble -> elements 0..15,
//                              high nibble -> elements 16..31)
//                          + 64 code rows of CODE_BYTES at slab[64 + CODE_BYTES*r]
//
//     tier   CODE_BYTES   CODE_WORDS   SLAB_BYTES   bpw (excl. row meta)
//     PXQ2        8            2           576          2.25
//     PXQ3       12            3           832          3.25
//     PXQ4       16            4          1088          4.25
//
// CODE PACKING (pxq23.cuh, and the format contract in the two engine headers):
//   PXQ2: 2 bits/elem, 4/byte. Two LE u32 words; word h covers elements 16h..16h+15,
//         element j at bits 2*(j&15). code(j) = (w[j>>4] >> 2*(j&15)) & 3.
//   PXQ3: 3 bits/elem, BIT-PLANE. Three LE u32 words w0 w1 w2:
//         w0 = LOW plane (2 b/elem) of elements  0..15
//         w1 = LOW plane (2 b/elem) of elements 16..31
//         w2 = HIGH plane: bit j (j=0..15) = element j's bit 2; bit 16+j = element 16+j's
//         code(j) = ((lo >> 2*(j&15)) & 3) | (((w2 >> j) & 1) << 2),  lo = w[j>>4]
//   Branch-free by construction; this is what makes one shared `pair()` interface work.
//
// ALIGNMENT, and why the code load is policy-dispatched rather than always a uint4:
//   PXQ4's 16 B rows sit at 64 + 16*r inside a 1088 B slab -- always 16 B aligned, so one
//   uint4. PXQ2's 8 B rows sit at 64 + 8*r inside a 576 B slab -- always 8 B aligned, so one
//   uint2. PXQ3's 12 B rows sit at 64 + 12*r inside an 832 B slab -- only 4 B aligned (r odd),
//   so THREE SCALAR u32 LOADS AND NEVER A VECTOR LOAD. The engine's pxq6_ldcodes makes the
//   same split for the same reason (pxq6.cuh:455-479); getting this wrong is a misaligned
//   address fault on odd rows, not a wrong number.
//
// BOOK SIZES: PXQ2 has 4 entries, PXQ3 has 8. Both are staged into the SAME 16-float shared
// tab[] the PXQ4 kernels use (entries past the book are zero-filled), exactly as the engine's
// stage_tabs does -- so the 16-float / 64-byte shared-table invariant that closes the
// bank-conflict question for PXQ4 holds unchanged for the new tiers.
//
// NO ZERO ENTRY AND absmax != 1 BY DESIGN: these are Lloyd-fit LM4/LM8 books, so the PXQ4/PX16
// invariants (book[7]==0, book[15]==1) deliberately do NOT apply. The self-check below is the
// engine's pxa_pxq23_book_ok: fp16-snapped, strictly ascending, sign-straddling, |v| < 1.

#pragma once

#include <stdint.h>

// ------------------------------------------------------------------------------ tier ids
// These are the ggml type ids, used verbatim as the `tier` argument of every op in this
// package so a tier can never be confused with an array index.
#define PXQ_TIER_PXQ2 254
#define PXQ_TIER_PXQ3 255
#define PXQ_TIER_PXQ4 252

// ------------------------------------------------------------------------------ PXQ2
#define PXQ2_QK          32
#define PXQ2_TYPE_SIZE   9       // 1 scale byte + 8 code bytes per 32-element row-block
#define PXQ2_BM          64
#define PXQ2_SLAB_BYTES  576     // 64 B scale SoA + 64 rows x 8 B codes
#define PXQ2_HDR_BYTES   128
#define PXQ2_ROW_META    2
#define PXQ2_CODE_OFF    64
#define PXQ2_CODE_BYTES  8
#define PXQ2_CODE_WORDS  2
#define PXQ2_NEFF        2
#define PXQ2_BOOK_N      4
#define PXQ2_ZIDX        2

// LM4 co-fit book (books.json b2_e16), fp16-snapped, exact fp32 hex -- the DEFAULT (v1) book.
// -0.70556640625, -0.1876220703125, 0.186767578125, 0.70263671875
#define PXQ2_BOOK_INIT { \
    -0x1.6940000000000p-1f, -0x1.8040000000000p-3f, 0x1.7e80000000000p-3f, 0x1.67c0000000000p-1f }

// v2 (PXA_PXQ_CEIL_V2) -- pure rescale to max|book| == 1.0.
#define PXQ2_BOOK_V2_INIT { \
    -0x1.0000000000000p+0f, -0x1.1040000000000p-2f, 0x1.0f00000000000p-2f, 0x1.fdc0000000000p-1f }

// v3 (PXA_PXQ2_V3) -- the model-family refit book (LM4R).
#define PXQ2_BOOK_V3_INIT { \
    -0x1.9500000000000p-1f, -0x1.15c0000000000p-2f, 0x1.9c80000000000p-4f, 0x1.54c0000000000p-1f }

// ------------------------------------------------------------------------------ PXQ3
#define PXQ3_QK          32
#define PXQ3_TYPE_SIZE   13      // 1 scale byte + 12 code bytes per 32-element row-block
#define PXQ3_BM          64
#define PXQ3_SLAB_BYTES  832     // 64 B scale SoA + 64 rows x 12 B codes
#define PXQ3_HDR_BYTES   128
#define PXQ3_ROW_META    2
#define PXQ3_CODE_OFF    64
#define PXQ3_CODE_BYTES  12
#define PXQ3_CODE_WORDS  3
#define PXQ3_NEFF        2
#define PXQ3_BOOK_N      8
#define PXQ3_ZIDX        4

// LM8 co-fit book (books.json b3_e16), fp16-snapped, exact fp32 hex -- the DEFAULT (v1) book.
#define PXQ3_BOOK_INIT { \
    -0x1.d040000000000p-1f, -0x1.1880000000000p-1f, -0x1.3100000000000p-2f, -0x1.7d80000000000p-4f, \
    0x1.7880000000000p-4f, 0x1.2ec0000000000p-2f, 0x1.1740000000000p-1f, 0x1.cfc0000000000p-1f }

// v2 (PXA_PXQ_CEIL_V2) -- pure rescale to max|book| == 1.0.
#define PXQ3_BOOK_V2_INIT { \
    -0x1.0000000000000p+0f, -0x1.3540000000000p-1f, -0x1.5040000000000p-2f, -0x1.a4c0000000000p-4f, \
    0x1.9f40000000000p-4f, 0x1.4e00000000000p-2f, 0x1.3400000000000p-1f, 0x1.ff80000000000p-1f }

// ------------------------------------------------------------------- host-side geometry
// The single place the C++ layer resolves a tier to its slab stride. Returns 0 for an
// unknown tier so every caller is forced to check rather than address into nothing.
static inline int pxq_tier_slab_bytes(int tier) {
    switch (tier) {
        case PXQ_TIER_PXQ2: return PXQ2_SLAB_BYTES;
        case PXQ_TIER_PXQ3: return PXQ3_SLAB_BYTES;
        case PXQ_TIER_PXQ4: return 1088;
        default:            return 0;
    }
}

static inline int pxq_tier_book_n(int tier) {
    switch (tier) {
        case PXQ_TIER_PXQ2: return PXQ2_BOOK_N;
        case PXQ_TIER_PXQ3: return PXQ3_BOOK_N;
        case PXQ_TIER_PXQ4: return 16;
        default:            return 0;
    }
}

static inline const char * pxq_tier_name(int tier) {
    switch (tier) {
        case PXQ_TIER_PXQ2: return "pxq2";
        case PXQ_TIER_PXQ3: return "pxq3";
        case PXQ_TIER_PXQ4: return "pxq4";
        default:            return "?";
    }
}
