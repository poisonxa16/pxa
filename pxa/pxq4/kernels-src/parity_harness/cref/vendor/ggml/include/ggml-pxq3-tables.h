// ggml-pxq3-tables.h -- PXQ3 frozen numeric tables. Format spec: this header; conversion: docs/PXQU-CONVERT.md.
//
// PXQ3 = 3-bit codes into the co-fit LM8 8-entry book + the PROVEN PXQ6 E16-row two-level
// scales, UNCHANGED (per-ROW fp16 anchor in the 128 B panel header + frozen PXQ6 SUB16
// 4-bit sub-scale per 16-elem block; slab scale SoA stays 64 B; dequant contract identical
// to PXQ6: eff = fp32(anchor)*SUB16[s4]; w = eff*fp32(book[c])).
//
// CODE PACKING -- BIT-PLANE (locked by the format contract so fp16-LUT and int8-LUT
// extraction both stay branch-free). Per 32-elem row-block, 12 code bytes laid out as
// three LE uint32 words w0 w1 w2:
//   w0 = LOW plane of elems  0..15 (2 bits/elem, elem j at bits 2j)
//   w1 = LOW plane of elems 16..31 (2 bits/elem, elem j-16 at bits 2(j-16))
//   w2 = HIGH plane: bit j (j=0..15) = elems 0..15 bit2; bit 16+j = elems 16..31 bit2
//   code(j) = ((lo >> 2*(j&15)) & 3) | (((w2 >> j) & 1) << 2)     [lo = w0 or w1 by j>>4]
// All three words are 4-byte aligned for every row (12 B rows, CODE_OFF 64).
//   slab = 64 B scale SoA + 64 x 12 B code rows = 832 B; panel = 128 B hdr + kslabs slabs.
//   bpw = 13*8/32 + 16/K = 3.25 + 16/K  (3.2656 @ K=2048, 3.28125 @ K=512).
//   Measured (lab, uniform-3bit, PROD imatrix, rng-42 eval protocol): wrel 0.1435315.
//
// SOURCES (sha256-locked): books.json ("b3_e16")
//   e3ef27d550d4538654bf46c9ca8dac39ff181af31871da336a48b9e54755089f
//   (internal calibration lab artifact; the sha256 pins the exact book used)
// Sub-scale LUT: PXQ6_SUB16_INIT from ggml-pxq6-tables.h REUSED VERBATIM (codebook-agnostic,
// measured bit-identical after fp16 snap when refit on the LM8 pool). No PXQ3 sub table.
// Book is the alternating-co-fit LM8 (kept round 2), fp16-snapped, exact fp32 hex.
// NO zero entry, absmax != 1 by design; min-|v| entry is index 4 (PXQ3_ZIDX).
#pragma once

#define PXQ3_QK          32
#define PXQ3_TYPE_SIZE   13      // 1 scale byte + 12 code bytes per 32-elem row-block
#define PXQ3_BM          64
#define PXQ3_SLAB_BYTES  832     // 64 B scale SoA + 64 rows x 12 B codes
#define PXQ3_HDR_BYTES   128     // 64 x fp16 row anchors at the head of every 64-row panel
#define PXQ3_ROW_META    2       // ggml row_meta_size: 2 B/row == 128 B / 64-row panel
#define PXQ3_ZIDX        4       // argmin |book| -- code written for exactly-zero blocks

// LM8 co-fit book (books.json b3_e16, val_wrel 0.144277; full-eval wrel 0.143531).
// fp16-snapped, strictly ascending: -0.90673828125, -0.5478515625, -0.2978515625,
// -0.0931396484375, 0.0919189453125, 0.295654296875, 0.54541015625, 0.90576171875
#define PXQ3_BOOK_INIT { \
    -0x1.d040000000000p-1f, -0x1.1880000000000p-1f, -0x1.3100000000000p-2f, -0x1.7d80000000000p-4f, \
    0x1.7880000000000p-4f, 0x1.2ec0000000000p-2f, 0x1.1740000000000p-1f, 0x1.cfc0000000000p-1f }

// ---------------------------------------------------------------------------------------------
// v2 book (PXA_PXQ_CEIL_V2, 2026-08-09) -- THE CEILING FIX. The frozen LM8 book rescaled by
// 1/max|book| = 1/0.90673828125 so max|book| == 1.0 exactly, then fp16-snapped. A pure rescale:
// same Lloyd grid shape, same ZIDX (4), same sign straddle. Restores the representable ceiling
// from 0.896*anchor to 0.987793*anchor = PXQ4/PXQ6 parity. Provenance: pxa.pxq3.version = 2 +
// these values in pxa.pxq3.book. Active ONLY under PXA_PXQ_CEIL_V2=1 on BOTH quantizer and
// runtime (older builds can decode v2 files via PXA_PXQ3_BOOK with these exact values).
// fp16-snapped, strictly ascending: -1.0, -0.60400390625, -0.328369140625, -0.1027221679688,
// 0.1013793945312, 0.326171875, 0.6015625, 0.9990234375
#define PXQ3_BOOK_V2_INIT { \
    -0x1.0000000000000p+0f, -0x1.3540000000000p-1f, -0x1.5040000000000p-2f, -0x1.a4c0000000000p-4f, \
    0x1.9f40000000000p-4f, 0x1.4e00000000000p-2f, 0x1.3400000000000p-1f, 0x1.ff80000000000p-1f }
