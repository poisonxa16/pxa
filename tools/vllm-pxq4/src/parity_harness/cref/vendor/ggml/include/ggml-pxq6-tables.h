// ggml-pxq6-tables.h -- PXQ6 frozen numeric tables. Format spec: this header; conversion: docs/PXQU-CONVERT.md.
//
// PXQ6 = PXQ5 numerics (frozen PX16 16-entry book, UNCHANGED) + E16-row two-level scales:
//   per-ROW fp16 anchor (128 B header per 64-row panel = 2 B/row via ggml row_meta_size)
//   + one 4-bit energy-weighted sub-scale per 16-elem block (two nibbles re-spend the old
//   SE8 byte -> slab scale SoA stays 64 B; slab layout/coalescing byte-count-identical).
//   dequant contract (parity-locked):
//     eff = fp32(anchor_fp16) * PXQ6_SUB16[s4]      (fp32 mul, once per 16-block)
//     w   = eff * fp32(book[c])                     (fp32 mul; GEMM snaps __float2half_rn(w))
//   Measured (lab, 36-slice rng-42 protocol): wrel 0.068086 = -12.6% vs frozen PXQ5 0.077906
//   at 4.2656 bpw. HQ tier (bs8 subs, PXQ6HQ type): wrel 0.058683 = -24.7% at 4.5156 bpw.
//
// SOURCES (sha256-locked; do NOT edit values by hand -- regenerate from the calibration sources):
//   sublevels.json  d8847db66fafc90644526597a71907889e1cf1d06bf9290bc7277d5fb6e22cab
//   books.json      047ef6431435339def7000571726e9a8fe8cd3e654448e70803c2bfa1fad3041
//   (internal calibration lab artifacts; the sha256s pin the exact tables used)
// All sublevels are fp16-snapped then emitted as exact fp32 hex literals. The book is the
// The frozen PX16 book (inherited verbatim from the retired PXQ5 type — values byte-frozen).
#pragma once

#define PXQ6_QK          32
#define PXQ6_TYPE_SIZE   17
#define PXQ6_BM          64
#define PXQ6_SLAB_BYTES  1088
#define PXQ6_HDR_BYTES   128     // 64 x fp16 row anchors at the head of every 64-row panel
#define PXQ6_ROW_META    2       // ggml row_meta_size: 2 B/row == 128 B / 64-row panel

// HQ tier (PXQ6HQ): 4-bit sub per 8-elem block -> scale SoA 128 B/slab, 18 B / 32 elems
#define PXQ6HQ_TYPE_SIZE 18
#define PXQ6HQ_SLAB_BYTES 1152

// frozen PX16 book -- BIT-IDENTICAL to PXQ5_BOOK_INIT (sorted asc, book[7]==0, absmax==1)
#define PXQ6_BOOK_INIT { \
    -0x1.f9c0000000000p-1f, -0x1.7880000000000p-1f, -0x1.1e00000000000p-1f, -0x1.adc0000000000p-2f, \
    -0x1.3440000000000p-2f, -0x1.8e40000000000p-3f, -0x1.8740000000000p-4f, 0x0.0p+0f, \
    0x1.5b00000000000p-4f, 0x1.5ec0000000000p-3f, 0x1.0c40000000000p-2f, 0x1.7140000000000p-2f, \
    0x1.e280000000000p-2f, 0x1.3380000000000p-1f, 0x1.8800000000000p-1f, 0x1.0000000000000p+0f }

// E16-row-4bit-EW sublevels (bs16 core tier), fp16-snapped, ascending, SUB16[0] != 0
#define PXQ6_SUB16_INIT { \
    0x1.b7c0000000000p-3f, 0x1.36c0000000000p-2f, 0x1.72c0000000000p-2f, 0x1.a2c0000000000p-2f, \
    0x1.ccc0000000000p-2f, 0x1.f300000000000p-2f, 0x1.0bc0000000000p-1f, 0x1.1e00000000000p-1f, \
    0x1.3040000000000p-1f, 0x1.4380000000000p-1f, 0x1.5800000000000p-1f, 0x1.6ec0000000000p-1f, \
    0x1.8880000000000p-1f, 0x1.a640000000000p-1f, 0x1.cac0000000000p-1f, 0x1.f9c0000000000p-1f }

// E8-row-4bit-EW sublevels (bs8 HQ tier), fp16-snapped, ascending
#define PXQ6_SUB8_INIT { \
    0x1.58c0000000000p-3f, 0x1.e440000000000p-3f, 0x1.2640000000000p-2f, 0x1.5280000000000p-2f, \
    0x1.7a80000000000p-2f, 0x1.a040000000000p-2f, 0x1.c4c0000000000p-2f, 0x1.e900000000000p-2f, \
    0x1.07c0000000000p-1f, 0x1.1c80000000000p-1f, 0x1.32c0000000000p-1f, 0x1.4bc0000000000p-1f, \
    0x1.68c0000000000p-1f, 0x1.8b40000000000p-1f, 0x1.b700000000000p-1f, 0x1.f380000000000p-1f }

// PXQ6 5-bit tier (display name "pxq6", GGML_TYPE_PXQ6 = 256; internal working name PXQ6R): the HQ+ composition
// (gate Q-G2b PASSED at wrel 0.034301 @ 5.2656 bpw, K=1024) — LM32 5-bit book x the frozen
// E16-row scale machinery (fp16 row anchor + SUB16 4-bit sub per 16-elem block; no new sub
// table — PXQ6_SUB16_INIT is shared verbatim, like PXQ2/PXQ3).
//   code row = 20 B / 32 elems: bytes 0..15 nibble plane (byte b = lo4(c[2b]) | lo4(c[2b+1])<<4,
//   byte-identical layout to the PXQ6 core rows) + bytes 16..19 one LE u32 hi-bit plane
//   (bit j = bit 4 of c[j], j = 0..31). Slab = 64 B scale SoA + 64 x 20 B code rows = 1344 B.
//   dequant contract (parity-locked, fp32):
//     eff = fp32(anchor_fp16) * PXQ6_SUB16[s4]; w = eff * fp32(PXQ6_LM32[c])
//   zero 16-block: s4 = 0, codes = PXQ6R_ZIDX (book[16] == +0.0f); zero row: anchor fp16 +0,
//   scale nibbles 0, codes PXQ6R_ZIDX -> exact 0.0 on every path.
#define PXQ6R_TYPE_SIZE   21      // bytes / 32 elems (1 scale byte + 20 code bytes)
#define PXQ6R_SLAB_BYTES  1344    // 64 B scale SoA + 64 rows x 20 B codes
#define PXQ6R_HDR_BYTES   128     // 64 x fp16 row anchors per 64-row panel (== PXQ6_HDR_BYTES)
#define PXQ6R_ROW_META    2       // ggml row_meta_size
#define PXQ6R_CODE_OFF    64
#define PXQ6R_ZIDX        16      // LM32 book zero index (book[16] == +0.0f)

// LM32 5-bit book (bs16 refit, refine-uniform5bitref BK_BS16), fp16-snapped. Wired as
// GGML_TYPE_PXQ6 (display pxq6) as of 2026-07-21; values remain frozen (byte-frozen —
// never edit). book zero index = 16.
#define PXQ6_LM32_INIT { \
    -0x1.0000000000000p+0f, -0x1.e500000000000p-1f, -0x1.b280000000000p-1f, -0x1.8400000000000p-1f, \
    -0x1.5a80000000000p-1f, -0x1.3480000000000p-1f, -0x1.1180000000000p-1f, -0x1.e200000000000p-2f, \
    -0x1.a480000000000p-2f, -0x1.69c0000000000p-2f, -0x1.3180000000000p-2f, -0x1.f640000000000p-3f, \
    -0x1.8d40000000000p-3f, -0x1.27c0000000000p-3f, -0x1.8900000000000p-4f, -0x1.8900000000000p-5f, \
    0x0.0p+0f, 0x1.a500000000000p-5f, 0x1.a740000000000p-4f, 0x1.3f80000000000p-3f, \
    0x1.ae00000000000p-3f, 0x1.1000000000000p-2f, 0x1.4bc0000000000p-2f, 0x1.8980000000000p-2f, \
    0x1.ca80000000000p-2f, 0x1.0800000000000p-1f, 0x1.2d80000000000p-1f, 0x1.55c0000000000p-1f, \
    0x1.8140000000000p-1f, 0x1.b080000000000p-1f, 0x1.e400000000000p-1f, 0x1.ff80000000000p-1f }
