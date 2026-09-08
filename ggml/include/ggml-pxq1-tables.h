// ggml-pxq1-tables.h — PXQ1 sub-2-bit tier: 1-bit sign codes x the PXQ E16-row scale
// machinery (per-row fp16 anchor + frozen SUB16 4-bit sub-scale per 16-elem block).
// The dominant tier of the pxq1-lab "adaptmix" winner (Tier0). ~1.25 bpw stored.
//
// FORMAT (frozen): 1-bit code = sign into the 2-level book {-1,+1}; magnitude carried
// entirely by the E16-row scale (per-row fp16 anchor x SUB16[s4]). Codes pack 8/byte ->
// 4 B / 32-elem row-block; slab = 64 B scale SoA + 64 x 4 B code rows = 320 B; panel =
// 128 B anchor header + kslabs x 320 B. Reconstruction: eff = fp32(anchor)*SUB16[s4];
// w = eff * book[sign]. Same scale search / zero-block convention as PXQ2/PXQ3.
#pragma once

#define PXQ1_QK          32
#define PXQ1_TYPE_SIZE   5       // 1 scale byte + 4 code bytes per 32-elem row-block
#define PXQ1_BM          64
#define PXQ1_SLAB_BYTES  320     // 64 B scale SoA + 64 rows x 4 B codes
#define PXQ1_HDR_BYTES   128     // 64 x fp16 row anchors at the head of every 64-row panel
#define PXQ1_ROW_META    2       // ggml row_meta_size: 2 B/row == 128 B / 64-row panel
#define PXQ1_ZIDX        0       // book index written for exactly-zero blocks (|book| symmetric)

// 2-level symmetric sign book (fp16-exact). The scale (anchor x SUB16) carries magnitude,
// so the LS-optimal 1-bit levels are +/-scale -> book = {-1, +1}.
#define PXQ1_BOOK_INIT { -1.0f, 1.0f }
