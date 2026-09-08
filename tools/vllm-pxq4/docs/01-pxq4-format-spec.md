# PXQ4 (GGML_TYPE_PXQ4 = 252) — complete byte-level format specification

Source of truth: `/path/to/engine-repo` @ branch `fable9-swa-kv`, HEAD `acf8f245` (READ ONLY).
Every claim below is tagged FACT (read in source, with file:line), INFERENCE, or ASSUMPTION.

**FILE-NAME TRAP (confirmed FACT).** The id-252 implementation lives in
`ggml/src/ggml-cuda/pxa/pxq6.cuh`. `ggml/src/ggml-cuda/pxa/pxq4.cuh:1-17` documents the *retired*
id-250 MXFP4-repack format and is now only a holder of shared macro constants
(`PXQ4_QK`, `PXQ4_BM`, `PXQ4_SLAB_BYTES`, `PXQ4_MMV_KSEG`) and tile/GLU helpers; its own
kernels were deleted (`pxq4.cuh:31-32, 62-63, 84-85, 117-119`). Read `pxq6.cuh` +
`src/pxq6-quantize.inc.cpp` + `ggml/include/ggml-pxq6-tables.h` for id 252.

Naming history (FACT, `ggml/include/ggml.h:461-467`): the type displayed as "PXQ6" before
2026-07-19 was re-laddered by bpw class and is now `GGML_TYPE_PXQ4 = 252`. All internal
identifiers still read `pxq6_*`. In this document **PXQ4 = id 252 = the "core" / bs16 tier
= policy `pxq6_pol_p6`**. Its sibling PXQ4HQ = id 253 = the bs8 tier = `pxq6_pol_p6hq`;
it is described only where it differs, because a PXQ4-backbone file can in principle contain
either (see §9).

---

## 0. One-paragraph summary

A PXQ4 tensor is a **64-row panel-interleaved** array. There is no per-row blob: a logical
weight row's bytes are scattered across every slab of its 64-row panel. A *panel* covers
64 consecutive weight rows and the full K axis: it starts with a **128-byte header of 64
fp16 per-row anchors**, followed by `K/32` **1088-byte slabs**, one per 32-column block.
A slab holds, in SoA order, **64 scale bytes** (one per row; each byte = two 4-bit indices
into a frozen 16-entry sub-scale table, one per 16-element half of the 32-column block)
then **64 × 16-byte nibble rows** (one per row; 32 4-bit codes into a frozen 16-entry
value book, packed as sequential pairs). Dequant is
`w = (fp32(anchor_row) * SUB16[s4]) * BOOK[code]`. Both tables are **global static
constants** compiled into the binary — nothing per-tensor, nothing per-row except the
fp16 anchor, nothing per-block except the 4-bit sub index.

---

## 1. Constants (FACT)

`ggml/include/ggml-pxq6-tables.h:21-27`:

```
PXQ6_QK          32      // elements per K-block (== ggml blck_size)
PXQ6_TYPE_SIZE   17      // bytes per 32 elements per row (== ggml type_size)
PXQ6_BM          64      // rows per panel
PXQ6_SLAB_BYTES  1088    // bytes per slab
PXQ6_HDR_BYTES   128     // 64 x fp16 row anchors, at the head of every 64-row panel
PXQ6_ROW_META    2       // ggml row_meta_size: 2 B/row == 128 B / 64-row panel
```
HQ tier, same header (`ggml-pxq6-tables.h:28-30`): `PXQ6HQ_TYPE_SIZE 18`,
`PXQ6HQ_SLAB_BYTES 1152`.

ggml type traits (FACT, `ggml/src/ggml.c:1421-1436`):

```c
[GGML_TYPE_PXQ4]   = { .type_name="pxq4",   .blck_size=32, .type_size=17,
                       .is_quantized=true, .nrows=1, .row_meta_size=2 },
[GGML_TYPE_PXQ4HQ] = { .type_name="pxq4hq", .blck_size=32, .type_size=18,
                       .is_quantized=true, .nrows=1, .row_meta_size=2 },
```
`to_float` / `from_float` / `vec_dot` are **deliberately NULL** and must stay NULL
(`ggml.c:1407-1414`, `ggml/src/pxq-cpu.h:4-12`): a ggml `to_float` receives a single-row
pointer, which is meaningless for a panel-interleaved format.

Geometry gate (FACT, `src/llama-quantize.cpp:1119-1122`):
```c
ggml_n_dims(t) >= 2 && t->ne[1] % 64 == 0 && t->ne[0] % 32 == 0
```
i.e. **rows % 64 == 0 && K % 32 == 0**. The CUDA dequant kernel hard-aborts on the same
condition (`pxq6.cuh:730-735`), and the CPU fallback documents it identically
(`pxq-cpu.h:44-47`).

ggml index convention (FACT): `ne[0] = K` (row length, the contracted axis),
`ne[1] = R` (number of weight output rows). Below I write **R = rows, K = columns**.

---

## 2. Container arithmetic — how ggml sizes a PXQ4 tensor

FACT, `ggml/src/ggml.c:4903-4906`:
```c
size_t ggml_row_size(type, ne) {
    return row_meta_size + type_size*ne/blck_size;
}
```
So for PXQ4: `row_size(K) = 2 + 17*K/32` bytes, and a tensor's nbytes is
`R * (2 + 17*K/32)`.

**This is a fiction that happens to be exact** (INFERENCE, arithmetic verified below):
there is no per-row region on disk. The real layout is panels. But

```
panel_bytes            = 128 + (K/32)*1088 = 128 + 34*K
total = (R/64)*panel_bytes = (R/64)*128 + (R/64)*34*K
                           = 2*R + (17*R*K)/32
row_size(K)*R          = R*(2 + 17*K/32) = 2*R + 17*R*K/32     ✓ identical
```
The `row_meta_size = 2` exists exactly so that ggml's `ggml_row_size` / `ggml_nbytes`
bookkeeping lands on the panel-header bytes without ggml knowing about panels
(FACT, comment `ggml.c:1415-1420`; `ggml-pxq6-tables.h:26`).

**Alignment (INFERENCE from the arithmetic):** `panel_bytes = 128 + 34*K` with `K ≡ 0 (mod 32)`
⇒ `34*K = 1088*(K/32)`, a multiple of 1088 and hence of 64. `128` is a multiple of 64.
Therefore **every panel base, every slab base, and the tensor base itself are 64-byte
aligned relative to the tensor base**, with no inter-panel padding. Panels are packed
back-to-back (`src/pxq6-quantize.inc.cpp:302-303`, `dst + p*panel_bytes`).

### 6. bpw check (FACT/arithmetic)
```
bits/weight = 8 * (2 + 17*K/32) / K = 4.25 + 16/K
```
which is exactly the "4.25 + 16/K bpw" in `ggml.h:465-467`. At K = 4096 → 4.25391;
at K = 1024 → 4.2656 (the number quoted in `ggml-pxq6-tables.h:10-11` and `pxq6.cuh:20-21`).
HQ: `8*(2 + 18*K/32)/K = 4.5 + 16/K` ✓ (`ggml.h:469-471`).

---

## 3. Memory layout (FACT)

Authoritative statements: `pxq6.cuh:14-17`, `src/pxq6-quantize.inc.cpp:8-13`,
and the writer `pxq6-quantize.inc.cpp:286-332`.

```
tensor data (dense 2D weight = a single "expert", E = 1):

  for e in [0, E):                                # experts OUTERMOST
    for p in [0, R/64):                           # panels row-major (panel p = rows 64p..64p+63)
      panel[e][p]:                                # panel_bytes = 128 + (K/32)*1088
        byte    0 ..  127 : ANCHOR HEADER   64 x fp16 little-endian
                            anchor[r] at offset 2*r, r = 0..63  (r = row within panel)
        for kb in [0, K/32):                      # slabs K-major within the panel
          slab[kb] at offset 128 + kb*1088:       # covers columns 32*kb .. 32*kb+31
            byte    0 ..   63 : SCALE SoA   scale_byte[r] at offset r, r = 0..63
                                   lo nibble (bits 0-3) = SUB16 index for elems  0..15
                                   hi nibble (bits 4-7) = SUB16 index for elems 16..31
            byte   64 .. 1087 : CODE ROWS   row r at offset 64 + 16*r, 16 bytes
                                   byte b (b = 0..15) = code(2b) | code(2b+1) << 4
                                   i.e. lo nibble = element 2b, hi nibble = element 2b+1
```

Addressing formulas as implemented in CUDA (FACT):
- `pxq6_panel_stride<POL>(kslabs) = POL::HDR + kslabs*POL::SLAB` — `pxq6.cuh:520-522`
- `pxq6_panel<POL>(W,e,panels,p,kslabs) = W + (e*panels + p)*panel_stride` — `pxq6.cuh:524-526`
  (this is the definitive statement that **experts are the outermost axis** and
  **panels are row-major within an expert**)
- `slab = panel + POL::HDR + kb*POL::SLAB` — `pxq6.cuh:703-704`, `947-957`
- `POL::CODE_OFF = 64` for PXQ4 (`pxq6.cuh:318`); code row pointer =
  `slab + CODE_OFF + row*CODE_BYTES` with `CODE_BYTES = 16` (`pxq6.cuh:330, 334-336`)
- HQ differs only in `CODE_OFF = 128`, `SLAB = 1152`, 2 scale bytes/row (`pxq6.cuh:347`)

**Element ordering answer (7):** the layout is **not** row-major-contiguous and **not**
a permutation of the K axis. Within one 32-element block the elements are in natural
order (element `i` at nibble `i` of the 16-byte code row: lo of byte `i/2` for even `i`,
hi for odd `i`). The permutation is entirely at the **row × K-block** level: the outer
loop nesting is `(expert, panel, K-block, row)` instead of `(row, K-block)`. Coalescing
comes from the SoA scale plane and from 64 consecutive rows' 16-byte code rows sitting
contiguously inside one slab (a 64-thread block does one 1024-byte fully-coalesced load;
`pxq6.cuh:680-726`).

---

## 4. The frozen tables — GLOBAL STATIC, not per-tensor

**PX16 book (the "book"):** 16 entries, **fp32 constants** in the header, every value
exactly representable in fp16 (verified: all 16 round-trip through fp16 unchanged).
It is a **global compile-time table**, identical for every tensor, every row, every model.
FACT: `ggml/include/ggml-pxq6-tables.h:33-37` (`PXQ6_BOOK_INIT`), instantiated as a
`__device__` array `pxq6_book_g[16]` (`pxq6.cuh:79`) and as a host array
`pxq6_book_q_[16]` (`src/pxq6-quantize.inc.cpp:41`).

Invariants enforced at runtime (FACT, `pxq6.cuh:99-110`): strictly ascending,
`book[7] == 0.0f`, `book[15] == 1.0f`, and all SUB entries fp16-snap-idempotent.

| idx | hex float (source literal) | decimal (fp32) | fp32 bits LE | fp16 bits |
|----|----|----|----|----|
| 0  | `-0x1.f9c0p-1` | -0.98779297 | 0xbf7ce000 | 0xbbe7 |
| 1  | `-0x1.7880p-1` | -0.73535156 | 0xbf3c4000 | 0xb9e2 |
| 2  | `-0x1.1e00p-1` | -0.55859375 | 0xbf0f0000 | 0xb878 |
| 3  | `-0x1.adc0p-2` | -0.41967773 | 0xbed6e000 | 0xb6b7 |
| 4  | `-0x1.3440p-2` | -0.30102539 | 0xbe9a2000 | 0xb4d1 |
| 5  | `-0x1.8e40p-3` | -0.19445801 | 0xbe472000 | 0xb239 |
| 6  | `-0x1.8740p-4` | -0.09552002 | 0xbdc3a000 | 0xae1d |
| 7  | `0x0.0p+0`     |  0.00000000 | 0x00000000 | 0x0000 |
| 8  | `0x1.5b00p-4`  |  0.08471680 | 0x3dad8000 | 0x2d6c |
| 9  | `0x1.5ec0p-3`  |  0.17126465 | 0x3e2f6000 | 0x317b |
| 10 | `0x1.0c40p-2`  |  0.26196289 | 0x3e862000 | 0x3431 |
| 11 | `0x1.7140p-2`  |  0.36059570 | 0x3eb8a000 | 0x35c5 |
| 12 | `0x1.e280p-2`  |  0.47119141 | 0x3ef14000 | 0x378a |
| 13 | `0x1.3380p-1`  |  0.60058594 | 0x3f19c000 | 0x38ce |
| 14 | `0x1.8800p-1`  |  0.76562500 | 0x3f440000 | 0x3a20 |
| 15 | `0x1.0p+0`     |  1.00000000 | 0x3f800000 | 0x3c00 |

Note the book is **asymmetric and non-uniform** (a learned/calibrated codebook), inherited
byte-frozen from the retired PXQ5 type (`ggml-pxq6-tables.h:18, 32`). There is no zero-point
and no offset: index 7 is the exact zero.

**SUB16 sub-scale table:** 16 entries, fp32 constants, again **global static**, all
fp16-exact, strictly ascending, `SUB16[0] > 0`. FACT: `ggml-pxq6-tables.h:40-44`
(`PXQ6_SUB16_INIT`), `pxq6.cuh:80` (`pxq6_sub16_g`), `pxq6-quantize.inc.cpp:42`.

| idx | hex float | decimal | fp32 bits LE | fp16 bits |
|----|----|----|----|----|
| 0  | `0x1.b7c0p-3` | 0.21472168 | 0x3e5be000 | 0x32df |
| 1  | `0x1.36c0p-2` | 0.30346680 | 0x3e9b6000 | 0x34db |
| 2  | `0x1.72c0p-2` | 0.36206055 | 0x3eb96000 | 0x35cb |
| 3  | `0x1.a2c0p-2` | 0.40893555 | 0x3ed16000 | 0x368b |
| 4  | `0x1.ccc0p-2` | 0.44995117 | 0x3ee66000 | 0x3733 |
| 5  | `0x1.f300p-2` | 0.48730469 | 0x3ef98000 | 0x37cc |
| 6  | `0x1.0bc0p-1` | 0.52294922 | 0x3f05e000 | 0x382f |
| 7  | `0x1.1e00p-1` | 0.55859375 | 0x3f0f0000 | 0x3878 |
| 8  | `0x1.3040p-1` | 0.59423828 | 0x3f182000 | 0x38c1 |
| 9  | `0x1.4380p-1` | 0.63183594 | 0x3f21c000 | 0x390e |
| 10 | `0x1.5800p-1` | 0.67187500 | 0x3f2c0000 | 0x3960 |
| 11 | `0x1.6ec0p-1` | 0.71630859 | 0x3f376000 | 0x39bb |
| 12 | `0x1.8880p-1` | 0.76660156 | 0x3f444000 | 0x3a22 |
| 13 | `0x1.a640p-1` | 0.82470703 | 0x3f532000 | 0x3a99 |
| 14 | `0x1.cac0p-1` | 0.89599609 | 0x3f656000 | 0x3b2b |
| 15 | `0x1.f9c0p-1` | 0.98779297 | 0x3f7ce000 | 0x3be7 |

`SUB8` (HQ tier only, `ggml-pxq6-tables.h:47-51`) is a different 16-entry table; a decoder
that only handles id 252 never needs it.

**Table overrides (must be honoured by any independent decoder that wants bit-parity):**
env vars `PXA_PXQ6_BOOK` / `PXA_PXQ6_SUB` / `PXA_PXQ6_SUB_HQ` replace the tables at runtime,
fp16-snapped (`pxq6.cuh:288-302`, `pxq6-quantize.inc.cpp:140-176`). The file records the
tables actually used in gguf KVs — see §8. **For the shipped
`/path/to/models/pxa-models/Qwen3.8-27B-PXQ4.gguf`, assume the frozen tables unless the KVs say
otherwise (ASSUMPTION — I did not read that file's KVs in this task; verify
`pxa.pxq6.book` / `pxa.pxq6.sub` before trusting the literals above for it).**

**Answer to "per-tensor / per-row / global":**
- BOOK: **global static** (compiled in; mirrored into the file's KVs for provenance).
- SUB16: **global static** (same).
- anchor: **per weight row**, fp16, in the panel header.
- 4-bit sub index: **per (row, 16-element block)** — i.e. 2 per row per 32-column slab.
- 4-bit code: **per element**.

---

## 5. E16-row anchor and SUB16 composition (FACT)

`row_effs` for PXQ4 (`pxq6.cuh:326-331`):
```c
__device__ static float anchor(const uint8_t * panel, int row) {
    return __half2float(((const half *)panel)[row]);        // panel[0..127] = 64 fp16
}
__device__ static void row_effs(const uint8_t * slab, int row, float anch,
                                const float * sub, float * eff) {
    const int sb = slab[row];
    eff[0] = anch * sub[sb & 0xf];    // elems 0-15
    eff[1] = anch * sub[sb >> 4];     // elems 16-31
}
```
`NEFF = 2` for PXQ4 (`pxq6.cuh:318`) — two distinct effective scales per 32-element block.

Mirrored bit-for-bit in the writer (`pxq6-quantize.inc.cpp:322-325`:
`slab[r] = s[0] | (s[1] << 4)` with the comment "lo nibble = elems 0-15") and in the CPU
reference dequant (`pxq6-quantize.inc.cpp:356-357`).

**Anchor semantics (FACT, `pxq6-quantize.inc.cpp:263-284`):** the anchor is the row's
fp16-snapped absmax (clamped to 65504), optionally refined by a ±2/16-octave search
(`PXA_PXQ_ANCHOR_FIT`) minimising weighted error. It is **not** a power of two and is
**not** derivable from the data — it must be read from the header. A zero row is
`anchor = fp16(+0)`.

**Two-level scale composition:** `eff = fp32(anchor_fp16) * SUB16[s4]`, one fp32 multiply
per 16-element block. Because `SUB16 ∈ [0.2147, 0.9878]` and `|BOOK| ≤ 1`, the maximum
representable magnitude in a row is `anchor * 0.98779 * 0.98779 ≈ 0.9757 * anchor`
(INFERENCE from the table values) — the anchor is an upper envelope, not an exact absmax
reproducer.

---

## 6. Exact dequant arithmetic (the parity-locked contract)

FACT — this is the contract, stated identically in three places:
`ggml-pxq6-tables.h:7-9`, `pxq6.cuh:13-15`, and implemented as the CPU reference at
`pxq6-quantize.inc.cpp:334-370`; the CUDA form is `pxq6.cuh:705-716`.

```
eff = fp32(anchor_fp16) * SUB16[s4]      (fp32 multiply, once per 16-element block)
w   = eff * fp32(BOOK[c])                (fp32 multiply, per element)
```
Rounding policy (FACT, `pxq6.cuh:13-15`): the **prefill GEMM** snaps `__float2half_rn(w)`;
the **decode mmv** accumulates in fp32 without snapping. So "the fp32 value above" is the
canonical weight; fp16 snapping is a consumer choice, not part of the format.

### Reference pseudocode (self-contained; independent decoder)

```python
BOOK  = [...16 fp32 constants from §4...]     # global
SUB16 = [...16 fp32 constants from §4...]     # global

HDR  = 128
SLAB = 1088

def pxq4_dequant(buf, R, K, E=1):
    """buf: bytes of the tensor. Returns w[E][R][K] as fp32.
       Requires R % 64 == 0 and K % 32 == 0."""
    assert R % 64 == 0 and K % 32 == 0
    KB          = K // 32                      # slabs per panel
    P           = R // 64                      # panels per expert
    panel_bytes = HDR + KB * SLAB              # == 128 + 34*K
    expert_bytes = P * panel_bytes
    w = zeros((E, R, K), float32)

    for e in range(E):
      for p in range(P):
        panel = e*expert_bytes + p*panel_bytes
        # 64 fp16 little-endian row anchors
        anchors = [fp16_to_fp32(u16_le(buf, panel + 2*r)) for r in range(64)]
        for r in range(64):
          anch = anchors[r]
          row  = 64*p + r
          for kb in range(KB):
            slab = panel + HDR + kb*SLAB
            sb   = buf[slab + r]               # scale byte, SoA plane
            eff_lo = anch * SUB16[ sb & 0x0F]   # elements  0..15 of this block
            eff_hi = anch * SUB16[(sb >> 4)  ]  # elements 16..31 of this block
            code_row = slab + 64 + 16*r         # 16 bytes = 32 nibbles
            for b in range(16):                 # b = element-PAIR index
              byte = buf[code_row + b]
              i0, i1 = 2*b, 2*b + 1
              e0 = eff_lo if i0 < 16 else eff_hi
              e1 = eff_lo if i1 < 16 else eff_hi
              w[e][row][kb*32 + i0] = e0 * BOOK[byte & 0x0F]   # lo nibble = even element
              w[e][row][kb*32 + i1] = e1 * BOOK[byte >> 4  ]   # hi nibble = odd element
    return w
```

Vectorised equivalent (INFERENCE, arithmetically identical — useful for an offline
converter):

```python
# buf -> uint8 array; drop nothing, slice by structure
hdr  = frombuffer(...)              # [P, 64] float16   at panel+0
sc   = ...                          # [P, KB, 64] uint8  at panel+128+kb*1088 + r
cod  = ...                          # [P, KB, 64, 16] uint8 at panel+128+kb*1088+64+16r
lo   = SUB16[sc & 0xF]              # [P,KB,64]
hi   = SUB16[sc >> 4]               # [P,KB,64]
nib  = stack([cod & 0xF, cod >> 4], axis=-1).reshape(P,KB,64,32)   # element order
val  = BOOK[nib]                                                    # [P,KB,64,32]
eff  = concat([repeat(lo,16), repeat(hi,16)], axis=-1)              # [P,KB,64,32]
w    = hdr[:,None,:,None] * eff * val            # [P,KB,64,32]
w    = w.transpose(0,2,1,3).reshape(R, K)        # panels/rows -> row-major [R,K]
```
The final `transpose(0,2,1,3).reshape` **is** the de-interleave: it is the only permutation
between PXQ4 order and row-major.

Zero handling (FACT, analogous statement for the sibling 5-bit tier at
`ggml-pxq6-tables.h:62-63`; for PXQ4 INFERENCE from the arithmetic): a genuinely zero row
is `anchor = +0.0` ⇒ every product is exactly `+0.0` regardless of the sub/code bytes.
There is no code that maps to exact zero *except* `BOOK[7] == 0.0`, which is available
per-element.

---

## 7. Sizes, worked

For a tensor with `R` rows and `K` columns:

```
slabs per panel   KB           = K / 32
panel bytes                    = 128 + 1088*KB      = 128 + 34*K
panels                         = R / 64
tensor bytes (per expert)      = (R/64) * (128 + 34*K) = 2*R + 17*R*K/32
ggml nbytes                    = R * (2 + 17*K/32)   — identical (§2)
```
Cross-check against the quantizer's own size formula (FACT,
`pxq6-quantize.inc.cpp:376`):
```c
exp_bytes = (R/64)*(PXQ6_HDR_BYTES + (K/32)*(tier ? PXQ6HQ_SLAB_BYTES : PXQ6_SLAB_BYTES));
```
✓ same expression.

Worked examples on this model's real shapes (hidden 5120):

| tensor | R | K | KB | panels | panel bytes | total bytes | bpw |
|---|---|---|---|---|---|---|---|
| `ffn_gate` | 17408 | 5120 | 160 | 272 | 174208 | 47,384,576 | 4.25313 |
| `ffn_down`  | 5120 | 17408 | 544 | 80 | 592000 | 47,360,000 | 4.25092 |
| `attn_q`   | 12288 | 5120 | 160 | 192 | 174208 | 33,447,936 | 4.25313 |
| `attn_output` | 5120 | 6144 | 192 | 80 | 209024 | 16,721,920 | 4.25260 |

(arithmetic only — no file was inspected for these; INFERENCE from the shapes given in the
brief plus the formula above.)

---

## 8. Provenance KVs written into the gguf (FACT)

`src/llama-quantize.cpp:1971-1988`:
```
pxa.pxq.backbone_rev   u32     (2 when rev-2 backbone table active, else 1)
pxa.pxq.backbone_map   string  (which class->type table built the file)
pxa.pxq6.version       u32  = 1
pxa.pxq6.tier          str  = "core" | "hq"        ("lm32" for the 5-bit id-256 type)
pxa.pxq6.book          f32[16]  the BOOK actually used
pxa.pxq6.sub           f32[16]  the SUB actually used (SUB16 for core, SUB8 for hq)
```
Key names keep the historical `pxq6` spelling on purpose — "historical key names —
file-format contract" (`llama-quantize.cpp:1751`).

**A converter should read `pxa.pxq6.book` / `pxa.pxq6.sub` from the file rather than
hard-coding the literals** — that is the only override-safe path, and it is 128 bytes of KV.

---

## 9. A "PXQ4 file" is NOT uniformly PXQ4 (FACT — critical for any loader)

The rev-2 backbone table (`src/llama-quantize.cpp:1314-1410`) resolves per tensor class:

| class | resolved type | file:line |
|---|---|---|
| `*_exps.weight` | expert path owns it (N/A on this dense model) | 1322 |
| `output.weight` | via `pxa_pxq_head_type()` (q8_0 per the brief) | 1323 |
| `*.nextn.*` (MTP) | `GGML_TYPE_Q8_0` | 1327 |
| `ssm_alpha.weight`, `ssm_beta.weight` | `GGML_TYPE_Q8_0` (32 rows < 64-row panel) | 1331-1333 |
| `ssm_out.weight` | left on legacy landing (`GGML_TYPE_COUNT`) in v1 | 1339 |
| `token_embd.weight` | `Q6_K` if `ne[0] % QK_K == 0` else `Q8_0` | 1343-1345 |
| `attn_k`, `attn_v`, `attn_v_b` | `Q8_0` (overridable by `PXA_PXQ_KV`) | 1355-1371 |
| `attn_gate.weight` with `ne[1] <= 256` (per-head) | `GGML_TYPE_F16` | 1373-1375 |
| any geometry failure (`rows%64` or `K%32`) | `GGML_TYPE_Q8_0` — never a silent demotion | 1399-1401 |
| the rest of the GEMM backbone | the native PXQ tier (PXQ4 for a PXQ4 build) | 1402+ |

Norms / routers / `f32` tensors are untouched. **Any vLLM-side loader must dispatch per
tensor on the gguf tensor type, not on a file-level "quantization = pxq4" flag.**

---

## 10. Sharding consequences (what §3 implies — supports the separate sharding analysis)

INFERENCE, but a direct consequence of the FACTs in §3:

1. **A panel is a self-contained contiguous byte range** covering exactly 64 weight rows
   and all of K. Its header holds only those 64 rows' anchors; its slabs hold only those
   64 rows' scale bytes and codes. Nothing in a panel references another panel.
   ⇒ **Column-parallel sharding (split output rows R) at any multiple of 64 is a pure
   `memcpy` of whole panels** — byte-identical, no re-quantization, no header rebuild.

2. **A slab is self-contained per 32-column block.** The only cross-K object is the fp16
   row anchor in the header, and it is a scalar multiplier with **no dependence on which
   K-blocks are present** (`eff = anchor * SUB16[s4]`, `pxq6.cuh:326-331`).
   ⇒ **Row-parallel sharding (split K) at any multiple of 32 is a byte-gather: for every
   panel, copy the 128 B header verbatim, then copy the contiguous run of slabs
   `[kb0, kb1)`.** Numerically bit-identical: each surviving element's `anchor`, `s4` and
   `code` bytes are unchanged. Note the anchor was *chosen* over the full row, so a K-shard
   is not the anchor an independent quantizer would pick — but it is the anchor the full
   product needs, and partial dot products summed across ranks reproduce the unsharded
   result exactly.
   ⇒ For a K-shard the resulting sub-tensor is itself a valid PXQ4 tensor with
   `K' = 32*(kb1-kb0)`, `panel_bytes' = 128 + 34*K'`. Its bpw rises to `4.25 + 16/K'`
   (the 128 B header is duplicated per rank — for TP=4 on K=17408 that is
   4.25 + 16/4352 = 4.2537 bpw, +0.09% bytes/rank vs unsharded).

3. **What you may NOT do:** slice at a row boundary that is not a multiple of 64, slice K
   at a boundary that is not a multiple of 32, or treat any row as a contiguous byte range.
   The last one is the specific reason the brief's warning about vLLM's generic GGUF
   sharder is correct: that sharder assumes per-row-contiguous blocks, which panel
   interleave violates (`pxq-cpu.h:5-9` states the same for ggml's own `to_float`).

---

## 11. Deltas for the neighbouring types (for a converter that must not confuse them)

FACT, all in `ggml/include/ggml-pxq6-tables.h` and `pxq6.cuh:346-380, 385-430`:

| type | id | codes | scale plane / slab | slab bytes | CODE_OFF | type_size | bpw |
|---|---|---|---|---|---|---|---|
| PXQ4 | 252 | 4-bit BOOK[16], 16 B/row | 1 B/row, 2 subs (bs16) | 1088 | 64 | 17 | 4.25 + 16/K |
| PXQ4HQ | 253 | 4-bit BOOK[16], 16 B/row | 2 B/row, 4 subs (bs8, SUB8) | 1152 | 128 | 18 | 4.5 + 16/K |
| PXQ6 | 256 | 5-bit LM32[32], 20 B/row (16 B nibble plane + LE u32 hi-bit plane) | 1 B/row, SUB16 | 1344 | 64 | 21 | 5.25 + 16/K |
| PXQ3 | 255 | 3-bit LM8, bit-plane packed | 1 B/row, SUB16 | — | — | 13 | 3.25 + 16/K |
| PXQ2 | 254 | 2-bit LM4 | 1 B/row, SUB16 | — | — | 9 | 2.25 + 16/K |
| PXQ1 | 248 | 1-bit sign | 1 B/row, SUB16 | — | — | 5 | 1.25 + 16/K |

All of them share the **identical** 128 B / 64-row fp16 anchor header and the SUB16 table
(`ggml.h:474-497`, `ggml.c:1437-1460`). The anchor header and the panel geometry are the
invariant; only the code plane changes.

---

## 12. Open items / not verified

- I did **not** open `/path/to/models/pxa-models/Qwen3.8-27B-PXQ4.gguf` in this task. Its
  actual per-tensor type map and its `pxa.pxq6.book` / `pxa.pxq6.sub` KVs are **not
  verified** here; §9 predicts the map from the backbone table source, and §4 gives the
  compiled-in defaults.
- `PXA_PXQ_KQW` (imatrix-derived per-column weights, `pxq6-quantize.inc.cpp:308`) and
  `PXA_PXQ_ANCHOR_FIT` affect only *encode*; they leave no trace in the decode path.
  Not verified whether the file records which of these were on.
- The HQ (`sub8`) table values are in `ggml-pxq6-tables.h:47-51`; I did not transcribe them
  here because a pure id-252 decoder never needs them.
