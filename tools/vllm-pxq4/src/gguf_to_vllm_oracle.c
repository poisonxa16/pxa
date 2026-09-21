/* gguf_to_vllm_oracle.c — gate G1 oracle: the ENGINE'S OWN CPU decoder, compiled standalone.
 *
 * WHY THIS FILE. reference.py claims to reproduce pxa_deq_row_pxq6 (ggml/src/pxq-cpu.c:135-158)
 * bit-for-bit in fp32. A claim like that is worth nothing if it is checked against a second
 * transcription by the same author. So this harness pulls the production function VERBATIM —
 * the body below is a byte copy of pxq-cpu.c:126-158, and the tables are a byte copy of
 * ggml-pxq6-tables.h:33-44 — links it against nothing, and prints float32 bits. Compare its
 * output to reference.dequant() and the gate is real.
 *
 * The two things NOT copied verbatim, and why they are safe:
 *   - GGML_COMPUTE_FP16_TO_FP32. In the real build this is a table lookup or an F16C/NEON
 *     intrinsic; here it is the IEEE-754 binary16 -> binary32 widening, which is exact for
 *     every input including subnormals, so every implementation agrees by definition. The
 *     harness asserts this against a brute-force sweep of all 65536 half patterns.
 *   - The env-var table overrides (PXA_PXQ6_BOOK / PXA_PXQ6_SUB, pxq-cpu.c:80-82). Omitted:
 *     the artifact records its tables in the gguf KVs and the converter cross-checks them.
 *
 * Nothing in /path/to/engine-repo was modified to produce this; the source lines were read
 * with sed and copied here.
 *
 * build: cc -O2 -std=c11 -o oracle gguf_to_vllm_oracle.c
 * usage: oracle <blob> <N> <K>          -> writes N*K float32 (little-endian) to stdout
 */
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- verbatim: ggml/include/ggml-pxq6-tables.h:22-27,33-44 ------------------------------ */
#define PXQ6_QK          32
#define PXQ6_TYPE_SIZE   17
#define PXQ6_BM          64
#define PXQ6_SLAB_BYTES  1088
#define PXQ6_HDR_BYTES   128     /* 64 x fp16 row anchors at the head of every 64-row panel */
#define PXQ6_ROW_META    2       /* ggml row_meta_size: 2 B/row == 128 B / 64-row panel */
#define PXQ6HQ_SLAB_BYTES 1152

#define PXQ6_BOOK_INIT { \
    -0x1.f9c0000000000p-1f, -0x1.7880000000000p-1f, -0x1.1e00000000000p-1f, -0x1.adc0000000000p-2f, \
    -0x1.3440000000000p-2f, -0x1.8e40000000000p-3f, -0x1.8740000000000p-4f, 0x0.0p+0f, \
    0x1.5b00000000000p-4f, 0x1.5ec0000000000p-3f, 0x1.0c40000000000p-2f, 0x1.7140000000000p-2f, \
    0x1.e280000000000p-2f, 0x1.3380000000000p-1f, 0x1.8800000000000p-1f, 0x1.0000000000000p+0f }

#define PXQ6_SUB16_INIT { \
    0x1.b7c0000000000p-3f, 0x1.36c0000000000p-2f, 0x1.72c0000000000p-2f, 0x1.a2c0000000000p-2f, \
    0x1.ccc0000000000p-2f, 0x1.f300000000000p-2f, 0x1.0bc0000000000p-1f, 0x1.1e00000000000p-1f, \
    0x1.3040000000000p-1f, 0x1.4380000000000p-1f, 0x1.5800000000000p-1f, 0x1.6ec0000000000p-1f, \
    0x1.8880000000000p-1f, 0x1.a640000000000p-1f, 0x1.cac0000000000p-1f, 0x1.f9c0000000000p-1f }

#define PXQ6_SUB8_INIT { \
    0x1.58c0000000000p-3f, 0x1.e440000000000p-3f, 0x1.2640000000000p-2f, 0x1.5280000000000p-2f, \
    0x1.7a80000000000p-2f, 0x1.a040000000000p-2f, 0x1.c4c0000000000p-2f, 0x1.e900000000000p-2f, \
    0x1.07c0000000000p-1f, 0x1.1c80000000000p-1f, 0x1.32c0000000000p-1f, 0x1.4bc0000000000p-1f, \
    0x1.68c0000000000p-1f, 0x1.8b40000000000p-1f, 0x1.b700000000000p-1f, 0x1.f380000000000p-1f }

/* ---- verbatim: ggml/src/pxq-cpu.c:53-55 ------------------------------------------------- */
static float pxa_tab_px16_book[16] = PXQ6_BOOK_INIT;     /* PXQ6/PXQ6HQ book */
static float pxa_tab_sub16[16]     = PXQ6_SUB16_INIT;    /* PXQ6-core / PXQ2 / PXQ3 subs */
static float pxa_tab_sub8[16]      = PXQ6_SUB8_INIT;     /* PXQ6HQ subs */

/* ---- IEEE binary16 -> binary32. Exact for every one of the 65536 patterns; see main(). --- */
static float ggml_compute_fp16_to_fp32(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu;
    const uint32_t m = h & 0x03FFu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) { bits = s; }
        else {
            /* subnormal half: renormalise into a normal float */
            uint32_t mm = m, ee = 127 - 15 + 1;
            while (!(mm & 0x0400u)) { mm <<= 1; ee--; }
            mm &= 0x03FFu;
            bits = s | (ee << 23) | (mm << 13);
        }
    } else if (e == 0x1Fu) {
        bits = s | 0x7F800000u | (m << 13);        /* inf / NaN */
    } else {
        bits = s | ((e + 127u - 15u) << 23) | (m << 13);
    }
    union { uint32_t u; float f; } v; v.u = bits; return v.f;
}
#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)

/* ---- verbatim: ggml/src/pxq-cpu.c:126-158 ----------------------------------------------- */
// 16 B nibble code rows: byte b = code(2b) | code(2b+1) << 4
static inline void pxa_deq_pairs16(const uint8_t * q, const float * book, const float * eff, int eff_shift, float * o) {
    for (int b = 0; b < 16; ++b) {
        const int i0 = 2*b, i1 = 2*b + 1;
        o[i0] = eff[i0 >> eff_shift] * book[q[b] & 0xf];
        o[i1] = eff[i1 >> eff_shift] * book[q[b] >> 4];
    }
}

static void pxa_deq_row_pxq6(const uint8_t * base, int64_t row, int64_t k, float * dst, bool hq) {
    const int64_t KB = k/32;
    const int     slab_bytes = hq ? PXQ6HQ_SLAB_BYTES : PXQ6_SLAB_BYTES;   // 1152 : 1088
    const int     code_off   = hq ? 128 : 64;
    const int64_t p = row >> 6;
    const int     r = (int)(row & 63);
    const uint8_t * panel = base + p*(PXQ6_HDR_BYTES + KB*slab_bytes);
    const float anchor = GGML_COMPUTE_FP16_TO_FP32(((const uint16_t *)panel)[r]);
    const float * sub = hq ? pxa_tab_sub8 : pxa_tab_sub16;
    for (int64_t kb = 0; kb < KB; ++kb) {
        const uint8_t * slab = panel + PXQ6_HDR_BYTES + kb*slab_bytes;
        float eff[4];
        if (hq) {
            eff[0] = anchor * sub[slab[2*r]   & 0xf];   // elems  0-7
            eff[1] = anchor * sub[slab[2*r]   >>  4];   // elems  8-15
            eff[2] = anchor * sub[slab[2*r+1] & 0xf];   // elems 16-23
            eff[3] = anchor * sub[slab[2*r+1] >>  4];   // elems 24-31
        } else {
            eff[0] = eff[1] = anchor * sub[slab[r] & 0xf];   // elems  0-15
            eff[2] = eff[3] = anchor * sub[slab[r] >>  4];   // elems 16-31
        }
        pxa_deq_pairs16(slab + code_off + r*16, pxa_tab_px16_book, eff, 3, dst + kb*32);
    }
}
/* ---- end verbatim ----------------------------------------------------------------------- */

/* ---- verbatim: ggml/src/ggml-common.h:182-187,227-231,382-387 (block structs) ------------ */
#define QK8_0 32
#define QK_K  256
#define QK_MXFP4 32
typedef uint16_t ggml_half;
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;
typedef struct { uint8_t e; uint8_t qs[QK_MXFP4/2]; } block_mxfp4;
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; ggml_half d; } block_q6_K;

#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)

/* ---- verbatim: ggml/src/ggml-impl.h:40-45 ----------------------------------------------- */
static inline float ggml_e8m0_to_fp32_half(uint8_t x) {
    static uint32_t val[2] = { 0x00200000, 0x00400000 };
    union { float f; uint32_t u; } helper;
    helper.u = x >= 2 ? (uint32_t)(x - 1) << 23u : val[x];
    return helper.f;
}
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

/* ---- verbatim: ggml/src/ggml-common.h:2244-2246 ----------------------------------------- */
static const int8_t kvalues_mxfp4[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

/* ---- verbatim: ggml/src/ggml-quants.c:1697-1711 ----------------------------------------- */
void dequantize_row_q8_0(const block_q8_0 * restrict x, float * restrict y, int64_t k) {
    static const int qk = QK8_0;
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

/* ---- verbatim: ggml/src/ggml-quants.c:3231-3260 ----------------------------------------- */
void dequantize_row_q6_K(const block_q6_K * restrict x, float * restrict y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * restrict ql = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* ---- verbatim: ggml/src/iqk/iqk_quantize.cpp:4300-4312 (C++ -> C: constexpr -> const) ---- */
void dequantize_row_mxfp4(const block_mxfp4 * x, float * y, int64_t k) {
    const int kBlockSize = QK_MXFP4;
    int nblock = k/kBlockSize;
    for (int ib = 0; ib < nblock; ++ib) {
        float d = GGML_E8M0_TO_FP32_HALF(x[ib].e);
        for (int j = 0; j < kBlockSize/2; ++j) {
            y[j             ] = d * kvalues_mxfp4[x[ib].qs[j] & 0xf];
            y[j+kBlockSize/2] = d * kvalues_mxfp4[x[ib].qs[j] >>  4];
        }
        y  += kBlockSize;
    }
}
/* ---- end verbatim ----------------------------------------------------------------------- */

static void selfcheck_fp16(void) {
    /* Every half pattern must widen exactly. Anything that fails here means the harness, not
     * the format, is wrong — and it would silently perturb the anchor for subnormal rows. */
    for (uint32_t i = 0; i < 65536u; ++i) {
        float a = ggml_compute_fp16_to_fp32((uint16_t)i);
        _Float16 h; memcpy(&h, &i, 2);
        float b = (float)h;
        uint32_t ua, ub; memcpy(&ua, &a, 4); memcpy(&ub, &b, 4);
        if (ua != ub && !(a != a && b != b)) {
            fprintf(stderr, "fp16 widen mismatch at 0x%04x: %08x vs %08x\n", i, ua, ub);
            exit(2);
        }
    }
}


static int other_type(const char * path, const char * type, int64_t N, int64_t K) {
    selfcheck_fp16();
    size_t blk, tsz;
    if      (!strcmp(type, "q8_0"))  { blk = 32;  tsz = sizeof(block_q8_0);  }
    else if (!strcmp(type, "q6_K"))  { blk = 256; tsz = sizeof(block_q6_K);  }
    else if (!strcmp(type, "mxfp4")) { blk = 32;  tsz = sizeof(block_mxfp4); }
    else { fprintf(stderr, "unknown type %s\n", type); return 1; }
    if ((size_t)K % blk) { fprintf(stderr, "K%%%zu required\n", blk); return 1; }
    const size_t rowb = (size_t)K / blk * tsz;
    const size_t need = (size_t)N * rowb;
    FILE * f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    uint8_t * blob = (uint8_t *)malloc(need);
    if (fread(blob, 1, need, f) != need) { fprintf(stderr, "short read (want %zu)\n", need); return 1; }
    fclose(f);
    float * row = (float *)malloc((size_t)K * sizeof(float));
    for (int64_t r = 0; r < N; ++r) {
        const void * p = blob + (size_t)r * rowb;
        if      (!strcmp(type, "q8_0"))  dequantize_row_q8_0 ((const block_q8_0 *)p,  row, K);
        else if (!strcmp(type, "q6_K"))  dequantize_row_q6_K ((const block_q6_K *)p,  row, K);
        else                             dequantize_row_mxfp4((const block_mxfp4 *)p, row, K);
        if (fwrite(row, sizeof(float), (size_t)K, stdout) != (size_t)K) return 1;
    }
    free(row); free(blob);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 2 && strcmp(argv[1], "--selfcheck") == 0) {
        selfcheck_fp16();
        fprintf(stderr, "fp16 widen: 65536/65536 exact\n");
        return 0;
    }
    if (argc == 5) return other_type(argv[1], argv[2], atoll(argv[3]), atoll(argv[4]));
    if (argc != 4) {
        fprintf(stderr, "usage: %s <blob> <N> <K>            (pxq4 panel blob)\n"
                        "       %s <blob> <type> <N> <K>     (type = q8_0|q6_K|mxfp4)\n"
                        "       %s --selfcheck\n", argv[0], argv[0], argv[0]);
        return 1;
    }
    selfcheck_fp16();
    const int64_t N = atoll(argv[2]), K = atoll(argv[3]);
    if (N % 64 || K % 32) { fprintf(stderr, "N%%64 and K%%32 required\n"); return 1; }
    const size_t need = (size_t)(N / 64) * (PXQ6_HDR_BYTES + (size_t)(K / 32) * PXQ6_SLAB_BYTES);

    FILE * f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    uint8_t * blob = (uint8_t *)malloc(need);
    if (fread(blob, 1, need, f) != need) { fprintf(stderr, "short read (want %zu)\n", need); return 1; }
    fclose(f);

    float * row = (float *)malloc((size_t)K * sizeof(float));
    for (int64_t r = 0; r < N; ++r) {
        pxa_deq_row_pxq6(blob, r, K, row, false);
        if (fwrite(row, sizeof(float), (size_t)K, stdout) != (size_t)K) return 1;
    }
    free(row); free(blob);
    return 0;
}
