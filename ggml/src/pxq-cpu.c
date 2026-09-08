// pxq-cpu.c — CPU panel-dequant + slow-but-correct matmul fallbacks for the PXQ slab types.
// See pxq-cpu.h for the contract and the layout-source citations.
//
// FORMAT SUMMARY (all verified against the native quantizers / repack tool):
//
//                 hdr   slab   sc B/row  code B/row  code_off  eff granularity
//   PXQ4 (252)    128   1088       1         16          64    per 16 (anchor x SUB16, 2x4b)
//   PXQ4HQ (253)  128   1152       2         16         128    per  8 (anchor x SUB8,  4x4b)
//   PXQ2 (254)    128    576       1          8          64    per 16 (anchor x SUB16)
//   PXQ3 (255)    128    832       1         12          64    per 16 (anchor x SUB16)
//   PXQ1 (248)    128    320       1          4          64    per 16 (anchor x SUB16)
//   PXQ6 (256)    128   1344       1         20          64    per 16 (anchor x SUB16)
//   (ids 250 + 251, the retired MXFP4-repack and PXQ5 legacy types, were removed 2026-07-21.)
//
//   PXQ1 + PXQ6 completed 2026-09-01 so that EVERY PXQ tier decodes without a GPU: that is what
//   lets llama-quantize requantize from a PXQ source and llama-pxq-export run --cpu. Layout
//   ground truth: ggml/include/ggml-pxq1-tables.h and the PXQ6R block of ggml-pxq6-tables.h,
//   cross-read against the CUDA policies pxq6_pol_p1 (ggml-cuda/pxq23.cuh) and pxq6_pol_p6r
//   (ggml-cuda/pxq6.cuh), which are the shipping decoders.
//
//   panel  = hdr (64 x fp16 row anchors when hdr==128) + (k/32) slabs; panels row-major.
//   slab   = 64-row scale SoA + 64 code rows.
//   16 B code rows (PXQ4/PXQ4HQ): byte b = code(elem 2b) | code(elem 2b+1) << 4.
//   8 B code rows (PXQ2): 2 bits/elem, elem j at bits 2*(j&3) of byte j>>2 (LE words).
//   12 B code rows (PXQ3): bit-plane, three LE u32 words: w0 = low 2 bits of elems 0-15,
//     w1 = low 2 bits of elems 16-31, w2 = bit2 plane (bit j = elem j, j = 0..31).
//   4 B code rows (PXQ1): one LE u32, 1 bit/elem, elem j at bit j; book = {-1,+1}.
//   20 B code rows (PXQ6): bytes 0-15 are the SAME nibble plane as PXQ4 (byte b = lo4(c[2b]) |
//     lo4(c[2b+1]) << 4), bytes 16-19 are one LE u32 hi-bit plane (bit j = bit 4 of c[j]),
//     giving 5-bit codes into the 32-entry LM32 book.
//   dequant (E16-row family contract, parity-locked):
//     eff = fp32(anchor_fp16) * SUB[s4];  w = eff * fp32(book[c])
//
// Table env overrides (PXA_PXQ6_BOOK/..., same names + fp16-snap as the quantizers and the
// CUDA kernels) are honored so a custom-table model keeps working on the CPU path too.

#include "pxq-cpu.h"
#include "pxq-dot.h"   // integer-domain PXQ x Q8 dot for the 4-bit tiers (phase 2)

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h" // block_mxfp4 / QK_MXFP4 / kvalues_mxfp4 for the MXFP4 dot below.
                          // The tables here expand to file-local statics (GGML_TABLE_BEGIN),
                          // exactly as in ggml-quants.c -- no duplicate symbols.

#include "ggml-impl.h"   // GGML_COMPUTE_FP16_TO_FP32 / GGML_COMPUTE_FP32_TO_FP16 (self-contained)

#include "ggml-pxq6-tables.h"
#include "ggml-pxq2-tables.h"
#include "ggml-pxq3-tables.h"
#include "ggml-pxq1-tables.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PXA_PXQ_ASSERT(x) \
    do { if (!(x)) { fprintf(stderr, "PXQ CPU fallback assert failed: %s at %s:%d\n", #x, __FILE__, __LINE__); abort(); } } while (0)

#if defined(_MSC_VER)
#define PXA_THREAD_LOCAL __declspec(thread)
#else
#define PXA_THREAD_LOCAL _Thread_local
#endif

// ---------------------------------------------------------------------------------------------
// tables (frozen headers + optional env overrides, fp16-snapped like the quantizers/kernels)
// ---------------------------------------------------------------------------------------------

static float pxa_tab_px16_book[16] = PXQ6_BOOK_INIT;     // PXQ6/PXQ6HQ book
static float pxa_tab_sub16[16]     = PXQ6_SUB16_INIT;    // PXQ6-core / PXQ2 / PXQ3 subs
static float pxa_tab_sub8[16]      = PXQ6_SUB8_INIT;     // PXQ6HQ subs
static float pxa_tab_lm4[4]        = PXQ2_BOOK_INIT;     // PXQ2 book
static float pxa_tab_lm8[8]        = PXQ3_BOOK_INIT;     // PXQ3 book
static float pxa_tab_lm32[32]      = PXQ6_LM32_INIT;     // PXQ6 (id 256) 5-bit book
static float pxa_tab_sign[2]       = PXQ1_BOOK_INIT;     // PXQ1 sign book {-1,+1}

static bool pxa_parse_n(const char * e, float * out, int want) {
    int n = 0;
    float v[32];   // 32: the PXQ6 LM32 book is the widest table an env override can carry
    char buf[1024];   // 32 hex-float entries do not fit in 512
    snprintf(buf, sizeof(buf), "%s", e);
    for (char * t = strtok(buf, ","); t && n < want; t = strtok(NULL, ",")) v[n++] = strtof(t, NULL);
    if (n != want) return false;
    for (int i = 0; i < want; ++i) {
        out[i] = GGML_COMPUTE_FP16_TO_FP32(GGML_COMPUTE_FP32_TO_FP16(v[i]));  // fp16-snap (spec)
    }
    return true;
}

// Idempotent, deterministic table init. May race on first use from multiple compute threads:
// every racer writes the exact same values (env parsed identically), aligned 4-byte float
// stores, so the race is benign — same pattern the iqk thread-local buffers rely on.
static void pxa_pxq_ensure_tables(void) {
    static volatile int done = 0;
    if (done) return;
    const char * e;
    float t[32];
    if ((e = getenv("PXA_PXQ6_BOOK"))   && pxa_parse_n(e, t, 16)) memcpy(pxa_tab_px16_book, t, sizeof(pxa_tab_px16_book));
    if ((e = getenv("PXA_PXQ6_SUB"))    && pxa_parse_n(e, t, 16)) memcpy(pxa_tab_sub16,     t, sizeof(pxa_tab_sub16));
    if ((e = getenv("PXA_PXQ6_SUB_HQ")) && pxa_parse_n(e, t, 16)) memcpy(pxa_tab_sub8,      t, sizeof(pxa_tab_sub8));
    // PXA_PXQ6R_BOOK overrides the LM32 book on the CUDA side (pxq6_maybe_upload_tables);
    // mirror it here or a custom-table PXQ6 file would decode differently on CPU than on GPU.
    // There is no PXQ1 book override on either side: the sign book {-1,+1} is not tunable.
    if ((e = getenv("PXA_PXQ6R_BOOK"))  && pxa_parse_n(e, t, 32)) memcpy(pxa_tab_lm32,      t, sizeof(pxa_tab_lm32));
    // PXA_PXQ_CEIL_V2 (2026-08-09): decode-side arm of the PXQ2/PXQ3 ceiling fix -- load the
    // v2 (max|book| == 1.0) LM4/LM8 books. Applied BEFORE the explicit PXA_PXQn_BOOK overrides
    // below so an explicit env table still wins. Default OFF: stock decode is byte-identical.
    if ((e = getenv("PXA_PXQ_CEIL_V2")) && atoi(e) != 0) {
        static const float pxa_lm4_v2[4] = PXQ2_BOOK_V2_INIT;
        static const float pxa_lm8_v2[8] = PXQ3_BOOK_V2_INIT;
        memcpy(pxa_tab_lm4, pxa_lm4_v2, sizeof(pxa_tab_lm4));
        memcpy(pxa_tab_lm8, pxa_lm8_v2, sizeof(pxa_tab_lm8));
        fprintf(stderr, "PXA_PXQ_CEIL_V2 ARMED (CPU decode): PXQ2/PXQ3 v2 books\n");
    }
    // PXA_PXQ2_V3 (2026-08-10): decode-side arm of the PXQ2 v3 refit book. PXQ2-only (PXQ3
    // keeps whatever the lines above chose); wins over CEIL_V2 for PXQ2; the explicit
    // PXA_PXQ2_BOOK override below still wins. Default OFF: stock decode is byte-identical.
    if ((e = getenv("PXA_PXQ2_V3")) && atoi(e) != 0) {
        static const float pxa_lm4_v3[4] = PXQ2_BOOK_V3_INIT;
        memcpy(pxa_tab_lm4, pxa_lm4_v3, sizeof(pxa_tab_lm4));
        fprintf(stderr, "PXA_PXQ2_V3 ARMED (CPU decode): PXQ2 v3 refit book\n");
    }
    if ((e = getenv("PXA_PXQ2_BOOK"))   && pxa_parse_n(e, t,  4)) memcpy(pxa_tab_lm4,       t, sizeof(pxa_tab_lm4));
    if ((e = getenv("PXA_PXQ3_BOOK"))   && pxa_parse_n(e, t,  8)) memcpy(pxa_tab_lm8,       t, sizeof(pxa_tab_lm8));
    // PXA_PXQ2_SUB / PXA_PXQ3_SUB alias the shared SUB16 (the quantizers keep separate copies
    // but always seed them from PXQ6_SUB16_INIT; an override of either must match PXA_PXQ6_SUB)
    if ((e = getenv("PXA_PXQ2_SUB"))    && pxa_parse_n(e, t, 16)) memcpy(pxa_tab_sub16,     t, sizeof(pxa_tab_sub16));
    if ((e = getenv("PXA_PXQ3_SUB"))    && pxa_parse_n(e, t, 16)) memcpy(pxa_tab_sub16,     t, sizeof(pxa_tab_sub16));
    done = 1;
}

void pxa_pxq_float_tables(const float ** book16, const float ** sub16, const float ** sub8) {
    pxa_pxq_ensure_tables();
    if (book16) *book16 = pxa_tab_px16_book;
    if (sub16)  *sub16  = pxa_tab_sub16;
    if (sub8)   *sub8   = pxa_tab_sub8;
}

// ---------------------------------------------------------------------------------------------
// per-type row dequant (row = global row index; data = 2D slice base)
// ---------------------------------------------------------------------------------------------

bool pxa_pxq_is_cpu_supported(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_PXQ4:
        case GGML_TYPE_PXQ4HQ:
        case GGML_TYPE_PXQ2:
        case GGML_TYPE_PXQ3:
        case GGML_TYPE_PXQ1:
        case GGML_TYPE_PXQ6:
            return true;
        default:
            return false;
    }
}

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

static void pxa_deq_row_pxq2(const uint8_t * base, int64_t row, int64_t k, float * dst) {
    const int64_t KB = k/32;
    const int64_t p = row >> 6;
    const int     r = (int)(row & 63);
    const uint8_t * panel = base + p*(PXQ2_HDR_BYTES + KB*PXQ2_SLAB_BYTES);
    const float anchor = GGML_COMPUTE_FP16_TO_FP32(((const uint16_t *)panel)[r]);
    for (int64_t kb = 0; kb < KB; ++kb) {
        const uint8_t * slab = panel + PXQ2_HDR_BYTES + kb*PXQ2_SLAB_BYTES;
        const float eff0 = anchor * pxa_tab_sub16[slab[r] & 0xf];   // elems  0-15
        const float eff1 = anchor * pxa_tab_sub16[slab[r] >>  4];   // elems 16-31
        const uint8_t * q = slab + 64 + r*8;
        float * o = dst + kb*32;
        for (int j = 0; j < 32; ++j) {
            const int c = (q[j >> 2] >> (2*(j & 3))) & 3;
            o[j] = (j < 16 ? eff0 : eff1) * pxa_tab_lm4[c];
        }
    }
}

static void pxa_deq_row_pxq3(const uint8_t * base, int64_t row, int64_t k, float * dst) {
    const int64_t KB = k/32;
    const int64_t p = row >> 6;
    const int     r = (int)(row & 63);
    const uint8_t * panel = base + p*(PXQ3_HDR_BYTES + KB*PXQ3_SLAB_BYTES);
    const float anchor = GGML_COMPUTE_FP16_TO_FP32(((const uint16_t *)panel)[r]);
    for (int64_t kb = 0; kb < KB; ++kb) {
        const uint8_t * slab = panel + PXQ3_HDR_BYTES + kb*PXQ3_SLAB_BYTES;
        const float eff0 = anchor * pxa_tab_sub16[slab[r] & 0xf];   // elems  0-15
        const float eff1 = anchor * pxa_tab_sub16[slab[r] >>  4];   // elems 16-31
        const uint8_t * in = slab + 64 + r*12;                      // three LE u32 words
        uint32_t w0 = 0, w1 = 0, w2 = 0;
        for (int i = 0; i < 4; ++i) {
            w0 |= (uint32_t)in[i]     << (8*i);
            w1 |= (uint32_t)in[4 + i] << (8*i);
            w2 |= (uint32_t)in[8 + i] << (8*i);
        }
        float * o = dst + kb*32;
        for (int j = 0; j < 16; ++j) {
            const int c0 = (int)(((w0 >> (2*j)) & 3) | (((w2 >> j)        & 1) << 2));
            const int c1 = (int)(((w1 >> (2*j)) & 3) | (((w2 >> (16 + j)) & 1) << 2));
            o[j]      = eff0 * pxa_tab_lm8[c0];
            o[16 + j] = eff1 * pxa_tab_lm8[c1];
        }
    }
}

// PXQ1 (248): 1 bit/elem into the sign book; all magnitude is in eff = anchor x SUB16[s4].
// Code row is ONE LE u32 at slab + 64 + row*4, element j at bit j -- the exact bit order
// pxq6_pol_p1::pair() reads (elem 2b at bit 2b, elem 2b+1 at bit 2b+1).
static void pxa_deq_row_pxq1(const uint8_t * base, int64_t row, int64_t k, float * dst) {
    const int64_t KB = k/32;
    const int64_t p = row >> 6;
    const int     r = (int)(row & 63);
    const uint8_t * panel = base + p*(PXQ1_HDR_BYTES + KB*PXQ1_SLAB_BYTES);
    const float anchor = GGML_COMPUTE_FP16_TO_FP32(((const uint16_t *)panel)[r]);
    for (int64_t kb = 0; kb < KB; ++kb) {
        const uint8_t * slab = panel + PXQ1_HDR_BYTES + kb*PXQ1_SLAB_BYTES;
        const float eff0 = anchor * pxa_tab_sub16[slab[r] & 0xf];   // elems  0-15
        const float eff1 = anchor * pxa_tab_sub16[slab[r] >>  4];   // elems 16-31
        const uint8_t * in = slab + 64 + r*4;
        uint32_t w = 0;
        for (int i = 0; i < 4; ++i) w |= (uint32_t)in[i] << (8*i);
        float * o = dst + kb*32;
        for (int j = 0; j < 32; ++j) {
            o[j] = (j < 16 ? eff0 : eff1) * pxa_tab_sign[(w >> j) & 1];
        }
    }
}

// PXQ6 (256): 5-bit LM32 codes. Bytes 0-15 of the 20 B code row are the same nibble plane as
// the 4-bit tiers (low 4 bits of each code); bytes 16-19 are one LE u32 whose bit j is bit 4
// of code j. Mirrors pxq6_pol_p6r::pair().
static void pxa_deq_row_pxq6r(const uint8_t * base, int64_t row, int64_t k, float * dst) {
    const int64_t KB = k/32;
    const int64_t p = row >> 6;
    const int     r = (int)(row & 63);
    const uint8_t * panel = base + p*(PXQ6R_HDR_BYTES + KB*PXQ6R_SLAB_BYTES);
    const float anchor = GGML_COMPUTE_FP16_TO_FP32(((const uint16_t *)panel)[r]);
    for (int64_t kb = 0; kb < KB; ++kb) {
        const uint8_t * slab = panel + PXQ6R_HDR_BYTES + kb*PXQ6R_SLAB_BYTES;
        const float eff0 = anchor * pxa_tab_sub16[slab[r] & 0xf];   // elems  0-15
        const float eff1 = anchor * pxa_tab_sub16[slab[r] >>  4];   // elems 16-31
        const uint8_t * in = slab + PXQ6R_CODE_OFF + r*20;
        uint32_t hi = 0;
        for (int i = 0; i < 4; ++i) hi |= (uint32_t)in[16 + i] << (8*i);
        float * o = dst + kb*32;
        for (int b = 0; b < 16; ++b) {
            const int c0 = (in[b] & 0xf) | (int)(((hi >> (2*b))     & 1) << 4);
            const int c1 = (in[b] >>  4) | (int)(((hi >> (2*b + 1)) & 1) << 4);
            o[2*b]     = (2*b     < 16 ? eff0 : eff1) * pxa_tab_lm32[c0];
            o[2*b + 1] = (2*b + 1 < 16 ? eff0 : eff1) * pxa_tab_lm32[c1];
        }
    }
}

void pxa_pxq_dequant_row(enum ggml_type type, const void * data, int64_t row, int64_t k, float * dst) {
    pxa_pxq_ensure_tables();
    PXA_PXQ_ASSERT(k % 32 == 0);
    const uint8_t * base = (const uint8_t *)data;
    switch (type) {
        case GGML_TYPE_PXQ4:   pxa_deq_row_pxq6 (base, row, k, dst, false); break;
        case GGML_TYPE_PXQ4HQ: pxa_deq_row_pxq6 (base, row, k, dst, true);  break;
        case GGML_TYPE_PXQ2:   pxa_deq_row_pxq2 (base, row, k, dst); break;
        case GGML_TYPE_PXQ3:   pxa_deq_row_pxq3 (base, row, k, dst); break;
        case GGML_TYPE_PXQ1:   pxa_deq_row_pxq1 (base, row, k, dst); break;
        case GGML_TYPE_PXQ6:   pxa_deq_row_pxq6r(base, row, k, dst); break;
        default: PXA_PXQ_ASSERT(!"pxa_pxq_dequant_row: not a PXQ type");
    }
}

void pxa_pxq_dequant_2d(enum ggml_type type, const void * data, float * dst, int64_t nrows, int64_t k) {
    // the quantizers only produce %64-row / %32-col tensors (pxq*_tensor_eligible in
    // llama-quantize.cpp); the CUDA dequant kernels abort on the same condition — mirror it.
    PXA_PXQ_ASSERT(nrows % 64 == 0 && k % 32 == 0);
    for (int64_t r = 0; r < nrows; ++r) {
        pxa_pxq_dequant_row(type, data, r, k, dst + r*k);
    }
}

// ---------------------------------------------------------------------------------------------
// fused up/gate + matmul fallbacks
// ---------------------------------------------------------------------------------------------

// per-thread scratch, grown on demand and cached for the lifetime of the thread (same
// strategy as iqk_mul_mat.cpp's thread_local_work_buffer(); compute threads are pooled,
// so this allocates once per thread and never churns; it is deliberately never freed).
static PXA_THREAD_LOCAL float * pxa_tls_buf  = NULL;
static PXA_THREAD_LOCAL size_t  pxa_tls_size = 0;

static float * pxa_scratch(size_t nfloats) {
    if (nfloats > pxa_tls_size) {
        float * p = (float *)realloc(pxa_tls_buf, nfloats*sizeof(float));
        PXA_PXQ_ASSERT(p != NULL);
        pxa_tls_buf  = p;
        pxa_tls_size = nfloats;
    }
    return pxa_tls_buf;
}

// scalar activations — mirror iqk_mul_mat.cpp MulMat::{gelu,relu,silu,swiglu_oai} exactly
// (tanh-approx GELU with the same constants; swiglu_oai alpha 1.702, hard limit 7)
static inline float pxa_activate(int op, float x) {
    switch (op) {
        case GGML_UNARY_OP_RELU: return x > 0.0f ? x : 0.0f;
        case GGML_UNARY_OP_SILU: return x/(1.0f + expf(-x));
        case GGML_UNARY_OP_GELU: {
            const float GELU_COEF_A    = 0.044715f;
            const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
            return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
        }
        case GGML_UNARY_OP_SWIGLU_OAI: {
            const float xi = x < 7.0f ? x : 7.0f;                 // k_swiglu_oai_limit
            return xi/(1.0f + expf(-xi*1.702f));                  // k_swiglu_oai_alpha
        }
        default:
            PXA_PXQ_ASSERT(!"pxa_activate: unsupported unary op for the PXQ CPU fallback");
            return 0.0f;
    }
}

static inline const float * pxa_x_row(const char * src1f, size_t nb11, size_t nb12,
                                      const struct pxa_pxq_rowmap * rows, int ne11, int64_t iy) {
    if (!rows) return (const float *)(src1f + (size_t)iy*nb11);
    const int i11 = rows[iy].i1 % ne11;
    const int i12 = rows[iy].i2;
    return (const float *)(src1f + (size_t)i12*nb12 + (size_t)i11*nb11);
}

static inline float * pxa_dst_row(char * dst, size_t nb1, size_t nb2,
                                  const struct pxa_pxq_rowmap * rows, int64_t iy) {
    if (!rows) return (float *)(dst + (size_t)iy*nb1);
    return (float *)(dst + (size_t)rows[iy].i1*nb1 + (size_t)rows[iy].i2*nb2);
}

static inline double pxa_dot(const float * w, const float * x, int64_t k) {
    double acc = 0.0;
    for (int64_t j = 0; j < k; ++j) acc += (double)w[j]*(double)x[j];
    return acc;
}

// ---------------------------------------------------------------------------------------------
// phase 2 fast path (pxq-dot.c): keep the weights in their nibble codes and take the dot in the
// integer domain against int8 activations, instead of materialising the row as f32 and running
// a double-precision f32 dot.
//
// The loop order flips as a result. The dequant path had to hoist the row decode out of the
// activation loop (4*k bytes of f32 per row, far too expensive to redo per token), which forced
// ix-outer/iy-inner with a k-float buffer in between. The integer path re-reads the row's CODES
// per (ix, iy) instead -- k/2 bytes, an eighth of the f32 row, and resident in L1 across the
// activation tile -- so the only thing that must be hoisted is the ACTIVATION quantisation.
//
// Activations are therefore quantised once per tile of PXA_PXQ_NYT rows and reused across every
// weight row this thread owns. The tile keeps the scratch bounded (16 * k/32 * 36 B = 69 KiB at
// k = 4096, an L2 resident) at the cost of re-walking the thread's weight rows once per tile;
// at decode ny == 1 and there is exactly one tile.
// ---------------------------------------------------------------------------------------------
#define PXA_PXQ_NYT 16   // activation rows quantised per pass

// scratch for one activation tile, as pxa_pxq_q8 blocks
static struct pxa_pxq_q8 * pxa_q8_scratch(int64_t kb_n) {
    static_assert(sizeof(struct pxa_pxq_q8) % sizeof(float) == 0,
                  "pxa_pxq_q8 must be a whole number of floats to ride pxa_scratch");
    return (struct pxa_pxq_q8 *)pxa_scratch((size_t)PXA_PXQ_NYT*(size_t)kb_n*
                                            (sizeof(struct pxa_pxq_q8)/sizeof(float)));
}

void pxa_pxq_moe_up_gate_cpu(
        enum ggml_type type_up,   const void * up,
        enum ggml_type type_gate, const void * gate,
        int64_t nr0, int64_t k,
        const float * up_bias, const float * gate_bias,
        const char * src1f, size_t nb11, size_t nb12,
        char * dst, size_t nb1, size_t nb2,
        const struct pxa_pxq_rowmap * rows, int ne11, int64_t ny,
        int unary_op, float limit,
        int ith, int nth) {

    PXA_PXQ_ASSERT(pxa_pxq_is_cpu_supported(type_up) && pxa_pxq_is_cpu_supported(type_gate));
    PXA_PXQ_ASSERT(k % 32 == 0);

    const int64_t chunk = (nr0 + nth - 1)/nth;
    const int64_t first = (int64_t)ith*chunk;
    const int64_t last  = first + chunk < nr0 ? first + chunk : nr0;
    if (first >= last) return;

    const bool oai = unary_op == GGML_UNARY_OP_SWIGLU_OAI;
    const bool has_limit = limit > 1e-6f;

    // fast path: both operands must be a tier pxq-dot.c covers. A mixed PXQ-UNIVERSAL pair with
    // only one 4-bit side (e.g. gate=pxq2, up=pxq4) falls through to the dequant path below
    // rather than running two different codecs in one loop.
    if (pxa_pxq_dot_supported(type_up) && pxa_pxq_dot_supported(type_gate)) {
        const int64_t kb_n = k/32;
        struct pxa_pxq_q8 * q8 = pxa_q8_scratch(kb_n);
        for (int64_t iy0 = 0; iy0 < ny; iy0 += PXA_PXQ_NYT) {
            const int64_t nyt = ny - iy0 < PXA_PXQ_NYT ? ny - iy0 : PXA_PXQ_NYT;
            for (int64_t t = 0; t < nyt; ++t) {
                pxa_pxq_quantize_row_q8(pxa_x_row(src1f, nb11, nb12, rows, ne11, iy0 + t),
                                        q8 + t*kb_n, k);
            }
            for (int64_t ix = first; ix < last; ++ix) {
                const float ub = up_bias   ? up_bias[ix]   : 0.0f;
                const float gb = gate_bias ? gate_bias[ix] : 0.0f;
                for (int64_t t = 0; t < nyt; ++t) {
                    const struct pxa_pxq_q8 * xq = q8 + t*kb_n;
                    float gv  = pxa_pxq_dot_q8(type_gate, gate, ix, k, xq) + gb;
                    float act = pxa_activate(unary_op, gv);
                    if (has_limit && act > limit) act = limit;
                    float uv = pxa_pxq_dot_q8(type_up, up, ix, k, xq) + ub;
                    if (oai) {
                        uv = 1.0f + (uv > 7.0f ? 7.0f : (uv < -7.0f ? -7.0f : uv));   // clamp_oai
                    } else if (has_limit) {
                        uv = uv > limit ? limit : (uv < -limit ? -limit : uv);
                    }
                    pxa_dst_row(dst, nb1, nb2, rows, iy0 + t)[ix] = uv*act;
                }
            }
        }
        return;
    }

    float * u = pxa_scratch(2*(size_t)k);
    float * g = u + k;

    for (int64_t ix = first; ix < last; ++ix) {
        pxa_pxq_dequant_row(type_up,   up,   ix, k, u);
        pxa_pxq_dequant_row(type_gate, gate, ix, k, g);
        const float ub = up_bias   ? up_bias[ix]   : 0.0f;
        const float gb = gate_bias ? gate_bias[ix] : 0.0f;
        for (int64_t iy = 0; iy < ny; ++iy) {
            const float * x = pxa_x_row(src1f, nb11, nb12, rows, ne11, iy);
            float gv = (float)pxa_dot(g, x, k) + gb;
            float act = pxa_activate(unary_op, gv);
            if (has_limit && act > limit) act = limit;
            float uv = (float)pxa_dot(u, x, k) + ub;
            if (oai) {
                uv = 1.0f + (uv > 7.0f ? 7.0f : (uv < -7.0f ? -7.0f : uv));   // clamp_oai
            } else if (has_limit) {
                uv = uv > limit ? limit : (uv < -limit ? -limit : uv);
            }
            pxa_dst_row(dst, nb1, nb2, rows, iy)[ix] = uv*act;
        }
    }
}

void pxa_pxq_mul_mat_cpu(
        enum ggml_type type, const void * a,
        int64_t nr0, int64_t k,
        const char * src1f, size_t nb11, size_t nb12,
        char * dst, size_t nb1, size_t nb2,
        const struct pxa_pxq_rowmap * rows, int ne11, int64_t ny,
        int ith, int nth) {

    PXA_PXQ_ASSERT(pxa_pxq_is_cpu_supported(type));
    PXA_PXQ_ASSERT(k % 32 == 0);

    const int64_t chunk = (nr0 + nth - 1)/nth;
    const int64_t first = (int64_t)ith*chunk;
    const int64_t last  = first + chunk < nr0 ? first + chunk : nr0;
    if (first >= last) return;

    if (pxa_pxq_dot_supported(type)) {
        const int64_t kb_n = k/32;
        struct pxa_pxq_q8 * q8 = pxa_q8_scratch(kb_n);
        for (int64_t iy0 = 0; iy0 < ny; iy0 += PXA_PXQ_NYT) {
            const int64_t nyt = ny - iy0 < PXA_PXQ_NYT ? ny - iy0 : PXA_PXQ_NYT;
            for (int64_t t = 0; t < nyt; ++t) {
                pxa_pxq_quantize_row_q8(pxa_x_row(src1f, nb11, nb12, rows, ne11, iy0 + t),
                                        q8 + t*kb_n, k);
            }
            for (int64_t ix = first; ix < last; ++ix) {
                for (int64_t t = 0; t < nyt; ++t) {
                    pxa_dst_row(dst, nb1, nb2, rows, iy0 + t)[ix] =
                        pxa_pxq_dot_q8(type, a, ix, k, q8 + t*kb_n);
                }
            }
        }
        return;
    }

    float * w = pxa_scratch((size_t)k);

    for (int64_t ix = first; ix < last; ++ix) {
        pxa_pxq_dequant_row(type, a, ix, k, w);
        for (int64_t iy = 0; iy < ny; ++iy) {
            const float * x = pxa_x_row(src1f, nb11, nb12, rows, ne11, iy);
            pxa_dst_row(dst, nb1, nb2, rows, iy)[ix] = (float)pxa_dot(w, x, k);
        }
    }
}

// ---- MXFP4 x q8_0 --------------------------------------------------------------------
// Semantics from dequantize_row_mxfp4: d = GGML_E8M0_TO_FP32_HALF(e); low nibble -> first
// half of the block, high nibble -> second half, through the 16-entry kvalues_mxfp4 table.
void pxa_mxfp4_dot_q8_0(int n, float * s, const void * vx, const void * vy) {
    const block_mxfp4 * restrict x = (const block_mxfp4 *)vx;
    const block_q8_0  * restrict y = (const block_q8_0  *)vy;
    const int nb = n / QK_MXFP4;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        const float d = GGML_E8M0_TO_FP32_HALF(x[ib].e) * GGML_FP16_TO_FP32(y[ib].d);
        int suml = 0;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            suml += y[ib].qs[j]              * kvalues_mxfp4[x[ib].qs[j] & 0xf]
                  + y[ib].qs[j + QK_MXFP4/2] * kvalues_mxfp4[x[ib].qs[j] >>  4];
        }
        sumf += d * (float)suml;
    }
    *s = sumf;
}

// ---- Q6_0 x q8_0 ---------------------------------------------------------------------
// Semantics from dequantize_row_q6_0: element j of the first half takes the low nibble of
// qs[j] plus bits 4-5 from qh[j % (QK6_0/4)]; element j of the second half takes the high
// nibble plus bits 4-5 of the same qh byte shifted two more. Both are biased by -32.
void pxa_q6_0_dot_q8_0(int n, float * s, const void * vx, const void * vy) {
    const block_q6_0 * restrict x = (const block_q6_0 *)vx;
    const block_q8_0 * restrict y = (const block_q8_0 *)vy;
    const int nb = n / QK6_0;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        int suml = 0;
        for (int j = 0; j < QK6_0/2; ++j) {
            const uint8_t h  = x[ib].qh[j % (QK6_0/4)] >> (4*(j/(QK6_0/4)));
            const int     x0 = ((x[ib].qs[j] & 0x0F) | ((h << 4) & 0x30)) - 32;
            const int     x1 = ((x[ib].qs[j] >>   4) | ((h << 2) & 0x30)) - 32;
            suml += x0 * y[ib].qs[j] + x1 * y[ib].qs[j + QK6_0/2];
        }
        sumf += (float)suml * GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d);
    }
    *s = sumf;
}

// ---- Q8_KV x Q8_KV -------------------------------------------------------------------
// Both operands are the same type: GGML_TYPE_Q8_KV's .vec_dot_type is GGML_TYPE_Q8_KV, so
// this dot feeds itself -- there is no repack and no interleave anywhere on the path.
//
// Q8_KV has NO block struct in ggml-common.h; a row is a flat header plus payload, and the
// layout is fixed by the three functions that read and write it:
//   iqk_quantize_row_q8_KV (iqk_quantize.cpp:4110) writes
//       dptr = (float *)vy;  q8 = (int8_t *)(dptr + 2);
//       dptr[0] = amax/127;  ((int32_t *)(dptr + 1))[0] = sum of the quants;  q8[i] = ...
//   dequantize_row_q8_KV (iqk_quantize.cpp:8449) reads
//       d = dptr[0];  q8 = (const int8_t *)(dptr + 2);  y[j] = d * q8[j];
//   the traits entry (ggml.c, GGML_TYPE_Q8_KV) agrees: blck_size 32, type_size 32,
//       row_meta_size 8 -- exactly the two leading 4-byte header words.
// So: [float d][int32 sum of quants][int8 qs[n]], one header per ROW, not per block.
//
// The int32 word is the row sum of the quants. It is there for kernels that pair this row
// with an asymmetric operand; Q8_KV is symmetric (a scale, no zero point and no min), so a
// Q8_KV x Q8_KV dot does not need it and must not subtract anything.
void pxa_q8_KV_dot_q8_KV(int n, float * s, const void * vx, const void * vy) {
    const float  * dx = (const float  *)vx;
    const float  * dy = (const float  *)vy;
    const int8_t * qx = (const int8_t *)(dx + 2);
    const int8_t * qy = (const int8_t *)(dy + 2);
    int64_t isum = 0;
    for (int j = 0; j < n; ++j) {
        isum += (int)qx[j] * (int)qy[j];
    }
    *s = dx[0] * dy[0] * (float)isum;
}

// =================================================================================================
// Graph ops and helpers reowned from ggml/src/iqk/iqk_cpu_ops.cpp (PXA 2026-08-21,
// ik separation phase 2).
//
// Everything below used to live in that file. None of it was ever an "accelerator" with a
// stock ggml path underneath: MUL_MULTI_ADD, HADAMARD and FUSED_RMS_RMS_ADD are fork ops
// whose ONLY implementation was there (grep for ggml_compute_forward_mul_multi_add /
// _hadamard / _fused_rms_rms_add returns nothing), SUM_ROWS+DIV was a fusion that fires on
// every MoE forward, and the two non-op helpers are called from llama.cpp and
// llama-sampling.cpp. Deleting the file without moving these is a link failure plus three
// dead fork ops, so they are moved, renamed into our namespace, and cited.
//
// The arithmetic is deliberately unchanged. Where ik had a SIMD loop with a scalar tail
// beneath it, both are carried over verbatim so the AVX2 build keeps producing exactly the
// bits it produced before. That is not an aspiration: a differential harness ran the old
// bodies and these side by side, same process, same flags, at nth=1 and nth=4, and required
// memcmp equality over the whole destination buffer. Every function below came out
// BIT-IDENTICAL on all three CPU trees (AVX2+FMA, AVX+F16C, pure scalar). So a divergence
// found by the CPU gauntlet after this commit is a porting mistake, never an intended
// numerics change.
//
// The one SIMD block deliberately NOT carried over is iqk_exp_with_thresh's AVX2 arm, which
// needed ik's v_expf and hsum_float_8 headers; it is a once-per-token sampler helper over
// n_vocab, its `#else` branch was already the reference, and dropping it is what makes this
// file free of ggml/src/iqk. That one agrees to 6.3e-07 relative (libm expf vs ik's
// polynomial), not to the bit, and is exact on any build without AVX2.
//
// Threading contract, inherited and preserved everywhere below: each compute thread calls
// with its own (ith, nth) and owns rows [ith*npt, MIN(first+npt, nrows)). Getting that
// arithmetic wrong drops or double-writes rows and shows up only at nth > 1, never in the
// single-threaded eval-callback dumps.
// =================================================================================================

#if defined(__AVX2__)
// Bit-for-bit ik's hsum_float_4/hsum_float_8 (iqk_common.h:230-237). The reduction ORDER of a
// horizontal float sum is observable in the last ulp, and this one feeds an RMS-norm rsqrt,
// so it is copied rather than re-derived.
static inline float pxa_hsum_float_4(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
static inline float pxa_hsum_float_8(__m256 x) {
    return pxa_hsum_float_4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)));
}
#endif

// -------------------------------------------------------------------------------------------------
// pxa_has_fancy_simd — was iqk_has_fancy_simd (iqk_cpu_ops.cpp:37-43).
//
// Not a ggml op; its only caller is src/llama.cpp, which logs whether the host has the
// AVX512 feature set ik called "fancy SIMD". ik got the answer from HAVE_FANCY_SIMD in
// iqk/iqk_config.h:46; the five feature macros are written out inline here instead so that
// nothing outside ggml/src/iqk is needed to answer the question.
// -------------------------------------------------------------------------------------------------
bool pxa_has_fancy_simd(void) {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && \
    defined(__AVX512BW__) && defined(__AVX512DQ__)
    return true;
#else
    return false;
#endif
}

// -------------------------------------------------------------------------------------------------
// pxa_sumrows_div — was iqk_sumrows_div (iqk_cpu_ops.cpp:156-176).
//
// The body of the SUM_ROWS+DIV fusion in ggml.c: given a DIV node whose numerator is the
// SUM_ROWS source and whose denominator is the SUM_ROWS result, compute the row-normalised
// values in one pass instead of materialising the sums. This is the MoE router softmax
// normalisation and it runs 40x per forward pass on a 35B-class graph.
//
// ik indexed rows as ir*nb[1] over ggml_nrows() with no contiguity check, which is only
// correct for a contiguous src. It has always been called with one, and the AVX2 build makes
// the same assumption, so this is not a divergence source -- but an assumption that load-bearing
// should be stated, so the assert is added here rather than left implied.
// -------------------------------------------------------------------------------------------------
void pxa_sumrows_div(struct ggml_tensor * div, int ith, int nth) {
    const struct ggml_tensor * src = div->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(div->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(div));

    const int ne00  = (int) src->ne[0];
    const int nrows = (int) ggml_nrows(src);
    const int npt   = (nrows + nth - 1)/nth;
    const int first = ith*npt;
    const int last  = MIN(first + npt, nrows);
    if (last < first) return;

    for (int ir = first; ir < last; ++ir) {
        const float * values = (const float *)((const char *)src->data + ir*src->nb[1]);
        float sum = 0;
        for (int j = 0; j < ne00; ++j) sum += values[j];
        const float norm = sum > 0 ? 1/sum : 0.0f;
        float * result = (float *)((char *)div->data + ir*div->nb[1]);
        for (int j = 0; j < ne00; ++j) result[j] = values[j]*norm;
    }
}

// -------------------------------------------------------------------------------------------------
// pxa_mul_multi_add — was iqk_mul_multi_add (iqk_cpu_ops.cpp:442-513).
//
// GGML_OP_MUL_MULTI_ADD is a fork op with no stock ggml body: this function IS the op. It
// collapses "multiply each of ne01 expert outputs by its routing weight, then add them" into
// one pass, and fires once per MoE layer (the routed_out-N nodes).
//
// Two shapes. When src[2] (f32 per-expert scales) and src[3] (i32 per-row expert ids) are
// both present, each term is additionally scaled by scales[ids[j]]; otherwise the weight in
// src1 is the whole story. src1 is a column vector (ne[0] == 1) of weights, one per term.
// -------------------------------------------------------------------------------------------------
void pxa_mul_multi_add(struct ggml_tensor * dst, int ith, int nth) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] ==  dst->ne[0]);
    GGML_ASSERT(src0->ne[2] ==  dst->ne[1]);
    GGML_ASSERT(src0->ne[1] == src1->ne[1]);
    GGML_ASSERT(src0->ne[2] == src1->ne[2]);
    GGML_ASSERT(src0->ne[3] == src1->ne[3]);
    GGML_ASSERT(src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[0] == 1);

    const int nrows = (int) dst->ne[1];
    const int npt   = (nrows + nth - 1)/nth;
    const int first = ith*npt;
    const int last  = MIN(nrows, first + npt);

    const int ne01 = (int) src0->ne[1];
    const int ne00 = (int) src0->ne[0];

    const struct ggml_tensor * src2 = dst->src[2];
    const struct ggml_tensor * src3 = dst->src[3];
    if (src2 && src3) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src3->type == GGML_TYPE_I32);
        GGML_ASSERT(src3->ne[0] == src0->ne[1]);

        const char  * cids   = (const char  *)src3->data;
        const float * scales = (const float *)src2->data;
        for (int ir = first; ir < last; ++ir) {
            const char * c0 = (const char *)src0->data + ir*src0->nb[2];
            const char * c1 = (const char *)src1->data + ir*src1->nb[2];
            float * y = (float *)((char *)dst->data + ir*dst->nb[1]);
            const float * x0 = (const float *)c0;
            const float * x1 = (const float *)c1;
            const int   * ids = (const int *)(cids + ir*src3->nb[1]);
            float s = scales[ids[0]] * x1[0];
            for (int k = 0; k < ne00; ++k) y[k] = x0[k] * s;
            for (int j = 1; j < ne01; ++j) {
                c0 += src0->nb[1];
                c1 += src1->nb[1];
                x0 = (const float *)c0;
                x1 = (const float *)c1;
                s  = x1[0] * scales[ids[j]];
                for (int k = 0; k < ne00; ++k) y[k] += x0[k] * s;
            }
        }
        return;
    }

    for (int ir = first; ir < last; ++ir) {
        const char * c0 = (const char *)src0->data + ir*src0->nb[2];
        const char * c1 = (const char *)src1->data + ir*src1->nb[2];
        float * y = (float *)((char *)dst->data + ir*dst->nb[1]);
        const float * x0 = (const float *)c0;
        const float * x1 = (const float *)c1;
        for (int k = 0; k < ne00; ++k) y[k] = x0[k] * x1[0];
        for (int j = 1; j < ne01; ++j) {
            c0 += src0->nb[1];
            c1 += src1->nb[1];
            x0 = (const float *)c0;
            x1 = (const float *)c1;
            for (int k = 0; k < ne00; ++k) y[k] += x0[k] * x1[0];
        }
    }
}

// -------------------------------------------------------------------------------------------------
// pxa_hadamard — was iqk_hadamard (iqk_cpu_ops.cpp:534-581), with fast_ht from :516-531.
//
// GGML_OP_HADAMARD is a fork op with no stock ggml body: this function IS the op. It applies
// a fast Walsh-Hadamard transform of length nh (a power of two, from op_params[0]) to every
// nh-wide chunk of every row, normalising by 2^(-log2(nh)/2) as it goes.
//
// ik's popcount() came from iqk/iqk_common.h:960, which sat outside that header's
// IQK_IMPLEMENT guard specifically so files like this could reach it. That is exactly the
// kind of hidden thread this campaign is cutting, so the power-of-two check is done here
// with a three-line portable helper instead.
// -------------------------------------------------------------------------------------------------
static inline bool pxa_is_pow2_u32(uint32_t x) {
    return x != 0 && (x & (x - 1)) == 0;
}

// In-place fast Walsh-Hadamard transform, length n (power of two). ik had this as a template
// instantiated only for float; de-templated it is plain C.
static void pxa_fast_ht_f32(int n, float * values) {
    const float ksqrt2 = 0.707106781f;
    float scale = 1;
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                const float x = values[j], y = values[j + h];
                values[j+0] = x + y;
                values[j+h] = x - y;
            }
        }
        scale *= ksqrt2;
    }
    for (int i = 0; i < n; ++i) values[i] *= scale;
}

void pxa_hadamard(struct ggml_tensor * dst, int ith, int nth) {
    const struct ggml_tensor * src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    const int nh = dst->op_params[0];
    GGML_ASSERT(nh > 1 && pxa_is_pow2_u32((uint32_t) nh));
    GGML_ASSERT(dst->ne[0] % nh == 0);

    const int nc  = (int) (dst->ne[0]/nh);
    const int nr  = (int) (ggml_nrows(dst) * nc);
    const int npt = (nr + nth - 1)/nth;
    const int first = npt*ith;
    const int last  = MIN(first + npt, nr);

    // ir enumerates (i3, i2, i1, chunk) quadruples; decompose it back into the four indices.
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];

    if (src->type == GGML_TYPE_F32) {
        for (int ir = first; ir < last; ++ir) {
            const int i3 = (int) ( ir / (ne1 * ne2 * nc));
            const int i2 = (int) ((ir - i3*ne1*ne2*nc) / (ne1 * nc));
            const int i1 = (int) ((ir - i3*ne1*ne2*nc - i2*ne1*nc) / nc);
            const int ic = (int) ( ir - i3*ne1*ne2*nc - i2*ne1*nc - i1*nc);

            const float * x = (const float *)((const char *)src->data + i3*src->nb[3] + i2*src->nb[2] + i1*src->nb[1]) + (size_t)ic*nh;
            float       * y = (      float *)((      char *)dst->data + i3*dst->nb[3] + i2*dst->nb[2] + i1*dst->nb[1]) + (size_t)ic*nh;
            memcpy(y, x, nh*sizeof(float));
            pxa_fast_ht_f32(nh, y);
        }
        return;
    }

    // Quantized source: dequantise the chunk into the destination and transform in place.
    // ggml_internal_get_type_traits is public (ggml.h), so this needs nothing from ik.
    ggml_type_traits_t traits = ggml_internal_get_type_traits(src->type);
    GGML_ASSERT(traits.to_float != NULL);
    const size_t blck_size = traits.blck_size;
    const size_t type_size = traits.type_size;
    GGML_ASSERT(blck_size > 0 && (nh % blck_size == 0 || blck_size % nh == 0));

    for (int ir = first; ir < last; ++ir) {
        const int i3 = (int) ( ir / (ne1 * ne2 * nc));
        const int i2 = (int) ((ir - i3*ne1*ne2*nc) / (ne1 * nc));
        const int i1 = (int) ((ir - i3*ne1*ne2*nc - i2*ne1*nc) / nc);
        const int ic = (int) ( ir - i3*ne1*ne2*nc - i2*ne1*nc - i1*nc);

        const char * x_row  = (const char *)src->data + i3*src->nb[3] + i2*src->nb[2] + i1*src->nb[1];
        const size_t offset = ((size_t)ic * nh / blck_size) * type_size;
        float      * y      = (float *)((char *)dst->data + i3*dst->nb[3] + i2*dst->nb[2] + i1*dst->nb[1]) + (size_t)ic*nh;
        traits.to_float(x_row + offset, y, nh);
        pxa_fast_ht_f32(nh, y);
    }
}

// -------------------------------------------------------------------------------------------------
// pxa_exp_with_thresh — was iqk_exp_with_thresh (iqk_cpu_ops.cpp:584-627).
//
// Not a ggml op: this is llama-sampling's adaptive-p accumulator. It rewrites logits[] in
// place as exp(logit - max) for every logit at or above `min`, zeroing the rest, and returns
// the sum. Called once per sampled token over n_vocab.
//
// ik had an AVX2 arm here using v_expf (iqk/iqk_utils.h) and hsum_float_8 (iqk/iqk_common.h),
// with this loop as its `#else`. The scalar branch was already the reference and the AVX2 arm
// bought a single vectorised exp per token, so the arm is dropped and the two ik headers with
// it. The comparison is `>=`, matching what the AVX2 arm's _CMP_GE_OQ did (fixed under
// separate cover before this move); it is an ordered compare, so a NaN logit scores 0 rather
// than poisoning the sum.
// -------------------------------------------------------------------------------------------------
float pxa_exp_with_thresh(int n, float * logits, float max, float min) {
    float sum = 0;
    for (int j = 0; j < n; ++j) {
        const float p = logits[j] >= min ? expf(logits[j] - max) : 0;
        sum += p;
        logits[j] = p;
    }
    return sum;
}

// -------------------------------------------------------------------------------------------------
// pxa_rms_rms_add — was iqk_rms_rms_add (iqk_cpu_ops.cpp:926-988) with its six helpers
// (:860-924).
//
// GGML_OP_FUSED_RMS_RMS_ADD is a fork op with no stock ggml body: this function IS the op. It
// computes two independent RMS norms of the same row shape and adds them:
//
//     dst[j] = c1[j]*x1[j]/sqrt(mean(x1^2) + eps) + c2[j]*x2[j]/sqrt(mean(x2^2) + eps)
//
// x1/x2 (src0/src2) share a type, one of f32/f16/bf16; the gains c1/c2 (src1/src3) are always
// f32 single rows. Only the f32 flavour had AVX2 loops in ik and both had correct scalar
// tails, so the no-AVX2 path was already right; both are carried over unchanged so the AVX2
// build's last ulp does not move.
// -------------------------------------------------------------------------------------------------
static inline float pxa_sum_row_squared_f32(int ncols, const float * x) {
    float sum = 0;
    int i = 0;
#ifdef __AVX2__
    __m256 vsum = _mm256_setzero_ps();
    for (; i < ncols - 7; i += 8) {
        const __m256 vx = _mm256_loadu_ps(x + i);
        vsum = _mm256_fmadd_ps(vx, vx, vsum);
    }
    sum = pxa_hsum_float_8(vsum);
#endif
    for (; i < ncols; ++i) sum += x[i]*x[i];
    return sum;
}
static inline float pxa_sum_row_squared_f16(int ncols, const ggml_half * x) {
    float sum = 0;
    for (int j = 0; j < ncols; ++j) {
        const float v = GGML_FP16_TO_FP32(x[j]);
        sum += v*v;
    }
    return sum;
}
static inline float pxa_sum_row_squared_bf16(int ncols, const ggml_bf16_t * x) {
    float sum = 0;
    for (int j = 0; j < ncols; ++j) {
        const float v = GGML_BF16_TO_FP32(x[j]);
        sum += v*v;
    }
    return sum;
}
static inline void pxa_rms_rms_add_f32(int ncols, float scale1, float scale2,
        const float * x1, const float * x2, const float * c1, const float * c2, float * dst) {
    int j = 0;
#ifdef __AVX2__
    const __m256 vs1 = _mm256_set1_ps(scale1);
    const __m256 vs2 = _mm256_set1_ps(scale2);
    for (; j < ncols - 7; j += 8) {
        const __m256 vx1 = _mm256_loadu_ps(x1 + j);
        const __m256 vx2 = _mm256_loadu_ps(x2 + j);
        const __m256 vc1 = _mm256_loadu_ps(c1 + j);
        const __m256 vc2 = _mm256_loadu_ps(c2 + j);
        const __m256 vy = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vs1, vc1), vx1),
                                        _mm256_mul_ps(_mm256_mul_ps(vs2, vc2), vx2));
        _mm256_storeu_ps(dst + j, vy);
    }
#endif
    for (; j < ncols; ++j) {
        dst[j] = scale1 * c1[j] * x1[j] + scale2 * c2[j] * x2[j];
    }
}
static inline void pxa_rms_rms_add_f16(int ncols, float scale1, float scale2,
        const ggml_half * x1, const ggml_half * x2, const float * c1, const float * c2, float * dst) {
    for (int j = 0; j < ncols; ++j) {
        const float v1 = GGML_FP16_TO_FP32(x1[j]);
        const float v2 = GGML_FP16_TO_FP32(x2[j]);
        dst[j] = scale1 * c1[j] * v1 + scale2 * c2[j] * v2;
    }
}
static inline void pxa_rms_rms_add_bf16(int ncols, float scale1, float scale2,
        const ggml_bf16_t * x1, const ggml_bf16_t * x2, const float * c1, const float * c2, float * dst) {
    for (int j = 0; j < ncols; ++j) {
        const float v1 = GGML_BF16_TO_FP32(x1[j]);
        const float v2 = GGML_BF16_TO_FP32(x2[j]);
        dst[j] = scale1 * c1[j] * v1 + scale2 * c2[j] * v2;
    }
}

void pxa_rms_rms_add(struct ggml_tensor * dst, int ith, int nth) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * src2 = dst->src[2];
    const struct ggml_tensor * src3 = dst->src[3];

    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src2) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src2, dst));
    GGML_ASSERT(ggml_nrows(src1) == 1 && ggml_nrows(src3) == 1);
    GGML_ASSERT(src0->ne[0] == src1->ne[0] && src2->ne[0] == src3->ne[0]);
    GGML_ASSERT(src0->type == src2->type);
    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F32);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps > 0.0f);

    const int nrows = (int) ggml_nrows(dst);
    const int nrows_per_thread = (nrows + nth - 1)/nth;
    const int first = ith*nrows_per_thread;
    const int last  = MIN(nrows, first + nrows_per_thread);

    const float * c1 = (const float *) src1->data;
    const float * c2 = (const float *) src3->data;

    const int ncols = (int) dst->ne[0];

    for (int ir = first; ir < last; ++ir) {
        float * y = (float *)dst->data + (size_t)ir*ncols;

        float sum1 = 0, sum2 = 0;
        if (src0->type == GGML_TYPE_F32) {
            sum1 = pxa_sum_row_squared_f32(ncols, (const float *)src0->data + (size_t)ir*ncols);
            sum2 = pxa_sum_row_squared_f32(ncols, (const float *)src2->data + (size_t)ir*ncols);
        } else if (src0->type == GGML_TYPE_F16) {
            sum1 = pxa_sum_row_squared_f16(ncols, (const ggml_half *)src0->data + (size_t)ir*ncols);
            sum2 = pxa_sum_row_squared_f16(ncols, (const ggml_half *)src2->data + (size_t)ir*ncols);
        } else {
            sum1 = pxa_sum_row_squared_bf16(ncols, (const ggml_bf16_t *)src0->data + (size_t)ir*ncols);
            sum2 = pxa_sum_row_squared_bf16(ncols, (const ggml_bf16_t *)src2->data + (size_t)ir*ncols);
        }

        const float mean1  = sum1/ncols;
        const float mean2  = sum2/ncols;
        const float scale1 = 1.0f/sqrtf(mean1 + eps);
        const float scale2 = 1.0f/sqrtf(mean2 + eps);
        if (src0->type == GGML_TYPE_F32) {
            pxa_rms_rms_add_f32(ncols, scale1, scale2,
                    (const float *)src0->data + (size_t)ir*ncols, (const float *)src2->data + (size_t)ir*ncols, c1, c2, y);
        } else if (src0->type == GGML_TYPE_F16) {
            pxa_rms_rms_add_f16(ncols, scale1, scale2,
                    (const ggml_half *)src0->data + (size_t)ir*ncols, (const ggml_half *)src2->data + (size_t)ir*ncols, c1, c2, y);
        } else {
            pxa_rms_rms_add_bf16(ncols, scale1, scale2,
                    (const ggml_bf16_t *)src0->data + (size_t)ir*ncols, (const ggml_bf16_t *)src2->data + (size_t)ir*ncols, c1, c2, y);
        }
    }
}
