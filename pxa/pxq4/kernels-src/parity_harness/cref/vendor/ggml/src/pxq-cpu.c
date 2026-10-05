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
//   (ids 250 + 251, the retired MXFP4-repack and PXQ5 legacy types, were removed 2026-07-21.
//    id 256, the 5-bit PXQ6 tier, has NO CPU fallback yet — pxa_pxq_is_cpu_supported returns
//    false for it.)
//
//   panel  = hdr (64 x fp16 row anchors when hdr==128) + (k/32) slabs; panels row-major.
//   slab   = 64-row scale SoA + 64 code rows.
//   16 B code rows (PXQ4/PXQ4HQ): byte b = code(elem 2b) | code(elem 2b+1) << 4.
//   8 B code rows (PXQ2): 2 bits/elem, elem j at bits 2*(j&3) of byte j>>2 (LE words).
//   12 B code rows (PXQ3): bit-plane, three LE u32 words: w0 = low 2 bits of elems 0-15,
//     w1 = low 2 bits of elems 16-31, w2 = bit2 plane (bit j = elem j, j = 0..31).
//   dequant (E16-row family contract, parity-locked):
//     eff = fp32(anchor_fp16) * SUB[s4];  w = eff * fp32(book[c])
//
// Table env overrides (PXA_PXQ6_BOOK/..., same names + fp16-snap as the quantizers and the
// CUDA kernels) are honored so a custom-table model keeps working on the CPU path too.

#include "pxq-cpu.h"

#include "ggml-impl.h"   // GGML_COMPUTE_FP16_TO_FP32 / GGML_COMPUTE_FP32_TO_FP16 (self-contained)

#include "ggml-pxq6-tables.h"
#include "ggml-pxq2-tables.h"
#include "ggml-pxq3-tables.h"

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

static bool pxa_parse_n(const char * e, float * out, int want) {
    int n = 0;
    float v[16];
    char buf[512];
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
    float t[16];
    if ((e = getenv("PXA_PXQ6_BOOK"))   && pxa_parse_n(e, t, 16)) memcpy(pxa_tab_px16_book, t, sizeof(pxa_tab_px16_book));
    if ((e = getenv("PXA_PXQ6_SUB"))    && pxa_parse_n(e, t, 16)) memcpy(pxa_tab_sub16,     t, sizeof(pxa_tab_sub16));
    if ((e = getenv("PXA_PXQ6_SUB_HQ")) && pxa_parse_n(e, t, 16)) memcpy(pxa_tab_sub8,      t, sizeof(pxa_tab_sub8));
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

// ---------------------------------------------------------------------------------------------
// per-type row dequant (row = global row index; data = 2D slice base)
// ---------------------------------------------------------------------------------------------

bool pxa_pxq_is_cpu_supported(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_PXQ4:
        case GGML_TYPE_PXQ4HQ:
        case GGML_TYPE_PXQ2:
        case GGML_TYPE_PXQ3:
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

void pxa_pxq_dequant_row(enum ggml_type type, const void * data, int64_t row, int64_t k, float * dst) {
    pxa_pxq_ensure_tables();
    PXA_PXQ_ASSERT(k % 32 == 0);
    const uint8_t * base = (const uint8_t *)data;
    switch (type) {
        case GGML_TYPE_PXQ4:   pxa_deq_row_pxq6 (base, row, k, dst, false); break;
        case GGML_TYPE_PXQ4HQ: pxa_deq_row_pxq6 (base, row, k, dst, true);  break;
        case GGML_TYPE_PXQ2:   pxa_deq_row_pxq2 (base, row, k, dst); break;
        case GGML_TYPE_PXQ3:   pxa_deq_row_pxq3 (base, row, k, dst); break;
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

    float * u = pxa_scratch(2*(size_t)k);
    float * g = u + k;

    const bool oai = unary_op == GGML_UNARY_OP_SWIGLU_OAI;
    const bool has_limit = limit > 1e-6f;

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

    float * w = pxa_scratch((size_t)k);

    for (int64_t ix = first; ix < last; ++ix) {
        pxa_pxq_dequant_row(type, a, ix, k, w);
        for (int64_t iy = 0; iy < ny; ++iy) {
            const float * x = pxa_x_row(src1f, nb11, nb12, rows, ne11, iy);
            pxa_dst_row(dst, nb1, nb2, rows, iy)[ix] = (float)pxa_dot(w, x, k);
        }
    }
}
