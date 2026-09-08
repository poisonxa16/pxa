// pxq23.cuh — PXQ2 (2-bit LM4) + PXQ3 (3-bit LM8, bit-plane) CUDA side: device tables, env
// gates, format policies, and the convert.cu dequant wrappers.
//
// Spec: PXQ-UNIVERSAL design notes, 2026-07-17 (internal lab). One COMBINED file (not pxq2.cuh+pxq3.cuh):
// the two types differ ONLY in book size and pair-decode (~40 lines each); every kernel they
// run on lives in pxq6.cuh's policy-templated family (k_pxq6_*), exactly how PXQ6HQ already
// rides PXQ6's kernels — a per-type file would be 90% boilerplate. This mirrors how pxq6.cuh
// hosts both PXQ6 and PXQ6HQ.
//
// INCLUSION POINT (load-bearing): #include'd from pxq6.cuh immediately BEFORE the
// "host-side dispatch" section (the PXA_PXQ_FMT_* defines), i.e. AFTER the generalized
// pxq6_dot32 / k_pxq6_dequant_matrix / pxq6_deq_slab_cm templates and BEFORE the
// PXQ6_PICK_FMT pickers, so that:
//   - the policies below can instantiate k_pxq6_dequant_matrix (wrappers at the bottom), and
//   - the picker switch arms in pxq6.cuh can name pxq6_pol_p2 / pxq6_pol_p3.
// It REQUIRES the generalized POL interface from the PXQ-UNIVERSAL apply plan
// (CODE_WORDS / CODE_BYTES / pair() / pairl() + pxq6_ldcodes) — see APPLY-PLAN.md §3.
//
// FORMATS (identical E16-row scale machinery to PXQ6 core — 128 B fp16 row-anchor panel
// header, frozen SUB16 4-bit sub-scale per 16-elem block, 64 B scale SoA per slab):
//   PXQ2: codes 2 bit, 4/byte -> 8 B/row/slab, SLAB 576.  code row = 2 LE u32 words,
//         word h = elems 16h..16h+15, elem j at bits 2*(j&15).
//   PXQ3: codes 3 bit, BIT-PLANE -> 12 B/row/slab, SLAB 832. code row = 3 LE u32 words:
//         w0/w1 = low planes (2b/elem) of elems 0-15 / 16-31, w2 = high plane (1b/elem,
//         bit j = elem j). code(j) = ((lo>>2*(j&15))&3) | (((w2>>j)&1)<<2). Branch-free.
//   PAIRLUT (K2) is 4-bit-only: both policies decode via pair(); the host pickers force
//   PAIR=false for these formats (pairl() below is a never-executed compile stub).
//   EXCEPTION (2026-08-04): PXQ3 has its OWN pair-LUT mode — PAIRL3 (env PXA_PXQ3_PAIRLUT,
//   default OFF), a 64-entry LUT keyed on the pair's packed bit-planes; see pairl3() below.
//   K2a/PAIRLUT itself stays 4-bit-only.
//   K6 WMMA is NOT extended to PXQ2/PXQ3 in v1 — the host driver masks wmma_mode for
//   fmt >= PXA_PXQ_FMT_P2 (see APPLY-PLAN §5). The K1/K1b/K2-VECX/K3/K4/K5 family
//   instantiates unmodified through the policies.
//
// Numeric tables: ggml-pxq2-tables.h / ggml-pxq3-tables.h (frozen, sha256-locked).
// PXA_PXQ2_BOOK / PXA_PXQ3_BOOK override the books (fp16-snapped; must match the file's
// pxa.pxq2.book / pxa.pxq3.book provenance KVs). The sub-scale LUT is pxq6_sub16_g —
// SHARED with PXQ6, so PXA_PXQ6_SUB overrides it for all three code widths at once.
#pragma once

#include "../../../include/ggml-pxq2-tables.h"
#include "../../../include/ggml-pxq3-tables.h"
#include "../../../include/ggml-pxq1-tables.h"

// per-TU device tables (ggml-cuda.cu and convert.cu each get a copy; env overrides are
// uploaded per-TU + per-device by pxq23_maybe_upload_books()).
static __device__ float pxq2_book_g[4] = PXQ2_BOOK_INIT;
static __device__ float pxq3_book_g[8] = PXQ3_BOOK_INIT;
static __device__ float pxq1_book_g[2] = PXQ1_BOOK_INIT;

// ---------------------------------------------------------------------------------------------
// master gates + host self-check (table integrity: fp16-snapped, strictly ascending,
// sign-straddling, |v| < 1). NOTE: unlike PX16 there is NO zero entry and absmax != 1 —
// those PXQ6 invariants deliberately do NOT apply to the Lloyd-fit LM4/LM8 books.
// ---------------------------------------------------------------------------------------------
static inline bool pxa_pxq23_book_ok(const float * b, int n) {
    bool ok = b[0] < 0.0f && b[n-1] > 0.0f;
    for (int i = 0; i < n - 1 && ok; ++i) ok = b[i] < b[i+1];
    for (int i = 0; i < n && ok; ++i) {
        ok = fabsf(b[i]) < 1.0f && __half2float(__float2half_rn(b[i])) == b[i];   // fp16-snap idempotence
    }
    return ok;
}

// PXQ1's book is the SIGN book {-1,+1}: the scale (anchor x SUB16) carries all magnitude, so the
// LS-optimal 1-bit levels are exactly +/-1. It therefore CANNOT use pxa_pxq23_book_ok(), which
// requires |v| < 1 (right for the Lloyd-fit LM4/LM8 books, wrong here) -- reusing it would fail
// the self-check and silently disable the fused kernels. Same shape of check, |v| <= 1.
static inline bool pxa_pxq1_book_ok(const float * b) {
    return b[0] < 0.0f && b[1] > 0.0f && b[0] < b[1]
        && fabsf(b[0]) <= 1.0f && fabsf(b[1]) <= 1.0f
        && __half2float(__float2half_rn(b[0])) == b[0]
        && __half2float(__float2half_rn(b[1])) == b[1];
}

static inline bool pxa_pxq1_enabled() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ1");
        bool v = !(e && atoi(e) == 0);
        if (v) {
            static const float book[2] = PXQ1_BOOK_INIT;
            if (!pxa_pxq1_book_ok(book)) {
                fprintf(stderr, "PXA_PXQ1: sign-book self-check FAILED — fused kernels DISABLED (fallback in use)\n");
                v = false;
            } else {
                fprintf(stderr, "PXA_PXQ1 fused kernels: ON (sign-book self-check PASS; PXA_PXQ1=0 disables)\n");
            }
        } else {
            fprintf(stderr, "PXA_PXQ1 fused kernels: OFF (dequant->cublas fallback)\n");
        }
        return v;
    }();
    return on;
}

static inline bool pxa_pxq2_enabled() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ2");
        bool v = !(e && atoi(e) == 0);
        if (v) {
            static const float book[4] = PXQ2_BOOK_INIT;
            if (!pxa_pxq23_book_ok(book, 4)) {
                fprintf(stderr, "PXA_PXQ2: LM4 table self-check FAILED — fused kernels DISABLED (fallback in use)\n");
                v = false;
            } else {
                fprintf(stderr, "PXA_PXQ2 fused kernels: ON (LM4 self-check PASS; PXA_PXQ2=0 disables)\n");
            }
        } else {
            fprintf(stderr, "PXA_PXQ2 fused kernels: OFF (dequant->cublas fallback)\n");
        }
        return v;
    }();
    return on;
}

static inline bool pxa_pxq3_enabled() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ3");
        bool v = !(e && atoi(e) == 0);
        if (v) {
            static const float book[8] = PXQ3_BOOK_INIT;
            if (!pxa_pxq23_book_ok(book, 8)) {
                fprintf(stderr, "PXA_PXQ3: LM8 table self-check FAILED — fused kernels DISABLED (fallback in use)\n");
                v = false;
            } else {
                fprintf(stderr, "PXA_PXQ3 fused kernels: ON (LM8 self-check PASS; PXA_PXQ3=0 disables)\n");
            }
        } else {
            fprintf(stderr, "PXA_PXQ3 fused kernels: OFF (dequant->cublas fallback)\n");
        }
        return v;
    }();
    return on;
}

// env book overrides (PXA_PXQ2_BOOK = 4 floats, PXA_PXQ3_BOOK = 8), fp16-snapped, per-device
// upload — same discipline as pxq6_maybe_upload_tables (which handles the SHARED sub LUT).
static inline void pxq23_maybe_upload_books(int device) {
    static bool parsed = false;
    static bool have_b2 = false, have_b3 = false;
    static float eb2[4], eb3[8];
    static bool uploaded[64] = {false};
    if (!parsed) {
        parsed = true;
        auto parse_n = [](const char * e, float * out, int want) -> bool {
            int n = 0; float v[16];
            char * dup = strdup(e);
            for (char * t = strtok(dup, ","); t && n < want; t = strtok(nullptr, ",")) v[n++] = strtof(t, nullptr);
            free(dup);
            if (n != want) return false;
            for (int i = 0; i < want; ++i) out[i] = __half2float(__float2half_rn(v[i]));
            return true;
        };
        if (const char * e = getenv("PXA_PXQ2_BOOK")) { have_b2 = parse_n(e, eb2, 4); fprintf(stderr, "PXA_PXQ2_BOOK: %s\n", have_b2 ? "custom book active" : "parse FAILED — ignored"); }
        if (const char * e = getenv("PXA_PXQ3_BOOK")) { have_b3 = parse_n(e, eb3, 8); fprintf(stderr, "PXA_PXQ3_BOOK: %s\n", have_b3 ? "custom book active" : "parse FAILED — ignored"); }
        // PXA_PXQ2_V3 (2026-08-10): decode-side arm of the PXQ2 v3 refit book (PXQ2 only).
        // Runs BEFORE the CEIL_V2 block so it wins for PXQ2 (CEIL_V2 only fills eb2 when
        // have_b2 is still false); an explicit PXA_PXQ2_BOOK override (above) already set
        // have_b2 and still wins. Default OFF: stock decode is byte-identical.
        if (const char * e = getenv("PXA_PXQ2_V3")) {
            if (atoi(e) != 0) {
                static const float v3b2[4] = PXQ2_BOOK_V3_INIT;
                if (!have_b2) { memcpy(eb2, v3b2, sizeof(eb2)); have_b2 = true; }
                fprintf(stderr, "PXA_PXQ2_V3 ARMED (CUDA decode): PXQ2 v3 refit book\n");
            }
        }
        // PXA_PXQ_CEIL_V2 (2026-08-09): decode-side arm of the PXQ2/PXQ3 ceiling fix -- upload
        // the v2 (max|book| == 1.0) LM4/LM8 books. An explicit PXA_PXQn_BOOK override (above)
        // still wins. Default OFF: stock decode is byte-identical (v1 tables untouched).
        if (const char * e = getenv("PXA_PXQ_CEIL_V2")) {
            if (atoi(e) != 0) {
                static const float v2b2[4] = PXQ2_BOOK_V2_INIT;
                static const float v2b3[8] = PXQ3_BOOK_V2_INIT;
                if (!have_b2) { memcpy(eb2, v2b2, sizeof(eb2)); have_b2 = true; }
                if (!have_b3) { memcpy(eb3, v2b3, sizeof(eb3)); have_b3 = true; }
                fprintf(stderr, "PXA_PXQ_CEIL_V2 ARMED (CUDA decode): PXQ2/PXQ3 v2 books\n");
            }
        }
    }
    if ((have_b2 || have_b3) && device >= 0 && device < 64 && !uploaded[device]) {
        uploaded[device] = true;
        int cur = 0; cudaGetDevice(&cur);
        if (cur != device) cudaSetDevice(device);
        if (have_b2) cudaMemcpyToSymbol(pxq2_book_g, eb2, sizeof(eb2));
        if (have_b3) cudaMemcpyToSymbol(pxq3_book_g, eb3, sizeof(eb3));
        if (cur != device) cudaSetDevice(cur);
    }
}

// ---------------------------------------------------------------------------------------------
// format policies (generalized POL interface — see APPLY-PLAN §3 for the pxq6.cuh side).
// NEFF = 2: two eff scales per 32-elem block (elems 0-15 / 16-31), identical to pxq6_pol_p6.
// eff-group split in the shared kernels is by ELEMENT-PAIR index b: eff[(b*NEFF)>>4].
// ---------------------------------------------------------------------------------------------
// PXQ1 (sub-2-bit tier): 1-bit sign codes, single LE u32 code word / 32-elem row-block.
// 2026-07-29: NO LONGER dequant-only. pxa_pxq_fmt(PXQ1) now returns PXA_PXQ_FMT_P1 and this
// policy is named by the decode/GEMM pickers exactly as pol_p2/pol_p3 are (TAB / TAB_CS modes
// only -- PRMT and PAIRLUT are 4-bit-nibble-addressed and are forced off for P1 by PICKM23).
struct pxq6_pol_p1 {
    static constexpr int SLAB = PXQ1_SLAB_BYTES, HDR = PXQ1_HDR_BYTES, CODE_OFF = 64, NEFF = 2;
    static constexpr int CODE_WORDS = 1, CODE_BYTES = 4;
    __device__ static void stage_tabs(float * tab, float * sub, int tid) {
        if      (tid < 16) tab[tid]      = tid < 2 ? pxq1_book_g[tid] : 0.f;
        else if (tid < 32) sub[tid - 16] = pxq6_sub16_g[tid - 16];
    }
    __device__ static float bookv(int i) { return pxq1_book_g[i & 1]; }
    __device__ static float anchor(const uint8_t * panel, int row) {
        return __half2float(((const half *)panel)[row]);
    }
    __device__ static void row_effs(const uint8_t * slab, int row, float anch, const float * sub, float * eff) {
        const int sb = slab[row];
        eff[0] = anch * sub[sb & 0xf];
        eff[1] = anch * sub[sb >> 4];
    }
    // pair b (elems 2b, 2b+1): single LE u32 word, elem j sign at bit j (1 bit/elem)
    __device__ static float2 pair(const uint32_t * q, int b, const float * tab) {
        const uint32_t w = q[0];
        return make_float2(tab[(w >> (2*b)) & 1], tab[(w >> (2*b + 1)) & 1]);
    }
    // K2d: identical bit extraction; the two gathers ride SHFL.IDX (lane l holds tab[l])
    __device__ static float2 pair_shfl(const uint32_t * q, int b, float bv) {
        const uint32_t w = q[0];
        return make_float2(__shfl_sync(0xffffffffu, bv, (int)((w >> (2*b)) & 1)),
                           __shfl_sync(0xffffffffu, bv, (int)((w >> (2*b + 1)) & 1)));
    }
    __device__ static float2 pairl(const uint32_t * q, int b, const float2 * plut) {
        (void)q; (void)b; (void)plut; return make_float2(0.f, 0.f);   // PAIR never offered for P1
    }
};

struct pxq6_pol_p2 {
    static constexpr int SLAB = PXQ2_SLAB_BYTES, HDR = PXQ2_HDR_BYTES, CODE_OFF = 64, NEFF = 2;
    static constexpr int CODE_WORDS = 2, CODE_BYTES = 8;
    __device__ static void stage_tabs(float * tab, float * sub, int tid) {
        if      (tid < 16) tab[tid]      = tid < 4 ? pxq2_book_g[tid] : 0.f;
        else if (tid < 32) sub[tid - 16] = pxq6_sub16_g[tid - 16];
    }
    __device__ static float bookv(int i) { return pxq2_book_g[i & 3]; }
    __device__ static float anchor(const uint8_t * panel, int row) {
        return __half2float(((const half *)panel)[row]);
    }
    __device__ static void row_effs(const uint8_t * slab, int row, float anch, const float * sub, float * eff) {
        const int sb = slab[row];
        eff[0] = anch * sub[sb & 0xf];    // elems 0-15
        eff[1] = anch * sub[sb >> 4];     // elems 16-31
    }
    // pair b (elems 2b, 2b+1): LE u32 word h = b>>3 covers elems 16h..16h+15, 2 bits/elem
    __device__ static float2 pair(const uint32_t * q, int b, const float * tab) {
        const uint32_t w  = q[b >> 3];
        const int      sh = 2 * ((2*b) & 15);
        return make_float2(tab[(w >> sh) & 3], tab[(w >> (sh + 2)) & 3]);
    }
    // K2d: identical field extraction; the two gathers ride SHFL.IDX (lane l holds tab[l])
    __device__ static float2 pair_shfl(const uint32_t * q, int b, float bv) {
        const uint32_t w  = q[b >> 3];
        const int      sh = 2 * ((2*b) & 15);
        return make_float2(__shfl_sync(0xffffffffu, bv, (int)((w >> sh) & 3)),
                           __shfl_sync(0xffffffffu, bv, (int)((w >> (sh + 2)) & 3)));
    }
    // PAIRLUT is 4-bit-only; the pickers force PAIR=false for P2 — compile stub, never executed
    __device__ static float2 pairl(const uint32_t * q, int b, const float2 * plut) {
        (void)q; (void)b; (void)plut; return make_float2(0.f, 0.f);
    }
};

struct pxq6_pol_p3 {
    static constexpr int SLAB = PXQ3_SLAB_BYTES, HDR = PXQ3_HDR_BYTES, CODE_OFF = 64, NEFF = 2;
    static constexpr int CODE_WORDS = 3, CODE_BYTES = 12;
    __device__ static void stage_tabs(float * tab, float * sub, int tid) {
        if      (tid < 16) tab[tid]      = tid < 8 ? pxq3_book_g[tid] : 0.f;
        else if (tid < 32) sub[tid - 16] = pxq6_sub16_g[tid - 16];
    }
    __device__ static float bookv(int i) { return pxq3_book_g[i & 7]; }
    __device__ static float anchor(const uint8_t * panel, int row) {
        return __half2float(((const half *)panel)[row]);
    }
    __device__ static void row_effs(const uint8_t * slab, int row, float anch, const float * sub, float * eff) {
        const int sb = slab[row];
        eff[0] = anch * sub[sb & 0xf];    // elems 0-15
        eff[1] = anch * sub[sb >> 4];     // elems 16-31
    }
    // pair b (elems 2b, 2b+1): bit-plane decode — lo word by half, high plane in q[2]
    __device__ static float2 pair(const uint32_t * q, int b, const float * tab) {
        const int      h  = b >> 3;               // 16-elem half (0 or 1)
        const int      j0 = (2*b) & 15;           // first elem within the half
        const uint32_t lo = q[h];
        const uint32_t hi = q[2] >> (16*h);       // this half's high plane in bits 0..15
        const int c0 = (int)((lo >> (2*j0))     & 3) | (int)(((hi >> j0)       & 1) << 2);
        const int c1 = (int)((lo >> (2*j0 + 2)) & 3) | (int)(((hi >> (j0 + 1)) & 1) << 2);
        return make_float2(tab[c0], tab[c1]);
    }
    // K2d: identical plane reassembly; the two gathers ride SHFL.IDX (lane l holds tab[l])
    __device__ static float2 pair_shfl(const uint32_t * q, int b, float bv) {
        const int      h  = b >> 3;               // 16-elem half (0 or 1)
        const int      j0 = (2*b) & 15;           // first elem within the half
        const uint32_t lo = q[h];
        const uint32_t hi = q[2] >> (16*h);       // this half's high plane in bits 0..15
        const int c0 = (int)((lo >> (2*j0))     & 3) | (int)(((hi >> j0)       & 1) << 2);
        const int c1 = (int)((lo >> (2*j0 + 2)) & 3) | (int)(((hi >> (j0 + 1)) & 1) << 2);
        return make_float2(__shfl_sync(0xffffffffu, bv, c0),
                           __shfl_sync(0xffffffffu, bv, c1));
    }
    __device__ static float2 pairl(const uint32_t * q, int b, const float2 * plut) {
        (void)q; (void)b; (void)plut; return make_float2(0.f, 0.f);   // PAIR forced off for P3
    }
    // PAIRL3 (PXA_PXQ3_PAIRLUT, MODE 5): the P3-specific instruction diet. pair() above
    // reassembles two 3-bit codes from the planes (~10 int ops + 2 smem gathers per pair;
    // the fired gateup loop measures 23% BFE). But the pair's 4 low-plane bits are
    // CONTIGUOUS in q[h] (bits 2*j0..2*j0+3) and its 2 high-plane bits are CONTIGUOUS in
    // q[2] (16*h + j0 == 2*b for BOTH halves, so bits 2b..2b+1). One 64-entry smem LUT
    // keyed [hi2:lo4] makes the whole decode BFE + BFE + BFI + LDS.64. Same book values
    // (the LUT is staged from the same pxq3_book_g via bookv), same FMA order — sourcing
    // only, bit-exact vs pair(). Distinct from the 256-entry byte-keyed K2a PAIRLUT
    // (4-bit tiers, +0.1%): K2a only fused the two gathers (the byte key was already one
    // BFE); this removes the bit-plane reassembly itself (~7 instructions per pair).
    __device__ static float2 pairl3(const uint32_t * q, int b, const float2 * plut) {
        const uint32_t lo4 = (q[b >> 3] >> (2*((2*b) & 15))) & 0xfu;   // low planes: elems 2b, 2b+1
        const uint32_t hi2 = (q[2]      >> (2*b))            & 0x3u;   // high plane: bits 2b, 2b+1
        return plut[lo4 | (hi2 << 4)];
    }
};

// ---------------------------------------------------------------------------------------------
// convert.cu dequant fallback wrappers (dequant->cublas keeps PXQ2/PXQ3 functional on every
// arch, incl. the 1080Ti/sm_61). Instantiate the SHARED k_pxq6_dequant_matrix.
// ---------------------------------------------------------------------------------------------
template <typename dst_t>
static void dequantize_row_pxq2_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row,
                                     cudaStream_t stream) {
    if (nrows % PXQ2_BM != 0 || n_per_row % PXQ2_QK != 0) {
        fprintf(stderr, "FATAL: dequantize_row_pxq2_cuda: nrows=%lld n_per_row=%lld not slab-aligned\n",
                (long long)nrows, (long long)n_per_row);
        abort();
    }
    int dev = -1; cudaGetDevice(&dev);
    pxq6_maybe_upload_tables(dev);    // shared SUB16 override
    pxq23_maybe_upload_books(dev);
    const int kslabs = (int)(n_per_row / PXQ2_QK);
    const int64_t nslabs = (nrows / PXQ2_BM) * (int64_t)kslabs;
    k_pxq6_dequant_matrix<pxq6_pol_p2, dst_t><<<nslabs, 64, 0, stream>>>((const uint8_t *)vx, y, kslabs, n_per_row);
}

template <typename dst_t>
static void dequantize_row_pxq3_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row,
                                     cudaStream_t stream) {
    if (nrows % PXQ3_BM != 0 || n_per_row % PXQ3_QK != 0) {
        fprintf(stderr, "FATAL: dequantize_row_pxq3_cuda: nrows=%lld n_per_row=%lld not slab-aligned\n",
                (long long)nrows, (long long)n_per_row);
        abort();
    }
    int dev = -1; cudaGetDevice(&dev);
    pxq6_maybe_upload_tables(dev);    // shared SUB16 override
    pxq23_maybe_upload_books(dev);
    const int kslabs = (int)(n_per_row / PXQ3_QK);
    const int64_t nslabs = (nrows / PXQ3_BM) * (int64_t)kslabs;
    k_pxq6_dequant_matrix<pxq6_pol_p3, dst_t><<<nslabs, 64, 0, stream>>>((const uint8_t *)vx, y, kslabs, n_per_row);
}

template <typename dst_t>
static void dequantize_row_pxq1_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row,
                                     cudaStream_t stream) {
    if (nrows % PXQ1_BM != 0 || n_per_row % PXQ1_QK != 0) {
        fprintf(stderr, "FATAL: dequantize_row_pxq1_cuda: nrows=%lld n_per_row=%lld not slab-aligned\n",
                (long long)nrows, (long long)n_per_row);
        abort();
    }
    int dev = -1; cudaGetDevice(&dev);
    pxq6_maybe_upload_tables(dev);    // shared SUB16 override
    pxq23_maybe_upload_books(dev);
    const int kslabs = (int)(n_per_row / PXQ1_QK);
    const int64_t nslabs = (nrows / PXQ1_BM) * (int64_t)kslabs;
    k_pxq6_dequant_matrix<pxq6_pol_p1, dst_t><<<nslabs, 64, 0, stream>>>((const uint8_t *)vx, y, kslabs, n_per_row);
}
