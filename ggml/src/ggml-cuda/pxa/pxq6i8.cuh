// pxq6i8.cuh — N13: int8 DP4A MMQ-style PREFILL tile for the PXQ formats (sm_61 first).
//
// WHY THIS FORM (design justification, N13): the two alternatives were (a) repack PXQ ->
// q8_0 and reuse the existing mul_mat_q_id MMQ path, or (b) a PXQ trait inside mmq.cuh.
// (a) needs either a resident q8_0 shadow of the expert weights (~2x the 2-bit tier's VRAM —
// impossible next to a ~10 GiB model on the 11 GiB 1080Ti) or a per-invocation full-tensor
// repack through global memory, AND q8_0's per-32 scale cannot carry PXQ's per-16 (per-8 for
// HQ) sub-scales without a second requantization error on top of the s8 book snap. (b) wedges
// panel-interleaved, row-meta'd PXQ layouts into trait machinery built for standard ggml block
// quants — touching shared MMQ code used by every other type/arch. So: a THIRD form — an int8
// twin of k_pxq6_gemm_grouped inside the existing policy-templated family. It reuses the whole
// proven grouped-prefill driver structure (tile map, row mapping, GLU, scatter, bias folding,
// mixed up/gate/down formats via per-GEMM pickers) and confines every new line to this file
// plus one env-gated dispatch branch. The "repack tile in smem" idea survives as the W-stage:
// codes -> s8 via the snapped book, per-16 scale folded into a per-tile fp32 rescale.
//
// NUMERIC CONTRACT (q3-s8-snap.md): book values snap to s8 as q_i = rint(book_i * 127/absmax);
// dequant w = (anchor * SUB[s] * (absmax/127)) * q_s8 — the /127 and book absmax fold into the
// per-16-block fp32 eff scale; the stored fp16 anchor is untouched. For the PX16 book
// (PXQ5/PXQ6/PXQ6HQ, absmax = 1.0) this reproduces the frozen Q3 s8 book
// [-125,-93,-71,-53,-38,-25,-12,0,11,22,33,46,60,76,97,127] exactly (+0.073% wrel refit /
// +0.219% post-hoc table swap — this kernel is the post-hoc form). PXQ2 (LM4) and PXQ3 (LM8)
// books are snapped by the same rule at runtime; their 2/3-bit quantization floors dwarf the
// <=0.4%-of-absmax grid displacement. PXQ4's E2M1 book is NOT snap-validated -> declined.
// Activations are quantized q8_1-style (per-32 absmax/127, round-to-nearest) — the same
// class of activation quantization the stock MMQ path applies to every quantized type.
//
// NOT bit-exact vs the fused fp16 path (int8 x int8 -> fp32 rescale vs strict-k half2 chains):
// G3-class gates (temp-0 coherence + semantic equivalence + top-5 logit spot-check) are
// mandatory before any live use. Flag OFF (default) = byte-identical dispatch to the old build.
//
// ENV: PXA_PXQ_INT8_PREFILL=0 (default, OFF) | 1 (sm_61 only — the ship gate) | 2 (all archs,
// TEST: sm_70+ has real dp4a; sm_60 falls to the emulated ggml_cuda_dp4a — never ship that).
//
// ---- PARKED (measured losses, 2026-07-22 sm_61 A/B — do NOT re-run blind) -------------------
// Two maturation candidates for this tile were implemented, gated (PXA_PXQ_I8_DBUF /
// PXA_PXQ_I8_BN=128), proven output-equivalent, and A/B'd on the 1080 Ti (PXQ2, fair-battle
// rev2: fa-off ub768 cold 5432-tok prompt, median of 3; same-binary parent 958-976 t/s pf):
//   I8-DBUF  (smem ping-pong, 1 barrier/slab, prefetch-into-regs): -0.4%  -> KILL.
//     ptxas: the shipped kernel already runs 223-228 regs, 0 spills, 4 blocks/SM — it is
//     ILP-saturated, not latency-bound; the ping-pong bought nothing.
//   I8-BN128 (64x128 tiles, 128 thr, halves W stream + W-decode per token): -23%   -> KILL
//     at ub768: this MoE routes ~96 tokens/expert per ubatch, so 128-token tiles run mostly
//     partial-fill (idle FMA lanes) and grid.y halves. Only plausible at >=2x per-expert
//     batches (ub2048-class), which the 11 GB card cannot hold next to a ~10 GB model.
//   Stack (both): -21% -> KILL.
// Even the <64,false> template restructure of this kernel (lambda-split stage/FMA) measured
// -2.5% on its own — keep the body below textually as-is. Full diff of the parked variants:
// the measurements below.
//
// ---- int8 double-buffer / 128-token tile evaluation (sm_61) ----
// Stream-K (splitting the K-reduction across extra blocks) was audited before building: at
// ub768 this MoE's tile count (grid.y = sum over active experts of ceil(tokens_e/64), grid.x =
// R/64 panels) already spans many multiples of the 1080Ti's 28 SMs x 4 blocks/SM = 112
// concurrent-block ceiling — the grid is wave-bound, not SM-starved, so adding more blocks via a
// K-split would only add partial-sum reduction overhead with no occupancy upside. Stream-K
// DECLINED without a build (kill line: "the -13% gap is layout maturity beyond economic reach").
// Instead: the fp16 sibling kernel (k_pxq6_gemm_grouped) already ships K4 RAGTAIL (bit-exact
// ragged-tile FMA skip, PXA_PXQ6_RAGTAIL, default ON) — this int8 tile NEVER got the same
// treatment (every block always ran the full 8x8 accumulate regardless of tile.nrows). Since
// MoE routing at ub768 leaves most experts under one 64-token tile (partial fill is the norm,
// not the exception — matches the I8-BN128 finding above), this is very likely the single
// biggest untapped win in this file. Ported verbatim (same fma_on pattern, same coarse per-8-
// token-group granularity, same "skipped FMAs only touch already-zeroed A data" bit-exact
// argument as the fp16 kernel) as RAG, gated PXA_PXQ_I8_RAGTAIL (default OFF until proven; A/B
// below). This is additive to N13 and does not touch the parked/killed variants above.
#pragma once

// requires: pxq6.cuh (policies incl. pxq6_pol_p2/p3 via pxq23.cuh, pxq6_panel, pxq6_ldcodes,
// pxq4_tile_info, pxq4_rowmap, pxq4_glu_apply, PXA_PXQ_FMT_*), common.cuh (ggml_cuda_dp4a).

#include "pxa-enhance.cuh"   // level default: ENHANCE -> mode 1 (sm_61 ship gate); env wins

static inline int pxa_pxq_int8_prefill() {
    static const int mode = [](){
        const int m = pxa_int8_prefill_mode_resolve();
        if (m) fprintf(stderr, "PXA_PXQ_INT8_PREFILL: mode %d (N13 dp4a int8 MMQ-tile prefill, %s; "
                       "s8 book snap — NOT bit-exact vs fused fp16, G3-gated)\n",
                       m, m == 1 ? "sm_61 only" : "ALL archs (TEST)");
        return m;
    }();
    return mode;
}

// K4-for-int8: ragged-tile FMA skip, ported from k_pxq6_gemm_grouped's RAG (bit-exact — the
// skipped FMAs only ever touch A data that was already forced to zero by the a_valid staging
// branch, so the accumulated sum is identical either way). Default OFF;
// PXA_PXQ_I8_RAGTAIL=1 to enable, independent of PXA_PXQ6_RAGTAIL (separate kernel family).
static inline bool pxa_pxq_i8_ragtail() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ_I8_RAGTAIL");
        const bool v = e ? atoi(e) != 0 : false;
        fprintf(stderr, "PXA_PXQ_I8_RAGTAIL: %s (K4-for-int8 ragged-tile FMA skip, bit-exact)\n",
                v ? "ON" : "OFF");
        return v;
    }();
    return on;
}

// ---------------------------------------------------------------------------------------------
// per-32-group s8 quantize of one staged f32 row (shared by the gather and GLU kernels).
// d = absmax/127, q = rint(x/d) in [-127,127]; d==0 group -> all-zero codes.
// ---------------------------------------------------------------------------------------------
static __device__ __forceinline__ void pxqi8_quant_row_groups(const float * __restrict__ xs,
        uint32_t * __restrict__ q_out, float * __restrict__ d_out, const int ngroups, const int nthr) {
    for (int g = threadIdx.x; g < ngroups; g += nthr) {
        const float * x = xs + g*PXQ4_QK;
        float amax = 0.f;
        #pragma unroll
        for (int t = 0; t < PXQ4_QK; ++t) amax = fmaxf(amax, fabsf(x[t]));
        const float inv = amax > 0.f ? 127.f/amax : 0.f;
        d_out[g] = amax / 127.f;
        #pragma unroll
        for (int w = 0; w < PXQ4_QK/4; ++w) {
            uint32_t pk = 0;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int v = (int)rintf(x[4*w + b]*inv);   // |x| <= amax -> v in [-127,127]
                pk |= (uint32_t)((uint8_t)v) << (8*b);
            }
            q_out[(size_t)g*(PXQ4_QK/4) + w] = pk;
        }
    }
}

// activation gather + q8 quantize: flat expert-grouped f32 rows -> s8 [total][K] (packed LE
// u32 words) + per-32-group scales f32 [total][K/32]. Same row mapping semantics as
// k_pxq4_gather_a_f16. dyn smem = K floats.
static __global__ void k_pxqi8_gather_quant(const char * __restrict__ src1,
        uint8_t * __restrict__ Aq, float * __restrict__ Ad,
        const pxq4_rowmap * __restrict__ map,
        const int64_t K, const int64_t ne11, const size_t nb11, const size_t nb12) {
    extern __shared__ float pxqi8_xs[];
    const int i = blockIdx.x;
    const int32_t i11 = map[i].i1 % ne11;
    const int32_t i12 = map[i].i2;
    const float * src = (const float *)(src1 + i11*nb11 + i12*nb12);
    for (int64_t j = threadIdx.x; j < K; j += blockDim.x) pxqi8_xs[j] = src[j];
    __syncthreads();
    pxqi8_quant_row_groups(pxqi8_xs, (uint32_t *)(Aq + (size_t)i*K), Ad + (size_t)i*(K/PXQ4_QK),
                           (int)(K/PXQ4_QK), blockDim.x);
}

// GLU epilogue + q8 quantize: C_gate/C_up f32 (flat [total][R]) -> H s8 + per-32 scales.
// The GLU arithmetic is pxq4_glu_apply verbatim. dyn smem = R floats.
static __global__ void k_pxqi8_glu_quant(const float * __restrict__ xg, const float * __restrict__ gu,
        uint8_t * __restrict__ Hq, float * __restrict__ Hd, const int R,
        const int unary, const float alpha, const float limit) {
    extern __shared__ float pxqi8_hs[];
    const int i = blockIdx.x;
    const float * g = xg + (size_t)i*R;
    const float * u = gu + (size_t)i*R;
    for (int j = threadIdx.x; j < R; j += blockDim.x)
        pxqi8_hs[j] = pxq4_glu_apply(g[j], u[j], unary, alpha, limit);
    __syncthreads();
    pxqi8_quant_row_groups(pxqi8_hs, (uint32_t *)(Hq + (size_t)i*R), Hd + (size_t)i*(R/PXQ4_QK),
                           R/PXQ4_QK, blockDim.x);
}

// ---------------------------------------------------------------------------------------------
// the int8 grouped prefill GEMM. Same grid/tile geometry as k_pxq6_gemm_grouped (64 threads,
// 64x64 C tile, one panel x one 64-token tile per block), but the inner product is dp4a on s8:
//   W-stage: thread = row; codes -> s8 via the snapped book (POL::pair over a smem table
//            holding the SNAPPED INTEGER values), packed 4/word into sWq[8][64]; the per-group
//            eff scale (anchor*SUB, POL::row_effs) x (book_absmax/127) goes to sEff[NG][64].
//   A-stage: thread = token; one aligned 32 B s8 read + the per-slab d scale.
//   FMA:     8 rows x 8 tokens per thread; per 32-k slab: NG integer sub-sums (dp4a chains of
//            8/NG words) -> fp32: acc += dA * sum_n eff[n]*sumi[n]. Token side is register-
//            cached in two 4-token panels to keep smem traffic ~3x below the naive form.
// NG = POL::NEFF (1 for P5's single per-32 scale, 2 for P2/P3/P6 per-16, 4 for P6HQ per-8).
// ---------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------
// N14 (BLUT): byte-keyed W-decode for the 2-bit tier.
//
// The W-stage was the measured wall of this tile (the RAGTAIL A/B root-caused the per-kb
// stage/sync pair, not the FMA loop, as the dominant cost). For PXQ2 the old stage spent, per
// thread per slab, 32 shared-memory loads of the snapped book PLUS 32 float->int conversions to
// build 8 packed s8 words -- for a book with FOUR entries. PXQ2 packs 4 codes per byte
// contiguously (elem 4w+i at bits 2i of code byte w), so ONE 256-entry uint32 LUT keyed on the
// code byte yields a whole packed s8 word per lookup: 8 smem loads and zero float conversions
// per thread per slab. This is the 2-bit twin of the 4-bit PAIRLUT idiom already in pxq6.cuh
// (there one byte = two 4-bit codes; here one byte = four 2-bit codes, so it is 2x denser).
//
// BIT-IDENTICAL by construction: the LUT is built FROM stab (the same snapped book the old path
// indexed) with the same (uint8_t)(int) narrowing and the same byte order, so the sWq words are
// literally the same bits. Only the 2-bit tier qualifies -- PXQ3 is bit-plane coded (codes are
// not byte-contiguous) and the 4-/5-bit tiers keep the pair() path.
//
// Cost: +1 KiB smem/block (5.1 -> 6.1 KiB; 4 blocks/SM stays well inside sm_61's 96 KiB) and a
// one-time 4-entries-per-thread LUT fill before the kslab loop.
// ---------------------------------------------------------------------------------------------
template <class POL> struct pxqi8_blut_ok { static constexpr bool value = false; };
template <> struct pxqi8_blut_ok<pxq6_pol_p2> { static constexpr bool value = true; };

// ---------------------------------------------------------------------------------------------
// N15 (JTAB / PXA_PXQ2_MMQ): the q8_0 loader route for the 2-bit tier.
//
// N13/N14 keep PXQ2's native per-16 scale structure inside the tile: NEFF=2 eff scales per
// 32-K slab, so the FMA loop runs TWO dp4a sub-chains per slab and pays, per (row, token,
// slab), two int->float converts, two float multiplies and an add. That is the shape the
// panel format has, not the shape the mature MMQ compute loop wants: q8_0 is ONE scale per 32
// values and one dp4a chain.
//
// So fold SUB16 into the int8 book image. For a row-slab whose two SUB groups are s0 (elems
// 0-15) and s1 (elems 16-31), let smax = max(s0, s1) and
//     scale = anchor * SUB16[smax] * bookabsmax / 127            (ONE fp32 scale per 32-K slab)
//     q[j]  = rint( SUB16[s(j)]/SUB16[smax] * BOOK[c(j)]/bookabsmax * 127 )
// so w[j] = scale * q[j] exactly reproduces the dequant contract up to the s8 grid. Because
// s(j) <= smax and |BOOK| <= bookabsmax, |q| <= 127 by construction -- no clipping. This is
// literally q8_0's layout (32 s8 + one scale), which is why the compute loop collapses to a
// single 8-word dp4a chain and a single scale multiply.
//
// DISPLACEMENT: the grid step is scale, i.e. half a step is 0.5/127 of the row-slab's LARGER
// group absmax. Relative to a group's OWN absmax that is 0.5/127 * SUB16[smax]/SUB16[s], and
// SUB16 spans [0.2147, 0.9878] (ratio 4.60), so the worst case over all inputs is 1.81% of
// that group's absmax -- against PXQ2's own 2-bit quantization floor (wrel 0.302), an order
// of magnitude smaller. Normalising per row-slab rather than to the global SUB16[15] is what
// keeps the typical case far below the bound (both groups of a slab are usually adjacent).
//
// DECODE COST: the joint table is (smax, s) -> the four s8 book values packed in a uint32, so
// 16*16 = 256 words = 1 KiB of smem, read ONCE per row-slab-group (two loads per row-slab).
// Expanding a code byte to a packed s8 word is then one PRMT: the byte's four 2-bit codes are
// spread into __byte_perm's four selector nibbles by a second 256-entry table (1 KiB). Net
// smem +1.75 KiB/block over N14 (sEff loses a row); still ~7.9 KiB, 4 blocks/SM unchanged.
//
// Only the 2-bit tier qualifies: the table is indexed by a 2-bit code, and PXQ3 is bit-plane
// coded so its codes are not byte-contiguous.
// ---------------------------------------------------------------------------------------------
template <class POL> struct pxqi8_jt_ok { static constexpr bool value = false; };
template <> struct pxqi8_jt_ok<pxq6_pol_p2> { static constexpr bool value = true; };

static inline bool pxa_pxq2_jtab() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ2_JTAB");
        const bool v = e ? atoi(e) != 0 : true;   // default ON (sm_61 ship gate is the caller's)
        fprintf(stderr, "PXA_PXQ2_JTAB: %s (N15 joint-table q8_0-layout 2-bit W-decode, "
                "<=1.81%% of a group's absmax vs the N13 per-16 form)\n", v ? "ON" : "OFF");
        return v;
    }();
    return on;
}

static inline bool pxa_pxq_i8_blut() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ_I8_BLUT");
        const bool v = e ? atoi(e) != 0 : true;   // default ON (bit-identical)
        fprintf(stderr, "PXA_PXQ_I8_BLUT: %s (N14 byte-keyed 2-bit W-decode, bit-identical)\n",
                v ? "ON" : "OFF");
        return v;
    }();
    return on;
}

typedef void (*pxqi8_gemm_fn)(const uint8_t *, const uint8_t *, const float *, float *,
                              const float *, const size_t, const pxq4_tile_info *, const int, const int);

template <class POL, bool RAG, bool BLUT = false, bool JT = false>
static __global__ void __launch_bounds__(64)
k_pxqi8_gemm_grouped(const uint8_t * __restrict__ W,
                     const uint8_t * __restrict__ Aq, const float * __restrict__ Ad,
                     float * __restrict__ C,
                     const float * __restrict__ bias, const size_t bias_nb1,
                     const pxq4_tile_info * __restrict__ tiles, const int R, const int K) {
    constexpr bool USE_JT = JT && pxqi8_jt_ok<POL>::value;
    constexpr int NG  = USE_JT ? 1 : POL::NEFF;   // integer sub-sum groups per 32-elem slab
    constexpr int WPG = 8 / NG;       // 4-byte k-words per group
    const int panels = R / PXQ4_BM, kslabs = K / PXQ4_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    if (tile.nrows <= 0) return;   // padding tile of a device-built tile list (uniform over the block)
    const uint8_t * pan = pxq6_panel<POL>(W, tile.e, panels, p, kslabs);
    const uint8_t * Aqt = Aq + (size_t)tile.row0*K;
    const float   * Adt = Ad + (size_t)tile.row0*kslabs;
    float         * Ct  = C  + (size_t)tile.row0*R + (size_t)p*PXQ4_BM;

    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float stab[16];               // s8-snapped book (integer values held as float)
    __shared__ uint32_t sWq[8][PXQ4_BM];     // packed s8 W tile, k-word major
    __shared__ uint32_t sAq[8][PXQ4_BN];     // packed s8 A tile, k-word major
    __shared__ float sEff[NG][PXQ4_BM];      // folded per-group W scales (anchor*SUB*absmax/127)
    __shared__ float sDa[PXQ4_BN];           // per-token per-slab A scale
    constexpr bool USE_LUT = BLUT && !USE_JT && pxqi8_blut_ok<POL>::value;
    __shared__ uint32_t sLUT[USE_LUT ? 256 : 1];   // N14: code byte -> 4 packed s8 weights
    __shared__ uint32_t sJT [USE_JT  ? 256 : 1];   // N15: (smax,s) -> 4 packed s8 book values
    __shared__ uint32_t sSEL[USE_JT  ? 256 : 1];   // N15: code byte -> __byte_perm selector
    const int tid = threadIdx.x;
    POL::stage_tabs(tab, sub, tid);
    __syncthreads();
    float bmax = 0.f;                        // book absmax (P2/P3 pad entries are 0 -> harmless)
    #pragma unroll
    for (int t = 0; t < 16; ++t) bmax = fmaxf(bmax, fabsf(tab[t]));
    const float bfold = bmax / 127.f;        // folds into the per-group eff scale (contract)
    if (tid < 16) stab[tid] = rintf(tab[tid] * (127.f/bmax));
    __syncthreads();

    if constexpr (USE_LUT) {   // N14: 256 entries / 64 threads = 4 each; same narrowing as pair()
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int b = tid*4 + i;
            uint32_t v = 0;
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                v |= (uint32_t)(uint8_t)(int)stab[(b >> (2*c)) & 3] << (8*c);
            sLUT[b] = v;
        }
        __syncthreads();
    }

    if constexpr (USE_JT) {   // N15: both 256-entry tables, 4 entries per thread
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int e    = tid*4 + i;
            const int smax = e >> 4, sg = e & 15;
            const float r  = sub[sg] / sub[smax];        // <= 1 whenever sg <= smax (the only use)
            uint32_t v = 0;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                int q = (int)rintf(r * tab[c] * (127.f/bmax));   // single rounding, same rule as stab
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                v |= (uint32_t)(uint8_t)q << (8*c);
            }
            sJT[e] = v;
            // code byte -> PRMT selector: nibble i of the selector picks book entry c[i]
            const int b = e;
            sSEL[b] = (uint32_t)((b & 0x03) | ((b & 0x0C) << 2) | ((b & 0x30) << 4) | ((b & 0xC0) << 6));
        }
        __syncthreads();
    }

    const float anch = POL::HDR ? POL::anchor(pan, tid) : 0.f;
    const int tx = tid & 7, ty = tid >> 3;   // rows 8*tx..+7, tokens 8*ty..+7

    float acc[8][8];
    #pragma unroll
    for (int r8 = 0; r8 < 8; ++r8)
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[r8][j] = 0.f;

    const bool a_valid = tid < tile.nrows;
    const bool fma_on  = !RAG || (8*ty) < tile.nrows;   // K4-for-int8: skip FMAs whose stores are masked

    for (int kb = 0; kb < kslabs; ++kb) {
        const uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
        {   // W-stage: this thread's row -> s8 words + folded eff scales
            uint32_t q[POL::CODE_WORDS];
            pxq6_ldcodes<POL>(slab + POL::CODE_OFF + tid*POL::CODE_BYTES, q);
            float eff[NG];
            if constexpr (USE_JT) {
                // N15: one scale per 32-K slab (q8_0 layout); SUB16 rides in the s8 values.
                const int sb = slab[tid];
                const int s0 = sb & 0xf, s1 = sb >> 4;
                const int smx = s0 > s1 ? s0 : s1;
                const uint32_t t0 = sJT[smx*16 + s0];    // elems 0-15
                const uint32_t t1 = sJT[smx*16 + s1];    // elems 16-31
                const uint8_t * cb = (const uint8_t *)q;
                #pragma unroll
                for (int w = 0; w < 4; ++w) sWq[w][tid] = __byte_perm(t0, 0u, sSEL[cb[w]]);
                #pragma unroll
                for (int w = 4; w < 8; ++w) sWq[w][tid] = __byte_perm(t1, 0u, sSEL[cb[w]]);
                eff[0] = anch * sub[smx];
            } else {
            POL::row_effs(slab, tid, anch, sub, eff);
            if constexpr (USE_LUT) {
                // PXQ2: code byte w holds elems 4w..4w+3 (elem 4w+i at bits 2i) -- exactly the
                // packing sWq[w] wants, so one LUT hit per output word.
                const uint8_t * cb = (const uint8_t *)q;
                #pragma unroll
                for (int w = 0; w < 8; ++w) sWq[w][tid] = sLUT[cb[w]];
            } else {
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                const float2 v0 = POL::pair(q, 2*w,   stab);
                const float2 v1 = POL::pair(q, 2*w+1, stab);
                sWq[w][tid] = (uint32_t)((uint8_t)(int)v0.x)
                            | ((uint32_t)((uint8_t)(int)v0.y) << 8)
                            | ((uint32_t)((uint8_t)(int)v1.x) << 16)
                            | ((uint32_t)((uint8_t)(int)v1.y) << 24);
            }
            }
            }
            #pragma unroll
            for (int n = 0; n < NG; ++n) sEff[n][tid] = eff[n]*bfold;
        }
        if (a_valid) {   // A-stage: this thread's token — one aligned 32 B sector
            const uint8_t * arow = Aqt + (size_t)tid*K + kb*PXQ4_QK;
            const uint4 w0 = *(const uint4 *)arow;
            const uint4 w1 = *(const uint4 *)(arow + 16);
            sAq[0][tid] = w0.x; sAq[1][tid] = w0.y; sAq[2][tid] = w0.z; sAq[3][tid] = w0.w;
            sAq[4][tid] = w1.x; sAq[5][tid] = w1.y; sAq[6][tid] = w1.z; sAq[7][tid] = w1.w;
            sDa[tid] = Adt[(size_t)tid*kslabs + kb];
        } else {
            #pragma unroll
            for (int w = 0; w < 8; ++w) sAq[w][tid] = 0;
            sDa[tid] = 0.f;
        }
        __syncthreads();
        if (fma_on) {
        #pragma unroll
        for (int jb = 0; jb < 2; ++jb) {     // two 4-token register panels
            uint32_t aw[4][8]; float da[4];
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const int t = 8*ty + 4*jb + jj;
                da[jj] = sDa[t];
                #pragma unroll
                for (int w = 0; w < 8; ++w) aw[jj][w] = sAq[w][t];
            }
            #pragma unroll
            for (int r8 = 0; r8 < 8; ++r8) {
                const int r = 8*tx + r8;
                uint32_t ww[8];
                #pragma unroll
                for (int w = 0; w < 8; ++w) ww[w] = sWq[w][r];
                float ef[NG];
                #pragma unroll
                for (int n = 0; n < NG; ++n) ef[n] = sEff[n][r];
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    float sum = 0.f;
                    #pragma unroll
                    for (int n = 0; n < NG; ++n) {
                        int s = 0;
                        #pragma unroll
                        for (int w = 0; w < WPG; ++w)
                            s = ggml_cuda_dp4a((int)ww[n*WPG + w], (int)aw[jj][n*WPG + w], s);
                        sum += ef[n]*(float)s;
                    }
                    acc[r8][4*jb + jj] += da[jj]*sum;
                }
            }
        }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r8 = 0; r8 < 8; ++r8) {
        const int row = 8*tx + r8;
        const float b = bias ? *(const float *)((const char *)bias + (size_t)tile.e*bias_nb1
                                                + (size_t)(p*PXQ4_BM + row)*sizeof(float)) : 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int t = 8*ty + j;
            if (t < tile.nrows) Ct[(size_t)t*R + row] = acc[r8][j] + b;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// N16 (PXA_PXQ2_MMQ): the q8_0 COMPUTE loop for the 2-bit tier.
//
// N15 gave this tile q8_0's NUMBERS -- one fp32 scale per 32-K slab, one dp4a chain -- but left
// N13's compute loop underneath: 64 threads, an 8x8 register accumulator block per thread,
// 223-228 registers, 4 blocks/SM, i.e. 256 of the 2048 threads an sm_61 SM can hold (12.5%).
// The measured ceiling was never the arithmetic, it was that operating point. Upstream's
// IQ2_KS rides mmq.cuh at a different one: WARP_SIZE lanes over ROWS, nwarps over TOKENS, and
// only (mmq_x/nwarps)*(mmq_y/WARP_SIZE) accumulators per thread -- 16 here instead of 64.
//
// This is that loop with a PXQ2 loader in front of it. The inner product is
// vec_dot_q8_0_q8_1_impl's shape verbatim (VDR = 8 four-byte words per dp4a group, which under
// the N15 contract is exactly one 32-element PXQ2 slab: 32 s8 against one fp32 scale on each
// side), and the x/y tiles use mmq.cuh's padded-row indexing. Only the loader is new: it turns
// a (panel, slab, row) into the (32 s8, one scale) pair the loop wants, through the same joint
// table N15 builds, so the numbers are N15's and only the schedule changes.
//
// Staging is 4 slabs (128 K-elements, 32 four-byte words) per step, which keeps the whole
// working set -- x 8.25 KiB, y 8.25 KiB, scales 2 KiB, tables 2 KiB -- near 20.7 KiB, so shared
// memory allows 4 blocks/SM and the accumulator budget is what decides occupancy. No global
// allocation is added: the compute buffer is untouched.
// ---------------------------------------------------------------------------------------------
#define PXQ2_MMQ_WS  32
#define PXQ2_MMQ_VDR 8      // == VDR_Q8_0_Q8_1_MMQ: 8 words = one 32-element slab

// One (panel, slab, row) -> 8 packed s8 words + that row-slab's single scale (the N15 contract).
static __device__ __forceinline__ void pxq2_q8_slab(
        const uint8_t * __restrict__ slab, const int row, const float anch,
        const float * __restrict__ sub, const uint32_t * __restrict__ jt,
        const uint32_t * __restrict__ sel, const float bfold,
        uint32_t * __restrict__ out8, float & scale) {
    const int sb  = slab[row];
    const int s0  = sb & 0xf, s1 = sb >> 4;
    const int smx = s0 > s1 ? s0 : s1;
    const uint32_t t0 = jt[smx*16 + s0];      // elems 0-15
    const uint32_t t1 = jt[smx*16 + s1];      // elems 16-31
    const uint8_t * cb = slab + pxq6_pol_p2::CODE_OFF + row*pxq6_pol_p2::CODE_BYTES;
    #pragma unroll
    for (int w = 0; w < 4; ++w) out8[w] = __byte_perm(t0, 0u, sel[cb[w]]);
    #pragma unroll
    for (int w = 4; w < 8; ++w) out8[w] = __byte_perm(t1, 0u, sel[cb[w]]);
    scale = anch * sub[smx] * bfold;
}

template <int MMQ_X, int MMQ_Y, int NWARPS, bool RAG>
static __global__ void __launch_bounds__(PXQ2_MMQ_WS*NWARPS)
k_pxq2_mmq_grouped(const uint8_t * __restrict__ W,
                   const uint8_t * __restrict__ Aq, const float * __restrict__ Ad,
                   float * __restrict__ C,
                   const float * __restrict__ bias, const size_t bias_nb1,
                   const pxq4_tile_info * __restrict__ tiles, const int R, const int K) {
    using POL = pxq6_pol_p2;
    constexpr int WS   = PXQ2_MMQ_WS;
    constexpr int KSL  = 4;                  // slabs staged per K step
    constexpr int KW   = KSL*8;              // four-byte words per row per K step
    constexpr int XSTR = KW + 1;             // padded row stride (mmq.cuh's +1 idiom)
    constexpr int NTHR = WS*NWARPS;
    constexpr int ACC  = (MMQ_X/NWARPS)*(MMQ_Y/WS);
    constexpr int RPS  = NTHR / MMQ_Y;       // (row, slab) pairs staged per thread-row sweep

    const int panels = R / MMQ_Y, kslabs = K / PXQ4_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    if (tile.nrows <= 0) return;
    const uint8_t * pan = pxq6_panel<POL>(W, tile.e, panels, p, kslabs);
    const uint8_t * Aqt = Aq + (size_t)tile.row0*K;
    const float   * Adt = Ad + (size_t)tile.row0*kslabs;
    float         * Ct  = C  + (size_t)tile.row0*R + (size_t)p*MMQ_Y;

    __shared__ float    tab[16], sub[16];
    __shared__ uint32_t sJT[256], sSEL[256];
    __shared__ uint32_t x_qs[MMQ_Y*XSTR];
    __shared__ float    x_df[MMQ_Y*KSL];
    __shared__ uint32_t y_qs[MMQ_X*XSTR];
    __shared__ float    y_df[MMQ_X*KSL];

    const int tid = threadIdx.y*WS + threadIdx.x;
    POL::stage_tabs(tab, sub, tid);
    __syncthreads();

    float bmax = 0.f;
    #pragma unroll
    for (int t = 0; t < 16; ++t) bmax = fmaxf(bmax, fabsf(tab[t]));
    const float bfold = bmax / 127.f;
    for (int e = tid; e < 256; e += NTHR) {          // joint table + PRMT selector table
        const int smax = e >> 4, sg = e & 15;
        const float r  = sub[sg] / sub[smax];
        uint32_t v = 0;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            int q = (int)rintf(r * tab[c] * (127.f/bmax));
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            v |= (uint32_t)(uint8_t)q << (8*c);
        }
        sJT[e]  = v;
        sSEL[e] = (uint32_t)((e & 0x03) | ((e & 0x0C) << 2) | ((e & 0x30) << 4) | ((e & 0xC0) << 6));
    }
    __syncthreads();

    const int lrow = tid % MMQ_Y;                    // row / token staged by this thread
    const int lsl  = tid / MMQ_Y;                    // its slab within the K step
    const float anch = __half2float(((const half *)pan)[lrow]);
    const bool  aok  = lrow < tile.nrows;

    float sum[ACC];
    #pragma unroll
    for (int a = 0; a < ACC; ++a) sum[a] = 0.f;

    for (int kb0 = 0; kb0 < kslabs; kb0 += KSL) {
        #pragma unroll
        for (int s = lsl; s < KSL; s += RPS) {       // W: PXQ2 panel -> q8_0-shaped x tile
            uint32_t w8[8] = {0,0,0,0,0,0,0,0}; float sc = 0.f;
            if (kb0 + s < kslabs) {
                const uint8_t * slab = pan + POL::HDR + (size_t)(kb0 + s)*POL::SLAB;
                pxq2_q8_slab(slab, lrow, anch, sub, sJT, sSEL, bfold, w8, sc);
            }
            #pragma unroll
            for (int w = 0; w < 8; ++w) x_qs[lrow*XSTR + s*8 + w] = w8[w];
            x_df[lrow*KSL + s] = sc;
        }
        #pragma unroll
        for (int s = lsl; s < KSL; s += RPS) {       // A: already q8-per-32 from the gather
            const bool ok = aok && (kb0 + s < kslabs);
            const uint8_t * arow = Aqt + (size_t)lrow*K + (size_t)(kb0 + s)*PXQ4_QK;
            uint4 v0 = make_uint4(0,0,0,0), v1 = v0;
            if (ok) { v0 = *(const uint4 *)arow; v1 = *(const uint4 *)(arow + 16); }
            uint32_t * d = &y_qs[lrow*XSTR + s*8];
            d[0]=v0.x; d[1]=v0.y; d[2]=v0.z; d[3]=v0.w;
            d[4]=v1.x; d[5]=v1.y; d[6]=v1.z; d[7]=v1.w;
            y_df[lrow*KSL + s] = ok ? Adt[(size_t)lrow*kslabs + kb0 + s] : 0.f;
        }
        __syncthreads();

        // Register-cached x panel. The naive mmq.cuh ordering re-reads this thread's weight
        // words once per token group, which on this tile is two shared loads per dp4a and
        // makes the loop LDS-bound long before it is occupancy-bound (measured: the 3.2x
        // occupancy win over N13 bought only +1.7% until this was fixed). MMQ_Y/WS is 2 here,
        // so both rows' 8 words live in 16 registers and each k-group reads x once, y once.
        for (int k01 = 0; k01 < KW; k01 += PXQ2_MMQ_VDR) {   // not unrolled: mmq.cuh does the same
            int   xw[MMQ_Y/WS][PXQ2_MMQ_VDR];
            float xd[MMQ_Y/WS];
            #pragma unroll
            for (int ii = 0; ii < MMQ_Y/WS; ++ii) {
                const int i = ii*WS + threadIdx.x;
                #pragma unroll
                for (int w = 0; w < PXQ2_MMQ_VDR; ++w) xw[ii][w] = (int)x_qs[i*XSTR + k01 + w];
                xd[ii] = x_df[i*KSL + k01/8];
            }
            #pragma unroll
            for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
                const int j = j0 + threadIdx.y;
                if (RAG && j >= tile.nrows) continue;
                int yw[PXQ2_MMQ_VDR];
                #pragma unroll
                for (int w = 0; w < PXQ2_MMQ_VDR; ++w) yw[w] = (int)y_qs[j*XSTR + k01 + w];
                const float dy = y_df[j*KSL + k01/8];
                #pragma unroll
                for (int ii = 0; ii < MMQ_Y/WS; ++ii) {
                    int sumi = 0;
                    #pragma unroll
                    for (int w = 0; w < PXQ2_MMQ_VDR; ++w) sumi = ggml_cuda_dp4a(xw[ii][w], yw[w], sumi);
                    sum[j0/NWARPS*(MMQ_Y/WS) + ii] += xd[ii]*dy*(float)sumi;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
        const int j = j0 + threadIdx.y;
        if (j >= tile.nrows) continue;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += WS) {
            const int i = i0 + threadIdx.x;
            const float b = bias ? *(const float *)((const char *)bias + (size_t)tile.e*bias_nb1
                                                    + (size_t)(p*MMQ_Y + i)*sizeof(float)) : 0.f;
            Ct[(size_t)j*R + i] = sum[j0/NWARPS*(MMQ_Y/WS) + i0/WS] + b;
        }
    }
}

// PXA_PXQ2_MMQ: 0 = N15 numbers on N13's loop; 1 = N15 numbers on mmq.cuh's loop (N16).
// PXA_PXQ2_MMQ_NW picks the warp count (4 or 8) so the accumulator/occupancy trade is A/B-able.
static inline bool pxa_pxq2_mmq() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ2_MMQ");
        const bool v = e ? atoi(e) != 0 : true;
        fprintf(stderr, "PXA_PXQ2_MMQ: %s (N16 mmq.cuh-geometry dp4a compute loop for the 2-bit tier)\n",
                v ? "ON" : "OFF");
        return v;
    }();
    return on;
}
// Warp count. 4 warps (128 threads, 32 accumulators/thread, 128 regs, 4 blocks/SM) measured
// FASTER than 8 warps (256 threads, 16 accumulators, 78 regs, 3 blocks/SM) on sm_61 --
// 1273 vs 1116 t/s cold prefill. Once the x panel is register-cached the loop is ILP-bound
// again, not occupancy-bound, so the wider per-thread accumulator block wins.
static inline int pxa_pxq2_mmq_nw() {
    static const int nw = [](){
        const char * e = getenv("PXA_PXQ2_MMQ_NW");
        const int v = e ? atoi(e) : 4;
        return (v == 8) ? 8 : 4;
    }();
    return nw;
}

// Ragged-tile token skip for the N16 twin. Unlike the N13 tile's PXA_PXQ_I8_RAGTAIL (measured
// a KILL there and still default OFF), skipping token columns past tile.nrows is a clear win
// here -- this MoE routes ~24 of 64 tile tokens per expert at ub768, so most of the token axis
// is padding. Default ON; PXA_PXQ2_MMQ_RAG=0 disables.
static inline bool pxa_pxq2_mmq_rag() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ2_MMQ_RAG");
        return e ? atoi(e) != 0 : true;
    }();
    return on;
}

// launch descriptor: the N16 twin needs mmq.cuh's block shape, not N13's flat 64.
struct pxqi8_kern { pxqi8_gemm_fn fn; dim3 block; };
static pxqi8_gemm_fn pxqi8_pick_gemm_impl(int fmt);
static pxqi8_gemm_fn pxqi8_pick_gemm(int fmt) { return pxqi8_pick_gemm_impl(fmt); }

// per-format picker. Unknown formats decline -> the caller falls back to the proven
// fp16/cublas paths for the WHOLE op. (The retired legacy P4/P5 formats were removed 2026-07-21.)
// RAG selection reads the PXA_PXQ_I8_RAGTAIL gate once (memoized inside pxa_pxq_i8_ragtail()).
static pxqi8_kern pxqi8_pick_kern(int fmt) {
    if (fmt == PXA_PXQ_FMT_P2 && pxa_pxq2_jtab() && pxa_pxq2_mmq()) {
        const bool rag = pxa_pxq2_mmq_rag();
        if (pxa_pxq2_mmq_nw() == 4)
            return { rag ? k_pxq2_mmq_grouped<64,64,4,true> : k_pxq2_mmq_grouped<64,64,4,false>, dim3(PXQ2_MMQ_WS,4) };
        return { rag ? k_pxq2_mmq_grouped<64,64,8,true> : k_pxq2_mmq_grouped<64,64,8,false>, dim3(PXQ2_MMQ_WS,8) };
    }
    return { pxqi8_pick_gemm(fmt), dim3(64) };
}

static pxqi8_gemm_fn pxqi8_pick_gemm_impl(int fmt) {
    const bool rag = pxa_pxq_i8_ragtail();
    const bool blut = pxa_pxq_i8_blut();
    const bool jt   = pxa_pxq2_jtab();
    switch (fmt) {
        case PXA_PXQ_FMT_P6:   return rag ? k_pxqi8_gemm_grouped<pxq6_pol_p6,   true> : k_pxqi8_gemm_grouped<pxq6_pol_p6,   false>;
        case PXA_PXQ_FMT_P6HQ: return rag ? k_pxqi8_gemm_grouped<pxq6_pol_p6hq, true> : k_pxqi8_gemm_grouped<pxq6_pol_p6hq, false>;
        case PXA_PXQ_FMT_P2:
            if (jt) return rag ? k_pxqi8_gemm_grouped<pxq6_pol_p2, true,  false, true>
                               : k_pxqi8_gemm_grouped<pxq6_pol_p2, false, false, true>;
            return rag ? (blut ? k_pxqi8_gemm_grouped<pxq6_pol_p2, true,  true> : k_pxqi8_gemm_grouped<pxq6_pol_p2, true,  false>)
                       : (blut ? k_pxqi8_gemm_grouped<pxq6_pol_p2, false, true> : k_pxqi8_gemm_grouped<pxq6_pol_p2, false, false>);
        case PXA_PXQ_FMT_P3:   return rag ? k_pxqi8_gemm_grouped<pxq6_pol_p3,   true> : k_pxqi8_gemm_grouped<pxq6_pol_p3,   false>;
        default:               return nullptr;
    }
}
