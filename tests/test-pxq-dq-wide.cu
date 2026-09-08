// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// test-pxq-dq-wide.cu — correctness harness for K8 (wide-store full-matrix PXQ dequant) and
// K8-C (the weight-stationary prefetch arena).
//
// PHASE 1  bit-exactness. k_pxq6_dequant_wide must be byte-for-byte identical to the incumbent
//          k_pxq6_dequant_matrix for every tier (PXQ4 / PXQ4HQ / PXQ6R), every shape, and every
//          variant in the NKS x launch_bounds sweep. Zero tolerance: the wide kernel reuses
//          POL::pair / POL::row_effs verbatim and preserves the multiply order, so any mismatch
//          is a real defect, and the equality is what keeps this change a G1/bitwise land that
//          needs no logprob-parity or ppl regate. An independent CPU reference written straight
//          from ggml-pxq6-tables.h validates the incumbent too, so a failure separates "the new
//          kernel is wrong" from "the format contract drifted". Raw slab bytes are synthesized
//          by a deterministic xorshift64 so the whole code space is covered, including fp16
//          anchor subnormals / signed zeros / max, and the PXQ6R hi-bit plane.
//
// PHASE 2  arena race + key aliasing. The only failure mode that matters in K8-C is a read that
//          outruns its write or a ring recycle under a live consumer, which would surface in
//          production only as a rare quality regression. 512 rounds of walk-style prefetch +
//          acquire + an FNV-1a consumer kernel on the main stream, with the arena forced small
//          so every reclaim path fires, must reproduce the reference hashes exactly. Then the
//          two lifetime guards: a hit must be refused while the consumer stream is capturing a
//          CUDA graph, and invalidate_all() must survive being called after the consumer stream
//          has been destroyed (the real llama_free / llama_free_model order).
//
// PHASE 3  achieved GB/s, informational only, never a gate.
//
// Nonzero exit on any mismatch.
#define PXA_DQC_NO_WALKER   // ggml_backend_buft_is_cuda_split has internal linkage in ggml-cuda.cu
#include "ggml-cuda/pxa/pxq6.cuh"
#include "ggml-cuda/pxa/pxa-dqcache.cuh"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)

static int g_fail = 0;
static int g_checks = 0;

static void ck(bool cond, const char * fmt, ...) {
    ++g_checks;
    if (cond) return;
    ++g_fail;
    va_list ap; va_start(ap, fmt);
    printf("  FAIL: "); vprintf(fmt, ap); printf("\n");
    va_end(ap);
}

static inline uint64_t rs(uint64_t & s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

// Self-contained IEEE-754 binary16 conversions. Deliberately NOT ggml's: the point of the CPU
// arm is to be an independent decode of the format spec, and the GPU side uses __half2float /
// __float2half_rn, so the reference has to implement those two operations itself.
static float h2f(uint16_t h) {
    const uint32_t sgn = (uint32_t)(h >> 15) << 31;
    const uint32_t e   = (h >> 10) & 0x1f;
    const uint32_t m   = h & 0x3ff;
    uint32_t o;
    if (e == 0) {
        if (m == 0) {
            o = sgn;
        } else {                                   // subnormal half -> normal float
            int      sh = -1;
            uint32_t mm = m;
            do { mm <<= 1; ++sh; } while (!(mm & 0x400));
            o = sgn | ((uint32_t)(127 - 15 - sh) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (e == 31) {
        o = sgn | 0x7f800000u | (m << 13);
    } else {
        o = sgn | ((e + 112) << 23) | (m << 13);
    }
    float f; memcpy(&f, &o, 4); return f;
}

static uint16_t f2h(float f) {                     // round to nearest, ties to even
    uint32_t x; memcpy(&x, &f, 4);
    const uint16_t sgn = (uint16_t)((x >> 16) & 0x8000);
    const uint32_t be  = (x >> 23) & 0xff;
    const uint32_t man = x & 0x7fffff;
    if (be == 0xff) {
        return (uint16_t)(sgn | 0x7c00 | (man ? (0x200 | (uint16_t)(man >> 13)) : 0));
    }
    const int32_t exp = (int32_t)be - 127;
    if (exp > 15)  return (uint16_t)(sgn | 0x7c00);
    if (exp >= -14) {                              // normal half
        const uint32_t hm  = man >> 13;
        const uint32_t rem = man & 0x1fff;
        uint16_t h = (uint16_t)(sgn | (uint16_t)((exp + 15) << 10) | (uint16_t)hm);
        if (rem > 0x1000 || (rem == 0x1000 && (hm & 1))) ++h;   // carry walks into the exponent
        return h;
    }
    if (exp < -25) return sgn;
    const uint32_t m2  = man | 0x800000;           // subnormal half
    const int      sh  = -exp - 1;                 // -exp - 14 + 13
    uint32_t       hm  = m2 >> sh;
    const uint32_t rem = m2 & ((1u << sh) - 1);
    const uint32_t hu  = 1u << (sh - 1);
    if (rem > hu || (rem == hu && (hm & 1))) ++hm;
    return (uint16_t)(sgn | (uint16_t)hm);
}

// ---------------------------------------------------------------------------------------------
// PHASE 1
// ---------------------------------------------------------------------------------------------
static const float g_book16[16] = PXQ6_BOOK_INIT;
static const float g_book32[32] = PXQ6_LM32_INIT;
static const float g_sub16[16]  = PXQ6_SUB16_INIT;
static const float g_sub8[16]   = PXQ6_SUB8_INIT;

// fp16 anchor generator: normals in +/-8, signed zeros, subnormals, +/-max.
static uint16_t gen_anchor(uint64_t & s) {
    const uint64_t r = rs(s);
    switch (r & 7) {
        case 0: return 0x0000;                                     // +0
        case 1: return 0x8000;                                     // -0
        case 2: return (uint16_t)(1 + (rs(s) % 0x3ff));             // + subnormal
        case 3: return (uint16_t)(0x8000 | (1 + (rs(s) % 0x3ff)));  // - subnormal
        case 4: return 0x7bff;                                     // +max normal
        case 5: return 0xfbff;                                     // -max normal
        default: {
            const float v = ((float)(rs(s) % 100001) / 100000.0f) * 16.0f - 8.0f;
            return f2h(v);
        }
    }
}

template <class POL>
static void build_raw(std::vector<uint8_t> & raw, int panels, int kslabs, uint64_t & seed) {
    const size_t stride = (size_t)POL::HDR + (size_t)kslabs*POL::SLAB;
    raw.assign(stride*panels, 0);
    for (int p = 0; p < panels; ++p) {
        uint8_t * pan = raw.data() + (size_t)p*stride;
        uint16_t * anc = (uint16_t *)pan;
        for (int r = 0; r < PXQ6_BM; ++r) anc[r] = gen_anchor(seed);
        for (size_t b = POL::HDR; b < stride; ++b) pan[b] = (uint8_t)(rs(seed) & 0xff);
    }
}

// Independent CPU reference, written from the format spec in ggml-pxq6-tables.h.
template <class POL>
static void cpu_ref(const std::vector<uint8_t> & raw, int panels, int kslabs, std::vector<uint16_t> & out) {
    const size_t stride = (size_t)POL::HDR + (size_t)kslabs*POL::SLAB;
    const int64_t K = (int64_t)kslabs*PXQ6_QK;
    out.assign((size_t)panels*PXQ6_BM*K, 0);
    const bool  is_p6r = (POL::CODE_WORDS == 5);
    const bool  is_hq  = (POL::NEFF == 4);
    const float * book = is_p6r ? g_book32 : g_book16;
    const float * sub  = is_hq  ? g_sub8   : g_sub16;
    for (int p = 0; p < panels; ++p) {
        const uint8_t * pan = raw.data() + (size_t)p*stride;
        for (int kb = 0; kb < kslabs; ++kb) {
            const uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
            for (int r = 0; r < PXQ6_BM; ++r) {
                const float anch = h2f(((const uint16_t *)pan)[r]);
                const uint8_t * codes = slab + POL::CODE_OFF + (size_t)r*POL::CODE_BYTES;
                uint32_t hi = 0;
                if (is_p6r) memcpy(&hi, codes + 16, 4);
                for (int b = 0; b < 16; ++b) {
                    float eff;
                    if (is_hq) {
                        const int sb = slab[2*r + (b >> 3)];
                        eff = anch * sub[((b >> 2) & 1) ? (sb >> 4) : (sb & 0xf)];
                    } else {
                        const int sb = slab[r];
                        eff = anch * sub[(b >> 3) ? (sb >> 4) : (sb & 0xf)];
                    }
                    const int byte = codes[b];
                    int c0 = byte & 0xf, c1 = byte >> 4;
                    if (is_p6r) {
                        c0 |= (int)((hi >> (2*b))     & 1u) << 4;
                        c1 |= (int)((hi >> (2*b + 1)) & 1u) << 4;
                    }
                    const size_t o = ((size_t)(p*PXQ6_BM + r))*K + (size_t)kb*PXQ6_QK + 2*b;
                    out[o]     = f2h(eff * book[c0]);
                    out[o + 1] = f2h(eff * book[c1]);
                }
            }
        }
    }
}

template <class POL>
static void phase1_shape(const char * tier, int nrows, int n_per_row, uint64_t & seed) {
    const int    panels = nrows / PXQ6_BM;
    const int    kslabs = n_per_row / PXQ6_QK;
    const int64_t nel   = (int64_t)nrows * n_per_row;
    const size_t  obytes = (size_t)nel*sizeof(half);

    std::vector<uint8_t> raw;
    build_raw<POL>(raw, panels, kslabs, seed);

    uint8_t * d_raw = nullptr;
    half * d_inc = nullptr, * d_w = nullptr, * d_w2 = nullptr, * d_w3 = nullptr;
    CUCK(cudaMalloc(&d_raw, raw.size()));
    CUCK(cudaMemcpy(d_raw, raw.data(), raw.size(), cudaMemcpyHostToDevice));
    CUCK(cudaMalloc(&d_inc, obytes));
    CUCK(cudaMalloc(&d_w,   obytes));
    CUCK(cudaMalloc(&d_w2,  obytes));
    CUCK(cudaMalloc(&d_w3,  obytes));

    const int64_t nslabs = (int64_t)panels*kslabs;
    CUCK(cudaMemset(d_inc, 0xCD, obytes));
    k_pxq6_dequant_matrix<POL, half><<<nslabs, 64>>>(d_raw, d_inc, kslabs, n_per_row);
    CUCK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_inc(nel), h_ref, h_w(nel), h_w2(nel), h_w3(nel);
    CUCK(cudaMemcpy(h_inc.data(), d_inc, obytes, cudaMemcpyDeviceToHost));

    // ASSERT 2 — the incumbent itself matches an independent CPU decode of the format spec.
    cpu_ref<POL>(raw, panels, kslabs, h_ref);
    size_t bad = 0; int64_t first = -1;
    for (int64_t i = 0; i < nel; ++i) if (h_inc[i] != h_ref[i]) { if (!bad) first = i; ++bad; }
    ck(bad == 0, "[%s %dx%d] ASSERT2 incumbent vs CPU reference: %zu/%lld halves differ (first @%lld: gpu %04x ref %04x)",
       tier, nrows, n_per_row, bad, (long long)nel, (long long)first,
       first >= 0 ? h_inc[first] : 0, first >= 0 ? h_ref[first] : 0);

    for (int var = 0; var <= 5; ++var) {
        CUCK(cudaMemset(d_w,  0xCD, obytes));
        CUCK(cudaMemset(d_w2, 0xCD, obytes));
        CUCK(cudaMemset(d_w3, 0xCD, obytes));
        pxq6_launch_dequant_wide<POL, half>(d_raw, d_w,  kslabs, n_per_row, nrows, 0, var);
        pxq6_launch_dequant_wide<POL, half>(d_raw, d_w2, kslabs, n_per_row, nrows, 0, var);
        pxq6_launch_dequant_wide<POL, half>(d_raw, d_w3, kslabs, n_per_row, nrows, 0, var);
        CUCK(cudaGetLastError());
        CUCK(cudaDeviceSynchronize());
        CUCK(cudaMemcpy(h_w.data(),  d_w,  obytes, cudaMemcpyDeviceToHost));
        CUCK(cudaMemcpy(h_w2.data(), d_w2, obytes, cudaMemcpyDeviceToHost));
        CUCK(cudaMemcpy(h_w3.data(), d_w3, obytes, cudaMemcpyDeviceToHost));

        // ASSERT 1 — the ship gate.
        bad = 0; first = -1;
        for (int64_t i = 0; i < nel; ++i) if (h_w[i] != h_inc[i]) { if (!bad) first = i; ++bad; }
        if (bad) {
            printf("  first differing elements (variant %d, %s %dx%d):\n", var, tier, nrows, n_per_row);
            int shown = 0;
            for (int64_t i = first; i < nel && shown < 8; ++i) {
                if (h_w[i] == h_inc[i]) continue;
                const int64_t row = i / n_per_row, k = i % n_per_row;
                printf("    panel %lld row %lld k %lld (slab %lld, elem %lld): wide %04x incumbent %04x\n",
                       (long long)(row/PXQ6_BM), (long long)(row%PXQ6_BM), (long long)k,
                       (long long)(k/PXQ6_QK), (long long)(k%PXQ6_QK), h_w[i], h_inc[i]);
                ++shown;
            }
        }
        ck(bad == 0, "[%s %dx%d var%d] ASSERT1 wide vs incumbent: %zu/%lld halves differ",
           tier, nrows, n_per_row, var, bad, (long long)nel);

        // ASSERT 3 — bit-stability across 3 launches into freshly poisoned buffers.
        ck(memcmp(h_w.data(), h_w2.data(), obytes) == 0 && memcmp(h_w.data(), h_w3.data(), obytes) == 0,
           "[%s %dx%d var%d] ASSERT3 wide kernel not bit-stable across 3 runs", tier, nrows, n_per_row, var);
    }

    CUCK(cudaFree(d_raw)); CUCK(cudaFree(d_inc));
    CUCK(cudaFree(d_w)); CUCK(cudaFree(d_w2)); CUCK(cudaFree(d_w3));
}

template <class POL>
static void phase1_tier(const char * tier, uint64_t & seed) {
    printf("PHASE 1 tier %s\n", tier);
    phase1_shape<POL>(tier,   64,   32, seed);   // single slab
    phase1_shape<POL>(tier,   64, 4096, seed);
    phase1_shape<POL>(tier, 1024, 1024, seed);
    phase1_shape<POL>(tier,  256,  288, seed);   // kslabs = 9: 9 % 8 == 1 exercises the ragged tail
    phase1_shape<POL>(tier, 2048, 5120, seed);
}

// ---------------------------------------------------------------------------------------------
// PHASE 2 — arena race + key aliasing
// ---------------------------------------------------------------------------------------------
static __global__ void k_fnv(const uint32_t * __restrict__ w, size_t nw, uint64_t * out) {
    __shared__ uint64_t sh[256];
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = threadIdx.x; i < nw; i += 256) { h ^= (uint64_t)w[i]; h *= 1099511628211ULL; }
    sh[threadIdx.x] = h;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t g = 1469598103934665603ULL;
        for (int t = 0; t < 256; ++t) { g ^= sh[t]; g *= 1099511628211ULL; }
        *out = g;
    }
}

static void phase2(uint64_t & seed) {
    printf("PHASE 2 arena race + key aliasing\n");
    setenv("PXA_DQC_MB",     "192", 1);
    setenv("PXA_DQC_MIN_NY", "1",   1);
    if (!pxa_dqc_on()) { ck(false, "PHASE2 arena refused to arm"); return; }

    const int N = 64, ROUNDS = 512;
    std::vector<uint8_t *> d_raw(N, nullptr);
    std::vector<int64_t>   rows(N), cols(N);
    size_t maxo = 0;
    uint64_t s2 = seed;
    for (int i = 0; i < N; ++i) {
        rows[i] = 64 * (4 + (i % 13));
        cols[i] = 32 * (64 + 16*(i % 9));
        const int kslabs = (int)(cols[i]/PXQ6_QK), panels = (int)(rows[i]/PXQ6_BM);
        std::vector<uint8_t> raw;
        build_raw<pxq6_pol_p6>(raw, panels, kslabs, s2);
        CUCK(cudaMalloc(&d_raw[i], raw.size()));
        CUCK(cudaMemcpy(d_raw[i], raw.data(), raw.size(), cudaMemcpyHostToDevice));
        maxo = std::max(maxo, (size_t)rows[i]*cols[i]*sizeof(half));
    }

    cudaStream_t main_s = nullptr;
    CUCK(cudaStreamCreateWithFlags(&main_s, cudaStreamNonBlocking));
    half * scratch = nullptr; CUCK(cudaMalloc(&scratch, maxo));
    uint64_t * d_h = nullptr; CUCK(cudaMalloc(&d_h, sizeof(uint64_t)*ROUNDS));
    uint64_t * d_r = nullptr; CUCK(cudaMalloc(&d_r, sizeof(uint64_t)*N));

    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_PXQ4);
    ck(to_fp16 != nullptr, "PHASE2 no to_fp16 for PXQ4");
    if (!to_fp16) return;

    // reference hashes, arena untouched
    for (int i = 0; i < N; ++i) {
        to_fp16(d_raw[i], scratch, rows[i], cols[i], main_s);
        k_fnv<<<1, 256, 0, main_s>>>((const uint32_t *)scratch, (size_t)rows[i]*cols[i]/2, d_r + i);
    }
    CUCK(cudaStreamSynchronize(main_s));
    std::vector<uint64_t> h_ref(N);
    CUCK(cudaMemcpy(h_ref.data(), d_r, sizeof(uint64_t)*N, cudaMemcpyDeviceToHost));

    // 512 rounds, no host sync inside the loop: the hash kernels read the arena on main_s while
    // later prefetches write it on the dq stream.
    int hits = 0;
    for (int r = 0; r < ROUNDS; ++r) {
        const int i = r % N;
        for (int d = 0; d <= 2; ++d) {
            const int j = (i + d) % N;
            pxa_dqc_prefetch(0, pxa_dqc_key{ d_raw[j], rows[j], cols[j], (int)GGML_TYPE_PXQ4 }, main_s);
        }
        const half * p = pxa_dqc_acquire(0, d_raw[i], rows[i], cols[i], (int)GGML_TYPE_PXQ4, main_s);
        const uint32_t * src;
        if (p) { ++hits; src = (const uint32_t *)p; }
        else   { to_fp16(d_raw[i], scratch, rows[i], cols[i], main_s); src = (const uint32_t *)scratch; }
        k_fnv<<<1, 256, 0, main_s>>>(src, (size_t)rows[i]*cols[i]/2, d_h + r);
    }
    CUCK(cudaStreamSynchronize(main_s));
    std::vector<uint64_t> h_got(ROUNDS);
    CUCK(cudaMemcpy(h_got.data(), d_h, sizeof(uint64_t)*ROUNDS, cudaMemcpyDeviceToHost));

    int mism = 0, firstr = -1;
    for (int r = 0; r < ROUNDS; ++r) if (h_got[r] != h_ref[r % N]) { if (mism == 0) firstr = r; ++mism; }
    ck(mism == 0, "PHASE2 %d/%d round hashes differ from the arena-free reference (first round %d)",
       mism, ROUNDS, firstr);
    printf("  arena: %d/%d rounds served from the arena\n", hits, ROUNDS);
    pxa_dqc_report_line();

    // negative test — the immutability gate must refuse a COMPUTE buffer, and accept a WEIGHTS one.
    {
        ggml_backend_buffer buf {};
        ggml_tensor s0 {}, s1 {}, node {};
        s0.type = GGML_TYPE_PXQ4;
        s0.ne[0] = 4096; s0.ne[1] = 4096; s0.ne[2] = 1; s0.ne[3] = 1;
        s0.nb[0] = ggml_type_size(GGML_TYPE_PXQ4);
        s0.nb[1] = ggml_row_size(GGML_TYPE_PXQ4, s0.ne[0]);
        s0.nb[2] = s0.nb[1]*s0.ne[1];
        s0.nb[3] = s0.nb[2];
        s0.buffer = &buf;
        s0.data   = d_raw[0];
        s1.type = GGML_TYPE_F32;
        s1.ne[0] = 4096; s1.ne[1] = 2048; s1.ne[2] = 1; s1.ne[3] = 1;
        node.op = GGML_OP_MUL_MAT; node.src[0] = &s0; node.src[1] = &s1;

        buf.usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        ck(pxa_dqc_node_ok(&node, 64, false), "PHASE2 walker refused a legitimate WEIGHTS MUL_MAT");
        buf.usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE;
        ck(!pxa_dqc_node_ok(&node, 64, false), "PHASE2 immutability gate ACCEPTED a COMPUTE buffer");
        buf.usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        ck(!pxa_dqc_node_ok(&node, 64, true),  "PHASE2 walker accepted a split buffer");
        s1.ne[1] = 8;
        ck(!pxa_dqc_node_ok(&node, 64, false), "PHASE2 walker accepted ny below PXA_DQC_MIN_NY");
        s1.ne[1] = 2048; s0.ne[2] = 2;
        ck(!pxa_dqc_node_ok(&node, 64, false), "PHASE2 walker accepted ne[2] != 1 (key would not match)");
    }

    const pxa_dqc_key k0{ d_raw[0], rows[0], cols[0], (int)GGML_TYPE_PXQ4 };

    // CAPTURE GUARD — a live region must never be served into a CUDA graph capture. Under
    // capture the hit's cudaStreamWaitEvent refers to an event recorded outside the capture
    // graph (cudaErrorStreamCaptureIsolation), and the arena pointer would be baked into the
    // executable graph and replayed after the ring recycled the region.
    {
        cudaStream_t cs = nullptr;
        CUCK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        pxa_dqc_prefetch(0, k0, cs);
        ck(pxa_dqc_acquire(0, k0.ptr, k0.rows, k0.cols, k0.type, cs) != nullptr,
           "PHASE2 arena refused the eager warm-up acquire");
        CUCK(cudaStreamSynchronize(cs));

        cudaGraph_t g = nullptr;
        CUCK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeRelaxed));
        const half * during = pxa_dqc_acquire(0, k0.ptr, k0.rows, k0.cols, k0.type, cs);
        const cudaError_t ce = cudaStreamEndCapture(cs, &g);
        ck(during == nullptr, "PHASE2 arena served a hit INTO a CUDA graph capture");
        ck(ce == cudaSuccess, "PHASE2 capture invalidated by an arena acquire: %s", cudaGetErrorString(ce));
        if (g) CUCK(cudaGraphDestroy(g));
        CUCK(cudaStreamDestroy(cs));
    }

    // INVALIDATE AFTER THE CONSUMER STREAM IS GONE — this is the real teardown order:
    // llama_free(ctx) destroys every stream, THEN llama_free_model(model) frees the weight
    // buffers and lands in pxa_dqc_invalidate_all(). Recording the wrap barrier on the stored
    // compute stream here would hand cudaEventRecord a destroyed handle.
    pxa_dqc_prefetch(0, k0, main_s);
    CUCK(cudaStreamSynchronize(main_s));
    CUCK(cudaStreamDestroy(main_s));
    main_s = nullptr;
    pxa_dqc_invalidate_all();
    {
        const cudaError_t ie = cudaGetLastError();
        ck(ie == cudaSuccess, "PHASE2 invalidate_all after stream teardown raised %s", cudaGetErrorString(ie));
        cudaStream_t fs = nullptr;
        CUCK(cudaStreamCreateWithFlags(&fs, cudaStreamNonBlocking));
        ck(pxa_dqc_acquire(0, k0.ptr, k0.rows, k0.cols, k0.type, fs) == nullptr,
           "PHASE2 acquire hit a region that pxa_dqc_invalidate_all() should have dropped");
        CUCK(cudaStreamDestroy(fs));
    }

    for (int i = 0; i < N; ++i) CUCK(cudaFree(d_raw[i]));
    CUCK(cudaFree(scratch)); CUCK(cudaFree(d_h)); CUCK(cudaFree(d_r));
}

// ---------------------------------------------------------------------------------------------
// PHASE 3 — achieved GB/s (informational)
// ---------------------------------------------------------------------------------------------
static void phase3_shape(int nrows, int n_per_row, uint64_t & seed) {
    const int panels = nrows/PXQ6_BM, kslabs = n_per_row/PXQ6_QK;
    const int64_t nel = (int64_t)nrows*n_per_row;
    std::vector<uint8_t> raw;
    build_raw<pxq6_pol_p6>(raw, panels, kslabs, seed);
    uint8_t * d_raw = nullptr; half * d_y = nullptr;
    CUCK(cudaMalloc(&d_raw, raw.size()));
    CUCK(cudaMemcpy(d_raw, raw.data(), raw.size(), cudaMemcpyHostToDevice));
    CUCK(cudaMalloc(&d_y, (size_t)nel*sizeof(half)));

    const double gb = ((double)nel*2.0 + (double)raw.size()) / (1024.0*1024.0*1024.0);
    cudaEvent_t a, b; CUCK(cudaEventCreate(&a)); CUCK(cudaEventCreate(&b));
    const int REP = 20;

    auto run = [&](int var) -> double {
        std::vector<float> ms(REP);
        for (int i = 0; i < REP; ++i) {
            CUCK(cudaEventRecord(a));
            if (var < 0) k_pxq6_dequant_matrix<pxq6_pol_p6, half><<<(int64_t)panels*kslabs, 64>>>(d_raw, d_y, kslabs, n_per_row);
            else         pxq6_launch_dequant_wide<pxq6_pol_p6, half>(d_raw, d_y, kslabs, n_per_row, nrows, 0, var);
            CUCK(cudaEventRecord(b));
            CUCK(cudaEventSynchronize(b));
            CUCK(cudaEventElapsedTime(&ms[i], a, b));
        }
        std::sort(ms.begin(), ms.end());
        return gb / (ms[REP/2] / 1000.0);
    };

    printf("  %4dx%-5d incumbent %6.1f GB/s", nrows, n_per_row, run(-1));
    for (int v = 0; v <= 5; ++v) printf(" | var%d %6.1f", v, run(v));
    printf("\n");

    CUCK(cudaEventDestroy(a)); CUCK(cudaEventDestroy(b));
    CUCK(cudaFree(d_raw)); CUCK(cudaFree(d_y));
}

// ---------------------------------------------------------------------------------------------
int main(int argc, char ** argv) {
    bool do_bw = true;
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], "--no-bw")) do_bw = false;

    int dev = 0; CUCK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUCK(cudaGetDeviceProperties(&prop, dev));
    printf("device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);

    uint64_t seed = 0x50584134ull;   // "PXA4"
    phase1_tier<pxq6_pol_p6>  ("PXQ4   (p6)  ", seed);
    phase1_tier<pxq6_pol_p6hq>("PXQ4HQ (p6hq)", seed);
    phase1_tier<pxq6_pol_p6r> ("PXQ6   (p6r) ", seed);

    phase2(seed);

    if (do_bw) {
        printf("PHASE 3 achieved bandwidth (informational, not a gate)\n");
        phase3_shape(2048, 5120, seed);
        phase3_shape(4096, 4096, seed);
    }

    printf("\n%s: %d checks, %d failures\n", g_fail ? "FAILED" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
