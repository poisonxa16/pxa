// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// test-pxq-lutdec-bench.cu — the GO/NO-GO microbench for activation-LUT codebook decode
// ("Psumbook") on PXQ2/PXQ3 dense decode. Measurement only: nothing here is a shipping kernel,
// and nothing in the engine calls it.
//
// THE QUESTION. The incumbent decode mmv (k_pxq6_mmv) does, per weight: extract the code,
// gather book[code] from shared memory, multiply by the activation, accumulate. An activation
// LUT precomputes, once per activation tile, the inner product of every codebook code-PAIR
// against that tile's two activations, so decode becomes one lookup + one add per PAIR instead
// of two gathers + two FMAs. Fewer ALU and LSU ops per weight; in exchange a table has to be
// built per activation chunk, which costs shared memory, two __syncthreads per chunk, and
// (because the table is indexed by data) shared-memory bank conflicts.
//
// Both effects are real and they pull in opposite directions on this silicon, which is why this
// file measures them SEPARATELY instead of A/B-ing one kernel:
//
//   A  incumbent      the shipping k_pxq6_mmv, unmodified — the number to beat.
//   A' incumbent-clone  a local copy with the same inner loop. Control: A' must land on A, or
//                     the clone-based arms below are not measuring what they claim.
//   B  memory ceiling same launch geometry, same global loads (codes, anchors, sub-scales, x
//                     staging), decode arithmetic deleted. B is the floor no decode-side
//                     optimization can go below. **(A - B) is the ENTIRE budget any LUT
//                     competes for.** If A - B is small, the answer is NO-GO no matter how
//                     clever the table is, because the kernel is bound by something the table
//                     does not touch.
//   C  no book gather FMAs kept, the shared-memory book gather replaced by a register value.
//                     Splits the budget into "book traffic on the LSU" vs "the FMAs themselves".
//   D1/D4/D8  Psumbook  the real thing: a pair-psum table rebuilt per 32-element chunk, with
//                     1, 4 or 8 64-row panels per block. The panel count is the amortization
//                     lever — the table build is paid once per BLOCK, so 8 panels spreads it
//                     over 512 rows instead of 64. Reports its own registers/smem/occupancy.
//
// Plus an M sweep (1/2/4/8 columns) on clone-vs-LUT arms that both read x from global, because
// the ratio moves with M: the incumbent gathers the book once and pays M FMAs, while the LUT
// pays M lookups. Which way that cuts is a measurement, not an argument.
//
// Reported per arm: median-of-7 ms, achieved GB/s over the weight bytes actually streamed,
// registers/thread, static+dynamic smem/block, and blocks-per-SM from the occupancy API — so a
// loss can be attributed to occupancy rather than guessed at.
//
// The LUT arms fold pair-psums instead of per-element products, so they are NOT bit-exact
// against the incumbent; max abs / max rel error against arm A is printed for every LUT arm.
// This harness never gates on that: it is a speed and occupancy instrument.
#include "ggml-cuda/pxa/pxq6.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)

static inline uint64_t rs(uint64_t & s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

// ---------------------------------------------------------------------------------------------
// per-format LUT traits: how many entries a PAIR key can take, and how to lift a pair key out of
// the packed code words. The key schemes are the ones the engine already uses — PXQ2's pair is
// one nibble of the 2-bit plane; PXQ3's is PAIRL3's 6-bit (lo4 | hi2<<4) bit-plane key — so a
// table entry means the same thing here as it does in pxq6_stage_pairlut3.
// ---------------------------------------------------------------------------------------------
template <class POL> struct lut_traits;

template <> struct lut_traits<pxq6_pol_p2> {
    static constexpr int ENT = 16;                 // 2 codes x 2 bits
    __device__ static int key(const uint32_t * q, int b) {
        return (int)((q[b >> 3] >> (2 * ((2*b) & 15))) & 0xF);
    }
    __device__ static void codes(int k, int & c0, int & c1) { c0 = k & 3; c1 = (k >> 2) & 3; }
};

template <> struct lut_traits<pxq6_pol_p3> {
    static constexpr int ENT = 64;                 // 2 codes x 3 bits, PAIRL3 key packing
    __device__ static int key(const uint32_t * q, int b) {
        const int      h  = b >> 3;
        const int      j0 = (2*b) & 15;
        const uint32_t lo = q[h];
        const uint32_t hi = q[2] >> (16*h);
        return (int)((lo >> (2*j0)) & 0xF) | (int)(((hi >> j0) & 3) << 4);
    }
    __device__ static void codes(int k, int & c0, int & c1) {
        c0 = (k & 3)        | (((k >> 4) & 1) << 2);
        c1 = ((k >> 2) & 3) | (((k >> 5) & 1) << 2);
    }
};

// ---------------------------------------------------------------------------------------------
// arms A' / B / C — one kernel, the decode body switched at compile time. Everything outside the
// body (grid, block, x staging, code loads, row_effs, the canonical two-level fold, the kseg
// reduction, the store) is k_pxq6_mmv's, verbatim, so the arms differ ONLY in the decode.
// ---------------------------------------------------------------------------------------------
#define BARM_INC   0   // pxq6_dot32's body
#define BARM_MEM   1   // loads kept, decode arithmetic deleted
#define BARM_NOLDS 2   // FMAs kept, book gather replaced by a register

template <class POL, int BARM>
static __global__ void __launch_bounds__(256)
k_bench_mmv(const uint8_t * __restrict__ W, const float * __restrict__ x,
            float * __restrict__ dst, const int R, const int K) {
    const int p = blockIdx.x;

    extern __shared__ float bsmem[];
    float * xs  = bsmem;
    float * red = bsmem + K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];

    __shared__ float tab[32];
    __shared__ float sub[16];
    POL::stage_tabs(tab, sub, threadIdx.x);
    __syncthreads();

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const uint8_t * pan = pxq6_panel<POL>(W, 0, panels, p, kslabs);
    const float anch = POL::anchor(pan, row);

    // arm C: the same value every gather would have produced for code 0, held in a register.
    const float bconst  = tab[0];
    const float bconst2 = tab[1];

    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_MMV_SPLIT_MAX);
    float su = 0.f;
    for (int c = 0; c < nfix; ++c) {
        const int b0 = (kslabs*c)/nfix, b1 = (kslabs*(c+1))/nfix;
        float t = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            const uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
            const float   * xk   = xs + kb*PXQ6_QK;
            float eff[POL::NEFF];
            POL::row_effs(slab, row, anch, sub, eff);
            uint32_t q[POL::CODE_WORDS];
            pxq6_ldcodes<POL, false>(slab + POL::CODE_OFF + row*POL::CODE_BYTES, q);

            if constexpr (BARM == BARM_MEM) {
                // Consume the loaded words and both eff scales so nothing is dead-code
                // eliminated; ~4 instructions per 32-element chunk instead of ~160.
                uint32_t acc = q[0];
                #pragma unroll
                for (int w = 1; w < POL::CODE_WORDS; ++w) acc ^= q[w];
                t += eff[0]*(float)(acc & 1u) + eff[1]*xk[0];
            } else {
                float tt[POL::NEFF];
                #pragma unroll
                for (int i = 0; i < POL::NEFF; ++i) tt[i] = 0.f;
                #pragma unroll
                for (int b = 0; b < 16; ++b) {
                    float2 pv;
                    if constexpr (BARM == BARM_NOLDS) {
                        // key extraction kept (it is real work); only the two LSU gathers go.
                        // one predicate + one select in place of the two LSU gathers; the key
                        // extraction and the FMAs stay exactly where the incumbent has them.
                        const int k = lut_traits<POL>::key(q, b);
                        pv = make_float2((k & 1) ? bconst : bconst2, (k & 2) ? bconst2 : bconst);
                    } else {
                        pv = POL::pair(q, b, tab);
                    }
                    tt[(b*POL::NEFF) >> 4] = pxq6_acc2(tt[(b*POL::NEFF) >> 4], pv.x, xk[2*b], pv.y, xk[2*b+1]);
                }
                t += (POL::NEFF == 1) ? eff[0]*tt[0] : eff[0]*tt[0] + eff[1]*tt[1];
            }
        }
        su += t;
    }
    red[kseg*64 + row] = su;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;
        #pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s*64 + row];
        dst[p*PXQ6_BM + row] = u;
    }
}

// ---------------------------------------------------------------------------------------------
// arm D — the Psumbook. RPB 64-row panels per block (64*RPB threads, one row per thread), the
// whole block walking chunks in lockstep so ONE table serves 64*RPB rows.
//
//   tbl[b*ENT + k] = book[c0(k)]*x[32*kb + 2b] + book[c1(k)]*x[32*kb + 2b + 1]
//
// so a pair of weights costs one indexed shared load and one add instead of two gathers and two
// FMAs. x is read from global for the build (every block reads the same words — L2-hot) rather
// than staged, which is what makes the table's shared-memory cost independent of K: the
// incumbent's x staging is 20 KB at K=5120 and is itself the occupancy limiter today.
// ---------------------------------------------------------------------------------------------
template <class POL, int RPB>
static __global__ void __launch_bounds__(64*RPB)
k_bench_lut(const uint8_t * __restrict__ W, const float * __restrict__ x,
            float * __restrict__ dst, const int R, const int K) {
    using LT = lut_traits<POL>;
    constexpr int ENT = LT::ENT;
    constexpr int NT  = 64*RPB;

    __shared__ float tbl[16*ENT];
    __shared__ float sub[16];
    __shared__ float tab[32];
    POL::stage_tabs(tab, sub, (int)threadIdx.x);
    __syncthreads();

    const int row = threadIdx.x & 63;
    const int p   = blockIdx.x*RPB + (int)(threadIdx.x >> 6);
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const uint8_t * pan = pxq6_panel<POL>(W, 0, panels, p, kslabs);
    const float anch = POL::anchor(pan, row);

    // the book in registers — the build needs it per entry and it is 4 or 8 floats.
    float bk[ENT <= 16 ? 4 : 8];
    #pragma unroll
    for (int i = 0; i < (ENT <= 16 ? 4 : 8); ++i) bk[i] = POL::bookv(i);

    float acc = 0.f;
    for (int kb = 0; kb < kslabs; ++kb) {
        const float * xk = x + kb*PXQ6_QK;
        // build: 16 pair positions x ENT entries, spread over the block's threads.
        __syncthreads();
        #pragma unroll 1
        for (int i = threadIdx.x; i < 16*ENT; i += NT) {
            const int b = i / ENT, k = i - b*ENT;
            int c0, c1; LT::codes(k, c0, c1);
            tbl[i] = __fmaf_rn(bk[c1], xk[2*b+1], bk[c0]*xk[2*b]);
        }
        __syncthreads();

        const uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
        float eff[POL::NEFF];
        POL::row_effs(slab, row, anch, sub, eff);
        uint32_t q[POL::CODE_WORDS];
        pxq6_ldcodes<POL, false>(slab + POL::CODE_OFF + row*POL::CODE_BYTES, q);

        float t0 = 0.f, t1 = 0.f;
        #pragma unroll
        for (int b = 0; b < 8; ++b)  t0 += tbl[b*ENT + LT::key(q, b)];
        #pragma unroll
        for (int b = 8; b < 16; ++b) t1 += tbl[b*ENT + LT::key(q, b)];
        acc += eff[0]*t0 + eff[1]*t1;
    }
    dst[p*PXQ6_BM + row] = acc;
}

// ---------------------------------------------------------------------------------------------
// M sweep. Both arms read x from GLOBAL (no staging) so the comparison isolates the decode:
// the incumbent-shaped arm gathers the book ONCE per pair and pays NCOL FMAs; the LUT arm pays
// NCOL indexed shared loads and NCOL adds. These two are comparable to each other and NOT to
// arms A/B/C above, which stage x.
// ---------------------------------------------------------------------------------------------
template <class POL, int NCOL>
static __global__ void __launch_bounds__(64)
k_bench_mcol_inc(const uint8_t * __restrict__ W, const float * __restrict__ x,
                 float * __restrict__ dst, const int R, const int K) {
    __shared__ float tab[32];
    __shared__ float sub[16];
    POL::stage_tabs(tab, sub, (int)threadIdx.x);
    __syncthreads();

    const int row = threadIdx.x & 63;
    const int p   = blockIdx.x;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const uint8_t * pan = pxq6_panel<POL>(W, 0, panels, p, kslabs);
    const float anch = POL::anchor(pan, row);

    float acc[NCOL];
    #pragma unroll
    for (int m = 0; m < NCOL; ++m) acc[m] = 0.f;

    for (int kb = 0; kb < kslabs; ++kb) {
        const uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
        float eff[POL::NEFF];
        POL::row_effs(slab, row, anch, sub, eff);
        uint32_t q[POL::CODE_WORDS];
        pxq6_ldcodes<POL, false>(slab + POL::CODE_OFF + row*POL::CODE_BYTES, q);
        float t[NCOL][2];
        #pragma unroll
        for (int m = 0; m < NCOL; ++m) { t[m][0] = 0.f; t[m][1] = 0.f; }
        #pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float2 pv = POL::pair(q, b, tab);       // ONE gather pair for all NCOL columns
            #pragma unroll
            for (int m = 0; m < NCOL; ++m) {
                const float * xm = x + (size_t)m*K + kb*PXQ6_QK;
                t[m][b >> 3] = pxq6_acc2(t[m][b >> 3], pv.x, xm[2*b], pv.y, xm[2*b+1]);
            }
        }
        #pragma unroll
        for (int m = 0; m < NCOL; ++m) acc[m] += eff[0]*t[m][0] + eff[1]*t[m][1];
    }
    #pragma unroll
    for (int m = 0; m < NCOL; ++m) dst[(size_t)m*R + p*PXQ6_BM + row] = acc[m];
}

template <class POL, int NCOL>
static __global__ void __launch_bounds__(64)
k_bench_mcol_lut(const uint8_t * __restrict__ W, const float * __restrict__ x,
                 float * __restrict__ dst, const int R, const int K) {
    using LT = lut_traits<POL>;
    constexpr int ENT = LT::ENT;

    __shared__ float tbl[NCOL][16*ENT];
    __shared__ float sub[16];
    __shared__ float tab[32];
    POL::stage_tabs(tab, sub, (int)threadIdx.x);
    __syncthreads();

    const int row = threadIdx.x & 63;
    const int p   = blockIdx.x;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const uint8_t * pan = pxq6_panel<POL>(W, 0, panels, p, kslabs);
    const float anch = POL::anchor(pan, row);

    float bk[ENT <= 16 ? 4 : 8];
    #pragma unroll
    for (int i = 0; i < (ENT <= 16 ? 4 : 8); ++i) bk[i] = POL::bookv(i);

    float acc[NCOL];
    #pragma unroll
    for (int m = 0; m < NCOL; ++m) acc[m] = 0.f;

    for (int kb = 0; kb < kslabs; ++kb) {
        __syncthreads();
        #pragma unroll 1
        for (int i = threadIdx.x; i < NCOL*16*ENT; i += 64) {
            const int m = i / (16*ENT), r = i - m*16*ENT;
            const int b = r / ENT,      k = r - b*ENT;
            int c0, c1; LT::codes(k, c0, c1);
            const float * xm = x + (size_t)m*K + kb*PXQ6_QK;
            tbl[m][r] = __fmaf_rn(bk[c1], xm[2*b+1], bk[c0]*xm[2*b]);
        }
        __syncthreads();

        const uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
        float eff[POL::NEFF];
        POL::row_effs(slab, row, anch, sub, eff);
        uint32_t q[POL::CODE_WORDS];
        pxq6_ldcodes<POL, false>(slab + POL::CODE_OFF + row*POL::CODE_BYTES, q);

        float t[NCOL][2];
        #pragma unroll
        for (int m = 0; m < NCOL; ++m) { t[m][0] = 0.f; t[m][1] = 0.f; }
        #pragma unroll
        for (int b = 0; b < 16; ++b) {
            const int k = LT::key(q, b);              // ONE key extraction for all NCOL columns
            #pragma unroll
            for (int m = 0; m < NCOL; ++m) t[m][b >> 3] += tbl[m][b*ENT + k];
        }
        #pragma unroll
        for (int m = 0; m < NCOL; ++m) acc[m] += eff[0]*t[m][0] + eff[1]*t[m][1];
    }
    #pragma unroll
    for (int m = 0; m < NCOL; ++m) dst[(size_t)m*R + p*PXQ6_BM + row] = acc[m];
}

// ---------------------------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------------------------
template <class POL>
static void build_raw(std::vector<uint8_t> & raw, int panels, int kslabs, uint64_t & seed) {
    raw.assign((size_t)panels*(POL::HDR + (size_t)kslabs*POL::SLAB), 0);
    for (int p = 0; p < panels; ++p) {
        uint8_t * pan = raw.data() + (size_t)p*(POL::HDR + (size_t)kslabs*POL::SLAB);
        // anchors: well-conditioned fp16 in [0.25, 1.0) so the reference sum stays in range
        // (this is a speed harness; degenerate anchors would only add noise to the error print).
        uint16_t * a = (uint16_t *)pan;
        for (int r = 0; r < 64; ++r) {
            const float v = 0.25f + (float)(rs(seed) & 0xffff)/65536.0f*0.75f;
            const __half h = __float2half_rn(v);
            memcpy(&a[r], &h, 2);
        }
        for (int kb = 0; kb < kslabs; ++kb) {
            uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
            for (int i = 0; i < POL::SLAB; ++i) slab[i] = (uint8_t)(rs(seed) & 0xff);
        }
    }
}

struct arm_stat { double ms; int regs; size_t smem; int bpsm; double gbs; double maxabs; double maxrel; };

static const int REP  = 7;    // medians of 7, the campaign's standard
static int       g_iter = 50;  // launches folded into ONE timing region

// A 5120x5120 decode mmv is 80 blocks on a 56-SM P100 — about 15 us of work, which is the same
// order as cudaEvent jitter and kernel launch overhead. Timing ONE launch would put a +/-13%
// error bar on every number in this table and make the whole comparison unreadable. So each
// timing region folds g_iter back-to-back launches and reports the per-launch mean of the
// median region. The kernels are idempotent (same inputs, same dst), so repeating them is
// legitimate; what it does NOT model is a cold L2, which is also true of decode in a real
// forward pass, where the same weights are streamed layer after layer.
template <typename F>
static double time_med(F && launch) {
    cudaEvent_t a, b; CUCK(cudaEventCreate(&a)); CUCK(cudaEventCreate(&b));
    launch(); CUCK(cudaDeviceSynchronize());          // warm
    std::vector<float> ms(REP);
    for (int i = 0; i < REP; ++i) {
        CUCK(cudaEventRecord(a));
        for (int it = 0; it < g_iter; ++it) launch();
        CUCK(cudaEventRecord(b));
        CUCK(cudaEventSynchronize(b));
        CUCK(cudaEventElapsedTime(&ms[i], a, b));
    }
    std::sort(ms.begin(), ms.end());
    CUCK(cudaEventDestroy(a)); CUCK(cudaEventDestroy(b));
    return ms[REP/2]/(double)g_iter;
}

template <typename K>
static void kattr(K k, int blk, size_t dsm, int & regs, size_t & smem, int & bpsm) {
    cudaFuncAttributes fa; CUCK(cudaFuncGetAttributes(&fa, (const void *)k));
    regs = fa.numRegs;
    smem = fa.sharedSizeBytes + dsm;
    CUCK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, (const void *)k, blk, dsm));
}

static void err_vs(const std::vector<float> & ref, const std::vector<float> & got,
                   double & maxabs, double & maxrel) {
    maxabs = 0.0; maxrel = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = fabs((double)got[i] - (double)ref[i]);
        if (d > maxabs) maxabs = d;
        const double den = fabs((double)ref[i]);
        if (den > 1e-6) { const double r = d/den; if (r > maxrel) maxrel = r; }
    }
}

static void row(const char * name, const arm_stat & s, const arm_stat & base, bool show_err) {
    printf("  %-22s %8.3f ms  %7.1f GB/s  %6.2fx  regs %3d  smem %6zu B  %d blk/SM",
           name, s.ms, s.gbs, base.ms/s.ms, s.regs, s.smem, s.bpsm);
    if (show_err) printf("  maxabs %.3e maxrel %.3e", s.maxabs, s.maxrel);
    printf("\n");
}

template <class POL>
static void run_tier(const char * label, int R, int K, uint64_t & seed) {
    const int panels = R/PXQ6_BM, kslabs = K/PXQ6_QK;
    std::vector<uint8_t> raw;
    build_raw<POL>(raw, panels, kslabs, seed);

    const int MMAX = 8;
    std::vector<float> hx((size_t)MMAX*K);
    for (auto & v : hx) v = ((float)(rs(seed) & 0xffff)/65536.0f - 0.5f)*2.0f;

    uint8_t * dW = nullptr; float * dx = nullptr; float * dy = nullptr;
    CUCK(cudaMalloc(&dW, raw.size()));
    CUCK(cudaMemcpy(dW, raw.data(), raw.size(), cudaMemcpyHostToDevice));
    CUCK(cudaMalloc(&dx, hx.size()*sizeof(float)));
    CUCK(cudaMemcpy(dx, hx.data(), hx.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUCK(cudaMalloc(&dy, (size_t)MMAX*R*sizeof(float)));

    // weight bytes streamed once per launch — the only DRAM traffic that scales with the matrix
    const double gb = (double)raw.size()/(1024.0*1024.0*1024.0);
    const size_t smem_inc = (size_t)K*sizeof(float) + PXQ4_MMV_KSEG*64*sizeof(float);

    int nsm = 1; { int d=0; CUCK(cudaGetDevice(&d)); cudaDeviceProp pr; CUCK(cudaGetDeviceProperties(&pr,d)); nsm = pr.multiProcessorCount; }
    printf("\n%s  R=%d K=%d  (%zu B of weights, %.2f MiB)  grid %d blocks on %d SMs\n",
           label, R, K, raw.size(), (double)raw.size()/(1024.0*1024.0), panels, nsm);
    if (smem_inc > 46*1024) { printf("  (K too wide for the incumbent's x staging — skipped)\n"); }

    std::vector<float> ref((size_t)R), got((size_t)R);
    int32_t * dids = nullptr;
    CUCK(cudaMalloc(&dids, sizeof(int32_t)));
    CUCK(cudaMemset(dids, 0, sizeof(int32_t)));

    arm_stat A{}, tmp{};

    // ---- arm A: the shipping kernel, called exactly as ggml-cuda.cu calls it ----
    {
        auto k = k_pxq6_mmv<POL, PXQ6_MODE_TAB, false>;
        A.ms = time_med([&]{
            k<<<dim3(panels,1,1), 256, smem_inc>>>(dW,
                (const char *)dx, 0, 0, (char *)dy, 0, 0,
                (const char *)dids, sizeof(int32_t), sizeof(int32_t), R, K, 1);
        });
        CUCK(cudaGetLastError());
        kattr(k, 256, smem_inc, A.regs, A.smem, A.bpsm);
        A.gbs = gb/(A.ms/1000.0);
        CUCK(cudaMemcpy(ref.data(), dy, (size_t)R*sizeof(float), cudaMemcpyDeviceToHost));
        row("A incumbent", A, A, false);
    }

    // ---- arms A' / B / C ----
    auto run_barm = [&](const char * name, auto k, bool cmp) {
        tmp.ms = time_med([&]{ k<<<dim3(panels,1,1), 256, smem_inc>>>(dW, dx, dy, R, K); });
        CUCK(cudaGetLastError());
        kattr(k, 256, smem_inc, tmp.regs, tmp.smem, tmp.bpsm);
        tmp.gbs = gb/(tmp.ms/1000.0);
        if (cmp) {
            CUCK(cudaMemcpy(got.data(), dy, (size_t)R*sizeof(float), cudaMemcpyDeviceToHost));
            err_vs(ref, got, tmp.maxabs, tmp.maxrel);
        }
        row(name, tmp, A, cmp);
    };
    run_barm("A' clone (control)", k_bench_mmv<POL, BARM_INC>,   true);
    run_barm("B memory ceiling",  k_bench_mmv<POL, BARM_MEM>,    false);
    run_barm("C no book gather",  k_bench_mmv<POL, BARM_NOLDS>,  false);

    // ---- arm D: the Psumbook, 1 / 4 / 8 panels per block ----
    auto run_lut = [&](const char * name, auto k, int rpb) {
        if (panels % rpb) { printf("  %-22s (panels %% %d != 0, skipped)\n", name, rpb); return; }
        const int nthr = 64*rpb;
        tmp.ms = time_med([&]{ k<<<dim3(panels/rpb,1,1), nthr>>>(dW, dx, dy, R, K); });
        CUCK(cudaGetLastError());
        kattr(k, nthr, 0, tmp.regs, tmp.smem, tmp.bpsm);
        tmp.gbs = gb/(tmp.ms/1000.0);
        CUCK(cudaMemcpy(got.data(), dy, (size_t)R*sizeof(float), cudaMemcpyDeviceToHost));
        err_vs(ref, got, tmp.maxabs, tmp.maxrel);
        row(name, tmp, A, true);
    };
    run_lut("D1 LUT 1 panel/blk",  k_bench_lut<POL, 1>, 1);
    run_lut("D4 LUT 4 panels/blk", k_bench_lut<POL, 4>, 4);
    run_lut("D8 LUT 8 panels/blk", k_bench_lut<POL, 8>, 8);

    // ---- M sweep ----
    printf("  -- M sweep (both arms read x from global; comparable to each other only) --\n");
    auto run_m = [&](const char * name, auto k, int ncol) {
        const double gbm = gb;   // weights are streamed once regardless of NCOL
        arm_stat s{};
        s.ms = time_med([&]{ k<<<dim3(panels,1,1), 64>>>(dW, dx, dy, R, K); });
        CUCK(cudaGetLastError());
        kattr(k, 64, 0, s.regs, s.smem, s.bpsm);
        s.gbs = gbm/(s.ms/1000.0);
        printf("  %-22s M=%d  %8.3f ms  %7.1f GB/s  regs %3d  smem %6zu B  %d blk/SM\n",
               name, ncol, s.ms, s.gbs, s.regs, s.smem, s.bpsm);
    };
    run_m("  Mi incumbent-shape", k_bench_mcol_inc<POL, 1>, 1);
    run_m("  Ml LUT",             k_bench_mcol_lut<POL, 1>, 1);
    run_m("  Mi incumbent-shape", k_bench_mcol_inc<POL, 2>, 2);
    run_m("  Ml LUT",             k_bench_mcol_lut<POL, 2>, 2);
    run_m("  Mi incumbent-shape", k_bench_mcol_inc<POL, 4>, 4);
    run_m("  Ml LUT",             k_bench_mcol_lut<POL, 4>, 4);
    run_m("  Mi incumbent-shape", k_bench_mcol_inc<POL, 8>, 8);
    run_m("  Ml LUT",             k_bench_mcol_lut<POL, 8>, 8);

    CUCK(cudaFree(dW)); CUCK(cudaFree(dx)); CUCK(cudaFree(dy)); CUCK(cudaFree(dids));
}

int main(int argc, char ** argv) {
    int R = 5120, K = 5120, R2 = 17408;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-R")    && i+1 < argc) R  = atoi(argv[++i]);
        if (!strcmp(argv[i], "-K")    && i+1 < argc) K  = atoi(argv[++i]);
        if (!strcmp(argv[i], "-R2")   && i+1 < argc) R2 = atoi(argv[++i]);
        if (!strcmp(argv[i], "-iter") && i+1 < argc) g_iter = atoi(argv[++i]);
    }
    int dev = 0; CUCK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUCK(cudaGetDeviceProperties(&prop, dev));
    printf("device %d: %s (sm_%d%d)  %d SMs  %zu B smem/SM  peak %.1f GB/s\n",
           dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (size_t)prop.sharedMemPerMultiprocessor,
           2.0*prop.memoryClockRate*1e3*(prop.memoryBusWidth/8)/1e9);
    printf("A' must land on A, or the clone-based arms are not measuring the incumbent.\n");
    printf("(A - B) is the whole budget any decode-side optimization competes for.\n");

    printf("timing: %d launches folded per region, medians of %d, per-launch means reported\n", g_iter, REP);

    uint64_t seed = 0x4c555444ull;   // "LUTD"
    // Two shapes: a square attention-sized linear (80 blocks — under one wave and therefore
    // latency-exposed, which is what a real decode launch looks like) and an ffn_up/gate-sized
    // one (272 blocks, ~5 waves, where the steady-state throughput shows).
    run_tier<pxq6_pol_p2>("PXQ2 (4-entry book, nibble pair key)", R, K, seed);
    run_tier<pxq6_pol_p3>("PXQ3 (8-entry book, PAIRL3 6-bit key)", R, K, seed);
    if (R2 > 0) {
        run_tier<pxq6_pol_p2>("PXQ2 wide", R2, K, seed);
        run_tier<pxq6_pol_p3>("PXQ3 wide", R2, K, seed);
    }
    return 0;
}
