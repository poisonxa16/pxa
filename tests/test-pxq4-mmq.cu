// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// test-pxq4-mmq.cu -- correctness harness for M1 (k_pxq4_mmq_2d, the dp4a MMQ prefill tile for
// the 4-bit PXQ tiers).
//
// The tile is NOT bit-exact against the route it replaces (dequant -> fp16 scratch -> cuBLAS
// HGEMM), so a memcmp is not the gate. What IS checkable exactly is the kernel against its own
// arithmetic contract, and that is what phase 1 does: a CPU emulation of the documented numeric
// contract (per-32 activation q8 snap, s8 book snap, per-16/per-8 fp32 eff scale, dp4a chains)
// accumulated in double must equal the GPU tile to fp32 rounding. A failure there is a kernel
// defect and nothing else. Phase 2 then measures how far that contract sits from an independent
// CPU decode of the format spec at full fp32 -- the quantization noise the parity/ppl gates on
// the live model are there to judge. Phase 3 sweeps ragged M (token counts that do not fill the
// 64-token tile) and ragged panel counts.
//
// Nonzero exit on any mismatch.
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/pxa/pxq6.cuh"
#include "ggml-cuda/pxa/pxq6i8.cuh"
#include "ggml-cuda/pxa/pxq4-mmq.cuh"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)

static int g_fail = 0, g_checks = 0;
static void ck(bool cond, const char * fmt, ...) {
    ++g_checks;
    if (cond) return;
    ++g_fail;
    va_list ap; va_start(ap, fmt);
    printf("  FAIL: "); vprintf(fmt, ap); printf("\n");
    va_end(ap);
}
static inline uint64_t rs(uint64_t & s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

// independent binary16 -> float (the reference must not borrow the GPU's converter)
static float h2f(uint16_t h) {
    const uint32_t sgn = (uint32_t)(h >> 15) << 31;
    const uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t o;
    if (e == 0) {
        if (m == 0) o = sgn;
        else { int sh = -1; uint32_t mm = m; while (!(mm & 0x400)) { mm <<= 1; ++sh; }
               o = sgn | ((uint32_t)(127 - 15 - sh) << 23) | ((mm & 0x3ff) << 13); }
    } else if (e == 31) {
        o = sgn | 0x7f800000u | (m << 13);
    } else {
        o = sgn | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
    }
    float f; memcpy(&f, &o, 4); return f;
}

static const float BOOK[16]  = PXQ6_BOOK_INIT;
static const float SUB16[16] = PXQ6_SUB16_INIT;
static const float SUB8[16]  = PXQ6_SUB8_INIT;

struct fmtdesc {
    const char * name;
    int slab, hdr, code_off, neff, sub_per_slab;   // sub_per_slab: scale bytes per row per slab
    const float * sub;
};
static const fmtdesc FMT_P6   = { "PXQ4",   PXQ6_SLAB_BYTES,   PXQ6_HDR_BYTES,  64, 2, 1, SUB16 };
static const fmtdesc FMT_P6HQ = { "PXQ4HQ", PXQ6HQ_SLAB_BYTES, PXQ6_HDR_BYTES, 128, 4, 2, SUB8  };

// eff scale n (of neff) for (slab, row)
static float eff_of(const fmtdesc & F, const uint8_t * slab, int row, float anch, int n) {
    if (F.neff == 2) { const int sb = slab[row]; return anch * F.sub[n == 0 ? (sb & 0xf) : (sb >> 4)]; }
    const int sb = slab[2*row + (n >> 1)];
    return anch * F.sub[(n & 1) ? (sb >> 4) : (sb & 0xf)];
}
// code of element e (0..31) of (slab, row)
static int code_of(const fmtdesc & F, const uint8_t * slab, int row, int e) {
    const uint8_t b = slab[F.code_off + row*16 + (e >> 1)];
    return (e & 1) ? (b >> 4) : (b & 0xf);
}

int main(int argc, char ** argv) {
    int dev = 0;
    if (argc > 1) dev = atoi(argv[1]);
    CUCK(cudaSetDevice(dev));
    cudaDeviceProp prop; CUCK(cudaGetDeviceProperties(&prop, dev));
    printf("device %d: %s (cc %d.%d)\n", dev, prop.name, prop.major, prop.minor);

    // s8 snapped book (absmax(BOOK) == 1.0)
    float bmax = 0.f; for (int i = 0; i < 16; ++i) bmax = fmaxf(bmax, fabsf(BOOK[i]));
    int qb[16];
    for (int i = 0; i < 16; ++i) { int q = (int)rintf(BOOK[i]*(127.f/bmax)); qb[i] = q < -127 ? -127 : (q > 127 ? 127 : q); }
    printf("s8 book:"); for (int i = 0; i < 16; ++i) printf(" %d", qb[i]); printf("\n");
    const double bfold = (double)(bmax/127.f);

    struct shape { int R, K, ny; };
    const shape shapes[] = {
        {  64,  128,  64 }, {  64,  128,  33 }, { 128,  256,  64 },
        { 256,  512, 129 }, { 192,  384, 200 }, { 128, 1024,  65 },
        { 320,  128,  32 }, { 128,  160,  97 },
    };
    const fmtdesc * fmts[2] = { &FMT_P6, &FMT_P6HQ };
    const int nws[2] = { 4, 8 };

    uint64_t seed = 0x9e3779b97f4a7c15ull;
    for (int fi = 0; fi < 2; ++fi) {
        const fmtdesc & F = *fmts[fi];
        for (size_t si = 0; si < sizeof(shapes)/sizeof(shapes[0]); ++si) {
            const int R = shapes[si].R, K = shapes[si].K, ny = shapes[si].ny;
            const int panels = R/64, kslabs = K/32;
            const size_t pstride = (size_t)F.hdr + (size_t)kslabs*F.slab;
            const size_t wbytes  = (size_t)panels*pstride;

            // --- synthesize raw panel bytes -------------------------------------------------
            std::vector<uint8_t> W(wbytes);
            for (size_t i = 0; i < wbytes; ++i) W[i] = (uint8_t)(rs(seed) & 0xff);
            for (int p = 0; p < panels; ++p) {            // sane fp16 anchors (no inf/nan/denorm)
                uint16_t * an = (uint16_t *)&W[(size_t)p*pstride];
                for (int r = 0; r < 64; ++r) {
                    const int e = 8 + (int)(rs(seed) % 12);           // exp 8..19 -> 2^-7..2^4
                    an[r] = (uint16_t)(((rs(seed) & 1) << 15) | (e << 10) | (rs(seed) & 0x3ff));
                }
            }
            // --- activations ----------------------------------------------------------------
            std::vector<float> A((size_t)ny*K);
            for (size_t i = 0; i < A.size(); ++i) {
                A[i] = (float)((int64_t)(rs(seed) % 20001) - 10000) / 3000.0f;
            }

            // --- CPU: q8 activation stage (must match pxqi8_quant_row_groups exactly) -------
            std::vector<int8_t> Aq_ref((size_t)ny*K);
            std::vector<float>  Ad_ref((size_t)ny*kslabs);
            for (int t = 0; t < ny; ++t) {
                for (int g = 0; g < kslabs; ++g) {
                    const float * x = &A[(size_t)t*K + (size_t)g*32];
                    float amax = 0.f; for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[j]));
                    const float inv = amax > 0.f ? 127.f/amax : 0.f;
                    Ad_ref[(size_t)t*kslabs + g] = amax/127.f;
                    for (int j = 0; j < 32; ++j) Aq_ref[(size_t)t*K + (size_t)g*32 + j] = (int8_t)(int)rintf(x[j]*inv);
                }
            }

            // --- CPU reference: contract emulation (double) + full-fp32 spec decode ---------
            std::vector<double> ref_mmq((size_t)ny*R), ref_spec((size_t)ny*R);
            for (int r = 0; r < R; ++r) {
                const int p = r/64, rr = r%64;
                const uint8_t * pan = &W[(size_t)p*pstride];
                const float anch = h2f(((const uint16_t *)pan)[rr]);
                for (int t = 0; t < ny; ++t) {
                    double acc_m = 0.0, acc_s = 0.0;
                    for (int kb = 0; kb < kslabs; ++kb) {
                        const uint8_t * slab = pan + F.hdr + (size_t)kb*F.slab;
                        const double dy = Ad_ref[(size_t)t*kslabs + kb];
                        const int epg = 32/F.neff;                     // elements per eff group
                        for (int n = 0; n < F.neff; ++n) {
                            const double ef = (double)eff_of(F, slab, rr, anch, n)*bfold;
                            long long sumi = 0;
                            for (int j = 0; j < epg; ++j) {
                                const int e = n*epg + j;
                                sumi += (long long)qb[code_of(F, slab, rr, e)]
                                      * (long long)Aq_ref[(size_t)t*K + (size_t)kb*32 + e];
                            }
                            acc_m += ef*(double)sumi*dy;
                        }
                        for (int e = 0; e < 32; ++e) {
                            const int n = e/(32/F.neff);
                            const double w = (double)eff_of(F, slab, rr, anch, n)*(double)BOOK[code_of(F, slab, rr, e)];
                            acc_s += w*(double)A[(size_t)t*K + (size_t)kb*32 + e];
                        }
                    }
                    ref_mmq[(size_t)t*R + r]  = acc_m;
                    ref_spec[(size_t)t*R + r] = acc_s;
                }
            }

            for (int wi = 0; wi < 2; ++wi) {
                const int nw = nws[wi];

                uint8_t * dW; float * dA; uint8_t * dAq; float * dAd; float * dC;
                pxq4_tile_info * dT;
                const int ntiles = (ny + 63)/64;
                CUCK(cudaMalloc(&dW, wbytes));
                CUCK(cudaMalloc(&dA, (size_t)ny*K*sizeof(float)));
                CUCK(cudaMalloc(&dAq, (size_t)ny*K));
                CUCK(cudaMalloc(&dAd, (size_t)ny*kslabs*sizeof(float)));
                CUCK(cudaMalloc(&dC, (size_t)ny*R*sizeof(float)));
                CUCK(cudaMalloc(&dT, (size_t)ntiles*sizeof(pxq4_tile_info)));
                CUCK(cudaMemcpy(dW, W.data(), wbytes, cudaMemcpyHostToDevice));
                CUCK(cudaMemcpy(dA, A.data(), A.size()*sizeof(float), cudaMemcpyHostToDevice));
                CUCK(cudaMemset(dC, 0xCD, (size_t)ny*R*sizeof(float)));

                k_pxq_tiles_2d<<<(ntiles+255)/256, 256>>>(dT, ny, ntiles);
                k_pxq4mmq_quant_rows<<<ny, 128>>>(dA, dAq, dAd, K);
                CUCK(cudaGetLastError());

                // the activation stage must reproduce the CPU q8 snap byte for byte
                std::vector<int8_t> Aq_gpu((size_t)ny*K); std::vector<float> Ad_gpu((size_t)ny*kslabs);
                CUCK(cudaMemcpy(Aq_gpu.data(), dAq, Aq_gpu.size(), cudaMemcpyDeviceToHost));
                CUCK(cudaMemcpy(Ad_gpu.data(), dAd, Ad_gpu.size()*sizeof(float), cudaMemcpyDeviceToHost));
                ck(memcmp(Aq_gpu.data(), Aq_ref.data(), Aq_ref.size()) == 0,
                   "%s R%d K%d ny%d nw%d: activation s8 codes differ from the CPU snap", F.name, R, K, ny, nw);
                ck(memcmp(Ad_gpu.data(), Ad_ref.data(), Ad_ref.size()*sizeof(float)) == 0,
                   "%s R%d K%d ny%d nw%d: activation scales differ from the CPU snap", F.name, R, K, ny, nw);

                // the templates directly: pxa_pxq4_mmq_nw() memoizes its env read, so the
                // picker cannot sweep the warp count inside one process.
                pxq4mmq_fn fn = nullptr; dim3 blk(PXQ4MMQ_WS, nw);
                if (&F == &FMT_P6) fn = nw == 8 ? pxq4mmq_pick_rag<pxq6_pol_p6, 8>()
                                                : pxq4mmq_pick_rag<pxq6_pol_p6, 4>();
                else               fn = nw == 8 ? pxq4mmq_pick_rag<pxq6_pol_p6hq, 8>()
                                                : pxq4mmq_pick_rag<pxq6_pol_p6hq, 4>();
                ck(fn != nullptr, "%s: no kernel", F.name);
                if (!fn) { return 1; }
                // the production picker must agree with the nw=4 default instantiation
                if (nw == 4) {
                    const int fmt = (&F == &FMT_P6) ? PXA_PXQ_FMT_P6 : PXA_PXQ_FMT_P6HQ;
                    ck(pxq4mmq_pick(fmt).fn == fn, "%s: picker disagrees with the nw=4 instantiation", F.name);
                }
                fn<<<dim3(panels, ntiles), blk>>>(dW, dAq, dAd, dC, dT, R, K);
                CUCK(cudaGetLastError());
                CUCK(cudaDeviceSynchronize());

                std::vector<float> C((size_t)ny*R);
                CUCK(cudaMemcpy(C.data(), dC, C.size()*sizeof(float), cudaMemcpyDeviceToHost));

                // scale for a relative metric: rms of the exact reference over the tile
                double rms = 0.0; for (double v : ref_spec) rms += v*v;
                rms = sqrt(rms/(double)ref_spec.size());

                double emax = 0.0, qmax = 0.0;
                bool finite = true;
                for (int t = 0; t < ny; ++t) for (int r = 0; r < R; ++r) {
                    const double got = (double)C[(size_t)t*R + r];
                    if (!std::isfinite(got)) finite = false;
                    emax = fmax(emax, fabs(got - ref_mmq[(size_t)t*R + r]));
                    qmax = fmax(qmax, fabs(ref_mmq[(size_t)t*R + r] - ref_spec[(size_t)t*R + r]));
                }
                ck(finite, "%s R%d K%d ny%d nw%d: non-finite output", F.name, R, K, ny, nw);
                // fp32 accumulation of ~K terms against a double emulation: 1e-4 relative is
                // several orders above float rounding and far below any real defect.
                ck(emax <= 1e-4*rms*sqrt((double)K),
                   "%s R%d K%d ny%d nw%d: kernel vs contract emulation max|d| %.6g (rms %.6g, bound %.6g)",
                   F.name, R, K, ny, nw, emax, rms, 1e-4*rms*sqrt((double)K));
                printf("  %-7s R%-4d K%-5d ny%-4d nw%d  kernel-vs-contract %.3e (%.2e rel) | "
                       "contract-vs-spec %.3e (%.2e rel)\n",
                       F.name, R, K, ny, nw, emax, emax/rms, qmax, qmax/rms);

                CUCK(cudaFree(dW)); CUCK(cudaFree(dA)); CUCK(cudaFree(dAq));
                CUCK(cudaFree(dAd)); CUCK(cudaFree(dC)); CUCK(cudaFree(dT));
            }
        }
    }

    printf("\n%d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
