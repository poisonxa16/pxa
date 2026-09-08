// pxq4_v5_gpu.cu -- device gate + microbenchmark for the v5 combined mmv path.
//
// v5 = chunk-major grid order (explicit nfix/panels args)
//    + single-launch fused split (arrival-counter reduce)
//    + occupancy-based split/mono dispatch.
//
// Everything is ONE translation unit (it #includes pxq4_kernel.cu) so there is exactly one
// copy of pxq4_book_g / pxq4_sub16_g and the REAL shipping launchers are the things measured.
//
//   modes:  parity | stress | bench | graph
//
// The v3 reference is reconstructed IN THIS FILE as k_part_v3 (the pre-swap (panels, nfix, M)
// grid, nfix off gridDim.y) plus the shipping k_pxq4_mmv_reduce, and its output is asserted
// bit-identical to k_pxq4_mmv before any timing is believed.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include "pxq4_kernel.cu"

#define CK(e) do{cudaError_t r_=(e); if(r_!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s @%d: %s\n",#e,__LINE__,cudaGetErrorString(r_)); exit(1);} }while(0)

// ------------------------------------------------------------------ v3 reference part kernel
// Byte-for-byte the shipped v3 body: panel on .x, chunk on .y, nfix from gridDim.y.
template <bool VECX>
static __global__ void __launch_bounds__(256)
k_part_v3(const uint8_t * __restrict__ slabs, const __half * __restrict__ anchor,
          const __half * __restrict__ x, float * __restrict__ part, const int K) {
    const int p = blockIdx.x, c = blockIdx.y, iy = blockIdx.z;
    const int nfix = gridDim.y;
    PXQ4_EXTERN_SHARED float pxq4_xs[];
    __shared__ float tab[16]; __shared__ float sub[16];
    pxq4_pol::stage_tabs(tab, sub, threadIdx.x);
    const int row = threadIdx.x & 63, kseg = threadIdx.x >> 6, kslabs = K / PXQ4_QK;
    const uint8_t * pan_slabs = slabs + (size_t)p * kslabs * pxq4_pol::SLAB;
    const float anch = __half2float(anchor[(size_t)p * PXQ4_BM + row]);
    const __half * xt = x + (size_t)iy * K;
    const int b0 = (kslabs*c)/nfix, b1 = (kslabs*(c+1))/nfix, n = (b1-b0)*PXQ4_QK;
    __syncthreads();
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x)
        pxq4_xs[idx] = __half2float(xt[b0*PXQ4_QK + idx]);
    __syncthreads();
    float t = 0.f;
    for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG)
        t += pxq4_dot32<VECX>(pan_slabs + (size_t)kb*pxq4_pol::SLAB, row, anch,
                              pxq4_xs + (size_t)(kb-b0)*PXQ4_QK, tab, sub);
    part[(((size_t)iy*gridDim.x + p)*nfix + c)*(PXQ4_MMV_KSEG*PXQ4_BM) + threadIdx.x] = t;
}

static void launch_split_v3(const uint8_t* w, const __half* a, const __half* x, float* part,
                            __half* out, int M, int panels, int kslabs, cudaStream_t s) {
    const int K = kslabs*PXQ4_QK, nfix = pxq4_mmv_nfix(kslabs), R = panels*PXQ4_BM;
    const size_t smem = (size_t)pxq4_mmv_smem_bytes(kslabs);
    k_part_v3<true><<<dim3(panels,nfix,M),256,smem,s>>>(w,a,x,part,K);
    k_pxq4_mmv_reduce<<<dim3(panels,M,1),PXQ4_BM,0,s>>>(part,out,nfix,R);
}

template <bool VECX>
static __device__ __forceinline__ void pxq4_dot32_rb2(const uint8_t * __restrict__ slab,
                                                      int row, float anchA, float anchB,
                                                      const float * __restrict__ xk,
                                                      const float * __restrict__ tab,
                                                      const float * __restrict__ sub,
                                                      float * dA, float * dB) {
    float effA[PXQ4_NEFF];
    float effB[PXQ4_NEFF];
    pxq4_pol::row_effs(slab, row,      anchA, sub, effA);
    pxq4_pol::row_effs(slab, row + 32, anchB, sub, effB);

    uint32_t qa[4];
    uint32_t qb[4];
    pxq4_ldcodes(slab + pxq4_pol::CODE_OFF + row        * pxq4_pol::CODE_BYTES, qa);
    pxq4_ldcodes(slab + pxq4_pol::CODE_OFF + (row + 32) * pxq4_pol::CODE_BYTES, qb);

    float tA[PXQ4_NEFF];
    float tB[PXQ4_NEFF];
#pragma unroll
    for (int i = 0; i < PXQ4_NEFF; ++i) { tA[i] = 0.f; tB[i] = 0.f; }

    if (VECX) {
#pragma unroll
        for (int b = 0; b < 16; b += 2) {
            const float4 xv = *(const float4 *)&xk[2 * b];
            const float2 pa0 = pxq4_pol::pair(qa, b,     tab);
            const float2 pa1 = pxq4_pol::pair(qa, b + 1, tab);
            tA[(b * PXQ4_NEFF) >> 4]       = pxq4_acc2(tA[(b * PXQ4_NEFF) >> 4],       pa0.x, xv.x, pa0.y, xv.y);
            tA[((b + 1) * PXQ4_NEFF) >> 4] = pxq4_acc2(tA[((b + 1) * PXQ4_NEFF) >> 4], pa1.x, xv.z, pa1.y, xv.w);
            const float2 pb0 = pxq4_pol::pair(qb, b,     tab);
            const float2 pb1 = pxq4_pol::pair(qb, b + 1, tab);
            tB[(b * PXQ4_NEFF) >> 4]       = pxq4_acc2(tB[(b * PXQ4_NEFF) >> 4],       pb0.x, xv.x, pb0.y, xv.y);
            tB[((b + 1) * PXQ4_NEFF) >> 4] = pxq4_acc2(tB[((b + 1) * PXQ4_NEFF) >> 4], pb1.x, xv.z, pb1.y, xv.w);
        }
    } else {
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float2 pa = pxq4_pol::pair(qa, b, tab);
            tA[(b * PXQ4_NEFF) >> 4] = pxq4_acc2(tA[(b * PXQ4_NEFF) >> 4], pa.x, xk[2 * b], pa.y, xk[2 * b + 1]);
            const float2 pb = pxq4_pol::pair(qb, b, tab);
            tB[(b * PXQ4_NEFF) >> 4] = pxq4_acc2(tB[(b * PXQ4_NEFF) >> 4], pb.x, xk[2 * b], pb.y, xk[2 * b + 1]);
        }
    }
    *dA = effA[0] * tA[0] + effA[1] * tA[1];
    *dB = effB[0] * tB[0] + effB[1] * tB[1];
}

// A 4-byte-aligned pair of fp16 activations. Used ONLY by the rb2 staging loop, to issue one
// 32-bit load where the scalar loop issues two 16-bit ones. __half2float of each member is the
// same exact widening the scalar loop performs, so the staged floats are bit-identical.
// Spelled as a local struct rather than CUDA's __half2 so the host simulator (which has no
// __half2) compiles the same source.
struct __align__(4) pxq4_h2 { __half x, y; };


// ------------------------------------------------------- EXPERIMENT: fused + rb2 register block
// The rb2 treatment (two output rows per lane, half2 activation staging) applied to the FUSED
// kernel rather than to the two-launch part kernel. 128 threads, r = tid&31, kseg = tid>>5.
// part[] layout is unchanged (kseg*64 + row), so the arrival barrier and the fold are the same
// adds in the same order -- bit-exact if it works at all. The open question is REGISTERS: the
// shipped fused kernel is 32 regs / 0 spill, and rb2 cost the standalone part kernel 40; the
// monolithic rb2 experiment spilled once a reduce tail was added to it.
template <bool VECX>
static __global__ void __launch_bounds__(PXQ4_MMV_KSEG * 32, 12)
k_pxq4_mmv_fused_rb2(const uint8_t * __restrict__ slabs, const __half * __restrict__ anchor,
                     const __half * __restrict__ x, float * __restrict__ part,
                     unsigned * __restrict__ ctr, __half * __restrict__ out,
                     const int R, const int K, const int nfix, const int panels) {
    const int c = blockIdx.x, p = blockIdx.y, iy = blockIdx.z;
    PXQ4_EXTERN_SHARED float pxq4_xs[];
    __shared__ float tab[16]; __shared__ float sub[16];
    __shared__ float red[PXQ4_MMV_KSEG * PXQ4_BM];
    __shared__ int   last;
    pxq4_pol::stage_tabs(tab, sub, threadIdx.x);
    const int r = threadIdx.x & 31, kseg = threadIdx.x >> 5, kslabs = K / PXQ4_QK;
    const uint8_t * pan_slabs = slabs + (size_t)p * kslabs * pxq4_pol::SLAB;
    const float anchA = __half2float(anchor[(size_t)p * PXQ4_BM + r]);
    const float anchB = __half2float(anchor[(size_t)p * PXQ4_BM + r + 32]);
    const __half * xt = x + (size_t)iy * K;
    const int b0 = (kslabs*c)/nfix, b1 = (kslabs*(c+1))/nfix, n = (b1-b0)*PXQ4_QK;
    __syncthreads();
    { const pxq4_h2 * src = (const pxq4_h2 *)(xt + (size_t)b0 * PXQ4_QK);
      const int nh = n >> 1;
      for (int i = threadIdx.x; i < nh; i += PXQ4_MMV_KSEG*32) {
          const pxq4_h2 h = src[i];
          pxq4_xs[2*i] = __half2float(h.x); pxq4_xs[2*i+1] = __half2float(h.y); } }
    __syncthreads();
    float tA=0.f, tB=0.f;
    for (int kb=b0+kseg; kb<b1; kb+=PXQ4_MMV_KSEG) {
        float dA,dB;
        pxq4_dot32_rb2<VECX>(pan_slabs+(size_t)kb*pxq4_pol::SLAB, r, anchA, anchB,
                             pxq4_xs+(size_t)(kb-b0)*PXQ4_QK, tab, sub, &dA, &dB);
        tA+=dA; tB+=dB;
    }
    const size_t tile = (size_t)(PXQ4_MMV_KSEG*PXQ4_BM);
    float * const pbase = part + (((size_t)iy*panels + p)*nfix)*tile;
    pbase[(size_t)c*tile + kseg*PXQ4_BM + r]      = tA;
    pbase[(size_t)c*tile + kseg*PXQ4_BM + r + 32] = tB;
    __syncthreads();                                   // LOAD-BEARING, as in k_pxq4_mmv_fused
    if (threadIdx.x == 0) {
        const unsigned old = pxq4_arrive_release(&ctr[(size_t)iy*panels + p]);
        last = (old == (unsigned)(nfix-1));
        if (last) { ctr[(size_t)iy*panels + p] = 0u; pxq4_fence_acq_rel(); }
    }
    __syncthreads();
    if (!last) return;
    // each lane folds TWO of the 256 tile slots; per-element order over cc is unchanged
    float s0=0.f, s1=0.f;
    for (int cc=0; cc<nfix; ++cc) {
        s0 += pxq4_ld_part(&pbase[(size_t)cc*tile + threadIdx.x]);
        s1 += pxq4_ld_part(&pbase[(size_t)cc*tile + threadIdx.x + PXQ4_MMV_KSEG*32]);
    }
    red[threadIdx.x] = s0; red[threadIdx.x + PXQ4_MMV_KSEG*32] = s1;
    __syncthreads();
    if (threadIdx.x < PXQ4_BM) {
        const int row = threadIdx.x; float u = 0.f;
#pragma unroll
        for (int s=0; s<PXQ4_MMV_KSEG; ++s) u += red[s*PXQ4_BM + row];
        out[(size_t)iy*R + p*PXQ4_BM + row] = __float2half_rn(u);
    }
}
static void launch_fused_rb2(const uint8_t* w,const __half* a,const __half* x,float* part,
                             unsigned* ctr,__half* out,int M,int panels,int kslabs,cudaStream_t s){
    const int K=kslabs*PXQ4_QK, nfix=pxq4_mmv_nfix(kslabs), R=panels*PXQ4_BM;
    const size_t smem=(size_t)pxq4_mmv_smem_bytes(kslabs);
    k_pxq4_mmv_fused_rb2<true><<<dim3(nfix,panels,M),PXQ4_MMV_KSEG*32,smem,s>>>(
        w,a,x,part,ctr,out,R,K,nfix,panels);
}

// ------------------------------------------------------------------------------- shapes
struct Case { const char* name; int N, K; };
// Real artifact shapes. TP4 is the shipping layout.
static const Case CASES[] = {
    {"tp4_gate_up", 8704, 5120}, {"tp4_down", 5120, 4352},
    {"tp4_qkvz",    4096, 5120}, {"tp4_o_proj", 5120, 1536},
    {"tp2_gate_up",17408, 5120}, {"tp2_down", 5120, 8704},
    {"tp2_qkvz",    8192, 5120}, {"tp2_o_proj", 5120, 3072},
    {"tp1_gate_up",34816, 5120}, {"tp1_down", 5120,17408},
    {"tp1_qkvz",   16384, 5120}, {"tp1_o_proj", 5120, 6144},
};
static const int NCASE = (int)(sizeof(CASES)/sizeof(CASES[0]));

struct Buf {
    int panels, kslabs, nfix, M, R, K;
    uint8_t* w=nullptr; __half* a=nullptr; __half* x=nullptr;
    float* part=nullptr; unsigned* ctr=nullptr;
    __half *o_mono=nullptr, *o_v3=nullptr, *o_v5s=nullptr, *o_fused=nullptr, *o_rb2=nullptr;
    float* part_ref=nullptr;
    size_t wbytes=0, partw=0;
};

static void fill_host(std::vector<uint8_t>& w, std::vector<__half>& a, std::vector<__half>& x,
                      int panels, int kslabs, int M, unsigned seed) {
    unsigned s = seed;
    auto rnd = [&]{ s = s*1664525u + 1013904223u; return s; };
    for (size_t i=0;i<w.size();++i) w[i] = (uint8_t)(rnd()>>17);
    // signed anchors, all non-zero except two deliberate specials
    for (size_t i=0;i<a.size();++i) {
        float v = ((float)(rnd()%20001) - 10000.f)/100000.f;
        if (v==0.f) v = 3e-3f;
        a[i] = __float2half(v);
    }
    if (a.size()>2) { a[0]=__float2half(0.f); a[1]=__float2half(6e-8f); }
    for (size_t i=0;i<x.size();++i)
        x[i] = __float2half(((float)(rnd()%20001)-10000.f)/30000.f);
    (void)M;
}

static void alloc_case(Buf& b, const Case& c, int Mmax) {
    b.panels = c.N/PXQ4_BM; b.kslabs = c.K/PXQ4_QK; b.K = c.K; b.R = c.N;
    b.nfix = pxq4_mmv_nfix(b.kslabs); b.M = Mmax;
    b.wbytes = (size_t)b.panels*b.kslabs*PXQ4_SLAB_BYTES;
    b.partw  = (size_t)Mmax*b.panels*b.nfix*(PXQ4_MMV_KSEG*PXQ4_BM);
    std::vector<uint8_t> hw(b.wbytes);
    std::vector<__half>  ha((size_t)b.panels*PXQ4_BM), hx((size_t)Mmax*c.K);
    fill_host(hw,ha,hx,b.panels,b.kslabs,Mmax, 12345u + (unsigned)c.K*7u + (unsigned)c.N);
    CK(cudaMalloc(&b.w,b.wbytes));  CK(cudaMemcpy(b.w,hw.data(),b.wbytes,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.a,ha.size()*2)); CK(cudaMemcpy(b.a,ha.data(),ha.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.x,hx.size()*2)); CK(cudaMemcpy(b.x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.part,b.partw*4)); CK(cudaMalloc(&b.part_ref,b.partw*4));
    CK(cudaMalloc(&b.ctr,(size_t)Mmax*b.panels*4)); CK(cudaMemset(b.ctr,0,(size_t)Mmax*b.panels*4));
    size_t ob = (size_t)Mmax*c.N*2;
    CK(cudaMalloc(&b.o_mono,ob)); CK(cudaMalloc(&b.o_v3,ob));
    CK(cudaMalloc(&b.o_v5s,ob));  CK(cudaMalloc(&b.o_fused,ob)); CK(cudaMalloc(&b.o_rb2,ob));
}
static void free_case(Buf& b){ cudaFree(b.w);cudaFree(b.a);cudaFree(b.x);cudaFree(b.part);
    cudaFree(b.part_ref);cudaFree(b.ctr);cudaFree(b.o_mono);cudaFree(b.o_v3);
    cudaFree(b.o_v5s);cudaFree(b.o_fused);cudaFree(b.o_rb2); }

// ------------------------------------------------------------------------------- parity
static int cmp16(const __half* da,const __half* db,size_t n,const char* what,const char* tag){
    std::vector<uint16_t> A(n),B(n);
    CK(cudaMemcpy(A.data(),da,n*2,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(B.data(),db,n*2,cudaMemcpyDeviceToHost));
    size_t d=0; for(size_t i=0;i<n;++i) if(A[i]!=B[i]) ++d;
    if(d) printf("    MISMATCH %s %s: %zu/%zu fp16 words\n",tag,what,d,n);
    return (int)d;
}
static int cmp32(const float* da,const float* db,size_t n,const char* what,const char* tag){
    std::vector<uint32_t> A(n),B(n);
    CK(cudaMemcpy(A.data(),da,n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(B.data(),db,n*4,cudaMemcpyDeviceToHost));
    size_t d=0; for(size_t i=0;i<n;++i) if(A[i]!=B[i]) ++d;
    if(d) printf("    MISMATCH %s %s: %zu/%zu fp32 words\n",tag,what,d,n);
    return (int)d;
}
static int nonzero_ctr(const unsigned* d,size_t n){
    std::vector<unsigned> h(n); CK(cudaMemcpy(h.data(),d,n*4,cudaMemcpyDeviceToHost));
    int c=0; for(size_t i=0;i<n;++i) if(h[i]) ++c; return c;
}
static size_t count_poison16(const __half* d,size_t n,uint16_t poison){
    std::vector<uint16_t> h(n); CK(cudaMemcpy(h.data(),d,n*2,cudaMemcpyDeviceToHost));
    size_t c=0; for(size_t i=0;i<n;++i) if(h[i]==poison) ++c; return c;
}

static int run_parity() {
    const int Ms[] = {1,2,3,4,8};
    int combos=0, fails=0;
    for (int ci=0; ci<NCASE; ++ci) {
        Buf b; alloc_case(b, CASES[ci], 8);
        for (int mi=0; mi<5; ++mi) {
            const int M = Ms[mi];
            for (int vx=1; vx>=0; --vx) {
                const size_t on = (size_t)M*b.R;
                const size_t pn = (size_t)M*b.panels*b.nfix*(PXQ4_MMV_KSEG*PXQ4_BM);
                // poison EVERYTHING before EVERY call: a kernel that writes nothing must fail
                CK(cudaMemset(b.o_mono,0xA5,on*2)); CK(cudaMemset(b.o_v3,0x5A,on*2));
                CK(cudaMemset(b.o_v5s,0xC3,on*2));  CK(cudaMemset(b.o_fused,0x3C,on*2));
                CK(cudaMemset(b.part,0xFF,pn*4));   CK(cudaMemset(b.part_ref,0xFF,pn*4));

                pxq4_launch_mmv_f16(b.w,b.a,b.x,b.o_mono,M,b.panels,b.kslabs,vx!=0,0);
                CK(cudaDeviceSynchronize());
                launch_split_v3(b.w,b.a,b.x,b.part_ref,b.o_v3,M,b.panels,b.kslabs,0);
                CK(cudaDeviceSynchronize());
                CK(cudaMemset(b.part,0xFF,pn*4));
                pxq4_launch_mmv_split_f16(b.w,b.a,b.x,b.part,b.o_v5s,M,b.panels,b.kslabs,vx!=0,0);
                CK(cudaDeviceSynchronize());
                int d_part = cmp32(b.part,b.part_ref,pn,"v5-part vs v3-part",CASES[ci].name);
                CK(cudaMemset(b.part,0xFF,pn*4));
                pxq4_launch_mmv_fused_f16(b.w,b.a,b.x,b.part,b.ctr,b.o_fused,M,b.panels,
                                          b.kslabs,vx!=0,0);
                CK(cudaDeviceSynchronize());
                int d_pf = cmp32(b.part,b.part_ref,pn,"fused-part vs v3-part",CASES[ci].name);

                CK(cudaMemset(b.o_rb2,0x7E,on*2)); CK(cudaMemset(b.part,0xFF,pn*4));
                launch_fused_rb2(b.w,b.a,b.x,b.part,b.ctr,b.o_rb2,M,b.panels,b.kslabs,0);
                CK(cudaDeviceSynchronize());
                int d_rb2 = cmp16(b.o_rb2,b.o_mono,on,"fused_rb2 vs mono",CASES[ci].name);
                int d1 = cmp16(b.o_fused,b.o_mono,on,"fused vs mono",CASES[ci].name);
                int d2 = cmp16(b.o_fused,b.o_v5s ,on,"fused vs v5-split",CASES[ci].name);
                int d3 = cmp16(b.o_v5s  ,b.o_v3  ,on,"v5-split vs v3-split",CASES[ci].name);
                int cz = nonzero_ctr(b.ctr,(size_t)M*b.panels);
                // mono output must be genuinely written and not all-poison
                size_t poi = count_poison16(b.o_mono,on,0xA5A5);
                int bad = d1|d2|d3|d_part|d_pf|d_rb2|cz|(poi==on);
                combos++;
                if (bad) { fails++;
                    printf("  FAIL %-12s M=%d vecx=%d  fm=%d fs=%d sv3=%d part=%d fpart=%d rb2=%d ctr=%d poison=%zu\n",
                           CASES[ci].name,M,vx,d1,d2,d3,d_part,d_pf,d_rb2,cz,poi); }
                else printf("  ok   %-12s M=%d vecx=%d  out=%zu part=%zu  fused==v5split==v3split==mono\n",
                            CASES[ci].name,M,vx,on,pn);
            }
        }
        free_case(b);
    }
    printf("\npxq4 v5 device parity: %d/%d combos bit-exact -> %s\n",
           combos-fails, combos, fails? "FAIL":"PASS");
    return fails?1:0;
}

// ------------------------------------------------------------------------------- stress
static int run_stress(int launches) {
    int bad=0;
    const int idx[4]={0,1,2,3};
    for (int k=0;k<4;++k) {
        Buf b; alloc_case(b,CASES[idx[k]],8);
        for (int M : {1,8}) {
            const size_t on=(size_t)M*b.R, pn=(size_t)M*b.panels*b.nfix*(PXQ4_MMV_KSEG*PXQ4_BM);
            CK(cudaMemset(b.part_ref,0xFF,pn*4));
            pxq4_launch_mmv_f16(b.w,b.a,b.x,b.o_mono,M,b.panels,b.kslabs,true,0);
            CK(cudaDeviceSynchronize());
            int mism=0, ctrz=0;
            for (int i=0;i<launches;++i) {
                CK(cudaMemset(b.part,0xFF,pn*4));
                CK(cudaMemset(b.o_fused,0x3C,on*2));
                pxq4_launch_mmv_fused_f16(b.w,b.a,b.x,b.part,b.ctr,b.o_fused,M,b.panels,
                                          b.kslabs,true,0);
                CK(cudaDeviceSynchronize());
                if (cmp16(b.o_fused,b.o_mono,on,"stress","")) ++mism;
                if (nonzero_ctr(b.ctr,(size_t)M*b.panels)) ++ctrz;
            }
            printf("  STRESS %-12s M=%d  %d launches: %d mismatching, %d ctr_nonzero -> %s\n",
                   CASES[idx[k]].name,M,launches,mism,ctrz,(mism||ctrz)?"FAIL":"PASS");
            if (mism||ctrz) ++bad;
        }
        free_case(b);
    }
    printf("\nSTRESS: %s\n", bad? "FAIL":"PASS");
    return bad?1:0;
}

// ------------------------------------------------------------------------------- bench
struct Rot { std::vector<uint8_t*> w; };
static double bench_ms(void(*fn)(void*,int), void* ctx, int iters) {
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i=0;i<20;++i) fn(ctx,i);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int i=0;i<iters;++i) fn(ctx,i);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    return ms/iters;
}
struct BC { Buf* b; std::vector<uint8_t*>* rot; int M; int mode; };
static void bc_call(void* p,int i){
    BC* c=(BC*)p; Buf* b=c->b;
    uint8_t* w = (*c->rot)[i % c->rot->size()];
    switch(c->mode){
      case 0: pxq4_launch_mmv_f16(w,b->a,b->x,b->o_mono,c->M,b->panels,b->kslabs,true,0); break;
      case 1: launch_split_v3(w,b->a,b->x,b->part,b->o_v3,c->M,b->panels,b->kslabs,0); break;
      case 2: pxq4_launch_mmv_split_f16(w,b->a,b->x,b->part,b->o_v5s,c->M,b->panels,b->kslabs,true,0); break;
      case 3: pxq4_launch_mmv_fused_f16(w,b->a,b->x,b->part,b->ctr,b->o_fused,c->M,b->panels,b->kslabs,true,0); break;
      case 4: launch_fused_rb2(w,b->a,b->x,b->part,b->ctr,b->o_rb2,c->M,b->panels,b->kslabs,0); break;
    }
}
static int run_bench(int M, int iters, int reps, int wsMB) {
    printf("# M=%d iters=%d reps=%d L2-defeating working set >= %d MB\n", M, iters, reps, wsMB);
    printf("%-13s %7s %5s %4s | %-16s | %-16s | %-16s | %-16s | %s\n",
           "shape","slabMB","pan","nfix","mono ms  GB/s","v3split ms GB/s",
           "v5fused ms GB/s","v5fused+rb2 GB/s","fused/v3  rb2/fused  fused/mono");
    for (int ci=0; ci<NCASE; ++ci) {
        Buf b; alloc_case(b,CASES[ci],std::max(M,8));
        const double mb = (double)b.wbytes/1048576.0;
        int R = (int)((wsMB + mb - 1)/mb); if (R<1) R=1; if (R>24) R=24;
        std::vector<uint8_t*> rot; rot.push_back(b.w);
        for (int i=1;i<R;++i){ uint8_t* p; if(cudaMalloc(&p,b.wbytes)!=cudaSuccess) break;
            CK(cudaMemcpy(p,b.w,b.wbytes,cudaMemcpyDeviceToDevice));
            // perturb one byte so no two copies are the same allocation-elidable object
            CK(cudaMemset(p, (int)(0x11*i), 1));
            rot.push_back(p); }
        double t[5]; for (int m=0;m<5;++m){ double best=1e30;
            for (int r=0;r<reps;++r){ BC c{&b,&rot,M,m}; double v=bench_ms(bc_call,&c,iters);
                best = std::min(best,v);} t[m]=best; }
        auto gbs=[&](double ms){ return (double)b.wbytes/ (ms*1e-3) /1e9; };
        printf("%-13s %7.2f %5d %4d | %8.5f %6.0f | %8.5f %6.0f | %8.5f %6.0f | %8.5f %6.0f | %5.3f %5.3f %5.3f  (R=%d)\n",
               CASES[ci].name, mb, b.panels, b.nfix,
               t[0],gbs(t[0]), t[1],gbs(t[1]), t[3],gbs(t[3]), t[4],gbs(t[4]),
               t[1]/t[3], t[3]/t[4], t[0]/t[3], (int)rot.size());
        for (size_t i=1;i<rot.size();++i) cudaFree(rot[i]);
        free_case(b);
    }
    return 0;
}

int main(int argc,char** argv){
    const std::string mode = argc>1? argv[1] : "parity";
    int dev=0; CK(cudaSetDevice(dev));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr,dev));
    printf("device: %s  SMs=%d  L2=%.1f MB\n",pr.name,pr.multiProcessorCount,pr.l2CacheSize/1048576.0);
    if (mode=="parity") return run_parity();
    if (mode=="stress") return run_stress(argc>2? atoi(argv[2]) : 200);
    if (mode=="bench")  return run_bench(argc>2?atoi(argv[2]):1, argc>3?atoi(argv[3]):300,
                                         argc>4?atoi(argv[4]):5, argc>5?atoi(argv[5]):48);
    fprintf(stderr,"unknown mode %s\n",mode.c_str()); return 2;
}
