// pxq4_v5_graph.cu -- the in-engine vehicle: one decode token's worth of PXQ4 linear work
// (240 modules per rank at TP=4), captured in a CUDA graph and replayed, exactly as vLLM does.
//
// WHY THIS AND NOT tok/s: the serving A/B on this box is contended (the owner's four cards run
// at 100% and share host CPU/PCIe), which puts several tok/s of noise on a ~10% effect. This
// harness has <0.1% run-to-run spread and isolates the PXQ4 linear time inside a real graph
// replay, which is the thing the kernel change can actually move.
//
// WHAT IT MODELS, and what the previous attempt got wrong:
//   * 240 DISTINCT weight tensors (~2.9 GB), because the engine streams a different tensor per
//     module -- reusing one buffer 64 times would leave it L2-warm and flatter every arm.
//   * ONE shared fp32 partials arena and ONE shared counter arena, because
//     mmv_partials_arena / mmv_counter_arena are single per-device tensors. Giving each module
//     its own partials (304 MB of working set) costs ~8 ms/token and slanders the split path.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include "pxq4_kernel.cu"

#define CK(e) do{cudaError_t r_=(e); if(r_!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s @%d: %s\n",#e,__LINE__,cudaGetErrorString(r_)); exit(1);} }while(0)

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
    const int b0=(kslabs*c)/nfix, b1=(kslabs*(c+1))/nfix, n=(b1-b0)*PXQ4_QK;
    __syncthreads();
    for (int idx=threadIdx.x; idx<n; idx+=blockDim.x)
        pxq4_xs[idx]=__half2float(xt[b0*PXQ4_QK+idx]);
    __syncthreads();
    float t=0.f;
    for (int kb=b0+kseg; kb<b1; kb+=PXQ4_MMV_KSEG)
        t += pxq4_dot32<VECX>(pan_slabs+(size_t)kb*pxq4_pol::SLAB,row,anch,
                              pxq4_xs+(size_t)(kb-b0)*PXQ4_QK,tab,sub);
    part[(((size_t)iy*gridDim.x+p)*nfix+c)*(PXQ4_MMV_KSEG*PXQ4_BM)+threadIdx.x]=t;
}

struct Mod { int panels, kslabs, nfix; uint8_t* w; __half* a; __half* x; __half* out; };

// TP=4 module inventory per rank, per decode token: 64 gate_up, 64 down, 48 qkvz, 64 o_proj.
struct Shape { const char* name; int N, K, count; };
static Shape SH[] = { {"gate_up",8704,5120,64}, {"down",5120,4352,64},
                      {"qkvz",4096,5120,48},    {"o_proj",5120,1536,64} };

enum Policy { P_V3, P_V5_GRID, P_V5_FUSED };

int main(int argc,char** argv){
    const int reps  = argc>1? atoi(argv[1]) : 50;
    CK(cudaSetDevice(0));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr,0));
    printf("device: %s SMs=%d L2=%.1f MB\n",pr.name,pr.multiProcessorCount,pr.l2CacheSize/1048576.0);

    std::vector<Mod> mods; size_t wtotal=0; size_t maxpart=0, maxctr=0;
    for (auto& s : SH) {
        const int panels=s.N/PXQ4_BM, kslabs=s.K/PXQ4_QK, nfix=pxq4_mmv_nfix(kslabs);
        const size_t wb=(size_t)panels*kslabs*PXQ4_SLAB_BYTES;
        std::vector<uint8_t> hw(wb);
        unsigned st=99u+s.K;
        for (size_t i=0;i<wb;++i){ st=st*1664525u+1013904223u; hw[i]=(uint8_t)(st>>17); }
        std::vector<__half> ha(panels*PXQ4_BM), hx(s.K);
        for (size_t i=0;i<ha.size();++i){ st=st*1664525u+1013904223u;
            ha[i]=__float2half(((float)(st%20001)-10000.f)/100000.f); }
        for (size_t i=0;i<hx.size();++i){ st=st*1664525u+1013904223u;
            hx[i]=__float2half(((float)(st%20001)-10000.f)/30000.f); }
        __half *da,*dx,*dout;
        CK(cudaMalloc(&da,ha.size()*2)); CK(cudaMemcpy(da,ha.data(),ha.size()*2,cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dx,hx.size()*2)); CK(cudaMemcpy(dx,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dout,(size_t)s.N*2));
        for (int i=0;i<s.count;++i) {
            uint8_t* dw; CK(cudaMalloc(&dw,wb));
            CK(cudaMemcpy(dw,hw.data(),wb,cudaMemcpyHostToDevice));
            CK(cudaMemset(dw,(int)(i*7+1),1));            // distinct tensors, as in the engine
            mods.push_back({panels,kslabs,nfix,dw,da,dx,dout}); wtotal+=wb;
        }
        maxpart=std::max(maxpart,(size_t)panels*nfix*(PXQ4_MMV_KSEG*PXQ4_BM));
        maxctr =std::max(maxctr ,(size_t)panels);
    }
    // interleave [gate_up, down, qkvz, o_proj] the way a layer does, so locality is realistic
    std::vector<Mod> chain; { size_t idx[4]={0,0,0,0}; size_t base[4]={0,64,128,176};
        for (int cyc=0;cyc<64;++cyc)
            for (int k=0;k<4;++k) if (idx[k]<(size_t)SH[k].count)
                chain.push_back(mods[base[k]+idx[k]++]);
    }
    printf("chain: %zu modules, %.2f GB of distinct weights, partials arena %.2f MB\n",
           chain.size(), wtotal/1e9, maxpart*4/1048576.0);

    float*    part; CK(cudaMalloc(&part,maxpart*4));      // ONE shared arena, as the engine has
    unsigned* ctr;  CK(cudaMalloc(&ctr ,maxctr*4)); CK(cudaMemset(ctr,0,maxctr*4));
    const int64_t kMaxMonoBlocks = 2*pr.multiProcessorCount;

    cudaStream_t s; CK(cudaStreamCreate(&s));
    auto emit=[&](Policy pol, cudaStream_t st){
        for (auto& m : chain) {
            const int K=m.kslabs*PXQ4_QK, R=m.panels*PXQ4_BM;
            const size_t sm=(size_t)pxq4_mmv_smem_bytes(m.kslabs);
            const bool split_v3 = m.nfix>1 &&
                (int64_t)m.panels*m.kslabs*PXQ4_SLAB_BYTES >= (8<<20);   // old byte rule
            const bool split_v5 = m.nfix>1 && (int64_t)m.panels*1 <= kMaxMonoBlocks;
            if (pol==P_V3) {
                if (split_v3) { k_part_v3<true><<<dim3(m.panels,m.nfix,1),256,sm,st>>>(m.w,m.a,m.x,part,K);
                                k_pxq4_mmv_reduce<<<dim3(m.panels,1,1),PXQ4_BM,0,st>>>(part,m.out,m.nfix,R); }
                else pxq4_launch_mmv_f16(m.w,m.a,m.x,m.out,1,m.panels,m.kslabs,true,st);
            } else if (pol==P_V5_GRID) {                  // grid swap only, old dispatch
                if (split_v3) pxq4_launch_mmv_split_f16(m.w,m.a,m.x,part,m.out,1,m.panels,m.kslabs,true,st);
                else pxq4_launch_mmv_f16(m.w,m.a,m.x,m.out,1,m.panels,m.kslabs,true,st);
            } else {                                      // v5: grid swap + fusion + dispatch
                if (split_v5) pxq4_launch_mmv_fused_f16(m.w,m.a,m.x,part,ctr,m.out,1,m.panels,m.kslabs,true,st);
                else pxq4_launch_mmv_f16(m.w,m.a,m.x,m.out,1,m.panels,m.kslabs,true,st);
            }
        }
    };
    const char* NAMES[3]={"v3 SHIPPING (ship grid + 8MB byte rule)",
                          "+ chunk-major grid only",
                          "+ grid + fusion + occupancy dispatch (v5)"};
    printf("\n%-42s %8s %10s %10s %8s\n","policy","nodes","eager ms","graph ms","g/e");
    for (int p=0;p<3;++p) {
        emit((Policy)p,s); CK(cudaStreamSynchronize(s));           // warm + arena settle
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
        emit((Policy)p,s);
        CK(cudaStreamEndCapture(s,&g));
        size_t nnodes=0; CK(cudaGraphGetNodes(g,nullptr,&nnodes));
        CK(cudaGraphInstantiate(&ge,g,nullptr,nullptr,0));
        cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        double eb=1e30, gb=1e30;
        for (int r=0;r<5;++r) {
            for(int i=0;i<3;++i) emit((Policy)p,s);
            CK(cudaStreamSynchronize(s));
            CK(cudaEventRecord(e0,s));
            for(int i=0;i<reps;++i) emit((Policy)p,s);
            CK(cudaEventRecord(e1,s)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); eb=std::min(eb,(double)ms/reps);
            for(int i=0;i<3;++i) CK(cudaGraphLaunch(ge,s));
            CK(cudaStreamSynchronize(s));
            CK(cudaEventRecord(e0,s));
            for(int i=0;i<reps;++i) CK(cudaGraphLaunch(ge,s));
            CK(cudaEventRecord(e1,s)); CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&ms,e0,e1)); gb=std::min(gb,(double)ms/reps);
        }
        printf("%-42s %8zu %10.4f %10.4f %8.3f   per-layer %.5f ms\n",
               NAMES[p],nnodes,eb,gb,gb/eb,gb/60.0);
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    }
    return 0;
}
