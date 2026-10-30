#pragma once
#include "mmvq-verify2.cuh"
#include "mmvq-templates.cuh"
#include "pxa-q6k-down2-eligibility.h"

// Same per-lane k-block walk, dot routine and reduction as the pinned Q6_K GEMV.
// Only routing/indexing is specialized. No expanded weights or workspace.
template<bool Pair>
static __device__ __forceinline__ void pxa_q6k_down2_row(const pxa_mv2_args &a,
        int row,int slot,int peer,int expert,int lane) {
    const char *x=(const char*)a.vx_u+uint64_t(expert)*a.nb02;
    const auto*y0=(const block_q8_1*)(a.vy+(slot>>3)*a.y_st+(slot&7)*a.y_sk);
    const auto*y1=Pair?(const block_q8_1*)(a.vy+a.y_st+peer*a.y_sk):nullptr;
    constexpr int qi=ggml_cuda_type_traits<GGML_TYPE_Q6_K>::qi;
    constexpr int vdr=get_vdr_mmvq(GGML_TYPE_Q6_K);
    float t0=0.0f,t1=0.0f;
    // Keep the original order of the two addends for each output.
    for(int kb=lane/(qi/vdr);kb<2;kb+=vdr*32/qi){
        const int kqs=vdr*(lane%(qi/vdr));
        t0+=vec_dot_q6_K_q8_1(x,y0+kb*8,row*2+kb,kqs);
        if constexpr(Pair)t1+=vec_dot_q6_K_q8_1(x,y1+kb*8,row*2+kb,kqs);
    }
    t0=warp_reduce_sum(t0);
    if constexpr(Pair)t1=warp_reduce_sum(t1);
    if(lane==0){
        ((float*)(a.dst+(slot>>3)*a.d_st+(slot&7)*a.d_sk))[row]=t0+0.0f;
        if constexpr(Pair)((float*)(a.dst+a.d_st+peer*a.d_sk))[row]=t1+0.0f;
    }
}

template<int Warps>
__launch_bounds__(32*Warps,1)
static __global__ void pxa_q6k_down2_kernel(const pxa_mv2_args a){
    const int lane=threadIdx.x&31,row=blockIdx.x*Warps+(threadIdx.x>>5),slot=blockIdx.y;
    int id=-1;
    if(lane<16)id=*(const int*)(a.ids+(lane>>3)*a.ids_nb1+(lane&7)*4);
    const int expert=__shfl_sync(0xffffffff,id,slot);
    if(expert<0||expert>=256)return;
    const unsigned matches=__ballot_sync(0xffffffff,lane<16&&id==expert);
    // Pair only one-to-one matches. Repeated IDs compute independently,
    // preserving every slot without multiple writers to a paired output.
    const bool pair = __popc(matches & 255) == 1 && __popc(matches >> 8) == 1;
    if(slot>=8 && pair)return;
    const int peer=slot<8 && pair?__ffs(matches>>8)-1:-1;
    if(peer>=0)pxa_q6k_down2_row<true>(a,row,slot,peer,expert,lane);
    else pxa_q6k_down2_row<false>(a,row,slot,0,expert,lane);
}

static bool pxa_q6k_down2_launch(ggml_type type,const pxa_mv2_args &a,
                              cudaStream_t stream,int warps=1){
    if(!pxa_q6k_down2_eligible(type,a)||(warps!=1&&warps!=4))return false;
    if(warps==1)pxa_q6k_down2_kernel<1><<<dim3(2048,16),32,0,stream>>>(a);
    else pxa_q6k_down2_kernel<4><<<dim3(512,16),128,0,stream>>>(a);
    return true;
}
