#pragma once
#include <cstdint>
#include <climits>
// Host-only contract: no CUDA calls on nullable/unsupported shapes.
// Requires eight expert IDs in [0,256) per token; repeated IDs run independently.
template<class Args> bool pxa_q6k_down2_eligible(ggml_type type,const Args &a) {
    const auto aligned=[](const void *p,uintptr_t n){return p && uintptr_t(p)%n==0;};
    return type==GGML_TYPE_Q6_K && !a.vx_g && !a.bias
        && aligned(a.vx_u,2) && aligned(a.vy,4) && aligned(a.dst,4) && aligned(a.ids,4)
        && a.ncols_x==512 && a.nrows_x==2048 && a.nrows_y==512 && a.ny==2 && a.n_ids==8
        && a.ids_nb0==4 && a.ids_nb1>=32 && a.ids_nb1<=INT64_MAX/2 && a.ids_nb1%4==0
        && a.y_sk>=576 && a.y_sk<=INT64_MAX/8 && a.y_sk%4==0
        && a.y_st>=8*a.y_sk && a.y_st<=INT64_MAX/2 && a.y_st%4==0
        && a.d_sk>=8192 && a.d_sk<=INT64_MAX/8 && a.d_sk%4==0
        && a.d_st>=8*a.d_sk && a.d_st<=INT64_MAX/2 && a.d_st%4==0
        && a.nb02>=860160 && a.nb02<=UINT64_MAX/256 && a.nb02%2==0;
}
