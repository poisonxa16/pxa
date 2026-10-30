#include "pxa-q6k-down2.cuh"
#include "pxa-q6k-down2-profile.h"
#include "pxa-q6k-down2-kernel.cuh"
#include <cstdlib>
#include <cstdio>
#include <atomic>

bool pxa_q6k_down2_try(int cc,const ggml_tensor *w,const ggml_tensor *x,
        const ggml_tensor *ids,const ggml_tensor *y,const pxa_mv2_args &a,cudaStream_t stream) {
    static const char *profile=std::getenv("PXA_Q6K_DOWN2");
    if(!pxa_q6k_down2_profile_matches(profile,cc,w,x,ids,y) || !pxa_q6k_down2_launch(w->type,a,stream,4))return false;
    static std::atomic<bool> logged{false};
    if(!logged.exchange(true))std::fprintf(stderr,"PXA_Q6K_DOWN2_V1 profile=1 warps=4\n");
    return true;
}
