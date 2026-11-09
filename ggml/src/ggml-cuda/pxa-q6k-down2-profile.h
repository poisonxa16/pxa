#pragma once
#include <cstring>
#include <climits>
// Experimental shape-specific opt-in; expert IDs must be in range.
static bool pxa_q6k_down2_profile_matches(const char *profile,int cc,const ggml_tensor *w,
        const ggml_tensor *x,const ggml_tensor *ids,const ggml_tensor *y) {
    if(!profile || std::strcmp(profile,"1") || cc!=700 || !w || !x || !ids || !y)return false;
    const auto match=[](const ggml_tensor *t,ggml_type type,
            int64_t a,int64_t b,int64_t c,size_t n0,size_t n1,size_t n2,size_t n3){
        return t->type==type && t->ne[0]==a && t->ne[1]==b && t->ne[2]==c && t->ne[3]==1
            && t->nb[0]==n0 && t->nb[1]==n1 && t->nb[2]==n2 && t->nb[3]==n3;
    };
    return match(w,GGML_TYPE_Q6_K,512,2048,256,210,420,860160,220200960)
        && match(x,GGML_TYPE_F32,512,8,2,4,2048,16384,32768)
        && ids->type==GGML_TYPE_I32 && ids->ne[0]==8 && ids->ne[1]==2 && ids->ne[2]==1 && ids->ne[3]==1
        && ids->nb[0]==4 && ids->nb[1]>=32 && ids->nb[1]<=INT64_MAX/2 && ids->nb[1]%4==0
        && match(y,GGML_TYPE_F32,2048,8,2,4,8192,65536,131072);
}
