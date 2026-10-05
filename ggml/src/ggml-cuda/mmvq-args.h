#pragma once

#include "common.cuh"

struct mmvq_args {
    const void * vx_u;
    const void * vx_g;
    const void * bias_u;
    const void * bias_g;
    const void * vy;
    float      * dst;
    const char * ids_data;
    const int    ncols_x;
    const int    nrows_x;
    const int    nrows_y;
    const int    ncols_y;
    const int    nrows_dst;
    const int    ne2;
    const uint64_t nb02;
    const uint64_t nb12;
    const uint64_t nb2;
    const uint64_t ids_nb0;
    const uint64_t bias_nb1;
    ggml_unary_op  unary_op;
    float          limit;
};


// PXA_MMVQ_GROUP: several ny = 1 GEMVs of one type on the same q8_1 activation in one launch (mmvq-templates.cuh)
#define MMVQ_GROUP_MAX 4
struct mmvq_group_args {
    const void * vx[MMVQ_GROUP_MAX];
    float      * dst[MMVQ_GROUP_MAX];
    int          nrows[MMVQ_GROUP_MAX];
    int          blk_end[MMVQ_GROUP_MAX];   // cumulative row-block counts, filled by the launcher
    int          n;
    const void * vy;
    int          ncols_x;
    int          nrows_y;                   // padded activation row length
};
