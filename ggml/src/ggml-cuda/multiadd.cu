#include "multiadd.cuh"

static __global__ void multi_add_f32(int nused, int64_t ne0, int64_t ne1, int64_t nb1, int64_t nb01, const char * src0, char * dst) {
    const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;
    int64_t k = ne0*ne1;
    if (i >= k) {
        return;
    }
    int i1 = i / ne0;
    int i0 = i % ne0;
    float * result = (float *)(dst + i1*nb1);
    const float * s = (const float *)(src0 + i1*nb01) + i0;
    if (nused == 1) {
        result[i0] = s[0];
    } else {
        float sum = s[0] + s[ne0];
        for (int j = 2; j < nused; ++j) sum += s[j*ne0];
        result[i0] = sum;
    }
}

static void multi_add_f32_cuda(int nused, int64_t ne0, int64_t ne1, int64_t nb1, int64_t nb01, const char * src0, char * dst, cudaStream_t stream) {
    int64_t k = ne0 * ne1;
    const int num_blocks = (k + CUDA_MULTI_ADD_BLOCK_SIZE - 1) / CUDA_MULTI_ADD_BLOCK_SIZE;
    multi_add_f32<<<num_blocks, CUDA_MULTI_ADD_BLOCK_SIZE, 0, stream>>>(nused, ne0, ne1, nb1, nb01, src0, dst);
}

void ggml_cuda_op_multi_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[2] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    int nused = dst->op_params[0];
    GGML_ASSERT(nused >= 1);
    const char * src0 = (const char *)dst->src[0]->data;
    cudaStream_t stream = ctx.stream();
    multi_add_f32_cuda(nused, dst->ne[0], dst->ne[1], dst->nb[1], dst->src[0]->nb[1], src0, (char *)dst->data, stream);
}


static __global__ void mul_multi_add_f32(int nused, int64_t ne0, int64_t ne1, int64_t nb1, int64_t nb01, int64_t nb02, int64_t nb11, int64_t nb12, const char * src0, const char * src1, char * dst,
        const char * resid, int64_t nbr1) {
    const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;
    int64_t k = ne0*ne1;
    if (i >= k) {
        return;
    }
    int i1 = i / ne0;
    int i0 = i % ne0;
    float * result = (float *)(dst + i1*nb1);

    auto c0 = src0 + i1*nb02;
    auto c1 = src1 + i1*nb12;

    float sum = 0;
    for (int j = 0; j < nused; ++j) {
        auto x0 = (const float *)c0;
        auto x1 = (const float *)c1;
        sum += x0[i0] * x1[0];
        c0 += nb01;
        c1 += nb11;
    }
    // G2-F4: residual folded in LAST (experts first, one final add) => bit-identical to the
    // standalone k_add_same in either ADD operand order (fp32 add is bitwise commutative).
    if (resid) {
        sum += ((const float *)(resid + i1*nbr1))[i0];
    }
    result[i0] = sum;
}

static void mul_multi_add_f32_cuda(int nused, int64_t ne0, int64_t ne1, int64_t nb1, int64_t nb01, int64_t nb02, int64_t nb11, int64_t nb12,
        const char * src0, const char * src1, char * dst, cudaStream_t stream,
        const char * resid = nullptr, int64_t nbr1 = 0) {
    int64_t k = ne0 * ne1;
    const int num_blocks = (k + CUDA_MULTI_ADD_BLOCK_SIZE - 1) / CUDA_MULTI_ADD_BLOCK_SIZE;
    mul_multi_add_f32<<<num_blocks, CUDA_MULTI_ADD_BLOCK_SIZE, 0, stream>>>(nused, ne0, ne1, nb1, nb01, nb02, nb11, nb12, src0, src1, dst, resid, nbr1);
}

static __global__ void mul_multi_add_f32(int nused, int64_t ne0, int64_t ne1, int64_t nb1, int64_t nb01, int64_t nb02, int64_t nb11, int64_t nb12, int64_t nb31,
        const char * src0, const char * src1, char * dst, const float * scales, const char * cids,
        const char * resid, int64_t nbr1) {
    const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;
    int64_t k = ne0*ne1;
    if (i >= k) {
        return;
    }
    int i1 = i / ne0;
    int i0 = i % ne0;
    float * result = (float *)(dst + i1*nb1);

    const int * ids = (const int *)(cids + i1 * nb31);

    auto c0 = src0 + i1*nb02;
    auto c1 = src1 + i1*nb12;

    float sum = 0;
    for (int j = 0; j < nused; ++j) {
        auto x0 = (const float *)c0;
        auto x1 = (const float *)c1;
        sum += x0[i0] * x1[0] * scales[ids[j]];
        c0 += nb01;
        c1 += nb11;
    }
    if (resid) {   // G2-F4: residual last, see the non-scaled kernel
        sum += ((const float *)(resid + i1*nbr1))[i0];
    }
    result[i0] = sum;
}

static void mul_multi_add_f32_cuda(int nused, int64_t ne0, int64_t ne1, int64_t nb1, int64_t nb01, int64_t nb02, int64_t nb11, int64_t nb12, int64_t nb31,
        const char * src0, const char * src1, char * dst, const float * scales, const char * ids, cudaStream_t stream,
        const char * resid = nullptr, int64_t nbr1 = 0) {
    int64_t k = ne0 * ne1;
    const int num_blocks = (k + CUDA_MULTI_ADD_BLOCK_SIZE - 1) / CUDA_MULTI_ADD_BLOCK_SIZE;
    mul_multi_add_f32<<<num_blocks, CUDA_MULTI_ADD_BLOCK_SIZE, 0, stream>>>(nused, ne0, ne1, nb1, nb01, nb02, nb11, nb12, nb31, src0, src1, dst, scales, ids, resid, nbr1);
}

static void ggml_cuda_op_mul_multi_add_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        char * out, int64_t out_nb1, const char * resid, int64_t nbr1) {
    auto src0 = dst->src[0];
    auto src1 = dst->src[1];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] ==  dst->ne[0]);
    GGML_ASSERT(src0->ne[2] ==  dst->ne[1]);
    GGML_ASSERT(src0->ne[1] == src1->ne[1]);
    GGML_ASSERT(src0->ne[2] == src1->ne[2]);
    GGML_ASSERT(src0->ne[3] == src1->ne[3]);
    GGML_ASSERT(src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[0] == 1);

    auto src2 = dst->src[2];
    auto src3 = dst->src[3];
    if (src2 && src3) {
        GGML_ASSERT(src3->ne[0] == src0->ne[1]);
        GGML_ASSERT(src3->type == GGML_TYPE_I32);
        GGML_ASSERT(src2->type == GGML_TYPE_F32);

        mul_multi_add_f32_cuda(src0->ne[1], dst->ne[0], dst->ne[1], out_nb1, src0->nb[1], src0->nb[2], src1->nb[1], src1->nb[2], src3->nb[1],
                (const char *)src0->data, (const char *)src1->data, out, (const float *)src2->data, (const char *)src3->data, ctx.stream(), resid, nbr1);

        return;
    }

    mul_multi_add_f32_cuda(src0->ne[1], dst->ne[0], dst->ne[1], out_nb1, src0->nb[1], src0->nb[2], src1->nb[1], src1->nb[2],
            (const char *)src0->data, (const char *)src1->data, out, ctx.stream(), resid, nbr1);
}

void ggml_cuda_op_mul_multi_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_mul_multi_add_impl(ctx, dst, (char *)dst->data, dst->nb[1], nullptr, 0);
}

void ggml_cuda_op_mul_multi_add_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * add) {
    const ggml_tensor * resid = add->src[0] == dst ? add->src[1] : add->src[0];
    GGML_ASSERT(add->src[0] == dst || add->src[1] == dst);
    GGML_ASSERT(resid->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(add, dst) && ggml_are_same_shape(resid, dst));
    ggml_cuda_op_mul_multi_add_impl(ctx, dst, (char *)add->data, add->nb[1], (const char *)resid->data, resid->nb[1]);
}
