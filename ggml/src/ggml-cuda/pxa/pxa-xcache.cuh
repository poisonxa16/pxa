// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// pxa-xcache.cuh -- PXA_XCACHE (expert-granular hot cache), the two graph ops it adds.
//
// A MoE layer whose expert stack does not fit keeps its most-routed experts resident (the HOT
// stack, VRAM) and the rest in pinned host RAM (the COLD stack: streamed through the VRAM ring by
// wide graphs, read by the CPU or in place by narrow ones). Both stacks are ordinary expert stacks
// and every MoE kernel runs on them unchanged; these two ops are the only glue:
//
//   GGML_OP_MOE_SPLIT_IDS  top-k ids -> the ids of one side through the layer's I32 map
//                          (map[e] >= 0: hot slot, map[e] <= -2: cold slot -2 - map[e]), -1 where
//                          the expert lives on the other side. -1 is the SER convention: every
//                          MoE driver skips such a slot (its output row is left untouched or 0).
//   GGML_OP_MOE_MERGE      in place over the hot output: each slot the hot stack did NOT serve
//                          takes the cold output. Slots the hot stack served are never written,
//                          so each slot holds exactly the bytes the one-stack graph computes for
//                          that (token, expert) -- the per-expert GEMMs see the same rows either
//                          way -- and the weighting/summation downstream is unchanged.
#pragma once

#include "../common.cuh"

static __global__ void k_pxa_moe_split_ids(const char * __restrict__ ids, const size_t nb0, const size_t nb1,
        const int32_t * __restrict__ map, const int n_map, int32_t * __restrict__ dst,
        const int n_used, const int n_rows, const int side) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_used*n_rows) return;
    const int k = i % n_used, t = i / n_used;
    const int32_t e = *(const int32_t *)(ids + (size_t)k*nb0 + (size_t)t*nb1);
    int32_t r = -1;
    if (e >= 0 && e < n_map) {
        const int32_t v = map[e];
        r = side == 0 ? (v >= 0 ? v : -1) : (v <= -2 ? -2 - v : -1);
    }
    dst[i] = r;
}

static void ggml_cuda_op_moe_split_ids(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ids = dst->src[0];
    const ggml_tensor * map = dst->src[1];
    const int side   = ((const int32_t *)dst->op_params)[0];
    const int n_used = (int)ids->ne[0];
    const int n_rows = (int)ids->ne[1];
    const int n = n_used*n_rows;
    if (n == 0) return;
    GGML_ASSERT(ggml_is_contiguous(dst));
    k_pxa_moe_split_ids<<<(n + 255)/256, 256, 0, ctx.stream()>>>((const char *)ids->data, ids->nb[0], ids->nb[1],
            (const int32_t *)map->data, (int)ggml_nelements(map), (int32_t *)dst->data, n_used, n_rows, side);
    CUDA_CHECK(cudaGetLastError());
}

// one block per (slot, token); the block returns at once for a hot-served slot, so a decode-width
// merge costs a launch and a prefill-width merge touches only the few percent of rows that went
// to the cold stack.
static __global__ void k_pxa_moe_merge(char * __restrict__ dst, const size_t nb1, const size_t nb2,
        const char * __restrict__ cold, const size_t cnb1, const size_t cnb2,
        const char * __restrict__ ids_hot, const size_t hnb0, const size_t hnb1,
        const char * __restrict__ ids_cold, const size_t cnb0i, const size_t cnb1i,
        const int ne0, const int n_used) {
    const int k = blockIdx.x % n_used, t = blockIdx.x / n_used;
    const int32_t eh = *(const int32_t *)(ids_hot + (size_t)k*hnb0 + (size_t)t*hnb1);
    if (eh >= 0) return;
    const int32_t ec = *(const int32_t *)(ids_cold + (size_t)k*cnb0i + (size_t)t*cnb1i);
    float * y = (float *)(dst + (size_t)k*nb1 + (size_t)t*nb2);
    const float * x = (const float *)(cold + (size_t)k*cnb1 + (size_t)t*cnb2);
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) y[i] = ec >= 0 ? x[i] : 0.0f;
}

static void ggml_cuda_op_moe_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cold     = dst->src[1];
    const ggml_tensor * ids_hot  = dst->src[2];
    const ggml_tensor * ids_cold = dst->src[3];
    const int ne0 = (int)dst->ne[0], n_used = (int)dst->ne[1], n_tok = (int)dst->ne[2];
    const int64_t nblk = (int64_t)n_used*n_tok;
    if (nblk == 0) return;
    GGML_ASSERT(nblk < INT_MAX);
    k_pxa_moe_merge<<<(unsigned)nblk, 256, 0, ctx.stream()>>>((char *)dst->data, dst->nb[1], dst->nb[2],
            (const char *)cold->data, cold->nb[1], cold->nb[2],
            (const char *)ids_hot->data, ids_hot->nb[0], ids_hot->nb[1],
            (const char *)ids_cold->data, ids_cold->nb[0], ids_cold->nb[1], ne0, n_used);
    CUDA_CHECK(cudaGetLastError());
}
