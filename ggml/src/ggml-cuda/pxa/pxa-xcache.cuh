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
//   (sides 2 and 3 of the same op: PXA_XCACHE_ASYNC, the cold half computed by a host worker thread -- see
//    ggml_moe_cold_submit / ggml_moe_cold_wait in ggml.h and the kernels at the end of this file)
//   GGML_OP_MOE_MERGE      in place over the hot output: each slot the hot stack did NOT serve
//                          takes the cold output. Slots the hot stack served are never written,
//                          so each slot holds exactly the bytes the one-stack graph computes for
//                          that (token, expert) -- the per-expert GEMMs see the same rows either
//                          way -- and the weighting/summation downstream is unchanged.
#pragma once

#include "../common.cuh"

// cnt != nullptr (side 0 only): the online-adaptation block that follows the map. cnt[0 .. n_map) are per-expert
// routing counters (every routed expert of every token, hot or cold, so the ranking sees the whole distribution),
// cnt[n_map + 0] = routings served by the hot stack, [+1] = routings that went to the cold stack, [+2] = tokens.
// All u32 counters only ever grow (the reader differences snapshots modulo 2^32). The counting costs one atomic
// per routed expert; the hit/miss pair is reduced per block first.
static __global__ void k_pxa_moe_split_ids(const char * __restrict__ ids, const size_t nb0, const size_t nb1,
        const int32_t * __restrict__ map, const int n_map, int32_t * __restrict__ dst,
        const int n_used, const int n_rows, const int side, uint32_t * __restrict__ cnt) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const bool valid = i < n_used*n_rows;
    int32_t r = -1;
    int hit = 0, miss = 0;
    if (valid) {
        const int k = i % n_used, t = i / n_used;
        const int32_t e = *(const int32_t *)(ids + (size_t)k*nb0 + (size_t)t*nb1);
        if (e >= 0 && e < n_map) {
            const int32_t v = map[e];
            r = side == 0 ? (v >= 0 ? v : -1) : (v <= -2 ? -2 - v : -1);
            if (cnt) { atomicAdd(cnt + e, 1u); hit = v >= 0; miss = v < 0; }
        }
        dst[i] = r;
    }
    if (cnt) {
        __shared__ int s_hit, s_miss;
        if (threadIdx.x == 0) { s_hit = 0; s_miss = 0; }
        __syncthreads();
        if (hit)  atomicAdd(&s_hit, 1);
        if (miss) atomicAdd(&s_miss, 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            if (s_hit)  atomicAdd(cnt + n_map + 0, (uint32_t) s_hit);
            if (s_miss) atomicAdd(cnt + n_map + 1, (uint32_t) s_miss);
            if (blockIdx.x == 0) atomicAdd(cnt + n_map + 2, (uint32_t) n_rows);
        }
    }
}

static void ggml_cuda_op_moe_cold_async(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static void ggml_cuda_op_moe_split_ids(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (((const int32_t *)dst->op_params)[0] >= 2) {
        ggml_cuda_op_moe_cold_async(ctx, dst);
        return;
    }
    const ggml_tensor * ids = dst->src[0];
    const ggml_tensor * map = dst->src[1];
    const int side   = ((const int32_t *)dst->op_params)[0];
    const int n_exp  = ((const int32_t *)dst->op_params)[1];   // > 0: the map tensor carries the adaptation block
    const int n_used = (int)ids->ne[0];
    const int n_rows = (int)ids->ne[1];
    const int n = n_used*n_rows;
    if (n == 0) return;
    GGML_ASSERT(ggml_is_contiguous(dst));
    const int n_map = n_exp > 0 ? n_exp : (int)ggml_nelements(map);
    uint32_t * cnt = (n_exp > 0 && side == 0 && ggml_nelements(map) >= 2*(int64_t)n_exp + 8)
            ? (uint32_t *)((int32_t *)map->data + n_exp) : nullptr;
    k_pxa_moe_split_ids<<<(n + 255)/256, 256, 0, ctx.stream()>>>((const char *)ids->data, ids->nb[0], ids->nb[1],
            (const int32_t *)map->data, n_map, (int32_t *)dst->data, n_used, n_rows, side, cnt);
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

// ---- PXA_XCACHE_ASYNC: the cold half on a host worker thread, no host sync on the compute stream -------------------------------
//
//   submit (side 2)  one block: the cold ids and the activation go to the slot's pinned host block, then the request flag is raised
//                    (a layer whose ids hold no cold slot raises the DONE flag itself: the host is never involved)
//   ... the hot half's kernels run on the stream while the worker thread computes the cold experts ...
//   wait (side 3)    one block: thread 0 spins on the done flag (volatile reads of host memory, a few hundred cycles apart), then the
//                    block copies the cold slots' rows of the host result into the tensor MOE_MERGE reads
//
// The request number lives in device memory (the submit kernel increments it, the wait kernel reads it), so neither node carries
// per-token state. Data words written by one side and read by the other are fenced at system scope before the flag that publishes
// them; the host side does the same (llama-pxa-xcache-async.cpp). The wait gives up after PXA_XCACHE_ASYNC_TIMEOUT_S (default 30 s)
// instead of hanging the card: the timeout is counted, reported at exit, and the rows of that layer are zero.
struct pxa_cold_dev {
    volatile uint64_t * req; volatile uint32_t * done;
    int32_t * ids; float * cur; float * out;
    uint32_t * seq; unsigned long long * stat;
};

static __global__ void k_pxa_cold_submit(const pxa_cold_dev s, const float * __restrict__ cur, const char * __restrict__ ids,
        const size_t inb0, const size_t inb1, const int n_embd, const int n_used, const int n_tok, int32_t * __restrict__ ticket) {
    __shared__ int s_cold;
    if (threadIdx.x == 0) s_cold = 0;
    __syncthreads();
    const int n_ids = n_used*n_tok;
    int mine = 0;
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        const int k = i % n_used, t = i / n_used;
        const int32_t e = *(const int32_t *)(ids + (size_t)k*inb0 + (size_t)t*inb1);
        s.ids[i] = e;
        mine += e >= 0;
    }
    if (mine) atomicAdd(&s_cold, mine);
    __syncthreads();
    if (s_cold > 0) {
        const int n = n_embd*n_tok;
        if ((n & 3) == 0 && (((uintptr_t) cur | (uintptr_t) s.cur) & 15) == 0) {
            const float4 * src = (const float4 *) cur;
            float4 * dstv = (float4 *) s.cur;
            for (int i = threadIdx.x; i < n/4; i += blockDim.x) dstv[i] = src[i];
        } else {
            for (int i = threadIdx.x; i < n; i += blockDim.x) s.cur[i] = cur[i];
        }
    }
    __threadfence_system();      // every thread's writes are visible to the host before the flag is
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t q = *s.seq + 1u;
        *s.seq = q;
        ticket[0] = (int32_t) q;
        ticket[1] = s_cold;
        if (s_cold > 0) *s.req = (uint64_t) q | ((uint64_t) (unsigned) n_tok << 32); else { *s.done = q; atomicAdd(s.stat + 2, 1ull); }
        __threadfence_system();
    }
}

static __global__ void k_pxa_cold_wait(const pxa_cold_dev s, const char * __restrict__ ids, const size_t inb0, const size_t inb1,
        float * __restrict__ dst, const size_t dnb1, const size_t dnb2, const int n_embd, const int n_used, const int n_tok,
        const long long timeout_clk) {
    __shared__ int s_timeout;
    if (threadIdx.x == 0) {
        const uint32_t q = *s.seq;
        const long long t0 = clock64();
        long long now = t0;
        int to = 0;
        while (*s.done != q) {                       // volatile: every poll is a read of host memory
            const long long c = clock64();
            if (c - t0 > timeout_clk) { to = 1; break; }
            while (clock64() - c < 160) { }          // ~0.1 us between polls: do not saturate the link
        }
        now = clock64();
        const unsigned long long w = (unsigned long long) (now - t0);
        atomicAdd(s.stat + 0, w);
        atomicAdd(s.stat + 1, 1ull);
        atomicMax(s.stat + 4, w);
        if (to) atomicAdd(s.stat + 3, 1ull);
        s_timeout = to;
    }
    __syncthreads();
    __threadfence_system();                          // the flag was seen: what it publishes is read after it
    const int n_ids = n_used*n_tok;
    if ((n_embd & 3) == 0 && (dnb1 & 15) == 0 && (dnb2 & 15) == 0 && (((uintptr_t) dst | (uintptr_t) s.out) & 15) == 0) {
        const int nv = n_embd/4;
        const int total = n_ids*nv;
        for (int base = threadIdx.x; base < total; base += blockDim.x*4) {
            float4 v[4];
            int slot[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int idx = base + u*blockDim.x;
                slot[u] = -1;
                if (idx < total) {
                    const int sl = idx / nv;
                    const int k = sl % n_used, t = sl / n_used;
                    const int32_t e = *(const int32_t *)(ids + (size_t)k*inb0 + (size_t)t*inb1);
                    if (e >= 0) {
                        slot[u] = sl;
                        v[u] = s_timeout ? make_float4(0.f, 0.f, 0.f, 0.f) : __ldcv(((const float4 *) s.out) + idx);
                    }
                }
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (slot[u] >= 0) {
                    const int idx = base + u*blockDim.x;
                    const int sl = slot[u];
                    const int k = sl % n_used, t = sl / n_used;
                    ((float4 *) ((char *) dst + (size_t)k*dnb1 + (size_t)t*dnb2))[idx - sl*nv] = v[u];
                }
            }
        }
    } else {
        for (int sl = 0; sl < n_ids; ++sl) {
            const int k = sl % n_used, t = sl / n_used;
            const int32_t e = *(const int32_t *)(ids + (size_t)k*inb0 + (size_t)t*inb1);
            if (e < 0) continue;
            float * y = (float *) ((char *) dst + (size_t)k*dnb1 + (size_t)t*dnb2);
            for (int i = threadIdx.x; i < n_embd; i += blockDim.x) y[i] = s_timeout ? 0.0f : __ldcv(s.out + (size_t) sl*n_embd + i);
        }
    }
}

// side 4 (PXA_XCACHE_ASYNC_CHECK): one block, bit-compare the rows the cold ids select
static __global__ void k_pxa_cold_check(const pxa_cold_dev s, const char * __restrict__ ids, const size_t inb0, const size_t inb1,
        const float * __restrict__ a, const size_t anb1, const size_t anb2, const float * __restrict__ b, const size_t bnb1, const size_t bnb2,
        const int n_embd, const int n_used, const int n_tok) {
    __shared__ unsigned long long s_rows, s_bad;
    __shared__ unsigned s_maxd;
    if (threadIdx.x == 0) { s_rows = 0; s_bad = 0; s_maxd = 0; }
    __syncthreads();
    for (int sl = 0; sl < n_used*n_tok; ++sl) {
        const int k = sl % n_used, t = sl / n_used;
        const int32_t e = *(const int32_t *)(ids + (size_t)k*inb0 + (size_t)t*inb1);
        if (e < 0) continue;                       // uniform over the block
        const float * ra = (const float *) ((const char *) a + (size_t)k*anb1 + (size_t)t*anb2);
        const float * rb = (const float *) ((const char *) b + (size_t)k*bnb1 + (size_t)t*bnb2);
        int bad = 0;
        float md = 0.0f;
        for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
            const float x = ra[i], y = rb[i];
            if (__float_as_uint(x) != __float_as_uint(y)) { bad = 1; md = fmaxf(md, fabsf(x - y)); }
        }
        const int any = __syncthreads_or(bad);
        if (bad) atomicMax(&s_maxd, __float_as_uint(md));
        if (threadIdx.x == 0) { ++s_rows; s_bad += any ? 1 : 0; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(s.stat + 5, s_rows);
        atomicAdd(s.stat + 6, s_bad);
        atomicMax(s.stat + 7, (unsigned long long) s_maxd);
    }
}

static void ggml_cuda_op_moe_cold_async(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int side = ((const int32_t *)dst->op_params)[0];
    ggml_cuda_cold_slot * slot = nullptr;
    memcpy(&slot, (const char *) dst->op_params + 2*sizeof(int32_t), sizeof(slot));
    GGML_ASSERT(slot != nullptr && slot->device == ctx.device);
    pxa_cold_dev s = { slot->d_req, slot->d_done, slot->d_ids, slot->d_cur, slot->d_out, slot->d_seq, slot->d_stat };
    const ggml_tensor * ids = dst->src[0];
    const int n_used = (int) ids->ne[0], n_tok = (int) ids->ne[1];
    GGML_ASSERT(n_used == slot->n_used && n_tok >= 1 && n_tok <= slot->n_tok_max);
    if (side == 4) {
        const ggml_tensor * a = dst->src[1];
        const ggml_tensor * b = dst->src[2];
        k_pxa_cold_check<<<1, 256, 0, ctx.stream()>>>(s, (const char *) ids->data, ids->nb[0], ids->nb[1],
                (const float *) a->data, a->nb[1], a->nb[2], (const float *) b->data, b->nb[1], b->nb[2], slot->n_embd, n_used, n_tok);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (side == 2) {
        const ggml_tensor * cur = dst->src[1];
        GGML_ASSERT(cur->ne[0] == slot->n_embd && ggml_nelements(cur) == (int64_t) slot->n_embd*n_tok && ggml_is_contiguous(cur));
        k_pxa_cold_submit<<<1, 256, 0, ctx.stream()>>>(s, (const float *) cur->data, (const char *) ids->data, ids->nb[0], ids->nb[1],
                slot->n_embd, n_used, n_tok, (int32_t *) dst->data);
    } else {
        GGML_ASSERT(side == 3 && dst->ne[0] == slot->n_embd && ggml_is_contiguous(dst));
        static const double timeout_s = [] { const char * e = getenv("PXA_XCACHE_ASYNC_TIMEOUT_S"); return e && atof(e) > 0 ? atof(e) : 30.0; }();
        const long long timeout_clk = (long long) (timeout_s * 1000.0 * ggml_backend_cuda_cold_clock_khz(ctx.device));
        k_pxa_cold_wait<<<1, 256, 0, ctx.stream()>>>(s, (const char *) ids->data, ids->nb[0], ids->nb[1],
                (float *) dst->data, dst->nb[1], dst->nb[2], slot->n_embd, n_used, n_tok, timeout_clk);
    }
    CUDA_CHECK(cudaGetLastError());
}
