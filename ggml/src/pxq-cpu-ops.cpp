// pxq-cpu-ops.cpp — ARGSORT and GROUPED_TOPK, reowned from ggml/src/iqk/iqk_cpu_ops.cpp.
//
// PXA 2026-08-21, ik separation phase 2. See pxq-cpu-ops.h for why these two, and only
// these two, need a C++ translation unit: std::partial_sort over (value, index) pairs.
//
// Provenance: the bodies below are the ik functions iqk_argsort (iqk_cpu_ops.cpp:240-278)
// and iqk_grouped_top_k (:178-238) plus their group_score helper (:57-64), transcribed
// with `auto` spelled out and the thread_local scratch made explicit. The comparison
// order, the row split across threads and the tie-break are all deliberately unchanged --
// this commit is a change of ownership, not of arithmetic. std::greater<std::pair<float,int>>
// is a total order (equal scores fall back to the larger column index), so the result is
// deterministic even though partial_sort is not a stable algorithm.
//
// Threading contract, inherited and preserved: every compute thread calls with its own
// (ith, nth) and takes rows [ith*npt, min(first+npt, nrows)). Get that arithmetic wrong
// and rows are silently dropped or written twice, which only shows up at nth > 1.

#include "pxq-cpu-ops.h"

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

namespace {

// Per-thread (value, index) scratch. Grows monotonically; never shrinks, so a long-running
// server pays the resize once. thread_local because every compute thread is in here at once.
std::vector<std::pair<float,int>> & pxa_pair_buffer(size_t size) {
    thread_local std::vector<std::pair<float,int>> buffer;
    if (buffer.size() < size) buffer.resize(size);
    return buffer;
}

// Per-thread float scratch for the group scorer.
//
// ik aliased this onto the head of the pair buffer -- `(float *)aux.data()` -- which was
// safe only because a pair<float,int> is 8 bytes and the scored region is always at least
// n_per_group pairs long. That is a load-bearing coincidence sitting one struct-layout
// change away from silent corruption, and it buys nothing: the scorer runs to completion
// before the pair buffer is filled. Separate buffer, same results.
std::vector<float> & pxa_float_buffer(size_t size) {
    thread_local std::vector<float> buffer;
    if (buffer.size() < size) buffer.resize(size);
    return buffer;
}

// Score one group by the sum of its top-nk values. aux must hold n_per_group floats.
float pxa_group_score(int n_per_group, int nk, const float * data, float * aux) {
    for (int j = 0; j < n_per_group; ++j) aux[j] = data[j];
    std::partial_sort(aux, aux + nk, aux + n_per_group, std::greater<float>{});
    float sum = 0;
    for (int j = 0; j < nk; ++j) sum += aux[j];
    return sum;
}

} // namespace

void pxa_grouped_top_k(struct ggml_tensor * dst, int ith, int nth) {
    const struct ggml_tensor * src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nrows(src) == ggml_nrows(dst));

    const int64_t nrows = ggml_nrows(src);
    const int64_t npt   = (nrows + nth - 1)/nth;
    const int64_t first = npt*ith;
    const int64_t last  = std::min(first + npt, nrows);
    if (last <= first) return;

    const int n_groups     = dst->op_params[0];
    const int n_top_groups = dst->op_params[1];
    const int nk           = dst->op_params[2];

    const int ne00 = src->ne[0];
    const int ne0  = dst->ne[0];
    GGML_ASSERT(ne0 <= ne00);
    GGML_ASSERT(ne00%n_groups == 0);
    const int n_per_group = ne00/n_groups;
    GGML_ASSERT(nk <= n_per_group);
    GGML_ASSERT(n_top_groups <= n_groups);

    const size_t work_size = (size_t)n_groups + (size_t)n_per_group*n_top_groups;
    std::vector<std::pair<float,int>> & aux = pxa_pair_buffer(work_size);
    std::vector<float> & score_aux = pxa_float_buffer((size_t)n_per_group);

    std::pair<float,int> * groups = aux.data() + (size_t)n_per_group*n_top_groups;

    for (int64_t ir = first; ir < last; ++ir) {
        const float * data = (const float *)((const char *)src->data + ir*src->nb[1]);
        int32_t * result   = (int32_t *)((char *)dst->data + ir*dst->nb[1]);
        // Degenerate routing: more slots than the surviving groups can supply, so every
        // expert is selected and the order does not matter.
        if (ne0 > n_per_group*n_top_groups) {
            for (int j = 0; j < ne0; ++j) result[j] = j;
            continue;
        }
        if (n_top_groups < n_groups) {
            for (int ig = 0; ig < n_groups; ++ig) {
                groups[ig] = { pxa_group_score(n_per_group, nk, data + ig*n_per_group, score_aux.data()), ig };
            }
            std::partial_sort(groups, groups + n_top_groups, groups + n_groups, std::greater<std::pair<float,int>>{});

            for (int ig = 0; ig < n_top_groups; ++ig) {
                const int i0 = n_per_group * ig;
                const int j0 = n_per_group * groups[ig].second;
                for (int j = 0; j < n_per_group; ++j) aux[i0 + j] = { data[j0 + j], j0 + j };
            }
        } else {
            for (int j = 0; j < ne00; ++j) aux[j] = { data[j], j };
        }
        if (ne0 < n_top_groups*n_per_group) {
            std::partial_sort(aux.begin(), aux.begin() + ne0, aux.begin() + n_top_groups*n_per_group, std::greater<std::pair<float,int>>{});
        } else {
            std::sort(aux.begin(), aux.begin() + ne0, std::greater<std::pair<float,int>>{});
        }
        for (int j = 0; j < ne0; ++j) result[j] = aux[j].second;
    }
}

void pxa_argsort(struct ggml_tensor * dst, int ith, int nth) {
    const struct ggml_tensor * src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);

    const int64_t nrows = ggml_nrows(src);
    const int64_t npt   = (nrows + nth - 1)/nth;
    const int64_t first = npt*ith;
    const int64_t last  = std::min(first + npt, nrows);
    if (last <= first) return;

    const enum ggml_sort_order order = (enum ggml_sort_order)dst->op_params[0];
    const int nk   = dst->op_params[1];
    const int ne00 = src->ne[0];

    std::vector<std::pair<float,int>> & aux = pxa_pair_buffer((size_t)ne00);

    for (int64_t ir = first; ir < last; ++ir) {
        const float * data = (const float *)((const char *)src->data + ir*src->nb[1]);
        for (int j = 0; j < ne00; ++j) aux[j] = { data[j], j };
        // nk < ne00 means the consumer (a VIEW + GET_ROWS pair) only reads the first nk
        // columns, so ordering the rest is wasted work. The tail is still written out --
        // it is the partition leftover, not garbage -- because dst is ne00 wide.
        if (nk < ne00) {
            if (order == GGML_SORT_ORDER_DESC) {
                std::partial_sort(aux.begin(), aux.begin() + nk, aux.begin() + ne00, std::greater<std::pair<float,int>>{});
            } else {
                std::partial_sort(aux.begin(), aux.begin() + nk, aux.begin() + ne00);
            }
        } else {
            if (order == GGML_SORT_ORDER_DESC) {
                std::sort(aux.begin(), aux.begin() + ne00, std::greater<std::pair<float,int>>{});
            } else {
                std::sort(aux.begin(), aux.begin() + ne00);
            }
        }
        int32_t * y = (int32_t *)((char *)dst->data + ir*dst->nb[1]);
        for (int j = 0; j < ne00; ++j) y[j] = aux[j].second;
    }
}
