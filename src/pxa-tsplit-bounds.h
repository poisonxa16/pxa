// Per-device tensor-split boundary computation (pure, no ggml dependency so tests/test-pxa-tsplit-bounds.cpp
// can run it without a GPU). Every boundary of a split dimension is a multiple of `granularity`.
#pragma once
#include <vector>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cassert>
#ifndef GGML_ASSERT
#define GGML_ASSERT(x) assert(x)
#endif

static inline std::vector<int> pxa_tsplit_create_split(int nr, int granularity, const std::vector<float> & splits, const std::vector<size_t> & mem_used,
        bool verbose = false) {
    (void) verbose;
    GGML_ASSERT(nr % granularity == 0);
    GGML_ASSERT(!splits.empty());
    if (granularity < 0) return std::vector<int>(splits.size(), nr);
    GGML_ASSERT(mem_used.size() == splits.size());
    size_t tot_memory_used = 1;
    for (auto & mem : mem_used) tot_memory_used += mem;
    int nchunk = nr / granularity;
    std::vector<int> result(splits.size());
    float last_split = 0;
    int sum = 0;
    for (int i = 0; i < (int)splits.size(); ++i) {
        float p = splits[i] - last_split;
        p += (p - 1.f*mem_used[i]/tot_memory_used);
        result[i] = roundf(p*nchunk);
        if (result[i] < 0) result[i] = 0;
        sum += result[i];
        last_split = splits[i];
    }
    while (sum > nchunk) {
        last_split = 0;
        float best_err = -INFINITY;
        int ibest = -1;
        for (int i = 0; i < (int)splits.size(); ++i) {
            if (result[i] > 0) {
                float p = splits[i] - last_split;
                p += (p - 1.f*mem_used[i]/tot_memory_used);
                float n_want = p*nchunk;
                float err = result[i] - n_want;
                if (err > best_err) {
                    best_err = err; ibest = i;
                }
            }
            last_split = splits[i];
        }
        GGML_ASSERT(ibest >= 0 && result[ibest] > 0);
        --result[ibest];
        --sum;
    }
    while (sum < nchunk) {
        last_split = 0;
        float best_err = -INFINITY;
        int ibest = -1;
        for (int i = 0; i < (int)splits.size(); ++i) {
            float p = splits[i] - last_split;
            p += (p - 1.f*mem_used[i]/tot_memory_used);
            float n_want = p*nchunk;
            float err = n_want - result[i];
            if (err > best_err) {
                best_err = err; ibest = i;
            }
            last_split = splits[i];
        }
        GGML_ASSERT(ibest >= 0);
        ++result[ibest];
        ++sum;
    }
    for (auto & r : result) r *= granularity;
    return result;
}


// PXQN RHT (ggml-pxqn.h PXQN_RHT_BLOCK): a rotated site rotates each device's activation slice per 128-block
// at K offset k0, so both the slice width and every slice offset must be multiples of 128. The dense/shared
// FFN cut (down dim 0, up/gate dim 1) is one chunk vector, so its granularity must be raised to 128 whenever
// the file rotates the FFN sites (PXQN4 has blck_size 32 -> panel 64, which let an uneven memory-weighted
// split land on 64 mod 128 and trip GGML_ASSERT in build_pxqn_rht). If nr is not a multiple of the raised
// granularity the old one is kept (that shape cannot be rotated either). `fell_back` reports that case.
static inline int pxa_tsplit_rht_granularity(int gran, int nr, bool rotated, bool * fell_back = nullptr) {
    if (fell_back) *fell_back = false;
    if (!rotated || gran % 128 == 0) return gran;
    int g = gran > 128 ? gran : 128;
    if (g % gran != 0) g = ((g + gran - 1) / gran) * gran;
    if (g % 128 != 0 || nr % g != 0) {
        if (fell_back) *fell_back = true;
        return gran;
    }
    return g;
}
