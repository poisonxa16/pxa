// pxq-cpu-ops.h — the two ggml graph ops whose bodies need the C++ standard library.
//
// PXA 2026-08-21, ik separation phase 2. Everything else that used to live in
// ggml/src/iqk/iqk_cpu_ops.cpp was pure scalar arithmetic and moved into pxq-cpu.c as C.
// These two did not: ARGSORT and GROUPED_TOPK both want a partial sort over
// (value, index) pairs, and C has no std::partial_sort. Rather than hand-roll a
// selection sort and change which expert wins a tie, we keep the STL and pay for one
// extra C++ translation unit. It is compiled unconditionally, exactly like pxq-cpu.c,
// and has no dependency on anything under ggml/src/iqk.
//
// ggml.c is C, so these are declared extern "C" and called from
// ggml_compute_forward_argsort / ggml_compute_forward_grouped_topk.

#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// GGML_OP_ARGSORT, f32 source. dst is I32: for every row of src[0], the column indices
// ordered by value. op_params[0] is the ggml_sort_order, op_params[1] is nk -- when
// nk < ne0 only the first nk entries of each output row are ordered (the rest are the
// leftovers of the partition, which is all the consumers of this op ever read).
//
// The stock C body, ggml_compute_forward_argsort_f32, is still alive in ggml.c as the
// reference: it is an O(n^2) bubble sort that ignores nk. It is not what we call.
void pxa_argsort(struct ggml_tensor * dst, int ith, int nth);

// GGML_OP_GROUPED_TOPK, f32 source. dst is I32. This op has NO stock ggml body anywhere
// in the tree -- ggml_compute_forward_grouped_topk is only the type switch -- so this
// function is the whole implementation, not an accelerator.
//
// op_params: [0] n_groups, [1] n_top_groups, [2] nk. The experts of a row are cut into
// n_groups contiguous groups; each group is scored by the sum of its top-nk values; the
// best n_top_groups groups survive; the top ne0 experts are then taken from the survivors.
void pxa_grouped_top_k(struct ggml_tensor * dst, int ith, int nth);

#ifdef __cplusplus
}
#endif
