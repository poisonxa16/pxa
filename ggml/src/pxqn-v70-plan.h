// pxqn-v70-plan.h -- integer decisions for the sm_70 resident seat.
// Header only, no CUDA. The closed library wraps these and can leave them
// unanswered (return 0) until a probe build turns the wrapper on.
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace pxqn_v70 {

// A prompt-copy draft versus the MTP ceiling for this round.
//   1  the copy is at least as long as the MTP draft would be: verify it, do not ask MTP
//  -1  the copy is shorter than that ceiling: discard it and ask MTP
//   0  no opinion (no copy, or no MTP stage)
inline int copy_wins(int ngram_len, int mtp_n_max) {
    if (ngram_len <= 0 || mtp_n_max <= 0) {
        return 0;
    }
    return ngram_len >= mtp_n_max ? 1 : -1;
}

// Next physical batch to try when `cap` checkpoint rows fit and `want` were asked.
// Steps down by four SM-waves so a boot does not rebuild once per wave, stays on a
// multiple of the SM count, and stops at 256. 0 = keep `ub`.
inline int ckpt_ub_next(int ub, int cap, int want, int nsm) {
    if (want <= 1 || cap >= want || nsm <= 0 || ub <= 256) {
        return 0;
    }
    const int jump = nsm * 4;
    int next = (ub - 1) / jump * jump;
    if (next >= ub) {
        next -= jump;
    }
    if (next < 256) {
        return 0;
    }
    return next;
}

// How many KV rows of an f16 stage fit in `free_bytes` after `margin`.
//   0  the whole cache fits (the existing single stage)
//  -1  not even 256 rows fit (keep the decline)
//  >0  a multiple of 256, at most n_kv
// `bytes_per_row` is one matrix. `n_mat` is 1 when V is a view of K, else 2.
inline int fa_kv_chunk(int64_t n_kv, int64_t bytes_per_row, int n_mat, size_t free_bytes, size_t margin) {
    if (n_kv <= 0 || bytes_per_row <= 0 || n_mat <= 0) {
        return 0;
    }
    const size_t row = (size_t) bytes_per_row * (size_t) n_mat;
    const size_t need = row * (size_t) n_kv;
    if (free_bytes > margin && need <= free_bytes - margin) {
        return 0;
    }
    const size_t budget = free_bytes > margin ? free_bytes - margin : 0;
    if (budget < row * 256) {
        return -1;
    }
    int64_t rows = (int64_t) (budget / row);
    rows -= rows % 256;
    if (rows < 256) {
        return -1;
    }
    if (rows > n_kv) {
        rows = n_kv - n_kv % 256;
    }
    return rows >= 256 ? (int) rows : -1;
}

}  // namespace pxqn_v70
