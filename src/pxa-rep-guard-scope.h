#pragma once
// Which weights make the repetition guard (common/sampling.cpp, PXA_REP_GUARD) eligible.
//
// The guard is a runaway CAP for the tiers that measurably fall into repetition attractors, not a
// quality knob, so it must stay a no-op on every file that does not carry such a tier:
//   * a classic PXQ1 (1-bit) tensor, any size: the 2026-07-23 rule, kept as it was (PXQU12/24 mixed
//     maps carry a few and still loop);
//   * PXQN1 (1.25 bit per weight): the same class of tier in the next-generation format. The
//     Flash-Next 32 GB file (gate_exps PXQN1 in 47 of 49 layers, up_exps in 41 of 49) loops to death
//     on 5 of 10 greedy prompts and on 5 of 8 default-sampling runs (2026-10-04,
//     encode log: PXQN1 relmse 0.34-0.41, output error 0.14-0.19). A stray PXQN1 tensor in an
//     otherwise high-tier file does not make a loop prone file, so PXQN1 counts by BYTE SHARE of the
//     PXQ-family weights (the denominator leaves out q8_0 / k-quant tensors: the Flash-Next file
//     carries a 54 GB q8_0 per-layer embedding table that would otherwise hide 11.7 GB of PXQN1 at 14%
//     of the file, while it is 42% of the quantised weights).
//   * PXQN2 does NOT count: same file, PXQN2 relmse 0.11, output error 0.037 -- 4x to 5x lower than
//     PXQN1 -- and the guard has a measured cost on tiers that do not need it (it flipped an exact
//     answer on a PXQ4 file, 2026-07-30). No PXQN2-only file has been observed to loop.
// The PXQ2/3/4/6 and PXQN3..PXQN5 tiers never count.
#include "ggml.h"

#include <cstdint>

struct pxa_rep_guard_scope {
    // PXQN1 is eligible once it is at least this share of the file's PXQ-family tensor bytes.
    static constexpr double pxqn1_min_share = 0.05;

    uint64_t bytes_total = 0;   // PXQ-family tensor bytes only
    uint64_t bytes_pxqn1 = 0;
    bool     any_pxq1    = false;

    // `is_pxq` = the tensor's type belongs to the PXQ / PXQN family (the caller's own classifier)
    void add(enum ggml_type type, uint64_t nbytes, bool is_pxq) {
        if (type == GGML_TYPE_PXQ1)  { any_pxq1 = true; }
        if (!is_pxq) { return; }
        bytes_total += nbytes;
        if (type == GGML_TYPE_PXQN1) { bytes_pxqn1 += nbytes; }
    }

    double pxqn1_share() const {
        return bytes_total > 0 ? (double) bytes_pxqn1 / (double) bytes_total : 0.0;
    }

    bool pxqn1_eligible() const {
        return bytes_pxqn1 > 0 && pxqn1_share() >= pxqn1_min_share;
    }

    bool eligible() const {
        return any_pxq1 || pxqn1_eligible();
    }
};
