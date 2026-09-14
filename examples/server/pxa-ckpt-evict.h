#pragma once

// PXA_CKPT_EVICT: which prompt checkpoint to drop when a slot's checkpoint ring is full.
//
// The shipped policy ("structural") is purely positional: it keeps the first and last checkpoints
// and removes the interior one whose removal leaves the remaining checkpoints most evenly spaced.
// That is a good prior when nothing is known about which checkpoint will be needed, but it has no
// memory: a checkpoint that has already saved a re-prefill three times is exactly as evictable as
// one that has never been touched.
//
// The "value" policy keeps the same structural term and multiplies it by a proven-usefulness
// weight, (1 + 4 * replay_hits), where replay_hits counts how many times THAT checkpoint was the
// one restored from. A checkpoint that has demonstrated it is a re-entry point therefore has to be
// four times worse structurally, per hit, before it is dropped. Checkpoints flagged as replay
// boundaries -- the first checkpoint created after a slot re-entered its cached prompt, i.e. the
// turn boundary an agent loop comes back to -- are protected outright, and only considered when
// every interior candidate is one (something must always be evicted to make room).
//
// Eviction changes only what has to be recomputed versus reused. It never changes a computed
// value, so neither policy can move a logit or a token; the only observable difference is prompt
// re-processing time and therefore TTFT.
//
// Lever: PXA_CKPT_EVICT=value turns the weighted policy on. Default (unset, or anything else) is
// the structural policy, byte-for-byte the pre-existing behaviour. Hit counting runs in both modes,
// because it is also the instrumentation that answers "how often does our checkpoint restore
// actually fire" for free.
//
// This header is pure: no server types, no llama types, no I/O. It is unit-tested directly
// (tests/test-ckpt-evict.cpp) against a verbatim copy of the pre-existing structural scoring.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

enum pxa_ckpt_evict_mode {
    PXA_CKPT_EVICT_STRUCTURAL = 0, // shipped default: gap-variance only
    PXA_CKPT_EVICT_VALUE      = 1, // gap-variance weighted by proven replay usefulness
};

struct pxa_ckpt_evict_entry {
    int64_t  pos_max     = 0;     // the checkpoint's newest position (its structural coordinate)
    uint32_t replay_hits = 0;     // how many restores have come from this exact checkpoint
    bool     boundary    = false; // deliberately placed at a replay boundary: protect it
};

// PXA_CKPT_EVICT=value -> weighted policy. Read once.
inline pxa_ckpt_evict_mode pxa_ckpt_evict_mode_from_env() {
    static const pxa_ckpt_evict_mode mode = [] {
        const char * e = getenv("PXA_CKPT_EVICT");
        if (e && (strcmp(e, "value") == 0 || strcmp(e, "1") == 0)) {
            return PXA_CKPT_EVICT_VALUE;
        }
        return PXA_CKPT_EVICT_STRUCTURAL;
    }();
    return mode;
}

// The structural score of interior checkpoint i: the product of its two adjacent gaps, normalised
// by the newest position. Minimising it minimises the gap variance left behind, because the first
// and last checkpoints are fixed and therefore so is the mean.
//
// Written exactly as the shipped loop computes it -- same operand order, same division -- so that
// the structural mode below stays bit-identical to the behaviour it replaces.
inline double pxa_ckpt_structural_score(const std::vector<pxa_ckpt_evict_entry> & ck, size_t i, double max_pos) {
    const double diff  = (double) (ck[i].pos_max     - ck[i - 1].pos_max);
    const double diff2 = (double) (ck[i + 1].pos_max - ck[i].pos_max);
    return diff * (diff2 / max_pos);
}

// Returns the index of the checkpoint to evict. Never returns an index outside [0, ck.size()).
//
// Fewer than three checkpoints leaves nothing interior to reason about, so the oldest goes, and
// with exactly three there is exactly one interior candidate. Both cases are identical in both
// modes -- the weighting can only choose BETWEEN candidates, never refuse to make room.
inline size_t pxa_ckpt_evict_pick(const std::vector<pxa_ckpt_evict_entry> & ck, pxa_ckpt_evict_mode mode) {
    const size_t n = ck.size();

    if (n < 3) {
        return 0;
    }
    if (n == 3) {
        return 1;
    }

    const size_t start = 1;     // never remove the first
    const size_t end   = n - 1; // never remove the last

    if (mode == PXA_CKPT_EVICT_STRUCTURAL) {
        const double max_pos = (double) ck[n - 1].pos_max;

        size_t best_idx      = start;
        double best_variance = pxa_ckpt_structural_score(ck, start, max_pos);

        for (size_t i = start + 1; i < end; i++) {
            const double variance = pxa_ckpt_structural_score(ck, i, max_pos);
            if (variance < best_variance) {
                best_variance = variance;
                best_idx      = i;
            }
        }
        return best_idx;
    }

    // value mode: same structural term, weighted by proven usefulness, boundaries protected.
    //
    // max_pos is guarded here (the structural path is left exactly as it shipped): a ring whose
    // newest checkpoint sits at position 0 would otherwise divide by zero and make every
    // comparison false, silently pinning the choice to the first interior slot.
    const double max_pos = ck[n - 1].pos_max > 0 ? (double) ck[n - 1].pos_max : 1.0;

    bool   have_best = false;
    size_t best_idx  = start;
    double best_val  = 0.0;

    // pass 1 considers only unprotected checkpoints; pass 2 runs only if every interior
    // checkpoint is a protected boundary, in which case one of them still has to go
    for (int pass = 0; pass < 2 && !have_best; ++pass) {
        const bool allow_boundary = pass == 1;

        for (size_t i = start; i < end; i++) {
            if (ck[i].boundary && !allow_boundary) {
                continue;
            }

            // (1 + 4*hits): one proven restore already makes a checkpoint five times as
            // expensive to drop as an equally-placed one that has never been used
            const double weight = 1.0 + 4.0 * (double) ck[i].replay_hits;
            const double value  = pxa_ckpt_structural_score(ck, i, max_pos) * weight;

            if (!have_best || value < best_val) {
                have_best = true;
                best_val  = value;
                best_idx  = i;
            }
        }
    }

    return best_idx;
}
