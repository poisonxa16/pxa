#pragma once

// PXA_SPEC_FIXED_WIDTH: verify every speculative step at ONE batch width.
//
// Why (spec-verify-batch-invariance, 2026-09-14). A verify step decodes M = 1 + n_draft rows in
// one forward pass, and every matmul picks its kernel, K-split and attention tile from the row
// count. n_draft changes from step to step (the drafter's confidence stop, the n-gram map carried
// between requests, the adaptive caps), so the same position's logits were computed by different
// kernels on different steps, and at a near-tie the greedy pick followed the draft length. With
// this lever the verify batch is padded to the slot's structural draft ceiling with filler rows
// that are never offered for acceptance: the width the target sees no longer depends on how long
// the draft happened to be. The filler rows are rolled back exactly like rejected drafts (KV
// positions trimmed, recurrent state restored from the per-step checkpoint).
//
// It does NOT make a speculative run equal a drafter-off run (width 1+n_max is still not width 1),
// and on a hybrid model the recurrent scan still starts where each step starts. What it removes is
// the kernel-route dependence on the draft length, which the 2026-09-14 measurement named as the
// mechanism ("it is the route DIFFERENCE that costs precision").
//
// Where it stands down (returns 0 filler rows):
//   * a recurrent/hybrid model whose checkpoint mode is not per-step: every padded step has a
//     rejected tail, and the non-per-step modes pay a full re-decode of the accepted run for that
//     (at yet another width), so padding there costs a decode per step and buys nothing;
//   * a per-step checkpoint smaller than the padded width (the save would refuse the batch);
//   * multi-slot recurrent steps, where the server's uniform-batch equalizer owns the width.
//
// Lever: PXA_SPEC_FIXED_WIDTH=1 turns it on. Default OFF until a speed probe says what the filler
// rows cost on each card (they are real rows; this build does not yet teach the width-sensitive
// kernels to skip them).

#include <algorithm>
#include <cstdio>
#include <cstdlib>

static inline bool pxa_spec_fixed_width_on() {
    static const bool on = [](){
        const char * e = getenv("PXA_SPEC_FIXED_WIDTH");
        const bool v = e && atoi(e) != 0;
        if (v) {
            fprintf(stderr, "PXA_SPEC_FIXED_WIDTH: ON -- every verify batch is padded to the slot's draft "
                            "ceiling with filler rows that are never accepted, so the verify width (and the "
                            "kernels it selects) no longer follows the draft length\n");
        }
        return v;
    }();
    return on;
}

struct pxa_spec_fixed_width_in {
    int  n_real;          // real drafted tokens this step (after every cap)
    int  n_max_struct;    // the slot's structural draft ceiling (largest stage n_max)
    int  ctx_room;        // n_ctx - n_past - 2: rows that still fit in the context
    bool recurrent;       // model has recurrent state
    int  ckpt_capacity;   // per-step checkpoint capacity in tokens (drafted + 1); <= 0 = not per-step
    bool equalized;       // the multi-slot uniform-batch equalizer capped this step
};

// Number of filler rows to append after the real draft (0 = verify as before).
static inline int pxa_spec_fixed_width_pad(const pxa_spec_fixed_width_in & in) {
    if (in.n_real <= 0 || in.equalized) {
        return 0;
    }
    int target = std::min(in.n_max_struct, in.ctx_room);
    if (in.recurrent) {
        if (in.ckpt_capacity <= 0) {
            return 0;
        }
        target = std::min(target, in.ckpt_capacity - 1);
    }
    return std::max(0, target - in.n_real);
}
