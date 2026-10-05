// PXA_SPEC_FIXED_WIDTH pad rule (examples/server/pxa-spec-fixed-width.h). Pure, no model, no GPU.
#include "pxa-spec-fixed-width.h"

#include <cstdio>

static int g_fail = 0;
#define CHECK_EQ(a, b) do { const int _a = (a), _b = (b); if (_a != _b) { \
    fprintf(stderr, "FAIL %s:%d: %s = %d, want %d\n", __FILE__, __LINE__, #a, _a, _b); ++g_fail; } } while (0)

static pxa_spec_fixed_width_in mk(int n_real, int n_max, int room, bool rec, int cap, bool eq) {
    pxa_spec_fixed_width_in in;
    in.n_real = n_real; in.n_max_struct = n_max; in.ctx_room = room;
    in.recurrent = rec; in.ckpt_capacity = cap; in.equalized = eq;
    return in;
}

int main() {
    // attention-only model: pad to the ceiling, whatever the draft length was
    CHECK_EQ(pxa_spec_fixed_width_pad(mk(1, 4, 1000, false, 0, false)), 3);
    CHECK_EQ(pxa_spec_fixed_width_pad(mk(3, 4, 1000, false, 0, false)), 1);
    CHECK_EQ(pxa_spec_fixed_width_pad(mk(4, 4, 1000, false, 0, false)), 0);
    // widths are fixed: real + pad is constant across draft lengths
    for (int n = 1; n <= 8; ++n) {
        CHECK_EQ(n + pxa_spec_fixed_width_pad(mk(n, 8, 1000, false, 0, false)), 8);
    }
    // no draft, no verify: nothing to pad
    CHECK_EQ(pxa_spec_fixed_width_pad(mk(0, 4, 1000, false, 0, false)), 0);
    // the context end bounds the pad
    CHECK_EQ(pxa_spec_fixed_width_pad(mk(1, 8, 3, false, 0, false)), 2);
    CHECK_EQ(pxa_spec_fixed_width_pad(mk(2, 8, 1, false, 0, false)), 0);
    // recurrent: only with a per-step checkpoint, and never past its capacity (drafted + 1)
    CHECK_EQ(pxa_spec_fixed_width_pad(mk(1, 4, 1000, true, 0, false)), 0);
    CHECK_EQ(pxa_spec_fixed_width_pad(mk(1, 4, 1000, true, 5, false)), 3);
    CHECK_EQ(pxa_spec_fixed_width_pad(mk(1, 4, 1000, true, 3, false)), 1);
    CHECK_EQ(pxa_spec_fixed_width_pad(mk(2, 4, 1000, true, 3, false)), 0);
    // the multi-slot equalizer owns the width
    CHECK_EQ(pxa_spec_fixed_width_pad(mk(1, 4, 1000, false, 0, true)), 0);
    if (g_fail) {
        fprintf(stderr, "test-pxa-spec-fixed-width: %d failure(s)\n", g_fail);
        return 1;
    }
    printf("test-pxa-spec-fixed-width: OK\n");
    return 0;
}
