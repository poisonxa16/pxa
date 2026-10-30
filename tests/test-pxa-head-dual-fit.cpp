// PXA_HEAD_DUAL_LAYER fit guard (src/llama-pxa-head-dual.h). Pure, no model, no GPU.
//   (1) the measured 35B-A3B P100 pair (-c 8192, the default-on cell) still arms the cut,
//   (2) a first card that the =0 run fills to just above the context floor refuses the cut (it used to arm it and then fail
//       at context creation), and so does any card short of the copy,
//   (3) whenever the guard arms, the measured real context on the first card still leaves the floor free after the copy,
//   (4) the boundary is exact: free == need arms, one byte less does not.
#include "llama-pxa-head-dual.h"

#include <cstdio>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static const size_t MiB = (size_t) 1 << 20;

int main() {
    // the 35B-A3B pair: 15928 MiB free per card before the load, 9060 MiB of weights on the first card; the head copy is
    // 257.7 MiB; the loader's estimate for the first card: KV 113.5 MiB, compute 28 MiB; floor 128 MiB (PXA_POOL_FLOOR_MB default)
    const size_t copy = (size_t) (257.7 * MiB), kv = (size_t) (113.5 * MiB), comp = 28 * MiB, floor_b = 128 * MiB;
    // what that context really put on the first card past the weights: KV 113.5 + compute 123.25 + per-step 102 + GEMM pool 38
    // + checkpoint shadow 1.5 MiB
    const size_t real_ctx = (size_t) ((113.5 + 123.25 + 102.0 + 38.0 + 1.5) * MiB);

    // (1)
    CHECK(pxa_head_dual_fits((15928 - 9060) * MiB, copy, kv, comp, floor_b));

    // (2) the =0 run boots with 100 MiB above the floor; the copy (258 MiB) would push it under -> refuse
    const size_t tight = real_ctx + floor_b + 100 * MiB;
    CHECK(tight - real_ctx >= floor_b);                 // =0 boots
    CHECK(tight < real_ctx + copy + floor_b);           // the cut would not
    CHECK(!pxa_head_dual_fits(tight, copy, kv, comp, floor_b));
    CHECK(!pxa_head_dual_fits(copy, copy, kv, comp, floor_b));
    CHECK(!pxa_head_dual_fits(0, copy, kv, comp, floor_b));

    // (3) every free size the guard arms on leaves the floor after the copy and the measured context
    for (size_t f = 0; f <= 4096 * MiB; f += MiB) {
        if (pxa_head_dual_fits(f, copy, kv, comp, floor_b)) {
            CHECK(f >= copy + real_ctx + floor_b);
            if (f < copy + real_ctx + floor_b) break;
        }
    }

    // (4)
    const size_t need = pxa_head_dual_need_bytes(copy, kv, comp, floor_b);
    CHECK(need == copy + kv + comp + floor_b + (size_t) PXA_HEAD_DUAL_SLACK_MB * MiB);
    CHECK(pxa_head_dual_fits(need, copy, kv, comp, floor_b));
    CHECK(!pxa_head_dual_fits(need - 1, copy, kv, comp, floor_b));

    // the floor is honoured as given (PXA_POOL_FLOOR_MB=0 lowers the need by exactly the floor)
    CHECK(pxa_head_dual_need_bytes(copy, kv, comp, 0) + floor_b == need);

    if (g_fail) { fprintf(stderr, "test-pxa-head-dual-fit: %d failure(s)\n", g_fail); return 1; }
    printf("test-pxa-head-dual-fit: OK (35B pair arms: need %.1f MiB of %.1f free; tight =0-fits card refuses)\n",
           need / (double) MiB, (double) (15928 - 9060));
    return 0;
}
