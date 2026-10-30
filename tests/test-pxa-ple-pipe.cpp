// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// PXA_PLE_PIPELINE auto scope (src/llama-pxa-ple-pipe.h): on only where measured (copy slots + exactly 4 sm_60 cards),
// =0 / =1 / PXA_REFERENCE=1 as documented, never without copy slots. No model, no GPU.

#include "llama-pxa-ple-pipe.h"

#include <cstdio>

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)

int main() {
    const int p100x4[4] = { 600, 600, 600, 600 };
    const int p100x2[2] = { 600, 600 };
    const int v100x4[4] = { 700, 700, 700, 700 };
    const int mixed4[4] = { 600, 600, 700, 700 };
    const int p40x4[4]  = { 610, 610, 610, 610 };

    // auto: the measured cell only
    CHECK( pxa_ple_pipeline_decide(nullptr, 4, p100x4, 4, false));
    CHECK( pxa_ple_pipeline_decide("",      2, p100x4, 4, false));
    CHECK(!pxa_ple_pipeline_decide(nullptr, 4, p100x2, 2, false));
    CHECK(!pxa_ple_pipeline_decide(nullptr, 4, v100x4, 4, false));
    CHECK(!pxa_ple_pipeline_decide(nullptr, 4, mixed4, 4, false));
    CHECK(!pxa_ple_pipeline_decide(nullptr, 4, p40x4,  4, false));
    CHECK(!pxa_ple_pipeline_decide(nullptr, 4, nullptr, 0, false));   // CPU build / no GPU

    // no copy slots: never, whatever is asked (the barrier is the only ordering then)
    CHECK(!pxa_ple_pipeline_decide(nullptr, 1, p100x4, 4, false));
    CHECK(!pxa_ple_pipeline_decide("1",     1, p100x4, 4, false));
    CHECK(!pxa_ple_pipeline_decide("1",     0, v100x4, 4, false));

    // off switch and force
    CHECK(!pxa_ple_pipeline_decide("0",     4, p100x4, 4, false));
    CHECK( pxa_ple_pipeline_decide("1",     4, v100x4, 4, false));
    CHECK( pxa_ple_pipeline_decide("1",     2, p100x2, 2, false));
    CHECK( pxa_ple_pipeline_decide("1",     2, nullptr, 0, false));

    // PXA_REFERENCE=1: off unless the lever is named
    CHECK(!pxa_ple_pipeline_decide(nullptr, 4, p100x4, 4, true));
    CHECK( pxa_ple_pipeline_decide("1",     4, p100x4, 4, true));

    if (g_fail) { fprintf(stderr, "test-pxa-ple-pipe: %d FAILED\n", g_fail); return 1; }
    printf("test-pxa-ple-pipe: OK\n");
    return 0;
}
