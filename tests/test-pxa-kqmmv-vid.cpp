// The KQMMV verify-identity width rule (ggml/src/ggml-cuda/pxa/kq-mmv-vid.h), pure host, no GPU.
// PXA_KQMMV_VERIFY_ID_SMALL is an on/off switch (the registry sets "1" for nemotron_h_moe and qwen4exp): "1" must mean ON
// (Nemotron's small attention k/v launches keep KQMMV at verify widths 3..4), never a width cap of 1. PXA_KQMMV_VERIFY_ID_SMALL_W
// (default 8) widens launches of <= 1024 rows only while the switch is on; with it off every launch keeps the base cap (4).
#include "../ggml/src/ggml-cuda/pxa/kq-mmv-vid.h"

#include <cstdio>
#include <initializer_list>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

int main() {
    const int MAXNY = 8, BASE = 4;   // KQMMV_MAX_NY, PXA_KQMMV_VERIFY_ID default
    // the switch: the registry's "1" is ON; unset / empty / "0" are OFF
    CHECK( pxa_kqmmv_vid_small_parse("1"));
    CHECK( pxa_kqmmv_vid_small_parse("8"));
    CHECK(!pxa_kqmmv_vid_small_parse("0"));
    CHECK(!pxa_kqmmv_vid_small_parse(""));
    CHECK(!pxa_kqmmv_vid_small_parse(nullptr));
    // the width: default 8, clamp, <= 0 = base
    CHECK(pxa_kqmmv_vid_small_w_parse(nullptr, MAXNY) == 8);
    CHECK(pxa_kqmmv_vid_small_w_parse("",      MAXNY) == 8);
    CHECK(pxa_kqmmv_vid_small_w_parse("6",     MAXNY) == 6);
    CHECK(pxa_kqmmv_vid_small_w_parse("99",    MAXNY) == MAXNY);
    CHECK(pxa_kqmmv_vid_small_w_parse("0",     MAXNY) == 0);
    const int W = pxa_kqmmv_vid_small_w_parse(nullptr, MAXNY);

    // Nemotron-H MoE under the registry (PXA_KQMMV_VERIFY_ID_SMALL=1): attention k/v (512 rows) at verify widths 3..4 take the
    // small grouped path -- the "1" is not read as a cap of 1
    {
        const bool on = pxa_kqmmv_vid_small_parse("1");
        const int cap = pxa_kqmmv_vid_cap(BASE, on, W, 512);
        CHECK(cap == 8);
        CHECK(pxa_kqmmv_vid_small_grouped(on, cap, 3));
        CHECK(pxa_kqmmv_vid_small_grouped(on, cap, 4));
        CHECK(!pxa_kqmmv_vid_small_grouped(on, cap, 1));
        // a grouped launch over 1024 rows keeps the base cap: widths 3..4 still on, 5 off
        const int cap2 = pxa_kqmmv_vid_cap(BASE, on, W, 2048);
        CHECK(cap2 == BASE);
        CHECK(pxa_kqmmv_vid_small_grouped(on, cap2, 4));
        CHECK(!pxa_kqmmv_vid_small_grouped(on, cap2, 5));
    }
    // qwen4exp (Hemlock) under the registry: DeltaNet alpha/beta (96 rows) at verify widths 5..8 are served
    {
        const bool on = true;
        for (int ny = 2; ny <= 8; ++ny) CHECK(ny <= pxa_kqmmv_vid_cap(BASE, on, W, 96));
        CHECK(pxa_kqmmv_vid_cap(BASE, on, W, -1) == BASE);                                       // rows unknown
        CHECK(pxa_kqmmv_vid_cap(BASE, on, pxa_kqmmv_vid_small_w_parse("0", MAXNY), 96) == BASE); // _W=0
        CHECK(pxa_kqmmv_vid_cap(0, on, W, 96) == 0);                                             // PXA_KQMMV_VERIFY_ID=0
    }
    // switch off (every other arch, e.g. the 27B W3): the previous build's rule exactly -- base cap, no small grouped path
    {
        const bool on = pxa_kqmmv_vid_small_parse(nullptr);
        for (long R : {48L, 96L, 512L, 1024L, 2048L, 8192L, -1L}) CHECK(pxa_kqmmv_vid_cap(BASE, on, W, R) == BASE);
        for (int ny = 1; ny <= 8; ++ny) CHECK(!pxa_kqmmv_vid_small_grouped(on, pxa_kqmmv_vid_cap(BASE, on, W, 512), ny));
    }
    printf("test-pxa-kqmmv-vid: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
