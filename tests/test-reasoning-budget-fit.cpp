// PXA 2026-10-26 (#361/#362): common_reasoning_budget_fit_ctx, the thinking budget a slot can still honour.
// A budget larger than the room after the prompt never fires; the server fits it to the room minus an answer
// reserve of min(1024, room/4). Budgets that fit, unlimited (-1) and an unknown n_ctx stay unchanged.
#include "reasoning-budget.h"

#include <cstdio>
#include <cstdlib>

static int n_fail = 0;

static void expect(int32_t budget, int32_t n_ctx, int32_t n_prompt, int32_t want, const char * what) {
    const int32_t got = common_reasoning_budget_fit_ctx(budget, n_ctx, n_prompt);
    if (got != want) {
        fprintf(stderr, "FAIL %s: fit(%d, %d, %d) = %d, want %d\n", what, budget, n_ctx, n_prompt, got, want);
        n_fail++;
    } else {
        printf("ok   %s: fit(%d, %d, %d) = %d\n", what, budget, n_ctx, n_prompt, got);
    }
}

int main() {
    // the bug report's shape: -c 4096, --reasoning-budget 128000, a short chat prompt
    expect(128000, 4096, 100, 2997, "budget over the room -> room minus room/4");
    expect(128000, 4096, 1916, 1635, "long prompt: room 2180, reserve 545");
    // budgets that fit are untouched (output identical to the old engine)
    expect(2000, 4096, 100, 2000, "budget that fits");
    expect(2997, 4096, 100, 2997, "budget exactly at the fit");
    expect(128000, 131072, 87, 128000, "Control's 128000 on a 128k context");
    expect(128000, 262144, 500, 128000, "256k context");
    expect(0, 4096, 100, 0, "budget 0 (thinking off) stays 0");
    // unlimited and unknown context: no change
    expect(-1, 4096, 100, -1, "unlimited");
    expect(128000, 0, 100, 128000, "n_ctx unknown");
    // big context: the reserve caps at 1024
    expect(200000, 65536, 1000, 63512, "reserve capped at 1024");
    // no room at all: close the thought at once
    expect(500, 4096, 4096, 0, "prompt fills the slot");
    expect(500, 4096, 5000, 0, "prompt beyond the slot");
    expect(500, 4096, -3, 500, "negative prompt count treated as 0");
    if (n_fail) {
        fprintf(stderr, "%d case(s) failed\n", n_fail);
        return 1;
    }
    printf("all cases passed\n");
    return 0;
}
