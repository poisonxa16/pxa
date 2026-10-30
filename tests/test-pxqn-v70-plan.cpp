// Host checks for the sm_70 plan decisions. No CUDA, no GPU.
#include "pxqn-v70-plan.h"

#include <cstdio>

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); ++g_fail; } } while (0)

int main() {
    using namespace pxqn_v70;

    CHECK(copy_wins(0, 3) == 0);
    CHECK(copy_wins(4, 0) == 0);
    CHECK(copy_wins(2, 3) == -1);
    CHECK(copy_wins(3, 3) == 1);
    CHECK(copy_wins(8, 3) == 1);

    CHECK(ckpt_ub_next(2048, 4, 4, 80) == 0);
    CHECK(ckpt_ub_next(2048, 1, 4, 80) == 1920);
    CHECK(ckpt_ub_next(1920, 1, 4, 80) == 1600);
    CHECK(ckpt_ub_next(320, 1, 4, 80) == 0);
    CHECK(ckpt_ub_next(2048, 1, 4, 0) == 0);
    CHECK(ckpt_ub_next(256, 1, 4, 80) == 0);

    const size_t row = 256 * 2; // one f16 head-row stand-in
    const size_t margin = 256ull << 20;
    const int64_t n_kv = 8192;
    const size_t whole = row * 2 * (size_t) n_kv;
    CHECK(fa_kv_chunk(n_kv, (int64_t) row, 2, whole + margin, margin) == 0);
    CHECK(fa_kv_chunk(n_kv, (int64_t) row, 2, margin, margin) == -1);
    const size_t budget = row * 2 * 1000;
    const int chunk = fa_kv_chunk(n_kv, (int64_t) row, 2, budget + margin, margin);
    CHECK(chunk == 768); // 1000 rows snap down to 768, a multiple of 256
    CHECK(fa_kv_chunk(0, (int64_t) row, 2, budget, margin) == 0);

    if (g_fail) {
        std::fprintf(stderr, "%d checks failed\n", g_fail);
        return 1;
    }
    std::printf("pxqn-v70-plan: ok\n");
    return 0;
}
