// Bug #209 (spec-ckpt-restore-rewinds-sampler-rng): rolling back a speculative step must rewind the
// sampler's token state but NOT its RNG.
//
// The server checkpoints the slot sampler before a verify and restores it after a rejection. It used
// common_sampler_clone(), which also copies the RNG, so after every rejection the next verify drew
// the SAME uniforms again: at temperature > 0 the first token of the next step was drawn at the same
// quantile, and the output silently stopped sampling the model's distribution.
//
// common_sampler is a plain struct and clone only touches its optional sub-samplers when they are
// set, so this runs with no model, no context and no GPU.

#include "sampling.h"

#include <cstdio>
#include <random>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); ++g_fail; } } while (0)

static uint32_t draw(std::mt19937 & rng) { return (uint32_t) rng(); }

int main() {
    const uint32_t seed = 1234;

    // Reference stream: what the slot's RNG should produce draw after draw.
    std::mt19937 ref(seed);
    std::vector<uint32_t> stream;
    for (int i = 0; i < 8; ++i) stream.push_back(draw(ref));

    for (int variant = 0; variant < 2; ++variant) {
        common_sampler * live = new common_sampler();
        common_sampler * ckpt = new common_sampler();
        live->rng.seed(seed);
        live->prev = {11, 12, 13};

        // save_speculative_checkpoint
        common_sampler_clone(live, ckpt);

        // the verify: three draws, three accepted tokens that the rejection has to take back
        for (int i = 0; i < 3; ++i) {
            (void) draw(live->rng);
            live->prev.push_back(99);
        }

        // restore_speculative_checkpoint
        if (variant == 0) {
            common_sampler_rewind_keep_rng(ckpt, live);
        } else {
            common_sampler_clone(ckpt, live); // the pre-fix restore, kept to show the defect
        }

        CHECK(live->prev == std::vector<llama_token>({11, 12, 13}), "token history not rewound");

        const uint32_t next = draw(live->rng);
        if (variant == 0) {
            CHECK(next == stream[3], "rewind must continue the RNG stream (4th draw)");
            CHECK(next != stream[0], "rewind replayed the first draw");
        } else {
            // documents the bug: the full clone replays the checkpoint's first draw
            CHECK(next == stream[0], "expected the old full clone to replay the RNG (defect witness)");
        }

        delete ckpt;
        delete live;
    }

    if (g_fail) {
        fprintf(stderr, "test-spec-rewind-rng: %d failure(s)\n", g_fail);
        return 1;
    }
    printf("test-spec-rewind-rng: OK\n");
    return 0;
}
