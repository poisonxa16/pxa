// test-pxq-ksplit.cpp -- produce-and-compare for the PXQ K-split, one row per tier.
//
// THE CLAIM UNDER TEST. Cutting a panel-addressed PXQ tensor along dim 0 (K) is a pure byte
// permutation: each shard gets the 128-byte panel header verbatim and then its own whole slabs, so
// every element that survives the cut dequantises to the SAME BITS it had in the unsplit tensor.
// Until now that claim was verified for PXQ4 only, which is why PXQ4 was the only tier a tensor
// split would cut on K -- every other tier refused at load. This test is what the other five rows
// of that table were waiting for.
//
// HOW IT IS TESTED. For each tier, and for 2-way and 4-way cuts:
//   1. build a panel-addressed buffer of the tier's own geometry and fill it with deterministic
//      pseudo-random bytes, with finite fp16 values in the panel headers (the anchors);
//   2. slice it with pxa_pxq_k_slice_2d() -- the function the CUDA split uploader itself calls, not
//      a copy of it;
//   3. dequantise the unsplit tensor and every shard with the CPU panel decoder;
//   4. compare each shard's output, element by element, against the SAME COLUMNS of the unsplit
//      output, BIT FOR BIT (memcmp on the float bits, so a NaN compares equal to itself).
// Any difference at all is a failure: this is not an approximation test.
//
// It needs no model file, no GGUF and no GPU, so it runs anywhere the build runs.

#include "ggml.h"
#include "pxa-pxq-slice.h"
#include "pxq-cpu.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>

#define PANEL_ROWS 64

static uint32_t rng_state = 0x9e3779b9u;
static uint32_t rnd() {
    // xorshift32: deterministic, so a failure is reproducible from the seed alone
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

// fp16 bit pattern with a finite, well-scaled value: sign, exponent in [0x0c,0x11] (roughly
// 2^-6 .. 2^2), random mantissa. Keeps the anchors out of inf/NaN so a failure prints a readable
// number rather than a nan.
static uint16_t rnd_half() {
    const uint32_t r = rnd();
    const uint16_t sign = (uint16_t) ((r & 1u) << 15);
    const uint16_t exp  = (uint16_t) ((0x0c + (r >> 1) % 6) << 10);
    const uint16_t mant = (uint16_t) ((r >> 8) & 0x3ff);
    return (uint16_t) (sign | exp | mant);
}

struct tier {
    ggml_type   type;
    const char * name;
};

// Fill a panel-addressed [k x nrows] buffer: the 128-byte header of every panel gets finite fp16
// anchors, everything else gets random bytes. The codes and sub-scale nibbles are just bytes to the
// dequant, so random is as good as real and is harder on the slicer than a real file would be.
static void fill_panels(std::vector<uint8_t> & buf, ggml_type type, int64_t nrows, int64_t k) {
    const int64_t row_size  = (int64_t) ggml_row_size(type, k);
    const int64_t bs        = ggml_blck_size(type);
    const int64_t ts        = ggml_type_size(type);
    const int64_t row_meta  = (int64_t) ggml_row_size(type, bs) - ts;
    const int64_t panel_hdr = PANEL_ROWS * row_meta;

    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = (uint8_t) (rnd() & 0xff);
    }
    for (int64_t p = 0; p < nrows; p += PANEL_ROWS) {
        uint8_t * hdr = buf.data() + p * row_size;
        for (int64_t r = 0; r < panel_hdr / 2; ++r) {
            const uint16_t h = rnd_half();
            memcpy(hdr + 2 * r, &h, sizeof(h));
        }
    }
}

static bool check_tier(const tier & t, int64_t nrows, int64_t k, int nsplit, bool verbose) {
    const int64_t bs = ggml_blck_size(t.type);
    if (k % (bs * nsplit) != 0) {
        printf("  %-7s k=%-5lld nsplit=%d : SKIP (k is not divisible by %lld*%d)\n",
                t.name, (long long) k, nsplit, (long long) bs, nsplit);
        return true;
    }

    const int64_t row_size = (int64_t) ggml_row_size(t.type, k);
    std::vector<uint8_t> src((size_t) (nrows * row_size));
    fill_panels(src, t.type, nrows, k);

    // reference: dequantise the whole thing
    std::vector<float> ref((size_t) (nrows * k));
    pxa_pxq_dequant_2d(t.type, src.data(), ref.data(), nrows, k);

    const int64_t k_shard = k / nsplit;
    const int64_t shard_row_size = (int64_t) ggml_row_size(t.type, k_shard);

    int64_t n_checked = 0;
    for (int s = 0; s < nsplit; ++s) {
        const int64_t k_off = s * k_shard;

        std::vector<uint8_t> shard((size_t) (nrows * shard_row_size));
        pxa_pxq_k_slice_2d(t.type, src.data(), shard.data(), nrows, k, k_off, k_shard, PANEL_ROWS);

        std::vector<float> got((size_t) (nrows * k_shard));
        pxa_pxq_dequant_2d(t.type, shard.data(), got.data(), nrows, k_shard);

        for (int64_t r = 0; r < nrows; ++r) {
            const float * a = ref.data() + r * k + k_off;
            const float * b = got.data() + r * k_shard;
            if (memcmp(a, b, (size_t) k_shard * sizeof(float)) != 0) {
                // find the first differing column so the failure names a coordinate
                for (int64_t c = 0; c < k_shard; ++c) {
                    if (memcmp(a + c, b + c, sizeof(float)) != 0) {
                        printf("  %-7s k=%-5lld nsplit=%d : FAIL at row %lld, shard %d, column %lld "
                               "(k=%lld): unsplit %.9g, shard %.9g\n",
                                t.name, (long long) k, nsplit, (long long) r, s, (long long) c,
                                (long long) (k_off + c), (double) a[c], (double) b[c]);
                        return false;
                    }
                }
            }
            n_checked += k_shard;
        }
    }

    if (verbose) {
        printf("  %-7s k=%-5lld nsplit=%d : OK  (%lld elements bit-identical, shard row %lld B, "
               "anchor duplication %+.4f%%)\n",
                t.name, (long long) k, nsplit, (long long) n_checked, (long long) shard_row_size,
                100.0 * ((double) nsplit * nrows * shard_row_size / (double) (nrows * row_size) - 1.0));
    }
    return true;
}

int main(int argc, char ** argv) {
    const bool verbose = argc < 2 || strcmp(argv[1], "-q") != 0;

    const tier tiers[] = {
        { GGML_TYPE_PXQ1,   "PXQ1"   },
        { GGML_TYPE_PXQ2,   "PXQ2"   },
        { GGML_TYPE_PXQ3,   "PXQ3"   },
        { GGML_TYPE_PXQ4,   "PXQ4"   },
        { GGML_TYPE_PXQ4HQ, "PXQ4HQ" },
        { GGML_TYPE_PXQ6,   "PXQ6"   },
    };

    // Shapes: one panel, two panels and three panels, with a small K and with both of the K values
    // the split policy actually cuts on Qwen3.8-27B -- 6144 (attn_output) and 17408 (ffn_down).
    // Three panels is deliberate: an odd panel count catches a slicer that assumed a whole number
    // of panels per shard, which is a different mistake from assuming a whole number of slabs.
    const int64_t shapes[][2] = {
        {  64,   256 },
        { 128,  6144 },
        { 192,   384 },
        { 128, 17408 },
    };

    int n_fail = 0;
    int n_run  = 0;
    for (const auto & t : tiers) {
        if (!pxa_pxq_is_cpu_supported(t.type)) {
            printf("  %-7s : SKIP (no CPU panel decoder, cannot produce-and-compare)\n", t.name);
            continue;
        }
        for (const auto & s : shapes) {
            for (int nsplit : { 2, 4 }) {
                ++n_run;
                if (!check_tier(t, s[0], s[1], nsplit, verbose)) {
                    ++n_fail;
                }
            }
        }
    }

    printf("\ntest-pxq-ksplit: %d case(s), %d failure(s)\n", n_run, n_fail);
    return n_fail == 0 ? 0 : 1;
}
