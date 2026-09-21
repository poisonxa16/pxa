// FLASH-ATTENTION BAND ALIGNMENT: the running softmax state must be accumulated in fp32.
//
// WHAT THIS PINS. On the no-tensor-core cards the batched flash-attention path is
// flash_attn_tile_ext_f16 (ggml/src/ggml-cuda/fattn-tile-f16.cu). It walks the key range in
// 64-cell tiles and keeps three pieces of running state: the running maximum kqmax, the softmax
// denominator kqsum, and the output numerator VKQ. In the original instantiation all three were
// HALF, and that is the whole defect this test exists for:
//
//   * WHICH key lands in WHICH tile is decided by the key's CELL index, not by its position in
//     the sequence. A request's cells are laid down wherever the unified ring has room, so the
//     same prompt tiles differently depending on what another sequence was still holding.
//   * kqsum is not one accumulator but 64 of them -- one per key-slot within a tile (the kernel
//     gives lane `tid` the half2 at slot `tid`, i.e. keys 2*tid and 2*tid+1 of every tile) --
//     folded at the very end by a butterfly sum, also in half. Move the band and every key
//     changes slot.
//   * VKQ is rescaled by exp(max_old - max_new) at every tile boundary, and the boundaries fall
//     between different keys once the band moves.
//
// So in half the ROUNDING of a twenty-thousand-term accumulation is a function of where the band
// starts, and at fp16's precision that rounding is not small: at a running sum of 2048 the gap
// between representable values is already 2.0. In fp32 the same reassociation is invisible.
// A shift of ONE cell keeps 63 of every 64 tile members together and barely moves the result; a
// shift deep into a tile rescrambles every tile and moves it by orders of magnitude more -- which
// is exactly the shape the 4x P100 seat showed on a 20,859-token prompt, where the same prompt
// answered correctly from cell 0 and returned nothing from cell 101.
//
// WHAT IT DOES. It mirrors the kernel's arithmetic term for term -- the tile walk, the 64 kqsum
// slots, the per-tile rescale, the accumulation order over keys, the butterfly fold -- once with
// fp32 running state and once with fp16 running state, and lays the same key band down at a set
// of starting cells. Then:
//
//   1. THE FP32 ARM IS PLACEMENT-INVARIANT. Every offset must agree with the offset-0 result to
//      fp32 round-off, and with a whole-range double-precision reference. This is the property
//      the engine promises and the one the fix restores.
//   2. THE FP16 ARM IS NOT, AND THE TEST SAYS BY HOW MUCH. The retired arithmetic is kept here
//      deliberately, the way test-kv-cell-max keeps the retired sampled scan, so the defect stays
//      visible and measurable instead of becoming folklore. The test requires the fp16 spread to
//      be at least an order of magnitude worse than the fp32 one -- if that ever stops being
//      true the comparison has gone stale and someone should look at why.
//
// Deterministic inputs from a fixed LCG. CPU only, no GPU, no model.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define TILE   64   // FATTN_KQ_STRIDE_TILE_F16
#define SLOTS  64   // one kqsum accumulator per key-slot in a tile (32 lanes x half2)

struct lcg {
    uint64_t s;
    explicit lcg(uint64_t seed) : s(seed) {}
    uint32_t next() { s = s*6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t) (s >> 33); }
    float unit() { return (float) next() / (float) 0x80000000u - 1.0f; } // [-1, 1)
};

// fp32 -> fp16 -> fp32, round to nearest even. Written out here rather than called from ggml so
// the test carries no initialisation order of its own (ggml's fp16 table is filled by ggml_init).
static float to_half_and_back(float x) {
    if (!std::isfinite(x)) {
        return x;
    }
    const float ax = fabsf(x);
    if (ax >= 65520.0f) {                       // rounds to fp16 infinity
        return x > 0.0f ? INFINITY : -INFINITY;
    }
    if (ax < 6.103515625e-05f) {                // fp16 subnormals: multiples of 2^-24
        return rintf(x*16777216.0f)/16777216.0f;
    }
    int e = 0;
    frexpf(ax, &e);                             // ax = m * 2^e, m in [0.5, 1)
    const float ulp = ldexpf(1.0f, e - 11);     // fp16 keeps 11 significant bits
    return rintf(x/ulp)*ulp;                    // rintf is round-to-nearest-even
}

// round a value through fp16 when the arm asks for it, so a "half" accumulator really is one
static inline float rnd(float x, bool half) {
    return half ? to_half_and_back(x) : x;
}

// The kernel's online softmax, mirrored. `scores` and `V` are the band's own keys in sequence
// order; the band occupies cells [offset, offset + n) of a key range padded up to `extent`, and
// every other cell is masked. Returns the normalised output row.
static std::vector<double> fattn_tile(const std::vector<float> & scores,
                                      const std::vector<float> & V, int Dv,
                                      int offset, int extent, bool half) {
    const int n = (int) scores.size();

    std::vector<float> acc(SLOTS, 0.0f);          // kqsum, one per key-slot
    std::vector<float> VKQ((size_t) Dv, 0.0f);    // output numerator
    float kqmax = half ? -65504.0f/2.0f : -3.402823466e+38f/2.0f;

    for (int t0 = 0; t0 < extent; t0 += TILE) {
        // the tile's scores, with the mask applied. Scores are half in both arms: the kernel
        // computes them in half registers whatever the accumulators are.
        float tile[TILE];
        bool  any = false;
        for (int k = 0; k < TILE; ++k) {
            const int cell = t0 + k;
            const bool mine = cell >= offset && cell < offset + n;
            tile[k] = mine ? to_half_and_back(scores[cell - offset]) : -INFINITY;
            any = any || mine;
        }
        if (!any) {
            continue;                              // PXA_FA_MASK_SKIP_TILE: a fully masked tile
        }

        float kqmax_new = kqmax;
        for (int k = 0; k < TILE; ++k) {
            if (tile[k] > kqmax_new) kqmax_new = tile[k];
        }
        const float scale = rnd(expf(kqmax - kqmax_new), half);
        kqmax = kqmax_new;

        float val[TILE];
        for (int k = 0; k < TILE; ++k) {
            val[k] = std::isfinite(tile[k]) ? rnd(expf(tile[k] - kqmax), half) : 0.0f;
        }
        for (int k = 0; k < TILE; ++k) {
            acc[k] = rnd(rnd(acc[k]*scale, half) + val[k], half);
        }
        for (int d = 0; d < Dv; ++d) {
            VKQ[d] = rnd(VKQ[d]*scale, half);
        }
        // the kernel walks the tile's keys in cell order, one fused multiply-add per key
        for (int k = 0; k < TILE; ++k) {
            const int cell = t0 + k;
            if (cell < offset || cell >= offset + n) {
                continue;                          // exp() of a masked key is exactly zero
            }
            const float * v = &V[(size_t) (cell - offset)*Dv];
            for (int d = 0; d < Dv; ++d) {
                VKQ[d] = rnd(VKQ[d] + rnd(v[d]*val[k], half), half);
            }
        }
    }

    // fold the two halves of each lane's half2, then the 32-lane butterfly -- in the arm's type
    std::vector<float> lane(SLOTS/2);
    for (int l = 0; l < SLOTS/2; ++l) {
        lane[l] = rnd(acc[2*l] + acc[2*l + 1], half);
    }
    for (int step = (SLOTS/2)/2; step >= 1; step /= 2) {
        for (int l = 0; l < step; ++l) {
            lane[l] = rnd(lane[l] + lane[l + step], half);
        }
    }
    const float S = lane[0];

    std::vector<double> out((size_t) Dv);
    for (int d = 0; d < Dv; ++d) {
        out[d] = (double) VKQ[d] / (double) S;
    }
    return out;
}

// whole-range softmax in double precision: what the kernel is trying to compute
static std::vector<double> reference(const std::vector<float> & scores, const std::vector<float> & V, int Dv) {
    const int n = (int) scores.size();
    double m = -INFINITY;
    std::vector<double> s((size_t) n);
    for (int i = 0; i < n; ++i) {
        s[i] = (double) to_half_and_back(scores[i]); // same score input as the arms
        if (s[i] > m) m = s[i];
    }
    double sum = 0.0;
    std::vector<double> out((size_t) Dv, 0.0);
    for (int i = 0; i < n; ++i) {
        const double p = exp(s[i] - m);
        sum += p;
        for (int d = 0; d < Dv; ++d) {
            out[d] += p*(double) V[(size_t) i*Dv + d];
        }
    }
    for (int d = 0; d < Dv; ++d) {
        out[d] /= sum;
    }
    return out;
}

static double reldiff(const std::vector<double> & a, const std::vector<double> & b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = fabs(a[i] - b[i]);
        if (d > num) num = d;
        if (fabs(b[i]) > den) den = fabs(b[i]);
    }
    return den > 0.0 ? num/den : num;
}

int main(void) {
    const int N   = 20859;   // the seat's own long prompt, in tokens
    const int Dv  = 16;      // the value dimensions are independent; sixteen is enough to see it
    const int PAD = 256;     // llama_kv_cache_get_padding() under flash attention

    lcg r(20260909ull);
    std::vector<float> scores((size_t) N);
    std::vector<float> V((size_t) N*Dv);
    for (int i = 0; i < N; ++i) {
        scores[i] = 1.5f*r.unit();                       // a diffuse long-context attention row
    }
    for (int i = 0; i < 12; ++i) {
        scores[(size_t) (r.next() % (uint32_t) N)] += 8.0f; // with a handful of strong keys
    }
    for (size_t i = 0; i < V.size(); ++i) {
        V[i] = r.unit();
    }

    // 0 and 4096 start on the attention window's own grain, 1/257/4097 sit one cell past it, and
    // 101 / 20833 land deep inside a tile -- the placements the seat answered differently at.
    const int offsets[] = { 0, 1, 101, 257, 4097, 20833 };
    const int n_off     = (int) (sizeof(offsets)/sizeof(offsets[0]));

    const std::vector<double> ref = reference(scores, V, Dv);

    printf("test-fa-band-align: the tile flash-attention running state must be fp32\n");
    printf("  band = %d keys, Dv = %d, tile = %d, window grain = %d\n", N, Dv, TILE, PAD);

    double worst[2] = { 0.0, 0.0 };   // [0] = fp32 arm, [1] = fp16 arm
    double vs_ref[2] = { 0.0, 0.0 };
    int    fail = 0;

    for (int arm = 0; arm < 2; ++arm) {
        const bool half = arm == 1;
        printf("  %s accumulators\n", half ? "fp16 (the retired arithmetic)" : "fp32 (the shipped arithmetic)");
        std::vector<double> base;
        for (int a = 0; a < n_off; ++a) {
            const int off    = offsets[a];
            const int extent = ((off + N + PAD - 1)/PAD)*PAD;
            const std::vector<double> out = fattn_tile(scores, V, Dv, off, extent, half);
            const double vr = reldiff(out, ref);
            if (vr > vs_ref[arm]) vs_ref[arm] = vr;
            if (a == 0) {
                base = out;
                printf("    offset %-6d rel to the double-precision reference = %.3e   (reference placement)\n", off, vr);
                continue;
            }
            const double d = reldiff(out, base);
            if (d > worst[arm]) worst[arm] = d;
            printf("    offset %-6d rel to offset 0 = %.3e   (vs reference %.3e)\n", off, d, vr);
        }
    }

    // 1. the shipped arithmetic is placement-invariant, and right
    const double BOUND = 1e-5;
    if (!(worst[0] < BOUND)) {
        printf("  FAIL: fp32 accumulators are not placement-invariant (%.3e >= %.3e)\n", worst[0], BOUND);
        fail = 1;
    }
    if (!(vs_ref[0] < BOUND)) {
        printf("  FAIL: fp32 accumulators disagree with the double-precision reference (%.3e >= %.3e)\n", vs_ref[0], BOUND);
        fail = 1;
    }
    // 2. and the retired arithmetic was not -- by a margin, so the comparison cannot go stale
    if (!(worst[1] > 10.0*worst[0])) {
        printf("  FAIL: the fp16 arm no longer shows the placement dependence this test exists for "
               "(fp16 %.3e vs fp32 %.3e) -- the mirror has drifted from the kernel\n", worst[1], worst[0]);
        fail = 1;
    }

    printf("  placement spread: fp32 %.3e, fp16 %.3e  (%.0fx)\n",
           worst[0], worst[1], worst[0] > 0.0 ? worst[1]/worst[0] : 0.0);
    printf("%s\n", fail ? "FAIL" : "OK");
    return fail;
}
