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
#include <utility>
#include <algorithm>

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


// ---------------------------------------------------------------------------------------------
// THE EXPLICIT-RANGES ARM (ssm_out). The DeltaNet armer does not cut ssm_out into one contiguous K
// block per device: with repeat_type 1 a device owns gqa_ratio separated head groups. The uploader
// slices that with pxa_pxq_k_slice_ranges_2d(), and this block proves it the same way -- shard
// through the shipping function, dequantise, compare bit-for-bit against the same columns (in range
// order) of the unsplit tensor.
//
// The ranges are generated with the SAME arithmetic as prepare_delta_split(ttype 3, ...) in
// src/llama-load-tensors.cpp, for the real DeltaNet geometries.

typedef std::vector<std::pair<int,int>> range_list;

static std::vector<range_list> delta_ranges_ssm_out(int repeat_type, int num_k_heads, int gqa_ratio,
        int head_v_dim, const std::vector<int> & split) {
    std::vector<range_list> out(split.size());
    int first = 0;
    for (size_t is = 0; is < split.size(); ++is) {
        const int s = split[is];
        if (!s) continue;
        const int m = head_v_dim;   // ttype 3: multiplied by head_v_dim
        if (repeat_type == 0) {
            out[is].push_back({first*gqa_ratio*m, s*gqa_ratio*m});
        } else {
            for (int j = 0; j < gqa_ratio; ++j) {
                out[is].push_back({(first + j*num_k_heads)*m, s*m});
            }
        }
        first += s;
    }
    return out;
}

static bool check_ranges(const tier & t, const char * label, int64_t nrows, int64_t k,
        const std::vector<range_list> & dev_ranges, bool verbose) {
    const int64_t row_size = (int64_t) ggml_row_size(t.type, k);
    std::vector<uint8_t> src((size_t) (nrows * row_size));
    fill_panels(src, t.type, nrows, k);

    std::vector<float> ref((size_t) (nrows * k));
    pxa_pxq_dequant_2d(t.type, src.data(), ref.data(), nrows, k);

    int64_t n_checked = 0, k_total = 0;
    for (size_t d = 0; d < dev_ranges.size(); ++d) {
        const range_list & rl = dev_ranges[d];
        if (rl.empty()) continue;
        std::vector<int> flat;
        int64_t k_shard = 0;
        for (auto & p : rl) { flat.push_back(p.first); flat.push_back(p.second); k_shard += p.second; }
        k_total += k_shard;
        const int64_t shard_row_size = (int64_t) ggml_row_size(t.type, k_shard);
        std::vector<uint8_t> shard((size_t) (nrows * shard_row_size));
        const int64_t k_got = pxa_pxq_k_slice_ranges_2d(t.type, src.data(), shard.data(), shard.size(), nrows, k,
                flat.data(), (int) rl.size(), PANEL_ROWS);
        // a range that is not a multiple of the codec's K block (128 for PXQN3/PXQN3S8, whose slabs are
        // 128 K wide) cannot be cut: the only correct answer is the slicer's refusal, and the uploader
        // then aborts naming the tensor instead of loading a wrong shard
        bool aligned = true;
        for (auto & p : rl) aligned = aligned && p.first % ggml_blck_size(t.type) == 0 && p.second % ggml_blck_size(t.type) == 0;
        if (!aligned) {
            if (k_got != PXA_PXQ_SLICE_ERR_RANGE) {
                printf("  %-7s %-26s : FAIL device %zu: a range off the %lld-element block was not refused (rc %lld)\n",
                        t.name, label, d, (long long) ggml_blck_size(t.type), (long long) k_got);
                return false;
            }
            printf("  %-7s %-26s : OK  (refused: ranges are not multiples of the %lld-element block)\n",
                    t.name, label, (long long) ggml_blck_size(t.type));
            return true;
        }
        if (k_got != k_shard) {
            printf("  %-7s %-26s : FAIL device %zu: slicer returned K %lld, expected %lld\n",
                    t.name, label, d, (long long) k_got, (long long) k_shard);
            return false;
        }
        std::vector<float> got((size_t) (nrows * k_shard));
        pxa_pxq_dequant_2d(t.type, shard.data(), got.data(), nrows, k_shard);
        for (int64_t r = 0; r < nrows; ++r) {
            int64_t c0 = 0;
            for (auto & p : rl) {
                const float * a = ref.data() + r * k + p.first;
                const float * b = got.data() + r * k_shard + c0;
                if (memcmp(a, b, (size_t) p.second * sizeof(float)) != 0) {
                    for (int64_t c = 0; c < p.second; ++c) {
                        if (memcmp(a + c, b + c, sizeof(float)) != 0) {
                            printf("  %-7s %-26s : FAIL device %zu row %lld k=%lld: unsplit %.9g, shard %.9g\n",
                                    t.name, label, d, (long long) r, (long long) (p.first + c),
                                    (double) a[c], (double) b[c]);
                            return false;
                        }
                    }
                }
                c0 += p.second;
                n_checked += p.second;
            }
        }
    }
    if (k_total != k) {
        printf("  %-7s %-26s : FAIL ranges cover K %lld of %lld\n", t.name, label, (long long) k_total, (long long) k);
        return false;
    }
    if (verbose) {
        printf("  %-7s %-26s : OK  (%lld elements bit-identical over %zu device(s))\n",
                t.name, label, (long long) n_checked, dev_ranges.size());
    }
    return true;
}

// A row-addressed type through the same function (panel_rows 1) must reproduce, byte for byte, the
// per-row range copy the uploader did before the panel path existed.
static bool check_rowaddr_ranges_bytes(bool verbose) {
    const ggml_type type = GGML_TYPE_Q8_0;
    const int64_t nrows = 7, k = 6144;
    const int64_t bs = ggml_blck_size(type), ts = ggml_type_size(type);
    const int64_t row_size = (int64_t) ggml_row_size(type, k);
    std::vector<uint8_t> src((size_t) (nrows * row_size));
    for (auto & b : src) b = (uint8_t) (rnd() & 0xff);
    const auto dev = delta_ranges_ssm_out(1, 16, 3, 128, {8, 8});
    for (size_t d = 0; d < dev.size(); ++d) {
        std::vector<int> flat; int64_t ks = 0;
        for (auto & p : dev[d]) { flat.push_back(p.first); flat.push_back(p.second); ks += p.second; }
        const int64_t srs = (int64_t) ggml_row_size(type, ks);
        std::vector<uint8_t> got((size_t) (nrows * srs)), want((size_t) (nrows * srs));
        if (pxa_pxq_k_slice_ranges_2d(type, src.data(), got.data(), got.size(), nrows, k, flat.data(), (int) dev[d].size(), 1) != ks) {
            printf("  Q8_0    row-addressed ranges       : FAIL slicer refused\n");
            return false;
        }
        uint8_t * w = want.data();
        for (int64_t r = 0; r < nrows; ++r) {
            for (auto & p : dev[d]) {
                memcpy(w, src.data() + r * row_size + (p.first / bs) * ts, (size_t) ((p.second / bs) * ts));
                w += (p.second / bs) * ts;
            }
        }
        if (memcmp(got.data(), want.data(), got.size()) != 0) {
            printf("  Q8_0    row-addressed ranges       : FAIL device %zu differs from the per-row copy\n", d);
            return false;
        }
    }
    if (verbose) printf("  Q8_0    row-addressed ranges       : OK  (byte-identical to the pre-panel per-row copy)\n");
    return true;
}

// Each refusal of the slicer, probed IN ISOLATION (everything else about the call valid), so each
// check proves the one condition it names; and nothing may be written when it refuses.
static bool check_refusals(bool verbose) {
    const int64_t nrows = 64, k = 256;
    const size_t full = ggml_row_size(GGML_TYPE_PXQ4, k) * nrows;
    std::vector<uint8_t> src(full, 0x5a);
    std::vector<uint8_t> dst(full, 0xee);
    const int good[2]      = {   0, 128 };
    const int misalig[2]   = {  16, 128 };
    const int past_k[2]    = { 128, 160 };
    const size_t need_good = ggml_row_size(GGML_TYPE_PXQ4, 128) * nrows;
    struct probe { const char * what; const int * r; int64_t rows; size_t cap; int64_t want; };
    const probe probes[] = {
        { "misaligned first",   misalig, nrows, full,          PXA_PXQ_SLICE_ERR_RANGE    },
        { "range past K",       past_k,  nrows, full,          PXA_PXQ_SLICE_ERR_RANGE    },
        { "rows not x64",       good,    32,    full,          PXA_PXQ_SLICE_ERR_PANEL    },
        { "dst one byte short", good,    nrows, need_good - 1, PXA_PXQ_SLICE_ERR_CAPACITY },
        { "valid control",      good,    nrows, need_good,     128                        },
    };
    bool ok = true;
    for (const auto & pr : probes) {
        std::fill(dst.begin(), dst.end(), 0xee);
        const int64_t rc = pxa_pxq_k_slice_ranges_2d(GGML_TYPE_PXQ4, src.data(), dst.data(), pr.cap, pr.rows, k, pr.r, 1, PANEL_ROWS);
        bool untouched = true;
        if (rc < 0) for (auto b : dst) if (b != 0xee) { untouched = false; break; }
        const bool pass = rc == pr.want && untouched;
        if (!pass || verbose) {
            printf("  PXQ4    refusal: %-18s : %s (rc %lld, want %lld%s)\n", pr.what, pass ? "OK " : "FAIL",
                    (long long) rc, (long long) pr.want, untouched ? "" : ", dst WRITTEN");
        }
        ok = ok && pass;
    }
    (void) verbose;
    return ok;
}

// DIM 1 (output rows), confirmed rather than assumed. The uploader's dim-1 arms copy whole rows as
// byte ranges (row_size * rows), relying on 64 consecutive rows being exactly one panel. This
// replays both arms -- contiguous shards and explicit row ranges (the wqkv_gate / ssm_beta_alpha
// shape) -- at 64-row granularity and requires every shard to dequantise to the same bits as the
// same rows of the unsplit tensor.
static bool check_dim1(const tier & t, bool verbose) {
    const int64_t nrows = 320, k = 512;   // five panels
    const int64_t row_size = (int64_t) ggml_row_size(t.type, k);
    std::vector<uint8_t> src((size_t) (nrows * row_size));
    fill_panels(src, t.type, nrows, k);
    std::vector<float> ref((size_t) (nrows * k));
    pxa_pxq_dequant_2d(t.type, src.data(), ref.data(), nrows, k);
    // contiguous arm: 128 + 192 rows; ranges arm: device 0 = rows [0,64)+[192,256), device 1 = the rest
    const std::vector<range_list> layouts[2] = {
        { { {0, 128} }, { {128, 192} } },
        { { {0, 64}, {192, 64} }, { {64, 128}, {256, 64} } },
    };
    for (int li = 0; li < 2; ++li) {
        for (const auto & dev : layouts[li]) {
            int64_t n = 0; for (auto & p : dev) n += p.second;
            std::vector<uint8_t> shard((size_t) (n * row_size));
            uint8_t * d = shard.data();
            for (auto & p : dev) {
                memcpy(d, src.data() + p.first * row_size, (size_t) (p.second * row_size));
                d += p.second * row_size;
            }
            std::vector<int> flat;
            for (auto & p : dev) { flat.push_back(p.first); flat.push_back(p.second); }
            if (pxa_pxq_row_ranges_check(nrows, flat.data(), (int) dev.size(), PANEL_ROWS) != n) {
                printf("  %-7s dim-1 : FAIL the shipping validator refused a panel-aligned layout\n", t.name);
                return false;
            }
            std::vector<float> got((size_t) (n * k));
            pxa_pxq_dequant_2d(t.type, shard.data(), got.data(), n, k);
            int64_t r0 = 0;
            for (auto & p : dev) {
                if (memcmp(ref.data() + p.first * k, got.data() + r0 * k, (size_t) (p.second * k) * sizeof(float)) != 0) {
                    printf("  %-7s dim-1 %-20s : FAIL rows [%d,+%d)\n", t.name, li ? "row ranges" : "contiguous", p.first, p.second);
                    return false;
                }
                r0 += p.second;
            }
        }
    }
    // a range that halves a panel must be refused by the validator the uploader calls
    const int half[4] = { 0, 96, 96, 224 };
    if (pxa_pxq_row_ranges_check(nrows, half, 2, PANEL_ROWS) != PXA_PXQ_SLICE_ERR_PANEL) {
        printf("  %-7s dim-1 : FAIL a 96-row range was not refused\n", t.name);
        return false;
    }
    if (verbose) printf("  %-7s dim-1 contiguous+ranges     : OK  (5 panels, 64-row granularity, bit-identical)\n", t.name);
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
        { GGML_TYPE_PXQN3,   "PXQN3"   },
        { GGML_TYPE_PXQN3S8, "PXQN3S8" },
        { GGML_TYPE_PXQN4,   "PXQN4"   },
        { GGML_TYPE_PXQN2,   "PXQN2"   },   // the PXQN ladder (spec section 8)
        { GGML_TYPE_PXQN1,   "PXQN1"   },
        { GGML_TYPE_PXQN4S8, "PXQN4S8" },
        { GGML_TYPE_PXQN5,   "PXQN5"   },
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

    // Explicit-ranges arm (ssm_out). Geometries: Qwen3.8-27B ssm_out [6144 x 5120] = 16 k-heads x
    // gqa 3 x head_v_dim 128, 2-way and 4-way, both repeat types, even and uneven head splits; and
    // the Flash-Next ssm_out [6144 x 2560] 4-way (4 x 1536). Rows: three panels (odd count).
    struct rcase { const char * label; int repeat, nk, gqa, hv; std::vector<int> split; int64_t rows; };
    const rcase rcases[] = {
        { "27B ssm_out r1 2x(8,8) 5120r", 1, 16, 3, 128, { 8, 8 },       5120 },   // real 27B rows
        { "27B ssm_out r0 2x(8,8)",       0, 16, 3, 128, { 8, 8 },        192 },
        { "27B ssm_out r1 2x(5,11)",      1, 16, 3, 128, { 5, 11 },       192 },
        { "27B ssm_out r1 4x(3,3,3,7)",   1, 16, 3, 128, { 3, 3, 3, 7 },  192 },   // uneven 4-way
        { "FN ssm_out r1 4x1536 2560r",   1, 16, 3, 128, { 4, 4, 4, 4 }, 2560 },   // real Flash-Next rows
        { "FN ssm_out r0 4x1536",         0, 16, 3, 128, { 4, 4, 4, 4 },  192 },
        { "Ornith ssm_out r1 2x(8,8)",    1, 16, 2, 128, { 8, 8 },        192 },   // [4096 x 2048]
        { "hv64 r1 3x(3,0,5)",            1,  8, 2,  64, { 3, 0, 5 },     192 },
    };
    for (const auto & t : tiers) {
        if (!pxa_pxq_is_cpu_supported(t.type)) continue;
        for (const auto & rc : rcases) {
            int nk_sum = 0; for (int x : rc.split) nk_sum += x;
            const int64_t k = (int64_t) nk_sum * rc.gqa * rc.hv;
            ++n_run;
            if (!check_ranges(t, rc.label, rc.rows, k,
                    delta_ranges_ssm_out(rc.repeat, rc.nk, rc.gqa, rc.hv, rc.split), verbose)) {
                ++n_fail;
            }
        }
    }
    for (const auto & t : tiers) {
        if (!pxa_pxq_is_cpu_supported(t.type)) continue;
        ++n_run;
        if (!check_dim1(t, verbose)) ++n_fail;
    }
    ++n_run; if (!check_rowaddr_ranges_bytes(verbose)) ++n_fail;
    ++n_run; if (!check_refusals(verbose))             ++n_fail;

    printf("\ntest-pxq-ksplit: %d case(s), %d failure(s)\n", n_run, n_fail);
    return n_fail == 0 ? 0 : 1;
}
