// PXQN activation dump (examples/imatrix/pxqn-dump.h): fp32 -> fp16 never writes inf. Values within +-65504 convert bit-identically to
// ggml_fp32_to_fp16_row (the dump of a run without overflow is byte-identical); larger magnitudes saturate to +-65504, NaN to 0, counted.
// CPU only, no model file.
#include "../examples/imatrix/pxqn-dump.h"
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); ++fails; } } while (0)
static uint16_t bits(ggml_fp16_t h) { uint16_t b; memcpy(&b, &h, 2); return b; }

int main() {
    std::vector<float> tmp;
    // 1. in range: every finite fp16 value, plus 2M random fp32 values of magnitude <= 65504 (all exponents, signs, denormals)
    std::vector<float> x;
    for (uint32_t b = 0; b < 0x10000; ++b) { if ((b & 0x7C00) == 0x7C00) continue; ggml_fp16_t h; uint16_t hb = (uint16_t) b; memcpy(&h, &hb, 2); x.push_back(ggml_fp16_to_fp32(h)); }
    std::mt19937 rng(20261023);
    while (x.size() < 2000000) { uint32_t u = rng(); float f; memcpy(&f, &u, 4); if (std::fabs(f) <= 65504.0f) x.push_back(f); }
    x.push_back(65504.0f); x.push_back(-65504.0f); x.push_back(65503.99f); x.push_back(-0.0f); x.push_back(std::numeric_limits<float>::denorm_min());
    std::vector<ggml_fp16_t> want(x.size()), got(x.size());
    ggml_fp32_to_fp16_row(x.data(), want.data(), (int64_t) x.size());
    const int64_t n0 = pxqn_dump_to_fp16(x.data(), got.data(), (int64_t) x.size(), tmp);
    CHECK(n0 == 0, "in-range values counted as saturated: %lld", (long long) n0);
    CHECK(memcmp(want.data(), got.data(), 2 * x.size()) == 0, "in-range conversion differs from ggml_fp32_to_fp16_row");
    CHECK(tmp.empty(), "scratch used on the fast path");
    // 2. out of range in a row with normal values around it
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> y = { 1.5f, 65520.0f, -70000.0f, 1e6f, inf, -inf, nan, -3.25f, 65504.0f, 2e38f };
    std::vector<ggml_fp16_t> o(y.size());
    const int64_t n1 = pxqn_dump_to_fp16(y.data(), o.data(), (int64_t) y.size(), tmp);
    CHECK(n1 == 7, "saturated count %lld, want 7", (long long) n1);
    const uint16_t exp[] = { 0x3E00, 0x7BFF, 0xFBFF, 0x7BFF, 0x7BFF, 0xFBFF, 0x0000, 0xC280, 0x7BFF, 0x7BFF };
    for (size_t i = 0; i < y.size(); ++i) CHECK(bits(o[i]) == exp[i], "value %zu (%g): fp16 bits %04x want %04x", i, (double) y[i], bits(o[i]), exp[i]);
    for (size_t i = 0; i < y.size(); ++i) CHECK((bits(o[i]) & 0x7C00) != 0x7C00, "inf/NaN written at %zu", i);
    // 3. the plain conversion really does write inf for these (what the dump wrote before)
    std::vector<ggml_fp16_t> p(y.size()); ggml_fp32_to_fp16_row(y.data(), p.data(), (int64_t) y.size());
    CHECK((bits(p[1]) & 0x7C00) == 0x7C00 && (bits(p[3]) & 0x7C00) == 0x7C00, "plain fp16 conversion of 65520 / 1e6 is not inf (test premise)");
    // 4. the routed-expert hook converts one (token, slot) row at a time with one shared scratch: a saturated row leaves the rows after
    //    it bit-identical to the plain conversion (no scratch carry-over), and the per-row counts add up
    {
        const int64_t nc = 64, nr = 4; std::vector<float> z((size_t) (nc * nr));
        for (auto & v : z) { uint32_t u = rng(); v = (float) ((int32_t) u % 20000) * 0.37f; }
        z[1 * nc + 3] = 1e5f; z[1 * nc + 60] = nan; z[3 * nc + 0] = -inf;
        std::vector<ggml_fp16_t> rw((size_t) (nc * nr)), pl((size_t) (nc * nr)); int64_t ns = 0;
        for (int64_t r = 0; r < nr; ++r) ns += pxqn_dump_to_fp16(z.data() + r * nc, rw.data() + r * nc, nc, tmp);
        ggml_fp32_to_fp16_row(z.data(), pl.data(), nc * nr);
        CHECK(ns == 3, "routed rows: saturated %lld, want 3", (long long) ns);
        CHECK(memcmp(rw.data(), pl.data(), 2 * nc) == 0 && memcmp(rw.data() + 2 * nc, pl.data() + 2 * nc, 2 * nc) == 0, "clean routed rows differ from the plain conversion");
        for (int64_t i = 0; i < nc * nr; ++i) CHECK((bits(rw[i]) & 0x7C00) != 0x7C00, "routed rows: inf/NaN written at %lld", (long long) i);
        CHECK(bits(rw[1 * nc + 3]) == 0x7BFF && bits(rw[1 * nc + 60]) == 0 && bits(rw[3 * nc]) == 0xFBFF, "routed rows: saturated values wrong");
    }
    printf("test-pxqn-dump-sat: %s (%d failure(s), %zu in-range values checked)\n", fails ? "FAIL" : "PASS", fails, x.size());
    return fails ? 1 : 0;
}
