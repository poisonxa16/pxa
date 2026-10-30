// CPU-only: uneven memory-weighted tensor split must land on multiples of the RHT block (128) for the 27B dims.
#include "../src/pxa-tsplit-bounds.h"
#include <cstdio>
#include <cstdlib>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static std::vector<float> cum(const std::vector<float> & w) {
    float t = 0; for (float x : w) t += x;
    std::vector<float> c; float a = 0;
    for (float x : w) { a += x / t; c.push_back(a); }
    c.back() = 1.0f;
    return c;
}

int main() {
    const std::vector<std::vector<float>> weights = {
        {1, 1}, {3401.9f, 3473.7f}, {3, 5}, {1, 1, 1}, {10, 12, 15}, {1, 1, 1, 1},
        {3401.9f, 3401.9f, 3473.7f, 3473.7f}, {20, 20, 25, 25}, {7, 9, 11, 13}, {1, 2, 3, 4},
        {45.7f, 45.7f, 65.1f, 65.1f}, {24, 24, 24, 12}};
    struct dim { const char * name; int nr; int gran; bool rht; } dims[] = {
        {"ffn 17408 g64->rht128", 17408, 64, true},
        {"ffn 17408 g128",        17408, 128, true},
        {"v-heads 48x128",        48,    1,   false},
        {"q-heads 24x256 (g6)",   6144,  1536, false},
        {"kv-heads 4",            4,     1,   false},
        {"n_embd 5120",           5120,  128, true},
    };
    for (auto & w : weights) {
        for (auto & d : dims) {
            int g = pxa_tsplit_rht_granularity(d.gran, d.nr, d.rht);
            if (d.rht) CHECK(g % 128 == 0, "%s gran %d", d.name, g);
            std::vector<size_t> mem(w.size(), 0);
            auto r = pxa_tsplit_create_split(d.nr, g, cum(w), mem);
            int sum = 0, off = 0;
            for (size_t i = 0; i < r.size(); ++i) {
                CHECK(r[i] % g == 0, "%s dev %zu width %d gran %d", d.name, i, r[i], g);
                CHECK(off % g == 0, "%s dev %zu offset %d", d.name, i, off);
                if (d.rht) CHECK(r[i] % 128 == 0 && off % 128 == 0, "%s rht dev %zu", d.name, i);
                sum += r[i]; off += r[i];
            }
            CHECK(sum == d.nr, "%s sum %d != %d", d.name, sum, d.nr);
        }
    }
    // the shipped crash: 4 devs, ffn 17408, gran 64 gave a non-128 width; with the fix it does not.
    {
        std::vector<size_t> mem(4, 0);
        auto c = cum({3401.9f, 3401.9f, 3473.7f, 3473.7f});
        int g = pxa_tsplit_rht_granularity(64, 17408, true);
        CHECK(g == 128, "raised gran %d", g);
        auto r = pxa_tsplit_create_split(17408, g, c, mem);
        printf("uneven 4-way ffn: %d %d %d %d\n", r[0], r[1], r[2], r[3]);
    }
    // equal split: byte-identical boundaries between granularity 64 and 128
    for (int n : {2, 4}) {
        std::vector<float> w(n, 1.0f); std::vector<size_t> mem(n, 0);
        auto a = pxa_tsplit_create_split(17408, 64, cum(w), mem);
        auto b = pxa_tsplit_create_split(17408, 128, cum(w), mem);
        CHECK(a == b, "equal %d-way differs between g64 and g128", n);
    }
    // not a multiple of 128: falls back, keeps old granularity
    { bool fb = false; int g = pxa_tsplit_rht_granularity(64, 17472, true, &fb); CHECK(g == 64 && fb, "fallback"); }
    // not rotated: untouched
    CHECK(pxa_tsplit_rht_granularity(64, 17408, false) == 64, "unrotated");
    // '-sm tensor' block-granular expert cut (pxa_tsplit_parity_split): Flash-Next n_ff 640 in 128-K blocks on two
    // cards is 3:2, the odd block goes to the card with less memory committed, never more than one block off even.
    {
        auto r = pxa_tsplit_parity_split(640, 128, { 1000, 2000 });
        CHECK(r == std::vector<int>({384, 256}), "parity 2-way to the emptier card: %d %d", r[0], r[1]);
        r = pxa_tsplit_parity_split(640, 128, { 3000, 2000 });
        CHECK(r == std::vector<int>({256, 384}), "parity 2-way other card: %d %d", r[0], r[1]);
        r = pxa_tsplit_parity_split(640, 128, { 0, (size_t) 1 << 40 });   // a huge imbalance still moves one block only
        CHECK(r == std::vector<int>({384, 256}), "parity bounded: %d %d", r[0], r[1]);
        r = pxa_tsplit_parity_split(640, 64, { 5, 1 });                  // even: no remainder, memory ignored
        CHECK(r == std::vector<int>({320, 320}), "parity even: %d %d", r[0], r[1]);
        r = pxa_tsplit_parity_split(1408, 128, { 4, 1, 3, 2 });          // 11 blocks on 4 cards: 2,3,3,3 by memory
        CHECK(r == std::vector<int>({256, 384, 384, 384}), "parity 4-way: %d %d %d %d", r[0], r[1], r[2], r[3]);
        int sum = 0; for (int v : r) sum += v;
        CHECK(sum == 1408, "parity sum %d", sum);
    }
    printf(fails ? "FAILED %d\n" : "OK\n", fails);
    return fails ? 1 : 0;
}
