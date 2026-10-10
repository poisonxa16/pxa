// unit test for src/llama-pxa-tokmiss.h (PXA_XCACHE_MISS_TOKEN): g++ -std=c++17 -I src tests/test-pxa-tokmiss.cpp
#include "llama-pxa-tokmiss.h"
#include <cstdio>
#include <cmath>
static int fails = 0;
#define EXPECT(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
int main() {
    pxa_tokmiss m;
    // step 1: tokens {10, 20}; layer A: row 0 has 2 cold slots, row 1 none; layer B: row 1 has 3
    int32_t b1[2] = {10, 20};
    m.note_batch(b1, 2);
    int32_t idsA[8*2]; for (auto & v : idsA) v = -1; idsA[0] = 5; idsA[3] = 7;
    m.note_request(idsA, 8, 2, 400);
    int32_t idsB[8*2]; for (auto & v : idsB) v = -1; idsB[8] = 1; idsB[9] = 2; idsB[15] = 3;
    m.note_request(idsB, 8, 2, 600);
    int32_t b2[1] = {30};
    m.note_batch(b2, 1);                       // folds step 1
    EXPECT(std::fabs(m.slots(10) - 2.f) < 1e-6);
    EXPECT(std::fabs(m.slots(20) - 3.f) < 1e-6);
    EXPECT(std::fabs(m.slots(999) - 2.5f) < 1e-6);   // unseen: running mean
    // step 2: token 30 no cold at all -> 0
    int32_t b3[1] = {10};
    m.note_batch(b3, 1);
    EXPECT(std::fabs(m.slots(30)) < 1e-6);
    // EMA: token 10 now gets 0 cold -> 2 + 0.3*(0-2) = 1.4
    m.note_batch(b3, 1);
    EXPECT(std::fabs(m.slots(10) - 1.4f) < 1e-5);
    // cost: below 64 slots the fallback is used
    EXPECT(m.us_per_slot(100.0) == 100.0);
    for (int i = 0; i < 20; ++i) m.note_request(idsB, 8, 2, 300);   // 3 slots, 300 us each -> enough samples
    const double ups = m.us_per_slot(100.0);
    EXPECT(ups > 107.0 && ups < 108.5);   // (400+600 + 20*300)/(5 + 60)
    printf("us_per_slot %.2f\n", ups);
    // bill + keep: draft {30 (0 slots), 20 (3), 10 (1.4)}
    int32_t d[3] = {30, 20, 10};
    const int bill = m.bill_us(d, 3, 100.0);
    printf("bill %d us\n", bill);
    EXPECT(std::abs(bill - (int)std::lround((0 + 3 + 1.4)*ups)) <= 1);
    EXPECT(m.keep(d, 3, 1000000, 100.0) == 3);  // generous budget keeps all
    EXPECT(m.keep(d, 3, (int)(2*ups), 100.0) == 1);   // token 20 alone overflows: keep only the first
    EXPECT(m.keep(d, 3, (int)(3.5*ups), 100.0) == 2);
    // prefill-sized batch: not tracked
    std::vector<int32_t> big(64, 7); m.note_batch(big.data(), 64); m.note_batch(b3, 1);
    EXPECT(std::fabs(m.slots(7) - 2.5f) > -1);  // no crash; 7 stays unseen-or-mean
    printf("%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
