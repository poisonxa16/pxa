// PXA_HOST_RESIDENT gate (pass 8e review R6/R7): full offload only; MAX_MIB parse never wraps to unlimited.
#include "../src/pxa-host-resident.h"
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    // R6: full offload = every repeating layer AND the head on the cards
    CHECK( pxa_host_resident_full_offload(0, 999, 64));   // -ngl 999 (server default)
    CHECK( pxa_host_resident_full_offload(0, 65, 64));    // -ngl n_layer+1
    CHECK(!pxa_host_resident_full_offload(0, 64, 64));    // head on the CPU
    CHECK(!pxa_host_resident_full_offload(32, 32, 64));   // partial offload
    CHECK(!pxa_host_resident_full_offload(64, 0, 64));    // -ngl 0
    CHECK(!pxa_host_resident_full_offload(1, 999, 64));   // a layer moved back to the CPU by a placement rule
    // R7: the cap
    bool bad = true;
    CHECK(pxa_host_resident_cap_bytes(nullptr, &bad) == ((size_t) 2048 << 20) && !bad);
    CHECK(pxa_host_resident_cap_bytes("1024", &bad) == ((size_t) 1024 << 20) && !bad);
    CHECK(pxa_host_resident_cap_bytes("-1", &bad) == 0 && bad);           // used to become (size_t)-1 << 20
    CHECK(pxa_host_resident_cap_bytes("0", &bad) == 0 && bad);
    CHECK(pxa_host_resident_cap_bytes("", &bad) == 0 && bad);
    CHECK(pxa_host_resident_cap_bytes("abc", &bad) == 0 && bad);
    CHECK(pxa_host_resident_cap_bytes("12x", &bad) == 0 && bad);
    CHECK(pxa_host_resident_cap_bytes("99999999999", &bad) == ((size_t) 65536 << 20) && !bad);   // clamped, no overflow
    printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
