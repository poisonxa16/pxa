// R3d (PXA_MTP_RESERVE_RECOMPUTE): the MTP chain reserve priced per step (unchanged, = the formula before R3d) and as replay
// rollback (one start state + per-token recurrence inputs + conv rows, = pxa_replay_pack_elems + the conv rows per_step_alloc
// claims). Host-only; Flash-Next dims (nv 48, head_v 128, d_state 128 x n_group 16, d_conv 4, 36 recurrent layers).
#include "../src/llama-pxa-place.h"
#include <cmath>
#include <cstdio>
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)
int main() {
    const int64_t nv = 48, head_v = 128, key_dim = 128*16, d_conv = 4, L = 36;
    // per step = the pre-R3d formula, byte for byte
    for (int64_t w = 2; w <= 9; ++w) {
        const int64_t conv_dim = key_dim*2 + head_v*nv, ssm_row = head_v*head_v*nv, conv_row = (d_conv - 1)*conv_dim;
        const size_t old = (size_t)((w - 1)*ssm_row + w*conv_row)*sizeof(float);
        CHECK(pxa_spec_reserve_deltanet_layer(w, nv, head_v, key_dim, d_conv, false) == old, "per-step w=%lld differs", (long long) w);
    }
    // replay = start state + w x (q, k, v, g, beta) + w conv rows (pxa_replay_pack_elems + the per_step_conv rows)
    for (int64_t w = 2; w <= 9; ++w) {
        const int64_t Sv = head_v, H = nv, Sk = 128, Hk = 16;
        const int64_t pack = Sv*Sv*H + w*(2*Sk*Hk + Sv*H + 2*H), conv = w*(d_conv - 1)*(key_dim*2 + head_v*nv);
        CHECK(pxa_spec_reserve_deltanet_layer(w, nv, head_v, key_dim, d_conv, true) == (size_t)(pack + conv)*sizeof(float), "replay w=%lld", (long long) w);
    }
    const double ps4 = L*pxa_spec_reserve_deltanet_layer(4, nv, head_v, key_dim, d_conv, false)/1048576.0;
    const double rp4 = L*pxa_spec_reserve_deltanet_layer(4, nv, head_v, key_dim, d_conv, true)/1048576.0;
    const double ps2 = L*pxa_spec_reserve_deltanet_layer(2, nv, head_v, key_dim, d_conv, false)/1048576.0;
    const double rp2 = L*pxa_spec_reserve_deltanet_layer(2, nv, head_v, key_dim, d_conv, true)/1048576.0;
    CHECK(std::fabs(ps4 - 340.90) < 0.05, "per-step w4 %.2f MiB, the G30 log says 340.90", ps4);   // the server log's budget line
    CHECK(rp4 < 0.4*ps4, "replay w4 %.2f not < 40%% of per-step %.2f", rp4, ps4);
    CHECK(rp2 > ps2, "at width 2 replay (%.2f) costs a little more than per-step (%.2f): one state either way + the input rows", rp2, ps2);
    for (int64_t w = 3; w <= 9; ++w)
        CHECK(pxa_spec_reserve_deltanet_layer(w, nv, head_v, key_dim, d_conv, true) < pxa_spec_reserve_deltanet_layer(w, nv, head_v, key_dim, d_conv, false), "w=%lld", (long long) w);
    printf("Flash-Next chain reserve x %lld layers: width 4 per-step %.2f MiB, replay %.2f MiB (saves %.2f); width 2 %.2f vs %.2f\n",
           (long long) L, ps4, rp4, ps4 - rp4, ps2, rp2);
    printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
