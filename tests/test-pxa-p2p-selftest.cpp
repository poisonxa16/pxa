// PXA_P2P_SELFTEST: the startup peer-copy self-test and its fallback branch. Needs two CUDA devices
// with a peer path; skips (77) otherwise.
//   1. the self-test passes on a healthy pair;
//   2. with PXA_P2P_SELFTEST_CORRUPT=1 (one received word flipped) it reports corruption;
//   3. the cached verdict the enable path uses then refuses the pair, so peer access stays off.
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>

int main() {
    // The hook is on from the start: backend init may already cache the per-pair verdicts (a build
    // with NCCL proves every pair before NCCL starts), and those must see the forced corruption.
    setenv("PXA_P2P_SELFTEST_CORRUPT", "1", 1);
    const int n = ggml_backend_cuda_get_device_count();
    int a = -1, b = -1;
    for (int i = 0; i < n && a < 0; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const int r = ggml_backend_cuda_p2p_selftest(i, j);
            if (r >= 0) { a = i; b = j; break; }
        }
    }
    if (a < 0) {
        printf("test-pxa-p2p-selftest: SKIP (no device pair with a peer path among %d device(s))\n", n);
        return 77;
    }
    int fail = 0;
    const int forced = ggml_backend_cuda_p2p_selftest(a, b);
    printf("forced-corrupt self-test %d<->%d: %d (want 0)\n", a, b, forced);
    if (forced != 0) ++fail;
    // the verdict the peer-access enable path reads was formed under the hook -> refused
    const bool trusted = ggml_backend_cuda_p2p_pair_trusted(a, b);
    printf("enable-path verdict under the hook: %s (want refused)\n", trusted ? "trusted" : "refused");
    if (trusted) ++fail;
    unsetenv("PXA_P2P_SELFTEST_CORRUPT");
    const int healthy = ggml_backend_cuda_p2p_selftest(a, b);
    printf("healthy self-test %d<->%d: %d (want 1)\n", a, b, healthy);
    if (healthy != 1) {
        fprintf(stderr, "FAIL: the pair corrupts data for real -- this system needs PXA_P2P=0\n");
        ++fail;
    }
    printf("test-pxa-p2p-selftest: %s\n", fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}
