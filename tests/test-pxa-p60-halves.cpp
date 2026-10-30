// test-pxa-p60-halves.cpp -- PXA_P60_MEGA identity (2026-10-19): the default one-token PXQN4
// MoE half of the P60 stack (closed library, ggml-cuda/pxa/pxqn-mega.cu) must be bit-identical to the reference
// reimplementation of the normal decode path (the closed reference kernel, itself greedy-sha-identical to the stack-off engine)
// on pseudo-random inputs of the 35B-A3B shape. Needs a CUDA device (ctest -L cuda; PXA_TEST_DEVICE=N picks the card,
// default 0); a CPU build or a tree without the PXQN library SKIPs. argv[1] = seeds (default 4), argv[2] = timing reps
// (default 20; the us/call lines are shape-level information, not a cell).
#include <cstdio>
#include <cstdlib>
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include "ggml-cuda-p60.h"
#endif

int main(int argc, char ** argv) {
    const int nseed = argc > 1 ? atoi(argv[1]) : 4;
    const int reps  = argc > 2 ? atoi(argv[2]) : 20;
#ifndef GGML_USE_CUDA
    (void) nseed; (void) reps;
    printf("test-pxa-p60-halves: SKIP (no CUDA backend in this build)\n");
    return 0;
#else
    if (ggml_backend_cuda_get_device_count() < 1) {
        printf("test-pxa-p60-halves: SKIP (no CUDA device)\n");
        return 0;
    }
    const char * dv = getenv("PXA_TEST_DEVICE");
    const int device = dv ? atoi(dv) : 0;
    int fails = 0;
    for (int s = 1; s <= nseed; ++s) {
        int v1 = -1;
        float us[24] = {0.f};   // [0..2] us/call default / V1 / reference, [3 + 7 w + i] phase i of kernel w
        const int r = ggml_backend_cuda_p60_mega_selftest(device, (unsigned) s, reps, &v1, us);
        if (r == -1) {
            printf("test-pxa-p60-halves: SKIP (no PXQN library)\n");
            return 0;
        }
        printf("seed %d: default vs reference %d of 2048 words differ (V1 kernel: %d); us/call default %.1f V1 %.1f reference %.1f -> %s\n",
               s, r, v1, us[0], us[1], us[2], r == 0 ? "OK" : r == -2 ? "FAIL (non-finite)" : r == -3 ? "FAIL (CUDA error)" : "FAIL");
        static const char * const ph[7] = { "P0", "P1", "P2pro", "P2", "P3pro", "P3", "P4" };
        static const char * const kn[3] = { "default", "V1", "reference" };
        for (int w = 0; w < 3; ++w) {
            printf("  phases %-9s us:", kn[w]);
            for (int i = 0; i < 7; ++i) printf(" %s %.2f", ph[i], us[3 + 7 * w + i]);
            printf("\n");
        }
        if (r != 0) ++fails;
    }
    // the verify-width check: PXA_TEST_VW=<ops> (e.g. "0,1": 0 = MoE, 1 = GDN) runs, per op and
    // width 2..6, token t of the multi-token half against the one-token computation of that token, bitwise
    const char * vw = getenv("PXA_TEST_VW");
    for (int op = 0; vw && op < 2; ++op) {
        bool want = false;
        for (const char * c = vw; *c; ++c) want |= (*c - '0') == op;
        if (!want) continue;
        for (int s = 1; s <= nseed; ++s) {
            for (int T = 2; T <= 6; ++T) {
                int d[8] = {0};
                float us[32] = {0.f};
                const int r = ggml_backend_cuda_p60_vw_selftest(device, op, (unsigned) s, T, s == 1 ? reps : 0, d, us);
                if (r == -1) { printf("vw: SKIP (no PXQN library)\n"); break; }
                if (op == 0) {
                    printf("vw moe seed %d T %d: pxt2<T> vs pxt2<1> x_out %d, folded %d, add %d words differ; us pxm1 %.1f pxt2<1> %.1f pxt2<T> %.1f fold %.1f -> %s\n",
                           s, T, d[0], d[1], d[2], us[0], us[1], us[2], us[3], r == 0 ? "OK" : r == -2 ? "FAIL (non-finite)" : r == -3 ? "FAIL (CUDA error)" : r == -4 ? "FAIL (unsupported)" : "FAIL");
                } else {
                    printf("vw gdn seed %d T %d: v2 x %d state %d conv %d ssm %d | v1 x %d state %d conv %d ssm %d words differ; us pxg1 %.1f pxgxT %.1f v1 %.1f v2 %.1f -> %s\n",
                           s, T, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], us[0], us[1], us[2], us[3],
                           r == 0 ? "OK" : r == -2 ? "FAIL (non-finite)" : r == -3 ? "FAIL (CUDA error)" : r == -4 ? "FAIL (unsupported)" : "FAIL");
                    if (s == 1) {
                        static const char * const ph[8] = { "P0", "P1", "s1", "P2", "s2", "P3", "s3", "P4" };
                        for (int w = 0; w < 2; ++w) {
                            printf("  phases %s us:", w ? "v1" : "v2");
                            for (int i = 0; i < 8; ++i) printf(" %s %.2f", ph[i], us[4 + 8 * w + i]);
                            printf("\n");
                        }
                    }
                }
                if (r != 0) ++fails;
            }
        }
    }
    printf("test-pxa-p60-halves: %s (%d seeds, device %d)\n", fails ? "FAIL" : "OK", nseed, device);
    return fails ? 1 : 0;
#endif
}
