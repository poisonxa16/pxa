// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// Port of public PR #13 (SM75 dense matmul): ggml_cuda_should_use_mmq_dense must be a no-op everywhere except
// cc 750, and on cc 750 it must equal the old selector at the default PXA_TURING_CUBLAS_NE11 (64).
//   test-pxa-mmq-dense-select              driver: one child process per (cc, lever value) -- levers cache per process
//   test-pxa-mmq-dense-select CC MODE      MODE = default | 0 | 128 | 256 (PXA_TURING_CUBLAS_NE11)
// Every ggml type x ne11 1..4096 is compared. Host only, no card.
#include "ggml.h"
#include "ggml-cuda/pxa/pxa-enhance.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11);
bool ggml_cuda_should_use_mmq_dense(enum ggml_type type, int cc, int64_t ne11);

static bool turing_dense_type(ggml_type t) {
    return t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K;
}

static int child(int cc, const char * mode) {
    for (const char * k : {"PXA_TURING_CUBLAS_NE11", "PXA_REFERENCE", "PXA_ENHANCE"}) unsetenv(k);
    long thr = 64;
    if (strcmp(mode, "default") != 0) {
        setenv("PXA_TURING_CUBLAS_NE11", mode, 1);
        thr = atol(mode);
    }
    pxa_enhance_init_topology(1, &cc);
    long n = 0, bad = 0, moved = 0;
    for (int ti = 0; ti < GGML_TYPE_COUNT; ++ti) {
        const ggml_type t = (ggml_type) ti;
        if (ggml_type_size(t) == 0) continue;   // removed type ids
        for (int64_t ne11 = 1; ne11 <= 4096; ++ne11) {
            const bool base  = ggml_cuda_should_use_mmq(t, cc, ne11);
            const bool dense = ggml_cuda_should_use_mmq_dense(t, cc, ne11);
            bool want = base;
            if (cc == 750 && turing_dense_type(t) && ggml_cuda_should_use_mmq(t, cc, 1)) {
                want = thr == 0 || ne11 < thr;
            }
            ++n;
            moved += dense != base;
            if (dense != want) {
                if (++bad <= 8) fprintf(stderr, "  MISMATCH cc %d %s ne11 %lld: dense %d want %d (base %d)\n",
                                        cc, ggml_type_name(t), (long long) ne11, dense, want, base);
            }
        }
    }
    const bool identity_expected = cc != 750 || thr == 64;
    if (identity_expected && moved) bad += moved;
    printf("cc %d PXA_TURING_CUBLAS_NE11=%s: %ld comparisons, %ld differ from the old selector, %ld wrong\n",
           cc, mode, n, moved, bad);
    return bad ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc == 3) return child(atoi(argv[1]), argv[2]);
    int fails = 0, runs = 0;
    for (int cc : {600, 610, 700, 750}) {
        for (const char * mode : {"default", "0", "128", "256"}) {
            const std::string cmd = std::string("\"") + argv[0] + "\" " + std::to_string(cc) + " " + mode;
            ++runs;
            fails += system(cmd.c_str()) != 0;
        }
    }
    printf("test-pxa-mmq-dense-select: %s (%d runs, %d failed)\n", fails ? "FAILED" : "OK", runs, fails);
    return fails ? 1 : 0;
}
