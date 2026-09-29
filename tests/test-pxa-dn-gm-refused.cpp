// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// PXA_FUSE_DELTANET bit2 (the delta-net state gather + reset-mask multiply, fused into one launch):
// a launch the driver REFUSES for memory must be a clean decline, not an abort (bug #207 follow-up,
// ws6-fix 2026-09-25).
//
// What happened: with lazy module loading (the CUDA 12 default) the first launch of the fused
// kernel on a device has to load it, and loading takes device memory. On a V100 that -ub 1024 had
// filled to 27 MiB (Qwen3.8-27B PXQ3-balanced, -c 65536 q8_0) the launch came back
// cudaErrorMemoryAllocation and pxa_try_deltanet_gather_mask's CUDA_CHECK aborted the server --
// although the eager GET_ROWS + MUL it replaces were still able to run.
//
// A refused launch cannot be produced on demand without filling a card, so the engine has a test
// hook, PXA_DN_GM_FAULT=1, that makes the launch behave as refused. This test builds the exact
// shape the fusion matches -- GET_ROWS of 4096-wide f32 rows by an i32 index, then MUL by a
// [1, n_seqs] f32 mask -- runs it on CUDA device 0 and requires, in both modes:
//   * the result equal to the reference BIT FOR BIT (a gather and one multiply are exact), and
//   * the mode's own evidence in the engine's log (the fused site fired / the refusal was declined).
//
//   test-pxa-dn-gm-refused fused     the fused launch runs
//   test-pxa-dn-gm-refused refused   PXA_DN_GM_FAULT=1: refused -> declined -> eager pair, no abort
//
// Needs a CUDA device (LABEL "cuda"); exits 0 with a SKIP line when there is none.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

// The engine reports on stderr; capture it around the compute so the test can read its evidence.
struct stderr_capture {
    int saved = -1;
    std::string path;
    bool begin() {
        char tmpl[] = "/tmp/pxa-dn-gm-XXXXXX";
        const int fd = mkstemp(tmpl);
        if (fd < 0) return false;
        path = tmpl;
        fflush(stderr);
        saved = dup(2);
        dup2(fd, 2);
        close(fd);
        return true;
    }
    std::string end() {
        fflush(stderr);
        dup2(saved, 2);
        close(saved);
        std::string out;
        if (FILE * f = fopen(path.c_str(), "rb")) {
            char buf[4096];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
            fclose(f);
        }
        unlink(path.c_str());
        return out;
    }
};

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "fused";
    if (mode != "fused" && mode != "refused") {
        fprintf(stderr, "usage: %s fused|refused\n", argv[0]);
        return 2;
    }
    // bit2 only, so no other delta-net fusion can take these nodes; print the first sites' verdicts.
    setenv("PXA_FUSE_DELTANET", "4", 1);
    setenv("PXA_DN_GM_DUMP", "4", 1);
    if (mode == "refused") {
        setenv("PXA_DN_GM_FAULT", "1", 1);
    } else {
        unsetenv("PXA_DN_GM_FAULT");
    }

    if (ggml_backend_cuda_get_device_count() < 1) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }
    ggml_backend_t be = ggml_backend_cuda_init(0, nullptr);
    if (be == nullptr) {
        printf("SKIP: CUDA device 0 did not initialise\n");
        return 0;
    }

    const int64_t ncols = 4096;   // the fusion's floor: narrower gathers are not the state row
    const int64_t nrows = 8;
    const int64_t nseqs = 3;

    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nseqs);
    ggml_tensor * msk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, nseqs);
    ggml_tensor * g   = ggml_get_rows(ctx, src, idx);
    ggml_tensor * m   = ggml_mul(ctx, g, msk);
    ggml_set_name(m, "gm_out");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    std::vector<float> h_src(ncols * nrows);
    for (size_t k = 0; k < h_src.size(); ++k) {
        h_src[k] = (float) ((k * 2654435761u) % 1000003u) / 1000003.0f - 0.5f;
    }
    const int32_t h_idx[nseqs] = { 5, 0, 7 };
    const float   h_msk[nseqs] = { 1.0f, 0.0f, 0.75f };   // keep, reset, and a non-trivial scale
    ggml_backend_tensor_set(src, h_src.data(), 0, ggml_nbytes(src));
    ggml_backend_tensor_set(idx, h_idx, 0, sizeof(h_idx));
    ggml_backend_tensor_set(msk, h_msk, 0, sizeof(h_msk));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, m);

    stderr_capture cap;
    const bool captured = cap.begin();
    const ggml_status st = ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);
    const std::string log = captured ? cap.end() : std::string();
    fputs(log.c_str(), stderr);   // keep the engine's lines visible in the ctest output

    std::vector<float> h_out(ncols * nseqs);
    ggml_backend_tensor_get(m, h_out.data(), 0, ggml_nbytes(m));

    size_t bad = 0;
    for (int64_t s = 0; s < nseqs; ++s) {
        for (int64_t c = 0; c < ncols; ++c) {
            const float want = h_src[h_idx[s] * ncols + c] * h_msk[s];
            const float got  = h_out[s * ncols + c];
            if (memcmp(&want, &got, sizeof(float)) != 0) {
                if (bad < 4) {
                    printf("  mismatch seq %lld col %lld: got %.9g want %.9g\n", (long long) s, (long long) c, got, want);
                }
                bad++;
            }
        }
    }

    const bool fired    = log.find("-> fired") != std::string::npos;
    const bool refused  = log.find("-> launch-refused-oom") != std::string::npos &&
                          log.find("the fused launch was refused for memory") != std::string::npos;
    const bool evidence = mode == "fused" ? fired && !refused : refused && !fired;

    printf("mode=%s compute=%s bitwise-mismatches=%zu evidence(%s)=%s\n", mode.c_str(),
           st == GGML_STATUS_SUCCESS ? "ok" : "FAILED", bad,
           mode == "fused" ? "site fired" : "refusal declined", evidence ? "yes" : "NO");
    const bool ok = st == GGML_STATUS_SUCCESS && bad == 0 && evidence;
    printf("%s\n", ok ? "PASS" : "FAIL");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return ok ? 0 : 1;
}
