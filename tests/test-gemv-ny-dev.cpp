// PXA_F16_GEMV_NY on the device: the small-R F16 GEMV over a verify batch, against the path it
// replaces and against an fp64 reference.
//
// WHAT IS BEING CLAIMED. The shipping small-R F16 GEMV (k_pxa_gemv_f16, default ON) only accepts
// src1->ne[1] == 1, so at a speculative verify width the same node class lands on the generic
// cuBLAS chain instead: convert x f32->f16, GemmEx COMPUTE_16F, convert dst f16->f32. The new
// kernel gives that shipping kernel a token grid dimension. Two separate claims follow and this
// harness separates them, because they need different bars:
//
//   (1) WIDTH IS THE ONLY THING THAT CHANGED. Column j of a ny>1 launch must be BIT-IDENTICAL to
//       what the ne11==1 kernel produces for the same activation column. Per (row, token) the
//       loop bounds, the per-thread k stride and the warp/smem fold are the shipping kernel's,
//       so anything other than byte equality here is a defect, not a numerics difference. This
//       check runs INSIDE the armed process: every case computes a ny==1 node (which this lever
//       declines outright, so it takes k_pxa_gemv_f16) beside the ny>1 nodes, all fed the same
//       activation column 0, and compares the bytes.
//
//   (2) IT IS NOT BIT-EQUAL TO cuBLAS, AND MUST NOT BE. The chain being replaced rounds x AND
//       the result through f16; this path keeps both in f32. So the cross-arm comparison is a
//       REPORTED DIFFERENCE, not a pass/fail — and the fp64 reference below says which of the
//       two is closer to the true dot product. A byte-equal result across the two arms would
//       mean the lever never fired.
//
// The lever resolves once per process (a function-local static, as everywhere in this tree), so
// one process cannot run both paths. This binary runs the case list once and dumps every output
// raw to the file named on the command line; tests/test-gemv-ny-dev.sh runs it twice, unset then
// =1, and reports the per-case differences. The driver requires the kernel's own ENGAGED line in
// the armed run: without it, two identical dumps would come from the SAME path and prove nothing.
//
// The ny==1 cases are in the list for a third reason: they must be byte-identical across the two
// arms. That is the default-path invariant — arming this lever may not move plain decode.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct gemv_case {
    const char * name;
    int64_t K;    // src0->ne[0], the contracted extent
    int64_t R;    // src0->ne[1], the output rows (the "small R")
};

// The two live classes plus the edges of the gate. hc_inject is the one the lever was written
// for: [10240 x 4] F16, two nodes per layer plus the speculative head's two, and the class a
// profile named as 11.4% of all GPU time under verify. attn_gate is the other F16 small-R node
// in this family. The last two sit on the R and K gate boundaries so a mis-stated bound shows up
// as a case that silently stayed on cuBLAS in the armed run.
static const gemv_case cases[] = {
    { "hc_inject K10240 R4",    10240,   4 },
    { "hc_inject K10240 R8",    10240,   8 },
    { "attn_gate K2560 R48",     2560,  48 },
    { "attn_gate K2560 R64",     2560,  64 },
    { "wide-R    K4096 R512",    4096, 512 },
    { "K-floor   K1024 R4",      1024,   4 },
};

// Every ny the gate admits, plus 1. ny==1 is the invariant case (the lever declines it); 2..8 is
// the verify window (1 accepted + n_max drafted, MMVQ_MAX_BATCH_SIZE convention).
static const int NY_MAX = 8;

struct lcg {
    uint64_t s;
    explicit lcg(uint64_t seed) : s(seed) {}
    uint32_t next() { s = s*6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t) (s >> 33); }
    float unit() { return (float) next() / (float) 0x80000000u - 1.0f; }
};

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <output-dump>\n", argv[0]);
        return 2;
    }

    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr);
    if (!cuda) {
        printf("no CUDA device -- cannot run\n");
        return 2;
    }

    FILE * out = fopen(argv[1], "wb");
    if (!out) {
        fprintf(stderr, "cannot open %s for writing\n", argv[1]);
        ggml_backend_free(cuda);
        return 2;
    }

    const char * lever = getenv("PXA_F16_GEMV_NY");
    printf("test-gemv-ny-dev: PXA_F16_GEMV_NY=%s, dumping to %s\n", lever ? lever : "(unset)", argv[1]);

    // ggml_fp16_to_fp32 reads a lookup table that ggml_init populates. The fp64 reference below
    // calls it BEFORE this file's first ggml_init, so without this the first case's reference came
    // back all zeros and its accuracy column was meaningless (caught 2026-09-20: case 1 reported
    // max|y|=0 and an "error" equal to the answer, in BOTH arms).
    { ggml_init_params p0 = { 16*1024, nullptr, true }; ggml_context * c0 = ggml_init(p0); ggml_free(c0); }

    int rc = 0;
    for (const gemv_case & c : cases) {
        // Host-side operands, generated once per case so that every ny variant is fed the SAME
        // activation columns and the same weight. Column j of x is used by every variant with
        // ny > j, which is what makes the column-0 bit-compare below mean anything.
        lcg r(0x9e3779b9ull + (uint64_t) (c.K*131 + c.R));
        std::vector<float>       wf((size_t) c.K*c.R);
        std::vector<ggml_fp16_t> wh((size_t) c.K*c.R);
        std::vector<float>       xf((size_t) c.K*NY_MAX);
        for (size_t i = 0; i < wf.size(); ++i) wf[i] = 0.5f*r.unit();
        for (size_t i = 0; i < xf.size(); ++i) xf[i] = 1.0f*r.unit();
        ggml_fp32_to_fp16_row(wf.data(), wh.data(), (int64_t) wf.size());

        // fp64 reference, from the SAME f16 weights the device will read (round the weight first,
        // then widen) and the same f32 activations. This is the true dot product of the operands
        // as stored, so it prices both arms rather than flattering either.
        std::vector<double> ref((size_t) c.R*NY_MAX);
        for (int64_t j = 0; j < NY_MAX; ++j) {
            for (int64_t row = 0; row < c.R; ++row) {
                double acc = 0.0;
                const ggml_fp16_t * wrow = wh.data() + (size_t) row*c.K;
                const float       * xcol = xf.data() + (size_t) j*c.K;
                for (int64_t k = 0; k < c.K; ++k) {
                    acc += (double) ggml_fp16_to_fp32(wrow[k]) * (double) xcol[k];
                }
                ref[(size_t) j*c.R + row] = acc;
            }
        }

        // One graph per case: a ny==1 node plus one node per verify width, all sharing a weight.
        ggml_init_params ip = { ggml_tensor_overhead()*(2*(NY_MAX+1) + 4) + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.K, c.R);
        ggml_tensor * x[NY_MAX + 1] = { nullptr };
        ggml_tensor * y[NY_MAX + 1] = { nullptr };
        ggml_cgraph * gf = ggml_new_graph(ctx);
        for (int ny = 1; ny <= NY_MAX; ++ny) {
            x[ny] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, ny);
            y[ny] = ggml_mul_mat(ctx, w, x[ny]);
            ggml_build_forward_expand(gf, y[ny]);
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cuda);
        if (!buf) {
            printf("  %-24s SKIPPED (allocation failed)\n", c.name);
            ggml_free(ctx);
            continue;
        }

        ggml_backend_tensor_set(w, wh.data(), 0, wh.size()*sizeof(ggml_fp16_t));
        for (int ny = 1; ny <= NY_MAX; ++ny) {
            ggml_backend_tensor_set(x[ny], xf.data(), 0, (size_t) c.K*ny*sizeof(float));
        }

        if (ggml_backend_graph_compute(cuda, gf) != GGML_STATUS_SUCCESS) {
            printf("  %-24s FAIL (graph compute)\n", c.name);
            rc = 1;
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            continue;
        }

        std::vector<std::vector<float>> res(NY_MAX + 1);
        for (int ny = 1; ny <= NY_MAX; ++ny) {
            res[ny].resize((size_t) c.R*ny);
            ggml_backend_tensor_get(y[ny], res[ny].data(), 0, res[ny].size()*sizeof(float));
        }

        // Scale of the answer, so an absolute error can be read.
        double maxabs = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) maxabs = std::max(maxabs, std::fabs(ref[i]));

        printf("  %-24s K=%5lld R=%4lld  max|y|=%.6g\n",
               c.name, (long long) c.K, (long long) c.R, maxabs);

        for (int ny = 1; ny <= NY_MAX; ++ny) {
            // (2) distance from the fp64 truth, over every column of this node.
            double worst = 0.0;
            int    bad   = 0;
            for (int j = 0; j < ny; ++j) {
                for (int64_t row = 0; row < c.R; ++row) {
                    const float got = res[ny][(size_t) j*c.R + row];
                    if (!std::isfinite(got)) { bad++; continue; }
                    worst = std::max(worst, std::fabs((double) got - ref[(size_t) j*c.R + row]));
                }
            }

            // (1) width is the only thing that changed: column j against the ny==1 node, which
            // this lever declines and which therefore always runs the shipping kernel. Only
            // column 0 shares an activation with it.
            size_t neq = 0;
            double coldiff = 0.0;
            for (int64_t row = 0; row < c.R; ++row) {
                const float a = res[1][(size_t) row];
                const float b = res[ny][(size_t) row];
                if (memcmp(&a, &b, sizeof(float)) != 0) neq++;
                coldiff = std::max(coldiff, std::fabs((double) a - (double) b));
            }

            if (bad) {
                printf("      ny=%d  FAIL %d non-finite\n", ny, bad);
                rc = 1;
            } else {
                printf("      ny=%d  max|err vs fp64| %.6e   col0 vs ne11==1: %s (max diff %.6e)\n",
                       ny, worst,
                       neq == 0 ? "BIT-IDENTICAL" : "differs",
                       coldiff);
            }

            fwrite(res[ny].data(), sizeof(float), res[ny].size(), out);
        }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    fclose(out);
    ggml_backend_free(cuda);
    printf("%s\n", rc ? "FAIL" : "dump written");
    return rc;
}
