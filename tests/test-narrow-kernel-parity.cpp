//
// Parity harness for the narrow-shape CUDA kernel gates:
//
//   PXA_GETROWS_NARROW   flattened get_rows for 1-2 element rows
//   PXA_CPY_FASTDIV      multiply-shift index math in the float cpy kernel
//   PXA_CONCAT_FLAT      one-thread-per-element non-contiguous concat
//   PXA_NORM_REGCACHE    register-cached rms/l2 norm
//
// Each of those claims to be bit-identical to the path it replaces, so the test
// is an EQUALITY test, not a tolerance test. Two things are checked per case:
//
//   1. CUDA == CPU, bit-exact, on random input.  (For the cases here every
//      kernel is a pure data movement or a two-pass reduction that both
//      backends perform in the same order, so exact equality is the right bar.
//      The norm cases are the one place a reduction order could differ between
//      backends, so they are compared with an exact-equality bar against the
//      CUDA arm's own OFF run rather than against the CPU, see below.)
//   2. A hash of every CUDA result is printed. Because the gates are read once
//      per process into a static, a single process cannot A/B them; the
//      orchestrator runs this binary twice --
//
//        PXA_ENHANCE=0 ./test-narrow-kernel-parity                    > off.txt
//        ./test-narrow-kernel-parity                                  > on.txt
//        diff off.txt on.txt      # must be empty
//
//      -- and an empty diff is the bit-identity proof. A PASS in one run alone
//      only proves the arm that ran is correct, not that the two agree.
//
//      NOTE (2026-09-03): all four gates now default ON at the ENHANCE config
//      level, which is itself the default, so the BARE run is the ON arm and
//      PXA_ENHANCE=0 is what selects the OFF arm. Setting the four variables to
//      1 by hand still works and is equivalent to the bare run.
//
// Shapes are chosen to be the ones the gates actually target: ne00 = 1 gathers
// with many rows (a top-k expert weight gather), a transposed float copy, a
// narrow non-contiguous concat, and 2048/4096-wide norms.
//
// Needs a CUDA device. Tiny: a few MB.
//

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;

static void report(const char * name, bool ok, const char * detail = "") {
    printf("%-58s %s %s\n", name, ok ? "PASS" : "**FAIL**", detail);
    if (!ok) g_fail++;
}

// FNV-1a over the raw bytes, so the printed value is sensitive to every bit.
static uint64_t hash_bytes(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

static bool exact_equal(const std::vector<float> & a, const std::vector<float> & b, size_t & first_bad) {
    if (a.size() != b.size()) { first_bad = 0; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        // bitwise, so a signed zero or a NaN payload difference is a failure
        uint32_t ua, ub;
        memcpy(&ua, &a[i], 4);
        memcpy(&ub, &b[i], 4);
        if (ua != ub) { first_bad = i; return false; }
    }
    return true;
}

enum case_kind {
    CASE_GET_ROWS,      // ne00 = 1 gather over many rows
    CASE_GET_ROWS_WIDE, // control: the wide path the gate must not touch
    CASE_CPY,           // transposed f32 -> f32 copy
    CASE_CPY_F16,       // f32 -> f16 copy
    CASE_CONCAT_NC,     // narrow non-contiguous concat on dim 0
    CASE_RMS_NORM,      // 2048-wide rms norm
    CASE_L2_NORM,       // 4096-wide l2 norm
};

struct kcase {
    const char * name;
    case_kind    kind;
    int64_t      ne0;
    int64_t      ne1;
};

static const kcase g_cases[] = {
    { "get_rows ne00=1  nrows=5120",   CASE_GET_ROWS,      1,    5120 },
    { "get_rows ne00=2  nrows=777",    CASE_GET_ROWS,      2,     777 },
    { "get_rows ne00=1  nrows=1",      CASE_GET_ROWS,      1,       1 },
    { "get_rows ne00=64 nrows=300",    CASE_GET_ROWS_WIDE, 64,    300 },
    { "cpy f32 transposed 96x257",     CASE_CPY,           96,    257 },
    { "cpy f32->f16 transposed 33x64", CASE_CPY_F16,       33,     64 },
    { "concat nc dim1 ne0=4 ne1=8192", CASE_CONCAT_NC,     4,    8192 },
    { "concat nc dim1 ne0=7 ne1=513",  CASE_CONCAT_NC,     7,     513 },
    { "rms_norm 2048 x 3",             CASE_RMS_NORM,      2048,    3 },
    { "rms_norm 4096 x 1",             CASE_RMS_NORM,      4096,    1 },
    { "l2_norm  2048 x 2",             CASE_L2_NORM,       2048,    2 },
};

// Builds and runs one case on `be`, returning the result rows as f32.
static bool run_case(ggml_backend_t be, const kcase & c, std::mt19937 & rng, std::vector<float> & out) {
    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    ggml_tensor * result = nullptr;

    std::vector<float>   src_h;
    std::vector<int32_t> idx_h;
    std::vector<float>   src1_h;

    std::uniform_real_distribution<float> uni(-3.0f, 3.0f);

    ggml_tensor * a = nullptr, * b = nullptr, * idx = nullptr;

    switch (c.kind) {
        case CASE_GET_ROWS:
        case CASE_GET_ROWS_WIDE: {
            const int64_t n_src_rows = 512;   // e.g. the expert count
            a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne0, n_src_rows);
            idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.ne1);
            result = ggml_get_rows(ctx, a, idx);

            src_h.resize(c.ne0*n_src_rows);
            for (auto & v : src_h) v = uni(rng);
            idx_h.resize(c.ne1);
            std::uniform_int_distribution<int> ui(0, (int) n_src_rows - 1);
            for (auto & v : idx_h) v = ui(rng);
        } break;

        case CASE_CPY:
        case CASE_CPY_F16: {
            // transpose forces the non-contiguous scalar cpy kernel
            a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne0, c.ne1);
            ggml_tensor * at = ggml_transpose(ctx, a);
            b = ggml_new_tensor_2d(ctx, c.kind == CASE_CPY ? GGML_TYPE_F32 : GGML_TYPE_F16, c.ne1, c.ne0);
            result = ggml_cpy(ctx, at, b);
            if (c.kind == CASE_CPY_F16) {
                result = ggml_cast(ctx, result, GGML_TYPE_F32);
            }
            src_h.resize(c.ne0*c.ne1);
            for (auto & v : src_h) v = uni(rng);
        } break;

        case CASE_CONCAT_NC: {
            // ggml_concat's own asserts require nb[0] == sizeof(float) on both sources
            // (ggml.c ggml_compute_forward_concat_f32), which a transposed view violates
            // (transpose swaps nb[0] and nb[1]). Get a non-contiguous f32 source without
            // breaking that: allocate `a` with `pad` extra elements per row, then view it
            // at the true row width. That view keeps nb[0] == sizeof(float) but has
            // nb[1] > ne0*sizeof(float), so it is non-contiguous along dim 1 -- the exact
            // shape PXA_CONCAT_FLAT's non-contiguous path targets (row width narrower than
            // CUDA_CONCAT_BLOCK_SIZE=256, concatenated along a non-dim0 axis).
            const int64_t pad = 3;
            a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne0 + pad, c.ne1);
            ggml_tensor * a_nc = ggml_view_2d(ctx, a, c.ne0, c.ne1, a->nb[1], 0);
            b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne0, c.ne1);
            result = ggml_concat(ctx, a_nc, b, 1);
            src_h.resize((c.ne0 + pad)*c.ne1);
            for (auto & v : src_h) v = uni(rng);
            src1_h.resize(c.ne0*c.ne1);
            for (auto & v : src1_h) v = uni(rng);
        } break;

        case CASE_RMS_NORM:
        case CASE_L2_NORM: {
            a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne0, c.ne1);
            result = c.kind == CASE_RMS_NORM ? ggml_rms_norm(ctx, a, 1e-6f)
                                             : ggml_l2_norm (ctx, a, 1e-12f);
            src_h.resize(c.ne0*c.ne1);
            for (auto & v : src_h) v = uni(rng);
        } break;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); return false; }

    ggml_backend_tensor_set(a, src_h.data(), 0, src_h.size()*sizeof(float));
    if (idx) ggml_backend_tensor_set(idx, idx_h.data(), 0, idx_h.size()*sizeof(int32_t));
    if (b && !src1_h.empty()) ggml_backend_tensor_set(b, src1_h.data(), 0, src1_h.size()*sizeof(float));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }

    out.resize(ggml_nelements(result));
    ggml_backend_tensor_get(result, out.data(), 0, out.size()*sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    (void) argc; (void) argv;

    printf("# narrow-kernel parity harness\n");
    // The four gates default to ON at the ENHANCE level and OFF below it; an explicit variable
    // overrides in either direction. Print the level (which is what decides when nothing is set)
    // and then any explicit override, so the arm this run measured is unambiguous from the log.
    printf("# PXA config level: %s -> gates default %s\n",
           ggml_pxa_config_level_name(), ggml_pxa_config_level() >= 2 ? "ON" : "OFF");
    printf("# explicit overrides: GETROWS_NARROW=%s CPY_FASTDIV=%s CONCAT_FLAT=%s NORM_REGCACHE=%s\n",
           getenv("PXA_GETROWS_NARROW") ? getenv("PXA_GETROWS_NARROW") : "(unset)",
           getenv("PXA_CPY_FASTDIV")    ? getenv("PXA_CPY_FASTDIV")    : "(unset)",
           getenv("PXA_CONCAT_FLAT")    ? getenv("PXA_CONCAT_FLAT")    : "(unset)",
           getenv("PXA_NORM_REGCACHE")  ? getenv("PXA_NORM_REGCACHE")  : "(unset)");

    ggml_backend_t cpu  = ggml_backend_cpu_init();
    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr);
    if (!cuda) {
        printf("no CUDA device -- nothing to compare, skipping\n");
        ggml_backend_free(cpu);
        return 0;
    }

    for (const auto & c : g_cases) {
        // same seed per case on both backends, so both see identical input
        std::mt19937 rng_cpu (0xC0FFEEu);
        std::mt19937 rng_cuda(0xC0FFEEu);

        std::vector<float> out_cpu, out_cuda;

        if (!run_case(cpu, c, rng_cpu, out_cpu)) {
            report(c.name, false, "(cpu arm failed to run)");
            continue;
        }
        if (!run_case(cuda, c, rng_cuda, out_cuda)) {
            report(c.name, false, "(cuda arm failed to run)");
            continue;
        }

        // the hash is what the two runs of this binary are diffed on
        const uint64_t h = hash_bytes(out_cuda.data(), out_cuda.size()*sizeof(float));

        size_t bad = 0;
        char detail[192];

        const bool is_norm = c.kind == CASE_RMS_NORM || c.kind == CASE_L2_NORM;
        if (is_norm) {
            // A norm sums the row on both backends but not necessarily in the same
            // association, so CPU/CUDA equality is not the claim here. The claim is that
            // the CUDA arm does not change when the gate flips, which is what the printed
            // hash proves across the two runs. Report a loose sanity bound so a grossly
            // wrong kernel still fails inside a single run.
            double worst = 0.0;
            for (size_t i = 0; i < out_cpu.size() && i < out_cuda.size(); ++i) {
                worst = std::max(worst, (double) std::fabs(out_cpu[i] - out_cuda[i]));
            }
            snprintf(detail, sizeof(detail), "cuda_hash=%016llx max|cpu-cuda|=%.3g",
                     (unsigned long long) h, worst);
            report(c.name, worst < 1e-4, detail);
        } else {
            const bool ok = exact_equal(out_cpu, out_cuda, bad);
            if (ok) {
                snprintf(detail, sizeof(detail), "cuda_hash=%016llx n=%zu",
                         (unsigned long long) h, out_cuda.size());
            } else {
                snprintf(detail, sizeof(detail), "cuda_hash=%016llx first_diff=%zu cpu=%.9g cuda=%.9g",
                         (unsigned long long) h, bad,
                         bad < out_cpu.size()  ? out_cpu[bad]  : 0.0f,
                         bad < out_cuda.size() ? out_cuda[bad] : 0.0f);
            }
            report(c.name, ok, detail);
        }
    }

    ggml_backend_free(cuda);
    ggml_backend_free(cpu);

    printf("\n%s (%d failures)\n", g_fail == 0 ? "ALL PASS" : "**FAILURES**", g_fail);
    return g_fail == 0 ? 0 : 1;
}
