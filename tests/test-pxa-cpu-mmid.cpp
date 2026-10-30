// test-pxa-cpu-mmid (2026-10-16): the CPU MUL_MAT_ID work queue + same-order Q8_0 tiles (and the
// work queue with the plain vec_dot for K-quant / legacy / IQ4_NL / F16 experts)
// (PXA_CPU_MMID_TILED=1, default) must equal the old per-expert vec_dot loop (=0) byte for byte.
// Shapes: Nemotron-3-Nano-30B-A3B experts (up 2688->1856, down 1856->2688, 128 experts, top-6) at T = 1, 7, 512
// plus ragged edges (odd rows, few experts, T not a multiple of 4). Usage: test-pxa-cpu-mmid [threads] [reps] [name-filter]
// With reps > 0 it also prints the mean wall time of each path per case (perf mode, the perf-flagged cases; modes 0/1/2
// interleaved per rep, routed experts relabeled every rep so small batches read DRAM as in a model).
#include "ggml.h"
#include "ggml-pxa-lean.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

struct mcase { const char * name; int64_t K, N, E, k, T; bool up; bool perf; ggml_type type = GGML_TYPE_Q8_0; };

static uint64_t fnv(const void * p, size_t n) {
    const unsigned char * b = (const unsigned char *) p; uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static bool run_case(const mcase & c, int nthreads, int reps) {
    const size_t wbytes = ggml_row_size(c.type, c.K) * c.N * c.E;
    const int64_t ne11 = c.up ? 1 : c.k;
    ggml_init_params ip = { wbytes + (size_t)(c.K*ne11*c.T + c.N*c.k*c.T)*4 + (size_t)c.k*c.T*4 + 320ull*1024*1024, nullptr, false }; // slack: graph + 18-thread work buffer
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * as  = ggml_new_tensor_3d(ctx, c.type, c.K, c.N, c.E);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, ne11, c.T);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.k, c.T);
    std::mt19937 rng(1234 + (int) c.K + (int) c.T);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    {   // weights: random rows quantized with the library quantizer, one expert at a time
        std::vector<float> tmp(c.K * c.N);
        const size_t eb = ggml_row_size(c.type, c.K) * c.N;
        for (int64_t e = 0; e < c.E; ++e) {
            for (auto & v : tmp) v = nd(rng) * 0.05f;
            ggml_quantize_chunk(c.type, tmp.data(), (char *) as->data + e*eb, 0, c.N, c.K, nullptr, nullptr);
        }
    }
    for (int64_t i = 0; i < ggml_nelements(b); ++i) ((float *) b->data)[i] = nd(rng);
    {   // routing: k distinct experts per token, skewed so some experts are hot and some unused
        std::vector<int> pool(c.E);
        for (int64_t t = 0; t < c.T; ++t) {
            for (int64_t e = 0; e < c.E; ++e) pool[e] = (int) e;
            for (int64_t j = 0; j < c.k; ++j) {
                std::uniform_int_distribution<int64_t> ud(j, c.E - 1);
                int64_t pick = ud(rng);
                if (pick > j + 3 && (rng() % 3) == 0) pick = j + (int64_t)(rng() % 4); // hot experts
                std::swap(pool[j], pool[pick]);
                ((int32_t *) ids->data)[t*c.k + j] = pool[j];
            }
        }
    }
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    const size_t ob = ggml_nbytes(out);
    std::vector<uint8_t> ref(ob);
    double tm[3] = {0, 0, 0};
    bool ok = true;
    // modes: 0 = old per-expert loop (reference), 1 = queue + tiles (default), 2 = queue + plain vec_dot
    for (int mode = 0; mode <= 2; ++mode) {
        ggml_pxa_set_cpu_mmid_tiled(mode);
        memset(out->data, 0xA5, ob);
        ggml_graph_compute_with_ctx(ctx, gf, nthreads);
        if (mode == 0) {
            memcpy(ref.data(), out->data, ob);
        } else if (memcmp(ref.data(), out->data, ob) != 0) {
            // only the routed (slot, token) rows are written by MUL_MAT_ID; every slot is routed here
            size_t nd2 = 0; const float * a = (const float *) ref.data(); const float * q = (const float *) out->data;
            for (size_t i = 0; i < ob/4; ++i) nd2 += memcmp(&a[i], &q[i], 4) != 0;
            printf("  %-22s MISMATCH mode %d: %zu of %zu floats differ\n", c.name, mode, nd2, ob/4);
            ok = false;
        }
    }
    if (reps > 0 && c.perf) {
        // perf: every rep relabels the routed experts (+7 per rep, mod E) so small batches stream weights from DRAM
        // as in a model (no L3 reuse across reps); the same relabel sequence for each mode, modes interleaved per rep
        std::vector<int32_t> ids0((const int32_t *) ids->data, (const int32_t *) ids->data + c.k*c.T);
        for (int r = 0; r < reps; ++r) {
            for (int mode = 0; mode <= 2; ++mode) {
                for (int64_t i = 0; i < c.k*c.T; ++i) ((int32_t *) ids->data)[i] = (int32_t) ((ids0[i] + 7*(r + 1)) % c.E);
                ggml_pxa_set_cpu_mmid_tiled(mode);
                const int64_t t0 = ggml_time_us();
                ggml_graph_compute_with_ctx(ctx, gf, nthreads);
                tm[mode] += (ggml_time_us() - t0) / 1000.0 / reps;
            }
        }
        memcpy(ids->data, ids0.data(), ids0.size()*sizeof(int32_t));
    }
    printf("  %-22s %-5s K=%5lld N=%5lld E=%4lld k=%lld T=%4lld %s  hash %016llx  %s", c.name, ggml_type_name(c.type), (long long) c.K, (long long) c.N,
           (long long) c.E, (long long) c.k, (long long) c.T, c.up ? "up  " : "down", (unsigned long long) fnv(ref.data(), ob),
           ok ? "IDENTICAL" : "DIFF");
    if (reps > 0 && c.perf) printf("  old %.3f ms  new %.3f ms  q2 %.3f ms  new x%.2f  q2 x%.2f", tm[0], tm[1], tm[2], tm[0] / tm[1], tm[0] / tm[2]);
    printf("\n");
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    const int nthreads = argc > 1 ? atoi(argv[1]) : 4;
    const int reps     = argc > 2 ? atoi(argv[2]) : 0;
    const char * only  = argc > 3 ? argv[3] : nullptr; // run only the cases whose name contains this
    ggml_time_init();
    const mcase cases[] = {
        { "30b-up-T1",      2688, 1856, 128, 6,   1, true,  true  },
        { "30b-down-T1",    1856, 2688, 128, 6,   1, false, true  },
        { "30b-up-T7",      2688, 1856, 128, 6,   7, true,  false },
        { "30b-up-T4",      2688, 1856, 128, 6,   4, true,  true  },
        { "30b-down-T4",    1856, 2688, 128, 6,   4, false, true  },
        { "30b-up-T512",    2688, 1856, 128, 6, 512, true,  true  },
        { "30b-down-T512",  1856, 2688, 128, 6, 512, false, true  },
        { "edge-odd-rows",   256,   37,   8, 2,  33, true,  false },
        { "edge-few-tok",    512,   65,   4, 3,   3, false, false },
        { "edge-one-row",     64,    1,   3, 2,   5, true,  false },
        { "edge-hot",        128,   70,   2, 2, 130, false, false },
        // other expert types take the work queue with their own vec_dot (one call per output, no tile)
        // Nemotron-3.5-Bolt-74B-A9B latent experts (Q4_K_M: up q4_K 1024->2304, down q4_K / q6_K 2304->1024, 304 experts, top-9)
        { "bolt-up-q4k-T512",   1024, 2304, 304, 9, 512, true,  true,  GGML_TYPE_Q4_K },
        { "bolt-down-q4k-T512", 2304, 1024, 304, 9, 512, false, true,  GGML_TYPE_Q4_K },
        { "bolt-down-q6k-T512", 2304, 1024, 304, 9, 512, false, true,  GGML_TYPE_Q6_K },
        { "bolt-up-q4k-T1",     1024, 2304, 304, 9,   1, true,  true,  GGML_TYPE_Q4_K },
        // the 99-token tail ubatch of a 611-token prompt (about 3 columns per expert)
        { "bolt-up-q4k-T99",    1024, 2304, 304, 9,  99, true,  true,  GGML_TYPE_Q4_K },
        { "bolt-down-q6k-T99",  2304, 1024, 304, 9,  99, false, true,  GGML_TYPE_Q6_K },
        { "30b-up-T99",     2688, 1856, 128, 6,  99, true,  true  },
        { "q5k-up-T7",      2560,  100,  16, 4,   7, true,  false, GGML_TYPE_Q5_K },
        { "q5_0-down-T33",  1856,   64,   8, 2,  33, false, false, GGML_TYPE_Q5_0 },
        { "iq4nl-up-T9",    1856,   48,   8, 2,   9, true,  false, GGML_TYPE_IQ4_NL },
        { "f16-up-T9",       256,   40,   4, 2,   9, true,  false, GGML_TYPE_F16 },
    };
    int bad = 0;
    printf("test-pxa-cpu-mmid: threads %d reps %d\n", nthreads, reps);
    for (const auto & c : cases) {
        if (only && !strstr(c.name, only)) continue;
        bad += run_case(c, nthreads, reps) ? 0 : 1;
    }
    printf("%s: %d case(s) differ\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
