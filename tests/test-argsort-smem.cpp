// PXA_ARGSORT_SMEM (argsort.cu, default ON, =0 off): the register/shuffle bitonic argsort kernel (k_argsort_f32_T_reg)
// must write exactly what the global-key kernel (k_argsort_f32_T) writes.
//
// The proof here is exact, not a tolerance: the host runs the SAME compare/swap network as the kernel (stage by stage;
// within a stage the pairs are disjoint, so a sequential walk equals the parallel one) on the same floats, and the
// device indices must equal it element for element. Ties are the point: the router logits Flint sorts come out of a
// BF16 GEMM, so equal keys are common, and the order among equal keys is decided by the network, not by the values.
// The rows below are drawn from a small set of values (many exact ties, +0.0 and -0.0, -inf) at the widths the MoE
// routers use (8..512 experts) plus odd and padded widths, ASC and DESC, one and many rows, and the threshold variant
// (ggml_argsort_thresh, the store_ser path).
//
// Each lever setting is one process (the switch is read once): run it as is (lever ON) and with PXA_ARGSORT_SMEM=0;
// both must print OK, and the HASH lines of the two runs must be equal. Exit 77 without a CUDA device.

#include "pxa-env-flag.h"
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

struct lcg {
    uint64_t s;
    explicit lcg(uint64_t seed) : s(seed) {}
    uint32_t next() { s = s*6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t) (s >> 33); }
};

// host copy of the kernel's bitonic network (argsort.cu: sort<order>), indices padded to a power of two
static void host_bitonic(const float * x, int ncols, bool desc, std::vector<int> & idx) {
    int pad = 1;
    while (pad < ncols) pad *= 2;
    idx.resize(pad);
    for (int i = 0; i < pad; ++i) idx[i] = i;
    for (int k = 2; k <= pad; k *= 2) {
        for (int j = k/2; j > 0; j /= 2) {
            for (int col = 0; col < pad; ++col) {
                const int ixj = col ^ j;
                if (ixj <= col) continue;
                const int a = idx[col], b = idx[ixj];
                bool swap;
                if ((col & k) == 0) {
                    swap = a >= ncols || (b < ncols && (desc ? x[a] < x[b] : x[a] > x[b]));
                } else {
                    swap = b >= ncols || (a < ncols && (desc ? x[a] > x[b] : x[a] < x[b]));
                }
                if (swap) { idx[col] = b; idx[ixj] = a; }
            }
        }
    }
}

static uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ULL) {
    const unsigned char * c = (const unsigned char *) p;
    for (size_t i = 0; i < n; ++i) { h ^= c[i]; h *= 1099511628211ULL; }
    return h;
}

int main() {
    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr);
    if (!cuda) {
        printf("no CUDA device -- skipped\n");
        return 77;
    }
    const char * e = getenv("PXA_ARGSORT_SMEM");
    printf("PXA_ARGSORT_SMEM=%s (lever %s)\n", e ? e : "(unset)", pxa_env_value_off(e) ? "OFF" : "ON");

    const int widths[] = { 2, 3, 6, 8, 16, 32, 60, 64, 128, 160, 256, 257, 384, 512, 1000, 1024 };
    const int rows[]   = { 1, 7, 64 };
    int rc = 0, ncase = 0;
    uint64_t all = 1469598103934665603ULL;

    for (int w : widths) {
        for (int nr : rows) {
            for (int mode = 0; mode < 3; ++mode) {   // 0 = ASC, 1 = DESC, 2 = DESC thresh (store_ser)
                lcg r(0x5EEDull + (uint64_t) w*131 + (uint64_t) nr*7 + mode);
                // a small value set: many exact ties, signed zeros, -inf, a few distinct outliers
                const float pool[] = { 0.5f, 0.25f, -0.25f, 0.0f, -0.0f, 0.125f, 0.5f, 1.0f, -1.0f, -INFINITY, 0.75f, 0.0078125f };
                std::vector<float> x((size_t) w*nr);
                for (size_t i = 0; i < x.size(); ++i) {
                    const uint32_t v = r.next();
                    x[i] = (v % 5 == 0) ? (float) (v % 1000) / 997.0f : pool[v % (sizeof(pool)/sizeof(pool[0]))];
                }
                const int   min_e  = 2;
                const float thresh = 0.4f;

                ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
                ggml_context * ctx = ggml_init(ip);
                ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w, nr);
                ggml_tensor * y = mode == 0 ? ggml_argsort(ctx, a, GGML_SORT_ORDER_ASC)
                                : mode == 1 ? ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC)
                                :             ggml_argsort_thresh(ctx, a, min_e, thresh);
                if (!ggml_backend_supports_op(cuda, y)) {
                    printf("  w=%-5d rows=%-3d mode=%d SKIPPED (op not supported)\n", w, nr, mode);
                    ggml_free(ctx);
                    continue;
                }
                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, y);
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cuda);
                ggml_backend_tensor_set(a, x.data(), 0, x.size()*sizeof(float));
                if (ggml_backend_graph_compute(cuda, gf) != GGML_STATUS_SUCCESS) {
                    printf("  w=%-5d rows=%-3d mode=%d FAIL (graph compute)\n", w, nr, mode);
                    rc = 1;
                    ggml_backend_buffer_free(buf);
                    ggml_free(ctx);
                    continue;
                }
                std::vector<int32_t> got((size_t) w*nr);
                ggml_backend_tensor_get(y, got.data(), 0, got.size()*sizeof(int32_t));

                size_t nbad = 0;
                std::vector<int> ref;
                for (int ir = 0; ir < nr; ++ir) {
                    const float * xr = x.data() + (size_t) ir*w;
                    host_bitonic(xr, w, mode != 0, ref);
                    for (int c = 0; c < w; ++c) {
                        int want = ref[c];
                        if (mode == 2) {
                            const float mx = xr[ref[0]];
                            want = (c < min_e || xr[ref[c]] >= thresh*mx) ? ref[c] : -1;
                        }
                        if (got[(size_t) ir*w + c] != want) nbad++;
                    }
                }
                const uint64_t h = fnv(got.data(), got.size()*sizeof(int32_t));
                all = fnv(&h, sizeof(h), all);
                // widths > 1024 go to CUB; these widths all stay on the bitonic kernel
                const bool ok = nbad == 0;
                if (!ok) rc = 1;
                ncase++;
                printf("  w=%-5d rows=%-3d %-11s %s  %zu/%zu indices differ from the host network  HASH %016llx\n", w, nr,
                       mode == 0 ? "asc" : mode == 1 ? "desc" : "desc-thresh", ok ? "OK  " : "FAIL", nbad, got.size(),
                       (unsigned long long) h);
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
            }
        }
    }
    ggml_backend_free(cuda);
    printf("HASH-ALL %016llx over %d cases\n", (unsigned long long) all, ncase);
    printf("%s\n", rc ? "FAIL" : "OK");
    return rc;
}
