// test-fa-qkv-direct -- the sm_60 quantized-KV attention routes:
//   PXA_FA_QKV_DIRECT  query width 1..8   (incumbent: vec-f32)
//   PXA_FA_QKV_TILE    query width > 8    (incumbent: tile-f16 over an f16 copy of the cache)
//
// Qwen3.8-27B shape: head 256, GQA 6 (24/4 heads, and 12/2 as one card of a tensor split), plus
// GQA 4 and 8; q4_0 / q8_0 K and V; widths 1..8 and 16 / 64 / 512; n_kv 256..131072 with the
// valid range not a multiple of the tile, split counts > 1 (the launcher's occupancy split at
// narrow widths), a causal tail, scattered masked cells and a -inf run longer than a split unit.
// For every case it:
//   - asks the CUDA backend to run the node (the route the environment selects);
//   - scores it against a double-precision reference over the dequantized K/V (normalised MSE and
//     max abs error);
//   - runs it 4 more times and requires all 5 outputs byte-identical (determinism);
//   - QKVD_DUMP=<file> writes the outputs to <file> and each case's errors to <file>.err;
//     QKVD_CMP=<file> reads such a pair (the incumbent arm) and prints the two routes' errors side
//     by side plus the max arm-vs-arm difference; with a comparison the case passes only if the
//     new route's max abs error is <= the incumbent's; with QKVD_EXACT=1 it passes only if every output byte equals
//     the incumbent arm's (an identity claim).
//   test-fa-qkv-direct colcheck                      every verify column vs its own one-token decode, byte compare
//     (PXA_FA_QKV_NY1 / PXA_FA_QKV_UNIF; red check: PXA_FA_QKV_NY1=0 must FAIL it)
//
//   test-fa-qkv-direct                               accuracy + determinism, all cases
//   test-fa-qkv-direct perf <type> <w> <n_kv> [iters] [gqa]
//                                                    time one node (us/call, median of iters)
//
// The two-arm run (what the lane ran before shipping):
//   PXA_FA_QKV_DIRECT=0 PXA_FA_QKV_TILE=0 QKVD_DUMP=/tmp/inc.bin test-fa-qkv-direct
//   PXA_CORE_ROUTES=1                     QKVD_CMP=/tmp/inc.bin  test-fa-qkv-direct

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static int g_fail = 0;

static float frand(unsigned & s) { s = s * 1664525u + 1013904223u; return (float) (s >> 8) / (float) (1u << 24) * 2.0f - 1.0f; }

struct fa_case {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * out = nullptr;
    std::vector<float> qh, kd, vd;   // kd / vd token-major: [(j*nh_kv + h)*D]
    std::vector<ggml_fp16_t> mh;
    std::vector<uint8_t> kq, vq;     // the quantized cache bytes (colcheck rebuilds width-1 nodes from them)
    size_t row_sz = 0;
    int D, nb, n_kv, nh, nh_kv, nb_pad;
    float scale;
    ~fa_case() { if (buf) ggml_backend_buffer_free(buf); if (ctx) ggml_free(ctx); }
};

// Build one node. valid = number of leading cells that hold keys (the rest is masked, as a partly
// filled cache is); the last nb cells of the valid range are causal for the nb queries.
static bool build_case(fa_case & fc, ggml_backend_t be, ggml_type tkv, int nb, int n_kv, int valid, int nh, int nh_kv,
                       unsigned seed, bool prec_f32) {
    const int D = 256;
    fc.D = D; fc.nb = nb; fc.n_kv = n_kv; fc.nh = nh; fc.nh_kv = nh_kv;
    fc.scale = 1.0f / sqrtf((float) D);
    unsigned s = seed;
    fc.qh.resize((size_t) D * nb * nh);
    // Q with a real model's spread (a few large channels), K/V unit-ish.
    for (size_t i = 0; i < fc.qh.size(); ++i) fc.qh[i] = frand(s) * ((i % 37) == 0 ? 6.0f : 1.5f);
    const size_t n_el = (size_t) D * n_kv * nh_kv;
    const size_t row_sz = ggml_row_size(tkv, D);
    std::vector<uint8_t> kq(row_sz * n_kv * nh_kv), vq(row_sz * n_kv * nh_kv);
    {
        // cache layout: [D, n_kv, nh_kv] with the head as the middle-fastest axis of a token row, like
        // the KV cache view (token stride = nh_kv rows, head stride = one row).
        std::vector<float> kt(n_el), vt(n_el);
        for (auto & x : kt) x = frand(s);
        for (auto & x : vt) x = frand(s) * 2.0f;
        ggml_quantize_chunk(tkv, kt.data(), kq.data(), 0, (int64_t) n_kv * nh_kv, D, nullptr, nullptr);
        ggml_quantize_chunk(tkv, vt.data(), vq.data(), 0, (int64_t) n_kv * nh_kv, D, nullptr, nullptr);
    }
    const auto tr = ggml_internal_get_type_traits(tkv);
    fc.kd.resize(n_el); fc.vd.resize(n_el);
    for (size_t r = 0; r < (size_t) n_kv * nh_kv; ++r) {
        tr.to_float(kq.data() + r * row_sz, fc.kd.data() + r * D, D);
        tr.to_float(vq.data() + r * row_sz, fc.vd.data() + r * D, D);
    }
    fc.nb_pad = GGML_PAD(nb, GGML_KQ_MASK_PAD);
    fc.mh.assign((size_t) n_kv * fc.nb_pad, ggml_fp32_to_fp16(-INFINITY));
    const int run0 = valid / 3, run1 = std::min(valid / 3 + 300, valid - nb); // a -inf run longer than a split unit
    for (int t = 0; t < nb; ++t) {
        const int last = valid - nb + t; // query t sees cells [0, last]
        for (int j = 0; j <= last && j < n_kv; ++j) {
            // a sprinkling of masked cells inside the window (other sequences' cells), and the run
            const bool hole = (j % 97) == 13 || (j >= run0 && j < run1);
            fc.mh[(size_t) t * n_kv + j] = ggml_fp32_to_fp16(hole ? -INFINITY : 0.0f);
        }
    }

    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    fc.ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(fc.ctx, GGML_TYPE_F32, D, nb, nh, 1);
    // K/V as the cache view: ne = [D, n_kv, nh_kv], nb1 = token stride, nb2 = head stride
    ggml_tensor * kc = ggml_new_tensor_1d(fc.ctx, tkv, (int64_t) D * n_kv * nh_kv);
    ggml_tensor * vc = ggml_new_tensor_1d(fc.ctx, tkv, (int64_t) D * n_kv * nh_kv);
    ggml_tensor * k = ggml_view_3d(fc.ctx, kc, D, n_kv, nh_kv, row_sz * nh_kv, row_sz, 0);
    ggml_tensor * v = ggml_view_3d(fc.ctx, vc, D, n_kv, nh_kv, row_sz * nh_kv, row_sz, 0);
    ggml_tensor * m = ggml_new_tensor_4d(fc.ctx, GGML_TYPE_F16, n_kv, fc.nb_pad, 1, 1);
    fc.out = ggml_flash_attn_ext(fc.ctx, q, k, v, m, fc.scale, 0.0f, 0.0f);
    if (prec_f32) ggml_flash_attn_ext_set_prec(fc.out, GGML_PREC_F32);
    if (!ggml_backend_supports_op(be, fc.out)) return false;
    fc.gf = ggml_new_graph(fc.ctx);
    ggml_build_forward_expand(fc.gf, fc.out);
    fc.buf = ggml_backend_alloc_ctx_tensors(fc.ctx, be);
    // q is [D, nb, nh]: host qh is laid out [h][t][D] -> tensor order is [h][t][D] as well
    ggml_backend_tensor_set(q, fc.qh.data(), 0, fc.qh.size() * sizeof(float));
    ggml_backend_tensor_set(kc, kq.data(), 0, kq.size());
    ggml_backend_tensor_set(vc, vq.data(), 0, vq.size());
    ggml_backend_tensor_set(m, fc.mh.data(), 0, fc.mh.size() * sizeof(ggml_fp16_t));
    fc.kq = std::move(kq); fc.vq = std::move(vq); fc.row_sz = row_sz;
    return true;
}

// colcheck (PXA_FA_QKV_NY1 / PXA_FA_QKV_UNIF): every column t of a width-nb verify node must equal, byte for byte, the
// one-token decode of that token: a width-1 node over the first pad256(last_t + 1) cells with mask row t.
static void run_colcheck(ggml_backend_t be, ggml_type tkv, int nb, int n_kv, int valid, int nh, int nh_kv) {
    char name[160];
    snprintf(name, sizeof(name), "colcheck %s width %d n_kv %d valid %d heads %d/%d", ggml_type_name(tkv), nb, n_kv, valid, nh, nh_kv);
    fa_case fw;
    if (!build_case(fw, be, tkv, nb, n_kv, valid, nh, nh_kv, 0xc01u + nb * 31u + (unsigned) valid, false)) {
        printf("  FAIL %s: CUDA declines the node\n", name); ++g_fail; return;
    }
    const int D = fw.D;
    std::vector<float> ow((size_t) D * nh * nb);
    if (ggml_backend_graph_compute(be, fw.gf) != GGML_STATUS_SUCCESS) { printf("  FAIL %s: compute\n", name); ++g_fail; return; }
    ggml_backend_tensor_get(fw.out, ow.data(), 0, ow.size() * sizeof(float));
    int bad_cols = 0; double maxd = 0.0;
    for (int t = 0; t < nb; ++t) {
        const int last = valid - nb + t;
        const int n1 = GGML_PAD(last + 1, 256);
        ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * c1 = ggml_init(ip);
        ggml_tensor * q = ggml_new_tensor_4d(c1, GGML_TYPE_F32, D, 1, nh, 1);
        ggml_tensor * kc = ggml_new_tensor_1d(c1, tkv, (int64_t) D * n1 * nh_kv);
        ggml_tensor * vc = ggml_new_tensor_1d(c1, tkv, (int64_t) D * n1 * nh_kv);
        ggml_tensor * k = ggml_view_3d(c1, kc, D, n1, nh_kv, fw.row_sz * nh_kv, fw.row_sz, 0);
        ggml_tensor * v = ggml_view_3d(c1, vc, D, n1, nh_kv, fw.row_sz * nh_kv, fw.row_sz, 0);
        const int pad1 = GGML_PAD(1, GGML_KQ_MASK_PAD);
        ggml_tensor * m = ggml_new_tensor_4d(c1, GGML_TYPE_F16, n1, pad1, 1, 1);
        ggml_tensor * o = ggml_flash_attn_ext(c1, q, k, v, m, fw.scale, 0.0f, 0.0f);
        ggml_cgraph * g1 = ggml_new_graph(c1);
        ggml_build_forward_expand(g1, o);
        ggml_backend_buffer_t b1 = ggml_backend_alloc_ctx_tensors(c1, be);
        std::vector<float> q1((size_t) D * nh);
        for (int h = 0; h < nh; ++h) memcpy(q1.data() + (size_t) h * D, fw.qh.data() + ((size_t) h * nb + t) * D, D * sizeof(float));
        ggml_backend_tensor_set(q, q1.data(), 0, q1.size() * sizeof(float));
        ggml_backend_tensor_set(kc, fw.kq.data(), 0, fw.row_sz * (size_t) n1 * nh_kv);
        ggml_backend_tensor_set(vc, fw.vq.data(), 0, fw.row_sz * (size_t) n1 * nh_kv);
        std::vector<ggml_fp16_t> m1((size_t) n1 * pad1, ggml_fp32_to_fp16(-INFINITY));
        memcpy(m1.data(), fw.mh.data() + (size_t) t * n_kv, (size_t) n1 * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(m, m1.data(), 0, m1.size() * sizeof(ggml_fp16_t));
        std::vector<float> o1((size_t) D * nh);
        const bool ok_c = ggml_backend_graph_compute(be, g1) == GGML_STATUS_SUCCESS;
        if (ok_c) ggml_backend_tensor_get(o, o1.data(), 0, o1.size() * sizeof(float));
        ggml_backend_buffer_free(b1); ggml_free(c1);
        if (!ok_c) { printf("  FAIL %s: width-1 compute t=%d\n", name, t); ++g_fail; return; }
        bool same = true;
        for (int h = 0; h < nh; ++h) {
            const float * a = ow.data() + ((size_t) t * nh + h) * D;
            const float * b = o1.data() + (size_t) h * D;
            if (memcmp(a, b, D * sizeof(float)) != 0) {
                same = false;
                for (int c = 0; c < D; ++c) maxd = std::max(maxd, (double) std::fabs(a[c] - b[c]));
            }
        }
        bad_cols += same ? 0 : 1;
    }
    printf("  %s %s: %d/%d columns byte-identical to their one-token decode%s\n", bad_cols ? "FAIL" : "ok  ", name,
           nb - bad_cols, nb, bad_cols ? "" : "");
    if (bad_cols) { printf("       max|diff| %.3g\n", maxd); ++g_fail; }
    fflush(stdout);
}

// Double reference for (t, h) rows, split over threads. Returns nmse and max abs error.
static void reference_errors(const fa_case & fc, const std::vector<float> & o, double & nmse, double & maxabs) {
    const int D = fc.D, nb = fc.nb, nh = fc.nh, nh_kv = fc.nh_kv, n_kv = fc.n_kv;
    const int nrows = nb * nh;
    int nthr = (int) std::thread::hardware_concurrency();
    nthr = std::max(1, std::min(nthr, 32));
    std::vector<double> t_err(nthr, 0.0), t_ref(nthr, 0.0), t_max(nthr, 0.0);
    auto work = [&](int ti) {
        std::vector<double> sc(n_kv), acc(D);
        for (int row = ti; row < nrows; row += nthr) {
            const int t = row / nh, h = row % nh;
            const int hk = h / (nh / nh_kv);
            const float * qr = fc.qh.data() + ((size_t) h * nb + t) * D;
            double mx = -1e300;
            for (int j = 0; j < n_kv; ++j) {
                const float mv = ggml_fp16_to_fp32(fc.mh[(size_t) t * n_kv + j]);
                if (std::isinf(mv)) { sc[j] = -INFINITY; continue; }
                const float * kr = fc.kd.data() + ((size_t) j * nh_kv + hk) * D;
                double d = 0; for (int c = 0; c < D; ++c) d += (double) qr[c] * kr[c];
                sc[j] = d * fc.scale + mv; if (sc[j] > mx) mx = sc[j];
            }
            double sum = 0; for (int j = 0; j < n_kv; ++j) { sc[j] = std::isinf(sc[j]) ? 0.0 : exp(sc[j] - mx); sum += sc[j]; }
            for (int c = 0; c < D; ++c) acc[c] = 0;
            for (int j = 0; j < n_kv; ++j) {
                if (sc[j] == 0.0) continue;
                const float * vr = fc.vd.data() + ((size_t) j * nh_kv + hk) * D;
                for (int c = 0; c < D; ++c) acc[c] += sc[j] * vr[c];
            }
            const float * orow = o.data() + ((size_t) t * nh + h) * D;
            for (int c = 0; c < D; ++c) {
                const double r = acc[c] / sum;
                const double e = orow[c] - r;
                t_err[ti] += e * e; t_ref[ti] += r * r;
                t_max[ti] = std::max(t_max[ti], std::fabs(e));
                if (!std::isfinite(orow[c])) { t_err[ti] += 1e30; t_max[ti] = 1e30; }
            }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < nthr; ++i) th.emplace_back(work, i);
    for (auto & x : th) x.join();
    double err = 0, ref2 = 0; maxabs = 0;
    for (int i = 0; i < nthr; ++i) { err += t_err[i]; ref2 += t_ref[i]; maxabs = std::max(maxabs, t_max[i]); }
    nmse = err / (ref2 > 0 ? ref2 : 1.0);
}

struct io_files { FILE * dump = nullptr; FILE * dump_err = nullptr; FILE * cmp = nullptr; FILE * cmp_err = nullptr; };

static void run_accuracy(ggml_backend_t be, ggml_type tkv, int nb, int n_kv, int valid, int nh, int nh_kv, bool prec_f32, io_files & io) {
    char name[160];
    snprintf(name, sizeof(name), "%s width %d n_kv %d valid %d heads %d/%d%s", ggml_type_name(tkv), nb, n_kv, valid, nh, nh_kv,
             prec_f32 ? " prec f32" : "");
    fa_case fc;
    if (!build_case(fc, be, tkv, nb, n_kv, valid, nh, nh_kv, 0x9e37u + nb * 131u + (unsigned) n_kv + (unsigned) tkv * 7u + (unsigned) nh, prec_f32)) {
        printf("  FAIL %s: CUDA declines the node\n", name); ++g_fail; return;
    }
    const size_t n_out = (size_t) fc.D * nh * nb;
    std::vector<float> o1(n_out), o2(n_out);
    if (ggml_backend_graph_compute(be, fc.gf) != GGML_STATUS_SUCCESS) { printf("  FAIL %s: compute\n", name); ++g_fail; return; }
    ggml_backend_tensor_get(fc.out, o1.data(), 0, n_out * sizeof(float));
    bool det = true;
    for (int rep = 0; rep < 4; ++rep) {
        if (ggml_backend_graph_compute(be, fc.gf) != GGML_STATUS_SUCCESS) { printf("  FAIL %s: compute %d\n", name, rep + 2); ++g_fail; return; }
        ggml_backend_tensor_get(fc.out, o2.data(), 0, n_out * sizeof(float));
        det = det && memcmp(o1.data(), o2.data(), n_out * sizeof(float)) == 0;
    }

    double nmse = 0, maxabs = 0;
    reference_errors(fc, o1, nmse, maxabs);

    double arm_diff = -1.0, inc_max = -1.0, inc_nmse = -1.0;
    bool arm_bytes_equal = true;   // QKVD_EXACT compares bytes (a -0 / +0 or NaN-payload difference counts)
    if (io.dump) fwrite(o1.data(), sizeof(float), n_out, io.dump);
    if (io.dump_err) fprintf(io.dump_err, "%.17g %.17g\n", maxabs, nmse); // round-trips exactly
    if (io.cmp) {
        std::vector<float> ob(n_out);
        if (fread(ob.data(), sizeof(float), n_out, io.cmp) == n_out) {
            arm_diff = 0.0;
            for (size_t i = 0; i < n_out; ++i) arm_diff = std::max(arm_diff, (double) std::fabs(o1[i] - ob[i]));
            arm_bytes_equal = memcmp(o1.data(), ob.data(), n_out * sizeof(float)) == 0;
        }
    }
    if (io.cmp_err && fscanf(io.cmp_err, "%lf %lf", &inc_max, &inc_nmse) != 2) { inc_max = inc_nmse = -1.0; }
    const bool vs_ok = inc_max < 0 || maxabs <= inc_max;
    // QKVD_EXACT=1 (with QKVD_CMP): an identity claim -- any nonzero byte difference against the incumbent arm fails
    static const bool exact = getenv("QKVD_EXACT") && atoi(getenv("QKVD_EXACT")) != 0;
    const bool exact_ok = !exact || !io.cmp || (arm_diff >= 0.0 && arm_bytes_equal);
    const bool ok = nmse < 1e-3 && det && vs_ok && exact_ok;
    printf("  %s %s: max|err| %.3g", ok ? "ok  " : "FAIL", name, maxabs);
    if (inc_max >= 0) printf(" (incumbent %.3g)", inc_max);
    printf("  nmse %.3g", nmse);
    if (inc_nmse >= 0) printf(" (incumbent %.3g)", inc_nmse);
    printf("  determinism %s", det ? "5/5 byte-identical" : "DIFFERS");
    if (arm_diff >= 0) printf("  max|arm diff| %.3g%s", arm_diff, exact && !arm_bytes_equal ? " (QKVD_EXACT: NOT BYTE-IDENTICAL)" : "");
    printf("\n");
    fflush(stdout);
    if (!ok) ++g_fail;
}

static int run_perf(ggml_backend_t be, const char * tname, int nb, int n_kv, int iters, int gqa) {
    ggml_type t = GGML_TYPE_Q4_0;
    if (!strcmp(tname, "q8_0")) t = GGML_TYPE_Q8_0;
    else if (!strcmp(tname, "f16")) t = GGML_TYPE_F16;
    // QKVD_NH_KV=<n>: KV heads of the node (default 4, the whole model; 2 is one card of a 2-way tensor split)
    const int nh_kv = getenv("QKVD_NH_KV") ? atoi(getenv("QKVD_NH_KV")) : 4, nh = nh_kv * gqa;
    fa_case fc;
    if (!build_case(fc, be, t, nb, n_kv, n_kv, nh, nh_kv, 0x5eedu, false)) { printf("perf: node declined\n"); return 1; }
    for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(be, fc.gf);
    std::vector<double> us;
    for (int i = 0; i < iters; ++i) {
        ggml_backend_synchronize(be);
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(be, fc.gf);
        ggml_backend_synchronize(be);
        const auto t1 = std::chrono::steady_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(us.begin(), us.end());
    printf("perf %s width %d n_kv %d gqa %d: median %.1f us  min %.1f us  (iters %d)\n", tname, nb, n_kv, gqa, us[us.size() / 2], us[0], iters);
    return 0;
}

int main(int argc, char ** argv) {
    { ggml_init_params ip0 = { 1024, nullptr, true }; ggml_free(ggml_init(ip0)); } // fp16 tables (see test-fa-route-qkv)
    if (ggml_backend_cuda_get_device_count() < 1) { printf("SKIP: no CUDA device\n"); return 0; }
    ggml_backend_t be = ggml_backend_cuda_init(0, nullptr);
    if (!be) { printf("SKIP: cuda init failed\n"); return 0; }
    if (argc > 1 && !strcmp(argv[1], "colcheck")) {
        // verify columns vs their one-token decodes, the PXA_FA_QKV_NY1 promise (and PXA_FA_QKV_UNIF's)
        for (ggml_type t : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }) {
            for (int nb = 2; nb <= 8; ++nb) {
                run_colcheck(be, t, nb, 2048, 1500 + nb, 24, 4);
                run_colcheck(be, t, nb, 32768, 20560 + nb, 24, 4);
            }
            run_colcheck(be, t, 4, 2048, 1025, 24, 4);   // straddles 1024: k_max 1024,1024,1024,1280
            run_colcheck(be, t, 4, 2048, 1026, 24, 4);
            run_colcheck(be, t, 3, 2048, 1025, 16, 4);   // GQA 4
            run_colcheck(be, t, 3, 4096, 3000, 32, 4);   // GQA 8
            run_colcheck(be, t, 2, 8192, 300, 24, 4);    // short cache in a long view
        }
        ggml_backend_free(be);
        printf("%s\n", g_fail ? "FAILED" : "PASSED");
        return g_fail ? 1 : 0;
    }
    if (argc > 1 && !strcmp(argv[1], "perf")) {
        const char * tn = argc > 2 ? argv[2] : "q4_0";
        const int w = argc > 3 ? atoi(argv[3]) : 1;
        const int nkv = argc > 4 ? atoi(argv[4]) : 32768;
        const int it = argc > 5 ? atoi(argv[5]) : 50;
        const int g = argc > 6 ? atoi(argv[6]) : 6;
        const int rc = run_perf(be, tn, w, nkv, it, g);
        ggml_backend_free(be);
        return rc;
    }
    const int cc = ggml_backend_cuda_get_device_cc(0);
    io_files io;
    if (const char * dp = getenv("QKVD_DUMP")) {
        io.dump = fopen(dp, "wb");
        io.dump_err = fopen((std::string(dp) + ".err").c_str(), "w");
    }
    if (const char * cp = getenv("QKVD_CMP")) {
        io.cmp = fopen(cp, "rb");
        io.cmp_err = fopen((std::string(cp) + ".err").c_str(), "r");
    }
    printf("test-fa-qkv-direct: device 0 cc %d, PXA_FA_QKV_DIRECT=%s PXA_FA_QKV_TILE=%s%s\n", cc,
           getenv("PXA_FA_QKV_DIRECT") ? getenv("PXA_FA_QKV_DIRECT") : "(unset)",
           getenv("PXA_FA_QKV_TILE") ? getenv("PXA_FA_QKV_TILE") : "(unset)",
           io.cmp_err ? ", comparing against the incumbent arm's errors" : "");
    for (ggml_type t : { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0 }) {
        // ---- narrow (PXA_FA_QKV_DIRECT) ----
        for (int nb = 1; nb <= 8; ++nb) {
            run_accuracy(be, t, nb, 1024, 1024, 24, 4, false, io);   // full window
            run_accuracy(be, t, nb, 8192, 5000, 24, 4, false, io);   // partly filled cache, deeper
        }
        run_accuracy(be, t, 1, 4096, 4096, 24, 4, true, io);          // f32 precision request
        run_accuracy(be, t, 3, 2048, 1900, 16, 4, false, io);         // GQA 4
        run_accuracy(be, t, 8, 2048, 1900, 16, 4, false, io);
        run_accuracy(be, t, 5, 2048, 2048, 32, 4, false, io);         // GQA 8
        run_accuracy(be, t, 7, 2048, 2000, 32, 4, false, io);         // GQA 8, two column tiles
        run_accuracy(be, t, 8, 2048, 2048, 32, 4, false, io);
        run_accuracy(be, t, 2, 256, 200, 24, 4, false, io);           // one split unit, partly valid
        run_accuracy(be, t, 4, 768, 700, 24, 4, false, io);
        // PXA_FA_QKV_NY1 verify columns that straddle a 256-key boundary (k_max_c differs inside a column pair:
        // the PXA_FA_QKV_NY1_SHARE kernel runs such a pair one column at a time), and a deep pair-shared case
        run_accuracy(be, t, 4, 2048, 1025, 24, 4, false, io);         // columns 1021..1024: k_max 1024,1024,1024,1280
        run_accuracy(be, t, 3, 2048, 1025, 24, 4, false, io);         // 1022..1024: k_max 1024,1024,1280
        run_accuracy(be, t, 4, 2048, 1026, 24, 4, false, io);         // 1022..1025: 1024,1024,1280,1280
        run_accuracy(be, t, 4, 32768, 20560, 24, 4, false, io);       // the beta-2 long-prompt depth (~20.5k)
        run_accuracy(be, t, 2, 32768, 20561, 16, 4, false, io);       // GQA 4
        run_accuracy(be, t, 3, 32768, 20560, 32, 4, false, io);       // GQA 8
        run_accuracy(be, t, 1, 131072, 131000, 12, 2, false, io);     // one card of a tensor split, deep
        run_accuracy(be, t, 3, 131072, 130001, 12, 2, false, io);
        run_accuracy(be, t, 8, 65536, 60000, 12, 2, false, io);
        // ---- wide (PXA_FA_QKV_TILE) ----
        run_accuracy(be, t, 16, 2048, 1500, 24, 4, false, io);
        run_accuracy(be, t, 64, 4096, 4096, 24, 4, false, io);
        run_accuracy(be, t, 512, 16384, 16384, 24, 4, false, io);     // prefill ubatch at depth
        run_accuracy(be, t, 16, 2048, 2000, 16, 4, false, io);        // GQA 4
        run_accuracy(be, t, 16, 2048, 2000, 32, 4, false, io);        // GQA 8
        run_accuracy(be, t, 64, 131072, 131072, 12, 2, false, io);    // wide at depth, one card of a split
    }
    for (FILE * f : { io.dump, io.dump_err, io.cmp, io.cmp_err }) if (f) fclose(f);
    ggml_backend_free(be);
    printf("%s\n", g_fail ? "FAILED" : "PASSED");
    return g_fail ? 1 : 0;
}
