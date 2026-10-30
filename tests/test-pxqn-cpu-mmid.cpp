// test-pxqn-cpu-mmid (2026-10-23): host PXQN routed experts on the CPU MUL_MAT_ID work queue
// (PXA_PXQN_CPU_MMID_Q=1, default) must equal the old per-expert library call (=0) byte for byte, and this is the dev timing
// harness for it. The experts are a real model's: one layer's ffn_{gate,up,down}_exps read from a GGUF file, so the same
// harness times a PXQN file and a k-quant file of the same model at equal shapes.
//
//   test-pxqn-cpu-mmid <model.gguf> [-l layer] [-t threads] [-r reps] [-T 1,4,32,512] [-k n_used] [--rc 64,128] [--mat gate,up,down]
//
// Every case runs mode 0 (old) and mode 1 (queue) at each --rc on the same routing and compares all routed outputs (PXQN only;
// other types run their default path and are timed only). With -r > 0 each rep relabels the routed experts (+7 mod E) so the
// weights stream from DRAM as in a model, the modes interleaved per rep; prints mean ms per op and GB/s over the routed experts'
// weight bytes. Exit 1 on any byte difference.
#include "ggml.h"
#include "ggml-pxa-lean.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::vector<int> parse_list(const char * s) {
    std::vector<int> v;
    std::string str(s);
    size_t p = 0;
    while (p < str.size()) {
        size_t q = str.find(',', p);
        if (q == std::string::npos) q = str.size();
        v.push_back(atoi(str.substr(p, q - p).c_str()));
        p = q + 1;
    }
    return v;
}

static uint64_t fnv(const void * p, size_t n) {
    const unsigned char * b = (const unsigned char *) p; uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [-l layer] [-t threads] [-r reps] [-T 1,4,32,512] [-k n_used] [--rc 64,128] [--mat gate,up,down]\n", argv[0]);
        return 2;
    }
    const char * fname = argv[1];
    int layer = -1, nthreads = 8, reps = 0, n_used = 0;
    std::vector<int> Ts = { 1, 4, 32, 512 }, rcs = { 64 };
    std::string mats = "gate,up,down";
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : (const char *) ""; };
        if      (a == "-l")    layer = atoi(next());
        else if (a == "-t")    nthreads = atoi(next());
        else if (a == "-r")    reps = atoi(next());
        else if (a == "-T")    Ts = parse_list(next());
        else if (a == "-k")    n_used = atoi(next());
        else if (a == "--rc")  rcs = parse_list(next());
        else if (a == "--mat") mats = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    ggml_time_init();

    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &meta };
    gguf_context * g = gguf_init_from_file(fname, gp);
    if (!g) { fprintf(stderr, "cannot read %s\n", fname); return 2; }
    // n_used: <arch>.expert_used_count unless given
    if (n_used <= 0) {
        for (int i = 0; i < gguf_get_n_kv(g); ++i) {
            const char * key = gguf_get_key(g, i);
            const size_t kl = strlen(key);
            if (kl > 18 && !strcmp(key + kl - 18, ".expert_used_count")) { n_used = (int) gguf_get_val_u32(g, i); break; }
        }
        if (n_used <= 0) n_used = 8;
    }
    if (layer < 0) {   // the middle layer that has routed experts
        int nl = 0;
        for (int il = 0; il < 1000; ++il) {
            char nm[128]; snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", il);
            if (gguf_find_tensor(g, nm) >= 0) nl = il + 1;
        }
        layer = nl / 2;
    }
    const size_t data_off = gguf_get_data_offset(g);
    FILE * f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", fname); return 2; }

    std::vector<std::string> mat_names;
    { std::string m = mats; size_t p = 0; while (p < m.size()) { size_t q = m.find(',', p); if (q == std::string::npos) q = m.size(); mat_names.push_back(m.substr(p, q - p)); p = q + 1; } }

    // weights context: the chosen matrices of this layer
    size_t wbytes = 0;
    for (const auto & mn : mat_names) {
        char nm[128]; snprintf(nm, sizeof(nm), "blk.%d.ffn_%s_exps.weight", layer, mn.c_str());
        ggml_tensor * t = ggml_get_tensor(meta, nm);
        if (!t) { fprintf(stderr, "no tensor %s\n", nm); return 2; }
        wbytes += ggml_nbytes(t) + 256;
    }
    ggml_init_params wp = { wbytes + 1024*1024, nullptr, false };
    ggml_context * wctx = ggml_init(wp);
    std::vector<ggml_tensor *> W;
    for (const auto & mn : mat_names) {
        char nm[128]; snprintf(nm, sizeof(nm), "blk.%d.ffn_%s_exps.weight", layer, mn.c_str());
        ggml_tensor * mt = ggml_get_tensor(meta, nm);
        ggml_tensor * t = ggml_new_tensor_3d(wctx, mt->type, mt->ne[0], mt->ne[1], mt->ne[2]);
        const int idx = gguf_find_tensor(g, nm);
        const size_t off = data_off + gguf_get_tensor_offset(g, idx);
        if (fseeko(f, (off_t) off, SEEK_SET) != 0 || fread(t->data, 1, ggml_nbytes(t), f) != ggml_nbytes(t)) {
            fprintf(stderr, "read failed %s\n", nm); return 2;
        }
        W.push_back(t);
    }
    fclose(f);

    printf("test-pxqn-cpu-mmid: %s layer %d n_used %d threads %d reps %d\n", fname, layer, n_used, nthreads, reps);
    int bad = 0;
    double sum_new_T1 = 0, sum_old_T1 = 0;
    for (size_t im = 0; im < W.size(); ++im) {
        ggml_tensor * as = W[im];
        const bool up = mat_names[im] != "down";
        const int64_t K = as->ne[0], N = as->ne[1], E = as->ne[2];
        const bool is_pxqn = (int) as->type >= 257 && (int) as->type <= 264;   // the PXQN family (ggml.h)
        for (int T : Ts) {
            const int64_t ne11 = up ? 1 : n_used;
            ggml_init_params ip = { (size_t) (K*ne11*T + N*n_used*T)*4 + (size_t) n_used*T*4 + 256ull*1024*1024, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, ne11, T);
            ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, T);
            std::mt19937 rng(4242 + (int) K + T);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (int64_t i = 0; i < ggml_nelements(b); ++i) ((float *) b->data)[i] = nd(rng);
            std::vector<int> pool(E);
            for (int t = 0; t < T; ++t) {
                for (int64_t e = 0; e < E; ++e) pool[e] = (int) e;
                for (int j = 0; j < n_used; ++j) {
                    std::uniform_int_distribution<int64_t> ud(j, E - 1);
                    std::swap(pool[j], pool[ud(rng)]);
                    ((int32_t *) ids->data)[t*n_used + j] = pool[j];
                }
            }
            ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            const size_t ob = ggml_nbytes(out);
            std::vector<uint8_t> ref(ob);
            // arms: 0 = old; 1.. = queue at each rc (PXQN only)
            const int n_arms = is_pxqn ? 1 + (int) rcs.size() : 1;
            auto set_arm = [&](int a) {
                if (!is_pxqn) return;
                ggml_pxa_set_cpu_mmid_pxqn(a == 0 ? 0 : 1);
                if (a > 0) ggml_pxa_set_cpu_mmid_pxqn_rc(rcs[a - 1]);
            };
            bool ok = true;
            for (int a = 0; a < n_arms; ++a) {
                set_arm(a);
                memset(out->data, 0xA5, ob);
                ggml_graph_compute_with_ctx(ctx, gf, nthreads);
                if (a == 0) memcpy(ref.data(), out->data, ob);
                else if (memcmp(ref.data(), out->data, ob) != 0) {
                    size_t nd2 = 0; const float * x = (const float *) ref.data(); const float * y = (const float *) out->data;
                    for (size_t i = 0; i < ob/4; ++i) nd2 += memcmp(&x[i], &y[i], 4) != 0;
                    printf("  %-5s T=%-4d MISMATCH rc %d: %zu of %zu floats differ\n", mat_names[im].c_str(), T, rcs[a - 1], nd2, ob/4);
                    ok = false;
                }
            }
            std::vector<double> tm(n_arms, 0.0);
            double gb = 0;
            if (reps > 0) {
                std::vector<int32_t> ids0((const int32_t *) ids->data, (const int32_t *) ids->data + n_used*T);
                for (int r = 0; r < reps; ++r) {
                    for (int64_t i = 0; i < (int64_t) n_used*T; ++i) ((int32_t *) ids->data)[i] = (int32_t) ((ids0[i] + 7*(r + 1)) % E);
                    if (r == 0) {
                        std::vector<char> used(E, 0);
                        for (int64_t i = 0; i < (int64_t) n_used*T; ++i) used[((int32_t *) ids->data)[i]] = 1;
                        int64_t nu = 0; for (char c : used) nu += c;
                        gb = (double) nu * (double) as->nb[2] / 1e9;
                    }
                    for (int a0 = 0; a0 < n_arms; ++a0) {
                        const int a = (a0 + r) % n_arms;   // rotate the arm order per rep
                        set_arm(a);
                        const int64_t t0 = ggml_time_us();
                        ggml_graph_compute_with_ctx(ctx, gf, nthreads);
                        tm[a] += (ggml_time_us() - t0) / 1000.0 / reps;
                    }
                }
                memcpy(ids->data, ids0.data(), ids0.size()*sizeof(int32_t));
            }
            printf("  %-5s %-7s K=%5lld N=%5lld E=%4lld k=%d T=%4d hash %016llx %s", mat_names[im].c_str(), ggml_type_name(as->type),
                   (long long) K, (long long) N, (long long) E, n_used, T, (unsigned long long) fnv(ref.data(), ob),
                   is_pxqn ? (ok ? "IDENTICAL" : "DIFF") : "(timed only)");
            if (reps > 0) {
                printf("  %s %.3f ms (%.1f GB/s)", is_pxqn ? "old" : "def", tm[0], gb / (tm[0] / 1000.0));
                for (int a = 1; a < n_arms; ++a) printf("  q%d %.3f ms x%.3f", rcs[a - 1], tm[a], tm[0] / tm[a]);
                if (T == 1) { sum_old_T1 += tm[0]; sum_new_T1 += n_arms > 1 ? tm[1] : tm[0]; }
            }
            printf("\n");
            bad += ok ? 0 : 1;
            ggml_free(ctx);
        }
    }
    if (reps > 0 && sum_old_T1 > 0) printf("T=1 layer sum over [%s]: old/def %.3f ms  q%d %.3f ms  x%.3f\n", mats.c_str(), sum_old_T1, rcs[0], sum_new_T1, sum_old_T1 / sum_new_T1);
    printf("%s: %d case(s) differ\n", bad ? "FAIL" : "PASS", bad);
    ggml_free(wctx);
    ggml_free(meta);
    gguf_free(g);
    return bad ? 1 : 0;
}
