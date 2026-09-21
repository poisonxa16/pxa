// pxq-mmvq-bench.cpp — cudaEvent-timed GEMV microbench on REAL model tensors.
//
// Loads one named tensor's raw quantized bytes straight out of a GGUF (no full-model load),
// puts it on a CUDA backend, and times ggml_mul_mat(W, x) for a list of M (= src1->ne[1]).
// Both the PXQ4 and the MXFP4 arms go through the ordinary ggml-cuda dispatch, so whatever
// the engine would really run for a decode step is what gets timed here.
//
//   pxq-mmvq-bench <gguf> <tensor[,tensor...]> [M-list] [iters]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out; std::string cur;
    for (char c : s) { if (c == sep) { if (!cur.empty()) out.push_back(cur); cur.clear(); } else cur += c; }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

struct tinfo { std::string name; ggml_type type; int64_t ne[2]; size_t off; size_t nbytes; };

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <gguf> <tensor[,tensor...]> [M-list] [iters]\n", argv[0]); return 1; }
    const std::string path = argv[1];
    const auto names = split(argv[2], ',');
    const auto Ms    = argc > 3 ? split(argv[3], ',') : std::vector<std::string>{"1"};
    const int  iters = argc > 4 ? atoi(argv[4]) : 200;

    // ---- locate the tensors in the GGUF without materialising the model -------------------
    // no_alloc + a metadata context: the tensors come back with real ne[]/type and no data.
    struct ggml_context * meta = nullptr;
    struct gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &meta };
    struct gguf_context * gguf = gguf_init_from_file(path.c_str(), gp);
    if (!gguf || !meta) { fprintf(stderr, "gguf_init_from_file failed: %s\n", path.c_str()); return 1; }
    const size_t data_off = gguf_get_data_offset(gguf);

    std::vector<tinfo> tis;
    for (const auto & n : names) {
        const int id = gguf_find_tensor(gguf, n.c_str());
        if (id < 0) { fprintf(stderr, "tensor not found: %s\n", n.c_str()); return 1; }
        struct ggml_tensor * mt = ggml_get_tensor(meta, n.c_str());
        if (!mt) { fprintf(stderr, "no metadata for %s\n", n.c_str()); return 1; }
        if (ggml_n_dims(mt) != 2) { fprintf(stderr, "%s: need a 2D tensor\n", n.c_str()); return 1; }
        tinfo ti;
        ti.name   = n;
        ti.type   = mt->type;
        ti.ne[0]  = mt->ne[0];
        ti.ne[1]  = mt->ne[1];
        ti.off    = data_off + gguf_get_tensor_offset(gguf, id);
        ti.nbytes = ggml_nbytes(mt);
        tis.push_back(ti);
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) { fprintf(stderr, "ggml_backend_cuda_init(0, nullptr) failed\n"); return 1; }

    printf("# %-22s %-8s %8s %8s %4s %10s %10s %10s\n",
           "tensor", "type", "K", "N", "M", "us/call", "GB/s", "MiB");

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "fopen failed\n"); return 1; }

    for (const auto & ti : tis) {
        // one context per tensor: weights stay resident only while we bench them
        struct ggml_init_params ip = { ggml_tensor_overhead()*8, nullptr, true };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, ti.type, ti.ne[0], ti.ne[1]);
        ggml_set_name(w, "w");

        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!wbuf) { fprintf(stderr, "%s: weight alloc failed (%zu MiB)\n", ti.name.c_str(), ti.nbytes>>20); return 1; }

        std::vector<uint8_t> host(ti.nbytes);
        if (fseek(f, (long) ti.off, SEEK_SET) != 0 || fread(host.data(), 1, ti.nbytes, f) != ti.nbytes) {
            fprintf(stderr, "%s: read failed\n", ti.name.c_str()); return 1;
        }
        ggml_backend_tensor_set(w, host.data(), 0, ti.nbytes);
        host.clear(); host.shrink_to_fit();

        for (const auto & Ms_ : Ms) {
            const int M = atoi(Ms_.c_str());

            struct ggml_init_params ip2 = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
            struct ggml_context * c2 = ggml_init(ip2);
            struct ggml_tensor * x = ggml_new_tensor_2d(c2, GGML_TYPE_F32, ti.ne[0], M);
            ggml_set_name(x, "x");
            struct ggml_tensor * y = ggml_mul_mat(c2, w, x);
            struct ggml_cgraph * gf = ggml_new_graph(c2);
            ggml_build_forward_expand(gf, y);

            ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "graph alloc failed\n"); return 1; }

            srand(1234);
            std::vector<float> xh((size_t) ti.ne[0]*M);
            for (auto & v : xh) v = (float) ((rand() % 2001) - 1000) / 1000.0f;
            ggml_backend_tensor_set(x, xh.data(), 0, ggml_nbytes(x));

            for (int i = 0; i < 20; ++i) ggml_backend_graph_compute(backend, gf);   // warmup
            ggml_backend_synchronize(backend);

            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            const auto t1 = std::chrono::high_resolution_clock::now();

            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
            const double gbs = (double) ti.nbytes / (us * 1e-6) / 1e9;
            printf("  %-22s %-8s %8lld %8lld %4d %10.1f %10.1f %10.1f\n",
                   ti.name.c_str(), ggml_type_name(ti.type), (long long) ti.ne[0], (long long) ti.ne[1],
                   M, us, gbs, ti.nbytes / 1048576.0);
            fflush(stdout);

            // optional: dump x and y for the offline float-reference parity check
            const char * dd = getenv("PXQBENCH_DUMP");
            if (dd && M == 1) {
                std::vector<float> yh((size_t) ti.ne[1]);
                ggml_backend_tensor_get(y, yh.data(), 0, ggml_nbytes(y));
                char fn[1024];
                snprintf(fn, sizeof(fn), "%s/%s.y.f32", dd, ti.name.c_str());
                FILE * o = fopen(fn, "wb"); if (o) { fwrite(yh.data(), 4, yh.size(), o); fclose(o); }
                snprintf(fn, sizeof(fn), "%s/%s.x.f32", dd, ti.name.c_str());
                o = fopen(fn, "wb"); if (o) { fwrite(xh.data(), 4, xh.size(), o); fclose(o); }
            }

            ggml_gallocr_free(alloc);
            ggml_free(c2);
        }

        ggml_backend_buffer_free(wbuf);
        ggml_free(ctx);
    }

    fclose(f);
    ggml_backend_free(backend);
    gguf_free(gguf);
    return 0;
}
