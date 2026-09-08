// pxqu_ref.cpp — standalone PXQ2/PXQ3 reference quantizer/dequantizer (CPU, parity contract).
// Clone of pxq6_ref.cpp for the PXQ-UNIVERSAL low-bit types (build gate Q-G1/Q-G2).
//
// Compiles src/pxq2-quantize.inc.cpp + src/pxq3-quantize.inc.cpp standalone (fp16 via F16C —
// the same IEEE RN conversion ggml uses when built with GGML_F16C=ON). Used by:
//   Q-G1  pxqu_golden.py — byte-parity vs the numpy reference on random + edge-case slabs
//   Q-G2  pxqu_wrel.py   — wrel reproduction on the frozen 36-slice rng-42 lab protocol
//
// Usage:
//   pxqu_ref quant     TYPE R K w.f32 sig.f32|- out_q.bin
//   pxqu_ref roundtrip TYPE R K w.f32 sig.f32|- out_rec.f32 [out_q.bin]
// TYPE: 2 = PXQ2 (LM4, 2-bit), 3 = PXQ3 (LM8, 3-bit bit-plane).
// sig = K float32 imatrix column weights or '-'.
//
// Build:
//   g++ -O2 -mf16c -std=c++17 -o pxqu_ref pxa-bench/pxqu_ref.cpp -pthread
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>
#include <immintrin.h>

typedef uint16_t ggml_fp16_t;
static inline float ggml_fp16_to_fp32(ggml_fp16_t h) { return _cvtsh_ss(h); }
static inline ggml_fp16_t ggml_fp32_to_fp16(float f) { return _cvtss_sh(f, 0 /*RN*/); }

// dead-column guard helper — mirror of the static in src/llama-quantize.cpp (the .inc.cpp
// quantizers grew a call to it 2026-07-2x; this standalone tool must supply the same symbol).
#include <atomic>
#include <cmath>
static std::atomic<int64_t> g_pxq_imx_dead_cols{0};
static bool pxq_imatrix_column_usable(const float * w, int64_t K) {
    if (!w) return false;
    double s = 0.0;
    for (int64_t i = 0; i < K; ++i) {
        if (!std::isfinite(w[i]) || w[i] < 0.0f) { g_pxq_imx_dead_cols.fetch_add(1); return false; }
        s += (double) w[i];
    }
    if (s > 0.0) return true;
    g_pxq_imx_dead_cols.fetch_add(1);
    return false;
}

#include "../src/pxq2-quantize.inc.cpp"
#include "../src/pxq3-quantize.inc.cpp"

static std::vector<uint8_t> slurp(const char * p) {
    FILE * f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", p); exit(2); }
    fclose(f);
    return v;
}
static void spew(const char * p, const void * d, size_t n) {
    FILE * f = fopen(p, "wb");
    if (!f) { fprintf(stderr, "cannot open %s for write\n", p); exit(2); }
    fwrite(d, 1, n, f); fclose(f);
}

int main(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s quant|roundtrip TYPE(2|3) R K w.f32 sig.f32|- out [out_q.bin]\n", argv[0]); return 1; }
    const char * mode = argv[1];
    const int type = atoi(argv[2]);
    const int64_t R = atoll(argv[3]), K = atoll(argv[4]);
    if (type != 2 && type != 3) { fprintf(stderr, "TYPE must be 2 or 3\n"); return 1; }
    if (R % 64 || K % 32) { fprintf(stderr, "R%%64/K%%32 violation\n"); return 1; }
    auto wraw = slurp(argv[5]);
    if ((int64_t)wraw.size() != R*K*4) { fprintf(stderr, "w.f32 size mismatch (%zu vs %lld)\n", wraw.size(), (long long)(R*K*4)); return 1; }
    const float * W = (const float *)wraw.data();
    std::vector<uint8_t> sraw;
    const float * sig = nullptr;
    if (strcmp(argv[6], "-") != 0) {
        sraw = slurp(argv[6]);
        if ((int64_t)sraw.size() != K*4) { fprintf(stderr, "sig size mismatch\n"); return 1; }
        sig = (const float *)sraw.data();
    }
    const int64_t slab  = type == 2 ? PXQ2_SLAB_BYTES : PXQ3_SLAB_BYTES;
    const int64_t qbytes = (R/64)*(128 + (K/32)*slab);
    std::vector<uint8_t> q(qbytes);
    if (type == 2) pxq2_quantize_tensor(W, q.data(), R, K, 1, sig, sig ? K : 0, 8);
    else           pxq3_quantize_tensor(W, q.data(), R, K, 1, sig, sig ? K : 0, 8);

    if (strcmp(mode, "quant") == 0) {
        spew(argv[7], q.data(), q.size());
    } else if (strcmp(mode, "roundtrip") == 0) {
        std::vector<float> rec(R*K);
        if (type == 2) pxq2_dequant_expert(q.data(), rec.data(), R, K);
        else           pxq3_dequant_expert(q.data(), rec.data(), R, K);
        spew(argv[7], rec.data(), rec.size()*4);
        if (argc > 8) spew(argv[8], q.data(), q.size());
    } else {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 1;
    }
    return 0;
}
