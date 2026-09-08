// pxq6_ref.cpp — standalone PXQ6 reference quantizer/dequantizer (CPU, bit-parity contract).
//
// Compiles src/pxq6-quantize.inc.cpp standalone (fp16 via F16C — the same IEEE RN conversion
// ggml uses when built with GGML_F16C=ON). Used by:
//   Q-G1  pxq6_golden.py  — byte-parity vs the numpy reference on random + edge-case slabs
//   Q-G2  pxq6_wrel.py    — wrel reproduction on the frozen 36-slice rng-42 lab protocol
//
// Usage:
//   pxq6_ref quant     R K tier w.f32 sig.f32|- out_q.bin
//   pxq6_ref roundtrip R K tier w.f32 sig.f32|- out_rec.f32 [out_q.bin]
// tier: 0 = PXQ6 core (bs16), 1 = PXQ6HQ (bs8). sig = K float32 imatrix column weights or '-'.
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

#include "../src/pxq6-quantize.inc.cpp"

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
    if (argc < 8) { fprintf(stderr, "usage: %s quant|roundtrip R K tier w.f32 sig.f32|- out [out_q.bin]\n", argv[0]); return 1; }
    const char * mode = argv[1];
    const int64_t R = atoll(argv[2]), K = atoll(argv[3]);
    const int tier = atoi(argv[4]);
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
    const int64_t qbytes = (R/64)*(PXQ6_HDR_BYTES + (K/32)*(int64_t)(tier ? PXQ6HQ_SLAB_BYTES : PXQ6_SLAB_BYTES));
    std::vector<uint8_t> q(qbytes);
    pxq6_quantize_tensor(W, q.data(), R, K, 1, sig, sig ? K : 0, 8, tier);

    if (!strcmp(mode, "quant")) {
        spew(argv[7], q.data(), q.size());
    } else if (!strcmp(mode, "roundtrip")) {
        std::vector<float> rec(R*K);
        pxq6_dequant_expert(q.data(), rec.data(), R, K, tier);
        spew(argv[7], rec.data(), rec.size()*4);
        if (argc > 8) spew(argv[8], q.data(), q.size());
    } else {
        fprintf(stderr, "unknown mode %s\n", mode); return 1;
    }
    return 0;
}
