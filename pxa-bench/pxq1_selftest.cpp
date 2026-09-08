// pxq1_selftest.cpp — standalone gate for the PXQ1 native codec.
//   ./pxq1_selftest R K w.f32 sig.f32|- out_q.bin out_rec.f32
// Quantizes one [R,K] expert with pxq1_quantize_tensor (the SHIPPED code), writes the raw
// PXQ1 bytes + the CPU-reference dequant. A numpy driver then (a) re-derives the bytes and
// checks byte-parity, and (b) compares the recon to the Q8 original (cos).
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
static inline ggml_fp16_t ggml_fp32_to_fp16(float f) { return _cvtss_sh(f, 0); }
#include "../src/pxq1-quantize.inc.cpp"

int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage: R K w.f32 sig.f32|- out_q.bin out_rec.f32\n"); return 2; }
    long R = atol(argv[1]), K = atol(argv[2]);
    std::vector<float> w(R*K);
    FILE * f = fopen(argv[3], "rb"); fread(w.data(), 4, R*K, f); fclose(f);
    std::vector<float> sig; const float * sptr = nullptr;
    if (strcmp(argv[4], "-") != 0) { sig.resize(K); f = fopen(argv[4], "rb"); fread(sig.data(), 4, K, f); fclose(f); sptr = sig.data(); }
    long ebytes = (R/64)*(PXQ1_HDR_BYTES + (K/32)*(long)PXQ1_SLAB_BYTES);
    std::vector<uint8_t> q(ebytes);
    pxq1_quantize_tensor(w.data(), q.data(), R, K, 1, sptr, sptr ? K : 0, 1);
    f = fopen(argv[5], "wb"); fwrite(q.data(), 1, ebytes, f); fclose(f);
    std::vector<float> rec(R*K);
    pxq1_dequant_expert(q.data(), rec.data(), R, K);
    f = fopen(argv[6], "wb"); fwrite(rec.data(), 4, R*K, f); fclose(f);
    printf("OK ebytes=%ld\n", ebytes);
    return 0;
}
