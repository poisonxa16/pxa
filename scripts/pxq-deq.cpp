// pxq-deq.cpp — dequantize a raw tensor-slice blob to f32 using the ENGINE'S OWN codecs.
// Helper for scripts/pxq-salience-gate.py. Authoritative for every ggml type with a
// to_float (IQ2_XXS / Q2_K / MXFP4 / Q8_0 / F16 ...) and for the panel-interleaved PXQ
// slab types via pxa_pxq_dequant_2d.
//
// Two traps this tool exists to not re-spring (DS4-FP8-REQUANT-2026-08-02.md):
//   1. ggml_init() populates ggml_table_f32_f16[]. Without it every fp16-scaled codec
//      (Q8_0, Q2_K, IQ2_XXS, ...) dequantises to ALL ZEROS while MXFP4 (arithmetic E8M0
//      scale) still works — a plausible-looking wrel of exactly 1.0000 instead of a crash.
//   2. IQ2/IQ3 grid tables are lazily built by ggml_quantize_init(); same silent-zeros
//      failure mode.
// Hence the all-zero guard at the bottom: refuse to write rather than report garbage.
//
// usage: pxq-deq <type_id> <nrows> <k> <in.raw> <out.f32>
#include "ggml.h"
#include "pxq-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 6) { fprintf(stderr, "usage: %s <type_id> <nrows> <k> <in.raw> <out.f32>\n", argv[0]); return 2; }
    struct ggml_init_params ip = { 1024*1024, NULL, true };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }

    const enum ggml_type t = (enum ggml_type) atoi(argv[1]);
    const int64_t nrows = atoll(argv[2]);
    const int64_t k     = atoll(argv[3]);

    FILE * fi = fopen(argv[4], "rb");
    if (!fi) { perror("in"); return 1; }
    fseek(fi, 0, SEEK_END); long insz = ftell(fi); fseek(fi, 0, SEEK_SET);
    std::vector<uint8_t> in(insz);
    if (fread(in.data(), 1, insz, fi) != (size_t) insz) { fprintf(stderr, "short read\n"); return 1; }
    fclose(fi);

    std::vector<float> out(nrows * k);

    if (pxa_pxq_is_cpu_supported(t)) {
        fprintf(stderr, "pxq-deq: type %d via pxa_pxq_dequant_2d (panel format), %lld bytes\n", (int)t, (long long)insz);
        pxa_pxq_dequant_2d(t, in.data(), out.data(), nrows, k);
    } else {
        ggml_quantize_init(t);
        ggml_type_traits_t tr = ggml_internal_get_type_traits(t);
        if (!tr.to_float) { fprintf(stderr, "pxq-deq: type %d has no to_float\n", (int)t); return 3; }
        const size_t rs = ggml_row_size(t, k);
        if ((int64_t)(rs * nrows) != (int64_t) insz) {
            fprintf(stderr, "pxq-deq: size mismatch: row_size %zu * nrows %lld = %lld vs blob %ld\n",
                    rs, (long long)nrows, (long long)(rs*nrows), insz);
            return 4;
        }
        fprintf(stderr, "pxq-deq: type %d via to_float, row_size=%zu\n", (int)t, rs);
        for (int64_t r = 0; r < nrows; ++r) tr.to_float(in.data() + r*rs, out.data() + r*k, k);
    }

    // guard: an all-zero dequant is almost always an uninitialised-table bug, not data.
    double s = 0, mx = 0; size_t nz = 0;
    for (size_t i = 0; i < out.size(); ++i) { double v = out[i]; s += v*v; if (v != 0) ++nz; if (fabs(v) > mx) mx = fabs(v); }
    fprintf(stderr, "pxq-deq: nonzero=%.4f%% rms=%.6g absmax=%.6g\n", 100.0*nz/out.size(), sqrt(s/out.size()), mx);
    if (nz == 0) { fprintf(stderr, "pxq-deq: ALL-ZERO OUTPUT — refusing to write\n"); return 5; }

    FILE * fo = fopen(argv[5], "wb");
    if (!fo) { perror("out"); return 1; }
    fwrite(out.data(), sizeof(float), out.size(), fo);
    fclose(fo);
    fprintf(stderr, "pxq-deq: wrote %lld floats\n", (long long)out.size());
    return 0;
}
