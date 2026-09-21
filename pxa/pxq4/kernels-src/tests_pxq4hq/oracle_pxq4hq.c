// oracle_pxq4hq.c -- CPU decode oracle for PXQ4HQ, used to gate tiers.dequant() and the
// device kernels.
//
// The decode function is NOT written here: it is EXTRACTED VERBATIM at build time from the
// engine's own ggml/src/pxq-cpu.c by build_oracle_pxq4hq.sh, which cuts out
// ``pxa_deq_row_pxq6`` -- the shared body that serves both the core tier and the HQ tier via
// its ``bool hq`` parameter -- together with the pair helper it calls, and pastes them where
// the marker below says. That is deliberate, and it is the same rule the pxq2/pxq3 oracle
// follows: an oracle that is a hand transcription of the reference proves only that two
// people read the same header the same way, and it silently rots the first time the engine
// changes. Extracting the source text means this test fails loudly if the engine's decode
// ever moves, which is exactly what a parity gate is for.
//
// Everything the extracted function needs -- the geometry macros, the PX16 book, the SUB8
// table and the fp16 conversion -- is provided above the marker, under the names the engine
// uses.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-pxq6-tables.h"

// The frozen tables, under the names the extracted function uses. pxa_tab_sub16 is present
// because the extracted body selects between the two at run time on its `hq` argument; only
// the SUB8 path is exercised by this binary.
static float pxa_tab_px16_book[16] = PXQ6_BOOK_INIT;
static float pxa_tab_sub16[16]     = PXQ6_SUB16_INIT;
static float pxa_tab_sub8[16]      = PXQ6_SUB8_INIT;

// IEEE half -> float. The engine uses ggml's table-driven macro; a bit-exact scalar
// conversion is equivalent for every input (including subnormals, inf and nan) and keeps this
// file free of a ggml dependency.
static float pxa_half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) << 31;
    const uint32_t e = (h >> 10) & 0x1f;
    const uint32_t m = h & 0x3ff;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) { bits = s; }
        else {
            int sh = 0;
            uint32_t mm = m;
            while (!(mm & 0x400)) { mm <<= 1; ++sh; }
            mm &= 0x3ff;
            bits = s | ((uint32_t)(127 - 15 - sh + 1) << 23) | (mm << 13);
        }
    } else if (e == 0x1f) {
        bits = s | 0x7f800000u | (m << 13);
    } else {
        bits = s | ((e + 127 - 15) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}
#define GGML_COMPUTE_FP16_TO_FP32(x) pxa_half_to_float((uint16_t)(x))

/* @@PXA_ENGINE_DECODE_FUNCTIONS@@ */

// ------------------------------------------------------------------------------------------
// CLI: oracle_pxq4hq <N> <K> <in.bin> <out.f32>
// Reads the raw panel blob for one [N, K] PXQ4HQ tensor and writes N*K float32 in row-major
// order. There is no tier argument: this binary decodes the HQ tier and nothing else, so a
// mis-invocation cannot silently produce a core-tier answer.
// ------------------------------------------------------------------------------------------
int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <N> <K> <in.bin> <out.f32>\n", argv[0]);
        return 2;
    }
    const long N = atol(argv[1]), K = atol(argv[2]);
    if (N <= 0 || K <= 0 || N % 64 || K % 32) {
        fprintf(stderr, "oracle: bad geometry (N must be %%64, K must be %%32)\n");
        return 2;
    }
    const long need = (N / 64) * (PXQ6_HDR_BYTES + (K / 32) * PXQ6HQ_SLAB_BYTES);

    FILE * f = fopen(argv[3], "rb");
    if (!f) { perror("open in"); return 3; }
    uint8_t * blob = (uint8_t *)malloc((size_t)need);
    if (!blob) { fprintf(stderr, "oracle: out of memory (%ld B)\n", need); return 3; }
    if (fread(blob, 1, (size_t)need, f) != (size_t)need) {
        fprintf(stderr, "oracle: short read, wanted %ld B\n", need);
        return 3;
    }
    fclose(f);

    float * row = (float *)malloc(sizeof(float) * (size_t)K);
    FILE * o = fopen(argv[4], "wb");
    if (!o) { perror("open out"); return 3; }
    for (long r = 0; r < N; ++r) {
        pxa_deq_row_pxq6(blob, r, K, row, true);        /* hq = true: the bs8 arm */
        if (fwrite(row, sizeof(float), (size_t)K, o) != (size_t)K) {
            fprintf(stderr, "oracle: short write\n");
            return 3;
        }
    }
    fclose(o);
    free(row);
    free(blob);
    return 0;
}
