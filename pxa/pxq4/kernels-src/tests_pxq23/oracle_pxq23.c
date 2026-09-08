// oracle_pxq23.c -- CPU decode oracle for PXQ2/PXQ3, used to gate tiers.dequant().
//
// The two decode functions are NOT written here: they are EXTRACTED VERBATIM at build time
// from the engine's own ggml/src/pxq-cpu.c by build_oracle.sh, which cuts out
// pxa_deq_row_pxq2 and pxa_deq_row_pxq3 between their signature line and the closing brace at
// column 0 and pastes them where the marker below says. That is deliberate: an oracle that is
// a hand transcription of the reference proves only that two people read the same header the
// same way, and it silently rots the first time the engine changes. Extracting the source
// text means this test fails loudly if the engine's decode ever moves, which is exactly what
// a parity gate is for.
//
// Everything the extracted functions need -- the geometry macros, the tables, and the fp16
// conversion -- is provided above the marker.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-pxq2-tables.h"
#include "ggml-pxq3-tables.h"

// The frozen v1 books and the SHARED SUB16 LUT, under the names the extracted functions use.
static float pxa_tab_lm4[4]    = PXQ2_BOOK_INIT;
static float pxa_tab_lm8[8]    = PXQ3_BOOK_INIT;
static float pxa_tab_sub16[16] = {
    0x1.b7c0000000000p-3f, 0x1.36c0000000000p-2f, 0x1.72c0000000000p-2f, 0x1.a2c0000000000p-2f,
    0x1.ccc0000000000p-2f, 0x1.f300000000000p-2f, 0x1.0bc0000000000p-1f, 0x1.1e00000000000p-1f,
    0x1.3040000000000p-1f, 0x1.4380000000000p-1f, 0x1.5800000000000p-1f, 0x1.6ec0000000000p-1f,
    0x1.8880000000000p-1f, 0x1.a640000000000p-1f, 0x1.cac0000000000p-1f, 0x1.f9c0000000000p-1f };

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
// CLI: oracle_pxq23 <tier 2|3> <N> <K> <in.bin> <out.f32>
// Reads the raw panel blob for one [N, K] tensor and writes N*K float32 in row-major order.
// ------------------------------------------------------------------------------------------
int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <2|3> <N> <K> <in.bin> <out.f32>\n", argv[0]);
        return 2;
    }
    const int tier = atoi(argv[1]);
    const long N = atol(argv[2]), K = atol(argv[3]);
    if ((tier != 2 && tier != 3) || N <= 0 || K <= 0 || N % 64 || K % 32) {
        fprintf(stderr, "oracle: bad tier/geometry (N must be %%64, K must be %%32)\n");
        return 2;
    }
    const long slab = (tier == 2) ? PXQ2_SLAB_BYTES : PXQ3_SLAB_BYTES;
    const long need = (N / 64) * (128 + (K / 32) * slab);

    FILE * f = fopen(argv[4], "rb");
    if (!f) { perror("open in"); return 3; }
    uint8_t * blob = (uint8_t *)malloc((size_t)need);
    if (!blob) { fprintf(stderr, "oracle: out of memory (%ld B)\n", need); return 3; }
    if (fread(blob, 1, (size_t)need, f) != (size_t)need) {
        fprintf(stderr, "oracle: short read, wanted %ld B\n", need);
        return 3;
    }
    fclose(f);

    float * row = (float *)malloc(sizeof(float) * (size_t)K);
    FILE * o = fopen(argv[5], "wb");
    if (!o) { perror("open out"); return 3; }
    for (long r = 0; r < N; ++r) {
        if (tier == 2) pxa_deq_row_pxq2(blob, r, K, row);
        else           pxa_deq_row_pxq3(blob, r, K, row);
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
