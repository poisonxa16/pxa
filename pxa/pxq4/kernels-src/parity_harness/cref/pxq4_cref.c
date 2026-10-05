/* pxq4_cref.c -- expose the SHIPPING engine's CPU dequant as a CLI, so the parity
 * harness can compare against the real thing instead of a transcription of it.
 *
 * vendor/ is a verbatim READ-ONLY copy of the files needed to compile
 * ggml/src/pxq-cpu.c out of /path/to/engine-repo (production tree, never modified).
 * pxa_pxq_dequant_2d() is the parity-locked contract (pxq-cpu.h:16-18): the CUDA GEMMs
 * are explicitly allowed to differ (fp16 MMA snap), the DEQUANT is not.
 *
 *   usage: pxq4_cref <rows> <K> <in.bin> <out.f32>
 *          rows % 64 == 0, K % 32 == 0
 *          in.bin  = exactly (rows/64)*(128 + (K/32)*1088) bytes of PXQ4 panel data
 *          out.f32 = rows*K little-endian float32, row-major
 *
 * The tool deliberately does NOT clear the environment: pxa_pxq_ensure_tables()
 * (pxq-cpu.c:75-106) lets PXA_PXQ6_BOOK / PXA_PXQ6_SUB -- and, note, PXA_PXQ2_SUB and
 * PXA_PXQ3_SUB, which overwrite the SAME sub16 table -- replace the frozen tables. If
 * one of those is set the comparison is meaningless, so we report them and refuse.
 */
#include "pxq-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *kTableEnv[] = {
    "PXA_PXQ6_BOOK", "PXA_PXQ6_SUB", "PXA_PXQ2_SUB", "PXA_PXQ3_SUB", NULL
};

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <rows> <K> <in.bin> <out.f32>\n", argv[0]);
        return 2;
    }
    for (const char **e = kTableEnv; *e; ++e) {
        if (getenv(*e)) {
            fprintf(stderr, "pxq4_cref: refusing to run with %s set -- it replaces the "
                            "frozen tables and would invalidate the parity comparison\n", *e);
            return 3;
        }
    }

    long long rows = atoll(argv[1]);
    long long K    = atoll(argv[2]);
    if (rows <= 0 || K <= 0 || rows % 64 || K % 32) {
        fprintf(stderr, "pxq4_cref: bad geometry rows=%lld K=%lld (need rows%%64==0, K%%32==0)\n",
                rows, K);
        return 2;
    }
    if (!pxa_pxq_is_cpu_supported(GGML_TYPE_PXQ4)) {
        fprintf(stderr, "pxq4_cref: this build does not support GGML_TYPE_PXQ4 on CPU\n");
        return 4;
    }

    size_t panel_bytes = (size_t)128 + (size_t)(K / 32) * 1088;
    size_t nbytes      = (size_t)(rows / 64) * panel_bytes;
    size_t nfloats     = (size_t)rows * (size_t)K;

    unsigned char *blob = (unsigned char *)malloc(nbytes);
    float *out = (float *)malloc(nfloats * sizeof(float));
    if (!blob || !out) { fprintf(stderr, "pxq4_cref: OOM\n"); return 5; }

    FILE *f = fopen(argv[3], "rb");
    if (!f) { perror("open in"); return 5; }
    if (fread(blob, 1, nbytes, f) != nbytes) {
        fprintf(stderr, "pxq4_cref: %s is shorter than %zu B\n", argv[3], nbytes);
        return 5;
    }
    fclose(f);

    pxa_pxq_dequant_2d(GGML_TYPE_PXQ4, blob, out, rows, K);

    f = fopen(argv[4], "wb");
    if (!f) { perror("open out"); return 5; }
    if (fwrite(out, sizeof(float), nfloats, f) != nfloats) {
        fprintf(stderr, "pxq4_cref: short write\n"); return 5;
    }
    fclose(f);
    fprintf(stderr, "pxq4_cref: ok rows=%lld K=%lld in=%zu B out=%zu B\n",
            rows, K, nbytes, nfloats * sizeof(float));
    return 0;
}
