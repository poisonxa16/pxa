// PXA_FA_TILE_V2 on the device: the re-scheduled kernel must be BITWISE equal to the shipping one.
//
// tests/test-fa-tile-v2.cpp proves the schedule change on the CPU, term for term. This is the same
// claim on silicon, and it is deliberately a DUMP-AND-COMPARE rather than a tolerance check: the
// v2 kernel performs the shipping kernel's arithmetic in the shipping kernel's order and only
// moves the operands differently, so anything other than byte equality is a defect, not a
// numerics difference.
//
// The lever is resolved once per process (a function-local static, like every other lever in
// pxa/pxa-enhance.cuh), so one process cannot run both schedules. This binary therefore runs the
// whole case list once and writes every output tensor, raw, to the file named on the command line;
// tests/test-fa-tile-v2-dev.sh runs it twice -- PXA_FA_TILE_V2=0 then =1 -- and compares the two
// dumps byte for byte. The driver also requires the kernel's own "ENGAGED" line in the second run,
// so a lever that silently declined every shape cannot pass as "equal".
//
// SCOPE. The tile-f16 kernel is reached only where fattn.cu sends it: a card with fast fp16 and no
// fp16 MMA (sm_60), GGML_PREC_DEFAULT, and a batch above the vec-kernel cutoff. On any other card
// this harness still runs, still passes, and proves nothing -- which is exactly why the driver
// checks for the ENGAGED line rather than trusting a green exit code.
//
// The cases cover both head sizes the seats use (128 and 256) plus 64, the GQA ratios of the two
// seat models (1, 4, 6, 8), every (cols_per_block, parallel_blocks) pair the dispatcher can pick,
// a masked and an unmasked run, ALiBi, softcap, and a 150,016-cell KV range -- the seat's own
// context depth, where the accumulation is twenty thousand terms deep and a schedule change has
// the most room to be wrong.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct fa_case {
    const char * name;
    int64_t D;        // head size
    int64_t nh;       // query heads
    int64_t nh_kv;    // key/value heads
    int64_t n_kv;     // KV cells (must be a multiple of FATTN_KQ_STRIDE == 256)
    int64_t nb;       // query batch
    bool    mask;
    float   max_bias; // ALiBi
    float   softcap;
};

static const fa_case cases[] = {
    // name                       D   nh  nhkv    n_kv    nb  mask  bias  softcap
    { "d128 gqa1 kv512 b16",     128,  8,    8,    512,   16, true,  0.0f,  0.0f }, // cols16 pb4
    { "d128 gqa1 kv512 b32",     128,  8,    8,    512,   32, true,  0.0f,  0.0f }, // cols32 pb4
    { "d128 gqa6 kv2048 b64",    128, 24,    4,   2048,   64, true,  0.0f,  0.0f }, // cols32 pb1
    { "d128 gqa8 kv4096 b512",   128, 32,    4,   4096,  512, true,  0.0f,  0.0f },
    { "d128 gqa1 kv1024 nomask", 128,  8,    8,   1024,   32, false, 0.0f,  0.0f },
    { "d128 gqa1 kv1024 alibi",  128,  8,    8,   1024,   32, true,  8.0f,  0.0f },
    { "d128 gqa1 kv1024 softcap",128,  8,    8,   1024,   32, true,  0.0f, 30.0f },
    { "d64  gqa1 kv1024 b32",     64,  8,    8,   1024,   32, true,  0.0f,  0.0f },
    { "d256 gqa1 kv512 b16",     256, 16,   16,    512,   16, true,  0.0f,  0.0f }, // cols16 pb4
    { "d256 gqa6 kv2048 b32",    256, 24,    4,   2048,   32, true,  0.0f,  0.0f }, // cols16 pb1
    { "d256 gqa8 kv8192 b128",   256,  8,    1,   8192,  128, true,  0.0f,  0.0f },
    { "d128 gqa1 kv150016 b32",  128,  4,    4, 150016,   32, true,  0.0f,  0.0f }, // seat depth
    { "d256 gqa1 kv150016 b16",  256,  4,    4, 150016,   16, true,  0.0f,  0.0f }, // seat depth
};

struct lcg {
    uint64_t s;
    explicit lcg(uint64_t seed) : s(seed) {}
    uint32_t next() { s = s*6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t) (s >> 33); }
    float unit() { return (float) next() / (float) 0x80000000u - 1.0f; }
};

static void fill_f32(ggml_tensor * t, lcg & r, float amp) {
    std::vector<float> v((size_t) ggml_nelements(t));
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = amp*r.unit();
    }
    ggml_backend_tensor_set(t, v.data(), 0, v.size()*sizeof(float));
}

static void fill_f16(ggml_tensor * t, lcg & r, float amp) {
    const size_t n = (size_t) ggml_nelements(t);
    std::vector<float>       f(n);
    std::vector<ggml_fp16_t> h(n);
    for (size_t i = 0; i < n; ++i) {
        f[i] = amp*r.unit();
    }
    ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t) n);
    ggml_backend_tensor_set(t, h.data(), 0, n*sizeof(ggml_fp16_t));
}

// A causal mask over the last `nb` positions of an `n_kv`-cell range, with the padding rows fully
// masked -- the shape a real prefill ubatch presents, and the one that makes whole tiles -inf so
// the mask-skip path is exercised.
static void fill_mask(ggml_tensor * t, int64_t n_kv, int64_t nb) {
    const int64_t rows = t->ne[1];
    std::vector<ggml_fp16_t> h((size_t) n_kv*rows);
    std::vector<float>       f((size_t) n_kv);
    for (int64_t j = 0; j < rows; ++j) {
        const int64_t pos = n_kv - nb + j;
        for (int64_t k = 0; k < n_kv; ++k) {
            f[(size_t) k] = (j < nb && k <= pos) ? 0.0f : -INFINITY;
        }
        ggml_fp32_to_fp16_row(f.data(), h.data() + (size_t) j*n_kv, n_kv);
    }
    ggml_backend_tensor_set(t, h.data(), 0, h.size()*sizeof(ggml_fp16_t));
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <output-dump>\n", argv[0]);
        return 2;
    }

    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr);
    if (!cuda) {
        printf("no CUDA device -- cannot run\n");
        return 2;
    }

    FILE * out = fopen(argv[1], "wb");
    if (!out) {
        fprintf(stderr, "cannot open %s for writing\n", argv[1]);
        return 2;
    }

    const char * lever = getenv("PXA_FA_TILE_V2");
    printf("test-fa-tile-v2-dev: PXA_FA_TILE_V2=%s, dumping to %s\n", lever ? lever : "(unset)", argv[1]);

    int rc = 0;
    for (const fa_case & c : cases) {
        ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.D, c.nb,   c.nh,    1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.D, c.n_kv, c.nh_kv, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.D, c.n_kv, c.nh_kv, 1);
        ggml_tensor * m = c.mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, GGML_PAD(c.nb, GGML_KQ_MASK_PAD), 1, 1)
                                 : nullptr;
        ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf((float) c.D), c.max_bias, c.softcap);

        if (!ggml_backend_supports_op(cuda, o)) {
            printf("  %-28s SKIPPED (backend declines the shape)\n", c.name);
            ggml_free(ctx);
            continue;
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cuda);
        if (!buf) {
            printf("  %-28s SKIPPED (allocation failed)\n", c.name);
            ggml_free(ctx);
            continue;
        }

        lcg r(0x5eed0000ull + (uint64_t) (c.D*1000 + c.nh*10 + c.nb));
        fill_f32(q, r, 1.0f);
        fill_f16(k, r, 1.0f);
        fill_f16(v, r, 1.0f);
        if (m) {
            fill_mask(m, c.n_kv, c.nb);
        }

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);
        if (ggml_backend_graph_compute(cuda, gf) != GGML_STATUS_SUCCESS) {
            printf("  %-28s FAIL (graph compute)\n", c.name);
            rc = 1;
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            continue;
        }

        std::vector<float> res((size_t) ggml_nelements(o));
        ggml_backend_tensor_get(o, res.data(), 0, res.size()*sizeof(float));

        // A degenerate output would make byte equality meaningless, so say what came back.
        double sum = 0.0;
        int    bad = 0;
        for (float x : res) {
            if (!std::isfinite(x)) bad++;
            sum += (double) x;
        }
        if (bad) {
            printf("  %-28s FAIL (%d non-finite outputs of %zu)\n", c.name, bad, res.size());
            rc = 1;
        } else {
            printf("  %-28s %8zu floats, mean %+.9e\n", c.name, res.size(), sum/(double) res.size());
        }
        fwrite(res.data(), sizeof(float), res.size(), out);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    fclose(out);
    ggml_backend_free(cuda);
    printf("%s\n", rc ? "FAIL" : "dump written");
    return rc;
}
