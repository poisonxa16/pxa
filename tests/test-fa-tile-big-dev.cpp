// PXA_FA_TILE_512 on the device: the 512/512 and 576/512 tile kernel against the same graph
// computed on the CPU backend.
//
// This is NOT a dump-and-compare like tests/test-fa-tile-v2-dev.cpp. That test could demand byte
// equality because the v2 schedule performs the shipping kernel's arithmetic in the shipping
// kernel's order. Here there is no shipping kernel to be equal to: before this lever a 512-wide
// FLASH_ATTN_EXT node is DECLINED by the CUDA backend on sm_60/sm_70, so the reference has to be
// built rather than borrowed.
//
// -------------------------------------------------------------------------------------------
// WHAT THE REFERENCE IS, AND THE TRAP THAT COST A NIGHT (2026-09-14).
//
// The first version of this test ran the same graph on the CPU backend with an F16 V and called
// that the reference. IT IS NOT ONE AT DEPTH. ggml's CPU flash-attention keeps its P.V running
// sum in an FP16 ACCUMULATOR whenever V is F16 (ggml.c: VKQ16, and ggml_vec_mad_f16 rounds the
// accumulator back to fp16 on EVERY key), while the softmax denominator S stays fp32. With a
// 512-wide V and terms of order 0.2, a term stops changing the accumulator once |VKQ16| exceeds
// about 819 -- i.e. once about 4096 keys have been attended. Past that the numerator stalls
// while the denominator keeps growing, so the REFERENCE's output decays toward zero while a
// correct kernel keeps returning the true answer. That produced a smooth, monotone, batch- and
// head-invariant "error" with a knee between 3072 and 4096 cells which was read as a kernel
// defect (bug fa-tile-512-depth-error) and is entirely an artefact of the reference.
//
// It was made as bad as it can be by the second defect, one line of this file: lcg::unit()
// divided a 31-BIT value by 2^31 and subtracted one, so every Q, K and V element in the whole
// suite lay in [-1,0) instead of [-1,1). Same-signed V turns that fp16 accumulator into a
// monotone ramp, which is the worst case that exists for it -- with signed operands it is a
// random walk and the wall is much further out.
//
// So: the operands are now signed, and the REFERENCE RUN USES AN F32 V carrying exactly the same
// fp16-rounded values, which routes ggml's CPU flash-attention to its fp32 VKQ32 accumulator.
// The CUDA run still gets an F16 V, as the kernel requires and as the engine's cache holds. K
// stays F16 on both sides and Q stays F32 on both sides (the CPU converts Q to fp16 for the dot
// exactly as the kernel does), so the ONLY difference between the two runs is the arithmetic
// under test.
//
// PXA_FA_TILE_512_REF_F16=1 puts the reference back on an F16 V. That is the standing
// demonstration that the accumulator matters: with it set, rms(d)/rms(v) sits about TWO ORDERS OF
// MAGNITUDE above the F32 reference's. It no longer draws the old catastrophic curve, and that is
// the point -- with SIGNED operands the fp16 accumulator is a random walk rather than a ramp, so
// it stays flat at roughly 1.6e-4 across the whole KV sweep instead of climbing to 5e-1. Both
// defects had to be present for the artefact to look like a kernel bug.
// -------------------------------------------------------------------------------------------
//
// WHAT EACH RUN PROVES.
//   * PXA_FA_TILE_512 unset: every case must print DECLINED. That is the positive control for the
//     lever -- it says the default really does leave these nodes to the scheduler, and it says the
//     support predicate is the thing being switched rather than some unrelated branch.
//   * PXA_FA_TILE_512=1 on a cc 6.0 or 7.0 card: every case must print ENGAGED and clear the error
//     bar. On any other card every case prints DECLINED and the run proves nothing about the
//     kernel -- which is why tests/test-fa-tile-big-dev.sh checks for ENGAGED rather than for a
//     zero exit code, and why the driver prints the device it ran on.
//
// THE BAR, AND IT IS TWO NUMBERS, NOT ONE. Normalised mean squared error < 5e-4 against the
// reference (the bar test-backend-ops applies to every flash-attention case) AND rms(d)/rms(v)
// < 5e-4, plus a hard failure on any non-finite output. The second bar is there because NMSE
// alone cannot be read across this table: with signed operands the reference output is the mean
// of n_kv random V rows, so rms(ref) falls as 1/sqrt(n_kv) and a CONSTANT absolute error reads as
// an NMSE rising with depth -- a flat 5e-4 NMSE bar silently tightens by sqrt(n_kv) from the
// 256-cell cases to the 20480-cell ones. rms(d)/rms(v) measures the error against the operand
// that actually sets the output's scale and does not move with the KV extent, so it is the column
// a depth defect cannot hide behind and the column a shrinking denominator cannot manufacture one
// in. The kernel's fp16 QK products and fp16 probability round-trip put it in the same numerical
// class as the shipped 64/128/256 tile kernels, so 5e-4 is the right number for both.
//
// READ THE BAR WITH ITS LIMIT IN MIND. NMSE is a RATIO, and on these random operands the
// denominator is not constant down the table: a near-flat softmax makes the reference output the
// mean of n_kv random V rows, whose magnitude falls as 1/sqrt(n_kv), so the same absolute error
// reads as a larger NMSE at a deeper case. The table spans 256 to 20480 cells, so every case also
// prints rms(ref), rms(d) and rms(d)/rms(v). The last of those is the scale they should be
// compared on, because the output is a convex combination of V rows and rms(v) does not move with
// the KV extent. A case that fails NMSE while rms(d)/rms(v) is flat has a shrinking signal, not a
// growing error; a case where BOTH rise has a real one.
//
// THE CASES cover both head-size pairs, the GQA ratio Gemma 4's global layers actually use
// (16 query heads over 1 KV head), every (ncols, parallel_blocks) pair the dispatcher can pick
// (nb <= 8 -> 8/4, nb <= 16 -> 16/4, else 16/1), a masked and an unmasked run, ALiBi, a KV extent
// deep enough that the online softmax has to rescale many times, a SHORT-KV block where the whole
// prompt sits inside the first partially-filled 64-wide tile, and a WINDOW SWEEP at one fixed
// extent where the number of keys attended is the only variable that moves.

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
    int64_t DK;       // K/Q head size
    int64_t DV;       // V head size
    int64_t nh;       // query heads
    int64_t nh_kv;    // key/value heads
    int64_t n_kv;     // KV cells (a multiple of FATTN_KQ_STRIDE == 256)
    int64_t nb;       // query batch
    bool    mask;
    float   max_bias; // ALiBi
    int64_t window;   // 0 = full causal; > 0 = only the last `window` keys are visible
    int64_t pos0;     // position of query row 0; < 0 means n_kv - nb (the batch sits at the end)
};

static const fa_case cases[] = {
    // name                          DK   DV   nh  nhkv    n_kv   nb   mask  bias  window  pos0
    { "g4 512 gqa16 kv512 b1",      512, 512, 16,    1,     512,   1, true,  0.0f,     0,   -1 }, // cols8  pb4
    { "g4 512 gqa16 kv512 b8",      512, 512, 16,    1,     512,   8, true,  0.0f,     0,   -1 }, // cols8  pb4
    { "g4 512 gqa16 kv1024 b16",    512, 512, 16,    1,    1024,  16, true,  0.0f,     0,   -1 }, // cols16 pb4
    { "g4 512 gqa16 kv2048 b32",    512, 512, 16,    1,    2048,  32, true,  0.0f,     0,   -1 }, // cols16 pb1
    { "g4 512 gqa16 kv8192 b512",   512, 512, 16,    1,    8192, 512, true,  0.0f,     0,   -1 },
    { "g4 512 gqa1  kv1024 b32",    512, 512,  4,    4,    1024,  32, true,  0.0f,     0,   -1 },
    { "g4 512 gqa16 kv1024 nomask", 512, 512, 16,    1,    1024,  32, false, 0.0f,     0,   -1 },
    { "g4 512 gqa16 kv1024 alibi",  512, 512, 16,    1,    1024,  32, true,  8.0f,     0,   -1 },
    { "g4 512 gqa16 kv20480 b16",   512, 512,  4,    1,   20480,  16, true,  0.0f,     0,   -1 }, // seat depth
    { "mla 576/512 gqa16 kv512 b8", 576, 512, 16,    1,     512,   8, true,  0.0f,     0,   -1 },
    { "mla 576/512 gqa16 kv2048b32",576, 512, 16,    1,    2048,  32, true,  0.0f,     0,   -1 },
    { "mla 576/512 gqa8 kv4096b64", 576, 512,  8,    1,    4096,  64, true,  0.0f,     0,   -1 },

    // THE KV SWEEP, ONE SHAPE. Every case above that fails changes the batch or the head count as
    // well as the KV extent, so "the error grows with KV" is not something that table can say.
    // These seven differ in n_kv and in NOTHING else, so the curve they draw is a curve in KV.
    { "sweep kv2048  b32",          512, 512, 16,    1,    2048,  32, true,  0.0f,     0,   -1 },
    { "sweep kv3072  b32",          512, 512, 16,    1,    3072,  32, true,  0.0f,     0,   -1 },
    { "sweep kv4096  b32",          512, 512, 16,    1,    4096,  32, true,  0.0f,     0,   -1 },
    { "sweep kv6144  b32",          512, 512, 16,    1,    6144,  32, true,  0.0f,     0,   -1 },
    { "sweep kv8192  b32",          512, 512, 16,    1,    8192,  32, true,  0.0f,     0,   -1 },
    { "sweep kv12288 b32",          512, 512, 16,    1,   12288,  32, true,  0.0f,     0,   -1 },
    { "sweep kv20480 b32",          512, 512, 16,    1,   20480,  32, true,  0.0f,     0,   -1 },

    // THE WINDOW SWEEP: ONE extent, 20480 cells, so the launch configuration and the number of
    // tiles walked never change; only the number of keys ACTUALLY ATTENDED moves. A defect in how
    // a fully masked tile folds into the running softmax shows up at every point of this sweep; a
    // defect in the accumulation of real terms grows along it. w64 and w1024 were the original
    // two-point discriminator and are kept as its ends.
    { "deep20480 window64",         512, 512, 16,    1,   20480,  32, true,  0.0f,    64,   -1 },
    { "deep20480 window1024",       512, 512, 16,    1,   20480,  32, true,  0.0f,  1024,   -1 },
    { "deep20480 window2048",       512, 512, 16,    1,   20480,  32, true,  0.0f,  2048,   -1 },
    { "deep20480 window3072",       512, 512, 16,    1,   20480,  32, true,  0.0f,  3072,   -1 },
    { "deep20480 window4096",       512, 512, 16,    1,   20480,  32, true,  0.0f,  4096,   -1 },
    { "deep20480 window6144",       512, 512, 16,    1,   20480,  32, true,  0.0f,  6144,   -1 },
    { "deep20480 window8192",       512, 512, 16,    1,   20480,  32, true,  0.0f,  8192,   -1 },

    // THE SHORT-KV EDGE. The engine's 29-token prompt is the one place where the fa arm's greedy
    // output differs from the incumbent's, and it is the SHALLOWEST prompt there is -- the
    // opposite end of the table from the depth cases. These put the whole prompt at position 0 so
    // the live keys sit inside the FIRST 64-wide tile and the rest of that tile is masked: the
    // partially-filled tile, its masked lanes in kqmax/kqsum, and the tail handling.
    { "shortkv 1tok  pos0",         512, 512, 16,    1,     256,   1, true,  0.0f,     0,    0 }, // cols8  pb4
    { "shortkv 8tok  pos0",         512, 512, 16,    1,     256,   8, true,  0.0f,     0,    0 }, // cols8  pb4
    { "shortkv 29tok pos0",         512, 512, 16,    1,     256,  29, true,  0.0f,     0,    0 }, // the p33 prompt
    { "shortkv 63tok pos0",         512, 512, 16,    1,     256,  63, true,  0.0f,     0,    0 }, // one short of a tile
    { "shortkv 64tok pos0",         512, 512, 16,    1,     256,  64, true,  0.0f,     0,    0 }, // exactly one tile
    { "shortkv 65tok pos0",         512, 512, 16,    1,     256,  65, true,  0.0f,     0,    0 }, // one key into tile 1
    { "mla576 shortkv 29tok pos0",  576, 512, 16,    1,     256,  29, true,  0.0f,     0,    0 },
};

struct lcg {
    uint64_t s;
    explicit lcg(uint64_t seed) : s(seed) {}
    uint32_t next() { s = s*6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t) (s >> 33); }
    // next() is a 31-BIT value, so the divisor that spans [0,2) is 2^30, not 2^31. Dividing by
    // 2^31 (as this did until 2026-09-14) puts every operand in [-1,0), which is not a
    // representative attention input and is the worst case for any same-signed accumulator.
    float unit() { return (float) next() / (float) 0x40000000u - 1.0f; }
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

// Fill V. The values are generated in fp32 and rounded to fp16 in BOTH cases; only the STORAGE
// type differs, so the CUDA run (F16 V, what the kernel and the engine's cache require) and the
// reference run (F32 V, which routes ggml's CPU flash-attention to its fp32 accumulator) see
// numerically identical operands. Consumes the same number of LCG draws either way.
// Returns the RMS of the values actually stored, so a caller can measure an error against the
// operand's own scale instead of against a constant that a later change of `amp` would invalidate.
static double fill_v(ggml_tensor * t, lcg & r, float amp) {
    const size_t n = (size_t) ggml_nelements(t);
    std::vector<float>       f(n);
    std::vector<ggml_fp16_t> h(n);
    for (size_t i = 0; i < n; ++i) {
        f[i] = amp*r.unit();
    }
    ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t) n);
    ggml_fp16_to_fp32_row(h.data(), f.data(), (int64_t) n); // f is now exactly what will be stored

    if (t->type == GGML_TYPE_F16) {
        ggml_backend_tensor_set(t, h.data(), 0, n*sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(t, f.data(), 0, n*sizeof(float));
    }

    double s2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        s2 += (double) f[i]*(double) f[i];
    }
    return n ? sqrt(s2/(double) n) : 0.0;
}

// A causal mask over `nb` query positions starting at `pos0`, with the padding rows fully masked
// -- the shape a real prefill ubatch presents. pos0 < 0 puts the batch at the END of the n_kv
// range (a deep prefill into a full cache); pos0 == 0 puts it at the START (a short prompt in a
// padded cache, where only the first cells are live).
static void fill_mask(ggml_tensor * t, int64_t n_kv, int64_t nb, int64_t window, int64_t pos0) {
    const int64_t rows = t->ne[1];
    const int64_t base = pos0 >= 0 ? pos0 : n_kv - nb;
    std::vector<ggml_fp16_t> h((size_t) n_kv*rows);
    std::vector<float>       f((size_t) n_kv);
    for (int64_t j = 0; j < rows; ++j) {
        const int64_t pos = base + j;
        for (int64_t k = 0; k < n_kv; ++k) {
            const bool in_window = window <= 0 || k > pos - window;
            f[(size_t) k] = (j < nb && k <= pos && in_window) ? 0.0f : -INFINITY;
        }
        ggml_fp32_to_fp16_row(f.data(), h.data() + (size_t) j*n_kv, n_kv);
    }
    ggml_backend_tensor_set(t, h.data(), 0, h.size()*sizeof(ggml_fp16_t));
}

// Build and run one case on one backend. Returns false if the backend declines the op or cannot
// allocate; `declined` distinguishes the two. `v_type` selects V's storage (see fill_v).
static bool run_case(ggml_backend_t backend, const fa_case & c, ggml_type v_type,
                     std::vector<float> & out, bool & declined, double * rms_v_out = nullptr) {
    declined = false;

    ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.DK, c.nb,   c.nh,    1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.DK, c.n_kv, c.nh_kv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, v_type,        c.DV, c.n_kv, c.nh_kv, 1);
    ggml_tensor * m = c.mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, GGML_PAD(c.nb, GGML_KQ_MASK_PAD), 1, 1)
                             : nullptr;
    ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf((float) c.DK), c.max_bias, 0.0f);

    if (!ggml_backend_supports_op(backend, o)) {
        declined = true;
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }

    // Deterministic from the case alone, so the CUDA run and the reference run see identical bytes.
    lcg r(0x5eed0000ull + (uint64_t) (c.DK*100000 + c.DV*100 + c.nh*10 + c.nb));
    fill_f32(q, r, 1.0f);
    fill_f16(k, r, 1.0f);
    const double rms_v = fill_v(v, r, 1.0f);
    if (rms_v_out) {
        *rms_v_out = rms_v;
    }
    if (m) {
        fill_mask(m, c.n_kv, c.nb, c.window, c.pos0);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;

    if (ok) {
        out.resize((size_t) ggml_nelements(o));
        ggml_backend_tensor_get(o, out.data(), 0, out.size()*sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);

    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr);
    if (!cuda) {
        printf("no CUDA device -- cannot run\n");
        return 2;
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        printf("no CPU backend -- cannot run\n");
        return 2;
    }

    const char * lever = getenv("PXA_FA_TILE_512");
    const char * prec  = getenv("PXA_FA_TILE_512_FP32");
    const char * reff  = getenv("PXA_FA_TILE_512_REF_F16");

    // The reference V type. F32 keeps ggml's CPU flash-attention on its fp32 accumulator; F16 puts
    // it back on the fp16 one and is the positive control for the trap described at the top.
    const bool      ref_f16  = reff != nullptr && atoi(reff) != 0;
    const ggml_type ref_vtyp = ref_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

    printf("test-fa-tile-big-dev: PXA_FA_TILE_512=%s, PXA_FA_TILE_512_FP32=%s, device 0 = %s\n",
           lever ? lever : "(unset)", prec ? prec : "(unset)", ggml_backend_name(cuda));
    printf("reference: same graph on the CPU backend with a %s V (%s)\n",
           ref_f16 ? "F16" : "F32",
           ref_f16 ? "fp16 VKQ16 accumulator -- POSITIVE CONTROL, the deep cases are EXPECTED to fail"
                   : "fp32 VKQ32 accumulator");

    int rc       = 0;
    int engaged  = 0;
    int declined = 0;

    for (const fa_case & c : cases) {
        std::vector<float> got;
        std::vector<float> ref;
        bool cuda_declined = false;
        bool cpu_declined  = false;

        double rms_v = 0.0;
        if (!run_case(cuda, c, GGML_TYPE_F16, got, cuda_declined, &rms_v)) {
            if (cuda_declined) {
                printf("  %-30s DECLINED by the CUDA backend\n", c.name);
                declined++;
            } else {
                printf("  %-30s FAIL (CUDA compute or allocation)\n", c.name);
                rc = 1;
            }
            continue;
        }
        engaged++;

        if (!run_case(cpu, c, ref_vtyp, ref, cpu_declined)) {
            printf("  %-30s ENGAGED, but NO REFERENCE (%s) -- cannot be read as a pass\n",
                   c.name, cpu_declined ? "CPU backend declines the shape" : "CPU compute failed");
            rc = 1;
            continue;
        }

        if (got.size() != ref.size()) {
            printf("  %-30s FAIL (size %zu vs %zu)\n", c.name, got.size(), ref.size());
            rc = 1;
            continue;
        }

        // Normalised mean squared error, the bar test-backend-ops uses for flash-attention.
        double sum_d2 = 0.0;
        double sum_r2 = 0.0;
        double max_ad = 0.0;
        int    bad    = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (!std::isfinite(got[i])) {
                bad++;
                continue;
            }
            const double d = (double) got[i] - (double) ref[i];
            sum_d2 += d*d;
            sum_r2 += (double) ref[i]*(double) ref[i];
            if (fabs(d) > max_ad) {
                max_ad = fabs(d);
            }
        }
        const double nmse = sum_r2 > 0.0 ? sum_d2/sum_r2 : sum_d2;

        // NMSE ALONE CANNOT BE READ ACROSS THE CASE LIST. These operands are random, so with a
        // near-flat softmax the reference output is the MEAN of n_kv random V rows and its
        // magnitude falls as 1/sqrt(n_kv). NMSE divides by that shrinking signal, so a flat 5e-4
        // bar silently tightens by sqrt(n_kv) down the table, and a CONSTANT absolute error reads
        // as an error rising with depth. So the absolute scale is printed next to it: rms(ref)
        // says how much signal there was to divide by, and rms(d)/rms(v) measures the error
        // against the operand that actually sets the output's scale (the output is a convex
        // combination of V rows, so rms(v) is the honest denominator and it does not move with
        // n_kv). A rms(ref) that FALLS down the table is itself a warning that the reference, not
        // the kernel, is the thing decaying -- that is exactly what the fp16 VKQ16 trap looks like.
        const size_t n     = got.size();
        const double rms_d = sqrt(sum_d2/(double) n);
        const double rms_r = sqrt(sum_r2/(double) n);
        const double rel_v = rms_v > 0.0 ? rms_d/rms_v : 0.0;

        if (bad) {
            printf("  %-30s FAIL (%d non-finite of %zu)\n", c.name, bad, n);
            rc = 1;
        } else if (!(nmse < 5e-4) || !(rel_v < 5e-4)) {
            printf("  %-30s FAIL nmse %.3e  max|d| %.3e  rms(ref) %.3e  rms(d) %.3e  rms(d)/rms(v) %.3e  (%zu floats)\n",
                   c.name, nmse, max_ad, rms_r, rms_d, rel_v, n);
            rc = 1;
        } else {
            printf("  %-30s OK   nmse %.3e  max|d| %.3e  rms(ref) %.3e  rms(d) %.3e  rms(d)/rms(v) %.3e  (%zu floats)\n",
                   c.name, nmse, max_ad, rms_r, rms_d, rel_v, n);
        }
    }

    printf("engaged %d, declined %d, of %d cases\n", engaged, declined, (int) (sizeof(cases)/sizeof(cases[0])));

    ggml_backend_free(cpu);
    ggml_backend_free(cuda);

    printf("%s\n", rc ? "FAIL" : "PASS");
    return rc;
}
