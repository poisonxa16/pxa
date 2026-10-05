// PXA_FA_F16_KV_CHUNK: the chunked flash-attention softmax merge must reconstruct the whole-range
// softmax.
//
// WHAT THIS PINS. The CUDA lever converts a quantized K/V cache to F16 in bounded chunks and runs
// the tile flash-attention kernel once per chunk, folding each chunk's UNNORMALISED numerator and
// its per-row (max, sum) pair into a running accumulator (pxa_flash_attn_chunk_fold in
// ggml/src/ggml-cuda/fattn-common.cuh). The claim the lever rests on is arithmetic, not hardware:
// online-softmax merging is exact in exact arithmetic and is a reassociation in floating point.
// This test proves the merge itself, on the CPU, deterministically, with no GPU and no model --
// so a failure here is a bug in the merge, never a GPU nondeterminism story.
//
// WHAT IT MIRRORS. mergeChunked() below is the fold kernel term for term, including the
// flush-to-zero mask on the rescale (the thing that makes an entirely-masked chunk contribute
// nothing and that disarms the NaN of subtracting two initial maxima). It reads the same
// float2(max, sum) meta convention the engine's existing online-softmax reduction
// (flash_attn_combine_results) uses, because there is one merge convention in this backend.
//
// THE THREE CHECKS, at context depths spanning 1, 2 and 5+ chunks:
//   1. ONE CHUNK IS BIT-IDENTICAL. With n_kv <= chunk the merge is a single adopt-and-divide, so
//      it must equal the unchunked online-softmax result BIT FOR BIT -- not close. This is the
//      CPU half of the lever's headline gate; the CUDA half is that the launcher does not even
//      enter the chunked path at or below the chunk size.
//   2. MANY CHUNKS AGREE WITH THE REFERENCE. Above the chunk size the accumulation order changes
//      by construction, so the requirement is a bound, not equality: the chunked result must
//      match a straightforward whole-range softmax computed in double precision to fp32
//      round-off.
//   3. CHUNK SIZE DOES NOT CHANGE THE ANSWER. Several chunk sizes over the same input must all
//      land inside the same bound, so the lever's value is not a tuning parameter of the output.
//
// Deterministic inputs from a fixed LCG. CPU only.

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define SOFTMAX_FTZ_THRESHOLD -20.0f // must match ggml/src/ggml-cuda/fattn-common.cuh

// --------------------------------------------------------------------------------------------
// deterministic inputs
// --------------------------------------------------------------------------------------------

struct lcg {
    uint64_t s;
    explicit lcg(uint64_t seed) : s(seed) {}
    uint32_t next() { s = s*6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t)(s >> 33); }
    float unit() { return (float) next() / (float) 0x80000000u - 1.0f; } // [-1, 1)
};

// --------------------------------------------------------------------------------------------
// the kernel's arithmetic, mirrored
// --------------------------------------------------------------------------------------------

// The FA kernel's per-chunk epilogue in the chunked mode: unnormalised numerator + (max, sum).
// Scores are accumulated in the order the kernel walks the key range: ascending key index.
struct chunk_partial {
    std::vector<float> num; // [Dv]
    float              max;
    float              sum;
};

static chunk_partial fattnChunk(const std::vector<float> & q,   // [Dk]
                                const std::vector<float> & k,   // [n_kv][Dk] (this chunk only)
                                const std::vector<float> & v,   // [n_kv][Dv] (this chunk only)
                                const std::vector<float> & mask,// [n_kv] additive, -inf == masked
                                const int Dk, const int Dv, const int n_kv, const float scale) {
    chunk_partial p;
    p.num.assign(Dv, 0.0f);
    p.max = -FLT_MAX/2.0f; // the tile kernel's initial kqmax, deliberately not -inf
    p.sum = 0.0f;

    for (int j = 0; j < n_kv; ++j) {
        float s = 0.0f;
        for (int d = 0; d < Dk; ++d) {
            s += q[d]*k[(size_t) j*Dk + d];
        }
        s = s*scale + mask[j];

        const float m_new = s > p.max ? s : p.max;
        const float rescale = expf(p.max - m_new);
        const float w       = expf(s - m_new);

        p.sum = p.sum*rescale + w;
        for (int d = 0; d < Dv; ++d) {
            p.num[d] = p.num[d]*rescale + w*v[(size_t) j*Dv + d];
        }
        p.max = m_new;
    }

    return p;
}

// pxa_flash_attn_chunk_fold, term for term.
static void fold(std::vector<float> & acc_num, float & acc_max, float & acc_sum,
                 const chunk_partial & p, const bool first) {
    if (first) {
        acc_num = p.num;
        acc_max = p.max;
        acc_sum = p.sum;
        return;
    }

    const float m_new = acc_max > p.max ? acc_max : p.max;

    const float diff_a = acc_max - m_new;
    const float diff_c = p.max   - m_new;

    float scale_a = expf(diff_a);
    float scale_c = expf(diff_c);

    // the FTZ mask: a scale whose exponent is below the threshold (or NaN, from two initial
    // maxima cancelling) is forced to exactly zero rather than left to underflow
    uint32_t bits;
    memcpy(&bits, &scale_a, 4); bits &= 0xFFFFFFFFu*(diff_a > SOFTMAX_FTZ_THRESHOLD); memcpy(&scale_a, &bits, 4);
    memcpy(&bits, &scale_c, 4); bits &= 0xFFFFFFFFu*(diff_c > SOFTMAX_FTZ_THRESHOLD); memcpy(&scale_c, &bits, 4);

    for (size_t d = 0; d < acc_num.size(); ++d) {
        acc_num[d] = scale_a*acc_num[d] + scale_c*p.num[d];
    }
    acc_sum = scale_a*acc_sum + scale_c*p.sum;
    acc_max = m_new;
}

// The whole pipeline: split the key range into chunks of `chunk`, run the kernel epilogue per
// chunk, fold, normalise. chunk >= n_kv gives the single-pass (unchunked) result exactly.
static std::vector<float> attnChunked(const std::vector<float> & q, const std::vector<float> & k,
                                      const std::vector<float> & v, const std::vector<float> & mask,
                                      const int Dk, const int Dv, const int n_kv,
                                      const float scale, const int chunk) {
    std::vector<float> acc_num;
    float acc_max = 0.0f, acc_sum = 0.0f;
    bool first = true;

    for (int kv0 = 0; kv0 < n_kv; kv0 += chunk) {
        const int cur = chunk < n_kv - kv0 ? chunk : n_kv - kv0;

        const std::vector<float> ks(k.begin() + (size_t) kv0*Dk, k.begin() + (size_t)(kv0 + cur)*Dk);
        const std::vector<float> vs(v.begin() + (size_t) kv0*Dv, v.begin() + (size_t)(kv0 + cur)*Dv);
        const std::vector<float> ms(mask.begin() + kv0, mask.begin() + kv0 + cur);

        fold(acc_num, acc_max, acc_sum, fattnChunk(q, ks, vs, ms, Dk, Dv, cur, scale), first);
        first = false;
    }

    std::vector<float> out(Dv);
    for (int d = 0; d < Dv; ++d) {
        out[d] = acc_num[d]/acc_sum;
    }
    return out;
}

// The kernel's UNCHUNKED epilogue: one pass over the whole key range, divided in the kernel.
// This is what the lever must reproduce exactly when the key range fits in one chunk -- the fold's
// first-chunk branch has to be a plain adopt, with no rescale multiply of its own, or the two
// diverge in the last bit.
static std::vector<float> attnUnchunked(const std::vector<float> & q, const std::vector<float> & k,
                                        const std::vector<float> & v, const std::vector<float> & mask,
                                        const int Dk, const int Dv, const int n_kv, const float scale) {
    const chunk_partial p = fattnChunk(q, k, v, mask, Dk, Dv, n_kv, scale);

    std::vector<float> out(Dv);
    for (int d = 0; d < Dv; ++d) {
        out[d] = p.num[d]/p.sum;
    }
    return out;
}

// Straightforward whole-range softmax in double precision: the thing the lever must reproduce.
static std::vector<double> attnReference(const std::vector<float> & q, const std::vector<float> & k,
                                         const std::vector<float> & v, const std::vector<float> & mask,
                                         const int Dk, const int Dv, const int n_kv, const float scale) {
    std::vector<double> s(n_kv);
    double smax = -INFINITY;
    for (int j = 0; j < n_kv; ++j) {
        double acc = 0.0;
        for (int d = 0; d < Dk; ++d) {
            acc += (double) q[d]*(double) k[(size_t) j*Dk + d];
        }
        s[j] = acc*(double) scale + (double) mask[j];
        if (s[j] > smax) smax = s[j];
    }

    std::vector<double> out(Dv, 0.0);
    double den = 0.0;
    for (int j = 0; j < n_kv; ++j) {
        const double w = std::isinf(s[j]) && s[j] < 0 ? 0.0 : exp(s[j] - smax);
        den += w;
        for (int d = 0; d < Dv; ++d) {
            out[d] += w*(double) v[(size_t) j*Dv + d];
        }
    }
    for (int d = 0; d < Dv; ++d) {
        out[d] /= den;
    }
    return out;
}

// --------------------------------------------------------------------------------------------

static int n_fail = 0;

static void check(const bool ok, const char * what) {
    printf("%s %s\n", ok ? "  PASS" : "  FAIL", what);
    if (!ok) {
        n_fail++;
    }
}

static double maxAbsDiff(const std::vector<float> & a, const std::vector<double> & b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = fabs((double) a[i] - b[i]);
        if (e > d) d = e;
    }
    return d;
}

static bool bitIdentical(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

// One case: build inputs, then run the three checks.
static void run_case(const char * name, const int Dk, const int Dv, const int n_kv,
                     const int chunk, const int causal_upto, const uint64_t seed) {
    lcg rng(seed);

    std::vector<float> q(Dk), k((size_t) n_kv*Dk), v((size_t) n_kv*Dv), mask(n_kv, 0.0f);

    for (int d = 0; d < Dk; ++d) {
        q[d] = rng.unit();
    }
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = rng.unit();
    }
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = rng.unit();
    }
    // causal_upto < n_kv masks the tail, which is what makes whole chunks fully masked at depth --
    // the case the fold's flush-to-zero has to survive.
    for (int j = causal_upto; j < n_kv; ++j) {
        mask[j] = -INFINITY;
    }

    const float scale = 1.0f/sqrtf((float) Dk);

    const int n_chunks = (n_kv + chunk - 1)/chunk;
    printf("%s: Dk=%d Dv=%d n_kv=%d chunk=%d -> %d chunk(s), keys unmasked=%d\n",
           name, Dk, Dv, n_kv, chunk, n_chunks, causal_upto);

    const std::vector<float>  whole = attnUnchunked(q, k, v, mask, Dk, Dv, n_kv, scale);
    const std::vector<float>  split = attnChunked(q, k, v, mask, Dk, Dv, n_kv, scale, chunk);
    const std::vector<double> ref   = attnReference(q, k, v, mask, Dk, Dv, n_kv, scale);

    if (n_chunks == 1) {
        check(bitIdentical(whole, split), "one chunk is BIT-IDENTICAL to the unchunked result");
    }

    const double d_whole = maxAbsDiff(whole, ref);
    const double d_split = maxAbsDiff(split, ref);

    // fp32 online softmax over a few thousand keys: the unchunked path itself sits around 1e-7.
    // The bound is on the chunked path being no worse in kind, not on it being closer.
    const double tol = 1e-5;
    printf("        max|chunked - ref| = %.3e   max|unchunked - ref| = %.3e   (tol %.0e)\n",
           d_split, d_whole, tol);
    check(d_split < tol, "chunked result matches the double-precision reference");
    check(d_whole < tol, "unchunked result matches the double-precision reference (control)");

    // The answer must not be a function of the chunk size.
    for (int c = 64; c < n_kv; c *= 2) {
        const std::vector<float> alt = attnChunked(q, k, v, mask, Dk, Dv, n_kv, scale, c);
        const double d = maxAbsDiff(alt, ref);
        if (d >= tol) {
            printf("        chunk=%d: max|chunked - ref| = %.3e\n", c, d);
            check(false, "chunk size does not change the answer");
            return;
        }
    }
    check(true, "chunk size does not change the answer");
}

int main() {
    printf("test-fa-chunk-merge: online-softmax merge across bounded K/V conversion chunks\n\n");

    // 1 chunk -- the bit-identity case
    run_case("case 1 (single chunk)",          128, 128,  512,  1024,  512, 0x5eed0001ULL);
    // 2 chunks
    run_case("case 2 (two chunks)",            128, 128, 2048,  1024, 2048, 0x5eed0002ULL);
    // 5+ chunks
    run_case("case 3 (eight chunks)",          128, 128, 8192,  1024, 8192, 0x5eed0003ULL);
    // 5+ chunks with a fully-masked tail: the last three chunks contribute nothing at all
    run_case("case 4 (masked tail)",           128, 128, 8192,  1024, 4608, 0x5eed0004ULL);
    // only the first chunk carries any unmasked key: every later fold must be a no-op
    run_case("case 5 (only chunk 0 unmasked)", 128, 128, 8192,  1024,  700, 0x5eed0005ULL);
    // a ragged last chunk, and the other supported head sizes
    run_case("case 6 (ragged tail, D=64)",      64,  64, 5000,  1024, 5000, 0x5eed0006ULL);
    run_case("case 7 (D=256)",                 256, 256, 4096,   768, 4096, 0x5eed0007ULL);

    printf("\n%s (%d failure%s)\n", n_fail == 0 ? "ALL PASS" : "FAILED", n_fail, n_fail == 1 ? "" : "s");
    return n_fail == 0 ? 0 : 1;
}
