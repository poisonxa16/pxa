//
// DSA sparse-attention correctness harness.
//
// Four questions, each with a control that can fail:
//
//  1. Does GGML_OP_MASK_TO_IDX agree between the CPU and CUDA forwards? Exact integer
//     comparison -- no tolerance to hide behind.
//
//  2. THE ADJUDICATOR. An independent attention reference computed here in double
//     precision from the same inputs. Both the ggml CPU dense path and CUDA DSA are
//     scored against it, so when they disagree we learn WHICH one is wrong instead of
//     guessing. Without this arm a CPU-vs-CUDA mismatch is unattributable.
//
//  3. Is CUDA DSA numerically equivalent to dense flash attention?
//     CONTROL: at head dim 128 CUDA also has a known-good dense kernel, so the
//     CPU-vs-CUDA-dense number there is the fp16 noise floor of this comparison.
//     At head 512 -- the case that matters -- no dense CUDA kernel exists on sm_70,
//     which is exactly why DSA is being ported.
//
//  4. Can the gate say NO? ggml_backend_supports_op() is probed with inputs that MUST
//     be rejected and one that must be accepted. A gate that only ever says yes is not
//     a gate.
//
// Deliberately tiny: a few hundred MB of VRAM, because the DGX is on loan and its
// owner's server is resident on all 8 cards.
//

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_fail = 0;

static void report(const char * name, bool ok, const char * detail = "") {
    printf("%-56s %s %s\n", name, ok ? "PASS" : "**FAIL**", detail);
    if (!ok) g_fail++;
}

struct attn_shapes {
    int64_t d, n_head, n_kv, n_tokens, n_visible, n_idx;
};

static int64_t mask_rows_of(const attn_shapes & s) { return GGML_PAD(s.n_tokens, GGML_KQ_MASK_PAD); }

// Inputs are generated ONCE and shared by every arm, so the reference and the two
// implementations are scored on literally the same numbers.
static void gen_inputs(const attn_shapes & s, uint32_t seed,
                       std::vector<float> & qh, std::vector<ggml_fp16_t> & kh,
                       std::vector<ggml_fp16_t> & mh) {
    const int64_t mr = mask_rows_of(s);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.25f);

    qh.resize((size_t) s.d*s.n_tokens*s.n_head);
    for (auto & x : qh) x = nd(rng);

    kh.resize((size_t) s.d*s.n_kv);
    for (auto & x : kh) x = ggml_fp32_to_fp16(nd(rng));

    // Scattered (not a prefix) visible set, so a kernel that quietly assumed contiguity
    // is caught.
    mh.assign((size_t) s.n_kv*mr, ggml_fp32_to_fp16(-INFINITY));
    for (int64_t t = 0; t < mr; ++t) {
        for (int64_t j = 0; j < s.n_visible; ++j) {
            mh[(size_t) t*s.n_kv + (j*7919 + t*31) % s.n_kv] = ggml_fp32_to_fp16(0.0f);
        }
    }
}

// Independent reference. Plain double-precision attention over the SAME fp16 K the
// kernels see. V == K, matching the graph. Result laid out like ggml_flash_attn_ext:
// [d, n_head, n_tokens].
static void ref_attn(const attn_shapes & s, const std::vector<float> & qh,
                     const std::vector<ggml_fp16_t> & kh, const std::vector<ggml_fp16_t> & mh,
                     std::vector<float> & out) {
    const double scale = 1.0/sqrt((double) s.d);
    out.assign((size_t) s.d*s.n_head*s.n_tokens, 0.0f);
    std::vector<double> p(s.n_kv);

    for (int64_t t = 0; t < s.n_tokens; ++t) {
        for (int64_t h = 0; h < s.n_head; ++h) {
            const float * q = &qh[(size_t) h*s.d*s.n_tokens + (size_t) t*s.d];
            double mx = -INFINITY;
            for (int64_t r = 0; r < s.n_kv; ++r) {
                const double m = ggml_fp16_to_fp32(mh[(size_t) t*s.n_kv + r]);
                if (m <= -INFINITY) { p[r] = -INFINITY; continue; }
                double dot = 0.0;
                const ggml_fp16_t * k = &kh[(size_t) r*s.d];
                for (int64_t i = 0; i < s.d; ++i) dot += (double) q[i]*ggml_fp16_to_fp32(k[i]);
                p[r] = scale*dot + m;
                if (p[r] > mx) mx = p[r];
            }
            double sum = 0.0;
            for (int64_t r = 0; r < s.n_kv; ++r) {
                p[r] = p[r] == -INFINITY ? 0.0 : exp(p[r] - mx);
                sum += p[r];
            }
            float * o = &out[((size_t) t*s.n_head + h)*s.d];
            for (int64_t r = 0; r < s.n_kv; ++r) {
                if (p[r] == 0.0) continue;
                const double w = p[r]/sum;
                const ggml_fp16_t * v = &kh[(size_t) r*s.d];
                for (int64_t i = 0; i < s.d; ++i) o[i] += (float) (w*ggml_fp16_to_fp32(v[i]));
            }
        }
    }
}

static bool run_attn(ggml_backend_t backend, const attn_shapes & s,
                     const std::vector<float> & qh, const std::vector<ggml_fp16_t> & kh,
                     const std::vector<ggml_fp16_t> & mh,
                     std::vector<float> & out, bool * op_supported = nullptr) {
    const int64_t mr = mask_rows_of(s);

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    // Q is [d, n_tokens, n_head, 1]: ggml_flash_attn_ext reads ne[1] as tokens and ne[2]
    // as heads (result ne = {v->ne[0], q->ne[2], q->ne[1], q->ne[3]}).
    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.d, s.n_tokens, s.n_head, 1);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, s.d, s.n_kv, 1, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, s.n_kv, mr, 1, 1);
    ggml_tensor * v    = k;   // V is literally K, as all three DeepSeek-V4 call sites do

    ggml_tensor * idx = s.n_idx > 0 ? ggml_mask_to_index(ctx, mask, s.n_idx) : nullptr;

    ggml_tensor * out_t = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/sqrtf((float) s.d), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out_t, GGML_PREC_F32);
    if (idx) out_t->src[5] = idx;

    if (op_supported) {
        *op_supported = ggml_backend_supports_op(backend, out_t);
        // Do not compute a node the backend just declined: on sm_70 a head-512
        // FLASH_ATTN_EXT falls through to the wmma kernel, which ABORTS the process
        // ("Unhandled head size 512"). That is the pre-port behaviour and it is what the
        // PXA_DSA_ATTN=0 control run is supposed to demonstrate -- gracefully.
        if (!*op_supported) { ggml_free(ctx); return true; }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }

    ggml_backend_tensor_set(q,    qh.data(), 0, qh.size()*sizeof(float));
    ggml_backend_tensor_set(k,    kh.data(), 0, kh.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(mask, mh.data(), 0, mh.size()*sizeof(ggml_fp16_t));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    if (idx) ggml_build_forward_expand(gf, idx);
    ggml_build_forward_expand(gf, out_t);

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }

    out.resize(ggml_nelements(out_t));
    ggml_backend_tensor_get(out_t, out.data(), 0, out.size()*sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static double rms_pct(const std::vector<float> & a, const std::vector<float> & b, double * max_abs) {
    double se = 0.0, sr = 0.0; *max_abs = 0.0;
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    if (n == 0) return INFINITY;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { *max_abs = INFINITY; return INFINITY; }
        const double d = (double) a[i] - (double) b[i];
        se += d*d; sr += (double) b[i]*(double) b[i];
        if (fabs(d) > *max_abs) *max_abs = fabs(d);
    }
    if (sr <= 0.0) return se > 0.0 ? INFINITY : 0.0;
    return 100.0*sqrt(se/n)/sqrt(sr/n);
}

enum probe_kind { PROBE_OK, PROBE_NO_IDX, PROBE_BAD_WIDTH, PROBE_ALIBI, PROBE_SOFTCAP };

static bool gate_probe(ggml_backend_t cuda, int64_t d, probe_kind kind) {
    const int64_t n_tokens = 4, n_head = 8, n_kv = 2048;
    const int64_t mr = GGML_PAD(n_tokens, GGML_KQ_MASK_PAD);

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, n_tokens, n_head, 1);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, n_kv, 1, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, mr, 1, 1);

    ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, k, mask, 1.0f/sqrtf((float) d),
                                          kind == PROBE_ALIBI ? 8.0f : 0.0f,
                                          kind == PROBE_SOFTCAP ? 30.0f : 0.0f);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    if (kind != PROBE_NO_IDX) {
        o->src[5] = ggml_mask_to_index(ctx, mask, kind == PROBE_BAD_WIDTH ? 300 : 256);
    }
    const bool sup = ggml_backend_supports_op(cuda, o);
    ggml_free(ctx);
    return sup;
}

int main() {
    ggml_backend_t cpu  = ggml_backend_cpu_init();
    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr);
    if (!cuda) { printf("no CUDA device -- cannot run\n"); return 2; }

    // DSA is opt-in (PXA_DSA_ATTN=1). The harness deliberately does NOT set it for
    // itself: that would make the default unobservable. Instead it reads the switch and
    // flips every expectation, so BOTH invocations are checks that can fail --
    //
    //   PXA_DSA_ATTN=1 ./test-dsa-attn   every DSA case must be accepted and correct
    //   ./test-dsa-attn                  every DSA case must be DECLINED (default-off)
    //
    // -- and a default that leaked back to on would fail the second one.
    const char * dsa_env = getenv("PXA_DSA_ATTN");
    const bool   dsa_on  = dsa_env && dsa_env[0] == '1';
    printf("PXA_DSA_ATTN=%s -> DSA %s\n\n", dsa_env ? dsa_env : "(unset)",
           dsa_on ? "ENABLED" : "DISABLED (running as the default-off control)");

    printf("=== 1. GGML_OP_MASK_TO_IDX: CPU vs CUDA (exact) ===\n");
    {
        const int64_t n_kv = 2048, rows = 16, w = 256;
        for (int rep = 0; rep < 2; ++rep) {
            const int64_t nv = rep == 0 ? 200 : 400;   // rep 1 exceeds the width on purpose
            std::vector<int32_t> a, b;
            for (int pass = 0; pass < 2; ++pass) {
                ggml_backend_t be = pass == 0 ? cpu : cuda;
                ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
                ggml_context * ctx = ggml_init(ip);
                ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, rows, 1, 1);
                ggml_tensor * o = ggml_mask_to_index(ctx, m, w);
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

                std::vector<ggml_fp16_t> mh((size_t) n_kv*rows, ggml_fp32_to_fp16(-INFINITY));
                for (int64_t t = 0; t < rows; ++t)
                    for (int64_t j = 0; j < nv; ++j)
                        mh[(size_t) t*n_kv + (j*7919 + t*31) % n_kv] = ggml_fp32_to_fp16(0.0f);
                ggml_backend_tensor_set(m, mh.data(), 0, mh.size()*sizeof(ggml_fp16_t));

                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, o);
                ggml_backend_graph_compute(be, gf);

                std::vector<int32_t> & dst = pass == 0 ? a : b;
                dst.resize(ggml_nelements(o));
                ggml_backend_tensor_get(o, dst.data(), 0, dst.size()*sizeof(int32_t));
                ggml_backend_buffer_free(buf); ggml_free(ctx);
            }
            size_t ndiff = 0;
            for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ndiff++;
            // Also assert ascending order on the CPU side, so "they agree" cannot mean
            // "they are both scrambled the same way".
            bool asc = true;
            for (int64_t t = 0; t < rows && asc; ++t)
                for (int64_t j = 1; j < w; ++j)
                    if (a[t*w+j] >= 0 && a[t*w+j-1] >= 0 && a[t*w+j] <= a[t*w+j-1]) { asc = false; break; }
            char d[160]; snprintf(d, sizeof d, "(%zu/%zu differ, cpu ascending=%s)", ndiff, a.size(), asc ? "yes" : "NO");
            report(rep == 0 ? "mask_to_idx CPU==CUDA, visible < width"
                            : "mask_to_idx CPU==CUDA, visible > width (clamped)",
                   ndiff == 0 && asc, d);
        }
    }

    printf("\n=== 2/3. attention numerics ===\n");
    printf("%-56s %s\n", "", "(ref = independent double-precision attention)");
    {
        // Bisect on head dim, head count and visible count. head 128 passes and head 512
        // does not, so walk the dimension and vary one thing at a time.
        // must_diverge: the index list is NARROWER than the visible set, so the sparse
        // path physically cannot see every row the mask admits. GGML_OP_MASK_TO_IDX
        // clamps and drops the overflow with no diagnostic, which is exactly the
        // silently-wrong-engine failure the graph builder's n_visible_max bound exists to
        // prevent. Nothing in the kernel can detect it -- the bound is an argument, not a
        // runtime check -- so this case exists to prove the argument is load-bearing: if
        // truncation were harmless the bound would not matter, and the fact that it is
        // NOT harmless is what must be demonstrated, not assumed.
        struct { const char * name; attn_shapes s; bool indep_ref; bool must_diverge; } cases[] = {
            //                                          d  nh   n_kv  nt  vis  n_idx
            { "head 128, no index list (CUDA dense)", { 128, 8, 2048,  4, 200,   0 }, true, false },
            { "head 128, DSA",                        { 128, 8, 2048,  4, 200, 256 }, true, false },
            { "head 256, DSA",                        { 256, 8, 2048,  4, 200, 256 }, true, false },
            { "head 384, DSA",                        { 384, 8, 2048,  4, 200, 256 }, true, false },
            { "head 512, DSA          (TARGET)",      { 512, 8, 2048,  4, 200, 256 }, true, false },
            { "head 512, DSA, 1 head",                { 512, 1, 2048,  4, 200, 256 }, true, false },
            { "head 512, DSA, 1 token",               { 512, 8, 2048,  1, 200, 256 }, true, false },
            { "head 512, DSA, 1 visible col",         { 512, 8, 2048,  4,   1, 256 }, true, false },
            { "head 512, DSA, idx width 512",         { 512, 8, 4096,  4, 200, 512 }, true, false },
            // 64 tokens > DSA_ATTN_MAX_ROWS(32), so this is the only case that exercises
            // the multi-step loop (nstep=2) and the `first` offsets into Q/idx/dst.
            // n_head trimmed to 2 purely to keep the O(n_kv*d*nt*nh) reference cheap.
            { "head 512, DSA, 64 tok (nstep=2)",      { 512, 2, 4096, 64, 200, 256 }, true, false },
            // 400 visible rows into a 256-wide list: 144 rows per query are dropped.
            { "head 512, idx TOO NARROW -> must diverge",{ 512, 2, 4096,  4, 400, 256 }, true, true  },
        };
        for (auto & c : cases) {
            std::vector<float> qh, ref, cpu_o, cuda_o; std::vector<ggml_fp16_t> kh, mh;
            gen_inputs(c.s, 1234, qh, kh, mh);

            bool sup = false;
            const bool ok_cpu  = run_attn(cpu,  c.s, qh, kh, mh, cpu_o);
            const bool ok_cuda = run_attn(cuda, c.s, qh, kh, mh, cuda_o, &sup);
            if (!ok_cpu || !ok_cuda) { report(c.name, false, "(compute failed)"); continue; }

            // With DSA off, every case that carries an index list must be DECLINED --
            // that is the whole content of the default-off control, and it fails if the
            // switch stops applying. The n_idx==0 case uses no DSA and must still run.
            if (!dsa_on && c.s.n_idx > 0) {
                report(c.name, !sup, sup ? "(ACCEPTED with DSA off -- switch leaked)"
                                         : "(declined, as the default-off control requires)");
                continue;
            }
            if (!sup)                { report(c.name, false, "(CUDA declined the node)"); continue; }

            double mx1 = 0.0, mx2 = 0.0, mx3 = 0.0;
            const double cuda_vs_cpu = rms_pct(cuda_o, cpu_o, &mx1);

            if (c.indep_ref) {
                ref_attn(c.s, qh, kh, mh, ref);
                const double cpu_vs_ref  = rms_pct(cpu_o,  ref, &mx2);
                const double cuda_vs_ref = rms_pct(cuda_o, ref, &mx3);
                char d[256];
                snprintf(d, sizeof d, "cuda-vs-ref %.4f%% | cpu-vs-ref %.4f%% | cuda-vs-cpu %.4f%%",
                         cuda_vs_ref, cpu_vs_ref, cuda_vs_cpu);
                // Scored against the INDEPENDENT reference, so a broken ggml CPU path
                // cannot make a correct CUDA path look wrong (or vice versa).
                if (c.must_diverge) {
                    // Inverted expectation: dropping 144 of 400 visible rows per query
                    // must show up. If this ever "passes" at <2% the truncation is
                    // invisible in the output, and the n_visible_max bound protecting the
                    // three graph call sites would be untestable rather than merely
                    // unasserted -- which is a reason to look at it, not to relax it.
                    report(c.name, cuda_vs_ref > 5.0, d);
                    continue;
                }
                report(c.name, cuda_vs_ref < 2.0, d);
                if (cuda_vs_ref >= 2.0) {
                    // Show the shape of the error, not just its size.
                    printf("   ref :");
                    for (int i = 0; i < 6; ++i) printf(" %9.5f", ref[i]);
                    printf("\n   cuda:");
                    for (int i = 0; i < 6; ++i) printf(" %9.5f", cuda_o[i]);
                    double sr = 0, sc = 0;
                    for (size_t i = 0; i < ref.size(); ++i) { sr += ref[i]*ref[i]; sc += cuda_o[i]*cuda_o[i]; }
                    printf("\n   |ref| rms %.5f   |cuda| rms %.5f   ratio %.2f\n",
                           sqrt(sr/ref.size()), sqrt(sc/cuda_o.size()), sqrt(sc/sr));
                }
                if (cpu_vs_ref >= 2.0) {
                    printf("   ^ NOTE: the ggml CPU dense path is itself %.3f%% off the independent"
                           " reference here (max|d| %.5f) -- it is not a trustworthy baseline at this shape\n",
                           cpu_vs_ref, mx2);
                }
            } else {
                char d[160];
                snprintf(d, sizeof d, "cuda-vs-cpu %.4f%%, max|d| %.5f", cuda_vs_cpu, mx1);
                report(c.name, cuda_vs_cpu < 2.0, d);
            }
        }
    }

    printf("\n=== 4. gate negative controls ===\n");
    // The positive probe follows the switch: accepted iff DSA is on. This is the
    // sharpest test of the default -- with the variable unset it must come back false.
    report(dsa_on ? "head 512 + valid index list -> SUPPORTED (positive)"
                  : "head 512 + valid index list -> rejected (DSA off)",
           gate_probe(cuda, 512, PROBE_OK) == dsa_on);
    report("head 512, no index list     -> rejected",             gate_probe(cuda, 512, PROBE_NO_IDX) == false);
    report("head 512, index width 300   -> rejected",             gate_probe(cuda, 512, PROBE_BAD_WIDTH) == false);
    report("head 512, ALiBi max_bias!=0 -> rejected",             gate_probe(cuda, 512, PROBE_ALIBI) == false);
    report("head 512, logit softcap!=0  -> rejected",             gate_probe(cuda, 512, PROBE_SOFTCAP) == false);

    ggml_backend_free(cuda);
    ggml_backend_free(cpu);
    printf("\n%s (%d failures)\n", g_fail == 0 ? "ALL PASS" : "FAILURES PRESENT", g_fail);
    return g_fail == 0 ? 0 : 1;
}
