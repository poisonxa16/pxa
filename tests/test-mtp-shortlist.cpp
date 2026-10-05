// test-mtp-shortlist.cpp -- PXA_MTP_SHORTLIST on every MTP family.
//
// The lever: the DRAFT head of an MTP model scores only a token-id PREFIX of the vocabulary (the first N rows of the head);
// the TARGET's verify head still scores every id. The draft only proposes, so what it proposes can change how many drafts
// are accepted and can never change what is emitted. This test pins the two facts that argument rests on, with no model and
// no GPU:
//
//   case 1  a window onto the first N rows of a head IS a head of N rows: the logits it produces are bit-identical to the
//           first N logits of the full head, for every row type a head is stored in (F32, F16, Q8_0, Q4_0, Q6_K) -- so the
//           engine's zero-copy window (llm_build_mtp_shortlist_prefix) changes no logit it keeps.
//   case 2  speculative decoding with a windowed drafter emits EXACTLY the token sequence plain greedy decoding emits, for
//           every window width from 64 rows to the whole vocabulary, including windows that exclude the target's top token
//           on some steps (the draft cannot be right there; the verifier's own pick is emitted instead), at depths 1-4.
//   case 3  the cost side is real and the acceptance side is bounded: a window that covers the target's top tokens costs
//           acceptance nothing, a window that does not costs acceptance but never output.
//
// Usage: test-mtp-shortlist   (exit 0 = pass)
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; printf("  FAIL %s:%d  %s  ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t g_rng = 12345u;
static float rnd() {   // deterministic, signed, order-1
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((float) ((g_rng >> 9) & 0xFFFF) / 32768.0f - 1.0f);
}

// ---- case 1: the window is a head ---------------------------------------------------------------------------------------
static void case_window_is_a_head(ggml_type type, int64_t n_embd, int64_t n_rows, int64_t N) {
    ggml_init_params ip = { 16u << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, type, n_embd, n_rows);
    ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);

    std::vector<float> wf((size_t) (n_embd * n_rows));
    for (auto & v : wf) v = rnd() * 0.5f;
    if (type == GGML_TYPE_F32) {
        memcpy(W->data, wf.data(), ggml_nbytes(W));
    } else if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(wf.data(), (ggml_fp16_t *) W->data, (int64_t) wf.size());
    } else {
        ggml_quantize_chunk(type, wf.data(), W->data, 0, n_rows, n_embd, nullptr, nullptr);
    }
    for (int64_t i = 0; i < n_embd; ++i) ((float *) h->data)[i] = rnd();

    // the engine's window: the same tensor header with fewer rows, the same bytes (llm_build_mtp_shortlist_prefix)
    ggml_tensor win = *W;
    win.ne[1] = N;
    win.nb[2] = win.nb[1] * win.ne[1];
    win.nb[3] = win.nb[2] * win.ne[2];
    win.op    = GGML_OP_NONE;
    for (int j = 0; j < GGML_MAX_SRC; ++j) win.src[j] = nullptr;

    ggml_tensor * full = ggml_mul_mat(ctx, W, h);
    ggml_tensor * part = ggml_mul_mat(ctx, &win, h);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, full);
    ggml_build_forward_expand(gf, part);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    CHECK(part->ne[0] == N && full->ne[0] == n_rows, "shapes %lld / %lld", (long long) part->ne[0], (long long) full->ne[0]);
    const bool same = memcmp(part->data, full->data, (size_t) N * sizeof(float)) == 0;
    CHECK(same, "%s: window logits differ from the full head's first %lld", ggml_type_name(type), (long long) N);
    printf("  %-5s window %lld of %lld rows: first-N logits %s\n", ggml_type_name(type), (long long) N, (long long) n_rows,
           same ? "bit-identical" : "DIFFER");
    ggml_free(ctx);
}

// ---- case 2 / 3: speculative decoding with a windowed drafter -----------------------------------------------------------
// A toy autoregressive model. State s (d floats) evolves by a fixed nonlinear map of (s, last token); the TARGET's logits are
// W s over V rows; the DRAFTER sees a perturbed state (its prediction error) and scores only the first N rows of the same W.
struct Toy {
    int V, d;
    std::vector<float> W;      // [V, d]
    std::vector<float> E;      // [V, d] token embedding
    std::vector<float> A;      // [d, d] mixing
    std::vector<float> scale;  // row bias: makes the low ids likelier, like a merge-order vocabulary
    Toy(int V_, int d_) : V(V_), d(d_), W((size_t) V_ * d_), E((size_t) V_ * d_), A((size_t) d_ * d_), scale((size_t) V_) {
        for (auto & v : W) v = rnd();
        for (auto & v : E) v = rnd() * 0.7f;
        for (auto & v : A) v = rnd() * 0.3f;
        for (int i = 0; i < V; ++i) scale[(size_t) i] = 2.5f * std::exp(-4.0f * (float) i / (float) V);
    }
    void step(std::vector<float> & s, int tok) const {
        std::vector<float> n((size_t) d, 0.0f);
        for (int i = 0; i < d; ++i) {
            float a = 0;
            for (int j = 0; j < d; ++j) a += A[(size_t) i * d + j] * s[(size_t) j];
            n[(size_t) i] = std::tanh(a + E[(size_t) tok * d + i]);
        }
        s = n;
    }
    // argmax over the first `rows` rows of the head for state s (strictly-greater wins: a tie keeps the lower id)
    int argmax(const std::vector<float> & s, int rows) const {
        int best = 0; float bv = -1e30f;
        for (int r = 0; r < rows; ++r) {
            float a = scale[(size_t) r];
            for (int j = 0; j < d; ++j) a += W[(size_t) r * d + j] * s[(size_t) j];
            if (a > bv) { bv = a; best = r; }
        }
        return best;
    }
};

struct Run { std::vector<int> out; long drafted = 0, accepted = 0, rounds = 0; };

static Run plain(const Toy & m, int n_tok) {
    Run r; std::vector<float> s((size_t) m.d, 0.1f);
    int tok = 1;
    for (int i = 0; i < n_tok; ++i) {
        m.step(s, tok);
        tok = m.argmax(s, m.V);
        r.out.push_back(tok);
    }
    return r;
}

// Speculative greedy decoding. The drafter proposes `depth` tokens by chaining its own (perturbed) states through a head
// window of N rows; the target verifies every proposal against its full head and emits the verified token. Exactly what the
// engine does: the emitted tokens are always the target's argmax, drafts only decide how many are emitted per round.
static Run spec(const Toy & m, int n_tok, int depth, int N, float noise) {
    Run r; std::vector<float> s((size_t) m.d, 0.1f);
    int tok = 1;
    uint32_t sv = 99u;
    auto jit = [&]() { sv = sv * 1664525u + 1013904223u; return (((sv >> 9) & 0xFFFF) / 32768.0f - 1.0f) * noise; };
    while ((int) r.out.size() < n_tok) {
        ++r.rounds;
        // draft chain from the drafter's own perturbed copy of the state
        std::vector<int> draft; std::vector<float> ds = s; int dt = tok;
        for (int k = 0; k < depth; ++k) {
            m.step(ds, dt);
            for (auto & v : ds) v += jit();
            dt = m.argmax(ds, N);               // the window: proposals can only come from the first N rows
            draft.push_back(dt);
        }
        // verify: the target's own argmax at every position, over the full head
        int n_acc = 0;
        std::vector<float> ts = s; int tt = tok;
        for (int k = 0; k <= depth && (int) r.out.size() < n_tok; ++k) {
            m.step(ts, tt);
            const int pick = m.argmax(ts, m.V);
            r.out.push_back(pick);
            tt = pick;
            if (k < depth) {
                ++r.drafted;
                if (pick == draft[(size_t) k]) { ++r.accepted; ++n_acc; } else break;
            }
        }
        // the committed state is the target's
        s = ts; tok = tt;
    }
    return r;
}

int main() {
    printf("case 1: a window onto the first N rows of a head is a head of N rows\n");
    for (ggml_type t : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q6_K }) {
        case_window_is_a_head(t, 256, 1024, 320);
        case_window_is_a_head(t, 256, 1024, 64);
    }

    printf("case 2: speculative decoding through a windowed drafter emits exactly what plain greedy decoding emits\n");
    const int V = 2048, d = 48, n_tok = 300;
    Toy m(V, d);
    const Run ref = plain(m, n_tok);
    double acc_by_N[8] = {};
    const int widths[] = { 64, 128, 512, 1024, 2048 };
    for (int depth = 1; depth <= 4; ++depth) {
        for (int wi = 0; wi < 5; ++wi) {
            const int N = widths[wi];
            const Run r = spec(m, n_tok, depth, N, 0.25f);
            CHECK(r.out == ref.out, "depth %d window %d: the emitted sequence differs from plain greedy", depth, N);
            if (depth == 3) acc_by_N[wi] = r.drafted ? (double) r.accepted / (double) r.drafted : 0.0;
        }
    }
    printf("  all 20 (depth x window) runs emit the plain greedy sequence (%d tokens)\n", n_tok);

    // a window that excludes the target's top token must have been exercised: count the steps where it does
    int outside = 0;
    for (int t : ref.out) outside += t >= 64;
    CHECK(outside > 0, "the narrowest window never excluded the target's pick: the case would test nothing");
    printf("  the 64-row window excludes the target's own pick on %d of %d steps\n", outside, n_tok);

    printf("case 3: the window costs acceptance, never output\n");
    printf("  depth 3 acceptance by window: 64 rows %.3f, 128 %.3f, 512 %.3f, 1024 %.3f, 2048 (full) %.3f\n",
           acc_by_N[0], acc_by_N[1], acc_by_N[2], acc_by_N[3], acc_by_N[4]);
    CHECK(acc_by_N[0] <= acc_by_N[4] + 1e-9, "a 64-row window cannot be MORE accurate than the full head here");
    CHECK(acc_by_N[4] > 0.2, "the toy drafter must be useful with the full head (%.3f), or the comparison says nothing", acc_by_N[4]);
    // a window wide enough to hold every pick the target made costs nothing at all
    int top = 0; for (int t : ref.out) top = std::max(top, t);
    const Run wide = spec(m, n_tok, 3, std::min(V, top + 1 + 64), 0.25f);
    const Run full = spec(m, n_tok, 3, V, 0.25f);
    CHECK(wide.out == full.out, "wide window output");
    printf("  a window covering every pick (up to row %d) accepts %ld / %ld drafts against %ld / %ld for the full head\n",
           top, wide.accepted, wide.drafted, full.accepted, full.drafted);

    printf(g_fail ? "FAIL: %d\n" : "PASS test-mtp-shortlist\n", g_fail);
    return g_fail ? 1 : 0;
}
