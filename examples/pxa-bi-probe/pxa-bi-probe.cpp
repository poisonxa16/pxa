// pxa-bi-probe -- batch-invariance probe for speculative verification (2026-10-16).
//
// Question: does one forward of K tokens (one ubatch, every row an output -- the speculative verify shape) compute,
// for each of its tokens, exactly the bytes that K successive one-token forwards (plain decode) compute? If it does,
// a greedy speculative run is byte-identical to plain greedy decode; every op that does not is a reason it is not.
//
// Method, per chunk of K tokens: snapshot the whole context state S; run the K tokens one at a time (the reference
// arm; it extends the greedy continuation as it goes); snapshot S'; restore S; run the same K tokens as one batch at
// the same positions; restore S'. Both arms start every chunk from the same bytes, so each chunk is a clean test and a
// difference in one chunk cannot leak into the next.
//
// Output (stdout): one line per token (logits bitwise equal or not, max |dlogit|, both argmaxes, the reference's top-1
// margin), a per-column summary, and with BI_DUMP the first graph node whose token slice differs, per token, plus a
// tally of first-divergent nodes. Without BI_DUMP no eval callback is installed, so the graph runs exactly as in the
// server (fusions, CUDA graphs); with it the scheduler splits the graph at every observed node, which can disable a
// backend fusion that spans one -- a node diff that comes back clean while the logits differ points at such a fusion.
//
// Usage: llama-pxa-bi-probe -m model.gguf -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -c 4096 -b 2048 -ub 64 -t 8 -f prompt.txt
// Env:   BI_KS="2,3,4" verify widths (each from the same post-prompt state), BI_N tokens compared per width (64),
//        BI_DUMP unset = logits only | "all" = every non-view node | "a,b,c" = nodes whose name contains one of these,
//        BI_PROMPT_MAX cut the prompt to this many tokens (0 = whole), BI_VERBOSE=1 every divergent node per token,
//        BI_PARSE_SPECIAL=1 parse special tokens (chat-templated prompt files), BI_PROMPT_TAIL=n last n prompt tokens in
//        their own forward (the server's recurrent-model prompt split), BI_CKPT=<mode> the server's recurrent speculative
//        checkpoint around every verify forward + a rollback test (see main).
//        With BI_DUMP a "source nodes" list follows the tally: nodes whose slice differs while none of their captured
//        inputs' slices does -- the width-dependent ops themselves, all of them in one run (BI_SOURCES).
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct bi_node {
    std::string key, op, src0, src1;
    std::vector<int> srci;   // BI_SOURCES: per src, index of the captured node it reads (view chains resolved), -1 = not captured
    ggml_type   type;
    int64_t     ne[4];
    size_t      nb[4];
    std::vector<uint8_t> data;
};

struct bi_cb {
    bool on = false, all = false;
    std::vector<std::string> flt;
    std::map<std::string, int> occ;
    std::map<const ggml_tensor *, int> idx;   // BI_SOURCES: captured tensor -> index in *sink (this forward)
    std::map<std::string, int> idxn;          // BI_SOURCES: by name (a scheduler split copy "CUDA1#name#0" reads "name")
    std::vector<bi_node> * sink = nullptr;
};

static bool bi_view_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE;
}

static bool bi_eval(struct ggml_tensor * t, bool ask, void * ud) {
    bi_cb * cb = (bi_cb *) ud;
    if (ask) {
        if (!cb->on || !cb->sink || bi_view_op(t->op)) return false;
        if (cb->all) return true;
        for (const auto & f : cb->flt) {
            if (strstr(t->name, f.c_str())) return true;
        }
        return false;
    }
    if (!cb->sink) return true;
    if (cb->sink->empty()) { cb->idx.clear(); cb->idxn.clear(); }   // a new forward (every sink starts empty)
    bi_node n;
    const int k = cb->occ[t->name]++;
    n.key  = std::string(t->name) + "#" + std::to_string(k);
    n.op   = ggml_op_desc(t);
    n.src0 = t->src[0] ? t->src[0]->name : "";
    n.src1 = t->src[1] ? t->src[1]->name : "";
    n.type = t->type;
    for (int i = 0; i < 4; ++i) { n.ne[i] = t->ne[i]; n.nb[i] = t->nb[i]; }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * s = t->src[i];
        if (!s) continue;
        while (s->view_src) s = s->view_src;
        auto it = cb->idx.find(s);
        int si = it == cb->idx.end() ? -1 : it->second;
        if (si < 0) {
            std::string nm = s->name;
            const size_t a = nm.find('#'), b = nm.rfind('#');
            if (a != std::string::npos && b != std::string::npos && b > a) nm = nm.substr(a + 1, b - a - 1);
            auto jt = cb->idxn.find(nm);
            if (jt != cb->idxn.end()) si = jt->second;
        }
        n.srci.push_back(si);
    }
    cb->idx[t] = (int) cb->sink->size();
    cb->idxn[t->name] = (int) cb->sink->size();
    n.data.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, n.data.data(), 0, n.data.size());
    cb->sink->push_back(std::move(n));
    return true;
}

static int env_int(const char * k, int d) { const char * e = getenv(k); return e && *e ? atoi(e) : d; }

// the memory state of sequence 0 (KV cells + recurrent state), without the logits / embeddings buffers
static size_t bi_seq_state(llama_context * ctx, std::vector<uint8_t> & buf) {
    buf.resize(llama_state_seq_get_size(ctx, 0, 0));
    return llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0, 0);
}

static int argmax_of(const float * v, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i) if (v[i] > v[b]) b = i;
    return b;
}

static float top_margin(const float * v, int n) {
    float a = -INFINITY, b = -INFINITY;
    for (int i = 0; i < n; ++i) {
        if (v[i] > a) { b = a; a = v[i]; } else if (v[i] > b) { b = v[i]; }
    }
    return a - b;
}

// Compare token slice jA of node A (reference, one token) with slice jB of node B (batch). Returns -1 when the shapes
// cannot be aligned, else the number of differing elements (rows for block-quantized types); *md = max |d| (floats).
static long long bi_cmp(const bi_node & A, const bi_node & B, int K, int jB, double * md, bool * whole) {
    *md = 0.0; *whole = false;
    // the token dimension d: B.ne[d] == K * A.ne[d], every other dim equal. A.ne[d] == 1 is the plain [.., n_tok] layout;
    // A.ne[d] = m > 1 is a merged one ([head_dim, heads*n_tok] and the like, tokens outermost), where token j of the batch
    // owns the index range [j*m, (j+1)*m) of that dim (BI_SOURCES: those nodes used to be skipped)
    int d = -1;
    int64_t m = 1;
    for (int i = 0; i < 4 && K > 1; ++i) {
        if (B.ne[i] == K * A.ne[i]) {
            bool rest = true;
            for (int q = 0; q < 4; ++q) if (q != i && A.ne[q] != B.ne[q]) rest = false;
            if (rest) { d = i; m = A.ne[i]; if (A.ne[i] == 1) break; }
        }
    }
    bool same = true;
    for (int i = 0; i < 4; ++i) if (A.ne[i] != B.ne[i]) same = false;
    if (d < 0 && !same) return -1;
    if (d < 0) {                       // no token dimension: a state or a constant; only the last slice is comparable
        if (jB != K - 1) return -2;
        *whole = true;
    }
    if (A.type != B.type) return -1;
    const size_t ts = ggml_type_size(A.type);
    const int64_t bs = ggml_blck_size(A.type);
    long long nd = 0;
    const bool flt = A.type == GGML_TYPE_F32 || A.type == GGML_TYPE_F16;
    if (bs > 1 || !flt) {
        // byte rows: compare ne[0]/bs blocks per row; the token dimension must not be ne[0]
        if (d == 0) return -1;
        const size_t rb = (size_t) (A.ne[0] / bs) * ts;
        for (int64_t i3 = 0; i3 < A.ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < A.ne[2]; ++i2)
        for (int64_t i1 = 0; i1 < A.ne[1]; ++i1) {
            int64_t b1 = i1, b2 = i2, b3 = i3;
            if (d == 1) b1 = jB*m + i1; else if (d == 2) b2 = jB*m + i2; else if (d == 3) b3 = jB*m + i3;
            const size_t oa = i1*A.nb[1] + i2*A.nb[2] + i3*A.nb[3];
            const size_t ob = b1*B.nb[1] + b2*B.nb[2] + b3*B.nb[3];
            if (oa + rb > A.data.size() || ob + rb > B.data.size()) return -1;
            if (memcmp(&A.data[oa], &B.data[ob], rb)) ++nd;
        }
        return nd;
    }
    for (int64_t i3 = 0; i3 < A.ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < A.ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < A.ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < A.ne[0]; ++i0) {
        int64_t b0 = i0, b1 = i1, b2 = i2, b3 = i3;
        if (d == 0) b0 = jB*m + i0; else if (d == 1) b1 = jB*m + i1; else if (d == 2) b2 = jB*m + i2; else if (d == 3) b3 = jB*m + i3;
        const size_t oa = i0*A.nb[0] + i1*A.nb[1] + i2*A.nb[2] + i3*A.nb[3];
        const size_t ob = b0*B.nb[0] + b1*B.nb[1] + b2*B.nb[2] + b3*B.nb[3];
        if (oa + ts > A.data.size() || ob + ts > B.data.size()) return -1;
        if (memcmp(&A.data[oa], &B.data[ob], ts)) {
            ++nd;
            float a, b;
            if (A.type == GGML_TYPE_F32) { memcpy(&a, &A.data[oa], 4); memcpy(&b, &B.data[ob], 4); }
            else { a = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &A.data[oa]); b = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &B.data[ob]); }
            const double x = std::fabs((double) a - (double) b);
            if (x > *md || std::isnan(x)) *md = x;
        }
    }
    return nd;
}

// Snapshot / restore of the probed sequence. The sequence-state blob (seq 0) carries what the whole-context blob does not:
// qwen4exp's PLE conv window and n-gram token tail (llama_state_ple_write is reached only from the seq path), so with the
// whole-context blob the batch arm of every chunk after the first started from the reference arm's PLE state, K tokens
// ahead -- a probe artifact (ple scores off by ~1e3 at chunk 1 on Hemlock). BI_FULLSTATE=1 = the whole-context blob.
static size_t bi_snap(llama_context * ctx, std::vector<uint8_t> & buf) {
    static const bool full = getenv("BI_FULLSTATE") && atoi(getenv("BI_FULLSTATE")) != 0;
    if (full) { buf.resize(llama_state_get_size(ctx)); return llama_state_get_data(ctx, buf.data(), buf.size()); }
    buf.resize(llama_state_seq_get_size(ctx, 0, 0));
    return llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0, 0);
}
static void bi_restore(llama_context * ctx, const std::vector<uint8_t> & buf, size_t n) {
    static const bool full = getenv("BI_FULLSTATE") && atoi(getenv("BI_FULLSTATE")) != 0;
    if (full) { llama_state_set_data(ctx, buf.data(), n); return; }
    llama_kv_cache_seq_rm(ctx, 0, -1, -1);
    if (llama_state_seq_set_data(ctx, buf.data(), n, 0, 0) == 0) { fprintf(stderr, "pxa-bi-probe: seq state restore failed\n"); exit(1); }
}

static bool decode_or_die(llama_context * ctx, llama_batch & b, const char * what) {
    if (llama_decode(ctx, b)) { fprintf(stderr, "pxa-bi-probe: llama_decode failed (%s)\n", what); return false; }
    return true;
}

int main(int argc, char ** argv) {
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }
    std::vector<int> ks;
    {
        const char * e = getenv("BI_KS");
        std::string s = e && *e ? e : "4";
        size_t p = 0;
        while (p <= s.size()) {
            size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
            if (q > p) ks.push_back(atoi(s.substr(p, q - p).c_str()));
            p = q + 1;
        }
    }
    const int N = env_int("BI_N", 64), PMAX = env_int("BI_PROMPT_MAX", 0), VERB = env_int("BI_VERBOSE", 0);
    static bi_cb cb;
    const char * dump = getenv("BI_DUMP");
    if (dump && *dump) {
        cb.on = true;
        if (!strcmp(dump, "all")) cb.all = true;
        else {
            std::string s = dump; size_t p = 0;
            while (p <= s.size()) {
                size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
                if (q > p) cb.flt.push_back(s.substr(p, q - p));
                p = q + 1;
            }
        }
        params.cb_eval = bi_eval;
        params.cb_eval_user_data = &cb;
    }
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);
    llama_init_result init = llama_init_from_gpt_params(params);
    llama_model * model = init.model;
    llama_context * ctx = init.context;
    if (!model || !ctx) { fprintf(stderr, "pxa-bi-probe: init failed\n"); return 1; }
    const int nv = llama_n_vocab(model);
    int kmax = 1;
    for (int k : ks) kmax = std::max(kmax, k);

    // BI_PARSE_SPECIAL=1 (10-19): parse special tokens in the prompt text (a chat-templated prompt file then
    // tokenizes like the server's --jinja prompt). BI_PROMPT_TAIL=n: the last n prompt tokens go in their own forward, after
    // the rest -- the server's prompt split for a recurrent model (a context checkpoint n tokens before the prompt end), so the
    // probed continuation is the server's plain-greedy one.
    // BI_CKPT=<mode> (10-19; llama_spec_ckpt_mode, 0 = auto = what the server asks for): arm the recurrent
    // speculative checkpoint like the server's speculative init (llama_spec_ckpt_init + discard, capacity = the widest K), then
    // run every verify forward the way the server does (init + save before it, so the per-step state capture runs) and after
    // it test the ROLLBACK: restore to accepted step r = c % (K-1) (a rejection after draft r), decode the next reference
    // token at width 1 and compare its logits with the one-token reference -- the server's state after a rejected draft.
    const int  CKPT = getenv("BI_CKPT") ? env_int("BI_CKPT", 0) : -1;
    if (CKPT >= 0) {
        const int am = llama_spec_ckpt_init(ctx, CKPT, kmax);
        printf("pxa-bi-probe: BI_CKPT %d -> llama_spec_ckpt_init mode %d (capacity %d)\n", CKPT, am, kmax);
        llama_spec_ckpt_discard(ctx);
        if (am < 0) { fprintf(stderr, "pxa-bi-probe: no recurrent checkpoint mode\n"); return 1; }
    }
    const bool PSPEC = env_int("BI_PARSE_SPECIAL", 0) != 0;
    const int  PTAIL = env_int("BI_PROMPT_TAIL", 0);
    std::vector<llama_token> prompt = common_tokenize(ctx, params.prompt, llama_should_add_bos_token(model), PSPEC);
    if (PMAX > 0 && (int) prompt.size() > PMAX) prompt.resize(PMAX);
    if (prompt.empty()) { fprintf(stderr, "pxa-bi-probe: empty prompt\n"); return 1; }
    const int nb = std::max<int>(params.n_batch, kmax);
    llama_batch batch = llama_batch_init(nb, 0, 1);
    int n_past = 0;
    const size_t phead = (PTAIL > 0 && (size_t) PTAIL < prompt.size()) ? prompt.size() - PTAIL : prompt.size();
    for (size_t i = 0; i < prompt.size(); ) {
        common_batch_clear(batch);
        const size_t lim = i < phead ? phead : prompt.size();
        const int n = (int) std::min<size_t>(params.n_batch, lim - i);
        for (int j = 0; j < n; ++j) common_batch_add(batch, prompt[i + j], n_past + j, {0}, i + j + 1 == prompt.size());
        if (!decode_or_die(ctx, batch, "prompt")) return 1;
        n_past += n;
        i += n;
    }
    const llama_token first = argmax_of(llama_get_logits_ith(ctx, batch.n_tokens - 1), nv);
    const int p0 = n_past;
    std::vector<uint8_t> S0;
    const size_t s0n = bi_snap(ctx, S0);
    printf("pxa-bi-probe: prompt %d tokens, n_vocab %d, widths", p0, nv);
    for (int k : ks) printf(" %d", k);
    printf(", %d tokens per width, dump %s\n", N, cb.on ? (cb.all ? "all" : dump) : "off (production graph)");

    int rc = 0;
    for (int K : ks) {
        if (K < 1) continue;
        const int nchunk = N / K, NT = nchunk * K;
        bi_restore(ctx, S0, s0n);
        n_past = p0;
        llama_token next = first;
        std::vector<float> la((size_t) NT * nv), lb((size_t) NT * nv);
        std::vector<llama_token> tok(NT);
        std::vector<std::vector<bi_node>> dA(cb.on ? K : 0), dB(cb.on ? 1 : 0);
        std::vector<uint8_t> S, S2, MA, MB;
        int st_eq = 0;
        printf("\n== K %d: %d chunks, positions %d..%d\n", K, nchunk, p0, p0 + NT - 1);
        std::vector<int> col_eq(K, 0), col_n(K, 0), col_am(K, 0);
        std::map<std::string, int> first_tally;
        std::map<std::string, int> src_tally;   // BI_SOURCES: diverging nodes none of whose captured inputs diverge
        int n_eq = 0, n_am = 0, rb_eq = 0, rb_n = 0;
        for (int c = 0; c < nchunk; ++c) {
            for (auto & v : dA) v.clear();
            for (auto & v : dB) v.clear();
            const size_t sn = bi_snap(ctx, S);
            for (int j = 0; j < K; ++j) {                                  // reference arm: one token per forward
                tok[c*K + j] = next;
                common_batch_clear(batch);
                common_batch_add(batch, next, n_past + j, {0}, true);
                cb.occ.clear(); cb.sink = cb.on ? &dA[j] : nullptr;
                if (!decode_or_die(ctx, batch, "reference")) return 1;
                cb.sink = nullptr;
                const float * lg = llama_get_logits_ith(ctx, 0);
                memcpy(&la[(size_t) (c*K + j) * nv], lg, sizeof(float) * nv);
                next = argmax_of(lg, nv);
            }
            const size_t mna = bi_seq_state(ctx, MA);
            const size_t sn2 = bi_snap(ctx, S2);
            bi_restore(ctx, S, sn);
            if (CKPT >= 0) {                                              // BI_CKPT: the server's pre-verify save
                const int am = llama_spec_ckpt_init(ctx, CKPT, K);
                if (am < 0 || !llama_spec_ckpt_save(ctx, 0)) { fprintf(stderr, "pxa-bi-probe: spec ckpt save failed (mode %d)\n", am); return 1; }
            }
            common_batch_clear(batch);                                    // verify arm: the same K tokens, one forward
            for (int j = 0; j < K; ++j) common_batch_add(batch, tok[c*K + j], n_past + j, {0}, true);
            cb.occ.clear(); cb.sink = cb.on ? &dB[0] : nullptr;
            if (!decode_or_die(ctx, batch, "batch")) return 1;
            cb.sink = nullptr;
            for (int j = 0; j < K; ++j) memcpy(&lb[(size_t) (c*K + j) * nv], llama_get_logits_ith(ctx, j), sizeof(float) * nv);
            {   // the memory the next step starts from: after the K one-token forwards vs after the one K-token forward
                const size_t mnb = bi_seq_state(ctx, MB);
                long long nd = 0, first = -1;
                if (mna == mnb) {
                    for (size_t q = 0; q < mna; ++q) if (MA[q] != MB[q]) { ++nd; if (first < 0) first = (long long) q; }
                }
                if (mna == mnb && nd == 0) ++st_eq;
                else printf("chunk %3d state DIFF  bytes %lld of %zu (batch %zu) first at %lld\n", c, mna == mnb ? nd : -1LL, mna, mnb, first);
            }
            if (CKPT >= 0 && K >= 2) {                                    // BI_CKPT: rollback to accepted step r, next token at width 1
                const int r = c % (K - 1);
                if (!llama_spec_ckpt_restore(ctx, 0, n_past, r)) { fprintf(stderr, "pxa-bi-probe: spec ckpt restore failed\n"); return 1; }
                common_batch_clear(batch);
                common_batch_add(batch, tok[c*K + r + 1], n_past + r + 1, {0}, true);
                if (!decode_or_die(ctx, batch, "rollback")) return 1;
                const float * a = &la[(size_t) (c*K + r + 1) * nv];
                const float * b = llama_get_logits_ith(ctx, 0);
                const bool eq = !memcmp(a, b, sizeof(float) * nv);
                rb_n++; rb_eq += eq;
                if (!eq) {
                    double md = 0.0;
                    for (int v = 0; v < nv; ++v) md = std::max(md, std::fabs((double) a[v] - (double) b[v]));
                    printf("chunk %3d rollback to step %d: next-token logits DIFF max|d| %.3g argmax %d/%d\n", c, r, md, argmax_of(a, nv), argmax_of(b, nv));
                }
            }
            if (CKPT >= 0) llama_spec_ckpt_discard(ctx);
            bi_restore(ctx, S2, sn2);
            n_past += K;
        for (int i = c*K; i < (c + 1)*K; ++i) {
            const float * a = &la[(size_t) i * nv];
            const float * b = &lb[(size_t) i * nv];
            const bool eq = !memcmp(a, b, sizeof(float) * nv);
            double md = 0.0;
            for (int v = 0; v < nv; ++v) md = std::max(md, std::fabs((double) a[v] - (double) b[v]));
            const int aa = argmax_of(a, nv), ab = argmax_of(b, nv);
            const int j = i % K;
            col_n[j]++; col_eq[j] += eq; col_am[j] += aa == ab; n_eq += eq; n_am += aa == ab;
            std::string fd;
            if (cb.on) {
                const std::vector<bi_node> & A = dA[i - c*K];
                const std::vector<bi_node> & B = dB[0];
                std::map<std::string, size_t> bi;
                for (size_t q = 0; q < B.size(); ++q) bi[B[q].key] = q;
                int shown = 0, nmiss = 0;
                std::vector<int> diff_of(A.size(), 0);   // BI_SOURCES: 1 differs, 0 equal, -1 not comparable
                for (size_t ia = 0; ia < A.size(); ++ia) {
                    const bi_node & na = A[ia];
                    auto it = bi.find(na.key);
                    if (it == bi.end()) { ++nmiss; diff_of[ia] = -1; continue; }
                    double dd; bool whole;
                    const long long nd = bi_cmp(na, B[it->second], K, j, &dd, &whole);
                    diff_of[ia] = nd > 0 ? 1 : nd == 0 ? 0 : -1;
                    if (nd > 0 && !whole) {   // a whole-tensor (state) compare at the last column is no width test
                        bool src_diff = false, src_unk = false;
                        for (int si : na.srci) {
                            if (si < 0) continue;                      // weight / leaf / host input: identical by construction
                            if (diff_of[si] > 0) src_diff = true; else if (diff_of[si] < 0) src_unk = true;
                        }
                        if (!src_diff) {
                            char sb[320];
                            snprintf(sb, sizeof(sb), "j%d %s %s(%s, %s) %s [%lld,%lld,%lld,%lld]%s", j, na.key.c_str(), na.op.c_str(),
                                     na.src0.c_str(), na.src1.c_str(), ggml_type_name(na.type), (long long) na.ne[0], (long long) na.ne[1],
                                     (long long) na.ne[2], (long long) na.ne[3], src_unk ? " (an input not comparable)" : "");
                            src_tally[sb]++;
                        }
                    }
                    if (nd == -2) continue;
                    if (nd == -1) {
                        if (VERB) printf("      shape? %s %s A[%lld,%lld,%lld,%lld] B[%lld,%lld,%lld,%lld]\n", na.key.c_str(), na.op.c_str(),
                            (long long) na.ne[0], (long long) na.ne[1], (long long) na.ne[2], (long long) na.ne[3],
                            (long long) B[it->second].ne[0], (long long) B[it->second].ne[1], (long long) B[it->second].ne[2], (long long) B[it->second].ne[3]);
                        continue;
                    }
                    if (nd > 0) {
                        char buf[512];
                        snprintf(buf, sizeof(buf), "%s %s(%s, %s) %s [%lld,%lld,%lld,%lld] ndiff %lld max|d| %.3g%s",
                                 na.key.c_str(), na.op.c_str(), na.src0.c_str(), na.src1.c_str(), ggml_type_name(na.type),
                                 (long long) na.ne[0], (long long) na.ne[1], (long long) na.ne[2], (long long) na.ne[3], nd, dd,
                                 whole ? " (whole)" : "");
                        if (fd.empty()) { fd = buf; first_tally[na.key + " " + na.op]++; }
                        if (VERB && shown < 12) { printf("      diff %s\n", buf); ++shown; }
                    }
                }
                if (nmiss && VERB) printf("      %d reference nodes have no batch twin\n", nmiss);
            }
            printf("t %3d pos %5d j %d tok %6d logits %s max|d| %-9.3g argmax %6d/%6d %s margin %.4f%s%s\n", i, p0 + i, j, tok[i],
                   eq ? "EQUAL" : "DIFF ", md, aa, ab, aa == ab ? "same" : "FLIP", top_margin(a, nv),
                   fd.empty() ? "" : "  first: ", fd.c_str());
        }
        fflush(stdout);
        }
        printf("== K %d summary: logits bitwise equal %d/%d, argmax equal %d/%d; per column j (equal/n):", K, n_eq, NT, n_am, NT);
        for (int j = 0; j < K; ++j) printf(" j%d %d/%d", j, col_eq[j], col_n[j]);
        printf("; memory state after the chunk equal %d/%d", st_eq, nchunk);
        if (CKPT >= 0 && K >= 2) printf("; rollback next-token logits equal %d/%d", rb_eq, rb_n);
        printf("\n");
        if (cb.on) {
            std::vector<std::pair<int, std::string>> v;
            for (auto & kv : first_tally) v.push_back({kv.second, kv.first});
            std::sort(v.rbegin(), v.rend());
            printf("== K %d first divergent node tally:\n", K);
            for (size_t q = 0; q < v.size() && q < 20; ++q) printf("   %4d  %s\n", v[q].first, v[q].second.c_str());
            // BI_SOURCES (10-19): every node whose token slice differs while none of its captured inputs' slices
            // does -- each one is an op whose arithmetic depends on the batch width (column j0 is the clean test: a later column
            // also reads the batch's own earlier rows through K/V or a recurrent state, so a j>0 entry may be inherited)
            std::vector<std::pair<int, std::string>> w;
            for (auto & kv : src_tally) w.push_back({kv.second, kv.first});
            std::sort(w.begin(), w.end(), [](const std::pair<int, std::string> & x, const std::pair<int, std::string> & y) {
                return x.second < y.second; });
            printf("== K %d source nodes (slice differs, no captured input slice differs): %zu\n", K, w.size());
            for (size_t q = 0; q < w.size() && q < 60; ++q) printf("   %4d  %s\n", w[q].first, w[q].second.c_str());
        }
        if (n_eq != NT || st_eq != nchunk || rb_eq != rb_n) rc = 2;
        fflush(stdout);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return rc;
}
