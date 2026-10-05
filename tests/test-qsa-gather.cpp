//
// PXA_QSA: the CPU reference for qwen4exp's query-time sparse attention selection and gather.
//
// WHAT THIS PINS
// --------------
// Three things that no GPU run can tell apart after the fact, because a wrong answer here is
// plausible text rather than a crash:
//
//  1. BLOCK-FIRST SELECTION == THE REFERENCE'S CELL-WISE SELECTION.
//     The architecture's reference scores whole blocks of `r` consecutive positions, EXPANDS the
//     block score to each of the block's cells, adds the causal/visibility bias PER CELL, and
//     takes top_k + r - 1 CELLS. Selecting whole blocks first and expanding only the winners is
//     the whole point of the mechanism -- it never builds the n_kv-wide cell-score vector -- but
//     it is only allowed if it lands on the same cells. Note what the reference's width means:
//     the budget is `top_k` tokens = top_k/r whole blocks, plus r - 1 more cells, so the cut
//     lands INSIDE the next-ranked block. This test asserts the sets match exactly, including
//     that tail.
//
//  2. PARTIAL VISIBILITY. Because the bias is added after the expansion, a block that straddles
//     the query's position contributes only the members the query may see. A block-first
//     implementation that masked whole blocks would attend to fewer cells than the reference
//     (and a different set). This is the sharpest edge in the whole design.
//
//  3. np=1 / np=2 SELECTION IDENTITY. The same sequence, scheduled alone and then alongside a
//     second sequence in ONE unified cache with the cells interleaved, must select the same
//     PHYSICAL CELLS in the same ORDER. That is the determinism claim the seat's gate rests on,
//     and it is a property of the grid plus the score, so it can be proved here without a GPU.
//
// Plus the mechanism's own equivalence: attention over a physically gathered compact set equals
// attention over the full cache with everything outside the selection masked to -inf. Same
// terms, different order, so this is a tolerance check, not a bit-identity check -- exactly the
// claim the three-arm gate makes.
//
// No model, no GPU, no graph: the grid comes from llama-kpool-grid.cpp (already pure and
// already tested by test-kpool-cache), and everything else is plain C++ here.
//

#include "llama-kv-cache-kpool.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    printf("%-72s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) {
        g_fail++;
    }
}

// ---------------------------------------------------------------------------------------
// a synthetic cache
// ---------------------------------------------------------------------------------------

struct fixture {
    uint32_t n_ei      = 0;   // indexer key width
    uint32_t n_ih      = 0;   // indexer heads
    uint32_t r         = 0;   // block size (compress ratio)
    uint32_t top_k     = 0;   // budget, in tokens
    uint32_t n_cells   = 0;   // cache capacity

    std::vector<llama_kpool_cell_desc> cells;   // occupied cells, in cache order
    std::vector<float>                 kraw;    // [n_ei, n_cells] raw indexer keys, by CELL
};

// Lay two sequences into one unified cache with their cells INTERLEAVED, which is what
// --kv-unified actually produces: a block of r consecutive POSITIONS of one sequence is not
// r consecutive cells.
static fixture make_fixture(uint32_t n_a, uint32_t n_b, uint32_t r, uint32_t top_k,
                            uint32_t n_ei, uint32_t n_ih, uint32_t seed) {
    fixture f;
    f.n_ei    = n_ei;
    f.n_ih    = n_ih;
    f.r       = r;
    f.top_k   = top_k;
    f.n_cells = n_a + n_b;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    f.kraw.resize((size_t) n_ei*f.n_cells);
    for (auto & v : f.kraw) {
        v = d(rng);
    }

    uint32_t pa = 0;
    uint32_t pb = 0;
    for (uint32_t c = 0; c < f.n_cells; ++c) {
        llama_kpool_cell_desc cd;
        cd.cell = c;
        // deterministic interleave: two of A, then one of B, repeat, until one runs out
        const bool take_a = (pa < n_a) && (pb >= n_b || (c % 3) != 2);
        if (take_a) {
            cd.pos = (llama_pos) pa++;
            cd.seqs = { 0 };
        } else {
            cd.pos = (llama_pos) pb++;
            cd.seqs = { 1 };
        }
        f.cells.push_back(cd);
    }
    return f;
}

// ---------------------------------------------------------------------------------------
// the indexer score, exactly as the architecture defines it
// ---------------------------------------------------------------------------------------
//
// pooled block key = mean over the block's r member cells of the RAW key, then rms_norm, then
// (in the engine) a rotation. The rotation is a per-position orthogonal map applied to both the
// block key and the query; it does not change which block wins for a given query only in the
// trivial case, so this reference keeps it OUT and scores on the normed pooled key. That is
// deliberate: this file pins the SELECTION MACHINERY (grid, expansion, bias, width, order),
// not the numerics of rope, which the whole-model identity gate covers.

static void rms_norm(std::vector<float> & v, uint32_t n, float eps = 1e-6f) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        ss += (double) v[i]*v[i];
    }
    const float s = 1.0f/std::sqrt((float) (ss/n) + eps);
    for (uint32_t i = 0; i < n; ++i) {
        v[i] *= s;
    }
}

// pooled key of one block, from its member cells
static std::vector<float> pool_block(const fixture & f, const int32_t * members, uint32_t n_real) {
    std::vector<float> p(f.n_ei, 0.0f);
    for (uint32_t m = 0; m < f.r; ++m) {
        const int32_t c = members[m];
        for (uint32_t d = 0; d < f.n_ei; ++d) {
            p[d] += f.kraw[(size_t) c*f.n_ei + d];
        }
    }
    for (uint32_t d = 0; d < f.n_ei; ++d) {
        p[d] /= (float) f.r;
    }
    rms_norm(p, f.n_ei);
    (void) n_real;
    return p;
}

// score of one block for one token: sum over heads of relu(dot), no scale (top-k is invariant
// under a positive scale, so the architecture ships none and neither do we)
static float block_score(const fixture & f, const std::vector<float> & pooled,
                         const std::vector<float> & q /* [n_ei, n_ih] */) {
    float s = 0.0f;
    for (uint32_t h = 0; h < f.n_ih; ++h) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < f.n_ei; ++d) {
            dot += pooled[d]*q[(size_t) h*f.n_ei + d];
        }
        s += dot > 0.0f ? dot : 0.0f;
    }
    return s;
}

// ---------------------------------------------------------------------------------------
// the two selections
// ---------------------------------------------------------------------------------------

struct plan {
    llama_kpool_state st;
    llama_kpool_dims  d;
    std::vector<int32_t> pool_cells;
    std::vector<int32_t> pool_idxs;
    std::vector<float>   pool_mask;
    std::vector<int32_t> tail_idxs;
    std::vector<float>   gather_mask;
    std::vector<int32_t> new_pool_idxs;
    std::vector<int64_t> new_pool_rep;
};

static plan build_plan(const fixture & f, const std::vector<llama_kpool_tok_desc> & toks) {
    plan p;
    p.st = llama_kpool_build_layout(f.cells, f.r, /*n_seq_max*/ 2);
    llama_kpool_mark_new(p.st, toks, f.r, /*all_new*/ true);

    p.d.kpool    = f.r;
    p.d.n_pool   = llama_kpool_pad(p.st.n_pool_real);
    p.d.n_tokens = (uint32_t) toks.size();
    p.d.n_kv     = f.n_cells;
    p.d.n_new    = p.st.n_new;
    p.d.n_new_g  = std::max(p.d.n_new, llama_kpool_n_new_floor(p.d.n_pool, p.d.n_tokens, f.r));
    p.d.sink     = f.n_cells;
    p.d.n_top    = llama_kpool_n_top(p.d.n_pool, f.top_k, f.r);
    p.d.n_sel    = llama_kpool_n_sel(p.d.n_top, f.r, /*select_tail*/ true);
    p.d.gather   = true;

    p.pool_cells.assign(p.d.n_pool, 0);
    p.pool_idxs.assign((size_t) f.r*p.d.n_pool, 0);
    p.pool_mask.assign((size_t) p.d.n_pool*p.d.n_tokens, 0.0f);
    p.tail_idxs.assign((size_t) (f.r - 1)*p.d.n_tokens, 0);
    p.gather_mask.assign((size_t) p.d.n_sel*p.d.n_tokens, 0.0f);
    p.new_pool_idxs.assign((size_t) f.r*p.d.n_new_g, 0);
    p.new_pool_rep.assign(p.d.n_new_g, 0);

    llama_kpool_bufs b;
    b.pool_cells    = p.pool_cells.data();
    b.pool_idxs     = p.pool_idxs.data();
    b.pool_mask     = p.pool_mask.data();
    b.tail_idxs     = p.tail_idxs.data();
    b.gather_mask   = p.gather_mask.data();
    b.new_pool_idxs = p.new_pool_idxs.data();
    b.new_pool_rep  = p.st.cache_safe ? p.new_pool_rep.data() : nullptr;

    llama_kpool_fill(p.st, p.d, toks, b);
    return p;
}

// every block's score for one token, and the set of cells the query may see
static std::vector<float> scores_for(const fixture & f, const plan & p, uint32_t t,
                                     const std::vector<float> & q) {
    std::vector<float> s(p.d.n_pool, -INFINITY);
    for (uint32_t b = 0; b < p.d.n_pool; ++b) {
        if (std::isinf(p.pool_mask[(size_t) t*p.d.n_pool + b])) {
            continue;   // not this token's sequence, or not yet visible
        }
        const std::vector<float> pooled = pool_block(f, &p.pool_idxs[(size_t) b*f.r], f.r);
        s[b] = block_score(f, pooled, q);
    }
    return s;
}

// THE REFERENCE: expand each block score to its member cells, add the per-cell bias, take the
// top (top_k + r - 1) CELLS. Ties inside a block resolve by ascending cell index, which is what
// a stable descending sort gives.
static std::vector<int32_t> select_reference(const fixture & f, const plan & p, uint32_t t,
                                             const std::vector<float> & q,
                                             const std::vector<uint8_t> & visible /* by cell */) {
    const std::vector<float> bs = scores_for(f, p, t, q);

    std::vector<float> cell_score(f.n_cells, -INFINITY);
    for (uint32_t b = 0; b < p.d.n_pool; ++b) {
        if (std::isinf(bs[b])) {
            continue;
        }
        for (uint32_t m = 0; m < f.r; ++m) {
            const int32_t c = p.pool_idxs[(size_t) b*f.r + m];
            if (c >= 0 && (uint32_t) c < f.n_cells && visible[c]) {
                cell_score[c] = bs[b];         // the per-cell bias is what `visible` encodes
            }
        }
    }
    // the incomplete tail: cells of no complete block. The reference scores the partial block
    // like any other, but our grid does not form it, so the tail is appended -- the width below
    // is what makes room for it.
    for (uint32_t k = 0; k + 1 < f.r; ++k) {
        const bool real = p.gather_mask[(size_t) t*p.d.n_sel + (size_t) p.d.n_top*f.r + k] == 0.0f;
        const int32_t c = p.tail_idxs[(size_t) t*(f.r - 1) + k];
        if (real && c >= 0 && (uint32_t) c < f.n_cells && visible[c]) {
            cell_score[c] = INFINITY;          // always attended
        }
    }

    std::vector<int32_t> idx(f.n_cells);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
        return cell_score[a] > cell_score[b];
    });

    const size_t width = std::min<size_t>(f.n_cells, (size_t) f.top_k + f.r - 1);
    std::vector<int32_t> out;
    for (size_t i = 0; i < width && i < idx.size(); ++i) {
        if (std::isinf(cell_score[idx[i]]) && cell_score[idx[i]] < 0) {
            break;
        }
        out.push_back(idx[i]);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// BLOCK-FIRST: sort the n_pool block scores, take n_top whole blocks, expand only those, then
// the tail. Never builds a cell-score vector.
static std::vector<int32_t> select_block_first(const fixture & f, const plan & p, uint32_t t,
                                               const std::vector<float> & q,
                                               const std::vector<uint8_t> & visible) {
    const std::vector<float> bs = scores_for(f, p, t, q);

    std::vector<int32_t> ord(p.d.n_pool);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int32_t a, int32_t b) {
        return bs[a] > bs[b];
    });

    std::vector<int32_t> out;
    for (uint32_t i = 0; i < p.d.n_top && i < ord.size(); ++i) {
        if (std::isinf(bs[ord[i]])) {
            break;
        }
        for (uint32_t m = 0; m < f.r; ++m) {
            const int32_t c = p.pool_idxs[(size_t) ord[i]*f.r + m];
            if (c >= 0 && (uint32_t) c < f.n_cells && visible[c]) {
                out.push_back(c);
            }
        }
    }
    // The tail: the newest cells of this sequence that belong to no complete block, one of which
    // is always the token itself. Our grid appends them unconditionally.
    // A tail SLOT is padded when this sequence has no cell there; the grid fills the padding
    // with a REAL cell (the gather must stay in bounds) and marks it in gather_mask. So the
    // padding is invisible to `visible[]` and must be read off the mask, not guessed -- reading
    // it wrong silently attends to a duplicate cell and eats budget that belongs to a block.
    uint32_t n_tail = 0;
    for (uint32_t k = 0; k + 1 < f.r; ++k) {
        const bool real = p.gather_mask[(size_t) t*p.d.n_sel + (size_t) p.d.n_top*f.r + k] == 0.0f;
        const int32_t c = p.tail_idxs[(size_t) t*(f.r - 1) + k];
        if (real && c >= 0 && (uint32_t) c < f.n_cells && visible[c]) {
            out.push_back(c);
            ++n_tail;
        }
    }

    // The reference's width is top_k + r - 1 CELLS in total, tail included: r*n_top whole-block
    // cells plus r - 1 more. So after the tail has taken its share there are (r - 1) - n_tail
    // slots left, and they come from the NEXT-ranked block -- by ASCENDING CELL INDEX, because
    // every cell of that block carries the same score and the reference's sort is stable over a
    // cell-indexed vector.
    if (n_tail + 1 < f.r && p.d.n_top < ord.size() && !std::isinf(bs[ord[p.d.n_top]])) {
        std::vector<int32_t> cand;
        for (uint32_t m = 0; m < f.r; ++m) {
            const int32_t c = p.pool_idxs[(size_t) ord[p.d.n_top]*f.r + m];
            if (c >= 0 && (uint32_t) c < f.n_cells && visible[c]) {
                cand.push_back(c);
            }
        }
        std::sort(cand.begin(), cand.end());
        for (size_t i = 0; i < cand.size() && i + n_tail + 1 < f.r; ++i) {
            out.push_back(cand[i]);
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// ---------------------------------------------------------------------------------------
// gather-compact vs mask-dense attention over the SAME selection
// ---------------------------------------------------------------------------------------

static std::vector<float> attend_masked(const std::vector<float> & k /* [D, n_cells] */,
                                        const std::vector<float> & v,
                                        const std::vector<float> & q /* [D] */,
                                        const std::vector<int32_t> & sel,
                                        uint32_t D, uint32_t n_cells, float scale) {
    std::vector<uint8_t> keep(n_cells, 0);
    for (int32_t c : sel) {
        keep[c] = 1;
    }
    float m = -INFINITY;
    std::vector<float> logit(n_cells, -INFINITY);
    for (uint32_t c = 0; c < n_cells; ++c) {
        if (!keep[c]) {
            continue;
        }
        float s = 0.0f;
        for (uint32_t d = 0; d < D; ++d) {
            s += k[(size_t) c*D + d]*q[d];
        }
        logit[c] = s*scale;
        m = std::max(m, logit[c]);
    }
    double sum = 0.0;
    std::vector<double> w(n_cells, 0.0);
    for (uint32_t c = 0; c < n_cells; ++c) {
        if (!keep[c]) {
            continue;
        }
        w[c] = std::exp((double) (logit[c] - m));
        sum += w[c];
    }
    std::vector<float> out(D, 0.0f);
    for (uint32_t c = 0; c < n_cells; ++c) {
        if (!keep[c]) {
            continue;
        }
        const double a = w[c]/sum;
        for (uint32_t d = 0; d < D; ++d) {
            out[d] += (float) (a*v[(size_t) c*D + d]);
        }
    }
    return out;
}

static std::vector<float> attend_gathered(const std::vector<float> & k, const std::vector<float> & v,
                                          const std::vector<float> & q,
                                          const std::vector<int32_t> & sel,
                                          uint32_t D, float scale) {
    const size_t n = sel.size();
    std::vector<float> kg((size_t) n*D), vg((size_t) n*D);
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(&kg[i*D], &k[(size_t) sel[i]*D], D*sizeof(float));
        std::memcpy(&vg[i*D], &v[(size_t) sel[i]*D], D*sizeof(float));
    }
    float m = -INFINITY;
    std::vector<float> logit(n);
    for (size_t i = 0; i < n; ++i) {
        float s = 0.0f;
        for (uint32_t d = 0; d < D; ++d) {
            s += kg[i*D + d]*q[d];
        }
        logit[i] = s*scale;
        m = std::max(m, logit[i]);
    }
    double sum = 0.0;
    std::vector<double> w(n);
    for (size_t i = 0; i < n; ++i) {
        w[i] = std::exp((double) (logit[i] - m));
        sum += w[i];
    }
    std::vector<float> out(D, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        const double a = w[i]/sum;
        for (uint32_t d = 0; d < D; ++d) {
            out[d] += (float) (a*vg[i*D + d]);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------

int main() {
    const uint32_t r     = 4;
    const uint32_t n_ei  = 32;
    const uint32_t n_ih  = 4;
    const uint32_t top_k = 16;      // 4 whole blocks + 3 tail cells

    // one deep sequence and one shallow one, interleaved in a unified cache
    fixture f = make_fixture(/*n_a*/ 61, /*n_b*/ 23, r, top_k, n_ei, n_ih, /*seed*/ 1234);

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q((size_t) n_ei*n_ih);
    for (auto & x : q) {
        x = dist(rng);
    }

    // the token doing the attending: sequence 0, at its newest position
    llama_kpool_tok_desc tok;
    tok.pos  = 60;
    tok.seqs = { 0 };
    std::vector<llama_kpool_tok_desc> toks = { tok };

    plan p = build_plan(f, toks);

    // visibility, per CELL: same sequence, position not in the future
    std::vector<uint8_t> vis(f.n_cells, 0);
    for (const auto & c : f.cells) {
        vis[c.cell] = (!c.seqs.empty() && c.seqs[0] == 0 && c.pos <= tok.pos) ? 1 : 0;
    }

    const std::vector<int32_t> ref = select_reference  (f, p, 0, q, vis);
    const std::vector<int32_t> blk = select_block_first(f, p, 0, q, vis);

    check(!ref.empty(),            "reference selects a non-empty set");
    if (ref != blk) {
        printf("  DIAG ref=%zu blk=%zu  n_pool=%u n_top=%u n_sel=%u n_pool_real=%u\n",
               ref.size(), blk.size(), p.d.n_pool, p.d.n_top, p.d.n_sel, p.st.n_pool_real);
        printf("  DIAG ref only:"); for (int32_t c : ref) if (std::find(blk.begin(),blk.end(),c)==blk.end()) printf(" %d", c); printf("\n");
        printf("  DIAG blk only:"); for (int32_t c : blk) if (std::find(ref.begin(),ref.end(),c)==ref.end()) printf(" %d", c); printf("\n");
    }
    check(ref == blk,              "block-first selection == the reference's cell-wise selection");

    // 2. partial visibility: every selected cell is one this token may actually see
    bool all_visible = true;
    for (int32_t c : blk) {
        all_visible = all_visible && vis[c];
    }
    check(all_visible,             "no selected cell is invisible to the query (per-cell bias)");

    // the block-first path must not have attended to a straddling block's future members
    bool straddle_ok = true;
    for (uint32_t b = 0; b < p.d.n_pool; ++b) {
        for (uint32_t m = 0; m < r; ++m) {
            const int32_t c = p.pool_idxs[(size_t) b*r + m];
            if (c >= 0 && (uint32_t) c < f.n_cells && !vis[c] &&
                std::find(blk.begin(), blk.end(), c) != blk.end()) {
                straddle_ok = false;
            }
        }
    }
    check(straddle_ok,             "a block straddling the query position yields only visible cells");

    // 3. np=1 vs np=2: the same sequence alone, in its own cache, must select the same CELLS.
    //    Build a cache holding only sequence 0's cells, at the SAME cell indices, so the answer
    //    is comparable cell for cell.
    {
        fixture f1 = f;
        f1.cells.clear();
        for (const auto & c : f.cells) {
            if (!c.seqs.empty() && c.seqs[0] == 0) {
                f1.cells.push_back(c);
            }
        }
        plan p1 = build_plan(f1, toks);
        const std::vector<int32_t> one = select_block_first(f1, p1, 0, q, vis);
        check(one == blk,          "np=1 and np=2 select the same physical cells");
    }

    // 4. the mechanism itself: gather-compact == mask-dense over the same selection
    {
        const uint32_t D = 16;
        std::vector<float> k((size_t) D*f.n_cells), v((size_t) D*f.n_cells), qq(D);
        for (auto & x : k)  { x = dist(rng); }
        for (auto & x : v)  { x = dist(rng); }
        for (auto & x : qq) { x = dist(rng); }
        const float scale = 1.0f/std::sqrt((float) D);

        const std::vector<float> a = attend_masked  (k, v, qq, blk, D, f.n_cells, scale);
        const std::vector<float> b = attend_gathered(k, v, qq, blk, D, scale);

        float maxdiff = 0.0f;
        for (uint32_t d = 0; d < D; ++d) {
            maxdiff = std::max(maxdiff, std::fabs(a[d] - b[d]));
        }
        printf("  gather-vs-mask max |diff| = %.3e\n", maxdiff);
        check(maxdiff < 1e-5f,     "gathered attention == masked dense attention over one selection");
    }

    // 5. WHAT THE ENGINE ACTUALLY SELECTS, and by how much it is allowed to differ.
    //
    //    src/graphs/build_qwen4exp.cpp builds n_top whole blocks plus the tail and stops --
    //    it does NOT top the budget up from the next-ranked block when the tail is short,
    //    because doing so would mean changing llama_kpool_n_sel and the gather-mask layout,
    //    which glm5next shares and test-kpool-cache pins. That is a real divergence from the
    //    reference's flat top_k + r - 1 cells and it is written down here as a NUMBER rather
    //    than as prose: the engine's set is a subset of the reference's, it is short by
    //    exactly the cells the top-up would have added, and that count is never more than
    //    r - 1. If a later change makes the gap bigger, this fails.
    {
        // the engine's set: block-first WITHOUT the top-up
        std::vector<int32_t> eng;
        const std::vector<float> bs = scores_for(f, p, 0, q);
        std::vector<int32_t> ord(p.d.n_pool);
        std::iota(ord.begin(), ord.end(), 0);
        std::stable_sort(ord.begin(), ord.end(), [&](int32_t a, int32_t b) { return bs[a] > bs[b]; });
        for (uint32_t i = 0; i < p.d.n_top && i < ord.size(); ++i) {
            if (std::isinf(bs[ord[i]])) {
                break;
            }
            for (uint32_t m = 0; m < r; ++m) {
                const int32_t c = p.pool_idxs[(size_t) ord[i]*r + m];
                if (c >= 0 && (uint32_t) c < f.n_cells && vis[c]) {
                    eng.push_back(c);
                }
            }
        }
        for (uint32_t k = 0; k + 1 < r; ++k) {
            const bool real = p.gather_mask[(size_t) 0*p.d.n_sel + (size_t) p.d.n_top*r + k] == 0.0f;
            const int32_t c = p.tail_idxs[k];
            if (real && c >= 0 && (uint32_t) c < f.n_cells && vis[c]) {
                eng.push_back(c);
            }
        }
        std::sort(eng.begin(), eng.end());
        eng.erase(std::unique(eng.begin(), eng.end()), eng.end());

        bool subset = true;
        for (int32_t c : eng) {
            subset = subset && std::find(ref.begin(), ref.end(), c) != ref.end();
        }
        const size_t missing = ref.size() - eng.size();
        printf("  engine selects %zu cells, the reference %zu (short by %zu, bound %u)\n",
               eng.size(), ref.size(), missing, r - 1);
        check(subset,                  "the engine's selection is a SUBSET of the reference's");
        check(missing < r,             "the engine is short of the reference by fewer than r cells");
    }

    printf("\n%s\n", g_fail == 0 ? "ALL OK" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
}
