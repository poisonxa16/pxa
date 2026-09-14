//
// PXA_QSA milestone 2: the SELECTION's cost, and what may be changed about it.
//
// Milestone 1's gather is correct and still slower than dense (P100 quad, 2026-09-09: 14.80
// t/s against 17.85 at 86,401 fill), and tests/test-qsa-gather.cpp already pins WHAT the
// selection must produce. This file is about what it COSTS and about the mechanisms that make
// it cheaper without changing the answer:
//
//   1. THE APPEND-ONLY POOL GRID. llama_kpool_build_layout() ran from a fresh scan of the whole
//      cache on every ubatch -- every decoded token. At decode the cache only grows, by the
//      ubatch's own cells, so the grid is a pure append. The claim is EQUALITY with the
//      from-scratch grid, over long streams, at np=1 and np=2, across ubatch widths and across
//      the sequence edits that must fall back. Asserted here, and timed here, because "4.5 ms
//      per token at 86k" is the number that justifies the mechanism.
//
//   2. THE CHEAP BLOCK TOP-K. The engine sorts all n_pool block scores (21,632 of them at 86k)
//      to keep 512. GGML_OP_QSA_TOPK is a radix SELECT instead, and it has to produce the SAME
//      SET as the full descending sort in the SAME ORDER (equal scores keep ascending block
//      index) or the two QSA arms stop agreeing. Asserted here over adversarial score vectors:
//      heavy ties, all-equal, half -inf, relu's exact zeros, k >= n.
//
//   3. THE SELECTION-RECALL PROBE. The one number that says whether a selection is doing its
//      job: the fraction of the DENSE attention mass that the selected cells carry. It is what
//      validates the pooling, the rope on the block key and the score's shape all at once --
//      a plausible-looking wrong selection scores badly here and nowhere else -- and it is the
//      only honest bound on reusing a selection for N tokens, which is milestone 2's deferred
//      mechanism (a). Pinned here on a synthetic layer; measured on the real model by a live
//      recall A/B.
//
// No model, no GPU, no graph.
//

#include "llama-kv-cache-kpool.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
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

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---------------------------------------------------------------------------------------
// a stand-in for the engine's KV cache: cells handed out contiguously from a head, exactly
// the way llama_kv_cache_find_slot does for a non-recurrent cache
// ---------------------------------------------------------------------------------------

struct fake_cache {
    uint32_t size = 0;
    uint32_t head = 0;
    std::vector<llama_pos>    pos;
    std::vector<llama_seq_id> seq;   // -1 = empty

    explicit fake_cache(uint32_t n) : size(n), pos(n, -1), seq(n, -1) {}

    // returns the cell indices this ubatch took
    std::vector<uint32_t> add(const std::vector<std::pair<llama_pos, llama_seq_id>> & toks) {
        std::vector<uint32_t> out;
        for (const auto & t : toks) {
            while (head < size && seq[head] >= 0) {
                ++head;
            }
            if (head >= size) {
                return out;   // full; the caller stops
            }
            pos[head] = t.first;
            seq[head] = t.second;
            out.push_back(head);
            ++head;
        }
        return out;
    }

    std::vector<llama_kpool_cell_desc> occupied() const {
        std::vector<llama_kpool_cell_desc> cells;
        cells.reserve(size);
        for (uint32_t i = 0; i < size; ++i) {
            if (seq[i] < 0) {
                continue;
            }
            llama_kpool_cell_desc d;
            d.pos = pos[i];
            d.cell = i;
            d.seqs = { seq[i] };
            cells.push_back(std::move(d));
        }
        return cells;
    }
};

static bool state_equal(const llama_kpool_state & a, const llama_kpool_state & b) {
    if (a.n_pool_real != b.n_pool_real || a.cache_safe != b.cache_safe ||
        a.n_new != b.n_new || a.seqs.size() != b.seqs.size()) {
        return false;
    }
    for (size_t s = 0; s < a.seqs.size(); ++s) {
        if (a.seqs[s].pos_min != b.seqs[s].pos_min ||
            a.seqs[s].cells   != b.seqs[s].cells   ||
            a.seqs[s].pools   != b.seqs[s].pools   ||
            a.seqs[s].is_new  != b.seqs[s].is_new) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// 1. the append-only grid == the from-scratch grid
// ---------------------------------------------------------------------------------------

// Drive `n_steps` ubatches of `width` tokens over `n_seq` sequences and compare, after EVERY
// ubatch, the grid the engine would have rebuilt with the grid the append produced.
static bool drive(uint32_t cache_size, uint32_t n_seq, uint32_t kpool,
                  uint32_t prefill, uint32_t n_steps, uint32_t width, bool * took_fast) {
    fake_cache kv(cache_size);
    std::vector<llama_pos> next(n_seq, 0);

    llama_kpool_state st;
    st.seqs.resize(n_seq);
    bool have = false;
    *took_fast = false;

    auto step = [&](const std::vector<std::pair<llama_pos, llama_seq_id>> & toks) -> bool {
        const std::vector<uint32_t> cells = kv.add(toks);
        if (cells.size() != toks.size()) {
            return true;   // cache full: stop, not a failure
        }

        std::vector<llama_kpool_tok_desc> td(toks.size());
        std::vector<llama_kpool_cell_desc> added(toks.size());
        for (size_t i = 0; i < toks.size(); ++i) {
            td[i].pos = toks[i].first;
            td[i].seqs = { toks[i].second };
            added[i].pos = toks[i].first;
            added[i].cell = cells[i];
            added[i].seqs = { toks[i].second };
        }

        bool fast = false;
        if (have) {
            std::vector<uint32_t> before(st.seqs.size());
            for (size_t s = 0; s < st.seqs.size(); ++s) {
                before[s] = (uint32_t) st.seqs[s].pools.size();
            }
            if (llama_kpool_append_layout(st, added, kpool)) {
                llama_kpool_mark_new_appended(st, before);
                fast = true;
                *took_fast = true;
            }
        }
        if (!fast) {
            st = llama_kpool_build_layout(kv.occupied(), kpool, n_seq);
            llama_kpool_mark_new(st, td, kpool, /*all_new*/ !have);
            have = true;
        }

        llama_kpool_state ref = llama_kpool_build_layout(kv.occupied(), kpool, n_seq);
        llama_kpool_mark_new(ref, td, kpool, /*all_new*/ false);
        // the first ubatch of a stream legitimately re-pools everything; compare the grid only
        if (!fast) {
            ref.n_new = st.n_new;
            for (size_t s = 0; s < ref.seqs.size(); ++s) {
                ref.seqs[s].is_new = st.seqs[s].is_new;
            }
        }
        return state_equal(st, ref);
    };

    // a prefill, in ubatches of `prefill` tokens of sequence 0..n_seq-1 round robin
    for (uint32_t done = 0; done < prefill; ) {
        const uint32_t w = std::min(prefill - done, 64u);
        for (uint32_t s = 0; s < n_seq; ++s) {
            std::vector<std::pair<llama_pos, llama_seq_id>> toks;
            for (uint32_t i = 0; i < w; ++i) {
                toks.emplace_back(next[s]++, (llama_seq_id) s);
            }
            if (!step(toks)) {
                return false;
            }
        }
        done += w;
    }

    // then decode-shaped ubatches, interleaving the sequences the way --kv-unified does
    for (uint32_t t = 0; t < n_steps; ++t) {
        for (uint32_t s = 0; s < n_seq; ++s) {
            std::vector<std::pair<llama_pos, llama_seq_id>> toks;
            for (uint32_t i = 0; i < width; ++i) {
                toks.emplace_back(next[s]++, (llama_seq_id) s);
            }
            if (!step(toks)) {
                return false;
            }
        }
    }
    return true;
}

static void test_grid_append() {
    printf("\n-- 1. the append-only pool grid --------------------------------------------\n");

    struct cfg { const char * name; uint32_t size, nseq, prefill, steps, width; };
    const cfg cfgs[] = {
        { "np=1, decode after a 300-token prefill",      4096, 1, 300,  200, 1 },
        { "np=2 unified, interleaved decode",            4096, 2, 300,  200, 1 },
        { "np=2, ubatch width 4 (spec verify shape)",    4096, 2, 300,   80, 4 },
        { "np=3, width 2, short prefill",                4096, 3,  17,  120, 2 },
        { "np=1, no prefill at all",                     4096, 1,   0,  400, 1 },
    };
    for (const auto & c : cfgs) {
        bool fast = false;
        const bool ok = drive(c.size, c.nseq, /*kpool*/ 4, c.prefill, c.steps, c.width, &fast);
        check(ok && fast, c.name);
    }

    // the precondition is a real gate, not a formality: an out-of-order position must be
    // REFUSED and must leave the layout untouched, so the caller's fallback is safe
    {
        std::vector<llama_kpool_cell_desc> cells;
        for (uint32_t i = 0; i < 40; ++i) {
            llama_kpool_cell_desc d;
            d.pos = (llama_pos) i;
            d.cell = i;
            d.seqs = { 0 };
            cells.push_back(d);
        }
        llama_kpool_state st = llama_kpool_build_layout(cells, 4, 2);
        const llama_kpool_state before = st;

        std::vector<llama_kpool_cell_desc> bad(1);
        bad[0].pos = 7;              // already cached: not strictly newer
        bad[0].cell = 40;
        bad[0].seqs = { 0 };
        check(!llama_kpool_append_layout(st, bad, 4), "an out-of-order position is refused");
        check(state_equal(st, before), "a refused append leaves the layout byte-identical");

        std::vector<llama_kpool_cell_desc> shared(1);
        shared[0].pos = 40;
        shared[0].cell = 41;
        shared[0].seqs = { 0, 1 };   // a cell shared between sequences flips cache_safe
        check(!llama_kpool_append_layout(st, shared, 4), "a cell shared between sequences is refused");
        check(state_equal(st, before), "a refused shared-cell append leaves the layout unchanged");

        // a gap: position 41 arrives with 40 missing. The grid must still match a rebuild --
        // the pool that would have straddled the gap simply never forms.
        std::vector<llama_kpool_cell_desc> gap(1);
        gap[0].pos = 41;
        gap[0].cell = 42;
        gap[0].seqs = { 0 };
        llama_kpool_state g = st;
        check(llama_kpool_append_layout(g, gap, 4), "a gapped append is accepted");
        cells.push_back(gap[0]);
        llama_kpool_state gref = llama_kpool_build_layout(cells, 4, 2);
        gref.n_new = g.n_new;
        for (size_t s = 0; s < gref.seqs.size(); ++s) {
            gref.seqs[s].is_new = g.seqs[s].is_new;
        }
        check(state_equal(g, gref), "a gapped append == the from-scratch grid");
    }
}

// ---------------------------------------------------------------------------------------
// 1b. what the mechanism is worth, in host microseconds at the seat's own depths
// ---------------------------------------------------------------------------------------

static void bench_grid() {
    printf("\n-- 1b. host cost of the grid, per decoded token -----------------------------\n");
    printf("   %10s %10s %14s %14s %10s\n", "capacity", "fill", "from-scratch", "append-only", "speedup");

    struct row { uint32_t size, fill; };
    const row rows[] = { { 32768, 20801 }, { 150016, 20801 }, { 150016, 86401 }, { 150016, 140000 } };
    const uint32_t kpool = 4;

    for (const auto & r : rows) {
        fake_cache kv(r.size);
        std::vector<std::pair<llama_pos, llama_seq_id>> toks;
        for (uint32_t i = 0; i < r.fill; ++i) {
            toks.emplace_back((llama_pos) i, (llama_seq_id) 0);
        }
        kv.add(toks);

        std::vector<llama_kpool_tok_desc> td(1);
        td[0].pos = (llama_pos) r.fill - 1;
        td[0].seqs = { 0 };

        const int reps = 10;

        // from scratch: the scan + build_layout + mark_new the engine ran every token
        llama_kpool_state st;
        double t_scratch = 0;
        for (int i = 0; i < reps; ++i) {
            auto t0 = clk::now();
            std::vector<llama_kpool_cell_desc> cells = kv.occupied();
            st = llama_kpool_build_layout(cells, kpool, 1);
            llama_kpool_mark_new(st, td, kpool, false);
            t_scratch += ms(t0, clk::now());
        }

        // append-only: one more cell onto the carried layout
        double t_append = 0;
        for (int i = 0; i < reps; ++i) {
            llama_kpool_state s2 = st;      // not timed: the engine carries this, it does not copy it
            std::vector<llama_kpool_cell_desc> added(1);
            added[0].pos  = (llama_pos) (r.fill + i);
            added[0].cell = r.fill + i;
            added[0].seqs = { 0 };
            std::vector<uint32_t> before(s2.seqs.size());
            for (size_t s = 0; s < s2.seqs.size(); ++s) {
                before[s] = (uint32_t) s2.seqs[s].pools.size();
            }
            auto t0 = clk::now();
            const bool ok = llama_kpool_append_layout(s2, added, kpool);
            llama_kpool_mark_new_appended(s2, before);
            t_append += ms(t0, clk::now());
            if (!ok) {
                printf("   (append refused at fill %u)\n", r.fill);
            }
        }

        const double a = t_scratch/reps;
        const double b = t_append/reps;
        printf("   %10u %10u %11.3f ms %11.3f ms %9.1fx\n", r.size, r.fill, a, b, b > 0 ? a/b : 0.0);
    }
    printf("   (single-threaded host time per ubatch, i.e. per decoded token)\n");
}

// ---------------------------------------------------------------------------------------
// 2. the cheap block top-k == the full descending sort
// ---------------------------------------------------------------------------------------

// the reference: a stable descending sort, which is what ggml_top_k (= argsort DESC, CUB radix
// above 1024 columns) produces -- equal scores keep ascending block index
static std::vector<int32_t> topk_reference(const std::vector<float> & s, int n_top) {
    std::vector<int32_t> ord((int32_t) s.size());
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int32_t a, int32_t b) { return s[a] > s[b]; });
    ord.resize(std::min<size_t>(ord.size(), (size_t) n_top));
    return ord;
}

static void test_topk() {
    printf("\n-- 2. the cheap block top-k ------------------------------------------------\n");

    std::mt19937 rng(20260909);

    struct scen { const char * name; int n_pool; int n_top; int mode; };
    const scen scens[] = {
        { "random scores, 21632 blocks, top 512",        21632, 512, 0 },
        { "random scores, tiny grid",                       64,   4, 0 },
        { "half the grid is -inf padding",               21632, 512, 1 },
        { "heavy ties (scores quantised to 8 values)",    4096, 512, 2 },
        { "every score identical",                        4096, 512, 3 },
        { "relu output: most scores exactly 0",           8192, 512, 4 },
        { "n_top >= n_pool",                               300, 512, 0 },
        { "one visible block only",                       1024, 512, 5 },
    };

    for (const auto & sc : scens) {
        std::vector<float> s((size_t) sc.n_pool);
        std::uniform_real_distribution<float> d(0.0f, 4.0f);
        for (int i = 0; i < sc.n_pool; ++i) {
            switch (sc.mode) {
                case 0: s[i] = d(rng); break;
                case 1: s[i] = (i % 2) ? -INFINITY : d(rng); break;
                case 2: s[i] = (float) (rng() % 8); break;
                case 3: s[i] = 1.5f; break;
                case 4: s[i] = (rng() % 4) ? 0.0f : d(rng); break;
                case 5: s[i] = i == 17 ? 3.0f : -INFINITY; break;
            }
        }

        const std::vector<int32_t> ref = topk_reference(s, sc.n_top);
        std::vector<int32_t> got((size_t) std::min(sc.n_top, sc.n_pool));
        ggml_qsa_topk_row_f32(s.data(), sc.n_pool, sc.n_top, got.data());

        const bool same_set   = [&]{
            std::vector<int32_t> a = ref, b = got;
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            return a == b;
        }();
        const bool same_order = ref == got;

        char buf[160];
        snprintf(buf, sizeof(buf), "%s: same SET", sc.name);
        check(same_set, buf);
        snprintf(buf, sizeof(buf), "%s: same ORDER (ties by lowest block index)", sc.name);
        check(same_order, buf);
    }

    // and it must be cheaper than the sort it replaces, at the shape that matters
    {
        const int n_pool = 21632, n_top = 512;
        std::vector<float> s((size_t) n_pool);
        std::uniform_real_distribution<float> d(0.0f, 4.0f);
        for (auto & v : s) { v = d(rng); }
        std::vector<int32_t> out((size_t) n_top);

        const int reps = 200;
        auto t0 = clk::now();
        for (int i = 0; i < reps; ++i) { (void) topk_reference(s, n_top); }
        const double t_sort = ms(t0, clk::now())/reps;
        t0 = clk::now();
        for (int i = 0; i < reps; ++i) { ggml_qsa_topk_row_f32(s.data(), n_pool, n_top, out.data()); }
        const double t_sel = ms(t0, clk::now())/reps;
        printf("   n_pool=%d n_top=%d: full sort %.3f ms, partial selection %.3f ms (%.1fx)\n",
               n_pool, n_top, t_sort, t_sel, t_sel > 0 ? t_sort/t_sel : 0.0);
        check(t_sel < t_sort, "the partial selection is cheaper than the full sort (CPU)");
    }
}

// ---------------------------------------------------------------------------------------
// 3. the selection-recall probe, and what a REUSED selection costs
// ---------------------------------------------------------------------------------------
//
// recall(t) = sum of the dense softmax weights of the SELECTED cells, over the sum of all of
// them. 1.0 means the selection lost nothing; the floor is roughly n_sel/n_kv, which is what a
// selection uncorrelated with the attention scores would get, so the probe is only meaningful
// on a layer whose attention is actually peaked -- a near-uniform softmax gives every selection
// the same score and measures nothing. The fixture below therefore builds ONE content vector
// per cell and derives both the indexer key and the attention key from it, which is the
// property the architecture's indexer is trained to have.

struct recall_fixture {
    uint32_t n_cells, n_ei, n_ih, r, top_k, D;
    std::vector<float> kidx;   // [n_ei, n_cells] raw indexer keys
    std::vector<float> k, v;   // [D, n_cells]    attention keys / values
};

static recall_fixture make_recall_fixture(uint32_t n_cells, uint32_t seed) {
    recall_fixture f;
    f.n_cells = n_cells; f.n_ei = 32; f.n_ih = 4; f.r = 4; f.top_k = 128; f.D = 32;
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    f.kidx.resize((size_t) f.n_ei*n_cells);
    f.k.resize((size_t) f.D*n_cells);
    f.v.resize((size_t) f.D*n_cells);
    for (uint32_t c = 0; c < n_cells; ++c) {
        // one content vector per cell; the indexer key IS the content, the attention key is the
        // content plus its own noise. A block that scores high therefore usually attends high,
        // which is the only regime in which recall is a meaningful number.
        for (uint32_t d = 0; d < f.n_ei; ++d) {
            const float x = g(rng);
            f.kidx[(size_t) c*f.n_ei + d] = x;
            f.k   [(size_t) c*f.D   + d] = x + 0.25f*g(rng);
        }
        for (uint32_t d = 0; d < f.D; ++d) {
            f.v[(size_t) c*f.D + d] = g(rng);
        }
    }
    return f;
}

// the architecture's block score: mean-pooled raw key, rms-normed, relu per head, summed
static std::vector<float> recall_block_scores(const recall_fixture & f, uint32_t upto,
                                              const std::vector<float> & q_idx) {
    const uint32_t n_blk = upto/f.r;
    std::vector<float> bs(n_blk);
    std::vector<float> pooled(f.n_ei);
    for (uint32_t b = 0; b < n_blk; ++b) {
        std::fill(pooled.begin(), pooled.end(), 0.0f);
        for (uint32_t m = 0; m < f.r; ++m) {
            const uint32_t c = b*f.r + m;
            for (uint32_t d = 0; d < f.n_ei; ++d) {
                pooled[d] += f.kidx[(size_t) c*f.n_ei + d];
            }
        }
        double ss = 0.0;
        for (uint32_t d = 0; d < f.n_ei; ++d) { pooled[d] /= (float) f.r; ss += (double) pooled[d]*pooled[d]; }
        const float sc = 1.0f/std::sqrt((float) (ss/f.n_ei) + 1e-6f);
        float s = 0.0f;
        for (uint32_t h = 0; h < f.n_ih; ++h) {
            float dot = 0.0f;
            for (uint32_t d = 0; d < f.n_ei; ++d) {
                dot += pooled[d]*sc*q_idx[(size_t) h*f.n_ei + d];
            }
            s += dot > 0.0f ? dot : 0.0f;
        }
        bs[b] = s;
    }
    return bs;
}

static std::vector<int32_t> recall_select(const recall_fixture & f, uint32_t upto,
                                          const std::vector<float> & q_idx) {
    const std::vector<float> bs = recall_block_scores(f, upto, q_idx);
    const uint32_t n_top = std::min<uint32_t>((uint32_t) bs.size(), f.top_k/f.r);
    std::vector<int32_t> ord((int32_t) bs.size());
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int32_t a, int32_t b) { return bs[a] > bs[b]; });
    std::vector<int32_t> sel;
    for (uint32_t i = 0; i < n_top; ++i) {
        for (uint32_t m = 0; m < f.r; ++m) {
            sel.push_back((int32_t) (ord[i]*(int32_t) f.r + (int32_t) m));
        }
    }
    for (uint32_t c = (uint32_t) bs.size()*f.r; c < upto; ++c) {   // the incomplete tail
        sel.push_back((int32_t) c);
    }
    return sel;
}

// fraction of the dense softmax mass over [0, upto) that `sel` carries
static double recall_of(const recall_fixture & f, uint32_t upto,
                        const std::vector<float> & q, const std::vector<int32_t> & sel) {
    // A GAIN on the logits, because recall is only a meaningful number when the dense attention
    // is actually peaked: with a near-uniform softmax every selection of the same size scores
    // the same n_sel/n_kv and the probe measures nothing. Real attention at depth is peaked;
    // this is the synthetic stand-in for that.
    const float scale = 4.0f/std::sqrt((float) f.D);
    std::vector<double> w(upto);
    double m = -1e30, tot = 0.0;
    for (uint32_t c = 0; c < upto; ++c) {
        float s = 0.0f;
        for (uint32_t d = 0; d < f.D; ++d) {
            s += f.k[(size_t) c*f.D + d]*q[d];
        }
        w[c] = (double) s*scale;
        m = std::max(m, w[c]);
    }
    for (uint32_t c = 0; c < upto; ++c) { w[c] = std::exp(w[c] - m); tot += w[c]; }
    double got = 0.0;
    std::vector<uint8_t> seen(upto, 0);
    for (int32_t c : sel) {
        if (c >= 0 && (uint32_t) c < upto && !seen[c]) { seen[c] = 1; got += w[c]; }
    }
    return tot > 0.0 ? got/tot : 0.0;
}

static void test_recall_and_staleness() {
    printf("\n-- 3. selection recall, fresh vs reused ------------------------------------\n");

    const uint32_t n_cells = 8192;
    recall_fixture f = make_recall_fixture(n_cells, 4242);

    std::mt19937 rng(7);
    std::normal_distribution<float> g(0.0f, 1.0f);

    const uint32_t base = 4096;
    const int steps = 96;

    // the query, and its indexer projection, both derived from one content vector so the
    // indexer predicts the attention -- and both drifting a little per decoded token, which is
    // what makes a selection go stale at all
    std::vector<float> qc((size_t) f.n_ei);
    for (auto & x : qc) { x = g(rng); }

    auto make_q = [&](const std::vector<float> & c) {
        std::vector<float> q((size_t) f.D);
        for (uint32_t d = 0; d < f.D; ++d) { q[d] = c[d % f.n_ei]; }
        return q;
    };
    auto make_qidx = [&](const std::vector<float> & c) {
        std::vector<float> qi((size_t) f.n_ei*f.n_ih);
        for (uint32_t h = 0; h < f.n_ih; ++h) {
            for (uint32_t d = 0; d < f.n_ei; ++d) {
                qi[(size_t) h*f.n_ei + d] = c[d];
            }
        }
        return qi;
    };

    // the floor: what an arbitrary selection of the same size would carry
    {
        std::vector<int32_t> arb;
        for (uint32_t c = 0; c < f.top_k; ++c) { arb.push_back((int32_t) c); }
        const double fl = recall_of(f, base, make_q(qc), arb);
        const double fr = recall_of(f, base, make_q(qc), recall_select(f, base, make_qidx(qc)));
        printf("   at %u cells, %u selected: arbitrary %.4f, indexer %.4f\n",
               base, f.top_k, fl, fr);
        check(fr > 3.0*fl, "the indexer's selection carries far more mass than an arbitrary one");
    }

    printf("   %6s %14s %14s %14s\n", "N", "fresh recall", "stale recall", "loss");
    double loss1 = 0.0;
    for (int N : { 1, 2, 4, 8, 16, 32 }) {
        // the SAME drift trajectory for every N, or the columns are not comparable
        std::mt19937 drift(31337);
        std::normal_distribution<float> dg(0.0f, 1.0f);
        std::vector<float> c = qc;
        double sum_fresh = 0.0, sum_stale = 0.0;
        std::vector<int32_t> held;
        uint32_t held_upto = base;
        for (int t = 0; t < steps; ++t) {
            const uint32_t upto = base + (uint32_t) t;
            if (t % N == 0) {
                held = recall_select(f, upto, make_qidx(c));
                held_upto = upto;
            }
            // a reused selection still gets every cell written since it was taken: the tail is
            // appended unconditionally and a token can always attend to itself
            std::vector<int32_t> stale = held;
            for (uint32_t cc = held_upto; cc < upto; ++cc) { stale.push_back((int32_t) cc); }

            sum_fresh += recall_of(f, upto, make_q(c), recall_select(f, upto, make_qidx(c)));
            sum_stale += recall_of(f, upto, make_q(c), stale);

            for (auto & x : c) { x += 0.08f*dg(drift); }   // the hidden state drifts per token
        }
        const double fr = sum_fresh/steps, sl = sum_stale/steps;
        printf("   %6d %13.4f  %13.4f  %13.4f\n", N, fr, sl, fr - sl);
        if (N == 1) {
            loss1 = fr - sl;
            check(std::fabs(loss1) < 1e-12, "N=1 reuses nothing: stale recall == fresh recall");
        }
    }
    printf("   (a synthetic layer: it pins the PROBE and the shape of the loss. The N that a\n"
           "    reused selection can afford is a property of the real model and comes from\n"
           "    a live recall A/B against the served model.)\n");
}

// ---------------------------------------------------------------------------------------

int main() {
    test_grid_append();
    bench_grid();
    test_topk();
    test_recall_and_staleness();

    printf("\n%s\n", g_fail == 0 ? "ALL OK" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
}
