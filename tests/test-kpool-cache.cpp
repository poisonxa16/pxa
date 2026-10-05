// PXA_GLM5NEXT: host-side unit test for the pooled ("k-pool") indexer cache.
//
// GLM-5.3-Flash's DSA indexer scores POOLS of kpool consecutive tokens, so every ubatch the
// host must hand the graph a pool grid derived from POSITIONS, not from cell order. That
// derivation (src/llama-kv-cache-kpool.cpp) is the single largest off-by-one surface of the
// whole architecture and it is exercised here with synthetic token streams: no model, no
// weights, no GPU, no ggml graph.
//
// The expectations below are computed independently inside each case (a pool of a
// single-sequence run starting at pos_min is analytically known: pools are
// [pos_min + 4i, pos_min + 4i + 3], the pooled key lives in the LAST member's cell, and the
// tail of a token at pos p is the newest (p - pos_min + 1) % 4 tokens). They are NOT read back
// out of the same helpers under test.
//
// The cache model here mirrors the tree's flat POD ring: llama_kv_cache_find_slot() hands out
// a CONTIGUOUS run of `n_tokens` cells starting at `head`, and `head` wraps.

#include "llama-kv-cache-kpool.h"

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static int g_fail = 0;
static const char * g_case = "";
static void begin(const char * name) { g_case = name; printf("  %s\n", name); fflush(stdout); }

static void fail(const char * fmt, ...) {
    fprintf(stderr, "  FAIL [%s]: ", g_case);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    ++g_fail;
}

#define CHECK(cond, ...) do { if (!(cond)) { fail(__VA_ARGS__); } } while (0)

#define CHECK_EQ_I(a, b, fmt, ...) do { \
    const long long _a = (long long)(a), _b = (long long)(b); \
    if (_a != _b) { fail(fmt " -- %s = %lld, expected %lld", ##__VA_ARGS__, #a, _a, _b); } \
} while (0)

//
// a toy ring cache
//

struct toy_cache {
    uint32_t size = 0;
    uint32_t head = 0;
    std::vector<llama_pos>                 pos;
    std::vector<std::vector<llama_seq_id>> seqs;

    explicit toy_cache(uint32_t size) : size(size), pos(size, -1), seqs(size) {}

    // append n tokens of sequence s at positions [p0, p0+n), contiguously from head
    std::vector<uint32_t> append(llama_seq_id s, llama_pos p0, uint32_t n) {
        std::vector<uint32_t> cells;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t c = (head + i) % size;
            pos[c] = p0 + (llama_pos) i;
            seqs[c].assign(1, s);
            cells.push_back(c);
        }
        head = (head + n) % size;
        return cells;
    }

    std::vector<llama_kpool_cell_desc> occupied() const {
        std::vector<llama_kpool_cell_desc> out;
        for (uint32_t i = 0; i < size; ++i) {
            if (seqs[i].empty()) {
                continue;
            }
            llama_kpool_cell_desc d;
            d.pos  = pos[i];
            d.cell = i;
            d.seqs = seqs[i];
            out.push_back(d);
        }
        return out;
    }
};

// everything one ubatch produces
struct filled {
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

static filled run_ubatch(const toy_cache & cache,
                         const std::vector<llama_kpool_tok_desc> & toks,
                         uint32_t kpool, uint32_t n_seq_max,
                         uint32_t indexer_top_k, bool select_tail,
                         uint32_t n_kv, bool stale, bool force_gather = false) {
    filled f;
    f.st = llama_kpool_build_layout(cache.occupied(), kpool, n_seq_max);
    llama_kpool_mark_new(f.st, toks, kpool, stale);

    f.d.kpool    = kpool;
    f.d.n_pool   = llama_kpool_pad(f.st.n_pool_real);
    f.d.n_tokens = (uint32_t) toks.size();
    f.d.n_kv     = n_kv;
    f.d.n_new    = f.st.n_new;
    f.d.n_new_g  = std::max(f.d.n_new, llama_kpool_n_new_floor(f.d.n_pool, f.d.n_tokens, kpool));
    f.d.sink     = n_kv;   // the engine uses the cache size; any row past the cells will do
    f.d.n_top    = llama_kpool_n_top(f.d.n_pool, indexer_top_k, kpool);
    f.d.n_sel    = llama_kpool_n_sel(f.d.n_top, kpool, select_tail);
    f.d.gather   = force_gather || (f.d.n_tokens <= 16 && f.d.n_kv > f.d.n_sel);

    f.pool_cells.assign(f.d.n_pool, -12345);
    f.pool_idxs.assign((size_t) f.d.n_pool*kpool, -12345);
    f.pool_mask.assign((size_t) f.d.n_pool*f.d.n_tokens, 12345.0f);
    f.tail_idxs.assign((size_t) f.d.n_tokens*(kpool - 1), -12345);

    llama_kpool_bufs b;
    b.pool_cells = f.pool_cells.data();
    b.pool_idxs  = f.pool_idxs.data();
    b.pool_mask  = f.pool_mask.data();
    b.tail_idxs  = f.tail_idxs.data();

    if (f.d.gather) {
        f.gather_mask.assign((size_t) f.d.n_sel*f.d.n_tokens, 12345.0f);
        b.gather_mask = f.gather_mask.data();
    }
    // the new-pool block is always built, at the fixed width n_new_g
    f.new_pool_idxs.assign((size_t) f.d.n_new_g*kpool, -12345);
    b.new_pool_idxs = f.new_pool_idxs.data();
    if (f.st.cache_safe) {
        f.new_pool_rep.assign(f.d.n_new_g, -12345);
        b.new_pool_rep = f.new_pool_rep.data();
    }

    llama_kpool_fill(f.st, f.d, toks, b);
    return f;
}

static std::vector<llama_kpool_tok_desc> toks_of(llama_seq_id s, llama_pos p0, uint32_t n) {
    std::vector<llama_kpool_tok_desc> out(n);
    for (uint32_t i = 0; i < n; ++i) {
        out[i].pos = p0 + (llama_pos) i;
        out[i].seqs.assign(1, s);
    }
    return out;
}

//
// cases
//

// One prefill ubatch. The grid, the pooled-key rows, the visibility mask, the tail and the
// completed-pool set are all analytically known for a contiguous single-sequence run.
static void case_prefill_one_ubatch() {
    begin("prefill/one-ubatch");
    const uint32_t kpool = 4, n_tok = 10, n_kv = 16;

    toy_cache cache(64);
    cache.append(0, 0, n_tok);

    auto toks = toks_of(0, 0, n_tok);
    auto f = run_ubatch(cache, toks, kpool, 4, /*top_k*/ 2048, /*select_tail*/ true, n_kv, /*stale*/ true);

    // 10 tokens, kpool 4 -> 2 complete pools (0-3, 4-7); 8 and 9 are the tail.
    CHECK_EQ_I(f.st.n_pool_real, 2, "complete pools");
    CHECK_EQ_I(f.st.n_new, 2, "every pool is completed by this ubatch");
    CHECK(f.st.cache_safe, "no shared cells -> cache_safe");
    CHECK_EQ_I(f.d.n_pool, 64, "padded pool count");
    CHECK(!f.d.gather, "a 10-token ubatch with n_kv 16 <= n_sel is the scatter path");

    // the pooled key of pool i lives in the cell of position 4i+3
    for (uint32_t ip = 0; ip < 2; ++ip) {
        CHECK_EQ_I(f.pool_cells[ip], 4*ip + 3, "pool %u representative cell", ip);
        for (uint32_t k = 0; k < kpool; ++k) {
            CHECK_EQ_I(f.pool_idxs[ip*kpool + k], 4*ip + k, "pool %u member %u", ip, k);
            CHECK_EQ_I(f.new_pool_idxs[ip*kpool + k], 4*ip + k, "new pool %u member %u", ip, k);
        }
        CHECK_EQ_I(f.new_pool_rep[ip], 4*ip + 3, "new pool %u write-back row", ip);
    }

    // padded pools: representative on a real row, members on the scatter sentinel (n_kv)
    for (uint32_t ip = 2; ip < f.d.n_pool; ++ip) {
        CHECK_EQ_I(f.pool_cells[ip], 0, "padded pool %u representative is the dummy cell", ip);
        for (uint32_t k = 0; k < kpool; ++k) {
            CHECK_EQ_I(f.pool_idxs[ip*kpool + k], (int32_t) n_kv, "padded pool %u member %u is the sentinel", ip, k);
        }
    }

    // visibility: token at pos p sees pool i iff 4i+3 <= p
    for (uint32_t i = 0; i < n_tok; ++i) {
        for (uint32_t ip = 0; ip < f.d.n_pool; ++ip) {
            const bool want_vis = ip < 2 && (int) (4*ip + 3) <= (int) i;
            const float m = f.pool_mask[(size_t) i*f.d.n_pool + ip];
            if (want_vis) {
                CHECK(m == 0.0f, "token %u should see pool %u (mask %g)", i, ip, m);
            } else {
                CHECK(std::isinf(m) && m < 0, "token %u must not see pool %u (mask %g)", i, ip, m);
            }
        }
    }

    // tail of the token at pos p: the newest (p+1) % 4 tokens, newest first, then sentinel
    for (uint32_t i = 0; i < n_tok; ++i) {
        const uint32_t n_tail = (i + 1) % kpool;
        for (uint32_t k = 0; k < kpool - 1; ++k) {
            const int32_t got = f.tail_idxs[(size_t) i*(kpool - 1) + k];
            const int32_t want = k < n_tail ? (int32_t) (i - k) : (int32_t) n_kv;
            CHECK_EQ_I(got, want, "token %u tail slot %u", i, k);
        }
    }
}

// The same 10 tokens fed one at a time must end in exactly the layout the single ubatch gave,
// and must complete a pool on exactly the 4th, 8th, ... token and on no other.
static void case_prefill_vs_stepwise() {
    begin("prefill-vs-stepwise");
    const uint32_t kpool = 4, n_tok = 10, n_kv = 16;

    toy_cache batched(64);
    batched.append(0, 0, n_tok);
    auto fb = run_ubatch(batched, toks_of(0, 0, n_tok), kpool, 4, 2048, true, n_kv, true);

    toy_cache step(64);
    filled fs;
    for (uint32_t i = 0; i < n_tok; ++i) {
        step.append(0, (llama_pos) i, 1);
        // only the very first ubatch of a batch is "stale"; the rest carry their cached pools
        fs = run_ubatch(step, toks_of(0, (llama_pos) i, 1), kpool, 4, 2048, true, n_kv,
                        /*stale*/ i == 0);

        // a pool completes exactly on positions 3, 7, ...
        const uint32_t want_new = ((i + 1) % kpool == 0) ? 1u : 0u;
        CHECK_EQ_I(fs.st.n_new, want_new, "step %u completes %u pool(s)", i, want_new);
        CHECK_EQ_I(fs.st.n_pool_real, (i + 1)/kpool, "step %u pool count", i);
        if (want_new) {
            CHECK_EQ_I(fs.new_pool_rep[0], (int32_t) i, "step %u writes the pooled key into its own cell", i);
            for (uint32_t k = 0; k < kpool; ++k) {
                CHECK_EQ_I(fs.new_pool_idxs[k], (int32_t) (i - kpool + 1 + k), "step %u new pool member %u", i, k);
            }
        }
    }

    // the grid the two paths arrive at must be identical
    CHECK_EQ_I(fs.st.n_pool_real, fb.st.n_pool_real, "final pool count");
    CHECK_EQ_I(fs.d.n_pool, fb.d.n_pool, "final padded pool count");
    // only the REAL pools: the padded entries deliberately carry each ubatch's own dummy cell
    for (uint32_t ip = 0; ip < fb.st.n_pool_real; ++ip) {
        CHECK_EQ_I(fs.pool_cells[ip], fb.pool_cells[ip], "final pool_cells[%u]", ip);
        for (uint32_t k = 0; k < kpool; ++k) {
            CHECK_EQ_I(fs.pool_idxs[ip*kpool + k], fb.pool_idxs[ip*kpool + k], "final pool_idxs[%u][%u]", ip, k);
        }
    }
    // ... and so must the last token's mask row and tail
    const uint32_t last = n_tok - 1;
    for (uint32_t ip = 0; ip < fb.d.n_pool; ++ip) {
        const float a = fs.pool_mask[ip];
        const float b = fb.pool_mask[(size_t) last*fb.d.n_pool + ip];
        CHECK((a == b) || (std::isinf(a) && std::isinf(b) && a < 0 && b < 0),
              "final pool_mask[%u]: stepwise %g vs batched %g", ip, a, b);
    }
    for (uint32_t k = 0; k < kpool - 1; ++k) {
        CHECK_EQ_I(fs.tail_idxs[k], fb.tail_idxs[(size_t) last*(kpool - 1) + k], "final tail slot %u", k);
    }
}

// The grid is derived from positions, so it must survive the ring wrapping: the same
// sequence, laid out with its cells in a rotated order, yields the same pools.
static void case_ring_wrap() {
    begin("ring-wrap");
    const uint32_t kpool = 4, n_kv = 16;

    toy_cache cache(12);
    cache.head = 10;              // the run wraps: cells 10,11,0,1,2,3,4,5
    cache.append(0, 0, 8);

    auto f = run_ubatch(cache, toks_of(0, 0, 8), kpool, 4, 2048, true, n_kv, true);

    CHECK_EQ_I(f.st.n_pool_real, 2, "pools across the wrap");
    const int32_t want_cells[8] = { 10, 11, 0, 1, 2, 3, 4, 5 };
    for (uint32_t ip = 0; ip < 2; ++ip) {
        CHECK_EQ_I(f.pool_cells[ip], want_cells[4*ip + 3], "wrapped pool %u representative", ip);
        for (uint32_t k = 0; k < kpool; ++k) {
            CHECK_EQ_I(f.pool_idxs[ip*kpool + k], want_cells[4*ip + k], "wrapped pool %u member %u", ip, k);
        }
    }
}

// A hole in the position line (an evicted / removed token) must NOT be pooled over, and must
// not shift the grid off the sequence's first position.
static void case_position_gap() {
    begin("position-gap");
    const uint32_t kpool = 4, n_kv = 16;

    toy_cache cache(64);
    cache.append(0, 0, 12);
    // drop position 5 (cell 5): pools [0..3] and [8..11] survive, [4..7] does not
    cache.pos[5] = -1;
    cache.seqs[5].clear();

    auto f = run_ubatch(cache, toks_of(0, 0, 12), kpool, 4, 2048, true, n_kv, true);

    CHECK_EQ_I(f.st.n_pool_real, 2, "a gap kills exactly the pool containing it");
    CHECK_EQ_I(f.pool_cells[0], 3,  "first pool survives");
    CHECK_EQ_I(f.pool_cells[1], 11, "the pool after the gap re-aligns on the pos_min grid");
    for (uint32_t k = 0; k < kpool; ++k) {
        CHECK_EQ_I(f.pool_idxs[kpool + k], (int32_t) (8 + k), "post-gap pool member %u", k);
    }
}

// Two sequences in one ubatch: pools are laid out per sequence and a token must only ever see
// its own sequence's pools.
static void case_two_sequences() {
    begin("two-sequences");
    const uint32_t kpool = 4, n_kv = 32;

    toy_cache cache(64);
    cache.append(0, 0, 8);    // cells 0..7,  seq 0
    cache.append(1, 0, 4);    // cells 8..11, seq 1

    std::vector<llama_kpool_tok_desc> toks(2);
    toks[0].pos = 8; toks[0].seqs.assign(1, 0);
    toks[1].pos = 4; toks[1].seqs.assign(1, 1);
    cache.append(0, 8, 1);    // cell 12
    cache.append(1, 4, 1);    // cell 13

    auto f = run_ubatch(cache, toks, kpool, 4, 2048, true, n_kv, true);

    CHECK_EQ_I(f.st.n_pool_real, 3, "2 pools for seq 0 + 1 for seq 1");
    CHECK(f.st.cache_safe, "distinct cells -> cache_safe");
    // seq 0's pools come first (pools are laid out in seq_id order)
    CHECK_EQ_I(f.pool_cells[0], 3,  "seq 0 pool 0");
    CHECK_EQ_I(f.pool_cells[1], 7,  "seq 0 pool 1");
    CHECK_EQ_I(f.pool_cells[2], 11, "seq 1 pool 0");

    // token 0 is seq 0 at pos 8: sees both of seq 0's pools, neither of seq 1's
    CHECK(f.pool_mask[0] == 0.0f, "seq0 token sees seq0 pool 0");
    CHECK(f.pool_mask[1] == 0.0f, "seq0 token sees seq0 pool 1");
    CHECK(std::isinf(f.pool_mask[2]), "seq0 token must not see seq1's pool");
    // token 1 is seq 1 at pos 4: sees only seq 1's pool
    const float * row1 = f.pool_mask.data() + f.d.n_pool;
    CHECK(std::isinf(row1[0]) && std::isinf(row1[1]), "seq1 token must not see seq0's pools");
    CHECK(row1[2] == 0.0f, "seq1 token sees its own pool");
}

// A cell shared by two sequences (seq_cp) makes the sequence-relative pooled key uncacheable:
// cache_safe must go false and every pool must be marked new.
static void case_shared_cells() {
    begin("shared-cells");
    const uint32_t kpool = 4, n_kv = 16;

    toy_cache cache(64);
    cache.append(0, 0, 8);
    for (uint32_t c = 0; c < 8; ++c) {
        cache.seqs[c].push_back(1);   // seq_cp 0 -> 1
    }
    cache.append(0, 8, 1);            // the decode token, seq 0 only, cell 8

    auto f = run_ubatch(cache, toks_of(0, 8, 1), kpool, 4, 2048, true, n_kv, /*stale*/ false);

    CHECK(!f.st.cache_safe, "shared cells -> !cache_safe");
    CHECK_EQ_I(f.st.n_pool_real, 4, "both sequences see 2 pools each");
    CHECK_EQ_I(f.st.n_new, 4, "!cache_safe re-pools everything");
    CHECK(f.new_pool_rep.empty(), "!cache_safe writes no pooled keys back");
}

// The decode gather path: padding points at a REAL row and is masked by gather_mask instead.
static void case_gather_path() {
    begin("gather");
    const uint32_t kpool = 4, n_kv = 4096;

    toy_cache cache(8192);
    cache.append(0, 0, 20);   // 5 complete pools + no tail (20 % 4 == 0)
    cache.append(0, 20, 1);   // the decode token, cell 20
    auto toks = toks_of(0, 20, 1);

    // top_k 16 -> at most 4 pools per token, so the selection is genuinely narrower than n_kv
    auto f = run_ubatch(cache, toks, kpool, 4, /*top_k*/ 16, /*select_tail*/ true, n_kv,
                        /*stale*/ false);

    CHECK(f.d.gather, "1 token and n_kv > n_sel -> gather");
    CHECK_EQ_I(f.d.n_top, 4, "top_k 16 / kpool 4");
    CHECK_EQ_I(f.d.n_sel, 4*4 + 3, "4 pools of 4 cells plus the 3 tail slots");
    CHECK_EQ_I(f.st.n_pool_real, 5, "5 complete pools");

    // gather padding never points outside the cache: the sentinel is the ubatch's own cell
    for (uint32_t ip = f.st.n_pool_real; ip < f.d.n_pool; ++ip) {
        for (uint32_t k = 0; k < kpool; ++k) {
            CHECK_EQ_I(f.pool_idxs[ip*kpool + k], 20, "gather padding is the dummy cell");
        }
    }

    // the token at pos 20 sees all 5 pools, but only 4 fit -> all 4 ranked slots are finite
    for (uint32_t j = 0; j < f.d.n_top*kpool; ++j) {
        CHECK(f.gather_mask[j] == 0.0f, "gather slot %u should be live", j);
    }
    // pos 20, pos_min 0 -> (20 - 0 + 1) % 4 == 1 tail token (itself)
    CHECK(f.gather_mask[f.d.n_top*kpool + 0] == 0.0f, "the token's own tail slot is live");
    CHECK(std::isinf(f.gather_mask[f.d.n_top*kpool + 1]), "unused tail slot 1 is masked");
    CHECK(std::isinf(f.gather_mask[f.d.n_top*kpool + 2]), "unused tail slot 2 is masked");
    CHECK_EQ_I(f.tail_idxs[0], 20, "tail slot 0 is the token's own cell");
    CHECK_EQ_I(f.tail_idxs[1], 20, "unused tail slot points at the dummy cell, not out of range");
}

// A short context, where fewer pools exist than the selection can hold: the mask must leave
// exactly the visible ones finite so the ranked slots line up with gather_mask.
static void case_gather_partial_visibility() {
    begin("gather/partial-visibility");
    const uint32_t kpool = 4, n_kv = 4096;

    toy_cache cache(8192);
    cache.append(0, 0, 9);    // 2 complete pools, 1 tail token
    cache.append(0, 9, 1);    // decode token at pos 9, cell 9
    auto f = run_ubatch(cache, toks_of(0, 9, 1), kpool, 4, /*top_k*/ 16, true, n_kv, false);

    CHECK(f.d.gather, "gather expected");
    CHECK_EQ_I(f.st.n_pool_real, 2, "2 complete pools");
    // 2 visible pools out of 4 ranked slots: slots 0..7 live, 8..15 masked
    for (uint32_t j = 0; j < 2*kpool; ++j) {
        CHECK(f.gather_mask[j] == 0.0f, "visible gather slot %u", j);
    }
    for (uint32_t j = 2*kpool; j < f.d.n_top*kpool; ++j) {
        CHECK(std::isinf(f.gather_mask[j]), "empty gather slot %u must be masked", j);
    }
    // pos 9, pos_min 0 -> (9+1) % 4 == 2 tail tokens: cells 9 and 8
    CHECK_EQ_I(f.tail_idxs[0], 9, "tail slot 0");
    CHECK_EQ_I(f.tail_idxs[1], 8, "tail slot 1");
    CHECK(f.gather_mask[f.d.n_top*kpool + 0] == 0.0f, "tail slot 0 live");
    CHECK(f.gather_mask[f.d.n_top*kpool + 1] == 0.0f, "tail slot 1 live");
    CHECK(std::isinf(f.gather_mask[f.d.n_top*kpool + 2]), "tail slot 2 masked");
}

// A sequence whose first cached position is not 0 (a trimmed / shifted context): the grid is
// anchored on pos_min, not on absolute position 0.
static void case_pos_min_offset() {
    begin("pos-min-offset");
    const uint32_t kpool = 4, n_kv = 32;

    toy_cache cache(64);
    cache.append(0, 6, 9);    // positions 6..14 in cells 0..8

    auto f = run_ubatch(cache, toks_of(0, 6, 9), kpool, 4, 2048, true, n_kv, true);

    // grid anchored at 6: pools [6..9] and [10..13]; 14 is the tail
    CHECK_EQ_I(f.st.n_pool_real, 2, "pools anchored on pos_min");
    CHECK_EQ_I(f.pool_cells[0], 3, "pool 0 ends at pos 9 = cell 3");
    CHECK_EQ_I(f.pool_cells[1], 7, "pool 1 ends at pos 13 = cell 7");

    // the token at pos 14: (14 - 6 + 1) % 4 == 1 tail token, itself (cell 8)
    const uint32_t last = 8;
    CHECK_EQ_I(f.tail_idxs[(size_t) last*(kpool - 1) + 0], 8, "tail slot 0");
    CHECK_EQ_I(f.tail_idxs[(size_t) last*(kpool - 1) + 1], (int32_t) n_kv, "unused tail slot 1");
}

// A stale grid (any seq_* edit) must re-pool everything from the still-valid key|gate rows.
static void case_stale_repools_all() {
    begin("stale-repools-all");
    const uint32_t kpool = 4, n_kv = 32;

    toy_cache cache(64);
    cache.append(0, 0, 16);
    cache.append(0, 16, 1);   // the decode token, cell 16

    auto fresh = run_ubatch(cache, toks_of(0, 16, 1), kpool, 4, 2048, true, n_kv, /*stale*/ false);
    CHECK_EQ_I(fresh.st.n_new, 0, "a decode token that completes no pool marks nothing new");

    auto stale = run_ubatch(cache, toks_of(0, 16, 1), kpool, 4, 2048, true, n_kv, /*stale*/ true);
    CHECK_EQ_I(stale.st.n_new, 4, "stale re-pools every pool");
    for (uint32_t ip = 0; ip < 4; ++ip) {
        CHECK_EQ_I(stale.new_pool_rep[ip], 4*ip + 3, "re-pooled pool %u writes back to its own row", ip);
    }
}

// The new-pool block is a FIXED width so the graph's node count does not move token to token.
// A decode step that completes NO pool must still fill n_new_g rows, and every padded row must
// read a real cache cell and write the sink -- never a row any pool reads back.
static void case_new_pool_padding() {
    begin("new-pool-padding-is-inert");
    const uint32_t kpool = 4, n_kv = 32;

    toy_cache cache(64);
    cache.append(0, 0, 16);       // four complete pools, cells 0..15
    cache.append(0, 16, 1);       // the decode token, cell 16: completes nothing

    auto f = run_ubatch(cache, toks_of(0, 16, 1), kpool, 4, 2048, true, n_kv, /*stale*/ false);

    CHECK_EQ_I(f.st.n_new,  0, "this decode token completes no pool");
    CHECK_EQ_I(f.d.n_new_g, 1, "the graph still builds one new-pool row");

    // dummy_cell is the ubatch's own first token's cell (cell 16 here)
    for (uint32_t k = 0; k < kpool; ++k) {
        CHECK_EQ_I(f.new_pool_idxs[k], 16, "padded member %u reads the ubatch's own cell", k);
    }
    CHECK_EQ_I(f.new_pool_rep[0], (int64_t) f.d.sink, "padded row writes the sink, not a pool row");

    // and the sink is not a row any pool reads back
    for (uint32_t ip = 0; ip < f.d.n_pool; ++ip) {
        CHECK(f.pool_cells[ip] != (int32_t) f.d.sink,
                   "no pool reads the sink row (pool %u)", ip);
    }

    // a step that DOES complete a pool needs no padding at all
    toy_cache c2(64);
    c2.append(0, 0, 20);
    auto g = run_ubatch(c2, toks_of(0, 19, 1), kpool, 4, 2048, true, n_kv, /*stale*/ false);
    CHECK_EQ_I(g.st.n_new,  1, "the 20th token completes the fifth pool");
    CHECK_EQ_I(g.d.n_new_g, 1, "and the graph width is the SAME as the step that completed none");
    CHECK_EQ_I(g.new_pool_rep[0], 19, "the real row writes back to its own last member");
}

// A prompt fed in ubatches of 3 (so pool boundaries fall INSIDE a ubatch, and one ubatch
// completes two pools) must reach the same grid as one big ubatch, and must mark every pool it
// touches — including a pool it only partly rewrote.
static void case_unaligned_ubatches() {
    begin("unaligned-ubatches");
    const uint32_t kpool = 4, n_tok = 12, n_kv = 32;

    toy_cache one(64);
    one.append(0, 0, n_tok);
    auto fb = run_ubatch(one, toks_of(0, 0, n_tok), kpool, 4, 2048, true, n_kv, true);

    toy_cache many(64);
    filled fm;
    uint32_t total_new = 0;
    for (uint32_t p = 0; p < n_tok; p += 3) {
        many.append(0, (llama_pos) p, 3);
        fm = run_ubatch(many, toks_of(0, (llama_pos) p, 3), kpool, 4, 2048, true, n_kv,
                        /*stale*/ p == 0);
        total_new += fm.st.n_new;
    }
    // ubatch [0,1,2]: no complete pool.      [3,4,5]: completes pool 0, touches pool 1's members.
    // ubatch [6,7,8]: completes pool 1, touches pool 2.  [9,10,11]: completes pool 2.
    CHECK_EQ_I(fm.st.n_pool_real, 3, "3 pools after 12 tokens");
    CHECK_EQ_I(fb.st.n_pool_real, 3, "same from one ubatch");
    CHECK(total_new >= 3, "every pool must be pooled at least once (got %u)", total_new);

    for (uint32_t ip = 0; ip < fb.st.n_pool_real; ++ip) {
        CHECK_EQ_I(fm.pool_cells[ip], fb.pool_cells[ip], "pool_cells[%u]", ip);
        for (uint32_t k = 0; k < kpool; ++k) {
            CHECK_EQ_I(fm.pool_idxs[ip*kpool + k], fb.pool_idxs[ip*kpool + k], "pool_idxs[%u][%u]", ip, k);
        }
    }
}

// A pool a ubatch only PARTLY rewrote must still be marked new: its pooled key mixes the old
// and the new members and is no longer whatever was cached.
static void case_partial_pool_is_new() {
    begin("partial-pool-is-new");
    const uint32_t kpool = 4, n_kv = 32;

    toy_cache cache(64);
    cache.append(0, 0, 8);
    // this ubatch wrote only positions 6 and 7 -> pool 1 (positions 4..7) is stale, pool 0 is not
    std::vector<llama_kpool_tok_desc> toks(2);
    toks[0].pos = 6; toks[0].seqs.assign(1, 0);
    toks[1].pos = 7; toks[1].seqs.assign(1, 0);

    auto f = run_ubatch(cache, toks, kpool, 4, 2048, true, n_kv, /*stale*/ false);

    CHECK_EQ_I(f.st.n_new, 1, "only the touched pool is re-pooled");
    CHECK_EQ_I(f.new_pool_rep[0], 7, "and it is pool 1");
    for (uint32_t k = 0; k < kpool; ++k) {
        CHECK_EQ_I(f.new_pool_idxs[k], (int32_t) (4 + k), "re-pooled member %u", k);
    }
}

// The padded pool count must always leave at least one spare entry, whatever the real count.
static void case_pad_leaves_a_spare() {
    begin("pad");
    for (uint32_t n : { 0u, 1u, 63u, 64u, 65u, 127u, 128u, 1000u }) {
        const uint32_t p = llama_kpool_pad(n);
        CHECK(p > n, "pad(%u) = %u must exceed the real count", n, p);
        CHECK(p % 64 == 0, "pad(%u) = %u must be a multiple of 64", n, p);
    }
}

int main() {
    printf("test-kpool-cache: GLM-5.3-Flash pooled indexer grid\n");

    case_prefill_one_ubatch();
    case_prefill_vs_stepwise();
    case_ring_wrap();
    case_position_gap();
    case_two_sequences();
    case_shared_cells();
    case_gather_path();
    case_gather_partial_visibility();
    case_pos_min_offset();
    case_stale_repools_all();
    case_new_pool_padding();
    case_unaligned_ubatches();
    case_partial_pool_is_new();
    case_pad_leaves_a_spare();

    if (g_fail) {
        printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    printf("ALL OK\n");
    return 0;
}
