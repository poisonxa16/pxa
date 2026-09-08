// DeepSeek-V4 memory module — IMPLEMENTATION.
//
// Adapted from llama.cpp `src/llama-kv-cache-dsv4.cpp` @ upstream commit db7d8b24b
// (PR #24162 plus the post-launch cache fixes 91d2fc387 / 7f575c39d / 13f2b28b0 /
// 024c46ae4, all of which are in that checkout's history). Copyright (c) 2023-2026
// The ggml authors. MIT.
//
// Upstream builds this on llama_memory_i / llama_memory_context_i / llama_kv_cache_iswa /
// llama_kv_cells / llama_io_{read,write}_i — none of which exist in this tree. Per the port
// spec §1.4 this is therefore a REWRITE against our flat POD cache, NOT a file-level port.
//
// WHAT IS TRANSCRIBED VERBATIM AND MUST NOT BE "IMPROVED":
//   dsv4_build_comp_plan() below is upstream's plan-builder arithmetic, copied with only
//   the ubatch->llama_batch accessor edits. Getting state_read_idxs / state_write_idxs
//   wrong does not crash — the model loads and emits FLUENT GARBAGE. Do not re-derive it.
//
// SCOPE OF THIS FIRST CUT (spec §3, deliberate, each one asserted at init rather than
// silently mis-served):
//   * -np 1 / single sequence. Upstream's own header says the cache is non-unified only,
//     and multi-seq is exactly where a wrong plan corrupts silently. With n_stream == 1
//     upstream's dsv4_stream_offset() is identically 0 and plan.n_stream is identically 1,
//     so those terms are folded out here rather than carried as dead generality.
//   * no session save/load (needs llama_io_*), no context shift (llama_model_n_swa()
//     returns 0 for DS4 upstream, so SWA cannot back a shift), no MTP.
//
// The three compressed stores are APPEND-ONLY BLOCK STREAMS, not ring buffers: a row is
// committed once per `ratio` tokens at a block boundary and never evicted. They therefore
// need a per-layer tensor plus a monotonic cursor — no slot allocation, no defrag, no
// eviction. Only the raw SWA cache (window 128) is a true ring.

#include "llama-kv-cache-dsv4.h"

#include "llama-context.h"
#include "llama-model.h"
#include "llama-hparams.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// compress ratios, as they appear in attention.compress_ratios
#define DSV4_CSA_RATIO 4u
#define DSV4_HCA_RATIO 128u

// number of compressed rows a stream needs to cover kv_size tokens
static uint32_t dsv4_comp_size(uint32_t kv_size, uint32_t ratio) {
    return std::max<uint32_t>(1, (kv_size + ratio - 1)/ratio);
}

//
// module state
//

struct dsv4_stream_state {
    uint32_t ratio        = 0;
    uint32_t state_size   = 0;   // rows of compressor ring state (2*ratio when overlapped)
    uint32_t n_embd_state = 0;
    uint32_t n_embd_k     = 0;
    uint32_t cache_size   = 0;   // compressed rows
    bool     overlap      = false;

    std::unordered_map<int32_t, size_t> map_il;   // layer id -> slot
    std::vector<ggml_tensor *> k;                 // [n_embd_k,     cache_size]
    std::vector<ggml_tensor *> st_kv;             // [n_embd_state, state_size]
    std::vector<ggml_tensor *> st_score;          // [n_embd_state, state_size]

    llama_dsv4_comp_plan plan;
};

struct llama_dsv4_memory {
    // raw SWA(128) token cache — the one true ring
    uint32_t raw_size = 0;
    uint32_t n_swa    = 128;
    std::unordered_map<int32_t, size_t> raw_map_il;
    std::vector<ggml_tensor *> raw_k;             // [n_embd_k_raw, raw_size]

    dsv4_stream_state s[3];                       // indexed by llama_dsv4_stream

    llama_dsv4_inputs inputs;

    std::vector<ggml_context *>        ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    // host-side staging for the input tensors (kept alive across set_inputs)
    std::vector<int64_t> raw_k_idxs_host;
};

static llama_dsv4_memory * g_dsv4 = nullptr;   // single-context module (first cut is -np 1)

static llama_dsv4_memory & dsv4_mem() {
    GGML_ASSERT(g_dsv4 && "DSV4 memory module used before init");
    return *g_dsv4;
}

//
// THE PLAN BUILDER — transcribed from upstream dsv4_build_comp_plan().
// Only the accessor shape changed (llama_ubatch -> llama_batch) and the n_stream == 1
// folding described at the top. The arithmetic is upstream's, deliberately unaltered.
//

static llama_seq_id dsv4_batch_seq(const llama_batch & batch, int32_t i, int32_t s) {
    if (batch.seq_id && batch.seq_id[i]) {
        return batch.seq_id[i][s];
    }
    return batch.all_seq_id;
}

static int32_t dsv4_batch_n_seq_id(const llama_batch & batch, int32_t i) {
    return batch.n_seq_id ? batch.n_seq_id[i] : 1;
}

// The reserve/worst-case graph is built straight from llama_batch_get_one(), whose pos
// array is NULL and whose positions are implied by all_pos_0/all_pos_1. llama_decode
// normalises that before its own graph build, but llama_build_graph() for the reserve
// pass does not -- so every read of a position here must go through this.
static llama_pos dsv4_batch_pos(const llama_batch & batch, int32_t i) {
    return batch.pos ? batch.pos[i] : batch.all_pos_0 + (llama_pos) i*batch.all_pos_1;
}

static llama_dsv4_comp_plan dsv4_build_comp_plan(
        const llama_batch & batch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size) {
    llama_dsv4_comp_plan plan;
    plan.n_visible.resize(batch.n_tokens);
    plan.n_stream = 1;

    // n_stream == 1 => dsv4_stream_offset() is identically 0 upstream; folded out.
    const int64_t state_rows = (int64_t) state_size;

    struct persist_row {
        int32_t   dst;
        int32_t   src;
        llama_pos pos;
    };

    std::vector<persist_row> persist_rows;

    // The overlap compressor consumes state_read_idxs as two contiguous halves: the first
    // ratio*n_blocks entries are the "previous-window" gather indices for every block,
    // followed by the "current-window" indices. Collected separately, appended prev-then-cur.
    std::vector<int32_t> overlap_prev_reads;
    std::vector<int32_t> overlap_cur_reads;

    std::map<std::pair<llama_seq_id, llama_pos>, int64_t> curr_token_idx_map;

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        for (int32_t s = 0; s < dsv4_batch_n_seq_id(batch, i); ++s) {
            curr_token_idx_map[std::make_pair(dsv4_batch_seq(batch, i, s), dsv4_batch_pos(batch, i))] = i;
        }
    }

    const auto state_source_idx = [&](llama_seq_id seq_id, llama_pos pos) -> int32_t {
        if (pos < 0) {
            // The overlap compressor needs a zero/-inf source for the first block's previous
            // half. The graph appends that row after the current-ubatch scratch rows.
            return (int32_t) (state_rows + batch.n_tokens);
        }

        const auto key = std::make_pair(seq_id, pos);
        if (curr_token_idx_map.find(key) != curr_token_idx_map.end()) {
            return (int32_t) (state_rows + curr_token_idx_map.at(key));
        }

        return (int32_t) (pos%state_size);
    };

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const llama_pos pos = dsv4_batch_pos(batch, i);

        if (pos < 0) {
            continue;
        }

        plan.state_pos.push_back((int32_t) (pos%ratio));

        const int64_t n_visible = (int64_t) (pos + 1)/ratio;
        plan.n_visible[i] = (int32_t) n_visible;
        plan.n_kv = std::max(plan.n_kv, n_visible);

        for (int32_t s = 0; s < dsv4_batch_n_seq_id(batch, i); ++s) {
            const llama_seq_id seq_id = dsv4_batch_seq(batch, i, s);
            const int32_t state_idx = (int32_t) (pos%state_size);

            const auto it = std::find_if(persist_rows.begin(), persist_rows.end(),
                    [state_idx](const persist_row & row) {
                        return row.dst == state_idx;
                    });
            if (it == persist_rows.end()) {
                persist_rows.push_back({ state_idx, (int32_t) i, pos });
            } else if (pos > it->pos) {
                it->src = (int32_t) i;
                it->pos = pos;
            }

            if ((pos + 1) % (llama_pos) ratio != 0) {
                continue;
            }

            const llama_pos source_start = pos + 1 - (llama_pos) ratio;

            plan.state_write_idxs.push_back(pos/(llama_pos) ratio);
            plan.state_write_pos .push_back((int32_t) source_start);

            if (overlap) {
                const llama_pos prev_start = source_start - (llama_pos) ratio;

                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_prev_reads.push_back(state_source_idx(seq_id, prev_start + (llama_pos) j));
                }
                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_cur_reads.push_back(state_source_idx(seq_id, source_start + (llama_pos) j));
                }
            } else {
                for (uint32_t j = 0; j < ratio; ++j) {
                    plan.state_read_idxs.push_back(state_source_idx(seq_id, source_start + (llama_pos) j));
                }
            }
        }
    }

    if (ratio == DSV4_CSA_RATIO && plan.state_write_idxs.empty() && !plan.state_pos.empty()) {
        // Non-boundary CSA steps still need a write op so their graph matches boundary steps.
        // Use a padded scratch row that is masked from attention.
        assert(kv_size > 0);

        int32_t i = 0;
        while (i < batch.n_tokens && dsv4_batch_pos(batch, i) < 0) {
            ++i;
        }
        assert(i < batch.n_tokens);

        const llama_pos    pos    = dsv4_batch_pos(batch, i);
        const llama_seq_id seq_id = dsv4_batch_seq(batch, i, 0);
        const int32_t source_idx  = state_source_idx(seq_id, pos);

        plan.state_write_idxs.push_back((int64_t) kv_size - 1);
        plan.state_write_pos .push_back(0);

        if (overlap) {
            for (uint32_t j = 0; j < ratio; ++j) {
                overlap_prev_reads.push_back(source_idx);
                overlap_cur_reads .push_back(source_idx);
            }
        } else {
            for (uint32_t j = 0; j < ratio; ++j) {
                plan.state_read_idxs.push_back(source_idx);
            }
        }
    }

    if (overlap) {
        // [ all blocks' prev-window indices | all blocks' cur-window indices ]
        plan.state_read_idxs.reserve(overlap_prev_reads.size() + overlap_cur_reads.size());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(),
                overlap_prev_reads.begin(), overlap_prev_reads.end());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(),
                overlap_cur_reads.begin(), overlap_cur_reads.end());
    }

    plan.n_kv = GGML_PAD(plan.n_kv, 256);

    std::sort(persist_rows.begin(), persist_rows.end(),
            [](const persist_row & a, const persist_row & b) {
                return a.dst < b.dst;
            });

    for (const persist_row & row : persist_rows) {
        plan.state_persist_src_idxs.push_back(row.src);
        plan.state_persist_dst_idxs.push_back(row.dst);
    }

    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    if (debug) {
        LLAMA_LOG_INFO("%s: ratio=%u n_tokens=%d n_persist=%zu n_write=%zu n_read=%zu n_kv=%lld\n",
                __func__, ratio, batch.n_tokens, plan.state_persist_dst_idxs.size(),
                plan.state_write_idxs.size(), plan.state_read_idxs.size(), (long long) plan.n_kv);
    }

    return plan;
}

//
// allocation
//

static bool dsv4_alloc(llama_dsv4_memory & mem, const llama_model & model, bool offload,
                       uint32_t n_layer) {
    const llama_hparams & hparams = model.hparams;

    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;

    // 4 stores per layer (raw k, comp k, state kv, state score) across 3 streams + raw
    const size_t n_tensors_max = 16u*n_layer;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it != ctx_map.end()) {
            return it->second;
        }
        ggml_init_params params = {
            /*.mem_size   =*/ n_tensors_max*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            return nullptr;
        }
        ctx_map[buft] = ctx;
        mem.ctxs.push_back(ctx);
        return ctx;
    };

    for (uint32_t il = 0; il < n_layer; ++il) {
        // This fork resolves the per-layer buffer type through model.buft_layer[],
        // not upstream's model.dev_layer() / ggml_backend_dev_buffer_type().
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (offload) {
            buft = model.buft_layer[il].buft;
        }

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to create ggml context for DSV4 layer %u\n", __func__, il);
            return false;
        }

        const uint32_t ratio = hparams.dsv4_compress_ratios[il];

        // raw SWA(128) K, every layer
        {
            ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                 hparams.n_embd_k_gqa(il), mem.raw_size);
            ggml_format_name(k, "dsv4_raw_k_l%u", il);
            mem.raw_map_il[il] = mem.raw_k.size();
            mem.raw_k.push_back(k);
        }

        // CSA + LID ride the ratio-4 layers; HCA rides the ratio-128 layers.
        for (int si = 0; si < 3; ++si) {
            dsv4_stream_state & st = mem.s[si];
            const bool want = (si == LLAMA_DSV4_HCA) ? (ratio == DSV4_HCA_RATIO)
                                                     : (ratio == DSV4_CSA_RATIO);
            if (!want) {
                continue;
            }

            ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, st.n_embd_k, st.cache_size);
            ggml_tensor * kv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, st.n_embd_state, st.state_size);
            ggml_tensor * sc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, st.n_embd_state, st.state_size);

            ggml_format_name(k,  "dsv4_s%d_k_l%u",        si, il);
            ggml_format_name(kv, "dsv4_s%d_state_kv_l%u", si, il);
            ggml_format_name(sc, "dsv4_s%d_state_sc_l%u", si, il);

            st.map_il[il] = st.k.size();
            st.k       .push_back(k);
            st.st_kv   .push_back(kv);
            st.st_score.push_back(sc);
        }
    }

    for (auto & kvp : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(kvp.second, kvp.first);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate DSV4 buffer\n", __func__);
            return false;
        }
        // DSV4 attention reads compressed-K / compressor-state rows that the current graph
        // does not necessarily overwrite. Uninitialized buffer contents would leak in as
        // instance-specific garbage and corrupt recall — zero everything up front so reads
        // of un-written rows are deterministic. (Upstream does this via clear_compressed().)
        ggml_backend_buffer_clear(buf, 0);
        mem.bufs.push_back(buf);
    }

    return true;
}

//
// lifecycle
//


// PXA_DSV4_STATE_RING_PAD — extra compressor-ring rows so a multi-row batch cannot
// overwrite the history the overlap compressor is about to read.
//
// MEASURED (harness dspark_m5_kvprobe, 2026-08-02): with the ring at its upstream size
// the compressor state after ONE 6-row batch differs from the same six tokens decoded
// one at a time in EXACTLY one compressed row per ratio-4 layer — 1344 of 387072 CSA
// floats, max_abs 3.4-6.0 against an RMS of 1.12. That is structural, not rounding, and
// it happens on a FULLY ACCEPTED batch: it is a property of the batch shape, not of
// whether the drafted tokens were right.
//
// WHY. state_size is 2*ratio for the overlapped streams and the overlap compressor reads
// exactly 2*ratio positions of history, so the ring holds precisely what it needs and not
// one row more. A batch at p..p+nt-1 writes rows p%S..(p+nt-1)%S; a boundary at q reads
// prev-window rows (q-2r+1)%S..(q-r)%S. With S == 2r those two sets intersect for every q
// in the batch, so the FIRST boundary of the batch reads a row this batch has already
// overwritten. Later boundaries are safe because their windows fall inside the batch and
// are served from the ubatch scratch rows — which is why exactly one row per layer is hit.
//
// FIX. Size the ring to history + the widest batch: S = H + n_ubatch. Then no in-batch
// write can alias an out-of-batch read, because the two are at most n_ubatch + H - 1
// positions apart and the ring is that wide. state_source_idx(), the scratch-row base and
// the graph all derive from state_size, so nothing else has to change.
//
// DEFAULT OFF, on the evidence. A/B'd on the GPU (V100, sm_70) with the M5 KV probe:
// enabling the pad did NOT make a fully-accepted batch state-equivalent to sequential
// decode (FULL 0/4 either way) and did NOT make the rejected batch self-heal (NOREST 0/4
// either way). The one arm it changed was RAWONLY - restore the raw ring, replay every
// position from p - which went 0/4 -> 4/4, i.e. WITH the pad the compressed streams do
// recover from a full single-token replay and without it they do not. That is a real
// effect and it is why the lever exists, but it is not the mechanism behind the
// batch-vs-sequential difference, and it does not change the M5 identity result.
//
// Since it also changes multi-row (prefill) numerics and would therefore invalidate every
// temp-0 baseline taken on an earlier binary, shipping it ON is not justified by what has
// been measured. =1 enables it. Byte-unchanged at the default.
static bool dsv4_state_ring_pad_enabled() {
    static const bool on = []() {
        const char * e = getenv("PXA_DSV4_STATE_RING_PAD");
        return e ? atoi(e) != 0 : false;
    }();
    return on;
}

// history the compressor reads: 2*ratio for the overlapped streams, ratio for the plain one
static uint32_t dsv4_state_rows(uint32_t history, uint32_t n_ubatch) {
    if (!dsv4_state_ring_pad_enabled()) {
        return history;
    }
    return history + (n_ubatch > 0 ? n_ubatch : 1u);
}

bool llama_dsv4_memory_init(llama_context & lctx, ggml_type type_k, ggml_type type_v,
                            uint32_t kv_size, bool offload) {
    const llama_model   & model   = lctx.model;
    const llama_hparams & hparams = model.hparams;

    if (g_dsv4) {
        llama_dsv4_memory_free(lctx);
    }

    // Scope guards. Each of these is a case the first cut does NOT serve; failing loudly
    // here is the whole point of the module — a silently mis-served one emits fluent
    // garbage rather than an error.
    if (lctx.cparams.n_seq_max > 1) {
        LLAMA_LOG_ERROR("%s: DeepSeek-V4 first cut supports -np 1 only (got n_seq_max=%u).\n"
                        "%s: Multi-sequence is where a wrong compression plan corrupts silently;\n"
                        "%s: it is deferred deliberately rather than served incorrectly.\n",
                        __func__, lctx.cparams.n_seq_max, __func__, __func__);
        return false;
    }
    if (type_k != GGML_TYPE_F32 || type_v != GGML_TYPE_F32) {
        LLAMA_LOG_WARN("%s: DSV4 compressor state is F32; -ctk/-ctv are ignored for the "
                       "compressed streams in this cut\n", __func__);
    }

    llama_dsv4_memory * mem = new llama_dsv4_memory();

    // raw SWA ring. ⚠ NOT sized from the window alone: unlike upstream (whose prefill
    // attends in-batch keys directly and only rings the residual window), THIS port reads
    // every raw key THROUGH the ring — set_rows(pos % raw_size) then attend over the ring.
    // A ring of GGML_PAD(n_swa+1) rows self-collides on any ubatch spanning more than
    // raw_size positions: rows written by position p are overwritten by p + raw_size in
    // the SAME pass while earlier queries' masks still address the original keys.
    // Measured: temp-0 divergence vs the reference onset at exactly prompt_n > 256,
    // position-dependent, text-independent (DS4-QUANT-LADDER-2026-08-01.md checkpoint 2).
    // Collision-free condition: raw_size >= n_ubatch + n_swa (+1 self row).
    const uint32_t n_swa = hparams.n_swa > 0 ? hparams.n_swa : 128u;
    const uint32_t n_ub  = lctx.cparams.n_ubatch > 0 ? lctx.cparams.n_ubatch : 512u;
    mem->raw_size = GGML_PAD(n_ub + n_swa + 1, 256u);
    mem->n_swa    = n_swa;

    const uint32_t n_embd_head_k = hparams.n_embd_head_k(0);   // method here, field upstream

    // stream parameterisation — upstream llama_kv_cache_dsv4 ctor:
    //   CSA  ratio 4    state 2*ratio (overlapped)  n_embd_state 2*n_embd_head_k
    //   HCA  ratio 128  state ratio   (plain)       n_embd_state   n_embd_head_k
    //   LID  ratio 4    state 2*ratio (overlapped)  n_embd_state 2*indexer_head_size
    mem->s[LLAMA_DSV4_CSA] = {};
    mem->s[LLAMA_DSV4_CSA].ratio        = DSV4_CSA_RATIO;
    mem->s[LLAMA_DSV4_CSA].state_size   = dsv4_state_rows(2*DSV4_CSA_RATIO, n_ub);
    mem->s[LLAMA_DSV4_CSA].n_embd_state = 2*n_embd_head_k;
    mem->s[LLAMA_DSV4_CSA].n_embd_k     = hparams.n_embd_k_gqa(0);
    mem->s[LLAMA_DSV4_CSA].cache_size   = GGML_PAD(dsv4_comp_size(kv_size, DSV4_CSA_RATIO), 256u);
    mem->s[LLAMA_DSV4_CSA].overlap      = true;

    mem->s[LLAMA_DSV4_HCA] = {};
    mem->s[LLAMA_DSV4_HCA].ratio        = DSV4_HCA_RATIO;
    mem->s[LLAMA_DSV4_HCA].state_size   = dsv4_state_rows(DSV4_HCA_RATIO, n_ub);
    mem->s[LLAMA_DSV4_HCA].n_embd_state = n_embd_head_k;
    mem->s[LLAMA_DSV4_HCA].n_embd_k     = hparams.n_embd_k_gqa(0);
    mem->s[LLAMA_DSV4_HCA].cache_size   = GGML_PAD(dsv4_comp_size(kv_size, DSV4_HCA_RATIO), 256u);
    mem->s[LLAMA_DSV4_HCA].overlap      = false;

    mem->s[LLAMA_DSV4_LID] = {};
    mem->s[LLAMA_DSV4_LID].ratio        = DSV4_CSA_RATIO;
    mem->s[LLAMA_DSV4_LID].state_size   = dsv4_state_rows(2*DSV4_CSA_RATIO, n_ub);
    mem->s[LLAMA_DSV4_LID].n_embd_state = 2*hparams.indexer_head_size;
    mem->s[LLAMA_DSV4_LID].n_embd_k     = hparams.indexer_head_size;
    mem->s[LLAMA_DSV4_LID].cache_size   = GGML_PAD(dsv4_comp_size(kv_size, DSV4_CSA_RATIO), 256u);
    mem->s[LLAMA_DSV4_LID].overlap      = true;

    if (!dsv4_alloc(*mem, model, offload, hparams.n_layer)) {
        delete mem;
        return false;
    }

    size_t total = 0;
    for (ggml_backend_buffer_t buf : mem->bufs) {
        total += ggml_backend_buffer_get_size(buf);
    }

    LLAMA_LOG_INFO("%s: DSV4 state ring: CSA %u, HCA %u, LID %u rows (pad %s, n_ubatch %u)\n",
                   __func__, mem->s[LLAMA_DSV4_CSA].state_size, mem->s[LLAMA_DSV4_HCA].state_size,
                   mem->s[LLAMA_DSV4_LID].state_size, dsv4_state_ring_pad_enabled() ? "ON" : "off",
                   lctx.cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: DSV4 memory: raw ring %u cells, CSA %u rows, HCA %u rows, LID %u rows, "
                   "%.2f MiB across %zu buffer(s)\n",
                   __func__, mem->raw_size,
                   mem->s[LLAMA_DSV4_CSA].cache_size,
                   mem->s[LLAMA_DSV4_HCA].cache_size,
                   mem->s[LLAMA_DSV4_LID].cache_size,
                   total/1024.0/1024.0, mem->bufs.size());

    g_dsv4 = mem;
    return true;
}

void llama_dsv4_memory_free(llama_context & /*lctx*/) {
    if (!g_dsv4) {
        return;
    }
    for (ggml_backend_buffer_t buf : g_dsv4->bufs) {
        ggml_backend_buffer_free(buf);
    }
    for (ggml_context * ctx : g_dsv4->ctxs) {
        ggml_free(ctx);
    }
    delete g_dsv4;
    g_dsv4 = nullptr;
}

// write a host vector into a graph input tensor, checking the element count matches the
// size the graph derived from the same plan (a mismatch means plan and graph disagree,
// which must be loud -- a short write leaves the tail uninitialised)
template <typename T>
static void dsv4_fill(ggml_tensor * t, const std::vector<T> & v) {
    if (!t) {
        return;
    }
    GGML_ASSERT((int64_t) v.size() == ggml_nelements(t) &&
                "DSV4 plan/graph input size disagreement");
    if (!v.empty()) {
        ggml_backend_tensor_set(t, v.data(), 0, v.size()*sizeof(T));
    }
}

void llama_dsv4_build_plans(llama_context & /*lctx*/, const llama_batch & batch) {
    llama_dsv4_memory & mem = dsv4_mem();

    for (int si = 0; si < 3; ++si) {
        dsv4_stream_state & st = mem.s[si];
        st.plan = dsv4_build_comp_plan(batch, st.ratio, st.overlap, st.state_size, st.cache_size);
    }
}

void llama_dsv4_set_inputs(llama_context & lctx, const llama_batch & batch) {
    llama_dsv4_memory & mem = dsv4_mem();
    const llama_hparams & hparams = lctx.model.hparams;

    // The plans were built by llama_dsv4_build_plans() from THIS batch, before the graph
    // that owns these input tensors. Re-deriving them here would be harmless (the builder
    // is a pure function of the batch) but wasted; asserting instead catches a build/fill
    // ordering regression loudly rather than as a short write into an ill-sized tensor.
    for (int si = 0; si < 3; ++si) {
        GGML_ASSERT((int32_t) mem.s[si].plan.n_visible.size() == batch.n_tokens &&
                    "DSV4 plans were not built for this batch (llama_dsv4_build_plans must run first)");
    }

    llama_dsv4_inputs & inp = mem.inputs;
    const int32_t nt = batch.n_tokens;

    // ---- raw stream ----

    // ring row for each token
    mem.raw_k_idxs_host.resize(nt);
    for (int32_t i = 0; i < nt; ++i) {
        mem.raw_k_idxs_host[i] = (int64_t) (dsv4_batch_pos(batch, i) % (llama_pos) mem.raw_size);
    }
    dsv4_fill(inp.raw_k_idxs, mem.raw_k_idxs_host);

    if (inp.raw_win_idxs && llama_dsv4_raw_window_mode(lctx) == 2) {
        // bisect: the FULL ring through the window plumbing, so the maths is the ring's
        std::vector<int32_t> ident((size_t) mem.raw_size);
        for (uint32_t j = 0; j < mem.raw_size; ++j) {
            ident[j] = (int32_t) j;
        }
        ggml_backend_tensor_set(inp.raw_win_idxs, ident.data(), 0, ident.size()*sizeof(int32_t));
    }

    if (inp.raw_kq_mask && llama_dsv4_raw_window_mode(lctx) == 1) {
        // window mode: K = [prior n_swa rows from the ring | the nt in-batch keys]
        const llama_pos n_swa    = (llama_pos) mem.n_swa;
        const int64_t   n_kv_raw = inp.raw_kq_mask->ne[0];
        GGML_ASSERT(n_kv_raw >= (int64_t) n_swa + nt && "raw window mask narrower than the window");

        llama_pos P = 0;
        bool have_p = false;
        for (int32_t i = 0; i < nt; ++i) {
            const llama_pos p = dsv4_batch_pos(batch, i);
            if (p >= 0 && (!have_p || p < P)) { P = p; have_p = true; }
        }

        // [prior window | this ubatch's own rows]. Row (P-n_swa+j) cannot have been
        // clobbered: raw_size >= n_ubatch + n_swa + 1, so the positions this pass writes
        // ([P, P+nt)) and the prior window ([P-n_swa, P)) never share a ring row.
        // safe filler row: a row this pass definitely wrote, so no gathered row is ever
        // uninitialised. The mask leaves every filler column -inf, so it contributes zero.
        // ORDER: [prior | in-batch] — KV positions monotonic, the ordering whose maths
        // was proven byte-identical to the ring on the soft_max graph (checkpoint 12
        // arms #4/#5). The swapped [in-batch | prior] order (v6) exists only to appease
        // the fallback FA kernel's leading-tile behaviour and produces WRONG TEXT there;
        // window mode is not enabled under flash-attn (see llama_dsv4_raw_window_mode).
        const int32_t safe = nt > 0 ? (int32_t) mem.raw_k_idxs_host[0] : 0;
        std::vector<int32_t> win((size_t) n_kv_raw, safe);
        for (llama_pos j = 0; j < n_swa; ++j) {
            const llama_pos pos_j = P - n_swa + j;
            win[(size_t) j] = pos_j >= 0 ? (int32_t) (pos_j % (llama_pos) mem.raw_size) : safe;
        }
        for (int32_t k = 0; k < nt; ++k) {
            win[(size_t) n_swa + (size_t) k] = (int32_t) mem.raw_k_idxs_host[k];
        }
        if (inp.raw_win_idxs) {
            ggml_backend_tensor_set(inp.raw_win_idxs, win.data(), 0, win.size()*sizeof(int32_t));
        }

        std::vector<float> mask((size_t) n_kv_raw*nt, -INFINITY);
        for (int32_t i = 0; i < nt; ++i) {
            const llama_pos p = dsv4_batch_pos(batch, i);
            if (p < 0) {
                continue;
            }
            for (llama_pos j = 0; j < n_swa; ++j) {
                const llama_pos pos_j = P - n_swa + j;
                if (pos_j >= 0 && pos_j <= p && p - pos_j < n_swa) {
                    mask[(size_t) i*n_kv_raw + (size_t) j] = 0.0f;
                }
            }
            for (int32_t k = 0; k < nt; ++k) {
                const llama_pos pk = dsv4_batch_pos(batch, k);
                if (pk >= 0 && pk <= p && p - pk < n_swa) {
                    mask[(size_t) i*n_kv_raw + (size_t) n_swa + (size_t) k] = 0.0f;
                }
            }
        }
        ggml_backend_tensor_set(inp.raw_kq_mask, mask.data(), 0, mask.size()*sizeof(float));
    } else if (inp.raw_kq_mask) {
        const int64_t n_kv_raw = inp.raw_kq_mask->ne[0];
        const llama_pos n_swa  = (llama_pos) (hparams.n_swa > 0 ? hparams.n_swa : 128u);

        std::vector<float> mask((size_t) n_kv_raw*nt, -INFINITY);
        for (int32_t i = 0; i < nt; ++i) {
            const llama_pos p = dsv4_batch_pos(batch, i);
            if (p < 0) {
                continue;
            }
            for (int64_t j = 0; j < n_kv_raw; ++j) {
                // most recent position stored in ring row j, at or before p
                const llama_pos back = (llama_pos) (((int64_t) p - j) % (int64_t) mem.raw_size);
                const llama_pos q    = p - (back < 0 ? back + (llama_pos) mem.raw_size : back);
                const llama_pos d    = p - q;
                if (q >= 0 && d >= 0 && d < n_swa) {
                    mask[(size_t) i*n_kv_raw + j] = 0.0f;
                }
            }
        }
        ggml_backend_tensor_set(inp.raw_kq_mask, mask.data(), 0, mask.size()*sizeof(float));
    }

    // ---- the three compressed streams ----

    for (int si = 0; si < 3; ++si) {
        const llama_dsv4_comp_plan & pl = mem.s[si].plan;
        llama_dsv4_comp_inputs     & ci = inp.comp[si];

        dsv4_fill(ci.state_pos,              pl.state_pos);
        dsv4_fill(ci.state_persist_src_idxs, pl.state_persist_src_idxs);
        dsv4_fill(ci.state_persist_dst_idxs, pl.state_persist_dst_idxs);
        dsv4_fill(ci.state_read_idxs,        pl.state_read_idxs);
        dsv4_fill(ci.state_write_idxs,       pl.state_write_idxs);
        dsv4_fill(ci.state_write_pos,        pl.state_write_pos);

        if (ci.kq_mask) {
            const int64_t n_kv = ci.kq_mask->ne[0];
            std::vector<float> mask((size_t) n_kv*nt, -INFINITY);
            for (int32_t i = 0; i < nt; ++i) {
                // n_visible[i] completed compressed rows are addressable by token i;
                // everything above (including the graph-width padding) stays masked
                const int64_t vis = std::min<int64_t>(pl.n_visible[i], n_kv);
                for (int64_t j = 0; j < vis; ++j) {
                    mask[(size_t) i*n_kv + j] = 0.0f;
                }
            }
            ggml_backend_tensor_set(ci.kq_mask, mask.data(), 0, mask.size()*sizeof(float));
        }
    }
}

//
// graph-side accessors
//

// ggml_set_rows() wants a 2D source: it asserts src->ne[1] == idxs->ne[0] and
// src->ne[2] == dst->ne[2]. The DSV4 graph hands us 3D tensors ([n_embd, 1, n_tokens]
// for raw K), so flatten to [dst_row_width, n_rows] first. Flattened against the
// DESTINATION width, not an assumed source layout, because the four stores have
// different widths (n_embd_k_gqa / n_embd_state / indexer_head_size).
static ggml_tensor * dsv4_rows_src(ggml_context * ctx, ggml_tensor * dst, ggml_tensor * src) {
    const int64_t w = dst->ne[0];
    GGML_ASSERT(w > 0 && ggml_nelements(src) % w == 0);
    if (src->ne[0] == w && ggml_n_dims(src) <= 2) {
        return src;
    }
    return ggml_reshape_2d(ctx, src, w, ggml_nelements(src)/w);
}

llama_dsv4_inputs & llama_dsv4_get_inputs(llama_context & /*lctx*/) {
    return dsv4_mem().inputs;
}

const llama_dsv4_comp_plan & llama_dsv4_get_plan(const llama_context & /*lctx*/, llama_dsv4_stream s) {
    return dsv4_mem().s[s].plan;
}

uint32_t llama_dsv4_get_raw_n_kv(const llama_context & /*lctx*/) {
    return dsv4_mem().raw_size;
}

uint32_t llama_dsv4_get_raw_swa(const llama_context & /*lctx*/) {
    return dsv4_mem().n_swa;
}

int llama_dsv4_raw_window_mode(const llama_context & lctx) {
    static const int env_mode = []() {
        const char * e = getenv("PXA_DSV4_RAW_WINDOW_KV");
        return e ? atoi(e) : -1;   // -1 = no explicit override
    }();
    if (env_mode >= 0) {
        return env_mode;
    }
    // DEFAULT: window (mode 1) on the non-flash-attn graph, ring (mode 0) under
    // flash-attn.
    //
    // The window path is proven byte-identical to the ring on the soft_max graph
    // (DS4-QUANT-LADDER checkpoint 12, arms #4/#5), and it is the arm that matches
    // upstream's raw-attention width (n_swa + n_tokens instead of the whole padded
    // ring: 640 vs 768 at ub=512 prefill, 129+pad vs 768 at decode). Under
    // cparams.flash_attn the executing FA kernel rejects the window layout — NaN
    // with [prior | in-batch] order (online softmax seeded from a fully-masked
    // leading tile computes exp(-inf - -inf)), wrong text with [in-batch | prior]
    // (the kernel relies on KV order) — so flash-attn contexts stay on the ring.
    // NOTE this is the *fallback* FA kernel: CUDA flash-attention itself never runs
    // for DS4 on sm_70 (head dim 512 needs new_mma / Ampere+, and
    // ggml_cuda_fattn_is_supported() returns false, see ggml-cuda/fattn.cu).
    return lctx.cparams.flash_attn ? 0 : 1;
}

bool llama_dsv4_raw_window_enabled(const llama_context & lctx) {
    return llama_dsv4_raw_window_mode(lctx) != 0;
}

// Window width padding. Default 64: enough that the soft_max path's kernels never
// see a degenerate tail, and prefill at ub=512 keeps the full lever (128+512 = 640
// is already a multiple of 64).
//
// The old default here was 256, imposed for the CUDA flash-attention contract
// (K->ne[1] % FATTN_KQ_STRIDE, fattn-common.cuh) — but that contract belongs to a
// path DS4 cannot take on sm_70: fattn_wmma_f16 supports head dims
// {64,80,96,112,128,256} and DS4 is 512, whose branch requires new_mma_available
// (Ampere+). ggml_cuda_fattn_is_supported() therefore returns false and the FA op
// never reaches a CUDA FA kernel here. Padding to 256 also erased the lever's
// prefill benefit entirely (640 -> 768 = exactly the ring). If a future target ever
// routes DS4 through CUDA FA, set PXA_DSV4_RAW_WINDOW_ALIGN=256.
static uint32_t dsv4_raw_window_align() {
    static const uint32_t a = []() {
        const char * e = getenv("PXA_DSV4_RAW_WINDOW_ALIGN");
        const int v = e ? atoi(e) : 64;
        uint32_t r = (uint32_t) (v > 0 ? v : 64);
        return r;
    }();
    return a;
}

uint32_t llama_dsv4_raw_window_width(const llama_context & /*lctx*/, int64_t n_tokens) {
    const uint32_t w = GGML_PAD((uint32_t) (dsv4_mem().n_swa + (uint32_t) n_tokens),
                                dsv4_raw_window_align());
    const uint32_t r = w < dsv4_mem().raw_size ? w : dsv4_mem().raw_size;
    return r;
}

int64_t llama_dsv4_get_raw_nrot(const llama_context & /*lctx*/) {
    return 0;   // raw stream carries no k_rot in this cut
}

int64_t llama_dsv4_get_comp_nrot(const llama_context & /*lctx*/, llama_dsv4_stream /*s*/) {
    return 0;   // compressed streams carry no k_rot in this cut
}

// The cache is STORED flat as [n_embd_k_gqa, n_rows] because ggml_set_rows() wants a
// row-major destination, but attention wants [n_embd_head_k, n_head_kv, n_kv]: MHA does
// ggml_permute(k, 0, 2, 1, 3) and then mul_mat, and ggml_can_mul_mat needs
// q->ne[2] % k->ne[2] == 0. Flat, k permutes to ne[2] = n_kv (256 here) and 64 heads
// % 256 fails; viewed, it permutes to ne[2] = n_head_kv = 1 and the check passes.
// n_embd_head_k*n_head_kv == n_embd_k_gqa, so this reinterprets the same rows — the
// write path (flat) and the read path (viewed) address identical memory.
static ggml_tensor * dsv4_k_attn_view(ggml_context * ctx, ggml_tensor * k,
                                      int64_t n_embd_head_k, int64_t n_head_kv) {
    GGML_ASSERT(k->ne[0] == n_embd_head_k*n_head_kv);
    return ggml_view_3d(ctx, k, n_embd_head_k, n_head_kv, k->ne[1],
                        ggml_row_size(k->type, n_embd_head_k),
                        k->nb[1], 0);
}

ggml_tensor * llama_dsv4_get_raw_k(const llama_context & lctx, ggml_context * ctx, int32_t il) {
    llama_dsv4_memory & mem = dsv4_mem();
    ggml_tensor * k = mem.raw_k[mem.raw_map_il.at(il)];
    const llama_hparams & hp = lctx.model.hparams;
    return dsv4_k_attn_view(ctx, k, hp.n_embd_head_k(il), hp.n_head_kv(il));
}

ggml_tensor * llama_dsv4_get_raw_win_k(const llama_context & lctx, ggml_context * ctx,
                                       ggml_tensor * ring_w, ggml_tensor * win_idxs, int32_t il) {
    // ONE gather over the ring: [prior window | this ubatch's rows].
    //
    // Gather from `ring_w` -- what llama_dsv4_cpy_raw_k() RETURNED -- not from the ring
    // leaf. The set_rows that writes this ubatch's keys returns its own handle, so reading
    // the leaf leaves the gather with NO dependency on the scatter, and the multi-backend
    // scheduler is then free to run the gather first and read unwritten rows (NaN). The
    // full-ring path is immune only because its K is a view aliasing the ring in place.
    //
    // ggml_get_rows() on an F16 source yields F32; cast straight back to the ring's type
    // so flash-attn takes K/V as F16 without build_dsv4_attn_mha casting the SAME tensor
    // twice (it is passed as both k and v). F16 -> F32 -> F16 is exact, so the window
    // path is bit-identical to the ring path, not merely close.
    llama_dsv4_memory & mem = dsv4_mem();
    ggml_tensor * ring = mem.raw_k[mem.raw_map_il.at(il)];        // [w, raw_size]
    GGML_ASSERT(ring_w != nullptr && "raw window gather needs the set_rows result");
    GGML_ASSERT(ring_w->ne[0] == ring->ne[0] && ring_w->ne[1] == ring->ne[1]);
    ggml_tensor * win = ggml_get_rows(ctx, ring_w, win_idxs);     // [w, n_swa+nt] F32
    if (win->type != ring->type) {
        win = ggml_cast(ctx, win, ring->type);
    }
    const llama_hparams & hp = lctx.model.hparams;
    return dsv4_k_attn_view(ctx, win, hp.n_embd_head_k(il), hp.n_head_kv(il));
}

ggml_tensor * llama_dsv4_cpy_raw_k(const llama_context & /*lctx*/, ggml_context * ctx,
                                   ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) {
    // write through the FLAT store, not the attention view: set_rows needs a row-major
    // [width, n_rows] destination
    llama_dsv4_memory & mem = dsv4_mem();
    ggml_tensor * dst = mem.raw_k[mem.raw_map_il.at(il)];
    return ggml_set_rows(ctx, dst, dsv4_rows_src(ctx, dst, k_cur), k_idxs);
}

ggml_tensor * llama_dsv4_get_comp_k(const llama_context & lctx, ggml_context * ctx,
                                    llama_dsv4_stream s, int32_t il) {
    dsv4_stream_state & st = dsv4_mem().s[s];
    ggml_tensor * k = st.k[st.map_il.at(il)];
    // LID carries indexer_head_size per head, the other two carry n_embd_head_k; both
    // are single-KV-head here, so the view is [width, 1, n_rows] either way.
    return dsv4_k_attn_view(ctx, k, st.n_embd_k, 1);
}

ggml_tensor * llama_dsv4_cpy_comp_k(const llama_context & /*lctx*/, ggml_context * ctx,
                                    llama_dsv4_stream s, ggml_tensor * k_cur,
                                    ggml_tensor * k_idxs, int32_t il) {
    dsv4_stream_state & st = dsv4_mem().s[s];
    ggml_tensor * dst = st.k[st.map_il.at(il)];   // flat store, see cpy_raw_k
    return ggml_set_rows(ctx, dst, dsv4_rows_src(ctx, dst, k_cur), k_idxs);
}

ggml_tensor * llama_dsv4_get_state_kv(const llama_context & /*lctx*/, ggml_context * /*ctx*/,
                                      llama_dsv4_stream s, int32_t il) {
    dsv4_stream_state & st = dsv4_mem().s[s];
    return st.st_kv[st.map_il.at(il)];
}

ggml_tensor * llama_dsv4_get_state_score(const llama_context & /*lctx*/, ggml_context * /*ctx*/,
                                         llama_dsv4_stream s, int32_t il) {
    dsv4_stream_state & st = dsv4_mem().s[s];
    return st.st_score[st.map_il.at(il)];
}

ggml_tensor * llama_dsv4_cpy_state_kv(const llama_context & lctx, ggml_context * ctx,
                                      llama_dsv4_stream s, ggml_tensor * cur,
                                      ggml_tensor * idxs, int32_t il) {
    ggml_tensor * dst = llama_dsv4_get_state_kv(lctx, ctx, s, il);
    return ggml_set_rows(ctx, dst, dsv4_rows_src(ctx, dst, cur), idxs);
}

ggml_tensor * llama_dsv4_cpy_state_score(const llama_context & lctx, ggml_context * ctx,
                                         llama_dsv4_stream s, ggml_tensor * cur,
                                         ggml_tensor * idxs, int32_t il) {
    ggml_tensor * dst = llama_dsv4_get_state_score(lctx, ctx, s, il);
    return ggml_set_rows(ctx, dst, dsv4_rows_src(ctx, dst, cur), idxs);
}

//
// =========================== SPECULATIVE ROLLBACK (M5) ===========================
//
// A DSpark verify batch of 1+k rows writes DSV4 state for EVERY row, including the rows a
// rejected draft occupies. Unlike the standard llama_kv_cache there is no seq_rm here, no
// crop and no truncate -- and the compressor state ring is not merely un-rolled-back, it
// is STRUCTURALLY too small to hold a speculative window alongside the history the overlap
// compressor still needs:
//
//   CSA/LID: ratio 4, state_size = 2*ratio = 8, overlap = true.
//   A boundary at position q reads the PREVIOUS window q-7 .. q-4, whose ring rows are
//   (q-7)%8 .. (q-4)%8  ==  (q+1)%8 .. (q+4)%8.
//   A batch at p .. p+5 has already written rows p%8 .. (p+5)%8.
//   For every q in [p, p+5] those two sets intersect, so the previous-window read returns
//   a FUTURE row instead of the history it asked for. The ring holds exactly 2*ratio
//   positions and the compressor needs exactly 2*ratio of history: there is no slack for
//   even one speculative row, let alone five.
//
// Therefore the rejected rows cannot be "healed" by re-decoding the correct token at the
// same position -- by the time we do, the history that position's compressor boundary
// needed has already been overwritten. The fix is a snapshot taken BEFORE the verify batch
// and restored when the batch is not fully accepted. Only the rows the batch writes need
// saving, and the compression plans already enumerate exactly those rows.
//
// The raw SWA ring is a different story and is saved separately so the two can be
// attributed independently: raw_size >= n_ubatch + n_swa + 1 by construction, so a
// speculative row and the window it would collide with never share a ring row. That is an
// argument, not a measurement, which is why RAW is its own restore component.
//

struct dsv4_saved_row {
    ggml_tensor * t   = nullptr;
    size_t        off = 0;    // byte offset into the tensor
    size_t        len = 0;    // bytes
    size_t        blob_off = 0;
};

struct dsv4_spec_snapshot_state {
    bool                        valid = false;
    std::vector<uint8_t>        blob;
    std::vector<dsv4_saved_row> rows[4];   // indexed by DSV4 component: RAW, CSA, HCA, LID
};

static dsv4_spec_snapshot_state g_dsv4_snap;

static void dsv4_snap_add(dsv4_spec_snapshot_state & S, int comp, ggml_tensor * t, int64_t row) {
    if (!t || row < 0 || row >= t->ne[1]) {
        return;
    }
    const size_t len = t->nb[1];
    dsv4_saved_row r;
    r.t        = t;
    r.off      = (size_t) row * len;
    r.len      = len;
    r.blob_off = S.blob.size();
    S.blob.resize(S.blob.size() + len);
    ggml_backend_tensor_get(t, S.blob.data() + r.blob_off, r.off, r.len);
    S.rows[comp].push_back(r);
}

size_t llama_dsv4_spec_snapshot_bytes(const llama_context & /*lctx*/) {
    return g_dsv4_snap.valid ? g_dsv4_snap.blob.size() : 0;
}

bool llama_dsv4_spec_snapshot(llama_context & /*lctx*/, const llama_batch & batch) {
    if (!g_dsv4) {
        return false;
    }
    llama_dsv4_memory & mem = *g_dsv4;

    dsv4_spec_snapshot_state & S = g_dsv4_snap;
    S.valid = false;
    S.blob.clear();
    for (int c = 0; c < 4; ++c) {
        S.rows[c].clear();
    }

    // RAW ring rows this batch will write. Deduplicated: two rows of a DSpark verify batch
    // share position p (the row-0 splice), and saving the same row twice would restore the
    // second copy over the first -- harmless here because both copies are pre-batch, but
    // it would silently mask an ordering bug later.
    {
        std::vector<int64_t> seen;
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            const llama_pos p = dsv4_batch_pos(batch, i);
            if (p < 0) {
                continue;
            }
            const int64_t row = (int64_t) (p % (llama_pos) mem.raw_size);
            if (std::find(seen.begin(), seen.end(), row) != seen.end()) {
                continue;
            }
            seen.push_back(row);
            for (ggml_tensor * t : mem.raw_k) {
                dsv4_snap_add(S, 0, t, row);
            }
        }
    }

    // Compressed streams. The plans are a PURE function of the batch, so they are rebuilt
    // locally rather than read from st.plan -- st.plan belongs to whatever batch is
    // currently mid-flight and this call deliberately runs before llama_decode().
    for (int si = 0; si < 3; ++si) {
        dsv4_stream_state & st = mem.s[si];
        if (st.k.empty()) {
            continue;
        }
        const llama_dsv4_comp_plan plan =
            dsv4_build_comp_plan(batch, st.ratio, st.overlap, st.state_size, st.cache_size);

        const int comp = si + 1;
        for (size_t j = 0; j < st.k.size(); ++j) {
            for (int32_t dst : plan.state_persist_dst_idxs) {
                dsv4_snap_add(S, comp, st.st_kv   [j], dst);
                dsv4_snap_add(S, comp, st.st_score[j], dst);
            }
            for (int64_t dst : plan.state_write_idxs) {
                dsv4_snap_add(S, comp, st.k[j], dst);
            }
        }
    }

    S.valid = true;
    return true;
}

bool llama_dsv4_spec_restore(llama_context & /*lctx*/, uint32_t components) {
    dsv4_spec_snapshot_state & S = g_dsv4_snap;
    if (!S.valid) {
        return false;
    }
    for (int c = 0; c < 4; ++c) {
        if (!(components & (1u << c))) {
            continue;
        }
        for (const dsv4_saved_row & r : S.rows[c]) {
            ggml_backend_tensor_set(r.t, S.blob.data() + r.blob_off, r.off, r.len);
        }
    }
    return true;
}

void llama_dsv4_spec_discard(llama_context & /*lctx*/) {
    g_dsv4_snap.valid = false;
    g_dsv4_snap.blob.clear();
    for (int c = 0; c < 4; ++c) {
        g_dsv4_snap.rows[c].clear();
    }
}

// FNV-1a over the raw bytes of a component's tensors. Debug instrument for the M5 KV
// probe: a digest cannot saturate the way a logit can, so a model whose forward pass has
// collapsed still gives a decisive answer about whether the STATE differs.
static uint64_t dsv4_digest_tensors(const std::vector<ggml_tensor *> & ts, uint64_t h) {
    std::vector<uint8_t> buf;
    for (ggml_tensor * t : ts) {
        if (!t) {
            continue;
        }
        const size_t n = ggml_nbytes(t);
        buf.resize(n);
        ggml_backend_tensor_get(t, buf.data(), 0, n);
        for (size_t i = 0; i < n; ++i) {
            h ^= (uint64_t) buf[i];
            h *= 1099511628211ull;
        }
    }
    return h;
}

// Count non-finite F32 elements across every DSV4 tensor. A digest alone cannot tell a
// live state machine from one that has NaN'd: two different NaN payloads hash differently
// and the probe would report "DIFFER" for a model that never computed anything. This is
// the guard against exactly that false positive.
static int64_t dsv4_count_nonfinite(const std::vector<ggml_tensor *> & ts) {
    int64_t n_bad = 0;
    std::vector<float> buf;
    for (ggml_tensor * t : ts) {
        if (!t || t->type != GGML_TYPE_F32) {
            continue;
        }
        const int64_t n = ggml_nelements(t);
        buf.resize((size_t) n);
        ggml_backend_tensor_get(t, buf.data(), 0, (size_t) n*sizeof(float));
        for (int64_t i = 0; i < n; ++i) {
            if (!std::isfinite(buf[(size_t) i])) {
                ++n_bad;
            }
        }
    }
    return n_bad;
}

uint64_t llama_dsv4_debug_digest(const llama_context & /*lctx*/, int component) {
    if (!g_dsv4) {
        return 0;
    }
    llama_dsv4_memory & mem = *g_dsv4;
    uint64_t h = 1469598103934665603ull;
    if (component == -2) {
        int64_t n_bad = dsv4_count_nonfinite(mem.raw_k);
        for (int si = 0; si < 3; ++si) {
            n_bad += dsv4_count_nonfinite(mem.s[si].k);
            n_bad += dsv4_count_nonfinite(mem.s[si].st_kv);
            n_bad += dsv4_count_nonfinite(mem.s[si].st_score);
        }
        return (uint64_t) n_bad;
    }
    if (component < 0) {
        h = dsv4_digest_tensors(mem.raw_k, h);
        for (int si = 0; si < 3; ++si) {
            h = dsv4_digest_tensors(mem.s[si].k,        h);
            h = dsv4_digest_tensors(mem.s[si].st_kv,    h);
            h = dsv4_digest_tensors(mem.s[si].st_score, h);
        }
        return h;
    }
    if (component == 0) {
        return dsv4_digest_tensors(mem.raw_k, h);
    }
    if (component >= 1 && component <= 3) {
        dsv4_stream_state & st = mem.s[component - 1];
        h = dsv4_digest_tensors(st.k,        h);
        h = dsv4_digest_tensors(st.st_kv,    h);
        h = dsv4_digest_tensors(st.st_score, h);
        return h;
    }
    return 0;
}

// Digest a stream's compressed K cache EXCLUDING each tensor's final row.
// The final row (cache_size-1) is the by-design masked dummy-write slot: ub=1
// non-boundary steps park a masked scratch write there while a batch containing a
// boundary does not, so the two submission styles leave different bytes in a row
// attention never reads. Hashing it makes state-equivalence gates fail on a slot
// with no consequences (measured 2026-08-02: the ENTIRE batched-vs-sequential
// FULL/CTRL difference on the CPU path is this slot -- 21 layers x 64 floats per
// stream, nothing else). The "live" digest answers the question those gates
// actually ask: is the ATTENDED state equivalent?
static uint64_t dsv4_digest_tensors_skip_last_row(const std::vector<ggml_tensor *> & ts, uint64_t h) {
    std::vector<uint8_t> buf;
    for (ggml_tensor * t : ts) {
        if (!t) {
            continue;
        }
        const size_t row_bytes = (size_t) t->nb[1];
        const size_t n_full    = (size_t) ggml_nbytes(t);
        const size_t n         = t->ne[1] > 1 ? n_full - row_bytes : n_full;
        buf.resize(n);
        ggml_backend_tensor_get(t, buf.data(), 0, n);
        for (size_t i = 0; i < n; ++i) {
            h ^= (uint64_t) buf[i];
            h *= 1099511628211ull;
        }
    }
    return h;
}

uint64_t llama_dsv4_debug_digest_live(const llama_context & /*lctx*/, int component) {
    if (!g_dsv4) {
        return 0;
    }
    llama_dsv4_memory & mem = *g_dsv4;
    uint64_t h = 1469598103934665603ull;
    if (component < 0) {
        h = dsv4_digest_tensors(mem.raw_k, h);
        for (int si = 0; si < 3; ++si) {
            h = dsv4_digest_tensors_skip_last_row(mem.s[si].k, h);
            h = dsv4_digest_tensors(mem.s[si].st_kv,    h);
            h = dsv4_digest_tensors(mem.s[si].st_score, h);
        }
        return h;
    }
    if (component == 0) {
        return dsv4_digest_tensors(mem.raw_k, h);
    }
    if (component >= 1 && component <= 3) {
        dsv4_stream_state & st = mem.s[component - 1];
        h = dsv4_digest_tensors_skip_last_row(st.k, h);
        h = dsv4_digest_tensors(st.st_kv,    h);
        h = dsv4_digest_tensors(st.st_score, h);
        return h;
    }
    return 0;
}

// Export a component's F32 state so a probe can measure HOW MUCH two states differ, not
// just THAT they differ. A digest answers "are these the same bytes"; it cannot tell a
// last-ulp reordering from a structurally wrong read, and those two have opposite
// consequences for whether a fully-accepted batch is usable.
size_t llama_dsv4_debug_export(const llama_context & /*lctx*/, int component,
                               float * out, size_t max_n) {
    if (!g_dsv4) {
        return 0;
    }
    llama_dsv4_memory & mem = *g_dsv4;
    std::vector<const std::vector<ggml_tensor *> *> groups;
    if (component == 0) {
        groups.push_back(&mem.raw_k);
    } else if (component >= 1 && component <= 3) {
        dsv4_stream_state & st = mem.s[component - 1];
        groups.push_back(&st.k);
        groups.push_back(&st.st_kv);
        groups.push_back(&st.st_score);
    } else {
        groups.push_back(&mem.raw_k);
        for (int si = 0; si < 3; ++si) {
            groups.push_back(&mem.s[si].k);
            groups.push_back(&mem.s[si].st_kv);
            groups.push_back(&mem.s[si].st_score);
        }
    }

    size_t n = 0;
    for (const std::vector<ggml_tensor *> * g : groups) {
        for (ggml_tensor * t : *g) {
            if (!t || t->type != GGML_TYPE_F32) {
                continue;
            }
            const size_t cnt = (size_t) ggml_nelements(t);
            if (out && n + cnt <= max_n) {
                ggml_backend_tensor_get(t, out + n, 0, cnt*sizeof(float));
            }
            n += cnt;
        }
    }
    return n;
}

//
// public C API (llama.h). Thin forwarders: the module is a process-global singleton, so
// the context argument is validated but otherwise unused -- exactly as everywhere else in
// this file. See llama_dsv4_get_inputs()'s note about the single-context first cut.
//

bool llama_dspark_kv_snapshot(struct llama_context * ctx, struct llama_batch batch) {
    if (!ctx || !g_dsv4) {
        return false;
    }
    return llama_dsv4_spec_snapshot(*ctx, batch);
}

bool llama_dspark_kv_restore(struct llama_context * ctx, uint32_t components) {
    if (!ctx || !g_dsv4) {
        return false;
    }
    return llama_dsv4_spec_restore(*ctx, components);
}

void llama_dspark_kv_discard(struct llama_context * ctx) {
    if (!ctx) {
        return;
    }
    llama_dsv4_spec_discard(*ctx);
}

size_t llama_dspark_kv_snapshot_bytes(struct llama_context * ctx) {
    if (!ctx) {
        return 0;
    }
    return llama_dsv4_spec_snapshot_bytes(*ctx);
}

uint64_t llama_dspark_kv_digest(struct llama_context * ctx, int component) {
    if (!ctx) {
        return 0;
    }
    return llama_dsv4_debug_digest(*ctx, component);
}

uint64_t llama_dspark_kv_digest_live(struct llama_context * ctx, int component) {
    if (!ctx) {
        return 0;
    }
    return llama_dsv4_debug_digest_live(*ctx, component);
}

// NEGATIVE CONTROL for the M5 identity gate. Perturbs exactly the rows the last snapshot
// covers - i.e. exactly the rows a speculative batch overwrites - by `amount`. If the
// identity gate still passes with a perturbation the size of the corruption the KV probe
// MEASURED, then that gate is insensitive at this scale and its "HOLDS" means nothing.
// A gate whose failure mode has never been demonstrated is decoration.
bool llama_dspark_kv_scramble(struct llama_context * ctx, float amount) {
    if (!ctx || !g_dsv4 || !g_dsv4_snap.valid) {
        return false;
    }
    std::vector<float> buf;
    // components 1..3 only: the compressed streams are what cannot self-heal
    for (int c = 1; c < 4; ++c) {
        for (const dsv4_saved_row & r : g_dsv4_snap.rows[c]) {
            if (r.t->type != GGML_TYPE_F32) {
                continue;
            }
            const size_t n = r.len/sizeof(float);
            buf.resize(n);
            ggml_backend_tensor_get(r.t, buf.data(), r.off, r.len);
            for (size_t i = 0; i < n; ++i) {
                buf[i] += ((i & 1) ? amount : -amount);
            }
            ggml_backend_tensor_set(r.t, buf.data(), r.off, r.len);
        }
    }
    return true;
}

size_t llama_dspark_kv_export(struct llama_context * ctx, int component, float * out, size_t max_n) {
    if (!ctx) {
        return 0;
    }
    return llama_dsv4_debug_export(*ctx, component, out, max_n);
}
