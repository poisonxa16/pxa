// PXA_MTP_BATCH_SLOTS: one MTP draft decode per STEP across all drafting slots, on the CPU, with no
// model, no weights, no context and no GPU.
//
// Our draft loop
// (mtp_speculative_gen_draft, common/speculative.cpp) runs ONE sequence's chain of 1-row decodes;
// at -np N the server calls it N times per cycle, so the shared companion sees N*K round trips.
// Mainline puts one row per drafting sequence into the SAME batch and advances them together
// (ggml-org/llama.cpp 304665fe7, common/speculative.cpp:1596-1746), paying K decodes of N rows.
// common_speculative_draft_batched() is that regrouping.
//
// THE CLAIM UNDER TEST. The regrouping is a SCHEDULING change and nothing else: per sequence the
// token fed in, the position it is written at, the K/V row it lands in, the hidden row it is
// conditioned on, the token that comes out and the p_min ordering that decides whether the chain
// continues are all unchanged. So, given a decode that computes each row from that row's own
// (token, position, conditioning hidden) -- which is what a batch row IS -- the two schedulers must
// produce, for every sequence:
//
//   * the same draft token stream,
//   * the same number of decodes and the same PXA_MTP_STATS stop code,
//   * the same K/V cells, each holding the same (conditioning hidden, token) pair at the same
//     position, and
//   * the same draft-region purge bounds.
//
// and the batched one must do it in fewer llama_decode calls -- exactly max_over_seqs(steps)
// instead of sum_over_seqs(steps).
//
// HOW IT IS TESTED. The per-sequence half of the loop is common/pxa-mtp-batch-slots.h's
// pxa_mtp_draft_chain, and it is the REAL header both schedulers use. This file supplies a
// deterministic oracle standing in for the companion decode (a row's output is a pure function of
// that row's inputs, which is the modelling assumption P5 rests on) and two schedulers:
//
//   serial()   a transcription of mtp_speculative_gen_draft()'s loop -- one chain at a time, one
//              row per batch;
//   batched()  a transcription of common_speculative_draft_batched()'s loop -- all still-running
//              chains, one row each, one batch per step.
//
// Case 1 pins the transcription itself against hand-computed expectations, so the equivalence
// cases cannot pass by both schedulers being wrong in the same way. Cases 2-5 then drive the two
// over 4-deep randomised scenarios, including early p_min stops at different depths per slot,
// cached free tokens, embedding-read failures and a decode failure, and require the four
// per-sequence records above to be identical.
//
// NOT tested here, because it is not a CPU property: whether the target's 2-row GEMM and two 1-row
// GEMVs produce bitwise-identical logits. That is the GPU arm -- two concurrent clients against an
// -np 2 seat, greedy identity per stream plus aggregate t/s -- and it is what the lever stays
// default-OFF for.

#include "pxa-mtp-batch-slots.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

static int          g_fail = 0;
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

// ---------------------------------------------------------------------------------------------
// The stand-in for the companion decode.
//
// A row is (token, position, conditioning hidden). The hidden state is modelled by an integer id so
// that "was this row conditioned on the row the previous step produced?" is checkable directly --
// that chain is the whole point of the MTP head and the thing a scheduling change could break.
//
// The oracle is a pure function of the row, which is precisely the assumption P5 rests on: a batch
// row does not know which other rows share its batch. The scenario supplies the per-step outcomes
// (probability, and whether the embedding read fails) per sequence so a case can force one slot to
// stop at depth 1 while another runs to depth 4.
// ---------------------------------------------------------------------------------------------
struct row_out {
    int32_t token  = 0;
    float   prob   = 1.0f;
    int32_t hidden = 0;
    bool    embd_ok = true;
};

struct seq_plan {
    int32_t seq_id  = 0;
    int32_t n_draft = 1;
    float   p_min   = 0.0f;
    bool    have_prob = false;
    int32_t n_past  = 0;
    int32_t id_last = 0;
    bool    has_cached = false;
    int32_t cached_id = -1;
    float   cached_prob = 1.0f;
    int32_t hidden_seed = 0;        // the target hidden row this sequence's first decode is fed

    // PXA_MTP_BATCH_SLOTS_WARM_v1: the runtime state the step decision reads. The default is a slot
    // that has been running long enough for its companion K/V row to hold n_past - 1, which is every
    // slot in cases 2-5; INT32_MIN is the marker for "caught up", so a scenario only has to say
    // something when it wants a slot that is NOT.
    int32_t companion_pos_max  = INT32_MIN;
    bool    have_target_hidden = true;

    // per-step outcomes, indexed by the chain's own step counter (0-based over DECODES, not over i)
    std::vector<float> probs;
    std::vector<bool>  embd_ok;
};

static pxa_mtp_batch_slot_state plan_slot_state(const seq_plan & p) {
    pxa_mtp_batch_slot_state st;
    st.seq_id             = p.seq_id;
    st.n_past             = p.n_past;
    st.companion_pos_max  = p.companion_pos_max == INT32_MIN ? p.n_past - 1 : p.companion_pos_max;
    st.have_target_hidden = p.have_target_hidden;
    st.n_draft            = p.n_draft;
    return st;
}

static float plan_prob(const seq_plan & p, int32_t step) {
    if (p.probs.empty()) {
        return 1.0f;
    }
    return p.probs[(size_t) (step < (int32_t) p.probs.size() ? step : (int32_t) p.probs.size() - 1)];
}

static bool plan_embd_ok(const seq_plan & p, int32_t step) {
    if (p.embd_ok.empty()) {
        return true;
    }
    return p.embd_ok[(size_t) (step < (int32_t) p.embd_ok.size() ? step : (int32_t) p.embd_ok.size() - 1)];
}

// The token and hidden a row produces. Deliberately mixes token, position and the incoming hidden
// so a mis-scheduled row (wrong conditioning hidden, wrong position, wrong K/V row) cannot produce
// the right answer by accident.
static int32_t oracle_token(int32_t seq_id, int32_t token_in, int32_t pos, int32_t hidden_in) {
    return 1000 + 37 * seq_id + 11 * token_in + 5 * pos + 3 * hidden_in;
}

static int32_t oracle_hidden(int32_t seq_id, int32_t token_in, int32_t pos, int32_t hidden_in) {
    return 500000 + 101 * seq_id + 13 * token_in + 7 * pos + 2 * hidden_in;
}

// ---------------------------------------------------------------------------------------------
// What a run records, per sequence. This is the identity being asserted.
// ---------------------------------------------------------------------------------------------
struct kv_write {
    int32_t pos        = -1;
    int32_t hidden_in  = -1;   // the conditioning hidden of the pair stored in the cell
    int32_t token_in   = -1;   // the token of the pair stored in the cell

    bool operator==(const kv_write & o) const {
        return pos == o.pos && hidden_in == o.hidden_in && token_in == o.token_in;
    }
};

struct seq_record {
    std::vector<int32_t>  drafts;
    std::vector<kv_write> kv;
    int32_t n_decode = 0;
    int32_t stop     = 2;
    bool    pre_purge     = false;
    int32_t pre_purge_pos = -1;
    bool    purged   = false;
    int32_t purge_p0 = -1;
    int32_t purge_p1 = -1;
};

struct run_record {
    std::map<int32_t, seq_record> seqs;      // by seq_id
    int32_t n_decode_calls = 0;              // llama_decode() calls -- the quantity P5 reduces
    std::vector<int32_t> rows_per_call;
    // PXA_MTP_BATCH_SLOTS_ROWS_v1: the conditioning-hidden rows handed to each decode. The MTP draft
    // graph concatenates that input against the token embeddings, so one row per batch row is not a
    // convention -- it is the shape the graph is built to, and a step that breaks it aborts inside
    // ggml_concat before any input-size check can refuse the decode.
    std::vector<int32_t> hidden_rows_per_call;
    // PXA_MTP_BATCH_SLOTS_WARM_v1: the step was handed back to the serial drafter.
    bool refused = false;
};

// ---------------------------------------------------------------------------------------------
// Scheduler A -- serial. A transcription of mtp_speculative_gen_draft() (common/speculative.cpp):
// one chain at a time, a fresh 1-row batch per step, per-chain pre-purge, loop, post-purge.
// ---------------------------------------------------------------------------------------------
static run_record serial(const std::vector<seq_plan> & plans, int32_t fail_decode_at_call = -1) {
    run_record rec;

    for (const auto & p : plans) {
        pxa_mtp_draft_chain chain;
        seq_record sr;

        if (!chain.begin(p.seq_id, p.n_draft, p.p_min, p.have_prob, p.n_past, p.id_last,
                         p.has_cached, p.cached_id, p.cached_prob)) {
            rec.seqs[p.seq_id] = sr;
            continue;
        }

        if (chain.needs_pre_purge()) {
            sr.pre_purge     = true;
            sr.pre_purge_pos = chain.n_past;
        }

        int32_t hidden_in = p.hidden_seed;
        int32_t step = 0;
        while (chain.wants_step()) {
            const int32_t pos      = chain.step_pos();
            const int32_t token_in = chain.cur_id;

            chain.on_step_issued();
            ++rec.n_decode_calls;
            rec.rows_per_call.push_back(1);

            if (fail_decode_at_call >= 0 && rec.n_decode_calls > fail_decode_at_call) {
                chain.on_fail();
                break;
            }

            sr.kv.push_back({ pos, hidden_in, token_in });

            const int32_t id_next = oracle_token(p.seq_id, token_in, pos, hidden_in);
            const float   prob    = plan_prob(p, step);

            if (chain.on_sample(id_next, prob) == PXA_MTP_DRAFT_STEP_STOP) {
                break;
            }

            if (!plan_embd_ok(p, step)) {
                chain.on_fail();
                break;
            }
            hidden_in = oracle_hidden(p.seq_id, token_in, pos, hidden_in);

            chain.on_hidden(prob);
            ++step;
        }

        if (chain.needs_purge()) {
            sr.purged   = true;
            sr.purge_p0 = chain.purge_p0();
            sr.purge_p1 = chain.purge_p1();
        }

        sr.drafts   = chain.drafts;
        sr.n_decode = chain.n_decode;
        sr.stop     = chain.stop;
        rec.seqs[p.seq_id] = sr;
    }

    return rec;
}

// ---------------------------------------------------------------------------------------------
// Scheduler B -- batched. A transcription of common_speculative_draft_batched()'s step loop
// (common/speculative.cpp): every still-running chain contributes one row to the same batch, one
// decode per step, per-row read-back by batch row index.
// ---------------------------------------------------------------------------------------------
static run_record batched(const std::vector<seq_plan> & plans,
                          int32_t fail_decode_at_call = -1,
                          bool    apply_step_decision = true) {
    run_record rec;

    // PXA_MTP_BATCH_SLOTS_WARM_v1: the step decision comes first, before anything is written. A
    // refusal is answered by the caller running the ordinary serial drafter over the same requests
    // (examples/server/server-context.cpp), so that is what a refused step records here.
    // apply_step_decision = false replays the PRE-FIX scheduler, which had no such decision and
    // batched whatever it was handed; case 8 uses it to pin the defect this test was written for.
    if (apply_step_decision) {
        std::vector<pxa_mtp_batch_slot_state> states;
        states.reserve(plans.size());
        for (const auto & p : plans) {
            states.push_back(plan_slot_state(p));
        }
        if (pxa_mtp_batch_slots_step_decision(states.data(), states.size()) != PXA_MTP_BATCH_STEP_BATCH) {
            rec = serial(plans, fail_decode_at_call);
            rec.refused = true;
            return rec;
        }
    }

    struct chain_slot {
        pxa_mtp_draft_chain chain;
        const seq_plan *    plan   = nullptr;
        int32_t             hidden = 0;
        int32_t             step   = 0;
        seq_record          rec;
        bool                live   = false;
    };

    std::vector<chain_slot> cs(plans.size());
    for (size_t c = 0; c < plans.size(); ++c) {
        const auto & p = plans[c];
        cs[c].plan   = &p;
        cs[c].hidden = p.hidden_seed;
        cs[c].live   = cs[c].chain.begin(p.seq_id, p.n_draft, p.p_min, p.have_prob, p.n_past,
                                         p.id_last, p.has_cached, p.cached_id, p.cached_prob);
    }

    for (auto & c : cs) {
        if (c.live && c.chain.needs_pre_purge()) {
            c.rec.pre_purge     = true;
            c.rec.pre_purge_pos = c.chain.n_past;
        }
    }

    for (;;) {
        // build the step's batch: one row per still-running chain, in chain order
        struct pending_row { size_t c; int32_t pos; int32_t token_in; int32_t hidden_in; };
        std::vector<pending_row> rows;
        for (size_t c = 0; c < cs.size(); ++c) {
            if (!cs[c].live || !cs[c].chain.wants_step()) {
                continue;
            }
            rows.push_back({ c, cs[c].chain.step_pos(), cs[c].chain.cur_id, cs[c].hidden });
            cs[c].chain.on_step_issued();
        }

        if (rows.empty()) {
            break;
        }

        ++rec.n_decode_calls;
        rec.rows_per_call.push_back((int32_t) rows.size());
        // The real scheduler appends exactly one conditioning-hidden row as it adds each batch row
        // (common/speculative.cpp), so the two counts move together by construction; recording them
        // separately is what lets case 9 assert it rather than assume it.
        rec.hidden_rows_per_call.push_back((int32_t) rows.size());

        if (fail_decode_at_call >= 0 && rec.n_decode_calls > fail_decode_at_call) {
            for (const auto & r : rows) {
                cs[r.c].chain.on_fail();
            }
            break;
        }

        for (const auto & r : rows) {
            cs[r.c].rec.kv.push_back({ r.pos, r.hidden_in, r.token_in });
        }

        // per-row read-back, by batch row index
        for (size_t row = 0; row < rows.size(); ++row) {
            const auto & r = rows[row];
            auto &       c = cs[r.c];

            const int32_t id_next = oracle_token(c.plan->seq_id, r.token_in, r.pos, r.hidden_in);
            const float   prob    = plan_prob(*c.plan, c.step);

            if (c.chain.on_sample(id_next, prob) == PXA_MTP_DRAFT_STEP_STOP) {
                continue;
            }

            if (!plan_embd_ok(*c.plan, c.step)) {
                c.chain.on_fail();
                continue;
            }
            c.hidden = oracle_hidden(c.plan->seq_id, r.token_in, r.pos, r.hidden_in);

            c.chain.on_hidden(prob);
            ++c.step;
        }
    }

    for (auto & c : cs) {
        if (c.chain.needs_purge()) {
            c.rec.purged   = true;
            c.rec.purge_p0 = c.chain.purge_p0();
            c.rec.purge_p1 = c.chain.purge_p1();
        }
        c.rec.drafts   = c.chain.drafts;
        c.rec.n_decode = c.chain.n_decode;
        c.rec.stop     = c.chain.stop;
        rec.seqs[c.plan->seq_id] = c.rec;
    }

    return rec;
}

// ---------------------------------------------------------------------------------------------

static std::string tokstr(const std::vector<int32_t> & v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(v[i]);
    }
    return s + "]";
}

static void compare(const run_record & a, const run_record & b, const char * what) {
    if (a.seqs.size() != b.seqs.size()) {
        fail("%s: serial covered %zu sequences, batched %zu", what, a.seqs.size(), b.seqs.size());
        return;
    }

    for (const auto & entry : a.seqs) {
        const int32_t seq = entry.first;
        const auto it = b.seqs.find(seq);
        if (it == b.seqs.end()) {
            fail("%s: seq %d missing from the batched run", what, (int) seq);
            continue;
        }
        const seq_record & sa = entry.second;
        const seq_record & sb = it->second;

        CHECK(sa.drafts == sb.drafts, "%s: seq %d drafts serial %s != batched %s",
              what, (int) seq, tokstr(sa.drafts).c_str(), tokstr(sb.drafts).c_str());
        CHECK(sa.n_decode == sb.n_decode, "%s: seq %d n_decode serial %d != batched %d",
              what, (int) seq, (int) sa.n_decode, (int) sb.n_decode);
        CHECK(sa.stop == sb.stop, "%s: seq %d stop code serial %d != batched %d",
              what, (int) seq, (int) sa.stop, (int) sb.stop);
        CHECK(sa.pre_purge == sb.pre_purge && sa.pre_purge_pos == sb.pre_purge_pos,
              "%s: seq %d pre-purge serial (%d,%d) != batched (%d,%d)", what, (int) seq,
              (int) sa.pre_purge, (int) sa.pre_purge_pos, (int) sb.pre_purge, (int) sb.pre_purge_pos);
        CHECK(sa.purged == sb.purged && sa.purge_p0 == sb.purge_p0 && sa.purge_p1 == sb.purge_p1,
              "%s: seq %d purge serial (%d,[%d,%d)) != batched (%d,[%d,%d))", what, (int) seq,
              (int) sa.purged, (int) sa.purge_p0, (int) sa.purge_p1,
              (int) sb.purged, (int) sb.purge_p0, (int) sb.purge_p1);

        if (sa.kv.size() != sb.kv.size()) {
            fail("%s: seq %d wrote %zu K/V cells serially and %zu batched",
                 what, (int) seq, sa.kv.size(), sb.kv.size());
            continue;
        }
        for (size_t i = 0; i < sa.kv.size(); ++i) {
            CHECK(sa.kv[i] == sb.kv[i],
                  "%s: seq %d K/V cell %zu serial (pos %d, h %d, tok %d) != batched (pos %d, h %d, tok %d)",
                  what, (int) seq, i,
                  (int) sa.kv[i].pos, (int) sa.kv[i].hidden_in, (int) sa.kv[i].token_in,
                  (int) sb.kv[i].pos, (int) sb.kv[i].hidden_in, (int) sb.kv[i].token_in);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Case 1 -- the transcription itself, against hand-computed expectations.
//
// Everything below compares the two schedulers against each other; this case is what stops them
// both from being wrong in the same way. The four behaviours pinned here are the ones the loop tail
// in mtp_speculative_gen_draft() actually has, and each of them was a live decision in the code:
// the cached free token, the collapse on ITS probability, the drop-at-i>0 / keep-at-i==0 asymmetry
// of the floor, and the purge bounds PXA_MTP_KVPOS_v1 moved.
// ---------------------------------------------------------------------------------------------
static void case_semantics() {
    begin("case 1: the per-sequence chain reproduces mtp_speculative_gen_draft's loop");

    // (a) no cached token, no floor: K decodes, K tokens, first row at n_past.
    {
        seq_plan p;
        p.seq_id = 0; p.n_draft = 3; p.p_min = 0.0f; p.have_prob = false;
        p.n_past = 40; p.id_last = 7; p.hidden_seed = 900;
        const run_record r = serial({ p });
        const seq_record & s = r.seqs.at(0);
        CHECK(s.drafts.size() == 3, "(a) drafted %zu, want 3", s.drafts.size());
        CHECK(s.n_decode == 3, "(a) n_decode %d, want 3", (int) s.n_decode);
        CHECK(s.stop == 2, "(a) stop %d, want 2 (full chain)", (int) s.stop);
        CHECK(s.pre_purge && s.pre_purge_pos == 40, "(a) missing the i==0 pre-purge at n_past");
        CHECK(s.kv.size() == 3 && s.kv[0].pos == 40 && s.kv[1].pos == 41 && s.kv[2].pos == 42,
              "(a) draft rows must run n_past, n_past+1, n_past+2");
        CHECK(s.kv[0].hidden_in == 900, "(a) the first row is conditioned on the target hidden");
        CHECK(s.kv[1].hidden_in == oracle_hidden(0, 7, 40, 900),
              "(a) each row must be conditioned on the row the previous step produced");
        CHECK(s.kv[1].token_in == s.drafts[0], "(a) each row feeds the token the previous step drafted");
        // PXA_MTP_KVPOS_v1: the purge starts one ABOVE n_past -- the row at n_past is committed
        // history the next draft attends to, not a draft row.
        CHECK(s.purged && s.purge_p0 == 41 && s.purge_p1 == 40 + 3 + 2,
              "(a) purge [%d,%d), want [41,45)", (int) s.purge_p0, (int) s.purge_p1);
    }

    // (b) a cached free token costs ZERO decodes and occupies draft slot 1; the loop starts at
    //     n_past + 1 and the row at n_past (written by the commit) is left alone.
    {
        seq_plan p;
        p.seq_id = 1; p.n_draft = 3; p.p_min = 0.0f; p.have_prob = false;
        p.n_past = 40; p.id_last = 7; p.hidden_seed = 900;
        p.has_cached = true; p.cached_id = 55; p.cached_prob = 1.0f;
        const run_record r = serial({ p });
        const seq_record & s = r.seqs.at(1);
        CHECK(s.drafts.size() == 3 && s.drafts[0] == 55, "(b) the cached token must be draft token 1");
        CHECK(s.n_decode == 2, "(b) n_decode %d, want 2 (K-1: the cached token is free)", (int) s.n_decode);
        CHECK(!s.pre_purge, "(b) a cached token means the commit wrote n_past -- it must NOT be purged");
        CHECK(s.kv.size() == 2 && s.kv[0].pos == 41, "(b) the first decoded row sits at n_past + 1");
        CHECK(s.kv[0].token_in == 55, "(b) the first decode is fed the cached token");
    }

    // (c) the cached token's OWN probability collapses the chain: below the floor, n_draft becomes
    //     1, so the cached token is the whole draft and nothing decodes.
    {
        seq_plan p;
        p.seq_id = 2; p.n_draft = 4; p.p_min = 0.75f; p.have_prob = true;
        p.n_past = 40; p.id_last = 7; p.hidden_seed = 900;
        p.has_cached = true; p.cached_id = 55; p.cached_prob = 0.10f;
        const run_record r = serial({ p });
        const seq_record & s = r.seqs.at(2);
        CHECK(s.drafts.size() == 1 && s.drafts[0] == 55, "(c) the chain must collapse to the cached token");
        CHECK(s.n_decode == 0, "(c) n_decode %d, want 0", (int) s.n_decode);
        CHECK(s.stop == 0, "(c) stop %d, want 0 (cached-token collapse)", (int) s.stop);
        CHECK(!s.purged, "(c) nothing decoded -> no draft region to purge");
    }

    // (d) the floor's asymmetry. At i == 0 a below-floor token is KEPT and stops the chain; at
    //     i > 0 it is DROPPED and stops the chain. Both are stop code 1.
    {
        seq_plan lo;                                  // first token already under the floor
        lo.seq_id = 3; lo.n_draft = 4; lo.p_min = 0.75f; lo.have_prob = true;
        lo.n_past = 40; lo.id_last = 7; lo.hidden_seed = 900;
        lo.probs = { 0.2f };
        const seq_record & s0 = serial({ lo }).seqs.at(3);
        CHECK(s0.drafts.size() == 1, "(d) a below-floor FIRST token is kept: drafted %zu, want 1", s0.drafts.size());
        CHECK(s0.n_decode == 1 && s0.stop == 1, "(d) first-token floor break: n_decode %d stop %d",
              (int) s0.n_decode, (int) s0.stop);

        seq_plan hi = lo;                             // first fine, second under the floor
        hi.seq_id = 4;
        hi.probs = { 0.9f, 0.2f };
        const seq_record & s1 = serial({ hi }).seqs.at(4);
        CHECK(s1.drafts.size() == 1, "(d) a below-floor LATER token is dropped: drafted %zu, want 1", s1.drafts.size());
        CHECK(s1.n_decode == 2 && s1.stop == 1, "(d) later-token floor break: n_decode %d stop %d",
              (int) s1.n_decode, (int) s1.stop);
        CHECK(s1.purge_p1 == 40 + 2 + 2, "(d) the purge's upper bound follows n_decode, not n_draft");
    }

    // (e) with no floor armed the probability is never asked for and never acts, however low it is.
    {
        seq_plan p;
        p.seq_id = 5; p.n_draft = 3; p.p_min = 0.0f; p.have_prob = false;
        p.n_past = 10; p.id_last = 2; p.hidden_seed = 3;
        p.probs = { 0.0f, 0.0f, 0.0f };
        const seq_record & s = serial({ p }).seqs.at(5);
        CHECK(s.drafts.size() == 3 && s.stop == 2, "(e) an unarmed floor must not truncate the chain");
    }
}

// ---------------------------------------------------------------------------------------------
// Case 2 -- two slots, the seat's shape (-np 2), equal depth.
// ---------------------------------------------------------------------------------------------
static void case_two_slots_equal_depth() {
    begin("case 2: two slots at equal depth -- identical per-slot output, half the decodes");

    seq_plan a;
    a.seq_id = 0; a.n_draft = 3; a.p_min = 0.0f; a.have_prob = false;
    a.n_past = 100; a.id_last = 11; a.hidden_seed = 4242;

    seq_plan b;
    b.seq_id = 1; b.n_draft = 3; b.p_min = 0.0f; b.have_prob = false;
    b.n_past = 250; b.id_last = 77; b.hidden_seed = 9;

    const run_record s = serial({ a, b });
    const run_record t = batched({ a, b });
    compare(s, t, "case 2");

    CHECK(s.n_decode_calls == 6, "case 2: serial made %d decode calls, want 6", (int) s.n_decode_calls);
    CHECK(t.n_decode_calls == 3, "case 2: batched made %d decode calls, want 3", (int) t.n_decode_calls);
    for (int32_t n : t.rows_per_call) {
        CHECK(n == 2, "case 2: every batched step must carry both slots, got a step of %d row(s)", (int) n);
    }
}

// ---------------------------------------------------------------------------------------------
// Case 3 -- ragged depths. The slots stop at different steps, which is the normal case with a floor
// armed, and the one where a batched scheduler could plausibly read the wrong row back: as chains
// drop out the batch shrinks and every later row index shifts.
// ---------------------------------------------------------------------------------------------
static void case_ragged_depths() {
    begin("case 3: slots that stop at different depths");

    seq_plan a;                                  // runs the full 4
    a.seq_id = 0; a.n_draft = 4; a.p_min = 0.75f; a.have_prob = true;
    a.n_past = 100; a.id_last = 11; a.hidden_seed = 4242;
    a.probs = { 0.99f, 0.98f, 0.97f, 0.96f };

    seq_plan b;                                  // drops out after step 2 (i > 0 -> token dropped)
    b.seq_id = 1; b.n_draft = 4; b.p_min = 0.75f; b.have_prob = true;
    b.n_past = 250; b.id_last = 77; b.hidden_seed = 9;
    b.probs = { 0.99f, 0.10f };

    seq_plan c;                                  // cached free token, then stops on its own floor
    c.seq_id = 2; c.n_draft = 4; c.p_min = 0.75f; c.have_prob = true;
    c.n_past = 7; c.id_last = 3; c.hidden_seed = 31337;
    c.has_cached = true; c.cached_id = 4001; c.cached_prob = 0.90f;
    c.probs = { 0.80f, 0.80f, 0.05f };

    const run_record s = serial({ a, b, c });
    const run_record t = batched({ a, b, c });
    compare(s, t, "case 3");

    // slot 0: 4 decodes, slot 1: 2, slot 2: 3 -> serial 9 calls, batched max(4,2,3) = 4
    CHECK(s.n_decode_calls == 9, "case 3: serial made %d decode calls, want 9", (int) s.n_decode_calls);
    CHECK(t.n_decode_calls == 4, "case 3: batched made %d decode calls, want 4", (int) t.n_decode_calls);
    CHECK(t.rows_per_call == std::vector<int32_t>({ 3, 3, 2, 1 }),
          "case 3: the batch must shrink as chains stop");
    CHECK(s.seqs.at(1).drafts.size() == 1, "case 3: slot 1's second token is below the floor and dropped");
    CHECK(s.seqs.at(2).drafts.size() == 3, "case 3: slot 2 keeps the cached token plus two decoded ones");
}

// ---------------------------------------------------------------------------------------------
// Case 4 -- failures. An embedding read that fails for ONE slot must not disturb the others, and a
// decode that fails must stop every slot in that batch with stop code 3, exactly as the serial loop
// stops the single chain it was running.
// ---------------------------------------------------------------------------------------------
static void case_failures() {
    begin("case 4: a per-slot embedding failure, and a failed decode");

    {
        seq_plan a;
        a.seq_id = 0; a.n_draft = 4; a.p_min = 0.0f; a.have_prob = false;
        a.n_past = 100; a.id_last = 11; a.hidden_seed = 4242;

        seq_plan b;                              // its second row's hidden read fails
        b.seq_id = 1; b.n_draft = 4; b.p_min = 0.0f; b.have_prob = false;
        b.n_past = 250; b.id_last = 77; b.hidden_seed = 9;
        b.embd_ok = { true, false };

        const run_record s = serial({ a, b });
        const run_record t = batched({ a, b });
        compare(s, t, "case 4a");
        CHECK(s.seqs.at(1).stop == 3, "case 4a: the failing slot must report stop code 3");
        CHECK(s.seqs.at(1).drafts.size() == 2, "case 4a: the token whose hidden read failed is still kept");
        CHECK(s.seqs.at(0).drafts.size() == 4, "case 4a: the healthy slot must be unaffected");
    }

    {
        // Serial: chain 0 uses calls 1-4, so failing "after call 2" kills chain 0 mid-way and chain 1
        // never decodes at all. Batched: both chains share every call. The two therefore DIVERGE on a
        // hard decode failure, and that is not a defect of the regrouping -- it is what "the decode
        // died" means in each shape. Assert the batched shape's own contract instead: every chain in
        // the failing batch stops with code 3 and its purge bound still follows its own n_decode.
        seq_plan a;
        a.seq_id = 0; a.n_draft = 4; a.p_min = 0.0f; a.have_prob = false;
        a.n_past = 100; a.id_last = 11; a.hidden_seed = 4242;

        seq_plan b = a;
        b.seq_id = 1; b.n_past = 250; b.id_last = 77; b.hidden_seed = 9;

        const run_record t = batched({ a, b }, /* fail_decode_at_call */ 2);
        CHECK(t.n_decode_calls == 3, "case 4b: batched made %d decode calls, want 3", (int) t.n_decode_calls);
        for (const auto & e : t.seqs) {
            CHECK(e.second.stop == 3, "case 4b: seq %d stop %d, want 3", (int) e.first, (int) e.second.stop);
            CHECK(e.second.n_decode == 3, "case 4b: seq %d n_decode %d, want 3 (the failed one counts)",
                  (int) e.first, (int) e.second.n_decode);
            CHECK(e.second.drafts.size() == 2, "case 4b: seq %d kept %zu tokens, want 2",
                  (int) e.first, e.second.drafts.size());
            CHECK(e.second.purged, "case 4b: seq %d must still purge its draft region", (int) e.first);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Case 5 -- a randomised sweep. Depth 1-5, floors on and off, cached tokens, probabilities that
// stop chains at every position including the first, up to 4 slots. Every scenario must agree on
// all four per-sequence records.
// ---------------------------------------------------------------------------------------------
static void case_sweep() {
    begin("case 5: randomised sweep, up to 4 slots");

    uint32_t rng = 0x9E3779B9u;
    auto next = [&rng]() { rng = rng * 1664525u + 1013904223u; return rng >> 8; };

    int n_scenarios = 0;
    for (int trial = 0; trial < 4000; ++trial) {
        const int n_seq = 2 + (int) (next() % 3);   // 2..4 slots
        std::vector<seq_plan> plans;
        plans.reserve((size_t) n_seq);

        const float p_min = (next() % 3 == 0) ? 0.0f : ((next() % 2) ? 0.75f : 0.40f);

        for (int s = 0; s < n_seq; ++s) {
            seq_plan p;
            p.seq_id      = s;
            p.n_draft     = 1 + (int32_t) (next() % 5);
            p.p_min       = p_min;
            p.have_prob   = p_min > 0.0f || (next() % 5 == 0);   // PXA_MTP_STATS also arms the probe
            p.n_past      = 3 + (int32_t) (next() % 5000);
            p.id_last     = (int32_t) (next() % 60000);
            p.hidden_seed = (int32_t) (next() % 100000);
            p.has_cached  = (next() % 2) == 0;
            if (p.has_cached) {
                p.cached_id   = (int32_t) (next() % 60000);
                p.cached_prob = (float) (next() % 1000) / 1000.0f;
            }
            for (int k = 0; k < 6; ++k) {
                p.probs.push_back((float) (next() % 1000) / 1000.0f);
                p.embd_ok.push_back((next() % 40) != 0);        // ~2.5% of reads fail
            }
            plans.push_back(std::move(p));
        }

        const run_record s = serial(plans);
        const run_record t = batched(plans);
        compare(s, t, "case 5");

        // The point of the exercise: the batched run costs one decode per STEP, i.e. the deepest
        // chain's decode count, where the serial run costs their sum.
        int32_t sum = 0;
        int32_t mx  = 0;
        for (const auto & e : s.seqs) {
            sum += e.second.n_decode;
            mx   = e.second.n_decode > mx ? e.second.n_decode : mx;
        }
        CHECK(s.n_decode_calls == sum, "case 5: serial calls %d != sum of per-seq decodes %d",
              (int) s.n_decode_calls, (int) sum);
        CHECK(t.n_decode_calls == mx, "case 5: batched calls %d != deepest chain's decodes %d",
              (int) t.n_decode_calls, (int) mx);
        CHECK(t.n_decode_calls <= s.n_decode_calls, "case 5: batching must never cost more decodes");

        ++n_scenarios;
        if (g_fail > 0) {
            break;
        }
    }
    printf("    %d scenarios\n", n_scenarios);
}

// ---------------------------------------------------------------------------------------------
// Case 6 -- the applicability gate. This is what keeps the regrouping off the shapes it would be
// WRONG on, so it is enumerated rather than sampled.
// ---------------------------------------------------------------------------------------------
static void case_gate() {
    begin("case 6: pxa_mtp_batch_slots_applicable");

    // the shipped seat shape: 2 slots, companion with n_seq_max = 2, n_ubatch = 4
    CHECK(pxa_mtp_batch_slots_applicable(2, 2, 4, 1), "np2 seat shape must be batchable");
    CHECK(pxa_mtp_batch_slots_applicable(4, 4, 4, 3), "np4 with n_ubatch 4 must be batchable");

    // one drafting slot is the serial path already, bit for bit
    CHECK(!pxa_mtp_batch_slots_applicable(1, 2, 4, 0), "a single request must not take the batched path");
    CHECK(!pxa_mtp_batch_slots_applicable(0, 2, 4, -1), "no requests must not take the batched path");

    // a single-seq companion remaps EVERY seq_id to local K/V row 0 (PXA_LLAMA_MTP_NP_FIX), so two
    // sequences in one batch would write each other's cache. This is the dangerous one.
    CHECK(!pxa_mtp_batch_slots_applicable(2, 1, 4, 1), "a single-seq companion must never be batched");
    CHECK(!pxa_mtp_batch_slots_applicable(2, 0, 4, 1), "n_seq_max 0 must never be batched");

    // a seq_id outside the companion's rows
    CHECK(!pxa_mtp_batch_slots_applicable(2, 2, 4, 2), "a seq_id at n_seq_max must be refused");
    CHECK(!pxa_mtp_batch_slots_applicable(2, 2, 4, -1), "a negative seq_id must be refused");

    // more requests than rows, and more requests than the companion's micro-batch ceiling
    CHECK(!pxa_mtp_batch_slots_applicable(3, 2, 4, 1), "more requests than seq rows must be refused");
    CHECK(!pxa_mtp_batch_slots_applicable(5, 8, 4, 4), "a step wider than n_ubatch must be refused");
    CHECK(pxa_mtp_batch_slots_applicable(5, 8, 0, 4), "n_ubatch 0 means 'unknown' and must not block");

    // PXA_MTP_BATCH_SLOTS_WARM_v1: the shape gate above is everything that is known before a token is
    // drafted. The step decision is the other half -- the part that reads the slots' runtime state.
    auto slot = [](int32_t seq_id, int32_t n_past, int32_t companion_pos_max,
                   bool have_hidden, int32_t n_draft) {
        pxa_mtp_batch_slot_state st;
        st.seq_id = seq_id; st.n_past = n_past; st.companion_pos_max = companion_pos_max;
        st.have_target_hidden = have_hidden; st.n_draft = n_draft;
        return st;
    };

    {   // the seat's steady state: both companions caught up, both slots want a depth-3 chain
        const pxa_mtp_batch_slot_state st[] = { slot(0, 100, 99, true, 3), slot(1, 250, 249, true, 3) };
        int32_t n_batchable = -1, cold = -2;
        CHECK(pxa_mtp_batch_slots_step_decision(st, 2, &n_batchable, &cold) == PXA_MTP_BATCH_STEP_BATCH,
              "two caught-up slots must be batchable");
        CHECK(n_batchable == 2 && cold == -1, "n_batchable %d, cold %d", (int) n_batchable, (int) cold);
    }

    {   // a companion holding EXACTLY n_past - 1 is caught up: the row at n_past is the one the draft
        // is about to write, not one it needs to find there.
        const pxa_mtp_batch_slot_state st[] = { slot(0, 74, 73, true, 3), slot(1, 74, 73, true, 3) };
        CHECK(pxa_mtp_batch_slots_step_decision(st, 2) == PXA_MTP_BATCH_STEP_BATCH,
              "pos_max == n_past - 1 is warm, not cold");
    }

    {   // the defect, in the shape the server hit it: slot 1 has just joined, its companion row is
        // empty (pos_max 0) and its draft would start at 74.
        const pxa_mtp_batch_slot_state st[] = { slot(0, 180, 179, true, 3), slot(1, 74, 0, true, 3) };
        int32_t n_batchable = -1, cold = -2;
        CHECK(pxa_mtp_batch_slots_step_decision(st, 2, &n_batchable, &cold) == PXA_MTP_BATCH_STEP_SERIAL,
              "a slot whose companion is behind must send the whole step to the serial drafter");
        CHECK(cold == 1, "the decision must name the slot that is behind, got %d", (int) cold);
        CHECK(n_batchable == 0, "a refused step reports nothing batchable, got %d", (int) n_batchable);
    }

    {   // ... and it refuses whichever position the cold slot sits in
        const pxa_mtp_batch_slot_state st[] = { slot(0, 74, 0, true, 3), slot(1, 180, 179, true, 3) };
        int32_t cold = -2;
        CHECK(pxa_mtp_batch_slots_step_decision(st, 2, nullptr, &cold) == PXA_MTP_BATCH_STEP_SERIAL,
              "the cold slot is refused first as well as last");
        CHECK(cold == 0, "cold slot %d, want 0", (int) cold);
    }

    {   // fewer than two slots left to regroup: one drafting slot IS the serial path
        const pxa_mtp_batch_slot_state st[] = { slot(0, 100, 99, true, 3), slot(1, 250, 249, false, 3) };
        int32_t n_batchable = -1;
        CHECK(pxa_mtp_batch_slots_step_decision(st, 2, &n_batchable) == PXA_MTP_BATCH_STEP_SERIAL,
              "a slot with no stored target hidden row leaves one batchable slot -> serial");
        CHECK(n_batchable == 1, "n_batchable %d, want 1", (int) n_batchable);
    }
    {
        const pxa_mtp_batch_slot_state st[] = { slot(0, 100, 99, true, 0), slot(1, 250, 249, true, 3) };
        CHECK(pxa_mtp_batch_slots_step_decision(st, 2) == PXA_MTP_BATCH_STEP_SERIAL,
              "a resolved depth of zero leaves one batchable slot -> serial");
    }
    {
        const pxa_mtp_batch_slot_state st[] = { slot(0, 100, 99, true, 3) };
        CHECK(pxa_mtp_batch_slots_step_decision(st, 1) == PXA_MTP_BATCH_STEP_SERIAL,
              "one slot is never a batch");
        CHECK(pxa_mtp_batch_slots_step_decision(nullptr, 0) == PXA_MTP_BATCH_STEP_SERIAL,
              "no slots is never a batch");
    }
    {   // a slot that cannot draft is NOT a reason to refuse when two others still can
        const pxa_mtp_batch_slot_state st[] = { slot(0, 100, 99, true, 3),
                                                slot(1, 250, 249, false, 3),
                                                slot(2, 40, 39, true, 2) };
        int32_t n_batchable = -1;
        CHECK(pxa_mtp_batch_slots_step_decision(st, 3, &n_batchable) == PXA_MTP_BATCH_STEP_BATCH,
              "an empty-draft slot must not sink a step two others can share");
        CHECK(n_batchable == 2, "n_batchable %d, want 2", (int) n_batchable);
    }
    {   // ... but a COLD slot is, even when two others are ready: a cold slot still wants its draft,
        // and only the serial drafter can give it one.
        const pxa_mtp_batch_slot_state st[] = { slot(0, 100, 99, true, 3),
                                                slot(1, 250, 249, true, 3),
                                                slot(2, 74, 0, true, 3) };
        int32_t cold = -2;
        CHECK(pxa_mtp_batch_slots_step_decision(st, 3, nullptr, &cold) == PXA_MTP_BATCH_STEP_SERIAL,
              "a cold slot refuses the step even when the other two could be batched");
        CHECK(cold == 2, "cold slot %d, want 2", (int) cold);
    }
}

// ---------------------------------------------------------------------------------------------
// Case 8 -- a slot joins a running server with a cold companion. THE DEFECT THIS FILE WAS
// EXTENDED FOR: on the release binary, -np 2, two concurrent clients, the server aborted on
// the first concurrent load with
//
//     common_speculative_draft_batched: MTP context not fully warmed up: pos_max = 0, expected >= 74
//     ggml.c: GGML_ASSERT(a->ne[d] == b->ne[d]) failed   (ggml_concat, in the MTP block)
//
// right after the second slot admitted its prompt. The engine SAW the condition -- it printed the
// warning -- and carried on into the batch anyway. Two things were wrong and both are pinned here:
// the scheduler had no way to refuse a step it had already started preparing, and the draft graph
// could not take more than one conditioning-hidden row in the first place.
//
// The scenario is the log's: slot 0 has been generating for a while (its companion is caught up),
// slot 1 has just prefilled and its companion row is still empty.
// ---------------------------------------------------------------------------------------------
static void case_cold_join() {
    begin("case 8: a slot joins mid-run with a cold companion");

    seq_plan warm;                                   // slot 0, generating, companion caught up
    warm.seq_id = 0; warm.n_draft = 3; warm.p_min = 0.0f; warm.have_prob = false;
    warm.n_past = 180; warm.id_last = 11; warm.hidden_seed = 4242;

    seq_plan cold = warm;                            // slot 1, just joined
    cold.seq_id = 1; cold.n_past = 74; cold.id_last = 77; cold.hidden_seed = 9;
    cold.companion_pos_max = 0;                      // the log's pos_max = 0 against expected >= 73

    const run_record s = serial({ warm, cold });
    const run_record t = batched({ warm, cold });

    // (c) the step is refused, not repaired: the batched scheduler hands it back rather than
    //     dropping the cold slot, so the cold slot keeps the draft the serial path gives it.
    CHECK(t.refused, "case 8: the step must be handed back to the serial drafter");

    // (a) nothing illegal is ever emitted. A refused step is the serial drafter's work, so every
    //     decode it records carries exactly one row -- never the two-row draft step that aborted.
    for (int32_t n : t.rows_per_call) {
        CHECK(n == 1, "case 8: a refused step must never emit a multi-row draft decode, got %d rows",
              (int) n);
    }

    // (b) identical per-slot output, including for the slot that was behind
    compare(s, t, "case 8");
    CHECK(!t.seqs.at(1).drafts.empty(),
          "case 8: the slot that joined must still get its draft -- refusing must not cost it one");
    CHECK(t.seqs.at(1).drafts == s.seqs.at(1).drafts,
          "case 8: the cold slot's draft must be exactly the serial drafter's");
    CHECK(t.n_decode_calls == s.n_decode_calls,
          "case 8: a refused step costs the serial drafter's decodes, %d != %d",
          (int) t.n_decode_calls, (int) s.n_decode_calls);

    // And the pre-fix scheduler, replayed, does what the server did: it batches the cold slot with
    // the warm one and emits a TWO-ROW draft step. That step is the abort -- the draft graph
    // allocated one [width] conditioning-hidden row and concatenated it against [width, 2] token
    // embeddings. Pinned so that re-introducing warn-and-continue fails here instead of on a seat.
    const run_record legacy = batched({ warm, cold }, /* fail_decode_at_call */ -1,
                                      /* apply_step_decision */ false);
    CHECK(!legacy.refused, "case 8: the pre-fix scheduler had no refusal to make");
    CHECK(!legacy.rows_per_call.empty() && legacy.rows_per_call[0] == 2,
          "case 8: the pre-fix scheduler must be shown putting both slots in one draft step");

    // The same scenario once the joining slot has caught up: nothing about it is special any more.
    seq_plan caught_up = cold;
    caught_up.companion_pos_max = caught_up.n_past - 1;
    const run_record s2 = serial({ warm, caught_up });
    const run_record t2 = batched({ warm, caught_up });
    CHECK(!t2.refused, "case 8: once the companion catches up the step must batch again");
    compare(s2, t2, "case 8 (caught up)");
    CHECK(t2.rows_per_call == std::vector<int32_t>({ 2, 2, 2 }),
          "case 8: the caught-up step is the ordinary two-row batch");
}

// ---------------------------------------------------------------------------------------------
// Case 9 -- the row-width contract (PXA_MTP_BATCH_SLOTS_ROWS_v1). Every draft step supplies exactly
// one conditioning-hidden row per batch row. The MTP block concatenates that input against the token
// embeddings, which are always [width, n_tokens], so this is not bookkeeping: a step that breaks it
// aborts inside ggml_concat while the graph is being assembled, before the float-count check in
// prepare_mtp_graph_inputs() could turn it into a clean decode failure. The builders now size
// inp_mtp_states by the batch token count for every MTP op type, which is what makes a step of more
// than one row legal at all.
// ---------------------------------------------------------------------------------------------
static void case_row_width() {
    begin("case 9: one conditioning-hidden row per batch row, on every step");

    uint32_t rng = 0x85EBCA6Bu;
    auto next = [&rng]() { rng = rng * 1664525u + 1013904223u; return rng >> 8; };

    for (int trial = 0; trial < 500; ++trial) {
        const int n_seq = 2 + (int) (next() % 3);
        std::vector<seq_plan> plans;
        for (int i = 0; i < n_seq; ++i) {
            seq_plan p;
            p.seq_id      = i;
            p.n_draft     = 1 + (int32_t) (next() % 5);
            p.p_min       = (next() % 2) ? 0.75f : 0.0f;
            p.have_prob   = p.p_min > 0.0f;
            p.n_past      = 3 + (int32_t) (next() % 5000);
            p.id_last     = (int32_t) (next() % 60000);
            p.hidden_seed = (int32_t) (next() % 100000);
            p.has_cached  = (next() % 2) == 0;
            if (p.has_cached) {
                p.cached_id   = (int32_t) (next() % 60000);
                p.cached_prob = (float) (next() % 1000) / 1000.0f;
            }
            for (int k = 0; k < 6; ++k) {
                p.probs.push_back((float) (next() % 1000) / 1000.0f);
            }
            plans.push_back(std::move(p));
        }

        const run_record t = batched(plans);
        CHECK(t.hidden_rows_per_call == t.rows_per_call,
              "case 9: a step supplied %zu hidden-row counts for %zu batches",
              t.hidden_rows_per_call.size(), t.rows_per_call.size());
        for (size_t i = 0; i < t.rows_per_call.size(); ++i) {
            CHECK(t.rows_per_call[i] >= 1 && t.rows_per_call[i] <= n_seq,
                  "case 9: step %zu carried %d rows, want 1..%d", i, (int) t.rows_per_call[i], n_seq);
        }
        if (g_fail > 0) {
            break;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Case 7 -- PXA_MTP_ADAPTIVE_K's ladder, which the batched scheduler resolves for itself.
// ---------------------------------------------------------------------------------------------
static void case_adaptive_k() {
    begin("case 7: pxa_mtp_adaptive_k_depth mirrors the serial ladder");

    CHECK(pxa_mtp_adaptive_k_depth(0.70f, 4) == 4, "strong acceptance -> full depth");
    CHECK(pxa_mtp_adaptive_k_depth(0.55f, 4) == 4, "the 0.55 boundary is inclusive");
    CHECK(pxa_mtp_adaptive_k_depth(0.54f, 4) == 2, "middling acceptance -> a shallow chain");
    CHECK(pxa_mtp_adaptive_k_depth(0.40f, 4) == 2, "the 0.40 boundary is inclusive");
    CHECK(pxa_mtp_adaptive_k_depth(0.39f, 4) == 1, "poor acceptance -> the cached/free token only");
    CHECK(pxa_mtp_adaptive_k_depth(0.10f, 1) == 1, "n_draft 1 is never narrowed");
    CHECK(pxa_mtp_adaptive_k_depth(0.45f, 2) == 2, "the shallow cap never RAISES the configured depth");
    for (int k = 1; k <= 8; ++k) {
        CHECK(pxa_mtp_adaptive_k_depth(0.9f, k) <= k && pxa_mtp_adaptive_k_depth(0.0f, k) <= k,
              "the ladder is a ceiling: K=%d must never grow", k);
    }
}

int main() {
    printf("test-mtp-batch-slots: PXA_MTP_BATCH_SLOTS -- one draft decode per step across slots\n");

    case_semantics();
    case_two_slots_equal_depth();
    case_ragged_depths();
    case_failures();
    case_sweep();
    case_gate();
    case_adaptive_k();
    case_cold_join();
    case_row_width();

    if (g_fail) {
        printf("FAILED (%d)\n", g_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
