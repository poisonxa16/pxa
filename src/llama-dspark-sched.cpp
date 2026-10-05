// DSpark speculation scheduler (M5).
//
// A transcription of ds4's ds4_session_dspark_scheduler_{should_skip,note,reset}
// (ds4.c:47792-48013 in the box copy). The constants, the ORDER of the three pause
// branches, and the integer-milli comparisons are deliberately identical: this is the
// governor antirez shipped because DSpark *loses* on low-acceptance traffic, and a
// re-derived approximation of it would be an unmeasurable difference from the reference.
//
// This is not an optimisation. Without it a low-acceptance regime pays the drafter
// forward plus the wider verify forward on every single token and never earns either
// back; the scheduler is what makes "DSpark is not worth it here" cost ~0 instead of
// costing 30%.
//
// Env knobs mirror ds4's names with a PXA_ prefix; defaults are ds4's defaults.

#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static uint32_t dspark_env_u32(const char * name, uint32_t dflt) {
    const char * e = getenv(name);
    if (!e || !e[0]) {
        return dflt;
    }
    char * end = nullptr;
    const unsigned long v = strtoul(e, &end, 10);
    if (end == e) {
        return dflt;
    }
    return (uint32_t) v;
}

struct llama_dspark_sched {
    // tunables, latched at init so a mid-run env change cannot make two cycles
    // incomparable
    uint32_t window                 = 4;
    uint32_t skip_cycles            = 2;
    uint32_t slow_skip_cycles       = 4;
    uint32_t min_avg_milli          = 1500;
    uint32_t max_ms_per_accept_milli= 28000;
    uint32_t max_extra_saved_ratio_milli = 1000;
    uint32_t break_even_window      = 0;
    uint32_t no_draft_skip          = 3;
    uint32_t short_accept_no_draft_skip = 4;
    uint32_t cold_low_conf_skip     = 7;
    uint32_t tail_min_tokens        = 10;
    float    cold_low_conf_thresh   = 0.5f;
    bool     enabled                = true;
    bool     log                    = false;

    // window accumulators
    uint32_t cycles   = 0;
    uint32_t accepted = 0;
    uint32_t no_draft = 0;
    double   extra_ms = 0.0;
    double   saved_ms = 0.0;

    // lifetime
    uint32_t skip               = 0;
    bool     skipped_cycle      = false;
    uint32_t lifetime_accepted  = 0;
    bool     long_accept_seen   = false;

    // counters the caller reports
    uint32_t n_sched_skips = 0;
    uint32_t n_tail_skips  = 0;
};

static void sched_reset_window(llama_dspark_sched * s) {
    s->cycles   = 0;
    s->accepted = 0;
    s->no_draft = 0;
    s->extra_ms = 0.0;
    s->saved_ms = 0.0;
}

llama_dspark_sched * llama_dspark_sched_init(void) {
    llama_dspark_sched * s = new llama_dspark_sched();
    const char * en = getenv("PXA_DSPARK_SCHED");
    s->enabled = !(en && en[0] && strcmp(en, "0") == 0);
    s->log     = getenv("PXA_DSPARK_SPEC_LOG") != nullptr;

    s->window                  = dspark_env_u32("PXA_DSPARK_SCHED_WINDOW", 4);
    if (s->window == 0) s->window = 4;
    s->skip_cycles             = dspark_env_u32("PXA_DSPARK_SCHED_SKIP", 2);
    s->slow_skip_cycles        = dspark_env_u32("PXA_DSPARK_SCHED_SLOW_SKIP", 4);
    s->min_avg_milli           = dspark_env_u32("PXA_DSPARK_SCHED_MIN_AVG_MILLI", 1500);
    s->max_ms_per_accept_milli = dspark_env_u32("PXA_DSPARK_SCHED_MAX_MS_PER_ACCEPT_MILLI", 28000);
    s->max_extra_saved_ratio_milli =
                                 dspark_env_u32("PXA_DSPARK_SCHED_MAX_EXTRA_SAVED_RATIO_MILLI", 1000);
    s->break_even_window       = dspark_env_u32("PXA_DSPARK_SCHED_BREAK_EVEN_WINDOW", 0);
    s->no_draft_skip           = dspark_env_u32("PXA_DSPARK_SCHED_NO_DRAFT_SKIP", 3);
    s->short_accept_no_draft_skip =
                                 dspark_env_u32("PXA_DSPARK_SCHED_SHORT_ACCEPT_NO_DRAFT_SKIP", 4);
    s->cold_low_conf_skip      = dspark_env_u32("PXA_DSPARK_SCHED_COLD_LOW_CONFIDENCE_SKIP", 7);
    s->tail_min_tokens         = dspark_env_u32("PXA_DSPARK_SCHED_TAIL_MIN_TOKENS", 10);
    s->cold_low_conf_thresh    = (float) dspark_env_u32("PXA_DSPARK_SCHED_COLD_LOW_CONFIDENCE_MILLI", 500)
                                 / 1000.0f;
    return s;
}

void llama_dspark_sched_free(llama_dspark_sched * s) {
    delete s;
}

bool llama_dspark_sched_should_skip(llama_dspark_sched * s) {
    if (!s || !s->enabled) {
        return false;
    }
    s->skipped_cycle = false;
    if (s->skip == 0) {
        return false;
    }
    s->skip--;
    s->skipped_cycle = true;
    s->n_sched_skips++;
    if (s->log) {
        fprintf(stderr, "dspark: scheduler skip remaining=%u\n", s->skip);
    }
    return true;
}

// ds4 checks the tail rule in the CALLER, before the proposal is prepared
// (ds4.c:64258-64274), not inside note(). Keeping it a separate call preserves that: a
// tail skip must not be recorded as a scheduler pause, or the window statistics blame the
// governor for the end of a generation.
bool llama_dspark_sched_tail_skip(llama_dspark_sched * s, int n_remaining) {
    if (!s || !s->enabled || s->tail_min_tokens == 0) {
        return false;
    }
    if (n_remaining < 0) {
        return false;                  // unknown budget: never tail-skip
    }
    if ((uint32_t) n_remaining >= s->tail_min_tokens) {
        return false;
    }
    s->n_tail_skips++;
    return true;
}

void llama_dspark_sched_note(llama_dspark_sched * s,
                             uint32_t accepted_drafts,
                             bool     no_draft,
                             double   extra_ms,
                             double   last_target_eval_ms,
                             bool     conf0_valid,
                             float    conf0) {
    if (!s || !s->enabled) {
        return;
    }
    // A cycle the scheduler itself skipped carries no information about acceptance.
    if (s->skipped_cycle) {
        s->skipped_cycle = false;
        return;
    }

    s->cycles++;
    s->accepted += accepted_drafts;
    if (accepted_drafts != 0) {
        if (s->lifetime_accepted <= UINT32_MAX - accepted_drafts) {
            s->lifetime_accepted += accepted_drafts;
        } else {
            s->lifetime_accepted = UINT32_MAX;
        }
        if (accepted_drafts > 2u) {
            s->long_accept_seen = true;
        }
    }
    if (no_draft) {
        s->no_draft++;
    }
    if (extra_ms > 0.0 && std::isfinite(extra_ms)) {
        s->extra_ms += extra_ms;
    }
    if (accepted_drafts != 0 && last_target_eval_ms > 0.0 && std::isfinite(last_target_eval_ms)) {
        s->saved_ms += last_target_eval_ms * (double) accepted_drafts;
    }

    // (1) the no-draft ladder
    if (no_draft && s->no_draft_skip != 0) {
        uint32_t skip = s->no_draft_skip;
        if (s->lifetime_accepted != 0 && !s->long_accept_seen) {
            if (skip < s->short_accept_no_draft_skip) {
                skip = s->short_accept_no_draft_skip;
            }
        } else if (s->lifetime_accepted == 0 && conf0_valid && conf0 <= s->cold_low_conf_thresh) {
            if (skip < s->cold_low_conf_skip) {
                skip = s->cold_low_conf_skip;
            }
        }
        if (s->skip < skip) {
            s->skip = skip;
        }
        if (s->log) {
            fprintf(stderr, "dspark: scheduler no-draft pause skip=%u accepted_total=%u "
                            "long_accept=%d confidence0=%s%.3f\n",
                    s->skip, s->lifetime_accepted, s->long_accept_seen ? 1 : 0,
                    conf0_valid ? "" : "n/a:", (double) conf0);
        }
    }

    const bool measured_unprofitable =
        s->max_extra_saved_ratio_milli != 0 &&
        s->accepted != 0 &&
        s->saved_ms > 0.0 &&
        s->extra_ms * 1000.0 > s->saved_ms * (double) s->max_extra_saved_ratio_milli;

    // (2) the break-even rule, which RESETS THE WINDOW AND RETURNS (off by default)
    if (s->break_even_window != 0 && s->cycles >= s->break_even_window && measured_unprofitable) {
        s->skip = s->slow_skip_cycles;
        if (s->log) {
            fprintf(stderr, "dspark: scheduler break-even pause cycles=%u accepted=%u "
                            "saved=%.3fms extra=%.3fms skip=%u\n",
                    s->cycles, s->accepted, s->saved_ms, s->extra_ms, s->skip);
        }
        sched_reset_window(s);
        return;
    }

    if (s->cycles < s->window) {
        return;
    }

    // (3) the four-condition window test
    const uint64_t avg_milli = ((uint64_t) s->accepted * 1000ull) / (uint64_t) s->cycles;
    const bool low_accept    = avg_milli < s->min_avg_milli;
    const bool many_no_draft = s->no_draft * 2u >= s->cycles;
    const double extra_per_accept_ms = s->accepted != 0 ? s->extra_ms / (double) s->accepted : 0.0;
    const bool slow_accept =
        s->max_ms_per_accept_milli != 0 &&
        s->accepted != 0 &&
        extra_per_accept_ms * 1000.0 > (double) s->max_ms_per_accept_milli;

    if (low_accept || many_no_draft || slow_accept || measured_unprofitable) {
        s->skip = s->skip_cycles;
        if (many_no_draft || slow_accept || measured_unprofitable) {
            if (s->skip < s->slow_skip_cycles) {
                s->skip = s->slow_skip_cycles;
            }
        }
        if (s->log) {
            fprintf(stderr, "dspark: scheduler pause cycles=%u accepted=%u avg=%.3f no_draft=%u "
                            "extra_per_accept=%.3fms saved=%.3fms extra=%.3fms skip=%u\n",
                    s->cycles, s->accepted, (double) avg_milli/1000.0, s->no_draft,
                    extra_per_accept_ms, s->saved_ms, s->extra_ms, s->skip);
        }
    }
    sched_reset_window(s);
}

void llama_dspark_sched_get(const llama_dspark_sched * s, struct llama_dspark_sched_state * out) {
    if (!s || !out) {
        return;
    }
    out->skip_remaining    = s->skip;
    out->cycles            = s->cycles;
    out->accepted          = s->accepted;
    out->no_draft          = s->no_draft;
    out->extra_ms          = s->extra_ms;
    out->saved_ms          = s->saved_ms;
    out->lifetime_accepted = s->lifetime_accepted;
    out->long_accept_seen  = s->long_accept_seen;
    out->n_sched_skips     = s->n_sched_skips;
    out->n_tail_skips      = s->n_tail_skips;
    out->enabled           = s->enabled;
}

// The accept rule, in one place so the harness, the server and any future caller cannot
// drift. EXACT greedy-argmax prefix matching, ds4.c:61096-61118: row i-1 of the verify
// batch predicts draft token i, and the first mismatch terminates the prefix. There is no
// probabilistic acceptance in DSpark and adding one would break its identity guarantee.
int llama_dspark_accept_prefix(const llama_token * draft,
                               int                 n_draft,
                               const llama_token * row_tops,
                               int                 n_rows) {
    if (n_draft <= 0 || n_rows < 1) {
        return 0;
    }
    // row_tops[0] is the argmax of the row that carried the last committed token, so it
    // predicts draft[0]; row_tops[i] predicts draft[i+1].
    int commit = 0;
    for (int i = 0; i < n_draft && i < n_rows; ++i) {
        if (row_tops[i] != draft[i]) {
            break;
        }
        ++commit;
    }
    return commit;
}
