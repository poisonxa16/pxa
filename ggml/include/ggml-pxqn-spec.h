// ggml-pxqn-spec.h -- the engine's side of the speculation selector hook. Open.
//
// The selector (cost model + policy) ships compiled inside libggml-pxqn. The engine only describes what a
// speculation round has on offer and executes the answer; it never sees the model behind it. Without the
// library, or with PXA_SPEC_SELECT=0, ggml_pxqn_spec_available() is false, nothing here is called, and the
// speculation chain runs exactly as its static rows say.
//
// One round per sequence = one choose() (before the verify decode) and, when a draft was verified, one
// observe() (after it). The library timestamps the choose() calls itself, so the interval between two of them
// is the round time of the first one.
#pragma once

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_pxqn_spec_kind {
    GGML_PXQN_SPEC_STATIC = -1,   // no opinion: run the chain as its static rows say
    GGML_PXQN_SPEC_PLAIN  =  0,   // no draft this round
    GGML_PXQN_SPEC_NGRAM  =  1,   // verify the table stage's draft, cut to `width` tokens
    GGML_PXQN_SPEC_MTP    =  2,   // ask the MTP stage for at most `width` tokens
};

#define GGML_PXQN_SPEC_FLAG_PLAIN_OK 1u   // a round without a draft is a legal choice (nothing to keep in step)

struct ggml_pxqn_spec_cfg {
    int32_t  n_seq_max;
    int32_t  has_ngram;    // the chain carries a table (n-gram) stage
    int32_t  has_mtp;      // the chain carries an MTP stage
    int32_t  mtp_n_max;    // configured MTP depth ceiling
    int32_t  ngram_n_max;  // configured table-stage depth ceiling
    uint32_t flags;        // GGML_PXQN_SPEC_FLAG_*
};

struct ggml_pxqn_spec_offer {
    int32_t seq_id;
    int32_t n_active;      // slots decoding this tick (the selector only decides for a lone slot)
    int32_t ngram_len;     // tokens the table stage offers this round (0 = none)
    int32_t mtp_n_max;     // MTP depth ceiling for this request
    int32_t bypass;        // 1 = the engine cannot honour a pick this round (MTP companion not live yet, ...)
};

struct ggml_pxqn_spec_pick {
    int32_t kind;          // enum ggml_pxqn_spec_kind
    int32_t width;         // tokens to draft / verify (PLAIN: 0)
    int32_t probe;         // 1 = this round measures an option, it is not the current best
};

// true when libggml-pxqn carries the selector and PXA_SPEC_SELECT does not turn it off
GGML_API bool  ggml_pxqn_spec_available(void);
GGML_API void * ggml_pxqn_spec_create(const struct ggml_pxqn_spec_cfg * cfg);
GGML_API void   ggml_pxqn_spec_destroy(void * h);
// a new generation starts on seq_id (-1 = every sequence): keep what was learned, forget the round in flight
GGML_API void   ggml_pxqn_spec_reset(void * h, int32_t seq_id);
GGML_API void   ggml_pxqn_spec_choose(void * h, const struct ggml_pxqn_spec_offer * offer, struct ggml_pxqn_spec_pick * pick);
// the target verified n_verified drafted tokens and kept n_accepted of them (n_verified < 0: not known)
GGML_API void   ggml_pxqn_spec_observe(void * h, int32_t seq_id, int32_t n_verified, int32_t n_accepted);
// one line of what the selector measured, for the end-of-request statistics; returns the length written
GGML_API int    ggml_pxqn_spec_describe(void * h, char * buf, size_t n);


// ---- per-model / per-card speculation defaults ---------------------------------------------------
// The engine describes the model it loaded and the cards it runs on; the library answers with the speculation
// shape that was measured best for that cell (which chain, how deep, how confident, whether the self-measuring
// selector drives it, how wide the draft head's vocabulary window is). The table of rows lives in libggml-pxqn.
// Without the library, with PXA_SPEC_DEFAULTS=0, or for a model the table has no row for, every call here says
// "no opinion" and the engine's own static rows run exactly as before. An explicit --spec-type, -md, environment
// value or per-request setting always wins over the answer; only absence is filled.
enum ggml_pxqn_specdef_chain {
    GGML_PXQN_SPECDEF_STATIC  = -1,   // no opinion: the engine's static rows
    GGML_PXQN_SPECDEF_NONE    =  0,   // no speculation
    GGML_PXQN_SPECDEF_NGRAM   =  1,   // the table stage alone
    GGML_PXQN_SPECDEF_MTP     =  2,   // the MTP head alone (a head in the file, or an assistant drafter)
    GGML_PXQN_SPECDEF_CASCADE =  3,   // the table stage in front of the MTP head
};

struct ggml_pxqn_specdef_in {
    char     arch[32];        // general.architecture of the TARGET model ("qwen35", "qwen4exp", "gemma4", ...)
    int32_t  n_dev;           // CUDA devices the model is spread over
    int32_t  cc_min;          // lowest compute capability among them, 100 * major + 10 * minor (600, 610, 700)
    int32_t  tensor_split;    // 1 = a row-parallel split (-sm tensor / attn / graph), 0 = one card or a layer split
    int32_t  has_nextn;       // the file carries MTP layers
    int32_t  has_assistant;   // an assistant drafter (a separate file) is configured
    int32_t  n_expert;        // routed experts per layer (0 = dense)
    int32_t  n_vocab;
    int64_t  headroom_mib;    // free VRAM left after the weights on the tightest card, -1 = unknown
    int32_t  n_ctx;           // 0 = unknown
    char     mtp_expert_type[16]; // ggml type name of the MTP block's routed experts ("PXQN4", "PXQN1", ...); "" = unknown / dense head
};

struct ggml_pxqn_specdef_out {
    int32_t chain;            // enum ggml_pxqn_specdef_chain
    int32_t mtp_n_max;        // MTP depth ceiling to arm, 0 = leave to the engine's row
    float   mtp_p_min;        // MTP confidence floor, < 0 = leave to the engine's row
    int32_t select;           // 1 = the self-measuring selector drives this chain, 0 = it does not, -1 = leave
    int32_t n_ubatch;         // physical batch to run this chain at when the user named none, 0 = leave (a smaller one frees the compute buffer the
                              // per-step recurrent checkpoint rows need: the depth the card can afford)
    int32_t depth_ramp_off;   // 1 = the context-depth ramp (PXA_SPEC_DEPTH_NMAX) is switched off: the confidence floor decides the depth
    int32_t logits_cap;       // 1 = reserve the worst-case logits of a capped row count (PXA_LOGITS_CAP) instead of the whole ubatch, when the
                              // user named none: the other way to free the room the checkpoint rows need, keeping the engine's -ub
    char    why[200];         // one sentence of provenance for the boot banner
};

GGML_API bool    ggml_pxqn_specdef_available(void);
GGML_API bool    ggml_pxqn_specdef_choose(const struct ggml_pxqn_specdef_in * in, struct ggml_pxqn_specdef_out * out);
// the selector's default for a chain that is already resolved: 1 arm it, 0 do not, -1 no opinion
GGML_API int32_t ggml_pxqn_specdef_select(const char * arch, int32_t has_ngram, int32_t has_mtp, int32_t has_assistant,
                                          int32_t n_dev, int32_t cc_min);
// rows of the draft head's vocabulary window (a token-id prefix of the head, a multiple of 64) for a head of
// n_vocab rows of ggml type head_type under the named architecture; 0 = the full head
GGML_API int32_t ggml_pxqn_specdef_shortlist(const char * arch, int32_t n_vocab, int32_t head_type, int32_t n_dev, int32_t cc_min);

#ifdef __cplusplus
}
#endif
