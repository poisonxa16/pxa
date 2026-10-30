#pragma once

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <vector>

enum common_reasoning_budget_state {
    REASONING_BUDGET_IDLE,         // waiting for start sequence
    REASONING_BUDGET_COUNTING,     // counting down tokens
    REASONING_BUDGET_FORCING,      // forcing budget message + end sequence
    REASONING_BUDGET_WAITING_UTF8, // budget exhausted, waiting for UTF-8 completion
    REASONING_BUDGET_DONE,         // passthrough forever
};

// Creates a reasoning budget sampler that limits token generation inside a
// reasoning block (e.g. between <think> and </think>).
//
// State machine: IDLE -> COUNTING -> WAITING_UTF8 -> FORCING -> DONE
//   IDLE:         passthrough, watching for start_tokens sequence
//   COUNTING:     counting down remaining tokens, watching for natural end_tokens
//   WAITING_UTF8: budget exhausted, allowing tokens to complete a UTF-8 sequence
//   FORCING:      forces forced_tokens token-by-token (all other logits -> -inf)
//   DONE:         passthrough forever
//
// Parameters:
//   vocab          - vocabulary (used for UTF-8 boundary detection; can be nullptr)
//   start_tokens   - token sequence that activates counting
//   end_tokens     - token sequence for natural deactivation
//   forced_tokens  - token sequence forced when budget expires
//   budget         - max tokens allowed in the reasoning block
//   initial_state  - initial state
//

struct common_reasoning_budget_ctx * common_reasoning_budget_init(
    const struct llama_vocab * vocab,
    const std::vector<llama_token> & start_tokens,
    const std::vector<llama_token> & end_tokens,
    const std::vector<llama_token> & forced_tokens,
    int32_t                          budget,
    common_reasoning_budget_state    initial_state = REASONING_BUDGET_IDLE);

common_reasoning_budget_state common_reasoning_budget_get_state(const common_reasoning_budget_ctx * smpl);

// PXA 2026-10-26 (#361/#362): the thinking budget a request can still honour in its slot. A budget larger than the
// room left after the prompt can never fire: the thought runs into the end of the context, a context shift (n_keep 1)
// drops the question, and the reply ends with an empty answer (Swift Flash-Next, -c 4096 with --reasoning-budget
// 128000: 28,791 thinking tokens, no content). Fit it to the room minus an answer reserve of min(1024, room/4)
// tokens. budget < 0 (unlimited: no sampler at all), n_ctx <= 0 (unknown) and budgets that already fit come back
// unchanged; no room at all -> 0 (the thought closes at once). The server's off switch: PXA_RBUDGET_FIT_CTX=0.
inline int32_t common_reasoning_budget_fit_ctx(int32_t budget, int32_t n_ctx, int32_t n_prompt) {
    if (budget < 0 || n_ctx <= 0) {
        return budget;
    }
    const int32_t room = n_ctx - std::max<int32_t>(n_prompt, 0);
    if (room <= 0) {
        return 0;
    }
    const int32_t fit = room - std::min<int32_t>(1024, room / 4);
    return budget > fit ? fit : budget;
}
