#pragma once

// Trust an inherited context-shift discard only when the cache is still that
// shift of this prompt. A shift keeps the first n_keep tokens and drops the
// next n_discarded ones:
//
//   cache[i]           == prompt[i]                        for i < n_keep
//   cache[n_keep + i]  == prompt[n_keep + n_discarded + i] for the suffix
//
// n_keep is 0 when the model does not add a BOS token. It is at least 1 when
// it does. Comparing cache[i] with prompt[n_discarded + i] from i = 0 then
// rejects every such resume, and the slot shifts from scratch.
//
// Header-only so tests/test-pxa-hotswap.cpp can run the Gemma and Qwen cases
// without the server. Prompt and Cache are indexed like server_tokens: a
// non-const operator[] is enough.

#include <algorithm>
#include <cstdint>
#include <string>

template <typename Prompt, typename Cache>
inline bool pxa_shift_keep_matches(Prompt & prompt, int32_t n_prompt,
                                   Cache & cache, int32_t n_cache,
                                   int32_t n_discarded, int32_t n_keep) {
    if (n_discarded <= 0 || n_keep < 0 || n_prompt < 0 || n_cache < 0) return false;
    if (n_keep > n_cache || n_keep > n_prompt) return false;
    for (int32_t i = 0; i < n_keep; ++i) {
        if (prompt[(size_t) i] != cache[(size_t) i]) return false;
    }
    const int32_t suffix_cache = n_cache - n_keep;
    const int32_t suffix_prompt = n_prompt - n_keep - n_discarded;
    if (suffix_cache <= 0 || suffix_prompt <= 0) return false;
    const int32_t ncheck = std::min<int32_t>(32, std::min(suffix_cache, suffix_prompt));
    for (int32_t i = 0; i < ncheck; ++i) {
        if (prompt[(size_t) n_keep + (size_t) n_discarded + (size_t) i]
            != cache[(size_t) n_keep + (size_t) i]) return false;
    }
    return true;
}

// PXA_MTMD_SHIFT_SNAP_v1 (bug #379): a prompt shift drops tokens [n_keep, n_keep + n_discard). The one-step
// n_discard is moved off image/audio chunks by adjust_n_to_support_context_shift, but a prompt longer than about
// 1.5 x n_ctx drops several steps, and that end can land INSIDE a media chunk: the chunk's map entry goes with the
// dropped part, its tail placeholders stay, and the first prompt batch throws "Chunk not found, or idx is not the
// first token of a chunk" (llama-server abort -6). The same holds for the cache tokens of a restored prompt.
// Move the end forward while it sits on a media placeholder that is not a chunk start, so the cut drops the whole
// chunk. An end on a text token or on a chunk start is left as it is (every case that worked before is unchanged).
// is_null(i): token i is a media placeholder (LLAMA_TOKEN_NULL); is_start(i): a media chunk starts at i.
template <typename IsNull, typename IsStart>
inline int32_t pxa_shift_snap_discard(int32_t n_tokens, int32_t n_keep, int32_t n_discard,
                                      IsNull is_null, IsStart is_start) {
    if (n_discard <= 0 || n_keep < 0) return n_discard;
    int32_t end = n_keep + n_discard;
    while (end < n_tokens && is_null(end) && !is_start(end)) ++end;
    return end - n_keep;
}

// PXA_MTMD_NO_PROMPT_SHIFT_v1 (bug #379, decision 11-05): a NEW prompt that carries media and
// does not fit the slot is refused with HTTP 400 instead of being shifted. A shift with n_keep 0 drops the
// system prompt, the question and the user header, and the model then answers with EOS at once (measured:
// SmolVLM2-2.2B, 5 images at -c 4096 -> a 1-token empty reply). Mainline refuses context shift with mtmd.
// Text-only prompts keep the old truncation; the snap above stays for the cache-restore and mid-generation
// shift paths.
inline bool pxa_mtmd_prompt_rejected(bool has_media, int32_t ga_n, int32_t n_prompt, int32_t n_ctx) {
    return has_media && ga_n == 1 && n_prompt >= n_ctx;
}

inline std::string pxa_mtmd_overflow_msg(int32_t n_prompt, int32_t n_ctx) {
    return "the images in this chat need " + std::to_string(n_prompt) + " tokens but Context is " +
           std::to_string(n_ctx) + "; raise Context (Control: model settings) or send fewer/smaller images";
}
