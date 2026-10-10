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
