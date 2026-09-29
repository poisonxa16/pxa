#pragma once

// bug #216: the parse decision and the two fallbacks of server_slot::update_chat_msg, header-only
// so tests/test-server-chat-parse.cpp can drive them without a server or a model.
//
// 1. Partial vs final. The chat parse is lenient (partial) unless generation ended on EOS, which is
//    upstream's rule. The former code keyed this on `server_slot::stop`, a member that was never
//    initialised or assigned, so which parse ran depended on whatever bytes the slot held.
// 2. Parse failure. A reply the final parse rejects (e.g. a malformed tool call before EOS) is
//    delivered as raw content instead of failing the request (PXA 2026-07-11).
// 3. Diff failure. That raw-content message is not a prefix-extension of what was streamed (the
//    streamed content excludes reasoning and tool-call text), so compute_diffs can throw. The throw
//    must not escape into update_slots(); the step just emits no delta.
// 4. What the slot keeps after a failed diff. A streamed step keeps the message the client already
//    has, so the next step diffs against what was actually sent; keeping the inconsistent message
//    (e.g. one that lost a streamed tool call) made the next step send that call again. The final
//    step keeps the new message: it is the authoritative reply for the non-streamed response.

#include "chat.h"

#include <exception>
#include <string>
#include <vector>

inline bool server_chat_parse_is_partial(bool stopped_eos) {
    return !stopped_eos;
}

// Parse `generated_text`; on a parse error return `prev` with the raw text as its content and set
// `*err` to the reason (left empty on success).
inline common_chat_msg server_chat_parse_or_raw(const std::string & generated_text, bool is_partial,
                                                const common_chat_parser_params & pp,
                                                const common_chat_msg & prev, std::string * err) {
    if (err != nullptr) {
        err->clear();
    }
    try {
        return common_chat_parse(generated_text, is_partial, pp);
    } catch (const std::exception & e) {
        if (err != nullptr) {
            *err = e.what();
        }
        common_chat_msg raw = prev;
        raw.role    = "assistant";
        raw.content = generated_text;
        return raw;
    }
}

// compute_diffs that never throws: on an inconsistent pair it returns no diffs and sets `*err`.
inline std::vector<common_chat_msg_diff> server_chat_diffs_nothrow(const common_chat_msg & prev,
                                                                   const common_chat_msg & cur,
                                                                   std::string * err) {
    if (err != nullptr) {
        err->clear();
    }
    try {
        return common_chat_msg_diff::compute_diffs(prev, cur);
    } catch (const std::exception & e) {
        if (err != nullptr) {
            *err = e.what();
        }
        return {};
    }
}

// The message update_chat_msg keeps in slot.chat_msg after one step (see 4 above).
inline const common_chat_msg & server_chat_msg_to_keep(const common_chat_msg & sent, const common_chat_msg & parsed,
                                                       bool diff_failed, bool is_partial) {
    return (diff_failed && is_partial) ? sent : parsed;
}
