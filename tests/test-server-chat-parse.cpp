// bug #216: the chat-parse decision and fallbacks of server_slot::update_chat_msg
// (examples/server/server-chat-parse.h), driven the way the server drives them: one lenient parse
// per streamed token, then the final parse keyed on how generation ended. No server, no model.
//
// The former code chose partial vs final from `server_slot::stop`, a member nothing initialised or
// assigned, and let compute_diffs throw out of update_chat_msg. These cases pin the replacement:
// strict only on EOS; a malformed EOS-terminated tool call falls back to raw content; the diff of
// that fallback, which the old code let escape into update_slots(), is contained.

#include "server-chat-parse.h"

#include "chat.h"
#include "testing.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

static std::string trim(const std::string & s) {
    const auto b = s.find_first_not_of(" \n\r\t");
    const auto e = s.find_last_not_of(" \n\r\t");
    return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
}

struct stream_state {
    common_chat_parser_params pp;
    common_chat_msg           msg;       // what update_chat_msg keeps in slot.chat_msg
    int                       parse_failures = 0;
    int                       diff_failures  = 0;
};

static stream_state make_state(testing & t) {
    stream_state st;
    std::ifstream fin("models/templates/Qwen-Qwen3-0.6B.jinja", std::ios::binary);
    std::ostringstream buf; buf << fin.rdbuf();
    const std::string src = buf.str();
    t.assert_true("Qwen3 template loaded", !src.empty());

    common_chat_templates_ptr tmpls(common_chat_templates_init(/* model = */ nullptr, src));
    common_chat_tool weather{
        /* .name        = */ "get_weather",
        /* .description = */ "Get current weather",
        /* .parameters  = */ R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})",
    };
    common_chat_msg user;
    user.role    = "user";
    user.content = "Weather in Atlanta?";
    common_chat_templates_inputs inputs;
    inputs.messages         = { user };
    inputs.tools            = { weather };
    inputs.enable_thinking  = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    const auto params = common_chat_templates_apply(tmpls.get(), inputs);

    // as launch_slot_with_task fills slot.params.chat_parser_params
    st.pp                   = common_chat_parser_params(params);
    st.pp.reasoning_format  = COMMON_REASONING_FORMAT_DEEPSEEK;
    st.pp.parse_tool_calls  = true;
    st.pp.parser.load(params.parser);
    st.msg.role = "assistant";
    return st;
}

// One update_chat_msg step: parse `text` (partial unless the stream ended on EOS), diff it
// against the previous message, keep the new one.
static std::vector<common_chat_msg_diff> step(stream_state & st, const std::string & text, bool stopped_eos,
                                              std::string * parse_err = nullptr, std::string * diff_err = nullptr) {
    std::string pe, de;
    const bool partial = server_chat_parse_is_partial(stopped_eos);
    common_chat_msg prev = st.msg;
    common_chat_msg cur  = server_chat_parse_or_raw(text, partial, st.pp, prev, &pe);
    st.parse_failures += !pe.empty();
    auto diffs = server_chat_diffs_nothrow(prev, cur, &de);
    st.diff_failures += !de.empty();
    st.msg = cur;
    if (parse_err) *parse_err = pe;
    if (diff_err)  *diff_err  = de;
    return diffs;
}

static void stream_all_but_last(stream_state & st, const std::string & text) {
    for (size_t i = 1; i < text.size(); ++i) {
        step(st, text.substr(0, i), /* stopped_eos = */ false);
    }
}

static const std::string k_head = "<think>\nChecking the tool.\n</think>\n\nSure.\n<tool_call>\n";
static const std::string k_call = "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Atlanta\"}}\n</tool_call>";

int main() {
    testing t(std::cout);

    t.test("partial unless EOS", [](testing & t) {
        t.assert_true("EOS -> final parse", !server_chat_parse_is_partial(true));
        t.assert_true("no EOS (limit / word / streaming) -> partial parse", server_chat_parse_is_partial(false));
    });

    t.test("valid call ending on EOS parses strictly", [](testing & t) {
        auto st = make_state(t);
        const std::string text = k_head + k_call;
        stream_all_but_last(st, text);
        std::string pe, de;
        step(st, text, /* stopped_eos = */ true, &pe, &de);
        t.assert_equal("no parse failure", std::string(""), pe);
        t.assert_equal("no diff failure", std::string(""), de);
        t.assert_equal("no failures while streaming", 0, st.parse_failures + st.diff_failures);
        if (t.assert_equal("one tool call", (size_t) 1, st.msg.tool_calls.size())) {
            t.assert_equal("name", std::string("get_weather"), st.msg.tool_calls[0].name);
        }
        t.assert_equal("reasoning", std::string("Checking the tool."), trim(st.msg.reasoning_content));
        t.assert_equal("content", std::string("Sure."), trim(st.msg.content));
    });

    // A complete call followed by stray text, then EOS. Every streamed prefix up to the last token is
    // a valid reply (content "Sure." plus one call); the last token makes the whole reply a hard
    // parse failure. The parser always runs leniently; the partial flag decides only what a hard
    // failure does: partial returns what was parsed so far, final throws. Ending on EOS, the final
    // parse rejects it and the raw text is delivered instead.
    const std::string malformed = k_head + k_call + "\nx";

    t.test("malformed reply ending on EOS falls back to raw content", [&](testing & t) {
        auto st = make_state(t);
        stream_all_but_last(st, malformed);
        t.assert_equal("streaming parsed cleanly", 0, st.parse_failures + st.diff_failures);
        const common_chat_msg streamed = st.msg;
        t.assert_equal("streamed content", std::string("Sure."), trim(streamed.content));
        t.assert_equal("streamed call", (size_t) 1, streamed.tool_calls.size());

        std::string pe, de;
        auto diffs = step(st, malformed, /* stopped_eos = */ true, &pe, &de);
        t.assert_true("final parse rejected", !pe.empty());
        t.assert_equal("raw text delivered as content", malformed, st.msg.content);

        // The fallback is not a prefix-extension of the streamed message (the streamed content has
        // no <think> block), so compute_diffs throws on it. The old update_chat_msg called it
        // unguarded, so this throw left update_chat_msg; the helper contains it.
        bool threw = false;
        try {
            common_chat_msg_diff::compute_diffs(streamed, st.msg);
        } catch (const std::exception &) {
            threw = true;
        }
        t.assert_true("raw compute_diffs throws on the fallback (the pre-fix escape)", threw);
        t.assert_true("helper reports the diff failure", !de.empty());
        t.assert_equal("and emits no delta", (size_t) 0, diffs.size());
    });

    t.test("same reply ending on the limit does not throw", [&](testing & t) {
        auto st = make_state(t);
        stream_all_but_last(st, malformed);
        std::string pe, de;
        step(st, malformed, /* stopped_eos = */ false, &pe, &de);
        t.assert_equal("partial parse returns instead of throwing", std::string(""), pe);
    });

    // bug #222 review: a streamed step whose parse lost a call that was already sent (the diff
    // throws "now finding less tool calls") must not become the slot's message: the next step would
    // diff from the 0-call message and send the same call again under a new header.
    t.test("failed streamed diff keeps the sent message, no duplicate call", [](testing & t) {
        common_chat_msg sent;
        sent.role = "assistant";
        common_chat_tool_call call;
        call.name      = "get_weather";
        call.arguments = R"({"city":"Atlanta"})";
        call.id        = "call_0";
        sent.tool_calls.push_back(call);

        common_chat_msg lost;              // an inconsistent partial parse: the call is gone
        lost.role = "assistant";
        std::string de;
        auto diffs = server_chat_diffs_nothrow(sent, lost, &de);
        t.assert_true("diff against the lost call fails", !de.empty());
        t.assert_equal("no delta", (size_t) 0, diffs.size());

        const common_chat_msg kept = server_chat_msg_to_keep(sent, lost, !de.empty(), /* is_partial = */ true);
        t.assert_equal("streamed step keeps the sent call", (size_t) 1, kept.tool_calls.size());

        // next step parses the call again: nothing new to send
        diffs = server_chat_diffs_nothrow(kept, sent, &de);
        t.assert_equal("next diff ok", std::string(""), de);
        size_t call_deltas = 0;
        for (const auto & d : diffs) {
            call_deltas += d.tool_call_index != std::string::npos;
        }
        t.assert_equal("the call is not sent again", (size_t) 0, call_deltas);

        // the old behaviour for contrast: keeping the lost message resends the call
        diffs = server_chat_diffs_nothrow(lost, sent, &de);
        call_deltas = 0;
        for (const auto & d : diffs) {
            call_deltas += d.tool_call_index != std::string::npos;
        }
        t.assert_equal("keeping the lost message would resend it", (size_t) 1, call_deltas);

        // the final step keeps the new message, even when its diff fails
        const common_chat_msg & fin = server_chat_msg_to_keep(sent, lost, true, /* is_partial = */ false);
        t.assert_equal("final step keeps the authoritative parse", (size_t) 0, fin.tool_calls.size());
    });

    return t.summary();
}
