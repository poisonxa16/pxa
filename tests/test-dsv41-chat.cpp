// DeepSeek-V4.1 chat template check.
//
//   test-dsv41-chat TEMPLATE.jinja ENCODING_TESTS_DIR
//
// ENCODING_TESTS_DIR holds the reference encoder's own golden pairs
// (test_input_N.json / test_output_N.txt from the V4.1 release's encoding/tests).
// They are read in place, not copied into this repo. Each case is rendered by
// our jinja engine and must equal the golden text byte for byte:
//   A: tools passed as the top-level `tools` variable (the llama.cpp route),
//   B: tools on messages[0] (the reference encoder's route; only cases with tools),
//   C: the full common_chat_templates_apply route (cases without tools/tasks;
//      checks detection, the generation prompt and the DSML handler's tags).
// Also runs the reference unit-test cases (no files needed). Missing directory -> golden part skipped.

#include "chat.h"
#include "common.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static const std::string BOS = "<｜begin▁of▁sentence｜>";

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string render_direct(const std::string & tmpl, const json & input) {
    jinja::lexer lexer;
    auto toks = lexer.tokenize(tmpl);
    jinja::program ast = jinja::parse_from_tokens(toks);
    jinja::context ctx(tmpl);
    jinja::global_from_json(ctx, input, true);
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(ast);
    auto parts = jinja::runtime::gather_string_parts(results);
    std::string out;
    for (const auto & part : parts->as_string().parts) {
        out += part.val;
    }
    return out;
}

// same as the chat.cpp workaround: JSON-string tool arguments become objects
static void args_to_objects(json & messages) {
    for (auto & m : messages) {
        if (!m.contains("tool_calls")) {
            continue;
        }
        for (auto & tc : m["tool_calls"]) {
            if (tc.contains("function") && tc["function"].contains("arguments") && tc["function"]["arguments"].is_string()) {
                try {
                    tc["function"]["arguments"] = json::parse(tc["function"]["arguments"].get<std::string>());
                } catch (...) {
                    // left as a string: the template renders it as one "arguments" parameter
                }
            }
        }
    }
}

static bool compare(const std::string & name, const std::string & got, const std::string & want) {
    if (got == want) {
        printf("PASS %s (%zu bytes)\n", name.c_str(), got.size());
        return true;
    }
    size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) {
        i++;
    }
    const size_t from = i > 60 ? i - 60 : 0;
    printf("FAIL %s: first difference at byte %zu (got %zu bytes, want %zu)\n", name.c_str(), i, got.size(), want.size());
    printf("  want: ...%s\n", want.substr(from, 160).c_str());
    printf("  got : ...%s\n", got.substr(from, 160).c_str());
    return false;
}

// Cases from the reference encoder's unit tests (encoding/test_encoding.py), expectations
// as asserted there. Each check: kind '=' equal, '+' contains, '-' does not contain, '$' ends with, '!' throws.
struct unit_check { char kind; std::string text; };
struct unit_case { std::string name; json input; std::vector<unit_check> checks; };

static json tool_lookup() {
    return json::parse(R"({"type": "function", "function": {"name": "lookup", "description": "Look up a value",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}}}})");
}
static json tool_call_msgs() {
    return json::parse(R"([{"role": "user", "content": "question"},
        {"role": "assistant", "reasoning_content": "  reason  ", "content": "summary",
         "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {"query": "value", "limit": 2}}}]}])");
}
static std::string effort_prefix(int b) {
    return "<｜System｜>Reasoning Effort: " + std::to_string(b) + " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
}
static const std::string TC_OUT =
    "  reason  </think>summary\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n"
    "<｜DSML｜ parameter name=\"query\" string=\"true\">value</｜DSML｜ parameter>\n"
    "<｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n"
    "</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>";

// json.dumps() with Python's default separators (", " and ": "), ensure_ascii=False
static std::string py_dumps(const json & j) {
    if (j.is_object()) {
        std::string out = "{";
        bool first = true;
        for (const auto & [k, v] : j.items()) {
            out += (first ? "" : ", ") + json(k).dump(-1, ' ', false) + ": " + py_dumps(v);
            first = false;
        }
        return out + "}";
    }
    if (j.is_array()) {
        std::string out = "[";
        for (size_t i = 0; i < j.size(); i++) {
            out += (i ? ", " : "") + py_dumps(j[i]);
        }
        return out + "]";
    }
    return j.dump(-1, ' ', false);
}

static std::vector<unit_case> unit_cases() {
    std::vector<unit_case> u;
    auto base = [](const std::string & mode, const json & msgs) {
        return json{{"bos_token", BOS}, {"add_generation_prompt", true}, {"thinking_mode", mode}, {"messages", msgs}};
    };
    const json q = json::parse(R"([{"role": "user", "content": "question"}])");
    u.push_back({"images", base("chat", json::parse(R"([{"role": "user", "content": [{"type": "text", "text": "inspect"},
        {"type": "image_url", "image_url": {"url": "/unused/image.png"}}]}])")),
        {{'=', BOS + "<｜User｜>inspect\n\n<｜deepseek_image｜><｜Assistant｜></think>"}}});
    for (auto [e, b] : std::vector<std::pair<json, int>>{{"low", 50}, {"high", 75}, {"max", 100}, {42, 42}, {100, 100}, {nullptr, 75}}) {
        json in = base("thinking", q);
        if (!e.is_null()) { in["reasoning_effort"] = e; }
        u.push_back({"effort-" + e.dump(), in, {{'=', BOS + effort_prefix(b) + "<｜User｜>question<｜Assistant｜><think>"}}});
    }
    for (json e : {json(-1), json(0), json(101), json("medium"), json(true), json(false), json(1.5)}) {
        json in = base("thinking", q);
        in["reasoning_effort"] = e;
        u.push_back({"effort-reject-" + e.dump(), in, {{'!', ""}}});
    }
    {
        json in = base("chat", json::parse(R"([{"role": "user", "content": "hello"}])"));
        in["reasoning_effort"] = "max";
        u.push_back({"chat-no-effort", in, {{'=', BOS + "<｜User｜>hello<｜Assistant｜></think>"}}});
    }
    u.push_back({"leading-system", base("chat", json::parse(R"([{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"}])")),
        {{'=', BOS + "<｜System｜>You are a helpful assistant.<｜User｜>hello<｜Assistant｜></think>"}}});
    {
        json in = base("thinking", json::parse(R"([{"role": "system", "content": "sys"}, {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"}, {"role": "system", "content": "mid sys"}])"));
        in["reasoning_effort"] = 88;
        u.push_back({"mid-system", in, {{'=', BOS + effort_prefix(88) +
            "sys<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜><｜System｜>mid sys<｜Assistant｜><think>"}}});
    }
    {
        json msgs = json::array({{{"role", "system"}, {"content", "system"}, {"tools", json::array({tool_lookup()})}},
                                 {{"role", "user"}, {"content", "question"}}});
        u.push_back({"tool-instructions", base("chat", msgs), {
            {'+', "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" "
                  "string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>"},
            {'-', "<｜DSML｜tool_calls>"}, {'-', "<｜DSML｜invoke"}, {'-', "<｜DSML｜parameter"}}});
    }
    u.push_back({"tool-call", base("thinking", tool_call_msgs()), {{'=', BOS + effort_prefix(75) + "<｜User｜>question<｜Assistant｜><think>" + TC_OUT}}});
    for (std::string loc : {"tool", "function"}) {
        for (json ns : {json("search"), json::parse(R"({"name": "search", "description": "Search tools."})")}) {
            json t = tool_lookup();
            (loc == "tool" ? t : t["function"])["namespace"] = ns;
            json schema = tool_lookup()["function"];
            schema["name"] = "search::lookup";
            if (ns.is_object()) { schema["description"] = "Search tools.\nLook up a value"; }
            json msgs = json::array({{{"role", "system"}, {"content", "system"}, {"tools", json::array({t})}}});
            u.push_back({"ns-schema-" + loc + "-" + (ns.is_object() ? "obj" : "str"), base("chat", msgs),
                {{'+', py_dumps(schema)}, {'-', "\"namespace\":"}}});
        }
    }
    for (std::string mode : {"chat", "thinking"}) {
        for (std::string loc : {"tool", "function", "qualified_name"}) {
            json msgs = tool_call_msgs();
            json & call = msgs[1]["tool_calls"][0];
            if (loc == "qualified_name") { call["function"]["name"] = "search::lookup"; }
            else { (loc == "tool" ? call : call["function"])["namespace"] = "search"; }
            std::string want = TC_OUT;
            want.replace(want.find("name=\"lookup\""), 13, "name=\"search::lookup\"");
            if (mode == "chat") { want = want.substr(want.find("</think>") + 8); }
            u.push_back({"ns-call-" + mode + "-" + loc, base(mode, msgs), {{'$', "<｜Assistant｜>" + std::string(mode == "thinking" ? "<think>" : "</think>") + want}}});
        }
    }
    {
        json msgs = json::parse(R"([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "reasoning_content": "r2", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "value"}])");
        u.push_back({"drop-thinking", base("thinking", msgs), {
            {'+', "<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>"}, {'-', "r1"}, {'+', "<think>r2</think>"},
            {'$', "<tool_result>value</tool_result><｜Assistant｜><think>"}}});
        msgs.push_back({{"role", "user"}, {"content", "q3"}});
        u.push_back({"drop-thinking-next", base("thinking", msgs), {{'-', "r2"}, {'$', "q3<｜Assistant｜><think>"}}});
    }
    u.push_back({"task-query", base("chat", json::parse(R"([{"role": "user", "content": "classify me", "task": "query"}])")),
        {{'$', "classify me<｜query｜>"}, {'-', "<｜Assistant｜>"}}});
    return u;
}

static int run_units(const std::string & tmpl, int & n_pass) {
    int n_fail = 0;
    for (const auto & uc : unit_cases()) {
        std::string out;
        bool threw = false;
        try {
            out = render_direct(tmpl, uc.input);
        } catch (const std::exception &) {
            threw = true;
        }
        bool ok = true;
        for (const auto & ck : uc.checks) {
            switch (ck.kind) {
                case '!': ok = ok && threw; break;
                case '=': ok = ok && !threw && out == ck.text; break;
                case '+': ok = ok && !threw && out.find(ck.text) != std::string::npos; break;
                case '-': ok = ok && !threw && out.find(ck.text) == std::string::npos; break;
                case '$': ok = ok && !threw && out.size() >= ck.text.size() && out.compare(out.size() - ck.text.size(), ck.text.size(), ck.text) == 0; break;
            }
        }
        if (ok) {
            n_pass++;
        } else {
            n_fail++;
            printf("FAIL unit %s%s\n  got: %s\n", uc.name.c_str(), threw ? " (threw)" : "", out.c_str());
        }
    }
    return n_fail;
}

// Output parsing through the chat handler the template is detected as (DSML, V4.1 spacing).
static int run_parse(const common_chat_templates * tmpls, int & n_pass) {
    int n_fail = 0;
    common_chat_tool get_time { "get_time", "Get the time in a city",
        R"({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})" };
    common_chat_tool magic_int { "magic_int", "Magic",
        R"({"type": "object", "properties": {"ref": {"type": "integer"}, "name": {"type": "string"}}, "required": ["ref"]})" };
    struct pcase { std::string name; bool thinking; std::vector<common_chat_tool> tools; std::string out;
                   std::string want_reasoning; std::string want_content; std::vector<std::pair<std::string, std::string>> want_calls; };
    const std::vector<pcase> cases = {
        { "content", false, {}, "Hello, world!\nWhat's up?", "", "Hello, world!\nWhat's up?", {} },
        { "thinking", true, {}, "I'm\nthinking</think>Hello", "I'm\nthinking", "Hello", {} },
        { "call", true, { get_time },
          "Let me check</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_time\">\n"
          "<｜DSML｜ parameter name=\"city\" string=\"true\">Tokyo</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
          "Let me check", "", { { "get_time", R"({"city":"Tokyo"})" } } },
        { "call-mixed", true, { magic_int },
          "Multi</think>Calling.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"magic_int\">\n"
          "<｜DSML｜ parameter name=\"ref\" string=\"false\">42</｜DSML｜ parameter>\n"
          "<｜DSML｜ parameter name=\"name\" string=\"true\">foo bar</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
          "Multi", "Calling.", { { "magic_int", R"({"ref":42,"name":"foo bar"})" } } },
        { "v4-tags-are-text", false, { get_time },
          "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_time\">\n"
          "<｜DSML｜parameter name=\"city\" string=\"true\">Tokyo</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
          "", "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_time\">\n"
          "<｜DSML｜parameter name=\"city\" string=\"true\">Tokyo</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>", {} },
    };
    for (const auto & pc : cases) {
        bool ok = true;
        std::string why;
        try {
            common_chat_templates_inputs in;
            common_chat_msg user;
            user.role = "user";
            user.content = "question";
            in.messages = { user };
            in.tools = pc.tools;
            in.add_generation_prompt = true;
            in.enable_thinking = pc.thinking;
            in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            in.parallel_tool_calls = true;
            auto params = common_chat_templates_apply(tmpls, in);
            common_chat_parser_params pp(params);
            pp.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            pp.parser.load(params.parser);
            auto msg = common_chat_parse(pc.out, false, pp);
            // compared after strip, as tests/test-chat.cpp does
            if (string_strip(msg.reasoning_content) != pc.want_reasoning) { ok = false; why += " reasoning=[" + msg.reasoning_content + "]"; }
            if (string_strip(msg.content) != pc.want_content) { ok = false; why += " content=[" + msg.content + "]"; }
            if (msg.tool_calls.size() != pc.want_calls.size()) {
                ok = false; why += " n_calls=" + std::to_string(msg.tool_calls.size());
            } else {
                for (size_t i = 0; i < msg.tool_calls.size(); i++) {
                    const auto & tc = msg.tool_calls[i];
                    if (tc.name != pc.want_calls[i].first || json::parse(tc.arguments) != json::parse(pc.want_calls[i].second)) {
                        ok = false; why += " call" + std::to_string(i) + "=" + tc.name + tc.arguments;
                    }
                }
            }
        } catch (const std::exception & e) {
            ok = false;
            why = std::string(" threw: ") + e.what();
        }
        if (ok) {
            n_pass++;
        } else {
            n_fail++;
            printf("FAIL parse %s:%s\n", pc.name.c_str(), why.c_str());
        }
    }
    return n_fail;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s TEMPLATE.jinja ENCODING_TESTS_DIR\n", argv[0]);
        return 2;
    }
    const std::string tmpl = read_file(argv[1]);
    const std::filesystem::path dir = argv[2];
    if (tmpl.empty()) {
        fprintf(stderr, "cannot read template %s\n", argv[1]);
        return 2;
    }
    auto tmpls = common_chat_templates_init(nullptr, tmpl, BOS, "<｜end▁of▁sentence｜>");
    int n_unit_pass = 0;
    int n_unit_fail = run_units(tmpl, n_unit_pass);
    printf("units: %d passed, %d failed\n", n_unit_pass, n_unit_fail);
    int n_parse_pass = 0;
    const int n_parse_fail = run_parse(tmpls.get(), n_parse_pass);
    printf("parse: %d passed, %d failed\n", n_parse_pass, n_parse_fail);
    n_unit_pass += n_parse_pass;
    n_unit_fail += n_parse_fail;
    if (!std::filesystem::is_directory(dir)) {
        printf("golden: SKIP, %s not found\n", argv[2]);
        return n_unit_fail == 0 ? 0 : 1;
    }

    std::vector<int> ids;
    for (const auto & e : std::filesystem::directory_iterator(dir)) {
        const std::string fn = e.path().filename().string();
        if (fn.rfind("test_input_", 0) == 0 && e.path().extension() == ".json") {
            ids.push_back(std::stoi(fn.substr(11)));
        }
    }
    std::sort(ids.begin(), ids.end());

    int n_pass = n_unit_pass;
    int n_fail = n_unit_fail;

    for (int id : ids) {
        const std::string in_path  = (dir / ("test_input_"  + std::to_string(id) + ".json")).string();
        const std::string out_path = (dir / ("test_output_" + std::to_string(id) + ".txt")).string();
        const std::string want = read_file(out_path);
        json data = json::parse(read_file(in_path));
        // load_cases(): a dict is one case, a list of messages is one case; the golden test uses case 0
        json c;
        if (data.is_object()) {
            c = data;
        } else if (!data.empty() && data[0].contains("role")) {
            c = json{{"messages", data}};
        } else {
            c = data[0];
        }
        json messages = c["messages"];
        args_to_objects(messages);
        const std::string mode = c.contains("thinking_mode") && c["thinking_mode"].is_string() ? c["thinking_mode"].get<std::string>() : "chat";

        json input = {
            {"bos_token", BOS},
            {"add_generation_prompt", true},
            {"thinking_mode", mode},
        };
        if (c.contains("reasoning_effort") && !c["reasoning_effort"].is_null()) {
            input["reasoning_effort"] = c["reasoning_effort"];
        }

        // A: top-level tools
        json a = input;
        a["messages"] = messages;
        if (c.contains("tools")) {
            a["tools"] = c["tools"];
        }
        const std::string name = "case" + std::to_string(id);
        try {
            (compare(name + "/A", render_direct(tmpl, a), want) ? n_pass : n_fail)++;
        } catch (const std::exception & e) {
            printf("FAIL %s/A: %s\n", name.c_str(), e.what());
            n_fail++;
        }

        // B: tools on messages[0], as the reference load_cases() does
        if (c.contains("tools")) {
            json b = input;
            b["messages"] = messages;
            b["messages"][0]["tools"] = c["tools"];
            try {
                (compare(name + "/B", render_direct(tmpl, b), want) ? n_pass : n_fail)++;
            } catch (const std::exception & e) {
                printf("FAIL %s/B: %s\n", name.c_str(), e.what());
                n_fail++;
            }
        }

        // C: through common_chat_templates_apply (plain system/user/assistant cases only)
        bool plain = !c.contains("tools") && !c.contains("reasoning_effort");
        for (const auto & m : messages) {
            const std::string role = m.value("role", "");
            if ((role != "system" && role != "user" && role != "assistant") || m.contains("task") || !m["content"].is_string()) {
                plain = false;
            }
        }
        if (plain) {
            common_chat_templates_inputs in;
            in.messages = common_chat_msgs_parse_oaicompat(messages);
            in.use_jinja = true;
            in.add_generation_prompt = true;
            in.enable_thinking = mode == "thinking";
            in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            try {
                auto params = common_chat_templates_apply(tmpls.get(), in);
                (compare(name + "/C", params.prompt, want) ? n_pass : n_fail)++;
            } catch (const std::exception & e) {
                printf("FAIL %s/C: %s\n", name.c_str(), e.what());
                n_fail++;
            }
        }
    }

    printf("test-dsv41-chat: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail == 0 && n_pass > 0 ? 0 : 1;
}
