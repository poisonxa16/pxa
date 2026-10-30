// chat-template-system-msg (public pxq_llama.cpp #3): a client (Claude Code and others) sends a system
// message after position 0; strict templates (Qwen3.5: "System message must be at the beginning")
// raise. common_chat_templates_apply merges every later system message into messages[0] for those
// templates only; templates that take a later system message render exactly as before.
//
// Usage: test-chat-system-merge [templates-dir]   (with a directory: also list which shipped templates
// the probe marks system-first-only, and check that each of them renders a late system message)

#include "chat.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail++; } } while (0)

static common_chat_msg msg(const std::string & role, const std::string & content) {
    common_chat_msg m;
    m.role    = role;
    m.content = content;
    return m;
}

static bool render(const std::string & tmpl_src, const std::vector<common_chat_msg> & messages, std::string & out, std::string & err) {
    try {
        auto tmpls = common_chat_templates_init(/* model= */ nullptr, tmpl_src, "<s>", "</s>");
        common_chat_templates_inputs inputs;
        inputs.use_jinja             = true;
        inputs.messages              = messages;
        inputs.add_generation_prompt = true;
        out = common_chat_templates_apply(tmpls.get(), inputs).prompt;
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

// Qwen3.5-style strict template (the literal message the source check keys on)
static const char * STRICT_LITERAL =
    "{%- for message in messages -%}"
    "{%- if message.role == 'system' and loop.index0 > 0 -%}{{ raise_exception('System message must be at the beginning.') }}{%- endif -%}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}";

// strict by behaviour only (different wording: found by the render probe)
static const char * STRICT_PROBE =
    "{%- for message in messages -%}"
    "{%- if message.role == 'system' and not loop.first -%}{{ raise_exception('only one system turn, and it goes first') }}{%- endif -%}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}";

// permissive ChatML: a later system message is rendered where it is
static const char * PERMISSIVE =
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}";

// alternation-strict but system-anywhere: two user turns in a row raise, a system message is rendered
// wherever it is. The probe's control conversation must be the same turns without the system message
// (user, user), or this template is flagged and a valid later system message would be moved.
static const char * ALTERNATING_SYSTEM_ANYWHERE =
    "{%- set ns = namespace(last='') -%}"
    "{%- for message in messages -%}"
    "{%- if message.role != 'system' -%}"
    "{%- if message.role == ns.last -%}{{ raise_exception('roles must alternate') }}{%- endif -%}"
    "{%- set ns.last = message.role -%}"
    "{%- endif -%}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}";

static std::string chatml(const std::vector<std::pair<std::string, std::string>> & turns) {
    std::string s;
    for (const auto & t : turns) {
        s += "<|im_start|>" + t.first + "\n" + t.second + "<|im_end|>\n";
    }
    return s + "<|im_start|>assistant\n";
}

static void test_flags() {
    CHECK(common_chat_template(STRICT_LITERAL, "", "").system_first_only, "literal strict template not flagged");
    CHECK(common_chat_template(STRICT_PROBE, "", "").system_first_only, "probe strict template not flagged");
    CHECK(!common_chat_template(PERMISSIVE, "", "").system_first_only, "permissive template flagged");
    CHECK(!common_chat_template(ALTERNATING_SYSTEM_ANYWHERE, "", "").system_first_only, "alternation-strict system-anywhere template flagged (probe control must be user,user)");
}

static void test_alternating_system_anywhere() {
    std::string out, err;
    bool ok = render(ALTERNATING_SYSTEM_ANYWHERE, { msg("system", "S1"), msg("user", "U1"), msg("assistant", "A1"), msg("system", "S2"), msg("user", "U2") }, out, err);
    CHECK(ok, "alternating: raises: %s", err.c_str());
    CHECK(out == chatml({ { "system", "S1" }, { "user", "U1" }, { "assistant", "A1" }, { "system", "S2" }, { "user", "U2" } }),
          "alternating: a valid later system message was moved:\n%s", out.c_str());
}

static void test_empty_late_system(const char * name, const char * src) {
    std::string out, err;
    // an empty late system message is dropped: no "\n\n" is appended and no empty system message is created
    bool ok = render(src, { msg("system", "S1"), msg("user", "U1"), msg("system", ""), msg("user", "U2") }, out, err);
    CHECK(ok && out == chatml({ { "system", "S1" }, { "user", "U1" }, { "user", "U2" } }), "%s: empty late system changed the first message:\n%s", name, out.c_str());
    ok = render(src, { msg("user", "U1"), msg("system", ""), msg("user", "U2") }, out, err);
    CHECK(ok && out == chatml({ { "user", "U1" }, { "user", "U2" } }), "%s: empty late system created an empty system message:\n%s", name, out.c_str());
}

static void test_strict(const char * name, const char * src) {
    std::string out, err;

    // leading system + a later one: merged into the first, blank line between
    bool ok = render(src, { msg("system", "S1"), msg("user", "U1"), msg("assistant", "A1"), msg("system", "S2"), msg("user", "U2") }, out, err);
    CHECK(ok, "%s: late system still raises: %s", name, err.c_str());
    CHECK(out == chatml({ { "system", "S1\n\nS2" }, { "user", "U1" }, { "assistant", "A1" }, { "user", "U2" } }),
          "%s: merged prompt wrong:\n%s", name, out.c_str());

    // no leading system: one is created at position 0; two late ones join in order
    ok = render(src, { msg("user", "U1"), msg("system", "S2"), msg("assistant", "A1"), msg("system", "S3"), msg("user", "U2") }, out, err);
    CHECK(ok, "%s: created-system case raises: %s", name, err.c_str());
    CHECK(out == chatml({ { "system", "S2\n\nS3" }, { "user", "U1" }, { "assistant", "A1" }, { "user", "U2" } }),
          "%s: created-system prompt wrong:\n%s", name, out.c_str());

    // nothing to merge: unchanged
    ok = render(src, { msg("system", "S1"), msg("user", "U1") }, out, err);
    CHECK(ok && out == chatml({ { "system", "S1" }, { "user", "U1" } }), "%s: plain prompt changed:\n%s", name, out.c_str());
    ok = render(src, { msg("user", "U1"), msg("assistant", "A1"), msg("user", "U2") }, out, err);
    CHECK(ok && out == chatml({ { "user", "U1" }, { "assistant", "A1" }, { "user", "U2" } }), "%s: no-system prompt changed:\n%s", name, out.c_str());
}

static void test_permissive() {
    std::string out, err;
    bool ok = render(PERMISSIVE, { msg("system", "S1"), msg("user", "U1"), msg("system", "S2"), msg("user", "U2") }, out, err);
    CHECK(ok, "permissive raises: %s", err.c_str());
    CHECK(out == chatml({ { "system", "S1" }, { "user", "U1" }, { "system", "S2" }, { "user", "U2" } }),
          "permissive prompt changed (must stay where the client put it):\n%s", out.c_str());
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static void sweep(const std::string & dir) {
    int flagged = 0, total = 0;
    for (const auto & e : std::filesystem::directory_iterator(dir)) {
        if (e.path().extension() != ".jinja") {
            continue;
        }
        const std::string src = read_file(e.path().string());
        bool flag = false;
        try {
            flag = common_chat_template(src, "<s>", "</s>").system_first_only;
        } catch (const std::exception &) {
            continue;  // not parseable by this engine: out of scope
        }
        total++;
        if (!flag) {
            continue;
        }
        flagged++;
        std::string out, err;
        bool ok = render(src, { msg("system", "S1"), msg("user", "U1"), msg("assistant", "A1"), msg("system", "S2"), msg("user", "U2") }, out, err);
        printf("  system-first-only: %s -> late system %s\n", e.path().filename().string().c_str(), ok ? "renders" : ("RAISES: " + err).c_str());
        CHECK(ok, "%s: late system still raises after the merge: %s", e.path().filename().string().c_str(), err.c_str());
        CHECK(!ok || out.find("S2") != std::string::npos, "%s: late system text lost", e.path().filename().string().c_str());
    }
    printf("sweep: %d of %d templates system-first-only\n", flagged, total);
}

int main(int argc, char ** argv) {
    test_flags();
    test_strict("literal", STRICT_LITERAL);
    test_strict("probe", STRICT_PROBE);
    test_permissive();
    test_alternating_system_anywhere();
    test_empty_late_system("literal", STRICT_LITERAL);
    test_empty_late_system("probe", STRICT_PROBE);
    if (argc > 1) {
        sweep(argv[1]);
    }
    if (g_fail) {
        fprintf(stderr, "test-chat-system-merge: %d failure(s)\n", g_fail);
        return 1;
    }
    printf("test-chat-system-merge: ALL PASS\n");
    return 0;
}
