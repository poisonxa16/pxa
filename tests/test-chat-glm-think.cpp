// glm-think-off (bug glm-enable-thinking-ignored): GLM-5.3's chat template opens the think block in
// the generation prompt unconditionally ("<|assistant|><think>") and never reads enable_thinking, so a
// request with thinking off still got a reasoning trace and an empty content field. The engine now
// closes the block the way the template writes an assistant turn without reasoning
// ("<|assistant|><think></think>") and the generation parses as content. Thinking on / default renders
// exactly as before, every other template renders exactly as before, PXA_GLM_THINK_OFF=0 is the off
// switch.
//
// Usage: test-chat-glm-think   (run from the source root: reads models/templates/*.jinja)

#include "chat.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail++; } else { g_pass++; } } while (0)

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "FAIL cannot read %s (run from the source root)\n", path.c_str());
        exit(1);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static common_chat_msg msg(const std::string & role, const std::string & content) {
    common_chat_msg m;
    m.role    = role;
    m.content = content;
    return m;
}

static bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

enum class think { DEFAULT, ON, OFF };

static common_chat_params apply(const common_chat_templates * tmpls, think t,
                                common_reasoning_format rf = COMMON_REASONING_FORMAT_DEEPSEEK,
                                const std::vector<common_chat_msg> & messages = {}) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja             = true;
    inputs.add_generation_prompt = true;
    inputs.reasoning_format      = rf;
    inputs.messages              = messages.empty() ? std::vector<common_chat_msg>{ msg("user", "Write a short story.") } : messages;
    if (t != think::DEFAULT) {
        inputs.enable_thinking = (t == think::ON);
        // what the server does with "chat_template_kwargs": {"enable_thinking": ...}
        inputs.chat_template_kwargs["enable_thinking"] = (t == think::ON) ? "true" : "false";
    }
    return common_chat_templates_apply(tmpls, inputs);
}

static common_chat_msg parse(const common_chat_params & p, const std::string & out, bool partial = false,
                             common_reasoning_format rf = COMMON_REASONING_FORMAT_DEEPSEEK) {
    common_chat_parser_params pp(p);
    pp.reasoning_format = rf;
    pp.parser.load(p.parser);
    return common_chat_parse(out, partial, pp);
}

static void set_switch(const char * v) {
    if (v) {
        setenv("PXA_GLM_THINK_OFF", v, 1);
    } else {
        unsetenv("PXA_GLM_THINK_OFF");
    }
}

static void test_glm53() {
    const std::string path = "models/templates/GLM-5.3-Flash.jinja";
    auto tmpls = common_chat_templates_init(nullptr, read_file(path));
    set_switch(nullptr);

    // the model's own template default: thinking ON (the server's startup probe reads supports_thinking)
    CHECK(common_chat_templates_support_enable_thinking(tmpls.get()), "GLM-5.3 must report thinking support");

    // thinking off: closed empty think block, answer parses as content
    for (auto rf : { COMMON_REASONING_FORMAT_DEEPSEEK, COMMON_REASONING_FORMAT_AUTO }) {
        auto off = apply(tmpls.get(), think::OFF, rf);
        CHECK(ends_with(off.prompt, "<|assistant|><think></think>"), "off prompt tail: ...%s", off.prompt.substr(off.prompt.size() > 40 ? off.prompt.size() - 40 : 0).c_str());
        CHECK(off.generation_prompt == "<|assistant|><think></think>", "off generation_prompt '%s'", off.generation_prompt.c_str());
        auto m = parse(off, "Once upon a time.", false, rf);
        CHECK(m.content == "Once upon a time.", "off content '%s'", m.content.c_str());
        CHECK(m.reasoning_content.empty(), "off reasoning '%s'", m.reasoning_content.c_str());
        auto mp = parse(off, "Once up", true, rf);
        CHECK(mp.content == "Once up", "off partial content '%s'", mp.content.c_str());
        CHECK(mp.reasoning_content.empty(), "off partial reasoning '%s'", mp.reasoning_content.c_str());
    }

    // thinking on and default: exactly the template's render, reasoning then content
    auto on  = apply(tmpls.get(), think::ON);
    auto def = apply(tmpls.get(), think::DEFAULT);
    CHECK(ends_with(on.prompt, "<|assistant|><think>"), "on prompt must end with the open think tag");
    CHECK(on.generation_prompt == "<|assistant|><think>", "on generation_prompt '%s'", on.generation_prompt.c_str());
    CHECK(on.prompt == def.prompt, "default prompt differs from thinking on");
    CHECK(on.generation_prompt == def.generation_prompt, "default generation_prompt differs from thinking on");
    CHECK(on.parser == def.parser, "default parser differs from thinking on");
    auto mo = parse(on, "The user wants a story.</think>Once upon a time.");
    CHECK(mo.reasoning_content == "The user wants a story.", "on reasoning '%s'", mo.reasoning_content.c_str());
    CHECK(mo.content == "Once upon a time.", "on content '%s'", mo.content.c_str());

    // off differs from on only by the close tag at the very end
    auto off = apply(tmpls.get(), think::OFF);
    CHECK(off.prompt == on.prompt + "</think>", "off prompt must be the on prompt + </think>");

    // multi-turn history (an earlier assistant turn without reasoning) keeps the template's render
    std::vector<common_chat_msg> conv = { msg("user", "Hi"), msg("assistant", "Hello!"), msg("user", "Tell me a joke.") };
    auto on2  = apply(tmpls.get(), think::ON, COMMON_REASONING_FORMAT_DEEPSEEK, conv);
    auto off2 = apply(tmpls.get(), think::OFF, COMMON_REASONING_FORMAT_DEEPSEEK, conv);
    CHECK(off2.prompt == on2.prompt + "</think>", "multi-turn off prompt must be the on prompt + </think>");
    CHECK(on2.prompt.find("<|assistant|><think></think>Hello!") != std::string::npos, "history assistant turn render");

    // the off switch: PXA_GLM_THINK_OFF=0 renders thinking off exactly as before the fix (the bug: the
    // answer would land in reasoning_content)
    set_switch("0");
    auto old = apply(tmpls.get(), think::OFF);
    CHECK(old.prompt == on.prompt, "PXA_GLM_THINK_OFF=0: off prompt must equal the template's render");
    CHECK(old.generation_prompt == "<|assistant|><think>", "PXA_GLM_THINK_OFF=0 generation_prompt '%s'", old.generation_prompt.c_str());
    bool old_clean = false;
    try {
        auto mold = parse(old, "Once upon a time.");
        old_clean = mold.content == "Once upon a time." && mold.reasoning_content.empty();
    } catch (const std::exception &) {
    }
    CHECK(!old_clean, "PXA_GLM_THINK_OFF=0 must reproduce the old (open think block) parse, not clean content");
    set_switch(nullptr);

    // no generation prompt (continue / prefill): untouched
    {
        common_chat_templates_inputs inputs;
        inputs.use_jinja             = true;
        inputs.add_generation_prompt = false;
        inputs.enable_thinking       = false;
        inputs.messages              = { msg("user", "Hi") };
        auto a = common_chat_templates_apply(tmpls.get(), inputs);
        inputs.enable_thinking = true;
        auto b = common_chat_templates_apply(tmpls.get(), inputs);
        CHECK(a.prompt == b.prompt, "no generation prompt: off must equal on");
    }
}

// templates that are not affected: every render with the fix equals the render with the off switch
static void test_untouched(const std::string & name) {
    const std::string path = "models/templates/" + name;
    auto tmpls = common_chat_templates_init(nullptr, read_file(path));
    for (auto t : { think::DEFAULT, think::ON, think::OFF }) {
        set_switch(nullptr);
        auto a = apply(tmpls.get(), t);
        set_switch("0");
        auto b = apply(tmpls.get(), t);
        set_switch(nullptr);
        CHECK(a.prompt == b.prompt && a.generation_prompt == b.generation_prompt && a.parser == b.parser,
              "%s (think %d): render changed by the GLM fix", name.c_str(), (int) t);
    }
}

int main() {
    test_glm53();
    // GLM templates that handle enable_thinking themselves, and forced-open think templates of other families
    for (const char * n : { "GLM-4.6.jinja", "GLM-4.7-Flash.jinja", "Qwen-Qwen3-0.6B.jinja", "Qwen3.5-4B.jinja",
                            "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B.jinja", "llama-cpp-deepseek-r1.jinja",
                            "MiMo-VL.jinja", "Kimi-K2-Thinking.jinja", "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja" }) {
        test_untouched(n);
    }
    if (g_fail) {
        fprintf(stderr, "test-chat-glm-think: %d FAILED, %d passed\n", g_fail, g_pass);
        return 1;
    }
    printf("test-chat-glm-think: ALL PASS (%d checks)\n", g_pass);
    return 0;
}
