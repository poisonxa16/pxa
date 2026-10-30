// A chat template that failed to load leaves a null common_chat_templates (the server keeps running without
// one). Every accessor must be safe on it: no crash, no assert; the ones that cannot answer throw a
// std::runtime_error (the server turns it into a request error), the others return an empty answer.

#include "chat.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail++; } } while (0)

template <typename F>
static bool throws_runtime_error(F && f) {
    try {
        f();
    } catch (const std::runtime_error &) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

int main() {
    const common_chat_templates * null_tmpls = nullptr;

    common_chat_msg user;
    user.role    = "user";
    user.content = "hi";

    common_chat_templates_inputs inputs;
    inputs.messages = { user };

    CHECK(throws_runtime_error([&] { common_chat_templates_apply(null_tmpls, inputs); }), "apply(nullptr) must throw runtime_error");
    for (bool jinja : { false, true }) {
        CHECK(throws_runtime_error([&] { common_chat_format_single(null_tmpls, {}, user, true, jinja); }),
              "format_single(nullptr, jinja=%d) must throw runtime_error", (int) jinja);
        CHECK(throws_runtime_error([&] { common_chat_format_single(null_tmpls, { user }, user, true, jinja); }),
              "format_single(nullptr, history, jinja=%d) must throw runtime_error", (int) jinja);
        CHECK(common_chat_format_example(null_tmpls, jinja, {}).empty(), "format_example(nullptr, jinja=%d) must be empty", (int) jinja);
    }
    CHECK(!common_chat_templates_was_explicit(null_tmpls), "was_explicit(nullptr) must be false");
    CHECK(common_chat_templates_source(null_tmpls).empty(), "source(nullptr) must be empty");
    CHECK(common_chat_templates_source(null_tmpls, "tool_use").empty(), "source(nullptr, tool_use) must be empty");
    CHECK(!common_chat_templates_support_enable_thinking(null_tmpls), "support_enable_thinking(nullptr) must be false");
    CHECK(common_chat_templates_get_caps(null_tmpls).empty(), "get_caps(nullptr) must be empty");

    // a real template still answers as before
    auto tmpls = common_chat_templates_init(/* model= */ nullptr, "chatml");
    CHECK(tmpls != nullptr, "chatml must load");
    CHECK(common_chat_templates_was_explicit(tmpls.get()), "an explicit template reports explicit");
    CHECK(!common_chat_templates_source(tmpls.get()).empty(), "source of a loaded template is not empty");
    CHECK(!common_chat_format_example(tmpls.get(), true, {}).empty(), "format_example of a loaded template is not empty");

    if (g_fail) {
        fprintf(stderr, "test-chat-null-tmpls: %d failure(s)\n", g_fail);
        return 1;
    }
    printf("test-chat-null-tmpls: OK\n");
    return 0;
}
