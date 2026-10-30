// test-chat-bos-probe: the chat path's "keep the template's BOS text" probe (common_chat_templates_init) changes
// nothing for vocabs outside its scope (Metaspace prepend_scheme "first"):
//   1. a template that starts with {{ bos_token }} renders with the BOS text stripped exactly when the vocab says
//      add_bos (the rule before the probe), for every vocab given, and kept for a prepend_scheme "first" vocab;
//   2. the probe logs no double-BOS warning for any vocab;
//   3. a vocab type whose tokenizer never adds the BOS token (RWKV; PLaMo-2 is the same) with add_bos_token=true
//      keeps the old strip (a GGUF crafted here from the first vocab given, tokenizer swapped for an RWKV one).
//
// usage: test-chat-bos-probe <vocab.gguf>...   (a vocab that does not load is reported and skipped)
#include "llama.h"
#include "common.h"
#include "chat.h"
#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static int g_double_bos = 0;

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (text && strstr(text, "starts with 2 BOS tokens")) {
        ++g_double_bos;
    }
    if (level == GGML_LOG_LEVEL_ERROR && text) {
        fputs(text, stderr);
    }
}

static const char * k_tmpl = "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}";

// returns 0 ok, 1 fail, 2 skipped
static int check_one(const std::string & fname, const char * label, int want_keep /* -1 = by vocab */) {
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(fname.c_str(), mparams);
    if (!model) {
        printf("SKIP %-40s does not load\n", label);
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    char scheme[64] = {0};
    const bool has_first = llama_model_meta_val_str(model, "tokenizer.ggml.metaspace.prepend_scheme", scheme, sizeof(scheme)) > 0 &&
                           std::string(scheme) == "first";
    const char * off = getenv("PXA_TOKENIZER_PREPEND_FIRST");
    const bool first_on = has_first && !(off && off[0] == '0');

    const llama_token bos = llama_vocab_bos(vocab);
    const std::string bos_text = bos == LLAMA_TOKEN_NULL ? std::string() : common_token_to_piece(vocab, bos, true);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    g_double_bos = 0;
    auto tmpls = common_chat_templates_init(model, k_tmpl);
    const int warned = g_double_bos;

    common_chat_templates_inputs in;
    common_chat_msg u; u.role = "user"; u.content = "hi"; in.messages.push_back(u);
    in.add_generation_prompt = false;
    in.use_jinja = true;
    const auto params = common_chat_templates_apply(tmpls.get(), in);
    const bool kept = !bos_text.empty() && params.prompt.rfind(bos_text, 0) == 0;

    // the rule before the probe: strip iff add_bos (and there is a BOS text to strip)
    bool want = want_keep >= 0 ? want_keep != 0 : !(add_bos && !bos_text.empty()) || first_on;
    if (bos_text.empty()) { want = false; }

    int rc = 0;
    if (kept != want) {
        printf("FAIL %-40s add_bos=%d first=%d: BOS text %s, want %s\n", label, add_bos, first_on,
               kept ? "kept" : "stripped", want ? "kept" : "stripped");
        rc = 1;
    }
    if (warned) {
        printf("FAIL %-40s %d double-BOS warning(s) during template init\n", label, warned);
        rc = 1;
    }
    if (rc == 0) {
        printf("ok   %-40s add_bos=%d first=%d BOS text %s\n", label, add_bos, first_on, kept ? "kept" : "stripped");
    }
    tmpls.reset();
    llama_free_model(model);
    return rc;
}

// a vocab-only GGUF with the base file's model keys and a small RWKV tokenizer: <s> (BOS, control), </s>, "\n",
// printable ASCII; add_bos_token = true. The RWKV tokenizer never adds BOS whatever add_bos says.
static bool make_rwkv_vocab(const std::string & base, const std::string & out) {
    gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * src = gguf_init_from_file(base.c_str(), ip);
    if (!src) { return false; }
    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);
    gguf_free(src);
    std::vector<std::string> drop;
    for (int i = 0; i < gguf_get_n_kv(dst); ++i) {
        const std::string k = gguf_get_key(dst, i);
        const bool vocab_size = k.size() > 11 && k.compare(k.size() - 11, 11, ".vocab_size") == 0;
        if (k.rfind("tokenizer.", 0) == 0 || vocab_size) { drop.push_back(k); }
    }
    for (const auto & k : drop) { gguf_remove_key(dst, k.c_str()); }

    std::vector<std::string> toks = { "<s>", "</s>", "\\n" };
    for (int c = 0x20; c < 0x7f; ++c) {
        toks.push_back(c == '\\' ? std::string("\\\\") : std::string(1, (char) c));
    }
    std::vector<const char *> ptrs;
    std::vector<int32_t> types;
    for (size_t i = 0; i < toks.size(); ++i) {
        ptrs.push_back(toks[i].c_str());
        types.push_back(i < 2 ? LLAMA_TOKEN_TYPE_CONTROL : LLAMA_TOKEN_TYPE_NORMAL);
    }
    gguf_set_val_str(dst, "tokenizer.ggml.model", "rwkv");
    gguf_set_arr_str(dst, "tokenizer.ggml.tokens", ptrs.data(), (int) ptrs.size());
    gguf_set_arr_data(dst, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), (int) types.size());
    gguf_set_val_u32(dst, "tokenizer.ggml.bos_token_id", 0);
    gguf_set_val_u32(dst, "tokenizer.ggml.eos_token_id", 1);
    gguf_set_val_bool(dst, "tokenizer.ggml.add_bos_token", true);
    gguf_write_to_file(dst, out.c_str(), false);
    gguf_free(dst);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <vocab.gguf>...\n", argv[0]); return 1; }
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    int n_fail = 0, n_ok = 0, n_skip = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string f = argv[i];
        const int rc = check_one(f, std::filesystem::path(f).filename().string().c_str(), -1);
        n_fail += rc == 1; n_ok += rc == 0; n_skip += rc == 2;
    }

    const std::string rwkv = (std::filesystem::temp_directory_path() /
                              ("test-chat-bos-probe-rwkv-" + std::to_string((long long) std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf")).string();
    if (!make_rwkv_vocab(argv[1], rwkv)) {
        printf("FAIL could not craft the RWKV vocab from %s\n", argv[1]);
        ++n_fail;
    } else {
        // old rule: add_bos=true -> the template's BOS text is stripped
        const int rc = check_one(rwkv, "crafted rwkv (add_bos_token=true)", 0);
        n_fail += rc != 0; n_ok += rc == 0;
        std::filesystem::remove(rwkv);
    }

    printf("chat-bos-probe: %d ok, %d failed, %d skipped\n", n_ok, n_fail, n_skip);
    llama_backend_free();
    return n_fail == 0 ? 0 : 1;
}
