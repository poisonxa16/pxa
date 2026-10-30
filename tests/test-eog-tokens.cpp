// common_eog_tokens (#361/#362, 2026-10-26): the server's ignore_eos bans every end-of-generation token, because a slot
// stops on any llama_token_is_eog token. On a Qwen vocab the EOS is <|im_end|>, and <|endoftext|> is a second EOG token
// that a raw-text prompt can produce first (Swift-1.5 Flash-Next did): banning only llama_token_eos ended the request
// after one token. Vocab-only load of models/ggml-vocab-qwen2.gguf; no weights, no GPU.
#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static llama_token one_token(const llama_model * model, const std::string & text) {
    const auto toks = common_tokenize(model, text, false, true);
    return toks.size() == 1 ? toks[0] : -1;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <vocab.gguf>\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr) {
        fprintf(stderr, "could not load %s\n", argv[1]);
        return 1;
    }
    const std::vector<llama_token> eog = common_eog_tokens(model);
    const auto has = [&](llama_token t) { return t >= 0 && std::find(eog.begin(), eog.end(), t) != eog.end(); };

    const llama_token eos = llama_token_eos(model);
    const llama_token im_end = one_token(model, "<|im_end|>");
    const llama_token endoftext = one_token(model, "<|endoftext|>");
    fprintf(stderr, "eog tokens: %zu (eos %d, <|im_end|> %d, <|endoftext|> %d)\n", eog.size(), eos, im_end, endoftext);

    CHECK(!eog.empty());
    CHECK(std::is_sorted(eog.begin(), eog.end()));
    CHECK(has(eos));                                 // the old single ban is still covered
    CHECK(im_end >= 0 && has(im_end));
    CHECK(endoftext >= 0 && has(endoftext));         // the token the old ban missed
    CHECK(eog.size() >= 2);
    for (const llama_token t : eog) {
        CHECK(llama_token_is_eog(model, t));         // nothing but EOG tokens (an ordinary token is never banned)
    }
    CHECK(!has(one_token(model, "Hello")));

    llama_free_model(model);
    llama_backend_free();
    if (fails) {
        fprintf(stderr, "test-eog-tokens: %d failure(s)\n", fails);
        return 1;
    }
    fprintf(stderr, "test-eog-tokens: OK\n");
    return 0;
}
