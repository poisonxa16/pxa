// test-tokenizer-hf-parity: our tokenizer (and the server chat path) vs ids from the model's own HF tokenizer.json.
// Nanbeige4.3 uses a Metaspace pre-tokenizer with prepend_scheme "first" (space prefix only at the
// start of the input, never after a special token), and its BOS is <|im_start|>, which every chat turn also starts with.
//
// usage: test-tokenizer-hf-parity <vocab.gguf>
//   <vocab.gguf>.hfinp : cases separated by "\n__ggml_vocab_test__\n"
//   <vocab.gguf>.hfout : one line per case, "<mode> id id id ..." (ids from HF tokenizers, see scripts in the lane dir)
//     N = tokenize(text, add_special=false, parse_special=true)  == HF encode(text, add_special_tokens=False)
//     S = tokenize(text, add_special=true,  parse_special=true)  == HF encode(text, add_special_tokens=True)
//     C = chat [user: text] + generation prompt, rendered by the GGUF template as the server does, then tokenized
//         with add_special=true, parse_special=true (the server's prompt path) == HF apply_chat_template(tokenize=True)
//     D = as C with a system message "You are a helpful assistant." first
//     T = as C with one tool (get_weather(city)) offered
#include "llama.h"
#include "common.h"
#include "chat.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string read_all(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <vocab.gguf>\n", argv[0]); return 1; }
    const std::string fname = argv[1];
    const std::string inp = read_all(fname + ".hfinp");
    std::ifstream fout(fname + ".hfout");
    if (inp.empty() || !fout) { fprintf(stderr, "missing %s.hfinp / .hfout\n", fname.c_str()); return 1; }

    std::vector<std::string> cases;
    const std::string sep = "\n__ggml_vocab_test__\n";
    for (size_t pos = 0;;) {
        size_t nx = inp.find(sep, pos);
        if (nx == std::string::npos) { break; }
        cases.push_back(inp.substr(pos, nx - pos));
        pos = nx + sep.size();
    }
    std::vector<std::pair<char, std::vector<llama_token>>> expect;
    for (std::string line; std::getline(fout, line);) {
        if (line.empty()) { continue; }
        std::istringstream ls(line); std::string mode; ls >> mode;
        std::vector<llama_token> ids; for (llama_token t; ls >> t;) { ids.push_back(t); }
        expect.emplace_back(mode[0], ids);
    }
    if (cases.size() != expect.size()) {
        fprintf(stderr, "case count %zu != expected count %zu\n", cases.size(), expect.size()); return 1;
    }

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(fname.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load %s\n", fname.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tmpls = common_chat_templates_init(model, "");

    int n_fail = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const char mode = expect[i].first;
        std::vector<llama_token> got;
        if (mode == 'N' || mode == 'S') {
            got = common_tokenize(vocab, cases[i], mode == 'S', true);
        } else if (mode == 'C' || mode == 'D' || mode == 'T') {
            common_chat_templates_inputs in;
            if (mode == 'D') { common_chat_msg s; s.role = "system"; s.content = "You are a helpful assistant."; in.messages.push_back(s); }
            common_chat_msg u; u.role = "user"; u.content = cases[i]; in.messages.push_back(u);
            if (mode == 'T') {
                common_chat_tool t;
                t.name        = "get_weather";
                t.description = "Get the current weather for a city";
                t.parameters  = R"({"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]})";
                in.tools.push_back(t);
            }
            in.add_generation_prompt = true;
            in.use_jinja = true;
            const auto params = common_chat_templates_apply(tmpls.get(), in);
            got = common_tokenize(vocab, params.prompt, true, true);
        } else {
            fprintf(stderr, "case %zu: unknown mode %c\n", i, mode); return 1;
        }
        const auto & want = expect[i].second;
        if (got != want) {
            ++n_fail;
            size_t k = 0; while (k < got.size() && k < want.size() && got[k] == want[k]) { ++k; }
            fprintf(stderr, "FAIL case %zu mode %c: n_ours %zu n_hf %zu, first diff at %zu (ours %d hf %d)\n", i, mode,
                    got.size(), want.size(), k, k < got.size() ? got[k] : -1, k < want.size() ? want[k] : -1);
        }
    }
    printf("hf-parity: %zu cases, %d failed\n", cases.size(), n_fail);
    tmpls.reset();
    llama_free_model(model);
    llama_backend_free();
    return n_fail == 0 ? 0 : 1;
}
