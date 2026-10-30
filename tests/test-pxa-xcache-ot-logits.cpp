// PXA_XCACHE_OT bit-identity driver (glm-mc-2-xcache-multidevice; CPU only).
//
// Loads the model with a STATIC -ot placement of every routed-expert tensor on the CPU buffer type (what the GLM argv does
// per card), prefills a fixed prompt, decodes n_predict greedy tokens and writes every logits row it saw, raw float32, to
// OUT. Run it twice -- once plain, once with PXA_XCACHE_OT=<n_hot> PXA_XCACHE_OT_COUNTS=<csv> -- and `cmp` the two files:
// tests/run-xcache-ot-bitid.sh does exactly that on the tiny glm5next file from tests/gen_tiny_glm5next.py.
//
//   test-pxa-xcache-ot-logits model.gguf OUT [n_prompt=12] [n_predict=16] [mmap=0]
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf OUT [n_prompt] [n_predict] [mmap]\n", argv[0]); return 2; }
    const int n_prompt  = argc > 3 ? atoi(argv[3]) : 12;
    const int n_predict = argc > 4 ? atoi(argv[4]) : 16;
    const bool mmap     = argc > 5 && atoi(argv[5]) != 0;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap = mmap;
    llama_model_tensor_buft_override ot[2] = { { "blk\\.[0-9]+\\.ffn_.*_exps\\.weight", ggml_backend_cpu_buffer_type() }, { nullptr, nullptr } };
    mp.tensor_buft_overrides = ot;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const int n_vocab = llama_n_vocab(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 64; cp.n_ubatch = 64; cp.n_seq_max = 1;
    cp.n_threads = 4; cp.n_threads_batch = 4;
    cp.flash_attn = false; cp.mla_attn = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    FILE * fo = fopen(argv[2], "wb");
    if (!fo) { perror(argv[2]); return 1; }
    llama_batch b = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; ++i) {
        b.token[i] = (llama_token)(5 + (i*37) % (n_vocab - 8)); b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = n_prompt;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill failed\n"); return 1; }
    for (int i = 0; i < n_prompt; ++i) fwrite(llama_get_logits_ith(ctx, i), sizeof(float), n_vocab, fo);
    llama_batch_free(b);
    llama_pos pos = n_prompt;
    for (int t = 0; t < n_predict; ++t) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        llama_token best = 0;
        for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
        printf("%d ", best);
        llama_batch d = llama_batch_init(1, 0, 1);
        d.token[0] = best; d.pos[0] = pos++; d.n_seq_id[0] = 1; d.seq_id[0][0] = 0; d.logits[0] = 1; d.n_tokens = 1;
        if (llama_decode(ctx, d) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        llama_batch_free(d);
        fwrite(llama_get_logits_ith(ctx, -1), sizeof(float), n_vocab, fo);
    }
    printf("\n");
    fclose(fo);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
