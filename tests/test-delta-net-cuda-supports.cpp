// GGML_OP_DELTA_NET on CUDA: supports_op must only claim the head widths the kernel has.
//
// The CUDA delta-net kernel (ggml/src/ggml-cuda/pxa/delta-net.cu) is instantiated for 64- and
// 128-wide heads and aborts with "Unsupported delta net head size" on anything else. Until
// 3456321f68 the backend's supports_op answered true for every DELTA_NET, so the scheduler
// never moved other widths to the CPU and tests/fixtures/tiny_qwen4exp.gguf (32-wide heads)
// aborted in llama_decode on every CUDA binary (todo qwen4exp-tiny-fixture-cuda-delta-head32).
//
// This test builds the op for widths 16, 32, 64, 96, 128 (both forget-gate variants) and asserts
// the CUDA answer is exactly "64 or 128". It never launches the kernel; it only needs a CUDA
// context, so it is not registered with llama_test(): run it by hand inside a GPU lock window.
// No CUDA device -> prints SKIP and exits 0.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>

static ggml_tensor * make_op(ggml_context * ctx, int S, bool per_channel) {
    const int T = 3, H = 2, n_seqs = 1;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, n_seqs);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, n_seqs);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, n_seqs);
    ggml_tensor * g = per_channel
        ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, n_seqs)
        : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T, 1, H, n_seqs);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, T, H, n_seqs);
    ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S*H, 1, n_seqs);
    return ggml_delta_net_ext(ctx, q, k, v, g, b, s, nullptr, per_channel ? 1 : 0);
}

int main() {
    printf("test-delta-net-cuda-supports: CUDA supports_op(DELTA_NET) == head width in {64, 128}\n");
    const int n_dev = ggml_backend_cuda_get_device_count();
    if (n_dev <= 0) {
        printf("SKIP (no CUDA device)\n");
        return 0;
    }
    ggml_backend_t be = ggml_backend_cuda_init(0, nullptr);
    if (!be) {
        printf("SKIP (ggml_backend_cuda_init(0) failed)\n");
        return 0;
    }

    // no_alloc: shapes only, nothing is computed
    ggml_init_params ip = { 16u*1024u*1024u, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    bool ok = true;
    const int widths[] = { 16, 32, 64, 96, 128 };
    for (int pc = 0; pc < 2; ++pc) {
        for (int S : widths) {
            ggml_tensor * op = make_op(ctx, S, pc != 0);
            const bool want = (S == 64 || S == 128);
            const bool got  = ggml_backend_supports_op(be, op);
            const bool pass = (got == want);
            ok &= pass;
            printf("  %-4s S=%-4d supports_op=%d want=%d  %s\n", pc ? "KDA" : "GDN", S, got, want, pass ? "OK" : "FAIL");
        }
    }

    ggml_free(ctx);
    ggml_backend_free(be);
    printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
