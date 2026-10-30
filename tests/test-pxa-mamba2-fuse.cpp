// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// PXA Nemotron-H phase 2: the in-place Mamba-2 ops (ggml_mamba2_conv + ggml_mamba2_scan) must equal the unfused
// graph BIT FOR BIT on CUDA -- the gated scan output and the recurrent state rows they write back.
//
// The unfused side is the exact op sequence src/llama-mamba2.cpp builds without PXA_MAMBA2_FUSE: get_rows of
// the state rows, * reset mask, ssm_conv over (window | x) with the per-token sequence map, + conv bias, silu,
// ssm_scan2 with dt + dt_b, y + D*x, * silu(z), and the new rows concatenated and scattered back with set_rows.
// Both sides run on CUDA device 0 from the same inputs; every case compares every float with memcmp.
//
// Cases: Nemotron-3-Nano-4B mixer dims (d_inner 7680, 96 heads x 80, 8 groups, d_state 128) and Nemotron-3-Nano-
// 30B-A3B dims (4096, 64 x 64, 8 groups, d_state 128); decode (1 token), short and long prefill (7 / 70 tokens:
// more than one conv token block), several sequences with one or five tokens each, reset masks on and off.
//
// Needs a CUDA device (LABEL "cuda"); exits 0 with a SKIP line when there is none.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct dims {
    const char * name;
    int64_t d_inner, n_head, n_group, d_state;
};

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static float frand(float lo, float hi) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return lo + (hi - lo) * (float) ((g_rng >> 11) & 0xFFFFFF) / (float) 0x1000000;
}

static size_t count_diff(const std::vector<float> & a, const std::vector<float> & b, const char * what) {
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            if (bad < 3) {
                printf("    %s mismatch at %zu: unfused %.9g fused %.9g\n", what, i, a[i], b[i]);
            }
            ++bad;
        }
    }
    return bad;
}

// one case: returns the number of mismatching floats
static size_t run_case(ggml_backend_t be, const dims & d, int64_t n_seqs, int64_t n_seq_tokens, bool with_reset) {
    const int64_t d_conv   = 4;
    const int64_t head_dim = d.d_inner / d.n_head;
    const int64_t conv_dim = d.d_inner + 2*d.n_group*d.d_state;
    const int64_t d_in     = 2*d.d_inner + 2*d.n_group*d.d_state + d.n_head;
    const int64_t conv_sd  = (d_conv - 1)*conv_dim;
    const int64_t ssm_sd   = d.d_state*d.d_inner;
    const int64_t state_w  = conv_sd + ssm_sd;
    const int64_t n_rows   = n_seqs + 2;
    const int64_t n_tok    = n_seqs*n_seq_tokens;

    ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    // inputs (shared by both graphs) and the two copies of the state rows
    ggml_tensor * rs_u   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_w, n_rows);
    ggml_tensor * rs_f   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_w, n_rows);
    ggml_tensor * zxBCdt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_in, n_tok);
    ggml_tensor * conv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, conv_dim);
    ggml_tensor * conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, conv_dim);
    ggml_tensor * A      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, d.n_head);
    ggml_tensor * D      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, d.n_head);
    ggml_tensor * dt_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.n_head);
    ggml_tensor * ids    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
    ggml_tensor * mask   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_seqs);
    ggml_tensor * seqmap = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_seqs, n_tok);

    const size_t ei  = ggml_element_size(zxBCdt);
    const size_t nbti = zxBCdt->nb[1];
    const size_t nbsi = nbti*n_seq_tokens;

    // ---------------- unfused (the src/llama-mamba2.cpp graph without PXA_MAMBA2_FUSE) ----------------
    ggml_tensor * y_u;
    ggml_tensor * set_u;
    {
        ggml_tensor * state = ggml_get_rows(ctx, rs_u, ids);
        state = ggml_mul(ctx, state, mask);
        ggml_tensor * conv_state_flat = ggml_view_2d(ctx, state, conv_sd, n_seqs, state->nb[1], 0);
        ggml_tensor * ssm_state_flat  = ggml_view_2d(ctx, state, ssm_sd,  n_seqs, state->nb[1], conv_sd*ggml_element_size(state));
        if (n_seqs > 1) {
            conv_state_flat = ggml_cont(ctx, conv_state_flat);
            ssm_state_flat  = ggml_cont(ctx, ssm_state_flat);
        }
        ggml_tensor * conv_states = ggml_reshape_3d(ctx, conv_state_flat, d_conv - 1, conv_dim, n_seqs);
        ggml_tensor * z   = ggml_view_2d(ctx, zxBCdt, d.d_inner, n_tok, nbti, 0);
        ggml_tensor * xBC = ggml_view_2d(ctx, zxBCdt, conv_dim, n_tok, nbti, d.d_inner*ei);
        ggml_tensor * dt  = ggml_view_2d(ctx, zxBCdt, d.n_head, n_tok, nbti, (2*d.d_inner + 2*d.n_group*d.d_state)*ei);
        ggml_tensor * conv_raw = ggml_ssm_conv(ctx, conv_states, ggml_cont(ctx, xBC), conv_w, seqmap, nullptr);
        ggml_tensor * xBC_c = ggml_view_2d(ctx, conv_raw, conv_dim, n_tok, conv_dim*ggml_element_size(conv_raw), 0);
        xBC_c = ggml_add(ctx, xBC_c, conv_b);
        xBC_c = ggml_silu(ctx, xBC_c);
        ggml_tensor * new_conv = ggml_view_3d(ctx, conv_raw, d_conv - 1, conv_dim, n_seqs,
                d_conv*ggml_element_size(conv_raw), d_conv*conv_dim*ggml_element_size(conv_raw),
                (conv_dim*n_tok + 1)*ggml_element_size(conv_raw));
        ggml_tensor * new_conv_flat = ggml_reshape_2d(ctx, ggml_cont(ctx, new_conv), conv_sd, n_seqs);
        const size_t es  = ggml_element_size(xBC_c);
        const size_t nbt = conv_dim*es;
        const size_t nbs = nbt*n_seq_tokens;
        ggml_tensor * x = ggml_view_4d(ctx, xBC_c, head_dim, d.n_head,  n_seq_tokens, n_seqs, head_dim*es, nbt, nbs, 0);
        ggml_tensor * B = ggml_view_4d(ctx, xBC_c, d.d_state, d.n_group, n_seq_tokens, n_seqs, d.d_state*es, nbt, nbs, d.d_inner*es);
        ggml_tensor * C = ggml_view_4d(ctx, xBC_c, d.d_state, d.n_group, n_seq_tokens, n_seqs, d.d_state*es, nbt, nbs, (d.d_inner + d.n_group*d.d_state)*es);
        dt = ggml_add(ctx, ggml_cont(ctx, dt), dt_b);
        dt = ggml_reshape_3d(ctx, dt, d.n_head, n_seq_tokens, n_seqs);
        ggml_tensor * s4 = ggml_reshape_4d(ctx, ssm_state_flat, d.d_state, head_dim, d.n_head, n_seqs);
        ggml_tensor * y_s = ggml_ssm_scan2(ctx, s4, x, dt, A, B, C);
        const int64_t n_y = head_dim*d.n_head*n_tok;
        ggml_tensor * new_ssm_flat = ggml_view_2d(ctx, y_s, ssm_sd, n_seqs, ssm_sd*ggml_element_size(y_s), n_y*ggml_element_size(y_s));
        ggml_tensor * new_row = ggml_concat(ctx, new_conv_flat, new_ssm_flat, 0);
        set_u = ggml_set_rows(ctx, rs_u, new_row, ids);
        ggml_tensor * y = ggml_view_3d(ctx, y_s, head_dim, d.n_head, n_tok, head_dim*ggml_element_size(y_s), head_dim*d.n_head*ggml_element_size(y_s), 0);
        ggml_tensor * x3 = ggml_view_3d(ctx, xBC_c, head_dim, d.n_head, n_tok, head_dim*es, nbt, 0);
        y = ggml_add(ctx, y, ggml_mul(ctx, x3, D));
        ggml_tensor * z3 = ggml_reshape_3d(ctx, ggml_cont(ctx, z), head_dim, d.n_head, n_tok);
        y_u = ggml_mul(ctx, y, ggml_silu(ctx, z3));
    }

    // ---------------- fused ----------------
    ggml_tensor * y_f;
    {
        ggml_tensor * xBC = ggml_view_2d(ctx, zxBCdt, conv_dim, n_tok, nbti, d.d_inner*ei);
        ggml_tensor * xBC_c = ggml_mamba2_conv(ctx, rs_f, xBC, conv_w, conv_b, ids, mask);
        const size_t es  = ggml_element_size(xBC_c);
        const size_t nbt = conv_dim*es;
        const size_t nbs = nbt*n_seq_tokens;
        ggml_tensor * x = ggml_view_4d(ctx, xBC_c, head_dim, d.n_head,  n_seq_tokens, n_seqs, head_dim*es, nbt, nbs, 0);
        ggml_tensor * B = ggml_view_4d(ctx, xBC_c, d.d_state, d.n_group, n_seq_tokens, n_seqs, d.d_state*es, nbt, nbs, d.d_inner*es);
        ggml_tensor * C = ggml_view_4d(ctx, xBC_c, d.d_state, d.n_group, n_seq_tokens, n_seqs, d.d_state*es, nbt, nbs, (d.d_inner + d.n_group*d.d_state)*es);
        ggml_tensor * z  = ggml_view_4d(ctx, zxBCdt, head_dim, d.n_head, n_seq_tokens, n_seqs, head_dim*ei, nbti, nbsi, 0);
        ggml_tensor * dt = ggml_view_3d(ctx, zxBCdt, d.n_head, n_seq_tokens, n_seqs, nbti, nbsi, (2*d.d_inner + 2*d.n_group*d.d_state)*ei);
        y_f = ggml_mamba2_scan(ctx, rs_f, x, dt, A, B, C, dt_b, D, z, ids, mask, conv_sd);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) {
        printf("    could not allocate the leaf tensors\n");
        ggml_free(ctx);
        return 1;
    }

    // fill
    std::vector<float> h_rs(state_w*n_rows), h_zx(d_in*n_tok), h_cw(d_conv*conv_dim), h_cb(conv_dim);
    std::vector<float> h_A(d.n_head), h_D(d.n_head), h_dtb(d.n_head);
    for (auto & v : h_rs)  v = frand(-1.0f, 1.0f);
    for (auto & v : h_zx)  v = frand(-3.0f, 3.0f);
    for (auto & v : h_cw)  v = frand(-0.5f, 0.5f);
    for (auto & v : h_cb)  v = frand(-0.2f, 0.2f);
    for (auto & v : h_A)   v = -frand(0.05f, 8.0f);
    for (auto & v : h_D)   v = frand(0.5f, 1.5f);
    for (auto & v : h_dtb) v = frand(-4.0f, 2.0f);
    // a few dt values beyond the softplus cut (dt + dt_b > 20)
    for (int64_t t = 0; t < n_tok; ++t) {
        h_zx[t*d_in + 2*d.d_inner + 2*d.n_group*d.d_state + (t % d.n_head)] = 25.0f;
    }
    std::vector<int32_t> h_ids(n_seqs), h_map(n_seqs*n_tok, -1);
    std::vector<float> h_mask(n_seqs);
    for (int64_t s = 0; s < n_seqs; ++s) {
        h_ids[s]  = (int32_t) (n_rows - 1 - s);   // unique rows, reversed order (the engine never shares a row)
        h_mask[s] = (with_reset && (s % 2 == 0)) ? 0.0f : 1.0f;
        for (int64_t t = 0; t < n_seq_tokens; ++t) {
            h_map[(s*n_seq_tokens + t)*n_seqs + 0] = (int32_t) s;
        }
    }
    ggml_backend_tensor_set(rs_u,   h_rs.data(),  0, ggml_nbytes(rs_u));
    ggml_backend_tensor_set(rs_f,   h_rs.data(),  0, ggml_nbytes(rs_f));
    ggml_backend_tensor_set(zxBCdt, h_zx.data(),  0, ggml_nbytes(zxBCdt));
    ggml_backend_tensor_set(conv_w, h_cw.data(),  0, ggml_nbytes(conv_w));
    ggml_backend_tensor_set(conv_b, h_cb.data(),  0, ggml_nbytes(conv_b));
    ggml_backend_tensor_set(A,      h_A.data(),   0, ggml_nbytes(A));
    ggml_backend_tensor_set(D,      h_D.data(),   0, ggml_nbytes(D));
    ggml_backend_tensor_set(dt_b,   h_dtb.data(), 0, ggml_nbytes(dt_b));
    ggml_backend_tensor_set(ids,    h_ids.data(), 0, ggml_nbytes(ids));
    ggml_backend_tensor_set(mask,   h_mask.data(),0, ggml_nbytes(mask));
    ggml_backend_tensor_set(seqmap, h_map.data(), 0, ggml_nbytes(seqmap));

    // compute both graphs (separate graphs: each one allocates its intermediates)
    size_t bad = 0;
    std::vector<float> yu(ggml_nelements(y_u)), yf(ggml_nelements(y_f));
    {
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_cgraph * gu = ggml_new_graph_custom(ctx, 1024, false);
        ggml_build_forward_expand(gu, set_u);
        ggml_build_forward_expand(gu, y_u);
        if (!ggml_gallocr_alloc_graph(ga, gu) || ggml_backend_graph_compute(be, gu) != GGML_STATUS_SUCCESS) {
            printf("    unfused graph failed\n");
            return 1;
        }
        ggml_backend_synchronize(be);
        ggml_backend_tensor_get(y_u, yu.data(), 0, ggml_nbytes(y_u));
        ggml_gallocr_free(ga);
    }
    {
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
        ggml_build_forward_expand(gf, y_f);
        if (!ggml_gallocr_alloc_graph(ga, gf) || ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            printf("    fused graph failed\n");
            return 1;
        }
        ggml_backend_synchronize(be);
        ggml_backend_tensor_get(y_f, yf.data(), 0, ggml_nbytes(y_f));
        ggml_gallocr_free(ga);
    }
    std::vector<float> su(state_w*n_rows), sf(state_w*n_rows);
    ggml_backend_tensor_get(rs_u, su.data(), 0, ggml_nbytes(rs_u));
    ggml_backend_tensor_get(rs_f, sf.data(), 0, ggml_nbytes(rs_f));

    bad += count_diff(yu, yf, "y");
    bad += count_diff(su, sf, "state");
    // the rows of this batch must have changed (the test would pass trivially otherwise)
    size_t moved = 0;
    for (int64_t s = 0; s < n_seqs; ++s) {
        moved += memcmp(&h_rs[h_ids[s]*state_w], &sf[h_ids[s]*state_w], state_w*sizeof(float)) != 0;
    }
    if (moved != (size_t) n_seqs) {
        printf("    state rows not written (%zu of %lld)\n", moved, (long long) n_seqs);
        ++bad;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return bad;
}

int main() {
    if (ggml_backend_cuda_get_device_count() < 1) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }
    ggml_backend_t be = ggml_backend_cuda_init(0, nullptr);
    if (be == nullptr) {
        printf("SKIP: CUDA device 0 did not initialise\n");
        return 0;
    }
    const dims ds[] = {
        { "nano-4b",  7680, 96, 8, 128 },
        { "nano-30b", 4096, 64, 8, 128 },
    };
    struct shape { int64_t n_seqs, nst; bool reset; };
    const shape ss[] = {
        { 1,  1, false }, { 1,  1, true }, { 1, 7, false }, { 1, 70, true },
        { 3,  1, true  }, { 2,  5, true }, { 4, 1, false },
        // the chunked prefill scan (nst >= 32): one exact chunk, ragged chunks, long prefill, multi-seq
        { 1, 32, false }, { 2, 33, true }, { 1, 512, false }, { 3, 100, true },
    };
    size_t total = 0;
    int ncase = 0;
    for (const dims & d : ds) {
        for (const shape & s : ss) {
            const size_t bad = run_case(be, d, s.n_seqs, s.nst, s.reset);
            printf("%-8s n_seqs=%lld n_seq_tokens=%-3lld reset=%d : %s (%zu mismatches)\n", d.name, (long long) s.n_seqs,
                   (long long) s.nst, (int) s.reset, bad == 0 ? "bit-identical" : "DIFFERS", bad);
            total += bad;
            ++ncase;
        }
    }
    printf("%d cases, %zu mismatching floats\n%s\n", ncase, total, total == 0 ? "PASS" : "FAIL");
    ggml_backend_free(be);
    return total == 0 ? 0 : 1;
}
