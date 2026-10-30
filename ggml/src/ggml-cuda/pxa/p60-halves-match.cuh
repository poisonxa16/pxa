// pxa / PXA kernel suite -- p60-halves-match.cuh: graph matchers for the sm_60 persistent layer-half kernels
// (PXA_P60_MEGA, kernels in pxa/p60-mega.cu). Included once by ggml-cuda.cu (host code only).
//
// A half of a layer starts at the norm of its input and ends at the residual ADD; the matcher identifies every node in
// that range by op and weight name and declines on anything else, so other graphs and other quant types keep the
// normal path. The matchers are written for the PXA/ik graph of build_qwen35moe():
//   MoE half: FUSED_RMS_NORM(x, post_attention_norm) .. MUL_MAT ffn_gate_inp, SOFT_MAX, ARGSORT, GET_ROWS, SUM_ROWS, DIV,
//             MOE_FUSED_UP_GATE (up, gate, silu), MUL_MAT_ID down, MUL_MULTI_ADD, ADD(+x), MUL_MAT ffn_gate_inp_shexp,
//             FUSED_UP_GATE (shexp up, gate, silu), MUL_MAT ffn_down_shexp, FUSED_MUL_UNARY(sigmoid) .. ADD l_out-<il>.
//
// Switches (all default OFF): PXA_P60_MEGA=1 arms the persistent halves on cc 6.0 devices; PXA_P60_MEGA_MOE=0 / PXA_P60_MEGA_GDN=0 /
// PXA_P60_MEGA_ATTN=0 keep that half on the normal path; PXA_P60_MEGA_DEBUG=1 logs why a match attempt failed (first failures per line),
// =3 also names the failing final check of a MoE half per token count (with the expert-ids shape).
#pragma once

#include "p60-mega.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

static bool pxa_p60_env(const char * name, bool dflt) {
    const char * e = getenv(name);
    return e ? atoi(e) != 0 : dflt;
}

static bool pxa_p60_mega_on(const ggml_backend_cuda_context & ctx) {
    static const bool on = [] {
        const bool v = pxa_p60_env("PXA_P60_MEGA", false);
        if (v) fprintf(stderr, "PXA_P60_MEGA: 1 (persistent single-token layer halves on cc 6.0; MoE half %s)\n",
                       pxa_p60_env("PXA_P60_MEGA_MOE", true) ? "on" : "off");
        return v;
    }();
    return on && ggml_cuda_info().devices[ctx.device].cc == 600;
}

static int pxa_p60_fail(int line, const ggml_tensor * at) {
    static const bool dbg = pxa_p60_env("PXA_P60_MEGA_DEBUG", false);
    static std::map<int, int> n;
    if (dbg && n[line]++ < 8) {
        fprintf(stderr, "PXA_P60_MEGA: no match at line %d at %s %s (src0 %s src1 %s)\n", line, at ? ggml_op_desc(at) : "-",
                at ? at->name : "-", at && at->src[0] ? at->src[0]->name : "-", at && at->src[1] ? at->src[1]->name : "-");
    }
    return 0;
}
#define PXA_P60_FAIL(t) return pxa_p60_fail(__LINE__, (t))

static bool pxa_p60_is_w(const ggml_tensor * t, int il, const char * suffix) {
    char buf[128];
    snprintf(buf, sizeof(buf), "blk.%d.%s", il, suffix);
    return t != nullptr && strcmp(t->name, buf) == 0;
}

static bool pxa_p60_named(const ggml_tensor * t, const char * base, int il) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s-%d", base, il);
    return t != nullptr && strcmp(t->name, buf) == 0;
}

static const ggml_tensor * pxa_p60_root(const ggml_tensor * t) {
    while (t->view_src != nullptr) {
        t = t->view_src;
    }
    return t;
}

// tensors computed in nodes [beg, end] are not graph outputs and are not read by later nodes, except `keep`
static bool pxa_p60_private(const ggml_cgraph * g, int beg, int end, const ggml_tensor * keep) {
    for (int k = beg; k <= end; k++) {
        const ggml_tensor * t = g->nodes[k];
        if (t != keep && (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return false;
        }
    }
    const int lim = std::min(g->n_nodes, end + 400);
    for (int k = end + 1; k < lim; k++) {
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const ggml_tensor * src = g->nodes[k]->src[j];
            if (src == nullptr) {
                continue;
            }
            const ggml_tensor * r = pxa_p60_root(src);
            if (r == keep || pxa_p60_root(keep) == r) {
                continue;
            }
            for (int m = beg; m <= end; m++) {
                if (g->nodes[m] == r || g->nodes[m] == src) {
                    return false;
                }
            }
        }
    }
    return true;
}

static bool pxa_p60_f32_vec(const ggml_tensor * t, int64_t n) {
    return t != nullptr && t->type == GGML_TYPE_F32 && ggml_nelements(t) == n && ggml_is_contiguous(t);
}

static bool pxa_p60_mat(const ggml_tensor * t, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2 = 1) {
    return t != nullptr && t->type == type && t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == ne2 && t->ne[3] == 1 &&
        ggml_is_contiguous(t) && ((uintptr_t) t->data) % 16 == 0;
}

// A matched half: the tensors whose ->data the kernel reads (resolved at launch, so a re-allocated graph is fine) and
// the scalars, cached per (graph, node) with a fingerprint of the nodes it was matched on.
struct pxa_p60_plan {
    int kind = 0; // 0 = none, 1 = MoE half, 2 = GDN half
    int n    = 0; // nodes covered after i
    int il   = -1;
    const ggml_tensor * t[16] = {};
    const ggml_tensor * t2 = nullptr;
    const ggml_tensor * t3 = nullptr; // GDN T > 1: per-step conv capture (SSM_CONV src[4]) or null
    const ggml_tensor * t4 = nullptr; // GDN T > 1: per-step state capture (DELTA_NET src[6]) or null
    int   T = 1;                      // GDN: tokens of the matched half
    float eps[4] = {};
    int   iv[4]  = {};
};

// MoE half. Returns the number of nodes after i that the kernel covered (the caller sets i += n), 0 = no match.
static int pxa_p60_match_moe(const ggml_cgraph * g, int i, pxa_p60_plan & P) {
    static const bool moe_on = pxa_p60_env("PXA_P60_MEGA_MOE", true);
    const ggml_tensor * norm = g->nodes[i];
    if (!moe_on || norm->op != GGML_OP_FUSED_RMS_NORM || norm->src[1] == nullptr) {
        return 0;
    }
    int il = -1;
    if (sscanf(norm->src[1]->name, "blk.%d.post_attention_norm.weight", &il) != 1 || !pxa_p60_is_w(norm->src[1], il, "post_attention_norm.weight")) {
        return 0;
    }
    const ggml_tensor * x_in = norm->src[0];
    {   // PXA_P60_MEGA_DEBUG=2: the op sequence of the first multi-token MoE half seen (to extend the matcher)
        static const int dbg = [] { const char * e = getenv("PXA_P60_MEGA_DEBUG"); return e ? atoi(e) : 0; }();
        static int dumped = 0;
        if (dbg >= 2 && norm->ne[1] > 1 && norm->ne[1] <= 6 && dumped < 2) {
            dumped++;
            for (int k = i; k < g->n_nodes && k < i + 48; k++) {
                const ggml_tensor * t = g->nodes[k];
                fprintf(stderr, "PXA_P60_MEGA_DUMP T=%d %3d %-18s %-40s src0 %s src1 %s\n", (int) norm->ne[1], k - i, ggml_op_desc(t), t->name,
                        t->src[0] ? t->src[0]->name : "-", t->src[1] ? t->src[1]->name : "-");
            }
        }
    }
    int e = -1;
    for (int k = i + 1; k < g->n_nodes && k < i + 64; k++) {
        const ggml_tensor * t = g->nodes[k];
        // the MTP head's MoE half ends at ffn_out-<il>, renamed result_norm by its last callback; in an MTP-armed target
        // the last layer's l_out-<il> is renamed result_mtp_embd (the head's conditioning row, a graph output)
        if (t->op == GGML_OP_ADD && (pxa_p60_named(t, "l_out", il) || pxa_p60_named(t, "ffn_out", il) || strcmp(t->name, "result_norm") == 0 ||
                                     strcmp(t->name, "result_mtp_embd") == 0)) {
            e = k;
            break;
        }
    }
    if (e < 0) {
        PXA_P60_FAIL(norm);
    }
    const ggml_tensor * router = nullptr, * sgate = nullptr, * gate = nullptr, * up = nullptr, * down = nullptr;
    const ggml_tensor * sg = nullptr, * su = nullptr, * sd = nullptr, * ids = nullptr;
    const ggml_tensor * sgate_mm = nullptr, * shexp_gated = nullptr, * routed_in = nullptr, * shexp_sig = nullptr;
    int n_softmax = 0, n_div = 0, n_argsort = 0, n_mma = 0;
    for (int k = i + 1; k < e; k++) {
        const ggml_tensor * t = g->nodes[k];
        switch (t->op) {
            case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_TRANSPOSE: case GGML_OP_PERMUTE: case GGML_OP_NONE:
            case GGML_OP_GET_ROWS: case GGML_OP_SUM_ROWS:
                break;
            case GGML_OP_ARGSORT:
                n_argsort++;
                break;
            case GGML_OP_DIV:
                n_div++;
                break;
            case GGML_OP_SOFT_MAX: {
                float scale, max_bias;
                memcpy(&scale, (const float *) t->op_params + 0, sizeof(float));
                memcpy(&max_bias, (const float *) t->op_params + 1, sizeof(float));
                if (t->src[1] != nullptr || scale != 1.0f || max_bias != 0.0f) PXA_P60_FAIL(t);
                n_softmax++;
                break;
            }
            case GGML_OP_MUL_MAT:
                if      (pxa_p60_is_w(t->src[0], il, "ffn_gate_inp.weight"))       { router = t->src[0]; if (t->src[1] != norm && pxa_p60_root(t->src[1]) != norm) PXA_P60_FAIL(t); }
                else if (pxa_p60_is_w(t->src[0], il, "ffn_gate_inp_shexp.weight")) { sgate  = t->src[0]; sgate_mm = t; if (t->src[1] != norm && pxa_p60_root(t->src[1]) != norm) PXA_P60_FAIL(t); }
                else if (pxa_p60_is_w(t->src[0], il, "ffn_down_shexp.weight"))     { sd     = t->src[0]; }
                else PXA_P60_FAIL(t);
                break;
            case GGML_OP_MOE_FUSED_UP_GATE:
                if (!pxa_p60_is_w(t->src[0], il, "ffn_up_exps.weight") || !pxa_p60_is_w(t->src[1], il, "ffn_gate_exps.weight") ||
                        t->op_params[0] != GGML_UNARY_OP_SILU || t->op_params[2] != 0 || t->src[4] != nullptr || t->src[5] != nullptr ||
                        pxa_p60_root(t->src[2]) != norm) {
                    PXA_P60_FAIL(t);
                }
                up = t->src[0]; gate = t->src[1]; ids = t->src[3];
                if (t->src[2]->ne[1] != 1 || t->src[2]->ne[2] < 1 || t->src[2]->ne[2] > 6) PXA_P60_FAIL(t); // 1..6 tokens
                break;
            case GGML_OP_MUL_MAT_ID:
                if (!pxa_p60_is_w(t->src[0], il, "ffn_down_exps.weight") || t->src[2] != ids) PXA_P60_FAIL(t);
                down = t->src[0];
                break;
            case GGML_OP_MUL_MULTI_ADD:
                n_mma++;
                break;
            case GGML_OP_ADD:
                if (t->src[1] != x_in && t->src[0] != x_in) PXA_P60_FAIL(t);
                routed_in = t;
                break;
            case GGML_OP_FUSED_UP_GATE:
                if (!pxa_p60_is_w(t->src[0], il, "ffn_up_shexp.weight") || !pxa_p60_is_w(t->src[1], il, "ffn_gate_shexp.weight") ||
                        t->op_params[0] != GGML_UNARY_OP_SILU || (t->src[2] != norm && pxa_p60_root(t->src[2]) != norm)) {
                    PXA_P60_FAIL(t);
                }
                su = t->src[0]; sg = t->src[1];
                break;
            case GGML_OP_FUSED_MUL_UNARY:
                if (t->op_params[0] != GGML_UNARY_OP_SIGMOID || t->src[0] != sgate_mm) PXA_P60_FAIL(t);
                shexp_gated = t;
                break;
            case GGML_OP_UNARY:   // 2+ tokens: the shared gate is SIGMOID then MUL (no fused broadcast form)
                if (ggml_get_unary_op(t) != GGML_UNARY_OP_SIGMOID || sgate_mm == nullptr || t->src[0] != sgate_mm || shexp_sig != nullptr) PXA_P60_FAIL(t);
                shexp_sig = t;
                break;
            case GGML_OP_MUL:
                if (shexp_sig == nullptr || (t->src[0] != shexp_sig && t->src[1] != shexp_sig)) PXA_P60_FAIL(t);
                shexp_gated = t;
                break;
            default:
                PXA_P60_FAIL(t);
        }
    }
    const ggml_tensor * add = g->nodes[e];
    if (!router || !sgate || !gate || !up || !down || !sg || !su || !sd || !ids || !shexp_gated || !routed_in ||
            n_softmax != 1 || n_div != 1 || n_argsort != 1 || n_mma != 1 ||
            !((add->src[0] == routed_in && add->src[1] == shexp_gated) || (add->src[1] == routed_in && add->src[0] == shexp_gated))) {
        PXA_P60_FAIL(add);
    }
    const ggml_type gut = gate->type, dnt = down->type;
    const int64_t T = add->ne[1];
    const ggml_type rt = router->type;
    if (ids->type != GGML_TYPE_I32 || ids->ne[0] != 8 || ids->ne[1] != T || T < 1 || T > 6 ||
            (T > 1 && gut != GGML_TYPE_PXQN4 && !(gut == GGML_TYPE_Q4_K || dnt == GGML_TYPE_Q6_K)) ||
            !(rt == GGML_TYPE_F32 || rt == GGML_TYPE_BF16) || sgate->type != rt || !pxa_p60_mat(router, rt, 2048, 256) ||
            ggml_nelements(sgate) != 2048 || !ggml_is_contiguous(sgate) ||
            !(gut == GGML_TYPE_Q4_K || gut == GGML_TYPE_Q5_K || gut == GGML_TYPE_PXQN4) || up->type != gut ||
            !pxa_p60_mat(gate, gut, 2048, 512, 256) || !pxa_p60_mat(up, gut, 2048, 512, 256) ||
            !(dnt == GGML_TYPE_Q5_K || dnt == GGML_TYPE_Q6_K || dnt == GGML_TYPE_PXQN4) || !pxa_p60_mat(down, dnt, 512, 2048, 256) ||
            ((gut == GGML_TYPE_PXQN4) != (dnt == GGML_TYPE_PXQN4)) || (gut == GGML_TYPE_PXQN4 && up->nb[2] != gate->nb[2]) ||
            !pxa_p60_mat(sg, GGML_TYPE_Q8_0, 2048, 512) || !pxa_p60_mat(su, GGML_TYPE_Q8_0, 2048, 512) || !pxa_p60_mat(sd, GGML_TYPE_Q8_0, 512, 2048) ||
            !pxa_p60_f32_vec(norm->src[1], 2048) || !pxa_p60_f32_vec(x_in, 2048 * T) || !pxa_p60_f32_vec(add, 2048 * T) ||
            !pxa_p60_private(g, i, e, add)) {
        {   // PXA_P60_MEGA_DEBUG=3 (diagnostic only): name the sub-condition that failed, per token count
            static const int dbg3 = [] { const char * s = getenv("PXA_P60_MEGA_DEBUG"); return s ? atoi(s) : 0; }();
            static int n3[8][8] = {};
            if (dbg3 >= 3) {
                int why = 7;
                if (ids->type != GGML_TYPE_I32 || ids->ne[0] != 8 || ids->ne[1] != T || T < 1 || T > 6) why = 0;
                else if (!(rt == GGML_TYPE_F32 || rt == GGML_TYPE_BF16) || sgate->type != rt || !pxa_p60_mat(router, rt, 2048, 256)) why = 1;
                else if (!pxa_p60_mat(gate, gut, 2048, 512, 256) || !pxa_p60_mat(up, gut, 2048, 512, 256) || !pxa_p60_mat(down, dnt, 512, 2048, 256)) why = 2;
                else if (!pxa_p60_f32_vec(x_in, 2048 * T)) why = 3;
                else if (!pxa_p60_f32_vec(add, 2048 * T)) why = 4;
                else if (!pxa_p60_private(g, i, e, add)) why = 5;
                const int tt = T >= 1 && T <= 7 ? (int) T : 0;
                if (n3[tt][why]++ < 4) {
                    fprintf(stderr, "PXA_P60_MEGA_DEBUG3: MoE layer %d T=%d fails check %d (0 ids 1 router 2 weights 3 x_in 4 out 5 private 7 other) "
                            "x_in %s view_src %s nb1 %zu ne %lld,%lld | out flags %d | ids %s op %s type %s ne %lld,%lld,%lld,%lld view_src %s (%s ne %lld,%lld) | n_nodes %d i %d\n",
                            il, (int) T, why, x_in->name,
                            x_in->view_src ? x_in->view_src->name : "-", x_in->nb[1], (long long) x_in->ne[0], (long long) x_in->ne[1], add->flags,
                            ids->name, ggml_op_desc(ids), ggml_type_name(ids->type), (long long) ids->ne[0], (long long) ids->ne[1], (long long) ids->ne[2], (long long) ids->ne[3],
                            ids->view_src ? ids->view_src->name : "-", ids->view_src ? ggml_op_desc(ids->view_src) : "-",
                            ids->view_src ? (long long) ids->view_src->ne[0] : -1LL, ids->view_src ? (long long) ids->view_src->ne[1] : -1LL, g->n_nodes, i);
                    if (why == 5) {
                        for (int k = i; k <= e; k++) {
                            if (g->nodes[k] != add && (g->nodes[k]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                                fprintf(stderr, "PXA_P60_MEGA_DEBUG3:   output flag on %s\n", g->nodes[k]->name);
                            }
                        }
                        const int lim = std::min(g->n_nodes, e + 400);
                        for (int k = e + 1; k < lim; k++) {
                            for (int j = 0; j < GGML_MAX_SRC; j++) {
                                const ggml_tensor * src = g->nodes[k]->src[j];
                                if (src == nullptr) continue;
                                const ggml_tensor * r = pxa_p60_root(src);
                                if (r == add || pxa_p60_root(add) == r) continue;
                                for (int m = i; m <= e; m++) {
                                    if (g->nodes[m] == r || g->nodes[m] == src) {
                                        fprintf(stderr, "PXA_P60_MEGA_DEBUG3:   later node %d %s (%s) reads %s\n", k - e, g->nodes[k]->name,
                                                ggml_op_desc(g->nodes[k]), g->nodes[m]->name);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        PXA_P60_FAIL(add);
    }
    P.kind = 1; P.n = e - i; P.il = il;
    P.iv[0] = (int) T;
    const ggml_tensor * TT[] = { x_in, add, norm->src[1], router, sgate, gate, up, down, sg, su, sd };
    for (int k = 0; k < 11; k++) P.t[k] = TT[k];
    memcpy(&P.eps[0], norm->op_params, sizeof(float));
    return e - i;
}

// the ADD feeding a matched MoE half (as its x_in) can be folded into the half: a plain same-shape f32 add
static bool pxa_p60_add_foldable(const ggml_tensor * add, const ggml_tensor * x_in) {
    const ggml_tensor * a = add->src[0], * b = add->src[1];
    return add == x_in && add->op == GGML_OP_ADD && add->type == GGML_TYPE_F32 && a != nullptr && b != nullptr &&
           a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && ggml_are_same_shape(a, add) && ggml_are_same_shape(b, add) &&
           ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(add);
}

static bool pxa_p60_launch_moe(ggml_backend_cuda_context & ctx, const pxa_p60_plan & P, const ggml_tensor * add = nullptr) {
    pxa_p60_moe_args a = {};
    a.x_in = (const float *) P.t[0]->data; a.x_out = (float *) P.t[1]->data;
    a.norm = (const float *) P.t[2]->data; a.eps = P.eps[0];
    if (P.t[3]->type == GGML_TYPE_BF16) {
        a.router = pxa_p60_bf16_to_f32(ctx, P.t[3]->data, 2048 * 256);
        a.sgate  = pxa_p60_bf16_to_f32(ctx, P.t[4]->data, 2048);
        if (a.router == nullptr || a.sgate == nullptr) {
            return false;
        }
    } else {
        a.router = (const float *) P.t[3]->data; a.sgate = (const float *) P.t[4]->data;
    }
    a.gate = (const char *) P.t[5]->data; a.up = (const char *) P.t[6]->data; a.down = (const char *) P.t[7]->data;
    a.sg = (const char *) P.t[8]->data; a.su = (const char *) P.t[9]->data; a.sd = (const char *) P.t[10]->data;
    a.gu_type = P.t[5]->type; a.dn_type = P.t[7]->type;
    a.gu_nb2 = P.t[5]->nb[2]; a.dn_nb2 = P.t[7]->nb[2];
    const int T = P.iv[0];
    static const bool multi = pxa_p60_env("PXA_P60_MEGA_T", true);
    // PXA_P60_MEGA_PXQN_T=1 (default OFF): the 2..6-token PXQN4 MoE half (v2 kernel, pxqn-mega.cu pxt2: 167 us per layer
    // at 4 tokens on a P100 vs ~260 for the routed-expert GEMV path; =9 selects the first version, 513 us)
    static const bool pxq_multi = pxa_p60_env("PXA_P60_MEGA_PXQN_T", false);
    if (add != nullptr) {   // PXA_P60_MEGA_ADDFOLD: only the 2..6-token PXQN4 v2 kernel takes the residual ADD
        if (a.gu_type != GGML_TYPE_PXQN4 || T < 2 || !multi || !pxq_multi || !pxa_p60_add_foldable(add, P.t[0]) ||
                !ggml_cuda_pxqn_mega_moe_t_fold(ctx, a, T, (const float *) add->src[0]->data, (const float *) add->src[1]->data,
                                                (float *) add->data)) {
            return false;
        }
        static bool logged_f = false;
        if (!logged_f) {
            logged_f = true;
            fprintf(stderr, "PXA_P60_MEGA_ADDFOLD: residual ADD folded into the MoE half (dev %d, layer %d, %d tokens)\n", ctx.device, P.il, T);
        }
        return true;
    }
    // PXA_P60_MEGA_MOE_INV (default 0 = off while measured; 1 = on): the 2..6-token PXQN4 MoE half per token
    // bit-identical to the single-token kernel (pxt3) instead of pxt2
    static const bool moe_inv = [] {
        const bool v = pxa_p60_env("PXA_P60_MEGA_MOE_INV", false);
        fprintf(stderr, "PXA_P60_MEGA_MOE_INV: %d (2..6-token PXQN4 MoE half per token = the single-token kernel)\n", (int) v);
        return v;
    }();
    // PXA_P60_MEGA_MOE_W1 (default 0 = off while measured; 1 = on): the one-token PXQN4 MoE half runs the
    // verify kernel's one-token instance (pxt2<1>) instead of pxm. pxt2's per-token arithmetic does not depend on the width
    // (PXA_P60_MEGA_MOE_W1_CHECK=N compares each token of a 2..6-token call with its own one-token call), so with it a
    // verify computes for every token what plain decode computes: MTP == plain for the MoE half. Not bit-identical to pxm
    // (the stack-off path); only with the pxt2 verify (PXA_P60_MEGA_PXQN_T=1, MOE_INV off).
    static const bool moe_w1 = [] {
        const bool v = pxa_p60_env("PXA_P60_MEGA_MOE_W1", false);
        fprintf(stderr, "PXA_P60_MEGA_MOE_W1: %d (one-token PXQN4 MoE half = the verify kernel's one-token instance)\n", (int) v);
        return v;
    }();
    if (a.gu_type == GGML_TYPE_PXQN4 && T == 1 && moe_w1 && multi && pxq_multi && !moe_inv) {
        if (!ggml_cuda_pxqn_mega_moe_t(ctx, a, 1)) {
            return false;
        }
    } else if (a.gu_type == GGML_TYPE_PXQN4) {
        if (T == 1 ? !ggml_cuda_pxqn_mega_moe(ctx, a)
                   : (!multi || !pxq_multi || !(moe_inv ? ggml_cuda_pxqn_mega_moe_t3(ctx, a, T) : ggml_cuda_pxqn_mega_moe_t(ctx, a, T)))) {
            return false;
        }
    } else if (T == 1 ? !pxa_p60_mega_moe(ctx, a) : (!multi || !pxa_p60_mega_moe_t(ctx, a, T))) {
        return false;
    }
    static bool logged[7] = {};
    if (!logged[T]) {
        logged[T] = true;
        fprintf(stderr, "PXA_P60_MEGA: MoE half FIRED (dev %d, layer %d, %d token(s), gate/up %s, down %s, router %s)\n", ctx.device, P.il, T,
                ggml_type_name(P.t[5]->type), ggml_type_name(P.t[7]->type), ggml_type_name(P.t[3]->type));
    }
    return true;
}

// Gated-DeltaNet half (build_layer_attn_linear of llama-delta-net.cpp, decode: one token, one sequence, no snapshots):
//   FUSED_RMS_NORM(x, attn_norm) .. MUL_MAT attn_qkv / attn_gate / ssm_beta / ssm_alpha, GET_ROWS(cache_s_l<il>) * mask,
//   SSM_CONV, SILU, CONT, L2_NORM q / k, ADD dt + SOFTPLUS + MUL a, DELTA_NET, CONCAT + SET_ROWS (state write-back),
//   FUSED_RMS_NORM(o, ssm_norm), FUSED_MUL_UNARY(silu z), MUL_MAT ssm_out .. ADD ssm_output-<il>.
// PXA_P60_MEGA_GDN_T (default 0 = off while measured; 1 = on): the same half for T = 2..6 tokens of one sequence
// (speculative verify) as one cooperative kernel whose per-token arithmetic is the single-token kernel's
// (ggml_cuda_pxqn_mega_gdn_t, PXQN4 projections only). The dense per-step captures of SSM_CONV / DELTA_NET are written in
// their own layouts; a strided (PXA_RS_RING) or replay capture declines.
static bool pxa_p60_gdn_t_on() {
    static const bool on = [] {
        const bool v = pxa_p60_env("PXA_P60_MEGA_GDN_T", false);
        fprintf(stderr, "PXA_P60_MEGA_GDN_T: %d (2..6-token GDN half as one launch, per-token = the single-token kernel)\n", (int) v);
        return v;
    }();
    return on;
}

static int pxa_p60_match_gdn(const ggml_cgraph * g, int i, pxa_p60_plan & P) {
    static const bool gdn_on = pxa_p60_env("PXA_P60_MEGA_GDN", true);
    const ggml_tensor * norm = g->nodes[i];
    if (!gdn_on || norm->op != GGML_OP_FUSED_RMS_NORM || norm->src[1] == nullptr) {
        return 0;
    }
    const int T = (int) norm->ne[1];
    if (T < 1 || T > 6 || norm->ne[2] != 1 || norm->ne[3] != 1 || (T > 1 && !pxa_p60_gdn_t_on())) {
        return 0;
    }
    const ggml_tensor * step_conv = nullptr, * step_ssm = nullptr;
    int il = -1;
    if (sscanf(norm->src[1]->name, "blk.%d.attn_norm.weight", &il) != 1 || !pxa_p60_is_w(norm->src[1], il, "attn_norm.weight")) {
        return 0;
    }
    const ggml_tensor * x_in = norm->src[0];
    int e = -1;
    for (int k = i + 1; k < g->n_nodes && k < i + 100; k++) {
        const ggml_tensor * t = g->nodes[k];
        if (t->op == GGML_OP_ADD && pxa_p60_named(t, "ssm_output", il)) {
            e = k;
            break;
        }
        if (t->op == GGML_OP_FLASH_ATTN_EXT) {
            return 0; // an attention layer
        }
    }
    if (e < 0) {
        PXA_P60_FAIL(norm);
    }
    const ggml_tensor * qkv = nullptr, * z = nullptr, * out = nullptr, * alpha = nullptr, * beta = nullptr;
    const ggml_tensor * dt = nullptr, * ssa = nullptr, * conv = nullptr, * snorm = nullptr, * getrows = nullptr, * setrows = nullptr;
    const ggml_tensor * maskmul = nullptr, * dn = nullptr, * concat = nullptr, * gnorm = nullptr, * gate = nullptr, * out_mm = nullptr;
    const ggml_tensor * rht_in = nullptr, * rht_out = nullptr;
    float eps_q = -1.0f, eps_k = -1.0f, eps_gn = -1.0f;
    int n_silu = 0, n_softplus = 0;
    for (int k = i + 1; k < e; k++) {
        const ggml_tensor * t = g->nodes[k];
        switch (t->op) {
            case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_TRANSPOSE: case GGML_OP_PERMUTE: case GGML_OP_NONE:
            case GGML_OP_CONT:
                break;
            case GGML_OP_PXQN_RHT:
                if (t->src[0] == norm && rht_in == nullptr) rht_in = t;
                else if (rht_out == nullptr && gate != nullptr && pxa_p60_root(t->src[0]) == gate) rht_out = t;
                else PXA_P60_FAIL(t);
                break;
            case GGML_OP_MUL_MAT: {
                const ggml_tensor * xin = rht_in ? rht_in : norm;
                if      (pxa_p60_is_w(t->src[0], il, "attn_qkv.weight"))  { qkv   = t->src[0]; if (t->src[1] != xin) PXA_P60_FAIL(t); }
                else if (pxa_p60_is_w(t->src[0], il, "attn_gate.weight")) { z     = t->src[0]; if (t->src[1] != xin) PXA_P60_FAIL(t); }
                else if (pxa_p60_is_w(t->src[0], il, "ssm_alpha.weight")) { alpha = t->src[0]; if (t->src[1] != xin) PXA_P60_FAIL(t); }
                else if (pxa_p60_is_w(t->src[0], il, "ssm_beta.weight"))  { beta  = t->src[0]; if (t->src[1] != xin) PXA_P60_FAIL(t); }
                else if (pxa_p60_is_w(t->src[0], il, "ssm_out.weight"))   { out   = t->src[0]; out_mm = t;
                                                                             if (rht_out && t->src[1] != rht_out) PXA_P60_FAIL(t); }
                else PXA_P60_FAIL(t);
                break;
            }
            case GGML_OP_GET_ROWS: {
                char nm[64];
                snprintf(nm, sizeof(nm), "cache_s_l%d", il);
                if (getrows != nullptr || strcmp(pxa_p60_root(t->src[0])->name, nm) != 0 || t->src[1]->type != GGML_TYPE_I32 ||
                        ggml_nelements(t->src[1]) != 1) {
                    PXA_P60_FAIL(t);
                }
                getrows = t;
                break;
            }
            case GGML_OP_MUL:
                if      (pxa_p60_is_w(t->src[1], il, "ssm_a")) { ssa = t->src[1]; }
                else if (getrows != nullptr && t->src[0] == getrows && ggml_nelements(t->src[1]) == 1) { maskmul = t; }
                else PXA_P60_FAIL(t);
                break;
            case GGML_OP_SSM_CONV:
                if (t->src[1]->ne[1] != T || (T == 1 && t->src[4] != nullptr)) PXA_P60_FAIL(t);
                if (T > 1 && (t->op_params[2] != 0 || t->op_params[3] != 0 || (t->src[3] != nullptr && t->src[3]->ne[0] != 1))) PXA_P60_FAIL(t);
                if (T > 1 && t->src[4] != nullptr) {
                    if (t->src[4]->type != GGML_TYPE_F32 || !ggml_is_contiguous(t->src[4]) ||
                            ggml_nelements(t->src[4]) < (int64_t) T * 3 * 8192) PXA_P60_FAIL(t);
                    step_conv = t->src[4];
                }
                conv = t->src[2];
                break;
            case GGML_OP_UNARY: {
                const ggml_unary_op u = ggml_get_unary_op(t);
                if (u == GGML_UNARY_OP_SILU) n_silu++;
                else if (u == GGML_UNARY_OP_SOFTPLUS) n_softplus++;
                else PXA_P60_FAIL(t);
                break;
            }
            case GGML_OP_L2_NORM: {
                float eps;
                memcpy(&eps, t->op_params, sizeof(float));
                // T > 1: the norm reads the (token, head) permuted view of q_conv / k_conv
                const ggml_tensor * ls = (T > 1 && t->src[0]->op == GGML_OP_PERMUTE) ? t->src[0]->src[0] : t->src[0];
                if      (pxa_p60_named(ls, "q_conv", il)) eps_q = eps;
                else if (pxa_p60_named(ls, "k_conv", il)) eps_k = eps;
                else PXA_P60_FAIL(t);
                break;
            }
            case GGML_OP_ADD:
                if (!pxa_p60_is_w(t->src[1], il, "ssm_dt.bias")) PXA_P60_FAIL(t);
                dt = t->src[1];
                break;
            case GGML_OP_DELTA_NET:
                if (dn != nullptr || t->op_params[1] != 0 || (T == 1 && t->src[6] != nullptr)) PXA_P60_FAIL(t);
                if (T > 1 && (t->op_params[2] != 0 || t->op_params[3] != 0 || t->op_params[4] != 0)) PXA_P60_FAIL(t);
                if (T > 1 && t->src[6] != nullptr) {
                    if (t->src[6]->type != GGML_TYPE_F32 || !ggml_is_contiguous(t->src[6]) ||
                            ggml_nelements(t->src[6]) < (int64_t) (T - 1) * 128 * 128 * 32) PXA_P60_FAIL(t);
                    step_ssm = t->src[6];
                }
                dn = t;
                break;
            case GGML_OP_CONCAT:
                if (concat != nullptr) PXA_P60_FAIL(t);
                concat = t;
                break;
            case GGML_OP_SET_ROWS:
                if (setrows != nullptr || t->src[0] != concat || t->src[1]->type != GGML_TYPE_I32 || ggml_nelements(t->src[1]) != 1) PXA_P60_FAIL(t);
                setrows = t;
                break;
            case GGML_OP_FUSED_RMS_NORM:
                if (gnorm != nullptr || !pxa_p60_is_w(t->src[1], il, "ssm_norm.weight")) PXA_P60_FAIL(t);
                snorm = t->src[1];
                memcpy(&eps_gn, t->op_params, sizeof(float));
                gnorm = t;
                break;
            case GGML_OP_FUSED_MUL_UNARY:
                if (gate != nullptr || t->op_params[0] != GGML_UNARY_OP_SILU || t->src[1] != gnorm) PXA_P60_FAIL(t);
                gate = t;
                break;
            default:
                PXA_P60_FAIL(t);
        }
    }
    const ggml_tensor * add = g->nodes[e];
    if (!qkv || !z || !out || !alpha || !beta || !dt || !ssa || !conv || !snorm || !getrows || !setrows || !maskmul || !dn ||
            !concat || !gnorm || !gate || !out_mm || n_silu != 1 || n_softplus != 1 || eps_q < 0.0f || eps_k < 0.0f || eps_gn < 0.0f ||
            !((add->src[0] == out_mm && add->src[1] == x_in) || (add->src[1] == out_mm && add->src[0] == x_in))) {
        PXA_P60_FAIL(add);
    }
    // one token, one sequence, the 35B-A3B shapes; state row = [conv 3*8192 | ssm 32*128*128]
    const ggml_tensor * dq = dn->src[0], * dv = dn->src[2];
    const bool pxq = qkv->type == GGML_TYPE_PXQN4;
    if (dq->ne[0] != 128 || dq->ne[1] != T || dq->ne[2] != 16 || dq->ne[3] != 1 || dv->ne[0] != 128 || dv->ne[2] != 32 ||
            (T > 1 && !pxq) ||
            ggml_nelements(concat->src[0]) != 3 * 8192 || ggml_nelements(concat->src[1]) != 128 * 128 * 32 ||
            getrows->src[0]->ne[0] != 3 * 8192 + 128 * 128 * 32 || setrows->ne[0] != getrows->src[0]->ne[0] ||
            getrows->src[0]->type != GGML_TYPE_F32 || setrows->type != GGML_TYPE_F32 ||
            !(pxq ? (pxa_p60_mat(qkv, GGML_TYPE_PXQN4, 2048, 8192) && pxa_p60_mat(z, GGML_TYPE_PXQN4, 2048, 4096) &&
                     pxa_p60_mat(out, GGML_TYPE_PXQN4, 4096, 2048) && pxa_p60_mat(alpha, GGML_TYPE_Q8_0, 2048, 32) &&
                     pxa_p60_mat(beta, GGML_TYPE_Q8_0, 2048, 32) && rht_in && rht_out)
                  : (pxa_p60_mat(qkv, GGML_TYPE_Q8_0, 2048, 8192) && pxa_p60_mat(z, GGML_TYPE_Q8_0, 2048, 4096) &&
                     pxa_p60_mat(out, GGML_TYPE_Q8_0, 4096, 2048) && pxa_p60_mat(alpha, GGML_TYPE_F32, 2048, 32) &&
                     pxa_p60_mat(beta, GGML_TYPE_F32, 2048, 32) && !rht_in && !rht_out)) ||
            !pxa_p60_mat(conv, GGML_TYPE_F32, 4, 8192) ||
            !pxa_p60_f32_vec(dt, 32) || !pxa_p60_f32_vec(ssa, 32) || !pxa_p60_f32_vec(snorm, 128) || !pxa_p60_f32_vec(norm->src[1], 2048) ||
            !pxa_p60_f32_vec(x_in, 2048 * T) || !pxa_p60_f32_vec(add, 2048 * T) || x_in->ne[0] != 2048 || add->ne[0] != 2048 ||
            maskmul->src[1]->type != GGML_TYPE_F32 ||
            !pxa_p60_private(g, i, e, add)) {
        PXA_P60_FAIL(add);
    }
    P.kind = 2; P.n = e - i; P.il = il; P.T = T; P.t3 = step_conv; P.t4 = step_ssm;
    const ggml_tensor * TG[] = { x_in, add, norm->src[1], qkv, z, out, alpha, beta, dt, ssa, conv, snorm, getrows, setrows, maskmul, rht_in };
    for (int k = 0; k < 16; k++) P.t[k] = TG[k];
    P.t2 = rht_out;
    memcpy(&P.eps[0], norm->op_params, sizeof(float));
    P.eps[1] = eps_q; P.eps[2] = eps_k; P.eps[3] = eps_gn;
    P.iv[0] = dn->op_params[0];
    return e - i;
}

static bool pxa_p60_launch_gdn(ggml_backend_cuda_context & ctx, const pxa_p60_plan & P) {
    pxa_p60_gdn_args a = {};
    a.x_in = (const float *) P.t[0]->data; a.x_out = (float *) P.t[1]->data;
    a.attn_norm = (const float *) P.t[2]->data; a.eps_norm = P.eps[0];
    a.qkv = (const char *) P.t[3]->data; a.z = (const char *) P.t[4]->data; a.out = (const char *) P.t[5]->data;
    a.alpha = (const float *) P.t[6]->data; a.beta = (const float *) P.t[7]->data;
    a.dt = (const float *) P.t[8]->data; a.a = (const float *) P.t[9]->data; a.conv = (const float *) P.t[10]->data;
    a.snorm = (const float *) P.t[11]->data;
    a.eps_l2q = P.eps[1]; a.eps_l2k = P.eps[2]; a.eps_gn = P.eps[3];
    a.repeat_type = P.iv[0];
    const ggml_tensor * getrows = P.t[12], * setrows = P.t[13], * maskmul = P.t[14];
    a.s_rd_base = (const char *) getrows->src[0]->data; a.rd_ids = (const int32_t *) getrows->src[1]->data; a.rd_stride = getrows->src[0]->nb[1];
    a.s_wr_base = (char *) setrows->data; a.wr_ids = (const int32_t *) setrows->src[1]->data; a.wr_stride = setrows->nb[1];
    a.mask = (const float *) maskmul->src[1]->data;
    a.proj_type = P.t[3]->type;
    bool ok;
    if (a.proj_type == GGML_TYPE_PXQN4) {
        const ggml_tensor * ri = P.t[15], * ro = P.t2;
        a.rht_seed = (uint64_t) (uint32_t) ri->op_params[0] | ((uint64_t) (uint32_t) ri->op_params[1] << 32);
        const uint64_t seed_out = (uint64_t) (uint32_t) ro->op_params[0] | ((uint64_t) (uint32_t) ro->op_params[1] << 32);
        if (seed_out != a.rht_seed) {
            return false;
        }
        a.rht_layer_in = ri->op_params[2]; a.rht_site_in = ri->op_params[3]; a.rht_k0_in = ri->op_params[4];
        a.rht_layer_out = ro->op_params[2]; a.rht_site_out = ro->op_params[3]; a.rht_k0_out = ro->op_params[4];
        a.alpha_q8 = (const char *) P.t[6]->data; a.beta_q8 = (const char *) P.t[7]->data;
        if (P.T > 1) {
            pxa_p60_gdn_t_args at = {};
            at.g = a;
            at.conv_steps = P.t3 ? (float *) P.t3->data : nullptr; at.conv_step = 3 * 8192;
            at.ssm_steps  = P.t4 ? (float *) P.t4->data : nullptr; at.ssm_step  = 128 * 128 * 32;
            ok = ggml_cuda_pxqn_mega_gdn_t(ctx, at, P.T);
        } else {
            ok = ggml_cuda_pxqn_mega_gdn(ctx, a);
        }
    } else {
        ok = pxa_p60_mega_gdn(ctx, a);
    }
    if (!ok) {
        return false;
    }
    static bool logged[7] = {};
    if (!logged[P.T]) {
        logged[P.T] = true;
        fprintf(stderr, "PXA_P60_MEGA: GDN half FIRED (dev %d, layer %d, %d token(s), %s projections, repeat_type %d, eps q/k/gn %g/%g/%g, steps conv %d ssm %d)\n",
                ctx.device, P.il, P.T, ggml_type_name((ggml_type) a.proj_type), a.repeat_type, a.eps_l2q, a.eps_l2k, a.eps_gn,
                P.t3 != nullptr, P.t4 != nullptr);
    }
    return true;
}

// Full-attention half (build_std_attention, decode, one token):
//   FUSED_RMS_NORM(x, attn_norm) .. MUL_MAT attn_q (q | gate interleaved per head) / attn_k / attn_v, CONT(VIEW q),
//   FUSED_RMS_NORM q_norm / k_norm, ROPE (multi) q / k, CPY k / v -> cache views, FLASH_ATTN_EXT, CONT(VIEW gate),
//   SIGMOID, MUL, MUL_MAT attn_output .. ADD(attn_out-<il>, x).
// PXA_P60_MEGA_ATTN_T (default 2 at ENHANCE since 2026-10-20, =0 off; 1 = the whole INV half): the PXQN4 attention half for T = 2..6 tokens of one
// sequence as pre / fa / post launches whose per-token arithmetic is the single-token kernels' (ggml_cuda_pxqn_mega_attn_t)
// =2 (C033 lever 2): only the TAIL -- FLASH_ATTN_EXT .. the residual ADD -- as attn_fa_t + attn_post_t; the input
// projections (k_pxqn_pre under PXA_P60_PRE_VW), q / k norms, RoPE and the cache writes stay graph nodes (pxa_p60_match_attn_tail)
static int pxa_p60_attn_t_mode() {
    static const int m = [] {
        // default 2 at the ENHANCE level (2026-10-20, q11d #29139: +6.1% code / +2.7% chat MTP decode on the 35B P100 pair;
        // not bit-identical to the graph's multi-token flash attention; KLD at the width band, not inside it: ub4 0.004185 vs
        // the ub1 band 0.003913, 7% above, about 1 SE; the needle gate and a same-bracket re-KLD are owed): reached
        // only inside the P60 mega stack (PXA_P60_MEGA, the PXA_AUTO_P60 cell); =0 puts the graph's attention tail back
        const char * e = getenv("PXA_P60_MEGA_ATTN_T");
        const int v = e && *e ? atoi(e) : (ggml_pxa_config_level() >= 2 ? 2 : 0);
        fprintf(stderr, "PXA_P60_MEGA_ATTN_T: %d (2..6-token PXQN4 attention half: 1 = whole half, per-token = the single-token kernels; "
                        "2 = flash attention + output half only)\n", v);
        return v;
    }();
    return m;
}
static bool pxa_p60_attn_t_on() {
    return pxa_p60_attn_t_mode() == 1;
}

static int pxa_p60_match_attn(const ggml_cgraph * g, int i, pxa_p60_plan & P) {
    static const bool attn_on = pxa_p60_env("PXA_P60_MEGA_ATTN", true);
    const ggml_tensor * norm = g->nodes[i];
    if (!attn_on || norm->op != GGML_OP_FUSED_RMS_NORM || norm->src[1] == nullptr) {
        return 0;
    }
    int il = -1;
    if (sscanf(norm->src[1]->name, "blk.%d.attn_norm.weight", &il) != 1 || !pxa_p60_is_w(norm->src[1], il, "attn_norm.weight")) {
        return 0;
    }
    const ggml_tensor * x_in = norm->src[0];
    int e = -1;
    for (int k = i + 1; k < g->n_nodes && k < i + 64; k++) {
        const ggml_tensor * t = g->nodes[k];
        if (t->op == GGML_OP_ADD && ((pxa_p60_named(t->src[0], "attn_out", il) && t->src[1] == x_in) ||
                                     (pxa_p60_named(t->src[1], "attn_out", il) && t->src[0] == x_in))) {
            e = k;
            break;
        }
        if (t->op == GGML_OP_DELTA_NET) {
            return 0; // a GDN layer
        }
    }
    if (e < 0) {
        return 0;
    }
    const ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr, * qmm = nullptr, * kmm = nullptr, * vmm = nullptr;
    const ggml_tensor * qn = nullptr, * kn = nullptr, * rq = nullptr, * rk = nullptr, * cpk = nullptr, * cpv = nullptr, * fa = nullptr;
    const ggml_tensor * qcont = nullptr, * gcont = nullptr, * sig = nullptr, * gmul = nullptr, * omm = nullptr;
    const ggml_tensor * rht_in = nullptr, * rht_out = nullptr;
    for (int k = i + 1; k < e; k++) {
        const ggml_tensor * t = g->nodes[k];
        switch (t->op) {
            case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_TRANSPOSE: case GGML_OP_PERMUTE: case GGML_OP_NONE:
                break;
            case GGML_OP_PXQN_RHT:
                if (t->src[0] == norm && rht_in == nullptr) rht_in = t;
                else if (gmul != nullptr && t->src[0] == gmul && rht_out == nullptr) rht_out = t;
                else PXA_P60_FAIL(t);
                break;
            case GGML_OP_MUL_MAT: {
                const ggml_tensor * xin = rht_in ? rht_in : norm;
                if      (pxa_p60_is_w(t->src[0], il, "attn_q.weight"))      { wq = t->src[0]; qmm = t; if (t->src[1] != xin) PXA_P60_FAIL(t); }
                else if (pxa_p60_is_w(t->src[0], il, "attn_k.weight"))      { wk = t->src[0]; kmm = t; if (t->src[1] != xin) PXA_P60_FAIL(t); }
                else if (pxa_p60_is_w(t->src[0], il, "attn_v.weight"))      { wv = t->src[0]; vmm = t; if (t->src[1] != xin) PXA_P60_FAIL(t); }
                else if (pxa_p60_is_w(t->src[0], il, "attn_output.weight")) { wo = t->src[0]; omm = t; }
                else PXA_P60_FAIL(t);
                break;
            }
            case GGML_OP_CONT:
                if (t->src[0]->op != GGML_OP_VIEW || t->src[0]->view_src != qmm || qmm == nullptr) PXA_P60_FAIL(t);
                if      (t->src[0]->view_offs == 0)                 qcont = t;
                else if (t->src[0]->view_offs == 256 * sizeof(float)) gcont = t;
                else PXA_P60_FAIL(t);
                if (t->src[0]->ne[0] != 256 || t->src[0]->nb[1] != 2 * 256 * sizeof(float) || ggml_nelements(t) % 4096 != 0) PXA_P60_FAIL(t);
                break;
            case GGML_OP_FUSED_RMS_NORM:
                if      (pxa_p60_is_w(t->src[1], il, "attn_q_norm.weight")) { qn = t; if (t->src[0] != qcont) PXA_P60_FAIL(t); }
                else if (pxa_p60_is_w(t->src[1], il, "attn_k_norm.weight")) { kn = t; if (pxa_p60_root(t->src[0]) != kmm) PXA_P60_FAIL(t); }
                else PXA_P60_FAIL(t);
                break;
            case GGML_OP_ROPE:
                if      (t->src[0] == qn && qn) rq = t;
                else if (t->src[0] == kn && kn) rk = t;
                else PXA_P60_FAIL(t);
                break;
            case GGML_OP_CPY:
                if      (t->src[0] == rk && rk)  cpk = t;
                else if (t->src[0] == vmm && vmm) cpv = t;
                else PXA_P60_FAIL(t);
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                if (fa != nullptr) PXA_P60_FAIL(t);
                fa = t;
                break;
            case GGML_OP_UNARY:
                if (ggml_get_unary_op(t) != GGML_UNARY_OP_SIGMOID || t->src[0] != gcont || gcont == nullptr) PXA_P60_FAIL(t);
                sig = t;
                break;
            case GGML_OP_MUL:
                if (sig == nullptr || (t->src[1] != sig && t->src[0] != sig)) PXA_P60_FAIL(t);
                gmul = t;
                break;
            default:
                PXA_P60_FAIL(t);
        }
    }
    const ggml_tensor * add = g->nodes[e];
    if (!wq || !wk || !wv || !wo || !qn || !kn || !rq || !rk || !cpk || !cpv || !fa || !qcont || !gcont || !sig || !gmul || !omm ||
            omm->src[1] != (rht_out ? rht_out : gmul) || (rht_in == nullptr) != (rht_out == nullptr)) {
        PXA_P60_FAIL(add);
    }
    const bool pxq = wq->type == GGML_TYPE_PXQN4;
    // rope: multi (M-RoPE / IMRoPE), shared parameters, no frequency factors, ext_factor 0, not flipped
    for (int k = 0; k < 16; k++) {
        if (rq->op_params[k] != rk->op_params[k]) PXA_P60_FAIL(rk);
    }
    const int mode = rq->op_params[2], n_dims = rq->op_params[1];
    float ext_factor;
    memcpy(&ext_factor, (const int32_t *) rq->op_params + 7, sizeof(float));
    if (!(mode & GGML_ROPE_TYPE_MROPE) || mode == GGML_ROPE_TYPE_VISION || ext_factor != 0.0f || rq->op_params[15] != 0 ||
            rq->src[2] != nullptr || rk->src[2] != nullptr || rq->src[1] != rk->src[1] || rq->src[1]->type != GGML_TYPE_I32 ||
            n_dims <= 0 || n_dims > 256 || n_dims % 2 != 0) {
        PXA_P60_FAIL(rq);
    }
    // flash attention over f16 cache views: GQA 8, head 256, one token
    const ggml_tensor * Q = fa->src[0], * K = fa->src[1], * V = fa->src[2], * mask = fa->src[3];
    float fscale, max_bias, softcap;
    memcpy(&fscale,   (const float *) fa->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) fa->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) fa->op_params + 2, sizeof(float));
    char nk[64], nv[64];
    snprintf(nk, sizeof(nk), "cache_k_l%d", il);
    snprintf(nv, sizeof(nv), "cache_v_l%d", il);
    const int64_t T = Q->ne[1];
    if (mask == nullptr || fa->src[4] != nullptr || max_bias != 0.0f || softcap != 0.0f || pxa_p60_root(Q) != rq ||
            T < 1 || T > 6 || mask->ne[1] < T || ggml_nelements(rq->src[1]) != 4 * T || ggml_nelements(qcont) != 4096 * T ||
            ggml_nelements(gcont) != 4096 * T ||
            Q->ne[0] != 256 || Q->ne[2] != 16 || Q->type != GGML_TYPE_F32 ||
            K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 ||
            K->ne[0] != 256 || V->ne[0] != 256 || K->ne[2] != 2 || V->ne[2] != 2 || K->ne[1] != V->ne[1] || K->ne[3] != 1 || V->ne[3] != 1 ||
            mask->ne[0] < K->ne[1] || mask->nb[0] != sizeof(half) || K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half) ||
            K->nb[1] % 16 != 0 || K->nb[2] % 16 != 0 || V->nb[1] % 16 != 0 || V->nb[2] % 16 != 0 ||
            strcmp(pxa_p60_root(K)->name, nk) != 0 || strcmp(pxa_p60_root(V)->name, nv) != 0 ||
            (gmul->src[0] != fa && pxa_p60_root(gmul->src[0]) != fa && gmul->src[1] != fa && pxa_p60_root(gmul->src[1]) != fa)) {
        PXA_P60_FAIL(fa);
    }
    // this token's cache rows: f16, contiguous, 2 heads x 256, inside the views flash attention reads
    if (cpk->type != GGML_TYPE_F16 || cpv->type != GGML_TYPE_F16 || ggml_nelements(cpk) != 512 * T || ggml_nelements(cpv) != 512 * T ||
            !ggml_is_contiguous(cpk) || !ggml_is_contiguous(cpv) || strcmp(pxa_p60_root(cpk)->name, nk) != 0 ||
            strcmp(pxa_p60_root(cpv)->name, nv) != 0 || ggml_nelements(cpk->src[0]) != 512 * T || ggml_nelements(cpv->src[0]) != 512 * T) {
        PXA_P60_FAIL(cpk);
    }
    if (!pxa_p60_mat(wq, pxq ? GGML_TYPE_PXQN4 : GGML_TYPE_Q8_0, 2048, 8192) || !pxa_p60_mat(wk, GGML_TYPE_Q8_0, 2048, 512) ||
            !pxa_p60_mat(wv, GGML_TYPE_Q8_0, 2048, 512) || !pxa_p60_mat(wo, pxq ? GGML_TYPE_PXQN4 : GGML_TYPE_Q8_0, 4096, 2048) ||
            pxq != (rht_in != nullptr) || (pxq && T != 1 && !pxa_p60_attn_t_on()) || !pxa_p60_f32_vec(qn->src[1], 256) || !pxa_p60_f32_vec(kn->src[1], 256) ||
            !pxa_p60_f32_vec(norm->src[1], 2048) || !pxa_p60_f32_vec(x_in, 2048 * T) || !pxa_p60_f32_vec(add, 2048 * T) ||
            !pxa_p60_private(g, i, e, add)) {
        PXA_P60_FAIL(add);
    }
    P.kind = 3; P.n = e - i; P.il = il;
    P.iv[0] = (int) T;
    P.t[13] = rht_in; P.t2 = rht_out;
    const ggml_tensor * TT[] = { x_in, add, norm->src[1], wq, wk, wv, wo, qn->src[1], kn->src[1], rq, cpk, cpv, fa };
    for (int k = 0; k < 13; k++) P.t[k] = TT[k];
    memcpy(&P.eps[0], norm->op_params, sizeof(float));
    memcpy(&P.eps[1], qn->op_params, sizeof(float));
    memcpy(&P.eps[2], kn->op_params, sizeof(float));
    return e - i;
}

static bool pxa_p60_launch_attn(ggml_backend_cuda_context & ctx, const pxa_p60_plan & P) {
    const ggml_tensor * rq = P.t[9], * cpk = P.t[10], * cpv = P.t[11], * fa = P.t[12];
    const ggml_tensor * K = fa->src[1], * V = fa->src[2], * mask = fa->src[3];
    pxa_p60_attn_args a = {};
    a.x_in = (const float *) P.t[0]->data; a.x_out = (float *) P.t[1]->data;
    a.attn_norm = (const float *) P.t[2]->data; a.eps_norm = P.eps[0];
    a.wq = (const char *) P.t[3]->data; a.wk = (const char *) P.t[4]->data; a.wv = (const char *) P.t[5]->data; a.wo = (const char *) P.t[6]->data;
    a.q_norm = (const float *) P.t[7]->data; a.k_norm = (const float *) P.t[8]->data; a.eps_q = P.eps[1]; a.eps_k = P.eps[2];
    a.pos = (const int32_t *) rq->src[1]->data;
    a.n_dims = rq->op_params[1];
    a.imrope = rq->op_params[2] == GGML_ROPE_TYPE_IMROPE;
    memcpy(a.sections, (const int32_t *) rq->op_params + 11, 4 * sizeof(int));
    memcpy(&a.freq_base,   (const int32_t *) rq->op_params + 5, sizeof(float));
    memcpy(&a.freq_scale,  (const int32_t *) rq->op_params + 6, sizeof(float));
    memcpy(&a.attn_factor, (const int32_t *) rq->op_params + 8, sizeof(float));
    a.k_dst = cpk->data; a.v_dst = cpv->data; a.kv_t = 512;
    a.K = K->data; a.k_p = K->nb[1] / sizeof(half); a.k_h = K->nb[2] / sizeof(half);
    a.V = V->data; a.v_p = V->nb[1] / sizeof(half); a.v_h = V->nb[2] / sizeof(half);
    a.n_kv = (int) K->ne[1];
    a.mask = mask->data; a.mask_row = mask->nb[1] / sizeof(half);
    memcpy(&a.scale, (const float *) fa->op_params + 0, sizeof(float));
    const int T = P.iv[0];
    static const bool multi = pxa_p60_env("PXA_P60_MEGA_T", true);
    a.proj_type = P.t[3]->type;
    if (((uintptr_t) a.K) % 16 != 0 || ((uintptr_t) a.V) % 16 != 0) {
        return false;
    }
    if (a.proj_type == GGML_TYPE_PXQN4) {
        const ggml_tensor * ri = P.t[13], * ro = P.t2;
        a.rht_seed = (uint64_t) (uint32_t) ri->op_params[0] | ((uint64_t) (uint32_t) ri->op_params[1] << 32);
        const uint64_t seed_out = (uint64_t) (uint32_t) ro->op_params[0] | ((uint64_t) (uint32_t) ro->op_params[1] << 32);
        if (seed_out != a.rht_seed || (T != 1 && !pxa_p60_attn_t_on())) {
            return false;
        }
        a.rht_layer_in = ri->op_params[2]; a.rht_site_in = ri->op_params[3]; a.rht_k0_in = ri->op_params[4];
        a.rht_layer_out = ro->op_params[2]; a.rht_site_out = ro->op_params[3]; a.rht_k0_out = ro->op_params[4];
        if (T == 1 ? !ggml_cuda_pxqn_mega_attn(ctx, a) : !ggml_cuda_pxqn_mega_attn_t(ctx, a, T)) {
            return false;
        }
    } else if ((T > 1 && !multi) || !pxa_p60_mega_attn(ctx, a, T)) {
        return false;
    }
    static bool logged[7] = {};
    if (!logged[T]) {
        logged[T] = true;
        fprintf(stderr, "PXA_P60_MEGA: attention half FIRED (dev %d, layer %d, %d token(s), n_dims %d, sections %d/%d/%d/%d, %s, n_kv %d)\n",
                ctx.device, P.il, T, a.n_dims, a.sections[0], a.sections[1], a.sections[2], a.sections[3], a.imrope ? "imrope" : "mrope", a.n_kv);
    }
    return true;
}

// all persistent halves that may start at node i (a FUSED_RMS_NORM). The structural match (names, ops, shapes, the
// privacy scan) runs once per (graph, node); later tokens re-check a fingerprint of the covered nodes and the 64 after
// them (pointer, op, src0, src1) and reuse the plan, so the per-token host cost is ~1 us per half.
// The fingerprint carries every node's shape, not only its identity: llama's all-experts WARMUP decode builds the same
// topology into the same graph-meta buffer as a real decode (same node pointers, ops and sources) and differs only in
// shapes (ffn_moe_topk is [n_expert, T] instead of [n_expert_used, T]). Without ne[] the MoE half's "no match" verdict
// on that warmup graph was replayed on every later decode of the same width, which is what kept an MTP-armed context
// (its target warms up at one token) on the generic MoE route for good.
// It also carries src[2..] and the SSM_CONV / DELTA_NET op_params the GDN matcher decides on (10-21):
// a verify graph that arms the per-step rollback captures (SSM_CONV src[4], DELTA_NET src[6]) is otherwise the same graph
// as the server's capture-less 2-token WARMUP decode (captures are sources, not nodes: same pointers, ops, src0/src1,
// shapes, node count). The T = 2 GDN plan matched on the warmup (no captures) was replayed on every T = 2 verify, so the
// GDN T-path wrote no per-step states and a rejected draft was rolled back from stale capture rows (broken MTP text).
struct pxa_p60_fp { const ggml_tensor * t; int op; const void * s0; const void * s1; int64_t ne[4];
                    const void * sx[GGML_MAX_SRC - 2]; int32_t opp[5]; };

static void pxa_p60_fp_extra(const ggml_tensor * t, const void ** sx, int32_t * opp) {
    for (int s = 2; s < GGML_MAX_SRC; s++) sx[s - 2] = t->src[s];
    const bool gdn_op = t->op == GGML_OP_SSM_CONV || t->op == GGML_OP_DELTA_NET;
    for (int k = 0; k < 5; k++) opp[k] = gdn_op ? t->op_params[k] : 0;
}
struct pxa_p60_cache_ent { pxa_p60_plan plan; int n_nodes = 0; std::vector<pxa_p60_fp> fp; };

static void pxa_p60_fingerprint(const ggml_cgraph * g, int i, int n, std::vector<pxa_p60_fp> & fp) {
    const int end = std::min(g->n_nodes, i + n + 1 + 64);
    fp.clear();
    for (int k = i; k < end; k++) {
        const ggml_tensor * t = g->nodes[k];
        pxa_p60_fp f = { t, (int) t->op, t->src[0], t->src[1], { t->ne[0], t->ne[1], t->ne[2], t->ne[3] }, {}, {} };
        pxa_p60_fp_extra(t, f.sx, f.opp);
        fp.push_back(f);
    }
}

static bool pxa_p60_fp_same(const ggml_cgraph * g, int i, const std::vector<pxa_p60_fp> & fp) {
    if (i + (int) fp.size() > g->n_nodes) {
        return false;
    }
    for (size_t k = 0; k < fp.size(); k++) {
        const ggml_tensor * t = g->nodes[i + k];
        if (t != fp[k].t || (int) t->op != fp[k].op || t->src[0] != fp[k].s0 || t->src[1] != fp[k].s1 ||
                t->ne[0] != fp[k].ne[0] || t->ne[1] != fp[k].ne[1] || t->ne[2] != fp[k].ne[2] || t->ne[3] != fp[k].ne[3]) {
            return false;
        }
        const void * sx[GGML_MAX_SRC - 2]; int32_t opp[5];
        pxa_p60_fp_extra(t, sx, opp);
        if (memcmp(sx, fp[k].sx, sizeof(sx)) != 0 || memcmp(opp, fp[k].opp, sizeof(opp)) != 0) {
            static int said = 0;
            if (said < 4) {
                said++;
                fprintf(stderr, "PXA_P60_MEGA: plan re-matched at node %d (%s, %s): src[2..] / op_params changed since the cached match\n",
                        i + (int) k, t->name, ggml_op_name(t->op));
            }
            return false;
        }
    }
    return true;
}

// PXA_P60_MEGA_ATTN_T=2: at FLASH_ATTN_EXT of a 2..6-token PXQN4 attention layer, the nodes up to the residual ADD:
//   FLASH_ATTN_EXT(PERMUTE(ROPE q), cache_k view, cache_v view, mask) .. RESHAPE .. CONT(VIEW gate of the q projection) ..
//   SIGMOID .. MUL .. PXQN_RHT .. MUL_MAT attn_output .. ADD(attn_out-<il>, x)
// run as ggml_cuda_pxqn_mega_attn_tail. The gate CONT / SIGMOID may also sit before the FLASH_ATTN_EXT (then they ran already).
static int pxa_p60_match_attn_tail(const ggml_cgraph * g, int i, pxa_p60_plan & P) {
    const ggml_tensor * fa = g->nodes[i];
    if (fa->op != GGML_OP_FLASH_ATTN_EXT || pxa_p60_attn_t_mode() != 2) {
        return 0;
    }
    const ggml_tensor * Q = fa->src[0], * K = fa->src[1], * V = fa->src[2], * mask = fa->src[3];
    if (Q == nullptr || K == nullptr || V == nullptr || mask == nullptr || fa->src[4] != nullptr) {
        return 0;
    }
    int il = -1;
    if (sscanf(pxa_p60_root(K)->name, "cache_k_l%d", &il) != 1) {
        return 0;
    }
    char nk[64], nv[64];
    snprintf(nk, sizeof(nk), "cache_k_l%d", il);
    snprintf(nv, sizeof(nv), "cache_v_l%d", il);
    const ggml_tensor * rq = pxa_p60_root(Q);
    const int64_t T = Q->ne[1];
    float fscale, max_bias, softcap;
    memcpy(&fscale,   (const float *) fa->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) fa->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) fa->op_params + 2, sizeof(float));
    if (T < 2 || T > 6 || rq->op != GGML_OP_ROPE || rq->type != GGML_TYPE_F32 || !ggml_is_contiguous(rq) ||
            rq->ne[0] != 256 || rq->ne[1] != 16 || rq->ne[2] != T || rq->ne[3] != 1 ||
            rq->src[1] == nullptr || rq->src[1]->type != GGML_TYPE_I32 || ggml_nelements(rq->src[1]) != 4 * T ||
            Q->type != GGML_TYPE_F32 || Q->ne[0] != 256 || Q->ne[2] != 16 || Q->ne[3] != 1 ||
            Q->nb[0] != sizeof(float) || Q->nb[1] != 16 * 256 * sizeof(float) || Q->nb[2] != 256 * sizeof(float) ||
            max_bias != 0.0f || softcap != 0.0f || mask->ne[1] < T ||
            K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 ||
            K->ne[0] != 256 || V->ne[0] != 256 || K->ne[2] != 2 || V->ne[2] != 2 || K->ne[1] != V->ne[1] || K->ne[3] != 1 || V->ne[3] != 1 ||
            mask->ne[0] < K->ne[1] || mask->nb[0] != sizeof(half) || K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half) ||
            K->nb[1] % 16 != 0 || K->nb[2] % 16 != 0 || V->nb[1] % 16 != 0 || V->nb[2] % 16 != 0 ||
            strcmp(pxa_p60_root(V)->name, nv) != 0 || strcmp(pxa_p60_root(K)->name, nk) != 0 ||
            fa->type != GGML_TYPE_F32 || !ggml_is_contiguous(fa)) {
        return 0;
    }
    const ggml_tensor * gmul = nullptr, * rht_out = nullptr, * omm = nullptr, * add = nullptr, * x_in = nullptr;
    int e = -1;
    for (int k = i + 1; k < g->n_nodes && k < i + 24 && e < 0; k++) {
        const ggml_tensor * t = g->nodes[k];
        switch (t->op) {
            case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_TRANSPOSE: case GGML_OP_PERMUTE: case GGML_OP_NONE:
                break;
            case GGML_OP_CONT:   // the gate CONT (checked through the MUL below)
                break;
            case GGML_OP_UNARY:
                if (ggml_get_unary_op(t) != GGML_UNARY_OP_SIGMOID) return 0;
                break;
            case GGML_OP_MUL:
                if (gmul != nullptr || !(pxa_p60_root(t->src[0]) == fa || pxa_p60_root(t->src[1]) == fa)) return 0;
                gmul = t;
                break;
            case GGML_OP_PXQN_RHT:
                if (gmul == nullptr || t->src[0] != gmul || rht_out != nullptr) return 0;
                rht_out = t;
                break;
            case GGML_OP_MUL_MAT:
                if (rht_out == nullptr || t->src[1] != rht_out || !pxa_p60_is_w(t->src[0], il, "attn_output.weight")) return 0;
                omm = t;
                break;
            case GGML_OP_ADD:
                if (omm == nullptr) return 0;
                if      (t->src[0] == omm) x_in = t->src[1];
                else if (t->src[1] == omm) x_in = t->src[0];
                else return 0;
                add = t; e = k;
                break;
            default:
                return 0;
        }
    }
    if (e < 0 || !gmul || !rht_out || !omm || !add || !x_in) {
        return 0;
    }
    // the gate: SIGMOID(CONT(VIEW(q projection, +256 floats, rows of 512))) on the other side of the MUL
    const ggml_tensor * sig = pxa_p60_root(gmul->src[0]) == fa ? gmul->src[1] : gmul->src[0];
    if (sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID) return 0;
    const ggml_tensor * gcont = sig->src[0];
    if (gcont->op != GGML_OP_CONT || gcont->src[0]->op != GGML_OP_VIEW || ggml_nelements(gcont) != 4096 * T) return 0;
    const ggml_tensor * gview = gcont->src[0], * qmm = gview->view_src;
    if (qmm == nullptr || qmm->op != GGML_OP_MUL_MAT || !pxa_p60_is_w(qmm->src[0], il, "attn_q.weight") ||
            gview->view_offs != 256 * sizeof(float) || gview->ne[0] != 256 || gview->nb[1] != 2 * 256 * sizeof(float) ||
            qmm->type != GGML_TYPE_F32 || !ggml_is_contiguous(qmm) || qmm->ne[0] != 8192 || qmm->ne[1] != T) {
        return 0;
    }
    // qmm must still hold its rows when the tail runs: one of its readers is inside the range (the gate CONT), or none
    // of the nodes before i that was allocated after its last reader could have reused it -- require the CONT in range
    bool gate_in = false;
    for (int k = i + 1; k <= e; k++) gate_in |= g->nodes[k] == gcont;
    const ggml_tensor * wo = omm->src[0];
    if (!gate_in || !pxa_p60_mat(wo, GGML_TYPE_PXQN4, 4096, 2048) || qmm->src[0]->type != GGML_TYPE_PXQN4 ||
            !pxa_p60_f32_vec(x_in, 2048 * T) || !pxa_p60_f32_vec(add, 2048 * T) || !pxa_p60_private(g, i, e, add)) {
        return 0;
    }
    P.kind = 4; P.n = e - i; P.il = il;
    P.iv[0] = (int) T;
    const ggml_tensor * TT[] = { x_in, add, wo, rq, fa, qmm, rht_out };
    for (int k = 0; k < 7; k++) P.t[k] = TT[k];
    return e - i;
}

static bool pxa_p60_launch_attn_tail(ggml_backend_cuda_context & ctx, const pxa_p60_plan & P) {
    const ggml_tensor * rq = P.t[3], * fa = P.t[4], * qmm = P.t[5], * ro = P.t[6];
    const ggml_tensor * K = fa->src[1], * V = fa->src[2], * mask = fa->src[3];
    pxa_p60_attn_args a = {};
    a.x_in = (const float *) P.t[0]->data; a.x_out = (float *) P.t[1]->data;
    a.wo = (const char *) P.t[2]->data;
    a.pos = (const int32_t *) rq->src[1]->data;
    a.K = K->data; a.k_p = K->nb[1] / sizeof(half); a.k_h = K->nb[2] / sizeof(half);
    a.V = V->data; a.v_p = V->nb[1] / sizeof(half); a.v_h = V->nb[2] / sizeof(half);
    a.n_kv = (int) K->ne[1];
    a.mask = mask->data; a.mask_row = mask->nb[1] / sizeof(half);
    memcpy(&a.scale, (const float *) fa->op_params + 0, sizeof(float));
    a.proj_type = GGML_TYPE_PXQN4;
    a.rht_seed = (uint64_t) (uint32_t) ro->op_params[0] | ((uint64_t) (uint32_t) ro->op_params[1] << 32);
    a.rht_layer_out = ro->op_params[2]; a.rht_site_out = ro->op_params[3]; a.rht_k0_out = ro->op_params[4];
    if (((uintptr_t) a.K) % 16 != 0 || ((uintptr_t) a.V) % 16 != 0) {
        return false;
    }
    const int T = P.iv[0];
    if (!ggml_cuda_pxqn_mega_attn_tail(ctx, a, T, (const float *) rq->data, (const float *) qmm->data)) {
        return false;
    }
    static bool logged[7] = {};
    if (!logged[T]) {
        logged[T] = true;
        fprintf(stderr, "PXA_P60_MEGA: attention tail FIRED (dev %d, layer %d, %d tokens, n_kv %d; PXA_P60_MEGA_ATTN_T=2)\n",
                ctx.device, P.il, T, a.n_kv);
    }
    return true;
}

static int pxa_p60_try_mega(ggml_backend_cuda_context & ctx, const ggml_cgraph * g, int i, bool dry, const ggml_tensor * add = nullptr) {
    static std::mutex mtx;
    static std::unordered_map<uint64_t, pxa_p60_cache_ent> cache;
    // the row count is part of the bucket: an MTP head alternates a 1-row draft graph and an N-row update graph built
    // into the same meta buffer (same node pointers), and with shapes in the fingerprint one shared bucket would re-match
    // on every alternation. The fingerprint still validates every reuse, so the key only picks the bucket.
    const uint64_t key = (uint64_t) (uintptr_t) g->nodes[i] ^ ((uint64_t) i << 44) ^ ((uint64_t) ctx.device << 60) ^
                         ((uint64_t) g->nodes[i]->ne[1] * 0x9E3779B97F4A7C15ull);
    pxa_p60_plan P;
    {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = cache.find(key);
        if (it != cache.end() && it->second.n_nodes == g->n_nodes && pxa_p60_fp_same(g, i, it->second.fp)) {
            P = it->second.plan;
        } else {
            pxa_p60_plan Q;
            if (pxa_p60_match_moe(g, i, Q) == 0) {
                Q = pxa_p60_plan();
                if (pxa_p60_match_gdn(g, i, Q) == 0) {
                    Q = pxa_p60_plan();
                    if (pxa_p60_match_attn(g, i, Q) == 0) {
                        Q = pxa_p60_plan();
                        if (pxa_p60_match_attn_tail(g, i, Q) == 0) {   // PXA_P60_MEGA_ATTN_T=2
                            Q = pxa_p60_plan();
                        }
                    }
                }
            }
            if (cache.size() > 4096) {
                cache.clear();
            }
            pxa_p60_cache_ent & ent = cache[key];
            ent.plan = Q;
            ent.n_nodes = g->n_nodes;
            pxa_p60_fingerprint(g, i, Q.n, ent.fp);
            P = Q;
        }
    }
    if (P.kind == 0 || (add != nullptr && P.kind != 1)) {
        return 0;
    }
    if (dry) {
        return P.n;
    }
    const bool ok = P.kind == 1 ? pxa_p60_launch_moe(ctx, P, add) : P.kind == 2 ? pxa_p60_launch_gdn(ctx, P) :
                    P.kind == 4 ? pxa_p60_launch_attn_tail(ctx, P) : pxa_p60_launch_attn(ctx, P);
    return ok ? P.n : 0;
}

// ---- PXA_P60_PRE_VW (default OFF) -------------------------------------------------------------------------------------
// At FUSED_RMS_NORM(x, blk.N.attn_norm) of a 2..5-token batch with a PXQN_RHT on its output: every MUL_MAT that reads the
// RHT output (GDN: attn_qkv / attn_gate PXQN4 + ssm_beta / ssm_alpha Q8_0; attention: attn_q PXQN4 + attn_k / attn_v
// Q8_0) runs with the norm and the RHT as one k_pxqn_pre launch (closed library). The RHT and those MUL_MAT nodes are
// recorded per device and skipped when the graph reaches them (they may sit anywhere after the norm). Declines (normal
// path) unless the norm output is read only by the RHT and the RHT output only by those MUL_MATs inside the layer window.
static bool pxa_p60_pre_on() {
    static const bool v = [] {
        const bool on = pxa_p60_env("PXA_P60_PRE_VW", false);
        if (on) fprintf(stderr, "PXA_P60_PRE_VW: 1 (verify-width norm + RHT + input projections in one launch on cc 6.0)\n");
        return on;
    }();
    return v;
}

struct pxa_p60_done_t { const ggml_tensor * t[8]; int n = 0; };
static pxa_p60_done_t pxa_p60_done_set[GGML_CUDA_MAX_DEVICES];

static bool pxa_p60_done_take(int dev, const ggml_tensor * t) {
    pxa_p60_done_t & d = pxa_p60_done_set[dev];
    for (int k = 0; k < d.n; k++) {
        if (d.t[k] == t) {
            d.t[k] = d.t[--d.n];
            return true;
        }
    }
    return false;
}

struct pxa_p60_pre_plan { int ok = 0; int n_nodes = 0; const ggml_tensor * rht = nullptr; const ggml_tensor * mm[4] = {}; int nmm = 0; };

static int pxa_p60_try_pre(ggml_backend_cuda_context & ctx, const ggml_cgraph * g, int i, bool dry) {
    const ggml_tensor * norm = g->nodes[i];
    if (!pxa_p60_pre_on() || norm->op != GGML_OP_FUSED_RMS_NORM || norm->src[1] == nullptr ||
            ggml_cuda_info().devices[ctx.device].cc != 600) {
        return 0;
    }
    int il = -1;
    if (sscanf(norm->src[1]->name, "blk.%d.attn_norm.weight", &il) != 1 || !pxa_p60_is_w(norm->src[1], il, "attn_norm.weight")) {
        return 0;
    }
    const int64_t T = norm->ne[1];
    if (T < 2 || T > 5 || norm->ne[0] != 2048 || norm->ne[2] != 1 || norm->ne[3] != 1 || norm->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(norm) || norm->src[0]->type != GGML_TYPE_F32 || norm->src[0]->nb[0] != sizeof(float) ||
            !pxa_p60_f32_vec(norm->src[1], 2048)) {
        return 0;
    }
    static std::mutex mtx;
    static std::unordered_map<uint64_t, pxa_p60_pre_plan> cache;
    const uint64_t key = (uint64_t) (uintptr_t) norm ^ ((uint64_t) i << 44) ^ ((uint64_t) ctx.device << 60);
    pxa_p60_pre_plan P;
    {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = cache.find(key);
        bool hit = it != cache.end() && it->second.n_nodes == g->n_nodes;
        if (hit && it->second.ok) {   // the covered nodes must still be the same tensors on the same inputs
            const pxa_p60_pre_plan & C = it->second;
            int found = 0;
            for (int k = i + 1; k < g->n_nodes && k < i + 64 && found < C.nmm + 1; k++) {
                const ggml_tensor * t = g->nodes[k];
                if (t == C.rht && t->src[0] == norm) found++;
                for (int m = 0; m < C.nmm; m++) if (t == C.mm[m] && t->src[1] == C.rht) found++;
            }
            hit = found == C.nmm + 1;
        }
        if (hit) {
            P = it->second;
        } else {
            P = pxa_p60_pre_plan();
            P.n_nodes = g->n_nodes;
            const int lim = std::min(g->n_nodes, i + 64);
            bool bad = false;
            for (int k = i + 1; k < lim && !bad; k++) {
                const ggml_tensor * t = g->nodes[k];
                if (t->op == GGML_OP_PXQN_RHT && t->src[0] == norm) { if (P.rht) bad = true; P.rht = t; continue; }
                if (P.rht && t->op == GGML_OP_MUL_MAT && t->src[1] == P.rht) { if (P.nmm == 4) bad = true; else P.mm[P.nmm++] = t; }
            }
            if (!bad && P.rht && P.nmm > 0 && !(norm->flags & GGML_TENSOR_FLAG_OUTPUT) && !(P.rht->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                // privacy over the layer window: norm read only by the RHT, the RHT output only by the matched MUL_MATs
                const int wl = std::min(g->n_nodes, i + 400);
                for (int k = i + 1; k < wl && !bad; k++) {
                    const ggml_tensor * t = g->nodes[k];
                    for (int j = 0; j < GGML_MAX_SRC && !bad; j++) {
                        const ggml_tensor * s = t->src[j];
                        if (s == nullptr) continue;
                        const ggml_tensor * r = pxa_p60_root(s);
                        if ((s == norm || r == norm) && t != P.rht) bad = true;
                        if (s == P.rht || r == P.rht) {
                            bool ours = false;
                            for (int m = 0; m < P.nmm; m++) ours |= t == P.mm[m];
                            if (!ours) bad = true;
                        }
                    }
                }
                // types / shapes: PXQN4 or Q8_0 weights [2048 x R] named blk.<il>.*, f32 contiguous outputs [R x T]
                int np = 0, nq = 0;
                for (int m = 0; m < P.nmm && !bad; m++) {
                    const ggml_tensor * t = P.mm[m], * w = t->src[0];
                    if (w->ne[0] != 2048 || w->ne[2] != 1 || w->ne[3] != 1 || !ggml_is_contiguous(w) || t->type != GGML_TYPE_F32 ||
                            t->ne[1] != T || t->ne[0] != w->ne[1] || t->nb[0] != sizeof(float) || t->ne[2] != 1 || t->ne[3] != 1) bad = true;
                    else if (w->type == GGML_TYPE_PXQN4 && w->ne[1] % 64 == 0) np++;
                    else if (w->type == GGML_TYPE_Q8_0 && w->nb[1] == 2176) nq++;
                    else bad = true;
                    char pre[32];
                    snprintf(pre, sizeof(pre), "blk.%d.", il);
                    if (strncmp(w->name, pre, strlen(pre)) != 0) bad = true;
                }
                if (np > 2 || nq > 2) bad = true;
                if (P.rht->ne[0] != 2048 || P.rht->ne[1] != T || !ggml_is_contiguous(P.rht)) bad = true;
                P.ok = bad ? 0 : 1;
            }
            if (cache.size() > 4096) cache.clear();
            cache[key] = P;
        }
    }
    if (!P.ok) {
        return 0;
    }
    if (dry) {
        return 1;
    }
    pxa_p60_pre_args a = {};
    a.x = (const float *) norm->src[0]->data; a.ldx = norm->src[0]->nb[1] / sizeof(float);
    a.norm = (const float *) norm->src[1]->data;
    memcpy(&a.eps, norm->op_params, sizeof(float));
    const ggml_tensor * ri = P.rht;
    a.rht_seed = (uint64_t) (uint32_t) ri->op_params[0] | ((uint64_t) (uint32_t) ri->op_params[1] << 32);
    a.rht_layer = ri->op_params[2]; a.rht_site = ri->op_params[3]; a.rht_k0 = ri->op_params[4];
    for (int m = 0; m < P.nmm; m++) {
        const ggml_tensor * t = P.mm[m], * w = t->src[0];
        if (w->type == GGML_TYPE_PXQN4) {
            a.pw[a.np] = (const char *) w->data; a.pr[a.np] = (int) w->ne[1]; a.py[a.np] = (float *) t->data; a.pldy[a.np] = t->nb[1] / sizeof(float); a.np++;
        } else {
            a.qw[a.nq] = (const char *) w->data; a.qr[a.nq] = (int) w->ne[1]; a.qy[a.nq] = (float *) t->data; a.qldy[a.nq] = t->nb[1] / sizeof(float);
            a.qnb1[a.nq] = w->nb[1]; a.nq++;
        }
    }
    if (!ggml_cuda_pxqn_pre_vw(ctx, a, (int) T)) {
        return 0;
    }
    pxa_p60_done_t & d = pxa_p60_done_set[ctx.device];
    d.n = 0;
    d.t[d.n++] = P.rht;
    for (int m = 0; m < P.nmm; m++) d.t[d.n++] = P.mm[m];
    static bool logged[6] = {};
    if (!logged[T]) {
        logged[T] = true;
        fprintf(stderr, "PXA_P60_PRE_VW: FIRED (dev %d, layer %d, %d token(s), %d PXQN4 + %d Q8_0 projections)\n", ctx.device, il, (int) T, a.np, a.nq);
    }
    return 1;
}

// ---- PXA_P60_POST_VW (default OFF) ------------------------------------------------------------------------------------
// At FUSED_RMS_NORM(o, blk.N.ssm_norm) of a 2..5-token GDN layer: FUSED_MUL_UNARY(silu z) -> PXQN_RHT (out_in) -> MUL_MAT
// ssm_out (PXQN4) -> ADD residual run as one k_pxqn_post launch; those four nodes are recorded and skipped.
static bool pxa_p60_post_on() {
    static const bool v = [] {
        const bool on = pxa_p60_env("PXA_P60_POST_VW", false);
        if (on) fprintf(stderr, "PXA_P60_POST_VW: 1 (verify-width gated norm + RHT + ssm_out + residual in one launch on cc 6.0)\n");
        return on;
    }();
    return v;
}

struct pxa_p60_post_plan { int ok = 0; int n_nodes = 0; const ggml_tensor * gate = nullptr, * rht = nullptr, * mm = nullptr, * add = nullptr, * res = nullptr; };

static int pxa_p60_try_post(ggml_backend_cuda_context & ctx, const ggml_cgraph * g, int i) {
    const ggml_tensor * norm = g->nodes[i];
    if (!pxa_p60_post_on() || norm->op != GGML_OP_FUSED_RMS_NORM || norm->src[1] == nullptr ||
            ggml_cuda_info().devices[ctx.device].cc != 600) {
        return 0;
    }
    int il = -1;
    if (sscanf(norm->src[1]->name, "blk.%d.ssm_norm.weight", &il) != 1 || !pxa_p60_is_w(norm->src[1], il, "ssm_norm.weight")) {
        return 0;
    }
    if (norm->ne[0] != 128 || norm->ne[1] % 32 != 0 || norm->ne[2] != 1 || norm->ne[3] != 1 || norm->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(norm) || !ggml_is_contiguous(norm->src[0]) || !pxa_p60_f32_vec(norm->src[1], 128)) {
        return 0;
    }
    const int64_t T = norm->ne[1] / 32;
    if (T < 2 || T > 5) {
        return 0;
    }
    static std::mutex mtx;
    static std::unordered_map<uint64_t, pxa_p60_post_plan> cache;
    const uint64_t key = (uint64_t) (uintptr_t) norm ^ ((uint64_t) i << 44) ^ ((uint64_t) ctx.device << 60);
    pxa_p60_post_plan P;
    {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = cache.find(key);
        bool hit = it != cache.end() && it->second.n_nodes == g->n_nodes;
        if (hit && it->second.ok) {
            const pxa_p60_post_plan & C = it->second;
            int found = 0;
            for (int k = i + 1; k < g->n_nodes && k < i + 48; k++) {
                const ggml_tensor * t = g->nodes[k];
                found += (t == C.gate && t->src[1] == norm) + (t == C.rht) + (t == C.mm && t->src[1] == C.rht) +
                         (t == C.add && (t->src[0] == C.mm || t->src[1] == C.mm));
            }
            hit = found == 4;
        }
        if (hit) {
            P = it->second;
        } else {
            P = pxa_p60_post_plan();
            P.n_nodes = g->n_nodes;
            const int lim = std::min(g->n_nodes, i + 48);
            bool bad = false;
            for (int k = i + 1; k < lim && !bad; k++) {
                const ggml_tensor * t = g->nodes[k];
                if (t->op == GGML_OP_FUSED_MUL_UNARY && t->src[1] == norm) { if (P.gate) bad = true; P.gate = t; continue; }
                if (P.gate && t->op == GGML_OP_PXQN_RHT && pxa_p60_root(t->src[0]) == P.gate) { if (P.rht) bad = true; P.rht = t; continue; }
                if (P.rht && t->op == GGML_OP_MUL_MAT && t->src[1] == P.rht) { if (P.mm) bad = true; P.mm = t; continue; }
                if (P.mm && t->op == GGML_OP_ADD && (t->src[0] == P.mm || t->src[1] == P.mm)) { if (P.add) bad = true; P.add = t; }
            }
            if (!bad && P.gate && P.rht && P.mm && P.add) {
                P.res = P.add->src[0] == P.mm ? P.add->src[1] : P.add->src[0];
                const ggml_tensor * w = P.mm->src[0], * z = P.gate->src[0];
                if (P.gate->op_params[0] != GGML_UNARY_OP_SILU || !pxa_p60_mat(w, GGML_TYPE_PXQN4, 4096, 2048) ||
                        !pxa_p60_is_w(w, il, "ssm_out.weight") || z->type != GGML_TYPE_F32 || !ggml_is_contiguous(z) ||
                        z->ne[0] != 128 || z->ne[1] != 32 * T || P.rht->ne[0] != 4096 || P.rht->ne[1] != T ||
                        P.mm->ne[0] != 2048 || P.mm->ne[1] != T || P.add->ne[0] != 2048 || P.add->ne[1] != T ||
                        P.add->type != GGML_TYPE_F32 || !ggml_is_contiguous(P.add) || P.res->type != GGML_TYPE_F32 ||
                        P.res->ne[0] != 2048 || P.res->ne[1] != T || P.res->nb[0] != sizeof(float) ||
                        (norm->flags | P.gate->flags | P.rht->flags | P.mm->flags) & GGML_TENSOR_FLAG_OUTPUT) {
                    bad = true;
                }
                // privacy over the layer window: norm -> gate only, gate -> rht / mm keep-alive / views, rht -> mm, mm -> add
                const int wl = std::min(g->n_nodes, i + 400);
                for (int k = i + 1; k < wl && !bad; k++) {
                    const ggml_tensor * t = g->nodes[k];
                    if (t == P.gate || t == P.rht || t == P.mm || t == P.add) continue;
                    const bool noop = t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE ||
                                      t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE;
                    for (int j = 0; j < GGML_MAX_SRC && !bad; j++) {
                        const ggml_tensor * s = t->src[j];
                        if (s == nullptr) continue;
                        const ggml_tensor * r = pxa_p60_root(s);
                        if (s == norm || r == norm || s == P.rht || r == P.rht || s == P.mm || r == P.mm) bad = true;
                        if ((s == P.gate || r == P.gate) && !noop) bad = true;
                    }
                }
                if (P.gate->src[1] != norm) bad = true;
            } else {
                bad = true;
            }
            P.ok = bad ? 0 : 1;
            if (cache.size() > 4096) cache.clear();
            cache[key] = P;
        }
    }
    if (!P.ok) {
        return 0;
    }
    pxa_p60_post_args a = {};
    a.o = (const float *) norm->src[0]->data; a.ldo = 4096;
    a.z = (const float *) P.gate->src[0]->data; a.ldz = 4096;
    a.snorm = (const float *) norm->src[1]->data;
    memcpy(&a.eps, norm->op_params, sizeof(float));
    const ggml_tensor * ri = P.rht;
    a.rht_seed = (uint64_t) (uint32_t) ri->op_params[0] | ((uint64_t) (uint32_t) ri->op_params[1] << 32);
    a.rht_layer = ri->op_params[2]; a.rht_site = ri->op_params[3]; a.rht_k0 = ri->op_params[4];
    a.w = (const char *) P.mm->src[0]->data;
    a.res = (const float *) P.res->data; a.ldr = P.res->nb[1] / sizeof(float);
    a.y = (float *) P.add->data; a.ldy = P.add->nb[1] / sizeof(float);
    if (!ggml_cuda_pxqn_post_vw(ctx, a, (int) T)) {
        return 0;
    }
    pxa_p60_done_t & d = pxa_p60_done_set[ctx.device];
    d.n = 0;
    d.t[d.n++] = P.gate; d.t[d.n++] = P.rht; d.t[d.n++] = P.mm; d.t[d.n++] = P.add;
    static bool logged[6] = {};
    if (!logged[T]) {
        logged[T] = true;
        fprintf(stderr, "PXA_P60_POST_VW: FIRED (dev %d, layer %d, %d token(s))\n", ctx.device, il, (int) T);
    }
    return 1;
}
