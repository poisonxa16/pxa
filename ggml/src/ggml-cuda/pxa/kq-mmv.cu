// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// kq-mmv.cu -- engine glue for the k-quant decode GEMV (kq-mmv.cuh): levers, shape checks, grouping.
//
// Levers
//   PXA_KQMMV        auto (default) | 0 | 1. auto = on at sm_60 (the h2 path) for Q4_K / Q5_K / Q6_K / Q8_0
//                    (REFERENCE level: off); 1 = on on every card (i8 path from sm_61: in the unit
//                    microbench on a V100 it is 0.5-0.9x the incumbent dp4a MMVQ, hence not auto there);
//                    0 = the incumbent q8_1 MMVQ (quantize + one launch per matrix), untouched.
//                    Taken where the block can stage the whole activation (ny <= 2 at K = 5120, ny = 1 at
//                    K = 8704) and a launch has >= KQMMV_MIN_ROWS rows, plus the wide form (PXA_KQMMV_WIDE).
//   PXA_KQMMV_PATH   auto | h2 | i8. auto = h2 (half2 dot) below sm_61, i8 (dp4a) from sm_61. Tests / A-B.
//   PXA_KQMMV_GROUP  1 (default) | 0. Consecutive MUL_MATs reading the same src1 share one launch
//                    (bit-identical to one launch per matrix).
//   PXA_KQMMV_BIAS   1 (default) | 0. A GEMV whose next node ADDs a per-row vector (the residual add on the
//                    last device of a tensor split) runs here with the add as epilogue (bit-identical to
//                    GEMV + ADD); 0 = those nodes stay on the incumbent mmvq_biased.
//   PXA_KQMMV_WIDE   1 (default) | 0. Q8_0 GEMVs whose activation does not fit the block (h2 ny 3..4
//                    at K = 5120: the LM head at spec-verify widths) in launches of >= 8192 rows run on
//                    the wide form k_kq_mmv_rw (x staged in K chunks, 16-32 rows per block, row pairs) instead
//                    of falling to the incumbent; column c stays bit-identical to the ny = 1 launch.
//                    0 = the previous rule (those widths decline to the incumbent q8_1 MMVQ).
//   PXA_KQMMV_GU     1 (default) | 0. FUSED_UP_GATE runs as one launch with the silu(gate)*up epilogue
//                    (bit-identical to two launches + the kernel's own GLU).
#include "../common.cuh"
#include "kq-mmv.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int kqmmv_kind(int type) {
    switch ((ggml_type) type) {
        case GGML_TYPE_Q4_K: return KQ_Q4_K;
        case GGML_TYPE_Q5_K: return KQ_Q5_K;
        case GGML_TYPE_Q6_K: return KQ_Q6_K;
        case GGML_TYPE_Q8_0: return KQ_Q8_0;
        default:             return -1;
    }
}

// -1 auto, 0 off, 1 on
static int kqmmv_mode() {
    static const int v = [](){
        const char * e = getenv("PXA_KQMMV");
        if (!e || !*e || strcmp(e, "auto") == 0) return -1;
        return atoi(e) != 0 ? 1 : 0;
    }();
    return v;
}

// -1 auto, KQMMV_H2, KQMMV_I8
static int kqmmv_path_env() {
    static const int v = [](){
        const char * e = getenv("PXA_KQMMV_PATH");
        if (!e || !*e || strcmp(e, "auto") == 0) return -1;
        if (strcmp(e, "h2") == 0) return (int) KQMMV_H2;
        if (strcmp(e, "i8") == 0) return (int) KQMMV_I8;
        fprintf(stderr, "PXA_KQMMV_PATH=%s not understood (auto|h2|i8): auto\n", e);
        return -1;
    }();
    return v;
}

static int kqmmv_path(int cc) {
    const int p = kqmmv_path_env();
    if (p >= 0) return p;
    return cc < 610 ? KQMMV_H2 : KQMMV_I8;
}

bool ggml_cuda_kqmmv_group_on(void) {
    static const bool v = [](){ const char * e = getenv("PXA_KQMMV_GROUP"); return !e || atoi(e) != 0; }();
    return v;
}

static bool kqmmv_gu_on() {
    static const bool v = [](){ const char * e = getenv("PXA_KQMMV_GU"); return !e || atoi(e) != 0; }();
    return v;
}

// PXA_KQMMV_BIAS 1 (default) | 0: a GEMV followed by a per-row ADD (the residual add that the last
// device of a tensor split puts after each partial wo / ssm_out / ffn_down) runs on this kernel with the
// add as its epilogue; 0 leaves those nodes on the incumbent mmvq_biased (the pre-lever behaviour).
static bool kqmmv_bias_on() {
    static const bool v = [](){ const char * e = getenv("PXA_KQMMV_BIAS"); return !e || atoi(e) != 0; }();
    return v;
}

static bool kqmmv_wide_on() {
    static const bool v = [](){ const char * e = getenv("PXA_KQMMV_WIDE"); return !e || atoi(e) != 0; }();
    return v;
}

// the wide form is taken where it beats the incumbent in the unit microbench (tests/test-kq-mmv -width),
// K <= 8192 (the measured domain: K = 5120):
// h2 (sm_60) at ny 3..4 -- the h2 dot is ALU-bound per column, so from ny = 5 the incumbent's dp4a-emulated
// q8_1 MMVQ is faster. The i8 path (sm_61+, opt-in PXA_KQMMV=1) has no measured wide-form win yet: not taken. Launches under KQMMV_WIDE_MIN_ROWS rows
// (the attention k / v and delta-net beta / alpha groups) stay on the incumbent: there x staging per K chunk
// dominates and the incumbent is 1.3-4x faster.
#define KQMMV_WIDE_MIN_ROWS 8192
static bool kqmmv_wide(int T, int P, int ny, int K) {
    return kqmmv_wide_on() && P == KQMMV_H2 && kqmmv_rw_fits(T, P, ny, K) && K <= 8192 && ny >= 3 && ny <= 4;
}

static bool kqmmv_dev_on(int device) {
    static int on[GGML_CUDA_MAX_DEVICES];
    static std::atomic<bool> inited[GGML_CUDA_MAX_DEVICES];
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return false;
    if (inited[device].load(std::memory_order_acquire)) return on[device] != 0;
    const int cc = ggml_cuda_info().devices[device].cc;
    const int mode = kqmmv_mode();
    bool v;
    const char * why;
    if (mode >= 0) {
        v = mode == 1 && cc >= 600;
        why = "explicit PXA_KQMMV";
    } else if (ggml_pxa_config_level() == 0) {
        v = false;
        why = "REFERENCE level";
    } else {
        v = cc >= 600 && cc < 610;
        why = "auto: sm_60";
    }
    on[device] = v ? 1 : 0;
    if (!inited[device].exchange(true, std::memory_order_acq_rel)) {
        fprintf(stderr, "PXA_KQMMV dev%d (cc %d): %s%s%s [%s] (Q4_K/Q5_K/Q6_K/Q8_0 decode GEMV ny<=8; group %s, gate/up %s)\n",
                device, cc, v ? "ON, path " : "OFF (incumbent q8_1 MMVQ)",
                v ? (kqmmv_path(cc) == KQMMV_H2 ? "h2" : "i8") : "", "", why,
                ggml_cuda_kqmmv_group_on() ? "on" : "off", kqmmv_gu_on() ? "fused" : "split");
    }
    return v;
}

// gu = the fused gate/up launch (no wide form: only the whole-x-staged shapes)
static bool kqmmv_take(int device, int type, int64_t K, int64_t ny, bool gu) {
    const int T = kqmmv_kind(type);
    if (T < 0 || ny < 1 || ny > KQMMV_MAX_NY || K <= 0 || K > (int64_t) 1 << 30) return false;
    if (!kqmmv_dev_on(device)) return false;
    const int P = kqmmv_path(ggml_cuda_info().devices[device].cc);
    if (kqmmv_xchunk(T, P, (int) ny, (int) K) >= K) return true;
    return !gu && kqmmv_wide(T, P, (int) ny, (int) K);
}

bool ggml_cuda_kqmmv_take(int device, int type, int64_t K, int64_t ny) {
    return kqmmv_take(device, type, K, ny, false);
}

bool ggml_cuda_kqmmv_shape_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    const int T = kqmmv_kind(src0->type);
    if (T < 0) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    const int64_t K = src0->ne[0], R = src0->ne[1], ny = src1->ne[1];
    if (src1->ne[0] != K || K % ggml_blck_size(src0->type) != 0 || K <= 0) return false;
    if (ny < 1 || ny > KQMMV_MAX_NY) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (dst->ne[0] != R || dst->ne[1] != ny || dst->ne[2] != 1 || dst->ne[3] != 1) return false;
    if (src0->nb[0] != ggml_type_size(src0->type) || src0->nb[1] < ggml_row_size(src0->type, K)) return false;
    // the activation is read as float4: 16-byte aligned columns
    if (src1->nb[0] != sizeof(float) || src1->nb[1] % 16 != 0 || ((uintptr_t) src1->data & 15) != 0) return false;
    if (dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0) return false;
    if (R <= 0 || R > (int64_t) 1 << 30 || K > (int64_t) 1 << 30) return false;
    if (!src0->data || !src1->data || !dst->data) return false;
    return true;
}

// PXA_DN_BA_KQMMV (default 1; 0 = off). The delta-net beta/alpha projections are two [n_embd x n_v_heads]
// matrices -- 24 rows each per card on a 2-card split of the 27B -- that read the same activation back to
// back. Under KQMMV_MIN_ROWS they stayed on the incumbent: a q8_1 quantize launch plus one MMVQ launch per
// matrix, i.e. four launches per delta-net layer. Taken here as ONE grouped launch (the pair is one
// "virtual" [n_embd x 2*n_v_heads] GEMV, same as a load-time concatenation would give, without a second
// copy of the weights or a new tensor layout), staged straight from the f32 activation (no quantize).
// The h2 dot rounds differently from the q8_1 dot, so the outputs change in the last bits.
static bool kqmmv_dn_ba(const ggml_tensor * const * src0, int n) {
    static const bool on = [](){ const char * e = getenv("PXA_DN_BA_KQMMV"); return !e || atoi(e) != 0; }();
    if (!on || n < 1) return false;
    for (int i = 0; i < n; ++i) {
        if (!strstr(src0[i]->name, "ssm_beta") && !strstr(src0[i]->name, "ssm_alpha")) return false;
    }
    return true;
}

// PXA_ATTN_KV_KQMMV (default 1; 0 = off). The same admission for the attention k and v projections: on a
// 2-card head split of the 27B they are two Q8_0 [n_embd x 512] matrices reading the same normed activation
// back to back, 1024 rows together, so KQMMV_MIN_ROWS left them on the incumbent (a q8_1 quantize launch and
// two MMVQ launches per attention layer). Taken as ONE grouped h2 launch from the f32 activation, like the
// delta-net beta/alpha pair above. The h2 dot rounds differently from the q8_1 dot: last-bit changes.
static bool kqmmv_attn_kv(const ggml_tensor * const * src0, int n) {
    static const bool on = [](){ const char * e = getenv("PXA_ATTN_KV_KQMMV"); return !e || atoi(e) != 0; }();
    if (!on || n < 2) return false;
    for (int i = 0; i < n; ++i) {
        if (!strstr(src0[i]->name, "attn_k.weight") && !strstr(src0[i]->name, "attn_v.weight")) return false;
    }
    return true;
}

static void kqmmv_run(int device, cudaStream_t stream, int T, const ggml_tensor * src1,
                      const ggml_tensor * const * src0, ggml_tensor * const * dst, int n, bool gu, float limit,
                      const float * bias = nullptr) {
    const int cc  = ggml_cuda_info().devices[device].cc;
    const int nsm = ggml_cuda_info().devices[device].nsm;
    kqmmv_mdesc ms[KQMMV_MAX_MATS];
    for (int i = 0; i < n; ++i) {
        ms[i].w     = src0[i]->data;
        ms[i].dst   = (float *) dst[i]->data;
        ms[i].nb01  = src0[i]->nb[1];
        ms[i].sd    = dst[i]->nb[1]/sizeof(float);
        ms[i].nrows = (int) src0[i]->ne[1];
        ms[i].bias  = bias;
    }
    CUDA_CHECK(kqmmv_run_mats(T, kqmmv_path(cc), (int) src1->ne[1], (const float *) src1->data, src1->nb[1]/sizeof(float),
                              (int) src1->ne[0], ms, n, gu, limit, nsm, stream, kqmmv_wide_on()));
    // engagement banner: the first launch of each form (plain, grouped, gate/up) per device
    static std::atomic<uint32_t> seen{0};
    const int form = gu ? 2 : bias ? 3 : (n > 1 ? 1 : 0);
    const uint32_t bit = 1u << (4*(device & 7) + form);
    if (!(seen.fetch_or(bit) & bit)) {
        int64_t rows = 0;
        for (int i = 0; i < n; ++i) rows += src0[i]->ne[1];
        fprintf(stderr, "PXA_KQMMV dev%d: FIRING %s (%s, ny=%d, K=%d, %d matrix(es), %lld rows, %s path%s)\n", device,
                form == 3 ? "single+add" : form == 2 ? "gate/up" : form == 1 ? "group" : "single", ggml_type_name(src0[0]->type), (int) src1->ne[1],
                (int) src1->ne[0], n, (long long) rows, kqmmv_path(cc) == KQMMV_H2 ? "h2" : "i8",
                !gu && kqmmv_wide(T, kqmmv_path(cc), (int) src1->ne[1], (int) src1->ne[0]) ? ", wide form" : "");
    }
}

bool ggml_cuda_kqmmv_mul_mats(int device, cudaStream_t stream, const ggml_tensor * src1,
                              const ggml_tensor * const * src0, ggml_tensor * const * dst, int n) {
    int64_t rows = 0;
    for (int i = 0; i < n; ++i) rows += src0[i]->ne[1];
    if (rows < KQMMV_MIN_ROWS && !kqmmv_dn_ba(src0, n) && !kqmmv_attn_kv(src0, n)) return false;
    {
        const int cc = ggml_cuda_info().devices[device].cc;
        const int T0 = kqmmv_kind(src0[0]->type);
        if (T0 >= 0 && kqmmv_wide(T0, kqmmv_path(cc), (int) src1->ne[1], (int) src1->ne[0]) && rows < KQMMV_WIDE_MIN_ROWS) return false;
    }
    const int gmax = ggml_cuda_kqmmv_group_on() ? KQMMV_MAX_MATS : 1;
    int i = 0;
    while (i < n) {
        const ggml_type t = src0[i]->type;
        int g = 1;
        while (i + g < n && g < gmax && src0[i + g]->type == t) ++g;
        kqmmv_run(device, stream, kqmmv_kind(t), src1, src0 + i, dst + i, g, false, 0.0f);
        i += g;
    }
    return true;
}

bool ggml_cuda_kqmmv_mul_mat_bias(int device, cudaStream_t stream, const ggml_tensor * src1, const ggml_tensor * src0,
                                  ggml_tensor * add, const ggml_tensor * bias) {
    if (!kqmmv_bias_on()) return false;
    if (src0->ne[1] < KQMMV_MIN_ROWS) return false;
    if (kqmmv_kind(src0->type) >= 0 && src0->ne[1] < KQMMV_WIDE_MIN_ROWS &&
        kqmmv_wide(kqmmv_kind(src0->type), kqmmv_path(ggml_cuda_info().devices[device].cc), (int) src1->ne[1], (int) src1->ne[0])) return false;
    if (!add || add->type != GGML_TYPE_F32 || !bias || bias->type != GGML_TYPE_F32 || !bias->data) return false;
    if (bias->ne[0] != src0->ne[1] || ggml_nrows(bias) != 1 || bias->nb[0] != sizeof(float)) return false;
    // the ADD has the MUL_MAT result shape; its buffer receives the sum
    if (!ggml_cuda_kqmmv_shape_ok(src0, src1, add)) return false;
    const ggml_tensor * s0[1] = { src0 };
    ggml_tensor * d0[1] = { add };
    kqmmv_run(device, stream, kqmmv_kind(src0->type), src1, s0, d0, 1, false, 0.0f, (const float *) bias->data);
    return true;
}

int ggml_cuda_kqmmv_up_gate(int device, cudaStream_t stream, ggml_tensor * dst) {
    if (!kqmmv_gu_on()) return -1;
    const ggml_tensor * up   = dst->src[0];
    const ggml_tensor * gate = dst->src[1];
    const ggml_tensor * src1 = dst->src[2];
    if (!up || !gate || !src1 || up->type != gate->type) return -1;
    if (dst->src[4] || dst->src[5]) return -1;                                  // biased (swiglu_oai) form
    if ((ggml_unary_op) dst->op_params[0] != GGML_UNARY_OP_SILU) return -1;
    if (!kqmmv_take(device, up->type, src1->ne[0], src1->ne[1], true)) return -1;
    if (up->ne[1] < KQMMV_MIN_ROWS/2) return -1;
    if (!ggml_cuda_kqmmv_shape_ok(up, src1, dst) || !ggml_cuda_kqmmv_shape_ok(gate, src1, dst)) return -1;
    if (up->ne[1] != gate->ne[1] || up->nb[1] != gate->nb[1]) return -1;
    const float limit = *(const float *)(dst->op_params + 1);
    const ggml_tensor * s0[2] = { up, gate };
    ggml_tensor * d0[2] = { dst, dst };
    kqmmv_run(device, stream, kqmmv_kind(up->type), src1, s0, d0, 2, true, limit);
    return 0;
}
