#include "levers.cuh"

#include "../../common.cuh"
#include "../pxa-enhance.cuh"   // the card set (g_pxa_topology) and the model profile the rules below read

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

// One row per declared lever. The row is the whole declaration: the variable a user sets, the
// short name the boot report prints, the value that means "behave as the base engine did", and
// the one line that says what it is for. The rule that turns an environment string into a value
// lives in pxa_lever_resolve() below, one case per row, because the rules are genuinely different
// from each other and hiding that behind a common parser would be a lie.
struct pxa_lever_row {
    const char * env;
    const char * report_name;
    int64_t      off;
    const char * doc;
};

static const pxa_lever_row g_pxa_lever_rows[PXA_LEVER_COUNT] = {
    { "PXA_FA_D512_VOLTA",     "FA_D512_VOLTA",   0,
      "sm_70 head 512/512 flash attention: 0 the unfused chain, 1 the vendored m8n8k4 MMA kernel, "
      "2 the vendored tile kernel (the ENHANCE default). Fenced by the graph builder's width cut, "
      "context floor, every-card probe, architecture and offload checks." },
    { "PXA_FA_D256_VOLTA_TILE", "FA_D256_VOLTA_TILE", 0,
      "sm_70 head 256/256 flash attention on the vendored no-tensor-core tile kernel: 0 the route "
      "this fork always had (vector at width 1, legacy WMMA above it), 1 the tile kernel for the "
      "speculative verify widths, 2 for width 1 as well. Fenced by a KV floor and a width window." },
    { "PXA_FA_D256_VOLTA_TILE_MINKV", "FA_D256_VOLTA_TILE_MINKV", 1280,
      "the KV length at or above which PXA_FA_D256_VOLTA_TILE takes a node. The gain is at depth, "
      "and a fused attention kernel on this architecture has a known short-KV defect at head 512." },
    { "PXA_FA_D256_VOLTA_TILE_MAXCOLS", "FA_D256_VOLTA_TILE_MAXCOLS", 8,
      "the widest query batch PXA_FA_D256_VOLTA_TILE takes. Above it the node keeps the route it "
      "always had, which at prefill width is the vendored m8n8k4 MMA kernel." },
    { "PXA_FA_MMA_VOLTA",      "FA_MMA_VOLTA",    0,
      "sm_70 large-batch flash attention on the vendored mainline m8n8k4 MMA kernel. Default on; "
      "=0 hands those shapes back to the tile route. Decode is never routed here." },
    { "PXA_FA_MMA_VOLTA_Q8",   "FA_MMA_VOLTA_Q8", 0,
      "sm_70 head-256 prefill with a MATCHED q8_0 K/V cache also reaches that MMA kernel, staged "
      "to f16 through the CUDA pool. Default on; =0 restores the f16-only predicate. The admission "
      "declines itself when the staging would not fit in free device memory." },
    { "PXA_FA_TILE_VOLTA",     "FA_TILE_VOLTA",   0,
      "sm_70 flash attention on the no-tensor-core tile kernel: 0 off (default -- this route emits "
      "corrupt output on this architecture), 1 every batch size, 2 large batch only. Benchmarking." },
    { "PXA_FA_TILE256",        "FA_TILE256",      0,
      "pre-Volta D=256 attention at batch > 8 uses the tile-f16 ncols=16 kernel instead of the "
      "single-column vec kernel. Default on; =0 is a kernel-selection rollback, not a fix." },
    { "PXA_FA_SWA_SLICE",      "FA_SWA_SLICE",    0,
      "restore the windowed KV slice at dispatch. UNSOUND with np>1 or slot reuse (it can cut a "
      "sequence's own cells out of the view and produce NaN logits). Single-sequence benchmarking only." },
    { "PXA_FA_SWA_KEEP",       "FA_SWA_KEEP",     0,
      "keep n_swa in op_params[4] so the kernels' mask-driven KV_min_max scan bounds the SWA "
      "iteration. Default on; =0 restores the old full-range dispatch behaviour." },
    { "PXQ_SM60_FA_VEC_F32",   "SM60_FA_VEC_F32", 0,
      "sm_60 decode (batch <= 8) uses the fp32-accumulating vec kernel; the fp16 one flips ~3-4% "
      "of top-1 tokens and is not faster on a bandwidth-bound card. Default on." },
    { "PXA_CORE_ROUTES",       "CORE_ROUTES",     0,
      "print, at exit, how many flash-attention nodes each route served on each device. Default off." },
    { "PXA_FA_QKV_DIRECT",     "FA_QKV_DIRECT",   0,
      "sm_60 narrow attention (width 1..8, head 256, q4_0 or q8_0 K/V) on a kernel that reads the "
      "quantized cache in place and serves a whole GQA group per block. Default on at the ENHANCE level; "
      "=0 restores the vec-f32 route." },
    { "PXA_FA_QKV_TILE",       "FA_QKV_TILE",     0,
      "sm_60 wide attention (width > 8, head 256, q4_0 or q8_0 K/V) reads the quantized cache in place "
      "instead of converting it to f16 for the tile kernel. Default on at the ENHANCE level; "
      "=0 keeps the tile-f16 route and its conversion." },
    { "PXA_FA_QKV_DIRECT_VOLTA", "FA_QKV_DIRECT_VOLTA", 0,
      "sm_70 narrow attention (width 1..8, head 256, q4_0 or q8_0 K/V) takes the FA_QKV_DIRECT kernel "
      "instead of WMMA ncols 8, which converts the whole K/V cache to f16 every decode step. Default on "
      "at the ENHANCE level; =0 restores the WMMA route." },

    // ---- from pxa-enhance.cuh: level-only rules -------------------------------------------------
    { "PXA_P100_FP16_GEMM",    "P100_FP16_GEMM",  0,
      "sm_60 dense GEMMs ride dequant->fp16 + cublasGemmEx COMPUTE_16F instead of dequant->fp32 SGEMM "
      "(GP100 has double-rate fp16). Default on at DEFAULT and ENHANCE, off at REFERENCE; =0 rolls back at any level." },
    { "PXA_SPEC_1ROW",         "SPEC_1ROW",       0,
      "a single-output-row GEMV also takes the 1-row path at spec-verify batch sizes (Ny<=8). Default on at "
      "DEFAULT and ENHANCE, off at REFERENCE; a measured no-op on sm_60, a documented +6.6% on sm_70." },
    { "PXA_FA_MASK_SKIP_TILE", "FA_MASK_SKIP_TILE", 0,
      "skip the fully masked KV tiles in the tile-f16 flash-attention kernel (sm_60 only: bit-identical by "
      "construction). Default on at DEFAULT and ENHANCE, off at REFERENCE; =0 rolls back." },
    { "PXA_FA_TILE_F32ACC",    "FA_TILE_F32ACC",  0,
      "fp32 running state (kqmax, kqsum, VKQ) in the tile-f16 flash-attention kernel: the f16 accumulator's "
      "rounding depended on where a request's band sat in the KV ring. Default on at DEFAULT and ENHANCE, off "
      "at REFERENCE (which keeps the original arithmetic); =0 restores the f16 accumulator for an A/B." },
    { "PXA_FA_TILE_V2",        "FA_TILE_V2",      0,
      "an alternative schedule for the tile-f16 flash-attention arithmetic (16-byte shared loads, no row pad); "
      "bitwise equal to the shipping kernel. Default off at every level; =1 arms it. No speed number exists." },
    { "PXA_FA_PREFILL_SPLIT",  "FA_PREFILL_SPLIT", 0,
      "an attention batch of at least this many tokens builds the non-FA batched-cuBLAS chain even under -fa on "
      "(values 1..8 clamp to 9). Default 0 (off) at every level and posture: the non-FA chain inflates the "
      "compute buffer ~2.35x. Opt in, e.g. =64." },
    { "PXA_MODE",              "MODE",            0,
      "the user-facing posture: balance (default) = -fa on serving, max = -fa off bulk ingest. Any value "
      "starting with m or M selects max. Drives the startup report; the -fa/-ub defaults live server-side." },

    // ---- from pxa-enhance.cuh: tier masks ------------------------------------------------------
    { "PXA_PXQ23_MMVQ",        "PXQ23_MMVQ",      0,
      "which of the two LOW tiers may ride the q8_1 MMVQ decode path: a bitmask, bit0 PXQ2, bit1 PXQ3, 3 both, "
      "0 off. DEFAULT OFF: the decode is a large win on sm_70 but the mean KL divergence against the file's own "
      "exact path is worse than the PXQ4 kernel's, so it is a lever to opt into. Refuses (and says so) when a "
      "PXQ2/PXQ3 book or sub-scale override is set, or when the frozen s8 books no longer match the codec's tables." },
    { "PXA_PXQ_MMV_H2",        "MMV_H2",      3,
      "half2 inner loop for the dense PXQ decode mmv, sm_60 (GP100) only: a bitmask, bit0 PXQ2, bit1 PXQ3, bit2 "
      "PXQ4. Default 3 (both low tiers) at DEFAULT and ENHANCE, off at REFERENCE; =0 returns to the exact fp32 "
      "loop, =7 arms PXQ4 as well. Not bit-exact against the fp32 arm; fidelity-gated." },

    // ---- from pxa-enhance.cuh: rules that read the card set and the model profile ----------------
    { "PXA_ROUTER_FUSE",       "ROUTER_FUSE",     0,
      "the MoE router-logits F32 GEMV (ffn_gate_inp x one decode token) on a dedicated warp-per-row kernel instead "
      "of a bare cublasSgemm. ENHANCE default 1 only on a PURE sm_70 (non-mixed) topology with a MoE model (or "
      "no model registered yet); 0 elsewhere (a -3.1% loss measured on a mixed V100+P100 rig, a +1.6% kill on "
      "sm_60). Explicit value wins at any level: 0 off, 1 the sm_70 gate, 2 test all-arch, 3 the sm_60-shaped "
      "GEMV; a value above 3 is reported and forced off." },
    { "PXA_VOLTA_CUBLAS_NE11", "VOLTA_CUBLAS_NE11", 0,
      "sm_70 routes dense quantized GEMMs with ne11 >= N to fp16 cuBLAS. Default 64 at DEFAULT and ENHANCE "
      "(+9.4% prefill on one V100), 0 at REFERENCE and for the muse-glimmer arch (a measured loss there); "
      "=0 is MMQ always, other values retune the threshold." },
    { "PXA_PXQ_INT8_PREFILL",  "INT8_PREFILL", 0,
      "the DP4A int8 prefill tile. Default 1 at ENHANCE (the sm_61-only ship gate: +182% prefill on the 1080 Ti), "
      "0 at DEFAULT and REFERENCE; 2 is a test mode on every arch; values outside 0..2 mean 0." },
    { "PXA_SPEC_RELAXED",      "SPEC_RELAXED",    0,
      "relaxed speculative acceptance. With the variable unset it is on only at ENHANCE and only where the "
      "lossless PXA_SPEC_SAMPLED rule is explicitly off (=0); common/sampling.cpp mirrors this rule." },

    // ---- from pxa-enhance.cuh (and the sites that read these variables): rules over the level, the card set and the model ----
    { "PXA_FA_GQA_PACK",       "FA_GQA_PACK",     0,
      "Q heads packed into one block of the f32 vec flash-attention kernel (0 = the stock per-head kernel; 2/4/8). "
      "Reached on D=256 f16-KV decode only. Default 4 at ENHANCE on a MULTI-card pure sm_60 topology, 0 on every "
      "other card set and level: not bit-exact against the shipped V pass and measured at one fill on the 4x P100 seat." },
    { "PXA_MOE_DEVICE_MAP",    "MOE_DEVICE_MAP",  0,
      "0 = the host MoE expert-row mapping; 1 = the device kernels (no D2H + stream sync inside graph compute); 2 = the "
      "device kernels plus a host cross-check. Default 1 at ENHANCE on a multi-card sm_60 topology, or for a Gemma 4 "
      "routed-expert model on a multi-card all-sm_70 topology (+35% 12.7k prefill, two V100); 0 elsewhere." },
    { "PXA_FUSE_DELTANET",     "FUSE_DELTANET",   0,
      "bitmask of the Gated-DeltaNet decode fusions (cluster, state carry, row absorb, out-gate). Default 55 at DEFAULT "
      "and ENHANCE, 0 at REFERENCE (the eager path)." },
    { "PXA_PXQ_MMVQ",          "MMVQ",        0,
      "PXQ4/PXQ4HQ decode through the q8_1 MMVQ kernel: 0 off, 1 the sm_70+ ship gate, 2 all-sm_61. The unset default "
      "is 1 when an sm_70+ card is present, 2 for an all-sm_61 fleet, 0 otherwise, and only for a model that carries an "
      "eligible tensor (PXQ4/PXQ4HQ, plus the low tiers PXA_PXQ23_MMVQ admits); 0 at REFERENCE." },
    { "PXA_PXQ_GEMM_2D",       "GEMM_2D",     0,
      "the 2D prefill GEMM for the PXQ4 tier: 1 sm_60 only, 2 sm_60 + sm_70. Default 0, except 1 at ENHANCE for a DENSE "
      "model that carries PXQ tensors on a box with an sm_60 card (+35% P100 dense prefill; the MoE cell is unmeasured)." },
    { "PXA_ENHANCE_DBG",       "ENHANCE_DBG",     0,
      "any non-zero value prints the detected topology and the per-topology ENHANCE choices at CUDA init. Diagnostic." },

    // ---- a resolver that lived in its own translation unit -------------------------------------------
    { "PXA_FA_MASK_SKIP_TILE_F32", "FA_MASK_SKIP_TILE_F32", 0,
      "skip the fully masked KV tiles in the tile-f32 flash-attention kernel (sm_61 prefill and forced-F32 precision "
      "at ne1 > 8). Default OFF at every level: the 32-wide tile makes the scan overhead ~2x the tile-f16 lever's "
      "relative cost and its own silicon A/B has not run. =1 enables." },

    // ---- the CUDA house levers: host-side / index-math, arch-independent (ENHANCE-default on) -----------
    { "PXA_NORM_REGCACHE",     "NORM_REGCACHE",   0,
      "register-cached norm variants: the second pass re-reads a cached value, the scaling expression is untouched, "
      "so the output is bit-identical. Default on at ENHANCE, off at DEFAULT and REFERENCE; block_size == 1024 only." },
    { "PXA_CONCAT_FLAT",       "CONCAT_FLAT",     0,
      "flattened non-contiguous concat: a launch-geometry change over a pure copy (one flat grid instead of a 3-D "
      "one). Default on at ENHANCE, off at DEFAULT and REFERENCE." },
    { "PXA_CPY_FASTDIV",       "CPY_FASTDIV",     0,
      "fastdiv index math in the float cpy kernel: an integer division replaced by the multiply-shift that computes "
      "the same quotient over the kernel's index range. Default on at ENHANCE, off at DEFAULT and REFERENCE." },
    { "PXA_GETROWS_NARROW",    "GETROWS_NARROW",  0,
      "flattened narrow-row get_rows launch: the same rows copied by the same code, one flat grid instead of one block "
      "per row. Default on at ENHANCE, off at DEFAULT and REFERENCE." },
    { "PXA_TOPK_MOE_MULTIROW", "TOPK_MOE_MULTIROW", 0,
      "the multi-row fused top-k MoE router processes several rows per block behind one block-wide aliasing barrier; "
      "the per-row arithmetic and its order are unchanged. Default on at ENHANCE, off at DEFAULT and REFERENCE." },
};

struct pxa_lever_state {
    int64_t      value;
    bool         set;   // the user set the variable to something this build understands
    const char * why;
};

static bool pxa_env_on_unless_zero(const char * v) { return !(v && v[0] == '0'); }
static bool pxa_env_on_if_one    (const char * v) { return   v && v[0] == '1';  }

// A lever whose value is a number rather than a mode. An unset, empty, unparsable or negative
// string keeps the row's default instead of silently meaning zero -- a KV floor of 0 and a KV floor
// of "nonsense" are very different instructions and only one of them was asked for.
static int64_t pxa_env_int(const char * env, const char * v, int64_t def) {
    if (!v || v[0] == '\0') {
        return def;
    }
    char * end = nullptr;
    const long long n = strtoll(v, &end, 10);
    if (end == v || *end != '\0' || n < 0) {
        fprintf(stderr, "%s=%s is not a non-negative integer; keeping %lld\n", env, v, (long long) def);
        return def;
    }
    return (int64_t) n;
}

// PXA_PXQ23_MMVQ's self-check: recompute the frozen s8 books from the codec's own float tables. If a
// table is ever re-frozen without re-deriving the snap, the two decoders would disagree by a whole book
// entry; this catches that at startup instead of in someone's perplexity.
static bool pxa_pxq23_mmvq_snap_ok() {
    static const float  b2[4] = PXQ2_BOOK_INIT,      b3[8] = PXQ3_BOOK_INIT;
    static const int8_t q2[4] = PXQ2_MMVQ_S8_INIT,   q3[8] = PXQ3_MMVQ_S8_INIT;
    float a2 = 0.0f, a3 = 0.0f;
    for (int i = 0; i < 4; ++i) a2 = fabsf(b2[i]) > a2 ? fabsf(b2[i]) : a2;
    for (int i = 0; i < 8; ++i) a3 = fabsf(b3[i]) > a3 ? fabsf(b3[i]) : a3;
    if (!(a2 > 0.0f) || !(a3 > 0.0f)) return false;
    if (a2 != PXQ2_MMVQ_ABSMAX || a3 != PXQ3_MMVQ_ABSMAX) return false;
    for (int i = 0; i < 4; ++i) if ((int) rintf(b2[i] * (127.0f/a2)) != (int) q2[i]) return false;
    for (int i = 0; i < 8; ++i) if ((int) rintf(b3[i] * (127.0f/a3)) != (int) q3[i]) return false;
    return true;
}

// The rules that hand a default over the level, the card set and the model. Each is exactly the
// function it replaces in pxa-enhance.cuh; the comments are the ones that explain the measured basis.
static int64_t pxa_lever_rule_default(int id) {
    const int level = ggml_pxa_config_level();
    const pxa_topology_t & t = g_pxa_topology;
    switch ((pxa_lever_id) id) {
        case PXA_LEVER_FA_GQA_PACK:
            // PXA_FA_GQA_PACK ENHANCE default (2026-09-03). The packed-GQA vec kernel is reached on ONE shape
            // only: the f32 vec FA kernel at Dk==Dv==256, cols_per_block==1, f16 K and V, no logit softcap --
            // i.e. D=256 f16-KV decode. Unlike the nine house levers this IS a distinct kernel with its own
            // register budget, so it arms only on a MULTI-CARD Pascal-GP100 (sm_60) topology and stays env-only
            // everywhere else -- a single P100 and every other architecture are unmeasured for it.
            //
            // CORRECTED 2026-09-12: no throughput basis is on record beyond NH=4 +6.9% at fill 8881 on the
            // 4x P100 seat (n=3, one fill); it is NOT bit-exact against the shipped V pass; the reviewer-cell
            // effect does not reproduce. Armed by the default level on this cell and disclosed in
            // docs/COOKBOOK.md; not to be quoted as a win. Bug 85 owns the firing counter that would settle it.
            if (level < 2) return 0;
            if (!t.valid || t.mixed || !t.multi || !t.has_sm60) return 0;
            return 4;
        case PXA_LEVER_MOE_DEVICE_MAP:
            // Mode 1 moves the MoE expert-row mapping onto the device, removing a D2H + cudaStreamSynchronize
            // from inside graph compute. It can only fire on a MoE model, so it arms on a MULTI-CARD sm_60
            // topology and, since 2026-09-20, for a Gemma 4 routed-expert model on a multi-card all-sm_70
            // topology (the ids readback blocks the host inside every fused PXQ expert op; two V100, lever the
            // only difference: prefill 2,347 -> 3,172 / 2,116 -> 2,976 / 1,799 -> 2,534 tok/s at 2.4k / 6.4k /
            // 12.7k, decode unchanged, greedy text byte-identical in eleven reps). One card, and other
            // routed-expert architectures on sm_70, stay env-only until measured. Mode 2 is env-only.
            if (level < 2) return 0;
            if (!t.valid || t.mixed) return 0;
            if (pxa_model_known() && !pxa_model_is_moe()) return 0;
            if (t.multi && t.has_sm60) return 1;
            if (t.multi && t.has_sm70 && !t.has_sm60 && !t.has_sm61 && !t.has_other &&
                pxa_model_known() && strncmp(pxa_model().arch_name, "gemma4", 6) == 0) return 1;
            return 0;
        case PXA_LEVER_FUSE_DELTANET:
            // REFERENCE -> 0 (eager path), DEFAULT/ENHANCE -> 55 = bits 0|1|2|4|5. The long history of the bits
            // (what each fuses, why bit 3 stays out, the bit-1 out-gate race and its 2026-09-27 fix) is in
            // pxa-enhance.cuh above pxa_fuse_deltanet_default().
            return level == 0 ? 0 : 55;
        case PXA_LEVER_PXQ_MMVQ: {
            // ENHANCE auto-default (device x model). The ship recipe (A4m) runs MMVQ ON: fidelity-neutral
            // measured PAIRED at matched batch (dppl +0.016%). Auto ON only when the model carries
            // MMVQ-eligible tensors AND a DP4A-capable device exists: sm_70+ present -> 1; all-sm_61 -> 2;
            // pure sm_60 / no census -> 0 (P100 has no DP4A). Armed at DEFAULT as well as ENHANCE since
            // 2026-07-31; REFERENCE (0) still opts out.
            if (level == 0) return 0;
            if (!pxa_model_known()) return 0;
            // A uniform PXQ2/PXQ3 file carries no PXQ4 tensor at all, so count the low tiers only when
            // PXA_PXQ23_MMVQ actually admits them.
            const int mask23 = (int) pxa_lever(PXA_LEVER_PXQ23_MMVQ);
            const int n_elig = pxa_model().n_pxq_mmvq_tensors
                             + ((mask23 & 1) ? pxa_model().n_pxq2_tensors : 0)
                             + ((mask23 & 2) ? pxa_model().n_pxq3_tensors : 0);
            if (n_elig <= 0) return 0;
            if (!t.valid) return 0;
            if (t.has_sm70 || t.has_other) return 1;
            if (t.has_sm61 && !t.has_sm60) return 2;
            return 0;
        }
        default:
            return g_pxa_lever_rows[id].off;
    }
}

// Resolve ONE row: read its environment variable, apply its rule and its default at the current
// config level (and, for a rule over the card set or the model, those), say what it dislikes about
// the value on stderr, and record the REASON beside the value. A row may read another row through
// pxa_lever() (PXQ_MMVQ reads PXQ23_MMVQ, the tile-f32 skip reads ENHANCE_DBG), so rows can be
// resolved in any order and, above all, one at a time -- but there is no cycle, and a PER_PROFILE row
// is resolved under one lock, so it must not read another PER_PROFILE row.
static pxa_lever_state pxa_lever_resolve(int i) {
    const int    level      = ggml_pxa_config_level();
    const char * level_name = ggml_pxa_config_level_name();

    const char * v   = getenv(g_pxa_lever_rows[i].env);
    const bool   set = v && v[0] != '\0';
    int64_t      value;

    switch ((pxa_lever_id) i) {
        case PXA_LEVER_FA_D512_VOLTA:
            // The default follows the config level like every other default: REFERENCE and
            // DEFAULT keep the unfused chain, ENHANCE (the shipping level) takes the tile kernel.
            value = !set ? (level >= 2 ? 2 : 0)
                         : (v[0] == '1' && v[1] == '\0') ? 1
                         : (v[0] == '2' && v[1] == '\0') ? 2 : 0;
            if (set && value == 0 && strcmp(v, "0") != 0) {
                fprintf(stderr, "PXA_FA_D512_VOLTA=%s is not a value this lever has (0 = off, 1 = MMA kernel, "
                                "2 = tile kernel); treating it as 0 and keeping the unfused attention chain\n", v);
            }
            break;

        case PXA_LEVER_FA_D256_VOLTA_TILE:
            // DEFAULT 2 AT THE ENHANCE LEVEL, like every other default that follows the
            // config level. 2 rather than 1 because width 1 -- plain decode -- turned out to
            // be the same win: on the 15k long class the plain step gains 9.3% and the
            // speculative arms 12-18%, and per node at width 1 the kernel beats not only the
            // vector kernel this engine had but mainline's own by 1.6-1.7x at depth.
            value = !set ? (level >= 2 ? 2 : 0)
                         : (v[0] == '1' && v[1] == '\0') ? 1
                         : (v[0] == '2' && v[1] == '\0') ? 2 : 0;
            if (set && value == 0 && strcmp(v, "0") != 0) {
                fprintf(stderr, "PXA_FA_D256_VOLTA_TILE=%s is not a value this lever has (0 = off, "
                                "1 = verify widths, 2 = width 1 as well); treating it as 0\n", v);
            }
            break;

        case PXA_LEVER_FA_D256_VOLTA_TILE_MINKV:
        case PXA_LEVER_FA_D256_VOLTA_TILE_MAXCOLS:
            value = pxa_env_int(g_pxa_lever_rows[i].env, v, g_pxa_lever_rows[i].off);
            break;

        case PXA_LEVER_FA_MMA_VOLTA:
            value = pxa_env_on_unless_zero(v) ? 1 : 0;
            // =2 also routed DECODE here. DISABLED 2026-09-20: the decode-width instances of
            // that kernel are stubs on sm_70, so lifting the gate faults the card with an
            // unspecified launch failure at widths 1, 2 and 4. Accepted and treated as =1, loudly.
            if (set && v[0] == '2') {
                fprintf(stderr, "PXA_FA_MMA_VOLTA=2 is disabled in this build (its decode-width kernel instances are not "
                                "built for sm_70 and the launch faults); treating it as =1\n");
            }
            break;

        case PXA_LEVER_FA_TILE_VOLTA:
            value = !set ? 0 : (v[0] == '0') ? 0 : (v[0] == '1') ? 1 : (v[0] == '2') ? 2 : 0;
            if (value == 2) {
                fprintf(stderr, "PXA_FA_TILE_VOLTA=2: sm_70 large-batch flash-attention -> tile kernel "
                                "(KNOWN TO PRODUCE CORRUPT OUTPUT ON THIS ARCH -- benchmarking only)\n");
            }
            break;

        case PXA_LEVER_FA_MMA_VOLTA_Q8: value = pxa_env_on_unless_zero(v) ? 1 : 0; break;
        case PXA_LEVER_FA_TILE256:      value = pxa_env_on_unless_zero(v) ? 1 : 0; break;
        case PXA_LEVER_FA_SWA_KEEP:     value = pxa_env_on_unless_zero(v) ? 1 : 0; break;
        case PXA_LEVER_SM60_FA_VEC_F32: value = pxa_env_on_unless_zero(v) ? 1 : 0; break;
        case PXA_LEVER_FA_SWA_SLICE:    value = pxa_env_on_if_one(v)     ? 1 : 0; break;
        case PXA_LEVER_CORE_ROUTES:     value = pxa_env_on_if_one(v)     ? 1 : 0; break;
        case PXA_LEVER_FA_QKV_DIRECT:
            // On at the ENHANCE level (the shipping default), off under REFERENCE / DEFAULT;
            // set, anything but 0 arms it.
            value = !set ? (level >= 2 ? 1 : 0) : (pxa_env_on_unless_zero(v) ? 1 : 0);
            break;
        case PXA_LEVER_FA_QKV_DIRECT_VOLTA:
            value = !set ? (level >= 2 ? 1 : 0) : (pxa_env_on_unless_zero(v) ? 1 : 0);
            break;
        case PXA_LEVER_FA_QKV_TILE:
            // On at the ENHANCE level like QKV_DIRECT: NMSE 2-17x below the tile-f16 route on all 12
            // wide cases; one case's single worst element is 7% above (1.07e-4 vs 9.99e-5). =0 restores
            // the tile-f16 route and its KV conversion.
            value = !set ? (level >= 2 ? 1 : 0) : (pxa_env_on_unless_zero(v) ? 1 : 0);
            break;

        // ---- from pxa-enhance.cuh: level-only rules. An empty value is a value here (atoi("") == 0): the
        // resolvers these rows replaced tested getenv() against null, not against "".
        case PXA_LEVER_P100_FP16_GEMM:
        case PXA_LEVER_SPEC_1ROW:
        case PXA_LEVER_FA_MASK_SKIP_TILE:
        case PXA_LEVER_FA_TILE_F32ACC:
            // REFERENCE forces the lever off (a baseline must be able to select the arithmetic it is
            // measured against); DEFAULT and ENHANCE keep it on; the variable wins at any level.
            value = v ? (atoi(v) != 0) : (level != 0);
            break;
        case PXA_LEVER_FA_TILE_V2:
            value = v && atoi(v) != 0;
            break;
        case PXA_LEVER_FA_PREFILL_SPLIT:
            // Opt-in at EVERY level: the non-FA prefill chain inflates the compute buffer ~2.35x
            // (1956 -> 4607 MiB) and OOMs a 16 GB card at ub 2048, so the level never arms it.
            if (v) {
                const int t = atoi(v);
                value = t <= 0 ? 0 : (t < 9 ? 9 : t);   // 1..8 clamp to 9 (decode / MTP-verify safety)
            } else {
                value = 0;
            }
            break;
        case PXA_LEVER_MODE:
            value = (v && (v[0] == 'm' || v[0] == 'M')) ? 1 : 0;
            break;

        // ---- from pxa-enhance.cuh: tier masks, each with the one line it printed when it armed ----
        case PXA_LEVER_PXQ23_MMVQ: {
            if (level == 0) {                        // REFERENCE opts out of everything
                value = 0;
                break;
            }
            // DEFAULT OFF, and the reason is measured, not cautionary. Decode is a large win on
            // sm_70 (uniform PXQ2 +36.4% at np1 and +47.8% at np2, uniform PXQ3 +36.4% / +51.9%,
            // 2026-09-10, 2xV100, medians of 7), but the pre-registered ship rule also required the
            // mean KL divergence against the file's own exact path to be no worse than the PXQ4
            // kernel already on this path shows against its own, and it is not: 0.004624 (PXQ2) and
            // 0.002186 (PXQ3) against the PXQ4 comparator's 0.000951, one session, -b 8 -ub 8.
            // So this is a lever a user opts into with the numbers in front of them, not a default.
            // docs/lab/LEVERS.md carries the full table and the confound (the three tiers are three
            // files, and the uniform-PXQ2 file's PPL 23.2 is a fragile operating point).
            int m = v ? atoi(v) : 0;
            if (m < 0 || m > 3) m = 0;
            if (m) {
                const char * ovr = getenv("PXA_PXQ2_BOOK") ? "PXA_PXQ2_BOOK"
                                 : getenv("PXA_PXQ3_BOOK") ? "PXA_PXQ3_BOOK"
                                 : getenv("PXA_PXQ_CEIL_V2") ? "PXA_PXQ_CEIL_V2"
                                 : getenv("PXA_PXQ2_V3") ? "PXA_PXQ2_V3"
                                 : getenv("PXA_PXQ6_SUB") ? "PXA_PXQ6_SUB"
                                 : getenv("PXA_PXQ2_SUB") ? "PXA_PXQ2_SUB"
                                 : getenv("PXA_PXQ3_SUB") ? "PXA_PXQ3_SUB" : nullptr;
                if (ovr) {
                    fprintf(stderr, "PXA_PXQ23_MMVQ: DISABLED — %s is set and this path holds frozen "
                                    "book/sub copies\n", ovr);
                    m = 0;
                } else if (!pxa_pxq23_mmvq_snap_ok()) {
                    fprintf(stderr, "PXA_PXQ23_MMVQ: DISABLED — the frozen s8 books no longer match "
                                    "PXQ2_BOOK_INIT/PXQ3_BOOK_INIT (re-derive the snap)\n");
                    m = 0;
                } else {
                    fprintf(stderr, "PXA_PXQ23_MMVQ: mode %d (%s%s%s decode via the q8_1 MMVQ kernel, "
                                    "s8 book snap — NOT bit-exact vs the fused fp16 mmv, fidelity-gated)\n",
                            m, (m & 1) ? "PXQ2" : "", (m == 3) ? "/" : "", (m & 2) ? "PXQ3" : "");
                }
            }
            value = m;
            break;
        }
        case PXA_LEVER_PXQ_MMV_H2: {
            if (level == 0) {                        // REFERENCE opts out of everything
                value = 0;
                break;
            }
            int m = v ? atoi(v) : 3;                 // DEFAULT ON, cc == 600 only: both low tiers, per the ship rule
            if (m < 0 || m > 7) m = 0;
            if (m) {
                fprintf(stderr, "PXA_PXQ_MMV_H2: mode %d (%s%s%s%s%s dense decode via the half2 pair-LUT "
                                "loop, sm_60 only — NOT bit-exact vs the fp32 mmv, fidelity-gated; the "
                                "fp16-exact book self-check runs at first dispatch)\n",
                        m, (m & 1) ? "PXQ2" : "", ((m & 1) && (m & 6)) ? "/" : "",
                           (m & 2) ? "PXQ3" : "", ((m & 3) && (m & 4)) ? "/" : "",
                           (m & 4) ? "PXQ4" : "");
            }
            value = m;
            break;
        }

        // ---- from pxa-enhance.cuh: rules that read the card set and the model profile ----------------
        case PXA_LEVER_ROUTER_FUSE: {
            // TOPOLOGY- AND MODEL-AWARE ENHANCE default: 1 ONLY on a PURE-sm_70 (non-mixed) topology, single or
            // multi V100, with a MoE model (or none registered yet: the router GEMV op only exists on MoE, so
            // this is self-gating). On a MIXED V100+P100 rig it is 0: T4 measured router_fuse NEUTRAL-to-NEGATIVE
            // there (-3.1% at the winning -ts 1.4,0.6), because -sm layer makes the P100 the decode bottleneck
            // and a V100-only GEMV fusion cannot pay while adding tie-flap risk. Pure P100 stays 0 (a +1.6% kill
            // on sm_60). REFERENCE/DEFAULT -> 0. Explicit env ALWAYS wins at any level and is NOT topology-masked.
            // G3-class: the fuse reorders FP math vs cuBLAS (ULP logit deltas can flap expert ties).
            int mode = 0;
            if (v) {
                int m = atoi(v);
                if (m < 0 || m > PXA_ROUTER_FUSE_MAX_MODE) {
                    // Highest mode this build implements. An env value above it is REPORTED, never silently
                    // folded into 0: that is what made a pre-mode-3 binary run two identical A/B arms and
                    // report a result.
                    fprintf(stderr, "PXA: PXA_ROUTER_FUSE=%d is out of range for this build "
                                    "(max %d) -- lever FORCED OFF (this is NOT a control arm)\n",
                            m, PXA_ROUTER_FUSE_MAX_MODE);
                    m = 0;
                }
                mode = m;   // 3 = sm_60-shaped 128-thr float4 GEMV (2026-08-03)
            } else if (level == 2 && g_pxa_topology.valid
                       && g_pxa_topology.has_sm70 && !g_pxa_topology.mixed
                       && (!pxa_model_known() || pxa_model_is_moe())) {
                mode = 1;
            }
            value = mode;
            break;
        }
        case PXA_LEVER_VOLTA_CUBLAS_NE11:
            // sm_70 routes dense quantized GEMMs with ne11 >= N to fp16 cuBLAS. REFERENCE -> 0 (MMQ-always);
            // DEFAULT/ENHANCE -> 64 (+9.4% prefill measured, single V100, public PXQ2; decode untouched).
            // MODEL-AWARE (muse-glimmer): the fp16-cuBLAS route is a measured LOSS there (pp2048 411 vs
            // 619 t/s with MMQ, 2x V100, Q4_K_XL), so the auto default is 0 for that arch. Env still wins.
            if (v) {
                value = atoi(v);
            } else if (level == 0) {
                value = 0;
            } else if (pxa_model_known() && strcmp(pxa_model().arch_name, "muse-glimmer") == 0) {
                value = 0;  // measured -34% prefill on this arch
            } else {
                value = 64;
            }
            break;
        case PXA_LEVER_PXQ_INT8_PREFILL: {
            // ENHANCE defaults mode 1 -- which IS the sm_61-only ship gate (the cc==610 dispatch check is
            // unchanged); +182% prefill measured on the 1080 Ti. REFERENCE/DEFAULT -> 0 (byte-identical
            // dispatch). Explicit env wins (2 = TEST all-arch).
            int m = v ? atoi(v) : (level == 2 ? 1 : 0);
            if (m < 0 || m > 2) m = 0;
            value = m;
            break;
        }
        case PXA_LEVER_SPEC_RELAXED: {
            // Level default (the actual consumer is common/sampling.cpp, which mirrors this logic): ENHANCE
            // -> on (spec lanes only, G3-class), but since v2026.10 the lossless PXA_SPEC_SAMPLED rule is the
            // default, so relaxed is the level default only where that rule is explicitly off.
            if (v) {
                value = atoi(v) != 0;
            } else {
                const char * s = getenv("PXA_SPEC_SAMPLED");
                value = !(s && atoi(s) == 0) ? 0 : (level == 2);
            }
            break;
        }

        // ---- the variable, else a rule over the level, the card set and the model ---------------------
        case PXA_LEVER_FA_GQA_PACK:
        case PXA_LEVER_MOE_DEVICE_MAP:
        case PXA_LEVER_FUSE_DELTANET:
        case PXA_LEVER_PXQ_MMVQ:
            value = v ? atoi(v) : pxa_lever_rule_default(i);
            break;
        case PXA_LEVER_PXQ_GEMM_2D: {
            // gemm_2d ENHANCE auto (2026-07-29, model-aware): mode 1 (sm_60 only) when an sm_60 device is
            // present AND the loaded model is DENSE AND it carries PXQ tensors. Basis: +35% P100 dense prefill
            // on the coalesced binary. The sm_60 x MoE cell was only ever measured against the PRE-coalescing
            // incumbent (+0.89% prefill, decode flat) -- post-coalescing it is UNMEASURED and the sm_70 dense
            // flip (-18.6%) shows the coalescing fix can invert a win, so MoE stays OFF until someone
            // measures it. Explicit env always wins; re-resolved when the profile lands.
            int m = v ? atoi(v) : 0;
            if (!v && level == 2 && g_pxa_topology.valid && g_pxa_topology.has_sm60
                && pxa_model_known() && !pxa_model_is_moe() && pxa_model().n_pxq_mmvq_tensors > 0) {
                m = 1;
                fprintf(stderr, "PXA_AUTO: PXQ_GEMM_2D=1 (ENHANCE x sm_60 present x dense PXQ model: "
                                "+35%% P100 dense prefill measured 2026-07-28; MoE stays off — post-"
                                "coalescing sm_60 MoE cell unmeasured; override PXA_PXQ_GEMM_2D)\n");
            }
            if (m < 0 || m > 2) m = 0;
            value = m;
            break;
        }
        case PXA_LEVER_ENHANCE_DBG:
            value = v && atoi(v) != 0;
            break;

        case PXA_LEVER_FA_MASK_SKIP_TILE_F32: {
            // Default OFF at every level: the silicon A/B has not run, and the surviving gain regime is np1
            // causal strictly-future tiles only. Deliberately NOT folded into FA_MASK_SKIP_TILE: that lever's own
            // silicon gate is a separate question and piggy-backing would ship a second unmeasured default.
            const bool on = v != nullptr && atoi(v) != 0;
            if (pxa_lever(PXA_LEVER_ENHANCE_DBG) != 0) {
                fprintf(stderr, "PXA_ENHANCE_DBG: fa_mask_skip_tile_f32=%s (%s)\n",
                        on ? "on" : "off", v ? "env" : "default");
            }
            value = on;
            break;
        }

        // The house levers: an explicit setting wins in both directions (so every A/B in the campaign log
        // reproduces bit for bit); unset, they follow the config level -- ON at ENHANCE, OFF below it.
        case PXA_LEVER_NORM_REGCACHE:
        case PXA_LEVER_CONCAT_FLAT:
        case PXA_LEVER_CPY_FASTDIV:
        case PXA_LEVER_GETROWS_NARROW:
        case PXA_LEVER_TOPK_MOE_MULTIROW:
            value = v ? (atoi(v) != 0) : (level >= 2);
            break;

        default:                        value = g_pxa_lever_rows[i].off; break;
    }

    pxa_lever_state st;
    st.value = value;
    // The rows that came over from pxa-enhance.cuh answer "was the variable present", as the code they
    // replaced did; the flash-attention rows above answer "was it set to something non-empty".
    st.set   = i >= PXA_LEVER_REPORT_END ? v != nullptr : set;
    // The reason is recorded WITH the value, at the moment the value is decided. This is the
    // whole point of the registry: the report cannot say "explicit env override" about a
    // default, because it does not re-derive anything -- it reads this field.
    st.why   = st.set ? "explicit env override" : level_name;
    return st;
}

// ONE ROW AT A TIME. A row is resolved on the first read of THAT row, never because some other row
// was read. The table used to be resolved whole on the first read of any row, which forced every
// row to settle at the moment the first flash-attention route was probed -- fine while a row was a
// function of the environment and the config level alone, wrong for a row that is a function of
// the card set or the model (neither is known that early, and a value frozen before they are
// registered is a value that is wrong for the rest of the process).
//
// HOW LONG A RESOLVED ROW STAYS RESOLVED is a property of the row, and it is the property the
// resolver it replaced had -- not a tidy-up. A rule that reads only the environment and the config
// level is final the first time it is read. A rule that also reads the model profile must be read
// again when the profile changes (the loader registers it, possibly after CUDA init). A rule that was
// never cached is a cheap function of live state and stays one.
enum pxa_lever_cache {
    PXA_CACHE_ONCE,          // resolved on the first read, final
    PXA_CACHE_PER_PROFILE,   // resolved again whenever the model-profile generation changes
    PXA_CACHE_LIVE,          // computed from the live environment / card set / model on every read
};

static pxa_lever_cache pxa_lever_cache_of(int id) {
    switch ((pxa_lever_id) id) {
        case PXA_LEVER_ROUTER_FUSE:
        case PXA_LEVER_VOLTA_CUBLAS_NE11:
        case PXA_LEVER_PXQ_GEMM_2D:
            return PXA_CACHE_PER_PROFILE;
        case PXA_LEVER_PXQ_INT8_PREFILL:
        case PXA_LEVER_SPEC_RELAXED:
        case PXA_LEVER_FA_GQA_PACK:
        case PXA_LEVER_MOE_DEVICE_MAP:
        case PXA_LEVER_FUSE_DELTANET:
        case PXA_LEVER_PXQ_MMVQ:
        case PXA_LEVER_ENHANCE_DBG:
            return PXA_CACHE_LIVE;
        default:
            return PXA_CACHE_ONCE;
    }
}

static pxa_lever_state       g_pxa_lever_state[PXA_LEVER_COUNT];
static std::once_flag        g_pxa_lever_once [PXA_LEVER_COUNT];
static std::atomic<bool>     g_pxa_lever_ready[PXA_LEVER_COUNT];

// A per-profile row is published as an immutable record tagged with the generation it was resolved
// at, so a reader never sees a value torn against its reason.
struct pxa_lever_gen_state {
    pxa_lever_state st;
    int             gen;
};
static std::atomic<const pxa_lever_gen_state *> g_pxa_lever_gen[PXA_LEVER_COUNT];
static std::mutex                               g_pxa_lever_gen_mu;

static pxa_lever_state pxa_lever_row(int id) {
    switch (pxa_lever_cache_of(id)) {
        case PXA_CACHE_PER_PROFILE: {
            const int gen = ggml_pxa_model_profile_generation();
            const pxa_lever_gen_state * cur = g_pxa_lever_gen[id].load(std::memory_order_acquire);
            if (cur && cur->gen == gen) {
                return cur->st;
            }
            std::lock_guard<std::mutex> lock(g_pxa_lever_gen_mu);
            cur = g_pxa_lever_gen[id].load(std::memory_order_acquire);
            if (cur && cur->gen == gen) {
                return cur->st;
            }
            // (a superseded record is not freed: a reader may still hold it, and there is one per
            // profile registration -- a handful per process)
            const pxa_lever_gen_state * next = new pxa_lever_gen_state{ pxa_lever_resolve(id), gen };
            g_pxa_lever_gen[id].store(next, std::memory_order_release);
            return next->st;
        }
        case PXA_CACHE_LIVE:
            return pxa_lever_resolve(id);
        case PXA_CACHE_ONCE:
        default:
            if (!g_pxa_lever_ready[id].load(std::memory_order_acquire)) {
                std::call_once(g_pxa_lever_once[id], [id] {
                    g_pxa_lever_state[id] = pxa_lever_resolve(id);
                    g_pxa_lever_ready[id].store(true, std::memory_order_release);
                });
            }
            return g_pxa_lever_state[id];
    }
}

int64_t pxa_lever(pxa_lever_id id) {
    if (id < 0 || id >= PXA_LEVER_COUNT) {
        return 0;
    }
    return pxa_lever_row(id).value;
}

// Whether the variable is set is a fact about the environment alone, so a row that is recomputed on
// every read need not be recomputed to be asked.
static bool pxa_lever_env_is_set(int id) {
    const char * v = getenv(g_pxa_lever_rows[id].env);
    return id >= PXA_LEVER_REPORT_END ? v != nullptr : (v && v[0] != '\0');
}

bool pxa_lever_set_by_user(pxa_lever_id id) {
    if (id < 0 || id >= PXA_LEVER_COUNT) {
        return false;
    }
    if (pxa_lever_cache_of(id) == PXA_CACHE_LIVE) {
        return pxa_lever_env_is_set(id);
    }
    return pxa_lever_row(id).set;
}

const char * pxa_lever_why(pxa_lever_id id) {
    if (id < 0 || id >= PXA_LEVER_COUNT) {
        return "?";
    }
    if (pxa_lever_cache_of(id) == PXA_CACHE_LIVE) {
        return pxa_lever_env_is_set(id) ? "explicit env override" : ggml_pxa_config_level_name();
    }
    return pxa_lever_row(id).why;
}

int64_t pxa_lever_default(pxa_lever_id id) {
    if (id < 0 || id >= PXA_LEVER_COUNT) {
        return 0;
    }
    return pxa_lever_rule_default(id);
}

const char * pxa_lever_env_text(pxa_lever_id id) {
    if (id < 0 || id >= PXA_LEVER_COUNT) {
        return nullptr;
    }
    return getenv(g_pxa_lever_rows[id].env);
}

const char * pxa_core_level_why(void) {
    const char * r = getenv("PXA_REFERENCE");
    if (r && atoi(r) != 0) {
        return "PXA_REFERENCE=1";
    }
    const char * e = getenv("PXA_ENHANCE");
    if (e) {
        return atoi(e) == 0 ? "PXA_ENHANCE=0" : "PXA_ENHANCE=1";
    }
    return "default";
}

void pxa_core_lever_report(void) {
    static std::atomic<bool> told{false};
    if (told.exchange(true)) {
        return;
    }
    const bool verbose = pxa_env_on_if_one(getenv("PXA_CORE_LEVERS"));
    // The report is the one reader that asks for every row, so it is the one place the whole
    // table is resolved at once -- exactly as the first read of any row used to do it. Resolving
    // the rows before printing the first line keeps a bad value's warning above the report, as it
    // was; a row an earlier caller already resolved is not resolved again.
    for (int i = 0; i < PXA_LEVER_REPORT_END; ++i) {
        (void) pxa_lever_row(i);
    }
    for (int i = 0; i < PXA_LEVER_REPORT_END; ++i) {
        const pxa_lever_state st = pxa_lever_row(i);
        fprintf(stderr, "PXA_AUTO: %s=%lld (%s; override %s)\n",
                g_pxa_lever_rows[i].report_name, (long long) st.value,
                st.why, g_pxa_lever_rows[i].env);
        if (verbose) {
            fprintf(stderr, "PXA_AUTO:   %s\n", g_pxa_lever_rows[i].doc);
        }
    }
}

void pxa_core_lever_check_unknown(void) {
    // Gated until the registry is complete: today most PXA_* levers are still read by their own
    // getenv, so an ungated warning would name dozens of variables that work perfectly well.
    // PXA_CORE_LEVERS=1 turns it on for the levers that HAVE moved.
    if (!pxa_env_on_if_one(getenv("PXA_CORE_LEVERS"))) {
        return;
    }
    extern char ** environ;
    for (char ** e = environ; e && *e; ++e) {
        if (strncmp(*e, "PXA_", 4) != 0 && strncmp(*e, "PXQ_", 4) != 0) { // the registry declares PXQ_ rows too
            continue;
        }
        const char * eq = strchr(*e, '=');
        if (!eq) {
            continue;
        }
        const size_t n = (size_t) (eq - *e);
        bool known = false;
        for (int i = 0; i < PXA_LEVER_COUNT && !known; ++i) {
            known = strlen(g_pxa_lever_rows[i].env) == n && strncmp(g_pxa_lever_rows[i].env, *e, n) == 0;
        }
        if (!known) {
            fprintf(stderr, "PXA_CORE_LEVERS: %.*s is not a lever the core declares (it may still be read "
                            "by its own site; the registry is being filled in step by step)\n", (int) n, *e);
        }
    }
}
