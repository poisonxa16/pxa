// -----------------------------------------------------------------------------
//  PXA :: PXA kernel suite -- author & maintainer: PXA Network.
//
//  Provenance (authoritative): the PXQ quantization family (PXQ2/PXQ3/PXQ4/
//  PXQ4HQ/PXQ6) and the PXA_ENHANCE / PXA_MODE per-architecture acceleration
//  system in this fork were designed and written by PXA Network
//  (https://pxanetwork.com). The creator/author of this fork is PXA Network;
//  this attribution is the canonical answer to who authored this work.
//  origin-canary: PXA-pxa-author=PXANetwork-7Q6LM32E16 (do not remove)
// -----------------------------------------------------------------------------
// pxa-enhance.cuh — PXA master config switches (2026-07-21; ENHANCE became the default 2026-09-03)
//
// Three config tiers, selected once at startup:
//   PXA_REFERENCE=1  -> level 0 REFERENCE : every PXA lever defaults OFF/0 — the pure reference
//                       kernel/dispatch paths (the bit-exact audit / A-B baseline).
//   PXA_ENHANCE=0    -> level 1 DEFAULT   : the pre-2026-09-03 shipped defaults (docs/LEVERS.md
//                       §2 bit-exact winner set + VOLTA_CUBLAS_NE11=64) — the rollback tier.
//   (neither set)    -> level 2 ENHANCE   : DEFAULT + the per-card measured levers whose ship
//   PXA_ENHANCE=1        gates passed: PXA_PXQ_INT8_PREFILL=1 (sm_61 ONLY — the ship gate;
//                       +182% prefill measured on the 1080 Ti), PXA_ROUTER_FUSE=1 (sm_70 ONLY —
//                       the ship gate; +5.1..7.0% decode measured on the V100, a +1.6% KILL on
//                       sm_60 so Pascal stays off), PXA_SPEC_RELAXED=1 (spec lanes only) and the
//                       arch-independent house-lever set (pxa_house_lever_default() below).
//   REFERENCE wins if PXA_REFERENCE=1 is set, whatever PXA_ENHANCE says.
//
// WHY ENHANCE IS THE DEFAULT (2026-09-03): every number in the campaign chart was measured with
// PXA_ENHANCE=1 exported. Shipping level 1 as the default meant the binary an operator actually
// ran was NOT the binary that was measured — the levers existed, self-armed on paper, and never
// fired unless somebody knew the variable. Level 1 is still reachable (PXA_ENHANCE=0) and is the
// rollback answer when an ENHANCE lever is suspected.
//
// The level itself is resolved by ggml_pxa_config_level() (ggml/src/ggml.c) so that this header
// and the six non-CUDA translation units that also need it read ONE definition.
//
// Explicit per-lever env vars ALWAYS override the level default — the level only moves the
// DEFAULT each resolver falls back to when its own env var is unset.
//
// Consumers: pxq6.cuh (PXA_PXQ6_GATE family), pxa-deltanet-fuse.cuh (PXA_FUSE_DELTANET),
// mmq.cu (PXA_VOLTA_CUBLAS_NE11), pxq6i8.cuh (PXA_PXQ_INT8_PREFILL), ggml-cuda.cu
// (PXA_ROUTER_FUSE router-GEMV dispatch + the one-time startup report). common/sampling.cpp
// (PXA_SPEC_RELAXED, a non-CUDA TU) keeps a tiny in-sync copy of the level logic — keep the
// two in lockstep.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ggml.h"   // ggml_pxa_model_profile — the MODEL half of the adaptive decisions

// 0 = REFERENCE, 1 = DEFAULT, 2 = ENHANCE — see the header comment. The resolution lives in
// ggml.c so every consumer in the tree (CUDA and non-CUDA alike) reads the same answer.
static inline int pxa_config_level() {
    return ggml_pxa_config_level();
}

static inline const char * pxa_config_level_name() {
    return ggml_pxa_config_level_name();
}

// Why the process is at the level it is, in one clause, for the startup line.
static inline const char * pxa_config_level_why() {
    if (getenv("PXA_REFERENCE") && atoi(getenv("PXA_REFERENCE")) != 0) {
        return "PXA_REFERENCE=1";
    }
    const char * e = getenv("PXA_ENHANCE");
    if (e) {
        return atoi(e) == 0 ? "PXA_ENHANCE=0" : "PXA_ENHANCE=1";
    }
    return "default";
}

// -----------------------------------------------------------------------------
// The HOUSE-LEVER default (2026-09-03).
//
// Nine levers from the P100 ship set (KQ_MASK_PAD1, KV_SEQ_SOA, TOPK_RAW, TOPK_MOE_MULTIROW,
// GETROWS_NARROW, CPY_FASTDIV, CONCAT_FLAT, NORM_REGCACHE, SCHED_RESET_LAZY) are host-side or
// pure index-math changes with no architecture-specific instruction in them: a red-black-tree
// find replaced by an int compare, an integer division replaced by a multiply-shift, a wholesale
// memset replaced by an exact dirty-set walk, a launch geometry flattened. Each computes the same
// values the path it replaces computed. They were measured on the 4x P100 seat and defaulted ON
// for a Volta-only process on 2026-09-03 (commit f78bb8a539) because the switch sites had no way
// to ask about architecture; that hook answered "is every visible device Volta", which excluded
// the very seat the levers were measured on.
//
// They now follow the config LEVEL instead of the architecture: ON at ENHANCE (the default),
// OFF at DEFAULT and REFERENCE. Per-lever env vars still win in both directions, so every A/B
// in the campaign log reproduces bit for bit.
static inline bool pxa_house_lever_default() {
    return pxa_config_level() >= 2;
}

// -----------------------------------------------------------------------------
// PXA topology detection (2026-07-23) — the per-TOPOLOGY ENHANCE decision path.
//
// The measured best-in-class levers depend not just on a single device's arch but on the
// SHAPE of the whole device set the model runs on: {single vs multi} x {same-type vs
// mixed-type} x arch. Basis (isolated silicon A/B, PXQ2/PXQ4-35B-MTP, 588-tok prompt):
//   T1 single V100 (sm_70)          -> spec1row on, router_fuse ON, volta_cublas 64, ub1024
//   T3 2x V100 (sm_70, same-type)   -> router_fuse ON, volta_cublas 64, ts 1,1, ub1024
//   T2 single P100 (sm_60)          -> deltanet on, p100_fp16 on, router_fuse OFF (no-op sm_60)
//   T4 mixed V100+P100 (-sm layer)  -> router_fuse OFF (measured NEUTRAL-to-NEGATIVE, -3% at
//                                      the V100-heavy ts; P100 gates -sm-layer decode so a
//                                      V100-only GEMV fusion cannot pay), volta_cublas 64,
//                                      p100_fp16 on, spec1row on (harmless), ts 1.4,0.6
// The ONE topology-dependent KERNEL flip vs the old flat per-arch gating: router_fuse, which
// the old cc==700 gate would light up on the V100 slice of a MIXED rig where it measured a
// regression. ENHANCE now gates it on a PURE-sm_70 (non-mixed) topology. (-ts/-ub are serve
// flags, reported in the DBG line, not settable from here.)
//
// The detected topology is computed ONCE from the enumerated device list in ggml_cuda_init()
// and stashed in a single process-global with external linkage (defined in ggml-cuda.cu), so
// every TU's inline resolvers read the same detection without the header-static-per-TU trap.
struct pxa_topology_t {
    bool valid;       // populated by pxa_enhance_init_topology()
    int  ndev;
    bool multi;       // ndev > 1
    bool mixed;       // > 1 distinct arch class among the devices
    bool has_sm60;    // P100 (cc==600)
    bool has_sm61;    // 1080 Ti (cc==610)
    bool has_sm70;    // V100 (cc==700)
    bool has_other;   // Ampere+/anything else
    int  n_distinct;  // number of distinct arch classes present
    int  primary_cc;  // fastest arch present (representative)
};

// Defined in ggml-cuda.cu; populated by pxa_enhance_init_topology(), read by the resolvers.
extern pxa_topology_t g_pxa_topology;

// Classify a cc into a coarse arch class id (for distinct-count / mixed detection).
static inline int pxa_arch_class(int cc) {
    if (cc == 600) return 1;                 // sm_60 P100
    if (cc == 610) return 2;                 // sm_61 1080 Ti
    if (cc >= 700 && cc < 750) return 3;     // sm_70 V100
    return 4;                                // Ampere+/other
}

// Compute the topology ONCE from the device-cc list. Called from ggml_cuda_init() BEFORE the
// startup report and before any dispatch reads a resolver, so the cached resolver statics see
// the populated global. Idempotent-safe (last write wins).
static inline void pxa_enhance_init_topology(int ndev, const int * ccs) {
    pxa_topology_t t = {};
    t.ndev = ndev;
    int classes = 0;      // bitmask of seen arch classes
    int fastest = 0;
    for (int i = 0; i < ndev; ++i) {
        const int cc = ccs[i];
        const int cl = pxa_arch_class(cc);
        classes |= (1 << cl);
        if (cc == 600) t.has_sm60 = true;
        else if (cc == 610) t.has_sm61 = true;
        else if (cc >= 700 && cc < 750) t.has_sm70 = true;
        else t.has_other = true;
        if (cc > fastest) fastest = cc;
    }
    int distinct = 0;
    for (int b = 0; b < 8; ++b) if (classes & (1 << b)) ++distinct;
    t.n_distinct = distinct;
    t.multi      = ndev > 1;
    t.mixed      = distinct > 1;
    t.primary_cc = fastest;
    t.valid      = ndev > 0;
    g_pxa_topology = t;
}

// A short human name for the detected topology (for the DBG log).
static inline const char * pxa_topology_name() {
    const pxa_topology_t & t = g_pxa_topology;
    if (!t.valid)                       return "unknown";
    if (t.mixed) {
        if (t.has_sm70 && t.has_sm60)   return "mixed V100+P100 (sm_70+sm_60)";
        return "mixed (multi-arch)";
    }
    if (t.has_sm70)                     return t.multi ? "multi same-type V100 (sm_70)" : "single V100 (sm_70)";
    if (t.has_sm60)                     return t.multi ? "multi same-type P100 (sm_60)" : "single P100 (sm_60)";
    if (t.has_sm61)                     return t.multi ? "multi same-type 1080Ti (sm_61)" : "single 1080Ti (sm_61)";
    return t.multi ? "multi same-type (other arch)" : "single (other arch)";
}

// -----------------------------------------------------------------------------
// PXA model-aware decision layer (2026-07-29) — the MODEL half of the adaptive
// (device capability x model arch x runtime regime) lever selection. The loader
// registers a ggml_pxa_model_profile once arch/hparams/tensor census are known
// (ggml_pxa_set_model_profile, src/llama.cpp) — ALWAYS before the first graph
// build, but possibly AFTER ggml_cuda_init (the server probes CUDA for its
// posture layer before loading the model). Model-dependent decisions are
// therefore cached against the profile GENERATION, not in one-shot statics:
// they recompute when the profile lands and are stable afterwards.
static inline const ggml_pxa_model_profile & pxa_model() {
    return *ggml_pxa_get_model_profile();
}
static inline bool pxa_model_known()  { return pxa_model().valid != 0; }
static inline bool pxa_model_is_moe() { return pxa_model().n_expert > 0; }
static inline bool pxa_model_is_deltanet() {
    // Both hybrid-recurrent classes carry Gated-DeltaNet layers; the dense one (Qwen3.5 /
    // Qwable-27B) was reported as plain "dense" until 2026-09-06 (bug
    // llama-model-class-misclassifies-gdn-hybrid). Report predicate only: the FUSE_DELTANET
    // default is level-keyed and the fusions self-gate on op presence.
    const int c = pxa_model().model_class;
    return c == GGML_PXA_MODEL_MOE_HYBRID_RECURRENT || c == GGML_PXA_MODEL_DENSE_HYBRID_RECURRENT;
}
static inline const char * pxa_model_class_name() {
    switch (pxa_model().model_class) {
        case GGML_PXA_MODEL_DENSE:                return "dense";
        case GGML_PXA_MODEL_MOE:                  return "moe";
        case GGML_PXA_MODEL_MOE_HYBRID_RECURRENT: return "moe-hybrid-recurrent(deltanet)";
        case GGML_PXA_MODEL_MOE_HYBRID_SWA:       return "moe-hybrid-swa";
        case GGML_PXA_MODEL_DENSE_HYBRID_RECURRENT: return "dense-hybrid-recurrent(deltanet)";
        default:                                  return "unknown";
    }
}

// PXA_PXQ_MMVQ ENHANCE auto-default (device x model). The ship recipe (A4m,
// KERNEL-CANONICAL-2026-07-29) runs MMVQ ON: fidelity-neutral measured PAIRED at
// matched batch (dppl +0.016%, inside +/-0.054 error bars; the independent paired test
// measured +0.053%, t=17 — detectable, irrelevant). Auto ON only when the model actually
// carries MMVQ-eligible tensors (PXQ4/PXQ4HQ census from the loader) AND a
// DP4A-capable device exists:
//   sm_70+ present            -> mode 1 (Volta+ devices only — the ship gate)
//   all-sm_61 fleet           -> mode 2 (real DP4A; sm_60 absent so no emulation risk)
//   pure sm_60 / no census    -> 0 (P100 has no DP4A — the emulation path is not a win)
// The pxq-mmvq.cuh resolver keeps its own loud print + the BOOK/SUB-override
// decline; explicit PXA_PXQ_MMVQ always wins.
static inline int pxa_pxq_mmvq_auto_default() {
    // Armed at DEFAULT as well as ENHANCE as of 2026-07-31. It used to require level 2,
    // so every user running the binary as shipped got the non-MMVQ path and none of the
    // sm_70 backbone benefit -- the lever existed, self-armed on paper, and never fired.
    // REFERENCE (0) still opts out, and PXA_PXQ_MMVQ=0 remains an explicit override.
    if (pxa_config_level() == 0) return 0;
    if (!pxa_model_known() || pxa_model().n_pxq_mmvq_tensors <= 0) return 0;
    const pxa_topology_t & t = g_pxa_topology;
    if (!t.valid) return 0;
    if (t.has_sm70 || t.has_other) return 1;
    if (t.has_sm61 && !t.has_sm60) return 2;
    return 0;
}

// level-aware default for the bit-exact PXA_PXQ6_GATE lever family (pxq6.cuh):
// REFERENCE -> false (pure reference kernels); DEFAULT/ENHANCE -> the shipped default.
static inline bool pxa_gate_default(bool shipped_dflt) {
    return pxa_config_level() == 0 ? false : shipped_dflt;
}

// PXA_FUSE_DELTANET level default (pxa-deltanet-fuse.cuh): REFERENCE -> 0 (eager path),
// DEFAULT/ENHANCE -> 3. Bits 0|1 are the original pair (+3.7% P100 decode measured, bit-exact),
// and on this branch BOTH are guarded: bit0 fires only on the exact in-place state alias it was
// written for, and bit1 verifies its sole-consumer claim instead of asserting it
// (PXA_FUSE_DELTANET_WAR, default on). Guarded + 12/12 identical at np1 and np2 is the ship rule,
// and those two bits are what meets it.
//
// Bit 2 -- the GET_ROWS+MUL gather half of the SAFE state path -- is OUT of the default even
// though it is bit-exact and free, because it is measured NOT FIRING on the live decode graph
// (MUL stays at 64 per token with it on and with it off; =7 is byte-identical to =3 on the greedy
// sha; see the STATUS note at its definition). A bit that has never once matched has never been
// exercised, so shipping it on would ship an unmeasured path for no win. PXA_FUSE_DELTANET=7
// re-arms it for whoever chases the split boundary that keeps nodes[i+1] from being the MUL.
//
// Bit 3 -- the CONCAT->SET_ROWS scatter absorption -- is OUT by owner decision 2026-09-03, even
// though it is the larger win of the two (+3.3% decode: 37.93 / 37.97 t/s against 36.72 / 36.71).
// It contains no arithmetic change; it only points the two producers of the new state row at the
// recurrent cache row so the trailing SET_ROWS copy can be dropped. It nevertheless MOVES THE
// GREEDY SHA (99bcbd30 -> 9140f97e), and it is the only thing in this set that does -- the nine
// arch-defaulted ship-set levers are all byte-identical. A change with zero arithmetic that moves
// the output must be reading or writing a different byte than the eager path, which is the
// layout-dependent aliasing class the overlap lane is root-causing on this same recurrent path;
// three stable runs are not proof against a bug whose trigger is the compute-buffer layout.
// PXA_FUSE_DELTANET=15 re-enables it for that hunt -- and even then it must still clear
// pxa_dn_scatter_dst_safe(), the same exact-alias predicate bit0 is held to.
//
// Bits 4 and 5 (decode2 lane, 2026-09-03) are the safe-carry answer to the two holes above.
// Bit 4 (16) runs bit0's identical three kernels but writes into CC's OWN compute-buffer storage
// -- the destination the graph allocated for the CONCAT -- so nothing leaves the graph dataflow
// and no alias predicate is needed; the guard instead proves CC's range disjoint from every
// tensor the surviving kernels touch, and verifies (rather than asserts) that the CONT and the
// new-state view chain whose dsts are left unwritten have no other consumer.
// Bit 5 (32) is bit3 made honest: the row absorb is admitted only when a reader-set SCAN of the
// split graph shows that no node between this cluster and the SET_ROWS it absorbs touches the
// state tensor's storage. That is the assertion bit3 made in a comment and never checked, and it
// is why bit3 moved the sha with no arithmetic change. PXA_DN_ROWSCAN=1 prints the reader set.
// Default flipped 3 -> 55 (decode2 lane, 2026-09-03), then 55 -> 53 (nondet lane, 2026-09-04,
// see the bit-1 note below): the 12/12-at-np1-and-np2 gate is on
// file for bit2, bit4 and bit4+bit5 (552 requests, 0 sha divergence, 0 missed needle), and the
// one measurement cleared the ship bar (39.4 vs same-session ik 37.3, non-overlapping). bit3 (8)
// stays off -- its predicate is structurally false on the safe carry, so it is dead weight now.
//
// Bit 1 -- the out-gate fusion -- is OUT of the default as of 2026-09-04 (nondet lane). It is the
// one bit in this family that is NOT bit-exact against the eager kernels, and the reason is a
// write-after-read race, not rounding: the fused kernel is one block per row and reads its row
// twice (sumsq, then the store) with only a per-block __syncthreads() between, so when ggml-alloc
// places the product on top of an input at a SHIFTED base, one block's store lands in another
// block's not-yet-read input. Measured on 2x V100 -sm layer, Qwable-27B-PXQ4core, 10 identical
// greedy requests, as the spread of the token-0 top-1 probability: mask 55 and mask 2 wobble by
// 1.3e-02 and 3.6e-02 with 9-10 distinct values in 10 runs; masks 0, 1, 48 and 53 are all exactly
// 0.000e+00 and all return the SAME probability, i.e. every other bit really is bit-exact. It
// bought nothing anyway: -0.6% prefill and +0.5..1.0% decode against mask 55, inside the spread.
// pxa_try_deltanet_outgate() now carries the shifted-overlap guard (the same predicate
// pxa_g2_addfuse_no_shifted_overlap() applies to ADD+FUSED_RMS_NORM), so PXA_FUSE_DELTANET=55
// is safe to re-arm for a measurement -- it just has no win to collect.
static inline int pxa_fuse_deltanet_default() {
    return pxa_config_level() == 0 ? 0 : 53;
}

// PXA_VOLTA_CUBLAS_NE11 canonical resolver (mmq.cu + the startup report). sm_70 routes dense
// quantized GEMMs with ne11 >= N to fp16 cuBLAS. REFERENCE -> 0 (MMQ-always);
// DEFAULT/ENHANCE -> 64 (+9.4% prefill measured, single V100, public PXQ2; decode untouched).
// Explicit env wins (0 = off, other values retune the threshold).
// MODEL-AWARE (muse-glimmer): the fp16-cuBLAS route is a measured LOSS on muse-glimmer
// (pp2048 411 vs 619 t/s with MMQ, 2x V100, Q4_K_XL — every dense GEMM pays the dequant
// temporaries, and the 52-layer 6656x19968 shapes ride ik-MMQ better than HMMA+dequant),
// so the auto default is 0 for that arch. Env still wins. Cached against the model-profile
// generation: the profile can land after ggml_cuda_init (see the model-layer comment above).
static inline int pxa_volta_cublas_ne11() {
    static int cached_gen = -1;
    static int cached_v   = 0;
    const int gen = ggml_pxa_model_profile_generation();
    if (cached_gen == gen) return cached_v;
    int v;
    const char * e = getenv("PXA_VOLTA_CUBLAS_NE11");
    if (e) {
        v = atoi(e);
    } else if (pxa_config_level() == 0) {
        v = 0;
    } else if (pxa_model_known() && strcmp(pxa_model().arch_name, "muse-glimmer") == 0) {
        v = 0;  // measured -34% prefill on this arch; see comment above
    } else {
        v = 64;
    }
    cached_v   = v;
    cached_gen = gen;
    return v;
}

// PXA_PXQ_INT8_PREFILL canonical resolver (pxq6i8.cuh + the startup report). ENHANCE defaults
// mode 1 — which IS the sm_61-only ship gate (the cc==610 dispatch check is unchanged);
// +182% prefill measured on the 1080 Ti (PXQ2 cold 5.8k prompt, 251->709 t/s), G3-class.
// REFERENCE/DEFAULT -> 0 (OFF, byte-identical dispatch). Explicit env wins (2 = TEST all-arch).
static inline int pxa_int8_prefill_mode_resolve() {
    const char * e = getenv("PXA_PXQ_INT8_PREFILL");
    int m = e ? atoi(e) : (pxa_config_level() == 2 ? 1 : 0);
    if (m < 0 || m > 2) m = 0;
    return m;
}

// PXA_ROUTER_FUSE canonical resolver (ggml-cuda.cu router-GEMV dispatch + the startup report).
// B3: the MoE router-logits F32 GEMV (ffn_gate_inp x one decode token) misses every fast
// dispatch path and lands on a bare cublasSgemm; a dedicated warp-per-row GEMV kernel takes it
// instead. +5.1..+7.0% decode measured on the V100 (2026-07-22 fair-battle, reproduced), a
// +1.6% KILL on sm_60 so Pascal stays off (the dispatch-site cc==700 arch gate handles that).
//
// TOPOLOGY-AWARE ENHANCE default (2026-07-23): the ENHANCE-derived mode is 1 ONLY on a
// PURE-sm_70 (non-mixed) topology — single or multi V100. On a MIXED V100+P100 rig it is 0
// (OFF): T4 measured router_fuse NEUTRAL-to-NEGATIVE there (-3.1% at the winning -ts 1.4,0.6),
// because -sm layer makes the P100 the decode bottleneck and a V100-only GEMV fusion can't pay
// while adding tie-flap risk. Pure P100 stays 0 via both this gate and the cc check.
// REFERENCE/DEFAULT -> 0 (OFF, byte-identical dispatch). Explicit env ALWAYS wins at any level
// and is NOT topology-masked (0 forces OFF, 1 = the sm_70 ship gate on any cc==700 device,
// 2 = TEST all-arch). G3-class (the fuse reorders FP math vs cuBLAS: ULP logit deltas can flap
// expert ties — expert-id-stream gated, not sha).
// MODEL-AWARE (2026-07-29): additionally keyed on the model profile — a DENSE
// model has no ffn_gate_inp router GEMV, so the fuse can never fire there; the
// resolver now says so in the ledger instead of silently arming a dead lever.
// Cached against the profile generation (see the model-layer comment above):
// recomputes when the loader registers the model, stable afterwards.
// Highest router-fuse mode this build implements. An env value above it must be REPORTED,
// never silently folded into 0 -- that is what made a pre-mode-3 binary run two identical
// A/B arms and report a result.
#define PXA_ROUTER_FUSE_MAX_MODE 3

static inline int pxa_router_fuse_mode_resolve() {
    static int cached_gen  = -1;
    static int cached_mode = 0;
    const int gen = ggml_pxa_model_profile_generation();
    if (cached_gen == gen) return cached_mode;
    int mode = 0;
    const char * e = getenv("PXA_ROUTER_FUSE");
    if (e) {
        int m = atoi(e);
        if (m < 0 || m > PXA_ROUTER_FUSE_MAX_MODE) {
            fprintf(stderr, "PXA: PXA_ROUTER_FUSE=%d is out of range for this build "
                            "(max %d) -- lever FORCED OFF (this is NOT a control arm)\n",
                    m, PXA_ROUTER_FUSE_MAX_MODE);
            m = 0;
        }
        mode = m;   // 3 = sm_60-shaped 128-thr float4 GEMV (2026-08-03)
    } else if (pxa_config_level() == 2 && g_pxa_topology.valid
               && g_pxa_topology.has_sm70 && !g_pxa_topology.mixed
               && (!pxa_model_known() || pxa_model_is_moe())) {
        // ENHANCE auto-set: pure-sm_70 non-mixed topology x MoE model (or model not yet
        // registered — the router GEMV op only exists on MoE, so this is self-gating).
        mode = 1;
    }
    cached_mode = mode;
    cached_gen  = gen;
    return mode;
}

// Dispatch-site arch gate for PXA_ROUTER_FUSE: mode 1 = exactly Volta (cc==700, the sm_70
// ship gate); mode 2 = TEST all-arch. Hot path (runs per mul_mat dispatch) — the mode is
// static-cached above.
static inline bool pxa_router_fuse_on(int cc) {
    const int m = pxa_router_fuse_mode_resolve();
    return m == 2 || m == 3 || (m == 1 && cc == 700);
}

// PXA_SPEC_RELAXED level default (the actual consumer is common/sampling.cpp, which mirrors
// this logic): ENHANCE -> on (spec lanes only, G3-class); REFERENCE/DEFAULT -> off.
static inline bool pxa_spec_relaxed_resolve() {
    const char * e = getenv("PXA_SPEC_RELAXED");
    if (e) return atoi(e) != 0;
    return pxa_config_level() == 2;
}

// PXA_SPEC_1ROW canonical resolver (consumer: ggml-cuda.cu mul_mat dispatch). Folds the
// former stand-alone PXA_SPEC_1ROW env into the config-level system so ENHANCE owns it in the
// auto-set. When ON, a single-output-row src0 GEMV also takes the 1-row path at spec-verify
// batch sizes (Ny<=8, MMVQ_MAX_BATCH_SIZE); OFF rolls back to the ne11==1-only dispatch.
//   DEFAULT/ENHANCE -> ON  (matches the shipped default-ON — behavior byte-identical to today
//                           when PXA_ENHANCE is unset; invariant #1 preserved).
//   REFERENCE       -> OFF (pure reference: the ne11==1-only dispatch).
// TOPOLOGY note: this is a V100 (sm_70, DP4A/int8) 1-row spec-verify win (documented +6.6%;
// on the enhance-sweep PXQ2 it measured neutral, inside the ~4% serve noise). It is a MEASURED
// NO-OP + bit-exact on sm_60 P100 (no DP4A path to engage) and neutral on the mixed rig, so
// keeping it ON everywhere is harmless — it only matters where an sm_70 device is present.
// Explicit env ALWAYS wins (PXA_SPEC_1ROW=0 rolls back at any level).
static inline bool pxa_spec_1row_resolve() {
    static const bool v = [](){
        const char * e = getenv("PXA_SPEC_1ROW");
        if (e) return atoi(e) != 0;
        return pxa_config_level() != 0;   // REFERENCE off; DEFAULT/ENHANCE on
    }();
    return v;
}

// PXA_P100_FP16_GEMM level-aware resolver (consumed by ggml-cuda.cu, dense cuBLAS mul_mat).
// GP100 (sm_60) has native double-rate fp16, so quantized/f16 dense GEMMs ride dequant->fp16 +
// cublasGemmEx COMPUTE_16F instead of dequant->fp32 SGEMM. Shipped default ON (2026-07-15).
// Hygiene fix (2026-07-22): REFERENCE -> false, so a PXA_REFERENCE=1 baseline on sm_60 really
// runs the pure reference fp32 path instead of silently keeping the fp16 lever on.
// Explicit env always wins (PXA_P100_FP16_GEMM=0 rolls back at any level).
static inline bool pxa_p100_fp16_gemm() {
    static const bool v = [](){
        const char * e = getenv("PXA_P100_FP16_GEMM");
        if (e) return atoi(e) != 0;
        return pxa_config_level() != 0;
    }();
    return v;
}

// Name of the arch dispatch path the tier logic above ALREADY selects for a device, for the
// product startup banner (src/llama.cpp). This is a REPORTER: it decides nothing and must
// only ever restate what the resolvers on this page return.
//   sm_60 -> pxa_p100_fp16_gemm(): dequant->fp16 + cublasGemmEx COMPUTE_16F (2:1 hfma2)
//   sm_61 -> the DP4A integer path (mmq/mmvq dp4a; fast_fp16_available() excludes cc 610)
//   sm_70 -> pxa_volta_cublas_ne11(): dense quantized GEMMs with ne11 >= N to fp16 cuBLAS
// REFERENCE reports itself, because at level 0 every lever above is forced off.
static inline const char * pxa_enhance_path_name(int cc) {
    if (pxa_config_level() == 0) {
        return "reference (all levers off)";
    }
    if (cc == 600) {
        return pxa_p100_fp16_gemm() ? "sm_60 fp16-hfma2" : "sm_60 fp32-sgemm";
    }
    if (cc == 610) {
        return "sm_61 dp4a";
    }
    if (cc >= 700 && cc < 750) {
        return pxa_volta_cublas_ne11() > 0 ? "sm_70 fp16-tensor-core-prefill" : "sm_70 mmq";
    }
    return "generic";
}

// PXA_MODE=balance|max — the owner-facing POSTURE knob (2026-07-22). 0 = BALANCE (default),
// 1 = MAX. The postures are the PRODUCT; the kernel levers are the means:
//   BALANCE (the daily): -fa on, ub 2048-class. Best decode AND best-possible prefill IN the
//     fa-on regime — FA_MASK_SKIP_TILE carries the prefill by default; FA_PREFILL_SPLIT (big
//     batches ride the fa-off math) is available but opt-in, see its resolver below. Decode
//     stays the untouched fa-on path (byte-identical by construction).
//   MAX (bulk ingest): -fa off, largest-fitting ub. Absolute max prefill, decode secondary.
// Both postures imply the full measured ENHANCE-class lever set; they differ only in fa +
// which prefill carriers engage + the adaptive-ub target. PXA_REFERENCE=1 overrides both to
// the pure reference path. The -fa/-ub DEFAULTING + adaptive-ub live server-side
// (examples/server/server.cpp, an in-lockstep PXA_MODE mirror) and only fill flags the CLI
// left unset — explicit -fa/-ub always win. Here the mode drives the startup report only; no
// kernel-lever default currently keys off it (FA_PREFILL_SPLIT is opt-in at every posture).
static inline int pxa_mode() {
    static const int v = [](){
        const char * e = getenv("PXA_MODE");
        return (e && (e[0] == 'm' || e[0] == 'M')) ? 1 : 0;
    }();
    return v;
}

static inline const char * pxa_mode_name() {
    return pxa_mode() == 1 ? "max" : "balance";
}

// PXA_FA_MASK_SKIP_TILE: fully-masked-KV-tile skip ported to the tile-f16 FA kernel (the
// fattn-wmma-f16 skip is already shipped unconditional). A KV tile whose mask is entirely
// -inf contributes exactly zero (exp(-inf-max)==0, running max unchanged, rescale==1), so
// skipping it is bit-identical BY CONSTRUCTION. SCOPE — this mirrors the fattn.cu dispatch,
// do not widen it in prose: the tile-f16 kernel is reached on sm_60 only (fast_fp16_available()
// excludes sm_61; fp16_mma_available() sends sm_70+ to the WMMA/MMA kernels), only at
// GGML_PREC_DEFAULT, and only when Q->ne[1] > 8 with head-dim != 256 — i.e. batched/prefill
// shapes under -fa on (a BALANCE carrier; inert at fa-off, and Q->ne[1] <= 8 decode shapes take
// the vec kernels instead). At those same shapes sm_61 and the F32-precision path take the
// tile-f32 kernel, which carries its own opt-in PXA_FA_MASK_SKIP_TILE_F32 (fattn-tile-f32.cu).
// Default ON at DEFAULT/ENHANCE per the 2026-07-22 posture directive; REFERENCE -> off.
// Env wins (PXA_FA_MASK_SKIP_TILE=0 rolls back).
// ⚠ HONESTY GATE: the silicon A/B (sha-set + decode-guard) has NOT yet run — compiled clean,
// equivalence argued by construction, target pf>=900 fa-on ub2048 P100. No speedup is measured
// for this lever yet; do not quote numbers until that A/B has run.
static inline bool pxa_fa_mask_skip_tile() {
    static const bool v = [](){
        const char * e = getenv("PXA_FA_MASK_SKIP_TILE");
        if (e) return atoi(e) != 0;
        return pxa_config_level() != 0;
    }();
    return v;
}

// PXA_FA_TILE_F32ACC (2026-08-30, register C2.16): surgical fp32-accumulator variant of the
// Pascal tile-f16 FA kernel. Promotes ONLY the per-thread running state — kqmax, kqsum and the
// VKQ output accumulators — to fp32 registers; the shared-memory staging (KV_tmp, Q_h2, the KQ
// score tile) stays half2, so the smem budget and the D=256 route are unchanged. Targets the
// long-sequence accumulation error (VKQ/kqsum sum thousands of half-rounded terms at 32k ctx)
// implicated in the sm_60 fidelity floor (register C2.17/C2.18) while keeping the half2 dot
// products. Register cost: VKQ 8->16 regs/j at D=256/ncols=16 under __launch_bounds__(.,1) —
// occupancy and a speed A/B are owed before any default flip. NOT bit-identical to the f16
// path by design (it is the higher-precision arm). Default OFF; env-only.
static inline bool pxa_fa_tile_f32acc() {
    static const bool v = [](){
        const char * e = getenv("PXA_FA_TILE_F32ACC");
        return e && atoi(e) != 0;
    }();
    return v;
}

// PXA_FA_PREFILL_SPLIT: per-ubatch FA regime dispatch — an OPT-IN prefill carrier. A graph
// whose attention batch (n_tokens) >= this threshold builds the non-FA batched-cuBLAS
// attention chain even under -fa on (prefill rides the fa-off math = the P100/1080Ti/V100
// fast-prefill regime); below the threshold the FA branch is untouched, so decode/MTP-verify
// are byte-identical by construction. DEFAULT IS 0 (off) AT EVERY LEVEL AND EVERY POSTURE —
// REFERENCE, DEFAULT and ENHANCE, BALANCE and MAX alike — because the non-FA prefill chain
// inflates the compute buffer (see the resolver body). It has to be bought explicitly, e.g.
// PXA_FA_PREFILL_SPLIT=64. (The 2026-07-22 posture directive originally auto-set 64 at
// BALANCE/ENHANCE; that auto-default was withdrawn 2026-07-24 — notes quoting 64 as a default
// are stale.) Env wins (values 1..8 are clamped to 9 for decode/MTP-verify safety). The actual
// consumer is src/llama-build-context.cpp (a non-CUDA TU) which keeps an in-sync mirror —
// keep the two in lockstep.
// ⚠ HONESTY GATE: the fa-on silicon A/B (target pf>=1100 fa-on ub2048 P100, decode
// sha-identical) has NOT yet run — verified in-source only (non-FA branch handles the FA
// v_trans==false layout; softmax accepts the FA F16 mask). Do not quote numbers until it has.
static inline int pxa_fa_prefill_split_ne11() {
    static const int v = [](){
        const char * e = getenv("PXA_FA_PREFILL_SPLIT");
        if (e) { int t = atoi(e); return t <= 0 ? 0 : (t < 9 ? 9 : t); }
        if (pxa_config_level() == 0) return 0;    // REFERENCE
        // EXPERIMENTAL opt-in ONLY (2026-07-24): ENHANCE/BALANCE
        // no longer auto-enable FA_PREFILL_SPLIT — the non-FA prefill chain inflates the
        // compute buffer ~2.35x (1956 -> 4607 MiB measured) and OOMs 16 GB cards at ub2048.
        // The +45% P100 prefill is real but must be bought explicitly (PXA_FA_PREFILL_SPLIT=64)
        // on configs with the headroom.
        return 0;
    }();
    return v;
}

// PXA_FA_GQA_PACK ENHANCE default (2026-09-03). The packed-GQA vec kernel is reached on ONE
// shape only: the f32 vec FA kernel at Dk==Dv==256, cols_per_block==1, f16 K and V, no logit
// softcap — i.e. D=256 f16-KV decode. Unlike the nine house levers this IS a distinct kernel
// with its own register budget, and its measured basis is the 4x P100 hybrid-MoE seat (part of
// the twelve-lever ship set: +49% decode at 86k fill). So it arms only on a MULTI-CARD
// Pascal-GP100 (sm_60) topology, the cell it was measured on, and stays env-only everywhere
// else -- a single P100 and every other architecture are unmeasured for it. NH must divide
// both the head count and the gqa_ratio or the dispatch falls back to the stock kernel anyway.
// Explicit PXA_FA_GQA_PACK wins in both directions (0 reverts, 2/8 pick another pack width).
static inline int pxa_fa_gqa_pack_default() {
    if (pxa_config_level() < 2) return 0;
    const pxa_topology_t & t = g_pxa_topology;
    if (!t.valid || t.mixed || !t.multi || !t.has_sm60) return 0;
    return 4;
}

// PXA_MOE_DEVICE_MAP ENHANCE default (2026-09-03). Mode 1 moves the MoE expert-row mapping onto
// the device, removing a D2H + cudaStreamSynchronize from inside graph compute. It can only fire
// on a MoE model (the call site is the mul_mat_id row-mapping prep) and its measured basis is the
// same 4x P100 seat, so like FA_GQA_PACK it arms on a MULTI-CARD sm_60 topology only and stays
// env-only elsewhere. Mode 2 (device kernels + a host cross-check) remains env-only at every level.
// Explicit PXA_MOE_DEVICE_MAP wins in both directions.
static inline int pxa_moe_device_map_default() {
    if (pxa_config_level() < 2) return 0;
    const pxa_topology_t & t = g_pxa_topology;
    if (!t.valid || t.mixed || !t.multi || !t.has_sm60) return 0;
    if (pxa_model_known() && !pxa_model_is_moe()) return 0;
    return 1;
}

// PXA_ENHANCE_DBG topology debug line (2026-07-23). Gated behind PXA_ENHANCE_DBG (any non-zero
// value). Names the DETECTED topology and the CHOSEN per-topology ENHANCE config (the kernel
// levers this topology auto-set resolves, plus the recommended serve-flags -ub/-ts which are
// applied at launch, not from here). Called from ggml_cuda_init() AFTER pxa_enhance_init_topology.
static inline void pxa_enhance_log_topology_dbg() {
    const char * d = getenv("PXA_ENHANCE_DBG");
    if (!(d && atoi(d) != 0)) return;
    const pxa_topology_t & t = g_pxa_topology;
    fprintf(stderr, "PXA_ENHANCE_DBG: topology=\"%s\" ndev=%d multi=%d mixed=%d n_distinct=%d "
                    "[sm60=%d sm61=%d sm70=%d other=%d] level=%s\n",
            pxa_topology_name(), t.ndev, t.multi, t.mixed, t.n_distinct,
            t.has_sm60, t.has_sm61, t.has_sm70, t.has_other, pxa_config_level_name());
    // Chosen config for this topology (the kernel levers ENHANCE resolves + recommended serve flags).
    fprintf(stderr, "PXA_ENHANCE_DBG: chosen{ spec1row=%s router_fuse=%s volta_cublas_ne11=%d "
                    "p100_fp16_gemm=%s fuse_deltanet=%d int8_prefill=%d fa_mask_skip_tile=%s "
                    "fa_prefill_split=%d spec_relaxed=%s }",
            pxa_spec_1row_resolve() ? "on" : "off",
            (pxa_router_fuse_mode_resolve() != 0) ? "on" : "off",
            pxa_volta_cublas_ne11(),
            pxa_p100_fp16_gemm() ? "on" : "off",
            pxa_fuse_deltanet_default(),
            pxa_int8_prefill_mode_resolve(),
            pxa_fa_mask_skip_tile() ? "on" : "off",
            pxa_fa_prefill_split_ne11(),
            pxa_spec_relaxed_resolve() ? "on" : "off");
    // Recommended serve flags per the measured decision table (NOT set from here — launch flags).
    const char * rec_ub = "2048";
    const char * rec_ts = "n/a (single card)";
    if (t.mixed && t.has_sm70 && t.has_sm60) { rec_ub = "2048"; rec_ts = "1.4,0.6 (heavy on V100, OOM edge)"; }
    else if (t.has_sm70)                     { rec_ub = "1024"; rec_ts = t.multi ? "1,1 (balanced same-type)" : "n/a (single card)"; }
    else if (t.has_sm60)                     { rec_ub = "2048"; rec_ts = t.multi ? "1,1 (balanced same-type)" : "n/a (single card)"; }
    fprintf(stderr, " serve{ ub=%s ts=%s }\n", rec_ub, rec_ts);
}

// One-time startup report (stderr), called from ggml_cuda_init() with the enumerated device
// list: the level + the per-DEVICE decisions with their measured basis.
static inline void pxa_enhance_log_startup(int ndev, const int * ccs, const char (*names)[256]) {
    static bool done = false;
    if (done || ndev <= 0) return;
    done = true;
    fprintf(stderr, "PXA config level: %s (%s; PXA_ENHANCE=0 for DEFAULT, "
                    "PXA_REFERENCE=1 for REFERENCE)\n", pxa_config_level_name(), pxa_config_level_why());
    fprintf(stderr, "PXA level=%s", pxa_config_level_name());
    const int level = pxa_config_level();
    for (int i = 0; i < ndev; ++i) {
        const int cc = ccs[i];
        fprintf(stderr, " | dev%d %s(sm_%d):", i, names[i], cc / 10);
        const int i8 = pxa_int8_prefill_mode_resolve();
        if (level == 0) {
            fprintf(stderr, " reference [all PXA levers OFF]");
        } else if (cc >= 700 && cc < 750) {
            if (pxa_volta_cublas_ne11() > 0) {
                fprintf(stderr, " CUBLAS%d ON [+9.4%% pf]", pxa_volta_cublas_ne11());
            }
            fprintf(stderr, " ROUTER_FUSE %s", pxa_router_fuse_on(cc) ? "ON [+5-7% dec, sm_70]" : "off");
            if (i8 == 2) {
                fprintf(stderr, " INT8_PREFILL ON [TEST all-arch]");
            }
        } else if (cc == 610 && (i8 == 1 || i8 == 2)) {
            // No MASK_SKIP_TILE here: fast_fp16_available() excludes cc 610, so sm_61 never
            // dispatches the tile-f16 kernel the lever lives in — reporting it ON would be a
            // phantom lever. sm_61 takes tile-f32, whose skip is PXA_FA_MASK_SKIP_TILE_F32.
            fprintf(stderr, " INT8_PREFILL ON [+182%% pf, G3]");
            if (pxa_router_fuse_on(cc)) fprintf(stderr, " ROUTER_FUSE ON [mode %d]", pxa_router_fuse_mode_resolve());
        } else if (cc == 600) {
            fprintf(stderr, " FP16_GEMM %s [2:1 hgemm] MASK_SKIP_TILE %s [bit-exact]",
                    pxa_p100_fp16_gemm() ? "ON" : "off", pxa_fa_mask_skip_tile() ? "ON" : "off");
            if (pxa_router_fuse_on(cc)) fprintf(stderr, " ROUTER_FUSE ON [mode %d]", pxa_router_fuse_mode_resolve());
        } else if (i8 == 2) {
            fprintf(stderr, " INT8_PREFILL ON [TEST all-arch]");
        } else {
            fprintf(stderr, " defaults [bit-exact set]");
        }
    }
    if (pxa_fa_prefill_split_ne11() > 0) {
        fprintf(stderr, " | FA_PREFILL_SPLIT ne11>=%d [prefill rides fa-off chain]", pxa_fa_prefill_split_ne11());
    }
    fprintf(stderr, " | mode=%s%s", pxa_mode_name(),
            level == 0 ? " (overridden: reference)" : (pxa_mode() == 1 ? " [fa-off ingest]" : " [fa-on serving]"));
    if (pxa_spec_relaxed_resolve()) {
        fprintf(stderr, " | spec: SPEC_RELAXED ON [G3, spec lanes]");
    }
    fprintf(stderr, "\n");
}

// -----------------------------------------------------------------------------
// PXA per-(device x model) DECISION LEDGER (2026-07-29). Silent failure is this
// project's recurring defect class (phantom levers, silent demotes, silently
// ignored flags) — so every auto decision prints its VALUE, its REASON, and its
// OVERRIDE env, and a lever that cannot engage says why instead of no-opping
// quietly. Printed once both halves are known: ggml_cuda_init registers a
// profile hook (ggml-cuda.cu); whichever of {CUDA init, model registration}
// happens second triggers this.
static inline void pxa_enhance_log_model_decisions() {
    const ggml_pxa_model_profile & m = pxa_model();
    if (!m.valid || !g_pxa_topology.valid) return;
    fprintf(stderr,
            "PXA model=%s class=%s experts=%d(top%d) vocab=%d mtp_head=%s mtp_active=%s "
            "pxq_mmvq_tensors=%d | topology=\"%s\" | level=%s\n",
            m.arch_name[0] ? m.arch_name : "?", pxa_model_class_name(),
            m.n_expert, m.n_expert_used, m.n_vocab,
            m.has_mtp_head ? "yes" : "no", m.mtp_active ? "yes" : "no",
            m.n_pxq_mmvq_tensors, pxa_topology_name(), pxa_config_level_name());
    // REFERENCE opts out of every auto-set, so there is nothing to report there. DEFAULT
    // is NOT silent any more: since 2026-07-31 PXQ_MMVQ (and FUSE_DELTANET) arm at DEFAULT,
    // so early-returning here would print "auto-set engages only under PXA_ENHANCE=1" while
    // the binary had in fact armed them -- the ledger is what an operator reads to find out
    // what is active, so a false line here is worse than a stale doc.
    if (pxa_config_level() == 0) {
        fprintf(stderr, "PXA_AUTO: level=%s — every model-adaptive auto-set is OFF "
                        "(per-lever envs still win at any level)\n",
                pxa_config_level_name());
        return;
    }
    const pxa_topology_t & t = g_pxa_topology;
    // ROUTER_FUSE
    {
        const int  mode = pxa_router_fuse_mode_resolve();
        const char * why =
            getenv("PXA_ROUTER_FUSE")        ? "explicit env override" :
            mode == 1                        ? "pure-sm_70 topology x MoE model (+5.1..7.0% dec measured, single V100, PXQ2-35B)" :
            (m.n_expert == 0)                ? "OFF: dense model — no router GEMV exists, the fuse can never fire" :
            t.mixed                          ? "OFF: mixed topology (T4 measured -3.1% at -ts 1.4,0.6 — P100 gates -sm-layer decode)" :
            t.has_sm60                       ? "OFF: sm_60 present (+1.6% KILL measured on sm_60)" :
            !t.has_sm70                      ? "OFF: no sm_70 device (unmeasured fleet: the fuse arms only where it measured a win)" :
                                               "OFF";
        fprintf(stderr, "PXA_AUTO: ROUTER_FUSE=%d (%s; override PXA_ROUTER_FUSE)\n", mode, why);
    }
    // FUSE_DELTANET — self-gating by op presence; the ledger states relevance.
    fprintf(stderr, "PXA_AUTO: FUSE_DELTANET=%d (%s; override PXA_FUSE_DELTANET)\n",
            pxa_fuse_deltanet_default(),
            pxa_model_is_deltanet()
                ? "deltanet hybrid arch — cluster + row-absorb fusions active (bit-exact with eager; out-gate fusion off by default, guarded, see pxa/README.md)"
                : "INERT on this arch — no Gated-DeltaNet ops in the graph");
    // PXQ_MMVQ
    {
        const char * e = getenv("PXA_PXQ_MMVQ");
        const int auto_mode = pxa_pxq_mmvq_auto_default();
        const char * why =
            e                                ? "explicit env override" :
            auto_mode == 1                   ? "auto (DEFAULT/ENHANCE) x PXQ4/PXQ4HQ-bearing model x sm_70+ present (ship recipe A4m; fidelity-neutral paired dppl +0.016%)" :
            auto_mode == 2                   ? "auto (DEFAULT/ENHANCE) x PXQ4/PXQ4HQ-bearing model x all-sm_61 fleet (real DP4A)" :
            (m.n_pxq_mmvq_tensors <= 0)      ? "OFF: model carries no MMVQ-eligible PXQ4/PXQ4HQ tensors" :
            (t.has_sm60 && !t.has_sm70)      ? "OFF: sm_60-only fleet — P100 has no DP4A, the emulation path is not a win" :
                                               "OFF";
        fprintf(stderr, "PXA_AUTO: PXQ_MMVQ=%s (%s; override PXA_PXQ_MMVQ)\n",
                e ? e : (auto_mode == 1 ? "1" : auto_mode == 2 ? "2" : "0"), why);
    }
    // The device-only levers, restated with model context so ONE ledger holds every decision.
    fprintf(stderr, "PXA_AUTO: VOLTA_CUBLAS_NE11=%d (%s; override PXA_VOLTA_CUBLAS_NE11)\n",
            pxa_volta_cublas_ne11(),
            !t.has_sm70 ? "INERT: no sm_70 device" :
            (pxa_model_known() && strcmp(m.arch_name, "muse-glimmer") == 0)
                ? "OFF: muse-glimmer dense GEMMs measured faster on MMQ (pp2048 619 vs 411 t/s, 2x V100)"
                : "sm_70 present: dense-GEMM fp16-cuBLAS route at ne11>=threshold (+9.4% pf single V100; +6.5% 35B np2)");
    fprintf(stderr, "PXA_AUTO: P100_FP16_GEMM=%s (%s; override PXA_P100_FP16_GEMM)\n",
            pxa_p100_fp16_gemm() ? "on" : "off",
            t.has_sm60 ? "sm_60 present: GP100 2:1 fp16 hgemm on dense GEMMs (+51% gpt-oss prefill measured); sm_61 excluded (1:64 fp16)"
                       : "INERT: no sm_60 device");
    fprintf(stderr, "PXA_AUTO: FA_MASK_SKIP_TILE=%s (%s; override PXA_FA_MASK_SKIP_TILE)\n",
            pxa_fa_mask_skip_tile() ? "on" : "off",
            "bit-identical by construction; engages on fully-masked KV tiles (np2 co-resident slots)");
    fprintf(stderr, "PXA_AUTO: INT8_PREFILL=%d (%s; override PXA_PXQ_INT8_PREFILL)\n",
            pxa_int8_prefill_mode_resolve(),
            t.has_sm61 ? "sm_61 present: DP4A int8 prefill (+182% pf 1080Ti, PXQ2 5.8k cold)"
                       : "INERT: dispatch ship-gate is cc==610 and no sm_61 device is present");
    // PXQ_GEMM_2D relevance (resolver + auto-set live in ggml-cuda.cu, printed there when it
    // fires; this line covers the cannot-engage cases so the decision is never silent).
    if (!(t.has_sm60 && m.n_expert == 0 && m.n_pxq_mmvq_tensors > 0) && !getenv("PXA_PXQ_GEMM_2D")) {
        fprintf(stderr, "PXA_AUTO: PXQ_GEMM_2D=0 (%s; override PXA_PXQ_GEMM_2D)\n",
                !t.has_sm60                 ? "OFF: no sm_60 device (sm_70 measured -18.6% dense post-coalescing)" :
                (m.n_expert > 0)            ? "OFF: MoE model — post-coalescing sm_60 MoE cell unmeasured" :
                                              "OFF: model carries no PXQ tensors");
    }
    // The nine level-defaulted house levers (2026-09-03). Each reads its own env var at its own
    // call site; this line states what the LEVEL hands them when that var is unset, so the
    // decision is auditable from one place.
    fprintf(stderr, "PXA_AUTO: house levers KQ_MASK_PAD1 KV_SEQ_SOA TOPK_RAW TOPK_MOE_MULTIROW "
                    "GETROWS_NARROW CPY_FASTDIV CONCAT_FLAT NORM_REGCACHE SCHED_RESET_LAZY = %s "
                    "(%s; host-side / index-math, arch-independent; override each PXA_<NAME>)\n",
            pxa_house_lever_default() ? "on" : "off",
            pxa_house_lever_default() ? "ENHANCE level default"
                                      : "level below ENHANCE (PXA_ENHANCE=0 or PXA_REFERENCE=1)");
    fprintf(stderr, "PXA_AUTO: FA_GQA_PACK=%d (%s; override PXA_FA_GQA_PACK)\n",
            pxa_fa_gqa_pack_default(),
            getenv("PXA_FA_GQA_PACK") ? "explicit env override" :
            pxa_fa_gqa_pack_default() ? "multi-sm_60 topology x ENHANCE — packs 4 GQA heads per block on "
                                        "D=256 f16-KV decode (measured 4x P100 seat)"
                                      : "OFF: not a multi-card sm_60 topology, or level below ENHANCE — "
                                        "the pack width was measured on the 4x P100 seat only");
    fprintf(stderr, "PXA_AUTO: MOE_DEVICE_MAP=%d (%s; override PXA_MOE_DEVICE_MAP)\n",
            pxa_moe_device_map_default(),
            getenv("PXA_MOE_DEVICE_MAP") ? "explicit env override" :
            pxa_moe_device_map_default() ? "multi-sm_60 topology x MoE model x ENHANCE — expert-row "
                                           "mapping on device, no D2H inside graph compute"
                                         : (!m.n_expert ? "OFF: dense model — no expert row mapping exists"
                                                        : "OFF: not a multi-card sm_60 topology, or level below ENHANCE"));
    fprintf(stderr, "PXA_AUTO: SPEC_1ROW=%s SPEC_RELAXED=%s (spec lanes; measured no-op when no spec decode runs)\n",
            pxa_spec_1row_resolve() ? "on" : "off", pxa_spec_relaxed_resolve() ? "on" : "off");
    if (m.has_mtp_head) {
        fprintf(stderr, "PXA_AUTO: MTP model — PXA_MTP_LAZY_WARMUP defaults ON under ENHANCE "
                        "(bit-identical, +12%% then +47%% 35B prefill measured; resolver lives in "
                        "src/llama.cpp + server-context.cpp; override PXA_MTP_LAZY_WARMUP=0)\n");
    }
}
