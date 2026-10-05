#pragma once

#include <stdint.h>

// ---------------------------------------------------------------------------------------------
// The PXA core: the lever registry.
//
// WHY THIS FILE EXISTS. A PXA_* lever used to be a `static const bool v = [] { getenv(...) }()`
// written wherever it was first needed -- there are several hundred of them in the tree -- and the
// PXA_AUTO boot report was a SECOND piece of code that re-read the same environment and re-derived
// the same default. The two could disagree, and on 2026-09-20 they did: the report printed a
// lever's topology default beside the words "explicit env override".
//
// A lever declared here is declared ONCE: its name, the value that means "behave as the base
// engine did", its default at each config level, the rule that reads it, and the one line a user
// should be shown. Each row is resolved once per process, on the first read of THAT row (never
// because another row was read first), and the boot report is GENERATED from the same table, so
// a banner cannot disagree with the value in effect. A PXA_* variable that no row declares is
// named at boot instead of silently doing nothing.
//
// The first instalment was the levers that decide a flash-attention route; the rest of
// pxa-enhance.cuh follows in the next steps.
// ---------------------------------------------------------------------------------------------

enum pxa_lever_id {
    PXA_LEVER_FA_D512_VOLTA = 0, // 0 off / 1 m8n8k4 MMA / 2 tile   (ENHANCE default 2)
    PXA_LEVER_FA_D256_VOLTA_TILE,        // 0 off / 1 verify widths / 2 also width 1
    PXA_LEVER_FA_D256_VOLTA_TILE_MINKV,  // KV floor below which the route is not taken
    PXA_LEVER_FA_D256_VOLTA_TILE_MAXCOLS,// widest query batch the route is taken for
    PXA_LEVER_FA_MMA_VOLTA,      // sm_70 large-batch FA on the vendored MMA kernel
    PXA_LEVER_FA_MMA_VOLTA_Q8,   // ... and with a matched q8_0 K/V cache at head 256
    PXA_LEVER_FA_TILE_VOLTA,     // 0 off / 1 every batch size / 2 large batch only
    PXA_LEVER_FA_TILE256,        // pre-Volta D=256 batch>8 -> tile-f16 ncols=16
    PXA_LEVER_FA_SWA_SLICE,      // the unsound windowed KV slice, benchmarking only
    PXA_LEVER_FA_SWA_KEEP,       // keep n_swa in op_params[4] for the mask-driven KV scan
    PXA_LEVER_SM60_FA_VEC_F32,   // sm_60 decode -> fp32-accumulating vec kernel
    PXA_LEVER_CORE_ROUTES,       // print the route census at exit
    PXA_LEVER_FA_QKV_DIRECT,     // sm_60 narrow FA reads a q4_0/q8_0 K/V cache in place, GQA group per block
    PXA_LEVER_FA_QKV_TILE,       // sm_60 wide FA reads a q4_0/q8_0 K/V cache in place (no f16 conversion)
    PXA_LEVER_FA_QKV_DIRECT_VOLTA, // sm_70 narrow FA takes the QKV_DIRECT kernel too (no whole-cache f16 conversion)
    PXA_LEVER_REPORT_END,        // the rows above this line print in the generated PXA_AUTO report

    // ---- the resolvers that lived in pxa-enhance.cuh (core step 3) ----------------------------
    // Resolved on first read like every other row, but NOT in the generated report: each of these
    // already has a hand-written ledger line that says more than a generated one could (it names the
    // card set and the model that decided it), and that text must not change. They are registered so
    // that one table owns the variable, the rule and the value, and so that the unknown-variable
    // check knows them.
    //
    // Level-only rules (the environment variable, else the config level):
    PXA_LEVER_P100_FP16_GEMM = PXA_LEVER_REPORT_END,   // sm_60 dequant->fp16 + cublasGemmEx COMPUTE_16F for dense GEMMs
    PXA_LEVER_SPEC_1ROW,         // 1-row GEMV also at spec-verify batch sizes
    PXA_LEVER_FA_MASK_SKIP_TILE, // skip fully masked KV tiles in the tile-f16 FA kernel
    PXA_LEVER_FA_TILE_F32ACC,    // fp32 running state in the tile-f16 FA kernel
    PXA_LEVER_FA_TILE_V2,        // alternative schedule for the tile-f16 FA arithmetic (env only)
    PXA_LEVER_FA_PREFILL_SPLIT,  // prefill batches at or above this width ride the non-FA attention chain
    PXA_LEVER_MODE,              // posture: 0 balance, 1 max
    //
    // Tier masks (the variable, else a fixed default; each prints one line when it arms):
    PXA_LEVER_PXQ23_MMVQ,        // which low tiers (bit0 PXQ2, bit1 PXQ3) may ride the q8_1 MMVQ path; default 0
    PXA_LEVER_PXQ_MMV_H2,        // which tiers (bit0 PXQ2, bit1 PXQ3, bit2 PXQ4) use the half2 pair-LUT loop; default 3
    //
    // Rules that read the card set and the model profile (and so are re-read when the profile changes,
    // or on every read, exactly as the resolvers they replace were):
    PXA_LEVER_ROUTER_FUSE,       // MoE router-logits GEMV kernel: 0 off / 1 sm_70 ship gate / 2 test all-arch / 3 sm_60-shaped GEMV
    PXA_LEVER_VOLTA_CUBLAS_NE11, // sm_70 dense quantized GEMMs with ne11 >= N go to fp16 cuBLAS (0 = MMQ always)
    PXA_LEVER_PXQ_INT8_PREFILL,  // int8 prefill tile (sm_61 ship gate; 2 = test all-arch)
    PXA_LEVER_SPEC_RELAXED,      // relaxed speculative acceptance (the level default only where PXA_SPEC_SAMPLED=0)
    //
    // The variable, else a default that is a RULE over the level, the card set and the model. The
    // call sites keep whatever they do with the value (validation, one-line notices, a per-TU cache);
    // what moved is the environment read and the rule:
    PXA_LEVER_FA_GQA_PACK,       // Q heads packed per block in the f32 vec FA kernel (0 off; 2/4/8); default 4 on multi sm_60 at ENHANCE
    PXA_LEVER_MOE_DEVICE_MAP,    // 0 host expert-row mapping / 1 device kernels / 2 device + host cross-check
    PXA_LEVER_FUSE_DELTANET,     // bitmask of the Gated-DeltaNet decode fusions; default 55 (0 at REFERENCE)
    PXA_LEVER_PXQ_MMVQ,          // PXQ4/PXQ4HQ decode via the q8_1 MMVQ kernel: 0 off / 1 sm_70+ / 2 all-sm_61 (raw, before the call site's refusals)
    PXA_LEVER_PXQ_GEMM_2D,       // the 2D prefill GEMM for sm_60 (1) / sm_60+sm_70 (2); auto 1 on a dense PXQ model with an sm_60 card
    PXA_LEVER_ENHANCE_DBG,       // PXA_ENHANCE_DBG: print the topology debug line at CUDA init
    //
    // Resolvers that lived in their own translation unit:
    PXA_LEVER_FA_MASK_SKIP_TILE_F32, // fully-masked-tile skip in the tile-f32 FA kernel (sm_61 / forced-F32); default OFF at every level
    //
    // The CUDA house levers: host-side / index-math changes, arch-independent. The variable, else ON at
    // ENHANCE (the shipping level) and OFF at DEFAULT and REFERENCE.
    PXA_LEVER_NORM_REGCACHE,     // register-cached norm variants (block_size == 1024)
    PXA_LEVER_CONCAT_FLAT,       // flattened non-contiguous concat launch
    PXA_LEVER_CPY_FASTDIV,       // fastdiv index math in the float cpy kernel
    PXA_LEVER_GETROWS_NARROW,    // flattened narrow-row get_rows launch
    PXA_LEVER_TOPK_MOE_MULTIROW, // block-wide aliasing barrier for the multi-row fused top-k MoE router
    PXA_LEVER_COUNT
};

// The resolved value. Cheap: a table read after the first call.
int64_t pxa_lever(pxa_lever_id id);

// The rows that came over from pxa-enhance.cuh, as an X-macro: the unit test walks it, and so can
// anything else that wants to name them all.
#define PXA_LEVER_MOVED_ROWS(X) \
    X(P100_FP16_GEMM) X(SPEC_1ROW) X(FA_MASK_SKIP_TILE) X(FA_TILE_F32ACC) X(FA_TILE_V2) X(FA_PREFILL_SPLIT) X(MODE) \
    X(PXQ23_MMVQ) X(PXQ_MMV_H2) X(ROUTER_FUSE) X(VOLTA_CUBLAS_NE11) X(PXQ_INT8_PREFILL) X(SPEC_RELAXED) \
    X(FA_GQA_PACK) X(MOE_DEVICE_MAP) X(FUSE_DELTANET) X(PXQ_MMVQ) X(PXQ_GEMM_2D) X(ENHANCE_DBG) \
    X(FA_MASK_SKIP_TILE_F32) X(NORM_REGCACHE) X(CONCAT_FLAT) X(CPY_FASTDIV) X(GETROWS_NARROW) X(TOPK_MOE_MULTIROW)

// Did the user set this lever's environment variable to something this build understands?
// (For the rows that came over from pxa-enhance.cuh: was the variable present, as their old
// resolvers tested it.)
bool pxa_lever_set_by_user(pxa_lever_id id);

// What the rule hands the lever when the variable is unset, at the current config level, card set
// and model profile. Defined for the rows whose default is a rule over those; the row's plain
// default elsewhere.
int64_t pxa_lever_default(pxa_lever_id id);

// The variable's text exactly as the user gave it (null when unset): for a ledger line that quotes it.
const char * pxa_lever_env_text(pxa_lever_id id);

// Why the process is at the config level it is, in one clause: "PXA_REFERENCE=1" / "PXA_ENHANCE=0" /
// "PXA_ENHANCE=1" / "default". The level itself is resolved in ggml.c (it is shared with the
// translation units outside the CUDA backend); this is only its provenance, for the startup line.
const char * pxa_core_level_why(void);

// "explicit env override" / "REFERENCE default" / "DEFAULT default" / "ENHANCE default".
const char * pxa_lever_why(pxa_lever_id id);

// The PXA_AUTO lines for every declared lever, generated from the table. Called from the boot
// report; safe to call more than once, prints once.
void pxa_core_lever_report(void);

// Name every PXA_* / PXQ_* variable in the environment that no row declares. A misspelled lever
// used to do nothing quietly, which has cost us at least two mis-read experiments.
void pxa_core_lever_check_unknown(void);
