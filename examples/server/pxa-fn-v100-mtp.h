#pragma once

// PXA_FLASHNEXT_V100_SET, MTP depth half (2026-10-26): Flash-Next (qwen4exp) on an all-V100 box at
// ENHANCE with GPU offload runs its MTP chain acceptance-adaptive by default: PXA_MTP_DEEP=3:0.5:0.8 with the draft
// ceiling raised to 5. The --spec-type row stays the base: below an acceptance EMA of 0.80 the chain is depth 3 at the
// row's own p_min; at or above it, up to 5 tokens at p_min 0.5. The raised ceiling sizes the verify batch, the
// recurrent checkpoint rows and the expert-cache chain reserve for width 6, exactly as `--spec-type mtp:n_max=5`.
//
// Measured (d1, 2026-10-26 12:21-12:52 PM, G31 2x V100 Flash-Next PXQN, build 940f2c5575cf, 12 visits mirrored,
// rep 1 dropped, n=6/arm): board row mtp:n_max=3,p_min=0.75 code 93.03 -> 95.62 t/s (+2.8%, 8.2 SE), prose 77.88 ->
// 78.75 (+1.1%, undetectable); n_max=4 row 92.98 / 78.34 (same bracket). Greedy512 sha identical in every arm
// (code d0b8497b, prose efbe1948): only the verify width changes and the qwen4exp verify columns are bitwise
// one-token decode up to width 8.
//
// Scoped to where it was measured: one slot (-np 1), an MTP stage whose row is n_max 3 or 4 at p_min 0.75, no
// PXA_SPEC_SELECT / PXA_SPEC_POLICY steering the depth. Anything else is left exactly as configured.
// Off: PXA_MTP_DEEP=0 (the --spec-type row runs as written), or PXA_FLASHNEXT_V100_SET=0 (the whole set).
// An explicit PXA_MTP_DEEP value is the user's: the rule then changes nothing (no ceiling raise either).

#include <cmath>
#include <cstdlib>

struct pxa_fn_v100_mtp_in {
    bool        qwen4exp      = false;   // model arch
    bool        has_nextn     = false;   // the file carries an MTP head
    bool        volta_only    = false;   // every visible CUDA device is sm_70
    int         config_level  = 0;       // ggml_pxa_config_level(): 2 = ENHANCE
    int         n_gpu_layers  = 0;
    int         n_parallel    = 1;
    bool        has_mtp_stage = false;   // the resolved chain has an MTP stage
    int         mtp_n_max     = -1;      // that stage's effective n_max
    float       mtp_p_min     = -1.0f;   // that stage's effective p_min
    bool        steered       = false;   // PXA_SPEC_SELECT / PXA_SPEC_POLICY live
    const char * set_env      = nullptr; // getenv("PXA_FLASHNEXT_V100_SET")
    const char * deep_env     = nullptr; // getenv("PXA_MTP_DEEP")
};

struct pxa_fn_v100_mtp_out {
    bool         on    = false;
    int          n_max = -1;          // the new MTP stage ceiling when on
    const char * deep  = nullptr;     // the PXA_MTP_DEEP value to set (overwrite=0) when on
    const char * why   = "";          // one-line reason (on or off)
};

static constexpr int   PXA_FN_V100_MTP_CEIL = 5;
static constexpr float PXA_FN_V100_MTP_PMIN = 0.75f;
static const char * const PXA_FN_V100_MTP_DEEP = "3:0.5:0.8";

static inline pxa_fn_v100_mtp_out pxa_fn_v100_mtp_rule(const pxa_fn_v100_mtp_in & in) {
    pxa_fn_v100_mtp_out o;
    if (in.set_env && *in.set_env && atoi(in.set_env) == 0) { o.why = "PXA_FLASHNEXT_V100_SET=0"; return o; }
    if (!in.qwen4exp || !in.volta_only || in.config_level < 2 || in.n_gpu_layers <= 0) {
        o.why = "not the Flash-Next all-V100 ENHANCE GPU layout"; return o;
    }
    if (!in.has_nextn || !in.has_mtp_stage) { o.why = "no MTP stage"; return o; }
    if (in.deep_env) { o.why = "PXA_MTP_DEEP is yours"; return o; }
    if (in.n_parallel != 1) { o.why = "more than one slot (unmeasured)"; return o; }
    if (in.steered) { o.why = "PXA_SPEC_SELECT / PXA_SPEC_POLICY steer the depth (unmeasured)"; return o; }
    if ((in.mtp_n_max != 3 && in.mtp_n_max != 4) || std::fabs(in.mtp_p_min - PXA_FN_V100_MTP_PMIN) > 0.005f) {
        o.why = "the MTP row is not n_max 3/4 at p_min 0.75 (unmeasured)"; return o;
    }
    o.on    = true;
    o.n_max = PXA_FN_V100_MTP_CEIL;
    o.deep  = PXA_FN_V100_MTP_DEEP;
    o.why   = "measured row";
    return o;
}
