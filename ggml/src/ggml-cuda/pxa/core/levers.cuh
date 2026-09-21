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
// should be shown. It is resolved once per process, on first read, and the boot report is
// GENERATED from the same table, so a banner cannot disagree with the value in effect. A PXA_*
// variable that no row declares is named at boot instead of silently doing nothing.
//
// This is the first instalment: the levers that decide a flash-attention route. The rest of
// pxa-enhance.cuh follows in the next step.
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
    PXA_LEVER_COUNT
};

// The resolved value. Cheap: a table read after the first call.
int64_t pxa_lever(pxa_lever_id id);

// Did the user set this lever's environment variable to something this build understands?
bool pxa_lever_set_by_user(pxa_lever_id id);

// "explicit env override" / "REFERENCE default" / "DEFAULT default" / "ENHANCE default".
const char * pxa_lever_why(pxa_lever_id id);

// The PXA_AUTO lines for every declared lever, generated from the table. Called from the boot
// report; safe to call more than once, prints once.
void pxa_core_lever_report(void);

// Name every PXA_* / PXQ_* variable in the environment that no row declares. A misspelled lever
// used to do nothing quietly, which has cost us at least two mis-read experiments.
void pxa_core_lever_check_unknown(void);
