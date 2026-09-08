// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// The PXQ quantization tiers below are PXA Network's original work; the creator
// of this fork is PXA Network. (provenance canary: PXA-7Q6LM32E16-ORIGIN)
// pxq6.cuh — PXQ6: E16-row two-level scales + the unified PXA kernel-fastpath family (K0..K6).
//
// Spec: PXQ 4-bit-tier optimization notes, 2026-07-17 (internal lab); deep-dives: K1 K-split
// on P100 + K6 V100 tensor-core headroom (same series).
//
// FORMAT (GGML_TYPE_PXQ4 = 252, core tier; GGML_TYPE_PXQ4HQ = 253, bs8 tier):
//   values : 4-bit codes into the frozen PX16 book
//   scales : per-ROW fp16 anchor (128 B header per 64-row panel; ggml row_meta_size = 2 B/row)
//            x 4-bit sub-scale per 16-elem block (core, SUB16) / per 8-elem block (HQ, SUB8)
//   dequant: eff = fp32(anchor) * SUB[s4];  w = eff * fp32(book[c])   — ONE extra fp32 mul
//            vs PXQ5; prefill GEMM snaps __float2half_rn(w); decode mmv accumulates fp32.
//   layout : slab = 64|128 B scale SoA + 64 x 16 B nibble rows (= the PXQ4/PXQ5 slab, so
//            coalescing / one-pass streaming / graph capture / 64-row panels are untouched);
//            panel = 128 B anchor header + kslabs slabs; panels row-major; experts outermost.
//   Measured quality (lab, frozen 36-slice rng-42 protocol): core wrel 0.068086 (−12.6% vs
//   PXQ5's 0.077906) @ 4.2656 bpw; HQ wrel 0.058683 (−24.7%) @ 4.5156 bpw.
//
// KERNEL FASTPATH FAMILY — format-independent (templated on a format policy: PXQ4, PXQ5,
// PXQ6, PXQ6HQ all instantiate), each behind its own env gate, DEFAULT OFF, clean rollback:
//   K1 PXA_PXQ6_KSPLIT=1    decode gateup K-split, THE BIT-EXACT FORM: the S=4 split blocks
//                           ARE the existing PXQ4_MMV_KSEG=4 k-segment chains (one 64-thread
//                           block per kseg; per-kseg fp32 partials to a persistent workspace;
//                           the reducer replicates the proven red[] combination order exactly,
//                           then bias + GLU). Per-thread chains and the final 4-way sum order
//                           are identical to the proven kernel => bit-identical output.
//   K1b PXA_PXQ6_KSPLIT_GEN=S  generic S-way split (2/4/8): 256-thr blocks, S k-chunks, the
//                           occupancy form from the K-split deep-dive. DETERMINISTIC but NOT
//                           bit-exact vs the proven kernel (different chain count) — gate G3
//                           before any live use. Staged for the GPU window.
//   K2 PXA_PXQ6_PAIRLUT=1   256-entry float2 byte-pair LUT (book[lo],book[hi] per byte) —
//                           same FMA order/operands => bit-exact.
//      PXA_PXQ6_VECX=1      float4 activation loads (2 bytes/iter) — same operand order =>
//                           bit-exact.
//   K2d PXA_PXQ6_SHFL=1     register-book __shfl decode (decode mmv family, every tier):
//                           lane l holds book[l] in one register; the two per-pair book
//                           gathers become SHFL.IDX, moving the codebook traffic (~63% of
//                           decode LSU bytes on the fired gateup loop) off the saturated
//                           LSU onto the idle shuffle pipe — same index, same float, same
//                           FMA order => bit-exact.
//   K3 PXA_PXQ6_GUFUSE=1    prefill: ONE fused up+gate GEMM kernel with the GLU epilogue
//                           (kills one full A-tile pass + the separate GLU kernel + 2 f32
//                           round-trips). Identical per-tensor accumulation chains => bit-exact.
//      PXA_PXQ6_SCATFUSE=1  prefill: the down GEMM scatters straight to the MoE output rows
//                           (kills the C_down buffer + copy kernel). Same values/rows => exact.
//   K4 PXA_PXQ6_RAGTAIL=1   prefill: threads whose 8-token column group lies beyond
//                           tile.nrows skip the FMA loop (stores were already masked) => exact.
//   K5 PXA_PXQ6_PIPE=1      prefill: 2-stage register prefetch of the next slab's codes + A
//                           tile (sm_60 latency tool; identical arithmetic DAG) => bit-exact.
//   K6 PXA_PXQ6_WMMA=1|2|3  V100 (cc==700) ONLY: fused dequant→skewed-smem-fp16→m16n16k16
//                           HMMA grouped GEMM. 1 = fp32 accumulator fragments, 2 = fp16-accum
//                           twin (closest-precision A/B), 3 = v2 REBUILD (2-stage register
//                           double-buffer + fused up+gate/GLU + WMMA scat down; fp32 accum).
//                           NOT bit-exact (fp16-mul/fp32-add-tree
//                           vs strict-k half2 chain) — G3 logprob parity + G4 ppl regate are
//                           MANDATORY before any live use; temp-0 sha is NOT a valid gate here.
//   PXA_PXQ6=0              master off for PXQ6/PXQ6HQ fused kernels (dequant→cublas fallback).
//   PXA_PXQ6_FORCE_PREFILL=1  TEST ONLY: bypass the sm_60/70 arch gate so correctness A/Bs can
//                           run on other CUDA archs (e.g. the 1080Ti). Never set in production.
//
// Numeric tables: ggml-pxq6-tables.h (frozen literals, sha256-locked to the lab artifacts).
// PXA_PXQ6_BOOK / PXA_PXQ6_SUB / PXA_PXQ6_SUB_HQ override the tables (fp16-snapped, must match
// the file's KV provenance pxa.pxq6.*) — same discipline as PXA_PXQ5_BOOK.
#pragma once

#include "pxq4.cuh"              // shared slab macros, tile structs, glu, gather (reused)
#include "pxa-enhance.cuh"       // PXA_REFERENCE / PXA_ENHANCE master config tiers
#include "../../../include/ggml-pxq6-tables.h"
#include <mma.h>
#include <mutex>
#include <atomic>
#include <type_traits>

// per-TU device tables (ggml-cuda.cu and convert.cu each get a copy; env overrides are
// uploaded per-TU + per-device by pxq6_maybe_upload_tables()).
static __device__ float pxq6_book_g[16]  = PXQ6_BOOK_INIT;
static __device__ float pxq6_sub16_g[16] = PXQ6_SUB16_INIT;
static __device__ float pxq6_sub8_g[16]  = PXQ6_SUB8_INIT;
static __device__ float pxq6_lm32_g[32]  = PXQ6_LM32_INIT;   // PXQ6 (id 256) 5-bit LM32 book

// ---------------------------------------------------------------------------------------------
// env gates (each cached once; every one logs its state on first query)
// ---------------------------------------------------------------------------------------------
static inline bool pxa_pxq6_env_flag(const char * name, bool dflt, bool * logged = nullptr) {
    const char * e = getenv(name);
    bool v = e ? atoi(e) != 0 : dflt;
    (void)logged;
    return v;
}

// master gate + host self-check (table integrity: fp16-snapped, ascending, book invariants)
static inline bool pxa_pxq6_enabled() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ6");
        bool v = !(e && atoi(e) == 0);
        if (v) {
            static const float book[16]  = PXQ6_BOOK_INIT;
            static const float sub16[16] = PXQ6_SUB16_INIT;
            static const float sub8[16]  = PXQ6_SUB8_INIT;
            bool ok = book[7] == 0.0f && book[15] == 1.0f;
            for (int i = 0; i < 15 && ok; ++i) ok = book[i] < book[i+1] && sub16[i] < sub16[i+1] && sub8[i] < sub8[i+1];
            ok = ok && sub16[0] > 0.0f && sub8[0] > 0.0f;
            for (int i = 0; i < 16 && ok; ++i) {   // fp16-snap idempotence (frozen contract)
                ok = __half2float(__float2half_rn(sub16[i])) == sub16[i] &&
                     __half2float(__float2half_rn(sub8[i]))  == sub8[i];
            }
            if (!ok) {
                fprintf(stderr, "PXA_PXQ6: table self-check FAILED — fused kernels DISABLED (fallback in use)\n");
                v = false;
            } else {
                fprintf(stderr, "PXA_PXQ6 fused kernels: ON (table self-check PASS; PXA_PXQ6=0 disables)\n");
            }
        } else {
            fprintf(stderr, "PXA_PXQ6 fused kernels: OFF (dequant->cublas fallback)\n");
        }
        return v;
    }();
    return on;
}

// Level-aware default (pxa-enhance.cuh): PXA_REFERENCE=1 -> every gate defaults OFF (pure
// reference path); DEFAULT/ENHANCE -> the shipped default. An explicit env var ALWAYS wins.
#define PXA_PXQ6_GATE(fn, env, dflt, desc) \
    static inline bool fn() { \
        static const bool on = [](){ \
            const char * e = getenv(env); \
            bool v = e ? atoi(e) != 0 : pxa_gate_default(dflt); \
            if (v != (dflt)) fprintf(stderr, "%s: %s (%s)\n", env, v ? "ON" : "OFF", desc); \
            return v; \
        }(); \
        return on; \
    }

// Defaults: the measured bit-exact winners from docs/LEVERS.md §2 are ON out of the box
// (published per-card numbers assume them; `<env>=0` reverts any one of them to the proven
// reference path). Config-specific / no-gain levers (§3) stay opt-in.
PXA_PXQ6_GATE(pxa_pxq6_ksplit,   "PXA_PXQ6_KSPLIT",   true,  "K1 decode gateup K-split, bit-exact kseg form")
PXA_PXQ6_GATE(pxa_pxq6_pairlut,  "PXA_PXQ6_PAIRLUT",  false, "K2 byte-pair float2 LUT, bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_vecx,     "PXA_PXQ6_VECX",     true,  "K2 float4 activation loads, bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_gufuse,   "PXA_PXQ6_GUFUSE",   true,  "K3 fused up+gate GEMM + GLU epilogue, bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_scatfuse, "PXA_PXQ6_SCATFUSE", true,  "K3 down-GEMM scatter fusion, bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_ragtail,  "PXA_PXQ6_RAGTAIL",  true,  "K4 ragged-tile FMA skip, bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_pipe,     "PXA_PXQ6_PIPE",     false, "K5 register-prefetch GEMM pipelining, bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_force_prefill, "PXA_PXQ6_FORCE_PREFILL", false, "TEST ONLY: bypass prefill arch gate")
PXA_PXQ6_GATE(pxa_g2_redfuse,    "PXA_G2_REDFUSE",    false, "G2-F1 absorb gateup ksplit-reduce + GLU into the down-mmv x-staging prologue, bit-exact")
PXA_PXQ6_GATE(pxa_g2_addfuse,    "PXA_G2_ADDFUSE",    true,  "G2-F4 residual-add fusion: ADD+FUSED_RMS_NORM pair + MUL_MULTI_ADD residual epilogue, bit-exact")
// G2-F4 sub-gates: the master gate above covers TWO independent fusions that were shipped
// together. They fire on different graph shapes and have different safety arguments, so they
// also need to be A/B-able on their own (a determinism bisect on the published PXQ2 tier is
// exactly the case the combined gate could not resolve). Both default ON under the master.
PXA_PXQ6_GATE(pxa_g2_addfuse_rms, "PXA_G2_ADDFUSE_RMS", true,  "G2-F4a ADD + FUSED_RMS_NORM pair fusion")
PXA_PXQ6_GATE(pxa_g2_addfuse_mma, "PXA_G2_ADDFUSE_MMA", true,  "G2-F4b MUL_MULTI_ADD + residual ADD epilogue fusion")
// G2-F5 (2026-09-03): the four remaining node-adjacency fusions had NO write-after-read guard
// and no same-device check, unlike the ADD/RMS family above. Two of them (ADD+SOFTPLUS+MUL and
// MUL+EXP+MUL) sit on the delta-net gate path, which is why an unguarded shifted alias there
// shows up as recurrent-state corruption (the all-"!" 539b277a signature) rather than as a
// small numeric wobble. Each is gated so it can be A/B-ed on its own; all default ON, now
// guarded.
PXA_PXQ6_GATE(pxa_g2_softplusfuse, "PXA_G2_SOFTPLUSFUSE", true, "G2-F5a ADD + SOFTPLUS + MUL fusion (delta-net gate)")
PXA_PXQ6_GATE(pxa_g2_mulexpfuse,   "PXA_G2_MULEXPFUSE",   true, "G2-F5b MUL + EXP + MUL fusion (delta-net decay)")
PXA_PXQ6_GATE(pxa_g2_cpy2fuse,     "PXA_G2_CPY2FUSE",     true, "G2-F5c CPY + VIEW + CPY pair fusion")
PXA_PXQ6_GATE(pxa_g2_sumrowsfuse,  "PXA_G2_SUMROWSFUSE",  true, "G2-F5d CONT + SUM_ROWS + TRANSPOSE fusion")
PXA_PXQ6_GATE(pxa_pxq6_prmt,     "PXA_PXQ6_PRMT",     false, "K2c prmt register-LUT book decode (4-bit tiers, decode mmv), bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_ldcs,     "PXA_PXQ6_LDCS",     false, "K7 streaming (evict-last-priority) weight code loads in decode mmv, bit-exact")
PXA_PXQ6_GATE(pxa_pxq3_pairlut,  "PXA_PXQ3_PAIRLUT",  false, "K2b3 P3 64-entry bit-plane pair LUT, decode mmv family, bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_shfl,     "PXA_PXQ6_SHFL",     false, "K2d register-book __shfl decode: book gathers ride the shuffle pipe, not the LSU; bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_rowx2,    "PXA_PXQ6_ROWX2",    false, "R2 two-rows-per-thread 128-thr decode mmv twins (halve warp-uniform x LDS writeback), bit-exact")

// R2 first-fire / first-decline, one line per dispatch site (phantom-lever discipline:
// an armed lever that never prints ENGAGED did not run — such an A/B is void, not null).
static inline void pxa_pxq6_rowx2_log(int site, bool engaged) {
    static std::atomic<uint32_t> seen{0};
    if (site < 0 || site >= 6) return;
    const uint32_t bit = 1u << (2*site + (engaged ? 0 : 1));
    if (seen.fetch_or(bit) & bit) return;
    static const char * names[6] = {"2d-mmv", "2d-mmv-split", "moe-gateup-split",
                                    "dense-gateup-split", "moe-down-split", "moe-down"};
    fprintf(stderr, "PXA_PXQ6_ROWX2 %s: %s\n", names[site],
            engaged ? "ENGAGED (128-thr two-row twin)"
                    : "DECLINED -> 256-thr incumbent (no x2 instantiation)");
}

// Q1 (PXA_PXQ6_QPF): one-slab-lookahead register prefetch in the decode chunk walk.
// 0 = off (default), 1 = single-weight mmv family, 2 = gateup family, 3 = both — split so
// the two register-cliff exposures A/B independently (the gateup twin carries 2x the
// prefetch state). At every site Q1 yields to an engaged R2 twin (no double-restructure).
static inline int pxa_pxq6_qpf() {
    static const int v = [](){
        const char * e = getenv("PXA_PXQ6_QPF");
        int m = e ? atoi(e) : 0;
        if (m < 0 || m > 3) { fprintf(stderr, "PXA_PXQ6_QPF=%d invalid (want 0-3) -- OFF\n", m); m = 0; }
        if (m) fprintf(stderr, "PXA_PXQ6_QPF: armed (%s%s) -- Q1 decode slab prefetch, bit-exact; "
                               "an A/B without the in-run ENGAGED banner is void\n",
                       (m & 1) ? "mmv " : "", (m & 2) ? "gateup" : "");
        return m;
    }();
    return v;
}

// Q1 first-fire / first-decline, one line per dispatch site (phantom-lever discipline).
static inline void pxa_pxq6_qpf_log(int site, bool engaged) {
    static std::atomic<uint32_t> seen{0};
    if (site < 0 || site >= 6) return;
    const uint32_t bit = 1u << (2*site + (engaged ? 0 : 1));
    if (seen.fetch_or(bit) & bit) return;
    static const char * names[6] = {"2d-mmv", "2d-mmv-split", "moe-gateup-split",
                                    "dense-gateup-split", "moe-down-split", "moe-down"};
    fprintf(stderr, "PXA_PXQ6_QPF %s: %s\n", names[site],
            engaged ? "ENGAGED (Q1 one-slab-lookahead twin)"
                    : "DECLINED -> incumbent (no qp instantiation)");
}


// PXQ6R (GGML_TYPE_PXQ6 = 256, display "pxq6") master gate + host self-check. Default ON;
// PXA_PXQ6R=0 disables (dequant->cublas fallback). Invariants of the frozen LM32 book:
// exactly 32 entries, strictly ascending, fp16-snap idempotent, book[16] == +0.0f (PXQ6R_ZIDX),
// book[0] == -1.0f, book[31] > 0. Failure -> one-time warning + fmt NONE.
static inline bool pxa_pxq6r_enabled() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ6R");
        bool v = !(e && atoi(e) == 0);
        if (v) {
            static const float book[32] = PXQ6_LM32_INIT;
            bool ok = book[16] == 0.0f && book[0] == -1.0f && book[31] > 0.0f;
            for (int i = 0; i < 31 && ok; ++i) ok = book[i] < book[i+1];
            for (int i = 0; i < 32 && ok; ++i) {   // fp16-snap idempotence (frozen contract)
                ok = __half2float(__float2half_rn(book[i])) == book[i];
            }
            if (!ok) {
                fprintf(stderr, "PXA_PXQ6R: LM32 table self-check FAILED — fused kernels DISABLED (fallback in use)\n");
                v = false;
            } else {
                fprintf(stderr, "PXA_PXQ6R fused kernels: ON (LM32 self-check PASS; PXA_PXQ6R=0 disables)\n");
            }
        } else {
            fprintf(stderr, "PXA_PXQ6R fused kernels: OFF (dequant->cublas fallback)\n");
        }
        return v;
    }();
    return on;
}

// K6 WMMA: 0 = off (default), 1 = fp32-accum fragments, 2 = fp16-accum twin,
// 3 = v2 (double-buffered + GUFUSE-fused + WMMA scat down; fp32 accum). cc==700 only.
static inline int pxa_pxq6_wmma() {
    static const int mode = [](){
        const char * e = getenv("PXA_PXQ6_WMMA");
        int m = e ? atoi(e) : 0;
        if (m < 0 || m > 3) m = 0;
        if (m) fprintf(stderr, "PXA_PXQ6_WMMA: mode %d (K6 V100 HMMA prefill%s — NOT bit-exact, G3+G4 gated)\n",
                       m, m == 3 ? " v2: dbuf+gufuse+scat" : "");
        return m;
    }();
    return mode;
}

// K1b generic S-split (2/4/8; 0 = off). NOT bit-exact — staged for the GPU window (G3 first).
static inline int pxa_pxq6_ksplit_gen() {
    static const int s = [](){
        const char * e = getenv("PXA_PXQ6_KSPLIT_GEN");
        int v = e ? atoi(e) : 0;
        if (v != 0 && v != 2 && v != 4 && v != 8) v = 0;
        if (v) fprintf(stderr, "PXA_PXQ6_KSPLIT_GEN: S=%d (generic split — NOT bit-exact, G3 required)\n", v);
        return v;
    }();
    return s;
}

// Arch-aware effective S for the decode gateup split (2026-07-26). An explicit env ALWAYS wins
// (including =0). Unset defaults to S=4 on the two target Teslas: the K1 64-thr form spreads
// the SAME warp count over more blocks, while the 256-thr S=4 chunk form multiplies
// warps-in-flight 4x — measured +2.6% decode on 2xV100 and +2.9% on 2xP100 (F35B pure-PXQ4,
// fill ~6k; S=8 was worse on V100, keep 4). Other archs keep the K1 bit-exact default.
static inline int pxa_pxq6_ksplit_gen_eff(int cc) {
    static const bool has_env = getenv("PXA_PXQ6_KSPLIT_GEN") != nullptr;
    if (has_env) return pxa_pxq6_ksplit_gen();
    const bool on = (cc == 700 || cc == 600);
    static const bool logged = [](bool o){
        if (o) fprintf(stderr, "PXA_PXQ6_KSPLIT_GEN: S=4 (sm_60/70 decode default; env overrides)\n");
        return true;
    }(on);
    (void)logged;
    return on ? 4 : 0;
}

// env table overrides (PXA_PXQ6_BOOK / _SUB / _SUB_HQ + the PXQ6R 32-entry PXA_PXQ6R_BOOK),
// fp16-snapped, per-device upload
static inline void pxq6_maybe_upload_tables(int device) {
    static bool parsed = false;
    static bool have_book = false, have_sub = false, have_sub8 = false, have_lm32 = false;
    static float ebook[16], esub[16], esub8[16], elm32[32];
    static bool uploaded[64] = {false};
    if (!parsed) {
        parsed = true;
        auto parse_nw = [](const char * e, float * out, int want) -> bool {
            int n = 0; float v[32];
            char * dup = strdup(e);
            for (char * t = strtok(dup, ","); t && n < want; t = strtok(nullptr, ",")) v[n++] = strtof(t, nullptr);
            free(dup);
            if (n != want) return false;
            for (int i = 0; i < want; ++i) out[i] = __half2float(__float2half_rn(v[i]));
            return true;
        };
        auto parse16 = [&](const char * e, float * out) -> bool { return parse_nw(e, out, 16); };
        if (const char * e = getenv("PXA_PXQ6_BOOK"))   { have_book = parse16(e, ebook);  fprintf(stderr, "PXA_PXQ6_BOOK: %s\n",   have_book ? "custom book active"  : "parse FAILED — ignored"); }
        if (const char * e = getenv("PXA_PXQ6_SUB"))    { have_sub  = parse16(e, esub);   fprintf(stderr, "PXA_PXQ6_SUB: %s\n",    have_sub  ? "custom SUB16 active" : "parse FAILED — ignored"); }
        if (const char * e = getenv("PXA_PXQ6_SUB_HQ")) { have_sub8 = parse16(e, esub8);  fprintf(stderr, "PXA_PXQ6_SUB_HQ: %s\n", have_sub8 ? "custom SUB8 active"  : "parse FAILED — ignored"); }
        if (const char * e = getenv("PXA_PXQ6R_BOOK"))  { have_lm32 = parse_nw(e, elm32, 32); fprintf(stderr, "PXA_PXQ6R_BOOK: %s\n", have_lm32 ? "custom LM32 book active" : "parse FAILED — ignored"); }
    }
    if ((have_book || have_sub || have_sub8 || have_lm32) && device >= 0 && device < 64 && !uploaded[device]) {
        uploaded[device] = true;
        int cur = 0; cudaGetDevice(&cur);
        if (cur != device) cudaSetDevice(device);
        if (have_book) cudaMemcpyToSymbol(pxq6_book_g,  ebook, sizeof(ebook));
        if (have_sub)  cudaMemcpyToSymbol(pxq6_sub16_g, esub,  sizeof(esub));
        if (have_sub8) cudaMemcpyToSymbol(pxq6_sub8_g,  esub8, sizeof(esub8));
        if (have_lm32) cudaMemcpyToSymbol(pxq6_lm32_g,  elm32, sizeof(elm32));
        if (cur != device) cudaSetDevice(cur);
    }
}

// ---------------------------------------------------------------------------------------------
// format policies. The WHOLE format difference is: slab/header geometry + how the per-8-elem
// effective scales are decoded. NEFF = distinct eff scales per 32-elem block (1 = single chain,
// preserving the exact PXQ4/PXQ5 accumulation shape).
// ---------------------------------------------------------------------------------------------
// (pxq6_pol_p4, the policy for the retired legacy type id 250, was removed 2026-07-21.)

// (pxq6_pol_p5, the policy for the retired legacy type id 251, was removed 2026-07-21.)

struct pxq6_pol_p6 {
    static constexpr int  SLAB = PXQ6_SLAB_BYTES, HDR = PXQ6_HDR_BYTES, CODE_OFF = 64, NEFF = 2;
    __device__ static void stage_tabs(float * tab, float * sub, int tid) {
        if      (tid < 16) tab[tid]      = pxq6_book_g[tid];
        else if (tid < 32) sub[tid - 16] = pxq6_sub16_g[tid - 16];
    }
    __device__ static float bookv(int i) { return pxq6_book_g[i]; }
    __device__ static float anchor(const uint8_t * panel, int row) {
        return __half2float(((const half *)panel)[row]);
    }
    __device__ static void row_effs(const uint8_t * slab, int row, float anch, const float * sub, float * eff) {
        const int sb = slab[row];
        eff[0] = anch * sub[sb & 0xf];    // elems 0-15
        eff[1] = anch * sub[sb >> 4];     // elems 16-31
    }
    static constexpr int CODE_WORDS = 4, CODE_BYTES = 16;
    __device__ static float2 pair(const uint32_t * q, int b, const float * tab) {
        const int byte = (q[b >> 2] >> (8*(b & 3))) & 0xff;   // LE byte b of the 16-B code row
        return make_float2(tab[byte & 0xf], tab[byte >> 4]);
    }
    // K2d: identical byte extraction; the two gathers ride SHFL.IDX (lane l holds tab[l])
    __device__ static float2 pair_shfl(const uint32_t * q, int b, float bv) {
        const int byte = (q[b >> 2] >> (8*(b & 3))) & 0xff;   // LE byte b of the 16-B code row
        return make_float2(__shfl_sync(0xffffffffu, bv, byte & 0xf),
                           __shfl_sync(0xffffffffu, bv, byte >> 4));
    }
    __device__ static float2 pairl(const uint32_t * q, int b, const float2 * plut) {
        return plut[(q[b >> 2] >> (8*(b & 3))) & 0xff];
    }
};

struct pxq6_pol_p6hq {
    static constexpr int  SLAB = PXQ6HQ_SLAB_BYTES, HDR = PXQ6_HDR_BYTES, CODE_OFF = 128, NEFF = 4;
    __device__ static void stage_tabs(float * tab, float * sub, int tid) {
        if      (tid < 16) tab[tid]      = pxq6_book_g[tid];
        else if (tid < 32) sub[tid - 16] = pxq6_sub8_g[tid - 16];
    }
    __device__ static float bookv(int i) { return pxq6_book_g[i]; }
    __device__ static float anchor(const uint8_t * panel, int row) {
        return __half2float(((const half *)panel)[row]);
    }
    __device__ static void row_effs(const uint8_t * slab, int row, float anch, const float * sub, float * eff) {
        const int sb0 = slab[2*row], sb1 = slab[2*row + 1];
        eff[0] = anch * sub[sb0 & 0xf];   // elems 0-7
        eff[1] = anch * sub[sb0 >> 4];    // elems 8-15
        eff[2] = anch * sub[sb1 & 0xf];   // elems 16-23
        eff[3] = anch * sub[sb1 >> 4];    // elems 24-31
    }
    static constexpr int CODE_WORDS = 4, CODE_BYTES = 16;
    __device__ static float2 pair(const uint32_t * q, int b, const float * tab) {
        const int byte = (q[b >> 2] >> (8*(b & 3))) & 0xff;   // LE byte b of the 16-B code row
        return make_float2(tab[byte & 0xf], tab[byte >> 4]);
    }
    // K2d: identical byte extraction; the two gathers ride SHFL.IDX (lane l holds tab[l])
    __device__ static float2 pair_shfl(const uint32_t * q, int b, float bv) {
        const int byte = (q[b >> 2] >> (8*(b & 3))) & 0xff;   // LE byte b of the 16-B code row
        return make_float2(__shfl_sync(0xffffffffu, bv, byte & 0xf),
                           __shfl_sync(0xffffffffu, bv, byte >> 4));
    }
    __device__ static float2 pairl(const uint32_t * q, int b, const float2 * plut) {
        return plut[(q[b >> 2] >> (8*(b & 3))) & 0xff];
    }
};

// PXQ6R "real PXQ6" (GGML_TYPE_PXQ6 = 256, display "pxq6"): LM32 5-bit codes on the UNCHANGED
// E16-row scale machinery (fp16 row anchor + shared SUB16 4-bit sub per 16-elem block).
// Code row = 20 B: bytes 0..15 nibble plane (byte-identical layout to the P6 core rows) +
// bytes 16..19 one LE u32 hi-bit plane (bit j = bit 4 of code j, j = 0..31). Code rows start
// at 64 + 20r — 4-byte aligned for every r, NOT 8/16-aligned for odd r => scalar u32 loads
// ONLY (the CODE_WORDS == 5 arm of pxq6_ldcodes; uint2/uint4 vector loads are ILLEGAL here).
// Modes: TAB/TAB_CS only — the 32-entry book rides neither PRMT (two 16-entry byte planes
// keyed by a nibble) nor PAIRLUT (256-entry LUT keyed on one byte = two 4-bit codes); the
// pickers demote exactly like P2/P3 (pairl() is a never-executed compile stub).
struct pxq6_pol_p6r {
    static constexpr int  SLAB = PXQ6R_SLAB_BYTES, HDR = PXQ6R_HDR_BYTES, CODE_OFF = PXQ6R_CODE_OFF, NEFF = 2;
    static constexpr int CODE_WORDS = 5, CODE_BYTES = 20;
    __device__ static void stage_tabs(float * tab, float * sub, int tid) {
        if      (tid < 32) tab[tid]      = pxq6_lm32_g[tid];
        else if (tid < 48) sub[tid - 32] = pxq6_sub16_g[tid - 32];
    }
    __device__ static float bookv(int i) { return pxq6_lm32_g[i & 31]; }
    __device__ static float anchor(const uint8_t * panel, int row) {
        return __half2float(((const half *)panel)[row]);
    }
    __device__ static void row_effs(const uint8_t * slab, int row, float anch, const float * sub, float * eff) {
        const int sb = slab[row];
        eff[0] = anch * sub[sb & 0xf];    // elems 0-15
        eff[1] = anch * sub[sb >> 4];     // elems 16-31
    }
    // pair b (elems 2b, 2b+1): nibble plane byte extraction == proven P6 form; 5th bit from the
    // register-resident plane word q[4] (bit j = element j -> elem 2b at bit 2b, 2b+1 at 2b+1)
    __device__ static float2 pair(const uint32_t * q, int b, const float * tab) {
        const int byte = (q[b >> 2] >> (8*(b & 3))) & 0xff;   // LE byte b of the 16-B nibble plane
        const uint32_t hi = q[4];                             // hi-bit plane word
        const int c0 = (byte & 0xf) | (int)(((hi >> (2*b    )) & 1) << 4);   // element 2b
        const int c1 = (byte >> 4)  | (int)(((hi >> (2*b + 1)) & 1) << 4);   // element 2b+1
        return make_float2(tab[c0], tab[c1]);
    }
    // K2d: identical plane reassembly; the two gathers ride SHFL.IDX (32-entry book:
    // every lane's tab[l] is staged, codes span the full 0..31 range)
    __device__ static float2 pair_shfl(const uint32_t * q, int b, float bv) {
        const int byte = (q[b >> 2] >> (8*(b & 3))) & 0xff;   // LE byte b of the 16-B nibble plane
        const uint32_t hi = q[4];                             // hi-bit plane word
        const int c0 = (byte & 0xf) | (int)(((hi >> (2*b    )) & 1) << 4);   // element 2b
        const int c1 = (byte >> 4)  | (int)(((hi >> (2*b + 1)) & 1) << 4);   // element 2b+1
        return make_float2(__shfl_sync(0xffffffffu, bv, c0),
                           __shfl_sync(0xffffffffu, bv, c1));
    }
    __device__ static float2 pairl(const uint32_t * q, int b, const float2 * plut) {
        (void)q; (void)b; (void)plut; return make_float2(0.f, 0.f);   // PAIRLUT demoted for P6R
    }
};

// per-row code load: CODE_WORDS LE u32 words. 16 B formats keep the proven uint4 load;
// 8 B uses uint2 (8-aligned: CODE_OFF 64 + row*8); 12 B uses 3 u32 (4-aligned: row*12);
// 4 B (P1) uses ONE u32.
// CS variant (K7): identical addresses/values through ld.global.cs — the decode weight stream
// is read exactly once per token, so mark it evict-first and keep L2 for the hot working set
// (activations/tables/KV). Cache policy cannot change loaded values => bit-exact by construction.
template <class POL, bool CS = false>
static __device__ __forceinline__ void pxq6_ldcodes(const uint8_t * p, uint32_t * q) {
    if constexpr (POL::CODE_WORDS == 4) {
        if constexpr (CS) { *(uint4 *)q = __ldcs((const uint4 *)p); }
        else              { *(uint4 *)q = *(const uint4 *)p; }
    } else if constexpr (POL::CODE_WORDS == 2) {
        if constexpr (CS) { *(uint2 *)q = __ldcs((const uint2 *)p); }
        else              { *(uint2 *)q = *(const uint2 *)p; }
    } else if constexpr (POL::CODE_WORDS == 1) {
        // P1 4 B row (CODE_OFF 64 + row*4): 4-aligned only -> ONE scalar u32.
        // Without this arm CODE_WORDS == 1 fell through to the trailing 3-load branch (the
        // P3 12 B case), which wrote q[1]/q[2] on a `uint32_t q[1]` array and read 8 B past
        // the row -- past the tensor entirely on the last row of the last slab. Latent because
        // pxq6_pol_p1 is only instantiated by k_pxq6_dequant_matrix and pol_p1::pair() reads
        // q[0] alone, so the over-read was never consumed; the OOB global access was real.
        const uint32_t * s = (const uint32_t *)p;
        if constexpr (CS) { q[0] = __ldcs(s); }
        else              { q[0] = s[0]; }
    } else if constexpr (POL::CODE_WORDS == 5) {
        // P6R 20 B row: only 4-aligned (odd rows) -> five scalar u32 loads, NEVER vector
        const uint32_t * s = (const uint32_t *)p;
        if constexpr (CS) { q[0] = __ldcs(s); q[1] = __ldcs(s + 1); q[2] = __ldcs(s + 2); q[3] = __ldcs(s + 3); q[4] = __ldcs(s + 4); }
        else              { q[0] = s[0]; q[1] = s[1]; q[2] = s[2]; q[3] = s[3]; q[4] = s[4]; }
    } else {
        const uint32_t * s = (const uint32_t *)p;
        if constexpr (CS) { q[0] = __ldcs(s); q[1] = __ldcs(s + 1); q[2] = __ldcs(s + 2); }
        else              { q[0] = s[0]; q[1] = s[1]; q[2] = s[2]; }
    }
}

// ---------------------------------------------------------------------------------------------
// K2c PRMT register-LUT book decode (B1 steal from the QTIP/Marlin lineage). The 16-entry
// fp16-snapped book is held as two byte planes (lo/hi) across 8 uniform registers; a nibble
// code indexes them with prmt (byte_perm) — zero smem traffic, zero bank conflicts, and the
// produced fp32 values are IDENTICAL to tab[] (the book is fp16-snapped, half->float is exact)
// with the same FMA order => bit-exact vs MODE_TAB. 4-bit tiers only (nibble == book index).
// ---------------------------------------------------------------------------------------------
struct pxq6_prmt_book {
    uint32_t L[4];   // lo bytes of book[0..15]
    uint32_t H[4];   // hi bytes of book[0..15]
};

// build once per block from the staged fp32 tab (post-__syncthreads); uniform across threads.
static __device__ __forceinline__ void pxq6_prmt_build(const float * __restrict__ tab, pxq6_prmt_book & B) {
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        uint32_t lo = 0, hi = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned short u = __half_as_ushort(__float2half_rn(tab[4*r + i]));
            lo |= (uint32_t)(u & 0xff) << (8*i);
            hi |= (uint32_t)(u >> 8)   << (8*i);
        }
        B.L[r] = lo; B.H[r] = hi;
    }
}

// Gather two book bytes from one 16-entry plane (4 regs). idx_lo/idx_hi land in OUTPUT bytes 0/1.
// KEY: __byte_perm selects each output byte via a 4-BIT NIBBLE of the selector (not a byte), so the
// two indices' low-3-bits go in nibbles 0 and 1 (perm_sel). The high-half (idx>=8) select is a
// per-BYTE 0xFF mask, so __vcmpgeu4 takes a byte-packed copy of the indices (mask_sel).
static __device__ __forceinline__ uint32_t pxq6_prmt_plane(const uint32_t * __restrict__ P,
                                                           uint32_t perm_sel, uint32_t mask_sel) {
    const uint32_t a = __byte_perm(P[0], P[1], perm_sel);   // out[0,1] = entries (idx&7) of the low half
    const uint32_t b = __byte_perm(P[2], P[3], perm_sel);   // out[0,1] = entries (idx&7)+8 of the high half
    const uint32_t m = __vcmpgeu4(mask_sel, 0x08080808u);   // 0xFF in byte i where idx_i >= 8
    return (a & ~m) | (b & m);
}

// one code byte (2 nibbles) -> float2(book[lo], book[hi]); same value/order contract as pair().
static __device__ __forceinline__ float2 pxq6_prmt_pair(const uint32_t * __restrict__ q, int b,
                                                        const pxq6_prmt_book & B) {
    const uint32_t byte   = (q[b >> 2] >> (8*(b & 3))) & 0xffu;
    const uint32_t idx_lo = byte & 0xfu, idx_hi = byte >> 4;
    const uint32_t perm   = (idx_lo & 7u) | ((idx_hi & 7u) << 4);   // nibble0=idx_lo&7, nibble1=idx_hi&7
    const uint32_t mask   = idx_lo | (idx_hi << 8);                 // byte0=idx_lo, byte1=idx_hi
    const uint32_t lo     = pxq6_prmt_plane(B.L, perm, mask);       // out[0,1]=lo(V_lo),lo(V_hi)
    const uint32_t hi     = pxq6_prmt_plane(B.H, perm, mask);       // out[0,1]=hi(V_lo),hi(V_hi)
    const uint32_t w      = __byte_perm(lo, hi, 0x5140u);           // [lo(Vlo),hi(Vlo),lo(Vhi),hi(Vhi)]=half2
    return __half22float2(*(const half2 *)&w);
}

// shared address helpers
template <class POL>
static __device__ __forceinline__ size_t pxq6_panel_stride(int kslabs) {
    return (size_t)POL::HDR + (size_t)kslabs*POL::SLAB;
}
template <class POL>
static __device__ __forceinline__ const uint8_t * pxq6_panel(const uint8_t * W, int e, int panels, int p, int kslabs) {
    return W + ((size_t)e*panels + p)*pxq6_panel_stride<POL>(kslabs);
}

// PAIRLUT staging: plut[b] = (book[lo(b)], book[hi(b)]) — built from the same global tables
template <class POL>
static __device__ __forceinline__ void pxq6_stage_pairlut(float2 * plut, int tid, int nthr) {
    for (int i = tid; i < 256; i += nthr) plut[i] = make_float2(POL::bookv(i & 0xf), POL::bookv(i >> 4));
}

// PAIRL3 staging (P3-only, PXA_PXQ3_PAIRLUT): 64 entries keyed by the PAIR's packed planes,
//   key = lo4 | (hi2 << 4); lo4 = both codes' low-plane 2-bit fields, hi2 = both high bits
//   => code0 = (key & 3) | ((key>>4 & 1) << 2), code1 = (key>>2 & 3) | ((key>>5 & 1) << 2)
// Values go through POL::bookv == the SAME per-TU global table stage_tabs() stages into
// tab[], so every float2 handed to pxq6_acc2 is bit-identical to the MODE_TAB gathers.
// 64 x float2 = 512 B static smem per block.
template <class POL>
static __device__ __forceinline__ void pxq6_stage_pairlut3(float2 * plut, int tid, int nthr) {
    for (int i = tid; i < 64; i += nthr) {
        const int c0 = (i & 3)        | (((i >> 4) & 1) << 2);
        const int c1 = ((i >> 2) & 3) | (((i >> 5) & 1) << 2);
        plut[i] = make_float2(POL::bookv(c0), POL::bookv(c1));
    }
}

// ---------------------------------------------------------------------------------------------
// the scaled 32-elem dot product — the ONE routine every decode kernel shares.
// NEFF=1 keeps the exact single-chain accumulation of the proven PXQ4/PXQ5 kernels
// (su += d*t with one t chain); NEFF=2/4 is the PXQ6/PXQ6HQ reference form.
// MODE/VECX only change operand SOURCING (never order) => bit-exact per variant:
//   MODE 0 = smem tab, 1 = smem PAIRLUT, 2 = PRMT register book,
//        3 = smem tab + ld.cs codes, 4 = PRMT + ld.cs codes,
//        5 = P3-only 64-entry bit-plane pair-LUT (PXA_PXQ3_PAIRLUT; smem, bit-exact).
//        6 = register-book __shfl decode (PXA_PXQ6_SHFL; every tier, bit-exact — lane l
//            holds tab[l] in one register and SHFL.IDX replaces each LDS book gather).
// ---------------------------------------------------------------------------------------------
#define PXQ6_MODE_TAB     0
#define PXQ6_MODE_PAIRL   1
#define PXQ6_MODE_PRMT    2
#define PXQ6_MODE_TAB_CS  3
#define PXQ6_MODE_PRMT_CS 4
#define PXQ6_MODE_PAIRL3  5   // P3-only (PXA_PXQ3_PAIRLUT): 64-entry bit-plane pair-LUT
#define PXQ6_MODE_SHFL    6   // K2d (PXA_PXQ6_SHFL): register-book __shfl decode, every tier

template <int MODE> struct pxq6_mode {
    static constexpr bool cs   = (MODE == PXQ6_MODE_TAB_CS) || (MODE == PXQ6_MODE_PRMT_CS);
    static constexpr bool prmt = (MODE == PXQ6_MODE_PRMT)   || (MODE == PXQ6_MODE_PRMT_CS);
    static constexpr bool pairl = (MODE == PXQ6_MODE_PAIRL);
    static constexpr bool pairl3 = (MODE == PXQ6_MODE_PAIRL3);
    static constexpr bool shfl = (MODE == PXQ6_MODE_SHFL);
};

// PXQ_CANON_v2 (2026-08-04) - accumulation SHAPE, not sourcing.
//
// The shipped inner loop writes  t += a0*x0 + a1*x1, which ptxas emits as FMUL + FFMA + FADD:
// 3 float ops and 3 roundings per pair, 48 per 32 weights. This is SASS-verified on the
// shipped sm_60 binary (16x FMUL.FTZ + 16x FFMA.FTZ + 16x FADD.FTZ per dot32), not inferred.
// ptxas cannot fix it on its own: --fmad=true CONTRACTS but never REASSOCIATES, and nvcc never
// sets LLVM reassoc - the same property that PXQ_CANON_v1 split==unsplit bit-exactness already
// depends on.
//
// The chained form below is 2 FFMA: 2 roundings instead of 3, and BOTH products are computed
// exactly (2-of-2 rather than 1-of-2). So this is a precision IMPROVEMENT, not a trade; the
// expected perplexity delta is <= 0.
//
// It is also register-NEGATIVE: the def-use graph shrinks from 3 float defs per pair to 1.
// That matters because k_pxq6_gateup_mmv_ksplit_gen sits at REG=80 with only ~5 registers of
// slack before the 85-reg / 3-blocks-per-SM cliff - the same cliff that killed PRMT (-11%) and
// full unroll (26.8 -> 18.6 t/s). A register-negative change cannot cross it.
//
// It DOES change the fp32 fold order, so it is a re-baselining event and therefore a BUILD
// constant rather than an env flag: every variant flips together and split==unsplit stays true
// by construction, exactly like PXQ6_CANON_CMAX. Default 0 = byte-identical to today.
// Build with -DPXQ_CANON_V2=1 to engage. Regenerate all controls on the engaged binary before
// comparing anything - pre-v2 numbers are not comparable.
#ifndef PXQ_CANON_V2
#define PXQ_CANON_V2 0
#endif

static __device__ __forceinline__ float pxq6_acc2(float acc, float a0, float x0, float a1, float x1) {
#if PXQ_CANON_V2
    return __fmaf_rn(a1, x1, __fmaf_rn(a0, x0, acc));
#else
    return acc + (a0*x0 + a1*x1);
#endif
}

// MODE-dispatched pair sourcing. `if constexpr` keeps discarded branches UNINSTANTIATED,
// so a policy only needs the members of the modes its pickers actually offer (pairl3()
// exists on pxq6_pol_p3 ONLY — no other policy is ever instantiated at MODE PAIRL3).
// For the pre-existing modes this inlines to exactly the calls the old ternary chain
// made: sourcing identical, bit-exact by construction.
// K2d rides a trailing param (bv): non-SHFL instantiations discard it under
// `if constexpr`, so every pre-existing mode keeps its exact sourcing and codegen.
// NO DEFAULT on bv (R2xK2d lesson): a defaulted 0.f let a bv-less caller compile
// clean and decode a zero book at MODE SHFL. Every caller stages bv explicitly.
template <class POL, int MODE>
static __device__ __forceinline__ float2 pxq6_pairx(const uint32_t * __restrict__ q, int b,
                                                    const float * __restrict__ tab,
                                                    const float2 * __restrict__ plut,
                                                    const pxq6_prmt_book & pb,
                                                    const float bv) {
    using M = pxq6_mode<MODE>;
    if constexpr (M::prmt)        return pxq6_prmt_pair(q, b, pb);
    else if constexpr (M::pairl)  return POL::pairl(q, b, plut);
    else if constexpr (M::pairl3) return POL::pairl3(q, b, plut);
    else if constexpr (M::shfl)   return POL::pair_shfl(q, b, bv);
    else                          return POL::pair(q, b, tab);
}

template <class POL, int MODE, bool VECX>
static __device__ __forceinline__ float pxq6_dot32(const uint8_t * __restrict__ slab, int row, float anch,
                                                   const float * __restrict__ xk,
                                                   const float * __restrict__ tab,
                                                   const float * __restrict__ sub,
                                                   const float2 * __restrict__ plut,
                                                   const pxq6_prmt_book & pb) {
    using M = pxq6_mode<MODE>;
    float eff[POL::NEFF];
    POL::row_effs(slab, row, anch, sub, eff);
    uint32_t q[POL::CODE_WORDS];
    pxq6_ldcodes<POL, M::cs>(slab + POL::CODE_OFF + row*POL::CODE_BYTES, q);
    // K2d (MODE SHFL): lane l holds tab[l] — the SAME staged smem values every other
    // mode gathers from — in one register. One loop-invariant LDS here replaces the 32
    // per-slab book gathers; garbage in lanes >= the book size is never selected (codes
    // index only valid entries). Dead code for every other mode.
    float bv = 0.f;
    if constexpr (M::shfl) bv = tab[threadIdx.x & 31];
    float t[POL::NEFF];
    #pragma unroll
    for (int i = 0; i < POL::NEFF; ++i) t[i] = 0.f;
    if (VECX) {
        #pragma unroll
        for (int b = 0; b < 16; b += 2) {
            const float4 xv = *(const float4 *)&xk[2*b];
            const float2 p0 = pxq6_pairx<POL, MODE>(q, b,   tab, plut, pb, bv);
            const float2 p1 = pxq6_pairx<POL, MODE>(q, b+1, tab, plut, pb, bv);
            t[(b*POL::NEFF) >> 4]     = pxq6_acc2(t[(b*POL::NEFF) >> 4],     p0.x, xv.x, p0.y, xv.y);
            t[((b+1)*POL::NEFF) >> 4] = pxq6_acc2(t[((b+1)*POL::NEFF) >> 4], p1.x, xv.z, p1.y, xv.w);
        }
    } else {
        #pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float2 p = pxq6_pairx<POL, MODE>(q, b, tab, plut, pb, bv);
            t[(b*POL::NEFF) >> 4] = pxq6_acc2(t[(b*POL::NEFF) >> 4], p.x, xk[2*b], p.y, xk[2*b+1]);
        }
    }
    if (POL::NEFF == 1) return eff[0]*t[0];
    if (POL::NEFF == 2) return eff[0]*t[0] + eff[1]*t[1];
    return (eff[0]*t[0] + eff[1]*t[1]) + (eff[2]*t[2] + eff[3]*t[3]);
}

// ---------------------------------------------------------------------------------------------
// full-matrix dequant (convert.cu fallback hook — dequant->cublas keeps PXQ6 functional on
// every arch, incl. the 1080Ti/sm_61). One block per slab, 64 threads (one row each).
// ---------------------------------------------------------------------------------------------
template <class POL, typename dst_t>
static __global__ void k_pxq6_dequant_matrix(const uint8_t * __restrict__ wq, dst_t * __restrict__ y,
                                             const int kslabs, const int64_t K) {
    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    // STORE COALESCING (2026-07-27). Decoding is naturally row-major -- one thread owns one row
    // and produces its PXQ6_QK consecutive outputs -- but writing that straight to y has 32
    // threads storing addresses K elements apart, so each store instruction touches 32 sectors
    // to deliver 64 useful bytes (~1/16 efficiency). This kernel is the incumbent for every PXQ
    // 2D MUL_MAT wider than PXA_PXQ4_2D_MAX_NY, which on a DENSE model is the entire FFN at
    // prefill, so that pattern was the dominant dense prefill cost (61.7 t/s vs MXFP4's 276 on
    // 2xV100). Stage the tile here and write it out along K below.
    //
    // +2 pads the row stride to 34, i.e. 17 4-byte banks per row for a 2-byte dst_t; gcd(17,32)
    // = 1, so the column-major fill in phase 1 is conflict-free. A 4-byte dst_t pads to 2-way,
    // which is not worth a second layout -- the f16 path is the one cuBLAS takes.
    __shared__ dst_t tile[PXQ6_BM][PXQ6_QK + 2];
    POL::stage_tabs(tab, sub, threadIdx.x);
    __syncthreads();
    const int64_t slab_id = blockIdx.x;
    const int64_t p  = slab_id / kslabs;
    const int     kb = (int)(slab_id % kslabs);
    const int     row = threadIdx.x;
    const uint8_t * panel = wq + p*pxq6_panel_stride<POL>(kslabs);
    const uint8_t * slab  = panel + POL::HDR + (size_t)kb*POL::SLAB;
    const float anch = POL::HDR ? POL::anchor(panel, row) : 0.f;
    float eff[POL::NEFF];
    POL::row_effs(slab, row, anch, sub, eff);
    uint32_t q[POL::CODE_WORDS];
    pxq6_ldcodes<POL>(slab + POL::CODE_OFF + row*POL::CODE_BYTES, q);
    #pragma unroll
    for (int b = 0; b < 16; ++b) {                 // b = element-pair index
        const float e = eff[(b*POL::NEFF) >> 4];
        const float2 v = POL::pair(q, b, tab);
        tile[row][2*b]   = (dst_t)(e * v.x);
        tile[row][2*b+1] = (dst_t)(e * v.y);
    }
    __syncthreads();
    // Same values, same addresses, different instruction mapping: one warp per row, lane == the
    // k offset, so a store covers PXQ6_QK contiguous elements. Bit-identical output.
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int nwarp = blockDim.x   >> 5;
    for (int r = warp; r < PXQ6_BM; r += nwarp) {
        y[(p*PXQ6_BM + r)*K + kb*PXQ6_QK + lane] = tile[r][lane];
    }
}

// ---------------------------------------------------------------------------------------------
// K8 — wide-store full-matrix dequant (PXA_PXQ_DQ_WIDE=1). Same values, same order, same
// addresses as k_pxq6_dequant_matrix above; only the block shape and the store instruction
// change, so the output is BIT-IDENTICAL by construction (asserted by tests/test-pxq-dq-wide.cu).
//
// WHY. The incumbent runs one 64-thread block per 1088 B slab: on a 4096x4096 PXQ4 tensor that
// is 16384 two-warp blocks, 16384 __syncthreads, a full smem round trip of the decoded tile,
// and a 64 B store per warp. Occupancy caps at 66% (2-warp blocks) and the store stream is
// 64 B/warp. This kernel takes a whole 64-row panel x NKS consecutive slabs per 256-thread
// block, stages the panel header + slabs into smem ONCE, and emits the decoded halves straight
// from registers with one 16 B vector store per lane. With NKS == 8 the 32 lanes of a warp
// cover 8 slabs x 4 element-groups of ONE row, i.e. 8*32 = 256 consecutive halves = 512
// contiguous, 64 B-aligned bytes per store instruction, and there is no output smem traffic.
//
// SMEM SLAB PADDING. The staged slab stride is POL::SLAB + 16 (1104 / 1168 / 1360 B). In
// 4-byte banks that is 276 / 292 / 340 words; 276 % 32 == 20 and gcd(20,32) == 4 gives the 8
// slab bases distinct bank offsets {0,20,8,28,16,4,24,12}, so the 8 slabs of a tile spread
// across the 32 banks rather than piling onto one. The +16 also keeps every slab base 16 B
// aligned, which is what makes the phase-1 uint4 staging stores legal.
//
// BIT-EXACTNESS. Each lane calls POL::pair() and POL::row_effs() verbatim, in the incumbent's
// pair order, with the incumbent's operand order (e * v.x then e * v.y, e = anchor*SUB[nibble]),
// and snaps with the same __float2half_rn. The eff selection eff[(b*NEFF)>>4] collapses to a
// per-lane constant — index g>>1 for NEFF==2, g for NEFF==4 — and is picked out of the
// row_effs() array by an unrolled compare so the array stays in registers.
//
// VARIANTS. PXA_DQ_WIDE_VARIANT selects (NKS, minBlocksPerSM):
//   0 = 8/8 (default)   1 = 8/6   2 = 8/4   3 = 4/8   4 = 4/6   5 = 4/4
//
// MEASURED 2026-09-03, 1x V100-PCIE-16GB, tools tests/test-pxq-dq-wide.cu phase 3, 20 reps,
// median, GB/s = (2*R*K written + raw PXQ bytes read) / time:
//   2048x5120 PXQ4: incumbent 603.6 | wide 290.9-301.8 across all six variants
//   4096x4096 PXQ4: incumbent 643.9 | wide 304.2-324.6 across all six variants
// The wide form is ~2x SLOWER, and flat against NKS, launch_bounds, unroll factor and per-lane
// read width (16 B full-code-row vs the 4 B form kept here) — none of those moved it more than
// 3%. The premise this kernel was built on does not hold: the incumbent already streams at
// 604-644 GB/s, roughly 70% of the card's 900 GB/s peak, not the ~368 GB/s that was inferred
// from the PXA_DQ_ABLATE prefill delta. There is no bandwidth headroom here to recover, so the
// ablation's +3.8%/+6.9% ceiling is NOT bought by making this kernel wider; whatever the
// ablation removes is not this kernel's DRAM time. THE GATE STAYS OFF. What is worth keeping is
// the harness and the bit-exactness result: any future rewrite of this kernel has a zero-
// tolerance reference to build against, and the next attempt should start from a profile of the
// incumbent rather than from a store-width argument.
// ---------------------------------------------------------------------------------------------
template <class POL, typename dst_t, int NKS, int OCC>
static __global__ void __launch_bounds__(256, OCC)
k_pxq6_dequant_wide(const uint8_t * __restrict__ wq, dst_t * __restrict__ y,
                    const int kslabs, const int64_t K, const int ktiles) {
    static_assert(NKS == 4 || NKS == 8, "k_pxq6_dequant_wide: NKS must be 4 or 8");
    static_assert(POL::SLAB % 16 == 0 && POL::HDR % 16 == 0, "16 B staging granule");

    constexpr int SPAD = POL::SLAB + 16;      // padded smem slab stride (bank spread + 16 B align)
    constexpr int LPR  = NKS * 4;             // lanes that cover one full row of the tile
    constexpr int RPW  = 32 / LPR;            // rows a warp covers per iteration (1 for NKS 8)
    constexpr int ITER = 8 / RPW;             // 64 rows / (8 warps * RPW)
    constexpr int HW   = POL::HDR  / 16;      // header 16 B words
    constexpr int SW   = POL::SLAB / 16;      // slab   16 B words

    __shared__ float   tab[32];               // 32 for the P6R LM32 book; 16-entry tiers leave 16..31
    __shared__ float   sub[16];
    __shared__ __align__(16) uint8_t sm[POL::HDR + NKS*SPAD];   // uint4 staging + uint4 code reads

    POL::stage_tabs(tab, sub, threadIdx.x);

    const int tile = blockIdx.x;
    const int p    = tile / ktiles;                 // panel
    const int kb0  = (tile - p*ktiles) * NKS;       // first slab of this tile
    const int nks  = min(NKS, kslabs - kb0);

    const uint8_t * panel = wq + (size_t)p * pxq6_panel_stride<POL>(kslabs);

    // phase 1 — cooperative 16 B stage of the panel header + this tile's slabs. Consecutive
    // threads read consecutive 16 B words, so the global side is fully coalesced; the incumbent
    // already relies on wq being 16 B aligned (pxq6_ldcodes' uint4 arm reads panel + 16*row).
    {
        const int total = HW + nks*SW;
        for (int i = threadIdx.x; i < total; i += 256) {
            int soff, doff;
            if (i < HW) {
                soff = i*16;
                doff = i*16;
            } else {
                const int j = i - HW;
                const int s = j / SW;
                const int w = j - s*SW;
                soff = POL::HDR + (kb0 + s)*POL::SLAB + w*16;   // kb0: this tile's first slab
                doff = POL::HDR + s*SPAD              + w*16;
            }
            *(uint4 *)(sm + doff) = *(const uint4 *)(panel + soff);
        }
    }
    __syncthreads();

    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int sub_r = lane / LPR;             // extra row this lane covers (0 for NKS 8)
    const int s     = (lane - sub_r*LPR) >> 2;// slab within the tile
    const int g     = lane & 3;               // 8-element group inside the 32-elem slab row
    if (s >= nks) {
        return;                               // ragged K tail: these lanes own no slab
    }

    const uint8_t * sslab = sm + POL::HDR + s*SPAD;
    dst_t * ybase = y + (int64_t)(kb0 + s)*PXQ6_QK + g*8;

    #pragma unroll
    for (int it = 0; it < ITER; ++it) {
        const int row = (it*8 + warp)*RPW + sub_r;

        const float anch = POL::HDR ? POL::anchor(sm, row) : 0.f;
        float eff[POL::NEFF];
        POL::row_effs(sslab, row, anch, sub, eff);
        const int ei = (POL::NEFF == 4) ? g : (g >> 1);   // == (b*NEFF)>>4 for every b this lane owns
        float e = eff[0];
        #pragma unroll
        for (int t = 1; t < POL::NEFF; ++t) {
            if (t == ei) e = eff[t];
        }

        // Each lane owns pairs b in [4g, 4g+4), and POL::pair() reads exactly q[b>>2] == q[g]
        // (plus q[4], the hi-bit plane, on P6R). So one 4 B word is all this lane needs; every
        // slot is filled with it so POL::pair() can stay verbatim and q[] stays in registers
        // (a runtime index into q[] would push the array to local memory). This is the whole
        // difference between a 4 B and a 16 B per-lane smem read at identical arithmetic.
        const uint8_t * crow = sslab + POL::CODE_OFF + row*POL::CODE_BYTES;
        uint32_t q[POL::CODE_WORDS];
        const uint32_t cw = *(const uint32_t *)(crow + 4*g);
        #pragma unroll
        for (int t = 0; t < 4; ++t) q[t] = cw;
        if constexpr (POL::CODE_WORDS == 5) {
            q[4] = *(const uint32_t *)(crow + 16);   // P6R hi-bit plane
        }

        // The decoded values never reach a local array through a pointer: taking the address of
        // one would push it to local memory and cost more than the wide store buys. acc[] is
        // indexed only by unrolled constants, so it stays in registers, and the vector store is
        // assembled with make_uint4 / make_float4.
        float acc[8];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int    b = g*4 + u;                     // element-pair index, incumbent order
            const float2 v = POL::pair(q, b, tab);
            acc[2*u]     = e * v.x;
            acc[2*u + 1] = e * v.y;
        }

        dst_t * dptr = ybase + (int64_t)(p*PXQ6_BM + row)*K;
        if constexpr (std::is_same<dst_t, half>::value) {
            uint32_t w[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                // (dst_t)(x) on half is __float2half_rn: same snap as the incumbent's store.
                w[u] = (uint32_t) __half_as_ushort(__float2half_rn(acc[2*u]))
                     | ((uint32_t)__half_as_ushort(__float2half_rn(acc[2*u + 1])) << 16);
            }
            *(uint4 *)dptr = make_uint4(w[0], w[1], w[2], w[3]);
        } else if constexpr (std::is_same<dst_t, float>::value) {
            *(float4 *)(dptr)     = make_float4(acc[0], acc[1], acc[2], acc[3]);
            *(float4 *)(dptr + 4) = make_float4(acc[4], acc[5], acc[6], acc[7]);
        } else {
            #pragma unroll
            for (int u = 0; u < 8; ++u) dptr[u] = (dst_t) acc[u];
        }
    }
}

// K8 gates. Default OFF: the land is inert until the measure window flips it.
static inline bool pxa_dq_wide() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ_DQ_WIDE");
        const bool v = e && atoi(e) != 0;   // measured 2x slower than the incumbent — see above
        fprintf(stderr, "PXA_PXQ_DQ_WIDE: %s\n", v ? "ON (wide-store dequant)" : "OFF (incumbent dequant)");
        return v;
    }();
    return on;
}

static inline int pxa_dq_wide_variant() {
    static const int v = [](){
        const char * e = getenv("PXA_DQ_WIDE_VARIANT");
        int x = e ? atoi(e) : 0;
        if (x < 0 || x > 5) x = 0;
        return x;
    }();
    return v;
}

#define PXA_DQW_LAUNCH(NKS_, OCC_)                                                             \
    do {                                                                                       \
        const int kt = (kslabs + (NKS_) - 1) / (NKS_);                                         \
        k_pxq6_dequant_wide<POL, dst_t, NKS_, OCC_>                                            \
            <<<(unsigned)((int64_t)panels * kt), 256, 0, stream>>>(wq, y, kslabs, n_per_row, kt); \
    } while (0)

// `variant` < 0 reads PXA_DQ_WIDE_VARIANT; the correctness harness passes 0..5 explicitly so a
// single process can sweep every instantiation.
template <class POL, typename dst_t>
static void pxq6_launch_dequant_wide(const uint8_t * wq, dst_t * y, const int kslabs,
                                     const int64_t n_per_row, const int64_t nrows, cudaStream_t stream,
                                     int variant = -1) {
    const int panels = (int)(nrows / PXQ6_BM);
    switch (variant < 0 ? pxa_dq_wide_variant() : variant) {
        case 1:  PXA_DQW_LAUNCH(8, 6); break;
        case 2:  PXA_DQW_LAUNCH(8, 4); break;
        case 3:  PXA_DQW_LAUNCH(4, 8); break;
        case 4:  PXA_DQW_LAUNCH(4, 6); break;
        case 5:  PXA_DQW_LAUNCH(4, 4); break;
        default: PXA_DQW_LAUNCH(8, 8); break;
    }
}
#undef PXA_DQW_LAUNCH

template <typename dst_t>
static void dequantize_row_pxq6_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row,
                                     cudaStream_t stream) {
    if (nrows % PXQ6_BM != 0 || n_per_row % PXQ6_QK != 0) {
        fprintf(stderr, "FATAL: dequantize_row_pxq6_cuda: nrows=%lld n_per_row=%lld not slab-aligned\n",
                (long long)nrows, (long long)n_per_row);
        abort();
    }
    int dev = -1; cudaGetDevice(&dev);
    pxq6_maybe_upload_tables(dev);
    const int kslabs = (int)(n_per_row / PXQ6_QK);
    const int64_t nslabs = (nrows / PXQ6_BM) * (int64_t)kslabs;
    if (pxa_dq_wide()) {
        pxq6_launch_dequant_wide<pxq6_pol_p6, dst_t>((const uint8_t *)vx, y, kslabs, n_per_row, nrows, stream);
        return;
    }
    k_pxq6_dequant_matrix<pxq6_pol_p6, dst_t><<<nslabs, 64, 0, stream>>>((const uint8_t *)vx, y, kslabs, n_per_row);
}

template <typename dst_t>
static void dequantize_row_pxq6hq_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row,
                                       cudaStream_t stream) {
    if (nrows % PXQ6_BM != 0 || n_per_row % PXQ6_QK != 0) {
        fprintf(stderr, "FATAL: dequantize_row_pxq6hq_cuda: nrows=%lld n_per_row=%lld not slab-aligned\n",
                (long long)nrows, (long long)n_per_row);
        abort();
    }
    int dev = -1; cudaGetDevice(&dev);
    pxq6_maybe_upload_tables(dev);
    const int kslabs = (int)(n_per_row / PXQ6_QK);
    const int64_t nslabs = (nrows / PXQ6_BM) * (int64_t)kslabs;
    if (pxa_dq_wide()) {
        pxq6_launch_dequant_wide<pxq6_pol_p6hq, dst_t>((const uint8_t *)vx, y, kslabs, n_per_row, nrows, stream);
        return;
    }
    k_pxq6_dequant_matrix<pxq6_pol_p6hq, dst_t><<<nslabs, 64, 0, stream>>>((const uint8_t *)vx, y, kslabs, n_per_row);
}

template <typename dst_t>
static void dequantize_row_pxq6r_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row,
                                      cudaStream_t stream) {
    if (nrows % PXQ6_BM != 0 || n_per_row % PXQ6_QK != 0) {
        fprintf(stderr, "FATAL: dequantize_row_pxq6r_cuda: nrows=%lld n_per_row=%lld not slab-aligned\n",
                (long long)nrows, (long long)n_per_row);
        abort();
    }
    int dev = -1; cudaGetDevice(&dev);
    pxq6_maybe_upload_tables(dev);   // handles the shared SUB16 + the PXQ6R LM32 override
    const int kslabs = (int)(n_per_row / PXQ6_QK);
    const int64_t nslabs = (nrows / PXQ6_BM) * (int64_t)kslabs;
    if (pxa_dq_wide()) {
        pxq6_launch_dequant_wide<pxq6_pol_p6r, dst_t>((const uint8_t *)vx, y, kslabs, n_per_row, nrows, stream);
        return;
    }
    k_pxq6_dequant_matrix<pxq6_pol_p6r, dst_t><<<nslabs, 64, 0, stream>>>((const uint8_t *)vx, y, kslabs, n_per_row);
}

// =============================================================================================
// PXQ_CANON_v1 (2026-07-28): ONE fixed, shape-only summation order for every PXQ decode mmv.
//
// Before this, the unsplit kernels accumulated each k-segment lane as a single left-fold over
// the WHOLE K axis, while the S-split (ksplit_gen) forms folded within their K-chunk first and
// then folded the S chunk partials — a different fp32 rounding order, so PXA_PXQ4_2D_SPLIT /
// PXA_PXQ6_KSPLIT_GEN were documented "NOT bit-exact vs unsplit" and temp-0 output depended on
// the split decision (which itself depends on DEVICE SM COUNT — the same file could produce
// different greedy output on a P100 vs a V100). A near-tie token flip from that rounding
// difference can drop a marginal model into a repetition attractor, which is a real failure
// mode, not a cosmetic one.
//
// The fix: define the canonical value of every output element as
//     u = fold_{s=0..KSEG-1} ( fold_{c=0..NFIX-1} t[c][s] )
// where the K axis is cut into NFIX FIXED chunks (NFIX depends only on kslabs — never on S,
// device, or env), t[c][s] is the left-fold of lane s's dot32 chain INSIDE chunk c, and every
// fold is ascending. Every variant now computes exactly this:
//   - unsplit / K1 re-grid kernels: two-level per-lane fold in registers (same cost, no extra
//     syncs), cross-lane epilogue unchanged;
//   - S-split kernels: each block writes its RAW per-(chunk,lane) partials t[c][s] to the
//     workspace (the in-block cross-lane reduce is gone entirely); the reducer / FUSERED
//     finisher performs the one canonical fold. S-chunk boundaries are (kslabs*c)/S with S a
//     power of two dividing NFIX, so they coincide with fixed-chunk boundaries exactly
//     (floor(kslabs*c/S) == floor(kslabs*(c*NFIX/S)/NFIX) when NFIX/S is an integer).
// Result: split == unsplit == any S, bitwise, by construction. The workspace grows from S to
// NFIX*KSEG slots per row (<= 64), which only split shapes pay, and only in partial traffic.
// NOTE: this changes the canonical rounding vs pre-canon binaries (a one-time, documented
// re-baselining — PPL-verified unchanged; see docs/LEVERS.md PXA_PXQ4_2D_SPLIT row).
// =============================================================================================
// PXQ6_CANON_CMAX (2026-08-03) — the canonical chunk count, which is ALSO the split workspace
// width. NFIX is a shape-only function of kslabs (that is what makes split == unsplit on every
// device), but it is capped, and the cap costs real bandwidth: every split node writes
// NFIX*KSEG*R floats of raw partials and the reducer reads them all back, whatever S is.
//
// Decode profile, 122B PXQU48-core on 4x P100 at fill 8881 (128 tokens, nsys): the reduce
// kernels alone are 1.87 ms/token (k_pxq_mmv_reduce_s 1.157 + k_pxq6_gateup_reduce_gen 0.708)
// out of ~32.6, with a matching write side inside the mmv kernels -- and the driver never picks
// S > 4 on ANY node of this model, so at CMAX 16 three quarters of every workspace row is
// written and read for nothing.
//
// A cap of 4 leaves EVERY grid geometry of this model unchanged (the S the driver selects is
// already <= 4 everywhere: attn_gate S=2, ssm_out/attn_output/ffn_down_shexp S=4, the MoE
// gateup family S=4) while cutting the workspace 2-4x per node. It stays shape-only, so
// split == unsplit still holds by construction. It does change the canonical fp32 fold (fewer,
// longer chunks), so it is a build-time constant with a re-baselining cost, not an env flag.
#ifndef PXQ6_CANON_CMAX
#define PXQ6_CANON_CMAX 16
#endif
// fixed workspace row width for the K8-2D split (slots per (iy,j) row; S is capped at this)
#define PXQ6_MMV_SPLIT_MAX PXQ6_CANON_CMAX
// gateup family S cap (PXA_PXQ6_KSPLIT_GEN allows 2/4/8)
#define PXQ6_GU_SPLIT_MAX  (PXQ6_CANON_CMAX < 8 ? PXQ6_CANON_CMAX : 8)

static __host__ __device__ __forceinline__ int pxq6_canon_nfix(int kslabs, int cmax) {
    int lim = kslabs / PXQ4_MMV_KSEG;   // >= KSEG slabs per chunk so every lane stays busy
    if (lim < 1) lim = 1;
    if (lim > cmax) lim = cmax;
    int n = 1;
    while (n*2 <= lim) n *= 2;          // largest power of two <= lim (so any pow2 S <= n divides n)
    return n;
}

// ---------------------------------------------------------------------------------------------
// decode: fused up+gate+GLU mmv + down mmv — the generic family (grid/threads/smem identical
// to the proven k_pxq5_* pair; POL/PAIR/VECX select format + K2 variants).
// ---------------------------------------------------------------------------------------------
template <class POLU, class POLG, int MODE, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq6_gateup_mmv(const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg,
                  const char * __restrict__ x_base, const size_t x_tok_stride,
                  char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
                  const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                  const float * __restrict__ bias_u, const size_t bias_u_nb1,
                  const float * __restrict__ bias_g, const size_t bias_g_nb1,
                  const int R, const int K, const int n_as,
                  const int unary, const float alpha, const float limit) {
    const int p  = blockIdx.x;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    extern __shared__ float pxq6_smem[];
    float * xs = pxq6_smem;
    float * red = pxq6_smem + K;

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride);
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tabU[32], subU[16], tabG[32], subG[16];   // 32-entry tabs: P6R LM32
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : (pxq6_mode<MODE>::pairl3 ? 64 : 1)];
    POLU::stage_tabs(tabU, subU, threadIdx.x);
    POLG::stage_tabs(tabG, subG, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POLU>(plut, threadIdx.x, 256);
    if constexpr (pxq6_mode<MODE>::pairl3) pxq6_stage_pairlut3<POLU>(plut, threadIdx.x, 256);
    __syncthreads();
    pxq6_prmt_book pbU{}, pbG{};
    if constexpr (pxq6_mode<MODE>::prmt) { pxq6_prmt_build(tabU, pbU); pxq6_prmt_build(tabG, pbG); }

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const uint8_t * panU = pxq6_panel<POLU>(Wu, e, panels, p, kslabs);
    const uint8_t * panG = pxq6_panel<POLG>(Wg, e, panels, p, kslabs);
    const float anchU = POLU::HDR ? POLU::anchor(panU, row) : 0.f;
    const float anchG = POLG::HDR ? POLG::anchor(panG, row) : 0.f;

    // PXQ_CANON_v1: two-level fixed-chunk fold per lane (see the canon comment above) — the
    // per-lane value is bit-identical to the K1/S-split forms of this node by construction.
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_GU_SPLIT_MAX);
    float su = 0.f, sg = 0.f;
    for (int c = 0; c < nfix; ++c) {
        const int b0 = (kslabs*c)/nfix, b1 = (kslabs*(c+1))/nfix;
        float tu = 0.f, tg = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            const float * xk = xs + kb*PXQ6_QK;
            tu += pxq6_dot32<POLU, MODE, VECX>(panU + POLU::HDR + (size_t)kb*POLU::SLAB, row, anchU, xk, tabU, subU, plut, pbU);
            tg += pxq6_dot32<POLG, MODE, VECX>(panG + POLG::HDR + (size_t)kb*POLG::SLAB, row, anchG, xk, tabG, subG, plut, pbG);
        }
        su += tu;
        sg += tg;
    }
    red[(kseg*64 + row)]                      = su;
    red[(PXQ4_MMV_KSEG*64) + (kseg*64 + row)] = sg;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f, g = 0.f;
        #pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
            u += red[s*64 + row];
            g += red[PXQ4_MMV_KSEG*64 + s*64 + row];
        }
        const int grow = p*PXQ6_BM + row;
        if (bias_u) u += *(const float *)((const char *)bias_u + (size_t)e*bias_u_nb1 + (size_t)grow*sizeof(float));
        if (bias_g) g += *(const float *)((const char *)bias_g + (size_t)e*bias_g_nb1 + (size_t)grow*sizeof(float));
        const float r = pxq4_glu_apply(g, u, unary, alpha, limit);
        float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
        out[grow] = r;
    }
}

template <class POL, int MODE, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq6_mmv(const uint8_t * __restrict__ W,
           const char * __restrict__ x_base, const size_t x_tok_stride, const size_t x_slot_stride,
           char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
           const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
           const int R, const int K, const int n_as) {
    const int p  = blockIdx.x;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    extern __shared__ float pxq6_smem[];
    float * xs = pxq6_smem;
    float * red = pxq6_smem + K;

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride + (size_t)j*x_slot_stride);
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : (pxq6_mode<MODE>::pairl3 ? 64 : 1)];
    POL::stage_tabs(tab, sub, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POL>(plut, threadIdx.x, 256);
    if constexpr (pxq6_mode<MODE>::pairl3) pxq6_stage_pairlut3<POL>(plut, threadIdx.x, 256);
    __syncthreads();
    pxq6_prmt_book pb{};
    if constexpr (pxq6_mode<MODE>::prmt) pxq6_prmt_build(tab, pb);

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const uint8_t * pan = pxq6_panel<POL>(W, e, panels, p, kslabs);
    const float anch = POL::HDR ? POL::anchor(pan, row) : 0.f;

    // PXQ_CANON_v1: two-level fixed-chunk fold per lane — bit-identical to the K1/S-split forms.
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_MMV_SPLIT_MAX);
    float su = 0.f;
    for (int c = 0; c < nfix; ++c) {
        const int b0 = (kslabs*c)/nfix, b1 = (kslabs*(c+1))/nfix;
        float t = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            t += pxq6_dot32<POL, MODE, VECX>(pan + POL::HDR + (size_t)kb*POL::SLAB, row, anch,
                                             xs + kb*PXQ6_QK, tab, sub, plut, pb);
        }
        su += t;
    }
    red[kseg*64 + row] = su;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;
        #pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s*64 + row];
        float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
        out[p*PXQ6_BM + row] = u;
    }
}

// ---------------------------------------------------------------------------------------------
// G2-F1 REDFUSE (2026-07-19): the down mmv with k_pxq6_gateup_reduce + GLU absorbed into its
// x-staging prologue. Instead of reading the gateup dst, each block reconstructs it from the
// KSPLIT partial workspace with EXACTLY the reducer's arithmetic (fixed s = 0..KSEG-1 ascending
// summation, then bias, then pxq4_glu_apply) => xs[] == the dst the standalone reduce would have
// written, bit-for-bit; everything after staging is k_pxq6_mmv verbatim => the final output is
// bit-identical while one kernel launch + one dst HBM round-trip disappear. Redundant reduce
// compute is per-block (panels x), but ws is tiny (2*KSEG*K floats/slot) and L2-hot.
// Driver-guarded: only when the gateup dst is SOLE-consumed by this down MUL_MAT_ID, only for
// the bit-exact KSPLIT=1 form (declined under KSPLIT_GEN).
// ---------------------------------------------------------------------------------------------
template <class POL, int MODE, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq6_mmv_redfuse(const uint8_t * __restrict__ W,
                   const float * __restrict__ ws,      // KSPLIT partials; gateup R == our K
                   char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
                   const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                   const float * __restrict__ bias_u, const size_t bias_u_nb1,
                   const float * __restrict__ bias_g, const size_t bias_g_nb1,
                   const int R, const int K, const int n_as, const int n_ids,
                   const int unary, const float alpha, const float limit) {
    const int p  = blockIdx.x;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    extern __shared__ float pxq6_smem[];
    float * xs = pxq6_smem;
    float * red = pxq6_smem + K;

    // staging prologue = the reducer, verbatim arithmetic (grow -> idx, R_gateup -> K)
    const float * wsj = ws + ((size_t)iy*n_ids + j)*2*PXQ4_MMV_KSEG*K;
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
        float u = 0.f, g = 0.f;
        #pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
            u += wsj[(size_t)s*K + idx];
            g += wsj[((size_t)PXQ4_MMV_KSEG + s)*K + idx];
        }
        if (bias_u) u += *(const float *)((const char *)bias_u + (size_t)e*bias_u_nb1 + (size_t)idx*sizeof(float));
        if (bias_g) g += *(const float *)((const char *)bias_g + (size_t)e*bias_g_nb1 + (size_t)idx*sizeof(float));
        xs[idx] = pxq4_glu_apply(g, u, unary, alpha, limit);
    }

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : (pxq6_mode<MODE>::pairl3 ? 64 : 1)];
    POL::stage_tabs(tab, sub, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POL>(plut, threadIdx.x, 256);
    if constexpr (pxq6_mode<MODE>::pairl3) pxq6_stage_pairlut3<POL>(plut, threadIdx.x, 256);
    __syncthreads();
    pxq6_prmt_book pb{};
    if constexpr (pxq6_mode<MODE>::prmt) pxq6_prmt_build(tab, pb);

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const uint8_t * pan = pxq6_panel<POL>(W, e, panels, p, kslabs);
    const float anch = POL::HDR ? POL::anchor(pan, row) : 0.f;

    // PXQ_CANON_v1: same two-level fixed-chunk fold as k_pxq6_mmv.
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_MMV_SPLIT_MAX);
    float su = 0.f;
    for (int c = 0; c < nfix; ++c) {
        const int b0 = (kslabs*c)/nfix, b1 = (kslabs*(c+1))/nfix;
        float t = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            t += pxq6_dot32<POL, MODE, VECX>(pan + POL::HDR + (size_t)kb*POL::SLAB, row, anch,
                                             xs + kb*PXQ6_QK, tab, sub, plut, pb);
        }
        su += t;
    }
    red[kseg*64 + row] = su;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;
        #pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s*64 + row];
        float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
        out[p*PXQ6_BM + row] = u;
    }
}

// ---------------------------------------------------------------------------------------------
// K1 KSPLIT — bit-exact kseg-as-blocks form. grid (panels*KSEG, n_ids, Ny), 64 threads: thread
// = row, block owns ONE of the proven kernel's PXQ4_MMV_KSEG chains (kb ≡ kseg mod 4, ascending)
// => per-thread fp32 chains are IDENTICAL to the proven kernel's. Partials go to the persistent
// workspace; k_pxq6_gateup_reduce then replicates the proven red[] combination order (s = 0..3,
// ascending) + bias + GLU exactly => bit-identical dst.
// workspace layout: ws[((iy*n_ids + j)*2 + {0=u,1=g})*KSEG*R + kseg*R + grow]
// ---------------------------------------------------------------------------------------------
template <class POLU, class POLG, int MODE, bool VECX>
static __global__ void __launch_bounds__(64)
k_pxq6_gateup_mmv_ksplit(const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg,
                         const char * __restrict__ x_base, const size_t x_tok_stride,
                         float * __restrict__ ws,
                         const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                         const int R, const int K, const int n_as, const int n_ids) {
    const int pk = blockIdx.x;
    const int p    = pk / PXQ4_MMV_KSEG;
    const int kseg = pk % PXQ4_MMV_KSEG;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    extern __shared__ float pxq6_smem[];
    float * xs = pxq6_smem;                       // K floats (no red[] needed)

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride);
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tabU[32], subU[16], tabG[32], subG[16];   // 32-entry tabs: P6R LM32
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : (pxq6_mode<MODE>::pairl3 ? 64 : 1)];
    POLU::stage_tabs(tabU, subU, threadIdx.x);
    POLG::stage_tabs(tabG, subG, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POLU>(plut, threadIdx.x, 64);
    if constexpr (pxq6_mode<MODE>::pairl3) pxq6_stage_pairlut3<POLU>(plut, threadIdx.x, 64);
    __syncthreads();
    pxq6_prmt_book pbU{}, pbG{};
    if constexpr (pxq6_mode<MODE>::prmt) { pxq6_prmt_build(tabU, pbU); pxq6_prmt_build(tabG, pbG); }

    const int row = threadIdx.x;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const uint8_t * panU = pxq6_panel<POLU>(Wu, e, panels, p, kslabs);
    const uint8_t * panG = pxq6_panel<POLG>(Wg, e, panels, p, kslabs);
    const float anchU = POLU::HDR ? POLU::anchor(panU, row) : 0.f;
    const float anchG = POLG::HDR ? POLG::anchor(panG, row) : 0.f;

    // PXQ_CANON_v1: this block IS one lane; fold its chunks in the canonical order so the
    // per-lane partial equals the unsplit kernel's bit-for-bit.
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_GU_SPLIT_MAX);
    float su = 0.f, sg = 0.f;
    for (int c = 0; c < nfix; ++c) {
        const int b0 = (kslabs*c)/nfix, b1 = (kslabs*(c+1))/nfix;
        float tu = 0.f, tg = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            const float * xk = xs + kb*PXQ6_QK;
            tu += pxq6_dot32<POLU, MODE, VECX>(panU + POLU::HDR + (size_t)kb*POLU::SLAB, row, anchU, xk, tabU, subU, plut, pbU);
            tg += pxq6_dot32<POLG, MODE, VECX>(panG + POLG::HDR + (size_t)kb*POLG::SLAB, row, anchG, xk, tabG, subG, plut, pbG);
        }
        su += tu;
        sg += tg;
    }
    const int grow = p*PXQ6_BM + row;
    float * wsj = ws + ((size_t)iy*n_ids + j)*2*PXQ4_MMV_KSEG*R;
    wsj[(size_t)kseg*R + grow]                        = su;
    wsj[((size_t)PXQ4_MMV_KSEG + kseg)*R + grow]      = sg;
}

// reducer: one thread per (iy, j, grow). Sums the KSEG partials in the proven fixed order,
// adds biases, applies GLU, writes dst — byte-identical epilogue to the proven kernel.
static __global__ void k_pxq6_gateup_reduce(const float * __restrict__ ws,
        char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
        const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
        const float * __restrict__ bias_u, const size_t bias_u_nb1,
        const float * __restrict__ bias_g, const size_t bias_g_nb1,
        const int R, const int n_as, const int n_ids,
        const int unary, const float alpha, const float limit) {
    const int grow = blockIdx.x*blockDim.x + threadIdx.x;
    if (grow >= R) return;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;   // SER slot: dst untouched (matches the proven kernel)
    const float * wsj = ws + ((size_t)iy*n_ids + j)*2*PXQ4_MMV_KSEG*R;
    float u = 0.f, g = 0.f;
    #pragma unroll
    for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
        u += wsj[(size_t)s*R + grow];
        g += wsj[((size_t)PXQ4_MMV_KSEG + s)*R + grow];
    }
    if (bias_u) u += *(const float *)((const char *)bias_u + (size_t)e*bias_u_nb1 + (size_t)grow*sizeof(float));
    if (bias_g) g += *(const float *)((const char *)bias_g + (size_t)e*bias_g_nb1 + (size_t)grow*sizeof(float));
    const float r = pxq4_glu_apply(g, u, unary, alpha, limit);
    float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
    out[grow] = r;
}

// ---------------------------------------------------------------------------------------------
// K1-2D KSPLIT — single-weight twin of k_pxq6_gateup_mmv_ksplit, for the plain 2D (non-MoE)
// MUL_MAT driver. grid (panels*KSEG, n_ids, Ny), 64 threads: thread = row, block owns ONE of
// k_pxq6_mmv's PXQ4_MMV_KSEG chains (kb ≡ kseg mod KSEG, ascending) => the per-thread fp32 chain
// is byte-identical to k_pxq6_mmv's. k_pxq_mmv_reduce then replays k_pxq6_mmv's red[] combination
// order (s = 0..KSEG-1 ascending) => bit-identical dst. Warps in flight are unchanged vs the
// 256-thr form (8 per panel either way); what changes is that they spread over KSEG x more blocks,
// which is what an under-filled launch (panels < n_SM) needs.
// workspace layout: ws[((iy*n_ids + j)*KSEG + kseg)*R + grow]
// ---------------------------------------------------------------------------------------------
template <class POL, int MODE, bool VECX>
static __global__ void __launch_bounds__(64)
k_pxq6_mmv_ksplit(const uint8_t * __restrict__ W,
                  const char * __restrict__ x_base, const size_t x_tok_stride, const size_t x_slot_stride,
                  float * __restrict__ ws,
                  const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                  const int R, const int K, const int n_as, const int n_ids) {
    const int pk = blockIdx.x;
    const int p    = pk / PXQ4_MMV_KSEG;
    const int kseg = pk % PXQ4_MMV_KSEG;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    extern __shared__ float pxq6_smem[];
    float * xs = pxq6_smem;                       // K floats (no red[] in the 64-thr form)

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride + (size_t)j*x_slot_stride);
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : (pxq6_mode<MODE>::pairl3 ? 64 : 1)];
    POL::stage_tabs(tab, sub, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POL>(plut, threadIdx.x, 64);
    if constexpr (pxq6_mode<MODE>::pairl3) pxq6_stage_pairlut3<POL>(plut, threadIdx.x, 64);
    __syncthreads();
    pxq6_prmt_book pb{};
    if constexpr (pxq6_mode<MODE>::prmt) pxq6_prmt_build(tab, pb);

    const int row = threadIdx.x;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const uint8_t * pan = pxq6_panel<POL>(W, e, panels, p, kslabs);
    const float anch = POL::HDR ? POL::anchor(pan, row) : 0.f;

    // PXQ_CANON_v1: this block IS one lane; canonical two-level fold => per-lane partial equals
    // the unsplit kernel's bit-for-bit.
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_MMV_SPLIT_MAX);
    float su = 0.f;
    for (int c = 0; c < nfix; ++c) {
        const int b0 = (kslabs*c)/nfix, b1 = (kslabs*(c+1))/nfix;
        float t = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            t += pxq6_dot32<POL, MODE, VECX>(pan + POL::HDR + (size_t)kb*POL::SLAB, row, anch,
                                             xs + kb*PXQ6_QK, tab, sub, plut, pb);
        }
        su += t;
    }
    float * wsj = ws + ((size_t)iy*n_ids + j)*PXQ4_MMV_KSEG*R;
    wsj[(size_t)kseg*R + p*PXQ6_BM + row] = su;
}

// 2D reducer: one thread per (iy, j, grow). Sums the KSEG partials in k_pxq6_mmv's fixed ascending
// order and writes dst — byte-identical to k_pxq6_mmv's kseg == 0 epilogue (no bias, no GLU).
static __global__ void k_pxq_mmv_reduce(const float * __restrict__ ws,
        char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
        const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
        const int R, const int n_as, const int n_ids) {
    const int grow = blockIdx.x*blockDim.x + threadIdx.x;
    if (grow >= R) return;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;   // SER slot: dst untouched (matches the proven kernel)
    const float * wsj = ws + ((size_t)iy*n_ids + j)*PXQ4_MMV_KSEG*R;
    float u = 0.f;
    #pragma unroll
    for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += wsj[(size_t)s*R + grow];
    float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
    out[grow] = u;
}

// (PXQ6_MMV_SPLIT_MAX moved up to the PXQ_CANON_v1 block)

// K8-2D S-SPLIT (2026-07-26): the single-weight 256-thr S-way twin of the gateup gen form
// below, built for the 2D decode mmv's two pathological launch shapes (measured, nvprof
// gpu-trace, Laguna-XS rev-2 backbone, 2xP100):
//   R=2048 K=8192 (attn_output / ffn_down): 32 blocks + 33.8 KB dyn smem = 1 block/SM on 32 of
//     56 SMs -> 87 GB/s achieved vs the SAME codec's 210 GB/s where the grid is healthy;
//   R=512 K=2048 (ffn_{up,gate}_shexp): 8 blocks total -> 20 GB/s.
// Where the plain grid already covers the machine (R=8192 K=2048 attn_q: 128 blocks, 9 KB) the
// UNSPLIT kernel measures 210 GB/s vs mmvq-MXFP4's 161 GB/s on the same tensor -- the dot path
// is not the deficit, the launch geometry is. Block pk = (panel p, chunk): the full 64x4 kseg
// structure INSIDE K-chunk [kb0,kb1); stages only its x slice (dyn smem Kc) so smem stops
// capping occupancy at large K.
// PXQ_CANON_v1 (2026-07-28): each block now writes its RAW per-(fixed-chunk, lane) partials
// t[c][s] to the workspace (NFIX*KSEG slots per row) and the reducer performs the ONE canonical
// fold -- so this kernel is bit-exact vs the unsplit kernel by construction, at any S, on any
// device. The old in-block cross-lane reduce (and its smem red[] + syncthreads) is gone.
// FUSERED (2026-07-26): when `counters` is non-null the LAST block of each (iy,j) performs the
// final canonical fold in-kernel (threadFenceReduction fence-and-flag pattern) and the driver
// skips the k_pxq_mmv_reduce_s launch (default OFF -- measured loss, kept as documented dead end).
template <class POL, int MODE, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq6_mmv_ksplit_gen(const uint8_t * __restrict__ W,
                      const char * __restrict__ x_base, const size_t x_tok_stride, const size_t x_slot_stride,
                      float * __restrict__ ws,
                      const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                      char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
                      int * __restrict__ counters,
                      const int R, const int K, const int n_as, const int n_ids, const int S) {
    const int pk = blockIdx.x;
    const int p     = pk / S;
    const int chunk = pk % S;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    const int kslabs = K / PXQ6_QK;
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_MMV_SPLIT_MAX);   // S divides nfix (driver-guaranteed)
    const int cpb  = nfix/S;                                        // fixed chunks per block
    const int c0 = chunk*cpb, c1 = c0 + cpb;
    // S-chunk boundaries == the enclosing fixed-chunk boundaries (floor((kslabs*c0)/nfix) ==
    // floor((kslabs*chunk)/S) exactly, since nfix/S is an integer)
    const int kb0 = (kslabs*c0)/nfix, kb1 = (kslabs*c1)/nfix;
    const int Kc  = (kb1 - kb0)*PXQ6_QK;

    extern __shared__ float pxq6_smem[];
    float * xs  = pxq6_smem;                      // Kc floats (chunk slice only)

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride + (size_t)j*x_slot_stride) + kb0*PXQ6_QK;
    for (int idx = threadIdx.x; idx < Kc; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : (pxq6_mode<MODE>::pairl3 ? 64 : 1)];
    POL::stage_tabs(tab, sub, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POL>(plut, threadIdx.x, 256);
    if constexpr (pxq6_mode<MODE>::pairl3) pxq6_stage_pairlut3<POL>(plut, threadIdx.x, 256);
    __syncthreads();
    pxq6_prmt_book pb{};
    if constexpr (pxq6_mode<MODE>::prmt) pxq6_prmt_build(tab, pb);

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ6_BM;
    const uint8_t * pan = pxq6_panel<POL>(W, e, panels, p, kslabs);
    const float anch = POL::HDR ? POL::anchor(pan, row) : 0.f;

    float * wsj = ws + ((size_t)iy*n_ids + j)*(size_t)(PXQ6_MMV_SPLIT_MAX*PXQ4_MMV_KSEG)*R;
    for (int c = c0; c < c1; ++c) {
        const int b0 = (kslabs*c)/nfix, b1 = (kslabs*(c+1))/nfix;
        float t = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            t += pxq6_dot32<POL, MODE, VECX>(pan + POL::HDR + (size_t)kb*POL::SLAB, row, anch,
                                             xs + (kb - kb0)*PXQ6_QK, tab, sub, plut, pb);
        }
        wsj[(size_t)(c*PXQ4_MMV_KSEG + kseg)*R + p*PXQ6_BM + row] = t;
    }
    if (counters == nullptr) return;                 // driver launches k_pxq_mmv_reduce_s instead
    __threadfence();                                 // ws writes visible before the flag
    __syncthreads();
    __shared__ int lastblk;
    if (threadIdx.x == 0) {
        const int done = atomicAdd(&counters[iy*n_ids + j], 1) + 1;
        lastblk = (done == (int)gridDim.x);
        if (lastblk) counters[iy*n_ids + j] = 0;     // self-clean for the next node on the stream
    }
    __syncthreads();
    if (!lastblk) return;
    // final reduction, SAME canonical order as k_pxq_mmv_reduce_s => identical values
    const float * wsj2 = wsj;
    float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
    for (int grow = threadIdx.x; grow < R; grow += blockDim.x) {
        float u = 0.f;
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
            float sus = 0.f;
            for (int c = 0; c < nfix; ++c) sus += wsj2[(size_t)(c*PXQ4_MMV_KSEG + s)*R + grow];
            u += sus;
        }
        out[grow] = u;
    }
}

// canonical reducer for k_pxq6_mmv_ksplit_gen: performs the ONE canonical fold (lane-major,
// fixed chunks ascending inside each lane) over the raw t[c][s] partials and writes dst.
// Bit-identical to the unsplit kernel's epilogue by construction. One thread per (iy, j, grow).
static __global__ void k_pxq_mmv_reduce_s(const float * __restrict__ ws,
        char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
        const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
        const int R, const int n_as, const int n_ids, const int kslabs) {
    const int grow = blockIdx.x*blockDim.x + threadIdx.x;
    if (grow >= R) return;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;   // SER slot: dst untouched (matches the proven kernel)
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_MMV_SPLIT_MAX);
    const float * wsj = ws + ((size_t)iy*n_ids + j)*(size_t)(PXQ6_MMV_SPLIT_MAX*PXQ4_MMV_KSEG)*R;
    float u = 0.f;
    for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
        float sus = 0.f;
        for (int c = 0; c < nfix; ++c) sus += wsj[(size_t)(c*PXQ4_MMV_KSEG + s)*R + grow];
        u += sus;
    }
    float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
    out[grow] = u;
}

// PXA_PXQ_MMV_TOK (2026-08-30): TOK-batched twin of k_pxq6_mmv_ksplit_gen — the dense-2D /
// MoE-down mmv arm of the PXA_PXQ_TOKBATCH recipe (gateup twin below at pxq6_gu_tok_pass).
// The per-token kernel carries the token on blockIdx.z, so an ny-token verify batch or an
// ny-request concurrent decode re-streams every weight panel from DRAM once PER TOKEN. This
// twin folds TOK consecutive tokens into one block: grid.z = ceil(ny/TOK), the TOK x
// chunk-slices are staged token-major in dynamic smem (TOK*Kc floats), and the token loop
// sits INSIDE the (kb, row) weight walk so tokens 2..TOK hit L1/registers instead of DRAM.
// BIT-EXACT by construction: per token t the (fixed-chunk, lane) fp32 chain is the per-token
// kernel's chain verbatim (same canon_v1 chunks, same kb order, same pxq6_dot32 sourcing;
// the acc[t] chains are independent, so the t-inner loop cannot change any fold), and it
// writes the SAME raw-partial ws slots; the canonical reducer (k_pxq_mmv_reduce_s) is reused
// unchanged. temp-0 sha vs the per-token arm is a valid gate. Register discipline follows the
// gateup twin: __launch_bounds__(256, PXA_PXQ_TOK_MINBLK); single-matrix walk keeps TOK
// accumulators + TOK panel pointers + TOK anchors live (roomier than the gateup U+G pair, so
// TOK=4 is offered here where the gateup arm caps at 3 — gate fired instantiations with
// cuobjdump -res-usage all the same). FUSERED is not offered on this arm (reducer always
// runs). A dead token slot (iy >= ny) rides e = -1: no staging, no dots, no ws writes.
// Default OFF (PXA_PXQ_MMV_TOK unset/0).
#ifndef PXA_PXQ_TOK_MINBLK
#define PXA_PXQ_TOK_MINBLK 4
#endif
template <class POL, int MODE, bool VECX, int TOK>
static __global__ void __launch_bounds__(256, PXA_PXQ_TOK_MINBLK)
k_pxq6_mmv_ksplit_gen_tok(const uint8_t * __restrict__ W,
                      const char * __restrict__ x_base, const size_t x_tok_stride, const size_t x_slot_stride,
                      float * __restrict__ ws,
                      const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                      const int R, const int K, const int n_as, const int n_ids, const int S,
                      const int ny) {
    const int pk = blockIdx.x;
    const int p     = pk / S;
    const int chunk = pk % S;
    const int j   = blockIdx.y;
    const int iy0 = (int)blockIdx.z * TOK;

    const int kslabs = K / PXQ6_QK;
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_MMV_SPLIT_MAX);   // S divides nfix (driver-guaranteed)
    const int cpb  = nfix/S;                                        // fixed chunks per block
    const int c0 = chunk*cpb, c1 = c0 + cpb;
    const int kb0 = (kslabs*c0)/nfix, kb1 = (kslabs*c1)/nfix;
    const int Kc  = (kb1 - kb0)*PXQ6_QK;

    extern __shared__ float pxq6_smem[];
    float * xs  = pxq6_smem;                      // TOK * Kc floats, token-major

    int e_t[TOK];
    #pragma unroll
    for (int t = 0; t < TOK; ++t) {
        const int iy = iy0 + t;
        e_t[t] = iy < ny ? *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0) : -1;
        if (e_t[t] >= n_as) e_t[t] = -1;
        if (e_t[t] >= 0) {
            const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride + (size_t)j*x_slot_stride) + kb0*PXQ6_QK;
            for (int idx = threadIdx.x; idx < Kc; idx += blockDim.x) xs[(size_t)t*Kc + idx] = x[idx];
        }
    }

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : (pxq6_mode<MODE>::pairl3 ? 64 : 1)];
    POL::stage_tabs(tab, sub, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POL>(plut, threadIdx.x, 256);
    if constexpr (pxq6_mode<MODE>::pairl3) pxq6_stage_pairlut3<POL>(plut, threadIdx.x, 256);
    __syncthreads();
    pxq6_prmt_book pb{};
    if constexpr (pxq6_mode<MODE>::prmt) pxq6_prmt_build(tab, pb);

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ6_BM;
    const uint8_t * pan_t[TOK];
    float anch_t[TOK];
    #pragma unroll
    for (int t = 0; t < TOK; ++t) {
        pan_t[t]  = e_t[t] >= 0 ? pxq6_panel<POL>(W, e_t[t], panels, p, kslabs) : W;
        anch_t[t] = (e_t[t] >= 0 && POL::HDR) ? POL::anchor(pan_t[t], row) : 0.f;
    }

    for (int c = c0; c < c1; ++c) {
        const int b0 = (kslabs*c)/nfix, b1 = (kslabs*(c+1))/nfix;
        float acc[TOK];
        #pragma unroll
        for (int t = 0; t < TOK; ++t) acc[t] = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            #pragma unroll
            for (int t = 0; t < TOK; ++t) {
                if (e_t[t] < 0) continue;
                acc[t] += pxq6_dot32<POL, MODE, VECX>(pan_t[t] + POL::HDR + (size_t)kb*POL::SLAB, row,
                            anch_t[t], xs + (size_t)t*Kc + (size_t)(kb - kb0)*PXQ6_QK, tab, sub, plut, pb);
            }
        }
        #pragma unroll
        for (int t = 0; t < TOK; ++t) {
            if (e_t[t] < 0) continue;
            float * wsj = ws + ((size_t)(iy0 + t)*n_ids + j)*(size_t)(PXQ6_MMV_SPLIT_MAX*PXQ4_MMV_KSEG)*R;
            wsj[(size_t)(c*PXQ4_MMV_KSEG + kseg)*R + p*PXQ6_BM + row] = acc[t];
        }
    }
}

// K1b generic S-split: 256-thr blocks, block pk handles its S-chunk with the full 64x4 kseg
// structure INSIDE the chunk. Stages only its K-chunk slice of x (dyn smem K/S).
// PXQ_CANON_v1 (2026-07-28): writes RAW per-(fixed-chunk, lane) partials (2 x NFIX*KSEG slots
// per row: u then g) and the reducer performs the ONE canonical fold + bias + GLU -- bit-exact
// vs the unsplit k_pxq6_gateup_mmv by construction, at any S, on any device.
// FUSERED: non-null `counters` => the last block of each (iy,j) applies the canonical fold +
// bias + GLU in-kernel (fence-and-flag) and the driver skips the standalone reduce launch.
// R1 (2026-08-04): pin the FIRED instantiations to 4 blocks/SM.
// The variants that actually launch in production are TAB (MODE 0) x VECX=1. canon_v2 pushed
// them 64 -> 74/75 regs, crossing the 65-reg cliff (65536/(256*4) = 64 exactly) and costing
// 25% of resident warps on 40 of 48 layers. Asking for min 4 blocks brings them back to <=64.
// NOT the killed cap-forcing experiment: that forced 36/32 regs (10-45 below natural) and
// spilled (-27.7%). This asks -10 from 74, and any LOCAL>0 in res-usage aborts the change.
// Only the fired shape is pinned; REFERENCE-only VECX=0 variants keep their natural allocation.
#ifndef PXQ_GU_MINBLK
#define PXQ_GU_MINBLK 0   // MEASURED NULL 2026-08-04: +0.14% median. Default OFF.
#endif
template <class POLU, class POLG, int MODE, bool VECX>
#if PXQ_GU_MINBLK > 0
static __global__ void __launch_bounds__(256, PXQ_GU_MINBLK)
#else
// No second argument: __launch_bounds__(256, 1) is NOT equivalent to __launch_bounds__(256).
// Declaring minBlocksPerMultiprocessor=1 lifts the register ceiling and ptxas spends 121 regs
// (2 blocks/SM) instead of the unhinted 64 (4 blocks/SM). Disabling the lever must restore the
// original declaration exactly.
static __global__ void __launch_bounds__(256)
#endif
k_pxq6_gateup_mmv_ksplit_gen(const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg,
                             const char * __restrict__ x_base, const size_t x_tok_stride,
                             float * __restrict__ ws,
                             const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                             char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
                             const float * __restrict__ bias_u, const size_t bias_u_nb1,
                             const float * __restrict__ bias_g, const size_t bias_g_nb1,
                             int * __restrict__ counters,
                             const int unary, const float alpha, const float limit,
                             const int R, const int K, const int n_as, const int n_ids, const int S) {
    const int pk = blockIdx.x;
    const int p     = pk / S;
    const int chunk = pk % S;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    const int kslabs = K / PXQ6_QK;
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_GU_SPLIT_MAX);   // S divides nfix (driver-guaranteed)
    const int cpb  = nfix/S;                                       // fixed chunks per block
    const int c0 = chunk*cpb, c1 = c0 + cpb;
    const int kb0 = (kslabs*c0)/nfix, kb1 = (kslabs*c1)/nfix;      // == the old (kslabs*chunk)/S bounds
    const int Kc  = (kb1 - kb0)*PXQ6_QK;

    extern __shared__ float pxq6_smem[];
    float * xs  = pxq6_smem;                      // Kc floats (chunk slice only)

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride) + kb0*PXQ6_QK;
    for (int idx = threadIdx.x; idx < Kc; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tabU[32], subU[16], tabG[32], subG[16];   // 32-entry tabs: P6R LM32
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : (pxq6_mode<MODE>::pairl3 ? 64 : 1)];
    POLU::stage_tabs(tabU, subU, threadIdx.x);
    POLG::stage_tabs(tabG, subG, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POLU>(plut, threadIdx.x, 256);
    if constexpr (pxq6_mode<MODE>::pairl3) pxq6_stage_pairlut3<POLU>(plut, threadIdx.x, 256);
    __syncthreads();
    pxq6_prmt_book pbU{}, pbG{};
    if constexpr (pxq6_mode<MODE>::prmt) { pxq6_prmt_build(tabU, pbU); pxq6_prmt_build(tabG, pbG); }

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ6_BM;
    const uint8_t * panU = pxq6_panel<POLU>(Wu, e, panels, p, kslabs);
    const uint8_t * panG = pxq6_panel<POLG>(Wg, e, panels, p, kslabs);
    const float anchU = POLU::HDR ? POLU::anchor(panU, row) : 0.f;
    const float anchG = POLG::HDR ? POLG::anchor(panG, row) : 0.f;

    // PXQ_CANON_v1: raw per-(fixed-chunk, lane) partials to ws; no in-block cross-lane reduce.
    const size_t guslots = (size_t)(PXQ6_GU_SPLIT_MAX*PXQ4_MMV_KSEG);   // 32 slots per half
    float * wsj = ws + ((size_t)iy*n_ids + j)*2*guslots*R;
    const int grow = p*PXQ6_BM + row;
    for (int c = c0; c < c1; ++c) {
        const int b0 = (kslabs*c)/nfix, b1 = (kslabs*(c+1))/nfix;
        float tu = 0.f, tg = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            const float * xk = xs + (kb - kb0)*PXQ6_QK;
            tu += pxq6_dot32<POLU, MODE, VECX>(panU + POLU::HDR + (size_t)kb*POLU::SLAB, row, anchU, xk, tabU, subU, plut, pbU);
            tg += pxq6_dot32<POLG, MODE, VECX>(panG + POLG::HDR + (size_t)kb*POLG::SLAB, row, anchG, xk, tabG, subG, plut, pbG);
        }
        wsj[(size_t)(c*PXQ4_MMV_KSEG + kseg)*R + grow]           = tu;
        wsj[(guslots + (size_t)(c*PXQ4_MMV_KSEG + kseg))*R + grow] = tg;
    }
    if (counters == nullptr) return;                 // driver launches k_pxq6_gateup_reduce_gen instead
    __threadfence();
    __syncthreads();
    __shared__ int lastblk;
    if (threadIdx.x == 0) {
        const int done = atomicAdd(&counters[iy*n_ids + j], 1) + 1;
        lastblk = (done == (int)gridDim.x);
        if (lastblk) counters[iy*n_ids + j] = 0;
    }
    __syncthreads();
    if (!lastblk) return;
    const float * wsj2 = wsj;
    float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
    for (int gr = threadIdx.x; gr < R; gr += blockDim.x) {
        float u = 0.f, g = 0.f;
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
            float sus = 0.f, sgs = 0.f;
            for (int c = 0; c < nfix; ++c) {
                sus += wsj2[(size_t)(c*PXQ4_MMV_KSEG + s)*R + gr];
                sgs += wsj2[(guslots + (size_t)(c*PXQ4_MMV_KSEG + s))*R + gr];
            }
            u += sus;
            g += sgs;
        }
        if (bias_u) u += *(const float *)((const char *)bias_u + (size_t)e*bias_u_nb1 + (size_t)gr*sizeof(float));
        if (bias_g) g += *(const float *)((const char *)bias_g + (size_t)e*bias_g_nb1 + (size_t)gr*sizeof(float));
        out[gr] = pxq4_glu_apply(g, u, unary, alpha, limit);
    }
}


// =============================================================================================
// Q1 — PXA_PXQ6_QPF (2026-08-04): one-slab-lookahead register prefetch twins of the fired
// decode kernels. WHY: with SHFL shipped, the residual term is EXPOSED PER-WARP DRAM LATENCY,
// not a throughput pipe — every issue-side lever pays ~0.2:1 (SHFL -18% loop instructions ->
// +4.2%; PAIRL3 ~-160 instructions -> +2.7%; +33% occupancy -> +0.14% null), and the SASS
// shows the code LDG consumed by the IMMEDIATELY next instruction with zero cross-iteration
// lookahead: each warp eats the full DRAM latency once per slab, serially. These twins issue
// slab i+1's code words + scale byte(s) BEFORE slab i's unpack/FMA chain, so the latency
// overlaps compute that was being issued anyway (the decode analog of the GEMM K5 PIPE
// prologue staging in k_pxq6_gemm_grouped).
//
// BIT-EXACT BY CONSTRUCTION: the (c, kb) walk order, per-chunk fold boundaries, per-lane fp32
// chains, ws slots, and epilogues are copied from the incumbents verbatim; pxq6_dot32_qp
// computes pxq6_dot32's arithmetic on operands loaded EARLIER from the same read-only
// addresses. Only load issue time changes => temp-0 sha must match the incumbents.
//
// WALK GUARANTEE the pipeline leans on: pxq6_canon_nfix caps nfix at kslabs/PXQ4_MMV_KSEG,
// so every fixed chunk spans >= KSEG slabs and every (chunk, kseg lane) runs >= 1 iteration
// — the successor step below never has to skip an empty chunk, and every ws slot the
// incumbent writes is written here (same value, same order).
//
// REGISTER DISCIPLINE: prefetch state = CODE_WORDS+1 regs per tensor per stage (P2: 3, P6: 5,
// P6R: 6; gateup doubles it). 256 threads, plain __launch_bounds__(256) — NEVER (256, 1):
// the neutral-looking minBlocksPerMultiprocessor=1 lifts the register ceiling (74->121 regs
// happened once). GATE every fired instantiation with cuobjdump -res-usage before GPU time.
// =============================================================================================

// per-(lane, slab) weight-side state: code words + packed scale byte(s), loaded EARLY
template <class POL>
struct pxq6_pfslab {
    uint32_t q[POL::CODE_WORDS];
    uint32_t sb;   // 1 B scale byte (NEFF 2) / LE u16 byte pair (NEFF 4)
};

template <class POL, bool CS>
static __device__ __forceinline__ void pxq6_qp_ld(const uint8_t * __restrict__ pan, int kb, int row,
                                                  pxq6_pfslab<POL> & st) {
    const uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
    pxq6_ldcodes<POL, CS>(slab + POL::CODE_OFF + row*POL::CODE_BYTES, st.q);
    if constexpr (POL::NEFF == 4) st.sb = *(const uint16_t *)(slab + 2*row);   // 2-B aligned (HDR/SLAB even)
    else                          st.sb = slab[row];
}

// successor of (c, kb) in the canonical chunk walk; false when exhausted. Single-step chunk
// advance is sufficient (no empty chunks — see the walk guarantee above).
static __device__ __forceinline__ bool pxq6_qp_next(int & c, int & kb, int & bnd,
                                                    const int c_end, const int kslabs,
                                                    const int nfix, const int kseg) {
    kb += PXQ4_MMV_KSEG;
    if (kb < bnd) return true;
    if (++c >= c_end) return false;
    kb  = (kslabs*c)/nfix + kseg;
    bnd = (kslabs*(c+1))/nfix;
    return true;
}

// pxq6_dot32 on a pre-loaded pxq6_pfslab: the eff computation replicates every policy's
// row_effs shape exactly (all NEFF==2 policies share the 1-byte SUB16 form; NEFF==4 is the
// pxq6_pol_p6hq 2-byte SUB8 form), and the pair loop is pxq6_dot32's verbatim with q -> st.q.
// Same operands, same order => bit-exact vs pxq6_dot32 for every MODE (incl. the K2d bv stage).
template <class POL, int MODE, bool VECX>
static __device__ __forceinline__ float pxq6_dot32_qp(const pxq6_pfslab<POL> & st, float anch,
                                                      const float * __restrict__ xk,
                                                      const float * __restrict__ tab,
                                                      const float * __restrict__ sub,
                                                      const float2 * __restrict__ plut,
                                                      const pxq6_prmt_book & pb) {
    using M = pxq6_mode<MODE>;
    float eff[POL::NEFF];
    if constexpr (POL::NEFF == 4) {
        const int sb0 = (int)(st.sb & 0xff), sb1 = (int)(st.sb >> 8);
        eff[0] = anch * sub[sb0 & 0xf];
        eff[1] = anch * sub[sb0 >> 4];
        eff[2] = anch * sub[sb1 & 0xf];
        eff[3] = anch * sub[sb1 >> 4];
    } else {
        const int sb = (int)st.sb;
        eff[0] = anch * sub[sb & 0xf];
        eff[1] = anch * sub[sb >> 4];
    }
    // K2d (MODE SHFL): same bv staging as pxq6_dot32 — dead code for every other mode.
    float bv = 0.f;
    if constexpr (M::shfl) bv = tab[threadIdx.x & 31];
    float t[POL::NEFF];
    #pragma unroll
    for (int i = 0; i < POL::NEFF; ++i) t[i] = 0.f;
    if (VECX) {
        #pragma unroll
        for (int b = 0; b < 16; b += 2) {
            const float4 xv = *(const float4 *)&xk[2*b];
            const float2 p0 = pxq6_pairx<POL, MODE>(st.q, b,   tab, plut, pb, bv);
            const float2 p1 = pxq6_pairx<POL, MODE>(st.q, b+1, tab, plut, pb, bv);
            t[(b*POL::NEFF) >> 4]     = pxq6_acc2(t[(b*POL::NEFF) >> 4],     p0.x, xv.x, p0.y, xv.y);
            t[((b+1)*POL::NEFF) >> 4] = pxq6_acc2(t[((b+1)*POL::NEFF) >> 4], p1.x, xv.z, p1.y, xv.w);
        }
    } else {
        #pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float2 p = pxq6_pairx<POL, MODE>(st.q, b, tab, plut, pb, bv);
            t[(b*POL::NEFF) >> 4] = pxq6_acc2(t[(b*POL::NEFF) >> 4], p.x, xk[2*b], p.y, xk[2*b+1]);
        }
    }
    if (POL::NEFF == 1) return eff[0]*t[0];
    if (POL::NEFF == 2) return eff[0]*t[0] + eff[1]*t[1];
    return (eff[0]*t[0] + eff[1]*t[1]) + (eff[2]*t[2] + eff[3]*t[3]);
}

// Q1 twin of k_pxq6_mmv: identical grid/smem/prologue/epilogue; only the chunk walk is
// software-pipelined one slab deep.
template <class POL, int MODE, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq6_mmv_qp(const uint8_t * __restrict__ W,
              const char * __restrict__ x_base, const size_t x_tok_stride, const size_t x_slot_stride,
              char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
              const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
              const int R, const int K, const int n_as) {
    const int p  = blockIdx.x;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    extern __shared__ float pxq6_smem[];
    float * xs = pxq6_smem;
    float * red = pxq6_smem + K;

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride + (size_t)j*x_slot_stride);
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : (pxq6_mode<MODE>::pairl3 ? 64 : 1)];
    POL::stage_tabs(tab, sub, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POL>(plut, threadIdx.x, 256);
    if constexpr (pxq6_mode<MODE>::pairl3) pxq6_stage_pairlut3<POL>(plut, threadIdx.x, 256);
    __syncthreads();
    pxq6_prmt_book pb{};
    if constexpr (pxq6_mode<MODE>::prmt) pxq6_prmt_build(tab, pb);

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const uint8_t * pan = pxq6_panel<POL>(W, e, panels, p, kslabs);
    const float anch = POL::HDR ? POL::anchor(pan, row) : 0.f;

    // PXQ_CANON_v1 walk, Q1-pipelined: same (c, kb) sequence, same per-chunk fold order,
    // same dot32 chains — only WHEN the read-only weight-side loads issue changes.
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_MMV_SPLIT_MAX);
    float su = 0.f;
    {
        int c = 0, kb = kseg, bnd = kslabs/nfix;
        pxq6_pfslab<POL> cur, nxt;
        pxq6_qp_ld<POL, pxq6_mode<MODE>::cs>(pan, kb, row, cur);
        float t = 0.f;
        for (;;) {
            int cn = c, kbn = kb, bn = bnd;
            const bool more = pxq6_qp_next(cn, kbn, bn, nfix, kslabs, nfix, kseg);
            if (more) pxq6_qp_ld<POL, pxq6_mode<MODE>::cs>(pan, kbn, row, nxt);   // in flight over this slab's math
            t += pxq6_dot32_qp<POL, MODE, VECX>(cur, anch, xs + kb*PXQ6_QK, tab, sub, plut, pb);
            if (cn != c) { su += t; t = 0.f; }        // chunk close: same fold order as incumbent
            if (!more) break;
            c = cn; kb = kbn; bnd = bn; cur = nxt;
        }
    }
    red[kseg*64 + row] = su;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;
        #pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s*64 + row];
        float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
        out[p*PXQ6_BM + row] = u;
    }
}

// Q1 twin of k_pxq6_mmv_ksplit_gen: identical grid/smem/ws layout/FUSERED tail; pipelined walk.
template <class POL, int MODE, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq6_mmv_ksplit_gen_qp(const uint8_t * __restrict__ W,
                         const char * __restrict__ x_base, const size_t x_tok_stride, const size_t x_slot_stride,
                         float * __restrict__ ws,
                         const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                         char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
                         int * __restrict__ counters,
                         const int R, const int K, const int n_as, const int n_ids, const int S) {
    const int pk = blockIdx.x;
    const int p     = pk / S;
    const int chunk = pk % S;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    const int kslabs = K / PXQ6_QK;
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_MMV_SPLIT_MAX);   // S divides nfix (driver-guaranteed)
    const int cpb  = nfix/S;                                        // fixed chunks per block
    const int c0 = chunk*cpb, c1 = c0 + cpb;
    const int kb0 = (kslabs*c0)/nfix, kb1 = (kslabs*c1)/nfix;
    const int Kc  = (kb1 - kb0)*PXQ6_QK;

    extern __shared__ float pxq6_smem[];
    float * xs  = pxq6_smem;                      // Kc floats (chunk slice only)

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride + (size_t)j*x_slot_stride) + kb0*PXQ6_QK;
    for (int idx = threadIdx.x; idx < Kc; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : (pxq6_mode<MODE>::pairl3 ? 64 : 1)];
    POL::stage_tabs(tab, sub, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POL>(plut, threadIdx.x, 256);
    if constexpr (pxq6_mode<MODE>::pairl3) pxq6_stage_pairlut3<POL>(plut, threadIdx.x, 256);
    __syncthreads();
    pxq6_prmt_book pb{};
    if constexpr (pxq6_mode<MODE>::prmt) pxq6_prmt_build(tab, pb);

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ6_BM;
    const uint8_t * pan = pxq6_panel<POL>(W, e, panels, p, kslabs);
    const float anch = POL::HDR ? POL::anchor(pan, row) : 0.f;

    float * wsj = ws + ((size_t)iy*n_ids + j)*(size_t)(PXQ6_MMV_SPLIT_MAX*PXQ4_MMV_KSEG)*R;
    {
        int c = c0, kb = kb0 + kseg, bnd = (kslabs*(c0+1))/nfix;
        pxq6_pfslab<POL> cur, nxt;
        pxq6_qp_ld<POL, pxq6_mode<MODE>::cs>(pan, kb, row, cur);
        float t = 0.f;
        for (;;) {
            int cn = c, kbn = kb, bn = bnd;
            const bool more = pxq6_qp_next(cn, kbn, bn, c1, kslabs, nfix, kseg);
            if (more) pxq6_qp_ld<POL, pxq6_mode<MODE>::cs>(pan, kbn, row, nxt);   // in flight over this slab's math
            t += pxq6_dot32_qp<POL, MODE, VECX>(cur, anch, xs + (kb - kb0)*PXQ6_QK, tab, sub, plut, pb);
            if (cn != c) {                            // chunk close: same slot, same value
                wsj[(size_t)(c*PXQ4_MMV_KSEG + kseg)*R + p*PXQ6_BM + row] = t;
                t = 0.f;
            }
            if (!more) break;
            c = cn; kb = kbn; bnd = bn; cur = nxt;
        }
    }
    if (counters == nullptr) return;                 // driver launches k_pxq_mmv_reduce_s instead
    __threadfence();                                 // ws writes visible before the flag
    __syncthreads();
    __shared__ int lastblk;
    if (threadIdx.x == 0) {
        const int done = atomicAdd(&counters[iy*n_ids + j], 1) + 1;
        lastblk = (done == (int)gridDim.x);
        if (lastblk) counters[iy*n_ids + j] = 0;     // self-clean for the next node on the stream
    }
    __syncthreads();
    if (!lastblk) return;
    // final reduction, SAME canonical order as k_pxq_mmv_reduce_s => identical values
    const float * wsj2 = wsj;
    float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
    for (int grow = threadIdx.x; grow < R; grow += blockDim.x) {
        float u = 0.f;
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
            float sus = 0.f;
            for (int c = 0; c < nfix; ++c) sus += wsj2[(size_t)(c*PXQ4_MMV_KSEG + s)*R + grow];
            u += sus;
        }
        out[grow] = u;
    }
}

// Q1 twin of k_pxq6_gateup_mmv_ksplit_gen: identical grid/smem/ws layout/FUSERED tail;
// pipelined walk with a U+G state pair per stage (2x the single-weight prefetch registers —
// that is why the env splits the families).
template <class POLU, class POLG, int MODE, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq6_gateup_mmv_ksplit_gen_qp(const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg,
                                const char * __restrict__ x_base, const size_t x_tok_stride,
                                float * __restrict__ ws,
                                const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                                char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
                                const float * __restrict__ bias_u, const size_t bias_u_nb1,
                                const float * __restrict__ bias_g, const size_t bias_g_nb1,
                                int * __restrict__ counters,
                                const int unary, const float alpha, const float limit,
                                const int R, const int K, const int n_as, const int n_ids, const int S) {
    const int pk = blockIdx.x;
    const int p     = pk / S;
    const int chunk = pk % S;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    const int kslabs = K / PXQ6_QK;
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_GU_SPLIT_MAX);   // S divides nfix (driver-guaranteed)
    const int cpb  = nfix/S;                                       // fixed chunks per block
    const int c0 = chunk*cpb, c1 = c0 + cpb;
    const int kb0 = (kslabs*c0)/nfix, kb1 = (kslabs*c1)/nfix;
    const int Kc  = (kb1 - kb0)*PXQ6_QK;

    extern __shared__ float pxq6_smem[];
    float * xs  = pxq6_smem;                      // Kc floats (chunk slice only)

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride) + kb0*PXQ6_QK;
    for (int idx = threadIdx.x; idx < Kc; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tabU[32], subU[16], tabG[32], subG[16];   // 32-entry tabs: P6R LM32
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : (pxq6_mode<MODE>::pairl3 ? 64 : 1)];
    POLU::stage_tabs(tabU, subU, threadIdx.x);
    POLG::stage_tabs(tabG, subG, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POLU>(plut, threadIdx.x, 256);
    if constexpr (pxq6_mode<MODE>::pairl3) pxq6_stage_pairlut3<POLU>(plut, threadIdx.x, 256);
    __syncthreads();
    pxq6_prmt_book pbU{}, pbG{};
    if constexpr (pxq6_mode<MODE>::prmt) { pxq6_prmt_build(tabU, pbU); pxq6_prmt_build(tabG, pbG); }

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ6_BM;
    const uint8_t * panU = pxq6_panel<POLU>(Wu, e, panels, p, kslabs);
    const uint8_t * panG = pxq6_panel<POLG>(Wg, e, panels, p, kslabs);
    const float anchU = POLU::HDR ? POLU::anchor(panU, row) : 0.f;
    const float anchG = POLG::HDR ? POLG::anchor(panG, row) : 0.f;

    const size_t guslots = (size_t)(PXQ6_GU_SPLIT_MAX*PXQ4_MMV_KSEG);   // slots per half
    float * wsj = ws + ((size_t)iy*n_ids + j)*2*guslots*R;
    const int grow = p*PXQ6_BM + row;
    {
        int c = c0, kb = kb0 + kseg, bnd = (kslabs*(c0+1))/nfix;
        pxq6_pfslab<POLU> curU, nxtU;
        pxq6_pfslab<POLG> curG, nxtG;
        pxq6_qp_ld<POLU, pxq6_mode<MODE>::cs>(panU, kb, row, curU);
        pxq6_qp_ld<POLG, pxq6_mode<MODE>::cs>(panG, kb, row, curG);
        float tu = 0.f, tg = 0.f;
        for (;;) {
            int cn = c, kbn = kb, bn = bnd;
            const bool more = pxq6_qp_next(cn, kbn, bn, c1, kslabs, nfix, kseg);
            if (more) {                               // both tensors' next slab in flight
                pxq6_qp_ld<POLU, pxq6_mode<MODE>::cs>(panU, kbn, row, nxtU);
                pxq6_qp_ld<POLG, pxq6_mode<MODE>::cs>(panG, kbn, row, nxtG);
            }
            const float * xk = xs + (kb - kb0)*PXQ6_QK;
            tu += pxq6_dot32_qp<POLU, MODE, VECX>(curU, anchU, xk, tabU, subU, plut, pbU);
            tg += pxq6_dot32_qp<POLG, MODE, VECX>(curG, anchG, xk, tabG, subG, plut, pbG);
            if (cn != c) {                            // chunk close: same slots, same values
                wsj[(size_t)(c*PXQ4_MMV_KSEG + kseg)*R + grow]             = tu;
                wsj[(guslots + (size_t)(c*PXQ4_MMV_KSEG + kseg))*R + grow] = tg;
                tu = 0.f; tg = 0.f;
            }
            if (!more) break;
            c = cn; kb = kbn; bnd = bn; curU = nxtU; curG = nxtG;
        }
    }
    if (counters == nullptr) return;                 // driver launches k_pxq6_gateup_reduce_gen instead
    __threadfence();
    __syncthreads();
    __shared__ int lastblk;
    if (threadIdx.x == 0) {
        const int done = atomicAdd(&counters[iy*n_ids + j], 1) + 1;
        lastblk = (done == (int)gridDim.x);
        if (lastblk) counters[iy*n_ids + j] = 0;
    }
    __syncthreads();
    if (!lastblk) return;
    const float * wsj2 = wsj;
    float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
    for (int gr = threadIdx.x; gr < R; gr += blockDim.x) {
        float u = 0.f, g = 0.f;
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
            float sus = 0.f, sgs = 0.f;
            for (int c = 0; c < nfix; ++c) {
                sus += wsj2[(size_t)(c*PXQ4_MMV_KSEG + s)*R + gr];
                sgs += wsj2[(guslots + (size_t)(c*PXQ4_MMV_KSEG + s))*R + gr];
            }
            u += sus;
            g += sgs;
        }
        if (bias_u) u += *(const float *)((const char *)bias_u + (size_t)e*bias_u_nb1 + (size_t)gr*sizeof(float));
        if (bias_g) g += *(const float *)((const char *)bias_g + (size_t)e*bias_g_nb1 + (size_t)gr*sizeof(float));
        out[gr] = pxq4_glu_apply(g, u, unary, alpha, limit);
    }
}

// =============================================================================================
// R2 — PXA_PXQ6_ROWX2 (2026-08-04): two-rows-per-thread 128-thread twins of the fired decode
// kernels. WHY: SASS LSU accounting on the fired gateup loop puts the warp-uniform x loads
// (8x LDS.U.128 = 4,096 B/iteration of LSU writeback — the address depends only on kb, and kb
// is warp-uniform because kseg is per-warp) at 29% of a 13,888 B LSU total against 1,088 B of
// useful LDG weight traffic; the family is LSU-throughput bound (~83% of the structural LSU
// ceiling at 304 GB/s, DRAM at 50%), NOT instruction-count bound (PAIRL3 cut instructions 36%
// and left LSU bytes identical: 13,632 = 13,632). Each thread owns panel rows (tid&31) and
// (tid&31)+32 and ONE x load feeds both rows' chains (pxq6_dot32x2 below); kseg = tid>>5 keeps
// the SAME KSEG=4 lane structure; grid, dyn-smem layout and red[]/ws slots are unchanged.
// x LSU bytes per weight byte are halved; codebook-gather and LDG bytes per weight byte are
// unchanged. BIT-EXACT by construction: each row's per-(fixed-chunk, lane) fp32 chain is the
// incumbent kernel's chain verbatim (same kb order, same pxq6_acc2 folds, same eff epilogue),
// the two chains are independent, and every partial lands in the same red[]/ws slot; the
// cross-lane epilogues replay the same ascending folds => temp-0 sha env ON vs OFF must match.
// Sourcing rides pxq6_pairx UNCHANGED (composes with any MODE / codebook rework).
// Register expectation: ~+8..16 over the 256-thr twins (2x accumulators/q/eff live); at 128
// threads the 4-blocks-per-SM point is 128 regs, so the 256-thr shapes' 64-reg cliff does not
// bind here. GATE with cuobjdump -res-usage: LOCAL > 0 aborts the change.
// __launch_bounds__(128) deliberately has NO second argument (the (256,1) trap: a minBlocks
// hint lifts the register ceiling; these are NEW kernels, the incumbents keep their exact
// declarations and the env-off binary path is byte-identical).
static_assert(PXQ4_MMV_KSEG == 4, "R2 twins assume the KSEG == 4 lane structure");

// two-row twin of pxq6_dot32 — ONE x load feeds BOTH rows' accumulation chains.
template <class POL, int MODE, bool VECX>
static __device__ __forceinline__ float2 pxq6_dot32x2(const uint8_t * __restrict__ slab,
                                                      int row0, int row1, float anch0, float anch1,
                                                      const float * __restrict__ xk,
                                                      const float * __restrict__ tab,
                                                      const float * __restrict__ sub,
                                                      const float2 * __restrict__ plut,
                                                      const pxq6_prmt_book & pb) {
    using M = pxq6_mode<MODE>;
    float eff0[POL::NEFF], eff1[POL::NEFF];
    POL::row_effs(slab, row0, anch0, sub, eff0);
    POL::row_effs(slab, row1, anch1, sub, eff1);
    uint32_t q0[POL::CODE_WORDS], q1[POL::CODE_WORDS];
    pxq6_ldcodes<POL, M::cs>(slab + POL::CODE_OFF + row0*POL::CODE_BYTES, q0);
    pxq6_ldcodes<POL, M::cs>(slab + POL::CODE_OFF + row1*POL::CODE_BYTES, q1);
    // K2d (MODE SHFL) — R2xK2d bv plumb: lane l holds tab[l]; ONE loop-invariant LDS
    // serves BOTH rows' sourcing (threadIdx.x & 31 is the lane id at any blockDim, so
    // the lane->entry map is identical to pxq6_dot32's; all pair_shfl sites in the x2
    // twins are warp-converged: kseg is per-warp constant, chunk bounds block-uniform).
    // The first ROWX2 drop omitted this and pxq6_pairx's then-DEFAULTED bv silently
    // decoded a zero book at MODE SHFL (NaN cascade -> the VOID stacked arm); the
    // default is now removed so that miswire can never compile again.
    float bv = 0.f;
    if constexpr (M::shfl) bv = tab[threadIdx.x & 31];
    float t0[POL::NEFF], t1[POL::NEFF];
    #pragma unroll
    for (int i = 0; i < POL::NEFF; ++i) { t0[i] = 0.f; t1[i] = 0.f; }
    if (VECX) {
        #pragma unroll
        for (int b = 0; b < 16; b += 2) {
            const float4 xv = *(const float4 *)&xk[2*b];        // ONE load, both rows
            const float2 p00 = pxq6_pairx<POL, MODE>(q0, b,   tab, plut, pb, bv);
            const float2 p01 = pxq6_pairx<POL, MODE>(q0, b+1, tab, plut, pb, bv);
            t0[(b*POL::NEFF) >> 4]     = pxq6_acc2(t0[(b*POL::NEFF) >> 4],     p00.x, xv.x, p00.y, xv.y);
            t0[((b+1)*POL::NEFF) >> 4] = pxq6_acc2(t0[((b+1)*POL::NEFF) >> 4], p01.x, xv.z, p01.y, xv.w);
            const float2 p10 = pxq6_pairx<POL, MODE>(q1, b,   tab, plut, pb, bv);
            const float2 p11 = pxq6_pairx<POL, MODE>(q1, b+1, tab, plut, pb, bv);
            t1[(b*POL::NEFF) >> 4]     = pxq6_acc2(t1[(b*POL::NEFF) >> 4],     p10.x, xv.x, p10.y, xv.y);
            t1[((b+1)*POL::NEFF) >> 4] = pxq6_acc2(t1[((b+1)*POL::NEFF) >> 4], p11.x, xv.z, p11.y, xv.w);
        }
    } else {
        #pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float xa = xk[2*b], xb = xk[2*b+1];           // ONE pair of loads, both rows
            const float2 p0 = pxq6_pairx<POL, MODE>(q0, b, tab, plut, pb, bv);
            t0[(b*POL::NEFF) >> 4] = pxq6_acc2(t0[(b*POL::NEFF) >> 4], p0.x, xa, p0.y, xb);
            const float2 p1 = pxq6_pairx<POL, MODE>(q1, b, tab, plut, pb, bv);
            t1[(b*POL::NEFF) >> 4] = pxq6_acc2(t1[(b*POL::NEFF) >> 4], p1.x, xa, p1.y, xb);
        }
    }
    float r0, r1;
    if (POL::NEFF == 1)      { r0 = eff0[0]*t0[0]; r1 = eff1[0]*t1[0]; }
    else if (POL::NEFF == 2) { r0 = eff0[0]*t0[0] + eff0[1]*t0[1]; r1 = eff1[0]*t1[0] + eff1[1]*t1[1]; }
    else                     { r0 = (eff0[0]*t0[0] + eff0[1]*t0[1]) + (eff0[2]*t0[2] + eff0[3]*t0[3]);
                               r1 = (eff1[0]*t1[0] + eff1[1]*t1[1]) + (eff1[2]*t1[2] + eff1[3]*t1[3]); }
    return make_float2(r0, r1);
}

// R2 twin of k_pxq6_mmv (the 120 GB/s MoE fused-down node's kernel, and the K8-2D plain form).
template <class POL, int MODE, bool VECX>
static __global__ void __launch_bounds__(128)
k_pxq6_mmv_x2(const uint8_t * __restrict__ W,
              const char * __restrict__ x_base, const size_t x_tok_stride, const size_t x_slot_stride,
              char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
              const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
              const int R, const int K, const int n_as) {
    const int p  = blockIdx.x;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    extern __shared__ float pxq6_smem[];
    float * xs = pxq6_smem;
    float * red = pxq6_smem + K;                  // same layout/size as k_pxq6_mmv

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride + (size_t)j*x_slot_stride);
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : (pxq6_mode<MODE>::pairl3 ? 64 : 1)];
    POL::stage_tabs(tab, sub, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POL>(plut, threadIdx.x, 128);
    if constexpr (pxq6_mode<MODE>::pairl3) pxq6_stage_pairlut3<POL>(plut, threadIdx.x, 128);
    __syncthreads();
    pxq6_prmt_book pb{};
    if constexpr (pxq6_mode<MODE>::prmt) pxq6_prmt_build(tab, pb);

    const int row0 = threadIdx.x & 31;            // R2: this thread owns rows row0 and row0+32
    const int row1 = row0 + 32;
    const int kseg = threadIdx.x >> 5;            // same KSEG=4 lane structure as the 256-thr twin
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const uint8_t * pan = pxq6_panel<POL>(W, e, panels, p, kslabs);
    const float anch0 = POL::HDR ? POL::anchor(pan, row0) : 0.f;
    const float anch1 = POL::HDR ? POL::anchor(pan, row1) : 0.f;

    // PXQ_CANON_v1: per-row two-level fixed-chunk fold — each row's chain is k_pxq6_mmv's verbatim.
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_MMV_SPLIT_MAX);
    float su0 = 0.f, su1 = 0.f;
    for (int c = 0; c < nfix; ++c) {
        const int b0 = (kslabs*c)/nfix, b1 = (kslabs*(c+1))/nfix;
        float t0 = 0.f, t1 = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            const float2 d = pxq6_dot32x2<POL, MODE, VECX>(pan + POL::HDR + (size_t)kb*POL::SLAB,
                                 row0, row1, anch0, anch1, xs + kb*PXQ6_QK, tab, sub, plut, pb);
            t0 += d.x;
            t1 += d.y;
        }
        su0 += t0;
        su1 += t1;
    }
    red[kseg*64 + row0] = su0;
    red[kseg*64 + row1] = su1;
    __syncthreads();
    if (kseg == 0) {                              // threads 0..31 finish both their rows
        float u0 = 0.f, u1 = 0.f;
        #pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) { // per-row fold: same ascending order as the twin
            u0 += red[s*64 + row0];
            u1 += red[s*64 + row1];
        }
        float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
        out[p*PXQ6_BM + row0] = u0;
        out[p*PXQ6_BM + row1] = u1;
    }
}

// R2 twin of k_pxq6_mmv_ksplit_gen (K8-2D + MoE-down S-split forms).
template <class POL, int MODE, bool VECX>
static __global__ void __launch_bounds__(128)
k_pxq6_mmv_ksplit_gen_x2(const uint8_t * __restrict__ W,
                      const char * __restrict__ x_base, const size_t x_tok_stride, const size_t x_slot_stride,
                      float * __restrict__ ws,
                      const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                      char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
                      int * __restrict__ counters,
                      const int R, const int K, const int n_as, const int n_ids, const int S) {
    const int pk = blockIdx.x;
    const int p     = pk / S;
    const int chunk = pk % S;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    const int kslabs = K / PXQ6_QK;
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_MMV_SPLIT_MAX);   // S divides nfix (driver-guaranteed)
    const int cpb  = nfix/S;
    const int c0 = chunk*cpb, c1 = c0 + cpb;
    const int kb0 = (kslabs*c0)/nfix, kb1 = (kslabs*c1)/nfix;
    const int Kc  = (kb1 - kb0)*PXQ6_QK;

    extern __shared__ float pxq6_smem[];
    float * xs  = pxq6_smem;                      // Kc floats (chunk slice only)

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride + (size_t)j*x_slot_stride) + kb0*PXQ6_QK;
    for (int idx = threadIdx.x; idx < Kc; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : (pxq6_mode<MODE>::pairl3 ? 64 : 1)];
    POL::stage_tabs(tab, sub, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POL>(plut, threadIdx.x, 128);
    if constexpr (pxq6_mode<MODE>::pairl3) pxq6_stage_pairlut3<POL>(plut, threadIdx.x, 128);
    __syncthreads();
    pxq6_prmt_book pb{};
    if constexpr (pxq6_mode<MODE>::prmt) pxq6_prmt_build(tab, pb);

    const int row0 = threadIdx.x & 31;
    const int row1 = row0 + 32;
    const int kseg = threadIdx.x >> 5;
    const int panels = R / PXQ6_BM;
    const uint8_t * pan = pxq6_panel<POL>(W, e, panels, p, kslabs);
    const float anch0 = POL::HDR ? POL::anchor(pan, row0) : 0.f;
    const float anch1 = POL::HDR ? POL::anchor(pan, row1) : 0.f;

    float * wsj = ws + ((size_t)iy*n_ids + j)*(size_t)(PXQ6_MMV_SPLIT_MAX*PXQ4_MMV_KSEG)*R;
    for (int c = c0; c < c1; ++c) {
        const int b0 = (kslabs*c)/nfix, b1 = (kslabs*(c+1))/nfix;
        float t0 = 0.f, t1 = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            const float2 d = pxq6_dot32x2<POL, MODE, VECX>(pan + POL::HDR + (size_t)kb*POL::SLAB,
                                 row0, row1, anch0, anch1, xs + (kb - kb0)*PXQ6_QK, tab, sub, plut, pb);
            t0 += d.x;
            t1 += d.y;
        }
        wsj[(size_t)(c*PXQ4_MMV_KSEG + kseg)*R + p*PXQ6_BM + row0] = t0;
        wsj[(size_t)(c*PXQ4_MMV_KSEG + kseg)*R + p*PXQ6_BM + row1] = t1;
    }
    if (counters == nullptr) return;                 // driver launches k_pxq_mmv_reduce_s instead
    __threadfence();                                 // ws writes visible before the flag
    __syncthreads();
    __shared__ int lastblk;
    if (threadIdx.x == 0) {
        const int done = atomicAdd(&counters[iy*n_ids + j], 1) + 1;
        lastblk = (done == (int)gridDim.x);
        if (lastblk) counters[iy*n_ids + j] = 0;     // self-clean for the next node on the stream
    }
    __syncthreads();
    if (!lastblk) return;
    // final reduction, SAME canonical order as k_pxq_mmv_reduce_s => identical values
    const float * wsj2 = wsj;
    float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
    for (int grow = threadIdx.x; grow < R; grow += blockDim.x) {
        float u = 0.f;
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
            float sus = 0.f;
            for (int c = 0; c < nfix; ++c) sus += wsj2[(size_t)(c*PXQ4_MMV_KSEG + s)*R + grow];
            u += sus;
        }
        out[grow] = u;
    }
}

// R2 twin of k_pxq6_gateup_mmv_ksplit_gen (MoE fused-gateup + dense-shexp gateup S-split forms).
template <class POLU, class POLG, int MODE, bool VECX>
static __global__ void __launch_bounds__(128)
k_pxq6_gateup_mmv_ksplit_gen_x2(const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg,
                             const char * __restrict__ x_base, const size_t x_tok_stride,
                             float * __restrict__ ws,
                             const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                             char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
                             const float * __restrict__ bias_u, const size_t bias_u_nb1,
                             const float * __restrict__ bias_g, const size_t bias_g_nb1,
                             int * __restrict__ counters,
                             const int unary, const float alpha, const float limit,
                             const int R, const int K, const int n_as, const int n_ids, const int S) {
    const int pk = blockIdx.x;
    const int p     = pk / S;
    const int chunk = pk % S;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    const int kslabs = K / PXQ6_QK;
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_GU_SPLIT_MAX);   // S divides nfix (driver-guaranteed)
    const int cpb  = nfix/S;
    const int c0 = chunk*cpb, c1 = c0 + cpb;
    const int kb0 = (kslabs*c0)/nfix, kb1 = (kslabs*c1)/nfix;
    const int Kc  = (kb1 - kb0)*PXQ6_QK;

    extern __shared__ float pxq6_smem[];
    float * xs  = pxq6_smem;                      // Kc floats (chunk slice only)

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride) + kb0*PXQ6_QK;
    for (int idx = threadIdx.x; idx < Kc; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tabU[32], subU[16], tabG[32], subG[16];   // 32-entry tabs: P6R LM32
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : (pxq6_mode<MODE>::pairl3 ? 64 : 1)];
    POLU::stage_tabs(tabU, subU, threadIdx.x);
    POLG::stage_tabs(tabG, subG, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POLU>(plut, threadIdx.x, 128);
    if constexpr (pxq6_mode<MODE>::pairl3) pxq6_stage_pairlut3<POLU>(plut, threadIdx.x, 128);
    __syncthreads();
    pxq6_prmt_book pbU{}, pbG{};
    if constexpr (pxq6_mode<MODE>::prmt) { pxq6_prmt_build(tabU, pbU); pxq6_prmt_build(tabG, pbG); }

    const int row0 = threadIdx.x & 31;
    const int row1 = row0 + 32;
    const int kseg = threadIdx.x >> 5;
    const int panels = R / PXQ6_BM;
    const uint8_t * panU = pxq6_panel<POLU>(Wu, e, panels, p, kslabs);
    const uint8_t * panG = pxq6_panel<POLG>(Wg, e, panels, p, kslabs);
    const float anchU0 = POLU::HDR ? POLU::anchor(panU, row0) : 0.f;
    const float anchU1 = POLU::HDR ? POLU::anchor(panU, row1) : 0.f;
    const float anchG0 = POLG::HDR ? POLG::anchor(panG, row0) : 0.f;
    const float anchG1 = POLG::HDR ? POLG::anchor(panG, row1) : 0.f;

    // PXQ_CANON_v1: raw per-(fixed-chunk, lane) partials to ws; no in-block cross-lane reduce.
    const size_t guslots = (size_t)(PXQ6_GU_SPLIT_MAX*PXQ4_MMV_KSEG);   // 32 slots per half
    float * wsj = ws + ((size_t)iy*n_ids + j)*2*guslots*R;
    const int grow0 = p*PXQ6_BM + row0;
    const int grow1 = p*PXQ6_BM + row1;
    for (int c = c0; c < c1; ++c) {
        const int b0 = (kslabs*c)/nfix, b1 = (kslabs*(c+1))/nfix;
        float tu0 = 0.f, tu1 = 0.f, tg0 = 0.f, tg1 = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            const float * xk = xs + (kb - kb0)*PXQ6_QK;
            const float2 du = pxq6_dot32x2<POLU, MODE, VECX>(panU + POLU::HDR + (size_t)kb*POLU::SLAB,
                                  row0, row1, anchU0, anchU1, xk, tabU, subU, plut, pbU);
            tu0 += du.x;
            tu1 += du.y;
            const float2 dg = pxq6_dot32x2<POLG, MODE, VECX>(panG + POLG::HDR + (size_t)kb*POLG::SLAB,
                                  row0, row1, anchG0, anchG1, xk, tabG, subG, plut, pbG);
            tg0 += dg.x;
            tg1 += dg.y;
        }
        wsj[(size_t)(c*PXQ4_MMV_KSEG + kseg)*R + grow0]             = tu0;
        wsj[(size_t)(c*PXQ4_MMV_KSEG + kseg)*R + grow1]             = tu1;
        wsj[(guslots + (size_t)(c*PXQ4_MMV_KSEG + kseg))*R + grow0] = tg0;
        wsj[(guslots + (size_t)(c*PXQ4_MMV_KSEG + kseg))*R + grow1] = tg1;
    }
    if (counters == nullptr) return;                 // driver launches k_pxq6_gateup_reduce_gen instead
    __threadfence();
    __syncthreads();
    __shared__ int lastblk;
    if (threadIdx.x == 0) {
        const int done = atomicAdd(&counters[iy*n_ids + j], 1) + 1;
        lastblk = (done == (int)gridDim.x);
        if (lastblk) counters[iy*n_ids + j] = 0;
    }
    __syncthreads();
    if (!lastblk) return;
    const float * wsj2 = wsj;
    float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
    for (int gr = threadIdx.x; gr < R; gr += blockDim.x) {
        float u = 0.f, g = 0.f;
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
            float sus = 0.f, sgs = 0.f;
            for (int c = 0; c < nfix; ++c) {
                sus += wsj2[(size_t)(c*PXQ4_MMV_KSEG + s)*R + gr];
                sgs += wsj2[(guslots + (size_t)(c*PXQ4_MMV_KSEG + s))*R + gr];
            }
            u += sus;
            g += sgs;
        }
        if (bias_u) u += *(const float *)((const char *)bias_u + (size_t)e*bias_u_nb1 + (size_t)gr*sizeof(float));
        if (bias_g) g += *(const float *)((const char *)bias_g + (size_t)e*bias_g_nb1 + (size_t)gr*sizeof(float));
        out[gr] = pxq4_glu_apply(g, u, unary, alpha, limit);
    }
}

// ---------------------------------------------------------------------------------------------
// PXA_PXQ_TOKBATCH (2026-08-04): TOK-batched verify for the gateup gen-split decode kernel.
//
// A k-token drafter/MTP verify batch reaches this family as src1->ne[2] == Ny > 1 and the
// per-token kernel carries the token index on blockIdx.z: every (panel-chunk, slot, token)
// block re-loads and re-decodes its expert weight slice from DRAM once PER TOKEN (measured,
// 122B verify(5): gateup wall 4.51x the 1-token launch, per-token weight re-read ~0.90x --
// L2 absorbs essentially nothing because a whole expert (~3.5 MB gateup) does not survive
// between grid-z slices). This kernel folds TOK consecutive tokens into ONE block:
// grid.z = ceil(Ny/TOK), the block stages the TOK x slices in dynamic smem (TOK*Kc floats,
// token-major), and the token loop sits INSIDE the (kb, row) weight walk, so when tokens of
// the group route slot j to the SAME expert the repeat code-row loads land in L1 with
// perfect temporal locality instead of a fresh DRAM walk. Routing is PER TOKEN (ids stride
// ids_nb1), so each token keeps its own expert pointer: disjoint routing issues the same
// loads as today (no traffic loss vector), overlapping routing stops paying DRAM. The
// realized win is therefore proportional to cross-token expert overlap -- the PXA_MOE_DEBUG
// E2 union probe measures exactly that quantity on live traffic.
//
// BIT-EXACT by construction: each token's per-(fixed-chunk, lane) fp32 chain is the
// per-token kernel's chain verbatim (same canon_v1 chunks, same kb order, same pxq6_dot32
// sourcing; U and G chains are independent so the two-pass order below cannot change any
// fold), and it writes the SAME raw-partial ws slots; the canonical reducer
// (k_pxq6_gateup_reduce_gen) is reused unchanged. temp-0 sha vs the per-token arm is a
// valid gate here.
//
// REGISTER DISCIPLINE: the two-pass form (U walk, then G walk) keeps only TOK accumulators
// + TOK panel pointers + TOK anchors live in the hot loop -- at TOK=2 that is the same
// 2-independent-chain shape the per-token kernel schedules (its U+G pair).
// __launch_bounds__(256, PXA_PXQ_TOK_MINBLK) with the default minblk 4 pins the 64-reg /
// 4-blocks-per-SM point (65536/(256*4) = 64 exactly). GATE every fired instantiation with
// cuobjdump -res-usage: REG > 64 or LOCAL > 0 aborts the change (documented fallback:
// -DPXA_PXQ_TOK_MINBLK=3 = the incumbent per-token kernel's own 85-reg/3-block point).
// TOK is capped at 3: ~4 live regs per extra token per pass on top of a ~50+ reg shared hot
// set puts TOK=4 past the ceiling. A TOK=2 that fits beats a TOK=5 that spills.
// FUSERED is deliberately not offered on this arm (measured dead end; reducer always runs).
// Default OFF (PXA_PXQ_TOKBATCH unset/0). Ny == 1 decode never reaches this kernel.
// ---------------------------------------------------------------------------------------------
#ifndef PXA_PXQ_TOK_MINBLK
#define PXA_PXQ_TOK_MINBLK 4
#endif

// one weight pass (U or G) over this block's fixed chunks for TOK tokens. Per token t the
// chunk fold, kb stride and dot32 calls are byte-identical to the per-token kernel's; the
// raw partial goes to the SAME ws slot (ws_stride2 = 2*guslots*R floats per (iy, j) row;
// ws_half = 0 for the U half, guslots*R for the G half).
template <class POL, int MODE, bool VECX, int TOK>
static __device__ __forceinline__ void pxq6_gu_tok_pass(const uint8_t * __restrict__ W,
        const int (&e_t)[TOK], float * __restrict__ ws, const size_t ws_stride2, const size_t ws_half,
        const int iy0, const int n_ids, const int j,
        const float * __restrict__ xs, const int Kc, const int kb0,
        const float * __restrict__ tab, const float * __restrict__ sub,
        const float2 * __restrict__ plut, const pxq6_prmt_book & pb,
        const int row, const int kseg, const int panels, const int p, const int kslabs,
        const int c0, const int c1, const int nfix, const int R, const int grow) {
    const uint8_t * pan_t[TOK];
    float anch_t[TOK];
    #pragma unroll
    for (int t = 0; t < TOK; ++t) {
        pan_t[t]  = e_t[t] >= 0 ? pxq6_panel<POL>(W, e_t[t], panels, p, kslabs) : W;
        anch_t[t] = (e_t[t] >= 0 && POL::HDR) ? POL::anchor(pan_t[t], row) : 0.f;
    }
    for (int c = c0; c < c1; ++c) {
        const int b0 = (kslabs*c)/nfix, b1 = (kslabs*(c+1))/nfix;
        float acc[TOK];
        #pragma unroll
        for (int t = 0; t < TOK; ++t) acc[t] = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            #pragma unroll
            for (int t = 0; t < TOK; ++t) {
                if (e_t[t] < 0) continue;
                acc[t] += pxq6_dot32<POL, MODE, VECX>(pan_t[t] + POL::HDR + (size_t)kb*POL::SLAB, row,
                            anch_t[t], xs + (size_t)t*Kc + (size_t)(kb - kb0)*PXQ6_QK, tab, sub, plut, pb);
            }
        }
        #pragma unroll
        for (int t = 0; t < TOK; ++t) {
            if (e_t[t] < 0) continue;
            float * wsj = ws + ((size_t)(iy0 + t)*n_ids + j)*ws_stride2 + ws_half;
            wsj[(size_t)(c*PXQ4_MMV_KSEG + kseg)*R + grow] = acc[t];
        }
    }
}

template <class POLU, class POLG, int MODE, bool VECX, int TOK>
static __global__ void __launch_bounds__(256, PXA_PXQ_TOK_MINBLK)
k_pxq6_gateup_mmv_ksplit_gen_tok(const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg,
                                 const char * __restrict__ x_base, const size_t x_tok_stride,
                                 float * __restrict__ ws,
                                 const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                                 const int R, const int K, const int n_as, const int n_ids,
                                 const int S, const int Ny) {
    const int pk = blockIdx.x;
    const int p     = pk / S;
    const int chunk = pk % S;
    const int j   = blockIdx.y;
    const int iy0 = (int)blockIdx.z*TOK;

    // per-token routed expert (block-uniform, so every branch below is convergent);
    // iy >= Ny (ragged tail group) or an SER-hole id disables that token's lane.
    int e_t[TOK];
    bool any = false;
    #pragma unroll
    for (int t = 0; t < TOK; ++t) {
        const int iy = iy0 + t;
        int e = -1;
        if (iy < Ny) e = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
        e_t[t] = (e >= 0 && e < n_as) ? e : -1;
        any = any || (e_t[t] >= 0);
    }
    if (!any) return;

    const int kslabs = K / PXQ6_QK;
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_GU_SPLIT_MAX);   // S divides nfix (driver-guaranteed)
    const int cpb  = nfix/S;                                       // fixed chunks per block
    const int c0 = chunk*cpb, c1 = c0 + cpb;
    const int kb0 = (kslabs*c0)/nfix, kb1 = (kslabs*c1)/nfix;
    const int Kc  = (kb1 - kb0)*PXQ6_QK;

    extern __shared__ float pxq6_smem[];
    float * xs = pxq6_smem;                       // TOK x Kc floats, token-major

    #pragma unroll
    for (int t = 0; t < TOK; ++t) {
        if (e_t[t] < 0) continue;                 // inactive lane: slice never read
        const float * x = (const float *)(x_base + (size_t)(iy0 + t)*x_tok_stride) + kb0*PXQ6_QK;
        for (int idx = threadIdx.x; idx < Kc; idx += blockDim.x) xs[(size_t)t*Kc + idx] = x[idx];
    }

    __shared__ float tabU[32], subU[16], tabG[32], subG[16];   // 32-entry tabs: P6R LM32
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : (pxq6_mode<MODE>::pairl3 ? 64 : 1)];
    POLU::stage_tabs(tabU, subU, threadIdx.x);
    POLG::stage_tabs(tabG, subG, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POLU>(plut, threadIdx.x, 256);
    if constexpr (pxq6_mode<MODE>::pairl3) pxq6_stage_pairlut3<POLU>(plut, threadIdx.x, 256);
    __syncthreads();
    pxq6_prmt_book pbU{}, pbG{};
    if constexpr (pxq6_mode<MODE>::prmt) { pxq6_prmt_build(tabU, pbU); pxq6_prmt_build(tabG, pbG); }

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ6_BM;
    const size_t guslots = (size_t)(PXQ6_GU_SPLIT_MAX*PXQ4_MMV_KSEG);   // 32 slots per half
    const int grow = p*PXQ6_BM + row;

    // U pass, then G pass: per-tensor chains are independent, so the pass split cannot
    // change any token's fold; it exists to halve hot-loop register liveness (see header).
    pxq6_gu_tok_pass<POLU, MODE, VECX, TOK>(Wu, e_t, ws, 2*guslots*R, 0,
            iy0, n_ids, j, xs, Kc, kb0, tabU, subU, plut, pbU,
            row, kseg, panels, p, kslabs, c0, c1, nfix, R, grow);
    pxq6_gu_tok_pass<POLG, MODE, VECX, TOK>(Wg, e_t, ws, 2*guslots*R, guslots*R,
            iy0, n_ids, j, xs, Kc, kb0, tabG, subG, plut, pbG,
            row, kseg, panels, p, kslabs, c0, c1, nfix, R, grow);
}

// PXA_PXQ_TOKBATCH env (0 = OFF default; 2/3 = TOK). 1 is meaningless (that IS the
// per-token kernel) and >3 is register-unsafe -- both clamp with a warning. Requires the
// gen split (PXA_PXQ6_KSPLIT_GEN, default S=4 on sm_60/70): with the gen split off this
// lever never engages and the ENGAGED banner will not print -- such an A/B is VOID, not null.
static inline int pxa_pxq_tokbatch() {
    static const int v = [](){
        const char * e = getenv("PXA_PXQ_TOKBATCH");
        int t = e ? atoi(e) : 0;
        if (t != 0 && (t < 2 || t > 3)) {
            fprintf(stderr, "PXA_PXQ_TOKBATCH=%d invalid (want 0/2/3) -- OFF\n", t);
            t = 0;
        }
        if (t) fprintf(stderr, "PXA_PXQ_TOKBATCH: TOK=%d armed (gateup verify token-batching, bit-exact; "
                               "an A/B without the in-run ENGAGED banner is void)\n", t);
        return t;
    }();
    return v;
}

// first-fire / first-decline banner per device (phantom-lever discipline, same as the
// down-split log): an armed lever that never prints ENGAGED did not run.
static inline void pxa_pxq_tokbatch_log(int device, bool fired, int tok, unsigned gx, unsigned gy,
                                        unsigned gz, int Ny, int S, size_t smem, int fu, int fg,
                                        int m, int vx) {
    static std::atomic<uint32_t> seen{0};
    if (device < 0 || device >= 16) return;
    const uint32_t bit = 1u << (2*device + (fired ? 0 : 1));
    if (seen.fetch_or(bit) & bit) return;
    if (fired) {
        fprintf(stderr, "PXA_PXQ_TOKBATCH dev%d: ENGAGED tok=%d grid=%ux%ux%u Ny=%d S=%d smem=%zuB fmt=%d/%d mode=%d vecx=%d\n",
                device, tok, gx, gy, gz, Ny, S, smem, fu, fg, m, vx);
    } else {
        fprintf(stderr, "PXA_PXQ_TOKBATCH dev%d: DECLINED -> per-token gateup (tok=%d Ny=%d S=%d smem=%zuB fmt=%d/%d mode=%d vecx=%d: no instantiation for this fmt/mode/vecx or smem > 46K)\n",
                device, tok, Ny, S, smem, fu, fg, m, vx);
    }
}

// canonical reducer for k_pxq6_gateup_mmv_ksplit_gen: the ONE canonical fold over the raw
// t[c][s] partials + bias + GLU -- bit-identical to k_pxq6_gateup_mmv's epilogue by construction.
static __global__ void k_pxq6_gateup_reduce_gen(const float * __restrict__ ws,
        char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
        const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
        const float * __restrict__ bias_u, const size_t bias_u_nb1,
        const float * __restrict__ bias_g, const size_t bias_g_nb1,
        const int R, const int n_as, const int n_ids,
        const int unary, const float alpha, const float limit, const int kslabs) {
    const int grow = blockIdx.x*blockDim.x + threadIdx.x;
    if (grow >= R) return;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_GU_SPLIT_MAX);
    const size_t guslots = (size_t)(PXQ6_GU_SPLIT_MAX*PXQ4_MMV_KSEG);
    const float * wsj = ws + ((size_t)iy*n_ids + j)*2*guslots*R;
    float u = 0.f, g = 0.f;
    for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
        float sus = 0.f, sgs = 0.f;
        for (int c = 0; c < nfix; ++c) {
            sus += wsj[(size_t)(c*PXQ4_MMV_KSEG + s)*R + grow];
            sgs += wsj[(guslots + (size_t)(c*PXQ4_MMV_KSEG + s))*R + grow];
        }
        u += sus;
        g += sgs;
    }
    if (bias_u) u += *(const float *)((const char *)bias_u + (size_t)e*bias_u_nb1 + (size_t)grow*sizeof(float));
    if (bias_g) g += *(const float *)((const char *)bias_g + (size_t)e*bias_g_nb1 + (size_t)grow*sizeof(float));
    const float r = pxq4_glu_apply(g, u, unary, alpha, limit);
    float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
    out[grow] = r;
}

// FUSERED gate (default OFF — measured LOSS, kept as a documented dead end; =1 enables).
// The fence-and-flag fused finish produces byte-identical output (verified) but the LAST
// split block then reduces S*R floats ALONE while the rest of the machine drains — the wide
// k_pxq_mmv_reduce_s runs the same reduction across R/256 parallel blocks in ~2.4us, and the
// launch it saves was being hidden by async submission anyway. Protocol (F35B matched-bytes,
// fill 5992, n=8/arm, 4 interleaved blocks): fused-ON decode 83.81 vs two-kernel 85.17 on
// 2xV100 (-1.6%), 59.45 vs 60.92 on 2xP100 (-2.4%). Verdict: launch count was NOT the wall
// gap; do not re-enable without new evidence.
static inline bool pxa_pxq_split_fusered() {
    static const bool v = [] {
        const char * e = getenv("PXA_PXQ_SPLIT_FUSERED");
        return e && atoi(e) != 0;
    }();
    return v;
}

// persistent per-device fence-and-flag counters for the FUSERED finish (1024 ints, zeroed once;
// finishers self-clean, so no per-call memset). nullptr mid-capture or on alloc failure ->
// callers fall back to the separate-reduce form.
static inline int * pxq6_ksplit_counters(int device, cudaStream_t stream) {
    static int * bufs[64] = {nullptr};
    static std::mutex mtx;
    if (device < 0 || device >= 64) return nullptr;
    if (bufs[device]) return bufs[device];
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &st);
    if (st != cudaStreamCaptureStatusNone) return nullptr;
    std::lock_guard<std::mutex> lk(mtx);
    if (!bufs[device]) {
        int * p = nullptr;
        if (cudaMalloc(&p, 4096) != cudaSuccess) { (void)cudaGetLastError(); return nullptr; }
        if (cudaMemset(p, 0, 4096) != cudaSuccess) { (void)cudaGetLastError(); cudaFree(p); return nullptr; }
        bufs[device] = p;
    }
    return bufs[device];
}

// persistent KSPLIT workspace (per device; grown OUTSIDE graph capture only — if capture is
// active and the buffer is too small, the driver falls back to the non-split kernel, which is
// bit-identical anyway, so replayed graphs stay correct whatever the capture saw).
struct pxq6_ksplit_ws_t { float * ptr = nullptr; size_t sz = 0; };
static inline float * pxq6_ksplit_workspace(int device, cudaStream_t stream, size_t need_floats) {
    static pxq6_ksplit_ws_t ws[64];
    if (device < 0 || device >= 64) return nullptr;
    pxq6_ksplit_ws_t & w = ws[device];
    const size_t need = need_floats*sizeof(float);
    if (w.sz >= need) return w.ptr;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &st);
    if (st != cudaStreamCaptureStatusNone) return nullptr;   // can't grow mid-capture -> decline
    if (w.ptr) cudaFree(w.ptr);
    w.ptr = nullptr; w.sz = 0;
    if (cudaMalloc(&w.ptr, need) != cudaSuccess) { w.ptr = nullptr; cudaGetLastError(); return nullptr; }
    w.sz = need;
    return w.ptr;
}

// ---------------------------------------------------------------------------------------------
// prefill: grouped fused GEMM — generic family. Tiling/accumulation identical to the proven
// k_pxq5_gemm_grouped (64 thr, 8r x 8t per thread, strict k-order half2 chains); POL selects the
// format's dequant; RAG = K4 ragged-tile FMA skip; PIPE = K5 register prefetch. All bit-exact.
// ---------------------------------------------------------------------------------------------
template <class POL>
static __device__ __forceinline__ void pxq6_deq_slab_cm(const uint8_t * __restrict__ slab, int tid, float anch,
                                                        const float * __restrict__ tab, const float * __restrict__ sub,
                                                        const uint32_t * __restrict__ q, half (* __restrict__ sW)[PXQ4_BM]) {
    float eff[POL::NEFF];
    POL::row_effs(slab, tid, anch, sub, eff);
    #pragma unroll
    for (int b = 0; b < 16; ++b) {
        const float e = eff[(b*POL::NEFF) >> 4];
        const float2 v = POL::pair(q, b, tab);
        sW[2*b][tid]   = __float2half_rn(e * v.x);
        sW[2*b+1][tid] = __float2half_rn(e * v.y);
    }
}

template <class POL, bool RAG, bool PIPE>
static __global__ void __launch_bounds__(64)
k_pxq6_gemm_grouped(const uint8_t * __restrict__ W, const half * __restrict__ A, float * __restrict__ C,
                    const float * __restrict__ bias, const size_t bias_nb1,
                    const pxq4_tile_info * __restrict__ tiles, const int R, const int K) {
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    if (tile.nrows <= 0) return;   // padding tile of a device-built tile list (uniform over the block)
    const uint8_t * pan = pxq6_panel<POL>(W, tile.e, panels, p, kslabs);
    const half    * At  = A + (size_t)tile.row0*K;
    float         * Ct  = C + (size_t)tile.row0*R + (size_t)p*PXQ6_BM;

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ half sW[PXQ6_QK][PXQ4_BM];
    __shared__ half sA[PXQ6_QK][PXQ4_BN];
    const int tid = threadIdx.x;
    POL::stage_tabs(tab, sub, tid);
    const float anch = POL::HDR ? POL::anchor(pan, tid) : 0.f;

    const int tx = tid & 7, ty = tid >> 3;
    half2 acc[8][4];
    #pragma unroll
    for (int r = 0; r < 8; ++r)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][j] = __floats2half2_rn(0.f, 0.f);

    const bool a_valid = tid < tile.nrows;
    const bool fma_on  = !RAG || (8*ty) < tile.nrows;   // K4: skip FMAs whose stores are masked

    uint32_t qn[POL::CODE_WORDS] = {};
    uint4 an0 = {0,0,0,0}, an1 = {0,0,0,0}, an2 = {0,0,0,0}, an3 = {0,0,0,0};
    if (PIPE) {   // prologue loads, kb = 0
        pxq6_ldcodes<POL>(pan + POL::HDR + (size_t)0*POL::SLAB + POL::CODE_OFF + tid*POL::CODE_BYTES, qn);
        if (a_valid) {
            const half * src = At + (size_t)tid*K;
            an0 = *(const uint4 *)(src);      an1 = *(const uint4 *)(src + 8);
            an2 = *(const uint4 *)(src + 16); an3 = *(const uint4 *)(src + 24);
        }
    }

    for (int kb = 0; kb < kslabs; ++kb) {
        const uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
        uint32_t q[POL::CODE_WORDS]; uint4 a0, a1, a2, a3;
        if (PIPE) {
            #pragma unroll
            for (int i = 0; i < POL::CODE_WORDS; ++i) q[i] = qn[i];
            a0 = an0; a1 = an1; a2 = an2; a3 = an3;
            if (kb + 1 < kslabs) {   // issue next-slab loads; they overlap this slab's FMAs
                const uint8_t * slabn = pan + POL::HDR + (size_t)(kb+1)*POL::SLAB;
                pxq6_ldcodes<POL>(slabn + POL::CODE_OFF + tid*POL::CODE_BYTES, qn);
                if (a_valid) {
                    const half * src = At + (size_t)tid*K + (kb+1)*PXQ6_QK;
                    an0 = *(const uint4 *)(src);      an1 = *(const uint4 *)(src + 8);
                    an2 = *(const uint4 *)(src + 16); an3 = *(const uint4 *)(src + 24);
                }
            }
        } else {
            pxq6_ldcodes<POL>(slab + POL::CODE_OFF + tid*POL::CODE_BYTES, q);
            if (a_valid) {
                const half * src = At + (size_t)tid*K + kb*PXQ6_QK;
                a0 = *(const uint4 *)(src);      a1 = *(const uint4 *)(src + 8);
                a2 = *(const uint4 *)(src + 16); a3 = *(const uint4 *)(src + 24);
            }
        }
        __syncthreads();
        pxq6_deq_slab_cm<POL>(slab, tid, anch, tab, sub, q, sW);
        if (a_valid) {
            const half * h0 = (const half *)&a0; const half * h1 = (const half *)&a1;
            const half * h2 = (const half *)&a2; const half * h3 = (const half *)&a3;
            #pragma unroll
            for (int i = 0; i < 8; ++i) { sA[i][tid] = h0[i]; sA[8+i][tid] = h1[i]; sA[16+i][tid] = h2[i]; sA[24+i][tid] = h3[i]; }
        } else {
            const half hz = __float2half_rn(0.f);
            #pragma unroll
            for (int i = 0; i < PXQ6_QK; ++i) sA[i][tid] = hz;
        }
        __syncthreads();
        if (fma_on) {
            #pragma unroll 4
            for (int kk = 0; kk < PXQ6_QK; ++kk) {
                half2 a2v[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) a2v[j] = *(const half2 *)&sA[kk][8*ty + 2*j];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const half2 wp  = *(const half2 *)&sW[kk][8*tx + 2*i];
                    const half2 wlo = __low2half2(wp), whi = __high2half2(wp);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[2*i][j]   = __hfma2(wlo, a2v[j], acc[2*i][j]);
                        acc[2*i+1][j] = __hfma2(whi, a2v[j], acc[2*i+1][j]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = 8*tx + r;
        const float b = bias ? *(const float *)((const char *)bias + (size_t)tile.e*bias_nb1 + (size_t)(p*PXQ6_BM + row)*sizeof(float)) : 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = 8*ty + 2*j;
            if (t < tile.nrows)     Ct[(size_t)t*R + row]     = __half2float(__low2half(acc[r][j]))  + b;
            if (t + 1 < tile.nrows) Ct[(size_t)(t+1)*R + row] = __half2float(__high2half(acc[r][j])) + b;
        }
    }
}

// K3 GUFUSE: ONE kernel does the up GEMM + gate GEMM + GLU epilogue. Per-tensor accumulation
// chains are identical to two k_pxq6_gemm_grouped launches; the epilogue applies the exact
// pxq4_glu_apply to the same (bias-folded) u/g floats and converts once => bit-exact vs the
// 3-kernel pipeline (GEMM up, GEMM gate, k_pxq4_glu).
template <class POL, typename dst_t, bool RAG, bool PIPE>
static __global__ void __launch_bounds__(64)
k_pxq6_gemm_gufuse(const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg,
                   const half * __restrict__ A, dst_t * __restrict__ H,
                   const float * __restrict__ bias_u, const size_t bias_u_nb1,
                   const float * __restrict__ bias_g, const size_t bias_g_nb1,
                   const pxq4_tile_info * __restrict__ tiles, const int R, const int K,
                   const int unary, const float alpha, const float limit) {
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    if (tile.nrows <= 0) return;   // padding tile of a device-built tile list (uniform over the block)
    const uint8_t * panU = pxq6_panel<POL>(Wu, tile.e, panels, p, kslabs);
    const uint8_t * panG = pxq6_panel<POL>(Wg, tile.e, panels, p, kslabs);
    const half    * At  = A + (size_t)tile.row0*K;
    dst_t         * Ht  = H + (size_t)tile.row0*R + (size_t)p*PXQ6_BM;

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ half sWu[PXQ6_QK][PXQ4_BM];
    __shared__ half sWg[PXQ6_QK][PXQ4_BM];
    __shared__ half sA[PXQ6_QK][PXQ4_BN];
    const int tid = threadIdx.x;
    POL::stage_tabs(tab, sub, tid);
    const float anchU = POL::HDR ? POL::anchor(panU, tid) : 0.f;
    const float anchG = POL::HDR ? POL::anchor(panG, tid) : 0.f;

    const int tx = tid & 7, ty = tid >> 3;
    half2 accU[8][4], accG[8][4];
    #pragma unroll
    for (int r = 0; r < 8; ++r)
        #pragma unroll
        for (int j = 0; j < 4; ++j) { accU[r][j] = __floats2half2_rn(0.f, 0.f); accG[r][j] = __floats2half2_rn(0.f, 0.f); }

    const bool a_valid = tid < tile.nrows;
    const bool fma_on  = !RAG || (8*ty) < tile.nrows;

    uint32_t qun[POL::CODE_WORDS] = {}, qgn[POL::CODE_WORDS] = {};
    uint4 an0 = {0,0,0,0}, an1 = {0,0,0,0}, an2 = {0,0,0,0}, an3 = {0,0,0,0};
    if (PIPE) {
        pxq6_ldcodes<POL>(panU + POL::HDR + POL::CODE_OFF + tid*POL::CODE_BYTES, qun);
        pxq6_ldcodes<POL>(panG + POL::HDR + POL::CODE_OFF + tid*POL::CODE_BYTES, qgn);
        if (a_valid) {
            const half * src = At + (size_t)tid*K;
            an0 = *(const uint4 *)(src);      an1 = *(const uint4 *)(src + 8);
            an2 = *(const uint4 *)(src + 16); an3 = *(const uint4 *)(src + 24);
        }
    }

    for (int kb = 0; kb < kslabs; ++kb) {
        const uint8_t * slabU = panU + POL::HDR + (size_t)kb*POL::SLAB;
        const uint8_t * slabG = panG + POL::HDR + (size_t)kb*POL::SLAB;
        uint32_t qu[POL::CODE_WORDS], qg[POL::CODE_WORDS]; uint4 a0, a1, a2, a3;
        if (PIPE) {
            #pragma unroll
            for (int i = 0; i < POL::CODE_WORDS; ++i) { qu[i] = qun[i]; qg[i] = qgn[i]; }
            a0 = an0; a1 = an1; a2 = an2; a3 = an3;
            if (kb + 1 < kslabs) {
                pxq6_ldcodes<POL>(panU + POL::HDR + (size_t)(kb+1)*POL::SLAB + POL::CODE_OFF + tid*POL::CODE_BYTES, qun);
                pxq6_ldcodes<POL>(panG + POL::HDR + (size_t)(kb+1)*POL::SLAB + POL::CODE_OFF + tid*POL::CODE_BYTES, qgn);
                if (a_valid) {
                    const half * src = At + (size_t)tid*K + (kb+1)*PXQ6_QK;
                    an0 = *(const uint4 *)(src);      an1 = *(const uint4 *)(src + 8);
                    an2 = *(const uint4 *)(src + 16); an3 = *(const uint4 *)(src + 24);
                }
            }
        } else {
            pxq6_ldcodes<POL>(slabU + POL::CODE_OFF + tid*POL::CODE_BYTES, qu);
            pxq6_ldcodes<POL>(slabG + POL::CODE_OFF + tid*POL::CODE_BYTES, qg);
            if (a_valid) {
                const half * src = At + (size_t)tid*K + kb*PXQ6_QK;
                a0 = *(const uint4 *)(src);      a1 = *(const uint4 *)(src + 8);
                a2 = *(const uint4 *)(src + 16); a3 = *(const uint4 *)(src + 24);
            }
        }
        __syncthreads();
        pxq6_deq_slab_cm<POL>(slabU, tid, anchU, tab, sub, qu, sWu);
        pxq6_deq_slab_cm<POL>(slabG, tid, anchG, tab, sub, qg, sWg);
        if (a_valid) {
            const half * h0 = (const half *)&a0; const half * h1 = (const half *)&a1;
            const half * h2 = (const half *)&a2; const half * h3 = (const half *)&a3;
            #pragma unroll
            for (int i = 0; i < 8; ++i) { sA[i][tid] = h0[i]; sA[8+i][tid] = h1[i]; sA[16+i][tid] = h2[i]; sA[24+i][tid] = h3[i]; }
        } else {
            const half hz = __float2half_rn(0.f);
            #pragma unroll
            for (int i = 0; i < PXQ6_QK; ++i) sA[i][tid] = hz;
        }
        __syncthreads();
        if (fma_on) {
            #pragma unroll 2
            for (int kk = 0; kk < PXQ6_QK; ++kk) {
                half2 a2v[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) a2v[j] = *(const half2 *)&sA[kk][8*ty + 2*j];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const half2 wpu  = *(const half2 *)&sWu[kk][8*tx + 2*i];
                    const half2 wpg  = *(const half2 *)&sWg[kk][8*tx + 2*i];
                    const half2 wulo = __low2half2(wpu), wuhi = __high2half2(wpu);
                    const half2 wglo = __low2half2(wpg), wghi = __high2half2(wpg);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        accU[2*i][j]   = __hfma2(wulo, a2v[j], accU[2*i][j]);
                        accU[2*i+1][j] = __hfma2(wuhi, a2v[j], accU[2*i+1][j]);
                        accG[2*i][j]   = __hfma2(wglo, a2v[j], accG[2*i][j]);
                        accG[2*i+1][j] = __hfma2(wghi, a2v[j], accG[2*i+1][j]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = 8*tx + r;
        const float bu = bias_u ? *(const float *)((const char *)bias_u + (size_t)tile.e*bias_u_nb1 + (size_t)(p*PXQ6_BM + row)*sizeof(float)) : 0.f;
        const float bg = bias_g ? *(const float *)((const char *)bias_g + (size_t)tile.e*bias_g_nb1 + (size_t)(p*PXQ6_BM + row)*sizeof(float)) : 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = 8*ty + 2*j;
            if (t < tile.nrows) {
                const float u = __half2float(__low2half(accU[r][j]))  + bu;
                const float g = __half2float(__low2half(accG[r][j]))  + bg;
                Ht[(size_t)t*R + row] = (dst_t)pxq4_glu_apply(g, u, unary, alpha, limit);
            }
            if (t + 1 < tile.nrows) {
                const float u = __half2float(__high2half(accU[r][j])) + bu;
                const float g = __half2float(__high2half(accG[r][j])) + bg;
                Ht[(size_t)(t+1)*R + row] = (dst_t)pxq4_glu_apply(g, u, unary, alpha, limit);
            }
        }
    }
}

// K3 SCATFUSE: down GEMM writing straight to the MoE output rows via the row mapping
// (replaces C_down buffer + k_copy_dst_from_contiguous). Same values, same rows => exact.
template <class POL, bool RAG, bool PIPE>
static __global__ void __launch_bounds__(64)
k_pxq6_gemm_down_scat(const uint8_t * __restrict__ W, const half * __restrict__ A,
                      char * __restrict__ dst, const size_t nb1, const size_t nb2,
                      const pxq4_rowmap * __restrict__ map,
                      const pxq4_tile_info * __restrict__ tiles, const int R, const int K) {
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    if (tile.nrows <= 0) return;   // padding tile of a device-built tile list (uniform over the block)
    const uint8_t * pan = pxq6_panel<POL>(W, tile.e, panels, p, kslabs);
    const half    * At  = A + (size_t)tile.row0*K;

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ half sW[PXQ6_QK][PXQ4_BM];
    __shared__ half sA[PXQ6_QK][PXQ4_BN];
    const int tid = threadIdx.x;
    POL::stage_tabs(tab, sub, tid);
    const float anch = POL::HDR ? POL::anchor(pan, tid) : 0.f;

    const int tx = tid & 7, ty = tid >> 3;
    half2 acc[8][4];
    #pragma unroll
    for (int r = 0; r < 8; ++r)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][j] = __floats2half2_rn(0.f, 0.f);

    const bool a_valid = tid < tile.nrows;
    const bool fma_on  = !RAG || (8*ty) < tile.nrows;

    uint32_t qn[POL::CODE_WORDS] = {};
    uint4 an0 = {0,0,0,0}, an1 = {0,0,0,0}, an2 = {0,0,0,0}, an3 = {0,0,0,0};
    if (PIPE) {
        pxq6_ldcodes<POL>(pan + POL::HDR + POL::CODE_OFF + tid*POL::CODE_BYTES, qn);
        if (a_valid) {
            const half * src = At + (size_t)tid*K;
            an0 = *(const uint4 *)(src);      an1 = *(const uint4 *)(src + 8);
            an2 = *(const uint4 *)(src + 16); an3 = *(const uint4 *)(src + 24);
        }
    }

    for (int kb = 0; kb < kslabs; ++kb) {
        const uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
        uint32_t q[POL::CODE_WORDS]; uint4 a0, a1, a2, a3;
        if (PIPE) {
            #pragma unroll
            for (int i = 0; i < POL::CODE_WORDS; ++i) q[i] = qn[i];
            a0 = an0; a1 = an1; a2 = an2; a3 = an3;
            if (kb + 1 < kslabs) {
                pxq6_ldcodes<POL>(pan + POL::HDR + (size_t)(kb+1)*POL::SLAB + POL::CODE_OFF + tid*POL::CODE_BYTES, qn);
                if (a_valid) {
                    const half * src = At + (size_t)tid*K + (kb+1)*PXQ6_QK;
                    an0 = *(const uint4 *)(src);      an1 = *(const uint4 *)(src + 8);
                    an2 = *(const uint4 *)(src + 16); an3 = *(const uint4 *)(src + 24);
                }
            }
        } else {
            pxq6_ldcodes<POL>(slab + POL::CODE_OFF + tid*POL::CODE_BYTES, q);
            if (a_valid) {
                const half * src = At + (size_t)tid*K + kb*PXQ6_QK;
                a0 = *(const uint4 *)(src);      a1 = *(const uint4 *)(src + 8);
                a2 = *(const uint4 *)(src + 16); a3 = *(const uint4 *)(src + 24);
            }
        }
        __syncthreads();
        pxq6_deq_slab_cm<POL>(slab, tid, anch, tab, sub, q, sW);
        if (a_valid) {
            const half * h0 = (const half *)&a0; const half * h1 = (const half *)&a1;
            const half * h2 = (const half *)&a2; const half * h3 = (const half *)&a3;
            #pragma unroll
            for (int i = 0; i < 8; ++i) { sA[i][tid] = h0[i]; sA[8+i][tid] = h1[i]; sA[16+i][tid] = h2[i]; sA[24+i][tid] = h3[i]; }
        } else {
            const half hz = __float2half_rn(0.f);
            #pragma unroll
            for (int i = 0; i < PXQ6_QK; ++i) sA[i][tid] = hz;
        }
        __syncthreads();
        if (fma_on) {
            #pragma unroll 4
            for (int kk = 0; kk < PXQ6_QK; ++kk) {
                half2 a2v[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) a2v[j] = *(const half2 *)&sA[kk][8*ty + 2*j];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const half2 wp  = *(const half2 *)&sW[kk][8*tx + 2*i];
                    const half2 wlo = __low2half2(wp), whi = __high2half2(wp);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[2*i][j]   = __hfma2(wlo, a2v[j], acc[2*i][j]);
                        acc[2*i+1][j] = __hfma2(whi, a2v[j], acc[2*i+1][j]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = 8*tx + r;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = 8*ty + 2*j;
            if (t < tile.nrows) {
                const pxq4_rowmap m = map[tile.row0 + t];
                float * o = (float *)(dst + (size_t)m.i1*nb1 + (size_t)m.i2*nb2);
                o[p*PXQ6_BM + row] = __half2float(__low2half(acc[r][j]));
            }
            if (t + 1 < tile.nrows) {
                const pxq4_rowmap m = map[tile.row0 + t + 1];
                float * o = (float *)(dst + (size_t)m.i1*nb1 + (size_t)m.i2*nb2);
                o[p*PXQ6_BM + row] = __half2float(__high2half(acc[r][j]));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// K6 WMMA-v1 — V100 (sm_70) fused dequant→skewed-smem-fp16→m16n16k16 HMMA grouped GEMM.
// NOT bit-exact vs the half2 kernels (fp16-mul/fp32-add-tree vs strict-k fp16 chain): G3 logprob
// parity + G4 ppl regate are mandatory gates; env default OFF; runtime cc==700 dispatch only.
// Block = 256 thr (8 warps), 64x64 C tile; warp (wm 0..3, wn 0..1) owns a 16x32 C sub-tile =
// 2 n-frags. Smem: W and A staged row-major [64][32+8] half (+8 half skew — no ldmatrix on
// sm_70, the fragment loader does plain smem reads, so stride/skew is the whole game).
// Epilogue stages fragments through smem for bias + ragged masking (frag layout is opaque).
// ---------------------------------------------------------------------------------------------
#define PXQ6_WMMA_LD 40   // 32 k + 8 half skew

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
// fragment -> smem staging overloads (fp32 accum direct; fp16-accum twin widens via a half stage)
static __device__ __forceinline__ void pxq6_wmma_stage_acc(float * mytile,
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> & a, int warp, int lane) {
    (void)warp; (void)lane;
    nvcuda::wmma::store_matrix_sync(mytile, a, 17, nvcuda::wmma::mem_col_major);
}
static __device__ __forceinline__ void pxq6_wmma_stage_acc(float * mytile,
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> & a, int warp, int lane) {
    __shared__ half hstage[8][16*17];
    nvcuda::wmma::store_matrix_sync(&hstage[warp][0], a, 17, nvcuda::wmma::mem_col_major);
    __syncwarp();
    for (int i = lane; i < 256; i += 32) {
        const int rr = i & 15, cc = i >> 4;
        mytile[cc*17 + rr] = __half2float(hstage[warp][cc*17 + rr]);
    }
}
#endif

template <class POL, bool F32ACC>
static __global__ void __launch_bounds__(256)
k_pxq6_gemm_grouped_wmma(const uint8_t * __restrict__ W, const half * __restrict__ A, float * __restrict__ C,
                         const float * __restrict__ bias, const size_t bias_nb1,
                         const pxq4_tile_info * __restrict__ tiles, const int R, const int K) {
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
    using namespace nvcuda;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    if (tile.nrows <= 0) return;   // padding tile of a device-built tile list (uniform over the block)
    const uint8_t * pan = pxq6_panel<POL>(W, tile.e, panels, p, kslabs);
    const half    * At  = A + (size_t)tile.row0*K;
    float         * Ct  = C + (size_t)tile.row0*R + (size_t)p*PXQ6_BM;

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ half sWA[2][64*PXQ6_WMMA_LD];      // [0] = W rows, [1] = A tokens (row-major, skewed)
    const int tid  = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    POL::stage_tabs(tab, sub, tid);

    // staging assignment: 4 threads per row/token, 8 halves (one uint4 of codes / one uint4 of A) each
    const int srow = tid >> 2, sseg = tid & 3;
    const float anch = POL::HDR ? POL::anchor(pan, srow) : 0.f;

    const int wm = warp & 3, wn = warp >> 2;      // 4 x 2 warp grid over the 64x64 C tile
    typedef typename std::conditional<F32ACC, float, half>::type acc_t;
    wmma::fragment<wmma::accumulator, 16, 16, 16, acc_t> acc[2];
    wmma::fill_fragment(acc[0], (acc_t)0.0f);
    wmma::fill_fragment(acc[1], (acc_t)0.0f);

    const bool a_valid = srow < tile.nrows;       // srow doubles as the token index for A staging

    for (int kb = 0; kb < kslabs; ++kb) {
        const uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
        __syncthreads();
        {   // W: dequant 8 values (4 code bytes) into sWA[0][srow][sseg*8..]
            float eff[POL::NEFF];
            POL::row_effs(slab, srow, anch, sub, eff);
            const float e = eff[(sseg*POL::NEFF) >> 2];
            const uint32_t qw = *(const uint32_t *)(slab + POL::CODE_OFF + srow*16 + sseg*4);
            half * o = &sWA[0][srow*PXQ6_WMMA_LD + sseg*8];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int byte = (qw >> (8*b)) & 0xff;
                o[2*b]   = __float2half_rn(e * tab[byte & 0xf]);
                o[2*b+1] = __float2half_rn(e * tab[byte >> 4]);
            }
        }
        {   // A: 8 halves per thread
            half * o = &sWA[1][srow*PXQ6_WMMA_LD + sseg*8];
            if (a_valid) {
                const uint4 v = *(const uint4 *)(At + (size_t)srow*K + kb*PXQ6_QK + sseg*8);
                *(uint4 *)o = v;
            } else {
                const half hz = __float2half_rn(0.f);
                #pragma unroll
                for (int i = 0; i < 8; ++i) o[i] = hz;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kf = 0; kf < 2; ++kf) {          // two k=16 fragments per 32-K slab
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
            wmma::load_matrix_sync(af, &sWA[0][(wm*16)*PXQ6_WMMA_LD + kf*16], PXQ6_WMMA_LD);
            #pragma unroll
            for (int nf = 0; nf < 2; ++nf) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
                wmma::load_matrix_sync(bf, &sWA[1][(wn*32 + nf*16)*PXQ6_WMMA_LD + kf*16], PXQ6_WMMA_LD);
                wmma::mma_sync(acc[nf], af, bf, acc[nf]);
            }
        }
    }

    // epilogue: stage each 16x16 fragment through smem (bias add + ragged masking need row/col
    // coordinates; the Volta fragment register layout is opaque). Reuses the sWA smem after sync.
    __syncthreads();
    float * scratch = (float *)sWA;               // 8 warps x 16*17 floats = 8.5 KB < 10.2 KB
    float * mytile = scratch + warp*16*17;
    #pragma unroll
    for (int nf = 0; nf < 2; ++nf) {
        pxq6_wmma_stage_acc(mytile, acc[nf], warp, lane);
        __syncwarp();
        const int col0 = wn*32 + nf*16, row0 = wm*16;
        for (int i = lane; i < 256; i += 32) {
            const int rr = i & 15, t = i >> 4;
            if (col0 + t < tile.nrows) {
                const int row = row0 + rr;
                const float b = bias ? *(const float *)((const char *)bias + (size_t)tile.e*bias_nb1 + (size_t)(p*PXQ6_BM + row)*sizeof(float)) : 0.f;
                Ct[(size_t)(col0 + t)*R + row] = mytile[t*17 + rr] + b;
            }
        }
        __syncwarp();
    }
#else
    // non-sm_70 compile: dead body (runtime dispatch never selects this kernel off-Volta)
    (void)W; (void)A; (void)C; (void)bias; (void)bias_nb1; (void)tiles; (void)R; (void)K;
#endif
}

// ---------------------------------------------------------------------------------------------
// K6 WMMA-v2 (2026-07-22) — the V100 tensor-core prefill REBUILD (PXA_PXQ6_WMMA=3).
// v1 measured +0.97% because it is single-buffered (sync-stage-sync-MMA, zero overlap), the
// driver drops GUFUSE under WMMA (2 GEMM launches + 2x total*R f32 C round-trips + a separate
// GLU kernel), drops the down GEMM back to the 64-thread half2 scat kernel — and its gate
// excludes the P2/P3 tiers + mixed up/gate pairs, which on PXQU mixed-tier models (e.g. the
// 35B: 21 pxq4 / 41 pxq2 / 58 pxq3 expert tensors) leaves MOST expert GEMMs on half2. v2:
//   (1) 2-stage register->smem DOUBLE-BUFFER over k-slabs: the next slab's full code row +
//       scale bytes + A uint4 are issued as global loads while the current slab's smem stores
//       + HMMA work retire (the K5 PIPE pattern applied to the WMMA loop).
//   (2) FUSED up+gate WMMA kernel with the GLU epilogue writing dst_t directly — GUFUSE
//       parity: one launch, A staged ONCE for both tensors, each B fragment loaded once and
//       reused for the U and G MMAs, no f32 C intermediates, no separate GLU kernel.
//   (3) a WMMA down GEMM with the SCATFUSE scatter epilogue, so the whole expert triple rides
//       the tensor cores under the default SCATFUSE config.
//   (4) POLICY-GENERIC dequant staging (P2/P3/P6/P6HQ via the proven POL::pair decode) and a
//       TWO-POLICY template so mixed-tier up/gate pairs ride WMMA too.
// fp32 accumulators only (fp16-mul/fp32-add HMMA is full-rate on V100; the quality-safe form).
// Same determinism contract as v1: NOT bit-exact vs the half2 chain (fp16-mul/fp32-add tree) —
// G3 logprob/coherence parity + a ppl regate are MANDATORY; temp-0 sha is NOT a valid gate.
// cc==700 runtime dispatch only; P6R excluded (no model need, keeps the test surface small).
// ---------------------------------------------------------------------------------------------

// prefetched-scale helpers: the raw sub-scale byte(s) of one row, loaded a slab ahead so the
// k-loop body touches no cold global memory. Layout per policy: NEFF==2 -> 1 byte at slab[row]
// (P2/P3/P6 share it); NEFF==4 -> 2 bytes at slab[2*row] (2-aligned, u16 load).
template <class POL>
static __device__ __forceinline__ uint16_t pxq6_wmma_ldscale(const uint8_t * __restrict__ slab, int srow) {
    if constexpr (POL::NEFF == 4) return *(const uint16_t *)(slab + 2*srow);
    else                          return (uint16_t)slab[srow];
}
// eff scale for the 8-elem segment sseg (elems sseg*8..sseg*8+7) — replicates
// POL::row_effs + v1's eff[(sseg*NEFF)>>2] selection exactly, from the prefetched byte(s).
template <class POL>
static __device__ __forceinline__ float pxq6_wmma_eff(uint16_t sc, int sseg, float anch,
                                                      const float * __restrict__ sub) {
    if constexpr (POL::NEFF == 4) {
        const int byte = (sseg < 2) ? (sc & 0xff) : (sc >> 8);
        return anch * sub[(sseg & 1) ? (byte >> 4) : (byte & 0xf)];
    } else {
        const int byte = sc & 0xff;
        return anch * sub[(sseg < 2) ? (byte & 0xf) : (byte >> 4)];
    }
}

// dequant this thread's 8 elems (pairs sseg*4..sseg*4+3) from a prefetched FULL code row via
// the policy's proven pair() decode — policy-generic (P2/P3/P6/P6HQ). The full-row load is
// 4x redundant across the 4 threads of a row; the rows are tiny (8-16 B) and L1-resident.
template <class POL>
static __device__ __forceinline__ void pxq6_wmma_deq8(const uint32_t * __restrict__ q, int sseg,
                                                      float e, const float * __restrict__ tab,
                                                      half * __restrict__ o) {
    #pragma unroll
    for (int b2 = 0; b2 < 4; ++b2) {
        const float2 v = POL::pair(q, sseg*4 + b2, tab);
        o[2*b2]   = __float2half_rn(e * v.x);
        o[2*b2+1] = __float2half_rn(e * v.y);
    }
}

// v2 fused up+gate grouped WMMA GEMM + GLU epilogue (dst_t = half for the fuse_down H, float
// for the no-fusion path). Two policies (mixed-tier up/gate pairs are common in PXQU models).
// Block 256 thr / 8 warps; 64x64 C tile; warp (wm,wn) owns 16x32.
template <class POLU, class POLG, typename dst_t>
static __global__ void __launch_bounds__(256)
k_pxq6_gemm_gufuse_wmma(const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg,
                        const half * __restrict__ A, dst_t * __restrict__ H,
                        const float * __restrict__ bias_u, const size_t bias_u_nb1,
                        const float * __restrict__ bias_g, const size_t bias_g_nb1,
                        const pxq4_tile_info * __restrict__ tiles, const int R, const int K,
                        const int unary, const float alpha, const float limit) {
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
    using namespace nvcuda;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    if (tile.nrows <= 0) return;   // padding tile of a device-built tile list (uniform over the block)
    const uint8_t * panU = pxq6_panel<POLU>(Wu, tile.e, panels, p, kslabs);
    const uint8_t * panG = pxq6_panel<POLG>(Wg, tile.e, panels, p, kslabs);
    const half    * At  = A + (size_t)tile.row0*K;
    dst_t         * Ht  = H + (size_t)tile.row0*R + (size_t)p*PXQ6_BM;

    __shared__ float tabU[32], subU[16], tabG[32], subG[16];
    __shared__ half sS[3][64*PXQ6_WMMA_LD];       // [0]=U rows, [1]=G rows, [2]=A tokens
    const int tid  = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    POLU::stage_tabs(tabU, subU, tid);
    POLG::stage_tabs(tabG, subG, tid);

    const int srow = tid >> 2, sseg = tid & 3;    // 4 threads/row, 8 halves each
    const float anchU = POLU::HDR ? POLU::anchor(panU, srow) : 0.f;
    const float anchG = POLG::HDR ? POLG::anchor(panG, srow) : 0.f;

    const int wm = warp & 3, wn = warp >> 2;      // 4 x 2 warp grid over the 64x64 C tile
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accU[2], accG[2];
    wmma::fill_fragment(accU[0], 0.f); wmma::fill_fragment(accU[1], 0.f);
    wmma::fill_fragment(accG[0], 0.f); wmma::fill_fragment(accG[1], 0.f);

    const bool a_valid = srow < tile.nrows;       // srow doubles as the token index for A

    // 2-stage register prefetch (kb = 0 prologue): full code rows + scale bytes + A vector
    uint32_t qun[POLU::CODE_WORDS], qgn[POLG::CODE_WORDS];
    uint16_t scun, scgn;
    uint4 an = {0,0,0,0};
    {
        const uint8_t * slabU = panU + POLU::HDR;
        const uint8_t * slabG = panG + POLG::HDR;
        pxq6_ldcodes<POLU>(slabU + POLU::CODE_OFF + srow*POLU::CODE_BYTES, qun);
        pxq6_ldcodes<POLG>(slabG + POLG::CODE_OFF + srow*POLG::CODE_BYTES, qgn);
        scun = pxq6_wmma_ldscale<POLU>(slabU, srow);
        scgn = pxq6_wmma_ldscale<POLG>(slabG, srow);
        if (a_valid) an = *(const uint4 *)(At + (size_t)srow*K + sseg*8);
    }

    for (int kb = 0; kb < kslabs; ++kb) {
        uint32_t qu[POLU::CODE_WORDS], qg[POLG::CODE_WORDS];
        #pragma unroll
        for (int i = 0; i < POLU::CODE_WORDS; ++i) qu[i] = qun[i];
        #pragma unroll
        for (int i = 0; i < POLG::CODE_WORDS; ++i) qg[i] = qgn[i];
        const uint16_t scu = scun, scg = scgn;
        const uint4 av = an;
        if (kb + 1 < kslabs) {   // issue next-slab loads; latency hides under stores+sync+MMA
            const uint8_t * slabU = panU + POLU::HDR + (size_t)(kb+1)*POLU::SLAB;
            const uint8_t * slabG = panG + POLG::HDR + (size_t)(kb+1)*POLG::SLAB;
            pxq6_ldcodes<POLU>(slabU + POLU::CODE_OFF + srow*POLU::CODE_BYTES, qun);
            pxq6_ldcodes<POLG>(slabG + POLG::CODE_OFF + srow*POLG::CODE_BYTES, qgn);
            scun = pxq6_wmma_ldscale<POLU>(slabU, srow);
            scgn = pxq6_wmma_ldscale<POLG>(slabG, srow);
            if (a_valid) an = *(const uint4 *)(At + (size_t)srow*K + (kb+1)*PXQ6_QK + sseg*8);
        }
        __syncthreads();   // prior MMA done reading smem (also publishes tabs on kb==0)
        pxq6_wmma_deq8<POLU>(qu, sseg, pxq6_wmma_eff<POLU>(scu, sseg, anchU, subU), tabU,
                             &sS[0][srow*PXQ6_WMMA_LD + sseg*8]);
        pxq6_wmma_deq8<POLG>(qg, sseg, pxq6_wmma_eff<POLG>(scg, sseg, anchG, subG), tabG,
                             &sS[1][srow*PXQ6_WMMA_LD + sseg*8]);
        {
            half * o = &sS[2][srow*PXQ6_WMMA_LD + sseg*8];
            if (a_valid) {
                *(uint4 *)o = av;
            } else {
                const half hz = __float2half_rn(0.f);
                #pragma unroll
                for (int i = 0; i < 8; ++i) o[i] = hz;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kf = 0; kf < 2; ++kf) {          // two k=16 fragments per 32-K slab
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> afU, afG;
            wmma::load_matrix_sync(afU, &sS[0][(wm*16)*PXQ6_WMMA_LD + kf*16], PXQ6_WMMA_LD);
            wmma::load_matrix_sync(afG, &sS[1][(wm*16)*PXQ6_WMMA_LD + kf*16], PXQ6_WMMA_LD);
            #pragma unroll
            for (int nf = 0; nf < 2; ++nf) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
                wmma::load_matrix_sync(bf, &sS[2][(wn*32 + nf*16)*PXQ6_WMMA_LD + kf*16], PXQ6_WMMA_LD);
                wmma::mma_sync(accU[nf], afU, bf, accU[nf]);
                wmma::mma_sync(accG[nf], afG, bf, accG[nf]);
            }
        }
    }

    // epilogue: per-warp 16x17 f32 scratch (reuses the k-loop smem; 8 warps x 1088 B < 15.3 KB).
    // Stage U, pull the lane's 8 elems to registers, stage G over the same tile, then
    // bias + pxq4_glu_apply + ONE dst_t store per element (the exact gufuse epilogue math).
    __syncthreads();
    float * mytile = (float *)sS + warp*16*17;
    #pragma unroll
    for (int nf = 0; nf < 2; ++nf) {
        const int col0 = wn*32 + nf*16, row0 = wm*16;
        wmma::store_matrix_sync(mytile, accU[nf], 17, wmma::mem_col_major);
        __syncwarp();
        float ureg[8];
        #pragma unroll
        for (int ii = 0; ii < 8; ++ii) {
            const int i = ii*32 + lane;
            ureg[ii] = mytile[(i >> 4)*17 + (i & 15)];
        }
        __syncwarp();
        wmma::store_matrix_sync(mytile, accG[nf], 17, wmma::mem_col_major);
        __syncwarp();
        #pragma unroll
        for (int ii = 0; ii < 8; ++ii) {
            const int i = ii*32 + lane, rr = i & 15, t = i >> 4;
            if (col0 + t < tile.nrows) {
                const int row = row0 + rr;
                const float bu = bias_u ? *(const float *)((const char *)bias_u + (size_t)tile.e*bias_u_nb1 + (size_t)(p*PXQ6_BM + row)*sizeof(float)) : 0.f;
                const float bg = bias_g ? *(const float *)((const char *)bias_g + (size_t)tile.e*bias_g_nb1 + (size_t)(p*PXQ6_BM + row)*sizeof(float)) : 0.f;
                Ht[(size_t)(col0 + t)*R + row] = (dst_t)pxq4_glu_apply(mytile[t*17 + rr] + bg, ureg[ii] + bu, unary, alpha, limit);
            }
        }
        __syncwarp();
    }
#else
    (void)Wu; (void)Wg; (void)A; (void)H; (void)bias_u; (void)bias_u_nb1; (void)bias_g; (void)bias_g_nb1;
    (void)tiles; (void)R; (void)K; (void)unary; (void)alpha; (void)limit;
#endif
}

// v2 down GEMM: double-buffered WMMA + the SCATFUSE scatter epilogue (same values/rows as
// k_pxq6_gemm_down_scat — the scatter is a pure addressing epilogue). Policy-generic.
template <class POL>
static __global__ void __launch_bounds__(256)
k_pxq6_gemm_down_scat_wmma(const uint8_t * __restrict__ W, const half * __restrict__ A,
                           char * __restrict__ dst, const size_t nb1, const size_t nb2,
                           const pxq4_rowmap * __restrict__ map,
                           const pxq4_tile_info * __restrict__ tiles, const int R, const int K) {
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
    using namespace nvcuda;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    if (tile.nrows <= 0) return;   // padding tile of a device-built tile list (uniform over the block)
    const uint8_t * pan = pxq6_panel<POL>(W, tile.e, panels, p, kslabs);
    const half    * At  = A + (size_t)tile.row0*K;

    __shared__ float tab[32];
    __shared__ float sub[16];
    __shared__ half sS[2][64*PXQ6_WMMA_LD];       // [0]=W rows, [1]=A tokens
    const int tid  = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    POL::stage_tabs(tab, sub, tid);

    const int srow = tid >> 2, sseg = tid & 3;
    const float anch = POL::HDR ? POL::anchor(pan, srow) : 0.f;

    const int wm = warp & 3, wn = warp >> 2;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    wmma::fill_fragment(acc[0], 0.f); wmma::fill_fragment(acc[1], 0.f);

    const bool a_valid = srow < tile.nrows;

    uint32_t qn[POL::CODE_WORDS]; uint16_t scn;
    uint4 an = {0,0,0,0};
    {
        const uint8_t * slab = pan + POL::HDR;
        pxq6_ldcodes<POL>(slab + POL::CODE_OFF + srow*POL::CODE_BYTES, qn);
        scn = pxq6_wmma_ldscale<POL>(slab, srow);
        if (a_valid) an = *(const uint4 *)(At + (size_t)srow*K + sseg*8);
    }

    for (int kb = 0; kb < kslabs; ++kb) {
        uint32_t q[POL::CODE_WORDS];
        #pragma unroll
        for (int i = 0; i < POL::CODE_WORDS; ++i) q[i] = qn[i];
        const uint16_t sc = scn; const uint4 av = an;
        if (kb + 1 < kslabs) {
            const uint8_t * slab = pan + POL::HDR + (size_t)(kb+1)*POL::SLAB;
            pxq6_ldcodes<POL>(slab + POL::CODE_OFF + srow*POL::CODE_BYTES, qn);
            scn = pxq6_wmma_ldscale<POL>(slab, srow);
            if (a_valid) an = *(const uint4 *)(At + (size_t)srow*K + (kb+1)*PXQ6_QK + sseg*8);
        }
        __syncthreads();
        pxq6_wmma_deq8<POL>(q, sseg, pxq6_wmma_eff<POL>(sc, sseg, anch, sub), tab,
                            &sS[0][srow*PXQ6_WMMA_LD + sseg*8]);
        {
            half * o = &sS[1][srow*PXQ6_WMMA_LD + sseg*8];
            if (a_valid) {
                *(uint4 *)o = av;
            } else {
                const half hz = __float2half_rn(0.f);
                #pragma unroll
                for (int i = 0; i < 8; ++i) o[i] = hz;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kf = 0; kf < 2; ++kf) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
            wmma::load_matrix_sync(af, &sS[0][(wm*16)*PXQ6_WMMA_LD + kf*16], PXQ6_WMMA_LD);
            #pragma unroll
            for (int nf = 0; nf < 2; ++nf) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
                wmma::load_matrix_sync(bf, &sS[1][(wn*32 + nf*16)*PXQ6_WMMA_LD + kf*16], PXQ6_WMMA_LD);
                wmma::mma_sync(acc[nf], af, bf, acc[nf]);
            }
        }
    }

    __syncthreads();
    float * mytile = (float *)sS + warp*16*17;    // 8 warps x 1088 B < 10.2 KB
    #pragma unroll
    for (int nf = 0; nf < 2; ++nf) {
        const int col0 = wn*32 + nf*16, row0 = wm*16;
        wmma::store_matrix_sync(mytile, acc[nf], 17, wmma::mem_col_major);
        __syncwarp();
        #pragma unroll
        for (int ii = 0; ii < 8; ++ii) {
            const int i = ii*32 + lane, rr = i & 15, t = i >> 4;
            if (col0 + t < tile.nrows) {
                const pxq4_rowmap m = map[tile.row0 + col0 + t];
                float * o = (float *)(dst + (size_t)m.i1*nb1 + (size_t)m.i2*nb2);
                o[p*PXQ6_BM + row0 + rr] = mytile[t*17 + rr];
            }
        }
        __syncwarp();
    }
#else
    (void)W; (void)A; (void)dst; (void)nb1; (void)nb2; (void)map; (void)tiles; (void)R; (void)K;
#endif
}

#include "pxq23.cuh"   // PXQ2/PXQ3: tables, gates, pol_p2/pol_p3, dequant wrappers

// ---------------------------------------------------------------------------------------------
// host-side dispatch: format codes + kernel pickers (runtime env -> template instantiation).
// fmt: 3 = PXQ4 core, 4 = PXQ4-HQ, 5 = PXQ2, 6 = PXQ3, 7 = PXQ6 (display pxq6, 5-bit LM32);
// 0 = not a PXA slab type / gated off. (1 and 2 = the retired legacy id-250/id-251 formats,
// removed 2026-07-21 — values reserved, never reuse.)
// ---------------------------------------------------------------------------------------------
#define PXA_PXQ_FMT_NONE 0
// 1 reserved — retired PXA_PXQ_FMT_P4 (legacy type id 250), removed 2026-07-21
// 2 reserved — retired PXA_PXQ_FMT_P5 (legacy type id 251), removed 2026-07-21
#define PXA_PXQ_FMT_P6   3
#define PXA_PXQ_FMT_P6HQ 4
#define PXA_PXQ_FMT_P2   5      // ADD
#define PXA_PXQ_FMT_P3   6      // ADD
#define PXA_PXQ_FMT_P6R  7      // ADD: real-PXQ6 tier (TAB/TAB_CS modes only, like P2/P3)
// ADD 2026-07-29: PXQ1, 1-bit sign tier (TAB/TAB_CS modes only, like P2/P3).
// ⚠ THE VALUE MUST STAY > PXA_PXQ_FMT_P2 — the driver tests RANGES, not equality:
//     fmt >= PXA_PXQ_FMT_P6 (3) == "is a PXA slab format"  -> upload the shared SUB16 LUT
//     fmt >= PXA_PXQ_FMT_P2 (5) == "sub-nibble code packing" -> no PRMT/PAIRLUT, no K6 v1 WMMA
// Numbering P1 below P2 would silently route 1-bit codes into the 4-bit nibble kernels.
#define PXA_PXQ_FMT_P1   8

static inline int pxa_pxq_fmt(ggml_type t) {
    switch (t) {
        case GGML_TYPE_PXQ4:   return pxa_pxq6_enabled() ? PXA_PXQ_FMT_P6   : PXA_PXQ_FMT_NONE;
        case GGML_TYPE_PXQ4HQ: return pxa_pxq6_enabled() ? PXA_PXQ_FMT_P6HQ : PXA_PXQ_FMT_NONE;
        case GGML_TYPE_PXQ2:   return pxa_pxq2_enabled() ? PXA_PXQ_FMT_P2   : PXA_PXQ_FMT_NONE;
        case GGML_TYPE_PXQ3:   return pxa_pxq3_enabled() ? PXA_PXQ_FMT_P3   : PXA_PXQ_FMT_NONE;
        case GGML_TYPE_PXQ6:  return pxa_pxq6r_enabled() ? PXA_PXQ_FMT_P6R : PXA_PXQ_FMT_NONE;
        case GGML_TYPE_PXQ1:   return pxa_pxq1_enabled() ? PXA_PXQ_FMT_P1   : PXA_PXQ_FMT_NONE;
        default:               return PXA_PXQ_FMT_NONE;
    }
}

typedef void (*pxq6_gateup_fn)(const uint8_t *, const uint8_t *, const char *, size_t,
        char *, size_t, size_t, const char *, size_t, size_t,
        const float *, size_t, const float *, size_t, int, int, int, int, float, float);
typedef void (*pxq6_mmv_fn)(const uint8_t *, const char *, size_t, size_t,
        char *, size_t, size_t, const char *, size_t, size_t, int, int, int);
typedef void (*pxq6_mmv_ks_fn)(const uint8_t *, const char *, size_t, size_t,
        float *, const char *, size_t, size_t, int, int, int, int);
typedef void (*pxq6_mmv_ksg_fn)(const uint8_t *, const char *, size_t, size_t,
        float *, const char *, size_t, size_t,
        char *, size_t, size_t, int *, int, int, int, int, int);
typedef void (*pxq6_mmv_redfuse_fn)(const uint8_t *, const float *, char *, size_t, size_t,
        const char *, size_t, size_t, const float *, size_t, const float *, size_t,
        int, int, int, int, int, float, float);
typedef void (*pxq6_gateup_ks_fn)(const uint8_t *, const uint8_t *, const char *, size_t,
        float *, const char *, size_t, size_t, int, int, int, int);
typedef void (*pxq6_gateup_ksg_fn)(const uint8_t *, const uint8_t *, const char *, size_t,
        float *, const char *, size_t, size_t,
        char *, size_t, size_t, const float *, size_t, const float *, size_t,
        int *, int, float, float, int, int, int, int, int);
typedef void (*pxq6_gemm_fn)(const uint8_t *, const half *, float *, const float *, size_t,
        const pxq4_tile_info *, int, int);
typedef void (*pxq6_gufuse_h_fn)(const uint8_t *, const uint8_t *, const half *, half *,
        const float *, size_t, const float *, size_t, const pxq4_tile_info *, int, int, int, float, float);
typedef void (*pxq6_gufuse_f_fn)(const uint8_t *, const uint8_t *, const half *, float *,
        const float *, size_t, const float *, size_t, const pxq4_tile_info *, int, int, int, float, float);
typedef void (*pxq6_scat_fn)(const uint8_t *, const half *, char *, size_t, size_t,
        const pxq4_rowmap *, const pxq4_tile_info *, int, int);

#define PXQ6_PICK2(K, POL, b1, b2) \
    ((b1) ? ((b2) ? (K<POL, true, true>) : (K<POL, true, false>)) \
          : ((b2) ? (K<POL, false, true>) : (K<POL, false, false>)))
#define PXQ6_PICK_FMT(RET, NAME, K) \
    static inline RET NAME(int fmt, bool b1, bool b2) { \
        switch (fmt) { \
            case PXA_PXQ_FMT_P6: return PXQ6_PICK2(K, pxq6_pol_p6,   b1, b2); \
            case PXA_PXQ_FMT_P2: return PXQ6_PICK2(K, pxq6_pol_p2, false, b2); \
            case PXA_PXQ_FMT_P3: return PXQ6_PICK2(K, pxq6_pol_p3, false, b2); \
            case PXA_PXQ_FMT_P6R: return PXQ6_PICK2(K, pxq6_pol_p6r, false, b2); \
            case PXA_PXQ_FMT_P1: return PXQ6_PICK2(K, pxq6_pol_p1, false, b2); \
            default:             return PXQ6_PICK2(K, pxq6_pol_p6hq, b1, b2); \
        } \
    }

// decode-family pickers: runtime sourcing MODE (0 tab / 1 pairlut / 2 prmt / 3 tab+cs /
// 4 prmt+cs / 6 shfl). P2/P3 (sub-nibble code packing) support only the tab + shfl paths —
// prmt/pairlut demote at compile time.
static inline int pxa_pxq6_decode_mode() {
    static const int m = [](){
        const bool cs = pxa_pxq6_ldcs();
        if (pxa_pxq6_shfl())    return PXQ6_MODE_SHFL;    // K2d wins when armed (shfl+cs not offered)
        if (pxa_pxq6_prmt())    return cs ? PXQ6_MODE_PRMT_CS : PXQ6_MODE_PRMT;
        if (pxa_pxq6_pairlut()) return PXQ6_MODE_PAIRL;   // pairlut+cs not offered (lever is OFF-verdict)
        return cs ? PXQ6_MODE_TAB_CS : PXQ6_MODE_TAB;
    }();
    return m;
}
#define PXQ6_PICKM(K, POL, m, vx) \
    ((m) == PXQ6_MODE_SHFL    ? ((vx) ? (K<POL, PXQ6_MODE_SHFL,    true>) : (K<POL, PXQ6_MODE_SHFL,    false>)) : \
     (m) == PXQ6_MODE_PAIRL   ? ((vx) ? (K<POL, PXQ6_MODE_PAIRL,   true>) : (K<POL, PXQ6_MODE_PAIRL,   false>)) : \
     (m) == PXQ6_MODE_PRMT    ? ((vx) ? (K<POL, PXQ6_MODE_PRMT,    true>) : (K<POL, PXQ6_MODE_PRMT,    false>)) : \
     (m) == PXQ6_MODE_TAB_CS  ? ((vx) ? (K<POL, PXQ6_MODE_TAB_CS,  true>) : (K<POL, PXQ6_MODE_TAB_CS,  false>)) : \
     (m) == PXQ6_MODE_PRMT_CS ? ((vx) ? (K<POL, PXQ6_MODE_PRMT_CS, true>) : (K<POL, PXQ6_MODE_PRMT_CS, false>)) : \
                                ((vx) ? (K<POL, PXQ6_MODE_TAB,     true>) : (K<POL, PXQ6_MODE_TAB,     false>)))
#define PXQ6_PICKM23(K, POL, m, vx) /* P2/P3: tab + shfl paths only */ \
    ((m) == PXQ6_MODE_SHFL \
        ? ((vx) ? (K<POL, PXQ6_MODE_SHFL,   true>) : (K<POL, PXQ6_MODE_SHFL,   false>)) \
        : ((m) == PXQ6_MODE_TAB_CS || (m) == PXQ6_MODE_PRMT_CS) \
        ? ((vx) ? (K<POL, PXQ6_MODE_TAB_CS, true>) : (K<POL, PXQ6_MODE_TAB_CS, false>)) \
        : ((vx) ? (K<POL, PXQ6_MODE_TAB,    true>) : (K<POL, PXQ6_MODE_TAB,    false>)))
// P3-only: PXA_PXQ3_PAIRLUT=1 promotes BOTH tab modes to the 64-entry bit-plane pair-LUT
// (PAIRL3+LDCS not offered, mirroring PAIRL). Bit-exact sourcing => the A/B is env-clean.
#define PXQ6_PICKM3(K, POL, m, vx) \
    (pxa_pxq3_pairlut() \
        ? ((vx) ? (K<POL, PXQ6_MODE_PAIRL3, true>) : (K<POL, PXQ6_MODE_PAIRL3, false>)) \
        : PXQ6_PICKM23(K, POL, m, vx))
#define PXQ6_PICKM_FMT(RET, NAME, K) \
    static inline RET NAME(int fmt, int m, bool vx) { \
        switch (fmt) { \
            case PXA_PXQ_FMT_P6: return PXQ6_PICKM(K, pxq6_pol_p6,   m, vx); \
            case PXA_PXQ_FMT_P2: return PXQ6_PICKM23(K, pxq6_pol_p2, m, vx); \
            case PXA_PXQ_FMT_P3: return PXQ6_PICKM3(K, pxq6_pol_p3, m, vx); \
            case PXA_PXQ_FMT_P6R: return PXQ6_PICKM23(K, pxq6_pol_p6r, m, vx); \
            case PXA_PXQ_FMT_P1: return PXQ6_PICKM23(K, pxq6_pol_p1, m, vx); \
            default:             return PXQ6_PICKM(K, pxq6_pol_p6hq, m, vx); \
        } \
    }

#define PXQ6_PICKMU(K, PU, PG, m, vx) \
    ((m) == PXQ6_MODE_SHFL    ? ((vx) ? (K<PU, PG, PXQ6_MODE_SHFL,    true>) : (K<PU, PG, PXQ6_MODE_SHFL,    false>)) : \
     (m) == PXQ6_MODE_PAIRL   ? ((vx) ? (K<PU, PG, PXQ6_MODE_PAIRL,   true>) : (K<PU, PG, PXQ6_MODE_PAIRL,   false>)) : \
     (m) == PXQ6_MODE_PRMT    ? ((vx) ? (K<PU, PG, PXQ6_MODE_PRMT,    true>) : (K<PU, PG, PXQ6_MODE_PRMT,    false>)) : \
     (m) == PXQ6_MODE_TAB_CS  ? ((vx) ? (K<PU, PG, PXQ6_MODE_TAB_CS,  true>) : (K<PU, PG, PXQ6_MODE_TAB_CS,  false>)) : \
     (m) == PXQ6_MODE_PRMT_CS ? ((vx) ? (K<PU, PG, PXQ6_MODE_PRMT_CS, true>) : (K<PU, PG, PXQ6_MODE_PRMT_CS, false>)) : \
                                ((vx) ? (K<PU, PG, PXQ6_MODE_TAB,     true>) : (K<PU, PG, PXQ6_MODE_TAB,     false>)))
#define PXQ6_PICKMU23(K, PU, PG, m, vx) /* any P2/P3 operand: tab + shfl paths only */ \
    ((m) == PXQ6_MODE_SHFL \
        ? ((vx) ? (K<PU, PG, PXQ6_MODE_SHFL,   true>) : (K<PU, PG, PXQ6_MODE_SHFL,   false>)) \
        : ((m) == PXQ6_MODE_TAB_CS || (m) == PXQ6_MODE_PRMT_CS) \
        ? ((vx) ? (K<PU, PG, PXQ6_MODE_TAB_CS, true>) : (K<PU, PG, PXQ6_MODE_TAB_CS, false>)) \
        : ((vx) ? (K<PU, PG, PXQ6_MODE_TAB,    true>) : (K<PU, PG, PXQ6_MODE_TAB,    false>)))
#define PXQ6_PICKMU3(K, PU, PG, m, vx) /* P3 up+gate pair: see PXQ6_PICKM3 */ \
    (pxa_pxq3_pairlut() \
        ? ((vx) ? (K<PU, PG, PXQ6_MODE_PAIRL3, true>) : (K<PU, PG, PXQ6_MODE_PAIRL3, false>)) \
        : PXQ6_PICKMU23(K, PU, PG, m, vx))
#define PXQ6_PICKM_FMT_GU(RET, NAME, K) \
    static inline RET NAME(int fu, int fg, int m, bool vx) { \
        if (fu == fg) switch (fu) { \
            case PXA_PXQ_FMT_P6: return PXQ6_PICKMU(K, pxq6_pol_p6, pxq6_pol_p6, m, vx); \
            case PXA_PXQ_FMT_P2: return PXQ6_PICKMU23(K, pxq6_pol_p2, pxq6_pol_p2, m, vx); \
            case PXA_PXQ_FMT_P3: return PXQ6_PICKMU3(K, pxq6_pol_p3, pxq6_pol_p3, m, vx); \
            case PXA_PXQ_FMT_P6R: return PXQ6_PICKMU23(K, pxq6_pol_p6r, pxq6_pol_p6r, m, vx); \
            case PXA_PXQ_FMT_P1: return PXQ6_PICKMU23(K, pxq6_pol_p1, pxq6_pol_p1, m, vx); \
            default:             return PXQ6_PICKMU(K, pxq6_pol_p6hq, pxq6_pol_p6hq, m, vx); \
        } \
        switch (fu*8 + fg) {   /* mixed pairs: universal files only (P2/P3/P6), tab modes only */ \
            case PXA_PXQ_FMT_P2*8+PXA_PXQ_FMT_P3: return PXQ6_PICKMU23(K, pxq6_pol_p2, pxq6_pol_p3, m, vx); \
            case PXA_PXQ_FMT_P2*8+PXA_PXQ_FMT_P6: return PXQ6_PICKMU23(K, pxq6_pol_p2, pxq6_pol_p6, m, vx); \
            case PXA_PXQ_FMT_P3*8+PXA_PXQ_FMT_P2: return PXQ6_PICKMU23(K, pxq6_pol_p3, pxq6_pol_p2, m, vx); \
            case PXA_PXQ_FMT_P3*8+PXA_PXQ_FMT_P6: return PXQ6_PICKMU23(K, pxq6_pol_p3, pxq6_pol_p6, m, vx); \
            case PXA_PXQ_FMT_P6*8+PXA_PXQ_FMT_P2: return PXQ6_PICKMU23(K, pxq6_pol_p6, pxq6_pol_p2, m, vx); \
            case PXA_PXQ_FMT_P6*8+PXA_PXQ_FMT_P3: return PXQ6_PICKMU23(K, pxq6_pol_p6, pxq6_pol_p3, m, vx); \
            case PXA_PXQ_FMT_P1*8+PXA_PXQ_FMT_P2: return PXQ6_PICKMU23(K, pxq6_pol_p1, pxq6_pol_p2, m, vx); \
            case PXA_PXQ_FMT_P1*8+PXA_PXQ_FMT_P3: return PXQ6_PICKMU23(K, pxq6_pol_p1, pxq6_pol_p3, m, vx); \
            case PXA_PXQ_FMT_P1*8+PXA_PXQ_FMT_P6: return PXQ6_PICKMU23(K, pxq6_pol_p1, pxq6_pol_p6, m, vx); \
            case PXA_PXQ_FMT_P2*8+PXA_PXQ_FMT_P1: return PXQ6_PICKMU23(K, pxq6_pol_p2, pxq6_pol_p1, m, vx); \
            case PXA_PXQ_FMT_P3*8+PXA_PXQ_FMT_P1: return PXQ6_PICKMU23(K, pxq6_pol_p3, pxq6_pol_p1, m, vx); \
            case PXA_PXQ_FMT_P6*8+PXA_PXQ_FMT_P1: return PXQ6_PICKMU23(K, pxq6_pol_p6, pxq6_pol_p1, m, vx); \
            default: return nullptr;   /* unsupported mix -> driver declines (fallback) */ \
        } \
    }
PXQ6_PICKM_FMT_GU(pxq6_gateup_fn,     pxq6_pick_gateup,            k_pxq6_gateup_mmv)
PXQ6_PICKM_FMT(pxq6_mmv_fn,        pxq6_pick_mmv,               k_pxq6_mmv)
PXQ6_PICKM_FMT(pxq6_mmv_ks_fn,     pxq6_pick_mmv_ksplit,        k_pxq6_mmv_ksplit)
PXQ6_PICKM_FMT(pxq6_mmv_ksg_fn,    pxq6_pick_mmv_ksplit_gen,    k_pxq6_mmv_ksplit_gen)
PXQ6_PICKM_FMT(pxq6_mmv_redfuse_fn, pxq6_pick_mmv_redfuse,      k_pxq6_mmv_redfuse)
PXQ6_PICKM_FMT_GU(pxq6_gateup_ks_fn,  pxq6_pick_gateup_ksplit,     k_pxq6_gateup_mmv_ksplit)
PXQ6_PICKM_FMT_GU(pxq6_gateup_ksg_fn, pxq6_pick_gateup_ksplit_gen, k_pxq6_gateup_mmv_ksplit_gen)
// R2 (PXA_PXQ6_ROWX2) 128-thread two-row twins — full fmt/mode/vecx parity with the incumbents
PXQ6_PICKM_FMT(pxq6_mmv_fn,        pxq6_pick_mmv_x2,               k_pxq6_mmv_x2)
PXQ6_PICKM_FMT(pxq6_mmv_ksg_fn,    pxq6_pick_mmv_ksplit_gen_x2,    k_pxq6_mmv_ksplit_gen_x2)
PXQ6_PICKM_FMT_GU(pxq6_gateup_ksg_fn, pxq6_pick_gateup_ksplit_gen_x2, k_pxq6_gateup_mmv_ksplit_gen_x2)

// Q1 (PXA_PXQ6_QPF) one-slab-lookahead prefetch twins — full fmt/mode/vecx parity with the
// incumbents (dot32_qp mirrors dot32 for every mode, incl. K2d SHFL and PAIRL3 sourcing)
PXQ6_PICKM_FMT(pxq6_mmv_fn,        pxq6_pick_mmv_qp,               k_pxq6_mmv_qp)
PXQ6_PICKM_FMT(pxq6_mmv_ksg_fn,    pxq6_pick_mmv_ksplit_gen_qp,    k_pxq6_mmv_ksplit_gen_qp)
PXQ6_PICKM_FMT_GU(pxq6_gateup_ksg_fn, pxq6_pick_gateup_ksplit_gen_qp, k_pxq6_gateup_mmv_ksplit_gen_qp)

// TOK-batched gateup gen-split picker (PXA_PXQ_TOKBATCH). Offered for the SHIPPED decode
// sourcing only -- VECX=1 with MODE TAB, plus the P3-P3 PAIRL3 promotion (mirroring
// PXQ6_PICKMU3). Every other mode/vecx combination returns nullptr; the driver logs a
// DECLINE and falls back to the per-token kernel, which produces the same bytes.
typedef void (*pxq6_gateup_ksg_tok_fn)(const uint8_t *, const uint8_t *, const char *, size_t,
        float *, const char *, size_t, size_t, int, int, int, int, int, int);

template <int TOK>
static inline pxq6_gateup_ksg_tok_fn pxq6_pick_gu_ksg_tok_t(int fu, int fg) {
    if (fu == fg) switch (fu) {
        case PXA_PXQ_FMT_P6:  return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p6,  pxq6_pol_p6,  PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P2:  return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p2,  pxq6_pol_p2,  PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P3:  return pxa_pxq3_pairlut()
            ? (pxq6_gateup_ksg_tok_fn)k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p3, pxq6_pol_p3, PXQ6_MODE_PAIRL3, true, TOK>
            : (pxq6_gateup_ksg_tok_fn)k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p3, pxq6_pol_p3, PXQ6_MODE_TAB,    true, TOK>;
        case PXA_PXQ_FMT_P6R: return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p6r, pxq6_pol_p6r, PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P1:  return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p1,  pxq6_pol_p1,  PXQ6_MODE_TAB, true, TOK>;
        default:              return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p6hq, pxq6_pol_p6hq, PXQ6_MODE_TAB, true, TOK>;
    }
    switch (fu*8 + fg) {   // mixed pairs: the same set the per-token picker serves, tab sourcing
        case PXA_PXQ_FMT_P2*8+PXA_PXQ_FMT_P3: return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p2, pxq6_pol_p3, PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P2*8+PXA_PXQ_FMT_P6: return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p2, pxq6_pol_p6, PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P3*8+PXA_PXQ_FMT_P2: return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p3, pxq6_pol_p2, PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P3*8+PXA_PXQ_FMT_P6: return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p3, pxq6_pol_p6, PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P6*8+PXA_PXQ_FMT_P2: return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p6, pxq6_pol_p2, PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P6*8+PXA_PXQ_FMT_P3: return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p6, pxq6_pol_p3, PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P1*8+PXA_PXQ_FMT_P2: return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p1, pxq6_pol_p2, PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P1*8+PXA_PXQ_FMT_P3: return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p1, pxq6_pol_p3, PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P1*8+PXA_PXQ_FMT_P6: return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p1, pxq6_pol_p6, PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P2*8+PXA_PXQ_FMT_P1: return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p2, pxq6_pol_p1, PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P3*8+PXA_PXQ_FMT_P1: return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p3, pxq6_pol_p1, PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P6*8+PXA_PXQ_FMT_P1: return k_pxq6_gateup_mmv_ksplit_gen_tok<pxq6_pol_p6, pxq6_pol_p1, PXQ6_MODE_TAB, true, TOK>;
        default: return nullptr;   // unsupported mix -> driver declines (per-token fallback)
    }
}

static inline pxq6_gateup_ksg_tok_fn pxq6_pick_gateup_ksplit_gen_tok(int fu, int fg, int m, bool vx, int tok) {
    if (!vx || m != PXQ6_MODE_TAB) return nullptr;   // shipped sourcing only (PAIRL3 promoted inside)
    if (tok == 2) return pxq6_pick_gu_ksg_tok_t<2>(fu, fg);
    if (tok == 3) return pxq6_pick_gu_ksg_tok_t<3>(fu, fg);
    return nullptr;
}

// TOK-batched mmv gen-split picker (PXA_PXQ_MMV_TOK) — the single-matrix twin of the gateup
// tok picker above, same shipped-sourcing restriction (TAB x VECX=1 only; every other
// mode/vecx returns nullptr and the driver logs a DECLINE and falls back to the per-token
// kernel, which produces the same bytes). TOK=4 is offered on this arm: one matrix walk
// leaves more register headroom than the gateup U+G pair (gate with cuobjdump -res-usage).
typedef void (*pxq6_mmv_ksg_tok_fn)(const uint8_t *, const char *, size_t, size_t, float *,
        const char *, size_t, size_t, int, int, int, int, int, int);
template <int TOK>
static inline pxq6_mmv_ksg_tok_fn pxq6_pick_mmv_ksg_tok_t(int fmt) {
    switch (fmt) {
        case PXA_PXQ_FMT_P6:  return k_pxq6_mmv_ksplit_gen_tok<pxq6_pol_p6,   PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P2:  return k_pxq6_mmv_ksplit_gen_tok<pxq6_pol_p2,   PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P3:  return k_pxq6_mmv_ksplit_gen_tok<pxq6_pol_p3,   PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P6R: return k_pxq6_mmv_ksplit_gen_tok<pxq6_pol_p6r,  PXQ6_MODE_TAB, true, TOK>;
        case PXA_PXQ_FMT_P1:  return k_pxq6_mmv_ksplit_gen_tok<pxq6_pol_p1,   PXQ6_MODE_TAB, true, TOK>;
        default:              return k_pxq6_mmv_ksplit_gen_tok<pxq6_pol_p6hq, PXQ6_MODE_TAB, true, TOK>;
    }
}
static inline pxq6_mmv_ksg_tok_fn pxq6_pick_mmv_ksplit_gen_tok(int fmt, int m, bool vx, int tok) {
    if (!vx || m != PXQ6_MODE_TAB) return nullptr;   // shipped decode sourcing only
    if (tok == 2) return pxq6_pick_mmv_ksg_tok_t<2>(fmt);
    if (tok == 3) return pxq6_pick_mmv_ksg_tok_t<3>(fmt);
    if (tok == 4) return pxq6_pick_mmv_ksg_tok_t<4>(fmt);
    return nullptr;
}

// PXA_PXQ_MMV_TOK env (0 = OFF default; 2/3/4 = TOK). 1 is meaningless (that IS the
// per-token kernel); >4 is register-unsafe. Requires the K8-2D S-split machinery
// (PXA_PXQ4_2D_SPLIT, default on): with the split lever off this lever never engages and
// the ENGAGED banner will not print -- such an A/B is VOID, not null.
static inline int pxa_pxq_mmv_tok() {
    static const int v = [](){
        const char * e = getenv("PXA_PXQ_MMV_TOK");
        int t = e ? atoi(e) : 0;
        if (t != 0 && (t < 2 || t > 4)) {
            fprintf(stderr, "PXA_PXQ_MMV_TOK=%d invalid (want 0/2/3/4) -- OFF\n", t);
            t = 0;
        }
        if (t) fprintf(stderr, "PXA_PXQ_MMV_TOK: TOK=%d armed (dense-2D mmv token-batching, bit-exact; "
                               "an A/B without the in-run ENGAGED banner is void)\n", t);
        return t;
    }();
    return v;
}

// first-fire / first-decline banner per device (phantom-lever discipline).
static inline void pxa_pxq_mmv_tok_log(int device, bool fired, int tok, unsigned gx, unsigned gy,
                                       unsigned gz, int ny, int S, size_t smem, int fmt,
                                       int m, int vx) {
    static std::atomic<uint32_t> seen{0};
    if (device < 0 || device >= 16) return;
    const uint32_t bit = 1u << (2*device + (fired ? 0 : 1));
    if (seen.fetch_or(bit) & bit) return;
    if (fired) {
        fprintf(stderr, "PXA_PXQ_MMV_TOK dev%d: ENGAGED tok=%d grid=%ux%ux%u ny=%d S=%d smem=%zuB fmt=%d mode=%d vecx=%d\n",
                device, tok, gx, gy, gz, ny, S, smem, fmt, m, vx);
    } else {
        fprintf(stderr, "PXA_PXQ_MMV_TOK dev%d: DECLINED -> per-token mmv (tok=%d ny=%d S=%d smem=%zuB fmt=%d mode=%d vecx=%d: no instantiation for this fmt/mode/vecx or smem > 46K)\n",
                device, tok, ny, S, smem, fmt, m, vx);
    }
}
PXQ6_PICK_FMT(pxq6_gemm_fn,       pxq6_pick_gemm,              k_pxq6_gemm_grouped)
PXQ6_PICK_FMT(pxq6_scat_fn,       pxq6_pick_down_scat,         k_pxq6_gemm_down_scat)

#define PXQ6_PICK2T(K, POL, T, b1, b2) \
    ((b1) ? ((b2) ? (K<POL, T, true, true>) : (K<POL, T, true, false>)) \
          : ((b2) ? (K<POL, T, false, true>) : (K<POL, T, false, false>)))
static inline pxq6_gufuse_h_fn pxq6_pick_gufuse_h(int fmt, bool rag, bool pipe) {
    switch (fmt) {
        case PXA_PXQ_FMT_P6: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p6,   half, rag, pipe);
        case PXA_PXQ_FMT_P2: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p2,   half, rag, pipe);
        case PXA_PXQ_FMT_P3: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p3,   half, rag, pipe);
        case PXA_PXQ_FMT_P6R: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p6r, half, rag, pipe);
        case PXA_PXQ_FMT_P1: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p1,  half, rag, pipe);
        default:             return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p6hq, half, rag, pipe);
    }
}
static inline pxq6_gufuse_f_fn pxq6_pick_gufuse_f(int fmt, bool rag, bool pipe) {
    switch (fmt) {
        case PXA_PXQ_FMT_P6: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p6,   float, rag, pipe);
        case PXA_PXQ_FMT_P2: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p2,   float, rag, pipe);
        case PXA_PXQ_FMT_P3: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p3,   float, rag, pipe);
        case PXA_PXQ_FMT_P6R: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p6r, float, rag, pipe);
        case PXA_PXQ_FMT_P1: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p1,  float, rag, pipe);
        default:             return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p6hq, float, rag, pipe);
    }
}
static inline pxq6_gemm_fn pxq6_pick_gemm_wmma(int fmt, bool f32acc) {
    switch (fmt) {
        case PXA_PXQ_FMT_P6: return f32acc ? (pxq6_gemm_fn)k_pxq6_gemm_grouped_wmma<pxq6_pol_p6,   true> : (pxq6_gemm_fn)k_pxq6_gemm_grouped_wmma<pxq6_pol_p6,   false>;
        default:             return f32acc ? (pxq6_gemm_fn)k_pxq6_gemm_grouped_wmma<pxq6_pol_p6hq, true> : (pxq6_gemm_fn)k_pxq6_gemm_grouped_wmma<pxq6_pol_p6hq, false>;
    }
}
// K6 v2 pickers: 4 policies (P2/P3/P6/P6HQ) + mixed up/gate pairs; P6R -> nullptr (excluded)
#define PXQ6_WMMA_GU_PICK(T, PU, fg) \
    ((fg) == PXA_PXQ_FMT_P6   ? (k_pxq6_gemm_gufuse_wmma<PU, pxq6_pol_p6,   T>) : \
     (fg) == PXA_PXQ_FMT_P6HQ ? (k_pxq6_gemm_gufuse_wmma<PU, pxq6_pol_p6hq, T>) : \
     (fg) == PXA_PXQ_FMT_P2   ? (k_pxq6_gemm_gufuse_wmma<PU, pxq6_pol_p2,   T>) : \
     (fg) == PXA_PXQ_FMT_P3   ? (k_pxq6_gemm_gufuse_wmma<PU, pxq6_pol_p3,   T>) : nullptr)
static inline pxq6_gufuse_h_fn pxq6_pick_gufuse_wmma_h(int fu, int fg) {
    switch (fu) {
        case PXA_PXQ_FMT_P6:   return PXQ6_WMMA_GU_PICK(half, pxq6_pol_p6,   fg);
        case PXA_PXQ_FMT_P6HQ: return PXQ6_WMMA_GU_PICK(half, pxq6_pol_p6hq, fg);
        case PXA_PXQ_FMT_P2:   return PXQ6_WMMA_GU_PICK(half, pxq6_pol_p2,   fg);
        case PXA_PXQ_FMT_P3:   return PXQ6_WMMA_GU_PICK(half, pxq6_pol_p3,   fg);
        default:               return nullptr;
    }
}
static inline pxq6_gufuse_f_fn pxq6_pick_gufuse_wmma_f(int fu, int fg) {
    switch (fu) {
        case PXA_PXQ_FMT_P6:   return PXQ6_WMMA_GU_PICK(float, pxq6_pol_p6,   fg);
        case PXA_PXQ_FMT_P6HQ: return PXQ6_WMMA_GU_PICK(float, pxq6_pol_p6hq, fg);
        case PXA_PXQ_FMT_P2:   return PXQ6_WMMA_GU_PICK(float, pxq6_pol_p2,   fg);
        case PXA_PXQ_FMT_P3:   return PXQ6_WMMA_GU_PICK(float, pxq6_pol_p3,   fg);
        default:               return nullptr;
    }
}
static inline pxq6_scat_fn pxq6_pick_down_scat_wmma(int fmt) {
    switch (fmt) {
        case PXA_PXQ_FMT_P6:   return k_pxq6_gemm_down_scat_wmma<pxq6_pol_p6>;
        case PXA_PXQ_FMT_P6HQ: return k_pxq6_gemm_down_scat_wmma<pxq6_pol_p6hq>;
        case PXA_PXQ_FMT_P2:   return k_pxq6_gemm_down_scat_wmma<pxq6_pol_p2>;
        case PXA_PXQ_FMT_P3:   return k_pxq6_gemm_down_scat_wmma<pxq6_pol_p3>;
        default:               return nullptr;
    }
}
