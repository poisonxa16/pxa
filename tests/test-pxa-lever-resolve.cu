// test-pxa-lever-resolve.cu -- the effective value of every PXA_* lever that pxa-enhance.cuh and the
// core lever registry (ggml-cuda/pxa/core/levers.cu) resolve, for a card set, a model profile and an
// environment given on the command line, and the log lines the resolvers print while doing it.
//
// Host only: no card is touched, no CUDA context is created. It reads the same inline resolvers the
// engine's translation units read, over the real library.
//
// WHY IT EXISTS. The registry takes over resolvers one group at a time under the rule "same
// defaults, same semantics, same log line". A rule like that is only worth anything if somebody
// can DIFF it: this binary is built once from the tree before a change and once from the tree
// after it, run over the same grid of environments (tests/test-pxa-lever-resolve.py), and the two
// outputs must be byte identical. The grid, and the output it produced on the tree before the
// registry took over the resolvers, are checked in as tests/pxa-lever-resolve.golden.
//
// One process per environment: every resolver caches (per process, or per model-profile
// generation) and the config level is read once, which is exactly the behaviour under test.
//
//   test-pxa-lever-resolve STEP...
//     topo CC[,CC...]|none        register the card set (compute capabilities, e.g. 700,700)
//     prof ARCH EXPERTS USED MMVQ PXQ2 PXQ3 CLASS MTP
//                                 register a model profile (bumps the profile generation)
//     set NAME VALUE | unset NAME mutate the environment mid-run
//     get WHAT                    print WHAT=value (the names are in get_one() and ALL[] below)
//     row NAME                    read ONE registry row (a flash-attention row's short name)
//     all                         get every resolver
//     startup CC[,CC...]          the one-time per-device startup report
//     decisions                   the per-(device x model) decision ledger
//     dbg                         the PXA_ENHANCE_DBG topology line
//     report                      the generated registry report
#include "ggml-cuda/pxa/pxa-enhance.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<int> parse_ccs(const char * s) {
    std::vector<int> v;
    if (!s || !strcmp(s, "none")) {
        return v;
    }
    const char * p = s;
    while (*p) {
        v.push_back(atoi(p));
        const char * c = strchr(p, ',');
        if (!c) break;
        p = c + 1;
    }
    return v;
}

static void out_i(const char * name, long long v) { printf("%s=%lld\n", name, v); }
static void out_s(const char * name, const char * v) { printf("%s=%s\n", name, v ? v : "(null)"); }

// the registry rows that exist in every tree this test is built from (the flash-attention levers)
#define FA_ROWS(X) \
    X(FA_D512_VOLTA) X(FA_D256_VOLTA_TILE) X(FA_D256_VOLTA_TILE_MINKV) X(FA_D256_VOLTA_TILE_MAXCOLS) \
    X(FA_MMA_VOLTA) X(FA_MMA_VOLTA_Q8) X(FA_TILE_VOLTA) X(FA_TILE256) X(FA_SWA_SLICE) X(FA_SWA_KEEP) \
    X(SM60_FA_VEC_F32) X(CORE_ROUTES) X(FA_QKV_DIRECT) X(FA_QKV_TILE) X(FA_QKV_DIRECT_VOLTA)

// ---- the call sites that read a registry row (or, in the tree before the registry took them over,
// evaluated the expression a row now evaluates). Each is written twice: the expression the site used to
// carry, which is what the golden file was produced from, and the read the site carries now. In a tree
// with the rows the harness evaluates the new read and CROSS-CHECKS it against the legacy expression;
// a difference is printed as SITE MISMATCH and fails the run. Nothing is printed when they agree, so the
// output of the two trees stays comparable byte for byte.
static int g_site_mismatch = 0;
static void site_check(const char * what, long long now, long long legacy) {
    if (now != legacy) {
        printf("SITE MISMATCH %s: registry %lld, legacy expression %lld\n", what, now, legacy);
        g_site_mismatch = 1;
    }
}

static int legacy_env_or(const char * env, int dflt) {
    const char * e = getenv(env);
    return e ? atoi(e) : dflt;
}

// pxa_pxq_gemm_2d_mode() as ggml-cuda.cu had it, cache and notice included. WITH_NOTICE=false is the same
// function and the same per-generation cache without the stderr line, for the cross-check.
template <bool WITH_NOTICE>
static int legacy_pxq_gemm_2d_mode() {
    static int cached_gen = -1;
    static int cached     = 0;
    const int gen = ggml_pxa_model_profile_generation();
    if (cached_gen == gen) return cached;
    const char * e = getenv("PXA_PXQ_GEMM_2D");
    int m = e ? atoi(e) : 0;
    if (!e && pxa_config_level() == 2 && g_pxa_topology.valid && g_pxa_topology.has_sm60
        && pxa_model_known() && !pxa_model_is_moe() && pxa_model().n_pxq_mmvq_tensors > 0) {
        m = 1;
        if (WITH_NOTICE) {
            fprintf(stderr, "PXA_AUTO: PXQ_GEMM_2D=1 (ENHANCE x sm_60 present x dense PXQ model: "
                            "+35%% P100 dense prefill measured 2026-07-28; MoE stays off — post-"
                            "coalescing sm_60 MoE cell unmeasured; override PXA_PXQ_GEMM_2D)\n");
        }
    }
    if (m < 0 || m > 2) m = 0;
    cached = m;
    cached_gen = gen;
    return m;
}

// pxa_cuda_house_lever() as ggml-cuda/common.cuh had it, and pxa_fa_mask_skip_tile_f32() as fattn-tile-f32.cu
// had it (a function-local static: one resolution, one optional debug line)
static bool legacy_house_lever(const char * name) {
    const char * e = getenv(name);
    if (e) {
        return atoi(e) != 0;
    }
    return ggml_pxa_config_level() >= 2;
}
template <bool WITH_NOTICE>
static bool legacy_fa_mask_skip_tile_f32() {
    static const bool v = [](){
        const char * e = getenv("PXA_FA_MASK_SKIP_TILE_F32");
        const bool on = e != nullptr && atoi(e) != 0;
        const char * d = getenv("PXA_ENHANCE_DBG");
        if (WITH_NOTICE && d && atoi(d) != 0) {
            fprintf(stderr, "PXA_ENHANCE_DBG: fa_mask_skip_tile_f32=%s (%s)\n", on ? "on" : "off", e ? "env" : "default");
        }
        return on;
    }();
    return v;
}

static void get_house_sites() {
#define HOUSE(n) X(n, "PXA_" #n)
#define HOUSE_ROWS(X) HOUSE(NORM_REGCACHE) HOUSE(CONCAT_FLAT) HOUSE(CPY_FASTDIV) HOUSE(GETROWS_NARROW) HOUSE(TOPK_MOE_MULTIROW)
#ifdef PXA_LEVER_MOVED_ROWS
// (each call site cached its answer in a function-local static: one static per lever here too)
#define X(n, env) { static const bool legacy = legacy_house_lever(env); const bool now = pxa_lever(PXA_LEVER_##n) != 0; \
                    site_check(env, now, legacy); out_i("site_house_" #n, now); }
    HOUSE_ROWS(X)
#undef X
    const bool f32 = pxa_lever(PXA_LEVER_FA_MASK_SKIP_TILE_F32) != 0;
    site_check("FA_MASK_SKIP_TILE_F32", f32, legacy_fa_mask_skip_tile_f32<false>());
#else
#define X(n, env) { static const bool legacy = legacy_house_lever(env); out_i("site_house_" #n, legacy); }
    HOUSE_ROWS(X)
#undef X
    const bool f32 = legacy_fa_mask_skip_tile_f32<true>();
#endif
    out_i("site_fa_mask_skip_tile_f32", f32);
}

static void get_sites() {
#ifdef PXA_LEVER_MOVED_ROWS
    const int gqa  = (int) pxa_lever(PXA_LEVER_FA_GQA_PACK);
    const int moe  = (int) pxa_lever(PXA_LEVER_MOE_DEVICE_MAP);
    const int dn   = (int) pxa_lever(PXA_LEVER_FUSE_DELTANET);
    const int mmvq = (int) pxa_lever(PXA_LEVER_PXQ_MMVQ);
    const bool mmvq_set = pxa_lever_set_by_user(PXA_LEVER_PXQ_MMVQ);
    const int g2d  = (int) pxa_lever(PXA_LEVER_PXQ_GEMM_2D);
    const int dbg  = (int) pxa_lever(PXA_LEVER_ENHANCE_DBG);
    site_check("FA_GQA_PACK", gqa, legacy_env_or("PXA_FA_GQA_PACK", pxa_fa_gqa_pack_default()));
    site_check("MOE_DEVICE_MAP", moe, legacy_env_or("PXA_MOE_DEVICE_MAP", pxa_moe_device_map_default()));
    site_check("FUSE_DELTANET", dn, legacy_env_or("PXA_FUSE_DELTANET", pxa_fuse_deltanet_default()));
    site_check("PXQ_MMVQ", mmvq, legacy_env_or("PXA_PXQ_MMVQ", pxa_pxq_mmvq_auto_default()));
    site_check("PXQ_MMVQ set", mmvq_set, getenv("PXA_PXQ_MMVQ") != nullptr);
    site_check("PXQ_GEMM_2D", g2d, legacy_pxq_gemm_2d_mode<false>());
    site_check("ENHANCE_DBG", dbg, getenv("PXA_ENHANCE_DBG") && atoi(getenv("PXA_ENHANCE_DBG")) != 0);
#else
    const int gqa  = legacy_env_or("PXA_FA_GQA_PACK", pxa_fa_gqa_pack_default());
    const int moe  = legacy_env_or("PXA_MOE_DEVICE_MAP", pxa_moe_device_map_default());
    const int dn   = legacy_env_or("PXA_FUSE_DELTANET", pxa_fuse_deltanet_default());
    const int mmvq = legacy_env_or("PXA_PXQ_MMVQ", pxa_pxq_mmvq_auto_default());
    const bool mmvq_set = getenv("PXA_PXQ_MMVQ") != nullptr;
    const int g2d  = legacy_pxq_gemm_2d_mode<true>();
    const int dbg  = getenv("PXA_ENHANCE_DBG") && atoi(getenv("PXA_ENHANCE_DBG")) != 0;
#endif
    out_i("site_fa_gqa_pack", gqa);
    out_i("site_moe_device_map", moe);
    out_i("site_fuse_deltanet", dn);
    out_i("site_pxq_mmvq_raw", mmvq);
    out_i("site_pxq_mmvq_set", mmvq_set);
    out_i("site_pxq_gemm_2d", g2d);
    out_i("site_enhance_dbg", dbg);
}

static bool get_one(const char * w) {
    if (!strcmp(w, "level"))        { out_i("level", pxa_config_level()); out_s("level_name", pxa_config_level_name());
                                      out_s("level_why", pxa_config_level_why()); return true; }
    if (!strcmp(w, "pxq23_mmvq"))   { out_i(w, pxa_pxq23_mmvq_mask()); return true; }
    if (!strcmp(w, "pxq_mmv_h2"))   { out_i(w, pxa_pxq_mmv_h2_mask()); return true; }
    if (!strcmp(w, "pxq_mmvq_auto")){ out_i(w, pxa_pxq_mmvq_auto_default()); return true; }
    if (!strcmp(w, "fuse_deltanet_default")) { out_i(w, pxa_fuse_deltanet_default()); return true; }
    if (!strcmp(w, "volta_cublas_ne11")) { out_i(w, pxa_volta_cublas_ne11()); return true; }
    if (!strcmp(w, "int8_prefill")) { out_i(w, pxa_int8_prefill_mode_resolve()); return true; }
    if (!strcmp(w, "router_fuse"))  { out_i(w, pxa_router_fuse_mode_resolve());
                                      out_i("router_fuse_on600", pxa_router_fuse_on(600));
                                      out_i("router_fuse_on610", pxa_router_fuse_on(610));
                                      out_i("router_fuse_on700", pxa_router_fuse_on(700)); return true; }
    if (!strcmp(w, "spec_relaxed")) { out_i(w, pxa_spec_relaxed_resolve()); return true; }
    if (!strcmp(w, "spec_1row"))    { out_i(w, pxa_spec_1row_resolve()); return true; }
    if (!strcmp(w, "p100_fp16_gemm")) { out_i(w, pxa_p100_fp16_gemm()); return true; }
    if (!strcmp(w, "mode"))         { out_i(w, pxa_mode()); out_s("mode_name", pxa_mode_name()); return true; }
    if (!strcmp(w, "fa_mask_skip_tile")) { out_i(w, pxa_fa_mask_skip_tile()); return true; }
    if (!strcmp(w, "fa_tile_f32acc")) { out_i(w, pxa_fa_tile_f32acc()); return true; }
    if (!strcmp(w, "fa_tile_v2"))   { out_i(w, pxa_fa_tile_v2()); return true; }
    if (!strcmp(w, "fa_prefill_split")) { out_i(w, pxa_fa_prefill_split_ne11()); return true; }
    if (!strcmp(w, "fa_gqa_pack_default")) { out_i(w, pxa_fa_gqa_pack_default()); return true; }
    if (!strcmp(w, "moe_device_map_default")) { out_i(w, pxa_moe_device_map_default()); return true; }
    if (!strcmp(w, "house"))        { out_i(w, pxa_house_lever_default()); return true; }
    if (!strcmp(w, "gate"))         { out_i("gate_true", pxa_gate_default(true)); out_i("gate_false", pxa_gate_default(false)); return true; }
    if (!strcmp(w, "path"))         { out_s("path600", pxa_enhance_path_name(600)); out_s("path610", pxa_enhance_path_name(610));
                                      out_s("path700", pxa_enhance_path_name(700)); out_s("path800", pxa_enhance_path_name(800)); return true; }
    if (!strcmp(w, "sites"))        { get_sites(); get_house_sites(); return true; }
    if (!strcmp(w, "fa_rows")) {
#define X(n) out_i("row_" #n, pxa_lever(PXA_LEVER_##n)); out_i("row_set_" #n, pxa_lever_set_by_user(PXA_LEVER_##n)); \
             out_s("row_why_" #n, pxa_lever_why(PXA_LEVER_##n));
        FA_ROWS(X)
#undef X
        return true;
    }
#ifdef PXA_LEVER_RESOLVE_CALL_SITES
    PXA_LEVER_RESOLVE_CALL_SITES(w)
#endif
    return false;
}

static const char * ALL[] = {
    "level", "pxq23_mmvq", "pxq_mmv_h2", "pxq_mmvq_auto", "fuse_deltanet_default", "volta_cublas_ne11", "int8_prefill",
    "router_fuse", "spec_relaxed", "spec_1row", "p100_fp16_gemm", "mode", "fa_mask_skip_tile", "fa_tile_f32acc",
    "fa_tile_v2", "fa_prefill_split", "fa_gqa_pack_default", "moe_device_map_default", "house", "gate", "path", "sites", "fa_rows",
};

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    std::vector<int> topo_ccs;
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (!strcmp(a, "topo") && i + 1 < argc) {
            topo_ccs = parse_ccs(argv[++i]);
            pxa_enhance_init_topology((int) topo_ccs.size(), topo_ccs.data());
        } else if (!strcmp(a, "prof") && i + 8 < argc) {
            ggml_pxa_model_profile p;
            memset(&p, 0, sizeof(p));
            snprintf(p.arch_name, sizeof(p.arch_name), "%s", argv[i + 1]);
            p.n_expert           = atoi(argv[i + 2]);
            p.n_expert_used      = atoi(argv[i + 3]);
            p.n_pxq_mmvq_tensors = atoi(argv[i + 4]);
            p.n_pxq2_tensors     = atoi(argv[i + 5]);
            p.n_pxq3_tensors     = atoi(argv[i + 6]);
            p.model_class        = atoi(argv[i + 7]);
            p.has_mtp_head       = atoi(argv[i + 8]);
            p.n_vocab            = 32000;
            i += 8;
            ggml_pxa_set_model_profile(&p);
        } else if (!strcmp(a, "noprof")) {
            ggml_pxa_set_model_profile(nullptr);
        } else if (!strcmp(a, "set") && i + 2 < argc) {
            setenv(argv[i + 1], argv[i + 2], 1);
            i += 2;
        } else if (!strcmp(a, "unset") && i + 1 < argc) {
            unsetenv(argv[++i]);
        } else if (!strcmp(a, "get") && i + 1 < argc) {
            const char * w = argv[++i];
            if (!get_one(w)) {
                printf("UNKNOWN %s\n", w);
            }
        } else if (!strcmp(a, "row") && i + 1 < argc) {
            // ONE registry row, by its short name: reading it must resolve it and nothing else
            const char * w = argv[++i];
            bool found = false;
#define X(n) if (!strcmp(w, #n)) { found = true; out_i("row_" #n, pxa_lever(PXA_LEVER_##n)); }
            FA_ROWS(X)
#undef X
            if (!found) {
                printf("UNKNOWN row %s\n", w);
            }
        } else if (!strcmp(a, "all")) {
            for (const char * n : ALL) {
                get_one(n);
            }
        } else if (!strcmp(a, "startup") && i + 1 < argc) {
            std::vector<int> ccs = parse_ccs(argv[++i]);
            std::vector<std::string> names;
            static char nm[8][256];
            for (size_t k = 0; k < ccs.size() && k < 8; ++k) {
                snprintf(nm[k], sizeof(nm[k]), "Device%zu", k);
            }
            pxa_enhance_log_startup((int) ccs.size(), ccs.data(), nm);
        } else if (!strcmp(a, "decisions")) {
            pxa_enhance_log_model_decisions();
        } else if (!strcmp(a, "dbg")) {
            pxa_enhance_log_topology_dbg();
        } else if (!strcmp(a, "report")) {
            pxa_core_lever_report();
        } else {
            printf("BAD STEP %s\n", a);
            return 2;
        }
    }
    return g_site_mismatch ? 3 : 0;
}
