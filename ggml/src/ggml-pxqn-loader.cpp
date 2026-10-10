// ggml-pxqn-loader.cpp -- finds and checks the closed PXQN library (see ggml-pxqn-api.h). Open.
#include "ggml-pxqn-api.h"
#include "ggml-pxqn-tune.h"
#include "ggml-pxqn-levers.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#if defined(_WIN32)
const struct ggml_pxqn_lib_api * ggml_pxqn_lib(void) { return nullptr; }
ggml_pxqn_cpu_mul_mat_fn ggml_pxqn_cpu_mul_mat_get(void) { return nullptr; }
#else
#include <dlfcn.h>

static const struct ggml_pxqn_lib_api * g_pxqn_lib = nullptr;
static void * g_pxqn_handle = nullptr;
static ggml_pxqn_cpu_mul_mat_fn          g_pxqn_mmv = nullptr;   // optional extra symbol of the library (ggml-pxqn-api.h)
static const struct ggml_pxqn_xcache_policy * g_pxqn_xc = nullptr; // optional: the PXA_XCACHE online-adaptation policy
static ggml_pxqn_xcache_learn_build_fn       g_pxqn_learn_build = nullptr;
static ggml_pxqn_xcache_counts_pick_fn       g_pxqn_counts_pick = nullptr;
static ggml_pxqn_xcache_speed_write_fn       g_pxqn_speed_write = nullptr;
static ggml_pxqn_xcache_calib_step_fn        g_pxqn_calib_step = nullptr;
static ggml_pxqn_xcache_calib_consider_fn    g_pxqn_calib_consider = nullptr;
static ggml_pxqn_xcache_cpupool_create_fn    g_pxqn_cpupool_create = nullptr;
static ggml_pxqn_xcache_cpupool_destroy_fn   g_pxqn_cpupool_destroy = nullptr;
static ggml_pxqn_xcache_cpupool_step_fn      g_pxqn_cpupool_step = nullptr;

static void * pxqn_try_open(const std::string & path, std::string & err) {
    void * h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { const char * e = dlerror(); err = e ? e : "dlopen failed"; }
    return h;
}

static const struct ggml_pxqn_lib_api * pxqn_load(void) {
    const char * dis = getenv("PXA_PXQN_DISABLE");
    if (dis && *dis && atoi(dis) != 0) {
        fprintf(stderr, "pxqn: PXA_PXQN_DISABLE set -- PXQN unavailable\n");
        return nullptr;
    }
    std::string err, tried;
    void * h = nullptr;
    const char * env = getenv("PXA_PXQN_LIB");
    if (env && *env) {
        h = pxqn_try_open(env, err);
        tried = env;
    } else {
        Dl_info di;   // libggml's own directory first: the release tarball ships the two side by side
        if (dladdr((void *) &ggml_pxqn_lib, &di) && di.dli_fname) {
            std::string dir(di.dli_fname);
            const size_t sl = dir.rfind('/');
            dir = sl == std::string::npos ? std::string(".") : dir.substr(0, sl);
            tried = dir + "/" GGML_PXQN_LIB_NAME;
            h = pxqn_try_open(tried, err);
        }
        if (!h) {
            std::string err2;
            h = pxqn_try_open(GGML_PXQN_LIB_NAME, err2);
            if (!h && err.empty()) err = err2;
        }
    }
    if (!h) {
        if (getenv("PXA_PXQN_VERBOSE")) fprintf(stderr, "pxqn: %s not loaded (%s)\n", GGML_PXQN_LIB_NAME, err.c_str());
        return nullptr;
    }
    auto get = (ggml_pxqn_get_api_fn) dlsym(h, "ggml_pxqn_get_api");
    const struct ggml_pxqn_lib_api * api = get ? get(GGML_PXQN_LIB_VERSION) : nullptr;
    char sig[128];
    ggml_pxqn_abi_sig(sig, sizeof(sig));
    if (!api || api->version != GGML_PXQN_LIB_VERSION || api->size != sizeof(struct ggml_pxqn_lib_api) ||
        !api->abi || strcmp(api->abi, sig) != 0 || !api->cpu || api->cpu->version != GGML_PXQN_LIB_VERSION ||
        api->cpu->size != sizeof(struct ggml_pxqn_cpu_api)) {
        fprintf(stderr, "pxqn: %s was built for a different engine build (want %s, got %s) -- PXQN unavailable\n",
                tried.c_str(), sig, api && api->abi ? api->abi : "?");
        dlclose(h);
        return nullptr;
    }
    fprintf(stderr, "pxqn: loaded %s (%s)\n", tried.empty() ? GGML_PXQN_LIB_NAME : tried.c_str(),
            api->build ? api->build : "?");
    g_pxqn_handle = h;
    // the fast CPU matmul is an optional extra symbol: absent in an older library; PXA_PXQN_CPU_MMV=0 leaves it unused
    const char * mmv = getenv("PXA_PXQN_CPU_MMV");
    if (!(mmv && atoi(mmv) == 0)) g_pxqn_mmv = (ggml_pxqn_cpu_mul_mat_fn) dlsym(h, GGML_PXQN_CPU_MUL_MAT_SYM);
    // the online-adaptation policy of the expert cache: an optional extra symbol too (PXA_XCACHE_ADAPT=0 is read by the engine)
    const char * xcp = getenv("PXA_PXQN_XCACHE");
    if (!(xcp && atoi(xcp) == 0)) {
        auto xg = (ggml_pxqn_xcache_get_fn) dlsym(h, GGML_PXQN_XCACHE_SYM);
        const struct ggml_pxqn_xcache_policy * xp = xg ? xg(GGML_PXQN_XCACHE_VERSION) : nullptr;
        if (xp && xp->version == GGML_PXQN_XCACHE_VERSION && xp->size == sizeof(struct ggml_pxqn_xcache_policy)) g_pxqn_xc = xp;
    }
    g_pxqn_learn_build = (ggml_pxqn_xcache_learn_build_fn) dlsym(h, GGML_PXQN_XCACHE_LEARN_BUILD_SYM);
    g_pxqn_counts_pick = (ggml_pxqn_xcache_counts_pick_fn) dlsym(h, GGML_PXQN_XCACHE_COUNTS_PICK_SYM);
    g_pxqn_speed_write = (ggml_pxqn_xcache_speed_write_fn) dlsym(h, GGML_PXQN_XCACHE_SPEED_WRITE_SYM);
    g_pxqn_calib_step = (ggml_pxqn_xcache_calib_step_fn) dlsym(h, GGML_PXQN_XCACHE_CALIB_STEP_SYM);
    g_pxqn_calib_consider = (ggml_pxqn_xcache_calib_consider_fn) dlsym(h, GGML_PXQN_XCACHE_CALIB_CONSIDER_SYM);
    g_pxqn_cpupool_create = (ggml_pxqn_xcache_cpupool_create_fn) dlsym(h, GGML_PXQN_XCACHE_CPUPOOL_CREATE_SYM);
    g_pxqn_cpupool_destroy = (ggml_pxqn_xcache_cpupool_destroy_fn) dlsym(h, GGML_PXQN_XCACHE_CPUPOOL_DESTROY_SYM);
    g_pxqn_cpupool_step = (ggml_pxqn_xcache_cpupool_step_fn) dlsym(h, GGML_PXQN_XCACHE_CPUPOOL_STEP_SYM);
    return api;   // never unloaded: its kernels and per-device state live for the process
}

const struct ggml_pxqn_lib_api * ggml_pxqn_lib(void) {
    static std::once_flag once;
    std::call_once(once, [] { g_pxqn_lib = pxqn_load(); });
    return g_pxqn_lib;
}

const struct ggml_pxqn_lock_api * ggml_pxqn_lock(void) {
    static const struct ggml_pxqn_lock_api * lock = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        if (!ggml_pxqn_lib() || !g_pxqn_handle) return;
        auto get = (ggml_pxqn_get_lock_api_fn) dlsym(g_pxqn_handle, "ggml_pxqn_get_lock_api");
        const struct ggml_pxqn_lock_api * api = get ? get(GGML_PXQN_LOCK_VERSION) : nullptr;
        if (api && api->version == GGML_PXQN_LOCK_VERSION && api->size == sizeof(struct ggml_pxqn_lock_api) && api->open && api->decrypt && api->close) lock = api;
    });
    return lock;
}
#endif

#if defined(_WIN32)
const struct ggml_pxqn_lock_api * ggml_pxqn_lock(void) { return nullptr; }
#endif

bool ggml_pxqn_available(void) {
    return ggml_pxqn_lib() != nullptr;
}

void * ggml_pxqn_lock_open(const char * const * keys, const char * const * vals, int n, char * err, int errlen) {
    const struct ggml_pxqn_lock_api * lk = ggml_pxqn_lock();
    if (!lk) {                           // no libggml-pxqn next to this engine (the open build), or an older one: the one plain message
        if (err && errlen > 0) snprintf(err, (size_t) errlen, "%s", ggml_pxqn_lib() ? GGML_PXQN_LOCKED_OLD_MSG : GGML_PXQN_LOCKED_MSG);
        return nullptr;
    }
    return lk->open(keys, vals, n, err, errlen);
}

void ggml_pxqn_lock_decrypt(void * handle, const char * name, uint64_t off, void * buf, size_t n) {
    const struct ggml_pxqn_lock_api * lk = ggml_pxqn_lock();
    if (lk && handle) lk->decrypt(handle, name, off, buf, n);
}

void ggml_pxqn_lock_close(void * handle) {
    const struct ggml_pxqn_lock_api * lk = ggml_pxqn_lock();
    if (lk && handle) lk->close(handle);
}

#if !defined(_WIN32)
ggml_pxqn_cpu_mul_mat_fn ggml_pxqn_cpu_mul_mat_get(void) {
    return ggml_pxqn_lib() ? g_pxqn_mmv : nullptr;
}
const struct ggml_pxqn_xcache_policy * ggml_pxqn_xcache_policy_get(void) {
    return ggml_pxqn_lib() ? g_pxqn_xc : nullptr;
}
ggml_pxqn_xcache_learn_build_fn ggml_pxqn_xcache_learn_build_get(void) {
    return ggml_pxqn_lib() ? g_pxqn_learn_build : nullptr;
}
ggml_pxqn_xcache_counts_pick_fn ggml_pxqn_xcache_counts_pick_get(void) {
    return ggml_pxqn_lib() ? g_pxqn_counts_pick : nullptr;
}
ggml_pxqn_xcache_speed_write_fn ggml_pxqn_xcache_speed_write_get(void) {
    return ggml_pxqn_lib() ? g_pxqn_speed_write : nullptr;
}
ggml_pxqn_xcache_calib_step_fn ggml_pxqn_xcache_calib_step_get(void) {
    return ggml_pxqn_lib() ? g_pxqn_calib_step : nullptr;
}
ggml_pxqn_xcache_calib_consider_fn ggml_pxqn_xcache_calib_consider_get(void) {
    return ggml_pxqn_lib() ? g_pxqn_calib_consider : nullptr;
}
ggml_pxqn_xcache_cpupool_create_fn ggml_pxqn_xcache_cpupool_create_get(void) {
    return ggml_pxqn_lib() ? g_pxqn_cpupool_create : nullptr;
}
ggml_pxqn_xcache_cpupool_destroy_fn ggml_pxqn_xcache_cpupool_destroy_get(void) {
    return ggml_pxqn_lib() ? g_pxqn_cpupool_destroy : nullptr;
}
ggml_pxqn_xcache_cpupool_step_fn ggml_pxqn_xcache_cpupool_step_get(void) {
    return ggml_pxqn_lib() ? g_pxqn_cpupool_step : nullptr;
}
#else
const struct ggml_pxqn_xcache_policy * ggml_pxqn_xcache_policy_get(void) { return nullptr; }
ggml_pxqn_xcache_learn_build_fn ggml_pxqn_xcache_learn_build_get(void) { return nullptr; }
ggml_pxqn_xcache_counts_pick_fn ggml_pxqn_xcache_counts_pick_get(void) { return nullptr; }
ggml_pxqn_xcache_speed_write_fn ggml_pxqn_xcache_speed_write_get(void) { return nullptr; }
ggml_pxqn_xcache_calib_step_fn ggml_pxqn_xcache_calib_step_get(void) { return nullptr; }
ggml_pxqn_xcache_calib_consider_fn ggml_pxqn_xcache_calib_consider_get(void) { return nullptr; }
ggml_pxqn_xcache_cpupool_create_fn ggml_pxqn_xcache_cpupool_create_get(void) { return nullptr; }
ggml_pxqn_xcache_cpupool_destroy_fn ggml_pxqn_xcache_cpupool_destroy_get(void) { return nullptr; }
ggml_pxqn_xcache_cpupool_step_fn ggml_pxqn_xcache_cpupool_step_get(void) { return nullptr; }
#endif

bool ggml_pxqn_cpu_mmv_available(void) {
    return ggml_pxqn_cpu_mul_mat_get() != nullptr;
}

// ---- the speculation selector hook (ggml-pxqn-spec.h): thin wrappers over the library's optional table ----
#if defined(_WIN32)
const struct ggml_pxqn_spec_api * ggml_pxqn_spec_table(void) { return nullptr; }
#else
const struct ggml_pxqn_spec_api * ggml_pxqn_spec_table(void) {
    static std::once_flag once;
    static const struct ggml_pxqn_spec_api * t = nullptr;
    std::call_once(once, [] {
        const char * off = getenv("PXA_SPEC_SELECT");
        if (off && *off && atoi(off) == 0) return;                       // lever off: never looked up
        if (!ggml_pxqn_lib() || !g_pxqn_handle) return;
        auto get = (ggml_pxqn_get_spec_api_fn) dlsym(g_pxqn_handle, "ggml_pxqn_get_spec_api");
        const struct ggml_pxqn_spec_api * api = get ? get(GGML_PXQN_SPEC_VERSION) : nullptr;
        if (api && api->version == GGML_PXQN_SPEC_VERSION && api->size == sizeof(struct ggml_pxqn_spec_api) &&
            api->create && api->destroy && api->reset && api->choose && api->observe && api->describe) {
            t = api;
        }
    });
    return t;
}
#endif

bool ggml_pxqn_spec_available(void) {
    return ggml_pxqn_spec_table() != nullptr;
}
void * ggml_pxqn_spec_create(const struct ggml_pxqn_spec_cfg * cfg) {
    const auto * t = ggml_pxqn_spec_table();
    return t ? t->create(cfg) : nullptr;
}
void ggml_pxqn_spec_destroy(void * h) {
    const auto * t = ggml_pxqn_spec_table();
    if (t && h) t->destroy(h);
}
void ggml_pxqn_spec_reset(void * h, int32_t seq_id) {
    const auto * t = ggml_pxqn_spec_table();
    if (t && h) t->reset(h, seq_id);
}
void ggml_pxqn_spec_choose(void * h, const struct ggml_pxqn_spec_offer * offer, struct ggml_pxqn_spec_pick * pick) {
    const auto * t = ggml_pxqn_spec_table();
    if (t && h) { t->choose(h, offer, pick); return; }
    pick->kind = GGML_PXQN_SPEC_STATIC; pick->width = 0; pick->probe = 0;
}
void ggml_pxqn_spec_observe(void * h, int32_t seq_id, int32_t n_verified, int32_t n_accepted) {
    const auto * t = ggml_pxqn_spec_table();
    if (t && h) t->observe(h, seq_id, n_verified, n_accepted);
}
int ggml_pxqn_spec_describe(void * h, char * buf, size_t n) {
    const auto * t = ggml_pxqn_spec_table();
    if (t && h) return t->describe(h, buf, n);
    if (buf && n) buf[0] = 0;
    return 0;
}

// ---- the per-model / per-card speculation defaults (ggml-pxqn-spec.h): the library's optional third table ----
#if defined(_WIN32)
const struct ggml_pxqn_specdef_api * ggml_pxqn_specdef_table(void) { return nullptr; }
#else
const struct ggml_pxqn_specdef_api * ggml_pxqn_specdef_table(void) {
    static std::once_flag once;
    static const struct ggml_pxqn_specdef_api * t = nullptr;
    std::call_once(once, [] {
        // PXA_SPEC_DEFAULTS: the per-model / per-card table is armed by default (main 2026-10-04, from an earlier
        // one measurement, ledger mtp-all-spec-defaults: Qwen3.8-27B one V100 40.9/41.2 -> 56.7/71.5 t/s with code shas
        // equal to plain, Gemma 4 with no flags +13%, the Flash-Next 32 GB auto row no longer loses to plain).
        // PXA_SPEC_DEFAULTS=0 never arms it; PXA_REFERENCE=1 is the audit baseline.
        static constexpr bool PXA_SPEC_DEFAULTS_DEFAULT = true;
        const char * en = getenv("PXA_SPEC_DEFAULTS");
        const char * ref = getenv("PXA_REFERENCE");
        if (ref && *ref && atoi(ref) != 0) return;
        if (en && *en ? atoi(en) == 0 : !PXA_SPEC_DEFAULTS_DEFAULT) return;
        if (!ggml_pxqn_lib() || !g_pxqn_handle) return;
        auto get = (ggml_pxqn_get_specdef_api_fn) dlsym(g_pxqn_handle, "ggml_pxqn_get_specdef_api");
        const struct ggml_pxqn_specdef_api * api = get ? get(GGML_PXQN_SPECDEF_VERSION) : nullptr;
        if (api && api->version == GGML_PXQN_SPECDEF_VERSION && api->size == sizeof(struct ggml_pxqn_specdef_api) &&
            api->choose && api->select && api->shortlist) {
            t = api;
        }
    });
    return t;
}
#endif

bool ggml_pxqn_specdef_available(void) {
    return ggml_pxqn_specdef_table() != nullptr;
}
bool ggml_pxqn_specdef_choose(const struct ggml_pxqn_specdef_in * in, struct ggml_pxqn_specdef_out * out) {
    const auto * t = ggml_pxqn_specdef_table();
    if (t && in && out) return t->choose(in, out);
    if (out) {
        out->chain = GGML_PXQN_SPECDEF_STATIC; out->mtp_n_max = 0; out->mtp_p_min = -1.0f; out->select = -1; out->n_ubatch = 0; out->depth_ramp_off = 0; out->why[0] = 0;
    }
    return false;
}
int32_t ggml_pxqn_specdef_select(const char * arch, int32_t has_ngram, int32_t has_mtp, int32_t has_assistant, int32_t n_dev, int32_t cc_min) {
    const auto * t = ggml_pxqn_specdef_table();
    return t ? t->select(arch, has_ngram, has_mtp, has_assistant, n_dev, cc_min) : -1;
}
int32_t ggml_pxqn_specdef_shortlist(const char * arch, int32_t n_vocab, int32_t head_type, int32_t n_dev, int32_t cc_min) {
    const auto * t = ggml_pxqn_specdef_table();
    return t ? t->shortlist(arch, n_vocab, head_type, n_dev, cc_min) : 0;
}

// ---- optional tuning switches (ggml-pxqn-tune.h): one more optional table of the library ----
#if defined(_WIN32)
uint32_t ggml_pxqn_tune_flags(void) { return 0; }
#else
uint32_t ggml_pxqn_tune_flags(void) {
    static std::once_flag once;
    static uint32_t f = 0;
    std::call_once(once, [] {
        if (!ggml_pxqn_lib() || !g_pxqn_handle) return;
        auto get = (ggml_pxqn_get_tune_api_fn) dlsym(g_pxqn_handle, GGML_PXQN_TUNE_SYM);
        const struct ggml_pxqn_tune_api * api = get ? get(GGML_PXQN_TUNE_VERSION) : nullptr;
        if (api && api->version == GGML_PXQN_TUNE_VERSION && api->size == sizeof(struct ggml_pxqn_tune_api) && api->flags) {
            f = api->flags();
        }
    });
    return f;
}
#endif

// ---- the levers the library reads (ggml-pxqn-levers.h): optional symbol of the library ----
#if defined(_WIN32)
int ggml_pxqn_lever_builtin(const char * name) { (void) name; return 0; }
#else
int ggml_pxqn_lever_builtin(const char * name) {
    static std::once_flag once;
    static ggml_pxqn_lever_known_fn fn = nullptr;
    std::call_once(once, [] {
        if (!ggml_pxqn_lib() || !g_pxqn_handle) return;
        fn = (ggml_pxqn_lever_known_fn) dlsym(g_pxqn_handle, GGML_PXQN_LEVER_SYM);
    });
    return name && fn ? fn(name) : 0;
}
#endif
