// ggml-pxqn-loader.cpp -- finds and checks the closed PXQN library (see ggml-pxqn-api.h). Open.
#include "ggml-pxqn-api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#if defined(_WIN32)
const struct ggml_pxqn_lib_api * ggml_pxqn_lib(void) { return nullptr; }
#else
#include <dlfcn.h>

static const struct ggml_pxqn_lib_api * g_pxqn_lib = nullptr;

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
    return api;   // never unloaded: its kernels and per-device state live for the process
}

const struct ggml_pxqn_lib_api * ggml_pxqn_lib(void) {
    static std::once_flag once;
    std::call_once(once, [] { g_pxqn_lib = pxqn_load(); });
    return g_pxqn_lib;
}
#endif

bool ggml_pxqn_available(void) {
    return ggml_pxqn_lib() != nullptr;
}
