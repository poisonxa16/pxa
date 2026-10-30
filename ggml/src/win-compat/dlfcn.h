// win-compat/dlfcn.h -- Windows build only (on the include path only when WIN32): the few dlfcn calls the PXQN
// loader (ggml-pxqn-loader.cpp) makes, over LoadLibrary / GetProcAddress. Paths come back with '/' separators so the
// loader's directory logic is the same text on both systems; the library file name maps libggml-pxqn.so -> ggml-pxqn.dll.
#pragma once
#ifndef _WIN32
#error "win-compat/dlfcn.h is for the Windows build only"
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <stdio.h>
#include <string.h>

#define RTLD_NOW   2
#define RTLD_LAZY  1
#define RTLD_LOCAL 0

typedef struct { const char * dli_fname; void * dli_fbase; const char * dli_sname; void * dli_saddr; } Dl_info;

static char pxa_dl_err[512];

static inline void * dlopen(const char * path, int flags) {
    (void) flags;
    char p[2048];
    snprintf(p, sizeof(p), "%s", path ? path : "");
    const size_t n = strlen(p), m = strlen("libggml-pxqn.so");
    if (n >= m && strcmp(p + n - m, "libggml-pxqn.so") == 0 && (n == m || p[n - m - 1] == '/' || p[n - m - 1] == '\\')) {
        snprintf(p + n - m, sizeof(p) - (n - m), "%s", "ggml-pxqn.dll");
    }
    for (char * c = p; *c; ++c) if (*c == '/') *c = '\\';
    HMODULE h = LoadLibraryExA(p, NULL, strchr(p, '\\') ? LOAD_WITH_ALTERED_SEARCH_PATH : 0);
    if (!h) snprintf(pxa_dl_err, sizeof(pxa_dl_err), "LoadLibrary(%s) failed, error %lu", p, (unsigned long) GetLastError());
    return (void *) h;
}

static inline void * dlsym(void * h, const char * name) {
    return (void *) GetProcAddress((HMODULE) h, name);
}

static inline int dlclose(void * h) {
    return FreeLibrary((HMODULE) h) ? 0 : -1;
}

static inline const char * dlerror(void) {
    return pxa_dl_err[0] ? pxa_dl_err : NULL;
}

static inline int dladdr(const void * addr, Dl_info * info) {
    static char fname[2048];
    HMODULE m = NULL;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCSTR) addr, &m) || !m) return 0;
    const DWORD n = GetModuleFileNameA(m, fname, (DWORD) sizeof(fname));
    if (n == 0 || n >= sizeof(fname)) return 0;
    for (char * c = fname; *c; ++c) if (*c == '\\') *c = '/';
    info->dli_fname = fname; info->dli_fbase = (void *) m; info->dli_sname = NULL; info->dli_saddr = NULL;
    return 1;
}
