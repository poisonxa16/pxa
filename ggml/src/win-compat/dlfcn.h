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
#include <string>

#ifndef LOAD_LIBRARY_SEARCH_APPLICATION_DIR   // Windows 8+ (and 7 with KB2533623); older SDK headers may lack the names
#define LOAD_LIBRARY_SEARCH_APPLICATION_DIR 0x00000200
#endif
#ifndef LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
#define LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR    0x00000100
#endif
#ifndef LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
#define LOAD_LIBRARY_SEARCH_DEFAULT_DIRS    0x00001000
#endif
#ifndef LOAD_LIBRARY_SEARCH_SYSTEM32
#define LOAD_LIBRARY_SEARCH_SYSTEM32        0x00000800
#endif

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
    HMODULE h = NULL;
    if (!strchr(p, '\\') && !strchr(p, ':')) {
        // a bare name: the executable's folder and System32 only, never the current folder or %PATH% (DLL planting)
        h = LoadLibraryExA(p, NULL, LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    } else {
        // a path: made absolute (a relative one is the caller's explicit choice, e.g. PXA_PXQN_LIB; the flags below
        // need an absolute path: fail closed when it cannot be made one). Its dependencies come from its own folder,
        // the executable's folder and System32 only, never the current folder or %PATH% (the shipped
        // ggml-pxqn.dll imports ggml, cudart, cublas and the MSVC runtime: all ship in bin\ beside the exe)
        char full[2048];
        const DWORD nf = GetFullPathNameA(p, (DWORD) sizeof(full), full, NULL);
        if (nf > 0 && nf < sizeof(full)) {
            snprintf(p, sizeof(p), "%s", full);
            h = LoadLibraryExA(p, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        } else {
            SetLastError(ERROR_BAD_PATHNAME);
        }
    }
    if (!h) snprintf(pxa_dl_err, sizeof(pxa_dl_err), "LoadLibrary(%s) failed, error %lu", p, (unsigned long) GetLastError());
    return (void *) h;
}

// the running executable's folder with '/' separators (the loader's fallback location on Windows); 0 on failure
static inline int pxa_dl_exe_dir(std::string & out) {
    char fname[2048];
    const DWORD n = GetModuleFileNameA(NULL, fname, (DWORD) sizeof(fname));
    if (n == 0 || n >= sizeof(fname)) return 0;
    for (char * c = fname; *c; ++c) if (*c == '\\') *c = '/';
    char * sl = strrchr(fname, '/');
    if (!sl) return 0;
    *sl = 0;
    out = fname;
    return 1;
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
