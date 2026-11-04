// pxa-test-portable.h: the few POSIX calls the PXA tests use, made to build and run on native Windows (MSVC / nvcc
// with the MSVC host compiler) too, so the Windows ctest build compiles every registered test (todo
// pxqn-backend-test-unistd). On Linux everything here is the plain POSIX call.
//   setenv / unsetenv        -> _putenv_s (Windows: an empty value removes the variable)
//   pxa_getpid()             -> getpid / _getpid
//   pxa_popcount / ...ll     -> __builtin_popcount(ll) / __popcnt(64)
//   pxa_self_exe()           -> readlink(/proc/self/exe) / GetModuleFileNameA (resolved here: inside a child shell
//                               /proc/self/exe would name the shell)
//   pxa_tmp_dir()            -> TMPDIR or /tmp / TEMP or TMP
//   pxa_fseeko(f, off, w)    -> fseeko (64-bit off_t) / _fseeki64 (MSVC off_t is 32-bit long)
//   pxa_getloadavg(la, n)    -> getloadavg / -1 on Windows (no load average there; callers treat < 0 as unknown)
//   pxa_tmp_path(name)       -> <tmp dir>/<name>
//   pxa_run_self(args, env)  -> runs this binary again through system() with "args" (each path quoted by the caller
//                               with pxa_q) after setting each "NAME=value" in env in this process (inherited by the
//                               child, no shell VAR=val prefix) and restoring them after. Returns system()'s value.
//   pxa_mkdtemp(stem)        -> a new directory <tmp dir>/<stem>-XXXXXX: mkdtemp / _mktemp_s + _mkdir ("" on failure)
//   pxa_rmdir(path)          -> rmdir / _rmdir
//   pxa_lib_open(path, err)  -> dlopen(RTLD_NOW | RTLD_LOCAL) / LoadLibraryExA(LOAD_WITH_ALTERED_SEARCH_PATH): a DLL's own
//                               imports come from its folder first (ggml.dll beside ggml-pxqn.dll), then the usual places,
//                               PATH included (the CUDA runtime of a CUDA build). err = dlerror() / the Windows error code
//   pxa_lib_sym(h, name)     -> dlsym / GetProcAddress
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#  include <direct.h>
#  include <io.h>
#  include <process.h>
#  include <intrin.h>
// declared by hand instead of <windows.h>, whose macros (ERROR, IN, OUT, near, ...) collide with test code. The types are
// the SDK's own (HMODULE = struct HINSTANCE__ *, FARPROC = INT_PTR (WINAPI *)()), so a later <windows.h> redeclares them
// compatibly.
#  ifndef _WINDOWS_
extern "C" __declspec(dllimport) unsigned long __stdcall GetModuleFileNameA(struct HINSTANCE__ * module, char * name, unsigned long size);
typedef long long (__stdcall * pxa_farproc)();
extern "C" __declspec(dllimport) struct HINSTANCE__ * __stdcall LoadLibraryExA(const char * name, void * file, unsigned long flags);
extern "C" __declspec(dllimport) pxa_farproc __stdcall GetProcAddress(struct HINSTANCE__ * module, const char * name);
extern "C" __declspec(dllimport) unsigned long __stdcall GetLastError(void);
#  endif

#  ifndef setenv   // src/pxa-win-compat.h (included by some tests) already maps setenv/unsetenv
static inline int setenv(const char * name, const char * value, int overwrite) {
    if (!overwrite && getenv(name)) return 0;
    return _putenv_s(name, value ? value : "") == 0 ? 0 : -1;
}
static inline int unsetenv(const char * name) { return _putenv_s(name, "") == 0 ? 0 : -1; }
#  endif
static inline int pxa_getpid(void) { return _getpid(); }
static inline int pxa_popcount(unsigned v) { return (int) __popcnt(v); }
static inline int pxa_popcountll(unsigned long long v) { return (int) __popcnt64(v); }
#else
#  include <dlfcn.h>
#  include <unistd.h>
static inline int pxa_getpid(void) { return (int) getpid(); }
static inline int pxa_popcount(unsigned v) { return __builtin_popcount(v); }
static inline int pxa_popcountll(unsigned long long v) { return __builtin_popcountll(v); }
#endif

static inline int pxa_fseeko(FILE * f, long long off, int whence) {
#ifdef _WIN32
    return _fseeki64(f, (__int64) off, whence);
#else
    return fseeko(f, (off_t) off, whence);
#endif
}

static inline int pxa_getloadavg(double * la, int n) {
#ifdef _WIN32
    (void) la; (void) n;
    return -1;
#else
    return getloadavg(la, n);
#endif
}

static inline std::string pxa_self_exe() {
#ifdef _WIN32
    char buf[4096];
    const unsigned long n = GetModuleFileNameA(nullptr, buf, (unsigned long) sizeof buf);
    return n > 0 && n < sizeof buf ? std::string(buf, n) : std::string();
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
    return n > 0 ? std::string(buf, (size_t) n) : std::string();
#endif
}

static inline std::string pxa_tmp_dir() {
#ifdef _WIN32
    const char * t = getenv("TEMP");
    if (!t || !*t) t = getenv("TMP");
    return t && *t ? std::string(t) : std::string(".");
#else
    const char * t = getenv("TMPDIR");
    return t && *t ? std::string(t) : std::string("/tmp");
#endif
}

static inline std::string pxa_tmp_path(const std::string & name) {
#ifdef _WIN32
    return pxa_tmp_dir() + "\\" + name;
#else
    return pxa_tmp_dir() + "/" + name;
#endif
}

static inline std::string pxa_mkdtemp(const std::string & stem) {
    const std::string t = pxa_tmp_path(stem + "-XXXXXX");
    std::vector<char> b(t.begin(), t.end());
    b.push_back('\0');
#ifdef _WIN32
    if (_mktemp_s(b.data(), b.size()) != 0 || _mkdir(b.data()) != 0) return std::string();
#else
    if (!mkdtemp(b.data())) return std::string();
#endif
    return std::string(b.data());
}

static inline int pxa_rmdir(const std::string & path) {
#ifdef _WIN32
    return _rmdir(path.c_str());
#else
    return rmdir(path.c_str());
#endif
}

static inline void * pxa_lib_open(const std::string & path, std::string & err) {
#ifdef _WIN32
    std::string p = path;
    for (char & c : p) if (c == '/') c = '\\';   // the altered search order wants an absolute path written with backslashes
    void * h = LoadLibraryExA(p.c_str(), nullptr, 0x00000008 /* LOAD_WITH_ALTERED_SEARCH_PATH */);
    if (!h) err = "LoadLibraryExA(" + p + ") failed, error " + std::to_string(GetLastError());
#else
    void * h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { const char * e = dlerror(); err = e ? e : "dlopen failed"; }
#endif
    return h;
}

static inline void * pxa_lib_sym(void * h, const char * name) {
#ifdef _WIN32
    return (void *) GetProcAddress((struct HINSTANCE__ *) h, name);
#else
    return dlsym(h, name);
#endif
}

// a double-quoted argument for system() (paths only: no embedded quotes)
static inline std::string pxa_q(const std::string & s) { return "\"" + s + "\""; }

static inline int pxa_run_self(const std::string & args, const std::vector<std::pair<std::string, std::string>> & env) {
    const std::string exe = pxa_self_exe();
    if (exe.empty()) return -1;
    std::vector<std::pair<bool, std::string>> saved;
    for (const auto & kv : env) {
        const char * v = getenv(kv.first.c_str());
        saved.emplace_back(v != nullptr, v ? std::string(v) : std::string());
        setenv(kv.first.c_str(), kv.second.c_str(), 1);
    }
    std::string cmd = pxa_q(exe) + " " + args;
#ifdef _WIN32
    cmd = "\"" + cmd + "\"";    // cmd /c strips one outer pair of quotes
#endif
    const int rc = system(cmd.c_str());
    for (size_t i = 0; i < env.size(); ++i) {
        if (saved[i].first) setenv(env[i].first.c_str(), saved[i].second.c_str(), 1);
        else                unsetenv(env[i].first.c_str());
    }
    return rc;
}
