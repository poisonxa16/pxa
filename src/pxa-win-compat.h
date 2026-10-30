// pxa-win-compat.h -- the few POSIX names the PXA engine files use, for an MSVC (Windows x64) build. Included INSTEAD of
// <unistd.h> / <dirent.h> under _WIN32 only, so a Linux build's preprocessed text is unchanged. Not a POSIX layer: just
// enough for access(), realpath(), lstat(), mkdir(path, mode), setenv(), unsetenv(), gmtime_r() and ssize_t to mean what
// the callers expect.
#pragma once
#ifdef _WIN32

#ifndef NOMINMAX
#  define NOMINMAX
#endif
#include <direct.h>
#include <io.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <errno.h>
#include <BaseTsd.h>

// Declared here so this header does not pull in windows.h. Skipped when a file already included it.
#if !defined(_WINDOWS_)
#  ifdef __cplusplus
extern "C" {
#  endif
__declspec(dllimport) int __stdcall SetEnvironmentVariableA(const char * lpName, const char * lpValue);
#  ifdef __cplusplus
}
#  endif
#endif

typedef SSIZE_T ssize_t;

#ifndef F_OK
#  define F_OK 0
#endif
#ifndef W_OK
#  define W_OK 2
#endif
#ifndef R_OK
#  define R_OK 4
#endif
#ifndef X_OK
#  define X_OK 0    // Windows has no execute bit: existence
#endif
#ifndef S_ISLNK
#  define S_ISLNK(m) 0   // _stat follows links; a Windows path is never reported as a symlink here
#endif
#ifndef S_ISDIR
#  define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif
#ifndef S_ISREG
#  define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#endif

static inline int pxa_win_access(const char * p, int mode) { return _access(p, mode & 6); }
static inline char * pxa_win_realpath(const char * p, char * out) { return _fullpath(out, p, out ? 260 : 0); }
static inline int pxa_win_mkdir(const char * p, int /*mode*/ = 0) { return _mkdir(p); }

// setenv(name, value, overwrite): overwrite = 0 never replaces a value already in the environment, an empty one
// included (getenv returns ""). _putenv_s deletes the name when the value is empty, so an empty value is stored by
// writing a one-byte placeholder and terminating it in the CRT block. The process block is set empty too, so a
// child inherits the empty value (the CRT copies that block at startup).
static inline int pxa_win_setenv(const char * n, const char * v, int overwrite) {
    if (!n || !*n) return -1;
    if (!v) v = "";
    if (!overwrite && getenv(n) != nullptr) return 0;
    if (v[0] != '\0') return _putenv_s(n, v);
    if (_putenv_s(n, "x") != 0) return -1;
    const size_t nl = strlen(n);
    for (char ** e = _environ; e && *e; ++e) {
        if (strncmp(*e, n, nl) == 0 && (*e)[nl] == '=') {
            (*e)[nl + 1] = '\0';
            SetEnvironmentVariableA(n, "");
            return 0;
        }
    }
    return -1;
}
// unsetenv: _putenv_s with an empty value removes the name from the CRT and the process block.
static inline int pxa_win_unsetenv(const char * n) {
    if (!n || !*n) return -1;
    return _putenv_s(n, "");
}
static inline struct tm * pxa_win_gmtime_r(const time_t * t, struct tm * out) { return gmtime_s(out, t) == 0 ? out : nullptr; }
// the O_DIRECT row reader (llama-pxa-ple-direct.h) is Linux-only: its aligned buffers are never allocated here
static inline int pxa_win_posix_memalign(void ** p, size_t, size_t) { *p = nullptr; return ENOMEM; }

#define setenv          pxa_win_setenv
#define unsetenv        pxa_win_unsetenv
#define gmtime_r        pxa_win_gmtime_r
#define posix_memalign  pxa_win_posix_memalign
#define access(p, m)    pxa_win_access((p), (m))
#define realpath(p, o)  pxa_win_realpath((p), (o))
#define lstat           stat
#define mkdir           pxa_win_mkdir
#define fsync           _commit
#ifndef O_DIRECTORY
#  define O_DIRECTORY   0x200000   // no such flag: opening a directory fails, the caller skips the directory fsync
#endif
#define PXA_PATH_CSTR(p) (p).string().c_str()   // path::c_str() is wchar_t on Windows

#endif // _WIN32
