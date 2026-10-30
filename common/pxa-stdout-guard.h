// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// pxa-stdout-guard.h - keep a tool's machine-readable stdout clean (beta 2 report, 2026-10-19:
// `llama-bench -o json` had status lines mixed into the JSON on stdout).
//
// A tool whose stdout is data (llama-bench -o json|csv|md|sql) calls pxa_stdout_guard_take() once,
// before the backends initialise. It returns a private FILE * on the ORIGINAL stdout for the data,
// and points file descriptor 1 at stderr, so every other write to stdout from then on - a printf
// in any library, a closed backend, a CUDA device printf, a status line - lands on stderr instead
// of inside the data. Nothing is dropped: the lines still show, only on the other stream.
//
// Off switch: PXA_BENCH_SHARED_STDOUT=1 keeps the old behaviour (one shared stdout).
// Header-only so a unit test (tests/test-pxa-stdout-guard.cpp) covers the exact code the tool runs.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#  include <io.h>
#  define PXA_SG_DUP    _dup
#  define PXA_SG_DUP2   _dup2
#  define PXA_SG_FILENO _fileno
#  define PXA_SG_FDOPEN _fdopen
#  define PXA_SG_CLOSE  _close
#else
#  include <unistd.h>
#  define PXA_SG_DUP    dup
#  define PXA_SG_DUP2   dup2
#  define PXA_SG_FILENO fileno
#  define PXA_SG_FDOPEN fdopen
#  define PXA_SG_CLOSE  close
#endif

// true when the guard is switched off by the environment
inline bool pxa_stdout_guard_disabled() {
    const char * e = std::getenv("PXA_BENCH_SHARED_STDOUT");
    return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
}

// Returns the FILE * the data goes to. On any failure (or when switched off) it returns stdout
// unchanged, so the caller never loses its output.
inline FILE * pxa_stdout_guard_take() {
    if (pxa_stdout_guard_disabled()) {
        return stdout;
    }
    std::fflush(stdout);
    std::fflush(stderr);
    const int data_fd = PXA_SG_DUP(PXA_SG_FILENO(stdout));
    if (data_fd < 0) {
        return stdout;
    }
    FILE * data = PXA_SG_FDOPEN(data_fd, "w");
    if (data == nullptr) {
        PXA_SG_CLOSE(data_fd);
        return stdout;
    }
    if (PXA_SG_DUP2(PXA_SG_FILENO(stderr), PXA_SG_FILENO(stdout)) < 0) {
        std::fclose(data);             // closes data_fd; stdout is untouched
        return stdout;
    }
    return data;
}

// Flush (and close, when it is the guard's own stream) at the end.
inline void pxa_stdout_guard_release(FILE * data) {
    if (data == nullptr) {
        return;
    }
    std::fflush(data);
    if (data != stdout) {
        std::fclose(data);
    }
}
