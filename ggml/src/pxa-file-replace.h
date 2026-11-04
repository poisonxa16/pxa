// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// pxa-file-replace.h -- replace a small file all or nothing. The one replace behind every file a session or a measurement leaves
// for the next boot to read: the engine's counts csv and learned .state (src/llama-pxa-counts.h: pxa_counts_csv_write,
// pxa_learned_state_write) and the closed PXQN library's .speed measurement (ggml/src/pxqn-xcache-learn.cpp,
// ggml_pxqn_xcache_speed_write). Header-only and open. It sits in ggml/src, below both: the library may not include src/, and
// src/ reaches it by a relative path.
//
// pxa_file_replace(path, text): the text goes to <path>.tmp in the same directory, reaches the disk, and only then moves over
// <path> in one step.
//   POSIX:   fwrite, fflush, fsync before fclose (EINVAL = a file system with nothing to sync, accepted), rename, then a
//            best-effort fsync of the directory, so the rename itself survives a crash.
//   Windows: fwrite, fflush, _commit (FlushFileBuffers) before fclose, then MoveFileExA(MOVEFILE_REPLACE_EXISTING |
//            MOVEFILE_WRITE_THROUGH): rename() refuses an existing target there.
// Binary mode: the bytes are the same on every OS (LF, never CRLF). The target is never opened for writing (no truncate-then-
// write): a crash, a full disk or a failed write leaves the old file as it was, and the temp file is removed on every failure.
// A reader at the same moment sees the old file or the new one. On Windows a reader that holds the target open (fopen does not
// share delete access) makes the move fail: the save reports false and the old file stays; the caller's next save retries.
// The temp name is fixed, so two processes saving the same file at the same instant share it (a per-process name would leave
// stale temps behind after a crash).
#pragma once

#include <cerrno>
#include <cstdio>
#include <string>
#ifdef _WIN32
#include <io.h>      // _commit
#else
#include <fcntl.h>   // open, O_DIRECTORY
#include <unistd.h>  // fsync, close
#endif

#ifdef _WIN32
// MoveFileExA (kernel32), declared here so this header does not pull in windows.h (as src/pxa-win-compat.h does for
// SetEnvironmentVariableA); skipped when a file already included it. The A form reads the path in the code page fopen reads it
// in, so it moves the very file fopen wrote.
#if !defined(_WINDOWS_)
extern "C" __declspec(dllimport) int __stdcall MoveFileExA(const char * lpExistingFileName, const char * lpNewFileName,
                                                           unsigned long dwFlags);
#endif
#endif

inline bool pxa_file_replace(const std::string & path, const std::string & text) {
    if (path.empty()) return false;
    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(text.data(), 1, text.size(), f) == text.size() && fflush(f) == 0;
#ifdef _WIN32
    ok = ok && _commit(_fileno(f)) == 0;
#else
    ok = ok && (fsync(fileno(f)) == 0 || errno == EINVAL);
#endif
    ok = fclose(f) == 0 && ok;
#ifdef _WIN32
    const unsigned long replace_existing = 0x1, write_through = 0x8;   // MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH
    ok = ok && MoveFileExA(tmp.c_str(), path.c_str(), replace_existing | write_through) != 0;
#else
    ok = ok && rename(tmp.c_str(), path.c_str()) == 0;
#endif
    if (!ok) {
        std::remove(tmp.c_str());
        return false;
    }
#ifndef _WIN32
    const size_t sl = path.find_last_of('/');
    const std::string dir = sl == std::string::npos ? std::string(".") : (sl == 0 ? std::string("/") : path.substr(0, sl));
    const int dfd = open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dfd >= 0) {
        (void) fsync(dfd);
        close(dfd);
    }
#endif
    return true;
}
