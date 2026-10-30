// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// PXA_XCACHE routing counts: where they are looked for. Header-only so tests/test-pxa-xcache-counts.cpp checks the exact
// code src/llama.cpp runs. An explicit PXA_XCACHE_COUNTS names the one file (read when it exists, written when it does not)
// - unless it can be neither, then it is ignored as if unset. Otherwise <model>.expert-counts.csv next to the GGUF under
// every name the file answers to, then the per-user cache (PXA_CACHE_DIR, else $XDG_CACHE_HOME/pxa, else $HOME/.cache/pxa).
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#ifndef _WIN32
#include <dirent.h>
#endif
#include <string>
#include <sys/stat.h>
#ifdef _WIN32
#include "pxa-win-compat.h"
#else
#include <unistd.h>
#endif
#include <vector>

// Every name the model file answers to: the path it was opened by, the file that path resolves to, and any sibling symlink
// (in either directory) resolving to the same file. The shipped release dir names the map after the SYMLINK
// (<link>.gguf.expert-counts.csv), so a file opened by its real name - or the other way round - missed a map that was sitting
// right next to it, and the cache fell back to whole-layer streaming (2026-10-12, Control repro).
inline void pxa_push_unique(std::vector<std::string> & v, const std::string & s) {
    if (std::find(v.begin(), v.end(), s) == v.end()) v.push_back(s);
}

inline std::string pxa_dir_of(const std::string & p) {
    const size_t sl = p.find_last_of('/');
    return sl == std::string::npos ? std::string(".") : (sl == 0 ? std::string("/") : p.substr(0, sl));
}

inline std::string pxa_base_of(const std::string & p) {
    const size_t sl = p.find_last_of('/');
    return sl == std::string::npos ? p : p.substr(sl + 1);
}

inline void pxa_model_names(const std::string & model_path, std::vector<std::string> & dirs, std::vector<std::string> & names) {
    pxa_push_unique(dirs, pxa_dir_of(model_path));
    pxa_push_unique(names, pxa_base_of(model_path));
    std::string real;
    if (char * r = realpath(model_path.c_str(), nullptr)) { real = r; free(r); }
    if (real.empty()) return;
    pxa_push_unique(dirs, pxa_dir_of(real));
    pxa_push_unique(names, pxa_base_of(real));
#ifndef _WIN32   // sibling symlinks: Windows model folders have none worth a directory scan
    for (const std::string d : std::vector<std::string>(dirs)) {
        DIR * dh = opendir(d.c_str());
        if (!dh) continue;
        while (const dirent * de = readdir(dh)) {
            const std::string n = de->d_name;
            if (n.size() < 6 || n.compare(n.size() - 5, 5, ".gguf") != 0) continue;
            const std::string fp = d + "/" + n;
            struct stat ls;
            if (lstat(fp.c_str(), &ls) != 0 || !S_ISLNK(ls.st_mode)) continue;
            if (char * r = realpath(fp.c_str(), nullptr)) {
                if (real == r) pxa_push_unique(names, n);
                free(r);
            }
        }
        closedir(dh);
    }
#endif
}

// An explicit PXA_XCACHE_COUNTS the engine can neither read nor write (empty, a dangling symlink, a file it may not read, a
// missing file in a directory it may not write) was WORSE than an unset one: unset makes the server profile the model, a dead
// path meant whole-layer expert streaming behind one WARN line (2026-10-12, arm C: a symlink into a read-only bind mount).
// It now says so as an ERROR and is treated as unset.
inline bool pxa_counts_env_usable(const std::string & e, std::string & why) {
    if (e.empty()) { why = "it is empty"; return false; }
    if (access(e.c_str(), R_OK) == 0) return true;
    if (access(e.c_str(), F_OK) == 0) { why = std::string("it exists but cannot be read (") + strerror(errno) + ")"; return false; }
    struct stat ls;
    if (lstat(e.c_str(), &ls) == 0 && S_ISLNK(ls.st_mode)) { why = "it is a symlink that does not resolve (dangling)"; return false; }
    if (access(pxa_dir_of(e).c_str(), W_OK) == 0) return true;   // missing but writable: the profile pass writes it there
    why = "it does not exist and its directory is not writable";
    return false;
}

inline bool pxa_counts_env_given() {   // set AND usable: an unusable value acts like an unset one
    std::string why;
    const char * p = getenv("PXA_XCACHE_COUNTS");
    return p != nullptr && pxa_counts_env_usable(p, why);
}

inline std::vector<std::string> pxa_xcache_counts_candidates(const std::string & model_path, std::string * ignored = nullptr) {
    std::vector<std::string> v;
    if (const char * p = getenv("PXA_XCACHE_COUNTS")) {
        std::string why;
        if (pxa_counts_env_usable(p, why)) { v.push_back(p); return v; }
        if (ignored) *ignored = why;
    }
    if (model_path.empty()) return v;
    std::vector<std::string> dirs, names;
    pxa_model_names(model_path, dirs, names);
    pxa_push_unique(v, model_path + ".expert-counts.csv");     // first, as before: the default write target
    for (const auto & d : dirs) for (const auto & n : names) pxa_push_unique(v, d + "/" + n + ".expert-counts.csv");
    std::string dir;
    if (const char * c = getenv("PXA_CACHE_DIR"); c && *c) dir = c;
    else if (const char * x = getenv("XDG_CACHE_HOME"); x && *x) dir = std::string(x) + "/pxa";
    else if (const char * h = getenv("HOME"); h && *h) dir = std::string(h) + "/.cache/pxa";
    if (!dir.empty()) for (const auto & n : names) pxa_push_unique(v, dir + "/" + n + ".expert-counts.csv");
    return v;
}
