// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// PXA_XCACHE routing counts: where they are looked for. Header-only so tests/test-pxa-xcache-counts.cpp checks the exact
// code src/llama.cpp runs. An explicit PXA_XCACHE_COUNTS names the one file (read when it exists, written when it does not)
// - unless it can be neither, then it is ignored as if unset. Otherwise <model>.expert-counts.csv next to the GGUF under
// every name the file answers to, then the per-user cache (PXA_CACHE_DIR, else $XDG_CACHE_HOME/pxa, else $HOME/.cache/pxa).
// Also the files the counts are written to (at the end): the csv writer (pxa_counts_csv_write, through pxa_file_replace) and
// the per-layer prior rows the online adaptation starts from (pxa_xc_adapt_layers, pxa_xc_prior_flat).
#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef _WIN32
#include <dirent.h>
#endif
#include <functional>
#include <string>
#include <sys/stat.h>
#ifdef _WIN32
#include "pxa-win-compat.h"
#else
#include <unistd.h>
#endif
#include <vector>

#include "../ggml/src/pxa-file-replace.h"   // pxa_file_replace

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

// pxa_file_replace(path, text): the one replace every counts csv write goes through, all or nothing (ggml/src/pxa-file-replace.h).

// The routing counts as the csv the expert cache reads (counts[il * n_expert + e]): "tensor,expert,count", then, for every
// layer with any count, one row per expert of its down-expert tensor (the planner reads the down rows first; the hot/cold
// split replaces that tensor, the loader still reads its name). widths[il] (optional) is a layer's router width: a layer with
// fewer experts than expert_count (Nemotron 3.5 Bolt) gets rows for its own experts only. Nothing is written without a row.
inline bool pxa_counts_csv_write(const std::string & path, const uint64_t * counts, int n_layers, int n_expert, const int * widths) {
    if (path.empty() || !counts || n_expert <= 0) return false;
    std::string text = "tensor,expert,count\n";
    long rows = 0;
    char line[160];
    for (int il = 0; il < n_layers; ++il) {
        uint64_t tot = 0;
        for (int e = 0; e < n_expert; ++e) tot += counts[(size_t) il*n_expert + e];
        if (tot == 0) continue;
        int n_l = n_expert;
        if (widths && widths[il] > 1 && widths[il] < n_expert) n_l = widths[il];
        for (int e = 0; e < n_l; ++e) {
            snprintf(line, sizeof(line), "blk.%d.ffn_down_exps.weight,%d,%llu\n", il, e, (unsigned long long) counts[(size_t) il*n_expert + e]);
            text += line;
            ++rows;
        }
    }
    return rows > 0 && pxa_file_replace(path, text);
}

// The online adaptation's per-layer setup after the plan (src/llama.cpp). prior[il] = the counts row the plan was made from, for
// every layer that has one n_expert wide, split or kept whole on the card (a split layer's row is the adaptor's prior; a layer
// the adaptor does not observe is carried into a learned save from its row). Every split layer (a hot and a cold part) also gets
// n_spare free cold slots (-1) at the end of its cold stack. XL is llama_model::pxa_xc_layer (hot, cold); counts are the plan's
// per-layer rows. Returns the split layers.
template <typename XL>
inline size_t pxa_xc_adapt_layers(std::vector<XL> & xc, std::vector<std::vector<double>> & prior,
                                  const std::vector<std::vector<double>> & counts, int n_layer, int n_expert, int n_spare) {
    prior.assign((size_t) std::max(n_layer, 0), {});
    size_t n_split = 0;
    for (int il = 0; il < n_layer && il < (int) xc.size(); ++il) {
        if (il < (int) counts.size() && (int) counts[(size_t) il].size() == n_expert) prior[(size_t) il] = counts[(size_t) il];
        auto & xl = xc[(size_t) il];
        if (xl.cold.empty() || xl.hot.empty()) continue;
        xl.cold.insert(xl.cold.end(), (size_t) std::max(n_spare, 0), -1);      // free cold slots at the end of the stack
        ++n_split;
    }
    return n_split;
}

// The prior rows as a learned save hands them to the PXQN library's builder (src/llama-pxa-xcache.cpp, persist_learned):
// [il * n_expert + e]; a layer with no row (or a row of another width) is all zero, which the builder reads as no prior.
inline std::vector<double> pxa_xc_prior_flat(const std::vector<std::vector<double>> & prior, int n_layer, int n_expert) {
    if (n_layer <= 0 || n_expert <= 0) return {};
    std::vector<double> flat((size_t) n_layer * (size_t) n_expert, 0.0);
    for (int il = 0; il < n_layer && il < (int) prior.size(); ++il) {
        const auto & row = prior[(size_t) il];
        if ((int) row.size() != n_expert) continue;
        for (int e = 0; e < n_expert; ++e) flat[(size_t) il * (size_t) n_expert + (size_t) e] = row[(size_t) e];
    }
    return flat;
}
