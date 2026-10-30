// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// Where the expert cache looks for its routing counts (src/llama-pxa-counts.h, the exact code src/llama.cpp runs):
//   1. the map is found under EVERY name the model answers to - the release dir names it after the symlink
//      (<link>.gguf.expert-counts.csv) while the engine was handed the real file (or the other way round);
//   2. a SET-but-unusable PXA_XCACHE_COUNTS (empty, dangling symlink, missing in a read-only dir) is ignored as if unset,
//      while a readable one, or a missing one in a writable dir (the profile pass writes it), is still the one file.
// No GPU, no model: plain files in a temp dir.

#include "../src/llama-pxa-counts.h"

#include <cstdio>
#include <fstream>

static int g_fail = 0;

static void check(const char * name, bool ok) {
    printf("%-64s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) g_fail++;
}

static std::string first_existing(const std::vector<std::string> & v) {
    for (const auto & c : v) if (access(c.c_str(), R_OK) == 0) return c;
    return "";
}

#ifdef _WIN32
// Sibling-symlink lookup is compiled out of the engine on Windows (a Windows path is never reported
// as a symlink). mkdtemp/symlink are POSIX. The checks below need both, so the test skips.
int main() {
    printf("SKIP test-pxa-xcache-counts: symlink count-map lookup is Linux-only\n");
    return 0;
}
#else
int main() {
    char tmpl[] = "/tmp/pxa-counts-XXXXXX";
    const char * root = mkdtemp(tmpl);
    if (!root) { printf("mkdtemp failed\n"); return 1; }
    const std::string R = root, store = R + "/store", rel = R + "/release";
    mkdir(store.c_str(), 0755);
    mkdir(rel.c_str(), 0755);
    const std::string real = store + "/Swift-FN-32GB.relaid.gguf";
    std::ofstream(real) << "x";
    const std::string link = rel + "/Swift-FN-32GB.gguf";
    if (symlink(real.c_str(), link.c_str()) != 0) { printf("symlink failed\n"); return 1; }
    const std::string csv_link = link + ".expert-counts.csv";
    std::ofstream(csv_link) << "tensor,expert,count\n";
    unsetenv("PXA_XCACHE_COUNTS");
    setenv("PXA_CACHE_DIR", (R + "/cache").c_str(), 1);

    // 1. the release layout: the map is named after the symlink
    check("opened by the symlink: the link-named map (as before)", first_existing(pxa_xcache_counts_candidates(link)) == csv_link);
    {
        const auto v = pxa_xcache_counts_candidates(link);
        check("the default write target stays <path given>.expert-counts.csv", !v.empty() && v.front() == csv_link);
        bool cache = false;
        for (const auto & c : v) cache |= c == R + "/cache/Swift-FN-32GB.relaid.gguf.expert-counts.csv";
        check("the cache dir is searched under the real file's name too", cache);
    }
    // opened by the REAL name, with the link beside it (a release dir that keeps both): the link-named map is found
    const std::string link2 = store + "/Swift-FN-32GB.gguf";
    symlink(real.c_str(), link2.c_str());
    const std::string csv_link2 = link2 + ".expert-counts.csv";
    std::ofstream(csv_link2) << "tensor,expert,count\n";
    check("opened by the REAL name: the sibling link's map is found", first_existing(pxa_xcache_counts_candidates(real)) == csv_link2);
    unlink(csv_link2.c_str()); unlink(link2.c_str());
    // and the other way round: the map named after the real file, the model opened by the link
    unlink(csv_link.c_str());
    const std::string csv_real = real + ".expert-counts.csv";
    std::ofstream(csv_real) << "tensor,expert,count\n";
    check("opened by the symlink: the real-named map is found", first_existing(pxa_xcache_counts_candidates(link)) == csv_real);

    // 2. PXA_XCACHE_COUNTS
    std::string why;
    setenv("PXA_XCACHE_COUNTS", csv_real.c_str(), 1);
    auto v = pxa_xcache_counts_candidates(link, &why);
    check("readable PXA_XCACHE_COUNTS: the one file", v.size() == 1 && v[0] == csv_real && why.empty());
    setenv("PXA_XCACHE_COUNTS", (R + "/new.csv").c_str(), 1);
    why.clear(); v = pxa_xcache_counts_candidates(link, &why);
    check("missing in a writable dir: still the one file (profiling writes it)", v.size() == 1 && why.empty());
    const std::string dang = R + "/dangling.csv";
    symlink((R + "/nowhere/x.csv").c_str(), dang.c_str());
    setenv("PXA_XCACHE_COUNTS", dang.c_str(), 1);
    why.clear(); v = pxa_xcache_counts_candidates(link, &why);
    check("dangling symlink: IGNORED, said why, the map beside the model found",
          why.find("dangling") != std::string::npos && first_existing(v) == csv_real);
    setenv("PXA_XCACHE_COUNTS", "", 1);
    why.clear(); v = pxa_xcache_counts_candidates(link, &why);
    check("empty: IGNORED as if unset", !why.empty() && first_existing(v) == csv_real);
    setenv("PXA_XCACHE_COUNTS", "/proc/pxa-no-such-dir/x.csv", 1);
    why.clear(); v = pxa_xcache_counts_candidates(link, &why);
    check("missing in a dir it may not write: IGNORED", why.find("not writable") != std::string::npos && first_existing(v) == csv_real);
    check("pxa_counts_env_given() is false for an ignored value", !pxa_counts_env_given());
    setenv("PXA_XCACHE_COUNTS", csv_real.c_str(), 1);
    check("pxa_counts_env_given() is true for a readable one", pxa_counts_env_given());
    unsetenv("PXA_XCACHE_COUNTS");

    unlink(dang.c_str()); unlink(csv_real.c_str()); unlink(link.c_str()); unlink(real.c_str());
    rmdir(store.c_str()); rmdir(rel.c_str()); rmdir(root);
    printf("%s\n", g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
#endif
