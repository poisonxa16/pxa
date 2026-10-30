// PXA_XCACHE_OT (glm-mc-2-xcache-multidevice, 2026-10-18; default OFF): the expert cache under a user -ot placement.
//
// The auto planner (llama-pxa-place.cpp) never runs when -ot places blk.* tensors ("user -ot places layer tensors"),
// and that is exactly how GLM (glm5next) is booted on several cards: whole layers pinned to devices with -ot. So GLM
// never got the expert-granular split. PXA_XCACHE_OT=<n_hot> keeps the user's -ot device of every layer and splits each
// MoE layer's routed experts on THAT device: the n_hot most-routed experts (routing counts csv, tensor,expert,count)
// stay in the layer's own buffer, the rest go to the cold stack (CUDA: the device's stream ring; CPU: the same host
// buffer). The split graph (ggml_moe_split_ids + two mul_mat_id + ggml_moe_merge) puts every routed slot back in place
// before the routing weights are applied, so on CPU the logits must be bit-identical to the static -ot placement
// (tests/test-pxa-xcache-ot.cpp for the split; tests/run-xcache-ot-bitid.sh for the end-to-end tiny-GLM check).
//
//   PXA_XCACHE_OT=<n_hot>        experts kept hot per MoE layer (0 / unset = off; >= n_expert = nothing cold)
//   PXA_XCACHE_OT_COUNTS=<csv>   routing counts (falls back to PXA_XCACHE_COUNTS, then <model>.expert-counts.csv)
//   PXA_XCACHE_OT_LAYERS=a-b     only split layers a..b (default: every layer with counts)
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct pxa_xc_ot_layer { std::vector<int32_t> hot, cold; };

// Per layer: the n_hot experts with the most routings stay hot (ties: lower expert id first), both lists ascending.
// A layer with no counts (empty row) or n_hot >= its expert count is left unsplit (empty hot and cold).
static inline std::vector<pxa_xc_ot_layer> pxa_xc_ot_split(const std::vector<std::vector<double>> & counts, int n_hot,
        int il_min = 0, int il_max = 1 << 30) {
    std::vector<pxa_xc_ot_layer> out(counts.size());
    if (n_hot < 0) n_hot = 0;
    for (size_t il = 0; il < counts.size(); ++il) {
        const auto & c = counts[il];
        const int ne = (int) c.size();
        if (ne == 0 || n_hot >= ne || (int) il < il_min || (int) il > il_max) continue;
        std::vector<int32_t> order(ne);
        for (int e = 0; e < ne; ++e) order[e] = e;
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return c[a] > c[b]; });
        auto & L = out[il];
        L.hot.assign(order.begin(), order.begin() + n_hot);
        L.cold.assign(order.begin() + n_hot, order.end());
        std::sort(L.hot.begin(), L.hot.end());
        std::sort(L.cold.begin(), L.cold.end());
    }
    return out;
}

// Reads tensor,expert,count rows; per layer the ffn_down_exps rows are used, else the first expert tensor seen.
// Returns false (and an error text) on a malformed row or an out-of-range layer/expert.
static inline bool pxa_xc_ot_read_counts(const char * path, int n_layer, int n_expert,
        std::vector<std::vector<double>> & counts, std::string & err) {
    counts.assign(n_layer, {});
    FILE * f = fopen(path, "r");
    if (!f) { err = std::string("cannot open ") + path; return false; }
    std::vector<std::string> src(n_layer);
    char line[1024];
    bool ok = true;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "blk.", 4) != 0) continue;            // header / comments
        char * c1 = strchr(line, ',');
        if (!c1) { err = "row without a comma"; ok = false; break; }
        *c1 = 0;
        const std::string nm = line;
        char * c2 = strchr(c1 + 1, ',');
        if (!c2) { err = "row without a count: " + nm; ok = false; break; }
        const int il = atoi(nm.c_str() + 4), e = atoi(c1 + 1);
        const double v = atof(c2 + 1);
        if (il < 0 || il >= n_layer || e < 0 || e >= n_expert || !(v >= 0)) { err = "bad row for " + nm; ok = false; break; }
        if (nm.find("_exps") == std::string::npos) continue;
        const bool is_down = nm.find("ffn_down_exps") != std::string::npos;
        if (src[il].empty() || (is_down && src[il].find("ffn_down_exps") == std::string::npos)) {
            if (src[il] != nm) { src[il] = nm; counts[il].assign(n_expert, 0.0); }
        }
        if (src[il] == nm) counts[il][e] = v;
    }
    fclose(f);
    return ok;
}

static inline int pxa_xc_ot_n_hot() {
    const char * e = getenv("PXA_XCACHE_OT");
    return e ? atoi(e) : 0;
}
