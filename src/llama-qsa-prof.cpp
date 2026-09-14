// PXA_QSA_PROF: implementation. See the header for what it measures and why.

#include "llama-qsa-prof.h"

#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr const char * st_names[PXA_QSA_ST_NB] = {
    "iproj", "pool", "score", "topk", "expand", "gather", "attn", "other"
};

constexpr const char * h_names[PXA_QSA_H_NB] = { "scan", "layout", "marknew", "fill" };

struct tag_rec {
    int8_t stage;
    int16_t il;
    bool terminal;
};

struct prof_state {
    // per-graph, rebuilt on every llama_build_graph
    std::unordered_map<const ggml_tensor *, tag_rec> tags;

    // resolved after seal: node -> stage, used by the level-2 callback
    std::unordered_map<const ggml_tensor *, int8_t> node_stage;

    // the last reported static shape, so the table is printed on change and not per token
    uint64_t shape_key = 0;

    // accumulators for the current step
    int64_t host_cur[PXA_QSA_H_NB] = {};
    int64_t node_cur[PXA_QSA_ST_NB] = {};
    int64_t node_t0 = 0;

    // rows: one per recorded decode step
    static constexpr int NCOL = (int) PXA_QSA_H_NB + (int) PXA_QSA_ST_NB;
    std::vector<std::array<int64_t, (size_t) NCOL>> rows;
};

prof_state & ps() { static prof_state s; return s; }

int prof_every() {
    static const int v = [] {
        const char * e = getenv("PXA_QSA_PROF_EVERY");
        const int n = e ? atoi(e) : 0;
        return n > 0 ? n : 32;
    }();
    return v;
}

// bytes a node writes -- the second-order term after the launch count
int64_t node_bytes(const ggml_tensor * t) {
    return (int64_t) ggml_nbytes(t);
}

} // namespace

int pxa_qsa_prof_level() {
    static const int v = [] {
        const char * e = getenv("PXA_QSA_PROF");
        const int n = e ? atoi(e) : 0;
        return n > 0 ? n : 0;
    }();
    return v;
}

void pxa_qsa_prof_tag(const ggml_tensor * t, int stage, int il, bool terminal) {
    if (!pxa_qsa_prof_on() || t == nullptr) {
        return;
    }
    ps().tags[t] = tag_rec{ (int8_t) stage, (int16_t) il, terminal };
}

void pxa_qsa_prof_begin_graph() {
    if (!pxa_qsa_prof_on()) {
        return;
    }
    ps().tags.clear();
    ps().node_stage.clear();
}

void pxa_qsa_prof_host_add(int bucket, int64_t us) {
    if (!pxa_qsa_prof_on() || bucket < 0 || bucket >= PXA_QSA_H_NB) {
        return;
    }
    ps().host_cur[bucket] += us;
}

void pxa_qsa_prof_node_begin() {
    if (pxa_qsa_prof_level() < 2) {
        return;
    }
    ps().node_t0 = ggml_time_us();
}

void pxa_qsa_prof_node_end(const ggml_tensor * t) {
    if (pxa_qsa_prof_level() < 2) {
        return;
    }
    auto & s = ps();
    if (s.node_t0 == 0) {
        return;
    }
    const int64_t us = ggml_time_us() - s.node_t0;
    s.node_t0 = 0;
    auto it = s.node_stage.find(t);
    const int st = it == s.node_stage.end() ? (int) PXA_QSA_ST_OTHER : (int) it->second;
    s.node_cur[st] += us;
}

void pxa_qsa_prof_seal(const ggml_cgraph * gf, int n_tokens, int n_kv,
                       int n_pool, int n_top, int n_sel, bool gather) {
    if (!pxa_qsa_prof_on() || gf == nullptr) {
        return;
    }
    auto & s = ps();

    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(gf));

    // A node's stage: its own tag if it has one, else the stage of the first source that has
    // already resolved to a real QSA stage and is not a stage TERMINAL. Graph order is
    // topological, so one forward pass resolves everything.
    std::unordered_map<const ggml_tensor *, int8_t> stage;
    std::unordered_set<const ggml_tensor *>         terminal;
    stage.reserve(n_nodes*2);

    struct acc { int64_t nodes = 0; int64_t bytes = 0; };
    std::array<acc, PXA_QSA_ST_NB> total{};
    // per-layer counts for the layers that actually carry QSA
    std::unordered_map<int, std::array<acc, PXA_QSA_ST_NB>> per_layer;
    std::unordered_map<const ggml_tensor *, int16_t> layer_of;

    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * nd = ggml_graph_node(const_cast<ggml_cgraph *>(gf), i);

        int8_t  st = PXA_QSA_ST_OTHER;
        int16_t il = -1;

        auto tit = s.tags.find(nd);
        if (tit != s.tags.end()) {
            st = tit->second.stage;
            il = tit->second.il;
            if (tit->second.terminal) {
                terminal.insert(nd);
            }
        } else {
            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                const ggml_tensor * src = nd->src[k];
                if (src == nullptr || terminal.count(src)) {
                    continue;
                }
                auto sit = stage.find(src);
                if (sit != stage.end() && sit->second != (int8_t) PXA_QSA_ST_OTHER) {
                    st = sit->second;
                    auto lit = layer_of.find(src);
                    il = lit == layer_of.end() ? (int16_t) -1 : lit->second;
                    break;
                }
            }
        }

        stage[nd]    = st;
        layer_of[nd] = il;

        total[st].nodes += 1;
        total[st].bytes += node_bytes(nd);
        if (il >= 0) {
            per_layer[il][st].nodes += 1;
            per_layer[il][st].bytes += node_bytes(nd);
        }
    }

    s.node_stage = std::move(stage);

    // Print the table only when the shape moves: at decode it is otherwise once per token.
    const uint64_t key = ((uint64_t) n_nodes << 40) ^ ((uint64_t) n_pool << 20) ^
                         ((uint64_t) n_tokens << 8) ^ (uint64_t) (gather ? 1 : 0);
    if (key == s.shape_key) {
        return;
    }
    s.shape_key = key;

    int64_t qsa_nodes = 0;
    int64_t qsa_bytes = 0;
    for (int st = 0; st < PXA_QSA_ST_OTHER; ++st) {
        qsa_nodes += total[st].nodes;
        qsa_bytes += total[st].bytes;
    }

    const int n_qsa_layers = (int) per_layer.size();

    fprintf(stderr,
            "PXA_QSA_PROF graph: n_tokens=%d n_kv=%d n_pool=%d n_top=%d n_sel=%d gather=%d "
            "nodes=%d qsa_nodes=%lld (%.1f%%) qsa_layers=%d\n",
            n_tokens, n_kv, n_pool, n_top, n_sel, gather ? 1 : 0,
            n_nodes, (long long) qsa_nodes,
            n_nodes ? 100.0*(double) qsa_nodes/(double) n_nodes : 0.0, n_qsa_layers);
    fprintf(stderr, "PXA_QSA_PROF %-8s %8s %10s %14s %14s\n",
            "stage", "nodes", "nodes/lyr", "MB_out", "MB_out/lyr");
    for (int st = 0; st < PXA_QSA_ST_NB; ++st) {
        const double mb = (double) total[st].bytes/1048576.0;
        fprintf(stderr, "PXA_QSA_PROF %-8s %8lld %10.2f %14.3f %14.3f\n",
                st_names[st], (long long) total[st].nodes,
                n_qsa_layers ? (double) total[st].nodes/n_qsa_layers : 0.0,
                mb, n_qsa_layers ? mb/n_qsa_layers : 0.0);
    }
    fprintf(stderr, "PXA_QSA_PROF total qsa: %lld nodes, %.3f MB written per token "
                    "(%.1f nodes and %.3f MB per QSA layer)\n",
            (long long) qsa_nodes, (double) qsa_bytes/1048576.0,
            n_qsa_layers ? (double) qsa_nodes/n_qsa_layers : 0.0,
            n_qsa_layers ? (double) qsa_bytes/1048576.0/n_qsa_layers : 0.0);
    fflush(stderr);
}

void pxa_qsa_prof_tick(int n_tokens) {
    if (!pxa_qsa_prof_on()) {
        return;
    }
    auto & s = ps();

    // prefill ubatches are not inter-token intervals and their host cost is out of scope
    if (n_tokens >= 32) {
        for (auto & c : s.host_cur) { c = 0; }
        for (auto & c : s.node_cur) { c = 0; }
        return;
    }

    std::array<int64_t, (size_t) prof_state::NCOL> row{};
    for (int i = 0; i < PXA_QSA_H_NB;  ++i) { row[i]                = s.host_cur[i]; s.host_cur[i] = 0; }
    for (int i = 0; i < PXA_QSA_ST_NB; ++i) { row[PXA_QSA_H_NB + i] = s.node_cur[i]; s.node_cur[i] = 0; }
    s.rows.push_back(row);

    if ((int) s.rows.size() < prof_every()) {
        return;
    }

    const size_t n = s.rows.size();
    fprintf(stderr, "PXA_QSA_PROF step n=%zu us(med/mean) host:", n);
    std::vector<int64_t> col(n);
    auto emit = [&](int c, const char * name) {
        double mean = 0;
        for (size_t i = 0; i < n; ++i) { col[i] = s.rows[i][c]; mean += (double) col[i]; }
        std::nth_element(col.begin(), col.begin() + n/2, col.end());
        fprintf(stderr, " %s=%lld/%lld", name, (long long) col[n/2], (long long) (mean/n));
    };
    for (int i = 0; i < PXA_QSA_H_NB; ++i) { emit(i, h_names[i]); }
    if (pxa_qsa_prof_level() >= 2) {
        fprintf(stderr, " | nodes:");
        for (int i = 0; i < PXA_QSA_ST_NB; ++i) { emit(PXA_QSA_H_NB + i, st_names[i]); }
    }
    fprintf(stderr, "\n");
    fflush(stderr);
    s.rows.clear();
}
