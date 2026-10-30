// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// PXA core step 3: the lever registry and the engine's own serve-flag defaults. See pxa-registry.h.

#include "pxa-env-flag.h"
#include "pxa-registry.h"

#include "ggml.h"
#ifdef _WIN32
#include "pxa-win-compat.h"   // setenv
#endif
#include "ggml-pxqn-levers.h"
#include "ggml-pxqn-overdrive.h"
#include "llama.h"

#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#if defined(__linux__)
#include <sched.h>
#endif

extern char ** environ;

// ---------------------------------------------------------------------------------------------
// The catalog. Generated; see scripts/pxa-lever-catalog.py.
// ---------------------------------------------------------------------------------------------
static const pxa_lever_decl k_pxa_levers[] = {
// the build copy (top-level CMakeLists.txt): the PUBLIC table; the closed library declares its own levers
#include "pxa-lever-catalog-build.inc"
};

const pxa_lever_decl * pxa_lever_catalog(size_t * n) {
    if (n) {
        *n = sizeof(k_pxa_levers) / sizeof(k_pxa_levers[0]);
    }
    return k_pxa_levers;
}

const pxa_lever_decl * pxa_lever_find(const char * name) {
    size_t n = 0;
    const pxa_lever_decl * t = pxa_lever_catalog(&n);
    for (size_t i = 0; i < n; ++i) {
        if (strcmp(t[i].name, name) == 0) {
            return &t[i];
        }
    }
    return nullptr;
}

void pxa_lever_catalog_dump(FILE * out) {
    size_t n = 0;
    const pxa_lever_decl * t = pxa_lever_catalog(&n);
    fprintf(out, "name\tdefault\tscope\tstatus\tevidence\trule\tset\n");
    for (size_t i = 0; i < n; ++i) {
        const char * v = getenv(t[i].name);
        fprintf(out, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", t[i].name, t[i].deflt, t[i].scope, t[i].status,
                t[i].evidence, t[i].rule, v ? v : "");
    }
}

// ---------------------------------------------------------------------------------------------
// Topology
// ---------------------------------------------------------------------------------------------
bool pxa_topology::same_cc() const {
    for (size_t i = 1; i < cc.size(); ++i) {
        if (cc[i] != cc[0]) {
            return false;
        }
    }
    return !cc.empty();
}

int pxa_topology::narrow_link() const {
    for (size_t i = 0; i < pcie_width.size(); ++i) {
        if (pcie_width[i] > 0 && pcie_width[i] < 4) {
            return (int) i;
        }
    }
    return -1;
}

pxa_topology pxa_topology_parse(const char * spec) {
    pxa_topology t;
    t.source = "PXA_TOPOLOGY";
    if (spec == nullptr) {
        return t;
    }
    std::string s(spec);
    // "NxCC[:MiB]" form
    const size_t x = s.find('x');
    if (x != std::string::npos && s.find(',') == std::string::npos) {
        const int n = atoi(s.substr(0, x).c_str());
        std::string rest = s.substr(x + 1);
        size_t mib = 0;
        const size_t colon = rest.find(':');
        if (colon != std::string::npos) {
            mib  = (size_t) atoll(rest.substr(colon + 1).c_str());
            rest = rest.substr(0, colon);
        }
        const int cc = atoi(rest.c_str());
        for (int i = 0; i < n && i < 64; ++i) {
            t.cc.push_back(cc);
            t.vram_mib.push_back(mib);
        }
    } else {
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (item.empty()) {
                continue;
            }
            size_t mib = 0;
            const size_t colon = item.find(':');
            if (colon != std::string::npos) {
                mib = (size_t) atoll(item.substr(colon + 1).c_str());
            }
            t.cc.push_back(atoi(item.c_str()));
            t.vram_mib.push_back(mib);
        }
    }
    t.n_dev = (int) t.cc.size();
    return t;
}

pxa_topology pxa_topology_detect(const std::string & devices, int max_gpu, bool allow_override) {
    const char * spec = getenv("PXA_TOPOLOGY");
    if (allow_override && spec && spec[0]) {
        return pxa_topology_parse(spec);
    }
    pxa_topology t;
    t.source = "none";
#if defined(GGML_USE_CUDA)
    const int n = ggml_backend_cuda_get_device_count();
    std::vector<int> ids;
    if (!devices.empty()) {
        // -dev CUDA0,CUDA2 (anything that is not a CUDA device does not take part in a CUDA split)
        std::stringstream ss(devices);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (item.rfind("CUDA", 0) == 0) {
                const int id = atoi(item.c_str() + 4);
                if (id >= 0 && id < n) {
                    ids.push_back(id);
                }
            }
        }
    }
    if (ids.empty()) {
        for (int i = 0; i < n; ++i) {
            ids.push_back(i);
        }
    }
    if (max_gpu > 0 && (int) ids.size() > max_gpu) {
        ids.resize(max_gpu);
    }
    for (int id : ids) {
        t.cc.push_back(ggml_backend_cuda_get_device_cc(id));
        t.pcie_width.push_back(ggml_backend_cuda_get_device_pcie_width(id));
        size_t vfree = 0, vtot = 0;   // total VRAM: the KV pick needs it (the server initialises these cards anyway)
        ggml_backend_cuda_get_device_memory(id, &vfree, &vtot);
        t.vram_mib.push_back(vtot / 1048576);
    }
    t.n_dev  = (int) t.cc.size();
    t.source = "cuda";
#else
    (void) devices;
    (void) max_gpu;
#endif
    return t;
}

// ---------------------------------------------------------------------------------------------
// Model probe (header only)
// ---------------------------------------------------------------------------------------------
static int pxa_kv_int(const gguf_context * g, int k) {
    switch (gguf_get_kv_type(g, k)) {
        case GGUF_TYPE_UINT8:  return (int) gguf_get_val_u8(g, k);
        case GGUF_TYPE_INT8:   return (int) gguf_get_val_i8(g, k);
        case GGUF_TYPE_UINT16: return (int) gguf_get_val_u16(g, k);
        case GGUF_TYPE_INT16:  return (int) gguf_get_val_i16(g, k);
        case GGUF_TYPE_UINT32: return (int) gguf_get_val_u32(g, k);
        case GGUF_TYPE_INT32:  return (int) gguf_get_val_i32(g, k);
        case GGUF_TYPE_UINT64: return (int) gguf_get_val_u64(g, k);
        case GGUF_TYPE_INT64:  return (int) gguf_get_val_i64(g, k);
        case GGUF_TYPE_ARRAY: {
            // per-layer arrays (e.g. head_count_kv on hybrid arches): the smallest non-zero entry
            const enum gguf_type at = gguf_get_arr_type(g, k);
            if (at == GGUF_TYPE_STRING || at == GGUF_TYPE_ARRAY) {
                return -1;
            }
            const int n = gguf_get_arr_n(g, k);
            const void * d = gguf_get_arr_data(g, k);
            int best = -1;
            for (int i = 0; i < n; ++i) {
                long long v = -1;
                switch (at) {
                    case GGUF_TYPE_UINT32: v = ((const uint32_t *) d)[i]; break;
                    case GGUF_TYPE_INT32:  v = ((const int32_t  *) d)[i]; break;
                    case GGUF_TYPE_UINT16: v = ((const uint16_t *) d)[i]; break;
                    case GGUF_TYPE_INT16:  v = ((const int16_t  *) d)[i]; break;
                    case GGUF_TYPE_UINT8:  v = ((const uint8_t  *) d)[i]; break;
                    case GGUF_TYPE_INT8:   v = ((const int8_t   *) d)[i]; break;
                    default: break;
                }
                if (v > 0 && (best < 0 || v < best)) {
                    best = (int) v;
                }
            }
            return best;
        }
        default: return -1;
    }
}

static std::string pxa_kv_text(const gguf_context * g, int k) {
    char buf[64];
    switch (gguf_get_kv_type(g, k)) {
        case GGUF_TYPE_STRING: return gguf_get_val_str(g, k);
        case GGUF_TYPE_BOOL:   return gguf_get_val_bool(g, k) ? "true" : "false";
        case GGUF_TYPE_FLOAT32: snprintf(buf, sizeof(buf), "%g", gguf_get_val_f32(g, k)); return buf;
        case GGUF_TYPE_FLOAT64: snprintf(buf, sizeof(buf), "%g", gguf_get_val_f64(g, k)); return buf;
        case GGUF_TYPE_ARRAY:
            snprintf(buf, sizeof(buf), "[%d x %s]", gguf_get_arr_n(g, k), gguf_type_name(gguf_get_arr_type(g, k)));
            return buf;
        default: {
            const int v = pxa_kv_int(g, k);
            snprintf(buf, sizeof(buf), "%d", v);
            return buf;
        }
    }
}

// ggml type id -> PXQ tier name (ggml/include/ggml.h). The launcher's tier_from_tensors() is the
// same rule: PXQ1 anywhere wins, more than one PXQ type is PXQ_UNIVERSAL.
static const char * pxa_pxq_tier_name(int t) {
    switch (t) {
        case 248: return "PXQ1";
        case 252: return "PXQ4";
        case 253: return "PXQ4-HQ";
        case 254: return "PXQ2";
        case 255: return "PXQ3";
        case 256: return "PXQ6";
        // PXQ-Next (ggml.h GGML_TYPE_PXQN*, 2026-09-25/26). Without these a PXQN file probed as
        // tier "" here while tools/pxa-launch.py (19c5ba4110) read it as its PXQN tier, so the
        // engine's own picks (the -sm default above all) never saw the new files as PXQ at all.
        case 257: return "PXQN3";
        case 258: return "PXQN3S8";
        case 259: return "PXQN4";
        case 260: return "PXQN2";
        case 261: return "PXQN1";
        case 262: return "PXQN4S8";
        case 263: return "PXQN5";
        default:  return nullptr;
    }
}

pxa_model_info pxa_model_probe(const std::string & path) {
    pxa_model_info m;
    m.path = path;
    if (path.empty()) {
        return m;
    }
    std::vector<std::string> shards = { path };
    std::set<int> pxq_types;
    for (size_t si = 0; si < shards.size(); ++si) {
        struct gguf_init_params ip = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
        gguf_context * g = gguf_init_from_file(shards[si].c_str(), ip);
        if (!g) {
            if (si == 0) {
                return m;
            }
            continue;   // a missing shard: the loader will say so; the probe counts what it can
        }
        {
            std::ifstream f(shards[si], std::ios::binary | std::ios::ate);
            if (f.good()) {
                m.bytes += (uint64_t) f.tellg();
            }
        }
        if (si == 0) {
            m.ok = true;
            const int ia = gguf_find_key(g, "general.architecture");
            if (ia >= 0) {
                m.arch = gguf_get_val_str(g, ia);
                const int ie = gguf_find_key(g, (m.arch + ".expert_count").c_str());
                m.n_expert = ie < 0 ? 0 : pxa_kv_int(g, ie);
                const int ih = gguf_find_key(g, (m.arch + ".attention.head_count_kv").c_str());
                m.n_head_kv = ih < 0 ? -1 : pxa_kv_int(g, ih);
                const int il = gguf_find_key(g, (m.arch + ".block_count").c_str());
                m.n_layer = il < 0 ? -1 : pxa_kv_int(g, il);
                const int ic = gguf_find_key(g, (m.arch + ".context_length").c_str());
                m.n_ctx_train = ic < 0 ? -1 : pxa_kv_int(g, ic);
                const int inx = gguf_find_key(g, (m.arch + ".nextn_predict_layers").c_str());
                m.n_nextn = inx < 0 ? 0 : pxa_kv_int(g, inx);
                const int ik = gguf_find_key(g, (m.arch + ".attention.key_length").c_str());
                const int iv = gguf_find_key(g, (m.arch + ".attention.value_length").c_str());
                const int ii = gguf_find_key(g, (m.arch + ".full_attention_interval").c_str());
                m.head_k = ik < 0 ? -1 : pxa_kv_int(g, ik);
                m.head_v = iv < 0 ? m.head_k : pxa_kv_int(g, iv);
                m.full_attn_interval = ii < 0 ? -1 : pxa_kv_int(g, ii);
            }
            const int nkv = gguf_get_n_kv(g);
            for (int k = 0; k < nkv; ++k) {
                const char * key = gguf_get_key(g, k);
                if (strncmp(key, "pxa.", 4) == 0) {
                    m.pxa_kv.emplace_back(key, pxa_kv_text(g, k));
                }
            }
            const int isc = gguf_find_key(g, "split.count");
            if (isc >= 0) {
                const int count = pxa_kv_int(g, isc);
                if (count > 1) {
                    m.n_shards = count;
                    char prefix[4096];
                    if (llama_split_prefix(prefix, sizeof(prefix), path.c_str(), 1, count) > 0) {
                        for (int s = 2; s <= count; ++s) {
                            char sp[4096];
                            llama_split_path(sp, sizeof(sp), prefix, s, count);
                            shards.push_back(sp);
                        }
                    }
                }
            }
        }
        const int nt = gguf_get_n_tensors(g);
        for (int i = 0; i < nt; ++i) {
            const int t = (int) gguf_get_tensor_type(g, i);
            if (pxa_pxq_tier_name(t)) {
                pxq_types.insert(t);
                m.n_pxq++;
            }
        }
        gguf_free(g);
    }
    for (int t : pxq_types) {
        m.pxq_mask |= 1u << (t - 248);
    }
    if (pxq_types.count(248)) {
        m.tier = "PXQ1";
    } else if (pxq_types.size() > 1) {
        m.tier = "PXQ_UNIVERSAL";
    } else if (pxq_types.size() == 1) {
        m.tier = pxa_pxq_tier_name(*pxq_types.begin());
    }
    return m;
}

// ---------------------------------------------------------------------------------------------
// Tier scopes for MIXED PXQN files (2026-10-24). tier_from_tensors reads every file that carries more than one PXQ
// tensor type as PXQ_UNIVERSAL, so a rule that tests tier == "PXQN4" never sees a PXQN4+PXQN5 / PXQN3+PXQN4+PXQN5 mix. pxq_mask (which
// types are present) lets a rule take the mix where the lever does not depend on the weight format.
// ---------------------------------------------------------------------------------------------
static constexpr uint32_t pxq_bit(int type_id) { return 1u << (type_id - 248); }
static bool pxa_pxq_mask_within(const pxa_model_info & m, uint32_t allowed) {
    return m.pxq_mask != 0 && (m.pxq_mask & ~allowed) == 0;
}
// Flint one-card rules (measured on PXQN4 files). The published Flint B (flint-pxqn4mix-r21) keeps the same routers (47 BF16
// ffn_gate_inp) and the same expert-cache / split-path layout, but its routed experts are PXQN4 + PXQN5, so it probes as
// PXQ_UNIVERSAL. None of the three levers reads the expert format (the planner prices layers by bytes, the router GEMV reads the BF16
// router, the V100 stack is bit-identical graph capture). Scope: tier PXQN4, or PXQ_UNIVERSAL whose PXQ tensors are all PXQN4/PXQN5.
static bool pxa_flint_pxqn_tier(const pxa_model_info & model) {
    return model.tier == "PXQN4" || (model.tier == "PXQ_UNIVERSAL" && pxa_pxq_mask_within(model, pxq_bit(259) | pxq_bit(263)));
}
// Dense 27B (qwen35) rules measured on the single-type PXQN4 ladder file. The published OneCard files (Qwen3.8-27B-PXQN-OneCard
// 8fccd7aa, Swift-1.5 4674a8c8) are PXQN4 (all attention / ssm_out / FFN of layers 32-47) + PXQN3 (FFN of the other 49 layers) + one
// PXQN5 output head, so they probe as PXQ_UNIVERSAL. The levers the rules arm do not read the FFN weight format: argmax / greedy
// readback, delta-net conv fuse, the fused rms-norm + RHT site (it rotates activations, whatever the consumer), the prompt-attention
// tile (f16 KV flash attention), the -ub fit (host memory arithmetic). PXA_PXQN_RBP / RBP_TP are compiled for the PXQN4 kernel kind
// only (pxqn-kern.cuh KIND == PXQN_K_N4): a PXQN3 tensor keeps its own kernel, so on the mix RBP covers the PXQN4 tensors and is
// bit-identical there. Scope: tier PXQN4, or PXQ_UNIVERSAL whose PXQ tensors are PXQN3/PXQN4/PXQN5 with at least one PXQN4.
static bool pxa_dense27_tier(const pxa_model_info & model) {
    return model.tier == "PXQN4" || (model.tier == "PXQ_UNIVERSAL" && (model.pxq_mask & pxq_bit(259)) != 0 &&
                                     pxa_pxq_mask_within(model, pxq_bit(257) | pxq_bit(259) | pxq_bit(263)));
}
// "a PXQN tier" for the one-P100 qwen35 stack (the rules that test the PXQN prefix): any PXQN file, or the 27B mix above
static bool pxa_dense27_pxqn_family(const pxa_model_info & model) {
    return model.tier.rfind("PXQN", 0) == 0 || pxa_dense27_tier(model);
}

// ---------------------------------------------------------------------------------------------
// The rules
// ---------------------------------------------------------------------------------------------

// Arches and tiers the tensor split is a DEFAULT for (the launcher's TSPLIT_AUTO_ARCHES /
// TSPLIT_AUTO_TIERS, which this mirrors exactly: tools/pxa-launch.py resolve_auto_split()). The
// capability check in src/llama.cpp admits more than this; a default needs a measurement, an
// admission only needs the builders to be complete.
static bool pxa_tsplit_auto_arch(const std::string & a) { return a == "qwen35"; }
// PXQN4 / PXQN4S8 / PXQN5 (the 2-card PXQ-Next sizes) joined PXQ4 on 2026-09-27.
// Evidence: the dense 27B on one binary (bigq-pp speedcheck table),
// -sm tensor leg against -sm layer leg, REPS 3: P100 pair decode +17.1/+24.3/+26.8%, prefill @22.6k
// +43.8/+44.0/+45.7%; V100 pair decode +12.1/+13.3/+28.0%, prefill +3.6/+3.0/+3.6%. The 1-card sizes
// (PXQN1/2/3/3S8) have no pair measurement and keep the layer split. PXA_AUTO_SM_PXQN=0 restores the
// PXQ4-only admission (read per call, like PXA_AUTO_SM, so the unit test can flip it).
static bool pxa_tsplit_auto_tier(const std::string & t) {
    if (t == "PXQ4") {
        return true;
    }
    const char * e = getenv("PXA_AUTO_SM_PXQN");
    if (e && e[0] == '0') {
        return false;
    }
    return t == "PXQN4" || t == "PXQN4S8" || t == "PXQN5";
}
// OneCard 27B mix (PXQN4 attention + PXQN3 FFN + PXQN5 head; tier PXQ_UNIVERSAL) on a P100 PAIR. 2026-10-24:
// public Qwen3.8-27B-PXQN-OneCard, 2x P100 cards 0+1, same build, boots layer/tensor/tensor/layer, 60 s warm-up, REPS 3 greedy512: tensor
// 40.10 +-0.09 prose / 40.26 +-0.07 code against layer(+P2L) 30.54 / 30.56 (+31.3% / +31.7%), both greedy shas identical to the layer arm.
// P100 pairs only: a V100 pair of this mix has no number yet. PXA_AUTO_SM_PXQN=0 restores the PXQN4/4S8/5-only rule.
static bool pxa_tsplit_onecard_pair(const pxa_model_info & m, const pxa_topology & topo) {
    const char * e = getenv("PXA_AUTO_SM_PXQN");
    if (e && e[0] == '0') {
        return false;
    }
    return m.tier == "PXQ_UNIVERSAL" && m.pxq_mask == (pxq_bit(257) | pxq_bit(259) | pxq_bit(263)) && topo.n_dev == 2 && topo.cc.size() == 2 && topo.cc[0] == 600 && topo.cc[1] == 600;
}
static bool pxa_is_gemma4(const std::string & a) { return a == "gemma4" || a == "gemma4_mtp" || a == "gemma4-assistant"; }

static const char * pxa_card_name(int cc) {
    switch (cc) {
        case 600: return "P100 (sm_60)";
        case 610: return "1080 Ti/P40 (sm_61)";
        case 700: return "V100 (sm_70)";
        default:  return "card";
    }
}

int pxa_registry_batch_cell(const pxa_topology & topo, const pxa_model_info & model, int level,
                            int * n_batch, int * n_ubatch, const char ** why,
                            const char ** status, int * n_ctx) {
    const char * st = "MEASURED";
    if (level < 2 || topo.n_dev <= 0) {
        return 0;
    }
    // Narrow file-layout cells; ordinary 11 GB Turing boards and other files keep their policy.
    const bool flashnext32 = model.ok && model.arch == "qwen4exp" && model.tier == "PXQ_UNIVERSAL" &&
        model.n_expert == 512 && model.n_head_kv == 2 && model.n_layer == 49 &&
        model.n_pxq == 681 && model.bytes == 83201745728ULL &&
        std::any_of(model.pxa_kv.begin(), model.pxa_kv.end(), [](const auto & kv) {
            return kv.first == "pxa.pxqn.encoder" && kv.second.compare(0, 4, "PXQN") == 0;
        });
    if (flashnext32 && topo.cc.size() == (size_t) topo.n_dev && topo.vram_mib.size() == topo.cc.size()) {
        int turing = 0, volta = 0;
        for (size_t i = 0; i < topo.cc.size(); ++i) {
            turing += topo.cc[i] == 750 && topo.vram_mib[i] >= 21504 && topo.vram_mib[i] < 23552;
            volta += topo.cc[i] == 700 && topo.vram_mib[i] >= 31744 && topo.vram_mib[i] < 33792;
        }
        if ((topo.n_dev == 1 && turing == 1) || (topo.n_dev == 2 && turing == 1 && volta == 1)) {
            if (n_batch) *n_batch = 2048;
            if (n_ubatch) *n_ubatch = 2048;
            if (n_ctx) *n_ctx = 0;
            if (status) *status = "FIT";
            if (why) *why = topo.n_dev == 1
                ? "Flash-Next PXQN 32GB layout, 1x SM75 22GB: prefill candidate -b/-ub 2048, verified at load (docs/SM75.md)"
                : "Flash-Next PXQN 32GB layout, SM75 22GB + V100 32GB: layer-split prefill candidate -b/-ub 2048, verified at load (docs/SM75.md)";
            return 1;
        }
    }
    if (!topo.same_cc()) {
        return 0;   // other mixed sets retain the existing adaptive policy
    }
    const int  n      = topo.n_dev;
    const int  cc     = topo.cc[0];
    const int  ne     = model.n_expert;
    const bool expert = ne > 0;
    const std::string & tier = model.tier;
    int b = 0, ub = 0, cctx = 0;
    const char * cell = nullptr;

    // Gemma 4 first (launcher GEMMA4_CELLS, 2026-09-20): the 26B-A4B expert file on ONE V100 (any
    // tier), a V100 pair or a P100 pair, all at -b 2048 -ub 512 (measured 2026-09-20). On a PAIR
    // the status follows the file's tier, exactly as GEMMA4_CELLS has it: the Google QAT q4_0 file
    // (tier "", not a PXQ tensor type) is MEASURED there; OUR OWN PXQ3 file is a mixed map, so
    // tier_from_tensors() reads it as PXQ_UNIVERSAL, and that pair row is INFERRED ("the pair was
    // not booted on the PXQ3 file" - arithmetic from the one-card boot). This rule used to return
    // "MEASURED" for both regardless of tier (#8382 problem 2: "the rule ignores tier"). The 1x
    // P100, 4x P100 and every dense-Gemma row are INFERRED in that table too and are not cells
    // here at all: those files fall through to the card table below, exactly as before step 3.
    // The generic 2x sm_70 cell (-ub 2048) was measured on the Qwen line.
    if (pxa_is_gemma4(model.arch) && expert && n == 1 && cc == 600) {
        // ONE P100 (16 GB): the card table's one-P100 cell (a Qwen measurement) did not leave room for the
        // 26B-A4B expert file, so llama-server fell back to weight streaming at 1.1 (v2026.10.2) / 1.8 t/s; at
        // -ub 256 the same server decodes 66.3 t/s and MTP n1 63.7 (v2026.10.3 gate, rel-gate2 #14827, 2026-10-05,
        // Google QAT q4_0, -c 10240).
        b = 2048; ub = 256;
        cell = "Gemma 4 expert file on 1x P100 (measured 2026-10-05 by the v2026.10.3 gate: -b 2048 -ub 256, "
               "66.3 t/s decode; the generic one-P100 cell did not fit)";
    } else if (pxa_is_gemma4(model.arch) && expert &&
        ((n == 1 && cc == 700) || (n == 2 && (cc == 700 || cc == 600)))) {
        b = 2048; ub = 512;
        cell = "Gemma 4 expert file on 1x V100 / 2x V100 / 2x P100 (measured 2026-09-20: -b 2048 -ub 512)";
        if (n == 2 && (tier == "PXQ3" || tier == "PXQ_UNIVERSAL")) {
            st = "INFERRED";
            cell = "Gemma 4 expert file on 2x V100 / 2x P100, our PXQ3 map (tier_from_tensors: "
                   "PXQ_UNIVERSAL): -b/-ub carried over from the Google QAT q4_0 pair measurement, "
                   "not booted on this tier (measured 2026-09-20 on 1x V100 only)";
        }
    } else if (n == 2 && cc == 700) {
        b = 8192; ub = 2048; cctx = 32768;
        cell = "2x V100 (sm_70) (measured: prefill 1,369 t/s @3k, 1,300 @20k; -ub 2048 beats 512 at -b 8192)";
        // The numbers above are the DENSE 27B PXQ4 pair boot (2026-09-03 FINAL V100 cells). The 35B MoE
        // flagship (expert PXQ4/PXQ6/PXQ4-HQ) on a V100 pair was never booted: tools/pxa-launch.py's
        // "2xpair-flagship-moe-v100" row borrows the P100 pair's flags and is INFERRED ("the V100 pair runs
        // the same command and its own number was NOT taken"). This branch used to hand every file on a
        // V100 pair the dense cell's MEASURED label; the picks are unchanged, only the label follows the
        // launcher now, until a V100-pair boot of that file is measured (todo
        // registry-flagship-moe-v100-label, same pattern as the gemma4 PXQ3 pair row above).
        if (expert && (tier == "PXQ4" || tier == "PXQ6" || tier == "PXQ4-HQ")) {
            st = "INFERRED";
            cell = "2x V100 (sm_70), expert PXQ4/PXQ6 file (the 35B MoE flagship): -b 8192 -ub 2048 carried over "
                   "from the 2x P100 flagship boot (decode 55.7, prefill ~843 t/s); the V100 pair was not booted "
                   "on this file, so no V100-pair number is quoted";
        }
    } else if (n == 2 && cc == 600) {
        if (expert && (tier == "PXQ4" || tier == "PXQ6" || tier == "PXQ4-HQ")) {
            b = 8192; ub = 2048; cctx = 8192;
            cell = "2x P100 (sm_60), expert PXQ4/PXQ6 file (the 35B MoE flagship cell: decode 55.7, prefill ~843 t/s at -ub 2048)";
        } else {
            b = 8192; ub = 256; cctx = 32768;
            cell = "2x P100 (sm_60) (measured: 340 t/s @3k, 317 @20k at -ub 256 against 231 at 2048)";
        }
    } else if (n == 1 && cc == 610 && !topo.vram_mib.empty() && topo.vram_mib[0] >= 20480) {
        // The P40 (24 GB) shares cc 610 with the 11 GB 1080 Ti but has the memory of a big card: it
        // takes the 16 GB-class batch (a user's P40 log, 2026-09-23, showed it on the 1080 Ti cell).
        b = 2048; ub = 2048; cctx = 8192;
        st = "INFERRED";
        cell = "1x P40-class 24 GB (sm_61) (-ub 2048 as on the 16 GB cards; not measured on a P40 here)";
    } else if (n == 1 && cc == 610) {
        b = 2048; ub = 768; cctx = 8192;
        cell = "1x 1080 Ti (sm_61) (measured: 1,306 t/s cold prefill; -ub 2048 does not fit on 11 GB)";
    } else if (n == 1 && (cc == 600 || cc == 700) && tier == "PXQ_UNIVERSAL" && expert) {
        b = 2048; ub = 2048; cctx = 8192;
        cell = cc == 600 ? "1x P100, PXQU MoE (measured: prefill 827-843 t/s at -ub 2048)"
                         : "1x V100, PXQU MoE (measured: prefill ~1,800-1,900 t/s at -ub 2048)";
    } else if (n == 4 && cc == 600) {
        const char * lv    = getenv("PXA_AUTO_UB_LONG");
        const int    lever = lv ? atoi(lv) : -1;
        b = 2048;
        if (lever == 0) {
            ub = 2048; cell = "4x P100 (sm_60), PXA_AUTO_UB_LONG=0 (pre-2026-09-13 flat cell)";
        } else if (lever == 1) {
            ub = 256;  cell = "4x P100 (sm_60), PXA_AUTO_UB_LONG=1 (small ubatch forced)";
        } else if (ne > 0) {
            ub = 2048; cell = "4x P100 (sm_60), expert file (measured on the 177B hybrid MoE seat: 478 t/s @3121 at -ub 2048)";
        } else if (ne == 0) {
            ub = 256;  cell = "4x P100 (sm_60), dense file (measured: +46/+101/+108/+111% at fills 505/3121/8223/20801 over -ub 2048)";
        } else {
            ub = 2048; cell = "4x P100 (sm_60), expert count unreadable - keeping the larger ubatch";
        }
    } else if (model.arch == "qwen4exp") {
        // Flash-Next off its 4x P100 seat (the launcher's ub1024 rule): +20% prefill over the
        // adaptive 512 on the six-card PXQ4 seat, decode unchanged.
        b = 1024; ub = 1024; st = "RULE";
        cell = "qwen4exp (Flash-Next) off the 4x P100 cell (measured: -ub 1024 +20% prefill over 512 on six cards)";
    } else {
        return 0;
    }
    // Flash-Next (qwen4exp) on 16 GB cards: never -b 8192. The 2x P100 cell above is a DENSE 27B measurement and handed this file
    // -b 8192 (CriticalTechGuy's 2x P100 seat, 2026-10-12); the measured Flash-Next seats ran -b 2048 (Control's click path,
    // 36.86 t/s prose, needle pass) and the -b 8192 arm measured no better (35.58). A card whose VRAM is not known is treated as
    // 16 GB-class when it is a P100 / 1080 Ti / V100 (the cards the cells above were measured on).
    if (model.arch == "qwen4exp" && b > 2048) {
        bool small = true;
        for (int i = 0; i < n; ++i) {
            const size_t v = i < (int) topo.vram_mib.size() ? topo.vram_mib[i] : 0;
            if (v > 16896 || (v == 0 && cc != 600 && cc != 610 && cc != 700)) small = false;
        }
        if (small) {
            static std::string fn_cell;
            b = std::max(2048, ub);
            fn_cell = std::string(cell) + "; -b capped at 2048 for the Flash-Next class on 16 GB cards (measured: -b 2048 36.86 "
                      "t/s prose vs -b 8192 35.58 on 2x P100)";
            cell = fn_cell.c_str();
        }
    }
    if (n_batch)  *n_batch  = b;
    if (n_ubatch) *n_ubatch = ub;
    if (why)      *why      = cell;
    if (status)   *status   = st;
    if (n_ctx)    *n_ctx    = cctx;
    return 1;
}

int pxa_registry_ub_rule_volta_q8(const pxa_topology & topo, int level, int n_ctx, bool kv_q8,
                                  const char ** why) {
    const char * lv = getenv("PXA_AUTO_UB_VOLTA_Q8");
    if (lv != nullptr && atoi(lv) == 0) {
        return 0;
    }
    if (level < 2 || topo.n_dev != 1 || topo.cc.empty() || topo.cc[0] != 700 || !kv_q8 || n_ctx < 49152) {
        return 0;
    }
    if (why) {
        *why = "1x V100 (sm_70), q8_0 KV, -c >= 49152: -ub 1024 keeps the Volta MMA q8 attention route "
               "(bug #207: 642.3 vs 387.5 t/s at pp65536 against -ub 2048); verified at load, stepped down "
               "while it leaves less than PXA_AUTO_UB_RESERVE_MB free";
    }
    return 1024;
}

int pxa_registry_ub_fit_rule(const pxa_topology & topo, const pxa_model_info & model, int level, int cell_ub,
                             const char ** why, bool kv_quant) {
    const char * e = getenv("PXA_AUTO_UB_FASTEST");
    if (e != nullptr && *e != '\0') {
        return 0;   // set by hand: =0 the measured cell as before, =1/=N the generic walk (common.cpp)
    }
    if (level < 2 || topo.n_dev != 1 || topo.cc.empty() || (topo.cc[0] != 600 && topo.cc[0] != 700) ||
        !pxa_is_gemma4(model.arch) || model.n_expert <= 0 || cell_ub <= 0 || cell_ub >= 1024) {
        return 0;
    }
    // 1x V100 / 1x P100, PXQN4 file, f16 KV (2026-10-17/18, ledger gemma-ub2048-v100-pxqn4): -ub 2048 beats
    // 1024 at pp16384 on both (V100 with the tensor-core expert GEMM 3985.32 vs 3680.99 t/s, +8.3%; P100 965.48 vs 946.88,
    // pp4096 1209.54 vs 1153.89), so the walk starts one rung higher there. A quantized KV cache keeps the 1024 ceiling
    // (its pp65536 cell was measured at 1024). PXA_AUTO_UB_GEMMA_2048=0 restores the 1024 ceiling.
    static const bool ub2048_on = [](){ const char * v = getenv("PXA_AUTO_UB_GEMMA_2048"); return !(v && *v && atoi(v) == 0); }();
    if (ub2048_on && !kv_quant && model.tier == "PXQN4") {
        if (why) {
            *why = topo.cc[0] == 700
                ? "Gemma 4 PXQN4 expert file on 1x V100, f16 KV: -ub 2048 down to the measured -ub 512, the largest that loads "
                  "resident and leaves PXA_AUTO_UB_RESERVE_MB free (measured 2026-10-17: pp16384 3681 -> 3985 t/s at -ub 2048 "
                  "over 1024; PXA_AUTO_UB_GEMMA_2048=0 for the 1024 ceiling)"
                : "Gemma 4 PXQN4 expert file on 1x P100, f16 KV: -ub 2048 down to the measured -ub 256, the largest that loads "
                  "resident and leaves PXA_AUTO_UB_RESERVE_MB free (measured 2026-10-18: pp4096 1154 -> 1210, pp16384 947 -> 965 t/s "
                  "at -ub 2048 over 1024; PXA_AUTO_UB_GEMMA_2048=0 for the 1024 ceiling)";
        }
        return 2048;
    }
    if (why) {
        *why = topo.cc[0] == 600
            ? "Gemma 4 expert file on 1x P100: -ub 1024 down to the measured -ub 256, the largest that loads resident and "
              "leaves PXA_AUTO_UB_RESERVE_MB free (measured 2026-10-16: server 16k prefill 322 -> 570 t/s at -ub 1024)"
            : "Gemma 4 expert file on 1x V100: -ub 1024 down to the measured -ub 512, the largest that loads resident and "
              "leaves PXA_AUTO_UB_RESERVE_MB free (measured 2026-10-16: server 16k prefill 2275 -> 2881 t/s at -ub 1024)";
    }
    return 1024;
}

int pxa_registry_ub_fit_dense(const pxa_topology & topo, const pxa_model_info & model, int level, bool kv_quant,
                              const char ** why) {
    // read at every call (not latched): the server, llama-bench and the unit test each ask once per model
    const char * v = getenv("PXA_AUTO_UB_DENSE_2048");
    if (v != nullptr && *v != '\0' && atoi(v) == 0) {
        return 0;
    }
    const char * e = getenv("PXA_AUTO_UB_FASTEST");
    if (e != nullptr && *e != '\0') {
        return 0;   // set by hand: the generic walk (or =0 the stock -ub) owns -ub
    }
    // one V100 (sm_70) or, since 2026-10-18, one P100 (sm_60): the same file, the same walk
    if (level < 2 || kv_quant || topo.n_dev != 1 || topo.cc.size() != 1 || (topo.cc[0] != 700 && topo.cc[0] != 600)) {
        return 0;
    }
    if (model.arch != "qwen35" || model.n_expert != 0 || !pxa_dense27_tier(model) ||
        (model.n_layer != 64 && model.n_layer != 65)) {   // 65 = 64 blocks + the nextn (MTP) block the shipped file carries
        return 0;
    }
    if (why) {
        *why = topo.cc[0] == 600
            ? "27B dense PXQN4 (qwen35, 64 blocks + nextn) on 1x P100, f16 KV, -ub and -b unset: -ub 2048 down to the stock -ub 512, "
              "the largest whose context allocates beside the weights and KV at the requested -c and leaves "
              "PXA_AUTO_UB_RESERVE_MB free (measured 2026-10-18, llama-bench vs v3.1 same bracket: pp4096 248.0 -> 253.4 t/s, "
              "pp16384 224.1 -> 230.8 t/s at -ub 2048); PXA_AUTO_UB_DENSE_2048=0 keeps -ub 512"
            : "27B dense PXQN4 (qwen35, 64 blocks + nextn) on 1x V100, f16 KV, -ub and -b unset: -ub 2048 down to the stock -ub 512, "
              "the largest whose context allocates beside the weights and KV at the requested -c and leaves "
              "PXA_AUTO_UB_RESERVE_MB free (measured 2026-10-18: llama-bench pp4096 986.5 -> 1127.8 t/s at -ub 2048; -ub 2048 "
              "does not allocate at -c 16384); PXA_AUTO_UB_DENSE_2048=0 keeps -ub 512";
    }
    return 2048;
}

int pxa_registry_stream_ub_decode(const pxa_topology & topo, const pxa_model_info & model, uint64_t file_bytes,
                                  uint64_t free_bytes, const char ** why) {
    const char * v = getenv("PXA_AUTO_UB_STREAM_DECODE");
    if (v != nullptr && *v != '\0' && atoi(v) == 0) {
        return 0;
    }
    if (topo.n_dev != 1 || topo.cc.size() != 1 || topo.cc[0] != 700) {
        return 0;
    }
    if (model.arch != "qwen35moe" || model.n_expert <= 0 || model.tier != "PXQN4") {
        return 0;
    }
    if (free_bytes == 0 || file_bytes <= free_bytes || 2*file_bytes > 3*free_bytes) {
        return 0;   // fits (no stream), or most experts stream (Flash-Next class: the 4096 prefill rule stands)
    }
    if (why) {
        *why = "decode-sized: qwen35moe PXQN4 on 1x V100 streams a small tail (file <= 1.5x free), the 4096 compute buffer "
               "would push ~350 more experts to the CPU path (35B-A3B decode 56.5 -> 61.8 t/s at 2048); PXA_AUTO_UB_STREAM_DECODE=0 keeps 4096";
    }
    return 2048;
}

bool pxa_registry_stream_ub_ckpt_on() {
    const char * v = getenv("PXA_STREAM_UB_CKPT");   // opt-in until a quiet A/B (r3: decode +1..+9% at N=2, NOT-QUIET)
    return v != nullptr && *v != '\0' && atoi(v) != 0;
}

int pxa_registry_stream_ub_ckpt_walk(int ub0, int cap0, int target, int floor_ub,
                                     const std::function<int(int)> & cap_at, std::string * trail) {
    if (target < 2 || cap0 >= target || ub0 <= floor_ub || !cap_at) {
        return ub0;
    }
    int  best_ub  = ub0;
    int  best_cap = cap0;
    char buf[64];
    for (int ub = ((ub0 - 1) / 512) * 512; ub >= floor_ub && ub > 0; ub -= 512) {
        const int cap = cap_at(ub);
        snprintf(buf, sizeof(buf), "-ub %d rows %d; ", ub, cap);
        if (trail) *trail += buf;
        if (cap <= 0) {
            break;   // did not allocate: a smaller rung was never the problem here, keep the best so far
        }
        if (cap > best_cap) {
            best_cap = cap;
            best_ub  = ub;
        }
        if (cap >= target) {
            break;
        }
    }
    return best_ub;
}

bool pxa_registry_xcache_smalltail(const pxa_topology & topo, const pxa_model_info & model, const char ** why) {
    const char * v = getenv("PXA_AUTO_XCACHE_SMALLTAIL");
    if (v != nullptr && *v != '\0' && atoi(v) == 0) {
        return false;
    }
    if (topo.n_dev != 1 || topo.cc.size() != 1 || topo.cc[0] != 700 || topo.vram_mib.size() != 1 || topo.vram_mib[0] == 0) {
        return false;
    }
    if (model.arch != "qwen35moe" || model.n_expert <= 0 || model.tier != "PXQN4") {
        return false;
    }
    const uint64_t vram = (uint64_t) topo.vram_mib[0] << 20;
    if (model.bytes <= vram || 2*model.bytes > 3*vram) {
        return false;   // fits (no cold experts), or most experts stream (the Flash-Next class has its own Overdrive cell)
    }
    if (why) {
        *why = "qwen35moe PXQN4 on 1x V100 overflows the card by a small tail (file <= 1.5x VRAM): async CPU-pool cold path";
    }
    return true;
}

// PXA_AUTO_FLINT_1CARD (2026-10-23; =0 off): a Flint (arch flint, 256 routed experts, top-6) PXQN4 file on ONE
// P100 that is larger than the card by up to half again (VRAM < file <= 1.5x VRAM; the 39.8B file is 20.1 GiB on 16 GB). The expert
// cache keeps the hottest experts on the card and runs each cold layer's experts on the CPU (the split path). Routing on this model is
// flat, so a cold layer is close to a whole layer on the CPU and every layer the plan keeps resident is worth ~5% of decode. On one
// V100 the C042 small-tail scope already plans with a one-layer ring and the 384 MiB slack credit; on one P100 that scope is closed in
// generic code, and this rule opens it for this file only (PXA_XCACHE_SMALLTAIL_SM60=1). Measured on one P100 (card 6, REPS 2 ABBA,
// box NOT quiet): the 384 MiB credit alone, 18 -> 17 cold layers, r16d greedy512 +4.4%, r16h (with staged inputs) +5.5%.
// Not bit-identical to the old plan (a different expert set runs on the GPU, so the greedy text changes): it ships on the gates.
// Rejected for this file (measured): the async CPU-pool cold path (impl flint-1card-async-l: V100 r19v greedy -33%, P100 r16e
// -25%), CUDA graphs on the split segments (flint-1card-cgraph: undetectable), staged split inputs (flint-1card-async-inputs:
// undetectable at N=4).
bool pxa_registry_flint_1card(const pxa_topology & topo, const pxa_model_info & model, const char ** why) {
    const char * v = getenv("PXA_AUTO_FLINT_1CARD");
    if (pxa_env_value_off(v)) {
        return false;
    }
    if (topo.n_dev != 1 || topo.cc.size() != 1 || topo.cc[0] != 600 || topo.vram_mib.size() != 1 || topo.vram_mib[0] == 0) {
        return false;   // one P100 only: one V100 is already in the generic small-tail scope
    }
    if (model.arch != "flint" || model.n_expert <= 0 || !pxa_flint_pxqn_tier(model)) {
        return false;
    }
    const uint64_t vram = (uint64_t) topo.vram_mib[0] << 20;
    if (model.bytes <= vram || 2*model.bytes > 3*vram) {
        return false;   // fits (no cold experts), or most experts are cold (unmeasured)
    }
    if (why) {
        *why = "flint PXQN4 on one P100 overflows the card by up to half again: the small-tail plan (one-layer ring, 384 MiB credit)";
    }
    return true;
}

// PXA_AUTO_FLINT_ROUTER (2026-10-23; =0 off, PXA_AUTO_FLINT_1CARD=0 too): the same Flint PXQN4 file on ONE P100 or
// V100 (VRAM < file <= 1.5x VRAM). Flint keeps its 47 routers (ffn_gate_inp) in BF16; the default path rounds the activation to
// BF16, runs the cuBLAS BF16 GEMM and widens the BF16 logits back (three kernels per layer). PXA_ROUTER_BF16=2 runs the router GEMV
// on the BF16 row widened to F32 (one kernel, x and the logits stay F32). P100 card 6 nvprof (64 decode tokens): 20.7 us/call
// (gemv2T 16.8 + converts 3.9) -> 8.9 us, 0.97 -> 0.42 ms per token and 94 fewer launches; llama-bench tg128 43.83 vs 42.37
// (r16z, REPS 2 ABBA, NOT quiet). Not bit-identical (F32 logits instead of BF16-rounded ones: the top-6 can change on near ties);
// it ships on the gates. PXA_ROUTER_BF16=0 in the environment keeps the cuBLAS path (setenv never overwrites).
bool pxa_registry_flint_1card_router(const pxa_topology & topo, const pxa_model_info & model, const char ** why) {
    for (const char * name : {"PXA_AUTO_FLINT_1CARD", "PXA_AUTO_FLINT_ROUTER"}) {
        const char * v = getenv(name);
        if (pxa_env_value_off(v)) {
            return false;
        }
    }
    if (topo.n_dev != 1 || topo.cc.size() != 1 || (topo.cc[0] != 600 && topo.cc[0] != 700) || topo.vram_mib.size() != 1 ||
            topo.vram_mib[0] == 0) {
        return false;
    }
    if (model.arch != "flint" || model.n_expert <= 0 || !pxa_flint_pxqn_tier(model)) {
        return false;
    }
    const uint64_t vram = (uint64_t) topo.vram_mib[0] << 20;
    if (model.bytes <= vram || 2*model.bytes > 3*vram) {
        return false;
    }
    if (why) {
        *why = "flint PXQN4 on one P100/V100 overflows the card by up to half again: BF16 router GEMV on an F32 widening (one kernel)";
    }
    return true;
}

// PXA_AUTO_FLINT_V100 (2026-10-23; =0 off, PXA_AUTO_FLINT_1CARD=0 too): the same Flint PXQN4 file on ONE V100
// (VRAM < file <= 1.5x VRAM). Decode there runs 17 GPU segments a token between the cold layers' CPU splits, ~70 kernels each; a V100
// runs those kernels about twice as fast as a P100, so the segments become launch-bound (P100 profile: 1228 kernels/token, ~10 us
// each, launches hidden; the same count at ~5 us is not). The rule arms the one-V100 plain-decode stack, every lever bit-identical:
// PXA_CUDA_GRAPH_DECODE (decode segments captured after two property-identical evals; the arena-generation recapture of aee131b73d
// covers the re-reserve that broke warm slots before), PXA_CUDA_GRAPH_MOE (the one-token MUL_MAT_ID stays in the capture),
// PXA_CUDA_GRAPH_LRU=256 (17 segments x shapes: 64 slots evicted 194 times on the P100 probe), PXA_SCHED_ASYNC_INPUTS (split inputs
// staged on the stream, no host wait), PXA_PLAIN_GREEDY_AMAX + PXA_ARGMAX_VAL_MB (a greedy step reads back one id). P100 probe r16s
// (card 6, REPS 1): greedy512 shas identical to the default on all three classes, 148 captures / 25848 replays, no warm-slot fault.
// Each lever's own =0 in the environment wins (setenv never overwrites).
bool pxa_registry_flint_1card_v100(const pxa_topology & topo, const pxa_model_info & model, const char ** why) {
    for (const char * name : {"PXA_AUTO_FLINT_1CARD", "PXA_AUTO_FLINT_V100"}) {
        const char * v = getenv(name);
        if (pxa_env_value_off(v)) {
            return false;
        }
    }
    if (topo.n_dev != 1 || topo.cc.size() != 1 || topo.cc[0] != 700 || topo.vram_mib.size() != 1 || topo.vram_mib[0] == 0) {
        return false;   // one V100 only: on one P100 the split segments are not launch-bound (flint-1card-cgraph: undetectable)
    }
    if (model.arch != "flint" || model.n_expert <= 0 || !pxa_flint_pxqn_tier(model)) {
        return false;
    }
    const uint64_t vram = (uint64_t) topo.vram_mib[0] << 20;
    if (model.bytes <= vram || 2*model.bytes > 3*vram) {
        return false;
    }
    if (why) {
        *why = "flint PXQN4 on one V100 overflows the card by up to half again: the plain-decode stack (graphs on the split segments)";
    }
    return true;
}

size_t pxa_registry_ub_reserve_bytes() {
    const char * e  = getenv("PXA_AUTO_UB_RESERVE_MB");
    const long   mb = e != nullptr && *e ? atol(e) : 512;
    return (size_t) (mb > 0 ? mb : 0) * 1024ull * 1024ull;
}

int pxa_registry_ub_next_rung(int ub) {
    static const int ladder[] = { 2048, 1024, 768, 512, 256 };
    for (int r : ladder) {
        if (r < ub) {
            return r;
        }
    }
    return 0;
}

bool pxa_registry_spec_stream_off(const pxa_spec_stream_in & in, std::string * why) {
    // PXA_GLM5NEXT_MTP auto scope (2026-10-18): glm5next (GLM-5.3-Flash) arms AUTO speculation only when its
    // routed experts are resident on the cards. With -ot ...=CPU expert placement or streamed experts (the Overdrive /
    // host-expert layouts) a 2-token verify reads ~2x the routed expert bytes on the host (the Nemotron-H measurement
    // below: -10%), so the auto layer stands down there. PXA_GLM5NEXT_MTP=1 lets it arm anywhere, =0 keeps it off everywhere.
    // An explicit --spec-type mtp still gets the head (only PXA_GLM5NEXT_MTP=0 refuses it, in the graph build).
    if (in.arch == "glm5next" && in.has_head) {
        const char * g = getenv("PXA_GLM5NEXT_MTP");
        const int gv = (g != nullptr && *g) ? atoi(g) : -1;
        if (gv > 0) return false;
        const bool host_experts = in.cpu_override || in.stream_forced || (!in.fits && (in.xcache_off || !in.xcache_mtp));
        if (gv != 0 && !host_experts) return false;
        if (why != nullptr) {
            *why = gv == 0 ? std::string("glm5next MTP head: PXA_GLM5NEXT_MTP=0, no speculation")
                           : std::string("glm5next with routed experts on the host (") +
                             (in.cpu_override ? "-ot CPU expert placement" : in.stream_forced ? "PXA_STREAM_WEIGHTS" : "file larger than the cards") +
                             "): a 2-token verify reads ~2x the host expert bytes, so no auto speculation "
                             "(PXA_GLM5NEXT_MTP=1 arms it; --spec-type mtp asks for one)";
        }
        return true;
    }
    const char * e = getenv("PXA_SPEC_STREAM_MTP");
    if (e != nullptr && *e && pxa_env_value_off(e)) return false;
    if (in.arch != "nemotron_h_moe" || !in.has_head || in.cpu_override) return false;
    const bool streamed = in.stream_forced || (!in.fits && (in.xcache_off || !in.xcache_mtp));
    if (!streamed) return false;
    if (why != nullptr) {
        *why = std::string("Nemotron-H MoE with an MTP head and experts streamed from host (") +
               (in.stream_forced ? "PXA_STREAM_WEIGHTS" : in.xcache_off ? "PXA_XCACHE=0, file larger than the cards"
                                                        : "file larger than the cards, expert cache declines with the head loaded") +
               "): MTP measured -10% here, so no auto speculation (PXA_SPEC_STREAM_MTP=0 restores it; --spec-type asks for one)";
    }
    return true;
}

bool pxa_registry_spec_head_present(int nextn_in_header, const char * head_file) {
    return nextn_in_header > 0 || (head_file != nullptr && *head_file);
}

int pxa_registry_ub_walk(int ub, int origin, size_t reserve,
                         const std::function<pxa_ub_attempt(int)> & build,
                         const std::function<void()> & discard, std::string * trail, int floor_ub) {
    char buf[160];
    for (bool first = true; ub > 0; first = false) {
        // floor_ub > 0: the floor is the last rung and is built exactly as without the walk (no reserve check)
        const bool last = floor_ub > 0 && ub <= floor_ub;
        const pxa_ub_attempt a = build(ub);
        if (a.allocated) {
            // the ladder's own first pick is kept exactly as before; everything else must leave the reserve
            const bool check = reserve > 0 && !last && !(origin == PXA_UB_LADDER && first);
            if (!check || a.free_min >= reserve) {
                return ub;
            }
            snprintf(buf, sizeof(buf), "-ub %d left %zu MiB free on CUDA%d; ", ub, a.free_min >> 20, a.worst_dev);
            if (trail) *trail += buf;
            discard();
        } else {
            if (!a.alloc_failure || last) {
                return 0;   // refused for a reason a smaller -ub cannot fix (or the floor itself failed, as it would have)
            }
            snprintf(buf, sizeof(buf), "-ub %d did not allocate; ", ub);
            if (trail) *trail += buf;
        }
        if (last) {
            return 0;
        }
        ub = pxa_registry_ub_next_rung(ub);
        if (floor_ub > 0 && ub < floor_ub) {
            ub = floor_ub;
        }
    }
    return 0;
}

bool pxa_registry_cuda_free_min(const std::string & devices, int only_dev, size_t * free_min, int * worst_dev) {
#if defined(GGML_USE_CUDA)
    const int n = ggml_backend_cuda_get_device_count();
    std::vector<int> ids;
    if (only_dev >= 0) {
        if (only_dev < n) {
            ids.push_back(only_dev);
        }
    } else {
        if (!devices.empty()) {
            std::stringstream ss(devices);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (item.rfind("CUDA", 0) == 0) {
                    const int id = atoi(item.c_str() + 4);
                    if (id >= 0 && id < n) {
                        ids.push_back(id);
                    }
                }
            }
        }
        if (ids.empty()) {
            for (int i = 0; i < n; ++i) {
                ids.push_back(i);
            }
        }
    }
    if (ids.empty()) {
        return false;
    }
    size_t lo = SIZE_MAX;
    int    wd = -1;
    for (int id : ids) {
        size_t f = 0, t = 0;
        ggml_backend_cuda_get_device_memory(id, &f, &t);
        if (f < lo) {
            lo = f;
            wd = id;
        }
    }
    if (free_min)  *free_min  = lo;
    if (worst_dev) *worst_dev = wd;
    return true;
#else
    (void) devices;
    (void) only_dev;
    (void) free_min;
    (void) worst_dev;
    return false;
#endif
}

static bool pxa_env_off(const char * name) {
    const char * v = getenv(name);
    return v && v[0] == '0';
}

// Past a pair the tensor split is a DEFAULT only where it was measured faster (2026-09-27):
// 4 identical P100s (sm_60), dense 27B, one binary,
// layer/tensor/layer -> tensor decode +4.6..5.6% (PXQN4) / +38.5..41.1% (PXQ4), prefill @22.6k +95%. Bug #206 (4-way garbage) is
// fixed underneath it (a failed NCCL group demotes to the peer route). 3 cards, 5+ cards and 4x
// V100 have no measurement and keep the layer split; PXA_TSPLIT_ALLOW_4WAY=0 (the engine's own
// >2-device guard) turns this off as well, so the one switch restores the pair-only default.
static bool pxa_tsplit_auto_quad(const pxa_topology & topo) {
    return topo.n_dev == 4 && topo.same_cc() && !topo.cc.empty() && topo.cc[0] == 600 &&
           !pxa_env_off("PXA_TSPLIT_ALLOW_4WAY");
}

static pxa_autoconfig pxa_autoconfig_resolve_picks(const pxa_topology & topo, const pxa_model_info & model,
                                                   const pxa_user_set & user, int level, int posture) {
    pxa_autoconfig ac;
    char buf[512];
    bool dense_ub_v100 = false;   // the -ub pick is the 27B dense one-V100 FIT ceiling (stands down on a quantized KV pick)

    // ---- -fa ---------------------------------------------------------------------------------
    {
        pxa_pick p;
        p.flag = "-fa";
        if (user.fa) {
            p.value = user.fa_value ? "on" : "off"; p.status = "USER"; p.why = "given on the command line";
        } else if (level == 0) {
            p.value = user.fa_value ? "on" : "off"; p.status = "OFF"; p.why = "PXA_REFERENCE=1: engine default kept";
        } else {
            bool all61 = topo.n_dev > 0;
            for (int c : topo.cc) {
                all61 = all61 && c == 610;
            }
            bool on = posture == 0;
            p.why = on ? "PXA_MODE=balance (the serving posture): flash attention is the decode win on P100/V100/1080 Ti"
                       : "PXA_MODE=max (the ingest posture): flash attention off is the cold-prefill win";
            if (model.arch == "deepseek2") {
                // the server's MLA exceptions (examples/server/server.cpp posture block), mirrored
                on = !all61;
                p.why = all61 ? "deepseek2 (MLA) on sm_61: fa off (fp16-starved FA path, measured 75-326% slower decode with fa on)"
                              : "deepseek2 (MLA): fa on in either posture (fa off degrades catastrophically with context)";
            }
            ac.flash_attn = on ? 1 : 0;
            p.value = on ? "on" : "off"; p.status = "RULE"; p.applied = true;
        }
        ac.picks.push_back(p);
    }
    const bool fa_on = user.fa ? user.fa_value : (ac.flash_attn >= 0 ? ac.flash_attn == 1 : user.fa_value);

    // ---- -sm ---------------------------------------------------------------------------------
    {
        pxa_pick p;
        p.flag = "-sm";
        p.evidence = "";
        const int n = topo.n_dev;
        bool tensor = false;
        if (user.sm) {
            p.value = ""; p.status = "USER"; p.why = "given on the command line"; p.evidence = "";
        } else if (level < 2 || pxa_env_off("PXA_AUTO_SM")) {
            p.value = "layer"; p.status = "OFF";
            p.why = level < 2 ? "config level below ENHANCE: the engine default (layer) is kept"
                              : "PXA_AUTO_SM=0: the engine default (layer) is kept";
        } else if (n < 2) {
            p.value = "layer"; p.status = "RULE";
            snprintf(buf, sizeof(buf), "%d card: a tensor split needs two", n); p.why = buf;
        } else if (n > 2 && !pxa_tsplit_auto_quad(topo)) {
            p.value = "layer"; p.status = "RULE"; p.evidence = "";
            if (pxa_env_off("PXA_TSPLIT_ALLOW_4WAY")) {
                p.why = "PXA_TSPLIT_ALLOW_4WAY=0: the tensor split stays on pairs (the pre-2026-09-27 bug #206 guard)";
            } else {
                snprintf(buf, sizeof(buf), "%d cards: past a pair the tensor split is a default only on 4 identical P100s, "
                                           "the one set it was measured faster on", n);
                p.why = buf;
            }
        } else if (topo.narrow_link() >= 0) {
            // 2026-09-27 (bug #280): x1/x2 risers. The split reduces every layer's output across the
            // cards each step; on a narrow link that traffic is the whole cost.
            p.value = "layer"; p.status = "RULE"; p.evidence = "";
            snprintf(buf, sizeof(buf), "card %d is on a PCIe x%d link (narrow): the tensor split reduces across the cards "
                                       "every step, which a narrow link cannot carry; layer split", topo.narrow_link(),
                     topo.pcie_width[topo.narrow_link()]);
            p.why = buf;
        } else if (!topo.same_cc()) {
            p.value = "layer"; p.status = "RULE";
            p.why = "the two cards are not the same class: an even tensor split runs at the slower card's pace every step";
        } else if (pxa_is_gemma4(model.arch)) {
            p.value = "layer"; p.status = "RULE";
            p.why = "Gemma 4: the tensor split is opt-in (PXA_TSPLIT_GEMMA4=1); faster on a P100 pair, slower on a V100 pair";
        } else if (model.arch == "qwen4exp" && n == 2 && topo.cc[0] == 600 && !pxa_env_off("PXA_AUTO_SM_FN") &&
                   !user.ts && !user.ngl_partial && fa_on) {
            // Flash-Next (qwen4exp) on a P100 PAIR (2026-10-15):
            // fn32-g6-pleq5, cards 0+1, both arms +PXA_QWEN4EXP_HC_FUSED, REPS 3 x 256:
            // MTP n3 tensor 40.94/47.29 vs layer 38.11/44.88 (+7.4/+5.4%), plain 35.82/36.00 vs 32.75/32.74
            // (+9.4/+10.0%); 10-14 (no HCF) +8.4/+5.3% MTP, +9.9/+9.2% plain. A V100 pair is NOT included:
            // there MTP is a tie (63.26/73.29 vs 63.31/74.37) and plain only +3.3%, because the FN GEMVs are
            // latency-bound on sm_70 at n_embd 2560 (24 us/launch split or not). PXA_AUTO_SM_FN=0 keeps layer.
            tensor = true;
            p.value = "tensor"; p.status = "MEASURED"; p.evidence = "";
            p.why = "2x P100 (sm_60), Flash-Next (qwen4exp): tensor split +7.4/+5.4% MTP, +9.4/+10.0% plain decode over the "
                    "layer split (2026-10-15); -sm layer or PXA_AUTO_SM_FN=0 is the way back";
            ac.env.emplace_back("PXA_TSPLIT_QWEN4EXP_HC", "1");
            ac.env.emplace_back("PXA_TSPLIT_UNPROVEN_ARCH", "1");
            ac.env.emplace_back("PXA_TSPLIT_QWEN4EXP_PAIRLOCAL", "1");
            ac.env.emplace_back("PXA_TSPLIT_WORKERS", "1");
            ac.env.emplace_back("PXA_TSPLIT_REPLAY", "1");
            ac.env.emplace_back("PXA_QWEN4EXP_HC_FUSED", "1");
        } else if (!pxa_tsplit_auto_arch(model.arch)) {
            p.value = "layer"; p.status = "RULE";
            snprintf(buf, sizeof(buf), "arch '%s' has no tensor-split measurement behind a default (qwen35 does)",
                     model.arch.empty() ? "?" : model.arch.c_str());
            p.why = buf;
        } else if (!pxa_tsplit_auto_tier(model.tier) && !pxa_tsplit_onecard_pair(model, topo)) {
            p.value = "layer"; p.status = "RULE";
            snprintf(buf, sizeof(buf), "tier %s: the split is a default only where it was measured (PXQ4, PXQN4, PXQN4S8, PXQN5)",
                     model.tier.empty() ? "none" : model.tier.c_str());
            p.why = buf;
        } else if (model.n_head_kv > 0 && model.n_head_kv < n) {
            p.value = "layer"; p.status = "RULE";
            snprintf(buf, sizeof(buf), "%d KV heads on %d cards: attention cannot divide that many ways", model.n_head_kv, n);
            p.why = buf;
        } else if (!fa_on) {
            p.value = "layer"; p.status = "RULE";
            p.why = "-fa off: the split attention builder has no non-FA path, and forcing FA back on would undo the posture";
        } else if (user.ts) {
            p.value = "layer"; p.status = "RULE";
            p.why = "-ts was given: a capacity split over layers; the tensor split would throw it away";
        } else if (user.ngl_partial) {
            p.value = "layer"; p.status = "RULE";
            p.why = "-ngl offloads only part of the model: the tensor split needs every layer on the cards";
        } else if (n == 4) {
            tensor = true;
            p.value = "tensor"; p.status = "MEASURED"; p.evidence = "";
            snprintf(buf, sizeof(buf), "4x %s, arch %s, tier %s: decode +5..41%% and prefill about 2x over the layer "
                                       "split (4x P100, 27B PXQN4/PXQ4, 2026-09-27); -sm layer is the one flag back",
                     pxa_card_name(topo.cc[0]), model.arch.c_str(), model.tier.c_str());
            p.why = buf;
        } else {
            tensor = true;
            p.value = "tensor"; p.status = "MEASURED";
            snprintf(buf, sizeof(buf), "2x %s, arch %s, tier %s: decode is faster at every measured context "
                                       "(prefill is slower on narrow PCIe links); -sm layer is the one flag back",
                     pxa_card_name(topo.cc[0]), model.arch.c_str(), model.tier.c_str());
            p.why = buf;
        }
        if (!user.sm && p.status != "OFF") {
            ac.split   = tensor ? PXA_SM_TENSOR : PXA_SM_LAYER;
            p.applied  = true;
        }
        if (tensor) {
            ac.ts_even = true;
            ac.env.emplace_back("PXA_TSPLIT_REDUCE", "fused");
            // PXA_TSPLIT_REDUCE_PREFILL is NOT emitted any more: the fused route is checked before the
            // two-device prefill DMA route (PXA_TSPLIT_PF, default on), so =1 pre-empted it and cost
            // prefill: V100 pair, dense 27B PXQN4, llama-server, 14,800-token prompt, PF 822.6/824.8
            // vs fused-prefill 738.8/747.9 t/s, greedy output identical, decode unchanged.
            // the split was CHOSEN, not asked for: a refusal at load demotes to layer instead of stopping
            ac.env.emplace_back("PXA_TSPLIT_FALLBACK", "1");
        }
        ac.picks.push_back(p);
    }

    // ---- the P60 verify stack: a Qwen3.6-35B-A3B-class MoE (qwen35moe, PXQN4) on a layer-split P100 pair ---------------
    // (2026-10-16)
    // Eleven levers that are OFF in the engine and ON in the board recipe: the persistent single-token layer halves (PXA_P60_MEGA, and
    // for a speculative verify the 2..6-token PXQN4 MoE half PXA_P60_MEGA_PXQN_T), the verify-width GEMVs (PXA_KQMMV_VW, PXA_PXQN_VW,
    // PXA_P60_PRE_VW) and the decode-width PXQN4 kernels (PXA_PXQN_RBP, PXA_PXQN_RMS_RHT_GB, PXA_ARGMAX_VAL_MB, PXA_PXQN_QPN) around the
    // delta-net conv cluster (PXA_DN_CONVFUSE). Same binary, same file, 2x P100 -sm layer, ON against OFF: plain decode 83.1 -> 133.1 t/s
    // (+60%), MTP 99.6 -> 138.1 (+38.7%) [board b3, ledger #7407 / #7409]. The kernels match the graph patterns of this architecture and
    // decline cleanly anywhere else, so the cell is where they were measured: arch qwen35moe, tier PXQN4, layer split, two or more cards
    // that are ALL sm_60. The kernels run per device on each card's own layers, so a layer split over four P100s runs the same code as
    // the pair; until 2026-10-20 the rule said "exactly two cards" and the 4x P100 cell ran the stack-off path (73 t/s, behind the rival's
    // -sm tensor 77.4): quick test 2026-10-20 (rc2.1 engine ee2245ced1, cards 0,1,5,6, greedy512 REPS 2, NOT-QUIET) the same
    // binary default 72.2 / 73.1 / 73.1 t/s (prose / code / chat) -> the 11 levers by env 124.6 / 125.6 / 125.0 (+71%), coherent text.
    // PXA_AUTO_P60=0 stands the whole pick down; any one lever named in the environment is kept.
    // DEFAULT ON (kP60DefaultOn). GATE VERDICT (2026-10-16, pass-3 engine, 35B-A3B MTP PXQN4, 2x P100, same binary, stack OFF vs ON, wiki400k -c 512 x 10 chunks):
    // prefill path bit-identical (KLD 0, same top-1 100%); decode width (-ub 1) mean KLD 2.5e-3 +- 0.2e-3, same top-1 98.35%, PPL +0.12% +- 0.16% (nil); verify width 2 (-ub 2) 2.4e-3, 98.16%;
    // determinism 12/12, needle 28k pass. So the stack is NOT bit-identical to OFF at decode / verify widths (K-chunk and fp16 staging order) and misses the 1e-4 / 99.5% bar the plan proposed.
    // If bit-identity by default is mandatory, set kP60DefaultOn = false: the pick then needs PXA_AUTO_P60=1 and presets/pxa-p100-moe35b-mtp.env carries the same levers for a seat that opts in.
    static const bool kP60DefaultOn = true;
    const char * p60_env = getenv("PXA_AUTO_P60");
    const bool p60_on = p60_env && *p60_env ? atoi(p60_env) != 0 : kP60DefaultOn;
    const bool p60_all_sm60 = topo.n_dev >= 2 && topo.cc.size() == (size_t) topo.n_dev &&
                              std::all_of(topo.cc.begin(), topo.cc.end(), [](int c) { return c == 600; });
    const bool p60_cell = level >= 2 && model.arch == "qwen35moe" && model.tier == "PXQN4" && p60_all_sm60 &&
                          (user.sm ? user.sm_value == PXA_SM_LAYER : ac.split == PXA_SM_LAYER) && p60_on;
    // measured on 2 and 4 cards; 3 or 5+ all-sm_60 cards take the same per-device kernels on the same layer split (INFERRED)
    const bool p60_measured_n = topo.n_dev == 2 || topo.n_dev == 4;

    // ---- the one-V100 qwen35 plain-decode stack (2026-10-16) ----
    // Four bit-identical levers: PXA_DN_CONVFUSE (fused delta-net conv cluster), PXA_DN_INPLACE (state carry in place),
    // PXA_ARGMAX_VAL_MB (many-block argmax_val) and PXA_PLAIN_GREEDY_AMAX (device argmax on a plain greedy step).
    // 27B PXQN4 on card 4 (V100-PCIE-16GB), same binary, server ABBA REPS 3 greedy512: prose 41.30 -> 42.63 (+3.2%),
    // code 40.97 -> 42.29 (+3.2%), greedy sha identical 6/6. The 2026-09-29 -31% conv-fusion row above predates the
    // conv-cluster fixes. Applies with no tensor split, exactly one sm_70 card, arch qwen35. PXA_AUTO_V70=0 stands the
    // pick down; each lever's own =0 in the environment wins. src/llama.cpp carries the same rule for the bench/cli.
    // 2026-10-19: the same six on the 2x V100 TENSOR pair. 27B PXQN4 mtpw3 on cards
    // 2+4 -sm tensor, server ABBA 2 boots x REPS 2 greedy512 (NOT-QUIET load 38-44, lever-ab g04-v70six-v24): prose 71.14 -> 76.21
    // (+7.1%), code 97.16 -> 102.23 (+5.2%), chat 68.86 -> 72.34 (+5.1%), greedy sha identical on all three, MTP acceptance
    // unchanged. Scoped to exactly what was measured (two sm_70 cards, tensor split, qwen35); -sm layer pairs stay as they were.
    const char * v70_env = getenv("PXA_AUTO_V70");
    const bool v70_tensor = user.sm ? user.sm_value == PXA_SM_TENSOR : ac.split == PXA_SM_TENSOR;
    const bool v70_one  = topo.n_dev == 1 && topo.cc.size() == 1 && topo.cc[0] == 700 && !v70_tensor;
    const bool v70_pair = topo.n_dev == 2 && topo.cc.size() == 2 && topo.cc[0] == 700 && topo.cc[1] == 700 && v70_tensor;
    const bool v70_cell = level >= 2 && model.arch == "qwen35" && (v70_one || v70_pair) &&
                          (v70_env && *v70_env ? atoi(v70_env) != 0 : true);
    if (v70_cell) {
        // PXA_PXQN_QPN=1 is NOT in this stack: re-measured 2026-10-18 (card 2, 27B PXQN4 mtpw3, bare
        // table MTP, greedy512 2x2 prompts): 84.29 -> 81.11 t/s (-3.8%), sha 4/4 equal -- the 10-14 +4% was on
        // the older depth-3 binary. It stays opt-in.
        static const char * const v70[] = { "PXA_DN_CONVFUSE", "PXA_DN_INPLACE", "PXA_ARGMAX_VAL_MB", "PXA_PLAIN_GREEDY_AMAX",
                                            "PXA_DN_CF_DIRECT", "PXA_SCHED_ASYNC_INPUTS" };
        for (const char * n : v70) {
            ac.env.emplace_back(n, "1");
        }
        pxa_pick p;
        p.flag = "PXA_AUTO_V70";
        // two plain assignments, not a ternary: scripts/pxa-release-scrub.py empties only `evidence = ""` (a release build
        // carries no record ids; test-pxa-release-strings caught the ternary)
        if (v70_pair) { p.evidence = ""; } else { p.evidence = ""; }
        p.value = "on"; p.status = "MEASURED"; p.applied = true;
        p.why = v70_pair
            ? "qwen35 on the 2x V100 tensor pair: the six one-V100 decode levers, +5.1..7.1% MTP server decode (27B mtpw3), "
              "greedy sha identical; PXA_AUTO_V70=0 or any one lever =0 in the environment overrides"
            : "qwen35 on one V100: CONVFUSE + INPLACE + ARGMAX_VAL_MB + PLAIN_GREEDY_AMAX (+3.2% server decode), DN_CF_DIRECT (+1.2%), "
              "SCHED_ASYNC_INPUTS (+0.6%), greedy sha identical; "
              "PXA_AUTO_V70=0 or any one lever =0 in the environment overrides";
        ac.picks.push_back(p);
    }

    // ---- the one-P100 qwen35 PXQN plain-decode additions (2026-10-18, owner DEFAULTS rule) ----
    // Three bit-identical levers on top of the one-P100 K3 cell below (RMS_RHT_GB + ARGMAX_VAL_MB + RBP):
    // PXA_DN_CONVFUSE (the fused delta-net conv cluster; the -21% one-P100 row of 2026-09-29 predates the host g_dead /
    // consumer index): 1x P100 card 6, 27B PXQN4, server ABBA REPS 3, GB+ARGMAX+CONVFUSE
    // 25.47 -> 26.40 t/s vs default, CONVFUSE ~+2.5% of it, every response sha identical), PXA_DN_CF_DIRECT (the conv
    // cluster reads its scratch directly, no copies) and PXA_PLAIN_GREEDY_AMAX (a plain greedy step reads back one id, not
    // the logits row) -- both bit-identical and already default on one V100, #22176.
    // Exactly one sm_60 card, arch qwen35, a PXQN tier, no tensor split, level 2. PXA_AUTO_P1=0 stands the three down;
    // each lever's own value in the environment wins (setenv never overwrites).
    const char * p1_env = getenv("PXA_AUTO_P1");
    const bool p1q35_cell = level >= 2 && model.arch == "qwen35" && pxa_dense27_pxqn_family(model) && topo.n_dev == 1 &&
                            topo.cc.size() == 1 && topo.cc[0] == 600 &&
                            !(user.sm ? user.sm_value == PXA_SM_TENSOR : ac.split == PXA_SM_TENSOR) &&
                            (p1_env && *p1_env ? atoi(p1_env) != 0 : true);
    if (p1q35_cell) {
        static const char * const p1[] = { "PXA_DN_CONVFUSE", "PXA_DN_CF_DIRECT", "PXA_PLAIN_GREEDY_AMAX" };
        for (const char * n : p1) {
            ac.env.emplace_back(n, "1");
        }
        pxa_pick p;
        p.flag = "PXA_AUTO_P1";
        p.evidence = "";
        p.value = "on"; p.status = "MEASURED"; p.applied = true;
        p.why = "qwen35 PXQN on one P100: DN_CONVFUSE (~+2.5% server decode, sha identical) + DN_CF_DIRECT + PLAIN_GREEDY_AMAX "
                "(bit-identical); PXA_AUTO_P1=0 or any one lever =0 in the environment overrides";
        ac.picks.push_back(p);
    }

    // ---- the dense 27B PXQN4 on the P100 LAYER pair: the one-P100 decode stack on both cards (2026-10-21) ----
    // A -sm layer pair runs every layer whole on one card, so its decode launches have exactly the one-card shapes; but every
    // one-P100 pick above is scoped to n_dev == 1 and the P2T pick to the tensor split, so the layer pair ran with none of them
    // (and with PXA_DN_CONVFUSE=0 from the no-tensor-split rule below). Board cell G10: the same 27B file on one P100 27.8 t/s,
    // on the layer pair 25.3 (v3.1 25.3 on both), i.e. the pair lost exactly the one-card stack. This cell takes the same six,
    // each bit-identical by construction: PXA_PXQN_RBP (persistent row-block one-column GEMV), PXA_PXQN_RMS_RHT_GB (8-block fused
    // norm + RHT site), PXA_ARGMAX_VAL_MB, PXA_DN_CONVFUSE (+ PXA_DN_CF_DIRECT), PXA_PLAIN_GREEDY_AMAX. With RBP on, the
    // the 27B fusions that ride on the persistent row-block launch (PXA_PXQN_GU_RHT and friends, default on) engage too.
    // Scope = what was measured: two sm_60 cards, layer split, arch qwen35, tier PXQN4, level 2. PXA_AUTO_P2L=0 stands the
    // pick down; each lever's own value in the environment wins (setenv never overwrites).
    const char * p2l_env = getenv("PXA_AUTO_P2L");
    const bool p2l_cell = level >= 2 && model.arch == "qwen35" && pxa_dense27_tier(model) && topo.n_dev == 2 &&
                          topo.cc.size() == 2 && topo.cc[0] == 600 && topo.cc[1] == 600 &&
                          (user.sm ? user.sm_value == PXA_SM_LAYER : ac.split == PXA_SM_LAYER) &&
                          (p2l_env && *p2l_env ? atoi(p2l_env) != 0 : true);
    if (p2l_cell) {
        static const char * const p2l[] = { "PXA_PXQN_RBP", "PXA_PXQN_RMS_RHT_GB", "PXA_ARGMAX_VAL_MB", "PXA_DN_CONVFUSE",
                                            "PXA_DN_CF_DIRECT", "PXA_PLAIN_GREEDY_AMAX" };
        for (const char * n : p2l) {
            ac.env.emplace_back(n, "1");
        }
        pxa_pick p;
        p.flag = "PXA_AUTO_P2L";
        p.evidence = "";
        p.value = "on"; p.status = "MEASURED"; p.applied = true;
        p.why = "dense 27B PXQN4 on the P100 layer pair: the one-P100 decode stack (RBP + RMS_RHT_GB + ARGMAX_VAL_MB + DN_CONVFUSE + "
                "DN_CF_DIRECT + PLAIN_GREEDY_AMAX) on both cards, all bit-identical; PXA_AUTO_P2L=0 or any of the six =0 in the "
                "environment overrides";
        ac.picks.push_back(p);
    }

    // ---- the one-V100 Gemma 4 MoE plain-decode stack (2026-10-17, ledger gemma-v100-decode-stack) ----
    // Four bit-identical levers: PXA_CUDA_GRAPH_MOE (the one-token MoE step is captured as a CUDA graph; its expert ids are
    // read on the device), PXA_ARGMAX_VAL_MB (many-block argmax_val over the 262k vocab), PXA_PLAIN_GREEDY_AMAX (a plain
    // greedy step reads back one id, not the logits row) and PXA_SCHED_ASYNC_INPUTS (split inputs without a host wait).
    // Gemma 4 26B-A4B PXQN4 on card 2 (V100-PCIE-16GB), quick REPS 2 greedy512 code, one bracket: eager 113.00 ->
    // 120.13 tok/s (+6.3%), v3.1 115.33 / 117.03, greedy sha 5a23668f identical on every arm (v3.1 too). Exactly one sm_70
    // card, arch gemma4 with experts, no tensor split. PXA_AUTO_V70_G4=0 stands the pick down; each lever's own =0 wins.
    // src/llama.cpp carries the same rule for the bench/cli.
    const char * v70g4_env = getenv("PXA_AUTO_V70_G4");
    const bool v70g4_cell = level >= 2 && model.arch == "gemma4" && model.n_expert > 0 && topo.n_dev == 1 &&
                            topo.cc.size() == 1 && topo.cc[0] == 700 &&
                            !(user.sm ? user.sm_value == PXA_SM_TENSOR : ac.split == PXA_SM_TENSOR) &&
                            (v70g4_env && *v70g4_env ? atoi(v70g4_env) != 0 : true);
    if (v70g4_cell) {
        static const char * const v70g4[] = { "PXA_CUDA_GRAPH_MOE", "PXA_ARGMAX_VAL_MB", "PXA_PLAIN_GREEDY_AMAX",
                                              "PXA_SCHED_ASYNC_INPUTS", "PXA_G2_NORMFUSE", "PXA_NORM_ADD_SCALE" };
        for (const char * n : v70g4) {
            ac.env.emplace_back(n, "1");
        }
        pxa_pick p;
        p.flag = "PXA_AUTO_V70_G4";
        p.evidence = "";
        p.value = "on"; p.status = "MEASURED"; p.applied = true;
        p.why = "Gemma 4 MoE on one V100: CUDA_GRAPH_MOE + ARGMAX_VAL_MB + PLAIN_GREEDY_AMAX + SCHED_ASYNC_INPUTS, "
                "26B-A4B server decode 113.0 -> 120.1 tok/s (+6.3%, v3.1 117.0), greedy sha identical; "
                "PXA_AUTO_V70_G4=0 or any one lever =0 in the environment overrides";
        ac.picks.push_back(p);
    }

    // ---- the one-V100 llama plain-decode stack (2026-10-19) ----
    // ----
    // Bit-identical levers: PXA_PLAIN_GREEDY_AMAX + PXA_ARGMAX_VAL_MB (a plain greedy step reads back the head's argmax, not the
    // 513 KB logits row), PXA_SAMPLED_TOPW (a plain sampled step reads back the top-64 window its top-k sampler draws from) and
    // PXA_ROPE_KV_FUSE=2 (Q rope + K rope + both f16 KV-cache stores in one launch). Nemotron-Nano-8B PXQN4 on card 2,
    // same binary, greedy512 shas identical (each alone 9/9 and the stack). Exactly one sm_70 card, arch llama, no tensor
    // split, level 2. PXA_AUTO_V70_LLAMA=0 stands the pick down; each lever's own value in the environment wins.
    const char * v70l_env = getenv("PXA_AUTO_V70_LLAMA");
    const bool v70l_cell = level >= 2 && model.arch == "llama" && topo.n_dev == 1 && topo.cc.size() == 1 && topo.cc[0] == 700 &&
                           !(user.sm ? user.sm_value == PXA_SM_TENSOR : ac.split == PXA_SM_TENSOR) &&
                           (v70l_env && *v70l_env ? atoi(v70l_env) != 0 : true);
    if (v70l_cell) {
        static const char * const v70l[][2] = { { "PXA_PLAIN_GREEDY_AMAX", "1" }, { "PXA_ARGMAX_VAL_MB", "1" },
                                                { "PXA_SAMPLED_TOPW", "1" }, { "PXA_ROPE_KV_FUSE", "2" } };
        for (const auto & kv : v70l) {
            ac.env.emplace_back(kv[0], kv[1]);
        }
        pxa_pick p;
        p.flag = "PXA_AUTO_V70_LLAMA";
        p.evidence = "";
        p.value = "on"; p.status = "MEASURED"; p.applied = true;
        p.why = "llama on one V100: device greedy argmax + many-block argmax_val + sampled top-W window + rope/KV-store fusion, "
                "bit-identical (Nemotron-Nano-8B PXQN4); PXA_AUTO_V70_LLAMA=0 or any one lever's own value overrides";
        ac.picks.push_back(p);
    }

    // ---- one sm_80+ card + Nemotron-H: async split inputs (2026-10-21, ledger rig-n4-decode-host-gap) ----
    // The Mamba-2 layers feed ~45 tiny host-resident split inputs per decode step, each a blocking copy behind a host wait.
    // PXA_SCHED_ASYNC_INPUTS=1 (copy scheduling only, greedy512 sha identical): Nemotron-3-Nano-4B PXQ4 on an RTX 3080 Ti
    // tg128 160.0 -> 200.0 tok/s (mainline Q4_K_M 187.2). Exactly one card with cc >= 800, arch nemotron_h, no tensor split,
    // level 2; P100 / V100 unmeasured. PXA_AUTO_A80_NEMOH=0 stands the pick down; PXA_SCHED_ASYNC_INPUTS in the environment wins.
    // src/llama.cpp carries the same rule for the bench/cli.
    const char * a80n_env = getenv("PXA_AUTO_A80_NEMOH");
    const bool a80n_cell = level >= 2 && model.arch == "nemotron_h" && topo.n_dev == 1 && topo.cc.size() == 1 && topo.cc[0] >= 800 &&
                           !(user.sm ? user.sm_value == PXA_SM_TENSOR : ac.split == PXA_SM_TENSOR) &&
                           (a80n_env && *a80n_env ? atoi(a80n_env) != 0 : true);
    if (a80n_cell) {
        ac.env.emplace_back("PXA_SCHED_ASYNC_INPUTS", "1");
        pxa_pick p;
        p.flag = "PXA_AUTO_A80_NEMOH";
        p.evidence = "";
        p.value = "on"; p.status = "MEASURED"; p.applied = true;
        p.why = "Nemotron-H on one sm_80+ card: async split inputs, Nano-4B decode 160 -> 200 tok/s on an RTX 3080 Ti "
                "(mainline 187), greedy sha identical; PXA_AUTO_A80_NEMOH=0 or PXA_SCHED_ASYNC_INPUTS=0 overrides";
        ac.picks.push_back(p);
    }

    // ---- PXA_DN_CONVFUSE (delta-net conv cluster) is a tensor-split lever ----------------------------
    // measured 2026-09-29, REL v2026.10, 27B one-card file, llama-bench tg128 ABAB, -fa 1 q4_0 KV:
    //   1x V100  fused 23.3/22.4 vs unfused 32.3/33.6 t/s (-31%);  1x P100  19.4/19.3 vs 24.6/24.2 (-21%);
    //   2x P100 -sm tensor fused 37.9/37.9 vs 36.3/36.3 (+4.4%);   2x V100 -sm tensor 56.3/56.1 vs 53.4/53.4 (+5.3%).
    // So the fused conv kernel is a default only where the split is tensor; anywhere else (one card,
    // -sm layer) the separate kernels are the default. PXA_DN_CONVFUSE=1/0 in the environment always wins.
    if (level >= 2 && topo.n_dev >= 1) {
        const bool tensor_eff = user.sm ? user.sm_value == PXA_SM_TENSOR : ac.split == PXA_SM_TENSOR;
        pxa_pick p;
        p.flag = "PXA_DN_CONVFUSE";
        p.evidence = "";
        if (tensor_eff) {
            p.value = "on"; p.status = "MEASURED"; p.applied = false;
            p.why = "tensor split: the fused delta-net conv cluster measured +4.4% (2x P100) / +5.3% (2x V100) decode";
        } else if (v70_cell) {
            // the one-V100 qwen35 plain-decode stack (block below emits the env entries)
            p.value = "on"; p.status = "MEASURED"; p.applied = false; p.evidence = "";
            p.why = "qwen35 on one V100: one lever of the bit-identical plain-decode stack (+3.2% server decode, 2026-10-16), "
                    "PXA_DN_CONVFUSE=0 puts it back";
        } else if (p1q35_cell) {
            // the one-P100 qwen35 PXQN additions (block above emits the env entry)
            p.value = "on"; p.status = "MEASURED"; p.applied = false; p.evidence = "";
            p.why = "qwen35 PXQN on one P100: the conv cluster with the host g_dead index measured ~+2.5% server decode, sha identical "
                    "(2026-10-14); PXA_DN_CONVFUSE=0 puts it back";
        } else if (p2l_cell) {
            // the dense 27B PXQN4 P100 layer pair takes the one-P100 stack (block above emits the env entry)
            p.value = "on"; p.status = "MEASURED"; p.applied = false; p.evidence = "";
            p.why = "dense 27B PXQN4 on the P100 layer pair: one lever of the one-P100 decode stack, run on both cards; "
                    "PXA_DN_CONVFUSE=0 puts it back";
        } else if (p60_cell) {
            // part of the measured board recipe on the 35B layer split; the P60 stack block below emits the env entry
            // the pair keeps its 2026-10-16 text byte for byte (only 3+ cards read differently)
            p.value = "on"; p.status = p60_measured_n ? "MEASURED" : "INFERRED"; p.applied = false; p.evidence = "";
            p.why = topo.n_dev == 2
                ? "35B-A3B on a layer-split P100 pair: one lever of the P60 verify stack (the board recipe), PXA_DN_CONVFUSE=0 puts it back"
                : "35B-A3B on layer-split P100s: one lever of the P60 verify stack (the board recipe), PXA_DN_CONVFUSE=0 puts it back";
        } else {
            ac.env.emplace_back("PXA_DN_CONVFUSE", "0");
            // MEASURED only where the A/B above was run: one P100 or one V100. Any other no-tensor-split
            // topology (a layer-split pair, 3-4 cards, a 1080 Ti) takes the same pick carried over from the
            // one-card runs, so it is INFERRED. Label only: the pick (off) and its application are unchanged.
            // qwen4exp (Flash-Next) on a V100 pair with -sm layer was measured too (2026-10-17):
            // the fused conv alone was slower on the Overdrive cell (prose -6.8%, code -6.1%,
            // greedy output identical), so the off pick is MEASURED there as well.
            const bool q4x_v100_pair = model.arch == "qwen4exp" && topo.n_dev == 2 && topo.cc.size() == 2 &&
                                       topo.cc[0] == 700 && topo.cc[1] == 700;
            const bool one_card_ab = (topo.n_dev == 1 && !topo.cc.empty() && (topo.cc[0] == 600 || topo.cc[0] == 700)) || q4x_v100_pair;
            p.value = "off"; p.status = one_card_ab ? "MEASURED" : "INFERRED"; p.applied = true;
            p.why = one_card_ab
                ? "no tensor split: the fused delta-net conv cluster measured -31% (1x V100) / -21% (1x P100) decode; "
                  "PXA_DN_CONVFUSE=1 puts it back"
                : "no tensor split: the fused delta-net conv cluster measured -31% (1x V100) / -21% (1x P100) decode; "
                  "this topology was not measured, the pick is carried over from the one-card runs; "
                  "PXA_DN_CONVFUSE=1 puts it back";
        }
        ac.picks.push_back(p);
    }

    // ---- PXA_PXQN_SM60_RBSHAPE (sm_60 row-block GEMV shapes) is off on a P100 tensor split ----------
    // measured 2026-10-05 by the v2026.10.3 gate (rel-gate2 #14803), Qwen3.8-27B PXQN4 on 2x P100 -sm tensor,
    // llama-server at shipping defaults (n-gram speculation on), mean of three alternating boots per arm,
    // prose/code/prose2 t/s: v2026.10.2 47.75/59.62/47.68; rule on 45.67/56.68/45.68 (-4.4/-4.9/-4.2%); rule off
    // 47.58/59.51/47.50 (= v2026.10.2); identical text in every arm. The new shapes help a single-column decode
    // (+1.5% plain on the pair, +7% on one P100) but cost the multi-column verify of the speculation on the split
    // halves. The shape stays a function of (arch, kind, R, K) so speculation keeps the plain summation order: the
    // rule is switched per topology here, not per batch width. One card keeps it. PXA_PXQN_SM60_RBSHAPE=1/0 wins.
    if (level >= 2 && topo.n_dev >= 2) {
        const bool tensor_eff = user.sm ? user.sm_value == PXA_SM_TENSOR : ac.split == PXA_SM_TENSOR;
        bool any_sm60 = false;
        for (int cc : topo.cc) any_sm60 = any_sm60 || cc == 600;
        if (tensor_eff && any_sm60) {
            pxa_pick p;
            p.flag = "PXA_PXQN_SM60_RBSHAPE";
            p.evidence = "";
            ac.env.emplace_back("PXA_PXQN_SM60_RBSHAPE", "0");
            p.value = "off"; p.status = "MEASURED"; p.applied = true;
            p.why = "P100 tensor split: the sm_60 row-block shapes cost the speculative verify -4.4..-4.9% server decode "
                    "(2x P100, 27B PXQN4); off restores v2026.10.2 speed; PXA_PXQN_SM60_RBSHAPE=1 puts it back";
            ac.picks.push_back(p);
        }
    }

    // ---- the dense 27B on the P100 TENSOR pair: the persistent row-block one-column GEMV (2026-10-20) ----
    // PXA_PXQN_RBP (the one-P100 pick below, +4.6% there) never engaged on the pair: the halves' launches resolve (RBSHAPE off,
    // the rule above) to the k4 s2 non-GU and k2 s2 fused up/gate shapes, which RBP did not instantiate. PXA_PXQN_RBP_TP adds
    // those two shapes; RBP is bit-identical to the row-block kernel at the same (ks, ns) (tests/test-pxqn-cuda.cu -rb27tp checks
    // the five per-card shapes byte for byte). Only the one-column (plain decode) launch changes; the speculative verify keeps its
    // multi-column kernels. Scope = what was measured: two sm_60 cards, tensor split, arch qwen35, tier PXQN4.
    // PXA_AUTO_P2T=0 stands the pick down; PXA_PXQN_RBP=0 / PXA_PXQN_RBP_TP=0 in the environment win.
    // The same cell also takes four bit-identical decode levers that were on only for the one-card cells (u07c, 2026-10-20,
    // same binary and bracket, greedy512 sha equal on prose/code/chat): PXA_ARGMAX_VAL_MB (many-block argmax_val over the 262k
    // vocab), PXA_PLAIN_GREEDY_AMAX (a plain greedy step reads back one id), PXA_DN_CF_DIRECT (the conv cluster reads its scratch
    // directly) and PXA_PXQN_RMS_RHT_GB (fused rms-norm + RHT off the split epilogue); each =0 in the environment wins.
    {
        const char * p2t_env = getenv("PXA_AUTO_P2T");
        const bool p2t_tensor = user.sm ? user.sm_value == PXA_SM_TENSOR : ac.split == PXA_SM_TENSOR;
        const bool p2t_cell = level >= 2 && topo.n_dev == 2 && topo.cc.size() == 2 && topo.cc[0] == 600 && topo.cc[1] == 600 &&
                              p2t_tensor && model.arch == "qwen35" && pxa_dense27_tier(model) &&
                              (p2t_env && *p2t_env ? atoi(p2t_env) != 0 : true);
        if (p2t_cell) {
            ac.env.emplace_back("PXA_PXQN_RBP", "1");
            ac.env.emplace_back("PXA_PXQN_RBP_TP", "1");
            ac.env.emplace_back("PXA_ARGMAX_VAL_MB", "1");
            ac.env.emplace_back("PXA_PLAIN_GREEDY_AMAX", "1");
            ac.env.emplace_back("PXA_DN_CF_DIRECT", "1");
            ac.env.emplace_back("PXA_PXQN_RMS_RHT_GB", "1");
            pxa_pick p;
            p.flag = "PXA_AUTO_P2T";
            p.evidence = "";
            p.value = "on"; p.status = "MEASURED"; p.applied = true;
            p.why = "dense 27B PXQN4 on the P100 tensor pair: the persistent row-block GEMV on the halves' one-column shapes + "
                    "ARGMAX_VAL_MB + PLAIN_GREEDY_AMAX + DN_CF_DIRECT + PXQN_RMS_RHT_GB, all bit-identical; PXA_AUTO_P2T=0 or "
                    "any of the six =0 in the environment overrides";
            ac.picks.push_back(p);
        }
    }

    // ---- the P60 verify stack (cell and evidence above) --------------------------------------------------------------
    if (p60_cell) {
        static const struct { const char * name; const char * val; } stack[] = {
            { "PXA_P60_MEGA", "1" }, { "PXA_P60_MEGA_PXQN_T", "1" }, { "PXA_KQMMV_VW", "1" }, { "PXA_PXQN_VW", "1" },
            { "PXA_P60_PRE_VW", "1" }, { "PXA_PXQN_RBP", "1" }, { "PXA_PXQN_RMS_RHT_GB", "1" }, { "PXA_ARGMAX_VAL_MB", "1" },
            { "PXA_DN_CONVFUSE", "1" }, { "PXA_FA_GQA_PACK", "0" }, { "PXA_PXQN_QPN", "1" },
        };
        for (const auto & e : stack) {
            ac.env.emplace_back(e.name, e.val);
        }
        pxa_pick p;
        p.flag = "PXA_P60_MEGA";
        if (topo.n_dev == 2) { p.evidence = ""; } else { p.evidence = ""; }
        p.value = "on"; p.status = p60_measured_n ? "MEASURED" : "INFERRED"; p.applied = true;
        if (topo.n_dev == 2) {
            p.why = "35B-A3B (qwen35moe PXQN4) on 2x P100 layer split: the P60 verify stack (11 levers) measured plain decode 72.7 -> 125 t/s (+72%), "
                    "bare launch 74 -> 138 / 151 / 130 with MTP n_max 3 p_min 0.85; decode KLD vs OFF 2.5e-3 (engine's own width noise 1.8e-3), prefill identical; "
                    "PXA_AUTO_P60=0 or any one PXA_* lever in the environment overrides";
        } else if (topo.n_dev == 4) {
            p.why = "35B-A3B (qwen35moe PXQN4) on 4x P100 layer split: the P60 verify stack (11 levers) measured plain decode 73 -> 125 t/s (+71%, greedy512 prose / code / chat); "
                    "correct, not bit-identical to the stack off (decode KLD 2.5e-3 on the pair, prefill identical); "
                    "PXA_AUTO_P60=0 or any one PXA_* lever in the environment overrides";
        } else {
            snprintf(buf, sizeof(buf), "35B-A3B (qwen35moe PXQN4) on %dx P100 layer split: the P60 verify stack (11 levers), measured on 2 and 4 "
                                       "P100s (plain decode +72%% on the pair), the same per-device kernels here; PXA_AUTO_P60=0 or any one "
                                       "PXA_* lever in the environment overrides", topo.n_dev);
            p.why = buf;
        }
        ac.picks.push_back(p);
    }

    // ---- long prefill on the 35B-A3B P100 pair: stable output-ubatch shape + sm_60 head-256 prompt attention ----
    // PXA_PF_STABLE_SHAPE=1 (keep the 1-row shared-expert gate unfused so every prompt ubatch has the same node count: no re-plan
    // + full drain per batch), PXA_FA_TILE_H2CHUNK=1 (tile-f16 prompt attention, half2 V*P chunks) and PXA_FA_LONGPREFILL=3 (the
    // lean head-256 kernel at KV >= 8192 only).
    // finding #22118: 35B-A3B PXQN4 on 2x P100, llama-bench -b 4096 -ub 1536, pp512/pp4096/pp16384 +0.7/+22.1/+43.5% over the bare
    // default; vs a popular community fork -25% at 16k -> +20%. CORRECT, NOT BIT-IDENTICAL: KLD vs f32acc 0.00148 (H2CHUNK) /
    // 0.00166 (LONGPREFILL) against the engine's own noise band 0.0021; greedy512 sha identical with PF_STABLE_SHAPE (b73f637ed7141ea8,
    // d8f618d3f88d51df); the long-prompt greedy sha moves with LONGPREFILL. GATE OWED before this ships: det 12/12 + needle 16k on
    // this cell. Same cell as the P60 stack minus PXA_AUTO_P60 (the split mode is free: prefill does
    // not depend on it). PXA_AUTO_PF16K=0 stands the pick down; any one of the three named in the environment is kept.
    {
        const char * pf_env = getenv("PXA_AUTO_PF16K");
        const bool pf_on = pf_env && *pf_env ? atoi(pf_env) != 0 : true;
        // two or more cards that are ALL sm_60 (prompt attention runs per device on its own layers): measured on 2 and on 4 (2026-10-20,
        // rc2.1 engine, 4x P100, llama-bench -b 4096 -ub 1536 REPS 2 NOT-QUIET: pp4096 2567.8 -> 2664.6 (+3.8%), pp16384
        // 2402.5 -> 2741.5 (+14.1%)); 3 or 5+ cards INFERRED. Until 2026-10-20 the rule said exactly two cards.
        const bool pf_cell = level >= 2 && pf_on && model.arch == "qwen35moe" && model.tier == "PXQN4" && topo.n_dev >= 2 &&
                             topo.cc.size() == (size_t) topo.n_dev &&
                             std::all_of(topo.cc.begin(), topo.cc.end(), [](int c) { return c == 600; });
        if (pf_cell) {
            // PXA_PF_STABLE_SHAPE=1 itself comes from the sm_60 MoE prefill rule below (order #22214, bit-identical); this
            // cell adds the two attention halves on top
            ac.env.emplace_back("PXA_FA_TILE_H2CHUNK", "1");
            ac.env.emplace_back("PXA_FA_LONGPREFILL", "3");
            pxa_pick p;
            p.flag = "PXA_FA_TILE_H2CHUNK";
            p.evidence = "";
            p.value = "on"; p.status = (topo.n_dev == 2 || topo.n_dev == 4) ? "MEASURED" : "INFERRED"; p.applied = true;
            if (topo.n_dev == 2) {
                p.why = "35B-A3B (qwen35moe PXQN4) on 2x P100: stable prompt-ubatch shape + H2CHUNK tile + lean long-KV attention measured "
                        "pp16384 +43.5%, pp4096 +22.1%; KLD 0.00148/0.00166 within the engine's noise 0.0021; PXA_AUTO_PF16K=0 or any one "
                        "of the three in the environment overrides";
            } else if (topo.n_dev == 4) {
                p.evidence = "";
                p.why = "35B-A3B (qwen35moe PXQN4) on 4x P100: H2CHUNK tile + lean long-KV attention on top of the stable prompt-ubatch shape "
                        "measured pp16384 2402 -> 2741 (+14.1%), pp4096 +3.8%; the same kernels as the pair (KLD 0.00148/0.00166 within the "
                        "engine's noise 0.0021); PXA_AUTO_PF16K=0 or any one of the three in the environment overrides";
            } else {
                p.evidence = "";
                snprintf(buf, sizeof(buf), "35B-A3B (qwen35moe PXQN4) on %dx P100: H2CHUNK tile + lean long-KV attention, measured on 2 and 4 "
                                           "P100s (pp16384 +43.5%% / +14.1%%), the same per-device kernels here; PXA_AUTO_PF16K=0 or any one "
                                           "of the three in the environment overrides", topo.n_dev);
                p.why = buf;
            }
            ac.picks.push_back(p);
        }
        // the dense 27B (qwen35 PXQN4, head 256, GQA 6, full attention every 4th layer) on the same 2x P100 pair, lane
        //  (cell C6, -sm tensor -fa 1 -ub 2048): the same two attention halves on top of PXA_PF_SLICE. QUICK q2/q3
        // (#23044 / #23049, REPS 2, NOT-QUIET): pp16384 353.3 -> 372.7 (+5.5%, the rival Q4_1 354.9: r 0.995 -> 1.050),
        // pp4096 399.2 -> 403.6. Correct, not bit-identical (the same kernels as the 35B cell above; LONGPREFILL only at
        // KV >= 8192). Gates (det 12/12, needle 16k, KLD) run once on the release candidate. Same off switches.
        // ONE P100 (2026-10-18, cells C017/C018, llama-bench defaults -ub 512, REPS 2, NOT-QUIET): pp16384
        // 211.67 -> 224.15 (+5.9%), pp4096 244.5 -> 248.3 (+1.5%); the same kernels, the same identity status.
        const bool pf_one  = topo.n_dev == 1 && topo.cc.size() == 1 && topo.cc[0] == 600;
        const bool pf_dense = level >= 2 && pf_on && model.arch == "qwen35" && pxa_dense27_tier(model) &&
                              (pf_one || (topo.n_dev == 2 && topo.cc.size() == 2 && topo.cc[0] == 600 && topo.cc[1] == 600));
        if (pf_dense) {
            ac.env.emplace_back("PXA_FA_TILE_H2CHUNK", "1");
            ac.env.emplace_back("PXA_FA_LONGPREFILL", "3");
            pxa_pick p;
            p.flag = "PXA_FA_TILE_H2CHUNK";
            p.evidence = "";
            p.value = "on"; p.status = "MEASURED"; p.applied = true;
            p.why = pf_one
                ? "27B dense (qwen35 PXQN4) on 1x P100: H2CHUNK tile + lean long-KV prompt attention measured pp16384 211.7 -> 224.2 "
                  "(+5.9%), pp4096 244.5 -> 248.3; correct, not bit-identical; PXA_AUTO_PF16K=0 or either flag in the environment overrides"
                : "27B dense (qwen35 PXQN4) on 2x P100: H2CHUNK tile + lean long-KV prompt attention measured pp16384 353.3 -> 372.7 "
                  "(+5.5%), pp4096 399.2 -> 403.6; correct, not bit-identical; PXA_AUTO_PF16K=0 or either flag in the environment overrides";
            ac.picks.push_back(p);
        }
    }

    // ---- one P100, PXQN file: the two bit-identical K3 decode kernels (2026-10-16) ----------------
    // PXA_PXQN_RMS_RHT_GB (the fused norm + RHT site on 8 blocks x 128 with one exchange instead of one 1024-thread block) and
    // PXA_ARGMAX_VAL_MB (the greedy argmax over many blocks instead of one). Both bit-identical by construction (576-case unit
    // test tests/test-pxqn-rmsrht-gb.cu; argmax keeps the lowest index on ties). Measured 2026-10-14
    // (mail #20867): 1x P100 card 6, 27B PXQN4, same binary, ABBA REPS 3 x 256, prose 25.56 -> 25.87 (+1.2%), code 25.49 -> 25.65
    // (+0.6%), prose2 25.38 -> 25.72 (+1.4%), every response sha identical (greedy512 cb0352c87a6a). One V100 gets RMS_RHT_GB from
    // llama_init_from_model and the 2x P100 35B cell from the P60 stack above; this is the one-P100 cell. PXA_PXQN_RMS_RHT_GB=0 /
    // PXA_ARGMAX_VAL_MB=0 in the environment put either one back (setenv never overwrites).
    // ARCH-AGNOSTIC ON PURPOSE (review finding 1): this one-P100 block never tested the arch. The old single-tier PXQN test let single-type Flint
    // PXQN4 in, and Flint B (PXQN4+PXQN5, arch flint) is single-type Flint to it, so a PXQN-family mix that includes PXQN4 takes K3 here too; RBP below stays qwen35-only.
    if (level >= 2 && topo.n_dev == 1 && topo.cc.size() == 1 && topo.cc[0] == 600 && pxa_dense27_pxqn_family(model) && !p60_cell) {
        ac.env.emplace_back("PXA_PXQN_RMS_RHT_GB", "1");
        ac.env.emplace_back("PXA_ARGMAX_VAL_MB", "1");
        pxa_pick p;
        p.flag = "PXA_PXQN_RMS_RHT_GB";
        p.evidence = "";
        p.value = "on"; p.status = "MEASURED"; p.applied = true;
        p.why = "one P100, PXQN file: the 8-block fused norm+RHT site and the many-block greedy argmax (PXA_ARGMAX_VAL_MB) measured "
                "+0.6..+1.4% decode on the 27B PXQN4, bit-identical; PXA_PXQN_RMS_RHT_GB=0 / PXA_ARGMAX_VAL_MB=0 put them back";
        ac.picks.push_back(p);
        // PXA_PXQN_RBP (2026-10-16): the persistent row-block one-column GEMV with the trimmed decode, bit-identical
        // by construction. Measured in-engine on the dense 27B PXQN4 file only (cell g1rbp-2109: 1x P100 card 0,
        // same binary, server REPS 6, greedy prose 25.26 -> 26.44 t/s (+4.6%), code 25.22 -> 26.40 (+4.8%), greedy512 sha equal; llama-bench
        // tg128 +5.7%), so the pick is scoped to that file class: arch qwen35, tier PXQN4. PXA_PXQN_RBP=0 in the environment puts it back.
        if (model.arch == "qwen35" && pxa_dense27_tier(model)) {
            ac.env.emplace_back("PXA_PXQN_RBP", "1");
            pxa_pick r;
            r.flag = "PXA_PXQN_RBP";
            r.evidence = "";
            r.value = "on"; r.status = "MEASURED"; r.applied = true;
            r.why = "one P100, dense 27B PXQN4: the persistent row-block GEMV measured +4.6% prose / +4.8% code server decode, "
                    "bit-identical (greedy512 sha equal); PXA_PXQN_RBP=0 puts it back";
            ac.picks.push_back(r);
        }
    }

    // The Overdrive cell's cold-expert CPU pool (PXA_XCACHE_CPUPOOL_CPUS) when the operator gave none: the launcher's own rule
    // (tools/pxa-launch.py overdrive_pool_cpus: the cards' node within the allowed CPUs, one thread per core first, 16 wide). By the
    // time the registry runs the CUDA backend has initialised and PXA_NUMA_BIND has already narrowed the process to the cards'
    // node, so the allowed CPUs ARE that node (or the operator's own cpuset). An unpinned pool time-slices with the decode thread:
    // quick test 2026-10-18 (Flash-Next 1x V100, node-1 cpuset, sha 4/4 equal) pinned 35.64 vs unpinned 31.53 t/s.
    const auto od_pool_cpus = []() -> std::string {
#if defined(__linux__)
        cpu_set_t cur; CPU_ZERO(&cur);
        if (sched_getaffinity(0, sizeof(cur), &cur) != 0) return "";
        std::vector<int> first, second;
        std::set<int> seen_core;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (!CPU_ISSET(c, &cur)) continue;
            std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
            std::string sl; std::getline(f, sl);
            // the core = the lowest sibling that is ALSO allowed (a sibling outside the cpuset does not share this set's work)
            int core = c;
            std::stringstream ss(sl); std::string tok;
            while (std::getline(ss, tok, ',')) {
                const size_t dash = tok.find('-');
                const int a = atoi(tok.c_str()), b = dash == std::string::npos ? a : atoi(tok.c_str() + dash + 1);
                for (int x = a; x <= b; ++x) if (x >= 0 && x < CPU_SETSIZE && CPU_ISSET(x, &cur)) { core = std::min(core, x); }
            }
            (seen_core.insert(core).second ? first : second).push_back(c);
        }
        first.insert(first.end(), second.begin(), second.end());
        if (first.size() < 4) return "";
        if (first.size() > 16) first.resize(16);
        std::string out;
        for (size_t i = 0; i < first.size(); ++i) out += (i ? "," : "") + std::to_string(first[i]);
        return out;
#else
        return "";
#endif
    };

    // ---- PXA Overdrive: the measured per-class lever set is the closed PXQN library's (ggml-pxqn-overdrive.h) --------
    // The library carries the rows, their values and their order, and the class/fit decision that applies them (recipe E
    // for the Flash-Next expert-cache seat; the measurements are recorded with it). The engine hands it what it knows (the
    // cards, the GGUF header, the user's split, the level) and appends the answer here, after the picks above, with
    // setenv that never overwrites (any one switch named in the environment is kept). Nothing of the set is spelled out
    // in the open tree (2026-10-19; before: an in-tree table, then encoded
    // names). Without the library, or with PXA_AUTO_OVERDRIVE=0 / PXA_NO_OVERDRIVE=1, nothing is appended and the engine
    // runs as before. The cold-expert CPU pool is the engine's own derivation (od_pool_cpus above: the NUMA node of the
    // cards) and rides along under the name the library asks for, when the operator gave none.
    // tests/test-pxa-autoconfig.cpp checks the hooks (library present / absent, log and json quiet); the rows themselves
    // are pinned by tests/test-pxqn-overdrive.cpp (closed tree).
    {
        std::vector<int32_t>  od_cc(topo.cc.begin(), topo.cc.end());
        std::vector<uint64_t> od_vram((size_t) std::max(topo.n_dev, 0), 0);
        for (size_t d = 0; d < od_vram.size() && d < topo.vram_mib.size(); ++d) {
            od_vram[d] = (uint64_t) topo.vram_mib[d] << 20;
        }
        od_cc.resize(od_vram.size(), 0);
        ggml_pxqn_od_in od_in = {};
        od_in.size        = (uint32_t) sizeof(od_in);
        od_in.arch        = model.arch.c_str();
        od_in.tier        = model.tier.c_str();
        od_in.n_expert    = model.n_expert;
        od_in.n_nextn     = model.n_nextn;
        od_in.n_layer     = model.n_layer;
        od_in.model_bytes = model.bytes;
        od_in.n_dev       = topo.n_dev;
        od_in.cc          = od_cc.empty() ? nullptr : od_cc.data();
        od_in.vram_bytes  = od_vram.empty() ? nullptr : od_vram.data();
        od_in.split       = user.sm ? user.sm_value : ac.split;
        od_in.level       = level;
        od_in.flags       = user.sm ? GGML_PXQN_OD_IN_SPLIT_FORCED : 0u;
        ggml_pxqn_od_out od_out = {};
        od_out.size = (uint32_t) sizeof(od_out);
        if (ggml_pxqn_overdrive_choose(&od_in, &od_out)) {
            for (int32_t i = 0; i < od_out.n_rows; ++i) {
                ac.env.emplace_back(od_out.rows[i].name, od_out.rows[i].value);
            }
            if (od_out.pool_name && *od_out.pool_name && !getenv(od_out.pool_name)) {
                const std::string pool = od_pool_cpus();
                if (!pool.empty()) ac.env.emplace_back(od_out.pool_name, pool);
            }
            pxa_pick p;
            p.flag = "PXA_AUTO_OVERDRIVE";
            p.evidence = "";
            p.value = "on"; p.status = "MEASURED"; p.applied = true;
            p.why = od_out.why;
            ac.picks.push_back(p);
        }
    }

    // ---- C042 small tail: the async CPU-pool cold path (2026-10-19) ----------
    // A qwen35moe PXQN4 file on ONE V100 that overflows the card by a small tail (pxa_registry_xcache_smalltail): the expert cache
    // runs the cold experts on the CPU. The async cold executor + the library's pinned CPU pool (the Overdrive cell's mechanism, no
    // scheduler split and no host sync per cold layer) is bit-identical to the split path at the same plan and makes a cold layer
    // cheap, so the planner (src/llama-pxa-place.cpp, small-tail async cost) spreads the cold set over more, thinner layers.
    // Pool CPUs derived like the Overdrive cell (od_pool_cpus); the pool's thread count stays -t. PXA_AUTO_XCACHE_SMALLTAIL=0 (or any
    // one of these switches in the environment) overrides; setenv never overwrites.
    if (level >= 2 && pxa_registry_xcache_smalltail(topo, model, nullptr)) {
        ac.env.emplace_back("PXA_XCACHE_ASYNC", "1");
        ac.env.emplace_back("PXA_XCACHE_CPUPOOL", "1");
        ac.env.emplace_back("PXA_XCACHE_IDLE_BACKOFF_MS", "200");
        if (!getenv("PXA_XCACHE_CPUPOOL_CPUS")) {
            const std::string pool = od_pool_cpus();
            if (!pool.empty()) ac.env.emplace_back("PXA_XCACHE_CPUPOOL_CPUS", pool);
        }
        pxa_pick p;
        p.flag = "PXA_AUTO_XCACHE_SMALLTAIL";
        p.evidence = "";
        p.value = "on"; p.status = "MEASURED"; p.applied = true;
        p.why = "qwen35moe PXQN4 on one V100 with a small streamed tail: async CPU-pool cold path (bit-identical to the split path) and "
                "more, thinner cold layers; 35B-A3B decode +13-16% over the split path; PXA_AUTO_XCACHE_SMALLTAIL=0 turns it off";
        ac.picks.push_back(p);
    }

    // ---- Flint on one P100 (2026-10-23): the C042 small-tail plan on sm_60 for this file (pxa_registry_flint_1card)
    // PXA_AUTO_FLINT_1CARD=0, or PXA_XCACHE_SMALLTAIL_SM60 in the environment, overrides; setenv never overwrites. The small-tail
    // scope's own switches still apply inside it (PXA_XCACHE_RING1=0 two-layer ring, PXA_XCACHE_SLACK_MB=N / =0 the credit).
    if (level >= 2 && pxa_registry_flint_1card(topo, model, nullptr)) {
        ac.env.emplace_back("PXA_XCACHE_SMALLTAIL_SM60", "1");
        pxa_pick p;
        p.flag = "PXA_AUTO_FLINT_1CARD";
        p.evidence = "";
        p.value = "on"; p.status = "MEASURED"; p.applied = true;
        p.why = "flint PXQN4 on one P100 (file up to 1.5x the card): the small-tail plan (one-layer ring, 384 MiB slack credit) keeps "
                "one more layer resident; PXA_AUTO_FLINT_1CARD=0 turns it off";
        ac.picks.push_back(p);
    }
    // ---- Flint on one P100/V100 (2026-10-23): the BF16 router GEMV (pxa_registry_flint_1card_router)
    // PXA_AUTO_FLINT_ROUTER=0 / PXA_AUTO_FLINT_1CARD=0, or PXA_ROUTER_BF16 in the environment, overrides; setenv never overwrites.
    if (level >= 2 && pxa_registry_flint_1card_router(topo, model, nullptr)) {
        ac.env.emplace_back("PXA_ROUTER_BF16", "2");
        pxa_pick p;
        p.flag = "PXA_AUTO_FLINT_ROUTER";
        p.evidence = "";
        p.value = "on"; p.status = "MEASURED"; p.applied = true;
        p.why = "flint PXQN4 on one P100/V100 (file up to 1.5x the card): the BF16 router runs as one F32-widened GEMV instead of "
                "cuBLAS BF16 + two conversions (P100 0.97 -> 0.42 ms/token); not bit-identical; PXA_AUTO_FLINT_ROUTER=0 turns it off";
        ac.picks.push_back(p);
    }

    // ---- Flint on one V100 (2026-10-23): the one-V100 plain-decode stack (pxa_registry_flint_1card_v100)
    // PXA_AUTO_FLINT_V100=0 / PXA_AUTO_FLINT_1CARD=0, or any of the six levers in the environment, overrides; setenv never overwrites.
    if (level >= 2 && pxa_registry_flint_1card_v100(topo, model, nullptr)) {
        static const char * const kv[][2] = { {"PXA_CUDA_GRAPH_DECODE", "1"}, {"PXA_CUDA_GRAPH_MOE", "1"}, {"PXA_CUDA_GRAPH_LRU", "256"},
                                              {"PXA_SCHED_ASYNC_INPUTS", "1"}, {"PXA_PLAIN_GREEDY_AMAX", "1"}, {"PXA_ARGMAX_VAL_MB", "1"} };
        for (const auto & e : kv) {
            ac.env.emplace_back(e[0], e[1]);
        }
        pxa_pick p;
        p.flag = "PXA_AUTO_FLINT_V100";
        p.evidence = "";
        p.value = "on"; p.status = "RULE"; p.applied = true;
        p.why = "flint PXQN4 on one V100 (file up to 1.5x the card): CUDA graphs on the decode segments between the cold layers "
                "(+ MoE capture, LRU 256, staged split inputs, greedy device argmax), all bit-identical; PXA_AUTO_FLINT_V100=0 turns it off";
        ac.picks.push_back(p);
    }

    // ---- sm_60 MoE prefill: PXA_PF_STABLE_SHAPE (order #22214, 2026-10-16) -------------------------------
    // Keeps the shared-expert gate unfused in every multi-token graph, so the output ubatch of a prompt batch has the same node
    // count as the others and the scheduler stops re-planning (and draining every backend) twice per batch. Measured by
    //  (#22196): 2x P100 35B-A3B PXQN4 pp16384 r3 1150.6 -> 1569.7 t/s (+36%), greedy sha
    // identical (the fork's 1505.5 on the same box). Decode graphs (one token) are untouched. Scoped to where it was measured:
    // qwen35moe on sm_60 cards. PXA_PF_STABLE_SHAPE=0 in the environment keeps the fused form (setenv never overwrites).
    if (level >= 2 && model.arch == "qwen35moe" && !topo.cc.empty() &&
        std::all_of(topo.cc.begin(), topo.cc.end(), [](int c) { return c == 600; })) {
        ac.env.emplace_back("PXA_PF_STABLE_SHAPE", "1");
        pxa_pick p;
        p.flag = "PXA_PF_STABLE_SHAPE";
        p.evidence = "";
        p.value = "on"; p.status = "MEASURED"; p.applied = true;
        p.why = "qwen35moe on sm_60: one graph shape for every prompt ubatch (no scheduler re-plan per batch): 2x P100 35B pp16384 "
                "1150.6 -> 1569.7 t/s, sha identical; PXA_PF_STABLE_SHAPE=0 keeps the fused 1-row gate";
        ac.picks.push_back(p);
    }

    // ---- PXA_PXQN_RHT_FUSE on one V100: no pick any more -------------------------------------------
    // 2026-09-29 the RHT->GEMV fusion measured -5..8% on one V100 and a pick here set PXA_PXQN_RHT_FUSE=0. Re-measured
    // 2026-10-14 (K3; 27B PXQN4 card 2, same binary, ABBA): all
    // fusions ON 41.04 vs 40.41 t/s tg128, server SA1 prose/code 40.24/40.26 vs 39.73/39.75, 1139 vs 1429 launches per
    // token, logits + greedy512 sha identical (the out_in site now runs RHT+q8_1 into the MMVQ kernel instead of
    // re-rotating in every GEMV block). So one V100 keeps the fusions like every other card; PXA_PXQN_RHT_FUSE=0 in the
    // environment still turns them off.

    // ---- -ngl ---------------------------------------------------------------------------------
    // The engine's own default offloads NOTHING on CUDA (llama_model_default_params: 0 layers), so a
    // bare `llama-server -m file` used to run on the CPU while pxa-launch passes -ngl 999. With a
    // card present and -ngl unset, every layer goes on the cards, as the launcher has always done.
    {
        pxa_pick p;
        p.flag = "-ngl";
        if (user.ngl) {
            p.status = "USER"; p.why = "given on the command line";
        } else if (level < 2) {
            // the level-1 rollback (PXA_ENHANCE=0) is the old behaviour end to end: no -ngl pick
            p.status = "OFF"; p.why = level == 0 ? "PXA_REFERENCE=1: engine default kept"
                                                 : "config level below ENHANCE: engine default kept (pass -ngl 999)";
        } else if (topo.n_dev < 1) {
            p.value = "0"; p.status = "RULE"; p.why = "no CUDA device: the model runs on the CPU";
        } else {
            ac.n_gpu_layers = 999;
            p.value = "999"; p.status = "RULE"; p.applied = true;
            p.why = "every layer on the cards (what pxa-launch passes); -ngl N for a partial offload";
        }
        ac.picks.push_back(p);
    }

    // ---- -b / -ub ----------------------------------------------------------------------------
    {
        int b = 0, ub = 0, cctx = 0;
        const char * cell = nullptr;
        const char * cst = nullptr;
        const bool have = pxa_registry_batch_cell(topo, model, level, &b, &ub, &cell, &cst, &cctx) != 0;
        pxa_pick pb, pu;
        pb.flag = "-b"; pu.flag = "-ub";
        pb.evidence = pu.evidence = "";
        if (level == 0) {
            pb.status = pu.status = "OFF"; pb.why = pu.why = "PXA_REFERENCE=1: engine defaults kept";
            pb.evidence = pu.evidence = "";
        } else if (have) {
            ac.n_batch = b; ac.n_ubatch = ub; ac.batch_cell = cell ? cell : ""; ac.batch_cell_ctx = cctx;
            pb.value = std::to_string(b); pu.value = std::to_string(ub);
            pb.status = pu.status = cst ? cst : "MEASURED"; pb.why = pu.why = cell;
            pb.applied = !user.b; pu.applied = !user.ub;
            // the cell's -ub as the FLOOR of a fit walk (Gemma 4 on one 16 GB card): the value is not a number on
            // purpose, so a caller that forwards numeric picks as flags (tools/pxa-launch.py) leaves -ub to the
            // engine, which walks it at load (common.cpp pxa_fastest_ub_init)
            const char * fit_why = nullptr;
            const int    fit     = user.ub ? 0 : pxa_registry_ub_fit_rule(topo, model, level, ub, &fit_why, user.kv_q8);
            if (fit > ub) {
                ac.ub_fit_ceiling = fit;
                pu.value  = "fit " + std::to_string(fit) + ">" + std::to_string(ub);
                pu.status = "RULE";
                pu.why    = fit_why;
            } else if (cst && strcmp(cst, "FIT") == 0 && !user.ub) {
                // a FIT cell (docs/SM75.md): the -ub is a candidate verified at load, not a number to forward
                pu.value = "fit " + std::to_string(ub);
            }
        } else {
            pb.status = "OFF"; pb.why = "no measured cell: the engine default -b is kept";
            pb.evidence = "";
            const char * ub_rule_why = nullptr;
            const int    ub_rule     = user.ub ? 0 : pxa_registry_ub_rule_volta_q8(topo, level, user.n_ctx_value,
                                                                              user.kv_q8, &ub_rule_why);
            // the 27B dense PXQN4 file on one V100 or one P100 (PXA_AUTO_UB_DENSE_2048): -ub 2048 is a ceiling walked down to the
            // stock 512 at load; the value is not a number on purpose (tools/pxa-launch.py then leaves -ub to the engine)
            const char * dense_why = nullptr;
            const int    dense_fit = (user.ub || user.b) ? 0 : pxa_registry_ub_fit_dense(topo, model, level,
                                                                          user.kv_q8 || user.kv, &dense_why);
            if (dense_fit > 0 && ub_rule <= 0) {
                ac.ub_fit_ceiling = dense_fit;
                pu.value = "fit " + std::to_string(dense_fit) + ">512"; pu.status = "RULE"; pu.applied = true;
                const bool dense_p100 = topo.cc.size() == 1 && topo.cc[0] == 600;
                dense_ub_v100 = !dense_p100;
                pu.why = dense_why;
                if (dense_p100) { pu.evidence = ""; } else { pu.evidence = ""; }
            } else if (ub_rule > 0) {
                ac.n_ubatch = ub_rule; ac.ub_rule = ub_rule_why;
                pu.value = std::to_string(ub_rule); pu.status = "RULE"; pu.applied = true;
                pu.why = ub_rule_why; pu.evidence = "";
            } else if (level >= 1 && !user.ub) {
                ac.ub_adaptive = true;
                pu.value = "adaptive"; pu.status = "ADAPTIVE"; pu.applied = true;
                pu.why = "no measured cell for this card set and file: the VRAM ladder picks at load (<= the card-type value)";
                pu.evidence = "";
            } else {
                pu.status = "OFF"; pu.why = "engine default kept"; pu.evidence = "";
            }
        }
        if (user.b)  { pb.status = "USER"; pb.why = "given on the command line"; pb.applied = false; }
        if (user.ub) { pu.status = "USER"; pu.why = "given on the command line"; pu.applied = false; }
        ac.picks.push_back(pb);
        ac.picks.push_back(pu);
    }

    // ---- -c ---------------------------------------------------------------------------------
    // Without -c the engine used the file's trained window (262,144 on Qwen3.8) and, with every
    // layer on the cards, that KV ring does not fit a 16 GiB card. When the -b/-ub cell just
    // picked above has its OWN measured/planned -c (ac.batch_cell_ctx), that number is used - the
    // SAME one tools/pxa-launch.py's matching recipe row passes, one source of truth instead of a
    // second independent formula that can drift from it. Everywhere else the fallback is
    // pxa-launch's anchor, np * 4096 per slot (ANCHOR_CTX_PER_SLOT), capped at the trained window.
    // EITHER WAY, -c is never left smaller than the -b just picked: src/llama.cpp clamps
    // cparams.n_batch to n_ctx, so a smaller auto -c would silently throw away part of the auto
    // -b/-ub pick instead of the engine refusing or saying so (#8382 problem 1: a bare 2x P100
    // boot picked -c 4096 here - np defaults to 1 with no -np given - while pxa-launch passed
    // -c 32768 for the identical cards and file, and the unmatched 4096 quietly clamped the -b
    // 8192 pick down to 4096).
    {
        pxa_pick p;
        p.flag = "-c";
        const int  np        = user.n_parallel > 0 ? user.n_parallel : 1;
        const bool from_cell = ac.batch_cell_ctx > 0;
        int        want      = from_cell ? ac.batch_cell_ctx : np * 4096;
        bool       raised    = false;
        if (ac.n_batch > 0 && want < ac.n_batch) {
            want   = ac.n_batch;   // never let the auto -c pick clamp the auto -b pick away
            raised = true;
        }
        const bool capped = model.n_ctx_train > 0 && want > model.n_ctx_train;
        if (capped) {
            want = model.n_ctx_train;
        }
        if (user.ctx) {
            p.status = "USER"; p.why = "given on the command line";
        } else if (level < 2) {
            p.status = "OFF"; p.why = "config level below ENHANCE: engine default kept (the trained window)";
        } else {
            ac.n_ctx = want;
            p.value = std::to_string(want); p.applied = true;
            if (from_cell) {
                p.status = "MEASURED"; p.evidence = "";
                p.why = std::string("the same cell as -b/-ub (") + ac.batch_cell +
                        "); pxa-launch passes this -c for the identical cards and file";
            } else {
                p.status = "RULE";
                p.why = "-np " + std::to_string(np) + " x 4096 per slot (pxa-launch's anchor; -c N for more)";
            }
            if (raised) {
                p.why += "; raised to the -b just picked above so -c never clamps it away";
            }
            if (capped) {
                p.why += ", capped at the trained window";
            }
        }
        ac.picks.push_back(p);
    }
    // ---- -ctk / -ctv (KV cache type): the FASTEST type that keeps the weights resident ---------------
    // f16 KV is the fastest where it fits (P100 27B one-card file, -c 8192: f16 24.9 vs q4_0 22.4 t/s), but at
    // -c 65536 it is 4.2 GB and tips a 16 GB card from resident into "spill dense ffn_down": 1.8 GB re-read from
    // host RAM over PCIe per graph, decode 24 -> 1.72 t/s, prefill 50 -> 14 (user report 2026-09-30, reproduced
    // on 1x P100, v2026.10.1). So with neither flag given: f16 if weights + f16 KV + compute + margin fit each
    // card, else q8_0 if that fits, else q4_0 (what every published number uses). Estimate, per card (MiB):
    //   weights = file - 1024 (the CPU-side output/embedding buffer) ; KV = 2 x K/V heads x head dim x
    //   bytes/elem x (layers / full_attention_interval) x n_ctx ; + 160 recurrent state ; + 1536 compute buffers
    //   + 512 margin, against total VRAM - 384. Both sides are divided by the card count.
    {
        pxa_pick pk, pv;
        pk.flag = "-ctk"; pv.flag = "-ctv";
        pk.evidence = pv.evidence = "";
        const bool sm_cards = topo.n_dev >= 1 && !topo.cc.empty() &&
                              (topo.cc[0] == 600 || topo.cc[0] == 610 || topo.cc[0] == 700);
        if (user.kv) {
            pk.status = pv.status = "USER"; pk.why = pv.why = "given on the command line";
        } else if (level >= 2 && sm_cards && model.arch == "qwen35" && !getenv("PXA_KV_AUTO_OFF")) {
            const int    n_ctx = user.ctx && user.n_ctx_value > 0 ? user.n_ctx_value
                               : ac.n_ctx > 0 ? ac.n_ctx : model.n_ctx_train;
            size_t vmin = 0;
            for (size_t v : topo.vram_mib) { if (v > 0 && (vmin == 0 || v < vmin)) vmin = v; }
            const int    itv   = model.full_attn_interval > 0 ? model.full_attn_interval : 4;
            const double lay   = model.n_layer > 0 ? (double) (model.n_layer / itv) : 0.0;
            const double kv_el = lay * (double) model.n_head_kv * ((double) model.head_k + (double) model.head_v) * (double) std::max(n_ctx, 0);
            const double n_dev = (double) topo.n_dev;
            const double wt    = std::max(0.0, (double) model.bytes / 1048576.0 - 1024.0) / n_dev;
            const double budget = (double) vmin - 384.0;
            const struct { const char * t; double b; } cand[3] = {{"f16", 2.0}, {"q8_0", 34.0 / 32.0}, {"q4_0", 18.0 / 32.0}};
            const char * pick = "q4_0";
            char why[600] = "";
            if (vmin == 0 || lay <= 0.0 || model.n_head_kv <= 0 || model.head_k <= 0 || n_ctx <= 0 || model.bytes == 0) {
                snprintf(why, sizeof(why), "no VRAM / KV geometry to estimate with: q4_0 (what every published number uses)");
            } else {
                for (int i = 0; i < 3; ++i) {
                    const double kv   = kv_el * cand[i].b / 1048576.0 / n_dev;
                    const double need = wt + kv + 160.0 + 1536.0 + 512.0;
                    if (need <= budget || i == 2) {
                        pick = cand[i].t;
                        snprintf(why, sizeof(why), "-c %d: weights %.0f + %s KV %.0f + 160 state + 1536 compute + 512 margin = %.0f MiB per card vs %.0f MiB budget (%.0f total - 384)%s",
                                 n_ctx, wt, cand[i].t, kv, need, budget, (double) vmin, need <= budget ? "" : "; nothing fits, smallest KV");
                        break;
                    }
                }
            }
            ac.kv_type = pick;
            pk.value = pv.value = pick; pk.status = pv.status = "MEASURED"; pk.applied = pv.applied = true;
            pk.why = pv.why = std::string(why) + "; f16 at 64k spills a 16 GB card's weights to host RAM (1.72 vs 24.1 t/s). "
                              "-ctk/-ctv force one; PXA_KV_AUTO_OFF=1 keeps the engine default f16";
        } else if (level >= 2 && model.arch == "qwen4exp" && !topo.cc.empty() && !getenv("PXA_KV_AUTO_OFF") &&
                   std::all_of(topo.cc.begin(), topo.cc.end(), [](int c) { return c == 600 || c == 700; })) {
            // Flash-Next on P100/V100: every Overdrive number was taken at q8_0 (1x P100 30.70 prose / 37.96 code; 2x P100
            // 35.58 / 36.44; Control's click path 36.86), and f16 KV with MTP was the 9-30 slowdown shape. No silent f16.
            ac.kv_type = "q8_0";
            pk.value = pv.value = "q8_0"; pk.status = pv.status = "MEASURED"; pk.applied = pv.applied = true;
            pk.evidence = pv.evidence = "";
            pk.why = pv.why = "the Flash-Next class on P100/V100: the Overdrive numbers were all measured at q8_0 KV (1x P100 "
                              "30.70/37.96, 2x P100 35.58/36.44 prose/code); -ctk/-ctv force one; PXA_KV_AUTO_OFF=1 keeps f16";
        } else {
            pk.status = pv.status = "OFF"; pk.why = pv.why = "no measured KV pick for this model/card: the engine default f16 is kept";
            pk.evidence = pv.evidence = "";
        }
        ac.picks.push_back(pk);
        ac.picks.push_back(pv);
    }
    // the 27B dense -ub 2048 ceiling (auto-ub-dense27-v100) is an f16-KV rule: when the KV pick above went quantized
    // (the server's default on one 16 GB V100: q4_0), it stands down and the server keeps its adaptive -ub as before
    if (ac.ub_fit_ceiling > 0 && ac.kv_type != nullptr && strcmp(ac.kv_type, "f16") != 0) {
        for (auto & p : ac.picks) {
            if (p.flag == "-ub" && dense_ub_v100) {   // keyed on the rule, not its evidence id (empty in a scrubbed build)
                ac.ub_fit_ceiling = 0;
                p.value = ""; p.status = "OFF"; p.applied = false;
                p.why = std::string("the 27B dense -ub 2048 ceiling is an f16-KV rule and the KV pick here is ") + ac.kv_type +
                        ": the stock / adaptive -ub stays";
            }
        }
    }
    // ---- Nemotron-H MoE: the speculative-verify identity pair -------------------------------------------------------------------------------
    // PXA_FA_VEC_NY1_SPLIT (verify columns of the one-column FA vec kernel take the one-token decode KV split) and PXA_KQMMV_VERIFY_ID_SMALL
    // (small grouped Q8_0 launches keep KQMMV at verify widths 3..4) make a Nemotron 3.5 Super NextN verify bitwise one-token decode (pxa-bi-
    // probe, PXQN2 2x P100: K=2 64/64, K=3 63/63, n_kv ~1860 32/32; without them 0/63 and 0/32). Scoped to this arch: on the 27B W3 MTP P01
    // cell they changed the MTP text and cost -1.1% prose / -2.1% code (#28441), so every other arch keeps its verify path byte-for-byte.
    // Either lever =0 in the environment puts it back (setenv never overwrites).
    if (level >= 1 && model.arch == "nemotron_h_moe") {
        ac.env.emplace_back("PXA_FA_VEC_NY1_SPLIT", "1");
        ac.env.emplace_back("PXA_KQMMV_VERIFY_ID_SMALL", "1");
        pxa_pick p;
        p.flag = "PXA_FA_VEC_NY1_SPLIT";
        p.evidence = "";
        p.value = "on"; p.status = "RULE"; p.applied = true;
        p.why = "Nemotron-H MoE: speculative verify columns bitwise one-token decode (FA decode KV split + small KQMMV launches at widths 3..4, "
                "PXA_KQMMV_VERIFY_ID_SMALL); PXA_FA_VEC_NY1_SPLIT=0 / PXA_KQMMV_VERIFY_ID_SMALL=0 put them back";
        ac.picks.push_back(p);
    }
    // ---- qwen4exp (Hemlock / Flash-Next): the small-launch verify-identity switch (option A)
    // PXA_KQMMV_VERIFY_ID_SMALL on: KQMMV launches of <= 1024 rows (DeltaNet alpha / beta Q8_0) keep the chunked form up to
    // PXA_KQMMV_VERIFY_ID_SMALL_W (8) verify columns, so a K = 5 / 6 / 8 verify forward is bitwise five / six / eight one-token forwards
    // (pxa-bi-probe 2x P100: 20/20, 24/24, 24/24; was 0/30). The same on/off switch as the Nemotron-H row above; every other arch keeps
    // the kernel default (off). =0 in the environment puts it back (setenv never overwrites).
    if (level >= 1 && model.arch == "qwen4exp") {
        ac.env.emplace_back("PXA_KQMMV_VERIFY_ID_SMALL", "1");
        pxa_pick p;
        p.flag = "PXA_KQMMV_VERIFY_ID_SMALL";
        p.evidence = "";
        p.value = "on"; p.status = "RULE"; p.applied = true;
        p.why = "qwen4exp: speculative verify columns bitwise one-token decode (small KQMMV launches up to PXA_KQMMV_VERIFY_ID_SMALL_W "
                "columns); PXA_KQMMV_VERIFY_ID_SMALL=0 puts them back";
        ac.picks.push_back(p);
    }
    // ---- qwen35 on sm_60: the same verify-identity pair --------------------------------------------------------------------
    // 27B W3 PXQN4 on 2x P100 (-sm tensor) with MTP: code greedy512 MTP != plain on every engine since pass 9. pxa-bi-probe: the K=3/4
    // verify forward differed from K one-token forwards in every token (logits + state 0/48): the delta-net beta/alpha (48 rows) and
    // attention k/v (1024 rows) Q8_0 groups decline to the q8_1 MMVQ at widths 3..4 while widths 1..2 run the h2 KQMMV (first divergent
    // node beta-0), and past n_kv ~256 the FA vec split of a 2..4-column verify differs from decode (K=2/4 0/60 at n_kv ~600). With
    // both levers K=2/3/4 are bitwise 48/48 (n_kv ~100) and 60/60 (n_kv ~600). The 27B's sm_60 KQMMV h2 path is what has the small
    // groups; V100 (KQMMV off, its own FA route) keeps its proven verify path. =0 in the environment turns either off.
    if (level >= 1 && model.arch == "qwen35" && std::any_of(topo.cc.begin(), topo.cc.end(), [](int c) { return c == 600; })) {
        ac.env.emplace_back("PXA_FA_VEC_NY1_SPLIT", "1");
        ac.env.emplace_back("PXA_KQMMV_VERIFY_ID_SMALL", "1");
        // the width the 27B measured: the small grouped launches at verify widths 2..4 (the base cap), not the
        // qwen4exp widening of launches <= 1024 rows to 8 (PXA_KQMMV_VERIFY_ID_SMALL_W default)
        ac.env.emplace_back("PXA_KQMMV_VERIFY_ID_SMALL_W", "4");
        pxa_pick p;
        p.flag = "PXA_KQMMV_VERIFY_ID_SMALL";
        p.evidence = "";
        p.value = "on"; p.status = "RULE"; p.applied = true;
        p.why = "qwen35 on sm_60: speculative verify columns bitwise one-token decode (small Q8_0 KQMMV groups at widths 3..4 + the FA "
                "decode KV split, PXA_FA_VEC_NY1_SPLIT; widths kept at 4, PXA_KQMMV_VERIFY_ID_SMALL_W=4): MTP greedy output == plain greedy; PXA_KQMMV_VERIFY_ID_SMALL=0 / "
                "PXA_FA_VEC_NY1_SPLIT=0 put them back";
        ac.picks.push_back(p);
    }
    return ac;
}

// ---- cost lines (2026-09-28) ---------------------------------------------------
// OWNER ORDER "PXA ENHANCE must always set the best defaults": a user who drives llama-server by
// hand (no launcher) and types a value that was MEASURED slower than the ENHANCE pick keeps that
// value, but is told in one line what it costs and what the better value is. Only values with a
// ledger row behind the comparison are named here; an unmeasured difference says nothing.
// Trigger case: a 2x P100 log package with PXA_TSPLIT_REDUCE=off and PXA_TSPLIT_REDUCE_PREFILL=1
// exported by hand on an older build.
static void pxa_autoconfig_costs(const pxa_topology & topo, const pxa_model_info & model,
                                 const pxa_user_set & user, int level, int posture, pxa_autoconfig & ac) {
    if (level < 2) {
        return;   // PXA_REFERENCE=1 / PXA_ENHANCE=0 are explicit rollbacks, not mistakes
    }
    const int  n       = topo.n_dev;
    const bool same_cc = topo.same_cc() && !topo.cc.empty();
    const int  cc      = same_cc ? topo.cc[0] : 0;
    // the split in effect: the user's -sm, else the pick above
    const bool tensor  = user.sm ? user.sm_value == PXA_SM_TENSOR : ac.split == PXA_SM_TENSOR;
    char buf[640];

    // (1) -sm layer typed on a set where ENHANCE picks the tensor split
    if (user.sm && user.sm_value == PXA_SM_LAYER) {
        pxa_user_set alt = user;
        alt.sm = false;
        alt.sm_value = -1;
        const pxa_autoconfig a2 = pxa_autoconfig_resolve_picks(topo, model, alt, level, posture);
        if (a2.split == PXA_SM_TENSOR) {
            const char * num = n == 4 ? "4x P100: decode +5..41% and prefill @22.6k about 2x over layer (27B PXQN4/PXQ4)"
                             : cc == 700 ? "V100 pair: decode +12..28%, prefill +3..4% over layer (27B PXQN4/PXQN4S8/PXQN5)"
                             : "P100 pair: decode +17..27%, prefill @22.6k +44% over layer (27B PXQN4/PXQN4S8/PXQN5)";
            snprintf(buf, sizeof(buf), "-sm layer on %dx %s (arch %s, tier %s): the tensor split measured faster here "
                                       "(%s); drop -sm to let ENHANCE pick it", n, pxa_card_name(cc),
                     model.arch.c_str(), model.tier.c_str(), num);
            ac.costs.push_back({ "-sm layer", n == 4 ? "" : "", buf });
        }
    }

    // (2) PXA_TSPLIT_REDUCE=off: not named here. The reduce itself prints that warning at its first
    // reduce, where it knows whether the group peers (P100
    // pair PXQ4 24.7 against 26.5/26.9 t/s decode); one line per value, not two.

    // (3) the fused route forced onto prefill-width reduces, pre-empting the two-device DMA route
    const char * rpf = getenv("PXA_TSPLIT_REDUCE_PREFILL");
    if (tensor && n == 2 && rpf && atoi(rpf) != 0) {
        snprintf(buf, sizeof(buf), "PXA_TSPLIT_REDUCE_PREFILL=%s: the fused route is tried first at prefill width and "
                                   "pre-empts the two-device prefill route (PXA_TSPLIT_PF): V100 pair 27B PXQN4, 14.8k "
                                   "prompt, 739.5/745.5 against 820.0/824.7 t/s (-10%% prefill), decode and greedy "
                                   "output unchanged; unset it", rpf);
        ac.costs.push_back({ std::string("PXA_TSPLIT_REDUCE_PREFILL=") + rpf, "", buf });
    }
}

pxa_autoconfig pxa_autoconfig_resolve(const pxa_topology & topo, const pxa_model_info & model,
                                      const pxa_user_set & user, int level, int posture) {
    // PXA_AUTOCONFIG=0 (2026-10-16): the ONE off switch for the whole resolve in
    // every tool (llama-server, llama-cli, llama-bench, every common tool): it resolves as the
    // REFERENCE level, so no -sm / -ngl / -c / KV pick and no lever env is applied. The rest of
    // the engine keeps its level (unlike PXA_REFERENCE=1 / PXA_ENHANCE=0, which roll back kernels too).
    if (pxa_env_off("PXA_AUTOCONFIG")) {
        level = 0;
    }
    pxa_autoconfig ac = pxa_autoconfig_resolve_picks(topo, model, user, level, posture);
    pxa_autoconfig_costs(topo, model, user, level, posture, ac);
    return ac;
}

// ---------------------------------------------------------------------------------------------
// Banner and JSON
// ---------------------------------------------------------------------------------------------
static std::string pxa_topo_str(const pxa_topology & t) {
    std::string s = std::to_string(t.n_dev) + " card(s)";
    for (size_t i = 0; i < t.cc.size(); ++i) {
        s += (i == 0 ? ": sm_" : ", sm_") + std::to_string(t.cc[i] / 10);
    }
    return s + " [" + t.source + "]";
}

void pxa_registry_banner(FILE * out, const pxa_topology & topo, const pxa_model_info & model,
                         const pxa_autoconfig & ac) {
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
    size_t n = 0;
    const pxa_lever_decl * cat = pxa_lever_catalog(&n);
    size_t n_rule = 0, n_on = 0;
    for (size_t i = 0; i < n; ++i) {
        n_rule += strcmp(cat[i].status, "rule") == 0;
        n_on   += strcmp(cat[i].status, "default-on") == 0;
    }
    fprintf(out, "PXA_REGISTRY: %zu levers declared (%zu default-on, %zu by rule); level %s; %s; model arch=%s tier=%s "
                 "experts=%d kv_heads=%d%s\n",
            n, n_on, n_rule, ggml_pxa_config_level_name(), pxa_topo_str(topo).c_str(),
            model.arch.empty() ? "?" : model.arch.c_str(), model.tier.empty() ? "none" : model.tier.c_str(),
            model.n_expert, model.n_head_kv, model.n_shards > 1 ? " (split file)" : "");
    if (pxa_env_off("PXA_AUTOCONFIG")) {
        fprintf(out, "PXA_REGISTRY: PXA_AUTOCONFIG=0: every pick and lever env below is stood down (engine defaults)\n");
    }
    for (const auto & kv : model.pxa_kv) {
        if (kv.first.rfind("pxa.quantizer.", 0) == 0 || kv.first.rfind("pxa.format.", 0) == 0 ||
            kv.first.rfind("pxa.codec.", 0) == 0 || kv.first.rfind("pxa.recipe.", 0) == 0) {
            fprintf(out, "PXA_REGISTRY: file %s = %s\n", kv.first.c_str(), kv.second.c_str());
        }
    }
    for (const auto & p : ac.picks) {
        fprintf(out, "PXA_REGISTRY: %-3s %-8s [%s%s%s] %s\n", p.flag.c_str(),
                p.value.empty() ? "(yours)" : p.value.c_str(), p.status.c_str(),
                p.evidence.empty() ? "" : "; ", p.evidence.c_str(), p.why.c_str());
    }
    for (const auto & c : ac.costs) {
        if (c.evidence.empty()) {    // a release build carries no evidence ids
            fprintf(out, "PXA_REGISTRY: COST %s %s\n", c.what.c_str(), c.text.c_str());
        } else {
            fprintf(out, "PXA_REGISTRY: COST %s [%s] %s\n", c.what.c_str(), c.evidence.c_str(), c.text.c_str());
        }
    }
    int n_closed_on = 0, n_closed_kept = 0;   // closed levers: counted, never named (the log is user-facing)
    for (const auto & e : ac.env) {
        const char * cur = getenv(e.first.c_str());
        const bool   kept = cur && strcmp(cur, e.second.c_str()) != 0;
        if (pxa_lever_is_closed(e.first)) {
            (kept ? n_closed_kept : n_closed_on) += 1;
            continue;
        }
        fprintf(out, "PXA_REGISTRY: env %s=%s (%s)\n", e.first.c_str(), cur ? cur : e.second.c_str(),
                kept ? "yours, kept" : "set by the -sm pick");
    }
    if (n_closed_on + n_closed_kept > 0) {
        fprintf(out, "PXA_REGISTRY: Overdrive: %d tuned library lever(s) on%s\n", n_closed_on,
                n_closed_kept > 0 ? (", " + std::to_string(n_closed_kept) + " kept at your value").c_str() : "");
    }
    // levers set in the environment: named, with their row, or flagged when no row declares them (a closed one: counted)
    int n_closed_user = 0;
    for (char ** e = environ; e && *e; ++e) {
        if (strncmp(*e, "PXA_", 4) != 0 && strncmp(*e, "PXQ_", 4) != 0) {
            continue;
        }
        const char * eq = strchr(*e, '=');
        if (!eq) {
            continue;
        }
        const std::string name(*e, eq - *e);
        bool ours = false;
        for (const auto & x : ac.env) {
            ours |= x.first == name;
        }
        if (ours) {
            continue;
        }
        if (pxa_lever_is_closed(name)) {
            ++n_closed_user;
            continue;
        }
        const pxa_lever_decl * d = pxa_lever_find(name.c_str());
        if (d) {
            fprintf(out, "PXA_REGISTRY: set %s=%s (default %s; %s)\n", name.c_str(), eq + 1, d->deflt, d->status);
        } else if (ggml_pxqn_lever_builtin(name.c_str())) {
            fprintf(out, "PXA_REGISTRY: set %s=%s (built-in: read by the closed PXQN code)\n", name.c_str(), eq + 1);
        } else {
            // name only: an undeclared PXA_* variable can hold anything (a token, a password), and these
            // lines reach logs and bug reports (beta 2 review, 2026-10-19)
            fprintf(out, "PXA_REGISTRY: WARNING %s is not a lever this build declares - a typo does nothing (value not shown)\n",
                    name.c_str());
        }
    }
    if (n_closed_user > 0) {
        fprintf(out, "PXA_REGISTRY: Overdrive: %d tuned library lever(s) set in the environment (read by the closed PXQN code)\n",
                n_closed_user);
    }
    fflush(out);
}

std::string pxa_autoconfig_json(const pxa_topology & topo, const pxa_model_info & model,
                                const pxa_autoconfig & ac, int level) {
    using json = nlohmann::ordered_json;
    json j;
    j["schema"] = "pxa-autoconfig/1";
    j["level"]  = level == 0 ? "REFERENCE" : level == 1 ? "DEFAULT" : "ENHANCE";
    json jt;
    jt["n"] = topo.n_dev;
    jt["cc"] = topo.cc;
    jt["source"] = topo.source;
    j["topology"] = jt;
    json jm;
    jm["path"] = model.path;
    jm["ok"] = model.ok;
    jm["arch"] = model.arch;
    jm["tier"] = model.tier;
    jm["n_expert"] = model.n_expert;
    jm["n_head_kv"] = model.n_head_kv;
    jm["n_layer"] = model.n_layer;
    jm["n_pxq_tensors"] = model.n_pxq;
    jm["n_shards"] = model.n_shards;
    jm["bytes"] = model.bytes;
    json kv = json::object();
    for (const auto & x : model.pxa_kv) {
        kv[x.first] = x.second;
    }
    jm["pxa"] = kv;
    j["model"] = jm;
    json picks = json::object();
    for (const auto & p : ac.picks) {
        json q;
        q["value"] = p.value;
        q["status"] = p.status;
        q["applied"] = p.applied;
        q["evidence"] = p.evidence;
        q["why"] = p.why;
        picks[p.flag[0] == '-' ? p.flag.substr(1) : p.flag] = q;
    }
    j["picks"] = picks;
    json env = json::object();
    for (const auto & e : ac.env) {
        // a library-only row of the Overdrive set leaves the process only as its alias (PXA_OVERDRIVE_LIB=1 sets the same
        // rows, setenv no-overwrite): the launcher / Control re-export this env, print it and write it into scripts
        if (e.second == "1" && ggml_pxqn_overdrive_is_member(e.first.c_str())) {
            env["PXA_OVERDRIVE_LIB"] = "1";
        } else {
            env[e.first] = e.second;
        }
    }
    j["env"] = env;
    j["ts_even"] = ac.ts_even;
    json costs = json::array();
    for (const auto & c : ac.costs) {
        costs.push_back({ { "what", c.what }, { "evidence", c.evidence }, { "text", c.text } });
    }
    j["costs"] = costs;
    size_t n = 0;
    pxa_lever_catalog(&n);
    j["levers_declared"] = n;
    return j.dump();
}

// ---- the registry in every tool (see pxa-registry.h; orders #22182 F1 + F3, 2026-10-16) -------------------
bool pxa_autoconfig_off() {
    return pxa_env_off("PXA_AUTOCONFIG");
}

bool pxa_lever_is_closed(const std::string & name) {
    // the levers only the closed PXQN library reads (its own list; the Overdrive set's library-only rows are in it)
    return ggml_pxqn_lever_builtin(name.c_str()) != 0;
}

std::string pxa_lever_shown(const std::string & name) {
    return pxa_lever_is_closed(name) ? std::string("an Overdrive library lever") : name;
}

int pxa_registry_env_apply(const std::vector<std::pair<std::string, std::string>> & env, bool primary,
                           std::vector<std::string> * differs) {
    const char * u = getenv("PXA_HS_ENV_UNION");
    const bool unite = u && atoi(u) != 0;
    int n_set = 0;
    for (const auto & e : env) {
        const char * cur = getenv(e.first.c_str());
        const bool unset = cur == nullptr || cur[0] == '\0';
        if (primary || unite) {
            if (unset) {
                setenv(e.first.c_str(), e.second.c_str(), /*overwrite =*/ 1);
                ++n_set;
            }
            if (primary) {
                continue;
            }
        }
        if (!primary && (unset ? !unite : e.second != cur) && differs) {
            differs->push_back(pxa_lever_is_closed(e.first) ? pxa_lever_shown(e.first)
                               : e.first + "=" + e.second + " (process: " + (unset ? std::string("unset") : std::string(cur)) + ")");
        }
    }
    return n_set;
}

int pxa_tool_kind_of(const char * argv0) {
    if (argv0 == nullptr) {
        return PXA_TOOL_CLI;
    }
    const char * b = strrchr(argv0, '/');
    b = b ? b + 1 : argv0;
    if (strstr(b, "perplexity") || strstr(b, "imatrix") || strstr(b, "kl-div") || strstr(b, "kld")) {
        return PXA_TOOL_GATE;
    }
    if (strstr(b, "llama-bench")) {
        return PXA_TOOL_BENCH;
    }
    if (strstr(b, "server")) {
        return PXA_TOOL_SERVER;
    }
    return PXA_TOOL_CLI;
}

pxa_user_set pxa_user_set_lever_only(int split_mode, bool ts_given, bool kv_q8) {
    pxa_user_set us;
    us.sm = true;
    us.sm_value = split_mode;
    us.b = us.ub = us.fa = us.ngl = us.ctx = us.kv = true;
    us.ts = ts_given;
    us.kv_q8 = kv_q8;
    return us;
}

int pxa_tool_ctx_pick(int pick_n_ctx, bool user_ctx, int n_ctx_train) {
    if (user_ctx || pick_n_ctx <= 0 || n_ctx_train <= 0 || pick_n_ctx < n_ctx_train) {
        return 0;
    }
    return pick_n_ctx;
}

std::vector<std::pair<std::string, std::string>> pxa_env_common(
        const std::vector<std::vector<std::pair<std::string, std::string>>> & per_instance,
        std::vector<std::string> * dropped) {
    std::vector<std::pair<std::string, std::string>> out;
    if (per_instance.empty()) {
        return out;
    }
    auto value_in = [](const std::vector<std::pair<std::string, std::string>> & v, const std::string & k) -> const std::string * {
        for (const auto & e : v) {
            if (e.first == k) return &e.second;
        }
        return nullptr;
    };
    std::set<std::string> seen;
    for (size_t i = 0; i < per_instance.size(); ++i) {
        for (const auto & e : per_instance[i]) {
            if (!seen.insert(e.first).second) {
                continue;
            }
            bool common = true;
            for (const auto & other : per_instance) {
                const std::string * v = value_in(other, e.first);
                common = common && v != nullptr && *v == e.second;
            }
            if (common) {
                out.push_back(e);
            } else if (dropped) {
                for (size_t j = 0; j < per_instance.size(); ++j) {
                    const std::string * v = value_in(per_instance[j], e.first);
                    dropped->push_back(pxa_lever_is_closed(e.first) ? pxa_lever_shown(e.first) + " (instance " + std::to_string(j) + ")"
                                       : e.first + "=" + (v ? *v : std::string("unset")) + " (instance " + std::to_string(j) + ")");
                }
            }
        }
    }
    return out;
}
