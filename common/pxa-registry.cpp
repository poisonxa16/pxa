// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// PXA core step 3: the lever registry and the engine's own serve-flag defaults. See pxa-registry.h.

#include "pxa-registry.h"

#include "ggml.h"
#include "ggml-pxqn-levers.h"
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
// The rules
// ---------------------------------------------------------------------------------------------

// Arches and tiers the tensor split is a DEFAULT for (the launcher's TSPLIT_AUTO_ARCHES /
// TSPLIT_AUTO_TIERS, which this mirrors exactly: tools/pxa-launch.py resolve_auto_split()). The
// capability check in src/llama.cpp admits more than this; a default needs a measurement, an
// admission only needs the builders to be complete.
static bool pxa_tsplit_auto_arch(const std::string & a) { return a == "qwen35"; }
// PXQN4 / PXQN4S8 / PXQN5 (the 2-card PXQ-Next sizes) joined PXQ4 on 2026-09-27.
// Evidence: the dense 27B on one binary (bigq-pp speedcheck, burst-logs/bigq-pp/speedcheck-table.md),
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
    if (level < 2 || topo.n_dev <= 0 || !topo.same_cc()) {
        return 0;   // mixed sets have no measured cell (their lever is -ts)
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

int pxa_registry_ub_walk(int ub, int origin, size_t reserve,
                         const std::function<pxa_ub_attempt(int)> & build,
                         const std::function<void()> & discard, std::string * trail) {
    char buf[160];
    for (bool first = true; ub > 0; first = false) {
        const pxa_ub_attempt a = build(ub);
        if (a.allocated) {
            // the ladder's own first pick is kept exactly as before; everything else must leave the reserve
            const bool check = reserve > 0 && !(origin == PXA_UB_LADDER && first);
            if (!check || a.free_min >= reserve) {
                return ub;
            }
            snprintf(buf, sizeof(buf), "-ub %d left %zu MiB free on CUDA%d; ", ub, a.free_min >> 20, a.worst_dev);
            if (trail) *trail += buf;
            discard();
        } else {
            if (!a.alloc_failure) {
                return 0;   // refused for a reason a smaller -ub cannot fix; the engine already said which
            }
            snprintf(buf, sizeof(buf), "-ub %d did not allocate; ", ub);
            if (trail) *trail += buf;
        }
        ub = pxa_registry_ub_next_rung(ub);
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

// Past a pair the tensor split is a DEFAULT only where it was measured faster (2026-09-27, lane
// launch-defaults, todo tsplit-4card-default): 4 identical P100s (sm_60), dense 27B, one binary,
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
        p.evidence = "launcher-auto-tensor-split";
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
            p.value = "layer"; p.status = "RULE"; p.evidence = "tsplit-4card-default";
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
            p.value = "layer"; p.status = "RULE"; p.evidence = "bug-280";
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
        } else if (!pxa_tsplit_auto_arch(model.arch)) {
            p.value = "layer"; p.status = "RULE";
            snprintf(buf, sizeof(buf), "arch '%s' has no tensor-split measurement behind a default (qwen35 does)",
                     model.arch.empty() ? "?" : model.arch.c_str());
            p.why = buf;
        } else if (!pxa_tsplit_auto_tier(model.tier)) {
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
            p.value = "tensor"; p.status = "MEASURED"; p.evidence = "tsplit-4card-default";
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
        p.evidence = "convfuse-single-device-2026-09-29";
        if (tensor_eff) {
            p.value = "on"; p.status = "MEASURED"; p.applied = false;
            p.why = "tensor split: the fused delta-net conv cluster measured +4.4% (2x P100) / +5.3% (2x V100) decode";
        } else {
            ac.env.emplace_back("PXA_DN_CONVFUSE", "0");
            // MEASURED only where the A/B above was run: one P100 or one V100. Any other no-tensor-split
            // topology (a layer-split pair, 3-4 cards, a 1080 Ti) takes the same pick carried over from the
            // one-card runs, so it is INFERRED. Label only: the pick (off) and its application are unchanged.
            const bool one_card_ab = topo.n_dev == 1 && !topo.cc.empty() && (topo.cc[0] == 600 || topo.cc[0] == 700);
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
            p.evidence = "rel-gate2-14803-2026-10-05";
            ac.env.emplace_back("PXA_PXQN_SM60_RBSHAPE", "0");
            p.value = "off"; p.status = "MEASURED"; p.applied = true;
            p.why = "P100 tensor split: the sm_60 row-block shapes cost the speculative verify -4.4..-4.9% server decode "
                    "(2x P100, 27B PXQN4); off restores v2026.10.2 speed; PXA_PXQN_SM60_RBSHAPE=1 puts it back";
            ac.picks.push_back(p);
        }
    }

    // ---- PXA_PXQN_RHT_FUSE (RHT->GEMV fusion) is off on one V100 ---------------------------------
    // measured 2026-09-29, 1x V100 (GPU 2), llama-bench tg128 -fa 1 q4_0 KV, REPS 3, 3 rounds interleaved:
    //   one-card 27B file  fused 32.3-33.4 vs unfused 34.8-35.0 t/s (+5..8%); PXQN4 ladder 31.0/32.6 vs 34.8/33.8;
    //   greedy512 sha identical (435d10f612f952d1). The fusion first fired in v2026.10.1 (d68c73cabc).
    // Pairs and P100 are unmeasured, so only the single-V100 case flips. PXA_PXQN_RHT_FUSE=1/0 in the environment wins.
    if (level >= 2 && topo.n_dev == 1 && !topo.cc.empty() && topo.cc[0] == 700) {
        pxa_pick p;
        p.flag = "PXA_PXQN_RHT_FUSE";
        p.evidence = "rhtfuse-single-v100-2026-09-29";
        ac.env.emplace_back("PXA_PXQN_RHT_FUSE", "0");
        p.value = "off"; p.status = "MEASURED"; p.applied = true;
        p.why = "1x V100: the RHT->GEMV fusion measured -5..8% decode (32.3-33.4 vs 34.8-35.0 t/s), sha identical; "
                "PXA_PXQN_RHT_FUSE=1 puts it back";
        ac.picks.push_back(p);
    }

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
        pb.evidence = pu.evidence = "prefill-auto-ubatch-model-aware";
        if (level == 0) {
            pb.status = pu.status = "OFF"; pb.why = pu.why = "PXA_REFERENCE=1: engine defaults kept";
            pb.evidence = pu.evidence = "";
        } else if (have) {
            ac.n_batch = b; ac.n_ubatch = ub; ac.batch_cell = cell; ac.batch_cell_ctx = cctx;
            pb.value = std::to_string(b); pu.value = std::to_string(ub);
            pb.status = pu.status = cst ? cst : "MEASURED"; pb.why = pu.why = cell;
            pb.applied = !user.b; pu.applied = !user.ub;
        } else {
            pb.status = "OFF"; pb.why = "no measured cell: the engine default -b is kept";
            pb.evidence = "";
            const char * ub_rule_why = nullptr;
            const int    ub_rule     = user.ub ? 0 : pxa_registry_ub_rule_volta_q8(topo, level, user.n_ctx_value,
                                                                              user.kv_q8, &ub_rule_why);
            if (ub_rule > 0) {
                ac.n_ubatch = ub_rule; ac.ub_rule = ub_rule_why;
                pu.value = std::to_string(ub_rule); pu.status = "RULE"; pu.applied = true;
                pu.why = ub_rule_why; pu.evidence = "bug-207";
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
                p.status = "MEASURED"; p.evidence = "prefill-auto-ubatch-model-aware";
                p.why = std::string("the same cell as -b/-ub (") + (ac.batch_cell ? ac.batch_cell : "") +
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
        pk.evidence = pv.evidence = "kv-f16-host-spill-2026-09-30";
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
        } else {
            pk.status = pv.status = "OFF"; pk.why = pv.why = "no measured KV pick for this model/card: the engine default f16 is kept";
            pk.evidence = pv.evidence = "";
        }
        ac.picks.push_back(pk);
        ac.picks.push_back(pv);
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
            ac.costs.push_back({ "-sm layer", n == 4 ? "tsplit-4card-default" : "launcher-auto-tensor-split", buf });
        }
    }

    // (2) PXA_TSPLIT_REDUCE=off: not named here. The reduce itself prints that warning at its first
    // reduce, where it knows whether the group peers (tsplit-reduce-off-warn: P100
    // pair PXQ4 24.7 against 26.5/26.9 t/s decode); one line per value, not two.

    // (3) the fused route forced onto prefill-width reduces, pre-empting the two-device DMA route
    const char * rpf = getenv("PXA_TSPLIT_REDUCE_PREFILL");
    if (tensor && n == 2 && rpf && atoi(rpf) != 0) {
        snprintf(buf, sizeof(buf), "PXA_TSPLIT_REDUCE_PREFILL=%s: the fused route is tried first at prefill width and "
                                   "pre-empts the two-device prefill route (PXA_TSPLIT_PF): V100 pair 27B PXQN4, 14.8k "
                                   "prompt, 739.5/745.5 against 820.0/824.7 t/s (-10%% prefill), decode and greedy "
                                   "output unchanged; unset it", rpf);
        ac.costs.push_back({ std::string("PXA_TSPLIT_REDUCE_PREFILL=") + rpf, "tsplit-prefill-pf-supersedes", buf });
    }
}

pxa_autoconfig pxa_autoconfig_resolve(const pxa_topology & topo, const pxa_model_info & model,
                                      const pxa_user_set & user, int level, int posture) {
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
        fprintf(out, "PXA_REGISTRY: COST %s [%s] %s\n", c.what.c_str(), c.evidence.c_str(), c.text.c_str());
    }
    for (const auto & e : ac.env) {
        const char * cur = getenv(e.first.c_str());
        fprintf(out, "PXA_REGISTRY: env %s=%s (%s)\n", e.first.c_str(), cur ? cur : e.second.c_str(),
                cur && strcmp(cur, e.second.c_str()) != 0 ? "yours, kept" : "set by the -sm pick");
    }
    // levers set in the environment: named, with their row, or flagged when no row declares them
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
        const pxa_lever_decl * d = pxa_lever_find(name.c_str());
        if (d) {
            fprintf(out, "PXA_REGISTRY: set %s=%s (default %s; %s)\n", name.c_str(), eq + 1, d->deflt, d->status);
        } else if (ggml_pxqn_lever_builtin(name.c_str())) {
            fprintf(out, "PXA_REGISTRY: set %s=%s (built-in: read by the closed PXQN code)\n", name.c_str(), eq + 1);
        } else {
            fprintf(out, "PXA_REGISTRY: WARNING %s=%s is not a lever this build declares - a typo does nothing\n",
                    name.c_str(), eq + 1);
        }
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
        env[e.first] = e.second;
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
