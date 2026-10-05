// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
#pragma once

// =============================================================================================
// PXA_EXPLAIN_PAGE_v1 (2026-09-25, showcase lane, PXA core step 3 follow-on): the engine already
// decides -sm/-b/-ub/-fa/-ngl and prints why (common/pxa-registry.{h,cpp}, PXA_EXPLAIN=1/levers).
// This file is the small, header-only, PURE layer that turns that plus a couple of live numbers
// the server already tracks into the one JSON object `GET /pxa/explain` serves (server.cpp) and
// `GET /pxa` renders. Nothing here is on the decode path: every function is read-only formatting
// of numbers the caller already has (getenv(), the lever catalog, or plain ints/strings the caller
// measured). No device I/O, no allocation beyond building the JSON object, no token ever passes
// through this file. See tests/test-pxa-explain.cpp for the schema/unit coverage (CPU only, no
// CUDA, no server, no model).
// =============================================================================================

#include "pxa-registry.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
#include <string>

// One CUDA device the caller already queried (ggml_backend_cuda_get_device_{count,description,
// memory} -- server.cpp is the only place that legitimately calls those; this function just
// formats the three numbers). Pure: no device touched here.
inline nlohmann::ordered_json pxa_explain_card_entry(int index, const std::string & name, int cc,
                                                      size_t free_bytes, size_t total_bytes) {
    nlohmann::ordered_json j;
    const size_t used_bytes = total_bytes > free_bytes ? total_bytes - free_bytes : 0;
    j["index"]           = index;
    j["name"]            = name;
    j["sm"]              = cc;                 // 60, 70, ... (pxa_topology.cc uses the same unit)
    j["vram_total_mib"]  = total_bytes >> 20;
    j["vram_free_mib"]   = free_bytes  >> 20;
    j["vram_used_mib"]   = used_bytes  >> 20;
    return j;
}

// Every PXA_*/PXQ_* lever the catalog declares, split by what decides it RIGHT NOW: a value the
// user set in the environment that differs from the row's shipped default (reported with both
// values, so a stale override is obvious), or a "default-on" row sitting at its default (nothing
// overrode it). Rows whose status is lever-off/rule/diagnostic/tool/site are not "active defaults"
// the way a default-on knob is, so they are counted (declared/rule_count) but not listed twice --
// a rule/diagnostic row DOES still show up under user_set if the environment sets it away from its
// default (e.g. PXA_EXPLAIN itself). This mirrors what pxa_registry_banner() already prints to
// stderr on every boot, as structured JSON instead of a log line, and needs no live pxa_autoconfig
// result to have run first.
inline nlohmann::ordered_json pxa_explain_levers_json() {
    nlohmann::ordered_json j;
    size_t n = 0;
    const pxa_lever_decl * cat = pxa_lever_catalog(&n);
    nlohmann::ordered_json default_on = nlohmann::ordered_json::array();
    nlohmann::ordered_json user_set   = nlohmann::ordered_json::array();
    size_t n_default_on = 0, n_rule = 0;
    for (size_t i = 0; i < n; ++i) {
        const pxa_lever_decl & d = cat[i];
        const bool is_default_on = strcmp(d.status, "default-on") == 0;
        n_default_on += is_default_on ? 1 : 0;
        n_rule       += strcmp(d.status, "rule") == 0 ? 1 : 0;
        const char * cur = getenv(d.name);
        if (cur && strcmp(cur, d.deflt) != 0) {
            nlohmann::ordered_json u;
            u["name"]    = d.name;
            u["value"]   = cur;
            u["default"] = d.deflt;
            u["status"]  = d.status;
            user_set.push_back(u);
        } else if (is_default_on) {
            nlohmann::ordered_json e;
            e["name"]     = d.name;
            e["value"]    = d.deflt;
            e["scope"]    = d.scope;
            e["evidence"] = d.evidence;
            default_on.push_back(e);
        }
    }
    j["declared"]          = n;
    j["default_on_count"]  = n_default_on;
    j["rule_count"]        = n_rule;
    j["default_on"]        = default_on;
    j["user_set"]          = user_set;
    return j;
}

// The full /pxa/explain object: pxa_autoconfig_json()'s own object (schema "pxa-autoconfig/1":
// level/topology/model/picks/env/ts_even/levers_declared -- parsed back so the caller merges
// rather than double-encodes) plus what this page adds: live per-card VRAM, the active-lever
// breakdown above, the server's own live decode/prefill/speculation numbers, and engine identity.
// `cards` and `live` are plain JSON the caller already built (server.cpp queries CUDA devices and
// the metrics task queue; tests/test-pxa-explain.cpp passes synthetic arrays instead) -- pulling
// the merge out here means the SHAPE of the final object is covered by a CPU-only unit test with
// no running server and no CUDA device required.
inline std::string pxa_explain_build(const std::string & autoconfig_json,
                                     const nlohmann::ordered_json & cards,
                                     const nlohmann::ordered_json & live,
                                     int engine_build, const std::string & engine_commit,
                                     const std::string & model_alias) {
    nlohmann::ordered_json j = nlohmann::ordered_json::parse(autoconfig_json, nullptr,
                                                              /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) {
        j = nlohmann::ordered_json::object();
    }
    j["schema"]      = "pxa-explain/1";
    j["engine"]      = { {"build", engine_build}, {"commit", engine_commit} };
    j["model_alias"] = model_alias;
    j["cards"]       = cards;
    j["levers"]      = pxa_explain_levers_json();
    j["live"]        = live;
    return j.dump();
}
