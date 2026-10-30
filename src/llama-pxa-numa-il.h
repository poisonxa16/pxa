// pxa / PXA_NUMA_INTERLEAVE -- authored by PXA Network (https://pxanetwork.com).
// llama-pxa-numa-il.h -- spread the host pages of a host-expert layout over every memory node the process may use.
//
// An Overdrive layout reads most expert bytes of every token from host RAM on the CPU. When the CPU threads do not sit on the
// node the cards hang off (PXA_NUMA_BIND found no allowed CPU there, or the cards span nodes), PXA_NUMA_BIND sets no memory
// policy and every page lands on the node of the thread that first touched it: one socket's memory controllers carry the whole
// stream. With MPOL_INTERLEAVE over all allowed nodes the stream is served by both sockets' controllers (the remote half over the
// inter-socket link), which is what `numactl --interleave=all` does for a whole process.
// Measured (2026-10-21): GLM-5.3-Flash bb5 on 1x V100 + host experts, threads on node 1, card on
// node 0: +9.3 / +17.6 / +9.0 % decode (prose / code / chat), greedy output identical (placement only, no numerics change).
//
// PXA_NUMA_INTERLEAVE: unset = the rule (glm5next with expert tensors in a host buffer, no memory policy set yet, >= 2 allowed
// nodes with memory); =0 off; =1 any model with a host buffer, still only when no memory policy is set; off at PXA_REFERENCE.
// The policy is set on the loading thread before the model buffers are allocated and kept, so the pinned host buffers, the loader
// threads and the CPU thread pool created later all inherit it (the same scope as numactl / the A/B shim it replaces).
#pragma once

#include "llama-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#endif

// mode: -1 unset (rule), 0 off, 1 forced
static inline int llama_pxa_numa_interleave_mode() {
    const char * e = getenv("PXA_NUMA_INTERLEAVE");
    if (e && *e) return atoi(e) != 0 ? 1 : 0;
    const char * r = getenv("PXA_REFERENCE");
    if (r && atoi(r) != 0) return 0;
    return -1;
}

// glm5next: the model arch is the rule's scope; host_experts: an *_exps tensor lives in host RAM (a host buffer type, or the
// PXA_XCACHE cold stack's pinned stream buffer)
static inline void llama_pxa_numa_interleave_apply(bool glm5next, bool host_experts, bool any_host) {
#if defined(__linux__)
    const int mode = llama_pxa_numa_interleave_mode();
    if (mode == 0) return;
    if (mode < 0 && !(glm5next && host_experts)) return;
    if (mode > 0 && !any_host) return;
    // an existing policy wins: PXA_NUMA_BIND's preferred node (threads on the cards' node), numactl --membind/--preferred, ...
    int cur = -1;
    if (syscall(SYS_get_mempolicy, &cur, nullptr, 0UL, nullptr, 0UL) != 0) return;
    if (cur != 0 /* MPOL_DEFAULT */) {
        LLAMA_LOG_INFO("PXA_NUMA_INTERLEAVE: a memory policy is already set (mode %d); host pages stay where it puts them\n", cur);
        return;
    }
    // allowed nodes (Mems_allowed_list of this process, i.e. the cpuset) that have memory
    std::string list;
    if (FILE * f = fopen("/proc/self/status", "r")) {
        char line[4096];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "Mems_allowed_list:", 18) == 0) { list = line + 18; break; }
        }
        fclose(f);
    }
    unsigned long mask[16] = {};
    int n_nodes = 0;
    std::string names;
    const char * q = list.c_str();
    while (*q) {
        while (*q == ' ' || *q == '\t' || *q == ',') ++q;
        if (*q < '0' || *q > '9') break;
        char * end = nullptr;
        const long a = strtol(q, &end, 10);
        long b = a;
        q = end;
        if (*q == '-') { b = strtol(q + 1, &end, 10); q = end; }
        for (long n = a; n <= b && n < 1024; ++n) {
            char path[96];
            snprintf(path, sizeof(path), "/sys/devices/system/node/node%ld/meminfo", n);
            long long kb = 0;
            if (FILE * f = fopen(path, "r")) {
                char line[256];
                while (fgets(line, sizeof(line), f)) {
                    const char * p = strstr(line, "MemTotal:");
                    if (p) { kb = atoll(p + 9); break; }
                }
                fclose(f);
            }
            if (kb <= 0) continue;
            mask[n / 64] |= 1UL << (n % 64);
            names += (n_nodes ? "," : "") + std::to_string(n);
            ++n_nodes;
        }
    }
    if (n_nodes < 2) return;   // one node: nothing to spread
    const long rc = syscall(SYS_set_mempolicy, 3 /* MPOL_INTERLEAVE */, mask, (unsigned long) (8 * sizeof(mask)));
    LLAMA_LOG_INFO("PXA_NUMA_INTERLEAVE: host pages interleaved over nodes %s (%s%s)%s\n", names.c_str(),
                   mode > 0 ? "forced" : "rule: glm5next host experts", ", =0 off", rc == 0 ? "" : ": set_mempolicy failed, not applied");
#else
    (void) glm5next; (void) host_experts; (void) any_host;
#endif
}
