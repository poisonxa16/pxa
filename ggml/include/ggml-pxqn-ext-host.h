// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
// ggml-pxqn-ext-host.h -- PXA_CLOSED_HOOK: role tables of the closed PXQN library (libggml-pxqn) asked by host-side
// engine code (common/). Open. Same rules as ggml-pxqn-ext.h: looked up through ggml_pxqn_ext_table(); without the
// library, with an older one, or when the library declines, the answer is NULL / -1 and the engine keeps its own rule.
#pragma once

#include "ggml-pxqn-ext.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---- role "chain": the library may end a draft chain before its next row ----
#define GGML_PXQN_CHAIN_ROLE    "chain"
#define GGML_PXQN_CHAIN_VERSION 1u

struct ggml_pxqn_chain_args {
    uint32_t size;     // sizeof(struct ggml_pxqn_chain_args)
    int32_t  row;      // index of the row about to be computed (>= 1)
    int32_t  n_rows;   // the chain's cap for this step
    int32_t  n_kept;   // rows kept so far (>= 1)
    float    p_min;    // the stage's floor
    float    p_last;   // probability of the last kept row
    float    p_all;    // product of the kept rows' probabilities
    uint32_t flags;    // none in version 1; a library answers -1 for a bit it does not know
};

struct ggml_pxqn_chain_api {
    uint32_t version;   // GGML_PXQN_CHAIN_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_chain_api)
    // 1 = compute the row, 0 = end the chain before it, -1 = the engine's own rule
    int (*next)(const struct ggml_pxqn_chain_args * a);
};

// the "chain" table, NULL when absent (read once)
GGML_API const struct ggml_pxqn_chain_api * ggml_pxqn_chain_api_get(void);
// a->next through the table; -1 without one
GGML_API int ggml_pxqn_chain_next(const struct ggml_pxqn_chain_args * a);

// ---- role "sched": the library may arm scheduler options for one scheduler of the engine ----
#define GGML_PXQN_SCHED_ROLE    "sched"
#define GGML_PXQN_SCHED_VERSION 1u

enum ggml_pxqn_sched_site {
    GGML_PXQN_SCHED_DRAFT = 1,   // the scheduler of the MTP draft companion context (draft + accepted-row update graphs)
};

enum ggml_pxqn_sched_flag {
    GGML_PXQN_SCHED_TSPLIT = 1u << 0,   // the scheduler runs the PXA tensor split
};

enum ggml_pxqn_sched_opt {
    GGML_PXQN_SCHED_ASYNC_INPUTS = 1u << 0,   // ggml_backend_sched_set_pxa_async_inputs(sched, true)
    GGML_PXQN_SCHED_KEEP_GRAPH   = 1u << 1,   // the context keeps its MTP reuse record across decodes (the end-of-decode
                                              // scheduler reset is skipped while it holds one), so a draft step can reuse
                                              // the previous step's graph
};
#define GGML_PXQN_SCHED_OPTS_KNOWN ((uint32_t) (GGML_PXQN_SCHED_ASYNC_INPUTS | GGML_PXQN_SCHED_KEEP_GRAPH))

struct ggml_pxqn_sched_args {
    uint32_t size;       // sizeof(struct ggml_pxqn_sched_args)
    int32_t  site;       // enum ggml_pxqn_sched_site
    int32_t  n_device;   // GPU devices the scheduler runs on
    uint32_t flags;      // enum ggml_pxqn_sched_flag; a library answers 0 for a bit it does not know
    int32_t  reserved[2];
};

struct ggml_pxqn_sched_api {
    uint32_t version;   // GGML_PXQN_SCHED_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_sched_api)
    // a mask of enum ggml_pxqn_sched_opt to arm; 0 = the engine's own rule
    uint32_t (*pick)(const struct ggml_pxqn_sched_args * a);
};

// the "sched" table, NULL when absent (read once)
GGML_API const struct ggml_pxqn_sched_api * ggml_pxqn_sched_api_get(void);
// a->pick through the table, masked to the options this engine knows; 0 without one
GGML_API uint32_t ggml_pxqn_sched_pick(const struct ggml_pxqn_sched_args * a);

// ---- role "upload": the library may set how many rows of a graph input the scheduler copies to a device ----
#define GGML_PXQN_UPLOAD_ROLE    "upload"
#define GGML_PXQN_UPLOAD_VERSION 1u

enum ggml_pxqn_upload_site {
    GGML_PXQN_UPLOAD_KQ_MASK      = 1,   // the attention mask input of a narrow (decode / verify width) graph
    GGML_PXQN_UPLOAD_KQ_MASK_ROWS = 2,   // reserved[0] = the live rows (> 1) of that mask: the leading rows the device copy
                                         // needs (the engine uses it only when >= the live rows), 0 = the whole tensor
    GGML_PXQN_UPLOAD_KQ_MASK_FILL = 3,   // reserved[0] = the live rows (> 1), reserved[1] = the engine's mask mode:
                                         // 1 = the host may skip filling the pad rows, 0 = fill them
};

struct ggml_pxqn_upload_args {
    uint32_t size;      // sizeof(struct ggml_pxqn_upload_args)
    int32_t  site;      // enum ggml_pxqn_upload_site
    int32_t  reserved[4];
};

struct ggml_pxqn_upload_api {
    uint32_t version;   // GGML_PXQN_UPLOAD_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_upload_api)
    // the site's mode: 0 = the engine's own rule (the whole input), > 0 = the mode the engine's site documents
    int32_t (*mode)(const struct ggml_pxqn_upload_args * a);
};

// the "upload" table, NULL when absent (read once)
GGML_API const struct ggml_pxqn_upload_api * ggml_pxqn_upload_api_get(void);
// a->mode through the table; 0 without one (or for an answer < 0)
GGML_API int32_t ggml_pxqn_upload_mode(const struct ggml_pxqn_upload_args * a);

// ---- role "host": per-site answers for host-side bookkeeping of the decode loop (2026-11-03) ----
// Every site's answer 0 = the engine's own (plain) rule; the engine documents what a > 0 answer lets it do.
#define GGML_PXQN_HOST_ROLE    "host"
#define GGML_PXQN_HOST_VERSION 1u

enum ggml_pxqn_host_site {
    GGML_PXQN_HOST_CKPT_INTERVAL_SKIP = 1,   // 1 = the server's generation-interval checkpoint test may return before its
                                             //     full-cache position scan when the slot cannot be due yet
    GGML_PXQN_HOST_KV_CELL_MAX        = 2,   // 1 = the KV cache's cell_max may answer from its epoch-validated memo,
                                             // 2 = the same, every memo answer cross-checked against the full scan
    GGML_PXQN_HOST_KQ_MASK_THREADS    = 3,   // r0 = n_kv of a decode mask fill: the threads that fill it (<= 1 = one)
    GGML_PXQN_HOST_KQ_MASK_COPY       = 4,   // r0 = bytes of one device copy of a graph's attention mask: 0 = the blocking
                                             //     copy, 1 = enqueued on the split backend's stream (the cards' copies
                                             //     overlap), 2 = enqueued on a side stream, the first reader waits for it
};

struct ggml_pxqn_host_args {
    uint32_t size;      // sizeof(struct ggml_pxqn_host_args)
    int32_t  site;      // enum ggml_pxqn_host_site
    int64_t  r0;        // the site's argument (see the site)
    int64_t  reserved[3];
};

struct ggml_pxqn_host_api {
    uint32_t version;   // GGML_PXQN_HOST_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_host_api)
    int32_t (*mode)(const struct ggml_pxqn_host_args * a);
};

// the "host" table, NULL when absent (read once)
GGML_API const struct ggml_pxqn_host_api * ggml_pxqn_host_api_get(void);
// the site's answer through the table; 0 without one (or for an answer < 0)
GGML_API int32_t ggml_pxqn_host_mode(int32_t site, int64_t r0);

#ifdef __cplusplus
}
#endif
