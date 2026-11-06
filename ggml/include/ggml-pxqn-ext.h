// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
// ggml-pxqn-ext.h -- PXA_CLOSED_HOOK: optional role tables of the closed PXQN library (libggml-pxqn). Open.
//
// A role table is looked up by its role name through ONE optional library symbol, after the main table
// (ggml-pxqn-api.h) passed its checks. Every table starts with { uint32 version; uint32 size; } and grows by
// appending members at the end; every argument struct starts with its own size. Without the library, with an
// older one, or when the library declines the role, the getter answers NULL and the engine runs its own route.
// An entry point answers 0 = done, -1 = declined (nothing was launched or written: the caller's own route runs).
#pragma once

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_PXQN_EXT_SYM "ggml_pxqn_get_ext_api"
typedef const void * (*ggml_pxqn_get_ext_api_fn)(const char * role, uint32_t version);

struct ggml_pxqn_ext_head {
    uint32_t version;
    uint32_t size;
};

// the role table `role` at `version`, NULL when absent; the table's size must be at least min_size
GGML_API const void * ggml_pxqn_ext_table(const char * role, uint32_t version, uint32_t min_size);

// ---- role "gemv": a matrix-vector launch over up to GGML_PXQN_GEMV_MAX_MATS weight matrices that share one input ----
#define GGML_PXQN_GEMV_ROLE     "gemv"
#define GGML_PXQN_GEMV_VERSION  1u
#define GGML_PXQN_GEMV_MAX_MATS 4

enum ggml_pxqn_gemv_flag {
    GGML_PXQN_GEMV_GATE_UP       = 1u << 0,   // mats[0] = up, mats[1] = gate; one output (mats[0].dst), act(gate) * up
    GGML_PXQN_GEMV_EXACT_COLUMNS = 1u << 1,   // every output column must equal a one-column launch of that column, bit for bit
    GGML_PXQN_GEMV_SMALL         = 1u << 2,   // the launch reached the engine through its small-launch rule
    GGML_PXQN_GEMV_WIDE_OK       = 1u << 3,   // the engine's own route may use its wide form for this launch
    GGML_PXQN_GEMV_DRAFT_HEAD    = 1u << 4,   // the weights are the MTP draft head (GGML_PXQN_DRAFT_HEAD_PREFIX): the
                                              // library may answer from its own lower-precision copy (drafts only; the
                                              // target's verify decides every token). Advisory: a library that does not
                                              // serve it declines, or offers the launch on without the bit.
};
// weight tensors whose name starts with this are the MTP draft head (src/llama.cpp's shortlist copies)
#define GGML_PXQN_DRAFT_HEAD_PREFIX "output_shortlist"

struct ggml_pxqn_gemv_mat {
    const void  * w;       // row 0 of the weights
    float       * dst;     // output column 0
    const float * bias;    // NULL, or a per-row vector added to every column
    int64_t       nb01;    // bytes per weight row
    int64_t       sd;      // output column stride (floats)
    int64_t       rows;
};

struct ggml_pxqn_gemv_args {
    uint32_t      size;    // sizeof(struct ggml_pxqn_gemv_args) of the caller
    int32_t       type;    // enum ggml_type of the weights
    int32_t       device;
    int32_t       cc;      // compute capability of the device (600, 700, ...)
    int32_t       nsm;     // multiprocessor count of the device
    int32_t       ny;      // input / output columns
    int64_t       K;       // input length (elements per column)
    const float * x;       // input column 0 (f32)
    int64_t       sx;      // input column stride (floats)
    int32_t       n_mats;
    uint32_t      flags;   // enum ggml_pxqn_gemv_flag
    float         limit;   // GATE_UP: the op's limit (< 1e-6 = the plain form)
    struct ggml_pxqn_gemv_mat mats[GGML_PXQN_GEMV_MAX_MATS];
};

struct ggml_pxqn_gemv_api {
    uint32_t version;   // GGML_PXQN_GEMV_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_gemv_api)
    // 0 = launched on `stream` (a cudaStream_t), -1 = declined
    int (*launch)(const struct ggml_pxqn_gemv_args * a, void * stream);
};

// the "gemv" table, NULL when absent (read once)
GGML_API const struct ggml_pxqn_gemv_api * ggml_pxqn_gemv_api_get(void);
// a->launch through the table; -1 without one
GGML_API int ggml_pxqn_gemv_launch(const struct ggml_pxqn_gemv_args * a, void * stream);

// ---- role "cache": request-scoped state of the expert cache (src/llama-pxa-xcache.cpp) ----
// The engine owns the routing counters, the slot copies, the maps and the files; the role decides. Without the table the
// engine keeps its hot set across requests as it is and runs no request-scoped step.
#define GGML_PXQN_CACHE_ROLE    "cache"
#define GGML_PXQN_CACHE_VERSION 1u

enum ggml_pxqn_cache_start {          // what a main-model batch starts (batch)
    GGML_PXQN_CACHE_START_NONE   = 0,
    GGML_PXQN_CACHE_START_STRICT = 1, // the batch starts a sequence at position 0 on an empty cache
    GGML_PXQN_CACHE_START_PROMPT = 2, // a request start the role asks for (the role's own step only)
};

enum ggml_pxqn_cache_act {            // what a select asks of the engine
    GGML_PXQN_CACHE_ACT_NONE = 0,     // nothing to do (no finish follows)
    GGML_PXQN_CACHE_ACT_KEEP = 1,     // keep the set as it is (finish follows)
    GGML_PXQN_CACHE_ACT_MOVE = 2,     // settle the copies in flight, then make the moves of plan (finish follows)
};

struct ggml_pxqn_cache_layer {
    int32_t il;          // model layer
    int32_t n_expert;
    int32_t n_slots;     // hot slots
};

struct ggml_pxqn_cache_args {
    uint32_t     size;         // sizeof(struct ggml_pxqn_cache_args) of the caller
    int32_t      n_layers;
    const struct ggml_pxqn_cache_layer * layers;
    double       n_used;       // expert selections per token
    int32_t      maxw;         // batches of at most maxw tokens are decode ticks
    uint64_t     model_size;   // bytes of the model file (0: unknown)
    const char * model_path;
};

struct ggml_pxqn_cache_view {
    uint32_t                size;       // sizeof(struct ggml_pxqn_cache_view) of the caller
    const uint32_t * const * counters;  // per layer: n_expert routing counters, then the device block's 8 words
    const uint8_t  * const * resident;  // per layer: n_expert flags, 1 = in a hot slot once the copies in flight land
    const double   * const * prior;     // per layer: the model's expected selections per token (n_expert), or NULL
};

struct ggml_pxqn_cache_move {
    int32_t layer;       // index into ggml_pxqn_cache_args.layers
    int32_t e_in;        // goes into the hot slot e_out leaves
    int32_t e_out;
};

struct ggml_pxqn_cache_host {         // the engine's own counts, for the role's summary line
    uint32_t size;
    int32_t  on;
    uint64_t waits, dropped_jobs, dropped_moves;
    double   ms_all, ms_moves;
};

#define GGML_PXQN_CACHE_NSTAT 32
struct ggml_pxqn_cache_stats {        // the role's counters (their meaning is the role's)
    uint32_t size;
    uint32_t n;
    double   v[GGML_PXQN_CACHE_NSTAT];
};

struct ggml_pxqn_cache_api {
    uint32_t     version;      // GGML_PXQN_CACHE_VERSION
    uint32_t     size;         // sizeof(struct ggml_pxqn_cache_api)
    const char * file_suffix;  // the role's state file: the learned counts' path with this in place of their suffix
    // NULL = the role does not run this session (msg: why); else msg = the boot line
    void * (*create)(const struct ggml_pxqn_cache_args * a, char * msg, size_t n);
    void   (*destroy)(void * st);
    // the state file's bytes: the entries read (msg: the line), -1 = ignored (msg: why)
    int    (*load)(void * st, const void * data, size_t n, char * msg, size_t nmsg);
    // the state to write (owned by st, valid until the next call): its bytes, 0 = nothing to write
    size_t (*save)(void * st, const void ** data);
    // one main-model batch (on = the role still runs): enum ggml_pxqn_cache_start
    int    (*batch)(void * st, int on, int strict, int n_tokens);
    // a request start / the running request's end (the engine read its counters)
    void   (*start)(void * st, const struct ggml_pxqn_cache_view * v, int prompt);
    void   (*end)(void * st, const struct ggml_pxqn_cache_view * v);
    // one tick of n_tokens: 1 = call select now (before the tick's own work)
    int    (*tick)(void * st, int n_tokens);
    // enum ggml_pxqn_cache_act
    int    (*select)(void * st, const struct ggml_pxqn_cache_view * v);
    // ACT_MOVE on the settled set: the moves in order (owned by st) and each one's value
    size_t (*plan)(void * st, const uint8_t * const * resident, const struct ggml_pxqn_cache_move ** moves, const double ** value);
    // the moves made (ok: all of them and the tables check out)
    void   (*moved)(void * st, int ok, size_t n_moves, double value);
    // after ACT_MOVE: the policy's prior for layer (n_expert into out); 0 = none
    int    (*prior)(void * st, int32_t layer, const double * model_prior, double * out);
    // after KEEP / MOVE: on = the role still runs, outcome 0 kept / 1 moved / 2 failed, ms = the moves' time; msg = the log line
    void   (*finish)(void * st, int on, int outcome, double ms, char * msg, size_t n);
    // the role stops for the session
    void   (*halt)(void * st);
    void   (*stats)(void * st, struct ggml_pxqn_cache_stats * out);
    void   (*summary)(void * st, const struct ggml_pxqn_cache_host * h, char * msg, size_t n);
};

// the "cache" table, NULL when absent (read once)
GGML_API const struct ggml_pxqn_cache_api * ggml_pxqn_cache_api_get(void);

// ---- role "learn": the book of the learned expert counts (<model>.expert-counts.learned.csv, src/llama-pxa-xcache.cpp) ----
// The engine counts the routings and writes the csv rows; the role keeps the record beside them and decides what a save carries.
// Without the table the engine neither reads nor writes learned counts (the curated map only).
#define GGML_PXQN_LEARN_ROLE    "learn"
#define GGML_PXQN_LEARN_VERSION 1u

enum ggml_pxqn_learn_arm {
    GGML_PXQN_LEARN_ARMED       = 0,
    GGML_PXQN_LEARN_NO_PATH     = 1,   // nowhere writable
    GGML_PXQN_LEARN_NOT_LEARNED = 2,   // the path is not a learned csv
    GGML_PXQN_LEARN_OFF         = 3,   // learned counts are off this session
};

struct ggml_pxqn_learn_api {
    uint32_t version;   // GGML_PXQN_LEARN_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_learn_api)
    // nonzero: learned counts are neither read nor written this session (msg: the line, may be NULL)
    int    (*off)(char * msg, size_t n);
    // the book of the learned csv at path ("" = nowhere writable); boot_tokens >= 0: this boot planned from learned counts
    // counted from that many tokens. NULL + *rc (enum ggml_pxqn_learn_arm) when not armed; msg = the line
    void * (*arm)(const char * path, double boot_tokens, int * rc, char * msg, size_t n);
    void   (*release)(void * book);
    double (*carry)(const void * book);          // the share of the boot's learned counts each save carries (0 = none)
    double (*carry_tokens)(const void * book);   // the tokens those counts were counted from
    void   (*stop)(void * book);                 // the session ends without a save
    // one save: write_csv(ud, path) writes the rows (nonzero = written), then the record follows (tokens = what the rows were
    // counted from). The sessions on record after it, -1 = nothing saved
    int    (*save)(void * book, int (*write_csv)(void * ud, const char * path), void * ud, double tokens, int shutdown);
    int    (*counted)(const void * book);        // this session is on record
    // the record beside the learned csv: 1 = tokens / sessions read, 0 = none
    int    (*record)(const char * csv, double * tokens, int * sessions);
    // the learned table of a save (the arguments of ggml_pxqn_xcache_learn_build, plus carry = the share of prior kept):
    // 0 = out_counts filled, 1 = nothing observed, 2 = *wait_layer is not ready, -1 = bad arguments
    int    (*rows)(int n_layer, int n_expert, int n_main, const double * prior, double carry, const int * obs_il, int n_obs,
                   const uint32_t * obs, const uint8_t * routed, uint64_t * out_counts, int * wait_layer);
};

// the "learn" table, NULL when absent (read once)
GGML_API const struct ggml_pxqn_learn_api * ggml_pxqn_learn_api_get(void);

// ---- role "place": the library may name the device of a graph node the engine joins from per-device parts ----
#define GGML_PXQN_PLACE_ROLE    "place"
#define GGML_PXQN_PLACE_VERSION 1u

enum ggml_pxqn_place_site {
    GGML_PXQN_PLACE_JOIN = 1,   // one node joining n_parts per-device parts of an output, over n_device devices
};

enum ggml_pxqn_place_flag {
    GGML_PXQN_PLACE_DRAFT    = 1u << 0,   // the output belongs to the draft head
    GGML_PXQN_PLACE_PARTS_OK = 1u << 1,   // the engine can also read the parts back unjoined (answer GGML_PXQN_PLACE_NO_JOIN)
};

// a pick that is not a device: no join node at all, each part is a graph output read back from its own device.
// Only ever answered when the caller set GGML_PXQN_PLACE_PARTS_OK.
#define GGML_PXQN_PLACE_NO_JOIN (-2)

struct ggml_pxqn_place_args {
    uint32_t size;       // sizeof(struct ggml_pxqn_place_args)
    int32_t  site;       // enum ggml_pxqn_place_site
    int32_t  n_parts;
    int32_t  n_device;
    uint32_t flags;      // enum ggml_pxqn_place_flag
    int32_t  reserved;
};

struct ggml_pxqn_place_api {
    uint32_t version;   // GGML_PXQN_PLACE_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_place_api)
    // the device index (0 .. n_device-1) that owns the node, -1 = the engine's own choice, or GGML_PXQN_PLACE_NO_JOIN
    // (only when the caller set GGML_PXQN_PLACE_PARTS_OK).
    // A library answers -1 for a site or flag bit it does not know.
    int (*pick)(const struct ggml_pxqn_place_args * a);
};

// the "place" table, NULL when absent (read once)
GGML_API const struct ggml_pxqn_place_api * ggml_pxqn_place_api_get(void);
// a->pick through the table; -1 without one
GGML_API int ggml_pxqn_place_pick(const struct ggml_pxqn_place_args * a);

// ---- role "xchg": one card's half of a tensor-split all-reduce (every rank's partial summed in ONE rank order, in
// place). The engine builds the exchange (its staging areas, arrival lines and ring counter: `task`, its own layout,
// checked by task_size) and offers the launch; the role answers 0 = launched, -1 = declined (nothing launched, the
// engine's own kernel runs). A role's answer depends only on what every rank of the group shares (the arguments below,
// the environment, the group's devices), so the ranks of one reduce always take the same route.
#define GGML_PXQN_XCHG_ROLE    "xchg"
#define GGML_PXQN_XCHG_VERSION 1u

struct ggml_pxqn_xchg_args {
    uint32_t     size;        // sizeof(struct ggml_pxqn_xchg_args) of the caller
    int32_t      device;      // this half's CUDA device
    int32_t      type;        // enum ggml_type of the reduced tensor
    int32_t      nrank;       // ranks in the group
    int64_t      nelem;       // elements of this launch (one chunk of the reduce)
    int64_t      ne0;         // row length (elements)
    float      * data;        // the partial in, the sum out (in place)
    const void * task;        // the engine's exchange descriptor
    uint32_t     task_size;   // its size
    int32_t      token;       // the arrival token when the descriptor carries no ring counter
    int        * err_flag;    // device-visible error flag
    size_t       area_bytes;  // bytes of one staging area (one slot, one source rank)
    int32_t      lines;       // arrival lines per rank and slot (the most blocks a launch may use)
    int32_t      nrows;       // rows of the whole reduced tensor (the same for every chunk)
};

struct ggml_pxqn_xchg_api {
    uint32_t version;   // GGML_PXQN_XCHG_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_xchg_api)
    int (*launch)(const struct ggml_pxqn_xchg_args * a, void * stream);
    // the most rows of one reduce (row length ne0, `lines` arrival lines per rank and slot) whose multi-row epilogue
    // (reduce + [add] + RMS norm + RHT) the library runs with this exchange; 0 = none (the engine plans as before).
    // The same answer on every rank of a group.
    int (*rows)(int32_t nrank, int64_t ne0, int32_t lines);
};

// the "xchg" table, NULL when absent (read once)
GGML_API const struct ggml_pxqn_xchg_api * ggml_pxqn_xchg_api_get(void);
// a->launch through the table; -1 without one
GGML_API int ggml_pxqn_xchg_launch(const struct ggml_pxqn_xchg_args * a, void * stream);
// rows() through the table; 0 without one
GGML_API int ggml_pxqn_xchg_rows(int32_t nrank, int64_t ne0, int32_t lines);

// ---- role "fa": the library may name which of the engine's own flash-attention kernels runs one node ----
// The engine asks only where its own rule has a choice and every kernel it may be told serves the node; it checks
// the answer against its support query again and runs its own route for -1, an unknown value or an unsupported one.
#define GGML_PXQN_FA_ROLE    "fa"
#define GGML_PXQN_FA_VERSION 1u

enum ggml_pxqn_fa_route {
    GGML_PXQN_FA_OWN      = -1,   // the engine's own rule
    GGML_PXQN_FA_VEC_F32  = 1,    // the engine's vec kernel, f32 accumulation
    GGML_PXQN_FA_TILE_F32 = 2,    // the engine's tile kernel, f32 accumulation
};

struct ggml_pxqn_fa_args {
    uint32_t size;      // sizeof(struct ggml_pxqn_fa_args)
    int32_t  cc;        // compute capability of the node's device (ggml-cuda numbering, 610 = sm_61)
    int32_t  d_k;       // K head size
    int32_t  d_v;       // V head size
    int64_t  ncols;     // query columns of the node
    int64_t  n_kv;      // KV length of the node
    int32_t  type_k;    // enum ggml_type of K
    int32_t  type_v;    // enum ggml_type of V
    int32_t  prec;      // enum ggml_prec of the node
    uint32_t flags;     // 0 (a library answers -1 for a bit it does not know)
};

struct ggml_pxqn_fa_api {
    uint32_t version;   // GGML_PXQN_FA_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_fa_api)
    // a value of enum ggml_pxqn_fa_route; -1 = the engine's own rule
    int (*pick)(const struct ggml_pxqn_fa_args * a);
};

// the "fa" table, NULL when absent (read once)
GGML_API const struct ggml_pxqn_fa_api * ggml_pxqn_fa_api_get(void);
// a->pick through the table; -1 without one or for a value outside enum ggml_pxqn_fa_route
GGML_API int ggml_pxqn_fa_pick(const struct ggml_pxqn_fa_args * a);

// ---- role "cbook": the codebook expert tier (GGML_TYPE_PXQNC): per-model tables + the decode entries ----
// The engine hands the library the file's pxa.pxqnc.* keys untouched (the library parses and checks them and keeps the
// tables: their layout, the decode and its kernels are the library's), then binds the bytes of every codebook-tier weight
// tensor (host or device addresses) to that model; every decode entry finds the table of a weight from its address. One
// registry per process (several models may be open). Without the table a file with such tensors is refused at load.
#define GGML_PXQN_CBOOK_ROLE       "cbook"
#define GGML_PXQN_CBOOK_VERSION    1u
#define GGML_PXQN_CBOOK_KEY_PREFIX "pxa.pxqnc."

static inline bool ggml_pxqn_cbook_type(enum ggml_type t) { return t == GGML_TYPE_PXQNC; }

// one GGUF key/value as the file has it: a scalar's data points at the value, a string's at its NUL-terminated text, an
// array's at its n elements of arr_type (a string array: n const char *)
struct ggml_pxqn_cbook_kv {
    const char * key;
    int32_t      type;       // enum gguf_type
    int32_t      arr_type;   // enum gguf_type of the elements (arrays only)
    uint64_t     n;          // elements (1 for a scalar)
    const void * data;
};

struct ggml_pxqn_cbook_api {
    uint32_t version;   // GGML_PXQN_CBOOK_VERSION
    uint32_t size;      // sizeof(struct ggml_pxqn_cbook_api)
    // the model's tables from its keys; NULL with a plain sentence in err when the keys are refused
    void * (*model_open)(const struct ggml_pxqn_cbook_kv * kv, int32_t n_kv, char * err, int32_t errlen);
    // unbinds every tensor of the model and frees its tables (host and device copies); call after its buffers are gone
    void   (*model_close)(void * model);
    // binds the bytes [data, data + nbytes) of weight tensor `name` (type, ne[0..3]) to the model: 0 = ok, -1 = refused (err)
    int    (*bind)(void * model, const char * name, int32_t type, const int64_t * ne, const void * data, size_t nbytes,
                   char * err, int32_t errlen);
    // drops the binding of the model's tensor that starts at `data` (its bytes are about to be freed or moved): 0 = done
    int    (*unbind)(void * model, const void * data);
    // 1 = the address lies inside a bound tensor
    int    (*bound)(const void * data);
    // CPU: row `row` of the bound 2-D slice at `data` (k = ne0) -> dst[0..k), f32, in the stored (rotated) basis
    void   (*deq_row)(int32_t type, const void * data, int64_t row, int64_t k, float * dst);
    // CUDA (NULL in a library without CUDA); cuda_ctx = the backend context (its layout is checked by the CUDA table)
    int    (*cuda_supports)(const struct ggml_tensor * op);                  // 1 = this MUL_MAT_ID is served
    int    (*cuda_mul_mat_id)(void * cuda_ctx, struct ggml_tensor * dst);   // 0 = done (every slot written), -1 = declined
    // the routed-expert rotation absorbed into the n_mm MUL_MAT_IDs that read `rht` (adjacent, its only readers): 0 = the
    // RHT (its output written too) and every MUL_MAT_ID done, -1 = declined (nothing launched)
    int    (*cuda_rht_mul_mat_id)(void * cuda_ctx, struct ggml_tensor * rht, struct ggml_tensor * const * mm, int32_t n_mm);
    // one line for the load log
    const char * (*describe)(const void * model);
};

// the "cbook" table, NULL when absent (read once)
GGML_API const struct ggml_pxqn_cbook_api * ggml_pxqn_cbook_api_get(void);

#ifdef __cplusplus
}
#endif
