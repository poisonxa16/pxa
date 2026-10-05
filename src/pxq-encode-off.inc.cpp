// pxq-encode-off.inc.cpp — the no-encoder build of the PXQ write side.
//
// PXQ files are PRODUCED by a separate tool. This build reads and runs them — the loader, the
// panel dequant and the CUDA kernels are all present and unchanged — it just cannot write one.
// llama_model_quantize() refuses a PXQ target before any of the entry points below could be
// reached, so every definition here is inert: each returns the same value its real counterpart
// returns for a non-PXQ target, which is what keeps Q4_K_M and every other type byte-identical
// to an encoder build. See src/llama-quantize.cpp for where the refusal is raised.

// The on-disk layout constants (panel header and slab sizes). These are the FORMAT, not the
// encoder: the loader and the panel dequant read the same headers, so they stay in the tree.
// The encoder build picks them up through the tier codecs instead.
#include "ggml-pxq1-tables.h"
#include "ggml-pxq2-tables.h"
#include "ggml-pxq3-tables.h"
#include "ggml-pxq6-tables.h"

static std::atomic<int64_t> g_pxq_imx_dead_cols{0};

static bool pxq_imatrix_column_usable(const float *, int64_t) {
    return false;
}

static unsigned pxa_pxq_native_mask() {
    return 0u;
}

static bool pxq4_legacy_native_class(const std::string &) {
    return false;
}

static bool pxa_pxq_policy_explicit(pxa_pxq_policy_id *) {
    return false;
}

static pxa_pxq_policy_id pxa_pxq_policy_for_tier(pxa_pxq_tier) {
    return PXA_POLICY_UNIFORM;
}

static bool pxa_pxq_policy_active_for(pxa_pxq_tier) {
    return false;
}

static bool pxa_pxq_backbone_native_class(const std::string &, const ggml_tensor *) {
    return false;
}

// GGML_TYPE_COUNT is "not mine — leave it to the legacy pipeline", which is what the resolver
// answers for every tensor when the run names no PXQ tier.
static ggml_type pxa_pxq_backbone_type_resolve(const std::string &, const ggml_tensor *, pxa_pxq_tier,
                                               bool) {
    return GGML_TYPE_COUNT;
}

static ggml_type pxa_pxq_backbone_type(const std::string &, const ggml_tensor *, pxa_pxq_tier,
                                       bool) {
    return GGML_TYPE_COUNT;
}

static bool pxq4_tensor_eligible(const std::string &, const ggml_tensor *) {
    return false;
}

static ggml_type pxa_pxq_landing_type(const std::string &, const ggml_tensor *,
                                      ggml_type, pxa_pxq_tier) {
    return GGML_TYPE_COUNT;
}

// Codebook accessors. The write loop reads them only while stamping a PXQ file's provenance
// KVs, which this build never reaches.
static inline bool pxq_ceil_v2_enabled()        { return false; }
static inline bool pxq2_v3_enabled()            { return false; }
static inline const float * pxq6_book_q()       { return nullptr; }
static inline const float * pxq6_sub_q(int)     { return nullptr; }
static inline const float * pxq6r_book_q()      { return nullptr; }
static inline const float * pxq6r_sub_q()       { return nullptr; }
static inline const float * pxq2_book_q()       { return nullptr; }
static inline const float * pxq2_sub_q()        { return nullptr; }
static inline const float * pxq3_book_q()       { return nullptr; }
static inline const float * pxq3_sub_q()        { return nullptr; }
static inline const float * pxq1_book_q()       { return nullptr; }
static inline const float * pxq1_sub_q()        { return nullptr; }

// The five tier codecs. Unreachable: a PXQ target is refused at llama_model_quantize().
#define PXQ_NO_ENCODER_ABORT() \
    GGML_ABORT("PXQ files are made with the separate `pxq-quantize` tool — see " PXA_PXQ_QUANTIZER_URL)

static void pxq1_quantize_tensor(const float *, uint8_t *, int64_t, int64_t, int64_t,
                                 const float *, int64_t, int)            { PXQ_NO_ENCODER_ABORT(); }
static void pxq2_quantize_tensor(const float *, uint8_t *, int64_t, int64_t, int64_t,
                                 const float *, int64_t, int)            { PXQ_NO_ENCODER_ABORT(); }
static void pxq3_quantize_tensor(const float *, uint8_t *, int64_t, int64_t, int64_t,
                                 const float *, int64_t, int)            { PXQ_NO_ENCODER_ABORT(); }
static void pxq6r_quantize_tensor(const float *, uint8_t *, int64_t, int64_t, int64_t,
                                  const float *, int64_t, int)           { PXQ_NO_ENCODER_ABORT(); }
static void pxq6_quantize_tensor(const float *, uint8_t *, int64_t, int64_t, int64_t,
                                 const float *, int64_t, int, int)       { PXQ_NO_ENCODER_ABORT(); }
