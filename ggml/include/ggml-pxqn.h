// ggml-pxqn.h -- PXQN (PXQ-Next revision 1): the type geometry, the revision and the activation-site ids
// the loader and the graph need. The decode itself (books, code layouts, rotation) is not part of the open
// engine: it ships compiled, in libggml-pxqn (see ggml/src/ggml-pxqn-api.h). A build without that library
// recognises PXQN files and refuses them with a clear message; classic PXQ and k-quants are unaffected.


#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define PXQN_HD static __host__ __device__ __forceinline__
#else
#define PXQN_HD static inline
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------------------------
// geometry
// ---------------------------------------------------------------------------------------------
#define PXQN_BM               64      // rows per panel
#define PXQN_HDR_BYTES        128     // 64 x fp16 row anchors at the head of every panel
#define PXQN_ROW_META         2       // ggml row_meta_size

#define PXQN3_QK              128     // K per slab (ggml blck_size)
#define PXQN3_TYPE_SIZE       52      // bytes per 128 elements of one row (4 B sub + 48 B codes)
#define PXQN3_SOA_BYTES       256     // 64 rows x u32 (8 SUB16 nibbles, one per 16-group)
#define PXQN3_CODE_ROW_BYTES  48      // 12 u32 words per row per slab
#define PXQN3_SLAB_BYTES      3328    // 256 + 64*48

#define PXQN3S8_QK            128
#define PXQN3S8_TYPE_SIZE     56      // 8 B sub + 48 B codes
#define PXQN3S8_SOA_BYTES     512     // 64 rows x u64 (16 SUB16 nibbles, one per 8-group)
#define PXQN3S8_SLAB_BYTES    3584    // 512 + 64*48

#define PXQN4_QK              32      // K per slab
#define PXQN4_TYPE_SIZE       17      // 1 B sub (2 nibbles) + 16 B nibble codes
#define PXQN4_SOA_BYTES       64
#define PXQN4_CODE_ROW_BYTES  16
#define PXQN4_SLAB_BYTES      1088

// --- ladder tiers (revision 1 additions, spec section 8) ---
#define PXQN2_QK              128
#define PXQN2_TYPE_SIZE       36      // 4 B sub + 32 B codes per 128 elements of one row
#define PXQN2_SOA_BYTES       256     // 64 rows x u32 (8 SUB16 nibbles, one per 16-group)
#define PXQN2_CODE_ROW_BYTES  32      // 8 u32 words per row per slab
#define PXQN2_SLAB_BYTES      2304    // 256 + 64*32

#define PXQN1_QK              128
#define PXQN1_TYPE_SIZE       20      // 4 B sub + 16 B sign bits
#define PXQN1_SOA_BYTES       256
#define PXQN1_CODE_ROW_BYTES  16      // 4 u32 words per row per slab
#define PXQN1_SLAB_BYTES      1280    // 256 + 64*16

#define PXQN4S8_QK            32
#define PXQN4S8_TYPE_SIZE     18      // 2 B sub (4 nibbles, one per 8-group) + 16 B nibble codes
#define PXQN4S8_SOA_BYTES     128     // 64 rows x u16
#define PXQN4S8_CODE_ROW_BYTES 16
#define PXQN4S8_SLAB_BYTES    1152    // 128 + 64*16

#define PXQN5_QK              128
#define PXQN5_TYPE_SIZE       84      // 4 B sub + 80 B codes
#define PXQN5_SOA_BYTES       256
#define PXQN5_CODE_ROW_BYTES  80      // 16 nibble words + 4 hi-bit words per row per slab
#define PXQN5_NIB_BYTES       64      // offset of the hi-bit plane inside a code row
#define PXQN5_SLAB_BYTES      5376    // 256 + 64*80

#define PXQN_REV              1u      // the revision this build reads and writes

// Rotation granularity along K: every rotated tensor (any type) cuts at multiples of this.
#define PXQN_RHT_BLOCK        128
#define PXQN_MIN_SHARD_K      128     // PXA_PXQ_MIN_SHARD_K for PXQN3/PXQN3S8 and rotated tensors

// activation sites a PXQN file may name in pxa.pxqn.rot.sites
enum pxqn_site {
    PXQN_SITE_NONE    = 0,
    PXQN_SITE_ATTN_IN = 1,   // attn_norm output: attn_qkv, attn_gate, ssm_alpha, ssm_beta / attn_q, attn_k, attn_v
    PXQN_SITE_FFN_IN  = 2,   // post_attention_norm output: ffn_gate, ffn_up
    PXQN_SITE_DOWN_IN = 3,   // SwiGLU output: ffn_down
    PXQN_SITE_OUT_IN  = 4,   // DeltaNet gated-norm output (ssm_out) / attention output (attn_output)
    PXQN_SITE_COUNT   = 5,
};

#define PXQN_SITE_BIT(site) (1u << (unsigned)(site))

#ifdef __cplusplus
}
#endif
