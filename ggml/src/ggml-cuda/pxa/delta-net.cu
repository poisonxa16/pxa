// -----------------------------------------------------------------------------
// Vendored from ik_llama.cpp, commit 3c58ae37, file
// ggml/src/ggml-cuda/delta-net.cu -- the DeltaNet linear-attention kernel for
// Qwen3-Next. First imported here 2026-07-17.
//
// ik_llama.cpp and the llama.cpp / ggml it derives from are MIT licensed:
//   MIT License, Copyright (c) 2023-2026 the ik_llama.cpp authors
//                Copyright (c) 2023-2026 The ggml authors
//   (full text: the LICENSE file at the root of each repository)
//
// The recurrence itself is upstream's and is unmodified. PXA's edits are the
// state-destination redirects that the DeltaNet decode fusion drives, and they
// are the only reason this file is not byte-identical to upstream's:
//   * ggml_cuda_op_delta_net_ex   -- PXA_FUSE_DELTANET, writes the new ssm state
//                                    into a caller-supplied destination instead
//                                    of through a separate CONCAT.
//   * ggml_cuda_op_delta_net_ex2  -- PXA_DN_SCATTER_FUSE, resolves the
//                                    destination ROW on the device from the
//                                    index tensor the fused-away SET_ROWS would
//                                    have read.
//   * ggml_cuda_op_delta_net_ex3  -- PXA_DN_INPLACE, additionally READS the
//                                    carried state from that same row, times
//                                    the per-sequence reset mask.
// All take nullptr to mean "do exactly what upstream does". Every edited site
// names the lever it belongs to in a comment; see ggml-cuda/pxa/README.md and
// pxa-deltanet-fuse.cuh for what drives them.
// -----------------------------------------------------------------------------
#include "../common.cuh"
#include "delta-net.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Delta Net Linear Attention Kernel for Qwen3-Next (HEAD_DIM=128)
// State layout: [S_v, S_v*H_v, 1, n_seqs] (column-major)

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <int block_size>
__device__ __forceinline__ float reduce_sum(float x, float * s) {
    x = warp_reduce_sum(x);
    if constexpr (block_size > WARP_SIZE) {
        //__shared__ float s[block_size/WARP_SIZE];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s[warp_id] = x;
        }
        __syncthreads();
        x = lane_id < block_size/WARP_SIZE ? s[lane_id] : 0.0f;
        x = warp_reduce_sum(x);
    }
    return x;
}

// PXA_DN_OCCUPANCY: the recurrence body, factored out of the __global__ entry point so that
// the default kernel and the opt-in __launch_bounds__ variant below run the identical
// instruction sequence -- the only difference between them is a launch-bounds attribute on
// the wrapper, never a code difference in the body itself.
//
// PXA_BISECT: g_per_channel is a TEMPLATE parameter, not a runtime argument. As a runtime
// argument the KDA-only per-key decay array `dcol` below was live on the Gated DeltaNet
// path too -- HEAD_DIM/num_warps = 32 extra floats per thread at <128,128>, which took the
// prefill instantiation from 96 to 128 registers on sm_70 (5 blocks/SM -> 4) and cost every
// non-KDA hybrid (qwen35 among them) 2-5% of prefill. Split into two instantiations: with
// g_per_channel == false every expression below is textually the pre-KDA one. The two
// rewrites compose: the split is on the body, so both entry points below inherit it.
// PXA_SPEC_CKPT_REPLAY: replay_cap == true is the verify of a replay-rollback round (one sequence):
// instead of a state row per step, saved_states is the replay pack (ggml.h,
// ggml_delta_net_set_replay_capture) and the kernel stores the state it STARTS from plus every
// token's inputs, exactly as it reads them. Every arithmetic expression is the replay_cap == false
// one; the captures are extra stores under `if constexpr`. qk_head_stride_in > 0 overrides the q/k
// head stride (HEAD_DIM * n_tokens): the replay reads a pack captured at a wider verify.
// PXA_DN_PREFETCH (pf): the per-token inputs of token t+1 (q, k, v, g, beta) are loaded into registers while
// token t is computed, so the three serialized global-load latencies per token (q/k at the top, g/beta after the
// first reduction, v after the second barrier) leave the critical path. Same values, same arithmetic in the same
// order: bit-identical to pf == false. Prefill class only (block_size == HEAD_DIM, scalar gate, no replay capture).
template <int HEAD_DIM, int block_size, bool g_per_channel, bool replay_cap = false, bool pf = false>
__device__ __forceinline__ void delta_net_recurrent_body(
    const float * __restrict__ q,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ k,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ v,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ g,         // [n_tokens, 1, n_heads, n_seqs]
    const float * __restrict__ beta_in,   // [1, n_tokens, n_heads, n_seqs]
    const float * __restrict__ state_in,  // [HEAD_DIM, HEAD_DIM*n_heads, 1, n_seqs]
    float * __restrict__ dst,             // output + new_state(s) concatenated
    float * __restrict__ saved_states,
    // PXA_RS_RING: strides (in elements) of the per-step capture destination. <= 0 means "derive
    // them from the dense [step][batch_pos][state] layout", which is bit-identical to before.
    const int64_t sv_row_stride_in,
    const int64_t sv_step_stride_in,
    float * __restrict__ state_ext,       // PXA_FUSE_DELTANET: optional external new-state destination
    const int32_t * __restrict__ state_ext_idx, // PXA_DN_SCATTER_FUSE: optional DEVICE-side row index
    const int64_t state_ext_row_stride,         //   (elements) row stride of the destination buffer
    const float * __restrict__ state_in_mask,   // PXA_DN_INPLACE: read the state IN PLACE from the destination row, times mask[0]
    const int64_t n_heads,
    const int64_t gqa_ratio,
    const int repeat_type,
    const int64_t n_tokens,
    const int64_t n_seqs,
    const int64_t output_offset,          // offset where state starts in output
    size_t vnb1, size_t vnb2, size_t vnb3,
    const int64_t qk_head_stride_in) {    // PXA_SPEC_CKPT_REPLAY: <= 0 = HEAD_DIM * n_tokens
    // PXA_GLM5NEXT: g_per_channel == false = one forget scalar per head (Gated DeltaNet), g is
    // [n_tokens,1,H,n_seqs]; true = one forget value per KEY CHANNEL (Kimi Delta Attention), g is
    // a contiguous [HEAD_DIM, n_tokens, H, n_seqs].
    constexpr int warps_per_head = HEAD_DIM/WARP_SIZE;
    const int batch_idx = blockIdx.x / (warps_per_head*n_heads);
    const int sub_head_idx  = blockIdx.x % (warps_per_head*n_heads);
    const int head_idx = sub_head_idx / warps_per_head;
    const int sub_idx  = sub_head_idx % warps_per_head;
    const int head_idx_kq = repeat_type == 0 ? head_idx / gqa_ratio : head_idx % (n_heads/gqa_ratio);
    const int tid = threadIdx.x;

    // Strides for input tensors (column-major)
    // Q/K/V: [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const int64_t qkv_stride_token = HEAD_DIM;
    const int64_t qkv_stride_head = qk_head_stride_in > 0 ? qk_head_stride_in : HEAD_DIM * n_tokens;
    const int64_t qkv_stride_batch = HEAD_DIM * n_tokens * n_heads;
    const int64_t qkv_stride_batch_kq = qkv_stride_batch / gqa_ratio;

    // G/Beta: [n_tokens, 1, n_heads, n_seqs] / [1, n_tokens, n_heads, n_seqs]
    //const int64_t g_stride_head = n_tokens;
    const int64_t g_stride_batch = n_tokens * n_heads;

    // State: [HEAD_DIM, HEAD_DIM*n_heads, 1, n_seqs]
    // For head h: columns h*HEAD_DIM to (h+1)*HEAD_DIM
    // state[row, col] for head h = state[row, h*HEAD_DIM + col]
    // Linear index: row + (h*HEAD_DIM + col) * HEAD_DIM = row + h*HEAD_DIM^2 + col*HEAD_DIM
    const int64_t state_head_offset = head_idx * HEAD_DIM * HEAD_DIM;
    const int64_t state_batch_stride = HEAD_DIM * HEAD_DIM * n_heads;

    // State step stride for save_all_states: HEAD_DIM^2 * n_heads * n_seqs
    const int64_t state_step_stride = HEAD_DIM * HEAD_DIM * n_heads * n_seqs;
    // PXA_RS_RING: the capture destination may be a strided window of a taller state tensor.
    const int64_t sv_row_stride  = sv_row_stride_in  > 0 ? sv_row_stride_in  : state_batch_stride;
    const int64_t sv_step_stride = sv_step_stride_in > 0 ? sv_step_stride_in : state_step_stride;

    // Pointers for this batch/head
    const float * q_ptr = q + batch_idx * qkv_stride_batch_kq + head_idx_kq * qkv_stride_head;
    const float * k_ptr = k + batch_idx * qkv_stride_batch_kq + head_idx_kq * qkv_stride_head;
    const float * v_ptr = v + batch_idx * vnb3 + head_idx * vnb2;
    const float * g_ptr = g + batch_idx * g_stride_batch + head_idx;
    const float * g_ptr_kda = g_per_channel
        ? g + batch_idx * (HEAD_DIM * n_tokens * n_heads) + head_idx * (HEAD_DIM * n_tokens)
        : nullptr;
    const float * beta_ptr = beta_in + batch_idx * g_stride_batch + head_idx;
    const float * state_src = state_in + batch_idx * state_batch_stride + state_head_offset;

    // Output layout: [head_v_dim, num_v_heads, n_seq_tokens, n_seqs]
    // For [dim, head, token, batch]: index = dim + head*S_v + token*S_v*H_v + batch*S_v*H_v*n_tokens
    float * out_base = dst + batch_idx * (HEAD_DIM * n_heads * n_tokens) + head_idx * HEAD_DIM;
    const int64_t out_token_stride = HEAD_DIM * n_heads;  // stride between tokens
    // PXA_DN_SCATTER_FUSE: when state_ext_idx is given, state_ext is the BASE of the recurrent
    // state buffer (already advanced past the conv prefix) and this sequence's row is chosen on the
    // DEVICE from the same index tensor the fused-away SET_ROWS would have used. The host cannot
    // do it: the index lives in a graph input buffer whose value is only known on the device
    // without a sync, and reading it host-side would also break CUDA-graph replay. Rows are
    // disjoint per sequence, so the per-batch packing stride is replaced (not added to) here.
    float * state_dst = (state_ext && state_ext_idx)
        ? state_ext + (int64_t)state_ext_idx[batch_idx] * state_ext_row_stride + state_head_offset
        : (state_ext ? state_ext : dst + output_offset) + batch_idx * state_batch_stride + state_head_offset;

    // PXA_SPEC_CKPT_REPLAY: pack = [start state | q | k | v | g | beta] of this (single) sequence
    float * rp_q = nullptr;
    float * rp_k = nullptr;
    float * rp_v = nullptr;
    float * rp_g = nullptr;
    float * rp_b = nullptr;
    bool rp_qk = false;   // this block stores its q/k head (one block per kq head)
    bool rp_gb = false;   // this thread stores g/beta of its head
    if constexpr (replay_cap) {
        const int64_t n_heads_kq = n_heads / gqa_ratio;
        rp_q = saved_states + state_batch_stride;
        rp_k = rp_q + HEAD_DIM * n_tokens * n_heads_kq;
        rp_v = rp_k + HEAD_DIM * n_tokens * n_heads_kq;
        rp_g = rp_v + HEAD_DIM * n_heads * n_tokens;
        rp_b = rp_g + n_heads * n_tokens;
        rp_q += head_idx_kq * HEAD_DIM * n_tokens;
        rp_k += head_idx_kq * HEAD_DIM * n_tokens;
        rp_qk = sub_idx == 0 && (repeat_type == 0 ? head_idx % gqa_ratio == 0 : head_idx < n_heads_kq);
        rp_gb = sub_idx == 0 && tid == 0;
    }

    // Shared memory for current token's Q, K, V (normalized), and intermediate results
    extern __shared__ float smem[];
    float * sQ = smem;                          // HEAD_DIM
    float * sK = sQ + HEAD_DIM;                 // HEAD_DIM

    const float scale = rsqrtf((float)HEAD_DIM);

    __shared__ float sum_helper[block_size/WARP_SIZE];

    constexpr int num_warps = block_size/WARP_SIZE;
    const int row = tid % WARP_SIZE;
    const int col_idx_0 = tid / WARP_SIZE;
    const int row_out = row + sub_idx * WARP_SIZE;

    // Keep the state in registers, copy the final state to its destination at the end
    float state_local[HEAD_DIM/num_warps];
    if (state_in_mask) {
        // PXA_DN_INPLACE: the carried state is read from the very row this thread writes at the end
        // (same thread, same element: an exact in-place alias), times the per-seq reset mask -- the
        // value the eager GET_ROWS + MUL would have staged, so the arithmetic is unchanged.
        const float sm = state_in_mask[0];
        for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
            int col = num_warps*i + col_idx_0;
            state_local[i] = __fmul_rn(state_dst[col*HEAD_DIM + row_out], sm);
        }
    } else {
        for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
            int col = num_warps*i + col_idx_0;
            state_local[i] = state_src[col*HEAD_DIM + row_out];
        }
    }
    if constexpr (replay_cap) {
        // the start state, exactly the register values the token loop begins from (mask applied)
        float * pre = saved_states + state_head_offset;
        for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
            int col = num_warps*i + col_idx_0;
            pre[col*HEAD_DIM + row_out] = state_local[i];
        }
    }

    constexpr int WARP_SIZE_S = WARP_SIZE + 1;
    constexpr int num_stored_rows = block_size/WARP_SIZE;
    __shared__ float all_sum[2*WARP_SIZE_S*num_stored_rows];
    auto all_sum1 = all_sum;
    auto all_sum2 = all_sum1 + WARP_SIZE_S*num_stored_rows;

    static_assert(!pf || (block_size == HEAD_DIM && !g_per_channel && !replay_cap), "PXA_DN_PREFETCH: prefill class only");
    float pf_q = 0.0f, pf_k = 0.0f, pf_v = 0.0f, pf_g = 0.0f, pf_b = 0.0f;
    if constexpr (pf) {
        if (n_tokens > 0) {
            pf_q = q_ptr[tid];
            pf_k = k_ptr[tid];
            pf_v = v_ptr[row_out];
            pf_g = g_ptr[0];
            pf_b = beta_ptr[0];
        }
    }

    for (int64_t t = 0; t < n_tokens; t++) {
        float sum_kq = 0.0f;
        float cur_v = 0.0f, cur_g = 0.0f, cur_b = 0.0f;
        if constexpr (pf) {
            const float cur_q = pf_q;
            const float cur_k = pf_k;
            cur_v = pf_v; cur_g = pf_g; cur_b = pf_b;
            if (t + 1 < n_tokens) {
                pf_q = q_ptr[(t + 1) * qkv_stride_token + tid];
                pf_k = k_ptr[(t + 1) * qkv_stride_token + tid];
                pf_v = v_ptr[(t + 1) * vnb1 + row_out];
                pf_g = g_ptr[(t + 1)*n_heads];
                pf_b = beta_ptr[(t + 1)*n_heads];
            }
            sQ[tid] = cur_q * scale;
            sK[tid] = cur_k;
            sum_kq += sK[tid] * sQ[tid];
        } else {
        for (int i = tid; i < HEAD_DIM; i += block_size) {
            sQ[i] = q_ptr[t * qkv_stride_token + i] * scale;
            sK[i] = k_ptr[t * qkv_stride_token + i];
            sum_kq += sK[i] * sQ[i];
        }
        }
        if constexpr (replay_cap) {
            if (rp_qk) {
                for (int i = tid; i < HEAD_DIM; i += block_size) {
                    rp_q[t * HEAD_DIM + i] = q_ptr[t * qkv_stride_token + i];
                    rp_k[t * HEAD_DIM + i] = k_ptr[t * qkv_stride_token + i];
                }
            }
            if (rp_gb) {
                rp_g[t * n_heads + head_idx] = g_ptr[t*n_heads];
                rp_b[t * n_heads + head_idx] = beta_ptr[t*n_heads];
            }
            if (col_idx_0 == 0) {
                rp_v[(t * n_heads + head_idx) * HEAD_DIM + row_out] = v_ptr[t * vnb1 + row_out];
            }
        }

        float attn_score = reduce_sum<block_size>(sum_kq, sum_helper);

        float beta_val = sigmoid_f(pf ? cur_b : beta_ptr[t*n_heads]);
        float decay    = g_per_channel ? 1.0f : expf(fminf(pf ? cur_g : g_ptr[t*n_heads], 50.0f));

        // PXA_GLM5NEXT: Kimi Delta Attention's forget gate is indexed by the KEY channel
        // (upstream PR #27773's gated_delta_net kernel does S[i][j] *= exp(g[i]) with i = key),
        // and each thread here owns whole COLUMNS of the state -- col is the key index, row_out
        // the value index. A per-key decay therefore cannot be factored out of the two sums the
        // way Gated DeltaNet's one-scalar-per-head gate can, so it is folded into the state as
        // it is read. With g_per_channel == 0 every expression below is textually the old one.
        // sized to 1 on the GDN path so it costs no registers there (see the note on the template)
        float dcol[g_per_channel ? HEAD_DIM/num_warps : 1];
        if constexpr (g_per_channel) {
#pragma unroll
            for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
                int col = num_warps*i + col_idx_0;
                dcol[i] = expf(fminf(g_ptr_kda[t*HEAD_DIM + col], 50.0f));
            }
        }

        float sum1 = 0, sum2 = 0;
#pragma unroll
        for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
            int col = num_warps*i + col_idx_0;
            if constexpr (g_per_channel) {
                const float s = state_local[i] * dcol[i];
                sum1 += s * sK[col];
                sum2 += s * sQ[col];
            } else {
                sum1 += state_local[i] * sK[col];
                sum2 += state_local[i] * sQ[col];
            }
        }
        all_sum1[col_idx_0*WARP_SIZE_S + row] = sum1;
        all_sum2[col_idx_0*WARP_SIZE_S + row] = sum2;

        __syncthreads();

        sum1 = sum2 = 0;
#pragma unroll
        for (int i = 0; i < block_size/WARP_SIZE; ++i) {
            sum1 += all_sum1[i*WARP_SIZE_S + row];
            sum2 += all_sum2[i*WARP_SIZE_S + row];
        }

        // the decay is already inside sum1/sum2 on the KDA path
        float sv_new = beta_val * ((pf ? cur_v : v_ptr[t * vnb1 + row_out]) - (g_per_channel ? sum1 : sum1 * decay));
        if (col_idx_0 == 0) {
            out_base[t * out_token_stride + row_out] =
                (g_per_channel ? sum2 : sum2 * decay) + sv_new * attn_score;
        }

        for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
            int col = num_warps*i + col_idx_0;
            float new_state_val;
            if constexpr (g_per_channel) {
                new_state_val = dcol[i] * state_local[i] + sv_new * sK[col];
            } else {
                new_state_val = decay * state_local[i] + sv_new * sK[col];
            }
            new_state_val = fminf(fmaxf(new_state_val, -1e6f), 1e6f);
            state_local[i] = new_state_val;
        }

        // Save per-step state if requested
        if constexpr (!replay_cap) {
        if (saved_states && t < n_tokens - 1) {
            float * state_step_dst = saved_states + batch_idx * sv_row_stride + state_head_offset + t * sv_step_stride;
            for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
                int col = num_warps*i + col_idx_0;
                state_step_dst[col*HEAD_DIM + row_out] = state_local[i];
            }
        }
        }

        // Barrier required: (a) sK reads in the state update above must complete
        // before next iteration overwrites sK at the top of the loop, and (b) this
        // single barrier also orders all_sum1/all_sum2 reads above vs. the next
        // iteration's writes — subsuming the prior barriers after the cross-warp
        // reduction and after the loop exit.
        __syncthreads();
    }
    // Copy the final state to its destination
    for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
        int col = num_warps*i + col_idx_0;
        state_dst[col*HEAD_DIM + row_out] = state_local[i];
    }
}

// Default entry point: today's kernel, byte-for-byte -- no launch-bounds attribute, same
// code path as before this file carried PXA_DN_OCCUPANCY.
template <int HEAD_DIM, int block_size, bool g_per_channel, bool replay_cap = false>
__global__ void delta_net_recurrent_f32(
    const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
    const float * __restrict__ g, const float * __restrict__ beta_in,
    const float * __restrict__ state_in, float * __restrict__ dst, float * __restrict__ saved_states,
    // PXA_RS_RING: strides (in elements) of the per-step capture destination; <= 0 = derive.
    const int64_t sv_row_stride, const int64_t sv_step_stride,
    float * __restrict__ state_ext, const int32_t * __restrict__ state_ext_idx,
    const int64_t state_ext_row_stride, const float * __restrict__ state_in_mask,
    const int64_t n_heads, const int64_t gqa_ratio,
    const int repeat_type, const int64_t n_tokens, const int64_t n_seqs,
    const int64_t output_offset, size_t vnb1, size_t vnb2, size_t vnb3, const int64_t qk_head_stride) {
    delta_net_recurrent_body<HEAD_DIM, block_size, g_per_channel, replay_cap>(
        q, k, v, g, beta_in, state_in, dst, saved_states, sv_row_stride, sv_step_stride,
        state_ext, state_ext_idx,
        state_ext_row_stride, state_in_mask, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, output_offset,
        vnb1, vnb2, vnb3, qk_head_stride);
}

// PXA_DN_PREFETCH=1: the prefill-class kernel with the next token's inputs in flight (see the body). Bit-identical.
template <int HEAD_DIM, int block_size>
__global__ void delta_net_recurrent_f32_pf(
    const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
    const float * __restrict__ g, const float * __restrict__ beta_in,
    const float * __restrict__ state_in, float * __restrict__ dst, float * __restrict__ saved_states,
    const int64_t sv_row_stride, const int64_t sv_step_stride,
    float * __restrict__ state_ext, const int32_t * __restrict__ state_ext_idx,
    const int64_t state_ext_row_stride, const float * __restrict__ state_in_mask,
    const int64_t n_heads, const int64_t gqa_ratio,
    const int repeat_type, const int64_t n_tokens, const int64_t n_seqs,
    const int64_t output_offset, size_t vnb1, size_t vnb2, size_t vnb3, const int64_t qk_head_stride) {
    delta_net_recurrent_body<HEAD_DIM, block_size, false, false, true>(
        q, k, v, g, beta_in, state_in, dst, saved_states, sv_row_stride, sv_step_stride,
        state_ext, state_ext_idx,
        state_ext_row_stride, state_in_mask, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, output_offset,
        vnb1, vnb2, vnb3, qk_head_stride);
}

// PXA_DN_PREFETCH: unset = on where it was measured (sm_60: 27B on one P100, pp4096 +0.26%, greedy output identical,
// 2026-10-19), off elsewhere; =0 off, =1 on everywhere. Bit-identical either way.
static bool pxa_dn_prefetch_on(int cc) {
    static const int v = [](){ const char * e = getenv("PXA_DN_PREFETCH"); return e && *e ? (atoi(e) != 0 ? 1 : 0) : -1; }();
    return v < 0 ? cc == 600 : v == 1;
}

// PXA_DN_OCCUPANCY=<n>, n>=1: opt-in variant carrying an explicit min-blocks-per-SM occupancy
// hint (MJ-PORT item 3, "the one thing that might still transfer"). Identical body to the
// default kernel above -- bit-identical output by construction, since only a compiler launch
// attribute differs; this is a register-allocation / occupancy experiment, not a numerics change.
template <int HEAD_DIM, int block_size, int MIN_BLOCKS, bool g_per_channel>
__global__ __launch_bounds__(block_size, MIN_BLOCKS) void delta_net_recurrent_f32_occ(
    const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
    const float * __restrict__ g, const float * __restrict__ beta_in,
    const float * __restrict__ state_in, float * __restrict__ dst, float * __restrict__ saved_states,
    // PXA_RS_RING: strides (in elements) of the per-step capture destination; <= 0 = derive.
    const int64_t sv_row_stride, const int64_t sv_step_stride,
    float * __restrict__ state_ext, const int32_t * __restrict__ state_ext_idx,
    const int64_t state_ext_row_stride, const float * __restrict__ state_in_mask,
    const int64_t n_heads, const int64_t gqa_ratio,
    const int repeat_type, const int64_t n_tokens, const int64_t n_seqs,
    const int64_t output_offset, size_t vnb1, size_t vnb2, size_t vnb3, const int64_t qk_head_stride) {
    delta_net_recurrent_body<HEAD_DIM, block_size, g_per_channel, false>(
        q, k, v, g, beta_in, state_in, dst, saved_states, sv_row_stride, sv_step_stride,
        state_ext, state_ext_idx,
        state_ext_row_stride, state_in_mask, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, output_offset,
        vnb1, vnb2, vnb3, qk_head_stride);
}

// PXA_DN_OCCUPANCY=<n>: 0/unset keeps today's kernel untouched; n in {1,2,3,4} launches the
// __launch_bounds__(block_size, n) variant above instead. Parsed once and cached; any other
// positive value is treated as unset (logged once) rather than silently guessing a hint.
static int pxa_dn_occupancy_hint() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("PXA_DN_OCCUPANCY");
        int v = (env && *env) ? atoi(env) : 0;
        if (v < 0 || v > 4) {
            if (v != 0) {
                fprintf(stderr, "PXA_DN_OCCUPANCY=%d is out of the supported {1,2,3,4} range; ignoring\n", v);
            }
            v = 0;
        }
        cached = v;
    }
    return cached;
}

template <int HEAD_DIM, int block_size, bool g_per_channel, typename... Args>
static void delta_net_dispatch(int occ, int num_blocks, size_t smem_size, cudaStream_t stream, Args... args) {
    switch (occ) {
        case 1: delta_net_recurrent_f32_occ<HEAD_DIM, block_size, 1, g_per_channel><<<num_blocks, block_size, smem_size, stream>>>(args...); break;
        case 2: delta_net_recurrent_f32_occ<HEAD_DIM, block_size, 2, g_per_channel><<<num_blocks, block_size, smem_size, stream>>>(args...); break;
        case 3: delta_net_recurrent_f32_occ<HEAD_DIM, block_size, 3, g_per_channel><<<num_blocks, block_size, smem_size, stream>>>(args...); break;
        case 4: delta_net_recurrent_f32_occ<HEAD_DIM, block_size, 4, g_per_channel><<<num_blocks, block_size, smem_size, stream>>>(args...); break;
        default: delta_net_recurrent_f32<HEAD_DIM, block_size, g_per_channel><<<num_blocks, block_size, smem_size, stream>>>(args...); break;
    }
}

// PXA_SPEC_CKPT_REPLAY: one head-dim/block-size class, gate kind and capture flavour. The capture
// verify (replay_cap) has its own instantiation (scalar gate, default entry); everything else is the
// switch above, unchanged.
template <int HEAD_DIM, int block_size, typename... Args>
static void delta_net_dispatch_cls(int occ, bool g_per_channel, bool replay_cap, bool pf, int num_blocks, size_t smem_size,
                                   cudaStream_t stream, Args... args) {
    if (replay_cap) {
        GGML_ASSERT(!g_per_channel);
        delta_net_recurrent_f32<HEAD_DIM, block_size, false, true><<<num_blocks, block_size, smem_size, stream>>>(args...);
    } else if (g_per_channel) {
        delta_net_dispatch<HEAD_DIM, block_size, true>(occ, num_blocks, smem_size, stream, args...);
    } else {
        if constexpr (block_size == HEAD_DIM) {
            if (occ == 0 && pf) {
                delta_net_recurrent_f32_pf<HEAD_DIM, block_size><<<num_blocks, block_size, smem_size, stream>>>(args...);
                return;
            }
        }
        delta_net_dispatch<HEAD_DIM, block_size, false>(occ, num_blocks, smem_size, stream, args...);
    }
}

// ---- PXA_DN_COLWARP (2026-10-15; default OFF until gated) ---------------------------------
// Prefill-class Gated DeltaNet (head 128, scalar gate, no per-step capture) with NO block barrier per token.
// A warp owns 4 value columns; inside a column 8 lanes split the 128 keys (lane phase j owns keys j, j+8, ...,
// j+120: conflict-free shared-memory reads, the 4 column groups read the same words = broadcasts), so the
// three per-token sums (S^T k, S^T q, q.k) finish with a 3-step xor shuffle inside the 8 lanes. Blocks of 4 warps
// = 16 columns of one head; q (scaled), k, v, g, beta of CH tokens are staged in shared memory with coalesced loads
// and one barrier pair per chunk. Same update formula, fp32 throughout; the sums are taken in a different order
// than delta_net_recurrent_f32 (not bit-identical; same math).
template <int CH>
__global__ void __launch_bounds__(128) delta_net_colwarp_f32(
    const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
    const float * __restrict__ g, const float * __restrict__ beta_in,
    const float * __restrict__ state_in, float * __restrict__ dst,
    float * __restrict__ state_ext, const int32_t * __restrict__ state_ext_idx,
    const int64_t state_ext_row_stride, const float * __restrict__ state_in_mask,
    const int64_t n_heads, const int64_t gqa_ratio, const int repeat_type,
    const int64_t n_tokens, const int64_t output_offset, size_t vnb1, size_t vnb2, size_t vnb3,
    const int64_t qk_head_stride_in) {
    constexpr int D = 128, COLS = 16;                       // head size, value columns per block
    const int blocks_per_head = D / COLS;
    const int batch_idx = blockIdx.x / (blocks_per_head * n_heads);
    const int bh        = blockIdx.x % (blocks_per_head * n_heads);
    const int head_idx  = bh / blocks_per_head;
    const int col0      = (bh % blocks_per_head) * COLS;
    const int head_idx_kq = repeat_type == 0 ? head_idx / gqa_ratio : head_idx % (n_heads/gqa_ratio);
    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    const int cg   = lane / 8, j = lane % 8;                // column group in the warp, key phase
    const int ccol = warp * 4 + cg;                         // column within the block
    const int vc   = col0 + ccol;                           // value column (row_out of the generic kernel)

    const int64_t qkv_stride_head  = qk_head_stride_in > 0 ? qk_head_stride_in : (int64_t) D * n_tokens;
    const int64_t qkv_stride_batch = (int64_t) D * n_tokens * n_heads;
    const int64_t qkv_stride_batch_kq = qkv_stride_batch / gqa_ratio;
    const int64_t g_stride_batch   = n_tokens * n_heads;
    const int64_t state_head_offset  = (int64_t) head_idx * D * D;
    const int64_t state_batch_stride = (int64_t) D * D * n_heads;

    const float * q_ptr = q + batch_idx * qkv_stride_batch_kq + head_idx_kq * qkv_stride_head;
    const float * k_ptr = k + batch_idx * qkv_stride_batch_kq + head_idx_kq * qkv_stride_head;
    const float * v_ptr = v + batch_idx * vnb3 + head_idx * vnb2;
    const float * g_ptr = g + batch_idx * g_stride_batch + head_idx;
    const float * b_ptr = beta_in + batch_idx * g_stride_batch + head_idx;
    const float * state_src = state_in + batch_idx * state_batch_stride + state_head_offset;
    float * out_base = dst + batch_idx * (D * n_heads * n_tokens) + head_idx * D;
    const int64_t out_token_stride = (int64_t) D * n_heads;
    float * state_dst = (state_ext && state_ext_idx)
        ? state_ext + (int64_t)state_ext_idx[batch_idx] * state_ext_row_stride + state_head_offset
        : (state_ext ? state_ext : dst + output_offset) + batch_idx * state_batch_stride + state_head_offset;

    __shared__ float sQ[CH][D];
    __shared__ float sK[CH][D];
    __shared__ float sV[CH][COLS];
    __shared__ float sDec[CH];
    __shared__ float sBeta[CH];

    float S[D/8];                                           // S[i] = state[key j+8i][value vc]
    if (state_in_mask) {
        const float sm = state_in_mask[0];
#pragma unroll
        for (int i = 0; i < D/8; ++i) S[i] = __fmul_rn(state_dst[(j + 8*i)*D + vc], sm);
    } else {
#pragma unroll
        for (int i = 0; i < D/8; ++i) S[i] = state_src[(j + 8*i)*D + vc];
    }
    const float scale = rsqrtf((float) D);

    // register double-buffer: chunk c+1's inputs are loaded while chunk c is computed (fully unrolled, so every
    // load of a chunk is in flight at once instead of one memory latency per staging iteration)
    constexpr int QK_PER_T = CH*D/128;                      // q (and k) values each thread stages per chunk
    constexpr int V_PER_T  = (CH*COLS + 127)/128;
    float rq[QK_PER_T], rk[QK_PER_T], rv[V_PER_T], rg = 0.0f, rb = 0.0f;
    auto load_chunk = [&](int64_t c0) {
        const int cnt = (int) min((int64_t) CH, n_tokens - c0);
#pragma unroll
        for (int u = 0; u < QK_PER_T; ++u) {
            const int e = tid + 128*u, tt = e / D, kk = e % D;
            rq[u] = tt < cnt ? q_ptr[(c0 + tt)*D + kk] : 0.0f;
            rk[u] = tt < cnt ? k_ptr[(c0 + tt)*D + kk] : 0.0f;
        }
#pragma unroll
        for (int u = 0; u < V_PER_T; ++u) {
            const int e = tid + 128*u, tt = e / COLS, cc = e % COLS;
            rv[u] = (e < CH*COLS && tt < cnt) ? v_ptr[(c0 + tt)*vnb1 + col0 + cc] : 0.0f;
        }
        if (tid < cnt) {
            rg = g_ptr[(c0 + tid)*n_heads];
            rb = b_ptr[(c0 + tid)*n_heads];
        }
    };
    if (n_tokens > 0) load_chunk(0);

    for (int64_t t0 = 0; t0 < n_tokens; t0 += CH) {
        const int nt = (int) min((int64_t) CH, n_tokens - t0);
        __syncthreads();                                    // the previous chunk's reads are done
#pragma unroll
        for (int u = 0; u < QK_PER_T; ++u) {
            const int e = tid + 128*u, tt = e / D, kk = e % D;
            sQ[tt][kk] = rq[u] * scale;
            sK[tt][kk] = rk[u];
        }
#pragma unroll
        for (int u = 0; u < V_PER_T; ++u) {
            const int e = tid + 128*u;
            if (e < CH*COLS) sV[e / COLS][e % COLS] = rv[u];
        }
        if (tid < nt) {
            sDec[tid]  = expf(fminf(rg, 50.0f));
            sBeta[tid] = 1.0f / (1.0f + expf(-rb));
        }
        __syncthreads();
        if (t0 + CH < n_tokens) load_chunk(t0 + CH);
        for (int tt = 0; tt < nt; ++tt) {
            float s1 = 0.0f, s2 = 0.0f, pqk = 0.0f;
#pragma unroll
            for (int i = 0; i < D/8; ++i) {
                const float kv = sK[tt][j + 8*i];
                const float qv = sQ[tt][j + 8*i];
                s1  += S[i] * kv;
                s2  += S[i] * qv;
                pqk += qv * kv;
            }
#pragma unroll
            for (int off = 4; off >= 1; off >>= 1) {
                s1  += __shfl_xor_sync(0xffffffff, s1,  off, 8);
                s2  += __shfl_xor_sync(0xffffffff, s2,  off, 8);
                pqk += __shfl_xor_sync(0xffffffff, pqk, off, 8);
            }
            const float decay = sDec[tt];
            const float sv_new = sBeta[tt] * (sV[tt][ccol] - s1 * decay);
            if (j == 0) {
                out_base[(t0 + tt)*out_token_stride + vc] = s2 * decay + sv_new * pqk;
            }
#pragma unroll
            for (int i = 0; i < D/8; ++i) {
                const float ns = decay * S[i] + sv_new * sK[tt][j + 8*i];
                S[i] = fminf(fmaxf(ns, -1e6f), 1e6f);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < D/8; ++i) state_dst[(j + 8*i)*D + vc] = S[i];
}

static bool pxa_dn_colwarp_on() {
    static const bool v = [](){ const char * e = getenv("PXA_DN_COLWARP"); return e && atoi(e) != 0; }();
    return v;
}

static void delta_net_f32_cuda(
    const float * q,
    const float * k,
    const float * v,
    const float * g,
    const float * beta,
    const float * state_in,
    float * dst,
    float * saved_states,
    const int64_t sv_row_stride,   // PXA_RS_RING (<=0: derive)
    const int64_t sv_step_stride,  // PXA_RS_RING (<=0: derive)
    float * state_ext,
    const int32_t * state_ext_idx,
    const int64_t state_ext_row_stride,
    const float * state_in_mask,
    const int64_t head_dim,
    const int64_t n_tokens,
    const int64_t n_heads,
    const int64_t gqa_ratio,
    const int     repeat_type,
    const int64_t n_seqs,
    size_t vnb1, size_t vnb2, size_t vnb3,
    const int g_per_channel,
    const int device_id,
    const int cc, // compute capability (e.g., 890 for SM 8.9, 1200 for SM 12.0)
    cudaStream_t stream,
    const int64_t qk_head_stride = 0,   // PXA_SPEC_CKPT_REPLAY: q/k head stride override (0 = HEAD_DIM*n_tokens)
    const int64_t n_tokens_class = 0,   // PXA_SPEC_CKPT_REPLAY: the width that picks the block size (0 = n_tokens)
    const bool    replay_cap = false) { // PXA_SPEC_CKPT_REPLAY: capture-verify instantiation
    GGML_UNUSED(device_id);

    const int64_t output_offset = head_dim * n_tokens * n_heads * n_seqs;

    if (head_dim != 64 && head_dim != 128) {
        GGML_ABORT("Unsupported delta net head size");
    }

    GGML_ASSERT(head_dim % WARP_SIZE == 0);
    const int num_blocks = n_seqs * n_heads * (head_dim/WARP_SIZE);
    const size_t smem_size = 2 * head_dim * sizeof(float);

    const int dn_occ = pxa_dn_occupancy_hint();
    const bool gpc = g_per_channel != 0;
    const bool dn_pf = pxa_dn_prefetch_on(cc);

    // PXA_SPEC_CKPT_REPLAY: the block size changes the reduction order, so a replay of k+1 tokens
    // must run in the class of the verify it replays, not in the class of k+1.
    const int64_t n_cls = n_tokens_class > 0 ? n_tokens_class : n_tokens;
    if (head_dim == 128 && !gpc && !replay_cap && saved_states == nullptr && n_cls > 8 && dn_occ == 0 && pxa_dn_colwarp_on()) {
        const int nb = (int) (n_seqs * n_heads * (128/16));
        delta_net_colwarp_f32<16><<<nb, 128, 0, stream>>>(q, k, v, g, beta, state_in, dst, state_ext, state_ext_idx,
                state_ext_row_stride, state_in_mask, n_heads, gqa_ratio, repeat_type, n_tokens, output_offset, vnb1, vnb2, vnb3,
                qk_head_stride);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (n_cls <= 8) {
        constexpr int threads_per_block = 256;
        if (head_dim == 64) {
            delta_net_dispatch_cls<64, threads_per_block>(dn_occ, gpc, replay_cap, dn_pf, num_blocks, smem_size, stream,
                    q, k, v, g, beta, state_in, dst, saved_states, sv_row_stride, sv_step_stride, state_ext, state_ext_idx, state_ext_row_stride, state_in_mask, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, output_offset, vnb1, vnb2, vnb3, qk_head_stride);
        } else {
            delta_net_dispatch_cls<128, threads_per_block>(dn_occ, gpc, replay_cap, dn_pf, num_blocks, smem_size, stream,
                    q, k, v, g, beta, state_in, dst, saved_states, sv_row_stride, sv_step_stride, state_ext, state_ext_idx, state_ext_row_stride, state_in_mask, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, output_offset, vnb1, vnb2, vnb3, qk_head_stride);
        }
    } else {
        constexpr int threads_per_block = 128;
        if (head_dim == 64) {
            delta_net_dispatch_cls<64, threads_per_block>(dn_occ, gpc, replay_cap, dn_pf, num_blocks, smem_size, stream,
                    q, k, v, g, beta, state_in, dst, saved_states, sv_row_stride, sv_step_stride, state_ext, state_ext_idx, state_ext_row_stride, state_in_mask, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, output_offset, vnb1, vnb2, vnb3, qk_head_stride);
        } else {
            delta_net_dispatch_cls<128, threads_per_block>(dn_occ, gpc, replay_cap, dn_pf, num_blocks, smem_size, stream,
                    q, k, v, g, beta, state_in, dst, saved_states, sv_row_stride, sv_step_stride, state_ext, state_ext_idx, state_ext_row_stride, state_in_mask, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, output_offset, vnb1, vnb2, vnb3, qk_head_stride);
        }
    }

    CUDA_CHECK(cudaGetLastError());

}

void ggml_cuda_op_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_delta_net_ex2(ctx, dst, nullptr, nullptr, 0);
}

void ggml_cuda_op_delta_net_ex(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_dst_override) {
    ggml_cuda_op_delta_net_ex2(ctx, dst, state_dst_override, nullptr, 0);
}

void ggml_cuda_op_delta_net_ex2(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_dst_override,
                                const int32_t * state_dst_row_idx, int64_t state_dst_row_stride) {
    ggml_cuda_op_delta_net_ex3(ctx, dst, state_dst_override, state_dst_row_idx, state_dst_row_stride, nullptr);
}

void ggml_cuda_op_delta_net_ex3(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_dst_override,
                                const int32_t * state_dst_row_idx, int64_t state_dst_row_stride,
                                const float * state_in_mask) {
    const ggml_tensor * src0 = dst->src[0];  // q
    const ggml_tensor * src1 = dst->src[1];  // k
    const ggml_tensor * src2 = dst->src[2];  // v
    const ggml_tensor * src3 = dst->src[3];  // g
    const ggml_tensor * src4 = dst->src[4];  // beta
    const ggml_tensor * src5 = dst->src[5];  // state
    const ggml_tensor * src6 = dst->src[6];  // when not null, state for token 0...n_token-1

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t head_dim = src0->ne[0];
    const int64_t n_tokens = src0->ne[1];
    const int64_t n_heads = src2->ne[2];
    const int64_t n_heads_kq = src0->ne[2];
    const int64_t n_seqs = src0->ne[3];
    GGML_ASSERT(n_heads % n_heads_kq == 0);
    const int64_t gqa_ratio = n_heads / n_heads_kq;
    // PXA_DN_INPLACE: the in-place masked read needs a device-resolved destination row and one sequence.
    // A per-step capture is allowed (PXA_DN_CONVFUSE_NY, verify widths): each thread loads its state
    // elements into registers ONCE before the token loop, the per-step snapshots go to the separate
    // capture buffer, and the row is written once after the loop -- no step reads the row back.
    GGML_ASSERT(!state_in_mask || (state_dst_override && state_dst_row_idx && n_seqs == 1));

    // Dimension validation
    // Q/K: [head_dim, n_tokens, n_heads, n_seqs]
    GGML_ASSERT(src1->ne[0] == head_dim && src1->ne[1] == n_tokens && src1->ne[2] == n_heads_kq && src1->ne[3] == n_seqs);
    // V: [head_dim, n_tokens, n_heads, n_seqs]
    GGML_ASSERT(src2->ne[0] == head_dim && src2->ne[1] == n_tokens && src2->ne[2] == n_heads && src2->ne[3] == n_seqs);
    // G: [n_tokens, 1, n_heads, n_seqs] (GDN) or [head_dim, n_tokens, n_heads, n_seqs] (KDA)
    const int g_per_channel = dst->op_params[1];
    if (g_per_channel) {
        GGML_ASSERT(src3->ne[0] == head_dim && src3->ne[1] == n_tokens && src3->ne[2] == n_heads && src3->ne[3] == n_seqs);
        // the kernel indexes g with the packed strides above, so it has to actually be packed
        GGML_ASSERT(ggml_is_contiguous(src3));
    } else {
        GGML_ASSERT(src3->ne[0] == n_tokens && src3->ne[1] == 1 && src3->ne[2] == n_heads && src3->ne[3] == n_seqs);
    }
    // Beta: [1, n_tokens, n_heads, n_seqs]
    GGML_ASSERT(src4->ne[0] == 1 && src4->ne[1] == n_tokens && src4->ne[2] == n_heads && src4->ne[3] == n_seqs);
    // State: [head_dim, head_dim*n_heads, 1, n_seqs]
    GGML_ASSERT(src5->ne[0] == head_dim && src5->ne[1] == head_dim * n_heads && src5->ne[2] == 1 && src5->ne[3] == n_seqs);

    // Verify output tensor size
    const int64_t output_size = head_dim * n_tokens * n_heads * n_seqs;
    const int64_t state_size = head_dim * head_dim * n_heads * n_seqs;

    int repeat_type = dst->op_params[0];
    // PXA_SPEC_CKPT_REPLAY: src6 is the replay pack, not per-step rows (ggml_delta_net_set_replay_capture)
    const bool replay_cap = dst->op_params[4] != 0;
    if (replay_cap) {
        GGML_ASSERT(src6 && src6->type == GGML_TYPE_F32 && n_seqs == 1 && !g_per_channel);
        GGML_ASSERT(dst->op_params[2] == 0 && dst->op_params[3] == 0);
        GGML_ASSERT(ggml_nelements(src6) >= head_dim*head_dim*n_heads
                    + n_tokens*(2*head_dim*n_heads_kq + head_dim*n_heads + 2*n_heads));
    } else if (src6) {
        GGML_ASSERT(src6->type == GGML_TYPE_F32);
        // PXA_RS_RING: with a retargeted capture the dense bound does not apply; check the
        // largest element the strided store can touch instead.
        const int64_t sv_row  = dst->op_params[2] > 0 ? (int64_t) dst->op_params[2] : head_dim*head_dim*n_heads;
        const int64_t sv_step = dst->op_params[3] > 0 ? (int64_t) dst->op_params[3] : state_size;
        const int64_t sv_max  = (n_tokens > 1 ? (n_tokens - 2)*sv_step : 0)
                              + (n_seqs   > 0 ? (n_seqs   - 1)*sv_row  : 0)
                              + head_dim*head_dim*n_heads;
        GGML_ASSERT(ggml_nelements(src6) >= sv_max);
    }

    const int64_t expected_size = output_size + state_size;
    GGML_ASSERT(ggml_nelements(dst) == expected_size);

    GGML_ASSERT(head_dim <= 256);  // Reasonable limit for shared memory

    // Get device info from ctx (avoids calling CUDA runtime APIs inside dispatch)
    const int device_id = ctx.device;
    const int cc = ggml_cuda_info().devices[device_id].cc;

    delta_net_f32_cuda(
        (const float *)src0->data,
        (const float *)src1->data,
        (const float *)src2->data,
        (const float *)src3->data,
        (const float *)src4->data,
        (const float *)src5->data,
        (float *)dst->data,
        src6 ? (float *)src6->data : nullptr,
        (int64_t) dst->op_params[2],   // PXA_RS_RING capture row stride  (0 = derive)
        (int64_t) dst->op_params[3],   // PXA_RS_RING capture step stride (0 = derive)
        state_dst_override,
        state_dst_row_idx,
        state_dst_row_stride,
        state_in_mask,
        head_dim, n_tokens, n_heads, gqa_ratio, repeat_type, n_seqs,
        src2->nb[1]/sizeof(float), src2->nb[2]/sizeof(float), src2->nb[3]/sizeof(float),
        g_per_channel,
        device_id, cc,
        ctx.stream(),
        0, 0, replay_cap);

}

// PXA_SPEC_CKPT_REPLAY: the rollback of a replay-capture verify. For each layer, re-run the SAME
// recurrence over the first n_replay tokens of the captured pack, from the captured start state,
// in the block-size class of the n_verify-token verify, writing the final state to state_dst. The
// state after token n_replay-1 is the value the verify held in registers after that token (the
// per-step row a per-step capture would have stored), bit for bit: same kernel, same inputs, same
// order. The attention output goes to scratch (>= head_dim*n_heads*n_replay floats) and is dropped.
void ggml_cuda_delta_net_replay(ggml_backend_cuda_context & ctx, int n, const pxa_dn_replay_desc * d) {
    const int device_id = ctx.device;
    const int cc = ggml_cuda_info().devices[device_id].cc;
    for (int i = 0; i < n; ++i) {
        const pxa_dn_replay_desc & r = d[i];
        GGML_ASSERT(r.n_replay >= 1 && r.n_replay <= r.n_verify);
        GGML_ASSERT(r.n_heads % r.n_heads_kq == 0);
        const int64_t hd = r.head_dim, H = r.n_heads, Hk = r.n_heads_kq, T = r.n_verify;
        const float * pre  = r.pack;
        const float * q    = pre + hd*hd*H;
        const float * k    = q + hd*T*Hk;
        const float * v    = k + hd*T*Hk;
        const float * g    = v + hd*H*T;
        const float * beta = g + H*T;
        delta_net_f32_cuda(q, k, v, g, beta, pre, r.scratch, nullptr, 0, 0,
                r.state_dst, nullptr, 0, nullptr,
                hd, r.n_replay, H, H/Hk, r.repeat_type, 1,
                (size_t)(hd*H), (size_t) hd, (size_t)(hd*H*T),
                0, device_id, cc, ctx.stream(),
                hd*T, T, false);
    }
}
