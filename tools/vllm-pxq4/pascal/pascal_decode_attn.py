# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton paged single-token-decode attention for Pascal (sm_60/sm_61).

Why this exists
---------------
The torch fallback in ``pascal_sdpa.py`` is correct but has two properties that
make it the wrong thing to run in a decode loop:

  * it is **shaped by host data** -- it reads ``seq_lens`` on the CPU to decide
    how many KV blocks to gather, so the tensor shapes change from step to
    step.  That makes it impossible to capture in a CUDA graph, which in turn
    forces the whole engine into eager mode, where ~90% of a decode step is
    Python dispatch rather than GPU work;
  * it *materialises* the gathered KV in fp32 (``k.float()``, plus a
    ``repeat_interleave`` to expand the GQA groups), so a step touches several
    times the memory the attention math actually needs.

This kernel fixes both.  The launch grid is a function of (batch, heads) only,
the per-sequence context length is read *inside* the kernel from a device
tensor, and the KV is streamed straight out of the paged cache in its stored
dtype.  Shapes are therefore static for a given batch size, so the whole decode
step becomes CUDA-graph capturable.

Pascal constraint: no ``tl.dot``.  Triton cannot lower MMA for sm_60, and the
tl.dot path miscompiles or refuses on this arch.  It is not needed here --
single-token decode has no matmul in it.  Both contractions are
elementwise-multiply-then-reduce:

    scores[j] = sum_d q[d] * k[j, d]        -> tl.sum(q[None, :] * k, axis=1)
    out[d]    = sum_j p[j] * v[j, d]        -> tl.sum(p[:, None] * v, axis=0)

Numerics: accumulation is fp32 with the standard online-softmax rescaling, so
the result matches the fp32 reference path to fp16 rounding.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_attn_kernel(
    q_ptr,            # [S, H, D]
    k_cache_ptr,      # [num_blocks, block_size, KVH, D]
    v_cache_ptr,      # [num_blocks, block_size, KVH, D]
    blk_tbl_ptr,      # [S, max_blocks] int32
    seq_lens_ptr,     # [S] int32
    out_ptr,          # [S, H, D]
    scale,
    stride_qs, stride_qh,
    stride_kb, stride_kp, stride_kh,
    stride_vb, stride_vp, stride_vh,
    stride_ts,
    stride_os, stride_oh,
    N_REP: tl.constexpr,        # query heads per kv head
    D: tl.constexpr,            # head dim (power of two)
    PAGE: tl.constexpr,         # tokens per KV block
    BLOCK_N: tl.constexpr,      # keys processed per iteration
):
    s = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // N_REP

    d = tl.arange(0, D)
    q = tl.load(q_ptr + s * stride_qs + h * stride_qh + d).to(tl.float32) * scale

    n = tl.load(seq_lens_ptr + s).to(tl.int32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    for start in range(0, n, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        valid = offs < n
        # Page lookup.  PAGE is a constexpr, so these become mul-shift, not div.
        blk = tl.load(blk_tbl_ptr + s * stride_ts + offs // PAGE,
                      mask=valid, other=0).to(tl.int32)
        pos = offs % PAGE

        base = blk[:, None] * stride_kb + pos[:, None] * stride_kp + kvh * stride_kh
        k = tl.load(k_cache_ptr + base + d[None, :],
                    mask=valid[:, None], other=0.0).to(tl.float32)

        sij = tl.sum(q[None, :] * k, axis=1)
        sij = tl.where(valid, sij, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(sij, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sij - m_new)

        vbase = blk[:, None] * stride_vb + pos[:, None] * stride_vp + kvh * stride_vh
        v = tl.load(v_cache_ptr + vbase + d[None, :],
                    mask=valid[:, None], other=0.0).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    # n == 0 happens for the padding slots of a cudagraph-captured batch; the
    # loop never ran, so l_i is 0 and acc/l_i would be NaN.  Those slots are
    # discarded downstream, but NaN in a captured buffer is a debugging trap.
    out = tl.where(l_i > 0.0, acc / tl.where(l_i > 0.0, l_i, 1.0), 0.0)
    tl.store(out_ptr + s * stride_os + h * stride_oh + d, out.to(out_ptr.dtype.element_ty))


def paged_decode_attention(
    query: torch.Tensor,        # [S, H, D]
    key_cache: torch.Tensor,    # [num_blocks, page, KVH, D]
    value_cache: torch.Tensor,  # [num_blocks, page, KVH, D]
    block_table: torch.Tensor,  # [S, max_blocks] int32
    seq_lens: torch.Tensor,     # [S] int32, on device
    output: torch.Tensor,       # [S, H, D]
    scale: float,
    block_n: int = 128,
) -> torch.Tensor:
    """One decode step of paged attention.  Shapes depend only on (S, H, D),
    never on the contents of ``seq_lens`` -- safe to capture in a CUDA graph."""
    S, H, D = query.shape
    KVH = key_cache.shape[2]
    page = key_cache.shape[1]
    assert D & (D - 1) == 0, f"head dim {D} must be a power of two"
    assert output.shape == query.shape

    _paged_decode_attn_kernel[(S, H)](
        query, key_cache, value_cache, block_table, seq_lens, output,
        scale,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        block_table.stride(0),
        output.stride(0), output.stride(1),
        N_REP=H // KVH,
        D=D,
        PAGE=page,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    return output
