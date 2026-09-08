"""dequant_ref.py — numpy decoders for every non-PXQ4 ggml type present in a PXA artifact.

WHY THIS FILE EXISTS AT ALL. A "PXQ4 file" is not uniformly PXQ4. PXA_PXQ_BACKBONE rev2
(docs/LEVERS.md, default ON since 2026-07-26; the exact table is recorded in the file as
``pxa.pxq.backbone_map``) promotes and demotes by tensor class, and the real artifact holds
FIVE types — measured, not assumed, by parsing its directory:

    pxq4  325 tensors  12,231,950,336 B   ffn_gate/up/down, attn_q, attn_qkv, attn_gate, attn_output
    q8_0  132          1,621,032,960 B    attn_k, attn_v, ssm_alpha, ssm_beta, output(lm_head), nextn.*
    q6_K    1          1,042,944,000 B    token_embd
    mxfp4  48            802,160,640 B    ssm_out  <-- the surprise: NOT pxq4, ggml type id 39
    f32   360             10,686,464 B    all norms, conv1d, ssm_a, dt_bias

    There is NO f16 tensor in the file: the ``attn_gate_head=f16`` backbone rule only fires
    for per-HEAD gates with ne[1] <= 256, and every attn_gate here is per-channel (ne[1]=6144).
    An f16 decoder is still provided because the rule exists and a future artifact will use it.

Every decoder here is transcribed from the exact C that ggml registers as that type's
``to_float``, cited per function. Getting one of these wrong is a silent catastrophe: a bad
``token_embd`` or ``output`` decode produces a model that loads, runs, and is subtly wrong,
with no error anywhere. ``gguf_to_vllm_oracle.c`` compiles the original C verbatim so the
tests can pin all of them by exact fp32 equality (gate G1).

CONVENTION. ggml stores ``ne = (K, N)`` with K fastest-varying. Every function here returns
row-major ``float32[N, K]`` — i.e. torch's ``[out_features, in_features]``, which is what
``nn.Linear.weight`` holds and what vLLM's loaders narrow. The one type that does NOT admit a
per-row decode is PXQ4 (see reference.py): its rows are scattered across a 64-row panel.
"""

from __future__ import annotations

import numpy as np

from . import gguf_raw as G

QK8_0 = 32
QK_K = 256
QK_MXFP4 = 32

#: e2m1 values, doubled. ggml-common.h:2244-2246 ``kvalues_mxfp4``. The doubling is undone by
#: the /2 baked into ``ggml_e8m0_to_fp32_half``.
KVALUES_MXFP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
                         dtype=np.float32)


def _e8m0_to_fp32_half(e: np.ndarray) -> np.ndarray:
    """``ggml_e8m0_to_fp32_half`` (ggml-impl.h:40-45): the E8M0 scale, halved.

        bits = (e >= 2) ? (e - 1) << 23 : {0x00200000, 0x00400000}[e]

    The two special cases keep e=0 and e=1 in the subnormal range instead of wrapping the
    exponent field, so they must be reproduced exactly rather than approximated by
    ``2.0**(e-128)`` — the naive formula underflows to 0 for e=0 and is wrong for e=1.
    """
    e = e.astype(np.uint32)
    bits = np.where(e >= 2, (e - 1) << np.uint32(23),
                    np.where(e == 1, np.uint32(0x00200000), np.uint32(0x00400000)))
    return bits.astype(np.uint32).view(np.float32)


def dequant_f32(blob, N: int, K: int) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4").reshape(N, K).astype(np.float32, copy=True)


def dequant_f16(blob, N: int, K: int) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f2").reshape(N, K).astype(np.float32)


def dequant_q8_0(blob, N: int, K: int) -> np.ndarray:
    """``dequantize_row_q8_0``: block of 32 = fp16 d + 32 int8, ``y[j] = d * qs[j]``.

    ``block_q8_0`` is ggml-common.h:227-231; 34 B per block, no padding, blocks laid out
    contiguously along K within a row and rows contiguous after each other.
    """
    if K % QK8_0:
        raise ValueError(f"q8_0: K={K} not a multiple of {QK8_0}")
    nb = K // QK8_0
    raw = np.frombuffer(blob, dtype=np.uint8).reshape(N, nb, 34)
    d = raw[:, :, :2].copy().view("<f2").reshape(N, nb).astype(np.float32)
    q = raw[:, :, 2:].view(np.int8).astype(np.float32)
    return (q * d[:, :, None]).reshape(N, K)


def dequant_q6_K(blob, N: int, K: int) -> np.ndarray:
    """``dequantize_row_q6_K`` (ggml-quants.c:3231-3260), transcribed exactly.

    ``block_q6_K`` (ggml-common.h:382-387) is ql[128] qh[64] scales[16] d, 210 B per 256
    elements. The C walks the super-block in two 128-element halves ``n``; within a half, for
    l in 0..31 it forms four quants from ``ql[l]``, ``ql[l+32]`` and the two-bit pairs of
    ``qh[l]``, writing them 32 apart and picking the 8-bit scale by ``is = l/16`` plus a
    0/2/4/6 stride. Reproduced index-for-index below because the interleave is not derivable
    from the block size.
    """
    if K % QK_K:
        raise ValueError(f"q6_K: K={K} not a multiple of {QK_K}")
    nb = K // QK_K
    raw = np.frombuffer(blob, dtype=np.uint8).reshape(N, nb, 210)
    ql = raw[:, :, 0:128]
    qh = raw[:, :, 128:192]
    sc = raw[:, :, 192:208].view(np.int8).astype(np.float32)     # [N,nb,16]
    d = raw[:, :, 208:210].copy().view("<f2").reshape(N, nb).astype(np.float32)

    out = np.empty((N, nb, QK_K), dtype=np.float32)
    for h in range(2):                                   # n = 0, 128
        qlh = ql[:, :, h * 64:(h + 1) * 64].astype(np.int16)      # the C's ql after h*64
        qhh = qh[:, :, h * 32:(h + 1) * 32].astype(np.int16)      # ... qh after h*32
        sch = sc[:, :, h * 8:(h + 1) * 8]                         # ... sc after h*8
        l = np.arange(32)
        is_ = l // 16
        q1 = ((qlh[:, :, l] & 0xF) | (((qhh[:, :, l] >> 0) & 3) << 4)) - 32
        q2 = ((qlh[:, :, l + 32] & 0xF) | (((qhh[:, :, l] >> 2) & 3) << 4)) - 32
        q3 = ((qlh[:, :, l] >> 4) | (((qhh[:, :, l] >> 4) & 3) << 4)) - 32
        q4 = ((qlh[:, :, l + 32] >> 4) | (((qhh[:, :, l] >> 6) & 3) << 4)) - 32
        base = h * 128
        dd = d[:, :, None]
        out[:, :, base + 0:base + 32] = dd * sch[:, :, is_ + 0] * q1
        out[:, :, base + 32:base + 64] = dd * sch[:, :, is_ + 2] * q2
        out[:, :, base + 64:base + 96] = dd * sch[:, :, is_ + 4] * q3
        out[:, :, base + 96:base + 128] = dd * sch[:, :, is_ + 6] * q4
    return out.reshape(N, K)


def dequant_mxfp4(blob, N: int, K: int) -> np.ndarray:
    """``dequantize_row_mxfp4`` (iqk_quantize.cpp:4300-4312) — the ``to_float`` ggml registers
    for GGML_TYPE_MXFP4 in this tree (ggml.c:1385-1401).

        d = e8m0_to_fp32_half(e)
        y[j]      = d * kvalues_mxfp4[qs[j] & 0xF]      j = 0..15
        y[j + 16] = d * kvalues_mxfp4[qs[j] >> 4]

    NOTE THE LAYOUT: this is SPLIT-HALVES, not the interleaved ``(2b, 2b+1)`` pairing PXQ4
    uses. Element j takes the low nibble and element j+16 the high nibble of the SAME byte.
    Assuming PXQ4's pairing here would produce a plausible, wrong ssm_out on all 48 GDN layers.

    ``block_mxfp4`` is 17 B (ggml-common.h:182-187) — one E8M0 byte then 16 nibble bytes —
    and is deliberately not 2-byte aligned, so the reshape below must stay byte-based.
    """
    if K % QK_MXFP4:
        raise ValueError(f"mxfp4: K={K} not a multiple of {QK_MXFP4}")
    nb = K // QK_MXFP4
    raw = np.frombuffer(blob, dtype=np.uint8).reshape(N, nb, 17)
    d = _e8m0_to_fp32_half(raw[:, :, 0])                          # [N,nb]
    qs = raw[:, :, 1:]                                            # [N,nb,16]
    out = np.empty((N, nb, QK_MXFP4), dtype=np.float32)
    out[:, :, :16] = KVALUES_MXFP4[qs & 0x0F]
    out[:, :, 16:] = KVALUES_MXFP4[qs >> 4]
    out *= d[:, :, None]
    return out.reshape(N, K)


#: type id -> per-row decoder. PXQ4 is deliberately absent: it has no per-row codec (its rows
#: are interleaved across a 64-row panel), which is exactly why ggml leaves its ``to_float``
#: NULL (ggml.c:1407-1414) and why vLLM's row-contiguous GGUF sharder cannot touch it.
DECODERS = {
    G.GGML_F32: dequant_f32,
    G.GGML_F16: dequant_f16,
    G.GGML_Q8_0: dequant_q8_0,
    G.GGML_Q6_K: dequant_q6_K,
    G.GGML_MXFP4: dequant_mxfp4,
}


def dequant_any(blob, type_id: int, ne: tuple[int, ...]) -> np.ndarray:
    """Decode any non-PXQ4 tensor to float32 in torch (row-major, reversed-ne) order.

    Handles the 1-D case (norms, biases, ``ssm_a``) by treating it as N=1 and squeezing, and
    the 3-D case (an expert stack, or ``conv1d`` written as ne=(4, 10240)) by folding all but
    the fastest axis into N and reshaping back afterwards.
    """
    if type_id == G.GGML_PXQ4:
        raise ValueError("pxq4 has no per-row decoder; use reference.dequant_blob")
    fn = DECODERS.get(type_id)
    if fn is None:
        raise ValueError(f"no decoder for ggml type {G.type_name(type_id)} ({type_id})")
    K = ne[0]
    N = 1
    for d in ne[1:]:
        N *= d
    out = fn(blob, N, K)
    return out.reshape(tuple(reversed(ne)))
