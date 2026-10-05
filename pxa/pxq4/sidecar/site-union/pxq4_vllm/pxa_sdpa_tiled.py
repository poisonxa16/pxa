# SPDX-License-Identifier: Apache-2.0
"""Tiled online-softmax prefill SDPA for the Pascal backend — kills the OOM class.

PascalSDPAImpl._sdpa_one materialises `scores = [H, qlen, ctx]` in fp32: at chunk 2048
against a 25k context that is a single 3.3 GiB allocation, and the 576 MiB instance of
it OOM-killed two boots this session (fragmentation-timing dependent). This module
replaces _sdpa_one with a flash-style tiled version: q-tiles x kv-tiles with a running
(max, denom, acc) — peak transient is H x TQ x TK fp32 (~33 MiB at 256x4096) instead of
H x qlen x ctx. Pure torch, eager-only path (prefill never captures), no Triton.

Numerics: fp32 throughout, standard online-softmax rescaling; agrees with the one-shot
softmax to fp32 rounding (parity-tested; not bit-identical — association differs).
Small problems (qlen==1 or qlen*ctx below the tile budget) keep the original one-shot
path, byte-identical behaviour there.

Env: PXA_SDPA_TILED=0 disables (original _sdpa_one kept).
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_TQ = 256
_TK = 4096
_patched = False


def maybe_patch() -> None:
    global _patched
    if _patched:
        return
    if os.getenv("PXA_SDPA_TILED", "1") == "0":
        return
    try:
        from vllm.v1.attention.backends import pascal_sdpa as _ps
    except Exception:
        return  # backend absent on this platform (e.g. V100 image): nothing to do
    _patched = True

    orig = _ps.PascalSDPAImpl._sdpa_one

    def _sdpa_one(self, q, k, v, causal):
        qlen, H, D = q.shape
        ctx = k.shape[0]
        # small problems: keep the original one-shot math (byte-identical there)
        if qlen == 1 or qlen * ctx <= _TQ * _TK:
            return orig(self, q, k, v, causal)

        qh = q.permute(1, 0, 2).contiguous()          # [H, qlen, D]
        kh = k.permute(1, 2, 0).contiguous()          # [H, D, ctx]
        vh = v.permute(1, 0, 2).contiguous()          # [H, ctx, D]
        out = torch.empty(H, qlen, D, dtype=q.dtype, device=q.device)
        base = ctx - qlen                             # absolute position of query 0

        for q0 in range(0, qlen, _TQ):
            tq = min(_TQ, qlen - q0)
            qt = qh[:, q0:q0 + tq]                    # [H, tq, D]
            qpos = torch.arange(base + q0, base + q0 + tq, device=q.device)
            m = torch.full((H, tq), float("-inf"), dtype=torch.float32, device=q.device)
            l = torch.zeros((H, tq), dtype=torch.float32, device=q.device)
            acc = torch.zeros((H, tq, D), dtype=torch.float32, device=q.device)
            kmax = int(qpos[-1]) + 1 if causal else ctx
            for k0 in range(0, kmax, _TK):
                tk = min(_TK, kmax - k0)
                s = torch.bmm(qt, kh[:, :, k0:k0 + tk]) * self.scale   # [H, tq, tk]
                if causal:
                    kpos = torch.arange(k0, k0 + tk, device=q.device)
                    s.masked_fill_((kpos.unsqueeze(0) > qpos.unsqueeze(1)).unsqueeze(0),
                                   float("-inf"))
                m_new = torch.maximum(m, s.amax(dim=-1))
                alpha = torch.exp(m - m_new)
                p = torch.exp(s - m_new.unsqueeze(-1))
                l = l * alpha + p.sum(dim=-1)
                acc = acc * alpha.unsqueeze(-1) + torch.bmm(p, vh[:, k0:k0 + tk])
                m = m_new
            out[:, q0:q0 + tq] = acc / l.unsqueeze(-1)
        return out.permute(1, 0, 2)

    _ps.PascalSDPAImpl._sdpa_one = _sdpa_one
    logger.info_once("pxa tiled prefill SDPA installed (PXA_SDPA_TILED=0 disables)")
