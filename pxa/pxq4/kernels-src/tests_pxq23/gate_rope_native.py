#!/usr/bin/env python3
"""Gate: rope works on a build whose _C has no rotary_embedding kernel.

Builds the same rotary layer the fork's QSA builds (via get_rope), runs it on
hardware, and checks it against an independent NumPy-free torch reference.
"""
import sys

import torch

FAIL, PASS = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")


def ref_rope(q, k, positions, head_size, rotary_dim, base, is_neox):
    inv = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32,
                                       device=q.device) / rotary_dim))
    freqs = positions.float()[:, None] * inv[None, :]
    cos, sin = freqs.cos(), freqs.sin()

    def rot(x):
        x = x.float().view(x.shape[0], -1, head_size)
        r, pas = x[..., :rotary_dim], x[..., rotary_dim:]
        if is_neox:
            x1, x2 = r[..., : rotary_dim // 2], r[..., rotary_dim // 2:]
            c, s = cos[:, None, :], sin[:, None, :]
            out = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
        else:
            x1, x2 = r[..., 0::2], r[..., 1::2]
            c, s = cos[:, None, :], sin[:, None, :]
            o1, o2 = x1 * c - x2 * s, x2 * c + x1 * s
            out = torch.stack([o1, o2], dim=-1).flatten(-2)
        return torch.cat([out, pas], dim=-1).flatten(1)

    return rot(q), rot(k)


def main():
    import pxq4_vllm.rope_native_fallback  # noqa: F401  (already imported via sitecustomize)

    have = hasattr(torch.ops._C, "rotary_embedding")
    check("_C.rotary_embedding is absent (this build)", not have,
          f"_C has {len(dir(torch.ops._C))} ops")

    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    check("RotaryEmbedding.forward_cuda is forward_native",
          RotaryEmbedding.forward_cuda is RotaryEmbedding.forward_native)
    from vllm.model_executor.custom_op import CustomOp
    check("CustomOp base was NOT dragged onto the native path",
          CustomOp.__dict__.get("forward_cuda") is not CustomOp.__dict__.get("forward_native")
          or "forward_cuda" not in CustomOp.__dict__,
          "the Pascal fused norm/activation providers must keep their forward_cuda")

    if not torch.cuda.is_available():
        check("cuda available", False)
        return finish()

    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name()}")
    torch.manual_seed(0)

    # a rotary layer is a CustomOp: it reads the current vLLM config at construction
    from vllm.config import VllmConfig, set_current_vllm_config
    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()

    for is_neox in (True, False):
        head_size, rotary_dim, base, n_heads, n_tok = 128, 128, 10000, 4, 9
        rope = get_rope(head_size=head_size, max_position=4096,
                        is_neox_style=is_neox, dtype=torch.float16,
                        rope_parameters={"rope_type": "default", "rope_theta": base,
                                         "partial_rotary_factor": 1.0}).to(dev)
        pos = torch.arange(n_tok, device=dev)
        q = torch.randn(n_tok, n_heads * head_size, device=dev, dtype=torch.float16)
        k = torch.randn(n_tok, n_heads * head_size, device=dev, dtype=torch.float16)
        gq, gk = rope(pos, q.clone(), k.clone())
        rq, rk = ref_rope(q, k, pos, head_size, rotary_dim, base, is_neox)
        eq = (gq.float() - rq).abs().max().item()
        ek = (gk.float() - rk).abs().max().item()
        check(f"rope neox={is_neox} query", eq < 5e-2, f"max abs err {eq:.3e}")
        check(f"rope neox={is_neox} key", ek < 5e-2, f"max abs err {ek:.3e}")

    ctx.__exit__(None, None, None)
    return finish()


def finish():
    total = len(PASS) + len(FAIL)
    print(f"\n{len(PASS)}/{total} PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
