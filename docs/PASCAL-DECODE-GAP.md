# Why vLLM decodes 3x slower than the llama engine on Pascal

Measured 2026-09-05 on the PXA box, two Tesla P100s (cards 1 and 5), 35B MoE PXQ4
(`coder35-moe-pxq4-m1`), image `pxa-vllm:sm60-v15`, TP=2 with custom all-reduce on.
Box not quiet: the V100 pair was in use throughout, so absolute rows are relative and
the ratios are the result.

## The mechanism

Modern vLLM does not ship fused CUDA kernels for its small ops. It ships
**decompositions** and relies on Inductor to fuse them back together. `vllm/ir/ops/
layernorm.py` defines `rms_norm` and `fused_add_rms_norm` as plain torch expression
trees, `RMSNorm.forward_cuda` returns `forward_native` unconditionally
(`layernorm.py:347-359`), and `SiluAndMul` resolves its op to
`torch.ops._C.silu_and_mul`, which the fork's own comment records as unbuildable
against torch 2.7 ("PASCAL PORT: basic activation ops live in `_C_stable_libtorch`").

On sm_60 there is no Inductor. `TORCHDYNAMO_DISABLE=1` is load-bearing — without it
`profile_run` compiles the language model and dies with `GPUTooOldForTriton` — and
`VLLM_USE_BREAKABLE_CUDAGRAPH=1` independently logs "disabling vLLM's torch.compile
pipeline". Two separate reasons, so nobody can quietly recover the fusions by dropping
one flag.

The result is that every norm and every activation in the model runs as its eager
decomposition. This is not a quantisation problem, not a collectives problem and not a
parallel-shape problem; the engine's fusion strategy is simply unavailable on the
architecture we serve, and the llama engine — which hand-writes its fused kernels and
needs no compiler — does not have the problem.

## The measurement

Torch profiler, 20 decode steps, single stream, both ranks within 1.5%.
2433 kernels and 20.74 ms of GPU per token; the step timer puts GPU busy at 80% of the
26.19 ms token, so this is a kernel problem and not a runner problem.

| category | kernels/step | ms/step | share |
|---|---|---|---|
| elementwise | 1766 | 9.196 | 44.3% |
| f16_gemv | 191 | 3.341 | 16.1% |
| pxq4_linear | 120 | 2.327 | 11.2% |
| moe_routed | 80 | 2.165 | 10.4% |
| collective | 82 | 1.575 | 7.6% |
| attn_gdn | 60 | 0.714 | 3.4% |
| topk_route | 40 | 0.595 | 2.9% |
| attn_full | 10 | 0.543 | 2.6% |

73% of every kernel in the token and 44% of the GPU time are the un-fused
`GemmaRMSNorm` and SwiGLU chains. The attribution is arithmetic, not inference:
`rsqrt_kernel_cuda` fires exactly 101.0 times per step, which is precisely this model's
101 `GemmaRMSNorm` invocations (40 input + 40 post-attention + 20 q/k + 1 final).

The llama engine decodes the same model on the same two cards in 10.5 ms — less than
our GPU time alone.

## What this closed

* **Collectives are not the problem.** 1.575 ms, 7.6%. The premise that ~80 all-reduces
  per token over PCIe was the cost is measured and small.
* **PP=2 does not fix decode** (-7.9% single, -16.0% aggregate) because it removes only
  those 1.575 ms while moving the whole vocab-parallel `lm_head` onto the last stage.
  It does, unexpectedly, **win prefill by +22.4%** (418.50 vs 341.85 at 3k), which is
  the shape to remember if a prefill-heavy P100 seat is ever wanted.
* **The GDN recurrent update is not the problem.** 0.714 ms, 3.4%.
* **Rotary is already fused** — one `_triton_mrope_forward` per full-attention layer,
  0.036 ms — and so are the 30 GDN `RMSNormGated` norms, which take the FLA Triton path.
  Two of four proposed kernels retired by measurement before any CUDA was written.
* **The routed MoE is 10.4%**, so a perfect fused MoE block is bounded at ~2.165 ms.
  moe-fused's form A measured +0.55% single decode against the same library with the
  fused path off — a clean three-point isolation — and is not byte-identical (7/20).

## What shipped and what did not

**Shipped, output-neutral:** the dead `torch.zeros` removed from `PXQ4MoEMethod.apply`'s
indexed path (40 dead memset kernels a token).

**Shipped, off by default:** `PXQ4_MOE_EPILOGUE=fused` (11 kernels to 10, bit-identical
at M=1/2 and 2e-6 apart at M=4/8); `PXA_STEP_TIMER`; the `vllm.ir` norm provider seam.

**Did not ship — the router-gate fp16 hook.** The MoE router gate (fp16 [256,2048], 40
per token) runs on cuBLAS as three kernels per call: 120 kernels and 1.129 ms. Adding
`gate` to the sidecar's existing fp16 mmv hook by environment alone armed 110 layers
instead of 70 and cut `f16_gemv` from 3.341 to 2.935 ms — a real, clean −0.406 ms with
identical kernel counts otherwise. But total GPU moved only 20.74 → 20.26 (−2.3%), and
the 20-prompt byte gate against the reference came back **9/20 byte-identical with 20/20
first-token agreement**: perturbing the last bit of a router logit flips a borderline
top-8 expert pick and everything downstream diverges. A measured 2% GPU knob that costs
byte-identity is not a default. Recorded here so it is not rediscovered as a fresh idea.

## The one thing to do next

A fused `GemmaRMSNorm` — the 1+w folded in, the residual add included, the mean in fp32,
both outputs written, matching the native arithmetic order bit for bit — is worth
roughly 35% of the decode token. It is worth more than PP, more than the collectives,
more than the fused MoE block and more than quantising `lm_head`, and unlike any of
those it pays on **every** model this stack serves on Pascal and Volta. `vllm.ir`'s
priority registry and the `_C` namespace are both open seams for it, with no fork edit.
