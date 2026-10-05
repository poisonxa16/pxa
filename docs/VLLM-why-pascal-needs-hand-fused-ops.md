## Why Pascal needs hand-fused ops

Modern vLLM does not ship fused CUDA kernels for its small operations any more. It ships
**decompositions** — `rms_norm` and `fused_add_rms_norm` are plain torch expression trees in
`vllm/ir/ops/layernorm.py`, `SiluAndMul.forward_native` is `F.silu(x[..., :d]) * x[..., d:]` —
and it relies on Inductor to fuse them back into single kernels at compile time. That is a
reasonable design on hardware Inductor can target.

On a Pascal card it produces a model that spends as much of every token on RMSNorm as on all
of its experts.

There is no Inductor on sm_60, for two independent reasons. `TORCHDYNAMO_DISABLE=1` is
load-bearing in the serving recipe: without it `profile_run` compiles the language model and
dies with `GPUTooOldForTriton` on a compute-capability-6.0 device. And `VLLM_USE_BREAKABLE_CUDAGRAPH=1`,
which the same recipe sets for unrelated reasons, logs on its own that it is "disabling vLLM's
torch.compile pipeline". Either flag alone is enough; nobody recovers the fusions by dropping one.

The fallback that would normally cover this is missing too, and this is the part that is easy to
assume rather than check. The entire layernorm / activation / positional-encoding / cache-write
kernel family lives in vLLM's `_C_stable_libtorch` extension, which is written against the
stable-ABI headers introduced in torch 2.8. This port is pinned to torch 2.7 — the last release
that ships sm_60 kernels at all — so `CMakeLists.txt` skips that target outright via
`VLLM_SKIP_C_STABLE`. There is no such shared library in the serving image, and `torch.ops._C`
contains ten unrelated quantisation ops and none of the ones the model wants. `RMSNorm.forward_cuda`
returns `forward_native` unconditionally; `SiluAndMul.__init__` does
`self.op = getattr(torch.ops._C, "silu_and_mul", None)` and, finding nothing, routes itself to the
native path. The fork's own source comment says so in as many words: *"PASCAL PORT: basic
activation ops live in `_C_stable_libtorch`, which torch 2.7 cannot build."*

So every norm and every activation in every model this stack serves on Pascal or Volta runs as its
eager decomposition, one launch per torch operation. Measured on the 35B MoE at TP=2, each call
site captured in its own CUDA graph so the counts are exact and architecture-independent:

| call site                          | calls / token | kernels each | kernels / token | ms / token |
|------------------------------------|--------------:|-------------:|----------------:|-----------:|
| GemmaRMSNorm, h=2048, with residual |            80 |           11 |             880 |      1.99  |
| GemmaRMSNorm, h=2048, no residual   |             1 |           10 |              10 |      0.02  |
| GemmaRMSNorm q_norm, h=256          |            10 |           10 |             100 |      0.24  |
| GemmaRMSNorm k_norm, h=256          |            10 |           10 |             100 |      0.24  |
| RMSNormGated, h=128 (GDN layers)    |            30 |           12 |             360 |      0.86  |
| SwiGLU (`SiluAndMul`)               |            80 |            2 |             160 |      0.81  |
| **total**                           |               |              |     **1,610**   | **4.16**   |

For scale: the entire routed-expert block of this MoE — the thing the PXQ codec exists for — is
440 kernels and 4.22 ms, in a 33.5 ms decode token. The norms cost the model as much as every
expert in it, and they are not doing arithmetic. Note the fourth column barely moves with the
width: 10 to 12 kernels and 24 to 29 microseconds whether the vector is 128, 256 or 2048 elements
long. These chains are pure launch and latency.

The pack in `pxa_pascal_ops.cu` replaces each of those chains with a single launch: one block per
row, fp32 accumulation, no atomics, nothing allocated, no host synchronisation, safe to capture in
a CUDA graph. **1,610 kernels per decode token become 211.**

Two things about it are worth stating because they are the parts that generalise.

**The seam is not where it looks.** vLLM's IR gives every op a provider registry and a priority
list, which looks like the obvious place to plug in a kernel — and for the model we actually serve
it reaches almost none of the work. Qwen3.5 aliases its norm to `GemmaRMSNorm`, which performs the
residual add itself in torch and only then calls `ir.ops.rms_norm`; it never calls
`fused_add_rms_norm` at all, so a provider registered there is dead code that arms, logs a
reassuring line at boot, and fires zero times. `RMSNormGated`, which the 30 gated-delta-net layers
use, does not touch the IR at any point. The registration that works is one level up, on the
`CustomOp` classes themselves — and even there, `CustomOp` binds *either* `forward_cuda` *or*
`forward_native` once at construction depending on how `CompilationConfig.custom_ops` resolved for
that boot, so patching only one of them arms an object that half the configurations never call.
The rule this taught us, at the cost of three separate arm-and-never-fire mistakes in one
afternoon: **never accept that a hook is installed because it registered. Only a changed kernel
count proves a dispatch moved.**

**A fused norm cannot be bit-identical to the decomposition it replaces, and it is worth knowing
why before you promise otherwise.** Everything elementwise can be matched exactly, and is — aten
evaluates half-precision operations in float and rounds the *result* back to half at every step,
so a three-operation torch chain is three roundings and not one fp32 expression, and reproducing
that (rather than improving on it) is what lets a kernel pass a byte-identity gate. But the
variance is a *reduction*, and torch's fp32 accumulation order for `x.pow(2).mean(-1)` is chosen
per shape by TensorIterator. A search over 366 candidate accumulation layouts found that every
real shape has some layout reproducing torch bit-for-bit, that no single layout reproduces two
shapes, and that one shape in the decode ladder is reproduced by none of them. The fold we ship
lands within 2 fp32 ULP of torch's variance, which the fp16 rounding absorbs for all but about
twenty output elements in a million. That is why the pack is gated behind `PXA_OPS_FUSED` and off
by default, and why its promotion gate is a twenty-prompt greedy byte-identity run on real
hardware rather than a unit test.
