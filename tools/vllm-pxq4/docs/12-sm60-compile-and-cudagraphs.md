# 12 — CUDA graphs and torch.compile on sm_60 (P100)

Decode on 2xP100 was not kernel-bound. It was bound by Python: ~265 ms of wall
clock per decode step against ~25 ms of GPU kernel time, because the engine ran
with `TORCHDYNAMO_DISABLE=1` and `--enforce-eager`, so nothing was ever captured
into a CUDA graph. This note records why those flags were there, what actually
blocks the compiler and the graph capture on Pascal, and what each one costs to
lift.

## Result

`qwen38-27b-unc-vllm-p1f`, greedy, 512-token streams, `--max-num-seqs 4`,
matched params against the llama.cpp bar. Every number is gated on a
`17x23 -> 391` check taken immediately before *and* after the timed runs — an
`ignore_eos` benchmark is coherence-blind, and a broken model benchmarks
beautifully.

**2xP100, TP=2** (256-token streams, n=4):

| config | decode tok/s | e2e tok/s | vs eager |
|---|---|---|---|
| eager, `TORCHDYNAMO_DISABLE=1` (starting point) | 3.90 | 3.90 | 1.00x |
| breakable CUDA graphs, no torch.compile | 13.08 | 12.83 | 3.35x |
| + packed GDN recurrent decode | 13.27 | 13.01 | 3.40x |

**4xP100, TP=4**, best config, 512-token streams, n=8:

| | median | sd | min | max |
|---|---|---|---|---|
| single-stream e2e | **14.76** | 0.18 | 14.35 | 14.79 |
| single-stream decode | **14.94** | 0.19 | 14.51 | 14.96 |

| concurrency | aggregate tok/s | per-stream median |
|---|---|---|
| 1 | 14.76 | 14.76 |
| 2 | **22.42** | 11.22 |
| 4 | **39.95** | 9.99 |

Bar: llama.cpp layer-split on the same P100s, **17.19 tok/s e2e** (18.2 decode),
single stream, `-np 1`.

* **Single-stream: 14.76 vs 17.19 — 86% of the bar. Not cleared.**
* Concurrency 2: 22.42 aggregate, 1.30x the bar.
* Concurrency 4: 39.95 aggregate, 2.32x the bar.

Correctness is not merely "coherent": greedy output from the graph-captured
TP=4 server is **byte-identical to llama.cpp on the same GGUF** across all three
check prompts (276 and 265 characters of exact match on the long ones).

## The short version

Nothing about Pascal prevents CUDA graphs. Capture works on a P100 exactly as
it does on a V100 — the very first thing to check, and it passes:

```
CUDAGRAPH_EAGER:  OK
CUDAGRAPH_TRITON: OK
```

What stood in the way was a chain of guards and one genuine correctness bug,
none of them a device limitation.

## Why the compiler was off, and what it takes to turn on

| # | Blocker | Where | Verdict |
|---|---------|-------|---------|
| 1 | `has_triton()` returns False below compute capability 7.0 | `torch/utils/_triton.py` | Policy check, not a capability. Triton 3.3 compiles and runs everything inductor emits on sm_60. |
| 2 | ptxas rejects `.L1::evict_first` / `.evict_last` below sm_70 | inductor codegen → triton nvidia backend | Real, but the modifier is a cache *hint*. Stripping it changes speed, never results. |
| 3 | `torch.ops._C.silu_and_mul` missing at import of the quant-fusion passes | `vllm/compilation/passes/pass_manager.py` | Real for this build: those kernels live in `_C_stable_libtorch`, which needs torch ≥ 2.8 stable-ABI headers. Every pass that uses them is off by default. |
| 4 | Two dynamo trace failures on `BasevLLMParameter` | `vllm/model_executor/parameter.py` | torch 2.7 dynamo only. |
| 5 | `torch.Size(...)` constructor untraceable | `vllm/model_executor/layers/attention/attention.py` | torch 2.7 dynamo only. |
| 6 | `guard_filter_fn` forwarded to the backend as an unknown kwarg | `vllm/compilation/wrapper.py` | torch 2.7 has no such option at all. |
| 7 | `torch._functorch.config` / `torch._inductor.config` keys that torch 2.7 lacks | several | Assertion and compile-cache knobs. |

Items 1 and 2 are lifted by `tools/patch_sm60_compile.py` against the venv;
3–7 are in `pascal/vllm-sm60-compile.patch` against the fork.

**All seven are now cleared, and the engine compiles and captures end to end —
but the compiled model produces garbage.** With `cudagraph_mode: NONE` and
inductor on, the model babbles; with inductor off and CUDA graphs on, it is
byte-correct. So the miscompile is in the torch-2.7 inductor/dynamo path, not
in graph capture. This is not chased further here: the win being sought is the
graph capture, and there is a way to have it without inductor (below). The
seven fixes are kept because they are each individually correct and because
they are what makes that diagnosis possible at all — before them the compile
died on an exception long before it could produce a wrong answer.

## What actually delivered: breakable CUDA graphs

The fork already carries `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
(`vllm/compilation/breakable_cudagraph.py`), written for the V100 lane. It
replaces torch.compile's FX-graph splitting with runtime stream-capture breaks:
one capture drives the whole forward and intercepts attention / kv-cache custom
ops at the dispatcher to end the capture, run the op eagerly, and resume. The
captured artifact is a list of zero-arg callables replayed in order.

That is precisely the combination Pascal needs: **CUDA graphs with the
torch.compile pipeline switched off**. It sets `-cc.mode=none`, so none of the
inductor miscompile applies, and it still collapses the per-step Python.
3.77 → 13.08 tok/s, coherent.

Two changes to `pascal_sdpa.py` were required to make the decode step
capturable at all, and both are worth having on their own merits.

### The decode attention had to stop being shaped by host data

The old batched decode path read `seq_lens` on the CPU to decide how many KV
blocks to gather. Its tensor shapes therefore changed from step to step, so a
captured graph would freeze whatever context length happened to be live at
capture time — silently, and wrongly. It also materialised `n_rep`-expanded
fp32 copies of K and V, several times the traffic the math needs.

`pascal_decode_attn.py` replaces it for the all-`qlen==1` case with a triton
kernel whose launch grid is `(batch, heads)` and which reads each sequence's
length from device memory *inside* the kernel. Shapes now depend only on
`(batch, heads, head_dim)`.

Pascal has no `tl.dot`, and single-token decode does not need one — neither
contraction is a matmul:

```
scores[j] = sum_d q[d] * k[j, d]   ->  tl.sum(q[None, :] * k, axis=1)
out[d]    = sum_j p[j] * v[j, d]   ->  tl.sum(p[:, None] * v, axis=0)
```

fp32 accumulation with the usual online-softmax rescaling. Against the torch
reference it matches to fp16 rounding across page sizes, GQA ratios and context
lengths, stays correct when replayed from a CUDA graph with `seq_lens` changing
underneath it, and is ~8x faster than the reference
(106 µs vs 837 µs at S=1, H=12, D=256, ctx=1000).

The builder now declares `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`
rather than inheriting `ALWAYS` from `TritonAttentionMetadataBuilder`. `ALWAYS`
was never true here: prefill and mixed batches still walk a host-driven python
loop over requests and must stay eager.

### The KV cache write had to stop syncing

`do_kv_cache_update` cannot use `reshape_and_cache_flash` — that op lives in
`_C_stable_libtorch` and does not exist in this build at all. The pure-torch
replacement selected the valid rows with a boolean mask, which needs
`bool(mask.all())` on the host: a GPU sync every step *and* a data-dependent
write shape. Either one alone makes the step uncapturable.

Padding rows (`slot_mapping == -1`) are now clamped into slot 0, which is
vLLM's reserved null block (`BlockPool` pops block 0 and marks it `is_null`, so
no request is ever handed it) and whose contents are never read back. Static
shapes, no sync.

## Traps

* **`ignore_eos` throughput numbers prove nothing about correctness.** Both
  configurations that turned out to be broken benchmarked fine. `bench/measure.py`
  refuses to report a number without a `391` check on both sides of the run.
* **The inductor compile-worker blind spot.** Any arch-dependent triton patch
  that probes the *active CUDA device* is a silent no-op inside inductor's
  compile workers, which never initialise CUDA. It appears to work with
  `TORCHINDUCTOR_COMPILE_THREADS=1` and fails with the default pool. Patch
  where the target capability is passed in explicitly — for the eviction
  policy that is the nvidia backend's `make_ptx()`.
* **`torch._functorch.config` is a `ConfigModule`.** Its `__setattr__` rejects
  any name that is not a declared config key, including the name of the method
  you are trying to wrap; the shim has to go into the module `__dict__`.
* **An attention backend that lies about `_cudagraph_support` fails silently.**
  Nothing errors — the graph simply captures a stale shape and the model starts
  producing plausible-looking wrong text.
* The working tree of the fork had an *untested* edit swapping the pure-torch
  KV write for `torch.ops._C_cache_ops.reshape_and_cache_flash`, an op this
  build does not have. It had never run: the serving container predated it.
  Check `git -C 1cat diff` before trusting that what is running is what is on
  disk.

## Where the remaining gap is

At TP=4 the step is ~67 ms. Weight traffic is only ~10 ms of that (22.9 GB over
four cards at ~550 GB/s), so the great majority is still spent outside the
graphs: the eager break segments, 64 per step, one per layer, each leaving and
re-entering stream capture for its attention or GDN op.

The evidence that it is the break machinery and not the kernels:

* going TP=2 → TP=4 (which halves the per-card weight read) buys only
  13.27 → 14.94 tok/s, far less than the kernel time it removes;
* enabling the packed GDN recurrent decode buys 1.4%;
* making full attention a single static triton kernel — an 8x kernel speedup on
  that op — bought 4% in eager.

Closing the last 14% to 17.19 single-stream means removing breaks, not making
kernels faster: make the GDN decode op capture-safe so those 48 layers need no
break at all, the same exercise `pascal_decode_attn.py` performed for the 16
full-attention layers. That is the next lever and it is a bigger one than
either TP=4 or a smaller artifact.

The other honest option is to stop optimising single-stream. The bar is a
`-np 1` llama.cpp number; this server clears it 1.3x at two concurrent streams
and 2.3x at four, which llama.cpp layer-split cannot do at all.
