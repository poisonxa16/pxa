"""tl.dot on Pascal: promote fp16/bf16 operands to fp32 in the Python front end.

Boot 7 of Flash-Next under vLLM on the four P100s (2026-09-06 22:55) reached CUDA-graph capture and
died compiling the QSA paged MQA scorer (ops/qsa.py:59), MLIR diagnostics on:

    Unsupported conversion from f16 to f16
    LLVM ERROR: Unsupported rounding mode for conversion.
    Pipeline failed while executing [`ConvertTritonGPUToLLVM` ...]

Root cause (reproduced standalone in this image with a 64x128 . 128x16 fp16 dot, num_warps=2):
Triton 3.3.1 has no MMA for capability 6.0, so AccelerateMatmul keeps the dot on the FMA path and
"promotes" the fp16 operands to the fp32 accumulator type with tt.fp_to_fp; the NVIDIA
ElementwiseOpToLLVM lowering has no fp16->fp32 entry in its conversion table (it expects f16 dots
to go to MMA), reports "f16 to f16" (it uses f16 as the fp32 intermediate) and aborts.

Fix: do the promotion ourselves, before the IR exists. Wrap triton.language.semantic.dot so that on
capability < 7.0 an fp16/bf16 operand pair is cast with semantic.cast (lowers to arith.extf, which
is supported) and the dot runs as an fp32 FMA dot with input_precision "ieee". Numerics: the FMA
path was going to compute in fp32 anyway; the standalone check matched torch fp32 matmul to 1.4e-5.
core.dot reaches the semantic through the module attribute, so rebinding it here covers EVERY
@triton.jit kernel in the process (QSA scorer, GDN, hyper-connection, MoE ...).

Second Pascal wall behind it (boot 8, 2026-09-06 23:35, ptxas): "Modifier '.evict_first' on 'ld' requires
.target sm_70 or higher" -- tl.load/tl.store eviction_policy="evict_first|evict_last" emit an L2 eviction hint
that ptxas rejects for sm_6x. The same wrapper strips the policy (it is only a cache hint; results are
identical) from semantic.load and semantic.store on capability < 7.0.

Third wall (boot 11, 2026-09-07 00:22, ptxas, last PP rank only): "Feature '.acq_rel' requires .target
sm_70 or higher" from the sampler's penalty bincount (v1/worker/gpu/sample/penalties.py) -- Triton's
tl.atomic_* default to sem="acq_rel", and PTX memory-ordering qualifiers need sm_70. sm_6x has only
plain (relaxed/monotonic) atomics, so every atomic is forced to sem="relaxed" here. Correct for the
counters/histograms/max-reductions these kernels use; a kernel that relied on acquire/release for
cross-CTA hand-off would need an explicit fence, and none of the Pascal-reachable ones do.

PXA_TRITON_DOT_FMA=auto (default; on when capability < 7.0) | on | off
"""

import logging
import os

log = logging.getLogger(__name__)

_MODE = os.environ.get("PXA_TRITON_DOT_FMA", "auto").strip().lower()


def _wanted() -> bool:
    if _MODE in ("off", "0", "no"):
        return False
    if _MODE in ("on", "1", "yes", "force"):
        return True
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability()
        return major < 7
    except Exception:
        return False


def _install() -> None:
    if not _wanted():
        log.info("triton_dot_fma_sm60: not active (mode=%s)", _MODE)
        return
    try:
        import triton.language as tl
        from triton.language import semantic
    except Exception as exc:
        log.debug("triton_dot_fma_sm60: triton not importable (%s)", exc)
        return
    orig = semantic.dot
    if getattr(orig, "_pxa_sm60_dot", False):
        return

    def dot_sm60(lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype, builder):
        try:
            small = (lhs.dtype.is_fp16() or lhs.dtype.is_bf16()) and (rhs.dtype.is_fp16() or rhs.dtype.is_bf16())
        except Exception:
            small = False
        if not small:
            return orig(lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype, builder)
        lhs32 = semantic.cast(lhs, tl.float32, builder)
        rhs32 = semantic.cast(rhs, tl.float32, builder)
        narrow_out = out_dtype is not None and (out_dtype.is_fp16() or out_dtype.is_bf16())
        if narrow_out:
            # fp32 operands cannot accumulate into fp16 in Triton: accumulate in fp32, cast back once.
            acc32 = semantic.cast(acc, tl.float32, builder) if acc is not None else None
            res = orig(lhs32, rhs32, acc32, "ieee", max_num_imprecise_acc, tl.float32, builder)
            return semantic.cast(res, out_dtype, builder)
        return orig(lhs32, rhs32, acc, "ieee", max_num_imprecise_acc, out_dtype, builder)

    dot_sm60._pxa_sm60_dot = True
    dot_sm60.__wrapped__ = orig
    semantic.dot = dot_sm60

    # eviction hints: drop them for loads and stores (positional or keyword), keep everything else
    import inspect
    stripped = []
    for name in ("load", "store"):
        fn = getattr(semantic, name, None)
        if fn is None or getattr(fn, "_pxa_sm60_evict", False):
            continue
        try:
            params = list(inspect.signature(fn).parameters)
            idx = params.index("eviction_policy")
        except (ValueError, TypeError):
            continue

        def make(fn, idx):
            def wrapped(*args, **kwargs):
                if "eviction_policy" in kwargs:
                    kwargs["eviction_policy"] = ""
                elif len(args) > idx:
                    args = args[:idx] + ("",) + args[idx + 1:]
                return fn(*args, **kwargs)
            wrapped._pxa_sm60_evict = True
            wrapped.__wrapped__ = fn
            return wrapped
        setattr(semantic, name, make(fn, idx))
        stripped.append(name)
    # atomics: PTX .acq_rel/.acquire/.release qualifiers need sm_70; force relaxed ordering
    relaxed = []
    for name in ("atomic_cas", "atomic_xchg", "atomic_add", "atomic_max", "atomic_min",
                 "atomic_and", "atomic_or", "atomic_xor"):
        fn = getattr(semantic, name, None)
        if fn is None or getattr(fn, "_pxa_sm60_relaxed", False):
            continue
        try:
            params = list(inspect.signature(fn).parameters)
            idx = params.index("sem")
        except (ValueError, TypeError):
            continue

        def make_relaxed(fn, idx):
            def wrapped(*args, **kwargs):
                if "sem" in kwargs:
                    kwargs["sem"] = "relaxed"
                elif len(args) > idx:
                    args = args[:idx] + ("relaxed",) + args[idx + 1:]
                else:
                    kwargs["sem"] = "relaxed"
                return fn(*args, **kwargs)
            wrapped._pxa_sm60_relaxed = True
            wrapped.__wrapped__ = fn
            return wrapped
        setattr(semantic, name, make_relaxed(fn, idx))
        relaxed.append(name)
    # Triton 3.3.1 front end rejects `a or b or c` ("chained boolean operators ... not supported");
    # the fork's sampler kernels (penalties.py, boot 12 2026-09-07) are written for a newer Triton
    # that accepts them. Fold the chain left-to-right into nested two-operand BoolOps -- the same
    # short-circuit-free semantics Triton gives `(a or b) or c`.
    chain = False
    try:
        import ast
        from triton.compiler import code_generator as cg
        orig_bool = cg.CodeGenerator.visit_BoolOp
        if not getattr(orig_bool, "_pxa_sm60_chain", False):
            def visit_BoolOp(self, node):
                if len(node.values) > 2:
                    folded = node.values[0]
                    for v in node.values[1:]:
                        nxt = ast.BoolOp(op=node.op, values=[folded, v])
                        ast.copy_location(nxt, node)
                        folded = nxt
                    node = folded
                return orig_bool(self, node)
            visit_BoolOp._pxa_sm60_chain = True
            visit_BoolOp.__wrapped__ = orig_bool
            cg.CodeGenerator.visit_BoolOp = visit_BoolOp
        chain = True
    except Exception as exc:
        log.warning("triton_dot_fma_sm60: could not patch visit_BoolOp (%s)", exc)
    log.info("triton_dot_fma_sm60: tl.dot promotes fp16/bf16 operands to fp32 (ieee FMA) on sm<70; eviction hints stripped from %s; relaxed atomics for %s; chained bool ops folded=%s",
             ",".join(stripped) or "nothing", ",".join(relaxed) or "nothing", chain)


_install()
