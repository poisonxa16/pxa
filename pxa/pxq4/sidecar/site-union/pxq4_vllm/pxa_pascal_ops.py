# SPDX-License-Identifier: Apache-2.0
"""Route vLLM's norm and activation decompositions to hand-fused Pascal/Volta kernels.

WHY THIS MODULE EXISTS. Modern vLLM does not ship fused CUDA kernels for its small ops.
It ships torch expression trees and relies on Inductor to fuse them. On sm_60 there is no
Inductor -- ``TORCHDYNAMO_DISABLE=1`` is load-bearing (without it ``profile_run`` compiles
the language model and dies with GPUTooOldForTriton), and this stack's recipe also sets
``VLLM_USE_BREAKABLE_CUDAGRAPH=1``, which on its own logs "disabling vLLM's torch.compile
pipeline". Two independent reasons, so nobody recovers the fusions by dropping one flag.
The whole layernorm / activation / positional-encoding family that would otherwise cover
this lives in the ``_C_stable_libtorch`` extension, which needs torch >= 2.8 stable-ABI
headers and is skipped on this build (``VLLM_SKIP_C_STABLE``); there is no such library in
the serving image, and ``torch.ops._C`` holds ten unrelated ops.

Measured on the 35B MoE at TP=2, each op captured in its own CUDA graph (vLLM pipeline-parallel seat):

    call site                          per token   kernels each   kernels/token   ms/token
    GemmaRMSNorm h=2048 with residual        80             11             880       1.987
    GemmaRMSNorm h=2048 no residual           1             10              10       0.023
    GemmaRMSNorm q_norm h=256                10             10             100       0.242
    GemmaRMSNorm k_norm h=256                10             10             100       0.237
    RMSNormGated h=128 (GDN layers)          30             12             360       0.861
    SwiGLU (SiluAndMul)                      80              2             160       0.809
    TOTAL                                                                  1610       4.159

against 440 kernels and 4.22 ms for every routed expert in the model, in a 33.5 ms token.

THREE SEAMS, AND WHY IT IS NOT ONE. The obvious hook -- ``vllm.ir``'s provider registry --
reaches almost none of this model. Qwen3_5 aliases its norm to ``GemmaRMSNorm``
(qwen3_5.py:41, qwen3_next.py:35), whose ``forward_native`` does the residual add ITSELF
in torch and then calls ``ir.ops.rms_norm`` -- never ``fused_add_rms_norm``, which this
model does not reach at all -- so an ir provider can only ever take the 8-kernel tail and
must leave the 3-kernel head. And ``RMSNormGated`` (the 30 GDN layers) never touches
``vllm.ir`` at any point. So:

  1. CLASS PATCH on ``GemmaRMSNorm.forward_cuda`` and ``RMSNormGated.forward_cuda``, the
     same shape of hook as ``pxa_sm60_f16`` on the dense linear path. This is the seam
     that gets the 1450 norm kernels, and it is the only one that can.
  2. IR PROVIDER on ``vllm.ir.ops.rms_norm`` / ``fused_add_rms_norm`` for models that use
     the standard ``RMSNorm`` class -- not this one, but the dense models on this stack.
  3. ``_C`` NAMESPACE FRAGMENT defining ``silu_and_mul``. activation.py:144-147 is
     literally ``self.op = getattr(torch.ops._C, "silu_and_mul", None); if self.op is
     None: self._forward_method = self.forward_native``, so the moment the name exists the
     fork takes itself off the native path with no patch at all. A FRAGMENT coexists with
     the one ``TORCH_LIBRARY(_C)`` the fork owns; on an image where the real op DOES exist
     (torch 2.10 / sm70-v15) a duplicate definition would be a hard crash at import, so
     the definition is hasattr-guarded and the whole thing is wrapped in try/except with a
     class patch as the fallback.

ARITHMETIC, stated plainly because it decides whether this can ever default ON. The
kernels reproduce the ORDER of the torch chains they replace, not merely their algebra.
But the variance is a REDUCTION, and a probe over 366 candidate accumulation layouts
(bench/reduce_probe.py) established that torch's fp32 order for ``x.pow(2).mean(-1)`` is
chosen per shape by TensorIterator: every real shape has some layout reproducing it
bit-for-bit, no layout reproduces two, and 8x2048 is reproduced by none. So bit-identity
with the native path is NOT attainable at any block geometry. The fold used lands within
2 fp32 ULP of torch's variance and changes about 20 fp16 output elements per million.
That is why this module is OFF by default and why the promotion gate is 20-prompt greedy
byte identity on a real pair, not a unit test.

Env:
  PXA_OPS_FUSED=0            change nothing at all
  PXA_OPS_FUSED unset        ARCH DEFAULT: _ARCH_DEFAULT_SM6X on sm_6x devices, off everywhere
                             else -- see _arch_default(). An explicit value always wins.
  PXA_OPS_ARCH_DEFAULT=...   what "unset on sm_6x" means (default "0" until the Phase P re-gate
                             on coder35 m3 passes; promotion = flipping this ONE string to
                             "gemma,silu" in a second commit)
  PXA_OPS_FUSED=all          arm every seam
  PXA_OPS_FUSED=gemma,gdn,silu,ir
                             arm a subset. 'gemma' is the 1090 kernels of GemmaRMSNorm,
                             'gdn' the 360 of RMSNormGated, 'silu' the 160 of SwiGLU,
                             'ir' the provider for standard-RMSNorm models.
  PXA_OPS_TRACE=1            count calls and declines per op and dump them every
                             PXA_OPS_TRACE_EVERY (default 2000) calls, so an in-engine
                             dispatch can be proved by count rather than by log line
  PXA_OPS_MAX_ROWS=<n>       decline the fused path above n rows, so prefill keeps the
                             native arithmetic and only the captured decode path is
                             perturbed. 0 (default) = no limit.
"""

from __future__ import annotations

import os

import torch

logger = None
_ARMED = False
_ARMED_LIST: list[str] = []

_OPS = (
    "gemma_add_rms_norm_out",
    "gemma_rms_norm_out",
    "rms_norm_gated_out",
    "rms_norm_out",
    "fused_add_rms_norm_out",
    "silu_and_mul_out",
)


def _have(name: str) -> bool:
    return hasattr(torch.ops, "pxq4") and hasattr(torch.ops.pxq4, name)


def _capturing() -> bool:
    """True while a CUDA graph is being captured on this stream.

    Every lazily built cache in this module refuses to allocate during capture and falls
    back to the native path instead. Allocating under capture would be served from the
    graph-private pool and pinned for the life of that graph; in practice the caches are
    all built during profile_run and warmup, which run before any capture, so the guard
    never fires in a healthy boot -- it is there so that an unhealthy one degrades to
    "slow" rather than to "wrong".
    """
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


# The default, when PXA_OPS_FUSED is unset, is decided by the ARCHITECTURE and not by a
# launcher flag. This pack exists because sm_60 has no Inductor, so vLLM's decomposition
# strategy has nothing to fuse it back together; on sm_70 the same seat runs
# mode=VLLM_COMPILE with backend=inductor and custom_ops="none", i.e. the strategy works
# as designed, and a fused opaque custom op there competes with an already-fused kernel and
# can also stop Inductor fusing across its boundary. Defaulting on both would be arming a
# pack whose premise holds on one card and not the other, and a default that depends on
# whoever sets an environment variable is not a default.
#
# So: sm_6x gets the shipping shape, everything else gets nothing, and an explicit
# PXA_OPS_FUSED always wins over both.
#
# NOT YET GRANTED (a pre-registered gate-1, commit 1 of 2). Until the re-gate with PXA_OPS_FUSED UNSET
# on the shipping checkpoint (coder35 m3) is 20/20 byte-identical vs the =0 control at np1
# AND np2, decisive top-token and needle green, the arch default is "0" and the pack arms
# only when someone sets PXA_OPS_FUSED explicitly. Flipping this one string to "gemma,silu"
# is the whole of the promotion (commit 2). Not "all": measured 40.37 < 41.34 on the pair.
_ARCH_DEFAULT_SM6X = os.environ.get("PXA_OPS_ARCH_DEFAULT", "0")


def _device_major() -> "int | None":
    """Compute-capability major of the first visible device, WITHOUT creating a CUDA context.

    This runs at plugin load in every process, including the API server, which never touches
    a GPU; torch.cuda.get_device_capability() there would plant a CUDA context (and its
    ~300 MiB) on a card that is already at 0.96 utilisation. NVML answers the same question
    from the driver. The torch call is the last resort and only if NVML is unavailable.
    """
    try:
        import pynvml  # noqa: PLC0415  (ships with vLLM)
        pynvml.nvmlInit()
        try:
            vis = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("NVIDIA_VISIBLE_DEVICES")
            idx = 0
            if vis and vis not in ("all", "void"):
                first = vis.split(",")[0].strip()
                if first.isdigit():
                    idx = int(first)
                elif first.startswith("GPU-"):
                    h = pynvml.nvmlDeviceGetHandleByUUID(first.encode() if isinstance(first, str) else first)
                    return int(pynvml.nvmlDeviceGetCudaComputeCapability(h)[0])
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            return int(pynvml.nvmlDeviceGetCudaComputeCapability(h)[0])
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.get_device_capability()[0])
    except Exception:
        pass
    return None


_ARCH_DEFAULT_CACHE: "dict[str, str]" = {}


def _arch_default() -> str:
    """Resolved once per process; "0" whenever the device cannot be identified."""
    if "v" not in _ARCH_DEFAULT_CACHE:
        major = _device_major()
        _ARCH_DEFAULT_CACHE["v"] = _ARCH_DEFAULT_SM6X if major == 6 else "0"
    return _ARCH_DEFAULT_CACHE["v"]


def _modes() -> set[str]:
    raw = os.getenv("PXA_OPS_FUSED")
    if raw is None or raw.strip() == "":
        raw = _arch_default()
    raw = raw.strip().lower()
    if raw in ("0", "off", "none", "false"):
        return set()
    if raw in ("all", "1", "on", "true"):
        return {"gemma", "gdn", "silu", "ir"}
    return {p.strip() for p in raw.split(",") if p.strip()}


def _max_rows() -> int:
    try:
        return int(os.getenv("PXA_OPS_MAX_ROWS", "0"))
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# PXA_OPS_TRACE: per-op call counting, so a dispatch can be proved by COUNT.
#
# This exists because a registration that succeeds and never fires looks exactly like one
# that works: the log line is the same. Three separate arm-and-never-fire mistakes were
# made building this pack, and every one of them was caught by a count and none by a log.
# With PXA_OPS_TRACE=1 each fused op counts its calls and its declines and dumps them
# every PXA_OPS_TRACE_EVERY calls, so an in-engine run can be checked against the expected
# calls per token instead of against a boot message.
#
# OFF by default and the flag is read ONCE at patch time into a module-level bool, so with
# tracing off the cost in the hot path is one global lookup and a branch.
# ---------------------------------------------------------------------------
_TRACE = False
_TRACE_EVERY = 2000
_CALLS: dict[str, int] = {}


def _tick(op: str, taken: bool) -> None:
    k = op if taken else op + ".declined"
    n = _CALLS.get(k, 0) + 1
    _CALLS[k] = n
    if n % _TRACE_EVERY == 0 and logger is not None:
        logger.info("PXA_OPS_TRACE pascal-ops calls: %s",
                    ", ".join(f"{a}={b}" for a, b in sorted(_CALLS.items())))


def _rows(t: torch.Tensor) -> int:
    n = 1
    for s in t.shape[:-1]:
        n *= s
    return n


# ---------------------------------------------------------------------------
# fake / meta kernels
#
# MANDATORY, not optional. Without a meta kernel the ops are opaque to dynamo tracing and
# to the shape-propagation that graph capture does, and the failure shows up as an
# incomprehensible model error rather than as a missing registration. Every out-parameter
# op's meta is a shape assertion and nothing else.
# ---------------------------------------------------------------------------
_FAKES_DONE = False


def _register_fakes() -> None:
    global _FAKES_DONE
    if _FAKES_DONE:
        return
    from torch.library import register_fake

    def _same(a, b):
        assert a.shape == b.shape, f"pxa_pascal: shape mismatch {a.shape} vs {b.shape}"

    if _have("gemma_add_rms_norm_out"):
        @register_fake("pxq4::gemma_add_rms_norm_out")
        def _(out, residual_out, x, residual, weight, eps):  # noqa: ANN001,ANN202
            _same(out, x)
            _same(residual_out, x)
            _same(residual, x)
            assert weight.shape[0] == x.shape[-1]
            return None

    if _have("gemma_rms_norm_out"):
        @register_fake("pxq4::gemma_rms_norm_out")
        def _(out, x, weight, eps):  # noqa: ANN001,ANN202
            _same(out, x)
            assert weight.shape[0] == x.shape[-1]
            return None

    if _have("rms_norm_gated_out"):
        @register_fake("pxq4::rms_norm_gated_out")
        def _(out, x, z, weight, eps, act):  # noqa: ANN001,ANN202
            _same(out, x)
            _same(z, x)
            assert weight.shape[0] == x.shape[-1]
            return None

    if _have("rms_norm_out"):
        @register_fake("pxq4::rms_norm_out")
        def _(out, x, weight, eps):  # noqa: ANN001,ANN202
            _same(out, x)
            return None

    if _have("fused_add_rms_norm_out"):
        @register_fake("pxq4::fused_add_rms_norm_out")
        def _(out, residual_out, x, residual, weight, eps):  # noqa: ANN001,ANN202
            _same(out, x)
            _same(residual_out, x)
            return None

    if _have("silu_and_mul_out"):
        @register_fake("pxq4::silu_and_mul_out")
        def _(out, x):  # noqa: ANN001,ANN202
            assert out.shape[-1] * 2 == x.shape[-1]
            return None

    _FAKES_DONE = True


# ---------------------------------------------------------------------------
# seam 1a: GemmaRMSNorm
# ---------------------------------------------------------------------------
def _patch_gemma() -> bool:
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    if getattr(GemmaRMSNorm, "_pxa_patched", False):
        return True
    # forward_NATIVE, not forward_cuda, and this is not a detail.
    #
    # CustomOp.dispatch_forward picks the branch ONCE at __init__ from
    # CompilationConfig.custom_ops: "all" gives forward_cuda, "none" gives forward_native.
    # Patching only forward_cuda arms an object that a "none" boot never calls, which is
    # an arm-and-never-fire failure -- measured, not theorised: bench/pascal_ops_census.py
    # showed the norms still at 11 kernels with the patch visibly installed, because that
    # config had resolved custom_ops to "none".
    #
    # forward_native covers BOTH, because GemmaRMSNorm.forward_cuda's last line is
    # `return self.forward_native(x, residual)` and every branch above it is gated on
    # capability 70 plus an env flag, none of which can fire on a Pascal card.
    original = GemmaRMSNorm.forward_native
    limit = _max_rows()

    def _w32(self) -> torch.Tensor | None:
        """The fp32 view of the raw Gemma weight, cached per module.

        The kernel folds the ``+ 1.0`` in itself, so caching the plain fp32 cast also
        deletes the two launches ``weight = self.weight.data.float() + 1.0`` costs on
        EVERY call, on top of the eleven the norm chain itself costs.
        """
        w = self.weight.data
        c = getattr(self, "_pxa_w32", None)
        if c is not None and getattr(self, "_pxa_w32_src", None) == w.data_ptr():
            return c
        if _capturing():
            return None
        c = w.float().contiguous()
        self._pxa_w32 = c
        self._pxa_w32_src = w.data_ptr()
        return c

    def forward_native(self, x, residual=None):  # noqa: ANN001,ANN202
        # The fast path is declined -- never approximated -- whenever the inputs are not
        # exactly the case the kernel reproduces. Every decline falls through to the
        # untouched original method, so an unexpected shape is slow and correct.
        if (
            x.is_cuda
            and x.dtype is torch.float16          # out dtype is x's ORIGINAL dtype
            and x.is_contiguous()
            and x.shape[-1] % 2 == 0
            and (limit <= 0 or _rows(x) <= limit)
        ):
            w32 = _w32(self)
            if w32 is not None:
                if residual is None:
                    out = torch.empty_like(x)
                    torch.ops.pxq4.gemma_rms_norm_out(
                        out, x, w32, self.variance_epsilon)
                    if _TRACE:
                        _tick("gemma_plain", True)
                    return out
                if (
                    residual.is_cuda
                    and residual.shape == x.shape
                    and residual.is_contiguous()
                    and residual.dtype in (torch.float16, torch.float32)
                ):
                    out = torch.empty_like(x)
                    res_out = torch.empty(x.shape, dtype=torch.float32,
                                          device=x.device)
                    torch.ops.pxq4.gemma_add_rms_norm_out(
                        out, res_out, x, residual, w32, self.variance_epsilon)
                    if _TRACE:
                        _tick("gemma_add", True)
                    return out, res_out
        if _TRACE:
            _tick("gemma", False)
        return original(self, x, residual)

    GemmaRMSNorm.forward_native = forward_native
    GemmaRMSNorm._pxa_patched = True
    return True


# ---------------------------------------------------------------------------
# seam 1b: RMSNormGated -- the 30 GDN layers
# ---------------------------------------------------------------------------
def _patch_gdn() -> bool:
    from vllm.model_executor.layers.layernorm import RMSNormGated

    if getattr(RMSNormGated, "_pxa_patched", False):
        return True
    # BOTH branches, because unlike GemmaRMSNorm these do not share a tail:
    # RMSNormGated.forward_cuda calls the FLA Triton rmsnorm_fn directly and never reaches
    # forward_native, so whichever one CompilationConfig.custom_ops selects must be
    # covered on its own. Each keeps its own original as the decline path, so the fused
    # kernel replaces exactly one shipped implementation and never chains them.
    orig_cuda = RMSNormGated.forward_cuda
    orig_native = RMSNormGated.forward_native
    limit = _max_rows()

    def _w32(self):  # noqa: ANN001,ANN202
        w = self.weight.data
        c = getattr(self, "_pxa_w32", None)
        if c is not None and getattr(self, "_pxa_w32_src", None) == w.data_ptr():
            return c
        if _capturing():
            return None
        c = w.float().contiguous()
        self._pxa_w32 = c
        self._pxa_w32_src = w.data_ptr()
        return c

    def _fast(self, x, z):  # noqa: ANN001,ANN202
        """Returns the fused result, or None if this call is declined."""
        # group_size is the one parameter that changes the op rather than the shape: a
        # grouped norm reduces over sub-vectors and is a different kernel, so it is
        # declined outright rather than approximated by the ungrouped one.
        if (
            z is not None
            and self.group_size is None
            and self.norm_before_gate
            and self.activation in ("silu", "swish", "sigmoid")
            and x.is_cuda
            and x.dtype in (torch.float16, torch.float32)
            and z.dtype == x.dtype
            and z.shape == x.shape
            and x.is_contiguous()
            and z.is_contiguous()
            and x.shape[-1] % 2 == 0
            and (limit <= 0 or _rows(x) <= limit)
        ):
            w32 = _w32(self)
            if w32 is not None:
                act = 1 if self.activation == "sigmoid" else 0
                out = torch.empty_like(x)
                torch.ops.pxq4.rms_norm_gated_out(out, x, z, w32, self.eps, act)
                if _TRACE:
                    _tick("rms_norm_gated", True)
                return out
        if _TRACE:
            _tick("rms_norm_gated", False)
        return None

    def forward_cuda(self, x, z=None):  # noqa: ANN001,ANN202
        r = _fast(self, x, z)
        return r if r is not None else orig_cuda(self, x, z)

    def forward_native(self, x, z=None):  # noqa: ANN001,ANN202
        r = _fast(self, x, z)
        return r if r is not None else orig_native(self, x, z)

    RMSNormGated.forward_cuda = forward_cuda
    RMSNormGated.forward_native = forward_native
    RMSNormGated._pxa_patched = True
    return True


# ---------------------------------------------------------------------------
# seam 2: the vllm.ir provider, for models that use the standard RMSNorm class
#
# NOT this model -- every norm in Qwen3_5 is a GemmaRMSNorm and is taken by seam 1 -- but
# the dense models on this stack do use it. Registered ahead of whatever the platform put
# there: on a real boot of this image the priority is ['vllm_c', 'native'], and vllm_c is
# a HOLLOW SHELL, because kernels/vllm_c.py falls straight back to the native
# decomposition the moment ``_has_c_op('rms_norm')`` is false, which it is on this build.
# So the shipped seat has a configured fast path that silently is not there.
# ---------------------------------------------------------------------------
def _patch_ir() -> bool:
    from vllm import ir

    def _ok(x, weight, variance_size) -> bool:
        return (
            variance_size is None          # a partial-variance norm is a different op
            and weight is not None
            and x.is_cuda
            and x.dtype is torch.float16
            and weight.dtype is torch.float16
            and x.is_contiguous()
            and x.shape[-1] % 2 == 0
        )

    def _sup_plain(x: torch.Tensor, weight: torch.Tensor | None, epsilon: float,
                   variance_size: int | None = None) -> bool:
        return _ok(x, weight, variance_size)

    def _sup_fused(x: torch.Tensor, x_residual: torch.Tensor,
                   weight: torch.Tensor | None, epsilon: float,
                   variance_size: int | None = None) -> bool:
        return (
            _ok(x, weight, variance_size)
            and x_residual.dtype == x.dtype
            and x_residual.shape == x.shape
            and x_residual.is_contiguous()
        )

    # The annotations are load-bearing: register_impl runs torch's infer_schema over the
    # function and refuses anything whose inferred schema is not character-identical to
    # the native op's, so the types, the names, the order and the defaults must all match
    # vllm/ir/ops/layernorm.py exactly.
    def _plain(x: torch.Tensor, weight: torch.Tensor | None, epsilon: float,
               variance_size: int | None = None) -> torch.Tensor:
        out = torch.empty_like(x)
        torch.ops.pxq4.rms_norm_out(out, x, weight, epsilon)
        if _TRACE:
            _tick("ir.rms_norm", True)
        return out

    def _fused(x: torch.Tensor, x_residual: torch.Tensor,
               weight: torch.Tensor | None, epsilon: float,
               variance_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty_like(x)
        res = torch.empty_like(x_residual)
        torch.ops.pxq4.fused_add_rms_norm_out(out, res, x, x_residual, weight, epsilon)
        if _TRACE:
            _tick("ir.fused_add_rms_norm", True)
        return out, res

    armed = []
    if _have("rms_norm_out") and "pxa" not in ir.ops.rms_norm.impls:
        ir.ops.rms_norm.register_impl("pxa", supported=True,
                                      supports_args=_sup_plain)(_plain)
        ir.ops.rms_norm.set_default(["pxa", "native"])
        armed.append("rms_norm")
    if (_have("fused_add_rms_norm_out")
            and "pxa" not in ir.ops.fused_add_rms_norm.impls):
        ir.ops.fused_add_rms_norm.register_impl("pxa", supported=True,
                                                supports_args=_sup_fused)(_fused)
        ir.ops.fused_add_rms_norm.set_default(["pxa", "native"])
        armed.append("fused_add_rms_norm")

    # The config installs its own priority later, in v1/worker/worker_base.py, which would
    # overwrite ours. Wrap that install point so we are re-prepended whenever it runs,
    # rather than depending on which of the two happens to go first.
    try:
        from vllm.config.kernel import IrOpPriorityConfig

        if not getattr(IrOpPriorityConfig, "_pxa_wrapped", False):
            _orig_set_default = IrOpPriorityConfig.set_default

            def set_default(self):  # noqa: ANN001,ANN202
                _orig_set_default(self)
                for name in ("rms_norm", "fused_add_rms_norm"):
                    op = getattr(ir.ops, name, None)
                    if op is not None and "pxa" in op.impls:
                        op.set_default(["pxa", "native"])

            IrOpPriorityConfig.set_default = set_default
            IrOpPriorityConfig._pxa_wrapped = True
    except Exception:
        logger.exception("pxa: could not wrap IrOpPriorityConfig.set_default")

    return bool(armed)


# ---------------------------------------------------------------------------
# seam 3: SwiGLU, through the _C namespace the fork already probes for
# ---------------------------------------------------------------------------
def _patch_silu() -> str:
    """Returns the mechanism(s) that armed: 'fragment', 'class', 'fragment+class', ''.

    BOTH are installed when both work, because which one the engine reaches depends on
    CompilationConfig.custom_ops, which is decided per boot: with custom_ops resolved to
    "all" -- the sm_60 case, since there is no Inductor -- CustomOp dispatches
    forward_cuda, which is the branch that consults torch.ops._C; with custom_ops resolved
    to "none" it dispatches forward_native and never looks at _C at all. Arming only the
    _C name would therefore be silently inert on any boot that took the second path, and
    that is exactly the kind of arm-and-never-fire failure this path already produced once.
    The two mechanisms call the same kernel, so having both cannot disagree.
    """
    how = ""
    # Preferred: define the name the fork looks for. hasattr-guarded, because on an image
    # where the real op exists a duplicate schema definition is a hard crash at import --
    # and that would take down every seat sharing this sidecar, not just this arm.
    try:
        if not hasattr(torch.ops._C, "silu_and_mul"):
            lib = torch.library.Library("_C", "FRAGMENT")
            lib.define("silu_and_mul(Tensor(a!) out, Tensor input) -> ()")

            def _impl(out, x):  # noqa: ANN001,ANN202
                torch.ops.pxq4.silu_and_mul_out(out, x)
                if _TRACE:
                    _tick("silu_and_mul[_C]", True)

            lib.impl("silu_and_mul", _impl, "CUDA")
            lib._register_fake("silu_and_mul", lambda out, x: None)
            # Keep the Library alive for the process lifetime; if it is garbage
            # collected the registration is torn down and the fork's cached
            # ``self.op`` becomes a dangling handle.
            globals()["_PXA_C_LIB"] = lib
            if hasattr(torch.ops._C, "silu_and_mul"):
                how = "fragment"
    except Exception:
        logger.exception("pxa: _C fragment for silu_and_mul failed; trying a class patch")

    # And the class patch, which covers the custom_ops="none" dispatch.
    try:
        from vllm.model_executor.layers.activation import SiluAndMul

        if getattr(SiluAndMul, "_pxa_patched", False):
            return how or "class"
        original = SiluAndMul.forward_native

        def forward_native(x):  # noqa: ANN001,ANN202
            if (x.is_cuda and x.dtype is torch.float16 and x.is_contiguous()
                    and x.shape[-1] % 2 == 0):
                out = torch.empty(x.shape[:-1] + (x.shape[-1] // 2,),
                                  dtype=x.dtype, device=x.device)
                torch.ops.pxq4.silu_and_mul_out(out, x)
                if _TRACE:
                    _tick("silu_and_mul[native]", True)
                return out
            if _TRACE:
                _tick("silu_and_mul", False)
            return original(x)

        SiluAndMul.forward_native = staticmethod(forward_native)
        SiluAndMul._pxa_patched = True
        return (how + "+class") if how else "class"
    except Exception:
        logger.exception("pxa: SiluAndMul class patch failed; native path kept")
    return how


# ---------------------------------------------------------------------------
def maybe_patch() -> None:
    global _ARMED, logger
    from vllm.logger import init_logger

    logger = init_logger("pxq4_vllm.pascal_ops")

    global _TRACE, _TRACE_EVERY
    _TRACE = os.getenv("PXA_OPS_TRACE", "0").strip() not in ("", "0", "off", "no")
    try:
        _TRACE_EVERY = max(1, int(os.getenv("PXA_OPS_TRACE_EVERY", "2000")))
    except ValueError:
        pass

    if _ARMED:
        return
    modes = _modes()
    if not modes:
        _ARMED = True
        return

    # LOAD THE LIBRARY FIRST. maybe_patch() runs from the plugin entry point, which fires
    # in load_general_plugins() long before any quantised layer calls ops.load_library() --
    # so on a real engine boot torch.ops.pxq4 does NOT yet carry this pack's ops at this
    # point, and an _have() check here would find nothing and disarm every seam. That is
    # exactly what happened on the first live boot: the warning below fired with all seven
    # ops "missing" from a library that demonstrably contains all seven. The unit harness
    # never saw it because it called torch.ops.load_library() itself before patching.
    # Loading here is safe and idempotent: ops.load_library() is flock-free, guarded by its
    # own lock and a _loaded flag, and does nothing if the library is already in.
    try:
        from . import ops as _pxq_ops  # noqa: PLC0415
        _pxq_ops.load_library(required=False)
    except Exception:
        logger.exception("pxa: could not load PXQ4_LIB before arming; "
                         "the seams will disarm and the decompositions are kept")

    missing = [n for n in _OPS if not _have(n)]
    if missing:
        logger.warning(
            "PXA_OPS_FUSED=%s but torch.ops.pxq4 is missing %s; the norms and "
            "activations will keep running as their torch decompositions. Load a lib "
            "built from pxa_pascal_ops.cu.",
            os.getenv("PXA_OPS_FUSED"), ", ".join(missing))
        _ARMED = True
        return

    try:
        _register_fakes()
    except Exception:
        logger.exception("pxa: fake kernel registration failed; not arming anything")
        _ARMED = True
        return

    ver = int(torch.ops.pxq4.pascal_ops_version())
    for name, fn in (("gemma", _patch_gemma), ("gdn", _patch_gdn), ("ir", _patch_ir)):
        if name not in modes:
            continue
        try:
            if fn():
                _ARMED_LIST.append(name)
                logger.info("pxa pascal-ops ARMED: %s (lib v%d)", name, ver)
        except Exception:
            logger.exception("pxa pascal-ops %s seam failed; native path kept", name)

    if "silu" in modes:
        try:
            how = _patch_silu()
            if how:
                _ARMED_LIST.append(f"silu[{how}]")
                logger.info("pxa pascal-ops ARMED: silu_and_mul via the %s seam "
                            "(lib v%d)", how, ver)
        except Exception:
            logger.exception("pxa pascal-ops silu seam failed; native path kept")

    _ARMED = True
    logger.info("pxa pascal-ops: armed %s; max rows %s",
                ", ".join(_ARMED_LIST) or "nothing",
                _max_rows() or "unlimited")
