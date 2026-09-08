# SPDX-License-Identifier: Apache-2.0
"""numpy-backed fakes for the slice of ``torch`` and ``vllm`` that
``pxq4_linear.py`` / ``pxq4_parameters.py`` / ``pxq4_ops.py`` touch.

DESTINATION IN THE REPO OF PLAN 09: ``tests/_stubs_linear.py``.

This is a TEST FIXTURE, not shipped code.  It exists so component B can be
exercised on a machine with neither torch nor vLLM installed (this workflow
runs no GPU and the DGX container must not be restarted).

Everything here is a transcription of code that was READ in /opt/1Cat-vLLM
(git 2ceb15066); every class carries the file:line it mirrors.  If the real
packages are importable the stubs are skipped, so the same test file runs
unchanged inside the container against the real vLLM -- which is the only way
these transcriptions can be confirmed rather than trusted.

The vLLM parameter classes are transcribed VERBATIM because they are the
component under test: the entire TP-correctness claim of PXQ4 is "the stock v2
loaders, given these shapes and these attributes, narrow on whole panels and
whole slabs".  A test against a paraphrase would prove nothing, so the bodies
below are copied line for line from parameter.py and linear.py.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from fractions import Fraction

import numpy as np


def have(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


USING_STUBS = not have("torch")

# --------------------------------------------------------------------------
# Fake torch
# --------------------------------------------------------------------------
ALLOC_COUNT = {"n": 0}


class _DType:
    def __init__(self, name: str, np_dtype) -> None:
        self.name = name
        self.np = np.dtype(np_dtype)

    def __repr__(self) -> str:
        return f"torch.{self.name}"


class _Device:
    def __init__(self, type_: str = "cuda", index: int | None = 0) -> None:
        self.type = type_
        self.index = index

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, _Device)
            and other.type == self.type
            and other.index == self.index
        )

    def __hash__(self) -> int:
        return hash((self.type, self.index))

    def __repr__(self) -> str:
        return f"device('{self.type}:{self.index}')"


class FakeTensor:
    """The subset of torch.Tensor the component uses. Views share memory."""

    def __init__(self, arr: np.ndarray, dtype: _DType, device: _Device) -> None:
        self._a = arr
        self.dtype = dtype
        self.device = device

    # -- introspection --------------------------------------------------
    @property
    def shape(self):
        return _Size(self._a.shape)

    @property
    def data(self):
        return self

    def dim(self) -> int:
        return self._a.ndim

    def numel(self) -> int:
        return int(self._a.size)

    def size(self, i: int | None = None):
        return self.shape if i is None else self._a.shape[i]

    def is_contiguous(self) -> bool:
        return self._a.flags["C_CONTIGUOUS"]

    def contiguous(self) -> "FakeTensor":
        if self.is_contiguous():
            return self
        return FakeTensor(np.ascontiguousarray(self._a), self.dtype, self.device)

    def numpy(self) -> np.ndarray:
        return self._a

    # -- shaping (all views, like torch) --------------------------------
    def narrow(self, dim: int, start: int, length: int) -> "FakeTensor":
        if start < 0 or start + length > self._a.shape[dim]:
            raise IndexError(
                f"narrow(dim={dim}, start={start}, length={length}) out of range "
                f"for shape {self._a.shape}"
            )
        sl = [slice(None)] * self._a.ndim
        sl[dim] = slice(start, start + length)
        return FakeTensor(self._a[tuple(sl)], self.dtype, self.device)

    def reshape(self, *shape) -> "FakeTensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return FakeTensor(self._a.reshape(shape), self.dtype, self.device)

    def view(self, *shape) -> "FakeTensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return FakeTensor(self._a.reshape(shape), self.dtype, self.device)

    def t(self) -> "FakeTensor":
        return FakeTensor(self._a.T, self.dtype, self.device)

    def __getitem__(self, idx) -> "FakeTensor":
        return FakeTensor(self._a[idx], self.dtype, self.device)

    # -- mutation --------------------------------------------------------
    def copy_(self, other: "FakeTensor") -> "FakeTensor":
        src = other._a if isinstance(other, FakeTensor) else np.asarray(other)
        if self._a.shape != src.shape:
            raise AssertionError(f"copy_ shape {src.shape} into {self._a.shape}")
        self._a[...] = src.astype(self._a.dtype, copy=False)
        return self

    def fill_(self, value) -> "FakeTensor":
        self._a[...] = value
        return self

    def add_(self, other) -> "FakeTensor":
        src = other._a if isinstance(other, FakeTensor) else np.asarray(other)
        self._a += src.astype(self._a.dtype, copy=False)
        return self

    # -- elementwise -----------------------------------------------------
    def eq(self, other):
        rhs = other._a if isinstance(other, FakeTensor) else other
        return FakeTensor(self._a == rhs, BOOL, self.device)

    def ne(self, other):
        rhs = other._a if isinstance(other, FakeTensor) else other
        return FakeTensor(self._a != rhs, BOOL, self.device)

    # -- reductions ------------------------------------------------------
    def all(self, dim: int | None = None):
        if dim is None:
            return _Scalar(bool(self._a.all()))
        return FakeTensor(self._a.all(axis=dim), BOOL, self.device)

    def any(self, dim: int | None = None):
        if dim is None:
            return _Scalar(bool(self._a.any()))
        return FakeTensor(self._a.any(axis=dim), BOOL, self.device)

    def sum(self, dim: int | None = None):
        if dim is None:
            return _Scalar(self._a.sum())
        return FakeTensor(self._a.sum(axis=dim), self.dtype, self.device)

    def __bool__(self) -> bool:
        return bool(self._a.all()) if self._a.size else False

    def __int__(self) -> int:
        return int(self._a)

    def __repr__(self) -> str:
        return f"FakeTensor(shape={self._a.shape}, dtype={self.dtype})"


class _Scalar:
    def __init__(self, v) -> None:
        self.v = v

    def __bool__(self) -> bool:
        return bool(self.v)

    def __int__(self) -> int:
        return int(self.v)

    def item(self):
        return self.v


class _Size(tuple):
    pass


FLOAT16 = _DType("float16", np.float16)
FLOAT32 = _DType("float32", np.float32)
UINT8 = _DType("uint8", np.uint8)
BOOL = _DType("bool", np.bool_)


def arr(x) -> np.ndarray:
    if isinstance(x, FakeTensor):
        return x._a
    if hasattr(x, "data") and isinstance(getattr(x, "data"), FakeTensor):
        return x.data._a
    return np.asarray(x)


def install_torch_stub() -> types.ModuleType:
    if have("torch"):  # pragma: no cover - container path
        import torch

        return torch

    torch = types.ModuleType("torch")
    torch.float16 = FLOAT16
    torch.half = FLOAT16
    torch.float32 = FLOAT32
    torch.uint8 = UINT8
    torch.bool = BOOL
    torch.dtype = _DType
    torch.Tensor = FakeTensor
    torch.device = _Device
    torch.Size = _Size

    def empty(*shape, dtype=FLOAT32, device=None):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        ALLOC_COUNT["n"] += 1
        return FakeTensor(np.zeros(shape, dtype=dtype.np), dtype, device or _Device())

    def zeros(*shape, dtype=FLOAT32, device=None):
        return empty(*shape, dtype=dtype, device=device)

    def as_tensor(data, dtype=FLOAT32):
        return FakeTensor(np.asarray(data, dtype=dtype.np), dtype, _Device("cpu", None))

    def isnan(x):
        a = arr(x)
        return FakeTensor(np.isnan(a.astype(np.float32)), BOOL, _Device())

    def mm(a, b, out=None):
        res = arr(a).astype(np.float32) @ arr(b).astype(np.float32)
        if out is None:
            return FakeTensor(res.astype(np.float16), FLOAT16, a.device)
        out._a[...] = res.astype(out._a.dtype)
        return out

    def _check(cond, msg=None):
        if not cond:
            raise RuntimeError(msg or "torch._check failed")

    torch.empty = empty
    torch.zeros = zeros
    torch.as_tensor = as_tensor
    torch.isnan = isnan
    torch.mm = mm
    torch._check = _check

    # torch.nn
    nn = types.ModuleType("torch.nn")

    class Parameter(FakeTensor):
        def __new__(cls, data, requires_grad=False):
            obj = object.__new__(cls)
            FakeTensor.__init__(obj, arr(data), data.dtype, data.device)
            obj.requires_grad = requires_grad
            return obj

        def __init__(self, data, requires_grad=False):  # noqa: D107
            pass

    class Module:
        def __init__(self) -> None:
            self._parameters: dict[str, object] = {}

        def register_parameter(self, name, param) -> None:
            self._parameters[name] = param
            object.__setattr__(self, name, param)

    nn.Parameter = Parameter
    nn.Module = Module
    torch.nn = nn

    # torch.ops
    class _OpNamespace:
        def __init__(self) -> None:
            self._libs: dict[str, object] = {}

        def load_library(self, path):
            raise OSError(f"fake torch cannot load {path}")

        def __getattr__(self, name):
            raise AttributeError(name)

    torch.ops = _OpNamespace()

    # torch.library
    library = types.ModuleType("torch.library")

    def register_fake(name):
        def _wrap(fn):
            REGISTERED_FAKES.append(name)
            return fn

        return _wrap

    library.register_fake = register_fake
    torch.library = library

    cuda = types.ModuleType("torch.cuda")
    cuda.current_device = lambda: 0
    torch.cuda = cuda

    compiler = types.ModuleType("torch.compiler")
    compiler.is_compiling = lambda: False
    torch.compiler = compiler

    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.library"] = library
    sys.modules["torch.cuda"] = cuda
    sys.modules["torch.compiler"] = compiler
    return torch


REGISTERED_FAKES: list[str] = []


# --------------------------------------------------------------------------
# Fake vLLM: parameter.py transcribed verbatim
# --------------------------------------------------------------------------
def install_vllm_stub(torch) -> None:
    if have("vllm"):  # pragma: no cover - container path
        return

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    sys.modules["vllm"] = vllm

    logger_mod = types.ModuleType("vllm.logger")

    def init_logger(name):
        import logging

        return logging.getLogger(name)

    logger_mod.init_logger = init_logger
    sys.modules["vllm.logger"] = logger_mod

    for pkg in ("vllm.model_executor", "vllm.model_executor.layers"):
        m = types.ModuleType(pkg)
        m.__path__ = []
        sys.modules[pkg] = m

    # ---- vllm.model_executor.parameter --------------------------------
    param_mod = types.ModuleType("vllm.model_executor.parameter")

    TP = {"rank": 0, "size": 1}  # set by the tests

    class BasevLLMParameter:
        """parameter.py:31-127. Held by composition rather than by Tensor
        subclassing; every attribute the component reads is delegated."""

        def __init__(self, data, weight_loader) -> None:
            self._data = data
            self._weight_loader = weight_loader
            self.tp_rank = TP["rank"]
            self.tp_size = TP["size"]

        @property
        def data(self):
            return self._data

        @property
        def weight_loader(self):
            return self._weight_loader

        def __getattr__(self, name):
            # delegate dim()/numel()/is_contiguous()/dtype/shape/device/...
            return getattr(object.__getattribute__(self, "_data"), name)

        # parameter.py:92-96
        def _assert_and_load(self, loaded_weight):
            assert tuple(self._data.shape) == tuple(loaded_weight.shape), (
                f"_assert_and_load {tuple(loaded_weight.shape)} into "
                f"{tuple(self._data.shape)}"
            )
            self._data.copy_(loaded_weight)

        def load_column_parallel_weight(self, loaded_weight):
            self._assert_and_load(loaded_weight)

        def load_row_parallel_weight(self, loaded_weight):
            self._assert_and_load(loaded_weight)

        def load_merged_column_weight(self, loaded_weight, **kwargs):
            self._assert_and_load(loaded_weight)

        def load_qkv_weight(self, loaded_weight, **kwargs):
            self._assert_and_load(loaded_weight)

    class _ColumnvLLMParameter(BasevLLMParameter):
        """parameter.py:129-201."""

        def __init__(self, output_dim: int, **kwargs) -> None:
            self._output_dim = output_dim
            super().__init__(**kwargs)

        @property
        def output_dim(self):
            return self._output_dim

        # parameter.py:145-151
        def load_column_parallel_weight(self, loaded_weight):
            shard_size = self.data.shape[self.output_dim]
            loaded_weight = loaded_weight.narrow(
                self.output_dim, self.tp_rank * shard_size, shard_size
            )
            assert tuple(self.data.shape) == tuple(loaded_weight.shape)
            self.data.copy_(loaded_weight)

        # parameter.py:153-173
        def load_merged_column_weight(self, loaded_weight, **kwargs):
            shard_offset = kwargs["shard_offset"]
            shard_size = kwargs["shard_size"]
            if (
                isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
                and self.packed_dim == self.output_dim
            ):
                shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                    shard_offset=shard_offset, shard_size=shard_size
                )
            param_data = self.data
            param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
            loaded_weight = loaded_weight.narrow(
                self.output_dim, self.tp_rank * shard_size, shard_size
            )
            assert tuple(param_data.shape) == tuple(loaded_weight.shape)
            param_data.copy_(loaded_weight)

        # parameter.py:175-201
        def load_qkv_weight(self, loaded_weight, **kwargs):
            shard_offset = kwargs["shard_offset"]
            shard_size = kwargs["shard_size"]
            shard_id = kwargs["shard_id"]
            num_heads = kwargs["num_heads"]
            if (
                isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
                and self.output_dim == self.packed_dim
            ):
                shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                    shard_offset=shard_offset, shard_size=shard_size
                )
            param_data = self.data
            shard_id_int = (
                self.tp_rank if shard_id == "q" else self.tp_rank // num_heads
            )
            param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
            loaded_weight = loaded_weight.narrow(
                self.output_dim, shard_id_int * shard_size, shard_size
            )
            assert tuple(param_data.shape) == tuple(loaded_weight.shape)
            param_data.copy_(loaded_weight)

    class RowvLLMParameter(BasevLLMParameter):
        """parameter.py:203-230."""

        def __init__(self, input_dim: int, **kwargs) -> None:
            self._input_dim = input_dim
            super().__init__(**kwargs)

        @property
        def input_dim(self):
            return self._input_dim

        # parameter.py:220-230 -- NOTE: no packed_factor anywhere.
        def load_row_parallel_weight(self, loaded_weight):
            shard_size = self.data.shape[self.input_dim]
            loaded_weight = loaded_weight.narrow(
                self.input_dim, self.tp_rank * shard_size, shard_size
            )
            assert tuple(self.data.shape) == tuple(loaded_weight.shape)
            self.data.copy_(loaded_weight)

    class ModelWeightParameter(_ColumnvLLMParameter, RowvLLMParameter):
        pass

    # parameter.py:605-616
    def _adjust_shard_indexes_for_packing(
        shard_size, shard_offset, packed_factor, marlin_tile_size
    ):
        shard_size = round(shard_size // packed_factor)
        shard_offset = round(shard_offset // packed_factor)
        if marlin_tile_size is not None:
            return shard_size * marlin_tile_size, shard_offset * marlin_tile_size
        return shard_size, shard_offset

    class _PackedMixin:
        def __init__(
            self, packed_factor, packed_dim, marlin_tile_size=None, **kwargs
        ) -> None:
            self._packed_factor = packed_factor
            self._packed_dim = packed_dim
            self._marlin_tile_size = marlin_tile_size
            super().__init__(**kwargs)

        @property
        def packed_dim(self):
            return self._packed_dim

        @property
        def packed_factor(self):
            return self._packed_factor

        @property
        def marlin_tile_size(self):
            return self._marlin_tile_size

        def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
            return _adjust_shard_indexes_for_packing(
                shard_size=shard_size,
                shard_offset=shard_offset,
                packed_factor=self.packed_factor,
                marlin_tile_size=self.marlin_tile_size,
            )

    class PackedColumnParameter(_PackedMixin, _ColumnvLLMParameter):
        """parameter.py:311-350."""

    class PackedvLLMParameter(_PackedMixin, ModelWeightParameter):
        """parameter.py:352-395."""

    param_mod.BasevLLMParameter = BasevLLMParameter
    param_mod._ColumnvLLMParameter = _ColumnvLLMParameter
    param_mod.RowvLLMParameter = RowvLLMParameter
    param_mod.ModelWeightParameter = ModelWeightParameter
    param_mod.PackedColumnParameter = PackedColumnParameter
    param_mod.PackedvLLMParameter = PackedvLLMParameter
    param_mod.TP = TP
    param_mod.Fraction = Fraction
    sys.modules["vllm.model_executor.parameter"] = param_mod

    # ---- vllm.model_executor.utils ------------------------------------
    utils_mod = types.ModuleType("vllm.model_executor.utils")

    def set_weight_attrs(weight, weight_attrs):
        # utils.py:13-40
        if weight_attrs is None:
            return
        for key, value in weight_attrs.items():
            assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"
            setattr(weight, key, value)

    utils_mod.set_weight_attrs = set_weight_attrs
    sys.modules["vllm.model_executor.utils"] = utils_mod

    # ---- vllm.model_executor.layers.linear -----------------------------
    linear_mod = types.ModuleType("vllm.model_executor.layers.linear")

    class LinearMethodBase:
        pass

    WEIGHT_LOADER_V2_SUPPORTED = ["UnquantizedLinearMethod"]

    def register_weight_loader_v2_supported_method(cls):
        WEIGHT_LOADER_V2_SUPPORTED.append(cls.__name__)
        return cls

    linear_mod.LinearMethodBase = LinearMethodBase
    linear_mod.WEIGHT_LOADER_V2_SUPPORTED = WEIGHT_LOADER_V2_SUPPORTED
    linear_mod.register_weight_loader_v2_supported_method = (
        register_weight_loader_v2_supported_method
    )
    sys.modules["vllm.model_executor.layers.linear"] = linear_mod


# --------------------------------------------------------------------------
# The two linear-layer loader drivers we need, from linear.py
# --------------------------------------------------------------------------
def merged_column_weight_loader_v2(param, loaded_weight, loaded_shard_id, output_sizes,
                                   tp_size):
    """linear.py:1140-1205, int-shard-id branch only (the tuple branch is
    handled by ``load_fused_module_from_checkpoint`` below)."""
    shard_offset = sum(output_sizes[:loaded_shard_id])
    shard_size = output_sizes[loaded_shard_id]
    shard_offset //= tp_size
    shard_size //= tp_size
    param.load_merged_column_weight(
        loaded_weight=loaded_weight,
        shard_id=loaded_shard_id,
        shard_offset=shard_offset,
        shard_size=shard_size,
        tp_rank=param.tp_rank,
    )


def load_fused_module_from_checkpoint(param, loaded_weight, output_sizes, all_sizes,
                                      tp_size):
    """linear.py:1100-1138. ``output_sizes`` is the subset selected by the
    tuple shard id; ``all_sizes`` is the layer's full ``self.output_sizes``
    (needed because the recursive call re-derives offsets from it)."""
    import sys as _sys

    pmod = _sys.modules["vllm.model_executor.parameter"]
    current = 0
    shard_offsets = []
    for i, output_size in enumerate(output_sizes):
        shard_offsets.append((i, current, output_size))
        current += output_size

    for shard_id, shard_offset, shard_size in shard_offsets:
        if (
            isinstance(
                param, (pmod.PackedColumnParameter, pmod.PackedvLLMParameter)
            )
            and param.packed_dim == param.output_dim
        ):
            shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                shard_size=shard_size, shard_offset=shard_offset
            )
        loaded_weight_shard = loaded_weight.narrow(
            param.output_dim, shard_offset, shard_size
        )
        merged_column_weight_loader_v2(
            param, loaded_weight_shard, shard_id, all_sizes, tp_size
        )
