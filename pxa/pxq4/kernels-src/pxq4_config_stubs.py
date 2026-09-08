# SPDX-License-Identifier: Apache-2.0
"""Minimal fakes for ``torch`` and the handful of ``vllm`` symbols that
``pxq4_config.py`` imports, so the config component can be unit-tested on a
machine with neither installed.

DESTINATION IN THE REPO OF PLAN 09: ``tests/_stubs.py``.

This is a test fixture, not shipped code.  It reproduces only the *contracts*
that were read in /opt/1Cat-vLLM (git 2ceb15066):

  * ``QuantizationConfig.__init__`` seeds ``packed_modules_mapping``
    (base_config.py:72-76).
  * ``register_quantization_config`` requires a ``QuantizationConfig`` subclass
    and records it (quantization/__init__.py:97-101).
  * ``LinearBase`` is the isinstance target used by our dispatch
    (linear.py:492).

If the real packages are importable the stubs are skipped, so the same test
file also runs unchanged inside the vLLM container.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import types
from abc import ABC, abstractmethod
from pathlib import Path

REGISTERED: dict[str, type] = {}
QUANTIZATION_METHODS: list[str] = ["awq", "compressed-tensors", "gguf"]


def _have(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def install_stubs() -> None:
    """Install fake ``torch`` and ``vllm`` modules into ``sys.modules``."""
    if not _have("torch"):
        torch = types.ModuleType("torch")

        class _DType:
            def __init__(self, name: str) -> None:
                self.name = name

            def __repr__(self) -> str:
                return f"torch.{self.name}"

        torch.float16 = _DType("float16")
        torch.half = torch.float16
        torch.bfloat16 = _DType("bfloat16")
        torch.float32 = _DType("float32")
        torch.dtype = _DType

        nn = types.ModuleType("torch.nn")

        class Module:  # noqa: D401 - stand-in for torch.nn.Module
            pass

        nn.Module = Module
        torch.nn = nn
        sys.modules["torch"] = torch
        sys.modules["torch.nn"] = nn

    if _have("vllm"):
        return

    import torch as _torch  # the real one, or the stub just installed

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []  # mark as a package so submodules can be registered
    sys.modules["vllm"] = vllm

    # ---- vllm.logger -----------------------------------------------------
    logger_mod = types.ModuleType("vllm.logger")

    def init_logger(name: str):
        import logging

        return logging.getLogger(name)

    logger_mod.init_logger = init_logger
    sys.modules["vllm.logger"] = logger_mod

    # ---- vllm.model_executor.layers.quantization.base_config -------------
    for pkg in (
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.quantization",
    ):
        m = types.ModuleType(pkg)
        m.__path__ = []
        sys.modules[pkg] = m

    base_config = types.ModuleType(
        "vllm.model_executor.layers.quantization.base_config"
    )

    class QuantizeMethodBase(ABC):
        @abstractmethod
        def create_weights(self, layer, *args, **kwargs):
            raise NotImplementedError

        @abstractmethod
        def apply(self, layer, *args, **kwargs):
            raise NotImplementedError

        def process_weights_after_loading(self, layer) -> None:
            return

    class QuantizationConfig(ABC):
        def __init__(self) -> None:
            super().__init__()
            self.packed_modules_mapping: dict[str, list[str]] = dict()

        @abstractmethod
        def get_name(self) -> str: ...

        @abstractmethod
        def get_supported_act_dtypes(self) -> list: ...

        @classmethod
        @abstractmethod
        def get_min_capability(cls) -> int: ...

        @staticmethod
        @abstractmethod
        def get_config_filenames() -> list[str]: ...

        @classmethod
        @abstractmethod
        def from_config(cls, config: dict): ...

        @classmethod
        def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
            return None

        @abstractmethod
        def get_quant_method(self, layer, prefix: str): ...

        def get_cache_scale(self, name: str):
            return None

        def apply_vllm_mapper(self, hf_to_vllm_mapper):
            pass

        def maybe_update_config(self, model_name, hf_config=None, revision=None):
            pass

        def is_mxfp4_quant(self, prefix, layer) -> bool:
            return False

    base_config.QuantizationConfig = QuantizationConfig
    base_config.QuantizeMethodBase = QuantizeMethodBase
    sys.modules["vllm.model_executor.layers.quantization.base_config"] = base_config

    quant_pkg = sys.modules["vllm.model_executor.layers.quantization"]

    def register_quantization_config(quantization: str):
        def _wrapper(cls):
            if quantization not in QUANTIZATION_METHODS:
                QUANTIZATION_METHODS.append(quantization)
            if not issubclass(cls, QuantizationConfig):
                raise ValueError(
                    "The quantization config must be a subclass of "
                    "`QuantizationConfig`."
                )
            REGISTERED[quantization] = cls
            return cls

        return _wrapper

    quant_pkg.register_quantization_config = register_quantization_config
    quant_pkg.QUANTIZATION_METHODS = QUANTIZATION_METHODS
    quant_pkg.QuantizationMethods = str

    # ---- vllm.model_executor.layers.linear -------------------------------
    linear = types.ModuleType("vllm.model_executor.layers.linear")

    class LinearBase(_torch.nn.Module):
        def __init__(self, prefix: str = "") -> None:
            self.prefix = prefix

    class ColumnParallelLinear(LinearBase):
        pass

    class RowParallelLinear(LinearBase):
        pass

    class MergedColumnParallelLinear(ColumnParallelLinear):
        pass

    class QKVParallelLinear(ColumnParallelLinear):
        pass

    class UnquantizedLinearMethod(QuantizeMethodBase):
        def create_weights(self, layer, *args, **kwargs):
            return None

        def apply(self, layer, *args, **kwargs):
            return None

    linear.LinearBase = LinearBase
    linear.ColumnParallelLinear = ColumnParallelLinear
    linear.RowParallelLinear = RowParallelLinear
    linear.MergedColumnParallelLinear = MergedColumnParallelLinear
    linear.QKVParallelLinear = QKVParallelLinear
    linear.UnquantizedLinearMethod = UnquantizedLinearMethod
    sys.modules["vllm.model_executor.layers.linear"] = linear


# --------------------------------------------------------------------------
# Assemble a throwaway ``pxq4_vllm`` package around the config module so its
# relative imports (``from . import layout``, ``from .linear import ...``)
# resolve.  The real package is built by components A and B; here we supply
# just enough of each to exercise dispatch.
# --------------------------------------------------------------------------

_LINEAR_STUB = '''
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase


class PXQ4LinearMethod(QuantizeMethodBase):
    """Stand-in for component B's real method; identity is all we test."""

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(self, layer, *args, **kwargs):
        raise NotImplementedError

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError
'''

_LAYOUT_STUB = '''
TYPE_ID = 252
PANEL_ROWS = 64
SLAB_COLS = 32
SLAB_BYTES = 1088
HEADER_BYTES = 128
CODE_OFF = 64
CODE_BYTES = 16
'''


def build_package(
    tmpdir: Path,
    config_src: Path,
    *,
    pkg_name: str = "pxq4_vllm",
    with_layout: bool = True,
    reference_tables: tuple[list[float], list[float]] | None = None,
) -> str:
    """Materialise a *pkg_name* package under *tmpdir* and return the path to
    add to ``sys.path``.  Returns the directory, not the package.

    ``pkg_name`` exists so that a test needing a differently-configured copy
    (e.g. one carrying mismatched reference tables) can build it under its own
    name instead of unloading the shared one -- unloading would rebind
    ``PXQ4LinearMethod`` to a fresh class object and break every ``isinstance``
    assertion elsewhere in the file."""
    pkg = tmpdir / pkg_name
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    shutil.copyfile(config_src, pkg / "config.py")
    (pkg / "linear.py").write_text(_LINEAR_STUB)
    if with_layout:
        (pkg / "layout.py").write_text(_LAYOUT_STUB)
    if reference_tables is not None:
        book, sub = reference_tables
        (pkg / "reference.py").write_text(f"BOOK = {book!r}\nSUB = {sub!r}\n")
    return str(tmpdir)
