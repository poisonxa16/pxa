"""
hostsim_bridge.py -- drive agent C's kernel on the CPU, via ctypes, with no GPU.

`pxq4_kernel_hostsim.cpp` compiles the REAL `k_pxq4_dequant_matrix` / `k_pxq4_mmv` device
code against a tiny host shim that fakes blockIdx/threadIdx/__syncthreads.  That means the
kernel gates -- which the plan schedules as G6/G8, behind a GPU and a lease -- can be run
here, today, against the actual kernel source rather than a description of it.

What this DOES prove: the layout arithmetic, the nibble and scale-SoA addressing, the
table values, the accumulation order, the canonical nfix fold, the fp16 store, and the
whole shard-then-dequant invariant, all in agent C's own code.

What it does NOT prove, and G6/G8 on real hardware still must: the launch configuration,
the dynamic-shared-memory opt-in, CUDA-graph capture, warp-level primitives (__shfl_sync,
prmt) if the sm_70 build selects a MODE that uses them, and nvcc's fp32 contraction
choices.  A hostsim PASS is necessary, not sufficient -- and the summary says so.

ASSUMPTION: the hostsim TU is compiled from the same headers as the CUDA TU, so a
divergence between them is a build problem rather than a silent semantic difference.
`test_f_hostsim.py` checks the built-in tables through this bridge, which catches the
most likely form of that (a stale object file).
"""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMPL = os.path.dirname(_HERE)

CANDIDATES = [
    os.environ.get("PXQ4_HOSTSIM"),
    os.path.join(_IMPL, "libpxq4_hostsim.so"),
    os.path.join(_HERE, "libpxq4_hostsim.so"),
]
BUILD_SCRIPT = os.path.join(_IMPL, "build_hostsim.sh")


class HostsimUnavailable(RuntimeError):
    pass


_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    path = next((p for p in CANDIDATES if p and os.path.exists(p)), None)
    if path is None and os.path.exists(BUILD_SCRIPT):
        try:
            subprocess.run(["sh", BUILD_SCRIPT], cwd=_IMPL, capture_output=True,
                           text=True, check=True)
        except Exception as e:
            raise HostsimUnavailable(f"build_hostsim.sh failed: {e}")
        path = next((p for p in CANDIDATES if p and os.path.exists(p)), None)
    if path is None:
        raise HostsimUnavailable(
            "libpxq4_hostsim.so not found; set $PXQ4_HOSTSIM or run build_hostsim.sh")
    lib = ctypes.CDLL(path)
    u8 = ctypes.POINTER(ctypes.c_uint8)
    u16 = ctypes.POINTER(ctypes.c_uint16)
    f32 = ctypes.POINTER(ctypes.c_float)
    ci = ctypes.c_int
    lib.pxq4_hostsim_dequant_f32.argtypes = [u8, u16, f32, ci, ci]
    lib.pxq4_hostsim_dequant_f32.restype = None
    lib.pxq4_hostsim_dequant_f16.argtypes = [u8, u16, u16, ci, ci]
    lib.pxq4_hostsim_dequant_f16.restype = None
    lib.pxq4_hostsim_mmv_f16.argtypes = [u8, u16, u16, u16, ci, ci, ci, ci]
    lib.pxq4_hostsim_mmv_f16.restype = None
    lib.pxq4_hostsim_canon_nfix.argtypes = [ci]
    lib.pxq4_hostsim_canon_nfix.restype = ci
    lib.pxq4_hostsim_canon_max_chunk.argtypes = [ci]
    lib.pxq4_hostsim_canon_max_chunk.restype = ci
    lib.pxq4_hostsim_builtin_tables.argtypes = [f32, f32]
    lib.pxq4_hostsim_builtin_tables.restype = None
    _lib = lib
    return lib


def available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False


def _p(a, ct):
    return np.ascontiguousarray(a).ctypes.data_as(ctypes.POINTER(ct))


def dequant_f32(slabs: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    lib = _load()
    P, S, _ = slabs.shape
    out = np.empty((P * 64, S * 32), dtype=np.float32)
    lib.pxq4_hostsim_dequant_f32(_p(slabs, ctypes.c_uint8),
                                 _p(anchor.view(np.uint16), ctypes.c_uint16),
                                 _p(out, ctypes.c_float), P, S)
    return out


def dequant_f16(slabs: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    lib = _load()
    P, S, _ = slabs.shape
    out = np.empty((P * 64, S * 32), dtype=np.uint16)
    lib.pxq4_hostsim_dequant_f16(_p(slabs, ctypes.c_uint8),
                                 _p(anchor.view(np.uint16), ctypes.c_uint16),
                                 _p(out, ctypes.c_uint16), P, S)
    return out.view(np.float16)


def mmv_f16(x: np.ndarray, slabs: np.ndarray, anchor: np.ndarray, vecx: int = 1) -> np.ndarray:
    """x float16[M,K] -> float16[M,N].  `vecx` selects the VECX template arm; both are
    supposed to be bit-identical (the b-loop accumulates into the same accumulators in
    the same ascending order either way), and test_f_hostsim asserts exactly that."""
    lib = _load()
    x = np.ascontiguousarray(x, dtype=np.float16)
    M, K = x.shape
    P, S, _ = slabs.shape
    assert S * 32 == K, f"x K={K} vs weights K={S*32}"
    out = np.empty((M, P * 64), dtype=np.uint16)
    lib.pxq4_hostsim_mmv_f16(_p(slabs, ctypes.c_uint8),
                             _p(anchor.view(np.uint16), ctypes.c_uint16),
                             _p(x.view(np.uint16), ctypes.c_uint16),
                             _p(out, ctypes.c_uint16), M, P, S, int(vecx))
    return out.view(np.float16)


def canon_nfix(kslabs: int) -> int:
    return _load().pxq4_hostsim_canon_nfix(int(kslabs))


def canon_max_chunk(kslabs: int) -> int:
    return _load().pxq4_hostsim_canon_max_chunk(int(kslabs))


def builtin_tables():
    lib = _load()
    book = np.empty(16, dtype=np.float32)
    sub = np.empty(16, dtype=np.float32)
    lib.pxq4_hostsim_builtin_tables(_p(book, ctypes.c_float), _p(sub, ctypes.c_float))
    return book, sub
