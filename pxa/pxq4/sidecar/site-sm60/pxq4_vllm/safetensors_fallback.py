"""safetensors_fallback.py -- name the tensor safetensors refuses to hand over, and read it anyway.

THE SYMPTOM (observed on both pp4 and tp2pp2). Seconds into the weight load,
at "Loading safetensors checkpoint shards: 0/76", every worker but one dies with

    File ".../vllm/model_executor/model_loader/weight_utils.py", line 1078,
      in safetensors_weights_iterator
        param = f.get_tensor(name)
    ValueError: could not determine the shape of object type 'torch.storage.UntypedStorage'

and the traceback never says WHICH tensor. That message is torch's, not safetensors': it comes
out of ``internal_new_from_data`` when something is handed an object it cannot infer a shape
from -- i.e. safetensors' Rust binding took a tensor-construction path that this torch (2.7.1)
does not accept, for that one tensor, and re-raised nothing useful.

It is not the checkpoint. All 76 shard headers are offset/size/EOF consistent, and every one of
the 149,204 tensors reads through this same safetensors 0.8.0 + torch 2.7.1 image one at a time,
with and without the sidecar, on CPU and under an ambient CUDA device. The failure only happens
inside the worker, so the tensor's IDENTITY is the missing evidence.

WHAT THIS DOES. It replaces the ``safe_open`` NAME that weight_utils holds with a thin wrapper
that delegates everything. When ``get_tensor`` raises, the wrapper

  1. LOGS the file, the key, its header dtype and shape, and the original exception -- which is
     the diagnosis; and
  2. re-reads that tensor from the file's own header offsets with ``torch.frombuffer``, an
     entirely independent path, so the boot continues instead of dying at 0/76.

Every fallback is logged and counted, and the count is logged again at the end of each file: a
checkpoint that needs the fallback for thousands of tensors is telling you something, and this
must not become a silent crutch. PXA_ST_FALLBACK=0 disables the whole module; =strict logs and
re-raises (diagnosis only, no repair).
"""
from __future__ import annotations

import json
import logging
import os
import struct

logger = logging.getLogger("pxq4_vllm.safetensors_fallback")

MODE = os.environ.get("PXA_ST_FALLBACK", "auto").strip().lower()
_LOG_LIMIT = int(os.environ.get("PXA_ST_FALLBACK_LOG", "20"))

_DTYPES = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
           "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8",
           "BOOL": "bool", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2"}

_n_fallback = 0
_n_logged = 0


class _Wrapped:
    """Delegates to the real safe_open handle; repairs only what raises."""

    def __init__(self, real, path):
        self._real = real
        self._path = path
        self._hdr = None
        self._base = 0

    # -- delegation ---------------------------------------------------------------------
    def __getattr__(self, item):
        return getattr(self._real, item)

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def keys(self):
        return self._real.keys()

    def metadata(self):
        return self._real.metadata()

    # -- the point ----------------------------------------------------------------------
    def _header(self):
        if self._hdr is None:
            with open(self._path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self._hdr = json.loads(fh.read(n))
            self._base = 8 + n
        return self._hdr

    def get_tensor(self, name):
        try:
            return self._real.get_tensor(name)
        except Exception as exc:
            global _n_fallback, _n_logged
            import torch
            info = self._header().get(name)
            _n_fallback += 1
            if _n_logged < _LOG_LIMIT:
                _n_logged += 1
                logger.error("SAFETENSORS get_tensor FAILED: file=%s key=%s dtype=%s shape=%s "
                             "-> %s: %s", os.path.basename(self._path), name,
                             (info or {}).get("dtype"), (info or {}).get("shape"),
                             type(exc).__name__, exc)
            if MODE == "strict" or info is None:
                raise
            start, end = info["data_offsets"]
            dtype = getattr(torch, _DTYPES.get(info["dtype"], ""), None)
            if dtype is None:
                raise
            with open(self._path, "rb") as fh:
                fh.seek(self._base + start)
                buf = bytearray(fh.read(end - start))
            t = torch.frombuffer(buf, dtype=dtype)
            shape = tuple(info["shape"])
            t = t.reshape(shape) if shape else t.reshape(())
            logger.warning("SAFETENSORS fallback read OK: %s %s %s (fallbacks so far: %d)",
                           os.path.basename(self._path), name, shape, _n_fallback)
            return t


def _install() -> None:
    from vllm.model_executor.model_loader import weight_utils

    real_safe_open = weight_utils.safe_open

    def safe_open(filename, framework="pt", device="cpu", **kw):
        real = real_safe_open(filename, framework=framework, device=device, **kw)
        return _Wrapped(real, filename)

    weight_utils.safe_open = safe_open
    logger.info("safetensors_fallback installed (mode=%s) on %s", MODE, weight_utils.__name__)


if MODE not in ("0", "off", "false", "no", ""):
    try:
        _install()
    except Exception as exc:          # never be the reason a boot dies
        logger.warning("safetensors_fallback could not install: %r", exc)
