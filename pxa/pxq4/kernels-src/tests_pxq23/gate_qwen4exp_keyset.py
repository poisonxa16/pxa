#!/usr/bin/env python3
"""gate_qwen4exp_keyset.py -- KEY-SET GATE: the fork's own model skeleton vs the keys we emitted.

The three earlier static gates (map-vs-producer, transforms-vs-emitted, emitted-bytes-vs-gguf)
prove the checkpoint says what the GGUF said. They cannot prove the ENGINE finds it: a key that
is spelled correctly for the producer and wrongly for vLLM is silently dropped
(``AutoWeightsLoader`` is built with ``ignore_unexpected_suffixes``) and the parameter nobody
filled keeps whatever ``torch.empty`` left in it. That is the silent-garbage class.

So this builds the REAL fork model -- same config.json, same ``--quantization pxq``, same
sidecar -- with every allocation redirected to the ``meta`` device, and replays the loader over
the emitted safetensors HEADERS (names/shapes/dtypes only; not one byte of weight data is read).
It then reports both halves:

  MISSING     model parameters that no checkpoint key filled.
  UNEXPECTED  checkpoint keys the loader had nowhere to put and dropped silently.
  FATAL       a checkpoint key that made the loader raise (it names the key).

PASS is 0/0/0. Run at pipeline-parallel-size 1 so there are no PP holes: every parameter of
every layer is present and must be filled by this one checkpoint.

WHY META WORKS HERE. pxq4_vllm's ``create_weights`` allocates with an explicit
``device=torch.cuda.current_device()``, so a bare ``torch.device("meta")`` context is not
enough; ``_force_meta`` redirects the explicit CUDA allocations too, and each parameter's
``weight_loader`` is replaced by a recorder. The recorder keeps ALL of the naming logic --
the expert mapping, the stacked-param routing, the shard ids -- and drops only the numerics,
which is precisely what a key-set gate is allowed to assume.

Usage:  python3 gate_qwen4exp_keyset.py --ckpt /path/to/vllm-fp8-checkpoint [--json out.json]
Exit:   0 = PASS, 1 = FAIL, 2 = could not run (reported, never silently green).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
import traceback

import torch

# ---------------------------------------------------------------------------------------------
# 1. every allocation goes to meta, including the ones that name a CUDA device explicitly
# ---------------------------------------------------------------------------------------------
_REAL: dict[str, object] = {}


def _is_cuda(dev) -> bool:
    if dev is None:
        return False
    if isinstance(dev, int):
        return True                      # device=<int> means cuda:<int> in torch
    try:
        return torch.device(dev).type == "cuda"
    except Exception:
        return False


def _force_meta() -> None:
    for name in ("empty", "zeros", "ones", "full", "empty_strided", "rand", "randn",
                 "arange", "eye", "linspace", "tensor"):
        fn = getattr(torch, name, None)
        if fn is None:
            continue
        _REAL[name] = fn

        def wrapper(*a, __fn=fn, **k):
            if _is_cuda(k.get("device")):
                k["device"] = "meta"
            return __fn(*a, **k)

        setattr(torch, name, wrapper)

    _real_to = torch.Tensor.to

    def to(self, *a, **k):
        if _is_cuda(k.get("device")):
            k["device"] = "meta"
        if a and _is_cuda(a[0]) and not isinstance(a[0], torch.Tensor):
            a = ("meta",) + tuple(a[1:])
        if self.is_meta:
            # A meta tensor cannot be moved onto real storage (the PLE table is built on the
            # host, so its loader asks for exactly that). Keep it on meta: the caller's next
            # step is a copy_, which is a no-op here, and the shape checks it did first are
            # what this gate measures.
            k.pop("device", None)
            a = tuple(x for x in a if isinstance(x, (torch.dtype,)) or isinstance(x, torch.Tensor))
        return _real_to(self, *a, **k)

    torch.Tensor.to = to
    torch.Tensor.cuda = lambda self, *a, **k: _real_to(self, "meta")

    # A meta tensor has no data, so any real copy raises. The loaders that copy directly
    # (copy_ple_embedding_shard_, for one) still do all of their SHAPE checking first, which is
    # the half this gate is about, so the copy itself becomes a no-op on meta.
    _real_copy = torch.Tensor.copy_

    def copy_(self, src, *a, **k):
        if self.is_meta or (isinstance(src, torch.Tensor) and src.is_meta):
            return self
        return _real_copy(self, src, *a, **k)

    torch.Tensor.copy_ = copy_


# ---------------------------------------------------------------------------------------------
# 2. the emitted checkpoint, from the safetensors headers only
# ---------------------------------------------------------------------------------------------
_DT = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
       "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
       "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
for _n, _a in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2")):
    if hasattr(torch, _a):
        _DT[_n] = getattr(torch, _a)


def read_headers(ckpt: str) -> dict[str, tuple[list, str]]:
    out: dict[str, tuple[list, str]] = {}
    # The index is the authority on which files belong to this checkpoint: the directory also
    # holds the PLE shards as symlinks, and a stale link would otherwise abort the gate.
    idx = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.exists(idx):
        files = [os.path.join(ckpt, f)
                 for f in sorted(set(json.load(open(idx))["weight_map"].values()))]
    else:
        files = sorted(glob.glob(os.path.join(ckpt, "*.safetensors")))
    if not files:
        raise SystemExit(f"no safetensors under {ckpt}")
    for path in files:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            out[k] = (v["shape"], v["dtype"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--quantization", default="pxq")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--json", default="")
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args()

    keys = read_headers(args.ckpt)
    print(f"[keyset] checkpoint {args.ckpt}: {len(keys)} tensors in the headers")

    _force_meta()

    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.model_executor.models.utils import AutoWeightsLoader

    cfg = EngineArgs(
        model=args.ckpt, quantization=args.quantization, dtype="float16",
        trust_remote_code=True, max_model_len=args.max_model_len,
        tensor_parallel_size=1, pipeline_parallel_size=1, enforce_eager=True,
        load_format="dummy", max_num_seqs=1,
    ).create_engine_config()

    # Both of these read get_current_vllm_config(), so they have to sit inside the context.
    ctx = set_current_vllm_config(cfg)
    ctx.__enter__()
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method="tcp://127.0.0.1:29591",
                                 backend="gloo")
    initialize_model_parallel(1, 1)

    ignored: list[str] = []
    orig_can_ignore = AutoWeightsLoader._can_ignore_unexpected

    def can_ignore(self, qualname: str) -> bool:
        ok = orig_can_ignore(self, qualname)
        if ok:
            ignored.append(qualname)
        return ok

    AutoWeightsLoader._can_ignore_unexpected = can_ignore

    torch.set_default_dtype(torch.float16)
    with torch.device("meta"):
        model = initialize_model(vllm_config=cfg)
    print(f"[keyset] model built on meta: {type(model).__name__}, "
          f"{sum(1 for _ in model.named_parameters())} parameters")

    # every weight_loader becomes a recorder: keeps the routing, drops the numerics
    seen_params: set[str] = set()
    name_of = {id(p): n for n, p in model.named_parameters()}

    def recorder(param, loaded_weight, *a, **k):
        n = name_of.get(id(param))
        if n is not None:
            seen_params.add(n)
        return None

    for _, p in model.named_parameters():
        p.weight_loader = recorder

    # FusedMoE calls self.weight_loader -- the LAYER's method, not the parameter's attribute --
    # so the per-parameter recorder above never sees an expert weight. Record there too.
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    def moe_recorder(self, param=None, loaded_weight=None, weight_name=None, shard_id=None,
                     expert_id=None, return_success=False, **kw):
        n = name_of.get(id(param))
        if n is not None:
            seen_params.add(n)
        return True if return_success else None

    FusedMoE.weight_loader = moe_recorder

    last = {"name": None, "shape": None, "dtype": None, "n": 0}

    def stream():
        for name, (shape, dt) in keys.items():
            last.update(name=name, shape=shape, dtype=dt, n=last["n"] + 1)
            t = _REAL["empty"](tuple(shape), dtype=_DT.get(dt, torch.float16), device="meta")
            yield name, t

    fatal = ""
    loaded: set[str] = set()
    try:
        got = model.load_weights(stream())
        loaded = set(got or ())
    except Exception:
        fatal = traceback.format_exc(limit=40)
        print("[keyset] LOADER RAISED after %d checkpoint keys; last key pulled: %s %s %s\n%s"
              % (last["n"], last["name"], last["shape"], last["dtype"], fatal))

    params = {n for n, _ in model.named_parameters()}
    filled = (loaded | seen_params) & params
    missing = sorted(params - filled)
    unexpected = sorted(set(ignored))

    print(f"[keyset] parameters={len(params)} filled={len(filled)} "
          f"MISSING={len(missing)} UNEXPECTED={len(unexpected)} FATAL={'yes' if fatal else 'no'}")
    for n in missing[:args.show]:
        print(f"[keyset] MISSING    {n}")
    for n in unexpected[:args.show]:
        print(f"[keyset] UNEXPECTED {n}")

    ok = not missing and not unexpected and not fatal
    if args.json:
        json.dump({"ckpt": args.ckpt, "ckpt_keys": len(keys), "params": len(params),
                   "filled": len(filled), "missing": missing, "unexpected": unexpected,
                   "fatal": fatal, "pass": ok}, open(args.json, "w"), indent=1)
    print("[keyset] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("[keyset] COULD NOT RUN")
        sys.exit(2)
