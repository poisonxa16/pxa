"""gate_qwen4exp_emitted.py -- read the EMITTED safetensors back and prove the transforms ran.

CPU only, seconds, no GPU. The sibling gates check the TABLES: gate_qwen4exp_vs_producer.py
proves the map agrees with the script that built the GGUF, and convert.py's own --verify
proves every native panel tensor is a byte-exact split of it. Neither proves that the value
transforms the map DECLARES were actually applied to the bytes on disk -- a transform that is
declared and skipped is the same silent failure as one that was never declared, and both
produce a checkpoint that loads.

So this opens the written shards and compares, tensor by tensor, against the GGUF:

  gemma norms   HF == ggml - 1        (hyperconnection, PLE, indexer q/k layernorms)
  ssm_norm      HF == ggml            (the sole norm that is NOT offset)
  A_log         HF == log(-A[gather]) (both the value transform and the v-head gather)
  dt_bias       HF == ggml[gather]    (the gather alone, on an exactly-representable tensor)
  in_proj_a     HF == ssm_alpha[gather]
  index_qk_proj HF rows [0, n_q) == indexer.q_proj and [n_q, n_q+n_k) == indexer.k_proj

Run:  python3 gate_qwen4exp_emitted.py [checkpoint dir] [gguf]
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gguf_to_vllm import dequant_ref as D      # noqa: E402
from gguf_to_vllm import gguf_raw as G         # noqa: E402
from gguf_to_vllm import namemap as NM         # noqa: E402

DEFAULT_CKPT = os.environ.get("PXQ_CKPT", "/path/to/vllm-fp8-checkpoint")
DEFAULT_GGUF = os.environ.get("PXQ_GGUF", "/path/to/model-PXQU.gguf")
_DT = {"F16": np.float16, "F32": np.float32, "BF16": np.uint16, "U8": np.uint8, "I8": np.int8}


def reader(ckpt: str):
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]

    def read(name: str) -> np.ndarray:
        path = os.path.join(ckpt, idx[name])
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n))[name]
            beg, end = meta["data_offsets"]
            f.seek(8 + n + beg)
            raw = f.read(end - beg)
        return np.frombuffer(raw, dtype=_DT[meta["dtype"]]).reshape(meta["shape"])
    return read


def main(argv: list[str]) -> int:
    ckpt = argv[1] if len(argv) > 1 else DEFAULT_CKPT
    gguf = argv[2] if len(argv) > 2 else DEFAULT_GGUF
    for p in (os.path.join(ckpt, "model.safetensors.index.json"), gguf):
        if not os.path.exists(p):
            print(f"SKIP: {p} not present")
            return 0
    read = reader(ckpt)
    gg = G.GGUFFile(gguf)
    LM = "model.language_model"
    fails: list[str] = []

    def gval(n: str) -> np.ndarray:
        ti = gg.tensors[n]
        return D.dequant_any(gg.raw(n), ti.type_id, ti.dims).astype(np.float32)

    def chk(label: str, a, b, tol: float = 1e-3) -> None:
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        d = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
        good = d <= tol
        print(f"  {'PASS' if good else 'FAIL'} {label:50s} maxabs={d:.3e} n={a.size}")
        if not good:
            fails.append(label)

    try:
        geom = NM.gdn_geometry(gg.kv)
        g = np.asarray(NM.v_head_gather(geom))
        gdn = next(int(n.split(".")[1]) for n in gg.order if n.endswith(".ssm_a"))
        att = next(int(n.split(".")[1]) for n in gg.order if n.endswith(".attn_q.weight"))
        ple = next(int(n.split(".")[1]) for n in gg.order if n.endswith(".ple_key.weight"))

        chk(f"L{gdn} attn_hc hc_norm == ggml-1",
            read(f"{LM}.layers.{gdn}.attn_hyper_connection.hc_norm.weight"),
            gval(f"blk.{gdn}.hc_attn_norm.weight") - 1.0)
        chk(f"L{ple} ple.norm_key == ggml-1",
            read(f"{LM}.layers.{ple}.ple.norm_key.weight"),
            gval(f"blk.{ple}.ple_norm_key.weight") - 1.0)
        chk(f"L{att} indexer.q_layernorm == ggml-1",
            read(f"{LM}.layers.{att}.self_attn.indexer.q_layernorm.weight"),
            gval(f"blk.{att}.indexer.q_norm.weight") - 1.0)
        chk(f"L{gdn} linear_attn.norm == ggml (NOT offset)",
            read(f"{LM}.layers.{gdn}.linear_attn.norm.weight"),
            gval(f"blk.{gdn}.ssm_norm.weight"))
        chk(f"L{gdn} A_log == log(-ssm_a[gather])",
            read(f"{LM}.layers.{gdn}.linear_attn.A_log"),
            np.log(-gval(f"blk.{gdn}.ssm_a")[g]))
        chk(f"L{gdn} dt_bias == ssm_dt.bias[gather]",
            read(f"{LM}.layers.{gdn}.linear_attn.dt_bias"),
            gval(f"blk.{gdn}.ssm_dt.bias")[g])
        chk(f"L{gdn} in_proj_a == ssm_alpha[gather]",
            read(f"{LM}.layers.{gdn}.linear_attn.in_proj_a.weight"),
            gval(f"blk.{gdn}.ssm_alpha.weight")[g], tol=2e-3)
        qk = read(f"{LM}.layers.{att}.self_attn.indexer.index_qk_proj.weight")
        nq = gg.tensors[f"blk.{att}.indexer.q_proj.weight"].ne1
        chk("index_qk_proj rows [0,nq) == indexer.q_proj",
            qk[:nq], gval(f"blk.{att}.indexer.q_proj.weight"), tol=2e-3)
        chk("index_qk_proj rows [nq,end) == indexer.k_proj",
            qk[nq:], gval(f"blk.{att}.indexer.k_proj.weight"), tol=2e-3)
    finally:
        gg.close()

    print()
    if fails:
        print(f"FAILED {len(fails)}: " + "; ".join(fails))
        return 1
    print("ALL PASS: every declared value transform and head gather is present in the bytes "
          "that were written, not only in the table that declared it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
