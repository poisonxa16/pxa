#!/usr/bin/env python3
"""usage: dequant_fp8_drafter.py <fp8 drafter dir> <out dir>  (FP8 E4M3 128x128 block scales -> BF16)"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BLOCK = 128


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    out = {}
    with safe_open(os.path.join(src, "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.endswith("weight_scale_inv"):
                continue
            t = f.get_tensor(k)
            if t.dtype == torch.float8_e4m3fn:
                s = f.get_tensor(k[: -len("weight")] + "weight_scale_inv").float()
                w = t.float()
                scale = s.repeat_interleave(BLOCK, 0)[: w.shape[0]].repeat_interleave(BLOCK, 1)[:, : w.shape[1]]
                t = (w * scale).to(torch.bfloat16)
            out[k] = t.contiguous()
    save_file(out, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg.pop("quantization_config", None)
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    for n in os.listdir(src):
        if n not in ("model.safetensors", "config.json", "SHA256SUMS") and os.path.isfile(os.path.join(src, n)):
            shutil.copy(os.path.join(src, n), dst)
    print(len(out), "tensors")


if __name__ == "__main__":
    main()
