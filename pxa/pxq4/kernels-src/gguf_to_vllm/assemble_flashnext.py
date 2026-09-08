"""assemble_flashnext.py -- put the Flash-Next vLLM checkpoint directory together from its parts.

    python3 -m gguf_to_vllm.assemble_flashnext --gguf <Flash-Next GGUF> --out $OUT_DIR
                                                [--ple-dir <out>/ple-fp8] [--model-index <convert.py out dir>]
                                                [--tokenizer-src <coder35 checkpoint dir>] [--dry-run]

Parts (Board B11, plan of record #1090):
  * config.json            from the GGUF's own KVs (config_qwen4exp.emit, every field traced to a source key)
                           PLUS ple_embedding_dtype=float8_e4m3fn / ple_offload_embedding=true (Blocker 2's format)
  * tokenizer files        copied from the coder35 checkpoint: pxq23 measured them byte-identical to Flash-Next's
                           (#1163: 248,077 ids, 247,587 merges in order, special ids, 50/50 probes)
  * PLE FP8 shards         ple-fp8-*.safetensors + ple.safetensors.index.json (gguf_to_vllm.ple_fp8), moved to the
                           top level of <out> (vLLM globs *.safetensors in the model dir only)
  * the rest of the model  model-*.safetensors + model.safetensors.index.json written by convert.py (Blocker 1
                           wiring of namemap_qwen4exp) -- when absent the assembly reports INCOMPLETE and W11 refuses

Writes <out>/model.safetensors.index.json as the UNION of both weight maps and checks that every file it names
exists. Idempotent: re-running only fills what is missing. Exit 0 = complete, 3 = incomplete (says what).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "chat_template.jinja",
                   "generation_config.json")


def log(m: str) -> None:
    print(m, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ple-dir", default=None, help="default <out>/ple-fp8")
    ap.add_argument("--model-index", default=None, help="dir holding convert.py's model.safetensors.index.json (default <out>)")
    ap.add_argument("--tokenizer-src", default=os.environ.get("PXA_MODELS_HOT", "./models") + "/coder35-moe-pxq4-m1")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    out = a.out; ple_dir = a.ple_dir or os.path.join(out, "ple-fp8"); midx_dir = a.model_index or out
    os.makedirs(out, exist_ok=True)
    incomplete = []

    # 1. config.json
    cfg_path = os.path.join(out, "config.json")
    if not os.path.exists(cfg_path):
        from .gguf_raw import GGUFHeaderOnly, GGUFFile
        from .config_qwen4exp import emit
        log("reading GGUF KVs (header parse, ~1 min on the 98 GB file) ...")
        try:
            g = GGUFHeaderOnly(a.gguf, os.path.getsize(a.gguf))
        except Exception:
            g = GGUFFile(a.gguf)
        cfg, missing = emit(g.kv, ple_fp8_offload=True)
        if missing:
            log(f"config: {len(missing)} expected KVs absent: {missing}"); incomplete.append("config KVs")
        if not a.dry_run:
            json.dump(cfg, open(cfg_path, "w"), indent=2)
        log(f"config.json: {len(cfg.get('text_config', {}))} text fields, ple_embedding_dtype/ple_offload_embedding set")
    else:
        c = json.load(open(cfg_path)).get("text_config", {})
        if c.get("ple_embedding_dtype") != "float8_e4m3fn" or c.get("ple_offload_embedding") is not True:
            log("config.json exists but lacks the PLE FP8/offload fields -- fix by hand or delete it to regenerate"); incomplete.append("config PLE fields")
        else:
            log("config.json present with the PLE fields")

    # 2. tokenizer
    for f in TOKENIZER_FILES:
        dst = os.path.join(out, f); src = os.path.join(a.tokenizer_src, f)
        if os.path.exists(dst):
            continue
        if os.path.exists(src):
            log(f"tokenizer: {f} <- {src}")
            if not a.dry_run:
                shutil.copy2(src, dst)
        elif f in ("tokenizer.json", "tokenizer_config.json"):
            log(f"tokenizer: MISSING {src}"); incomplete.append(f)

    # 3. PLE shards to the top level
    ple_index = os.path.join(ple_dir, "ple.safetensors.index.json")
    weight_map: dict[str, str] = {}
    if os.path.exists(ple_index):
        pi = json.load(open(ple_index))
        report = os.path.join(ple_dir, "PLE-FP8-REPORT.json")
        if os.path.exists(report) and not json.load(open(report)).get("pass"):
            log("PLE-FP8-REPORT.json says the gate FAILED -- not assembling a checkpoint from it"); incomplete.append("PLE gate")
        for name, fname in pi["weight_map"].items():
            src = os.path.join(ple_dir, fname); dst = os.path.join(out, fname)
            if not os.path.exists(dst):
                if not os.path.exists(src):
                    incomplete.append(f"PLE file {fname}"); continue
                if not a.dry_run:
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.move(src, dst)
            weight_map[name] = fname
        log(f"PLE: {len(pi['weight_map'])} tensors from {ple_index}")
    else:
        log(f"PLE index missing at {ple_index} (gguf_to_vllm.ple_fp8 not finished?)"); incomplete.append("PLE shards")

    # 4. the rest of the model (convert.py output)
    midx = os.path.join(midx_dir, "model.safetensors.index.json")
    model_map: dict[str, str] = {}
    if os.path.exists(midx):
        mi = json.load(open(midx))
        model_map = {k: v for k, v in mi["weight_map"].items() if ".ple.ngram_embedding." not in k}
        for name, fname in model_map.items():
            if not os.path.exists(os.path.join(out, fname)):
                incomplete.append(f"model file {fname}"); break
        log(f"model: {len(model_map)} tensors from {midx}")
    elif os.path.exists(os.path.join(midx_dir, "model.safetensors")):
        log("model: single model.safetensors present (no index); vLLM loads it directly")
    else:
        log("model tensors MISSING: convert.py has not produced the qwen4exp checkpoint (Blocker 1 wiring: namemap_qwen4exp -> convert.py policy)")
        incomplete.append("model tensors (Blocker 1)")

    # 5. union index
    union = {**model_map, **weight_map}
    if union and not a.dry_run:
        total = 0
        for fname in set(union.values()):
            p = os.path.join(out, fname)
            if os.path.exists(p):
                total += os.path.getsize(p)
        json.dump({"metadata": {"total_size": total}, "weight_map": union}, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
        log(f"model.safetensors.index.json: {len(union)} tensors, {total/1e9:.1f} GB on disk")
    if incomplete:
        log("INCOMPLETE: " + "; ".join(incomplete)); return 3
    log(f"COMPLETE: {out} (W11 preconditions met)"); return 0


if __name__ == "__main__":
    sys.exit(main())
