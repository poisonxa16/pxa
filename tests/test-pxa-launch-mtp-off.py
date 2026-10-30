#!/usr/bin/env python3
"""--spec mtp on a file without a draft head: the flag is dropped with a one-line note (public pxa #7).

  nokv   no nextn_predict_layers KV, no nextn tensors -> note "no draft (MTP) head"; no refusal, exit 0,
         no --spec-type in the command, and it must NOT say the file lies or name other files.
  liar   nextn_predict_layers=1 in the header, 0 nextn tensors -> note "declares a draft head ... no tensors".
  head   nextn tensor present -> --spec-type mtp IS in the command (positive control, unchanged behaviour).
Fake GPU table, tiny GGUFs; no GPU needed.

usage: test-pxa-launch-mtp-off.py [--launcher tools/pxa-launch.py]    exit 0 = PASS
"""
import argparse, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def tiny(path, nextn, head=False):
    sys.path.insert(0, os.path.join(HERE, "..", "gguf-py"))
    import numpy as np
    import gguf
    w = gguf.GGUFWriter(path, "qwen35")
    w.add_block_count(1); w.add_context_length(256); w.add_embedding_length(64)
    w.add_feed_forward_length(128); w.add_head_count(4); w.add_head_count_kv(4)
    w.add_layer_norm_rms_eps(1e-5); w.add_file_type(0)
    if nextn is not None:
        w.add_uint32("qwen35.nextn_predict_layers", nextn)
    w.add_tensor("token_embd.weight", np.zeros((32, 64), dtype=np.float32))
    if head:
        w.add_tensor("blk.1.nextn.eh_proj.weight", np.zeros((32, 64), dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launcher", default=os.path.join(HERE, "..", "tools", "pxa-launch.py"))
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="pxa-launch-mtp-")
    fails = []
    try:
        cases = {"nokv": (None, False), "liar": (1, False), "head": (1, True)}
        out = {}
        for name, (n, hd) in cases.items():
            f = os.path.join(tmp, f"{name}.gguf"); tiny(f, n, hd)
            env = dict(os.environ, PXA_LAUNCH_FAKE_GPUS="2x700", HOME=tmp)
            r = subprocess.run([sys.executable, a.launcher, "--explain", "--gpus", "0,1", "--model", f,
                                "--ctx", "128", "--spec", "mtp"], env=env, capture_output=True, text=True, timeout=300)
            out[name] = (r.returncode, r.stdout + r.stderr)

        def line(t, key):
            return next((l.strip()[:140] for l in t.splitlines() if key in l), "none")

        def check(name, ok, what):
            print(("PASS" if ok else "FAIL"), name + ":", what)
            ok or fails.append(name)

        for name, key, phrase in (("nokv", "MTP off:", "has no draft (MTP) head"),
                                  ("liar", "MTP off:", "declares a draft head")):
            rc, t = out[name]
            ok = (rc in (0, 5) and phrase in t and "REFUSING" not in t and "--spec-type" not in t
                  and "PXA-Agent-9B-f16" not in t and not re.search(r"\blie\b", t))
            check(name, ok, f"exit {rc}; " + line(t, key))
        rc, t = out["liar"]
        check("liar-kv", "nextn_predict_layers 1)" in t, line(t, "nextn_predict_layers"))
        rc, t = out["head"]
        check("head", rc in (0, 5) and "MTP off" not in t and "--spec-type mtp" in t, f"exit {rc}; " + line(t, "--spec-type"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT", "PASS" if not fails else "FAIL " + ",".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
