#!/usr/bin/env python3
"""Refuse --hot-model when either model uses sliding-window attention.

    tests/hotswap-refuse-swa.py LLAMA_SERVER MODEL_A MODEL_B

MODEL_A is the file on the cards at open. MODEL_B is the registered extra.
Exit 0 when the server exits non-zero and its log carries the refusal sentence,
before either model is served. A Qwen pair must not be refused here: this script
is only the refusal row.
"""
import os
import subprocess
import sys

if len(sys.argv) != 4:
    sys.stderr.write("usage: hotswap-refuse-swa.py LLAMA_SERVER MODEL_A MODEL_B\n")
    sys.exit(2)

server, model_a, model_b = sys.argv[1:]
phrase = "hot swap supports models without sliding-window attention for now; Gemma support is coming"
env = os.environ.copy()
# Pin the two Volta cards so a server that failed to refuse cannot spill onto the others.
env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "2,4")
env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
proc = subprocess.run(
    [server, "-m", model_a, "--hot-model", "beta=" + model_b,
     "-c", "512", "-ngl", "0", "--host", "127.0.0.1", "--port", "18151"],
    env=env, capture_output=True, text=True, timeout=60)
log = proc.stdout + proc.stderr
sys.stdout.write(log)
if proc.returncode != 0 and phrase in log:
    sys.exit(0)
sys.stderr.write("hot-swap sliding-window refusal did not fire (rc=%s)\n" % proc.returncode)
sys.exit(1)
