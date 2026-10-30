#!/usr/bin/env python3
"""pxa-launch mmproj auto-find for a text GGUF (todo gemma-mmproj-support): a projector whose file name
carries the model's name, or the only projector beside the only model file, is attached; ambiguous
cases are listed, never guessed. Pure-function test, no GPU.   usage: test-pxa-launch-mmproj-autofind.py
exit 0 = PASS"""
import importlib.util, os, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pxl", os.path.join(HERE, "..", "tools", "pxa-launch.py"))
pxl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pxl)
fails = []


def case(name, files, model, want):
    d = tempfile.mkdtemp(prefix="mmproj-af-")
    try:
        for f in files:
            open(os.path.join(d, f), "wb").close()
        cands, _ = pxl.find_mmproj(os.path.join(d, model), "gguf")
        got, why = pxl.pick_mmproj_auto(os.path.join(d, model), cands)
        got = os.path.basename(got) if got else None
        ok = got == want
        print(("PASS" if ok else "FAIL"), name, "->", got, f"({why})")
        if not ok:
            fails.append(name)
    finally:
        shutil.rmtree(d, ignore_errors=True)


G = "gemma-4-26B-A4B-it-PXQN4-embq8.gguf"
M = "mmproj-gemma-4-26B-A4B-it-f16.gguf"
case("gemma name match beside other models", [G, "gemma-4-26B-A4B-it-PXQ4.gguf", "other-9B-Q4.gguf", M], G, M)
case("only projector, only model (unrelated name)", ["model.gguf", "mmproj-F16.gguf"], "model.gguf", "mmproj-F16.gguf")
case("one projector, two models, no name match", ["a-7B-Q4.gguf", "b-7B-Q4.gguf", "mmproj-zzz-f16.gguf"], "a-7B-Q4.gguf", None)
case("f16 and q8_0 both match: not guessed", [G, M, "mmproj-gemma-4-26B-A4B-it-Q8_0.gguf"], G, None)
case("name boundary: 26B projector must not match 26B-A4B", ["gemma-4-26B-A4B-it-Q4_K_M.gguf", "x.gguf", "mmproj-gemma-4-26B-f16.gguf"], "gemma-4-26B-A4B-it-Q4_K_M.gguf", None)
case("quant tag after the name matches", ["Qwen3-VL-8B-Instruct-Q4_K_M.gguf", "x.gguf", "mmproj-Qwen3-VL-8B-Instruct-F16.gguf"], "Qwen3-VL-8B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3-VL-8B-Instruct-F16.gguf")
case("no projector", [G], G, None)
print("RESULT", "PASS" if not fails else "FAIL " + ",".join(fails))
sys.exit(0 if not fails else 1)
