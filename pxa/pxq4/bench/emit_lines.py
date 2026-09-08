#!/usr/bin/env python3
"""Turn a window log into the machine-readable SPEED / QUALITY lines main's #1011 mandates.

The format is fixed by that order and the keys are exact. This exists so the lines are
generated from the log rather than retyped from it, which is the same reason the op
inventory is a build step and not a habit.

  emit_lines.py <window.log> <model> <build> <lib-md5> <ckpt> <cards> <shape>
"""
import re
import sys


def main():
    log, model, build, lib, ckpt, cards, shape = sys.argv[1:8]
    quiet = sys.argv[8] if len(sys.argv) > 8 else "unknown"
    txt = open(log, errors="replace").read()

    # CELL <arm> <cell> median <v> tok/s
    for m in re.finditer(r"^CELL (\S+) (\S+) median ([0-9.]+) tok/s", txt, re.M):
        arm, cell, val = m.groups()
        if cell == "warm":
            continue
        armname = "fused-off" if arm == "0" else f"fused-{arm}"
        print(f"SPEED model={model} engine=pxa-vllm build={build} lib={lib} "
              f"ckpt={ckpt} cards={cards} shape={shape} cell={cell}-{armname} "
              f"value={val} unit=tok/s quiet={quiet} "
              f"reason=PXA_OPS_FUSED={arm}")

    # <a> vs <b>: byte-identical X/Y, same first token Z/Y
    for m in re.finditer(r"^(\S+) vs (\S+): byte-identical (\d+)/(\d+), "
                         r"same first token (\d+)/(\d+)", txt, re.M):
        a, b, ex, n, ft, _ = m.groups()
        np_ = "np2" if a.endswith("np2") else "np1"
        print(f"QUALITY model={model} ckpt={ckpt} engine=pxa-vllm build={build} "
              f"metric=byte-gate value={ex}/{n} pass={'yes' if ex == n else 'no'} "
              f"evidence={log}:{a}-vs-{b}-{np_}")
        print(f"QUALITY model={model} ckpt={ckpt} engine=pxa-vllm build={build} "
              f"metric=top-token value={ft}/{n} pass={'yes' if ft == n else 'no'} "
              f"evidence={log}:{a}-vs-{b}-{np_}")

    for m in re.finditer(r"^PROBE (PASS|FAIL)", txt, re.M):
        print(f"QUALITY model={model} ckpt={ckpt} engine=pxa-vllm build={build} "
              f"metric=short-prefill-probe value={m.group(1)} "
              f"pass={'yes' if m.group(1) == 'PASS' else 'no'} evidence={log}")


if __name__ == "__main__":
    main()
