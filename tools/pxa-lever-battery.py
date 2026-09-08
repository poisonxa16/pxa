#!/usr/bin/env python3
"""launcher-levers.py - exercise every CLI lever pxa-launch.py accepts, against real
hardware, and check the ONE property the launcher promises about all of them.

WHY THIS EXISTS SEPARATELY FROM --selftest
  --selftest drives the DECISION TABLE with synthetic model facts. It never parses an
  argv, never resolves a real model, and never emits a command. So it proves the routing
  logic and proves nothing about the flags. Every lever below is reachable only through
  argparse and the command builder, which --selftest does not touch.

THE PROPERTY
  pxa-launch's stated design rule is: "REFUSES rather than silently dropping a parameter
  that does not translate." That makes one outcome, and only one, a defect:

    PASSED      the flag reached the emitted command             ok
    REFUSED     the launcher stopped and named a refusal code    ok
    ANNOUNCED   not passed, but the output says so and why       ok
    DROPPED     accepted, absent from the command, never
                mentioned again                                  DEFECT

  A DROPPED lever is the worst failure mode this launcher can have, because the operator
  gets a seat that silently ignores what they asked for and every later measurement is
  attributed to a configuration that was never applied.

Usage:  python3 launcher-levers.py [--launcher PATH] [--gguf PATH] [--vdir PATH]
"""
import argparse, os, re, subprocess, sys, tempfile

AP = argparse.ArgumentParser()
AP.add_argument("--launcher", default="tools/pxa-launch.py")
AP.add_argument("--gguf", required=True, help="a PXQ4 GGUF (llama path)")
AP.add_argument("--vdir", required=True, help="a PXQ4-converted vLLM directory")
AP.add_argument("--mmproj", dest="mmproj_file", required=True, help="an mmproj GGUF")
AP.add_argument("--vllm-image", default="pxa-sm60-dev",
                help="a MEASURED, PRESENT image, so the vLLM command builder is reachable")
AP.add_argument("--cards60", default="0,6")
AP.add_argument("--cards70", default="2,4")
A = AP.parse_args()

# The projector-resolution branches need a GGUF with a SIBLING mmproj, and no such pair
# exists on this box: a VL text GGUF carries NO vision tensors (all 809 of Qwen2.5-VL's
# live in the mmproj), and the one mmproj here sits alone in its own directory. Build the
# shape out of SYMLINKS -- no bytes are copied, no pool space is consumed, no model file
# is touched -- so the test carries its own fixture instead of depending on a directory
# somebody made by hand once.
FIXTURE = os.path.join(tempfile.gettempdir(), "pxa-lever-fixture")
os.makedirs(FIXTURE, exist_ok=True)
for src, dst in ((A.gguf, "model.gguf"), (A.mmproj_file, "mmproj-F16.gguf")):
    link = os.path.join(FIXTURE, dst)
    if os.path.islink(link) or os.path.exists(link):
        os.unlink(link)
    os.symlink(os.path.realpath(src), link)


def run(extra):
    cmd = [sys.executable, A.launcher, "--explain"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return r.returncode, (r.stdout + r.stderr)


def emitted(out):
    """The command line the launcher would exec, or '' if it refused before building one."""
    m = re.search(r"^\s*command:\s*(.+)$", out, re.M)
    return m.group(1) if m else ""


def classify(out, rc, needles, mention):
    """needles: substrings that prove the lever reached the command.
       mention: regex that proves the launcher talked about it instead."""
    cmd = emitted(out)
    hit = next((n for n in needles if cmd and n in cmd), None)
    if hit:
        return "PASSED", "emitted: " + hit
    if re.search(r"REFUSING:|\[R-\d+\]", out):
        code = re.search(r"\[(R-\d+)\]", out)
        return "REFUSED", (code.group(1) if code else "refusal")
    if mention and re.search(mention, out, re.I):
        m = re.search(mention, out, re.I)
        return "ANNOUNCED", m.group(0)[:80]
    return "DROPPED", (cmd[:100] if cmd else "no command emitted, no refusal printed")


# (label, extra argv, needles proving it landed, regex proving it was announced instead)
GG = ["--model", A.gguf, "--cards", A.cards60]
VL = ["--model", os.path.join(FIXTURE, "model.gguf"), "--cards", A.cards60]
# The vLLM command builder is only REACHABLE with an image whose caps cover the cards.
# Without one the launcher stops at R-07 (correctly), and every vLLM lever below would
# score "REFUSED" for a reason that has nothing to do with the lever being tested.
VD = ["--model", A.vdir, "--cards", A.cards60, "--accept-unmeasured",
      "--vllm-image", A.vllm_image]

CASES = [
    # ---- llama-path levers -------------------------------------------------------
    ("--ctx 8192",        GG + ["-c", "8192"],                 ["-c 8192"], None),
    ("--np 4",            GG + ["--np", "4"],                  ["-np 4"], None),
    ("--threads 32",      GG + ["--threads", "32"],            ["-t 32"], None),
    ("--ngl 40",          GG + ["--ngl", "40"],                ["-ngl 40"], None),
    ("--sm row",          GG + ["--sm", "row"],                ["-sm row"], None),
    ("--ts 60,40",        GG + ["--ts", "60,40"],              ["-ts 60,40", "--tensor-split 60,40"], None),
    ("--ctk q8_0",        GG + ["--ctk", "q8_0"],              ["-ctk q8_0"], None),
    ("--ctv q8_0",        GG + ["--ctv", "q8_0"],              ["-ctv q8_0"], None),
    # R-13L refuses q8_0/f16 because no FA vec kernel exists at head 128. It must be a
    # KERNEL-SPECIFIC refusal, not a blanket ban on quantized KV: a pair the build
    # actually compiled has to get through, or the refusal is just breaking the lever.
    ("--ctk q8_0 --ctv q6_0 (compiled pair, must PASS)",
                          GG + ["--ctk", "q8_0", "--ctv", "q6_0"],
                                                             ["-ctk q8_0"], None),
    ("--no-mmap",         GG + ["--no-mmap"],                  ["--no-mmap"], None),
    ("--host/--port",     GG + ["--host", "127.0.0.1", "--port", "9911"],
                                                               ["--port 9911"], None),
    # -b/-ub is deliberately not passed (adaptive-ub probes at startup). An explicit
    # --ub must therefore either land or be announced -- never vanish.
    ("--ub 512",          GG + ["--ub", "512"],                ["-ub 512", "-b 512"],
                          r"-b/-ub:.*NOT PASSED|--ub[^\n]*ignor|adaptive-ub"),
    ("--workload longdoc", GG + ["--workload", "longdoc"],     ["-c ", "-np "],
                          r"workload=longdoc"),
    ("--workload serve",  GG + ["--workload", "serve"],        ["-np "],
                          r"workload=serve"),
    # speculation: A5 says a bare 'mtp' can only ever mean n_max=1.
    ("--spec mtp",        GG + ["--spec", "mtp"],              ["mtp", "draft-max", "--spec"],
                          r"spec|mtp|no .*mtp tensors|n_max"),
    ("--draft-model",     GG + ["--draft-model", A.gguf],      ["-md ", "--model-draft"],
                          r"draft|speculat"),
    ("--mmproj explicit",  VL + ["--mmproj", A.mmproj_file],        ["--mmproj"], None),
    ("sibling projector, no flag (must ANNOUNCE)",
                          VL,                                  ["__never__"],
                          r"mmproj: NOT attached"),
    # --no-mmproj SUPPRESSES the flag, so "absent from the command" is the CORRECT
    # outcome and the needle test would invert the verdict. What must not happen is
    # silence: on a model that carries vision tensors the launcher has to say the seat
    # is text-only. Tested on a VL model for that reason - on a text model the flag is
    # vacuous and the case proves nothing.
    ("--no-mmproj (must ANNOUNCE the suppression)",
                          VL + ["--no-mmproj"],                ["__never__"],
                          r"SUPPRESSED by --no-mmproj"),
    ("--tier PXQ4",       GG + ["--tier", "PXQ4"],             ["-m "],
                          r"tier:\s*PXQ4|--tier"),
    ("--allow-busy",      GG + ["--allow-busy"],               ["-m "], r"busy|resident"),
    # ---- vllm-path levers --------------------------------------------------------
    ("--engine vllm",     VD + ["--engine", "vllm"],           ["vllm", "docker"],
                          r"ENGINE = vllm"),
    ("--gmu 0.80",        VD + ["--engine", "vllm", "--gmu", "0.80"],
                                                               ["0.80", "0.8"],
                          r"gpu-memory-utilization|gmu"),
    # --vllm-image never appears in the emitted command: the launcher emits a bare
    # `vllm serve` and runs in whatever container it is already inside. The flag is a
    # PREMISE for the eligibility decision, not an instruction. It must therefore be
    # ANNOUNCED - a flag that reads like it selects a runtime and silently does not is
    # how a measurement gets attributed to an image that was never involved.
    ("--vllm-image (must ANNOUNCE the container contract)",
                          VD + ["--engine", "vllm"],           ["__never__"],
                          r"CONTAINER CONTRACT"),
    ("--vllm-image sm70 on sm_60 cards (must refuse)",
                          VD[:-2] + ["--engine", "vllm", "--vllm-image", "pxa-vllm:sm70"],
                                                               ["__never__"],
                          r"R-07|REFUS"),
    # A3 at the CLI, not in the synthetic table: a non-FDO capture mode must be REFUSED.
    ("--cudagraph-mode PIECEWISE (must refuse)",
                          VD + ["--engine", "vllm", "--cudagraph-mode", "PIECEWISE"],
                                                               ["__never__"],
                          r"FULL_DECODE_ONLY|R-08|REFUS"),
    # sm_70 routing: caps is set() until the gate passes, so this must NOT silently run.
    ("V100 pair, no image forced (must not route)",
                          ["--model", A.vdir, "--cards", A.cards70],
                                                               ["__never__"],
                          r"no vLLM-eligible card|REFUS"),
]

print("=" * 78)
print("launcher lever battery -- the property is: never SILENTLY DROPPED")
print("=" * 78)
width = max(len(c[0]) for c in CASES) + 2
defects, refused, passed, announced = [], 0, 0, 0
for label, argv, needles, mention in CASES:
    try:
        rc, out = run(argv)
    except subprocess.TimeoutExpired:
        print(f"  {label:<{width}} TIMEOUT")
        defects.append(label)
        continue
    verdict, detail = classify(out, rc, needles, mention)
    # A case whose label names its own required outcome is checked against THAT, not
    # against the default "anything but DROPPED is fine". Otherwise a case written to
    # prove a refusal is specific would score green when the launcher refused it too.
    want = None
    if "must refuse" in label or "must not route" in label:
        want = ("REFUSED", "ANNOUNCED")
    elif "must PASS" in label:
        want = ("PASSED",)
    elif "must ANNOUNCE" in label:
        want = ("ANNOUNCED",)
    if want and verdict not in want:
        verdict += "  <-- EXPECTED " + "/".join(want)
        defects.append(label)
    elif want is None and verdict == "DROPPED":
        defects.append(label)
    passed += verdict == "PASSED"
    refused += verdict.startswith("REFUSED")
    announced += verdict.startswith("ANNOUNCED")
    print(f"  {label:<{width}} {verdict:<10} {detail[:90]}")

print("-" * 78)
print(f"  {passed} passed through   {refused} refused   {announced} announced   "
      f"{len(defects)} silently dropped")
if defects:
    print("\n  FAILING LEVERS:")
    for d in defects:
        print("    -", d)
    sys.exit(1)
print("\n  Every lever either landed in the command, was refused with a code, or was\n"
      "  announced. None vanished silently.")
