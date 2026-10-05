#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
"""Driver for test-pxa-lever-resolve: the lever registry must resolve every lever exactly as the
resolvers it replaced did.

    test-pxa-lever-resolve.py HARNESS                  run the grid, compare with tests/pxa-lever-resolve.golden
    test-pxa-lever-resolve.py HARNESS --write-golden   rewrite the golden from this build (only from a tree whose
                                                       resolvers are the ones the golden is meant to pin)
    test-pxa-lever-resolve.py HARNESS --diff OTHER     run the grid through two builds and print the first differences

HARNESS is built from tests/test-pxa-lever-resolve.cu. It prints, for one card set / model profile /
environment, the value of every resolver in ggml-cuda/pxa/pxa-enhance.cuh and the core registry, and the log
lines the resolvers print while doing it. One process per scenario: every resolver caches (per process, or per
model-profile generation) and the config level is read once, which is the behaviour under test.

The golden holds one line per scenario, "<first 20 hex of sha256 of the output> <scenario name>". It was
produced from the tree BEFORE the registry took over the pxa-enhance.cuh resolvers (the commit that precedes
"lever registry: resolve each row on the first read of that row"), so passing it means the registry answers
what the inline resolvers answered: the same values at every config level, card set and model profile, the same
caching (once, per model-profile generation, or live), the same one-line notices.

The second half of the test is the property the golden cannot show, because the tree before the change had no
rows: a row is resolved on the first read of THAT row, and a variable's warning appears when its row is read,
not when some other row is.
"""
import concurrent.futures as cf
import hashlib
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "pxa-lever-resolve.golden")

# ---------------------------------------------------------------------------------------------- the grid
TOPOS = ["none", "600", "600,600", "600,600,600,600", "610", "610,610", "700", "700,700", "600,700", "800", "700,800", "610,700"]
# prof ARCH EXPERTS USED MMVQ PXQ2 PXQ3 CLASS MTP
PROFS = {
    "dense":      ["qwen35", "0", "0", "0", "0", "0", "0", "0"],
    "dense_pxq4": ["qwen35", "0", "0", "300", "0", "0", "5", "0"],
    "moe":        ["qwen35moe", "256", "8", "400", "0", "0", "3", "1"],
    "moe_pxq23":  ["qwen35moe", "256", "8", "10", "120", "80", "3", "0"],
    "pxq2_only":  ["qwen35", "0", "0", "0", "300", "0", "0", "0"],
    "pxq3_only":  ["qwen35", "0", "0", "0", "0", "300", "0", "0"],
    "gemma4":     ["gemma4", "128", "8", "200", "0", "0", "1", "0"],
    "muse":       ["muse-glimmer", "0", "0", "100", "0", "0", "0", "0"],
    "swa":        ["gemma4-x", "64", "4", "0", "0", "0", "4", "0"],
}
LEVELS = [{}, {"PXA_REFERENCE": "1"}, {"PXA_ENHANCE": "0"}, {"PXA_ENHANCE": "1"}, {"PXA_REFERENCE": "0"},
          {"PXA_REFERENCE": "1", "PXA_ENHANCE": "0"}]

VALS_BOOL = [None, "0", "1", "2", "", "abc", "-1", "00", " 1"]
VALS_MASK = [None, "0", "1", "2", "3", "4", "5", "7", "8", "-1", "", "x"]
VALS = {
    "PXA_P100_FP16_GEMM": VALS_BOOL, "PXA_SPEC_1ROW": VALS_BOOL, "PXA_FA_MASK_SKIP_TILE": VALS_BOOL,
    "PXA_FA_TILE_F32ACC": VALS_BOOL, "PXA_FA_TILE_V2": VALS_BOOL, "PXA_FA_MASK_SKIP_TILE_F32": VALS_BOOL,
    "PXA_FA_PREFILL_SPLIT": [None, "0", "1", "8", "9", "10", "64", "-5", "", "abc"],
    "PXA_MODE": [None, "balance", "max", "MAX", "m", "M", "", "x"],
    "PXA_PXQ23_MMVQ": VALS_MASK, "PXA_PXQ_MMV_H2": VALS_MASK,
    "PXA_ROUTER_FUSE": [None, "0", "1", "2", "3", "4", "9", "-1", "", "x"],
    "PXA_VOLTA_CUBLAS_NE11": [None, "0", "1", "16", "64", "128", "-1", "", "x"],
    "PXA_PXQ_INT8_PREFILL": [None, "0", "1", "2", "3", "-1", "", "x"],
    "PXA_SPEC_RELAXED": VALS_BOOL,
    "PXA_FA_GQA_PACK": [None, "0", "2", "4", "8", "3", "", "x"],
    "PXA_MOE_DEVICE_MAP": [None, "0", "1", "2", "3", "", "x"],
    "PXA_FUSE_DELTANET": [None, "0", "1", "3", "7", "15", "53", "55", "63", "", "x"],
    "PXA_PXQ_MMVQ": [None, "0", "1", "2", "3", "-1", "", "x"],
    "PXA_PXQ_GEMM_2D": [None, "0", "1", "2", "3", "-1", "", "x"],
    "PXA_ENHANCE_DBG": [None, "0", "1", "2", "", "x"],
    "PXA_SPEC_SAMPLED": [None, "0", "1", ""],
    "PXA_NORM_REGCACHE": VALS_BOOL, "PXA_CONCAT_FLAT": VALS_BOOL, "PXA_CPY_FASTDIV": VALS_BOOL,
    "PXA_GETROWS_NARROW": VALS_BOOL, "PXA_TOPK_MOE_MULTIROW": VALS_BOOL,
}
OVERRIDE_ENVS = ["PXA_PXQ2_BOOK", "PXA_PXQ3_BOOK", "PXA_PXQ_CEIL_V2", "PXA_PXQ2_V3", "PXA_PXQ6_SUB", "PXA_PXQ2_SUB", "PXA_PXQ3_SUB"]

# the flash-attention rows that were already in the registry before the pxa-enhance.cuh resolvers moved in
FA_ENVS = ["PXA_FA_D512_VOLTA", "PXA_FA_D256_VOLTA_TILE", "PXA_FA_D256_VOLTA_TILE_MINKV", "PXA_FA_D256_VOLTA_TILE_MAXCOLS",
           "PXA_FA_MMA_VOLTA", "PXA_FA_MMA_VOLTA_Q8", "PXA_FA_TILE_VOLTA", "PXA_FA_TILE256", "PXA_FA_SWA_SLICE",
           "PXA_FA_SWA_KEEP", "PXQ_SM60_FA_VEC_F32", "PXA_CORE_ROUTES", "PXA_FA_QKV_DIRECT", "PXA_FA_QKV_TILE",
           "PXA_FA_QKV_DIRECT_VOLTA"]
for _e in FA_ENVS:
    VALS[_e] = [None, "0", "1", "2", "3", "", "x", "-1", "100", "2000"]

GETS = ["level", "pxq23_mmvq", "pxq_mmv_h2", "pxq_mmvq_auto", "fuse_deltanet_default", "volta_cublas_ne11", "int8_prefill",
        "router_fuse", "spec_relaxed", "spec_1row", "p100_fp16_gemm", "mode", "fa_mask_skip_tile", "fa_tile_f32acc",
        "fa_tile_v2", "fa_prefill_split", "fa_gqa_pack_default", "moe_device_map_default", "house", "gate", "path", "sites",
        "fa_rows"]


def prof_args(p):
    return ["prof"] + PROFS[p]


def scenarios():
    out = []

    def alls():
        a = []
        for g in GETS:
            a += ["get", g]
        return a

    # 1. every lever x every value, on three representative rigs; the report first, as the engine does
    for lever, vals in VALS.items():
        for v in vals:
            for topo, prof in (("700,700", "gemma4"), ("600,600,600,600", "moe"), ("610", "dense_pxq4")):
                env = {} if v is None else {lever: v}
                out.append((f"lever:{lever}={v}:{topo}:{prof}", env, ["topo", topo] + prof_args(prof) + ["report"] + alls()))
    # 2. levels x card sets x models: every resolver, the ledger, the startup line, the debug line
    for li, lv in enumerate(LEVELS):
        for topo in TOPOS:
            for prof in PROFS:
                ccs = topo if topo != "none" else "700"
                out.append((f"level{li}:{topo}:{prof}", dict(lv),
                            ["topo", topo] + prof_args(prof) + ["decisions", "startup", ccs, "dbg"] + alls()))
    # 3. order: reads before the card set / the model are registered, then registration (generation caching)
    for li, lv in enumerate(LEVELS[:4]):
        for topo in ("700,700", "600,600", "600,700", "none"):
            for p1, p2 in (("dense", "moe"), ("moe", "gemma4"), ("muse", "dense_pxq4"), ("dense_pxq4", "moe_pxq23")):
                env = dict(lv)
                out.append((f"order-a{li}:{topo}:{p1}:{p2}", env,
                            alls() + ["topo", topo] + alls() + prof_args(p1) + alls() + prof_args(p2) + alls()))
                out.append((f"order-b{li}:{topo}:{p1}:{p2}", env,
                            prof_args(p1) + alls() + ["topo", topo] + alls() + prof_args(p2) + alls() + ["noprof"] + alls()))
    # 4. the environment changed mid-run: what is cached keeps what it resolved, what is per-generation or live re-reads
    for lever in ("PXA_ROUTER_FUSE", "PXA_VOLTA_CUBLAS_NE11", "PXA_PXQ_INT8_PREFILL", "PXA_SPEC_RELAXED", "PXA_FA_GQA_PACK",
                  "PXA_MOE_DEVICE_MAP", "PXA_FUSE_DELTANET", "PXA_PXQ_MMVQ", "PXA_PXQ_GEMM_2D", "PXA_P100_FP16_GEMM",
                  "PXA_PXQ23_MMVQ", "PXA_PXQ_MMV_H2", "PXA_FA_PREFILL_SPLIT", "PXA_MODE", "PXA_ENHANCE_DBG",
                  "PXA_NORM_REGCACHE", "PXA_FA_MASK_SKIP_TILE_F32"):
        for v1 in (None, "0", "1", "2"):
            for v2 in ("0", "1", "3"):
                env = {} if v1 is None else {lever: v1}
                out.append((f"mutate:{lever}:{v1}->{v2}", env,
                            ["topo", "700,700"] + prof_args("moe") + alls() + ["set", lever, v2] + alls() +
                            prof_args("gemma4") + alls() + ["unset", lever] + alls() + prof_args("dense") + alls()))
    # 5. the book / sub-scale overrides that make the low-tier MMVQ path decline
    for ov in OVERRIDE_ENVS:
        for m in ("1", "2", "3"):
            for topo in ("700,700", "600,600"):
                out.append((f"pxq23-override:{ov}:{m}:{topo}", {ov: "1", "PXA_PXQ23_MMVQ": m},
                            ["topo", topo] + prof_args("pxq2_only") + alls() + ["decisions"]))
    # 6. random interactions among the moved levers (fixed seed)
    rnd = random.Random(20261004)
    names = [n for n in VALS if n not in FA_ENVS]
    for i in range(400):
        env = {}
        for n in rnd.sample(names, rnd.randint(2, 5)):
            v = rnd.choice(VALS[n])
            if v is not None:
                env[n] = v
        env.update(rnd.choice(LEVELS))
        topo = rnd.choice(TOPOS)
        prof = rnd.choice(list(PROFS))
        out.append((f"rand{i}", env, ["topo", topo] + prof_args(prof) + ["decisions"] + alls() +
                    prof_args(rnd.choice(list(PROFS))) + alls()))
    # 7. odd flash-attention row values, read through the report (the engine's order)
    for e in FA_ENVS:
        for v in VALS[e]:
            if v is None:
                continue
            for li in (0, 1, 2):
                env = dict(LEVELS[li])
                env[e] = v
                out.append((f"fa:{e}={v}:l{li}", env, ["topo", "700"] + prof_args("dense") + ["report", "get", "fa_rows"]))
    return out


# ---------------------------------------------------------------------------------------------- running
BASE_ENV = {k: v for k, v in os.environ.items() if not k.startswith("PXA_") and not k.startswith("PXQ_")}


def run(binp, env, args):
    e = dict(BASE_ENV)
    e.update(env)
    p = subprocess.run([binp] + args, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    return p.returncode, p.stdout.decode("utf-8", "replace")


def digest(rc, text):
    return hashlib.sha256((str(rc) + "\n" + text).encode("utf-8")).hexdigest()[:20]


def run_all(binp):
    scs = scenarios()
    with cf.ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as ex:
        results = list(ex.map(lambda sc: run(binp, sc[1], sc[2]), scs))
    return [(sc[0], sc[1], sc[2], rc, text) for sc, (rc, text) in zip(scs, results)]


def read_golden():
    g = {}
    with open(GOLDEN, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                h, name = line.split(" ", 1)
                g[name] = h
    return g


# ---------------------------------------------------------------------------------------------- the laziness property
def lazy_checks(binp):
    fails = []

    def expect(cond, what):
        if not cond:
            fails.append(what)

    bad = {"PXA_FA_D512_VOLTA": "7", "PXA_FA_TILE_VOLTA": "2", "PXA_FA_MMA_VOLTA": "2"}
    w512 = "PXA_FA_D512_VOLTA=7 is not a value this lever has"
    wtile = "PXA_FA_TILE_VOLTA=2: sm_70 large-batch flash-attention"
    wmma = "PXA_FA_MMA_VOLTA=2 is disabled in this build"

    rc, t = run(binp, bad, ["row", "FA_TILE256"])
    expect(rc == 0 and "row_FA_TILE256=1" in t, "row FA_TILE256 did not resolve")
    expect(w512 not in t and wtile not in t and wmma not in t,
           "reading one row resolved the others: a bad value's warning printed for a row nobody read\n" + t)

    rc, t = run(binp, bad, ["row", "FA_TILE256", "row", "FA_D512_VOLTA"])
    expect(w512 in t and wtile not in t and wmma not in t,
           "the D512 warning must appear when (and only when) FA_D512_VOLTA is read\n" + t)
    expect(t.index("row_FA_TILE256=") < t.index(w512), "the warning appeared before the row was read\n" + t)

    # the report resolves the rows it prints, and only those
    rc, t = run(binp, dict(bad, PXA_ROUTER_FUSE="9", PXA_PXQ23_MMVQ="3", PXA_PXQ_MMV_H2="3"), ["topo", "700", "report"])
    expect(w512 in t and wtile in t and wmma in t, "the report did not resolve the rows it prints\n" + t)
    expect("PXA_ROUTER_FUSE=9 is out of range" not in t and "PXA_PXQ23_MMVQ: mode" not in t and "PXA_PXQ_MMV_H2: mode" not in t,
           "the report resolved a row it does not print\n" + t)

    # a moved row prints its own notice when it is read, and not before
    rc, t = run(binp, {"PXA_PXQ_MMV_H2": "3", "PXA_PXQ23_MMVQ": "3"}, ["get", "pxq23_mmvq"])
    expect("PXA_PXQ23_MMVQ: mode 3" in t and "PXA_PXQ_MMV_H2: mode" not in t,
           "reading PXQ23_MMVQ resolved PXQ_MMV_H2 too\n" + t)
    rc, t = run(binp, {"PXA_ROUTER_FUSE": "9"}, ["get", "spec_1row"])
    expect("out of range" not in t, "reading SPEC_1ROW resolved ROUTER_FUSE\n" + t)

    # a row that reads the card set and the model must not be frozen by a read that came before they exist
    rc, t = run(binp, {}, ["get", "router_fuse", "topo", "700,700", "prof", "qwen35moe", "256", "8", "400", "0", "0", "3", "0",
                          "get", "router_fuse"])
    lines = [l for l in t.splitlines() if l.startswith("router_fuse=")]
    expect(lines == ["router_fuse=0", "router_fuse=1"],
           "ROUTER_FUSE was not re-resolved when the model profile arrived: %r\n%s" % (lines, t))
    return fails


# ---------------------------------------------------------------------------------------------- main
def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    binp = args[0]
    if "--diff" in args:
        other = args[args.index("--diff") + 1]
        a, b = run_all(binp), run_all(other)
        n = 0
        for x, y in zip(a, b):
            if (x[3], x[4]) != (y[3], y[4]):
                n += 1
                if n <= 5:
                    import difflib
                    print("=== DIFF", x[0], x[1])
                    for l in list(difflib.unified_diff(x[4].splitlines(), y[4].splitlines(), "A", "B", lineterm="", n=1))[:30]:
                        print("   ", l)
        print("%d scenarios, %d differ" % (len(a), n))
        return 1 if n else 0

    res = run_all(binp)
    if "--write-golden" in args:
        pass
        with open(GOLDEN, "w", encoding="utf-8") as f:
            for name, env, a, rc, text in res:
                f.write("%s %s\n" % (digest(rc, text), name))
        print("wrote %s: %d scenarios" % (GOLDEN, len(res)))
        return 0

    golden = read_golden()
    names = set(r[0] for r in res)
    failed = []
    for name, env, a, rc, text in res:
        if rc not in (0,):
            failed.append((name, "exit status %d" % rc, text))
        elif golden.get(name) != digest(rc, text):
            failed.append((name, "output differs from the golden", text))
    missing = sorted(set(golden) - names)
    for name, why, text in failed[:5]:
        print("FAIL %s: %s" % (name, why))
        print("\n".join("    " + l for l in text.splitlines()[:60]))
    if failed:
        print("%d of %d scenarios failed" % (len(failed), len(res)))
    if missing:
        print("%d scenarios in the golden are no longer generated" % len(missing))
    lazy = lazy_checks(binp)
    for f in lazy:
        print("FAIL lazy: " + f)
    ok = not failed and not missing and not lazy
    print("test-pxa-lever-resolve: %s (%d scenarios against the golden, %d laziness checks)" %
          ("OK" if ok else "FAILED", len(res), 6))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
