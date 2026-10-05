#!/usr/bin/env python3
"""pxa-isa-scan.py -- count the instructions past baseline x86-64 in the code of an ELF library, from `objdump -d` text.

    objdump -d -w --no-show-raw-insn -M intel LIB | scripts/pxa-isa-scan.py [--name LIB] [--require-baseline]

Prints one block: VEX instructions (every AVX / AVX2 / FMA / F16C instruction is VEX-encoded, so that count IS the AVX count),
the FMA and F16C counts on their own, ymm / zmm register references, and (information only) SSSE3 / SSE4.x / popcnt-class
instructions. With --require-baseline it exits 1 when the library has any VEX, FMA, F16C, ymm or zmm instruction: that is the
check make-release-tarball.sh runs on lib-compat/libggml.so, and the reason a CUDA object served out of a shared ccache (which
does not key nvcc's -Xcompiler host flags) can never reach the package unseen.

Exit status: 0 = printed (and, with --require-baseline, clean); 1 = --require-baseline and not clean; 2 = no instructions read.
"""
import argparse
import re
import sys

NONVEX_V = {"verr", "verw", "vmcall", "vmlaunch", "vmresume", "vmptrld", "vmptrst", "vmread", "vmwrite", "vmxoff", "vmxon", "vmclear",
            "vmmcall", "vmload", "vmsave", "vmrun", "vmfunc"}
PREFIX = {"rep", "repz", "repnz", "repe", "repne", "lock", "data16", "notrack", "bnd", "cs", "ds", "es", "fs", "gs", "addr32"}
F16C = {"vcvtph2ps", "vcvtps2ph"}
SSSE3 = set("pshufb pabsb pabsw pabsd palignr phaddw phaddd phaddsw phsubw phsubd phsubsw pmaddubsw pmulhrsw psignb psignw psignd".split())
SSE41 = set(("pblendw blendps blendpd blendvps blendvpd pblendvb pminsb pmaxsb pminuw pmaxuw pminsd pmaxsd pminud pmaxud pmulld pmuldq ptest "
             "pinsrb pinsrd pinsrq pextrb pextrd pextrq roundss roundsd roundps roundpd dpps dppd pmovsxbw pmovsxbd pmovsxbq pmovsxwd pmovsxwq "
             "pmovsxdq pmovzxbw pmovzxbd pmovzxbq pmovzxwd pmovzxwq pmovzxdq packusdw pcmpeqq insertps extractps movntdqa mpsadbw phminposuw").split())
SSE42 = set("pcmpgtq crc32 pcmpestri pcmpestrm pcmpistri pcmpistrm".split())
OTHER = set("popcnt lzcnt tzcnt andn bextr blsi blsmsk blsr bzhi pdep pext mulx rorx sarx shlx shrx movbe cmpxchg16b adcx adox rdrand rdseed".split())
HDR = re.compile(r"^[0-9a-f]+ <(.+)>:$")


def scan(lines):
    c = {"insn": 0, "vex": 0, "fma": 0, "f16c": 0, "ymm": 0, "zmm": 0, "ssse3": 0, "sse41": 0, "sse42": 0, "other": 0}
    funcs = {}
    fn = None
    for line in lines:
        m = HDR.match(line.rstrip("\n"))
        if m:
            fn = m.group(1)
            continue
        if "\t" not in line:
            continue
        ins = line.rstrip("\n").split("\t")[-1].strip()
        if not ins or ins.startswith((".", "<")):
            continue
        toks = ins.split()
        while toks and toks[0] in PREFIX and len(toks) > 1:
            toks = toks[1:]
        mn = toks[0]
        c["insn"] += 1
        hit = None
        if mn.startswith("v") and mn not in NONVEX_V:
            c["vex"] += 1
            hit = "vex"
            if mn.startswith(("vfmadd", "vfmsub", "vfnmadd", "vfnmsub")):
                c["fma"] += 1
            if mn in F16C:
                c["f16c"] += 1
        elif mn in SSSE3:
            c["ssse3"] += 1
        elif mn in SSE41:
            c["sse41"] += 1
        elif mn in SSE42 or mn.startswith("crc32"):
            c["sse42"] += 1
        elif mn in OTHER:
            c["other"] += 1
        if "ymm" in ins:
            c["ymm"] += 1
            hit = hit or "ymm"
        if "zmm" in ins:
            c["zmm"] += 1
            hit = hit or "zmm"
        if hit and fn:
            funcs[fn] = funcs.get(fn, 0) + 1
    return c, funcs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="(stdin)")
    ap.add_argument("--require-baseline", action="store_true", help="exit 1 when any VEX / FMA / F16C / ymm / zmm instruction is present")
    ap.add_argument("--top", type=int, default=8, help="how many offending functions to list")
    a = ap.parse_args()
    c, funcs = scan(sys.stdin)
    if c["insn"] == 0:
        print("pxa-isa-scan: no instructions read from %s" % a.name, file=sys.stderr)
        return 2
    print("%s: %d instructions" % (a.name, c["insn"]))
    print("  VEX %d  (FMA %d, F16C %d)   ymm refs %d   zmm refs %d" % (c["vex"], c["fma"], c["f16c"], c["ymm"], c["zmm"]))
    print("  info: SSSE3 %d  SSE4.1 %d  SSE4.2 %d  popcnt/lzcnt/bmi-class %d" % (c["ssse3"], c["sse41"], c["sse42"], c["other"]))
    if funcs:
        print("  functions carrying VEX/ymm/zmm: %d" % len(funcs))
        for f, n in sorted(funcs.items(), key=lambda kv: -kv[1])[:a.top]:
            print("    %6d  %s" % (n, f[:120]))
    bad = c["vex"] + c["fma"] + c["f16c"] + c["ymm"] + c["zmm"]
    if a.require_baseline and bad:
        print("pxa-isa-scan: %s is NOT baseline x86-64 (%d AVX-class instructions)" % (a.name, bad), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
