#!/usr/bin/env python3
# test-release-scan-allow.py -- scripts/release-scan.py allow rules. @secrets-allow: a secret file's literals are allowed
# in the named package files only, and still block everywhere else. @archive-allow: a binary archive at a named path is a
# warning, any other binary archive still blocks. Synthetic data, temp dirs only.
import json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = os.path.join(ROOT, "scripts", "release-scan.py")
LIT = "zq7Hx2Lm9Pw4Rt6Yv8Nb3Kc5Jd1Fg0Sa"   # synthetic, 32 chars


def run(blocklist, pkg, out):
    p = subprocess.run([sys.executable, "-I", SCAN, "--blocklist", blocklist, "--json", out, pkg],
                       capture_output=True, text=True)
    return p.returncode, json.load(open(out)) if os.path.exists(out) else {}


def hits(js, cat):
    d = js[next(iter(js))] if "blocks" not in js else js
    return d.get("blocks", {}).get(cat, {}).get("by_file", {})


def main():
    fails = 0
    with tempfile.TemporaryDirectory() as t:
        sec = os.path.join(t, "sec"); pkg = os.path.join(t, "pkg")
        os.makedirs(sec); os.makedirs(os.path.join(pkg, "inc"))
        open(os.path.join(sec, "table-record.txt"), "w").write(LIT + "\n")
        open(os.path.join(pkg, "inc", "tables.h"), "w").write('static const char * t = "%s";\n' % LIT)
        open(os.path.join(pkg, "other.c"), "w").write('const char * k = "%s";\n' % LIT)
        base = "@category demo scope=text\n\\bDEMOBLOCKWORD\\b\n@selftest demo DEMOBLOCKWORD\n@secrets-dir %s\n" % sec
        for name, extra, want in (("no allow", "", {"inc/tables.h", "other.c"}),
                                  ("allow tables.h", "@secrets-allow ^table-record\\.txt$ inc/tables.h\n", {"other.c"}),
                                  ("allow other record", "@secrets-allow ^nope\\.txt$ inc/*\n", {"inc/tables.h", "other.c"})):
            bl = os.path.join(t, "bl-%d.txt" % len(extra)); open(bl, "w").write(base + extra)
            rc, js = run(bl, pkg, os.path.join(t, "out-%d.json" % len(extra)))
            got = set(hits(js, "secret-literal"))
            ok = got == want and rc == 2
            print("%s  %-20s rc=%d hits=%s" % ("ok  " if ok else "FAIL", name, rc, sorted(got)))
            fails += not ok
        bad = os.path.join(t, "bad.txt"); open(bad, "w").write(base + "@secrets-allow ^x$\n")
        rc, _ = run(bad, pkg, os.path.join(t, "bad.json"))
        print("%s  malformed directive rc=%d (want 3)" % ("ok  " if rc == 3 else "FAIL", rc)); fails += rc != 3
    with tempfile.TemporaryDirectory() as t:
        pkg = os.path.join(t, "pkg"); os.makedirs(os.path.join(pkg, "up"))
        for n in ("up/wrapper.jar", "other.jar"):
            open(os.path.join(pkg, n), "wb").write(b"PK\x03\x04\x00\x01\x02binary")
        base = "@category demo scope=text\n\\bDEMOBLOCKWORD\\b\n@selftest demo DEMOBLOCKWORD\n"
        for name, extra, want in (("no archive allow", "", {"up/wrapper.jar", "other.jar"}),
                                  ("allow up/*.jar", "@archive-allow up/*.jar\n", {"other.jar"})):
            bl = os.path.join(t, "a-%d.txt" % len(extra)); open(bl, "w").write(base + extra)
            rc, js = run(bl, pkg, os.path.join(t, "a-%d.json" % len(extra)))
            got = set(hits(js, "nested-archive"))
            ok = got == want and rc == 2
            print("%s  %-20s rc=%d hits=%s" % ("ok  " if ok else "FAIL", name, rc, sorted(got)))
            fails += not ok
    print("test-release-scan-allow: %s" % ("PASS" if not fails else "FAIL %d" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
