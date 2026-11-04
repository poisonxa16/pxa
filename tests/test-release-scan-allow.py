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
    # @shape-allow: a secret-shape hit in a known upstream text file (cryptography's ssh.py names the
    # OpenSSH private-key header as a parsing constant) is allowed only at that path AND with those exact bytes
    import hashlib
    with tempfile.TemporaryDirectory() as t:
        pkg = os.path.join(t, "pkg"); os.makedirs(os.path.join(pkg, "lib", "up"))
        hdr = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"
        up = ('_SK_START = b"%s"\n' % hdr).encode()
        open(os.path.join(pkg, "lib", "up", "ssh.py"), "wb").write(up)
        open(os.path.join(pkg, "lib", "up", "other.py"), "wb").write(up)          # same bytes, other path: blocks
        open(os.path.join(pkg, "lib", "mine.py"), "wb").write(up + b"# ours\n")   # other bytes: blocks
        sha = hashlib.sha256(up).hexdigest()
        base = "@category demo scope=text\n\\bDEMOBLOCKWORD\\b\n@selftest demo DEMOBLOCKWORD\n"
        allp = {"lib/up/ssh.py", "lib/up/other.py", "lib/mine.py"}
        for name, extra, want, rcw in (
                ("no shape allow", "", allp, 2),
                ("allow ssh.py by sha", "@shape-allow secret-shape-pem %s lib/up/ssh.py\n" % sha, allp - {"lib/up/ssh.py"}, 2),
                ("allow glob by sha", "@shape-allow secret-shape-pem %s lib/*\n" % sha, {"lib/mine.py"}, 2),
                ("wrong sha", "@shape-allow secret-shape-pem %s lib/up/ssh.py\n" % ("0" * 64), allp, 2),
                ("other shape", "@shape-allow secret-shape-github %s lib/up/ssh.py\n" % sha, allp, 2)):
            bl = os.path.join(t, "s-%d.txt" % len(name)); open(bl, "w").write(base + extra)
            rc, js = run(bl, pkg, os.path.join(t, "s-%d.json" % len(name)))
            got = set(hits(js, "secret-shape-pem"))
            ok = got == want and rc == rcw
            print("%s  %-20s rc=%d hits=%s" % ("ok  " if ok else "FAIL", name, rc, sorted(got)))
            fails += not ok
        os.remove(os.path.join(pkg, "lib", "up", "other.py")); os.remove(os.path.join(pkg, "lib", "mine.py"))
        bl = os.path.join(t, "s-only.txt"); open(bl, "w").write(base + "@shape-allow secret-shape-pem %s lib/up/ssh.py\n" % sha)
        rc, js = run(bl, pkg, os.path.join(t, "s-only.json"))
        print("%s  only allowed file rc=%d (want 0)" % ("ok  " if rc == 0 else "FAIL", rc)); fails += rc != 0
        for bad_line in ("@shape-allow secret-shape-pem %s\n" % sha, "@shape-allow demo %s lib/*\n" % sha,
                         "@shape-allow secret-shape-pem %s lib/*\n" % sha[:63]):
            bl = os.path.join(t, "s-bad.txt"); open(bl, "w").write(base + bad_line)
            rc, _ = run(bl, pkg, os.path.join(t, "s-bad.json"))
            print("%s  malformed shape-allow rc=%d (want 3)" % ("ok  " if rc == 3 else "FAIL", rc)); fails += rc != 3
    print("test-release-scan-allow: %s" % ("PASS" if not fails else "FAIL %d" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
