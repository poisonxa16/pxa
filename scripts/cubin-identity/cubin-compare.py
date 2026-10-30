"""cubin-compare.py old.json new.json [archs...]: every listed arch's cubin set identical old vs new, per binary; new arch counts shown."""
import json, sys
o, n = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
archs = sys.argv[3:] or ["sm_60", "sm_61", "sm_70", "sm_75"]
bad = 0
for b in sorted(set(o) | set(n)):
    if b not in o or b not in n:
        print("MISSING", b, "old" if b not in o else "new"); bad += 1; continue
    for a in archs:
        x, y = o[b]["archs"].get(a, []), n[b]["archs"].get(a, [])
        ok = x == y and len(x) > 0
        bad += not ok
        print("%-5s %s %s cubins old %d new %d%s" % ("SAME" if ok else "DIFF", b, a, len(x), len(y), "" if ok else " <-- differs"))
    extra = {a: c for a, c in n[b]["counts"].items() if a not in archs}
    print("      %s new-only archs %s (old had %s)" % (b, extra, {a: c for a, c in o[b]["counts"].items() if a not in archs}))
print("CUBIN-IDENTITY", "PASS" if bad == 0 else "FAIL (%d)" % bad)
sys.exit(1 if bad else 0)
