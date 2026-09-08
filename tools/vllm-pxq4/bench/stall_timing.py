import json, statistics, sys
d = json.load(open(sys.argv[1]))
for r in d["single"]["runs"]:
    ts = r["times"]
    gaps = [(ts[i+1]-ts[i])*1000 for i in range(len(ts)-1)]
    med = statistics.median(gaps)
    stalls = [(i, ts[i+1], g) for i, g in enumerate(gaps) if g > 2.5*med]
    if len(stalls) >= 4:
        at = [round(s[1], 2) for s in stalls]
        spac = [round(at[j+1]-at[j], 2) for j in range(len(at)-1)]
        print("run%2d tps=%.1f stalls_at_s=%s" % (r["i"], r["e2e_tps"], at[:14]))
        print("      spacings=%s" % (spac[:13],))
