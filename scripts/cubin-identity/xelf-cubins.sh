#!/bin/bash
# Per-arch cubin digests with cuobjdump -xelf all (docker nvidia/cuda:12.8.1-devel-ubuntu22.04, cores 36-53):
#   xelf-cubins.sh <work_dir> <out.json> <binary>...
# ELF (.so): cuobjdump -xelf all on the file. PE (.dll): each fatbin of its .nv_fatb section is written out (check-win-package's PE
# reader) and cuobjdump -xelf all runs on every fatbin. out.json = {binary: {sm_XX: sorted sha256 of every cubin}}.
set -eu
WK=$1; OUTJ=$2; shift 2
CW=${CW:-$(cd "$(dirname "$0")/.." && pwd)/check-win-package.py}
rm -rf "$WK.prev" 2>/dev/null; [ -e "$WK" ] && mv "$WK" "$WK.prev-$(date +%s)"; mkdir -p "$WK"
i=0
for b in "$@"; do
  i=$((i+1)); d="$WK/b$i"; mkdir -p "$d/in" "$d/x"; echo "$b" > "$d/name"
  if head -c2 "$b" | grep -q MZ; then
    python3 - "$CW" "$b" "$d/in" <<'PY'
import importlib.util, struct, sys
spec = importlib.util.spec_from_file_location("cwp", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
data = open(sys.argv[2], "rb").read(); blob = m.PE(data).section(".nv_fatb"); off = k = 0
while off + 16 <= len(blob):
    magic, _v, hsz, fsz = struct.unpack_from("<IHHQ", blob, off)
    if magic != m.FATBIN_MAGIC:
        off += 8; continue
    open("%s/f%04d.fatbin" % (sys.argv[3], k), "wb").write(blob[off:off + hsz + fsz]); k += 1; off += hsz + fsz
print(sys.argv[2], "fatbins", k)
PY
  else
    cp "$b" "$d/in/"
  fi
done
docker run --rm --network none --cpuset-cpus 36-53 -v "$WK:/w" nvidia/cuda:12.8.1-devel-ubuntu22.04 bash -c '
  for d in /w/b*; do for f in $d/in/*; do mkdir -p $d/x/$(basename $f); (cd $d/x/$(basename $f) && nice -n 15 cuobjdump -xelf all $f >/dev/null 2>$d/x/$(basename $f).err || echo "cuobjdump rc=$? $f"); done; done; chmod -R a+rwX /w'
python3 - "$WK" "$OUTJ" <<'PY'
import hashlib, json, os, re, sys
wk, outp = sys.argv[1:3]; res = {}
for b in sorted(os.listdir(wk)):
    d = os.path.join(wk, b)
    if not os.path.isdir(os.path.join(d, "x")):
        continue
    name = os.path.basename(open(os.path.join(d, "name")).read().strip()); per = {}
    for root, _, fs in os.walk(os.path.join(d, "x")):
        for f in fs:
            mm = re.search(r"\.sm_(\d+a?)\.cubin$", f)
            if mm:
                per.setdefault("sm_" + mm.group(1), []).append(hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest())
    res[name] = {"archs": {a: sorted(v) for a, v in per.items()}, "counts": {a: len(v) for a, v in sorted(per.items())}}
    print(name, res[name]["counts"])
json.dump(res, open(outp, "w"), indent=1)
PY
