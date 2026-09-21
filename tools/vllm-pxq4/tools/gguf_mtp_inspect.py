import struct, sys
p = sys.argv[1]
f = open(p, "rb")
magic, ver, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
assert magic == 0x46554747, hex(magic)
def rd_str():
    (n,) = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")
def rd_val(t):
    T = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}
    if t == 8: return rd_str()
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4)); (n,) = struct.unpack("<Q", f.read(8))
        return [rd_val(et) for _ in range(n)]
    s = T[t]; sz = struct.calcsize(s); return struct.unpack(s, f.read(sz))[0]
kv = {}
for _ in range(n_kv):
    k = rd_str(); (t,) = struct.unpack("<I", f.read(4)); kv[k] = rd_val(t)
print("block_count", {k: v for k, v in kv.items() if "block_count" in k or "nextn" in k})
names = []
for _ in range(n_tensors):
    nm = rd_str(); (nd,) = struct.unpack("<I", f.read(4))
    f.read(8 * nd); f.read(4); f.read(8)
    names.append(nm)
mtp = [n for n in names if "nextn" in n or n.startswith("blk.64.")]
print("total tensors", len(names), "| blk.64/nextn tensors:", len(mtp))
for n in sorted(mtp): print("  ", n)
