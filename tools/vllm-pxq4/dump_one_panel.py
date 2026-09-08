# Pull ONE 64-row panel of one PXQ4 tensor out of the artifact and print it base64.
# Read-only; touches nothing but the file. Runs on the DGX (python3 + stdlib only).
import base64, struct, sys

PATH = "/mnt/models/pxa-models/Qwen3.8-27B-PXQ4.gguf"
WANT = sys.argv[1] if len(sys.argv) > 1 else "blk.0.attn_gate.weight"

f = open(PATH, "rb")
def rd(n): return f.read(n)
def u32(): return struct.unpack("<I", rd(4))[0]
def u64(): return struct.unpack("<Q", rd(8))[0]
def s():
    n = u64(); return rd(n).decode("utf-8", "replace")

def val(t):
    if t == 0: return struct.unpack("<b", rd(1))[0]
    if t == 1: return struct.unpack("<B", rd(1))[0]
    if t == 2: return struct.unpack("<h", rd(2))[0]
    if t == 3: return struct.unpack("<H", rd(2))[0]
    if t == 4: return struct.unpack("<i", rd(4))[0]
    if t == 5: return struct.unpack("<I", rd(4))[0]
    if t == 6: return struct.unpack("<f", rd(4))[0]
    if t == 7: return struct.unpack("<?", rd(1))[0]
    if t == 8: return s()
    if t == 9:
        et = u32(); n = u64(); return [val(et) for _ in range(n)]
    if t == 10: return struct.unpack("<q", rd(8))[0]
    if t == 11: return struct.unpack("<Q", rd(8))[0]
    if t == 12: return struct.unpack("<d", rd(8))[0]
    raise ValueError(t)

assert rd(4) == b"GGUF"
ver = u32(); nt = u64(); nkv = u64()
align = 32
for _ in range(nkv):
    k = s(); t = u32(); v = val(t)
    if k == "general.alignment": align = v

tensors = []
for _ in range(nt):
    name = s(); nd = u32(); ne = [u64() for _ in range(nd)]; ty = u32(); off = u64()
    tensors.append((name, ne, ty, off))
pos = f.tell()
data_start = (pos + align - 1) // align * align

for name, ne, ty, off in tensors:
    if name == WANT:
        K, R = ne[0], ne[1]
        panel = 128 + (K // 32) * 1088
        f.seek(data_start + off)
        blob = f.read(panel)
        print("META", name, "type", ty, "K", K, "R", R, "panel_bytes", panel)
        print("B64", base64.b64encode(blob).decode())
        break
else:
    print("NOT FOUND", WANT)
