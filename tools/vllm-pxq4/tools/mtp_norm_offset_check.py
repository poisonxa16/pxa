"""Compare GGUF blk.64 norm means against p2a-nf's mtp.* norm means.

If the converter applied the gemma -1 offset to a tensor, the safetensors mean
will be the GGUF mean minus 1.
"""
import json, os, struct, sys
import numpy as np
from safetensors import safe_open

GGUF = sys.argv[1]; ART = sys.argv[2]

GGML_T = {0: ("f4", 4, 1), 1: ("f2", 2, 1)}   # F32, F16 only (norms are one of these)

f = open(GGUF, "rb")
magic, ver, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
def rd_str():
    (n,) = struct.unpack("<Q", f.read(8)); return f.read(n).decode()
def rd_val(t):
    T = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}
    if t == 8: return rd_str()
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4)); (n,) = struct.unpack("<Q", f.read(8))
        return [rd_val(et) for _ in range(n)]
    s = T[t]; return struct.unpack(s, f.read(struct.calcsize(s)))[0]
kv = {}
for _ in range(n_kv):
    k = rd_str(); (t,) = struct.unpack("<I", f.read(4)); kv[k] = rd_val(t)
align = int(kv.get("general.alignment", 32))
infos = []
for _ in range(n_tensors):
    nm = rd_str(); (nd,) = struct.unpack("<I", f.read(4))
    dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
    (tt,) = struct.unpack("<I", f.read(4)); (off,) = struct.unpack("<Q", f.read(8))
    infos.append((nm, dims, tt, off))
base = f.tell()
base = (base + align - 1) // align * align

gg = {}
for nm, dims, tt, off in infos:
    if not nm.startswith("blk.64.") or "norm" not in nm:
        continue
    if tt not in GGML_T:
        gg[nm] = None; continue
    dt, esz, _ = GGML_T[tt]
    n = int(np.prod(dims))
    f.seek(base + off)
    a = np.frombuffer(f.read(n * esz), dtype=np.dtype(dt)).astype(np.float64)
    gg[nm] = a.mean()

idx = json.load(open(os.path.join(ART, "model.safetensors.index.json")))["weight_map"]
st = {}
for k, shard in idx.items():
    if not k.startswith("mtp.") or "norm" not in k:
        continue
    p = os.path.join(ART, shard)
    if not os.path.exists(p):
        st[k] = None; continue
    with safe_open(p, framework="np") as h:
        st[k] = np.asarray(h.get_tensor(k)).astype(np.float64).mean()

PAIRS = [
    ("blk.64.attn_norm.weight",            "mtp.layers.0.input_layernorm.weight"),
    ("blk.64.post_attention_norm.weight",  "mtp.layers.0.post_attention_layernorm.weight"),
    ("blk.64.attn_q_norm.weight",          "mtp.layers.0.self_attn.q_norm.weight"),
    ("blk.64.attn_k_norm.weight",          "mtp.layers.0.self_attn.k_norm.weight"),
    ("blk.64.nextn.enorm.weight",          "mtp.pre_fc_norm_embedding.weight"),
    ("blk.64.nextn.hnorm.weight",          "mtp.pre_fc_norm_hidden.weight"),
    ("blk.64.nextn.shared_head_norm.weight","mtp.norm.weight"),
]
print(f"{'ggml tensor':38s} {'gguf mean':>11s} {'hf mean':>11s}  offset applied?")
for g, s in PAIRS:
    gv, sv = gg.get(g), st.get(s)
    if gv is None or sv is None:
        print(f"{g:38s} {'?':>11s} {'?':>11s}  (missing)"); continue
    d = sv - gv
    verdict = "YES (-1)" if abs(d + 1) < 0.02 else ("no (verbatim)" if abs(d) < 0.02 else f"?? delta={d:+.4f}")
    print(f"{g:38s} {gv:11.5f} {sv:11.5f}  {verdict}")
