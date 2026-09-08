import sys, ctypes, time
import numpy as np
sys.path.insert(0, "/mnt/models/pxa-p2a/src")
from gguf_to_vllm import gguf_raw, layout as L, reference as R
from gguf_to_vllm.encoder import NativeEncoder, encode_and_check

SO = "/mnt/models/pxa-p2a/libpxq4_encode.so"
GGUF = "/mnt/models/pxa-models/Qwen3.8-27B-PXQ4.gguf"

lib = ctypes.CDLL(SO)
lib.pxq4_decode.restype = ctypes.c_int
lib.pxq4_decode.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                            ctypes.c_int, ctypes.c_int]
enc = NativeEncoder(SO)

g = gguf_raw.GGUFFile(GGUF)
px = [t for t in g.tensors.values() if t.type_id == gguf_raw.GGML_PXQ4 and len(t.dims) == 2]
print(f"pxq4 2-D tensors in file: {len(px)}")
shapes = {}
for t in px:
    shapes.setdefault(t.logical_shape, []).append(t.name)
for s, names in sorted(shapes.items()):
    print(f"  shape {s}: {len(names)} tensors, e.g. {names[0]}")

# pick the smallest tensor and one large one
px_sorted = sorted(px, key=lambda t: t.nbytes)
picks = [px_sorted[0], px_sorted[len(px_sorted)//2], px_sorted[-1]]
seen = set()
for ti in picks:
    if ti.name in seen: continue
    seen.add(ti.name)
    N, K = ti.logical_shape
    blob = np.frombuffer(g.raw(ti.name), dtype=np.uint8)
    print(f"\n=== {ti.name} [{N},{K}] {ti.nbytes} B ===")
    t0 = time.time()
    ref = R.dequant_blob(blob, N, K)
    print(f"  reference.dequant_blob: {time.time()-t0:.2f}s")

    # 1) native decode vs numpy reference, bit-exact?
    nat = np.empty((N, K), dtype=np.float32)
    rc = lib.pxq4_decode(blob.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                         nat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         ctypes.c_int(N), ctypes.c_int(K))
    assert rc == 0, rc
    bit_eq = np.array_equal(nat.view(np.uint32), ref.view(np.uint32))
    print(f"  native pxq4_decode vs reference.dequant: bit-identical = {bit_eq}")
    if not bit_eq:
        nd = int((nat.view(np.uint32) != ref.view(np.uint32)).sum())
        print(f"    differing float slots: {nd} / {N*K}")

    # 2) tripwire + re-encode
    t0 = time.time()
    blob2, stats = encode_and_check(enc, ref, ti.name)
    print(f"  encode_and_check: {time.time()-t0:.2f}s  wrel={stats['wrel']:.6f}  (tripwire max 0.35)")
    b2 = np.frombuffer(blob2, dtype=np.uint8)
    assert b2.size == blob.size
    same = int((b2 == blob).sum())
    print(f"  re-encode byte identity: {same}/{blob.size} = {100.0*same/blob.size:.4f}%")
    # split header vs slab bytes
    P = N // 64; S = K // 32
    pb = 128 + S * 1088
    v_orig = blob.reshape(P, pb); v_new = b2.reshape(P, pb)
    hdr_same = int((v_new[:, :128] == v_orig[:, :128]).sum()); hdr_tot = P * 128
    slabs_o = v_orig[:, 128:].reshape(P, S, 1088); slabs_n = v_new[:, 128:].reshape(P, S, 1088)
    sc_same = int((slabs_n[:, :, :64] == slabs_o[:, :, :64]).sum()); sc_tot = P * S * 64
    cd_same = int((slabs_n[:, :, 64:] == slabs_o[:, :, 64:]).sum()); cd_tot = P * S * 1024
    print(f"    anchors: {hdr_same}/{hdr_tot}  sub-scales: {sc_same}/{sc_tot}  codes: {cd_same}/{cd_tot}")
    # double round-trip error
    back2 = R.dequant_blob(blob2, N, K)
    num = float(np.linalg.norm(back2 - ref)); den = float(np.linalg.norm(ref)) or 1.0
    print(f"  D(E(D(orig))) vs D(orig): wrel = {num/den:.6f}, elementwise max abs diff = {float(np.abs(back2-ref).max()):.6g}")

# 3) synthetic fresh-weights tripwire (not on-grid input)
rng = np.random.default_rng(42)
w = rng.standard_normal((256, 1024), dtype=np.float32) * 0.02
blob3, stats = encode_and_check(enc, w, "synthetic-256x1024")
print(f"\nsynthetic gaussian [256,1024]: wrel={stats['wrel']:.6f} (expected ~0.07 for gaussian weights)")
# and with an imatrix
imx = rng.random(1024, dtype=np.float32) + 0.1
blob4, stats4 = encode_and_check(enc, w, "synthetic-imx", imatrix=imx)
print(f"synthetic with imatrix:       wrel={stats4['wrel']:.6f}")
# determinism across thread counts
import os
blob5 = enc.encode(w)
print(f"repeat encode deterministic: {blob5 == blob3}")
print("\nVALIDATION DONE")
