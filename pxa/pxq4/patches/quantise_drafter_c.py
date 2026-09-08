"""Quantise the DFlash2 drafter's per-layer o_proj / gate_proj / up_proj / down_proj to
PXQ4 (ggml type id 252) with the native encoder, and emit a vLLM-loadable checkpoint.

WHAT STAYS BF16, AND WHY (all verified in the fork's source, not assumed):
  * (2026-09-07, self_attn.{q,k,v}_proj NO LONGER stay bf16 -- see
    QUANT_SUFFIXES below.  The blocker was _build_context_kv_buffers reading
    `a.qkv_proj.weight[a.q_size:]`; the mounted qwen3_dflash.py now dequantises the loaded
    PXQ4 panels instead, so there is nothing left to block on.)
  * fc (ReplicatedLinear 25600->5120).  LinearBase sets tp_size = TP world size even for a
    replicated layer (linear.py:526), and ReplicatedLinear passes the v1 loader
    (linear.py:559-567); the sidecar refuses that combination whenever tp_size > 1
    (pxq4_vllm/linear.py:508-519).  Needs `disable_tp=True` on the layer first.
  * attention_conv/mlp_conv.kernel_projection and candidate_selector.hidden_projection are
    constructed by the fork with quant_config=None (qwen3_dflash2.py:146, :304).
  * every norm, conv base kernel, codebook and the selector tables: not linears.
"""
import ctypes, hashlib, json, os, re, struct, sys, time
import numpy as np

sys.path.insert(0, os.environ.get("PXQ4_KERNELS_SRC", "../kernels-src"))
from gguf_to_vllm import layout as L, reference as R, safetensors_io as ST
from gguf_to_vllm.encoder import NativeEncoder, bf16_to_f32

SRC = os.environ.get("SRC", "./qwen38-27b-dflash2")
DST = os.environ.get("DST", "./qwen38-27b-dflash2-pxq4c")
SO = os.environ.get("PXQ4_ENCODE_SO", "./build/libpxq4_encode.so")

# Suffixes served as PXQ4.  Matched against the checkpoint tensor name.
QUANT_SUFFIXES = (".self_attn.o_proj.weight", ".mlp.gate_proj.weight",
                  ".mlp.up_proj.weight", ".mlp.down_proj.weight",
                  # 2026-09-07: q/k/v join the PXQ4 set.  They are the last
                  # 314.6 MB of bf16 in the drafter (q [4096,5120] 41.9 MB + k/v [1024,5120]
                  # 10.5 MB each, per layer, 5 layers).  They were excluded from -pxq4 and
                  # -pxq4b for exactly ONE reason, which is now fixed in the mounted
                  # qwen3_dflash.py: _build_context_kv_buffers (:547) read
                  # `a.qkv_proj.weight[a.q_size:]` at load time and a PXQ4 layer owns no
                  # `.weight`.  That buffer is now built by dequantising the loaded PXQ4
                  # panels, so the context K/V and the incremental K/V come from ONE weight.
                  #
                  # The three tensors are loaded into ONE fused QKVParallelLinear, so plan 09
                  # sec.3.1 (a fused module is uniformly PXQ4 or not at all) requires all
                  # three together -- which is why q/k/v are added as a set, never singly.
                  # The stock loader is sufficient: PXQ4SlabParameter/PXQ4AnchorParameter are
                  # Packed*Parameters with packed_dim == output_dim, which arms the panel
                  # adjustment inside parameter.py load_qkv_weight (:175-201); pxq4_vllm/
                  # parameters.py states this explicitly.  Every offset it computes is an
                  # exact panel multiple at TP1/2/4 -- q 4096/2048/1024 -> 64/32/16 panels,
                  # k and v 1024/512/256 -> 16/8/4 -- so the silent floor-division truncation
                  # in _adjust_shard_indexes_for_packing (parameter.py:605-616) cannot bite.
                  ".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                  ".self_attn.v_proj.weight")
# 2026-09-07: `fc` joins the PXQ4 set.  It is the drafter's
# aux-hidden-state ReplicatedLinear (25600 -> 5120, 262 MB bf16, the single
# 1.43 ms k_pxa_f16_mmv_mt<3> call in the k=3 profile).  It was excluded from
# the -pxq4 checkpoint only because LinearBase gives a ReplicatedLinear
# tp_size=2 at TP2 and the sidecar refuses the v1 loader there
# (pxq4_vllm/linear.py:508-519); the mounted qwen3_dflash.py now builds it with
# disable_tp=True, which sets tp_size=1 and clears that guard.  It has NO
# suffix match here because the tensor is named exactly "fc.weight" at the top
# level of the checkpoint, so it gets its own name test.
FC_NAME = "fc.weight"
PXQ4_MODULES = ["mlp.gate_up_proj", "mlp.down_proj", "self_attn.o_proj",
                "self_attn.qkv_proj", "fc"]
IGNORE = ["linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.in_proj_ba",
          "lm_head", "embed_tokens"]

os.environ.setdefault("PXQ4_ENCODE_THREADS", "16")


def read_header(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    return f, hdr, 8 + n


def main():
    os.makedirs(DST, exist_ok=True)
    f, hdr, base = read_header(f"{SRC}/model.safetensors")
    meta = hdr.pop("__metadata__", None)
    enc = NativeEncoder(SO)

    tensors, report, tp = [], [], 0
    for name in sorted(hdr):
        e = hdr[name]
        b0, b1 = e["data_offsets"]
        shape = tuple(e["shape"])
        f.seek(base + b0)
        raw = f.read(b1 - b0)
        assert e["dtype"] == "BF16", (name, e["dtype"])
        is_fc = name == FC_NAME
        if not (is_fc or name.endswith(QUANT_SUFFIXES)):
            tensors.append(ST.Tensor(name, "BF16", shape, raw))
            continue
        N, K = shape
        L.assert_geometry(N, K)
        if is_fc:
            # Replicated (disable_tp=True): never sharded on either axis, so the
            # only requirement is whole panels / whole slabs, which
            # assert_geometry already pins.  tp_sizes=(1,) keeps the call honest
            # rather than asserting a sharding this layer will never see.
            L.assert_shardable(N, K, (1,), row_parallel=False, name=name)
        else:
            L.assert_shardable(N, K, (1, 2, 4),
                               row_parallel=name.endswith((".o_proj.weight", ".down_proj.weight")),
                               name=name)
        w = bf16_to_f32(raw, shape)
        del raw
        t0 = time.time()
        blob = enc.encode(w, None)                       # imatrix: none (as convert.py does)
        slabs, anchor = L.split_blob(blob, N, K)
        back = R.dequant_blob(blob, N, K)                # reference decode of OUR OWN bytes
        rel = float(np.linalg.norm(back - w) / np.linalg.norm(w))
        amax = float(np.abs(w).max())
        assert np.isfinite(anchor.astype(np.float32)).all(), f"{name}: non-finite anchor"
        assert rel < 0.35, f"{name}: rel-L2 {rel} - layout/table mismatch"
        stem = name[: -len(".weight")]
        tensors.append(ST.Tensor(stem + ".pxq4_slabs", "U8", slabs.shape, slabs.tobytes()))
        tensors.append(ST.Tensor(stem + ".pxq4_anchor", "F16", anchor.shape,
                                 np.ascontiguousarray(anchor).tobytes()))
        report.append((name, N, K, rel, amax, len(blob)))
        tp += 1
        print(f"  {name:52s} [{N},{K}] rel-L2 {rel:.5f} absmax {amax:.4f} "
              f"{len(blob)/1e6:7.1f} MB  {time.time()-t0:5.1f}s", flush=True)
        del w, back, blob, slabs, anchor
    f.close()

    total = ST.write_file(f"{DST}/model.safetensors", tensors,
                          metadata={"format": "pt",
                                    "pxa_provenance": "PXQ4 re-encode of qwen38-27b-dflash2 (+fc, +qkv)"})
    print(f"\nwrote {DST}/model.safetensors  {total/1e9:.3f} GB ({tp} tensors quantised)")

    cfg = json.load(open(f"{SRC}/config.json"))
    cfg["quantization_config"] = {
        "quant_method": "pxq4",
        "pxq4_version": 1,
        "tier": "core",
        "type_id": L.TYPE_ID,
        "panel_rows": L.PANEL_ROWS,
        "slab_cols": L.SLAB_COLS,
        "slab_bytes": L.SLAB_BYTES,
        "header_bytes": L.HEADER_BYTES,
        "book": [float(x) for x in R.BOOK],
        "sub": [float(x) for x in R.SUB],
        "backbone_rev": None,
        "backbone_map": None,
        "pxq4_modules": sorted(PXQ4_MODULES),
        "ignore": IGNORE,
        "modules_to_not_convert": [],
    }
    with open(f"{DST}/config.json", "w") as fh:
        json.dump(cfg, fh, indent=1)

    h = hashlib.sha256()
    with open(f"{DST}/model.safetensors", "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    sha = h.hexdigest()
    worst = max(report, key=lambda r: r[3])
    print(f"sha256(model.safetensors) = {sha}")
    print(f"max rel-L2 = {worst[3]:.5f} on {worst[0]}")
    json.dump({"sha256": sha, "bytes": total,
               "tensors": [{"name": r[0], "N": r[1], "K": r[2], "rel_l2": r[3],
                            "absmax": r[4], "pxq4_bytes": r[5]} for r in report]},
              open("./build/libpxq4_encode.so"
                   "tp4close/logs/quantise-report-c.json", "w"), indent=1)
    return sha, total, worst


if __name__ == "__main__":
    main()
