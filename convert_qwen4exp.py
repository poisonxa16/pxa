#!/usr/bin/env python3
"""Convert Qwen4Exp (qwen4_exp) HF safetensors to a qwen4exp GGUF.

Scope: the text model only, matching what the engine loads. The vision tower
(model.visual.*) and the multi-token-prediction module (mtp.*) are skipped and
counted; a vision projector would be a separate mmproj file.

Layout rules below were established by byte-comparison against a known-good
reference GGUF, not by inference. See --explain.

Memory: per_layer_token_embd is 128 separate checkpoint shards totalling ~95 GiB
in bf16. They are NEVER concatenated in RAM -- a streaming writer feeds them to
the output file one shard at a time. Concatenating them is what took the host
down once already; the reference implementation's own comment notes the
concatenation peaks near 300 GB of RSS.
"""

import argparse, json, os, struct, sys, glob
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------- layout rules
# v-heads: HF stores 48 value heads grouped k-head-major (16 groups of 3);
# the engine wants the transpose. gguf[i] = hf[PERM48[i]].
PERM48 = np.arange(48).reshape(16, 3).T.ravel()

# Every RMSNorm weight is stored by HF as (w - 1): the module computes x*(1+w).
# The engine applies a plain x*w, so the converter must add 1.0.
# linear_attn.norm is the sole exception -- it already stores the full weight.
NORM_PLUS_ONE = True


def bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(f: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32)
    # round-to-nearest-even on the truncated 16 bits
    rounded = (u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


class SafeTensorsStore:
    """Random access to tensors across many safetensors shards, lazily mmapped."""

    def __init__(self, folder: Path):
        self.index = {}          # tensor name -> (path, data_start, info)
        self._maps = {}
        files = sorted(folder.glob("model-*-of-*.safetensors"))
        if not files:
            raise SystemExit(f"no safetensors shards under {folder}")
        for fn in files:
            with open(fn, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                self.index[k] = (fn, 8 + n, v)
        self.n_files = len(files)

    def _mm(self, path):
        m = self._maps.get(path)
        if m is None:
            m = np.memmap(path, dtype=np.uint8, mode="r")
            self._maps[path] = m
        return m

    def has(self, name): return name in self.index

    def raw_u16(self, name):
        """bf16 tensor as a shaped uint16 view -- no copy, no dtype conversion."""
        path, base, info = self.index[name]
        if info["dtype"] != "BF16":
            raise TypeError(f"{name} is {info['dtype']}, expected BF16")
        s, e = info["data_offsets"]
        buf = self._mm(path)[base + s: base + e]
        return np.frombuffer(buf, dtype="<u2").reshape(info["shape"])

    def f32(self, name):
        return bf16_bits_to_f32(self.raw_u16(name))

    def i64(self, name):
        path, base, info = self.index[name]
        assert info["dtype"] in ("I64", "INT64"), (name, info["dtype"])
        s, e = info["data_offsets"]
        buf = self._mm(path)[base + s: base + e]
        return np.frombuffer(buf, dtype="<i8").reshape(info["shape"])

    def shape(self, name): return tuple(self.index[name][2]["shape"])


QK8_0 = 32
BLOCK_Q8_0_BYTES = 2 + QK8_0        # ggml_half d + int8 qs[32]


def quantize_q8_0(x: np.ndarray) -> np.ndarray:
    """f32 -> ggml block_q8_0 bytes. Rows must be a multiple of 32 elements.

    Matches ggml: d = max|x| / 127 in f32, stored as f16; quants use the f32 d.
    """
    b = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, QK8_0)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    idd = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    q = np.rint(b * idd[:, None]).astype(np.int8)
    out = np.empty((b.shape[0], BLOCK_Q8_0_BYTES), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out


class ShardStreamQ8:
    """per_layer_token_embd, quantized to Q8_0 one checkpoint shard at a time.

    The gather table is quantized HERE rather than by llama-quantize because
    that quantizer dequantizes a whole tensor into one f32 buffer -- 51.2B
    parameters is ~205 GB, far past this host. Q8_0 is also block-32, so it is
    legal at ne0=160; Q4_K is not (it needs ne0 % 256 == 0)."""

    def __init__(self, store, names, nbytes, row_len):
        self.store, self.names, self.nbytes, self.row_len = store, names, nbytes, row_len

    def tofile(self, fout):
        written = 0
        for i, nm in enumerate(self.names):
            a = self.store.raw_u16(nm)
            blocks = quantize_q8_0(bf16_bits_to_f32(a))
            blocks.tofile(fout)
            written += blocks.nbytes
            del a, blocks
            if (i + 1) % 16 == 0 or i + 1 == len(self.names):
                print(f"      per_layer_token_embd -> Q8_0: shard {i+1}/{len(self.names)} "
                      f"({written / 2**30:.1f} GiB)", flush=True)
        assert written == self.nbytes, (written, self.nbytes)


class Lazy:
    """Duck-types just enough of np.ndarray for GGUFWriter.write_tensors_to_file
    (.nbytes and .tofile). Nothing is read or transformed until the writer asks
    for the bytes, so exactly one tensor is resident at a time.

    Without this the writer holds every tensor in memory until the first byte is
    written -- ~257 GiB of anonymous memory for this model, on a host with 188 GB
    and no swap."""

    def __init__(self, nbytes, produce, note=None):
        self.nbytes = nbytes
        self._produce = produce
        self._note = note

    def tofile(self, fout):
        a = self._produce()
        assert a.nbytes == self.nbytes, (self._note, a.nbytes, self.nbytes)
        a.tofile(fout)


class ShardStream:
    """per_layer_token_embd: 128 checkpoint shards written back-to-back."""

    def __init__(self, store, names, nbytes):
        self.store = store
        self.names = names
        self.nbytes = nbytes

    def tofile(self, fout):
        written = 0
        for i, nm in enumerate(self.names):
            a = self.store.raw_u16(nm)
            a.tofile(fout)
            written += a.nbytes
            del a
            if (i + 1) % 16 == 0 or i + 1 == len(self.names):
                print(f"      per_layer_token_embd: shard {i+1}/{len(self.names)} "
                      f"({written / 2**30:.1f} GiB)", flush=True)
        assert written == self.nbytes, (written, self.nbytes)


def add_vocab(w, model_dir, n_vocab, tcfg_text, gguf):
    """GPT-2 style BPE vocab out of tokenizer.json."""
    tj = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tok_cfg = {}
    p = model_dir / "tokenizer_config.json"
    if p.exists():
        tok_cfg = json.loads(p.read_text(encoding="utf-8"))

    model = tj["model"]
    vocab = model["vocab"]
    id2tok = {v: k for k, v in vocab.items()}
    added = {int(a["id"]): a for a in tj.get("added_tokens", [])}

    T = gguf.TokenType
    tokens, toktypes = [], []
    n_pad = 0
    for i in range(n_vocab):
        if i in added:
            tokens.append(added[i]["content"])
            toktypes.append(T.CONTROL if added[i].get("special") else T.USER_DEFINED)
        elif i in id2tok:
            tokens.append(id2tok[i]); toktypes.append(T.NORMAL)
        else:
            tokens.append(f"[PAD{i}]"); toktypes.append(T.UNUSED); n_pad += 1

    merges_raw = model.get("merges", [])
    merges = []
    for m in merges_raw:
        merges.append(m if isinstance(m, str) else " ".join(m))

    w.add_tokenizer_model("gpt2")
    # Must name a pre-tokenizer the engine actually implements; the reference
    # build of this model uses the qwen35 split regex.
    w.add_tokenizer_pre("qwen35")
    w.add_token_list(tokens)
    w.add_token_types(toktypes)
    w.add_token_merges(merges)

    eos = tok_cfg.get("eos_token")
    if isinstance(eos, dict): eos = eos.get("content")
    eos_id = vocab.get(eos) if eos else None
    if eos_id is None:
        eos_id = next((i for i, a in added.items() if a["content"] == eos), None)
    w.add_eos_token_id(eos_id if eos_id is not None else tcfg_text["eos_token_id"])
    w.add_bos_token_id(tcfg_text["bos_token_id"])
    w.add_pad_token_id(tcfg_text.get("pad_token_id") or tcfg_text["bos_token_id"])
    w.add_add_bos_token(bool(tok_cfg.get("add_bos_token", False)))

    ct = tok_cfg.get("chat_template")
    if ct is None and (model_dir / "chat_template.jinja").exists():
        ct = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    if ct:
        w.add_chat_template(ct)

    print(f"vocab: {len(tokens)} tokens ({n_pad} pad), {len(merges)} merges, "
          f"eos={eos_id}, chat_template={'yes' if ct else 'NO'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--split-max-size", type=int, default=0,
                    help="bytes per output shard (0 = single file)")
    ap.add_argument("--kv-only", type=Path, default=None,
                    help="build only the KV header and diff it against this reference GGUF")
    ap.add_argument("--self-test-max-layer", type=int, default=None,
                    help="with --self-test, only gate layers < N (every tensor kind\n                          appears within the first 8 layers)")
    ap.add_argument("--self-test", type=Path, default=None,
                    help="materialise every tensor thunk and compare against this reference GGUF")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the full tensor plan and KV header, write nothing")
    ap.add_argument("--gguf-py", type=Path, default=None)
    ap.add_argument("--tokenizer-dir", type=Path, default=None,
                    help="where tokenizer.json lives (default: model_dir)")
    ap.add_argument("--ple-type", choices=["q8_0", "bf16"], default="q8_0",
                    help="codec for per_layer_token_embd (default q8_0: block-32,\n                          legal at ne0=160, ~54 GiB instead of 95 GiB bf16)")
    ap.add_argument("--name", default="Qwen3.8 Flash Next")
    ap.add_argument("--size-label", default="56B")
    args = ap.parse_args()

    sys.path.insert(0, str(args.gguf_py or (Path(__file__).parent / "gguf-py")))
    import gguf
    from gguf.constants import GGMLQuantizationType

    cfg = json.loads((args.model_dir / "config.json").read_text())
    t = cfg["text_config"]

    n_layer   = t["num_hidden_layers"]
    n_embd    = t["hidden_size"]
    hc        = t["hc_count"]
    hc_dim    = hc * n_embd
    n_v_heads = t["linear_num_value_heads"]
    v_head_d  = t["linear_value_head_dim"]
    layer_types = t["layer_types"]
    assert len(layer_types) == n_layer
    assert n_v_heads == 48, "PERM48 is derived for 48 value heads"

    store = SafeTensorsStore(args.model_dir)
    print(f"indexed {len(store.index)} tensors across {store.n_files} shards")

    # ---- locate the PLE module by where its tensors actually live.
    # config's ple_layer_ids is NOT the module index (it says [2]; the module is
    # at layers.1, which is also what the reference GGUF records).
    ple_layers = sorted({
        int(k.split(".")[3]) for k in store.index
        if k.startswith("model.language_model.layers.") and ".ple." in k
    })
    if len(ple_layers) != 1:
        raise SystemExit(f"expected exactly one PLE layer, found {ple_layers}")
    ple_il = ple_layers[0]
    print(f"PLE module found at layer {ple_il} "
          f"(config ple_layer_ids={t.get('ple_layer_ids')} -- not the module index)")

    P = f"model.language_model.layers.{ple_il}.ple."
    # These three are consumed as UINT64 KV entries rather than as tensors --
    # record them as claimed so the completeness check does not flag them.
    ple_const_names = [P + "ple_embedding.layer_multipliers",
                       P + "ple_embedding.ngram_heads_offsets",
                       P + "ple_embedding.ngram_heads_vocab_sizes"]
    mult, offs, vocabs = (store.i64(n) for n in ple_const_names)

    shard_names = []
    i = 0
    while store.has(P + f"ple_embedding.ngram_embedding.shard_{i}.weight"):
        shard_names.append(P + f"ple_embedding.ngram_embedding.shard_{i}.weight")
        i += 1
    n_ple_shards = len(shard_names)
    if n_ple_shards != t.get("split_ngram_parts", n_ple_shards):
        raise SystemExit(f"found {n_ple_shards} ngram shards, config says "
                         f"{t.get('split_ngram_parts')}")
    ple_rows = sum(store.shape(s)[0] for s in shard_names)
    ple_dim  = store.shape(shard_names[0])[1]
    print(f"PLE table: {n_ple_shards} shards, {ple_rows} rows x {ple_dim} "
          f"({ple_rows * ple_dim * 2 / 2**30:.1f} GiB bf16, streamed)")

    w = gguf.GGUFWriter(path=None, arch="qwen4exp",
                        split_max_size=args.split_max_size)

    # KV keys and their GGUF types are written literally, matched against a
    # known-good reference header. The bundled constants.py is older than
    # gguf_writer.py in this tree (several add_* helpers reference Keys entries
    # that do not exist), so the helpers are not safe to rely on here.
    U32, I32, F32V, STR, ARR = (gguf.GGUFValueType.UINT32, gguf.GGUFValueType.INT32,
                                gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.STRING,
                                gguf.GGUFValueType.ARRAY)
    U64 = gguf.GGUFValueType.UINT64

    def kv(key, val, vtype, sub=None):
        w.add_key_value(key, val, vtype, sub_type=sub)

    A = "qwen4exp."
    kv("general.type", "model", STR)
    kv("general.name", args.name, STR)
    kv("general.size_label", f"{t['num_experts']}x{args.size_label}", STR)

    kv(A + "block_count", n_layer, U32)
    kv(A + "context_length", t["max_position_embeddings"], U32)
    kv(A + "embedding_length", n_embd, U32)
    kv(A + "attention.head_count", t["num_attention_heads"], U32)
    kv(A + "attention.head_count_kv", t["num_key_value_heads"], U32)
    kv(A + "attention.key_length", t["head_dim"], U32)
    kv(A + "attention.value_length", t["head_dim"], U32)
    kv(A + "attention.layer_norm_rms_epsilon", t["rms_norm_eps"], F32V)
    kv(A + "expert_count", t["num_experts"], U32)
    kv(A + "expert_used_count", t["num_experts_per_tok"], U32)
    kv(A + "expert_feed_forward_length", t["moe_intermediate_size"], U32)
    kv(A + "expert_shared_feed_forward_length", t["shared_expert_intermediate_size"], U32)

    rp = t["rope_parameters"]
    kv(A + "rope.freq_base", rp["rope_theta"], F32V)
    kv(A + "rope.dimension_count", int(t["head_dim"] * rp["partial_rotary_factor"]), U32)
    kv(A + "rope.dimension_sections", list(rp["mrope_section"]) + [0], ARR, I32)

    kv(A + "ssm.conv_kernel", t["linear_conv_kernel_dim"], U32)
    kv(A + "ssm.state_size", t["linear_key_head_dim"], U32)
    kv(A + "ssm.group_count", t["linear_num_key_heads"], U32)
    kv(A + "ssm.time_step_rank", n_v_heads, U32)
    kv(A + "ssm.inner_size", n_v_heads * v_head_d, U32)

    kv(A + "full_attention_interval", t["full_attention_interval"], U32)
    kv(A + "hyper_connection.count", hc, U32)
    kv(A + "hyper_connection.low_rank", t["hc_lowrank"], U32)

    kv(A + "attention.indexer.head_count", t["indexer_n_heads"], U32)
    kv(A + "attention.indexer.key_length", t["indexer_head_dim"], U32)
    kv(A + "attention.indexer.top_k", t["indexer_budget"], U32)
    kv(A + "attention.compress_ratios",
       [t["indexer_compress_ratio"] if lt == "full_attention" else 0 for lt in layer_types],
       ARR, I32)

    kv(A + "embedding_length_per_layer_input", ple_dim, U32)
    kv(A + "ple.layers", [ple_il], ARR, I32)
    kv(A + "ple.ngram_size", t["ngram_size"], U32)
    kv(A + "ple.heads_per_ngram", t["heads_per_ngram"], U32)
    kv(A + "ple.conv_kernel", t["ple_conv_kernel_size"], U32)
    kv(A + "ple.eos_token_id", t["eos_token_id"], U32)
    kv(A + "ple.image_token_id", cfg["image_token_id"], U32)
    # UINT64 arrays, read straight from the checkpoint -- not derived. The
    # engine's loader reads these as uint64; both get_arr overloads used to
    # throw on that type, which is why this arch could not load at all.
    kv(A + "ple.layer_multipliers", [int(x) for x in mult], ARR, U64)
    kv(A + "ple.head_offsets", [int(x) for x in offs], ARR, U64)
    kv(A + "ple.head_vocab_sizes", [int(x) for x in vocabs], ARR, U64)

    # ------------------------------------------------------------------ vocab
    add_vocab(w, args.tokenizer_dir or args.model_dir, t["vocab_size"], t, gguf)

    w.add_quantization_version(2)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)

    if args.kv_only is not None:
        ref = gguf.GGUFReader(str(args.kv_only))
        mine = {}
        for d in w.kv_data:
            for k, v in d.items():
                mine[k] = v.value
        skip = ("general.quantization_version", "general.file_type",
                "general.description", "general.quantized_by", "general.repo_url",
                "general.tags", "split.", "quantize.imatrix", "general.sampling")
        same = diff = missing = 0
        print("\n=== KV header vs %s ===" % args.kv_only.name)
        for k, f in ref.fields.items():
            if k.startswith("GGUF.") or any(k.startswith(x) for x in skip):
                continue
            if k.startswith("tokenizer.") and k not in (
                    "tokenizer.ggml.model", "tokenizer.ggml.pre",
                    "tokenizer.ggml.eos_token_id", "tokenizer.ggml.bos_token_id",
                    "tokenizer.ggml.padding_token_id", "tokenizer.ggml.add_bos_token"):
                continue
            if k not in mine:
                print("  MISSING   %s" % k); missing += 1; continue
            rv = f.contents()
            mv = mine[k]
            if isinstance(rv, (list, tuple)) or isinstance(mv, (list, tuple)):
                a = [int(x) if isinstance(x, (int, np.integer)) else x for x in (rv or [])]
                b = [int(x) if isinstance(x, (int, np.integer)) else x for x in (mv or [])]
                eq = a == b
            elif isinstance(rv, float) or isinstance(mv, float):
                eq = abs(float(rv) - float(mv)) <= 1e-9 * max(1.0, abs(float(rv)))
            else:
                eq = str(rv) == str(mv)
            if eq:
                same += 1
            else:
                diff += 1
                print("  DIFFERS   %-46s ref=%s  mine=%s" % (k, str(rv)[:40], str(mv)[:40]))
        extra = [k for k in mine if k not in ref.fields and not any(k.startswith(x) for x in skip)]
        for k in extra:
            print("  EXTRA     %-46s mine=%s" % (k, str(mine[k])[:40]))
        print("\n  matched %d, differ %d, missing %d, extra %d" % (same, diff, missing, len(extra)))
        return

    # ------------------------------------------------------------- tensors
    stats = {"f32": 0, "bf16": 0, "skipped_visual": 0, "skipped_mtp": 0}
    claimed = set(ple_const_names)   # consumed as KV, not as tensors

    def add_f32(name, shape, produce):
        n = int(np.prod(shape)) * 4
        w.add_tensor_info(name, tuple(shape), np.dtype(np.float32), n,
                          raw_dtype=GGMLQuantizationType.F32)
        w.tensors[-1][name].tensor = Lazy(
            n, lambda p=produce: np.ascontiguousarray(p(), dtype=np.float32), name)
        stats["f32"] += 1

    def add_bf16(name, shape, produce):
        n = int(np.prod(shape)) * 2
        # NB: write_ti_data_to_file reverses shape itself -- pass numpy order.
        w.add_tensor_info(name, tuple(shape), np.dtype(np.uint16), n,
                          raw_dtype=GGMLQuantizationType.BF16)
        w.tensors[-1][name].tensor = Lazy(
            n, lambda p=produce: np.ascontiguousarray(p()), name)
        stats["bf16"] += 1

    def src(name):
        claimed.add(name)
        return name

    def norm(name, hf_name, plus_one=True):
        h = src(hf_name)
        off = 1.0 if (plus_one and NORM_PLUS_ONE) else 0.0
        add_f32(name, store.shape(h),
                lambda h=h, off=off: store.f32(h).astype(np.float32) + off)

    LM = "model.language_model."
    sh = store.shape

    # globals
    for cand in (LM + "embed_tokens.weight", "model.embed_tokens.weight"):
        if store.has(cand):
            c = src(cand)
            add_bf16("token_embd.weight", sh(c), lambda c=c: store.raw_u16(c)); break
    else:
        if args.self_test is None:
            raise SystemExit("embed_tokens not found")
        print("  (self-test: embed_tokens not downloaded yet, skipping)")
    for cand in ("lm_head.weight", LM + "lm_head.weight"):
        if store.has(cand):
            c = src(cand)
            add_bf16("output.weight", sh(c), lambda c=c: store.raw_u16(c)); break
    else:
        if args.self_test is None:
            raise SystemExit("lm_head not found")
        print("  (self-test: lm_head not downloaded yet, skipping)")

    HCM = LM + "hyper_connection_mixer."
    norm("output_hc_norm.weight", HCM + "hc_norm.weight")
    for gg, hf_n in (("output_hc_down.weight", "input_mix_weight_down.weight"),
                     ("output_hc_up.weight",   "input_mix_weight_up.weight")):
        c = src(HCM + hf_n)
        add_bf16(gg, sh(c), lambda c=c: store.raw_u16(c))

    # the streamed PLE table
    _pn = [src(x) for x in shard_names]
    if args.ple_type == "q8_0":
        assert ple_dim % QK8_0 == 0, (ple_dim, QK8_0)
        ple_nbytes = (ple_rows * ple_dim // QK8_0) * BLOCK_Q8_0_BYTES
        w.add_tensor_info("per_layer_token_embd.weight", (ple_rows, ple_dim),
                          np.dtype(np.float32), ple_nbytes,
                          raw_dtype=GGMLQuantizationType.Q8_0)
        w.tensors[-1]["per_layer_token_embd.weight"].tensor = ShardStreamQ8(
            store, _pn, ple_nbytes, ple_dim)
    else:
        ple_nbytes = ple_rows * ple_dim * 2
        w.add_tensor_info("per_layer_token_embd.weight", (ple_rows, ple_dim),
                          np.dtype(np.uint16), ple_nbytes,
                          raw_dtype=GGMLQuantizationType.BF16)
        w.tensors[-1]["per_layer_token_embd.weight"].tensor = ShardStream(
            store, _pn, ple_nbytes)
    print(f"per_layer_token_embd codec: {args.ple_type} "
          f"({ple_nbytes / 2**30:.1f} GiB)")
    stats["bf16"] += 1

    n_kv_in = t["linear_num_key_heads"] * t["linear_key_head_dim"]
    conv_k  = t["linear_conv_kernel_dim"]

    def perm_v(a, n_lead):
        """permute the value-head block of a row-major [n_lead + 48*128, ...]."""
        head, tail = a[:n_lead], a[n_lead:]
        tail = tail.reshape(n_v_heads, v_head_d, -1)[PERM48].reshape(tail.shape[0], -1)
        return np.concatenate([head, tail.reshape(tail.shape[0], *a.shape[1:])], 0)

    skipped_layers = []
    for il in range(n_layer):
        L = f"{LM}layers.{il}."
        B = f"blk.{il}."

        if args.self_test is not None:
            # only gate layers whose source tensors are all actually on disk
            need = [L + f"{pfx}.{leaf}"
                    for pfx in ("attn_hyper_connection", "mlp_hyper_connection")
                    for leaf in ("hc_norm.weight", "input_mix_weight_down.weight",
                                 "input_mix_weight_up.weight", "block_inject_weight.weight")]
            need += [L + "mlp." + x for x in
                     ("gate.weight", "shared_expert_gate.weight",
                      "experts.gate_up_proj", "experts.down_proj",
                      "shared_expert.gate_proj.weight", "shared_expert.up_proj.weight",
                      "shared_expert.down_proj.weight")]
            if layer_types[il] == "linear_attention":
                need += [L + "linear_attn." + x for x in
                         ("in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight",
                          "A_log", "dt_bias", "in_proj_a.weight", "in_proj_b.weight",
                          "conv1d.weight", "norm.weight")]
            else:
                need += [L + "self_attn." + x for x in
                         ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                          "o_proj.weight", "q_norm.weight", "k_norm.weight",
                          "indexer.index_qk_proj.weight",
                          "indexer.q_layernorm.weight", "indexer.k_layernorm.weight")]
            if il == ple_il:
                need += [P + x for x in ("key_proj.weight", "value_proj.weight",
                                         "conv1d.weight", "norm_conv.weight",
                                         "norm_key.weight", "norm_query.weight")]
            if (args.self_test_max_layer is not None
                    and il >= args.self_test_max_layer) \
                    or not all(store.has(x) for x in need):
                skipped_layers.append(il)
                continue

        for hf_pfx, gg_pfx in (("attn_hyper_connection.", "hc_attn_"),
                               ("mlp_hyper_connection.",  "hc_ffn_")):
            norm(B + gg_pfx + "norm.weight", L + hf_pfx + "hc_norm.weight")
            for gg_s, hf_s in (("down", "input_mix_weight_down.weight"),
                               ("up",   "input_mix_weight_up.weight")):
                c = src(L + hf_pfx + hf_s)
                add_bf16(B + gg_pfx + gg_s + ".weight", sh(c), lambda c=c: store.raw_u16(c))
            c = src(L + hf_pfx + "block_inject_weight.weight")
            add_f32(B + gg_pfx + "inject.weight", sh(c), lambda c=c: store.f32(c))

        # ---- MoE
        c = src(L + "mlp.gate.weight")
        add_f32(B + "ffn_gate_inp.weight", sh(c), lambda c=c: store.f32(c))
        c = src(L + "mlp.shared_expert_gate.weight")
        add_f32(B + "ffn_gate_inp_shexp.weight", (n_embd,),
                lambda c=c: store.f32(c).reshape(-1))

        gu = src(L + "mlp.experts.gate_up_proj")
        n_e, n_gu, n_in = sh(gu)
        half = n_gu // 2
        add_bf16(B + "ffn_gate_exps.weight", (n_e, half, n_in),
                 lambda c=gu, h=half: store.raw_u16(c)[:, :h, :])
        add_bf16(B + "ffn_up_exps.weight", (n_e, half, n_in),
                 lambda c=gu, h=half: store.raw_u16(c)[:, h:, :])
        c = src(L + "mlp.experts.down_proj")
        add_bf16(B + "ffn_down_exps.weight", sh(c), lambda c=c: store.raw_u16(c))
        for nm in ("gate", "up", "down"):
            c = src(L + f"mlp.shared_expert.{nm}_proj.weight")
            add_bf16(B + f"ffn_{nm}_shexp.weight", sh(c), lambda c=c: store.raw_u16(c))

        if layer_types[il] == "linear_attention":
            A = L + "linear_attn."
            c = src(A + "in_proj_qkv.weight")
            add_bf16(B + "attn_qkv.weight", sh(c),
                     lambda c=c: perm_v(store.raw_u16(c), 2 * n_kv_in))
            c = src(A + "in_proj_z.weight")
            add_bf16(B + "attn_gate.weight", sh(c),
                     lambda c=c: store.raw_u16(c).reshape(n_v_heads, v_head_d, -1)[PERM48]
                                      .reshape(n_v_heads * v_head_d, -1))
            c = src(A + "out_proj.weight")
            add_bf16(B + "ssm_out.weight", sh(c),
                     lambda c=c: store.raw_u16(c).reshape(n_embd, n_v_heads, v_head_d)[:, PERM48, :]
                                      .reshape(n_embd, -1))
            # A_log -> -exp(A_log), then permute
            c = src(A + "A_log")
            add_f32(B + "ssm_a", sh(c),
                    lambda c=c: (-np.exp(store.f32(c).astype(np.float32)))[PERM48])
            c = src(A + "dt_bias")
            add_f32(B + "ssm_dt.bias", sh(c),
                    lambda c=c: store.f32(c).astype(np.float32)[PERM48])
            for gg, hf_n in (("ssm_alpha", "in_proj_a"), ("ssm_beta", "in_proj_b")):
                c = src(A + hf_n + ".weight")
                add_f32(B + gg + ".weight", sh(c), lambda c=c: store.f32(c)[PERM48])
            c = src(A + "conv1d.weight")
            add_f32(B + "ssm_conv1d.weight", (sh(c)[0], conv_k),
                    lambda c=c: perm_v(store.f32(c).reshape(-1, conv_k), 2 * n_kv_in))
            # the sole norm that is NOT offset by one
            norm(B + "ssm_norm.weight", A + "norm.weight", plus_one=False)
        else:
            S = L + "self_attn."
            for gg, hf_n in (("attn_q", "q_proj"), ("attn_k", "k_proj"),
                             ("attn_v", "v_proj"), ("attn_output", "o_proj")):
                c = src(S + hf_n + ".weight")
                add_bf16(B + gg + ".weight", sh(c), lambda c=c: store.raw_u16(c))
            norm(B + "attn_q_norm.weight", S + "q_norm.weight")
            norm(B + "attn_k_norm.weight", S + "k_norm.weight")
            c = src(S + "indexer.index_qk_proj.weight")
            n_iq = t["indexer_n_heads"] * t["indexer_head_dim"]
            n_ik = sh(c)[0] - n_iq
            add_bf16(B + "indexer.q_proj.weight", (n_iq, sh(c)[1]),
                     lambda c=c, n=n_iq: store.raw_u16(c)[:n])
            add_bf16(B + "indexer.k_proj.weight", (n_ik, sh(c)[1]),
                     lambda c=c, n=n_iq: store.raw_u16(c)[n:])
            norm(B + "indexer.q_norm.weight", S + "indexer.q_layernorm.weight")
            norm(B + "indexer.k_norm.weight", S + "indexer.k_layernorm.weight")

        if il == ple_il:
            for gg, hf_n in (("ple_key", "key_proj"), ("ple_value", "value_proj")):
                c = src(P + hf_n + ".weight")
                add_bf16(B + gg + ".weight", sh(c), lambda c=c: store.raw_u16(c))
            c = src(P + "conv1d.weight")
            add_f32(B + "ple_conv1d.weight", (sh(c)[0], t["ple_conv_kernel_size"]),
                    lambda c=c, k=t["ple_conv_kernel_size"]: store.f32(c).reshape(-1, k))
            for nm in ("conv", "key", "query"):
                norm(B + f"ple_norm_{nm}.weight", P + f"norm_{nm}.weight")

    if args.self_test is not None:
        from gguf.quants import dequantize
        ref = gguf.GGUFReader(str(args.self_test))
        RT = {tt.name: tt for tt in GGMLQuantizationType}
        rmap = {x.name: x for x in ref.tensors}
        # loose bounds: pure quantisation noise for the reference's own type.
        TOL = {"F32": 1e-6, "BF16": 1e-6, "Q8_0": 0.02, "Q6_K": 0.05, "Q5_K": 0.08,
               "IQ4_NL": 0.15, "IQ4_XS": 0.15, "IQ2_XXS": 0.9, "IQ1_S": 0.9, "Q4_K": 0.10}
        ok = bad = skip = 0
        checked = []
        for shard in w.tensors:
            for name, ti in shard.items():
                rt = rmap.get(name)
                if rt is None or name == "per_layer_token_embd.weight":
                    skip += 1; continue
                mine = ti.tensor._produce() if isinstance(ti.tensor, Lazy) else None
                if mine is None:
                    skip += 1; continue
                if mine.dtype == np.uint16:
                    mine = bf16_bits_to_f32(mine)
                mine = np.asarray(mine, dtype=np.float32).ravel()
                tyn = rt.tensor_type.name
                # For the 3-D expert stacks, compare ONE expert: dequantising
                # 838M IQ1_S elements per tensor buys no extra signal and takes
                # ~500x longer. The expert axis is slowest-moving, so expert 0
                # is a contiguous prefix of both the raw blocks and my array.
                rd, n_exp = rt.data, 1
                if len(ti.shape) == 3:
                    n_exp = ti.shape[0]
                    if rd.shape[0] % n_exp == 0:
                        rd = rd[: rd.shape[0] // n_exp]
                        mine = mine[: mine.size // n_exp]
                    else:
                        n_exp = 1
                if rt.tensor_type == GGMLQuantizationType.F32:
                    r = np.array(rd, dtype=np.float32).ravel()
                elif rt.tensor_type == GGMLQuantizationType.BF16:
                    r = bf16_bits_to_f32(np.frombuffer(rd.tobytes(), dtype="<u2")).ravel()
                else:
                    r = dequantize(rd, rt.tensor_type).astype(np.float32).ravel()
                if r.size != mine.size:
                    print(f"  SIZE  {name:<34} ref {r.size} vs mine {mine.size}")
                    bad += 1; continue
                rel = float(np.abs(mine - r).mean()) / (float(np.abs(r).mean()) + 1e-12)
                tol = TOL.get(tyn, 0.2)
                if rel <= tol:
                    ok += 1
                else:
                    bad += 1
                    print(f"  FAIL  {name:<34} [{tyn}] rel_err={rel:.4f} > {tol}")
                checked.append((name, tyn, rel))
                if len(checked) % 25 == 0:
                    print(f"  ...{len(checked)} tensors checked, worst so far "
                          f"{max(checked, key=lambda x: x[2])[2]:.4f}", flush=True)
                del mine, r
        print(f"\n=== self-test vs {args.self_test.name} ===")
        if skipped_layers:
            print(f"  layers not yet on disk, not gated: {len(skipped_layers)} "
                  f"({skipped_layers[:6]}{'...' if len(skipped_layers)>6 else ''})")
        worst = sorted(checked, key=lambda x: -x[2])[:8]
        for n, tyn, rel in worst:
            print(f"  worst: {n:<36} [{tyn}] rel_err={rel:.4f}")
        print(f"  passed {ok}, failed {bad}, skipped {skip} (not in reference / streamed)")
        if bad:
            raise SystemExit("self-test FAILED")
        return

    # ------------------------------------------------------- coverage report
    for k in store.index:
        if k.startswith("model.visual."):
            stats["skipped_visual"] += 1
        elif k.startswith("mtp."):
            stats["skipped_mtp"] += 1
    unclaimed = [k for k in store.index
                 if k not in claimed
                 and not k.startswith("model.visual.")
                 and not k.startswith("mtp.")]
    n_out = sum(len(d) for d in w.tensors)
    print(f"\ntensors written : {n_out}  (bf16 {stats['bf16']}, f32 {stats['f32']})")
    print(f"skipped         : {stats['skipped_visual']} vision (separate mmproj), "
          f"{stats['skipped_mtp']} mtp (not modelled by the engine)")
    if unclaimed:
        print(f"\nUNCONSUMED text tensors ({len(unclaimed)}) -- these would be "
              f"silently dropped, refusing:")
        for k in sorted(unclaimed)[:40]:
            print(f"   {k}  {store.shape(k)}")
        raise SystemExit("refusing to write an incomplete model")

    if args.dry_run:
        print("\n--dry-run: plan built, KV header complete, nothing written")
        return

    w.write_header_to_file(path=args.outfile)
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    print(f"\nwrote {args.outfile}")


if __name__ == "__main__":
    main()
