#!/usr/bin/env python3
"""PXA_GLM5NEXT: emit a tiny random-weight F16/F32 `glm5next` GGUF.

The point is the IDENTITY GATE: a model small enough to run on a CPU in a second, but with
every structural feature of GLM-5.3-Flash switched on, so that our engine and the reference
build (llama.cpp PR #27773) can be run on the same prompt and their logits compared. If the
two agree, the port's arithmetic is right; if they diverge, the divergence is in a graph this
small enough to bisect by hand.

Everything the real model has, in miniature:

  * 8 trunk blocks on 4 mHC residual streams, KDA on 0,1,2,4,5,6 and MLA+DSA on 3 and 7
  * NoPE MLA (rope.dimension_count = 0) with the q/kv LoRA pair and the k_b/v_b absorb pair
  * the k-pool indexer: kpool 4, top_k 8 (so at most 2 pools per token -- the selection is
    genuinely narrower than the context, which is what makes the decode gather path fire)
  * 3 leading dense blocks, then MoE: 8 routed experts, 2 used, 1 shared, sigmoid noaux_tc
    with exp_probs_b, expert_weights_scale/_norm, clamped SwiGLU
  * the KDA gate pair, the short causal conv on each of q/k/v, and a FINITE
    kda.gate_lower_bound so the bounded-sigmoid gate is the one exercised

The GGUF is written by hand (struct + numpy) rather than through gguf-py so the generator does
not depend on which tree's gguf-py is importable, and so the ggml `ne` order of every tensor is
explicit at the call site rather than implied by a numpy transpose.

Usage:  python3 tests/gen_tiny_glm5next.py out.gguf [--seed 1234]
"""

import argparse
import struct
import sys

import numpy as np

# --- GGUF constants -------------------------------------------------------------------

GGUF_MAGIC   = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT    = 32

T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64 = 4, 5, 6, 7, 8, 9, 10

GGML_TYPE_F32 = 0


class Writer:
    def __init__(self):
        self.kv = []        # (key, encoded value bytes)
        self.tensors = []   # (name, ne list, bytes)

    # -- kv encoders
    @staticmethod
    def _str(s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def u32(self, k, v):   self.kv.append((k, struct.pack("<I", T_UINT32) + struct.pack("<I", v)))
    def i32(self, k, v):   self.kv.append((k, struct.pack("<I", T_INT32) + struct.pack("<i", v)))
    def f32(self, k, v):   self.kv.append((k, struct.pack("<I", T_FLOAT32) + struct.pack("<f", v)))
    def bl(self, k, v):    self.kv.append((k, struct.pack("<I", T_BOOL) + struct.pack("<B", 1 if v else 0)))
    def st(self, k, v):    self.kv.append((k, struct.pack("<I", T_STRING) + self._str(v)))

    def arr_u32(self, k, vs):
        body = struct.pack("<I", T_ARRAY) + struct.pack("<I", T_UINT32) + struct.pack("<Q", len(vs))
        body += b"".join(struct.pack("<I", int(v)) for v in vs)
        self.kv.append((k, body))

    def arr_i32(self, k, vs):
        body = struct.pack("<I", T_ARRAY) + struct.pack("<I", T_INT32) + struct.pack("<Q", len(vs))
        body += b"".join(struct.pack("<i", int(v)) for v in vs)
        self.kv.append((k, body))

    def arr_f32(self, k, vs):
        body = struct.pack("<I", T_ARRAY) + struct.pack("<I", T_FLOAT32) + struct.pack("<Q", len(vs))
        body += b"".join(struct.pack("<f", float(v)) for v in vs)
        self.kv.append((k, body))

    def arr_str(self, k, vs):
        body = struct.pack("<I", T_ARRAY) + struct.pack("<I", T_STRING) + struct.pack("<Q", len(vs))
        body += b"".join(self._str(v) for v in vs)
        self.kv.append((k, body))

    # -- tensors. `ne` is in GGML order (ne[0] is the fastest-varying axis), and `data` is
    # laid out with ne[0] contiguous, i.e. numpy shape == reversed(ne).
    def tensor(self, name, ne, data):
        a = np.ascontiguousarray(data, dtype=np.float32)
        assert list(a.shape) == list(reversed(ne)), f"{name}: {a.shape} vs ne {ne}"
        self.tensors.append((name, list(ne), a.tobytes()))

    def write(self, path):
        head = bytearray()
        head += GGUF_MAGIC
        head += struct.pack("<I", GGUF_VERSION)
        head += struct.pack("<Q", len(self.tensors))
        head += struct.pack("<Q", len(self.kv))
        for k, v in self.kv:
            head += self._str(k) + v

        offset = 0
        infos = bytearray()
        for name, ne, data in self.tensors:
            infos += self._str(name)
            infos += struct.pack("<I", len(ne))
            for d in ne:
                infos += struct.pack("<Q", d)
            infos += struct.pack("<I", GGML_TYPE_F32)
            infos += struct.pack("<Q", offset)
            offset += (len(data) + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

        body = bytearray(head) + infos
        pad = (-len(body)) % ALIGNMENT
        body += b"\x00" * pad
        for _, _, data in self.tensors:
            body += data
            body += b"\x00" * ((-len(data)) % ALIGNMENT)

        with open(path, "wb") as f:
            f.write(body)
        return len(body)


# --- the model ------------------------------------------------------------------------

def build(path, seed, idx_topk=8):
    rng = np.random.default_rng(seed)

    ARCH   = "glm5next"
    N_LAYER = 8
    MLA_AT  = {3, 7}            # attention.head_count_kv == 1 there, 0 (KDA) elsewhere
    N_EMBD  = 256
    N_HEAD  = 4
    N_VOCAB = 512
    N_CTX   = 4096

    # MLA
    Q_LORA   = 64
    KV_LORA  = 32
    HEAD_MLA = 16               # the real per-head k/v width (the *_mla keys)

    # KDA
    KDA_HEAD = 64
    D_INNER  = KDA_HEAD * N_HEAD   # 256
    D_CONV   = 4
    GATE_LB  = -5.0

    # DSA indexer
    IDX_HEAD = 4
    IDX_DIM  = 32          # a multiple of the CUDA warp: this tree's norm kernel requires it
    IDX_TOPK = idx_topk         # 8 -> at most 2 pools per token; 2048 mirrors the real
                                # model, where n_top saturates at the 64-pool pad floor
    KPOOL    = 4

    # MoE
    N_EXPERT, N_USED, N_SHARED = 8, 2, 1
    N_FF_EXP = 64
    N_FF     = 128
    N_DENSE  = 3

    HC = 4
    HC_MIX = (2 + HC) * HC      # 24

    w = Writer()

    # --- general
    w.st("general.architecture", ARCH)
    w.st("general.name", "tiny-glm5next")
    w.u32("general.file_type", 0)              # ALL_F32
    w.u32("general.quantization_version", 2)

    # --- geometry
    w.u32(f"{ARCH}.block_count", N_LAYER)
    w.u32(f"{ARCH}.context_length", N_CTX)
    w.u32(f"{ARCH}.embedding_length", N_EMBD)
    w.u32(f"{ARCH}.feed_forward_length", N_FF)
    w.u32(f"{ARCH}.attention.head_count", N_HEAD)
    # 0 = KDA (recurrent), 1 = MLA. This per-layer array IS the layer-regime table.
    w.arr_u32(f"{ARCH}.attention.head_count_kv", [1 if i in MLA_AT else 0 for i in range(N_LAYER)])
    w.f32(f"{ARCH}.attention.layer_norm_rms_epsilon", 1e-5)
    w.f32(f"{ARCH}.attention.layer_norm_epsilon", 1e-5)
    w.u32(f"{ARCH}.vocab_size", N_VOCAB)
    w.u32(f"{ARCH}.nextn_predict_layers", 0)

    # NoPE: there is no rope half at all
    w.u32(f"{ARCH}.rope.dimension_count", 0)
    w.f32(f"{ARCH}.rope.freq_base", 10000.0)

    # MLA. attention.key_length/value_length are the width of the ABSORBED LATENT (they equal
    # kv_lora_rank in the released weights); the real per-head widths are the *_mla keys.
    w.u32(f"{ARCH}.attention.key_length", KV_LORA)
    w.u32(f"{ARCH}.attention.value_length", KV_LORA)
    w.u32(f"{ARCH}.attention.key_length_mla", HEAD_MLA)
    w.u32(f"{ARCH}.attention.value_length_mla", HEAD_MLA)
    w.u32(f"{ARCH}.attention.q_lora_rank", Q_LORA)
    w.u32(f"{ARCH}.attention.kv_lora_rank", KV_LORA)

    # KDA
    w.u32(f"{ARCH}.ssm.conv_kernel", D_CONV)
    w.u32(f"{ARCH}.kda.head_dim", KDA_HEAD)
    w.f32(f"{ARCH}.kda.gate_lower_bound", GATE_LB)

    # DSA indexer with k-pool compression
    w.u32(f"{ARCH}.attention.indexer.head_count", IDX_HEAD)
    w.u32(f"{ARCH}.attention.indexer.key_length", IDX_DIM)
    w.u32(f"{ARCH}.attention.indexer.top_k", IDX_TOPK)
    w.u32(f"{ARCH}.attention.indexer.kpool", KPOOL)
    w.bl(f"{ARCH}.attention.indexer.kpool_select_tail", True)

    # mHC
    w.u32(f"{ARCH}.hyper_connection.count", HC)
    w.u32(f"{ARCH}.hyper_connection.sinkhorn_iterations", 20)
    w.f32(f"{ARCH}.hyper_connection.epsilon", 1e-6)

    # MoE
    w.u32(f"{ARCH}.expert_count", N_EXPERT)
    w.u32(f"{ARCH}.expert_used_count", N_USED)
    w.u32(f"{ARCH}.expert_shared_count", N_SHARED)
    w.u32(f"{ARCH}.expert_feed_forward_length", N_FF_EXP)
    w.u32(f"{ARCH}.leading_dense_block_count", N_DENSE)
    w.f32(f"{ARCH}.expert_weights_scale", 2.5)
    w.bl(f"{ARCH}.expert_weights_norm", True)
    w.u32(f"{ARCH}.expert_gating_func", 2)     # sigmoid / noaux_tc
    w.arr_f32(f"{ARCH}.swiglu_clamp_exp", [10.0] * N_LAYER)
    w.arr_f32(f"{ARCH}.swiglu_clamp_shexp", [10.0] * N_LAYER)

    # --- vocab. A byte-fallback SPM vocab: 3 control pieces, the 256 single-byte tokens, and
    # filler. Every printable ASCII prompt therefore tokenises to one token per byte in BOTH
    # engines, which is what makes a logit comparison meaningful.
    toks, scores, ttypes = [], [], []
    for t in ("<unk>", "<s>", "</s>"):
        toks.append(t); scores.append(0.0); ttypes.append(3)          # CONTROL (unk is 2, close enough)
    ttypes[0] = 2                                                      # UNKNOWN
    for b in range(256):
        toks.append(f"<0x{b:02X}>"); scores.append(0.0); ttypes.append(6)   # BYTE
    i = 0
    while len(toks) < N_VOCAB:
        toks.append(f"▁tok{i}"); scores.append(-float(i) - 1.0); ttypes.append(1)  # NORMAL
        i += 1
    assert len(toks) == N_VOCAB

    w.st("tokenizer.ggml.model", "llama")
    w.arr_str("tokenizer.ggml.tokens", toks)
    w.arr_f32("tokenizer.ggml.scores", scores)
    w.arr_i32("tokenizer.ggml.token_type", ttypes)
    w.u32("tokenizer.ggml.bos_token_id", 1)
    w.u32("tokenizer.ggml.eos_token_id", 2)
    w.u32("tokenizer.ggml.unknown_token_id", 0)
    w.bl("tokenizer.ggml.add_bos_token", False)
    w.bl("tokenizer.ggml.add_eos_token", False)

    # --- weights. Small enough that eight blocks of random projections do not saturate, and
    # deterministic from `seed` so both engines are handed byte-identical numbers.
    def R(*ne, s=0.05):
        """random tensor with the given GGML ne; numpy shape is reversed(ne)"""
        return (rng.standard_normal(tuple(reversed(ne))) * s).astype(np.float32)

    def ONES(n, jitter=0.02):
        return (np.ones(n) + rng.standard_normal(n) * jitter).astype(np.float32)

    w.tensor("token_embd.weight", [N_EMBD, N_VOCAB], R(N_EMBD, N_VOCAB, s=0.2))
    w.tensor("output_norm.weight", [N_EMBD], ONES(N_EMBD))
    w.tensor("output.weight", [N_EMBD, N_VOCAB], R(N_EMBD, N_VOCAB, s=0.2))

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.tensor(p + "attn_norm.weight", [N_EMBD], ONES(N_EMBD))
        w.tensor(p + "ffn_norm.weight",  [N_EMBD], ONES(N_EMBD))

        # mHC mixers
        for tag in ("attn", "ffn"):
            w.tensor(p + f"hc_{tag}_fn.weight",    [HC * N_EMBD, HC_MIX], R(HC * N_EMBD, HC_MIX, s=0.02))
            w.tensor(p + f"hc_{tag}_base.weight",  [HC_MIX], R(HC_MIX, s=0.1))
            w.tensor(p + f"hc_{tag}_scale.weight", [3], np.array([1.0, 1.0, 1.0], dtype=np.float32))

        if il in MLA_AT:
            w.tensor(p + "attn_q_a_norm.weight",  [Q_LORA],  ONES(Q_LORA))
            w.tensor(p + "attn_kv_a_norm.weight", [KV_LORA], ONES(KV_LORA))
            w.tensor(p + "attn_q_a.weight",       [N_EMBD, Q_LORA],            R(N_EMBD, Q_LORA))
            w.tensor(p + "attn_q_b.weight",       [Q_LORA, N_HEAD * HEAD_MLA], R(Q_LORA, N_HEAD * HEAD_MLA))
            w.tensor(p + "attn_kv_a_mqa.weight",  [N_EMBD, KV_LORA],           R(N_EMBD, KV_LORA))
            w.tensor(p + "attn_k_b.weight",       [HEAD_MLA, KV_LORA, N_HEAD], R(HEAD_MLA, KV_LORA, N_HEAD))
            w.tensor(p + "attn_v_b.weight",       [KV_LORA, HEAD_MLA, N_HEAD], R(KV_LORA, HEAD_MLA, N_HEAD))
            w.tensor(p + "attn_output.weight",    [N_HEAD * HEAD_MLA, N_EMBD], R(N_HEAD * HEAD_MLA, N_EMBD))

            w.tensor(p + "indexer.k_norm.weight", [IDX_DIM], ONES(IDX_DIM))
            w.tensor(p + "indexer.k_norm.bias",   [IDX_DIM], R(IDX_DIM, s=0.05))
            w.tensor(p + "indexer.proj.weight",   [N_EMBD, IDX_HEAD], R(N_EMBD, IDX_HEAD))
            w.tensor(p + "indexer.attn_k.weight", [N_EMBD, IDX_DIM],  R(N_EMBD, IDX_DIM))
            w.tensor(p + "indexer.attn_q_b.weight", [Q_LORA, IDX_HEAD * IDX_DIM], R(Q_LORA, IDX_HEAD * IDX_DIM))
            w.tensor(p + "indexer_compressor_gate.weight", [N_EMBD, IDX_DIM], R(N_EMBD, IDX_DIM))
            w.tensor(p + "indexer_compressor_ape.weight",  [IDX_DIM, KPOOL],  R(IDX_DIM, KPOOL, s=0.2))
        else:
            for t in ("q", "k", "v"):
                w.tensor(p + f"ssm_conv1d_{t}.weight", [D_CONV, 1, D_INNER], R(D_CONV, 1, D_INNER, s=0.3))
            w.tensor(p + "attn_q.weight",      [N_EMBD, D_INNER], R(N_EMBD, D_INNER))
            w.tensor(p + "attn_k.weight",      [N_EMBD, D_INNER], R(N_EMBD, D_INNER))
            w.tensor(p + "attn_v.weight",      [N_EMBD, D_INNER], R(N_EMBD, D_INNER))
            w.tensor(p + "attn_output.weight", [D_INNER, N_EMBD], R(D_INNER, N_EMBD))

            w.tensor(p + "ssm_f_a.weight", [N_EMBD, KDA_HEAD],  R(N_EMBD, KDA_HEAD))
            w.tensor(p + "ssm_f_b.weight", [KDA_HEAD, D_INNER], R(KDA_HEAD, D_INNER))
            w.tensor(p + "ssm_dt.bias",    [D_INNER],           R(D_INNER, s=0.5))
            # ssm_a holds -exp(A_log), one per head
            w.tensor(p + "ssm_a", [N_HEAD],
                     (-np.exp(rng.uniform(-2.0, 0.0, N_HEAD))).astype(np.float32))
            w.tensor(p + "ssm_beta.weight", [N_EMBD, N_HEAD],   R(N_EMBD, N_HEAD, s=0.2))
            w.tensor(p + "ssm_g_a.weight",  [N_EMBD, KDA_HEAD], R(N_EMBD, KDA_HEAD))
            w.tensor(p + "ssm_g_b.weight",  [KDA_HEAD, D_INNER],R(KDA_HEAD, D_INNER))
            w.tensor(p + "ssm_norm.weight", [KDA_HEAD],         ONES(KDA_HEAD))

        if il < N_DENSE:
            w.tensor(p + "ffn_gate.weight", [N_EMBD, N_FF], R(N_EMBD, N_FF))
            w.tensor(p + "ffn_up.weight",   [N_EMBD, N_FF], R(N_EMBD, N_FF))
            w.tensor(p + "ffn_down.weight", [N_FF, N_EMBD], R(N_FF, N_EMBD))
        else:
            w.tensor(p + "ffn_gate_inp.weight", [N_EMBD, N_EXPERT], R(N_EMBD, N_EXPERT, s=0.2))
            w.tensor(p + "exp_probs_b.bias",    [N_EXPERT],         R(N_EXPERT, s=0.1))
            w.tensor(p + "ffn_gate_exps.weight", [N_EMBD, N_FF_EXP, N_EXPERT], R(N_EMBD, N_FF_EXP, N_EXPERT))
            w.tensor(p + "ffn_up_exps.weight",   [N_EMBD, N_FF_EXP, N_EXPERT], R(N_EMBD, N_FF_EXP, N_EXPERT))
            w.tensor(p + "ffn_down_exps.weight", [N_FF_EXP, N_EMBD, N_EXPERT], R(N_FF_EXP, N_EMBD, N_EXPERT))
            w.tensor(p + "ffn_gate_shexp.weight", [N_EMBD, N_FF_EXP * N_SHARED], R(N_EMBD, N_FF_EXP * N_SHARED))
            w.tensor(p + "ffn_up_shexp.weight",   [N_EMBD, N_FF_EXP * N_SHARED], R(N_EMBD, N_FF_EXP * N_SHARED))
            w.tensor(p + "ffn_down_shexp.weight", [N_FF_EXP * N_SHARED, N_EMBD], R(N_FF_EXP * N_SHARED, N_EMBD))

    n = w.write(path)
    print(f"wrote {path}: {len(w.tensors)} tensors, {len(w.kv)} kv, {n/1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1234)
    # PXA_GLM5NEXT: the real model ships indexer.top_k = 2048, which with kpool = 4 makes
    # n_top saturate at the 64-pool pad floor and n_sel = 259 -- so every short context
    # takes the SCATTER branch. The default 8 keeps the original identity-gate artifact.
    ap.add_argument("--top-k", type=int, default=8, dest="top_k")
    a = ap.parse_args()
    build(a.out, a.seed, a.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
