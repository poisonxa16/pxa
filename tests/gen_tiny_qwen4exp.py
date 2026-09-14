#!/usr/bin/env python3
"""PXA_QSA: emit a tiny random-weight F32 `qwen4exp` GGUF for the QSA identity gates.

The point is a model small enough to run to completion on a CPU in a couple of seconds, but
with the STRUCTURE that decides whether the query-time sparse-attention path is right:

  * 12 trunk blocks with full_attention_interval = 4, so full attention on il = 3, 7, 11 and
    Gated DeltaNet on the other nine -- the shipped model's 12-of-48 pattern reduced
    proportionally, and the same "the QSA layers are a minority of the trunk" shape.
  * the indexer at its REAL geometry: 4 heads x 128, compress ratio 4. Both matter. 4 heads is
    what lets the score ride the fused ggml_kpool_score (which wants a power of two in [4,32]),
    and 128 is a multiple of the CUDA warp, which this tree's norm kernel requires.
  * a SMALL budget: indexer.top_k = 16, so n_top = 4 blocks = 16 cells plus a 3-cell tail. Any
    prompt longer than ~20 tokens therefore has a context genuinely wider than the selection,
    which is the condition the decode gather branch is chosen on -- a tiny model with a large
    top_k would silently test nothing.
  * GQA with n_head = 4 over n_head_kv = 2, so the gather's 4-arg mul_mat broadcast
    (n_head % n_head_kv == 0) is exercised rather than assumed.
  * hyper connections (count 4), the MoE with a shared expert, and the gated Q projection that
    packs the attention output gate into wq -- everything the QSA block has to reproduce from
    build_std_attention.

Deliberately OFF: the PLE side path (no `ple.layers` key) and the NextN/MTP graft. Neither is
on the QSA path, both carry their own persistent state, and leaving them out keeps a failure
here attributable.

What the fixture is FOR (see tests/README-qsa-fixture.md-free note in the commit message):

    PXA_QSA unset          must reproduce the pre-QSA binary's greedy output byte for byte
    PXA_QSA=1 GATHER=0     the architecture's reference algorithm: the selection as a mask
    PXA_QSA=1 GATHER=1     the same selection, physically gathered

Arms 2 and 3 must agree on the greedy text; arm 1 is a different model output by design.

The GGUF is written by hand (struct + numpy) rather than through gguf-py so the generator does
not depend on which tree's gguf-py is importable, and so the ggml `ne` order of every tensor is
explicit at the call site rather than implied by a numpy transpose. Same Writer as
tests/gen_tiny_glm5next.py.

Usage:  python3 tests/gen_tiny_qwen4exp.py out.gguf [--seed 1234]
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

def build(path, seed):
    rng = np.random.default_rng(seed)

    ARCH     = "qwen4exp"
    N_LAYER  = 12
    FULL_INT = 4                                   # full attention on il = 3, 7, 11
    FULL     = {i for i in range(N_LAYER) if (i + 1) % FULL_INT == 0}
    N_EMBD   = 128
    N_VOCAB  = 512
    N_CTX    = 4096

    # attention: GQA 2:1, so the gather's mul_mat broadcast is real
    N_HEAD    = 4
    N_HEAD_KV = 2
    HEAD_K    = 32                                 # attention.key_length / value_length
    N_ROT     = 32
    SECTIONS  = [8, 4, 4, 0]                       # IMRoPE, sums to N_ROT/2

    # the QSA indexer, at the shipped model's own geometry
    IDX_HEAD = 4
    IDX_DIM  = 128
    IDX_TOPK = 16                                  # -> 4 whole blocks (16 cells) + a 3-cell tail
    RATIO    = 4

    # Gated DeltaNet. ssm_d_inner MUST equal ssm_d_state * ssm_dt_rank (the loader checks it).
    D_CONV   = 4
    D_STATE  = 32                                  # head_k_dim == head_v_dim
    DT_RANK  = 4                                   # n_v_heads
    N_GROUP  = 2                                   # n_k_heads
    D_INNER  = D_STATE * DT_RANK                   # 128
    KEY_DIM   = D_STATE * N_GROUP                  # 64
    VALUE_DIM = D_STATE * DT_RANK                  # 128
    CONV_DIM  = KEY_DIM * 2 + VALUE_DIM            # 256

    # hyper connections
    HC     = 4
    HC_LR  = 16
    HC_DIM = HC * N_EMBD                           # 512

    # MoE on every layer -- this arch has no dense-FFN fallback
    N_EXPERT, N_USED, N_SHARED = 4, 2, 1
    N_FF_EXP   = 32
    N_FF_SHEXP = 32
    N_FF       = 64

    w = Writer()

    # --- general
    w.st("general.architecture", ARCH)
    w.st("general.name", "tiny-qwen4exp")
    w.u32("general.file_type", 0)              # ALL_F32
    w.u32("general.quantization_version", 2)

    # --- geometry
    w.u32(f"{ARCH}.block_count", N_LAYER)
    w.u32(f"{ARCH}.context_length", N_CTX)
    w.u32(f"{ARCH}.embedding_length", N_EMBD)
    w.u32(f"{ARCH}.feed_forward_length", N_FF)
    w.u32(f"{ARCH}.vocab_size", N_VOCAB)
    w.u32(f"{ARCH}.attention.head_count", N_HEAD)
    w.u32(f"{ARCH}.attention.head_count_kv", N_HEAD_KV)
    w.u32(f"{ARCH}.attention.key_length", HEAD_K)
    w.u32(f"{ARCH}.attention.value_length", HEAD_K)
    w.f32(f"{ARCH}.attention.layer_norm_rms_epsilon", 1e-6)
    w.u32(f"{ARCH}.full_attention_interval", FULL_INT)
    w.u32(f"{ARCH}.nextn_predict_layers", 0)

    # IMRoPE
    w.u32(f"{ARCH}.rope.dimension_count", N_ROT)
    w.arr_i32(f"{ARCH}.rope.dimension_sections", SECTIONS)
    w.f32(f"{ARCH}.rope.freq_base", 10000.0)

    # the QSA indexer. compress_ratios is PER LAYER and is 0 on the recurrent ones -- the
    # loader refuses a ratio on a recurrent layer, and refuses a non-uniform one, rather than
    # silently mis-blocking, so this array is part of what the fixture tests.
    w.u32(f"{ARCH}.attention.indexer.head_count", IDX_HEAD)
    w.u32(f"{ARCH}.attention.indexer.key_length", IDX_DIM)
    w.u32(f"{ARCH}.attention.indexer.top_k", IDX_TOPK)
    w.arr_u32(f"{ARCH}.attention.compress_ratios",
              [RATIO if i in FULL else 0 for i in range(N_LAYER)])

    # Gated DeltaNet
    w.u32(f"{ARCH}.ssm.conv_kernel", D_CONV)
    w.u32(f"{ARCH}.ssm.inner_size", D_INNER)
    w.u32(f"{ARCH}.ssm.state_size", D_STATE)
    w.u32(f"{ARCH}.ssm.time_step_rank", DT_RANK)
    w.u32(f"{ARCH}.ssm.group_count", N_GROUP)

    # hyper connections
    w.u32(f"{ARCH}.hyper_connection.count", HC)
    w.u32(f"{ARCH}.hyper_connection.low_rank", HC_LR)

    # MoE
    w.u32(f"{ARCH}.expert_count", N_EXPERT)
    w.u32(f"{ARCH}.expert_used_count", N_USED)
    w.u32(f"{ARCH}.expert_shared_count", N_SHARED)
    w.u32(f"{ARCH}.expert_feed_forward_length", N_FF_EXP)
    w.u32(f"{ARCH}.expert_shared_feed_forward_length", N_FF_SHEXP)

    # --- vocab. A byte-fallback SPM vocab: 3 control pieces, the 256 single-byte tokens, and
    # filler, so every printable ASCII prompt tokenises to one token per byte and a prompt of a
    # known length produces a context of a known length -- which is what lets the fixture put
    # the cache reliably past the selection width.
    toks, scores, ttypes = [], [], []
    for t in ("<unk>", "<s>", "</s>"):
        toks.append(t); scores.append(0.0); ttypes.append(3)
    ttypes[0] = 2
    for b in range(256):
        toks.append(f"<0x{b:02X}>"); scores.append(0.0); ttypes.append(6)
    i = 0
    while len(toks) < N_VOCAB:
        toks.append(f"▁tok{i}"); scores.append(-float(i) - 1.0); ttypes.append(1)
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

    # --- weights, deterministic from `seed`
    def R(*ne, s=0.05):
        """random tensor with the given GGML ne; numpy shape is reversed(ne)"""
        return (rng.standard_normal(tuple(reversed(ne))) * s).astype(np.float32)

    def ONES(n, jitter=0.02):
        return (np.ones(n) + rng.standard_normal(n) * jitter).astype(np.float32)

    w.tensor("token_embd.weight", [N_EMBD, N_VOCAB], R(N_EMBD, N_VOCAB, s=0.2))
    w.tensor("output.weight",     [N_EMBD, N_VOCAB], R(N_EMBD, N_VOCAB, s=0.2))

    # there is no output_norm for this arch: the final low-rank hc mixer carries it
    w.tensor("output_hc_norm.weight", [HC_DIM],         ONES(HC_DIM))
    w.tensor("output_hc_down.weight", [HC_DIM, HC_LR],  R(HC_DIM, HC_LR))
    w.tensor("output_hc_up.weight",   [HC_LR, HC_DIM],  R(HC_LR, HC_DIM))

    for il in range(N_LAYER):
        p = f"blk.{il}."

        for tag in ("attn", "ffn"):
            w.tensor(p + f"hc_{tag}_norm.weight",   [HC_DIM],        ONES(HC_DIM))
            w.tensor(p + f"hc_{tag}_down.weight",   [HC_DIM, HC_LR], R(HC_DIM, HC_LR))
            w.tensor(p + f"hc_{tag}_up.weight",     [HC_LR, HC_DIM], R(HC_LR, HC_DIM))
            w.tensor(p + f"hc_{tag}_inject.weight", [HC_DIM, HC],    R(HC_DIM, HC))

        if il in FULL:
            # wq holds [q|gate] per head, hence the *2
            w.tensor(p + "attn_q.weight",      [N_EMBD, HEAD_K * N_HEAD * 2], R(N_EMBD, HEAD_K * N_HEAD * 2))
            w.tensor(p + "attn_k.weight",      [N_EMBD, HEAD_K * N_HEAD_KV],  R(N_EMBD, HEAD_K * N_HEAD_KV))
            w.tensor(p + "attn_v.weight",      [N_EMBD, HEAD_K * N_HEAD_KV],  R(N_EMBD, HEAD_K * N_HEAD_KV))
            w.tensor(p + "attn_output.weight", [HEAD_K * N_HEAD, N_EMBD],     R(HEAD_K * N_HEAD, N_EMBD))
            w.tensor(p + "attn_q_norm.weight", [HEAD_K], ONES(HEAD_K))
            w.tensor(p + "attn_k_norm.weight", [HEAD_K], ONES(HEAD_K))

            # the indexer. ONE shared 128-wide key per token, four query heads, and a norm on
            # each -- no per-head weight tensor and no compressor, which is exactly why the
            # block summary is parameter-free and the head sum is unweighted.
            w.tensor(p + "indexer.q_proj.weight", [N_EMBD, IDX_HEAD * IDX_DIM], R(N_EMBD, IDX_HEAD * IDX_DIM))
            w.tensor(p + "indexer.k_proj.weight", [N_EMBD, IDX_DIM],            R(N_EMBD, IDX_DIM))
            w.tensor(p + "indexer.q_norm.weight", [IDX_DIM], ONES(IDX_DIM))
            w.tensor(p + "indexer.k_norm.weight", [IDX_DIM], ONES(IDX_DIM))
        else:
            w.tensor(p + "attn_qkv.weight",   [N_EMBD, CONV_DIM],  R(N_EMBD, CONV_DIM))
            w.tensor(p + "attn_gate.weight",  [N_EMBD, VALUE_DIM], R(N_EMBD, VALUE_DIM))
            w.tensor(p + "ssm_conv1d.weight", [D_CONV, CONV_DIM],  R(D_CONV, CONV_DIM, s=0.3))
            w.tensor(p + "ssm_dt.bias",       [DT_RANK],           R(DT_RANK, s=0.5))
            # ssm_a is -exp(A_log), one per v-head, and carries no ".weight" suffix
            w.tensor(p + "ssm_a", [DT_RANK], (-np.exp(rng.uniform(-2.0, 0.0, DT_RANK))).astype(np.float32))
            w.tensor(p + "ssm_beta.weight",  [N_EMBD, DT_RANK],   R(N_EMBD, DT_RANK, s=0.2))
            w.tensor(p + "ssm_alpha.weight", [N_EMBD, DT_RANK],   R(N_EMBD, DT_RANK, s=0.2))
            w.tensor(p + "ssm_norm.weight",  [D_STATE],           ONES(D_STATE))
            w.tensor(p + "ssm_out.weight",   [VALUE_DIM, N_EMBD], R(VALUE_DIM, N_EMBD))

        # MoE on every layer, plus one shared expert
        w.tensor(p + "ffn_gate_inp.weight",   [N_EMBD, N_EXPERT], R(N_EMBD, N_EXPERT, s=0.2))
        w.tensor(p + "ffn_gate_exps.weight",  [N_EMBD, N_FF_EXP, N_EXPERT], R(N_EMBD, N_FF_EXP, N_EXPERT))
        w.tensor(p + "ffn_up_exps.weight",    [N_EMBD, N_FF_EXP, N_EXPERT], R(N_EMBD, N_FF_EXP, N_EXPERT))
        w.tensor(p + "ffn_down_exps.weight",  [N_FF_EXP, N_EMBD, N_EXPERT], R(N_FF_EXP, N_EMBD, N_EXPERT))

        w.tensor(p + "ffn_gate_inp_shexp.weight", [N_EMBD],               R(N_EMBD, s=0.2))
        w.tensor(p + "ffn_gate_shexp.weight",     [N_EMBD, N_FF_SHEXP],   R(N_EMBD, N_FF_SHEXP))
        w.tensor(p + "ffn_up_shexp.weight",       [N_EMBD, N_FF_SHEXP],   R(N_EMBD, N_FF_SHEXP))
        w.tensor(p + "ffn_down_shexp.weight",     [N_FF_SHEXP, N_EMBD],   R(N_FF_SHEXP, N_EMBD))

    n = w.write(path)
    print(f"wrote {path}: {len(w.tensors)} tensors, {len(w.kv)} kv, {n/1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    build(a.out, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
