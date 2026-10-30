#!/usr/bin/env python3
"""PXA_NEMOTRON_MTP: a tiny random-weight `nemotron_h_moe` trunk GGUF + a SEPARATE NextN head GGUF.

The pair mirrors how NVIDIA ships Nemotron 3.5 Super-VL: the trunk file (88 blocks there) carries no head, and
the draft head is its own file `mtp-*.gguf` whose metadata describes block_count = trunk + 1 and
nextn_predict_layers = 1, with per-layer arrays one entry longer, and whose tensors are the tail block
blk.<n_trunk>.* (attention + post_attention_norm + latent MoE + relu^2 shared expert + nextn.{eh_proj, enorm, hnorm,
shared_head_norm}) plus its own copies of token_embd / output / output_norm (the engine uses the trunk's). The
engine attaches it with PXA_MTP_HEAD_FILE=<head> and runs it with -mtp / --spec-type mtp.

Trunk, in miniature (every structural feature of the 120B-A12B on, at sizes a CPU runs in a second):
  6 blocks  M E M * E M   (M = Mamba-2 mixer, E = latent MoE, * = NoPE GQA attention)
  Mamba-2: d_inner 128, d_state 16, 2 groups, 8 heads, conv 4
  MoE: 8 routed experts (2 used) at latent width 32 behind latent_down/up, relu^2, sigmoid + exp_probs_b,
       weights norm + scale 5.0, one relu^2 shared expert on the full width

THE GATE NEEDS DRAFTS THAT ARE SOMETIMES RIGHT. A random head is right once in |vocab| tries, so every draft is
rejected and the accept path never runs. So the weights are not purely random:
  * the trunk is a noisy successor machine: output row j is the sum of the embeddings of the tokens whose
    successor is j (a fixed random map onto the plain `tokN` pieces), so with small block outputs the next token
    is succ(current); --trunk-mix scales every block's output projection (the relu^2 latent MoE at scale 5.0 is
    the loud one): at 0 the text is the pure successor chain (drafts 100% accepted, the rollback never runs),
    at 0.04 it follows the successor 48/95 steps and depends on history (MTP1/2/4 accept 84/59/36%, and the
    NO_RESTORE negative control changes the text), at >= 0.1 the blocks drown the successor (~1% accepted);
  * the head's eh_proj passes the normalised embedding of its input token through (identity on the e half) plus
    --head-noise times a random matrix on the hidden half, so the head proposes succ(draft input) and is right
    whenever the trunk follows the successor map.
The gate is token-for-token identity of greedy text with and without the head; the acceptance rate printed by
the server says the accept AND the reject path both ran.

Usage: python3 -I tests/gen_tiny_nemotron_h_moe.py TRUNK.gguf HEAD.gguf [--seed 7] [--trunk-mix 0.04] [--head-noise 0.3]
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
GGML_TYPE_BF16 = 30


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
    def tensor(self, name, ne, data, bf16=False):
        a = np.ascontiguousarray(data, dtype=np.float32)
        assert list(a.shape) == list(reversed(ne)), f"{name}: {a.shape} vs ne {ne}"
        if bf16:   # round-to-nearest-even to the upper 16 bits (the shipped head's router is BF16)
            u = a.view(np.uint32).astype(np.uint64)
            u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
            self.tensors.append((name, list(ne), u.tobytes(), GGML_TYPE_BF16))
        else:
            self.tensors.append((name, list(ne), a.tobytes(), GGML_TYPE_F32))

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
        for name, ne, data, gt in self.tensors:
            infos += self._str(name)
            infos += struct.pack("<I", len(ne))
            for d in ne:
                infos += struct.pack("<Q", d)
            infos += struct.pack("<I", gt)
            infos += struct.pack("<Q", offset)
            offset += (len(data) + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

        body = bytearray(head) + infos
        pad = (-len(body)) % ALIGNMENT
        body += b"\x00" * pad
        for _, _, data, _gt in self.tensors:
            body += data
            body += b"\x00" * ((-len(data)) % ALIGNMENT)

        with open(path, "wb") as f:
            f.write(body)
        return len(body)


# --- the model ------------------------------------------------------------------------

ARCH     = "nemotron_h_moe"
N_TRUNK  = 6
KIND     = ["M", "E", "M", "A", "E", "M"]      # head block (index N_TRUNK) is attention + MoE
N_EMBD   = 128
N_HEAD   = 4
HEAD_DIM = 32
N_HEAD_KV = 2
N_VOCAB  = 512
N_CTX    = 4096
D_CONV, D_INNER, D_STATE, N_GROUP, N_SSM_HEAD = 4, 128, 16, 2, 8
N_EXPERT, N_USED, LATENT, N_FF_EXP, N_FF_SHEXP = 8, 2, 32, 48, 96


def vocab():
    toks, scores, ttypes = [], [], []
    for t in ("<unk>", "<s>", "</s>"):
        toks.append(t); scores.append(0.0); ttypes.append(3)
    ttypes[0] = 2
    for b in range(256):
        toks.append(f"<0x{b:02X}>"); scores.append(0.0); ttypes.append(6)
    i = 0
    while len(toks) < N_VOCAB:
        toks.append(f"\u2581tok{i}"); scores.append(-float(i) - 1.0); ttypes.append(1)
        i += 1
    return toks, scores, ttypes


def meta(w, n_layer, head):
    kinds = KIND + (["H"] if head else [])
    w.st("general.architecture", ARCH)
    w.st("general.name", "tiny-nemotron-h-moe" + ("-mtp" if head else ""))
    w.u32("general.file_type", 0)
    w.u32("general.quantization_version", 2)
    w.u32(f"{ARCH}.block_count", n_layer)
    w.u32(f"{ARCH}.context_length", N_CTX)
    w.u32(f"{ARCH}.embedding_length", N_EMBD)
    w.arr_u32(f"{ARCH}.feed_forward_length", [N_FF_EXP if k in ("E", "H") else 0 for k in kinds])
    w.u32(f"{ARCH}.attention.head_count", N_HEAD)
    w.arr_u32(f"{ARCH}.attention.head_count_kv", [N_HEAD_KV if k in ("A", "H") else 0 for k in kinds])
    w.f32(f"{ARCH}.attention.layer_norm_epsilon", 1e-5)
    w.u32(f"{ARCH}.expert_count", N_EXPERT)
    w.u32(f"{ARCH}.expert_used_count", N_USED)
    w.u32(f"{ARCH}.expert_group_count", 1)
    w.u32(f"{ARCH}.expert_group_used_count", 1)
    w.u32(f"{ARCH}.vocab_size", N_VOCAB)
    w.u32(f"{ARCH}.rope.dimension_count", HEAD_DIM)
    w.u32(f"{ARCH}.ssm.conv_kernel", D_CONV)
    w.u32(f"{ARCH}.ssm.state_size", D_STATE)
    w.u32(f"{ARCH}.ssm.group_count", N_GROUP)
    w.u32(f"{ARCH}.ssm.inner_size", D_INNER)
    w.u32(f"{ARCH}.ssm.time_step_rank", N_SSM_HEAD)
    w.u32(f"{ARCH}.attention.key_length", HEAD_DIM)
    w.u32(f"{ARCH}.attention.value_length", HEAD_DIM)
    w.u32(f"{ARCH}.expert_feed_forward_length", N_FF_EXP)
    w.u32(f"{ARCH}.expert_shared_feed_forward_length", N_FF_SHEXP)
    w.u32(f"{ARCH}.expert_shared_count", 1)
    w.bl(f"{ARCH}.expert_weights_norm", True)
    w.f32(f"{ARCH}.expert_weights_scale", 5.0)
    w.u32(f"{ARCH}.moe_latent_size", LATENT)
    if head:
        w.u32(f"{ARCH}.nextn_predict_layers", 1)
    toks, scores, ttypes = vocab()
    w.st("tokenizer.ggml.model", "llama")
    w.arr_str("tokenizer.ggml.tokens", toks)
    w.arr_f32("tokenizer.ggml.scores", scores)
    w.arr_i32("tokenizer.ggml.token_type", ttypes)
    w.u32("tokenizer.ggml.bos_token_id", 1)
    w.u32("tokenizer.ggml.eos_token_id", 2)
    w.u32("tokenizer.ggml.unknown_token_id", 0)
    w.bl("tokenizer.ggml.add_bos_token", False)
    w.bl("tokenizer.ggml.add_eos_token", False)


def build(trunk_path, head_path, seed, mix, noise):
    rng = np.random.default_rng(seed)

    def R(*ne, s=0.05):
        return (rng.standard_normal(tuple(reversed(ne))) * s).astype(np.float32)

    def ONES(n, jitter=0.02):
        return (np.ones(n) + rng.standard_normal(n) * jitter).astype(np.float32)

    # embeddings ~ N(0,1): self dot ~ N_EMBD, cross dot ~ N(0, sqrt(N_EMBD)) -> a clear successor margin
    embd = rng.standard_normal((N_VOCAB, N_EMBD)).astype(np.float32)
    plain = np.arange(3 + 256, N_VOCAB)                       # the "tokN" pieces: valid UTF-8, never control
    succ = rng.choice(plain, size=N_VOCAB)                    # successor of every token (controls too, unused)
    out = np.zeros((N_VOCAB, N_EMBD), dtype=np.float32)
    for t in range(3, N_VOCAB):
        out[succ[t]] += embd[t]
    out[:3] = -0.5 * embd[:3]                                 # control pieces never win
    out_norm = ONES(N_EMBD)

    def mix_s(base):
        return base * mix

    def block(w, i, kind):
        p = f"blk.{i}."
        w.tensor(p + "attn_norm.weight", [N_EMBD], ONES(N_EMBD))
        if kind == "M":
            conv_dim = D_INNER + 2 * N_GROUP * D_STATE
            d_in_proj = 2 * D_INNER + 2 * N_GROUP * D_STATE + N_SSM_HEAD
            w.tensor(p + "ssm_in.weight", [N_EMBD, d_in_proj], R(N_EMBD, d_in_proj, s=0.09))
            w.tensor(p + "ssm_conv1d.weight", [D_CONV, conv_dim], R(D_CONV, conv_dim, s=0.4))
            w.tensor(p + "ssm_conv1d.bias", [conv_dim], R(conv_dim, s=0.05))
            w.tensor(p + "ssm_dt.bias", [N_SSM_HEAD], R(N_SSM_HEAD, s=0.5) - 1.0)
            w.tensor(p + "ssm_a", [1, N_SSM_HEAD], -np.exp(R(1, N_SSM_HEAD, s=0.5)))
            w.tensor(p + "ssm_d", [1, N_SSM_HEAD], ONES(N_SSM_HEAD, 0.1).reshape(N_SSM_HEAD, 1))
            w.tensor(p + "ssm_norm.weight", [D_INNER // N_GROUP, N_GROUP], ONES(D_INNER).reshape(N_GROUP, D_INNER // N_GROUP))
            w.tensor(p + "ssm_out.weight", [D_INNER, N_EMBD], R(D_INNER, N_EMBD, s=mix_s(0.09)))
        elif kind in ("A", "H"):
            w.tensor(p + "attn_q.weight", [N_EMBD, N_HEAD * HEAD_DIM], R(N_EMBD, N_HEAD * HEAD_DIM, s=0.15))
            w.tensor(p + "attn_k.weight", [N_EMBD, N_HEAD_KV * HEAD_DIM], R(N_EMBD, N_HEAD_KV * HEAD_DIM, s=0.15))
            w.tensor(p + "attn_v.weight", [N_EMBD, N_HEAD_KV * HEAD_DIM], R(N_EMBD, N_HEAD_KV * HEAD_DIM, s=0.09))
            w.tensor(p + "attn_output.weight", [N_HEAD * HEAD_DIM, N_EMBD], R(N_HEAD * HEAD_DIM, N_EMBD, s=mix_s(0.09)))
        if kind in ("E", "H"):
            if kind == "H":
                w.tensor(p + "post_attention_norm.weight", [N_EMBD], ONES(N_EMBD))
            w.tensor(p + "ffn_gate_inp.weight", [N_EMBD, N_EXPERT], R(N_EMBD, N_EXPERT, s=0.1), bf16=(kind == "H"))
            w.tensor(p + "exp_probs_b.bias", [N_EXPERT], R(N_EXPERT, s=0.05))
            w.tensor(p + "ffn_latent_down.weight", [N_EMBD, LATENT], R(N_EMBD, LATENT, s=0.09))
            w.tensor(p + "ffn_latent_up.weight", [LATENT, N_EMBD], R(LATENT, N_EMBD, s=mix_s(0.18)))
            w.tensor(p + "ffn_down_exps.weight", [N_FF_EXP, LATENT, N_EXPERT], R(N_FF_EXP, LATENT, N_EXPERT, s=0.15))
            w.tensor(p + "ffn_up_exps.weight", [LATENT, N_FF_EXP, N_EXPERT], R(LATENT, N_FF_EXP, N_EXPERT, s=0.5))
            w.tensor(p + "ffn_down_shexp.weight", [N_FF_SHEXP, N_EMBD], R(N_FF_SHEXP, N_EMBD, s=mix_s(0.1)))
            w.tensor(p + "ffn_up_shexp.weight", [N_EMBD, N_FF_SHEXP], R(N_EMBD, N_FF_SHEXP, s=0.1))
        if kind == "H":
            eh = np.zeros((N_EMBD, 2 * N_EMBD), dtype=np.float32)        # numpy (out, in) = ggml ne [2E, E]
            eh[:, :N_EMBD] = np.eye(N_EMBD, dtype=np.float32)
            eh[:, N_EMBD:] = rng.standard_normal((N_EMBD, N_EMBD)).astype(np.float32) * (noise / np.sqrt(N_EMBD))
            w.tensor(p + "nextn.eh_proj.weight", [2 * N_EMBD, N_EMBD], eh)
            w.tensor(p + "nextn.enorm.weight", [N_EMBD], ONES(N_EMBD))
            w.tensor(p + "nextn.hnorm.weight", [N_EMBD], ONES(N_EMBD))
            w.tensor(p + "nextn.shared_head_norm.weight", [N_EMBD], ONES(N_EMBD))

    t = Writer()
    meta(t, N_TRUNK, head=False)
    t.tensor("token_embd.weight", [N_EMBD, N_VOCAB], embd)
    t.tensor("output_norm.weight", [N_EMBD], out_norm)
    t.tensor("output.weight", [N_EMBD, N_VOCAB], out)
    for i, k in enumerate(KIND):
        block(t, i, k)
    nt = t.write(trunk_path)

    h = Writer()
    meta(h, N_TRUNK + 1, head=True)
    h.tensor("token_embd.weight", [N_EMBD, N_VOCAB], embd)      # the shipped head carries copies; unused
    h.tensor("output_norm.weight", [N_EMBD], out_norm)
    h.tensor("output.weight", [N_EMBD, N_VOCAB], out)
    block(h, N_TRUNK, "H")
    nh = h.write(head_path)
    print(f"wrote {trunk_path} ({nt} B, {len(t.tensors)} tensors) and {head_path} ({nh} B, {len(h.tensors)} tensors); "
          f"seed {seed} trunk-mix {mix} head-noise {noise}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trunk")
    ap.add_argument("head")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--trunk-mix", type=float, default=0.04)
    ap.add_argument("--head-noise", type=float, default=0.3)
    a = ap.parse_args()
    build(a.trunk, a.head, a.seed, a.trunk_mix, a.head_noise)


if __name__ == "__main__":
    main()
