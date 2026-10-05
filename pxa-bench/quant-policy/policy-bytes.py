#!/usr/bin/env python3
"""Byte and decode-bandwidth accounting for the PXQ tier policies (POLICY_REV 3).

Reads a GGUF header only (no weights, no GPU) and prints, per tensor class:
  * the exact byte cost under each profile, using ggml's own row-size arithmetic
        row_size = row_meta_size + type_size * ne0/blck_size
    so the numbers are what the quantizer will write, not a bpw estimate; and
  * the PER-TOKEN decode read, which is the number that actually sets decode speed.
    A dense class is read in full on every token. A routed expert stack is read
    k-of-N: with 512 experts and top-10, 98% of those bytes are never touched by a
    given token, which is why a MoE's always-resident backbone dominates its decode
    traffic even when it is a rounding error in the file size.

Usage: policy-bytes.py <model.gguf> [--experts N --topk K]
"""
import os, sys, collections, json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))
from gguf.gguf_reader import GGUFReader

# (blck_size, type_size, row_meta_size) straight out of ggml/src/ggml.c
TRAITS = {
    'pxq1':   (32,   5, 2), 'pxq2': (32,  9, 2), 'pxq3':   (32, 13, 2),
    'pxq4':   (32,  17, 2), 'pxq4hq': (32, 18, 2), 'pxq6': (32, 21, 2),
    'mxfp4':  (32,  17, 0),
    'q8_0':   (32,  34, 0), 'q6_K': (256, 210, 0), 'q5_K': (256, 176, 0),
    'q4_K':   (256, 144, 0),
    'f16':    (1,    2, 0), 'f32':  (1,   4, 0), 'bf16': (1, 2, 0),
}

def row_size(t, ne0):
    b, ts, rm = TRAITS[t]
    return rm + ts * ne0 // b

def tensor_bytes(t, ne):
    rows = 1
    for x in ne[1:]:
        rows *= x
    return row_size(t, ne[0]) * rows

# ---------------------------------------------------------------------------------
# The role split the policy layer uses, plus the extra buckets the accounting needs
# (embed and head are not GEMMs; routed experts are the k-of-N class).
# ---------------------------------------------------------------------------------
def ends(name, leaf):
    # same rule as pxa_pxq_policy_name_is(): the leaf must start at a '.' boundary, so
    # "blk.0.hc_ffn_down.weight" is NOT an ffn_down (it is a hyper-connection mixer, f16)
    return name.endswith(leaf) and (len(name) == len(leaf) or name[-len(leaf)-1] == '.')

def role(name, ne):
    if name == 'token_embd.weight':                       return 'embed'
    if name == 'output.weight':                           return 'head'
    if ends(name, 'per_layer_token_embd.weight'):      return 'ple_table'
    # plain suffix, no dot boundary: the leaf is ffn_down_exps.weight, matching the C++
    # pxa_name_ends() the expert path uses (NOT pxa_name_is(), which is the boundary rule)
    if name.endswith('_exps.weight'):                  return 'routed_experts'
    if ends(name, 'ffn_gate_inp.weight'):              return 'router'
    if '_shexp.weight' in name:                           return 'shared_expert'
    for k in ('attn_output.weight', 'attn_output_a.weight', 'attn_output_b.weight'):
        if ends(name, k):                              return 'attn_out'
    for k in ('attn_q.weight', 'attn_qkv.weight', 'attn_q_a.weight', 'attn_q_b.weight'):
        if ends(name, k):                              return 'attn_q/qkv'
    if ends(name, 'ssm_out.weight'):                   return 'deltanet_out'
    if ends(name, 'attn_gate.weight'):                 return 'attn_gate_ch' if ne[1] > 256 else 'attn_gate_head'
    for k in ('attn_k.weight', 'attn_v.weight', 'attn_v_b.weight', 'attn_kv.weight'):
        if ends(name, k):                              return 'attn_kv'
    for k in ('ffn_up.weight', 'ffn_gate.weight', 'ffn_down.weight'):
        if ends(name, k):                              return 'ffn'
    return 'other'

# roles the tier ladder actually moves; everything else is pinned by the backbone table
LADDER = {'attn_q/qkv', 'attn_out', 'attn_gate_ch', 'deltanet_out', 'ffn', 'shared_expert',
          'routed_experts'}
ATTN_ROLES = {'attn_q/qkv', 'attn_gate_ch', 'deltanet_out'}
RANK = ['', 'pxq1', 'pxq2', 'pxq3', 'pxq4', 'pxq4hq', 'pxq6']
CAP = 4   # pxq4: MMVQ-admitted, faster than pxq4hq, and the only one both engines read

def resolve(r, level, policy, moe):
    """What POLICY_REV 3 assigns this role. Mirrors src/pxa-pxq-policy.h."""
    if r in ('embed', 'ple_table'):  return 'q6_K' if r == 'embed' else 'q8_0'
    if r == 'head':                  return 'q8_0'
    if r == 'attn_kv':               return 'q8_0'
    if r == 'attn_gate_head':        return 'f16'
    if r in ('router', 'other'):     return None          # unchanged by any tier decision
    if r == 'routed_experts':        return level         # the level always names these
    lr = RANK.index(level)
    if moe:
        base = 4 if level in ('pxq1', 'pxq2') else 5 if level in ('pxq3',) else lr  # rev-2 backbone
        if r == 'attn_out':
            base = 5 if level in ('pxq1', 'pxq2', 'pxq3') else base
        if policy == 'uniform':
            return RANK[base]
        return RANK[4]
    if policy == 'uniform':          return level
    if r == 'ffn':                   return level
    if policy == 'attn4':            want = 4
    else:                            want = lr + (2 if r == 'attn_out' else 1)
    want = min(want, CAP)
    want = max(want, lr)
    return RANK[want]

def main():
    path = sys.argv[1]
    n_exp = topk = None
    for i, a in enumerate(sys.argv):
        if a == '--experts': n_exp = int(sys.argv[i+1])
        if a == '--topk':    topk  = int(sys.argv[i+1])
    r = GGUFReader(path)
    rows = []
    for t in r.tensors:
        ne = [int(x) for x in t.shape]
        rows.append((t.name, ne, role(t.name, ne), t.tensor_type.name.lower()))

    moe = any(x[2] == 'routed_experts' for x in rows)
    frac = (topk / n_exp) if (moe and n_exp and topk) else 1.0

    profiles = [('pxq2', 'uniform'), ('pxq2', 'balanced'), ('pxq2', 'attn4'),
                ('pxq3', 'uniform'), ('pxq3', 'balanced'), ('pxq3', 'attn4'),
                ('pxq4', 'uniform'), ('pxq4', 'balanced')]

    agg = collections.OrderedDict()
    for name, ne, rl, src in rows:
        nel = 1
        for x in ne: nel *= x
        a = agg.setdefault(rl, dict(n=0, nel=0, per={}, src=set()))
        a['n'] += 1; a['nel'] += nel; a['src'].add(src)
        for lv, po in profiles:
            ty = resolve(rl, lv, po, moe)
            key = f'{lv}-{po}'
            a['per'].setdefault(key, [0, None])
            a['per'][key][0] += tensor_bytes(ty, ne) if ty else tensor_bytes(src, ne)
            a['per'][key][1] = ty or src

    order = sorted(agg, key=lambda k: -agg[k]['nel'])
    print(f'== {path}   ({"MoE, %d experts top-%d" % (n_exp, topk) if moe and n_exp else "dense"})')
    hdr = f'{"role":<16}{"n":>4}{"Melem":>11}  '
    for lv, po in profiles: hdr += f'{lv[3:]+"-"+po[:3]:>13}'
    print(hdr)
    tot = {f'{lv}-{po}': 0 for lv, po in profiles}
    bw  = {f'{lv}-{po}': 0 for lv, po in profiles}
    for rl in order:
        a = agg[rl]
        line = f'{rl:<16}{a["n"]:>4}{a["nel"]/1e6:>11.1f}  '
        for lv, po in profiles:
            k = f'{lv}-{po}'
            b, ty = a['per'][k]
            tot[k] += b
            bw[k]  += b*frac if rl == 'routed_experts' else (0 if rl in ('embed','ple_table') else b)
            line += f'{ty+" "+format(b/2**30,".2f"):>13}'
        print(line)
    print('-'*len(hdr))
    l1 = f'{"FILE GiB":<16}{"":>4}{"":>11}  '
    l2 = f'{"DECODE MiB/tok":<16}{"":>4}{"":>11}  '
    l3 = f'{"vs uniform":<16}{"":>4}{"":>11}  '
    for lv, po in profiles:
        k = f'{lv}-{po}'; u = f'{lv}-uniform'
        l1 += f'{tot[k]/2**30:>13.2f}'
        l2 += f'{bw[k]/2**20:>13.1f}'
        l3 += f'{100*(tot[k]-tot[u])/tot[u]:>12.1f}%'
    print(l1); print(l2); print(l3)
    print('\nDECODE MiB/tok counts every weight a single token reads: dense classes in full,'
          f' routed experts at {frac:.4f} of the stack. embed/ple are row gathers and are excluded.')

main()
