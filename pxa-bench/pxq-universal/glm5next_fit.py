#!/usr/bin/env python3
"""GLM-5.3-Flash (arch glm5next) PXQU size ledger + per-card VRAM fit for the PXA box.

Every number here comes from the REAL GGUF header of the unsloth UD-Q2_K_XL file
(a GGUF header dump, 676 tensors = blocks 0..22 plus the head and
the embeddings) with the MoE blocks 22..44 completed from their class template; the modelled
total lands within 0.12 GiB of the real 4-shard file size (101.25 GiB), which is the check
that the extrapolation is honest.

The output side applies this tree's actual rules: the glm5next keep-list
(src/pxa-glm5next-quant.h), the rev-2 PXQ backbone under the PXQ_UNIVERSAL tier (attn_k /
attn_v -> q8_0, every other eligible 2-D class -> pxq4hq), and the .tiers map on the routed
experts. The card side uses the measured overhead constants from the 2026-08-28/31 campaign.

  python3 glm5next_fit.py            # ledger + both card sets at -c 8192 and -c 32768
  python3 glm5next_fit.py --reserve 0.8

2026-09-08.
"""
import os
import argparse, collections, itertools, re, sys

GIB = 1024**3; MIB = 1024**2
HDR = os.environ.get('PXA_HDR_DUMP', './hdr-shard2.txt')

# ggml type traits (blck, type_size)
T = {'F32':(1,4),'F16':(1,2),'BF16':(1,2),'Q8_0':(32,34),'Q6_K':(256,210),'Q5_K':(256,176),
     'Q4_K':(256,144),'IQ4_XS':(256,136),'IQ3_XXS':(256,98),'IQ2_XS':(256,74)}
# PXQ slabs: 64-row panels, (fp16 anchor header bytes per panel, slab bytes per 64r x 32c)
PXQ = {'pxq1':(128,320),'pxq2':(128,576),'pxq3':(128,832),'pxq4':(128,1088),'pxq4hq':(128,1152)}

def tsz(t, ne):
    b, s = T[t]; n = 1
    for x in ne: n *= x
    return n // b * s

def pxq_size(t, K, R, E=1):
    h, s = PXQ[t]
    assert R % 64 == 0 and K % 32 == 0, (t, K, R)
    return E * (R // 64) * (h + (K // 32) * s)

# ---------------------------------------------------------------- the model
def load_model():
    tensors = []
    for line in open(HDR):
        m = re.match(r'TENSOR (\S+)\s+(\S+)\s+\[([0-9, ]+)\]', line)
        if m: tensors.append((m.group(1), m.group(2), [int(x) for x in m.group(3).split(',')]))
    def tmpl(il):
        p = f'blk.{il}.'
        return [(n[len(p):], t, ne) for n, t, ne in tensors if n.startswith(p)]
    TM, TK = tmpl(3), tmpl(4)                      # MLA+MoE and KDA+MoE class templates
    is_mla = lambda il: il >= 3 and (il - 3) % 4 == 0
    have = {n for n, _, _ in tensors}
    full = list(tensors)
    for il in range(3, 45):
        for suf, t, ne in (TM if is_mla(il) else TK):
            nm = f'blk.{il}.{suf}'
            if nm not in have: full.append((nm, t, ne))
    # blk.45, the NextN/MTP head: an MLA+MoE block plus the usual deepseek MTP companions.
    # It lives in shard 4, which was not on disk when this was written, so the companion
    # shapes are the standard ones; only its total is used, and only as a disk figure.
    for suf, t, ne in TM: full.append((f'blk.45.{suf}', t, ne))
    full += [('blk.45.nextn.eh_proj.weight','Q8_0',[8192,4096]),
             ('blk.45.nextn.enorm.weight','F32',[4096]),
             ('blk.45.nextn.hnorm.weight','F32',[4096]),
             ('blk.45.nextn.shared_head_norm.weight','F32',[4096]),
             ('blk.45.nextn.shared_head_head.weight','Q4_K',[4096,154880]),
             ('blk.45.nextn.embed_tokens.weight','Q5_K',[4096,154880])]
    return full

KEEP_SUB = ['indexer.','indexer_compressor_','hc_attn_','hc_ffn_','ssm_f_a.weight',
            'ssm_f_b.weight','ssm_g_a.weight','ssm_g_b.weight','ssm_beta.weight','ssm_a',
            'ssm_dt.','ssm_norm','ssm_conv1d','attn_k_b.weight','attn_v_b.weight',
            'attn_kv_a_mqa.weight','attn_q_a.weight','attn_q_b.weight','exp_probs_b',
            'ffn_gate_inp.weight']
is_exp  = lambda n: n.endswith('_exps.weight')
def is_keep(n):
    if n.startswith('blk.45.'): return True          # NextN: TENSOR_SKIP at load without --mtp
    if n.endswith('_norm.weight') or n.endswith('_norm.bias') or n == 'output_norm.weight': return True
    if n in ('token_embd.weight','output.weight'): return True
    return any(s in n for s in KEEP_SUB)

def out_size(n, t, ne, exp_tier='pxq2'):
    if is_keep(n): return tsz(t, ne)          # the keep-list wins, incl. all of blk.45
    if is_exp(n):  return pxq_size(exp_tier, ne[0], ne[1], ne[2])
    if 'attn_k.weight' in n or 'attn_v.weight' in n: return tsz('Q8_0', ne)
    K = ne[0]; R = ne[1] if len(ne) > 1 else 1; E = ne[2] if len(ne) > 2 else 1
    if R % 64 == 0 and K % 32 == 0: return pxq_size('pxq4hq', K, R, E)
    return tsz('Q8_0', ne)

# ---------------------------------------------------------------- cards
P100 = 16270/1024.0          # nvidia-smi accounted total per card, GiB (16384 MiB nominal)
V100 = 16145/1024.0
# KV + recurrent state, one sequence, f16 KV:
#   KDA state 34x64x128x128x4B = 142.6 MiB (context-INDEPENDENT) ; KDA conv 34x3x3x8192x4B
#   MLA latent 11x512x2B/token ; indexer cache 11x3x128x4B/token
KV = {'8192': (142.6+10.1+ 88.0+132.0)/1024.0,
      '32768':(142.6+10.1+352.0+528.0)/1024.0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reserve', type=float, default=1.2, help='GiB kept free per card')
    a = ap.parse_args()

    full = load_model()
    lay_bb = collections.defaultdict(int); lay_exp = collections.defaultdict(int); nonlayer = 0
    src_keep = src_req = src_exp = 0
    for n, t, ne in full:
        m = re.match(r'blk\.(\d+)\.', n)
        if is_exp(n) and not n.startswith('blk.45.'):
            src_exp += tsz(t, ne)
        elif is_keep(n): src_keep += tsz(t, ne)
        else:            src_req  += tsz(t, ne)
        if not m: nonlayer += out_size(n, t, ne); continue
        il = int(m.group(1))
        (lay_exp if is_exp(n) else lay_bb)[il] += out_size(n, t, ne)
    BB = sum(lay_bb.values()); EX = sum(lay_exp.values())
    BB45 = lay_bb[45]; EX45 = lay_exp[45]          # NextN block: on disk, never on a card
    FILE = 9429859 + 49294975936 + 49949266048 + 9466399584

    print("=== SOURCE (unsloth UD-Q2_K_XL) ===")
    print(f"  kept at source (non-expert) {src_keep/GIB:8.3f} GiB")
    print(f"  requantised    (non-expert) {src_req /GIB:8.3f} GiB")
    print(f"  routed experts blk 3..44    {src_exp /GIB:8.3f} GiB")
    print(f"  modelled total {(src_keep+src_req+src_exp)/GIB:.3f} GiB vs real file {FILE/GIB:.3f} GiB "
          f"(delta {abs((src_keep+src_req+src_exp)-FILE)/GIB:.3f})")
    print()
    print("=== OUTPUT (uniform pxq2 experts) ===")
    print(f"  non-expert (incl. head + embeddings)           {(BB+nonlayer)/GIB:8.3f} GiB")
    print(f"  routed experts (blk 3..44 pxq2, blk.45 source) {EX/GIB:8.3f} GiB")
    print(f"  FILE ON DISK                                   {(BB+EX+nonlayer)/GIB:8.3f} GiB")
    print(f"    of which blk.45 NextN, never allocated        {(BB45+EX45)/GIB:8.3f} GiB")
    print(f"    -> RESIDENT backbone {(BB+nonlayer-BB45)/GIB:.3f} GiB + experts {(EX-EX45)/GIB:.3f} GiB")
    for tier in ('pxq1','pxq2','pxq3'):
        g = pxq_size(tier,4096,2048,288); d = pxq_size(tier,2048,4096,288)
        print(f"    experts blk 3..44 all-{tier}: {42*(2*g+d)/GIB:7.2f} GiB "
              f"({(2*g+d)/GIB:.4f} GiB per block)")
    print()

    EXB = pxq_size('pxq2',4096,2048,288)*2 + pxq_size('pxq2',2048,4096,288)   # bytes per block
    KB  = {'ffn_up_exps':  pxq_size('pxq2',4096,2048,288)/GIB,
           'ffn_gate_exps':pxq_size('pxq2',4096,2048,288)/GIB,
           'ffn_down_exps':pxq_size('pxq2',2048,4096,288)/GIB}
    bbw = [lay_bb[i] for i in range(45)]; bbw[0] += nonlayer

    def plan(cards, names, ctx, label):
        n = len(cards); kv = KV[ctx]/n
        fixed = [(980 if i == 0 else 570)/1024.0 + 280/1024.0 + a.reserve + kv for i in range(n)]
        cap   = [cards[i] - fixed[i] for i in range(n)]
        best = None
        for cuts in itertools.combinations(range(1,45), n-1):
            segs = []; prev = 0
            for c in list(cuts)+[45]: segs.append((prev,c)); prev = c
            ok = True; res = 0.0; det = []
            for i,(s,e) in enumerate(segs):
                bbc = sum(bbw[s:e])/GIB
                if bbc > cap[i]: ok = False; break
                room = cap[i]-bbc; used = 0.0; keep = []
                for il in reversed([x for x in range(s,e) if x >= 3]):
                    for k in ('ffn_down_exps','ffn_gate_exps','ffn_up_exps'):
                        if used+KB[k] <= room: used += KB[k]; keep.append((il,k))
                res += used; det.append((s,e,bbc,used,keep,len([x for x in range(s,e) if x>=3])*3))
            if ok and (best is None or res > best[0]): best = (res, segs, det)
        res, segs, det = best
        host = []
        for (s,e,bbc,used,keep,nt) in det:
            kept = set(keep)
            for il in [x for x in range(s,e) if x >= 3]:
                for k in ('ffn_up_exps','ffn_gate_exps','ffn_down_exps'):
                    if (il,k) not in kept: host.append((il,k))
        print(f"--- {label} | -c {ctx} | reserve {a.reserve} GiB/card ---")
        print(f"    {'card':>5} {'total':>6} {'fixed':>6} {'cap':>6} {'layers':>8} {'bbone':>6} "
              f"{'expT':>7} {'weights':>8} {'headroom':>9}")
        for i,(s,e,bbc,used,keep,nt) in enumerate(det):
            print(f"    {names[i]:>5} {cards[i]:6.2f} {fixed[i]:6.2f} {cap[i]:6.2f} "
                  f"{str(s)+'-'+str(e-1):>8} {bbc:6.2f} {len(keep):3d}/{nt:<3d} {bbc+used:8.2f} "
                  f"{cap[i]-bbc-used+a.reserve:9.2f}")
        onc = (BB+nonlayer-BB45)/GIB + res
        print(f"    on card {onc:.2f} GiB | on host {(BB+EX+nonlayer-BB45-EX45)/GIB-onc:.2f} GiB "
              f"({len(host)} of 126 expert tensors = {len(host)/3:.1f} block-equivalents)")
        print(f"    -ts {','.join(str(e-s) for s,e in segs)}")
        byk = collections.defaultdict(list)
        for il,k in host: byk[k].append(il)
        for k in ('ffn_up_exps','ffn_gate_exps','ffn_down_exps'):
            if byk[k]:
                print(f"    -ot \"blk\\.({'|'.join(str(x) for x in sorted(byk[k]))})\\.{k}\\.weight=CPU\"")
        print()

    for ctx in ('8192','32768'):
        plan([P100]*4,               ['gpu0','gpu1','gpu5','gpu6'],
             ctx, "A) 4x P100 (0,1,5,6)")
        plan([P100]*4+[V100]*2,      ['gpu0','gpu1','gpu5','gpu6','gpu2','gpu4'],
             ctx, "B) 6 cards (4x P100 0,1,5,6 + 2x V100 2,4)")
    return 0

sys.exit(main())
