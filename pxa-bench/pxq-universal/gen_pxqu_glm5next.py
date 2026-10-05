#!/usr/bin/env python3
"""Generate a PXQ-UNIVERSAL tier map for the GLM-5.3-Flash (arch glm5next) routed experts.

Same house policy as gen_pxqu_flashnext.py: a Lagrangian knapsack over the routed-expert
tensors, minimising weighted reconstruction error under a hard byte budget, which comes out
depth-graded (late layers richer) with the down-projection favoured.

Shapes are the REAL ones, read out of the unsloth UD-Q2_K_XL GGUF header with a raw parser:
  ffn_gate_exps  [4096, 2048, 288]      ffn_up_exps  [4096, 2048, 288]
  ffn_down_exps  [2048, 4096, 288]
42 MoE blocks (3..44; blocks 0..2 are dense, block 45 is the NextN head), 288 experts each.
That is 3 * 42 * 4096*2048*288 = 304.4 G parameters, 94.9% of the whole model.

HONEST LIMIT, same as the flashnext generator: the per-tensor weight below is a documented
PROXY (depth x kind), not a measured sens.json. No sensitivity sweep exists for this
architecture. Regenerate from a real imatrix before treating an assignment as final.
"""
import argparse, sys

LAYER_FIRST = 3      # first MoE block
LAYER_LAST  = 44     # last MoE block of the trunk (45 is NextN, quantised with -mtp only)
LAYERS  = list(range(LAYER_FIRST, LAYER_LAST + 1))
KINDS   = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")

NE0     = {"ffn_gate_exps": 4096, "ffn_up_exps": 4096, "ffn_down_exps": 2048}
NPARAMS = 4096 * 2048 * 288          # identical for all three kinds: 2.4159e9

# bpw = base + 16/K.  wrel measured on the rng-42 lab protocol, lower is better.
# pxq1 has NO measured wrel: 0.62 below is a PROXY (roughly 2x pxq2) so the knapsack has a
# usable slope. Any map that actually places pxq1 says so in its header.
TIERS = {"pxq1": 1.25, "pxq2": 2.25, "pxq3": 3.25, "pxq4": 4.25, "pxq6": 5.25}
WREL  = {"pxq1": 0.6200, "pxq2": 0.3020, "pxq3": 0.1435, "pxq4": 0.0696, "pxq6": 0.034301}

# down_exps writes straight back into the residual stream, so it outranks gate/up;
# deeper layers outrank shallow ones.
KIND_W  = {"ffn_down_exps": 1.30, "ffn_gate_exps": 1.00, "ffn_up_exps": 1.00}
def depth_w(il): return 1.0 + ((il - LAYER_FIRST) / (LAYER_LAST - LAYER_FIRST))

def nbytes(kind, tier):
    return NPARAMS * (TIERS[tier] + 16.0 / NE0[kind]) / 8.0

GIB = 1024 ** 3

def solve(budget_bytes, floor_tier):
    items = [(il, k) for il in LAYERS for k in KINDS]
    order = [t for t in ("pxq1", "pxq2", "pxq3", "pxq4", "pxq6")
             if TIERS[t] >= TIERS[floor_tier]]
    assign = {it: order[0] for it in items}
    spent  = sum(nbytes(k, order[0]) for _, k in items)
    if spent > budget_bytes:
        return None, spent
    while True:
        best, best_ratio = None, 0.0
        for it in items:
            il, k = it
            cur = order.index(assign[it])
            if cur + 1 >= len(order): continue
            nxt   = order[cur + 1]
            dcost = nbytes(k, nxt) - nbytes(k, assign[it])
            if spent + dcost > budget_bytes: continue
            gain  = (WREL[assign[it]] - WREL[nxt]) * KIND_W[k] * depth_w(il)
            r = gain / dcost
            if r > best_ratio:
                best, best_ratio = (it, nxt, dcost), r
        if best is None: break
        it, nxt, dcost = best
        assign[it] = nxt
        spent += dcost
    return assign, spent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gib", type=float, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--floor", default="pxq2", choices=list(TIERS))
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    assign, spent = solve(a.budget_gib * GIB, a.floor)
    if assign is None:
        print(f"INFEASIBLE: even all-{a.floor} needs {spent/GIB:.2f} GiB > {a.budget_gib} GiB",
              file=sys.stderr)
        return 1

    hist = {}
    for t in assign.values(): hist[t] = hist.get(t, 0) + 1
    total_params = NPARAMS * len(assign)
    avg_bpw = spent * 8 / total_params

    lines = [
        f"# PXQU tier map '{a.name}': glm5next routed experts, blocks {LAYER_FIRST}..{LAYER_LAST} x 3 tensors.",
        f"# expert budget {a.budget_gib:.2f} GiB -> {spent/GIB:.2f} GiB used, avg {avg_bpw:.3f} bpw, hist {hist}",
    ]
    if a.note: lines.append(f"# {a.note}")
    lines += [
        "# Shapes are the real ones from the unsloth UD-Q2_K_XL GGUF header.",
        "# Solved by a Lagrangian knapsack on a PROXY sensitivity (depth x kind), not a",
        "# measured sens.json -- no sensitivity sweep exists for glm5next yet. Regenerate",
        "# from a real imatrix before treating the assignment as final.",
    ]
    if "pxq1" in hist:
        lines.append("# WARNING: this map places pxq1, whose wrel is an unmeasured proxy here.")
    lines += [
        "# The precision-sensitive tensors (indexer.*, indexer_compressor_*, hc_*, ssm_a,",
        "# ssm_dt.bias, ssm_f_*, ssm_g_*, ssm_norm, attn_k_b, attn_v_b, *_norm) are NOT in",
        "# this map and must be left at source precision by the quantiser's keep list.",
        "# Consumed by llama-quantize --pxq-universal.",
    ]
    for il in LAYERS:
        for k in KINDS:
            lines.append(rf"^blk\.{il}\.{k}\.weight$={assign[(il,k)]}")
    open(a.out, "w").write("\n".join(lines) + "\n")

    print(f"{a.name}: {spent/GIB:.2f} GiB, avg {avg_bpw:.3f} bpw, hist {hist}")
    return 0

sys.exit(main())
