#!/usr/bin/env python3
"""Generate a PXQ-UNIVERSAL tier map for the qwen4exp experts under a byte budget.

Lagrangian knapsack over the 144 routed-expert tensors, exactly the shape of the
house policy in pxa-bench/pxq-universal/recipes: minimise total weighted
reconstruction error subject to a hard byte budget, which comes out depth-graded
(late layers richer) with the down-projection favoured.

HONEST LIMIT: the existing recipes were solved against a measured sens.json.
No sensitivity data exists for this architecture yet, so the per-tensor weight
here is a documented PROXY (depth x kind), not a measurement. Regenerating this
map from a real imatrix/sensitivity sweep is expected to move assignments.
"""
import argparse, json, re, sys

N_LAYER = 48
KINDS   = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")

# ne0 (row length) per kind, from the tensor directory:
#   ffn_gate_exps 2560x640x512 ; ffn_up_exps 2560x640x512 ; ffn_down_exps 640x2560x512
NE0     = {"ffn_gate_exps": 2560, "ffn_up_exps": 2560, "ffn_down_exps": 640}
NPARAMS = 2560 * 640 * 512          # identical for all three kinds

# bpw = base + 16/K ; measured wrel from ggml.h (rng-42 lab protocol), lower is better
TIERS = {"pxq2": 2.25, "pxq3": 3.25, "pxq4": 4.25, "pxq6": 5.25}
WREL  = {"pxq2": 0.3020, "pxq3": 0.1435, "pxq4": 0.0696, "pxq6": 0.034301}

# Proxy sensitivity. down_exps writes straight back into the residual stream, so it
# is weighted above gate/up; deeper layers are weighted above shallow ones.
KIND_W  = {"ffn_down_exps": 1.30, "ffn_gate_exps": 1.00, "ffn_up_exps": 1.00}
def depth_w(il): return 1.0 + (il / (N_LAYER - 1))

def nbytes(kind, tier):
    return NPARAMS * (TIERS[tier] + 16.0 / NE0[kind]) / 8.0

GIB = 1024 ** 3

def solve(budget_bytes):
    items = [(il, k) for il in range(N_LAYER) for k in KINDS]
    order = ["pxq2", "pxq3", "pxq4", "pxq6"]
    # start everyone at the cheapest tier, then buy upgrades by best error-reduction per byte
    assign = {it: "pxq2" for it in items}
    spent  = sum(nbytes(k, "pxq2") for _, k in items)
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
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    assign, spent = solve(a.budget_gib * GIB)
    if assign is None:
        print(f"INFEASIBLE: even all-pxq2 needs {spent/GIB:.2f} GiB > {a.budget_gib} GiB", file=sys.stderr)
        return 1

    hist = {}
    for t in assign.values(): hist[t] = hist.get(t, 0) + 1
    total_params = NPARAMS * len(assign)
    avg_bpw = spent * 8 / total_params

    lines = [
        f"# PXQU tier map '{a.name}': qwen4exp routed experts, 48 layers x 3 tensors.",
        f"# expert budget {a.budget_gib:.2f} GiB -> {spent/GIB:.2f} GiB used, avg {avg_bpw:.3f} bpw, hist {hist}",
        f"# {a.note}" if a.note else "#",
        "# Solved by a Lagrangian knapsack on a PROXY sensitivity (depth x kind), not a",
        "# measured sens.json -- no sensitivity sweep exists for this arch yet. Regenerate",
        "# from a real imatrix before treating the assignment as final.",
        "# Consumed by llama-quantize --pxq-universal.",
    ]
    for il in range(N_LAYER):
        for k in KINDS:
            lines.append(rf"^blk\.{il}\.{k}\.weight$={assign[(il,k)]}")
    open(a.out, "w").write("\n".join(lines) + "\n")

    print(f"{a.name}: {spent/GIB:.2f} GiB, avg {avg_bpw:.3f} bpw, hist {hist}")
    for t in ("pxq2","pxq3","pxq4","pxq6"):
        ls = sorted({il for (il,k),v in assign.items() if v==t})
        if ls: print(f"  {t}: {len(ls)} layers touched  {ls[:8]}{'...' if len(ls)>8 else ''}")
    return 0

sys.exit(main())
