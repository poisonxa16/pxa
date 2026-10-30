#!/usr/bin/env python3
"""usage: test_lenient_acceptance.py <stock rejection_sampler_utils.py> <patched rejection_sampler_utils.py>   (needs one CUDA GPU)

Flag off must match the stock module bit for bit; LOGR=-1e30 must accept every in-support draft; a finite LOGR must never accept fewer
than stock. Covers the standard greedy, standard sampled and DFlash2 sparse top-k paths."""
import importlib.util
import sys

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


orig, new = load("rs_orig", sys.argv[1]), load("rs_new", sys.argv[2])
dev, V, S = "cuda", 248320, 7
N = S + 1
torch.manual_seed(0)
fails = 0


def std_args(temp, seed, drafts, logits):
    ds = torch.zeros(N, dtype=torch.int64, device=dev)
    ds[1:] = drafts
    return (logits, None, ds, torch.tensor([0, N], dtype=torch.int32, device=dev),
            torch.arange(100, 100 + N, device=dev, dtype=torch.int64), torch.zeros(1, dtype=torch.int32, device=dev),
            torch.zeros(N, dtype=torch.int32, device=dev), torch.arange(N, dtype=torch.int32, device=dev),
            torch.tensor([temp], device=dev, dtype=torch.float32), torch.tensor([seed], device=dev, dtype=torch.int64), S)


def run_std(mod, flag, logr, args):
    mod.PXA_LENIENT, mod.PXA_LENIENT_LOGR = flag, logr
    sampled, num = mod.rejection_sample(*args)
    torch.cuda.synchronize()
    return sampled.clone(), num.clone()


for trial in range(40):
    logits = torch.randn(N, V, device=dev) * 3
    top2 = logits.topk(2, dim=1).indices
    drafts = top2[:S].gather(1, torch.randint(0, 2, (S,), device=dev)[:, None]).squeeze(1)
    for temp in (0.0, 0.6):
        a = std_args(temp, trial, drafts, logits)
        so, no = run_std(orig, False, 0.0, a)
        sl, nl = run_std(new, False, -1e30, a)
        if not (torch.equal(no, nl) and torch.equal(so[0, :int(no)], sl[0, :int(nl)])):
            fails += 1
            print("flag-off mismatch", trial, temp)
        if int(run_std(new, True, -1e30, a)[1]) != N:
            fails += 1
            print("accept-all failed", trial, temp)
        if int(run_std(new, True, -1.2, a)[1]) < int(no):
            fails += 1
            print("lenient accepted fewer", trial, temp)

K, DK = 20, 32
for trial in range(40):
    tl_ = torch.randn(N, K, device=dev).sort(dim=1, descending=True).values * 0.7
    tid = torch.stack([torch.randperm(1000, device=dev)[:K] for _ in range(N)])
    dlog = torch.randn(1, S, DK, device=dev)
    did = torch.stack([torch.randperm(1000, device=dev)[:DK] for _ in range(S)])[None]
    drafts = tid[:S].gather(1, torch.randint(0, 2, (S,), device=dev)[:, None]).squeeze(1)
    did[0, :, 0] = drafts
    ds = torch.zeros(N, dtype=torch.int64, device=dev)
    ds[1:] = drafts
    common = (tid, tl_, did, dlog, ds, torch.tensor([0, N], dtype=torch.int32, device=dev),
              torch.arange(100, 100 + N, device=dev, dtype=torch.int64), torch.zeros(1, dtype=torch.int32, device=dev),
              torch.tensor([0.6], device=dev), torch.tensor([0.95], device=dev), torch.tensor([trial], device=dev, dtype=torch.int64), S)
    res = {}
    for tag, mod, flag, logr in (("o", orig, False, 0.0), ("f", new, False, -1e30), ("a", new, True, -1e30), ("l", new, True, -1.2)):
        mod.PXA_LENIENT, mod.PXA_LENIENT_LOGR = flag, logr
        sampled, num = mod.dflash2_sparse_topk_rejection_sample(*common)
        torch.cuda.synchronize()
        res[tag] = (sampled.clone(), int(num))
    if not (res["o"][1] == res["f"][1] and torch.equal(res["o"][0][0, :res["o"][1]], res["f"][0][0, :res["f"][1]])):
        fails += 1
        print("sparse flag-off mismatch", trial)
    if res["l"][1] < res["o"][1]:
        fails += 1
        print("sparse lenient accepted fewer", trial)
print("fails", fails)
sys.exit(1 if fails else 0)
