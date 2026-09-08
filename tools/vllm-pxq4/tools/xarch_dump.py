#!/usr/bin/env python3
"""Dump PXQ4 kernel outputs for cross-arch bit-exactness comparison.
usage: xarch_dump.py <libpath> <outfile.pt>
Inputs are generated on CPU with a fixed seed so they are identical on every device."""
import sys, torch
torch.ops.load_library(sys.argv[1])
dev = "cuda:0"
print("device:", torch.cuda.get_device_name(0), "op version:", torch.ops.pxq4.version())
shapes = [("tp4_gate_up",136,160),("tp4_down",80,136),("tp4_qkvz",64,160),("tp4_o_proj",80,48)]
g = torch.Generator().manual_seed(20260819)
res = {}
for name, panels, kslabs in shapes:
    N, K = panels*64, kslabs*32
    slabs_c = torch.randint(0,256,(panels,kslabs,1088),dtype=torch.uint8,generator=g)
    anchor_c = (torch.randn(panels,64,generator=g)*0.05).half()
    anchor_c.view(-1)[0] = 0.0
    slabs, anchor = slabs_c.to(dev), anchor_c.to(dev)
    # full-matrix dequant
    w = torch.empty(N, K, dtype=torch.float16, device=dev)
    torch.ops.pxq4.dequant_out(w, slabs, anchor)
    res[(name,"dequant")] = w.cpu()
    for M in (1,2,3,4,5,6,7,8):
        x_c = (torch.randn(M,K,generator=g)*0.1).half()
        x = x_c.to(dev)
        out_a = torch.full((M,N),float("nan"),dtype=torch.float16,device=dev)
        out_b = torch.full((M,N),float("nan"),dtype=torch.float16,device=dev)
        torch.ops.pxq4.mmv_out(out_a, x, slabs, anchor)
        torch.ops.pxq4.mmv_out_mono(out_b, x, slabs, anchor)
        torch.cuda.synchronize()
        res[(name,"mmv",M)] = out_a.cpu()
        res[(name,"mmv_mono",M)] = out_b.cpu()
    # sliced linear path M=12 (mmv slices) -- default SLICE_MAX
    x_c = (torch.randn(12,K,generator=g)*0.1).half()
    x = x_c.to(dev)
    out = torch.full((12,N),float("nan"),dtype=torch.float16,device=dev)
    torch.ops.pxq4.linear_out(out, x, slabs, anchor)
    torch.cuda.synchronize()
    res[(name,"linear12")] = out.cpu()
torch.save({str(k): v for k,v in res.items()}, sys.argv[2])
print("saved", sys.argv[2], "entries:", len(res))
