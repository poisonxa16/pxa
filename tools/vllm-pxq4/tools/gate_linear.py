#!/usr/bin/env python3
"""v7 linear_out parity: M<=8 == mmv_mono; 8<M<=16 == per-slice mmv_mono; M>16 == dequant+mm."""
import sys, torch
torch.ops.load_library(sys.argv[1])
print("op version:", torch.ops.pxq4.version())
dev="cuda:0"; torch.manual_seed(11); fails=0
for (name,panels,kslabs) in [("gate_up",136,160),("down",80,136),("o_proj",80,48)]:
    N,K = panels*64, kslabs*32
    slabs = torch.randint(0,256,(panels,kslabs,1088),dtype=torch.uint8,device=dev)
    anchor = (torch.randn(panels,64,device=dev)*0.05).half()
    for M in (1,2,5,8,9,12,16,17,64,477,2048):
        x=(torch.randn(M,K,device=dev)*0.1).half()
        out=torch.full((M,N),float("nan"),dtype=torch.float16,device=dev)
        torch.ops.pxq4.linear_out(out,x,slabs,anchor)
        ref=torch.empty_like(out)
        if M<=16:
            for r0 in range(0,M,8):
                rows=min(8,M-r0)
                torch.ops.pxq4.mmv_out_mono(ref[r0:r0+rows],x[r0:r0+rows].contiguous(),slabs,anchor)
        else:
            w=torch.empty(N,K,dtype=torch.float16,device=dev)
            torch.ops.pxq4.dequant_out(w,slabs,anchor)
            ref=torch.mm(x,w.t())
        torch.cuda.synchronize()
        ok=torch.equal(out.view(torch.int16),ref.view(torch.int16))
        print(f"{name} M={M} exact={ok}")
        if not ok: fails+=1
print("LINEAR GATE:","FAIL" if fails else "PASS"); sys.exit(1 if fails else 0)
