#!/usr/bin/env python3
"""GPU parity + timing for the v6 multi-token mmv. Run inside the serving image on ONE GPU.
usage: python gpu_gate_v6.py /path/to/libpxq4_sm70.so [--stress]
"""
import sys, time, torch
lib = sys.argv[1]
torch.ops.load_library(lib)
print("op version:", torch.ops.pxq4.version())
dev = "cuda:0"
shapes = [("tp4_gate_up",136,160),("tp4_down",80,136),("tp4_qkvz",64,160),("tp4_o_proj",80,48)]
Ms = [1,2,3,4,6,8]
torch.manual_seed(7)
fails = 0
for name,panels,kslabs in shapes:
    N, K = panels*64, kslabs*32
    slabs = torch.randint(0,256,(panels,kslabs,1088),dtype=torch.uint8,device=dev)
    anchor = (torch.randn(panels,64,device=dev)*0.05).half()
    anchor.view(-1)[0] = 0.0
    for M in Ms:
        x = (torch.randn(M,K,device=dev)*0.1).half()
        out_a = torch.full((M,N),float("nan"),dtype=torch.float16,device=dev)
        out_b = torch.full((M,N),float("inf"),dtype=torch.float16,device=dev)
        torch.ops.pxq4.mmv_out(out_a,x,slabs,anchor)
        torch.ops.pxq4.mmv_out_mono(out_b,x,slabs,anchor)
        torch.cuda.synchronize()
        exact = torch.equal(out_a.view(torch.int16), out_b.view(torch.int16))
        if not exact: fails += 1
        for _ in range(30): torch.ops.pxq4.mmv_out(out_a,x,slabs,anchor)
        torch.cuda.synchronize()
        t0=time.perf_counter(); iters=300
        for _ in range(iters): torch.ops.pxq4.mmv_out(out_a,x,slabs,anchor)
        torch.cuda.synchronize()
        dt=(time.perf_counter()-t0)/iters*1e6
        gbs = (panels*kslabs*1088)/ (dt*1e-6) / 1e9
        print(f"{name} M={M} exact={exact} t={dt:6.1f}us weightGB/s={gbs:6.0f}")
if "--stress" in sys.argv:
    # barrier stress: many back-to-back launches at M=2..8 on the biggest shape, spot-check parity
    name,panels,kslabs = shapes[0]
    N, K = panels*64, kslabs*32
    slabs = torch.randint(0,256,(panels,kslabs,1088),dtype=torch.uint8,device=dev)
    anchor = (torch.randn(panels,64,device=dev)*0.05).half()
    bad = 0
    for it in range(400):
        M = (it % 7) + 2
        x = (torch.randn(M,K,device=dev)*0.1).half()
        out_a = torch.empty(M,N,dtype=torch.float16,device=dev)
        torch.ops.pxq4.mmv_out(out_a,x,slabs,anchor)
        if it % 40 == 0:
            out_b = torch.empty_like(out_a)
            torch.ops.pxq4.mmv_out_mono(out_b,x,slabs,anchor)
            torch.cuda.synchronize()
            if not torch.equal(out_a.view(torch.int16), out_b.view(torch.int16)): bad += 1
    torch.cuda.synchronize()
    print("stress parity fails:", bad)
    fails += bad
print("GPU GATE:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
