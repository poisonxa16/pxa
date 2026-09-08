import sys, time, torch
sys.path.insert(0, "/c/pxq4-sm60/work")
from pascal_decode_attn import paged_decode_attention
torch.manual_seed(0)
dev = "cuda"
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

def ref(q, kc, vc, bt, lens, scale):
    S, H, D = q.shape
    KVH = kc.shape[2]; page = kc.shape[1]; n_rep = H // KVH
    out = torch.empty_like(q)
    for s in range(S):
        n = int(lens[s])
        nb = (n + page - 1) // page
        blocks = bt[s, :nb].to(torch.long)
        k = kc[blocks].reshape(-1, KVH, D)[:n].float()
        v = vc[blocks].reshape(-1, KVH, D)[:n].float()
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1); v = v.repeat_interleave(n_rep, dim=1)
        sc = torch.einsum("hd,jhd->hj", q[s].float(), k) * scale
        p = torch.softmax(sc, -1)
        out[s] = torch.einsum("hj,jhd->hd", p, v).to(q.dtype)
    return out

for (S, H, KVH, D, page, nb_total, lens_list) in [
    (1, 12, 2, 256, 784, 8, [37]),
    (1, 12, 2, 256, 784, 8, [783]),
    (1, 12, 2, 256, 784, 8, [800]),
    (1, 12, 2, 256, 784, 8, [1600]),
    (4, 12, 2, 256, 784, 16, [5, 784, 1000, 2350]),
    (2, 24, 4, 128, 16, 64, [1, 63]),
]:
    q = torch.randn(S, H, D, device=dev, dtype=torch.float16)
    kc = torch.randn(nb_total, page, KVH, D, device=dev, dtype=torch.float16)
    vc = torch.randn(nb_total, page, KVH, D, device=dev, dtype=torch.float16)
    maxb = (max(lens_list) + page - 1)//page
    bt = torch.randperm(nb_total, device=dev)[:S*maxb].reshape(S, maxb).to(torch.int32)
    lens = torch.tensor(lens_list, device=dev, dtype=torch.int32)
    out = torch.empty_like(q)
    scale = D ** -0.5
    paged_decode_attention(q, kc, vc, bt, lens, out, scale)
    torch.cuda.synchronize()
    r = ref(q, kc, vc, bt, lens, scale)
    err = (out.float() - r.float()).abs().max().item()
    rel = err / r.float().abs().max().item()
    print(f"S={S} H={H} KVH={KVH} D={D} page={page} lens={lens_list}: maxabs={err:.4g} rel={rel:.3g} {'OK' if rel < 3e-3 else 'FAIL'}")

# cudagraph capture safety
q = torch.randn(1, 12, 256, device=dev, dtype=torch.float16)
kc = torch.randn(8, 784, 2, 256, device=dev, dtype=torch.float16)
vc = torch.randn(8, 784, 2, 256, device=dev, dtype=torch.float16)
bt = torch.arange(8, device=dev, dtype=torch.int32).reshape(1, 8)
lens = torch.tensor([100], device=dev, dtype=torch.int32)
out = torch.empty_like(q)
scale = 256 ** -0.5
st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    for _ in range(3): paged_decode_attention(q, kc, vc, bt, lens, out, scale)
torch.cuda.current_stream().wait_stream(st)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    paged_decode_attention(q, kc, vc, bt, lens, out, scale)
for L in (100, 900, 3000):
    lens.fill_(L); out.zero_(); g.replay(); torch.cuda.synchronize()
    r = ref(q, kc, vc, bt, lens, scale)
    rel = (out.float()-r.float()).abs().max().item() / r.float().abs().max().item()
    print(f"cudagraph replay len={L}: rel={rel:.3g} {'OK' if rel < 3e-3 else 'FAIL'}")

# timing
lens.fill_(1000)
for _ in range(5): paged_decode_attention(q, kc, vc, bt, lens, out, scale)
torch.cuda.synchronize(); t0=time.time()
for _ in range(200): paged_decode_attention(q, kc, vc, bt, lens, out, scale)
torch.cuda.synchronize()
print(f"triton kernel: {(time.time()-t0)/200*1e6:.1f} us/call (S=1,H=12,D=256,ctx=1000)")
torch.cuda.synchronize(); t0=time.time()
for _ in range(50): ref(q, kc, vc, bt, lens, scale)
torch.cuda.synchronize()
print(f"torch reference: {(time.time()-t0)/50*1e6:.1f} us/call")
