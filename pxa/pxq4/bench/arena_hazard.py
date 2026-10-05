#!/usr/bin/env python3
"""Deterministic reproduction of the PXQ4 shared-arena use-after-free (v12 -> v12b).

WHAT IS BEING TESTED. mmv_partials_arena / mmv_counter_arena are per-device scratch buffers
shared by every mmv path. A CUDA graph captured against one records the arena's RAW ADDRESS.
Up to v12 the only rule was "do not grow during capture", so a LATER, LARGER request from a
DIFFERENT path -- made eagerly, which is legal under that rule -- reassigned the tensor and
returned the captured block to the caching allocator. Every replay of the earlier graph then
writes partials into, and reduces out of, memory the allocator has handed to somebody else.

HOW THE HAZARD IS MADE VISIBLE. Two independent signals, and the second is the decisive one:

  (1) the replayed graph's output vs a reference taken before the arena moved; and
  (2) a POISON tensor allocated, all-NaN, right after the arena grows. It is sized to the
      block the growth just freed, so the caching allocator's best-fit hands that exact block
      back. Nothing in a correct library may touch it. If the graph replay overwrites those
      NaNs, the graph is provably writing memory it does not own -- which is the defect,
      independent of whether the numbers it produced happened to look plausible.

Signal (1) alone is not sufficient: the wmma path writes its whole partial tile before
reducing it, so a poisoned arena can still yield the right answer while corrupting a
neighbour. Signal (2) has no such escape.

Usage:  python bench/arena_hazard.py --lib <path/to/libpxq4_sm70_vXX.so> [--replays 400]
Exit status is 0 only if every case passes.
"""
import argparse, os, sys, torch

BM, QK, SLAB, KSEG, CMAX = 64, 32, 1088, 4, 16
MMA_CTAS = int(os.environ.get("PXQ4_MMA_CTAS", "512"))
MMA_GSLAB = 4


def canon_nfix(kslabs):
    lim = max(1, min(kslabs // KSEG, CMAX))
    n = 1
    while n * 2 <= lim:
        n *= 2
    return n


def mma_part_floats(panels, kslabs):
    want = max(1, min(16, (MMA_CTAS + panels - 1) // panels))
    while want > 1 and kslabs // want < MMA_GSLAB:
        want -= 1
    return want * panels * BM * 16


def mt_part_floats(panels, kslabs, M):
    return max(M, 8) * panels * canon_nfix(kslabs) * KSEG * BM


def make(N, K, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    P, S = N // BM, K // QK
    slabs = torch.randint(0, 256, (P, S, SLAB), dtype=torch.uint8, device=dev, generator=g)
    anchor = torch.full((P, BM), 0.05, dtype=torch.float16, device=dev)
    return slabs, anchor, P, S


def capture(op, slabs, anchor, M, N, K, dev):
    """Warm eagerly (this is the call that sizes the arena), then capture one mmv_out."""
    x = (torch.randn(M, K, device=dev) * 0.1).half()
    out = torch.zeros(M, N, device=dev, dtype=torch.half)
    op.mmv_out(out, x, slabs, anchor)
    torch.cuda.synchronize()
    ref = out.clone()
    assert torch.isfinite(ref).all(), "eager reference is not finite; bad fixture"
    st = torch.cuda.Stream()                        # side-stream warmup, as vLLM's capture does
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            op.mmv_out(out, x, slabs, anchor)
    torch.cuda.current_stream().wait_stream(st)
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph):
        op.mmv_out(out, x, slabs, anchor)
    out.zero_()
    gph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref), "capture itself changed the value; fixture is unstable"
    return gph, x, out, ref


def poison(nfloats):
    """A NaN tensor sized to the block a growth just freed, so best-fit reuses that block."""
    t = torch.empty(nfloats, dtype=torch.float32, device="cuda")
    t.fill_(float("nan"))
    return t


def run_case(name, op, dev, shapes, cap_ms, grow_m, freed_floats_fn, replays, results):
    print(f"\n=== {name} ===", flush=True)
    N, K = shapes
    slabs, anchor, P, S = make(N, K, dev, seed=1234 + N)
    graphs = []
    for M in cap_ms:                                   # vLLM order: largest capture size first
        gph, x, out, ref = capture(op, slabs, anchor, M, N, K, dev)
        # x and out MUST stay referenced: a captured graph holds their raw addresses, and
        # letting the loop variable drop the previous size's x hands that buffer straight back
        # to the allocator -- the same class of use-after-free this test is about, injected by
        # the test itself. It cost one confusing run of case B (rows 8..15 of the M=16 replay
        # came back NaN on the FIXED library, from x, not from the arena).
        graphs.append((M, gph, out, ref, x))
        print(f"  captured M={M:<3} graph", flush=True)

    # The eager call from the OTHER path. Legal under the v12 rule (not inside capture).
    xg = (torch.randn(grow_m, K, device=dev) * 0.1).half()
    og = torch.zeros(grow_m, N, device=dev, dtype=torch.half)
    grow_err = None
    try:
        op.mmv_out(og, xg, slabs, anchor)
        torch.cuda.synchronize()
        print(f"  eager M={grow_m} call on the other path: completed", flush=True)
    except RuntimeError as e:
        grow_err = str(e).strip().split("\n")[0]
        print(f"  eager M={grow_m} call on the other path: REFUSED -> {grow_err[:150]}", flush=True)

    pois = [poison(freed_floats_fn(P, S))] if grow_err is None else []
    if pois:
        print(f"  poison: {pois[0].numel()} floats "
              f"({pois[0].numel()*4/2**20:.2f} MiB) of NaN at 0x{pois[0].data_ptr():x}", flush=True)

    ok = True
    for M, gph, out, ref, _x in graphs:
        for _ in range(replays):
            gph.replay()
        torch.cuda.synchronize()
        same = torch.equal(out, ref)
        nn = int(torch.isnan(out).sum())
        print(f"  replay x{replays} M={M:<3} output identical={same} nan_words={nn}", flush=True)
        ok &= same and nn == 0
    for i, t in enumerate(pois):
        live = int(torch.isnan(t).sum())
        clob = t.numel() - live
        print(f"  poison[{i}] survived: {live}/{t.numel()} still NaN, {clob} words CLOBBERED",
              flush=True)
        ok &= clob == 0
    results.append((name, ok, grow_err))
    print(f"  -> {name}: {'PASS' if ok else 'FAIL (arena hazard reproduced)'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--case", required=True, choices=["A", "B", "C"])
    ap.add_argument("--replays", type=int, default=400)
    a = ap.parse_args()
    torch.ops.load_library(a.lib)
    op = torch.ops.pxq4
    dev = "cuda:0"
    torch.cuda.set_device(0)
    print(f"lib={a.lib} case={a.case} version={op.version()} "
          f"PXQ4_MMV_MMA={os.environ.get('PXQ4_MMV_MMA','unset')}", flush=True)

    # ONE CASE PER PROCESS. The freeze latch the fix installs is per device and sticky for the
    # life of the process, exactly as a captured graph's claim on the arena is; running two
    # capture scenarios back to back in one process would have the second legitimately refused.
    results = []
    gate_up = (17408, 5120)
    if a.case == "A":
        # The shipped ordering, minimal form. M=8 routes to the wmma arm, whose need is
        # SHAPE-ONLY and small (0.56M floats); the M=4 warmup then routes to MT and asks for
        # ~16x more (8.9M floats), which under v12 reallocates the arena the graph holds.
        run_case("A  gate_up  capture M=8 (mma), then eager M=4 (mt)", op, dev, gate_up,
                 [8], 4, lambda P, S: mma_part_floats(P, S), a.replays, results)
    elif a.case == "B":
        # The full vLLM decode-graph ordering: sizes captured LARGEST FIRST, then the M=4
        # warmup. Two live graphs are holding the block that the M=4 call frees.
        run_case("B  gate_up  capture M=16,M=8 (mma), then eager M=4 (mt)", op, dev, gate_up,
                 [16, 8], 4, lambda P, S: mma_part_floats(P, S), a.replays, results)
    else:
        # C: the fused arm, and with it the arrival-counter arena. N=640 K=12288 is the one
        # decode shape whose panels*M stays under the split/mono occupancy crossover at M=16,
        # so M=8 and M=16 both land on a fused kernel and the m_cap term (8 -> 16) doubles
        # BOTH the partials and the counters. A stale counter is the worse failure: no block
        # ever observes old == nfix-1, so out[] is never written at all.
        run_case("C  k12288   capture M=8 (fused), then eager M=16 (fused, m_cap 8->16)", op,
                 dev, (640, 12288), [8], 16, lambda P, S: mt_part_floats(P, S, 8), a.replays,
                 results)

    print("\n" + "=" * 78)
    bad = 0
    for name, ok, err in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("   [growth refused, loudly]" if err else ""))
        bad += 0 if ok else 1
    print(f"arena_hazard case {a.case}: {len(results)} cases, {bad} failing -> "
          f"{'FAIL' if bad else 'PASS'}")
    sys.exit(1 if bad else 0)


main()
