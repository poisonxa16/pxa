# device_gates — the gates that need a real card

The hostsim gates in `../` compile the REAL kernel source for the CPU and are the primary
bit-exactness oracle, but they run blocks STRICTLY SEQUENTIALLY. That makes them
structurally incapable of observing two things:

  * the chunk-major grid order, which is semantically a no-op when blocks run one at a time;
  * the fused kernel's arrival barrier, where the atomic degenerates to `++` and the fences
    become no-ops.

Both of those need a device. These two harnesses are that gate, and they are checked in so
the numbers in `docs/10-kernel-speed.md` are reproducible by anyone with a V100.

Build (inside the production image, no vLLM headers needed):

    nvcc -O3 -arch=sm_70 --expt-relaxed-constexpr -I../ pxq4_v5_gpu.cu   -o v5gpu
    nvcc -O3 -arch=sm_70 --expt-relaxed-constexpr -I../ pxq4_v5_graph.cu -o v5graph

Both `#include "pxq4_kernel.cu"` so the whole thing is ONE translation unit: there is exactly
one copy of `pxq4_book_g` / `pxq4_sub16_g`, and what gets measured is the REAL shipping
launcher, not a re-implementation of it.

## pxq4_v5_gpu.cu

    ./v5gpu parity            12 shapes x M in {1,2,3,4,8} x vecx in {1,0} = 120 combos
    ./v5gpu stress [N]        N fused launches per group, 8 groups (barrier race)
    ./v5gpu bench [M] [iters] [reps] [wsMB]

`parity` compares, per combo: the fp32 `part[]` of the v5 part kernel against an in-file
reconstruction of the v3 pre-swap part kernel as uint32; the fp16 output of the fused path
against BOTH `k_pxq4_mmv` (monolithic) and the two-launch split; and the arrival counters,
which must read back zero. Every buffer is poisoned before every call and the monolithic
output is asserted non-poison, so a kernel that writes nothing cannot pass.

`bench` rotates R distinct weight buffers so no launch finds its weights in the 6 MB L2 —
`wsMB` is the target working set (48 is the default and is the decode regime). Reusing one
buffer flatters the small shapes badly: o_proj is 3.98 MB and reads ~30% fast L2-warm.

## pxq4_v5_graph.cu

One decode token's PXQ4 linear work — 240 modules per rank at TP=4 — captured in a CUDA
graph and replayed, which is the only vehicle that shows how much of a per-kernel win
survives graph replay. It models two things the obvious version gets wrong:

  * 240 DISTINCT weight tensors (~2.9 GB). Reusing one buffer per shape leaves it L2-warm.
  * ONE shared partials arena and ONE shared counter arena, because `mmv_partials_arena` and
    `mmv_counter_arena` are single per-device tensors. Giving each module its own partials
    (~304 MB of working set) costs several ms per token and slanders the split path.

It prints all three policies — v3 shipping, grid-swap-only, and full v5 — so the win can be
decomposed.
