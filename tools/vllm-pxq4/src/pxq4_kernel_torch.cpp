// pxq4_kernel_torch.cpp — torch operator bindings for the PXQ4 sm_70 kernels.
//
// NAMESPACE. The library is `pxq4`, deliberately NOT `_C`: the host vLLM fork
// (github.com/KewaiiGamer/1Cat-vLLM) already owns `torch.ops._C` with 54 registered sm70 ops,
// and a second TORCH_LIBRARY(_C, ...) in the same process is a hard registration conflict.
//
// FROZEN ABI (plan §7.1) — components B and C agree on exactly this and nothing else:
//     pxq4::dequant_out(Tensor(a!) out, Tensor slabs, Tensor anchor) -> ()
//     pxq4::mmv_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()
//     pxq4::version() -> int
// The remaining entry points below are ADDITIVE setup/introspection helpers. They are eager-
// mode only and are never called from inside a captured region, so adding them cannot disturb
// the frozen contract.
//
// META / FAKE KERNELS ARE NOT REGISTERED HERE. Per plan §6.7 the fake implementations live in
// the runtime package (`src/pxq4_vllm/ops.py`, agent B) via torch.library.register_fake.
// Registering a Meta kernel here as well would make that call raise on a duplicate
// registration, so this TU registers CUDA implementations only. The two ops mutate their
// first argument and return nothing, so the fakes are shape checks with no return value:
//
//     torch.ops.load_library(_LIB)          # must happen before register_fake
//
//     @torch.library.register_fake("pxq4::dequant_out")
//     def _(out, slabs, anchor):
//         torch._check(out.shape == (slabs.shape[0] * 64, slabs.shape[1] * 32))
//         return None
//
//     @torch.library.register_fake("pxq4::mmv_out")
//     def _(out, x, slabs, anchor):
//         torch._check(x.shape[1] == slabs.shape[1] * 32)
//         torch._check(out.shape == (x.shape[0], slabs.shape[0] * 64))
//         return None
//
// The `Tensor(a!)` annotations in the schemas below are what tell the functionalisation pass
// that `out` is written in place; without them torch.compile silently drops the call.
//
// CAPTURE SAFETY. dequant_out and mmv_out allocate nothing, launch one kernel each, use only
// preallocated caller memory and static/dynamic shared memory, and never synchronise or read
// a device value on the host. They are cuda-graph-capture safe. set_tables is NOT: it is a
// cudaMemcpyToSymbol and must be called once, eagerly, at load time.

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

#include "pxq4_kernel_launch.h"
#include "pxq4_kernel_tables.h"

namespace {

// Default token-count ceiling for the mmv path. Mirrors the engine's PXA_PXQ4_2D_MAX_NY
// (ggml-cuda.cu:4019-4021, default 8). Above it the weight-reread cost of the mmv (grid.y is
// the token axis, so each block reads its whole panel once per token) loses to
// dequant + cuBLAS. The Python side owns the actual policy; this is the vendored default.
constexpr int64_t kPxq4MmvMaxM = 8;

struct Geom {
    int panels;
    int kslabs;
    int64_t N;
    int64_t K;
};

Geom check_weight(const at::Tensor & slabs, const at::Tensor & anchor) {
    TORCH_CHECK(slabs.is_cuda() && anchor.is_cuda(), "pxq4: slabs/anchor must be CUDA tensors");
    TORCH_CHECK(slabs.scalar_type() == at::kByte, "pxq4: slabs must be uint8, got ", slabs.scalar_type());
    TORCH_CHECK(anchor.scalar_type() == at::kHalf, "pxq4: anchor must be float16, got ", anchor.scalar_type());
    TORCH_CHECK(slabs.dim() == 3, "pxq4: slabs must be [panels, kslabs, 1088], got dim ", slabs.dim());
    TORCH_CHECK(anchor.dim() == 2, "pxq4: anchor must be [panels, 64], got dim ", anchor.dim());
    TORCH_CHECK(slabs.size(2) == PXQ4_SLAB_BYTES, "pxq4: slab stride must be ", PXQ4_SLAB_BYTES,
                ", got ", slabs.size(2));
    TORCH_CHECK(anchor.size(1) == PXQ4_BM, "pxq4: anchor row must be ", PXQ4_BM, ", got ", anchor.size(1));
    TORCH_CHECK(slabs.size(0) == anchor.size(0),
                "pxq4: panel count mismatch: slabs ", slabs.size(0), " vs anchor ", anchor.size(0));
    // Contiguity is not cosmetic: the vendored code does a 16-byte uint4 load at
    // slab_base + 64 + 16*row, which is only guaranteed aligned when the slab stride is the
    // natural 1088 and the base comes from a torch allocation. A narrow()ed, non-contiguous
    // view would silently misalign it.
    TORCH_CHECK(slabs.is_contiguous(), "pxq4: slabs must be contiguous");
    TORCH_CHECK(anchor.is_contiguous(), "pxq4: anchor must be contiguous");
    TORCH_CHECK(slabs.size(0) > 0 && slabs.size(1) > 0, "pxq4: empty weight");

    Geom g;
    g.panels = (int)slabs.size(0);
    g.kslabs = (int)slabs.size(1);
    g.N = (int64_t)g.panels * PXQ4_BM;
    g.K = (int64_t)g.kslabs * PXQ4_QK;
    return g;
}

void dequant_out(at::Tensor & out, const at::Tensor & slabs, const at::Tensor & anchor) {
    const Geom g = check_weight(slabs, anchor);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "pxq4: out must be a CUDA float16 tensor");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == g.N && out.size(1) == g.K,
                "pxq4: out must be [", g.N, ", ", g.K, "], got [", out.size(0), ", ",
                out.dim() == 2 ? out.size(1) : -1, "]");
    TORCH_CHECK(out.is_contiguous(), "pxq4: out must be contiguous");
    TORCH_CHECK(out.device() == slabs.device(), "pxq4: out and slabs on different devices");

    pxq4_launch_dequant_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), out.data_ptr(),
                            g.panels, g.kslabs, at::cuda::getCurrentCUDAStream());
}


// Persistent per-device fp32 partials arena for the split mmv. Grown ONLY outside CUDA graph
// capture; steady-state (and in-capture) calls are allocation-free. The first v2 TP=4 boot
// allocated the partials with a per-call at::empty instead, and the alloc/free churn inside
// the FULL decode-graph capture broke the fork's custom-allreduce graph-buffer registration
// (cudaIpcGetMemHandle returned 'invalid argument' at custom_all_reduce.cuh:976 on all four
// ranks, right after the capture bar hit 2/2). With the arena, the split path performs ZERO
// in-capture allocations -- exactly the allocation behaviour of the v1 monolithic kernel,
// which captured cleanly in the same config. Sizing uses max(M, mmv_max_m) so the eager
// pre-capture warmup vLLM always runs (every layer, decode-shaped M) warms the arena past
// anything a captured decode graph can need; if a capture would still outgrow it, we refuse
// loudly instead of allocating.
at::Tensor & mmv_partials_arena(const at::Tensor & like, int64_t need_floats) {
    static std::array<at::Tensor, 64> cache;  // one slot per CUDA device index
    const int64_t di = (int64_t)like.get_device();
    TORCH_CHECK(di >= 0 && di < (int64_t)cache.size(), "pxq4: bad device index ", di);
    at::Tensor & t = cache[(size_t)di];
    if (!t.defined() || t.numel() < need_floats) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &st);
        TORCH_CHECK(st == cudaStreamCaptureStatusNone,
                    "pxq4: mmv partials arena would have to grow (to ", need_floats,
                    " floats) during CUDA graph capture. Every PXQ4 layer must run eagerly "
                    "once before capture (vLLM's pre-capture warmup does this); hitting this "
                    "means the warmup skipped a layer or M exceeded mmv_max_m inside a graph.");
        t = at::empty({need_floats}, like.options().dtype(at::kFloat));
    }
    return t;
}

// Persistent per-device ARRIVAL-COUNTER arena for the fused single-launch split mmv
// (k_pxq4_mmv_fused). M*panels unsigned words, ZERO on entry to every launch. A completed
// launch rearms its own slots, so this is zeroed exactly ONCE, at allocation, which the same
// not-during-capture invariant as the partials arena guarantees is outside graph capture.
//
// TWO PROPERTIES OF THIS BUFFER ARE LOAD-BEARING, and neither is enforceable in code here:
//   * ONE PXQ4 mmv IN FLIGHT PER DEVICE AT A TIME. Two concurrent mmv calls on different
//     streams would corrupt both the counters and part[]. This is NOT a new constraint -- the
//     partials arena above is already shared and reused across every PXQ4 module and has the
//     identical hazard -- but with the barrier it is a CORRECTNESS dependency, not only a
//     scratch-aliasing one.
//   * A LAUNCH TORN DOWN MID-FLIGHT LEAVES A COUNTER NON-ZERO, after which no block of that
//     (panel, token) ever observes old == nfix-1, out[] is never written, and the caller
//     silently consumes stale fp16. The failure is SILENT, not loud. Any error path that
//     abandons a launch must reset this arena (drop the tensor so the next call reallocates
//     and re-zeroes).
at::Tensor & mmv_counter_arena(const at::Tensor & like, int64_t need_words) {
    static std::array<at::Tensor, 64> cache;
    const int64_t di = (int64_t)like.get_device();
    TORCH_CHECK(di >= 0 && di < (int64_t)cache.size(), "pxq4: bad device index ", di);
    at::Tensor & t = cache[(size_t)di];
    if (!t.defined() || t.numel() < need_words) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &st);
        TORCH_CHECK(st == cudaStreamCaptureStatusNone,
                    "pxq4: mmv counter arena would have to grow (to ", need_words,
                    " words) during CUDA graph capture; see mmv_partials_arena for why every "
                    "PXQ4 layer must run eagerly once before capture.");
        t = at::zeros({need_words}, like.options().dtype(at::kInt));  // ZERO on entry
    }
    return t;
}

void mmv_out(at::Tensor & out, const at::Tensor & x, const at::Tensor & slabs,
             const at::Tensor & anchor) {
    const Geom g = check_weight(slabs, anchor);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "pxq4: x must be a CUDA float16 tensor");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "pxq4: out must be a CUDA float16 tensor");
    TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "pxq4: x and out must be 2D");
    TORCH_CHECK(x.size(1) == g.K, "pxq4: x K=", x.size(1), " does not match weight K=", g.K);
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == g.N,
                "pxq4: out must be [", x.size(0), ", ", g.N, "]");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "pxq4: x and out must be contiguous");
    TORCH_CHECK(x.device() == slabs.device() && out.device() == slabs.device(),
                "pxq4: tensors on different devices");
    TORCH_CHECK(pxq4_mmv_supported(g.kslabs),
                "pxq4: K=", g.K, " needs ", pxq4_mmv_smem_bytes(g.kslabs),
                " B of dynamic shared memory, which exceeds the 48 KiB sm_70 budget; "
                "use the dequant + GEMM path for this layer");

    const int64_t M = x.size(0);
    if (M == 0) return;
    TORCH_CHECK(M <= 65535, "pxq4: mmv grid.y limit exceeded, M=", M);

    // Deliberately NOT enforcing M <= kPxq4MmvMaxM here: the ceiling is a performance policy
    // owned by the Python caller, and a hard check would make the op unusable for the
    // crossover sweep that decides where the ceiling actually belongs (plan risk 4).

    // v6: MULTI-TOKEN fast path. For a decode batch of 2..8 tokens the per-token kernels
    // above re-read the whole weight tensor once per token, which is why served throughput
    // collapsed under concurrency. k_pxq4_mmv_fused_mt gives one block all M tokens of its
    // (chunk, panel): weight bytes are decoded once and folded into M accumulators, so the
    // step cost is ~flat in M. Values are per-token bit-identical to the monolithic kernel
    // (see the kernel's header comment; gated by hostsim + GPU parity).
    // Kill switch: PXQ4_MMV_MT=0 restores the v5 dispatch untouched.
    static const bool kMtEnabled = [] {
        const char * e = getenv("PXQ4_MMV_MT");
        return !(e && *e && atoll(e) == 0);
    }();
    {
        const int nfix_mt = pxq4_mmv_nfix(g.kslabs);
        if (kMtEnabled && M >= 2 && M <= 8 && nfix_mt >= 2 &&
            pxq4_mmv_mt_supported(g.kslabs, (int)M)) {
            const int64_t m_cap = std::max<int64_t>(M, kPxq4MmvMaxM);
            at::Tensor & part = mmv_partials_arena(
                x, m_cap * (int64_t)g.panels * nfix_mt * (int64_t)(PXQ4_MMV_KSEG * PXQ4_BM));
            at::Tensor & ctr = mmv_counter_arena(x, m_cap * (int64_t)g.panels);
            pxq4_launch_mmv_fused_mt_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(),
                                         x.data_ptr(), part.data_ptr<float>(),
                                         (unsigned *)ctr.data_ptr<int32_t>(), out.data_ptr(),
                                         (int)M, g.panels, g.kslabs, /*vecx=*/true,
                                         at::cuda::getCurrentCUDAStream());
            return;
        }
    }

    // Split-vs-mono dispatch: an OCCUPANCY rule, not a byte-count rule.
    //
    // What decides the winner is whether the MONOLITHIC grid -- panels*M blocks of 256
    // threads -- has enough blocks to fill the SMs. It does not depend on how many bytes the
    // tensor holds. The old rule (slab_bytes >= 8 MB) used byte count as a proxy and got the
    // model's out_proj/o_proj class wrong: 64 of the 240 PXQ4 modules touched per token
    // (48 linear_attn.out_proj + 16 self_attn.o_proj, panels=80 kslabs=48 nfix=8, 3.98 MB)
    // sat just under the threshold and shipped on mono at 12.5% occupancy.
    //
    // The stale comment this replaces recorded "TP4 o_proj, 4.2 MB, mono 21 us vs split
    // 32 us". That does not reproduce in ANY L2 regime. Re-measured on sm_70 with a 48 MB
    // L2-defeating working set (the o_proj slab is 3.98 MB against a 6 MB L2, so a
    // single-copy benchmark reads ~30% fast and flatters mono): mono 24.4 us vs split
    // 12.0 us. L2-warm, the regime that presumably produced the stale number: mono 17.7 us
    // vs split 10.4 us. Split wins by 1.70-2.03x in both. The 8 MB threshold was costing
    // ~0.79 ms per decode token.
    //
    // Crossover calibrated over 48 (shape, M) points on an 80-SM V100: every point with
    // panels*M <= 160 wins on split (worst 1.04x), panels*M = 256 is a coin flip, every
    // point with panels*M >= 272 wins on mono. 2*multiProcessorCount is the conservative end
    // of that interval and is device-derived rather than hard-coded, since the constant is a
    // property of this kernel's 256-thread blocks. Env-tunable for a different part.
    static const int64_t kMaxMonoBlocks = [] {
        if (const char * e = getenv("PXQ4_MMV_SPLIT_MAX_BLOCKS")) {
            if (*e) return (int64_t)atoll(e);
        }
        int dev = 0, sm = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            sm <= 0) {
            sm = 80;  // V100; only reached if the driver query fails
        }
        return (int64_t)(2 * sm);
    }();
    // Second veto, retained so the byte threshold can still force a shape back onto mono
    // without a rebuild. Defaults to 0 = inactive (the old default was 8 MB). NOTE for
    // operators: anyone with PXQ4_MMV_SPLIT_MIN_BYTES set in their environment today will
    // keep getting the old byte veto ON TOP of the new occupancy rule.
    static const int64_t kSplitMinBytes = [] {
        const char * e = getenv("PXQ4_MMV_SPLIT_MIN_BYTES");
        return e && *e ? (int64_t)atoll(e) : (int64_t)0;
    }();
    const int64_t slab_bytes = (int64_t)g.panels * g.kslabs * PXQ4_SLAB_BYTES;
    const int nfix = pxq4_mmv_nfix(g.kslabs);
    const bool use_split = nfix > 1 && slab_bytes >= kSplitMinBytes &&
                           (int64_t)g.panels * M <= kMaxMonoBlocks;
    if (use_split) {
        // K-chunk-split fast path: bit-identical to the monolithic kernel (see
        // k_pxq4_mmv_part) but with nfix-times the blocks, which is what keeps an 80-SM
        // V100 busy at decode on this model's small-N TP shapes. The fp32 partials come
        // from a persistent per-device arena warmed before capture (see
        // mmv_partials_arena above) so this path allocates nothing at steady state or
        // under CUDA graph capture.
        const int64_t m_cap = std::max<int64_t>(M, kPxq4MmvMaxM);
        at::Tensor & part = mmv_partials_arena(
            x, m_cap * (int64_t)g.panels * nfix * (int64_t)(PXQ4_MMV_KSEG * PXQ4_BM));
        // v4: single launch. The reduce runs in whichever block of a (panel, token) arrives
        // last, so the device drain + refill between the two kernels disappears and ~240
        // kernel nodes leave the decode graph. Values are unchanged -- the atomic is an
        // arrival counter, never an accumulator (see k_pxq4_mmv_fused).
        at::Tensor & ctr = mmv_counter_arena(x, m_cap * (int64_t)g.panels);
        TORCH_CHECK(nfix >= 2, "pxq4: fused split mmv needs nfix >= 2, got ", nfix);
        pxq4_launch_mmv_fused_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                                  part.data_ptr<float>(), (unsigned *)ctr.data_ptr<int32_t>(),
                                  out.data_ptr(),
                                  (int)M, g.panels, g.kslabs, /*vecx=*/true,
                                  at::cuda::getCurrentCUDAStream());
    } else {
        pxq4_launch_mmv_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                            out.data_ptr(), (int)M, g.panels, g.kslabs, /*vecx=*/true,
                            at::cuda::getCurrentCUDAStream());
    }
}

// The pre-split monolithic mmv, kept callable so the parity gate can assert the split path
// bit-identical against it on device, and as a one-line fallback if a future shape ever
// misbehaves. Same ABI as mmv_out.
void mmv_out_mono(at::Tensor & out, const at::Tensor & x, const at::Tensor & slabs,
                  const at::Tensor & anchor) {
    const Geom g = check_weight(slabs, anchor);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == g.K && x.scalar_type() == at::kHalf, "pxq4: bad x");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == x.size(0) && out.size(1) == g.N &&
                out.scalar_type() == at::kHalf, "pxq4: bad out");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "pxq4: x and out must be contiguous");
    TORCH_CHECK(pxq4_mmv_supported(g.kslabs), "pxq4: K too large for mmv smem budget");
    if (x.size(0) == 0) return;
    pxq4_launch_mmv_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                        out.data_ptr(), (int)x.size(0), g.panels, g.kslabs, /*vecx=*/true,
                        at::cuda::getCurrentCUDAStream());
}

// The v3 two-launch split, kept callable so the device parity gate can assert the fused path
// bit-identical against it (and against mmv_out_mono) on a real card. Do not delete: the fused
// path's only strong gate is a device differential -- the hostsim runs blocks sequentially and
// is structurally incapable of observing the barrier race.
void mmv_out_split2(at::Tensor & out, const at::Tensor & x, const at::Tensor & slabs,
                    const at::Tensor & anchor) {
    const Geom g = check_weight(slabs, anchor);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == g.K && x.scalar_type() == at::kHalf, "pxq4: bad x");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == x.size(0) && out.size(1) == g.N &&
                out.scalar_type() == at::kHalf, "pxq4: bad out");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "pxq4: x and out must be contiguous");
    TORCH_CHECK(pxq4_mmv_supported(g.kslabs), "pxq4: K too large for mmv smem budget");
    const int64_t M = x.size(0);
    if (M == 0) return;
    const int nfix = pxq4_mmv_nfix(g.kslabs);
    TORCH_CHECK(nfix >= 2, "pxq4: split mmv needs nfix >= 2, got ", nfix);
    const int64_t m_cap = std::max<int64_t>(M, kPxq4MmvMaxM);
    at::Tensor & part = mmv_partials_arena(
        x, m_cap * (int64_t)g.panels * nfix * (int64_t)(PXQ4_MMV_KSEG * PXQ4_BM));
    pxq4_launch_mmv_split_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                              part.data_ptr<float>(), out.data_ptr(),
                              (int)M, g.panels, g.kslabs, /*vecx=*/true,
                              at::cuda::getCurrentCUDAStream());
}

// Persistent per-device fp16 DEQUANT arena for linear_out's large-M path. Same growth
// contract as mmv_partials_arena: grows only outside CUDA-graph capture. vLLM's eager
// warmup runs every layer at every capture size before capture, so any M that can appear
// inside a graph has already sized the arena.
at::Tensor & dequant_arena(const at::Tensor & like, int64_t need_elems) {
    static std::array<at::Tensor, 64> cache;
    const int64_t di = (int64_t)like.get_device();
    TORCH_CHECK(di >= 0 && di < (int64_t)cache.size(), "pxq4: bad device index ", di);
    at::Tensor & t = cache[(size_t)di];
    if (!t.defined() || t.numel() < need_elems) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &st);
        TORCH_CHECK(st == cudaStreamCaptureStatusNone,
                    "pxq4: dequant arena would have to grow (to ", need_elems,
                    " fp16 elems) during CUDA graph capture; the eager pre-capture warmup "
                    "must touch every PXQ4 layer first.");
        t = at::empty({need_elems}, like.options().dtype(at::kHalf));
    }
    return t;
}

// v7: SINGLE-OP LINEAR DISPATCHER. out[M,N] = x[M,K] @ W^T with the small-M mmv vs
// large-M dequant+cuBLAS policy decided HERE, per call, in C++.
//
// WHY THIS OP EXISTS: the Python-side branch (linear.py apply(): `M <= mmv_max_m`) is
// traced ONCE by torch.compile per compile range, and this fork runs backed dynamic
// shapes with evaluate_guards=False, so whichever branch the tracer saw is baked for the
// entire range. That baked the per-token mmv into the (1,2048) prefill range and produced
// served prefill at exactly one full weight read per token (measured 229 tok/s, 4.37 ms/token,
// perfectly linear in prompt length, vs 3160 tok/s for the AWQ arm). A custom op is opaque
// to dynamo/inductor, so the policy runs at execution time no matter what was traced.
// Under CUDA-graph capture the branch is evaluated at capture time with the real M, which
// is exactly what the captured kernels should be.
void linear_out(at::Tensor & out, const at::Tensor & x, const at::Tensor & slabs,
                const at::Tensor & anchor) {
    const Geom g = check_weight(slabs, anchor);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "pxq4: x must be a CUDA float16 tensor");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "pxq4: out must be a CUDA float16 tensor");
    TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "pxq4: x and out must be 2D");
    TORCH_CHECK(x.size(1) == g.K, "pxq4: x K=", x.size(1), " does not match weight K=", g.K);
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == g.N,
                "pxq4: out must be [", x.size(0), ", ", g.N, "]");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "pxq4: x and out must be contiguous");
    const int64_t M = x.size(0);
    if (M == 0) return;
    static const int64_t kMaxM = [] {
        const char * e = getenv("PXQ4_MMV_MAX_M");
        return (e && *e) ? (int64_t)atoll(e) : kPxq4MmvMaxM;
    }();
    if (M <= kMaxM && pxq4_mmv_supported(g.kslabs)) {
        mmv_out(out, x, slabs, anchor);
        return;
    }
    // Medium batches (kMaxM < M <= kSliceMax): ceil(M/kMaxM) sliced mmv calls. Each slice is
    // one multi-token weight pass (~4.3 ms/slice/rank on this model at TP4), which beats the
    // dequant+GEMM floor (~38 ms of dequant traffic) up to roughly M ~ 64; default 16 is the
    // conservative, measured-safe end. Each token's fold is the canonical engine mmv fold,
    // bit-identical to a direct mmv_out on that row. Slices are sequential launches on one
    // stream, so the shared partials/counter arenas see one mmv in flight at a time.
    static const int64_t kSliceMax = [] {
        const char * e = getenv("PXQ4_MMV_SLICE_MAX");
        return (e && *e) ? (int64_t)atoll(e) : (int64_t)16;
    }();
    if (M <= kSliceMax && kMaxM > 0 && pxq4_mmv_supported(g.kslabs)) {
        for (int64_t r0 = 0; r0 < M; r0 += kMaxM) {
            const int64_t rows = std::min<int64_t>(kMaxM, M - r0);
            at::Tensor xs = x.narrow(0, r0, rows);
            at::Tensor os = out.narrow(0, r0, rows);
            mmv_out(os, xs, slabs, anchor);
        }
        return;
    }
    // Prefill / large-batch path: coalesced dequant into the fp16 arena, then cuBLAS HMMA.
    at::Tensor & wbuf = dequant_arena(x, g.N * g.K);
    at::Tensor w = wbuf.narrow(0, 0, g.N * g.K).view({g.N, g.K});
    pxq4_launch_dequant_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), w.data_ptr(),
                            g.panels, g.kslabs, at::cuda::getCurrentCUDAStream());
    at::mm_out(out, x, w.t());
}

// Debug twin of mmv_out with the float4 activation loads disabled. Same values, different
// load width; used to isolate an alignment fault from an arithmetic fault at G6/G8.
void mmv_out_scalar(at::Tensor & out, const at::Tensor & x, const at::Tensor & slabs,
                    const at::Tensor & anchor) {
    const Geom g = check_weight(slabs, anchor);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == g.K && x.scalar_type() == at::kHalf, "pxq4: bad x");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == x.size(0) && out.size(1) == g.N &&
                out.scalar_type() == at::kHalf, "pxq4: bad out");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "pxq4: x and out must be contiguous");
    if (x.size(0) == 0) return;
    pxq4_launch_mmv_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(), out.data_ptr(),
                        (int)x.size(0), g.panels, g.kslabs, /*vecx=*/false,
                        at::cuda::getCurrentCUDAStream());
}

// 2 = K-chunk-split mmv; 3 = capture-safe partials arena; 4 = single-launch fused split mmv;
// 5 = chunk-major grid + occupancy-based split/mono dispatch; 6 = multi-token fused mmv for
// decode batches 2..8 (weight bytes read once per step instead of once per token)
// 7 = linear_out single-op dispatcher (mmv-vs-dequant+GEMM policy runs in C++ per call,
// immune to torch.compile per-range branch baking)
int64_t version() { return 7; }

int64_t mmv_max_m() { return kPxq4MmvMaxM; }

bool mmv_supported(int64_t K) {
    if (K <= 0 || K % PXQ4_QK != 0) return false;
    return pxq4_mmv_supported((int)(K / PXQ4_QK));
}

int64_t mmv_smem_bytes(int64_t K) {
    TORCH_CHECK(K > 0 && K % PXQ4_QK == 0, "pxq4: K must be a positive multiple of ", PXQ4_QK);
    return pxq4_mmv_smem_bytes((int)(K / PXQ4_QK));
}

// Overwrite the device book / sublevel tables from the checkpoint's recorded values
// (gguf KVs pxa.pxq6.book / pxa.pxq6.sub, mirrored into config.json's quantization_config).
// The engine allows PXA_PXQ6_BOOK / PXA_PXQ6_SUB to override the frozen literals at quantize
// time, so a checkpoint is only self-describing if we honour what it recorded.
// EAGER ONLY: this is a cudaMemcpyToSymbol and must happen before any cuda-graph capture.
void set_tables(const at::Tensor & book, const at::Tensor & sub) {
    TORCH_CHECK(book.numel() == 16 && sub.numel() == 16, "pxq4: book and sub must have 16 entries");
    const at::Tensor b = book.to(at::kCPU).to(at::kFloat).contiguous();
    const at::Tensor s = sub.to(at::kCPU).to(at::kFloat).contiguous();
    pxq4_upload_tables(b.data_ptr<float>(), s.data_ptr<float>());
}

// Read the tables back as [2, 16] float32 on CPU: row 0 = book, row 1 = sub.
at::Tensor get_tables() {
    at::Tensor t = at::empty({2, 16}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    pxq4_download_tables(t.data_ptr<float>(), t.data_ptr<float>() + 16);
    return t;
}

// The frozen compile-time literals, so a caller can verify the checkpoint's tables against
// the values this .so was built with WITHOUT touching the device (plan §5.6 check 3).
at::Tensor builtin_tables() {
    static const float book[16] = PXQ4_BOOK_INIT;
    static const float sub[16]  = PXQ4_SUB16_INIT;
    at::Tensor t = at::empty({2, 16}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    std::memcpy(t.data_ptr<float>(),      book, sizeof(book));
    std::memcpy(t.data_ptr<float>() + 16, sub,  sizeof(sub));
    return t;
}

}  // namespace

TORCH_LIBRARY(pxq4, m) {
    m.def("dequant_out(Tensor(a!) out, Tensor slabs, Tensor anchor) -> ()");
    m.def("mmv_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("linear_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("mmv_out_scalar(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("mmv_out_mono(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("mmv_out_split2(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("version() -> int");
    m.def("mmv_max_m() -> int");
    m.def("mmv_supported(int K) -> bool");
    m.def("mmv_smem_bytes(int K) -> int");
    m.def("set_tables(Tensor book, Tensor sub) -> ()");
    m.def("get_tables() -> Tensor");
    m.def("builtin_tables() -> Tensor");
}

TORCH_LIBRARY_IMPL(pxq4, CUDA, m) {
    m.impl("dequant_out", &dequant_out);
    m.impl("mmv_out", &mmv_out);
    m.impl("linear_out", &linear_out);
    m.impl("mmv_out_scalar", &mmv_out_scalar);
    m.impl("mmv_out_mono", &mmv_out_mono);
    m.impl("mmv_out_split2", &mmv_out_split2);
}

// Device-independent entry points. CompositeExplicitAutograd puts them on every backend
// without claiming an autograd formula (these ops have no gradient and never will).
TORCH_LIBRARY_IMPL(pxq4, CompositeExplicitAutograd, m) {
    m.impl("version", &version);
    m.impl("mmv_max_m", &mmv_max_m);
    m.impl("mmv_supported", &mmv_supported);
    m.impl("mmv_smem_bytes", &mmv_smem_bytes);
    m.impl("builtin_tables", &builtin_tables);
    // set_tables/get_tables take CPU tensors (or none at all), so they cannot be dispatched
    // on the CUDA key by argument device -- they must be registered device-independently even
    // though their bodies touch the current CUDA device.
    m.impl("set_tables", &set_tables);
    m.impl("get_tables", &get_tables);
}
