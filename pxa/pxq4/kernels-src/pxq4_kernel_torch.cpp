// pxq4_kernel_torch.cpp — torch operator bindings for the PXQ4 sm_70 kernels.
//
// NAMESPACE. The library is `pxq4`, deliberately NOT `_C`: the host vLLM fork
// (the Volta vLLM port) already owns `torch.ops._C` with 54 registered sm70 ops,
// and a second TORCH_LIBRARY(_C, ...) in the same process is a hard registration conflict.
//
// FROZEN ABI — components B and C agree on exactly this and nothing else:
//     pxq4::dequant_out(Tensor(a!) out, Tensor slabs, Tensor anchor) -> ()
//     pxq4::mmv_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()
//     pxq4::version() -> int
// The remaining entry points below are ADDITIVE setup/introspection helpers. They are eager-
// mode only and are never called from inside a captured region, so adding them cannot disturb
// the frozen contract.
//
// META / FAKE KERNELS ARE NOT REGISTERED HERE. Per plan §6.7 the fake implementations live in
// the runtime package (`src/pxq4_vllm/ops.py`, the vLLM plugin (ops.py/linear.py)) via torch.library.register_fake.
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
//
// The mmv paths do use per-device scratch arenas, and a captured graph records the arena's RAW
// ADDRESS. Refusing to grow an arena during capture is therefore only half the rule: growing
// one AFTER a capture frees the block that graph still writes to. See the arena block below
// (v12b) for the ordering that reached this in the shipped configuration, for why the arena is
// now sized from the SHAPE rather than from whichever path asked first, and for the
// freeze-on-first-capture guard that turns any residual case into a hard error.

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>

#include "pxq4_kernel_launch.h"
#include "pxq4_kernel_tables.h"
#include "pxq4_mma.h"

namespace {

// Default token-count ceiling for the mmv path. Mirrors the engine's PXA_PXQ4_2D_MAX_NY
// (ggml-cuda.cu:4019-4021, default 8). Above it the weight-reread cost of the mmv (grid.y is
// the token axis, so each block reads its whole panel once per token) loses to
// dequant + cuBLAS. The Python side owns the actual policy; this is the vendored default.
constexpr int64_t kPxq4MmvMaxM = 8;

// v14 SMALL-M MASTER SWITCH (2026-09-07). PXA_PXQ4_SMALLM=0 restores the v13
// routing exactly: tensor-core arm disarmed unless PXQ4_MMV_MMA=1, MMA floor M>=5, and the
// token-folded SIMT arm capped at M<=8. Default 1 = the new routing. It is read ONCE.
static bool pxa_pxq4_smallm() {
    static const bool v = [] {
        const char * e = getenv("PXA_PXQ4_SMALLM");
        return !(e && *e && atoll(e) == 0);
    }();
    return v;
}


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


// ---------------------------------------------------------------------------------------------
// PERSISTENT PER-DEVICE SCRATCH ARENAS (fp32 partials, arrival counters, fp16 dequant).
//
// Grown ONLY outside CUDA graph capture; steady-state (and in-capture) calls are
// allocation-free. The first v2 TP=4 boot allocated the partials with a per-call at::empty
// instead, and the alloc/free churn inside the FULL decode-graph capture broke the fork's
// custom-allreduce graph-buffer registration (cudaIpcGetMemHandle returned 'invalid argument'
// at custom_all_reduce.cuh:976 on all four ranks, right after the capture bar hit 2/2). With
// the arena, the split path performs ZERO in-capture allocations -- exactly the allocation
// behaviour of the v1 monolithic kernel, which captured cleanly in the same config.
//
// v12b -- USE-AFTER-FREE OF A CAPTURED ARENA BLOCK. Refusing growth *during* capture is not
// sufficient, and up to v12 it was the only rule. A CUDA graph does not own the memory it was
// captured against: it records raw device addresses. Reassigning the arena tensor AFTER a
// capture returns the old block to the caching allocator while every graph captured against it
// still writes partials into, and reduces out of, those addresses. The next eager tensor of a
// similar size is handed that block and the two silently share it. Under the fused
// last-arriver variant the counter arena has the same hole, with a worse symptom: a garbage
// counter means no block ever observes old == nfix-1, so out[] is never written at all and the
// caller consumes stale fp16 with no error anywhere.
//
// The ordering that reached this in the shipped configuration -- it needs no unusual shape and
// no missing warmup, only the arena being sized by whichever path asked first:
//
//   * vLLM captures decode batch sizes LARGEST FIRST (cudagraph_dispatcher.py builds the
//     size list with reverse=True) and does warm-then-capture per size.
//   * With PXQ4_MMV_MMA=1 the M=16 and M=8 sizes route to the tensor-core path, whose need is
//     SHAPE-ONLY (gate_up: 0.56M floats). Both graphs are captured holding that block, P1.
//   * The M=4 warmup then routes to the MT path, which sizes the SAME arena to
//     m_cap(8)*panels*nfix*KSEG*BM = 8.9M floats (35.6 MB) for gate_up. That is a growth, it
//     happens OUTSIDE capture, so the old rule permitted it: P1 is freed.
//   * Every subsequent replay of the M=8 / M=16 graphs now writes into freed memory.
//
// THE FIX IS TO TAKE THE SIZE OFF THE PATH AND PUT IT ON THE SHAPE. pxq4_arena_part_floats()
// below returns the MAXIMUM over every path that can serve (panels, kslabs, M) -- MT / split /
// fused / MoE and the wmma tensor-core path -- so the arena size no longer depends on which
// arm the dispatcher picked, only on the shape and on m_cap = max(M, kPxq4MmvMaxM). m_cap is
// non-increasing as M decreases, and vLLM warms and captures largest-M first, so the sequence
// of requests a decode graph set produces is monotonically NON-GROWING by construction. The
// M=16 request for gate_up is 17.8M floats (71.3 MB); every later M and every later path for
// that shape asks for less and is satisfied in place. Cost of the unified sizing is that the
// MMA-armed configuration now holds the SIMT-sized arena it would have held anyway with the
// tensor-core path disarmed -- no new peak for the library, only a new peak for that one arm.
//
// SECOND LINE OF DEFENCE, because "monotone by construction" is an argument about the caller.
// Each slot latches `frozen` the first time it is touched inside a capture, recording the
// data_ptr at that moment. After that:
//   * a request larger than the arena is a HARD ERROR naming the shape, never a silent
//     reallocation -- growth after any capture is exactly the defect above; and
//   * every call asserts the pointer still equals the recorded one, so a reallocation from any
//     future code path is caught at the first call rather than as wrong logits days later.
// The capture query is skipped once frozen, so the steady state costs one bool test. That
// matters: cudaStreamIsCapturing is a driver call (~2-4 us here), and running it on every mmv
// of every layer would be a real host-side tax on a process that never captures and therefore
// never freezes. Such a process seals the slot instead, after PXQ4_ARENA_SEAL_CALLS quiet
// requests -- at which point the arena is immutable for the rest of the process and no probe
// is needed. Below the M floor the tensor-core arm is never consulted at all, so an armed
// build and a disarmed one execute the identical instructions for M = 1..4 (bench/m1_route.py
// asserts this in one process, with the route flipped in place).
//
// ONE PXQ4 mmv IN FLIGHT PER DEVICE AT A TIME remains load-bearing and is not enforceable
// here: two concurrent mmv calls on different streams would corrupt both the counters and
// part[]. For the counter arena that is a CORRECTNESS dependency, not only scratch aliasing --
// a launch torn down mid-flight leaves a counter non-zero, after which no block of that
// (panel, token) observes old == nfix-1, out[] is never written, and the caller silently
// consumes stale fp16. Any error path that abandons a launch must reset that arena.
// ---------------------------------------------------------------------------------------------

// fp32 partial words the arena must hold to serve ANY path for this shape at this M.
// PATH-INDEPENDENT BY CONSTRUCTION -- this is the whole fix; every call site uses it and no
// call site computes its own need.
// v14: the M ceiling the arena must cover. With the small-M switch on, the token-folded SIMT
// arm serves up to M=16 in ONE launch (pxq4_mmv_mt_supported), so an M=16 request can arrive
// after capture on a card with no tensor cores. Sizing every request at 16 keeps the sequence
// non-growing by construction -- the property the whole freeze contract above rests on -- at
// the cost of one doubling of the partials peak (gate_up: 35.6 -> 71.3 MB, once per device).
static int64_t pxq4_arena_m_ceiling() {
    return pxa_pxq4_smallm() ? (int64_t)16 : kPxq4MmvMaxM;
}

static int64_t pxq4_arena_part_floats(int panels, int kslabs, int64_t M) {
    const int64_t m_cap = std::max<int64_t>(M, pxq4_arena_m_ceiling());
    const int64_t nfix  = std::max<int64_t>((int64_t)pxq4_mmv_nfix(kslabs), 1);
    // MT / split / fused / MoE: part[m_cap, panels, nfix, KSEG*BM].
    int64_t need = m_cap * (int64_t)panels * nfix * (int64_t)(PXQ4_MMV_KSEG * PXQ4_BM);
    // wmma tensor-core path: split-K partials, shape only.
    const int64_t mma = (int64_t)pxq4_mma_part_floats(panels, kslabs);
    if (mma > need) need = mma;
    return need;
}

// Arrival-counter words for the same (shape, M). Same rule: the max over every path, so the
// counter arena cannot be resized by a later path either.
static int64_t pxq4_arena_ctr_words(int panels, int64_t M) {
    return std::max<int64_t>(M, pxq4_arena_m_ceiling()) * (int64_t)panels;
}

// How many consecutive requests a slot may serve without needing to grow before it SEALS --
// freezes even though no capture has been observed. Without this the capture probe below would
// run on every call for the life of an eager-only process, and cudaStreamIsCapturing is a
// driver call, not a memory read: measured ~2-4 us per call on this box, against a 20-100 us
// kernel and ~240 PXQ4 modules per token. Sealing bounds that to the first few thousand calls
// and makes the steady state a bool test. 4096 is ~17 eager decode tokens on this model, and
// vLLM reaches its FIRST capture at ~240 calls (one warmup pass over the modules), so a
// graph-captured serve freezes on the capture long before the count is reached; the count only
// governs a process that never captures. 0 disables count sealing.
static int64_t pxq4_arena_seal_calls() {
    static const int64_t v = [] {
        const char * e = getenv("PXQ4_ARENA_SEAL_CALLS");
        return (e && *e) ? (int64_t)atoll(e) : (int64_t)4096;
    }();
    return v;
}

struct ArenaSlot {
    at::Tensor t;
    void *     pinned = nullptr;   // data_ptr recorded when the slot froze
    int64_t    quiet  = 0;         // consecutive requests served without growing
    bool       frozen = false;     // a capture touched this slot, or it sealed by count
};

// Common body for the three arenas. `zero` selects at::zeros (the counter arena must be ZERO
// on entry to every launch; a completed launch rearms its own slots, so it is zeroed exactly
// once, at allocation, which the not-during-capture invariant guarantees is outside capture).
static at::Tensor & pxq4_arena_get(std::array<ArenaSlot, 64> & cache, const char * what,
                                   const at::Tensor & like, at::ScalarType dtype, bool zero,
                                   int64_t need, const char * shape_note, int64_t a, int64_t b,
                                   int64_t m) {
    const int64_t di = (int64_t)like.get_device();
    TORCH_CHECK(di >= 0 && di < (int64_t)cache.size(), "pxq4: bad device index ", di);
    ArenaSlot & s = cache[(size_t)di];

    if (!s.frozen) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &st);
        if (st != cudaStreamCaptureStatusNone) {
            s.frozen = true;                                   // a graph is recording this address
            s.pinned = s.t.defined() ? s.t.data_ptr() : nullptr;
        }
        if (!s.t.defined() || s.t.numel() < need) {
            TORCH_CHECK(st == cudaStreamCaptureStatusNone,
                        "pxq4: the ", what, " arena on device ", di, " would have to grow to ",
                        need, " elements during CUDA graph capture (", shape_note, "=", a,
                        ",", b, " M=", m, "). Every PXQ4 layer must run eagerly once, at the "
                        "largest capture size, before capture; hitting this means the warmup "
                        "skipped a layer or a shape first appeared inside a graph.");
            s.t = zero ? at::zeros({need}, like.options().dtype(dtype))
                       : at::empty({need}, like.options().dtype(dtype));
            s.pinned = s.t.data_ptr();
            s.quiet  = 0;                                      // growth restarts the seal window
        } else {
            const int64_t seal = pxq4_arena_seal_calls();
            if (seal > 0 && ++s.quiet >= seal) {
                s.frozen = true;                               // sealed by quiescence
                s.pinned = s.t.data_ptr();
            }
        }
    } else {
        // FROZEN: a graph holds raw addresses into this block. Reallocating would free it
        // under that graph, so an oversized request is a hard error, never a silent regrow.
        TORCH_CHECK(s.t.defined() && s.t.numel() >= need,
                    "pxq4: the ", what, " arena on device ", di, " holds ",
                    s.t.defined() ? s.t.numel() : 0, " elements but this call needs ", need,
                    " (", shape_note, "=", a, ",", b, " M=", m, "). The arena is frozen -- a "
                    "CUDA graph has been captured against it, or it sealed after ",
                    pxq4_arena_seal_calls(), " quiet requests -- so growing it would free "
                    "memory a graph may still write to. Warm every PXQ4 shape eagerly at the "
                    "largest capture size before the first capture; PXQ4_ARENA_SEAL_CALLS "
                    "raises or (=0) disables the quiescence seal.");
    }
    // The captured graphs' addresses are only valid while this stays put.
    TORCH_CHECK(!s.frozen || s.pinned == nullptr || s.t.data_ptr() == s.pinned,
                "pxq4: the ", what, " arena on device ", di, " moved after a CUDA graph was "
                "captured against it (", s.pinned, " -> ", s.t.data_ptr(),
                "); every graph holding the old address is now writing freed memory.");
    return s.t;
}

// fp32 split-K partials, shared by every mmv path.
at::Tensor & mmv_partials_arena(const at::Tensor & like, int panels, int kslabs, int64_t M) {
    static std::array<ArenaSlot, 64> cache;
    return pxq4_arena_get(cache, "mmv partials", like, at::kFloat, /*zero=*/false,
                          pxq4_arena_part_floats(panels, kslabs, M),
                          "panels,kslabs", panels, kslabs, M);
}

// Arrival counters for the fused single-launch split mmv (k_pxq4_mmv_fused / _mt / moe).
at::Tensor & mmv_counter_arena(const at::Tensor & like, int panels, int64_t M) {
    static std::array<ArenaSlot, 64> cache;
    return pxq4_arena_get(cache, "mmv counter", like, at::kInt, /*zero=*/true,
                          pxq4_arena_ctr_words(panels, M),
                          "panels,-", panels, -1, M);
}

// ---------------------------------------------------------------------------------------------
// Dispatch observability. The banner makes the tensor-core arm provable from a serving log
// without a profiler; the tally makes it provable that it is actually carrying the decode
// traffic (and not, say, silently falling through because M never reaches the floor).
// ---------------------------------------------------------------------------------------------
enum { kDispMma = 0, kDispMt = 1, kDispN = 2 };
static const char * const kDispName[kDispN] = { "mma", "mt" };
static std::atomic<unsigned long long> g_disp_counts[kDispN];

// THE ARM MUST COST NOTHING ON THE ROUTE IT DOES NOT TAKE. Everything that decides the arm is
// resolved once and read as a bool/int64 afterwards; the dispatch hook below tests the plain
// integer `M >= kMmaMinM` FIRST, so a decode at M = 1..4 leaves the hook on an integer compare
// and executes byte-for-byte the same work as a build with the arm disarmed. In particular no
// getenv, no atomic, no device query and no allocation is reachable below the M floor --
// pxq4_mma_supported(), the only remaining driver call on the routed path, now caches the arch
// probe (pxq4_mma.cu) and is in any case unreachable at M < kMmaMinM by short-circuit.
//
// The flag is a mutable static rather than a const one so that mma_route_set() can flip it in
// process, which is what makes an M=1 A/B measurable without the confound of two separate
// processes. DIAGNOSTIC ONLY: eager mode, before any capture. A captured graph records the
// kernels the route chose at capture time; flipping the route afterwards does not re-record
// them, it only makes the eager and captured paths disagree.
static bool & pxq4_mma_route_flag() {
    static bool v = [] {
        const char * e = getenv("PXQ4_MMV_MMA");
        if (e && *e) return atoll(e) != 0;
        // v14: ARMED BY DEFAULT on the small-M switch. The arm is still a no-op on sm_60 --
        // pxq4_mma_supported() fails the arch probe there and the SIMT arms keep the work.
        return pxa_pxq4_smallm();
    }();
    return v;
}

static bool pxq4_mma_route_enabled() { return pxq4_mma_route_flag(); }

static int64_t pxq4_mma_min_m() {
    static const int64_t v = [] {
        const char * e = getenv("PXQ4_MMV_MMA_MIN_M");
        if (e && *e) return (int64_t)atoll(e);
        // v14 (2026-09-07): MEASURED, and the floor STAYS AT 5. The hypothesis was
        // that the two-slot global lookahead would drop the HMMA kernel's flat cost far
        // enough to move the crossover down to 2. It did not: the kernel is not global-latency
        // bound (bench/smallm_verify.py, V100, us at M=1/2/4/8/16, gate_up 95.9/107.0/167.7/
        // 170.0/175.8 before and 96.0/169.5/170.3/172.1/179.1 after), and a CTA-budget sweep
        // {256,512,1024,2048,4096} moves nothing either -- 512 is still the optimum and every
        // other value is worse, which rules out occupancy as well. Weighted over the real
        // rank-local call mix (gate_up 63, down 63, qkv/z 47, out_proj 63 calls per token) the
        // SIMT token-fold still wins M=4 by 3% (22.8 vs 23.5 ms) and M=2 by 40%, while HMMA
        // wins M=8 by 1.64x (23.9 vs 39.0 ms). So 5 is where the two curves actually cross.
        // What the kernel IS bound by is shared-memory wavefronts: wmma::load_matrix_sync
        // requires ldm to be a multiple of 8 halves, so every A and B fragment load is 4-way
        // bank conflicted by construction and no layout expressible through the wmma API can
        // fix it. Closing that needs raw mma.sync.m8n8k4 with a XOR-swizzled operand layout.
        return (int64_t)5;
    }();
    return v;
}

static void pxq4_mma_banner(int64_t min_m) {
    static const bool once = [min_m] {
        int dev = -1, major = 0, minor = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        fprintf(stderr, "pxq4: PXQ4_MMV_MMA: armed, M>=%lld -> tensor-core path "
                        "(sm_%d%d wmma m16n16k16, fp32 accumulate, split-K)\n",
                (long long)min_m, major, minor);
        return true;
    }();
    (void)once;
}

static void pxq4_dispatch_tally(int kind) {
    static const long long every = [] {
        const char * e = getenv("PXQ4_LOG_DISPATCH");
        return (e && *e) ? atoll(e) : 0LL;               // 0 = off; N = report every N
    }();
    if (every <= 0) return;
    const unsigned long long n =
        g_disp_counts[kind].fetch_add(1ull, std::memory_order_relaxed) + 1ull;
    if (n % (unsigned long long)every) return;
    fprintf(stderr, "pxq4: dispatch %s=%llu (mma=%llu mt=%llu)\n", kDispName[kind],
            (unsigned long long)n,
            g_disp_counts[kDispMma].load(std::memory_order_relaxed),
            g_disp_counts[kDispMt ].load(std::memory_order_relaxed));
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

    // v11: TENSOR-CORE multi-token path (sm_70 HMMA). ADDITIVE and OFF by default; set
    // PXQ4_MMV_MMA=1 to arm it. Above M ~ 4 the MT kernel stops being weight-stream bound and
    // becomes fp32-FMA bound -- its cost is linear in M from there (gate_up: 100 us at M=1,
    // 298 us at M=8) -- while the tensor-core kernel is flat in M because the same weight
    // stream feeds HMMA at 8x the fp32 rate. Measured V100-PCIe, all five decode shapes:
    //
    //     M      1      2      4      8     16        (gate_up N=17408 K=5120, us)
    //     MT   100    109    159    298    600
    //     MMA  169    169    170    171    177
    //
    // so the crossover is M = 4-5 and the ceiling is set HERE, not by the kernel: below M = 5
    // the MT kernel wins by up to 1.8x and must keep the work. At M = 8 this is 1.63-1.74x
    // across the shapes, at M = 16 (which the slice loop in linear_out reaches) 3.1-3.4x.
    //
    // NOT bit-identical to the MT kernel -- different accumulation order, by construction.
    // Both are fp32-accumulated over the whole of K; measured rel-L2 against an fp32
    // reference is 2.07e-4 (MT) vs 2.92e-4 (MMA), both dominated by the shared fp16 output
    // rounding. See pxq4_mma.cu's header for the full numerics argument and for why this is
    // not a repeat of PXQ4_GEMM2D (which carried an fp16 accumulator across K).
    const int64_t kMmaMinM = pxq4_mma_min_m();
    // Integer compare FIRST: M = 1..4 leaves this line without touching the flag, the tally,
    // the banner, the arena or any driver call. See pxq4_mma_route_flag()'s comment.
    if (M >= kMmaMinM && M <= 16 && pxq4_mma_route_enabled() &&
        pxq4_mma_supported(g.panels, g.kslabs, (int)M)) {
        // v12b: SIZE THE ARENA FOR THE SHAPE, NOT FOR THIS ARM. The wmma need is shape-only
        // and much smaller than the MT need (gate_up: 0.56M floats vs 8.9M), so asking for it
        // here was what let a later, larger MT request reallocate the arena out from under the
        // graphs captured on this line. pxq4_arena_part_floats takes the max over every path.
        at::Tensor & part = mmv_partials_arena(x, g.panels, g.kslabs, M);
        pxq4_mma_banner(kMmaMinM);
        pxq4_dispatch_tally(kDispMma);
        pxq4_launch_mma_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                            part.data_ptr<float>(), out.data_ptr(), (int)M, g.panels,
                            g.kslabs, at::cuda::getCurrentCUDAStream());
        return;
    }

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
        static const int64_t kMtMaxM = pxa_pxq4_smallm() ? (int64_t)16 : (int64_t)8;
        if (kMtEnabled && M >= 2 && M <= kMtMaxM && nfix_mt >= 2 &&
            pxq4_mmv_mt_supported(g.kslabs, (int)M)) {
            at::Tensor & part = mmv_partials_arena(x, g.panels, g.kslabs, M);
            at::Tensor & ctr = mmv_counter_arena(x, g.panels, M);
            pxq4_dispatch_tally(kDispMt);
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
        at::Tensor & part = mmv_partials_arena(x, g.panels, g.kslabs, M);
        // v4: single launch. The reduce runs in whichever block of a (panel, token) arrives
        // last, so the device drain + refill between the two kernels disappears and ~240
        // kernel nodes leave the decode graph. Values are unchanged -- the atomic is an
        // arrival counter, never an accumulator (see k_pxq4_mmv_fused).
        at::Tensor & ctr = mmv_counter_arena(x, g.panels, M);
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
    at::Tensor & part = mmv_partials_arena(x, g.panels, g.kslabs, M);
    pxq4_launch_mmv_split_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                              part.data_ptr<float>(), out.data_ptr(),
                              (int)M, g.panels, g.kslabs, /*vecx=*/true,
                              at::cuda::getCurrentCUDAStream());
}

static bool pxq4_gemm2d_route_enabled();

// Persistent per-device fp16 DEQUANT arena for linear_out's large-M path. N*K is shape-only,
// so no path can disagree about its size the way they did about the partials arena, but it is
// still shared across every PXQ4 layer and a bigger layer arriving after a capture would free
// the block that capture holds. Same slot type, same freeze-on-first-capture rule.
at::Tensor & dequant_arena(const at::Tensor & like, int64_t N, int64_t K) {
    static std::array<ArenaSlot, 64> cache;
    return pxq4_arena_get(cache, "dequant", like, at::kHalf, /*zero=*/false, N * K,
                          "N,K", N, K, /*m=*/-1);
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
    // v11: with the tensor-core path armed, 9 <= M <= 16 is ONE launch, not ceil(M/8) sliced
    // weight passes. The slice loop exists because the SIMT mmv re-reads the whole weight
    // tensor per slice; the MMA kernel is flat in M to 16, so slicing it would just pay the
    // 47 MB stream twice. Measured gate_up M=16: 177 us as one call vs 340 us as two slices
    // vs 600 us for the sliced SIMT path. mmv_out owns the M >= kMmaMinM policy and the
    // arena sizing; this branch only declines to slice.
    if (M > kMaxM && M <= 16 && pxq4_mma_route_enabled() &&
        pxq4_mma_supported(g.panels, g.kslabs, (int)M)) {
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
    // v14: the slice loop existed because every arm re-read the whole weight tensor per
    // slice. With the small-M switch on both arms are flat to M=16 -- HMMA on sm_70, the
    // token-folded SIMT tile on sm_60 -- so 9..16 is ONE weight pass, not two.
    if (pxa_pxq4_smallm() && M <= 16 && pxq4_mmv_supported(g.kslabs) &&
        pxq4_mmv_mt_supported(g.kslabs, (int)M)) {
        mmv_out(out, x, slabs, anchor);
        return;
    }
    if (M <= kSliceMax && kMaxM > 0 && pxq4_mmv_supported(g.kslabs)) {
        for (int64_t r0 = 0; r0 < M; r0 += kMaxM) {
            const int64_t rows = std::min<int64_t>(kMaxM, M - r0);
            at::Tensor xs = x.narrow(0, r0, rows);
            at::Tensor os = out.narrow(0, r0, rows);
            mmv_out(os, xs, slabs, anchor);
        }
        return;
    }
    // Prefill / large-batch path.
    if (pxq4_gemm2d_route_enabled()) {
        pxq4_launch_gemm2d_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                               out.data_ptr(), (int)M, g.panels, g.kslabs,
                               at::cuda::getCurrentCUDAStream());
        return;
    }
    // Coalesced dequant into the fp16 arena, then cuBLAS.
    at::Tensor & wbuf = dequant_arena(x, g.N, g.K);
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


// ---------------------------------------------------------------------------------------------
// v8: MoE expert-indexed mmv. out[S, N] = x[S, K] @ W[ids[s]]^T with ids ON DEVICE, so the
// FusedMoE path needs no host read of topk_ids and is legal under CUDA-graph capture. Rows
// with ids outside [0, E) are written as zeros (vLLM can emit -1 padding slots). Values per
// row are bit-identical to mmv_out on that expert's 2-D slice (see pxq4_moe_kernel.cuh).
// Dispatch mirrors mmv_out's occupancy rule: fused-split when the mono grid (panels*S blocks)
// cannot fill the SMs, mono otherwise. Arenas are the shared per-device ones; the Python side
// must warm them eagerly pre-capture at the largest capturable S (PXQ4MoEMethod.
// process_weights_after_loading does).
// ---------------------------------------------------------------------------------------------
struct MoeGeom {
    int E;
    int panels;
    int kslabs;
    int64_t N;
    int64_t K;
};

MoeGeom check_moe_weight(const at::Tensor & slabs, const at::Tensor & anchor) {
    TORCH_CHECK(slabs.is_cuda() && anchor.is_cuda(), "pxq4: moe slabs/anchor must be CUDA tensors");
    TORCH_CHECK(slabs.scalar_type() == at::kByte, "pxq4: moe slabs must be uint8");
    TORCH_CHECK(anchor.scalar_type() == at::kHalf, "pxq4: moe anchor must be float16");
    TORCH_CHECK(slabs.dim() == 4, "pxq4: moe slabs must be [E, panels, kslabs, 1088], got dim ",
                slabs.dim());
    TORCH_CHECK(anchor.dim() == 3, "pxq4: moe anchor must be [E, panels, 64], got dim ",
                anchor.dim());
    TORCH_CHECK(slabs.size(3) == PXQ4_SLAB_BYTES, "pxq4: moe slab stride must be ",
                PXQ4_SLAB_BYTES, ", got ", slabs.size(3));
    TORCH_CHECK(anchor.size(2) == PXQ4_BM, "pxq4: moe anchor row must be ", PXQ4_BM);
    TORCH_CHECK(slabs.size(0) == anchor.size(0) && slabs.size(1) == anchor.size(1),
                "pxq4: moe slabs/anchor expert or panel count mismatch");
    TORCH_CHECK(slabs.is_contiguous() && anchor.is_contiguous(),
                "pxq4: moe slabs/anchor must be contiguous");
    TORCH_CHECK(slabs.size(0) > 0 && slabs.size(1) > 0 && slabs.size(2) > 0,
                "pxq4: empty moe weight");
    MoeGeom g;
    g.E      = (int)slabs.size(0);
    g.panels = (int)slabs.size(1);
    g.kslabs = (int)slabs.size(2);
    g.N = (int64_t)g.panels * PXQ4_BM;
    g.K = (int64_t)g.kslabs * PXQ4_QK;
    return g;
}

void moe_mmv_out(at::Tensor & out, const at::Tensor & x, const at::Tensor & ids,
                 const at::Tensor & slabs, const at::Tensor & anchor) {
    const MoeGeom g = check_moe_weight(slabs, anchor);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "pxq4: x must be a CUDA float16 tensor");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "pxq4: out must be a CUDA float16 tensor");
    TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == at::kInt, "pxq4: ids must be a CUDA int32 tensor");
    TORCH_CHECK(x.dim() == 2 && out.dim() == 2 && ids.dim() == 1, "pxq4: x/out 2D, ids 1D");
    TORCH_CHECK(x.size(1) == g.K, "pxq4: x K=", x.size(1), " does not match moe weight K=", g.K);
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == g.N,
                "pxq4: out must be [", x.size(0), ", ", g.N, "]");
    TORCH_CHECK(ids.size(0) == x.size(0), "pxq4: ids rows ", ids.size(0),
                " != x rows ", x.size(0));
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && ids.is_contiguous(),
                "pxq4: x/out/ids must be contiguous");
    TORCH_CHECK(x.device() == slabs.device() && out.device() == slabs.device() &&
                ids.device() == slabs.device(), "pxq4: tensors on different devices");
    TORCH_CHECK(pxq4_mmv_supported(g.kslabs),
                "pxq4: moe K=", g.K, " exceeds the mmv smem budget");
    const int64_t S = x.size(0);
    if (S == 0) return;
    TORCH_CHECK(S <= 65535, "pxq4: moe mmv grid limit exceeded, S=", S);

    static const int64_t kMaxMonoBlocksMoe = [] {
        if (const char * e = getenv("PXQ4_MMV_SPLIT_MAX_BLOCKS")) {
            if (*e) return (int64_t)atoll(e);
        }
        int dev = 0, sm = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            sm <= 0) {
            sm = 80;
        }
        return (int64_t)(2 * sm);
    }();
    const int nfix = pxq4_mmv_nfix(g.kslabs);
    const bool use_split = nfix > 1 && (int64_t)g.panels * S <= kMaxMonoBlocksMoe;
    if (use_split) {
        at::Tensor & part = mmv_partials_arena(x, g.panels, g.kslabs, S);
        at::Tensor & ctr = mmv_counter_arena(x, g.panels, S);
        pxq4_launch_moe_mmv_fused_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(),
                                      x.data_ptr(), ids.data_ptr<int32_t>(),
                                      part.data_ptr<float>(),
                                      (unsigned *)ctr.data_ptr<int32_t>(), out.data_ptr(),
                                      (int)S, g.E, g.panels, g.kslabs, /*vecx=*/true,
                                      at::cuda::getCurrentCUDAStream());
    } else {
        pxq4_launch_moe_mmv_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                                ids.data_ptr<int32_t>(), out.data_ptr(),
                                (int)S, g.E, g.panels, g.kslabs, /*vecx=*/true,
                                at::cuda::getCurrentCUDAStream());
    }
}

// Mono-only twin, kept callable so the parity gate can assert the fused moe path
// bit-identical against it on device (same reason mmv_out_mono exists).
void moe_mmv_out_mono(at::Tensor & out, const at::Tensor & x, const at::Tensor & ids,
                      const at::Tensor & slabs, const at::Tensor & anchor) {
    const MoeGeom g = check_moe_weight(slabs, anchor);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == g.K && x.scalar_type() == at::kHalf, "pxq4: bad x");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == x.size(0) && out.size(1) == g.N &&
                out.scalar_type() == at::kHalf, "pxq4: bad out");
    TORCH_CHECK(ids.dim() == 1 && ids.size(0) == x.size(0) &&
                ids.scalar_type() == at::kInt && ids.is_cuda(), "pxq4: bad ids");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && ids.is_contiguous(),
                "pxq4: x/out/ids must be contiguous");
    TORCH_CHECK(pxq4_mmv_supported(g.kslabs), "pxq4: K too large for mmv smem budget");
    if (x.size(0) == 0) return;
    pxq4_launch_moe_mmv_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                            ids.data_ptr<int32_t>(), out.data_ptr(),
                            (int)x.size(0), g.E, g.panels, g.kslabs, /*vecx=*/true,
                            at::cuda::getCurrentCUDAStream());
}


// v9: plain-fp16 dense decode GEMV (sm_60 fast path for unquantized linears; cuBLAS
// gemv2T measured at 22.7% of the P100 decode step). M 1..8 only; larger M belongs to
// cuBLAS GEMM. No allocation, no host reads: capture-safe.
void f16_mmv_out(at::Tensor & out, const at::Tensor & x, const at::Tensor & w) {
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kHalf && w.dim() == 2 && w.is_contiguous(),
                "pxq4: w must be contiguous CUDA fp16 [N, K]");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 && x.is_contiguous(),
                "pxq4: x must be contiguous CUDA fp16 [M, K]");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf && out.dim() == 2 &&
                out.is_contiguous(), "pxq4: out must be contiguous CUDA fp16 [M, N]");
    const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(w.size(1) == K, "pxq4: f16 mmv K mismatch");
    TORCH_CHECK(out.size(0) == M && out.size(1) == N, "pxq4: f16 mmv out shape");
    TORCH_CHECK(K % 8 == 0, "pxq4: f16 mmv requires K % 8 == 0");
    if (M == 0) return;
    // POLICY LIVES HERE, per call, in C++ — the Python caller must have NO M branch:
    // under torch.compile with backed no-guard dynamic shapes a Python branch is baked
    // for the whole compile range (the linear_out lesson, re-learned when the V100 arm
    // died with "supports M 1..8, got 16" from a baked M<=8 branch).
    if (M <= 3) {
        pxa_launch_f16_mmv_mt(w.data_ptr(), x.data_ptr(), out.data_ptr(),
                              (int)M, (int)N, (int)K, at::cuda::getCurrentCUDAStream());
        return;
    }
    if (M <= 16) {
        // v10: smem-staged tile kernel; one weight read serves all M tokens without the
        // per-token __ldg re-reads that sagged v9 above M=3.
        pxa_launch_f16_mmv_smem(w.data_ptr(), x.data_ptr(), out.data_ptr(),
                                (int)M, (int)N, (int)K, at::cuda::getCurrentCUDAStream());
        return;
    }
    at::mm_out(out, x, w.t());   // prefill / large batch: cuBLAS
}


// v11: fused 4-bit prefill GEMM (engine k_pxq6_gemm_grouped, E==1). fp16 accumulation —
// default OFF everywhere; PXQ4_GEMM2D=1 routes linear_out's large-M branch through it on
// sm_60 ONLY (the engine measured -18.6% on sm_70). Standalone op kept for parity/bench.
void gemm2d_out(at::Tensor & out, const at::Tensor & x, const at::Tensor & slabs,
                const at::Tensor & anchor) {
    const Geom g = check_weight(slabs, anchor);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
                x.is_contiguous(), "pxq4: bad x");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf && out.dim() == 2 &&
                out.is_contiguous(), "pxq4: bad out");
    TORCH_CHECK(x.size(1) == g.K && out.size(0) == x.size(0) && out.size(1) == g.N,
                "pxq4: gemm2d shape mismatch");
    const int64_t M = x.size(0);
    if (M == 0) return;
    pxq4_launch_gemm2d_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                           out.data_ptr(), (int)M, g.panels, g.kslabs,
                           at::cuda::getCurrentCUDAStream());
}

static bool pxq4_gemm2d_route_enabled() {
    static const bool v = [] {
        const char * e = getenv("PXQ4_GEMM2D");
        if (!(e && *e && atoll(e) != 0)) return false;
        int dev = 0, maj = 0, min = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            return false;
        }
        if (!(maj == 6 && min == 0)) {
            fprintf(stderr, "pxq4: PXQ4_GEMM2D requested but device is sm_%d%d, not sm_60; "
                            "keeping dequant+cuBLAS (engine measured -18.6%% on sm_70)\n",
                    maj, min);
            return false;
        }
        return true;
    }();
    return v;
}

// 2 = K-chunk-split mmv; 3 = capture-safe partials arena; 4 = single-launch fused split mmv;
// 5 = chunk-major grid + occupancy-based split/mono dispatch; 6 = multi-token fused mmv for
// decode batches 2..8 (weight bytes read once per step instead of once per token)
// 7 = linear_out single-op dispatcher (mmv-vs-dequant+GEMM policy runs in C++ per call,
// immune to torch.compile per-range branch baking)
// 8 = moe_mmv_out expert-indexed mmv (device-resident topk ids; the FusedMoE path no longer
// host-syncs and is legal under CUDA-graph capture)
// 9 = f16_mmv_out plain-fp16 dense decode GEMV (sm_60 unquantized-linear fast path)
// 10 = smem-staged f16 mmv tile kernel for M 4..16 (fixes the M>=4 bandwidth sag)
// 11 = gemm2d_out fused 4-bit prefill GEMM (sm_60-only route, PXQ4_GEMM2D=1, default off)
// Direct tensor-core entry, same ABI as mmv_out, no policy: always takes the MMA path.
// Exists so the parity sweep can compare the two arms without env juggling, and so that
// `strings -a lib | grep '^mma_out$'` identifies a build that carries this path (the ops are
// registered through TORCH_LIBRARY string schemas, so nm will not find them -- see README).
void mma_out(at::Tensor & out, const at::Tensor & x, const at::Tensor & slabs,
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
    TORCH_CHECK(pxq4_mma_supported(g.panels, g.kslabs, (int)M),
                "pxq4: mma path does not admit M=", M, " panels=", g.panels,
                " kslabs=", g.kslabs, " (needs sm_70+ and 1 <= M <= 16)");
    at::Tensor & part = mmv_partials_arena(x, g.panels, g.kslabs, M);
    pxq4_launch_mma_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                        part.data_ptr<float>(), out.data_ptr(), (int)M, g.panels, g.kslabs,
                        at::cuda::getCurrentCUDAStream());
}

int64_t version() { return 11; }

int64_t mmv_max_m() { return kPxq4MmvMaxM; }

// DIAGNOSTIC, EAGER ONLY. Flips the tensor-core route in process so an A/B of a route the arm
// is NOT supposed to change (M = 1..4) can be measured in one process, against one fixture,
// one allocator state and one clock -- two processes cannot separate a route cost from a
// process-to-process difference. Returns the previous value. Must not be called after any
// CUDA graph capture: a graph records the kernels the route chose when it was captured.
bool mma_route_set(bool on) {
    const bool prev = pxq4_mma_route_flag();
    pxq4_mma_route_flag() = on;
    return prev;
}

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
    // ARCH GUARD (2026-09-05, reported ). The second upload targets
    // pxq4_mma_book_g / pxq4_mma_sub16_g, which live in pxq4_mma.cu -- a translation unit that
    // is compiled sm_70 ONLY, and correctly so: it uses wmma m16n16k16, which does not exist on
    // sm_60 and will not compile for it. On a Pascal card that module therefore has no device
    // image, the cudaMemcpyToSymbol returns "no kernel image is available for execution on the
    // device", and PXQ4_MMA_CHECK calls abort() -- the process dies with one stderr line, no
    // traceback, and nothing catchable from Python, AFTER the weights are already resident.
    //
    // It stayed latent only because every shipping Pascal recipe still loads a pre-v13 library,
    // which predates this TU entirely. The v13 libraries carry it, so any Pascal config that
    // moves to v13 AND whose checkpoint records a pxq4 book would have died at load.
    //
    // set_tables was the one caller that did not already ask whether the tensor-core path is
    // available; pxq4_mma_supported(1, 1, 1) is exactly pxq4_mma_arch_ok() with the shape
    // checks satisfied (pxq4_mma.cu:236-242), and pxq4_mma_arch_ok itself is static to that TU.
    // Skipping the upload on an arch that cannot run the kernel loses nothing: the tables it
    // feeds are read only by a kernel that will never launch there.
    if (pxq4_mma_supported(1, 1, 1)) {
        pxq4_mma_upload_tables(b.data_ptr<float>(), s.data_ptr<float>());
    }
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
    m.def("mma_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("linear_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("mmv_out_scalar(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("mmv_out_mono(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("mmv_out_split2(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("moe_mmv_out(Tensor(a!) out, Tensor x, Tensor ids, Tensor slabs, Tensor anchor) -> ()");
    m.def("moe_mmv_out_mono(Tensor(a!) out, Tensor x, Tensor ids, Tensor slabs, Tensor anchor) -> ()");
    m.def("f16_mmv_out(Tensor(a!) out, Tensor x, Tensor w) -> ()");
    m.def("gemm2d_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("version() -> int");
    m.def("mmv_max_m() -> int");
    m.def("mma_route_set(bool on) -> bool");
    m.def("mmv_supported(int K) -> bool");
    m.def("mmv_smem_bytes(int K) -> int");
    m.def("set_tables(Tensor book, Tensor sub) -> ()");
    m.def("get_tables() -> Tensor");
    m.def("builtin_tables() -> Tensor");
}

TORCH_LIBRARY_IMPL(pxq4, CUDA, m) {
    m.impl("dequant_out", &dequant_out);
    m.impl("mmv_out", &mmv_out);
    m.impl("mma_out", &mma_out);
    m.impl("linear_out", &linear_out);
    m.impl("mmv_out_scalar", &mmv_out_scalar);
    m.impl("mmv_out_mono", &mmv_out_mono);
    m.impl("mmv_out_split2", &mmv_out_split2);
    m.impl("moe_mmv_out", &moe_mmv_out);
    m.impl("moe_mmv_out_mono", &moe_mmv_out_mono);
    m.impl("f16_mmv_out", &f16_mmv_out);
    m.impl("gemm2d_out", &gemm2d_out);
}

// Device-independent entry points. CompositeExplicitAutograd puts them on every backend
// without claiming an autograd formula (these ops have no gradient and never will).
TORCH_LIBRARY_IMPL(pxq4, CompositeExplicitAutograd, m) {
    m.impl("version", &version);
    m.impl("mmv_max_m", &mmv_max_m);
    m.impl("mma_route_set", &mma_route_set);
    m.impl("mmv_supported", &mmv_supported);
    m.impl("mmv_smem_bytes", &mmv_smem_bytes);
    m.impl("builtin_tables", &builtin_tables);
    // set_tables/get_tables take CPU tensors (or none at all), so they cannot be dispatched
    // on the CUDA key by argument device -- they must be registered device-independently even
    // though their bodies touch the current CUDA device.
    m.impl("set_tables", &set_tables);
    m.impl("get_tables", &get_tables);
}
