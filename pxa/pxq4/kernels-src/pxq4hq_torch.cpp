// pxq4hq_torch.cpp -- torch op bindings for the PXQ4HQ tier.
//
// TORCH_LIBRARY_FRAGMENT, not TORCH_LIBRARY. The `pxq4` library is DEFINED in
// pxq4_kernel_torch.cpp, which is frozen; a fragment extends it from a third TU without
// touching that file or pxq23_torch.cpp, and without opening a second namespace (a second
// TORCH_LIBRARY on one name in one process is a hard registration conflict).
//
// The library name therefore stays `pxq4` while the tiers it serves are pxq2/pxq3/pxq4/pxq4hq.
// That is a WIRE-FORMAT name, exactly as the on-disk key suffix pxq4_slabs/pxq4_anchor is: the
// tier travels in the checkpoint's config.json, never in an op or parameter name.
//
// OP NAMING: one op per tier rather than one op with a `tier` argument, matching the pxq2/pxq3
// fragment. The frozen PXQ4 ABI is then literally untouched, the fake/meta kernels stay trivial
// shape checks, and a tier can never be passed as a runtime value that a captured graph baked
// in wrongly. This TU also keeps its OWN table ops (pxq4hq_set_tables / _get_book / _get_sub)
// rather than extending the shared pxq_set_book / pxq_set_sub: those upload the SUB16 LUT that
// every other tier shares, and PXQ4HQ's sub is a different table. One entry point, one table.
//
// WHAT THIS TIER DOES NOT HAVE, and why it is a refusal rather than a fallback. There is no
// pxq4hq_moe_mmv_out. PXQ4HQ exists on this engine to serve the ATTENTION block of a promoted
// tier profile -- the tier a policy buys attention UP to -- so its modules are dense LINEAR
// modules. An expert tensor at this tier is refused by the offline converter, with a message
// naming the missing op, so the decision is visible at conversion time rather than as a silent
// dequant-to-fp16 at serving time.

#include <torch/library.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

#include "pxq4hq_kernel_tables.h"
#include "pxq4hq_kernel_launch.h"
#include "pxq4hq_mma.h"

namespace {

// Default token-count ceiling for the SIMT mmv path, mirroring the engine's
// PXA_PXQ4_2D_MAX_NY (default 8). Above it the weight-reread cost of the mmv (grid.y is the
// token axis, so each block reads its whole panel once per token) loses to the other arms.
constexpr int64_t kPxq4hqMmvMaxM = 8;

// Library revision. Bumped when an op is added or its semantics change, so a Python side can
// require a floor rather than guessing from the .so file name.
constexpr int64_t kPxq4hqVersion = 1;

int64_t env_i64(const char * name, int64_t dflt) {
    const char * e = getenv(name);
    return (e && *e) ? (int64_t)atoll(e) : dflt;
}

struct Geom { int panels; int kslabs; int64_t N; int64_t K; };

Geom check_weight(const at::Tensor & slabs, const at::Tensor & anchor) {
    TORCH_CHECK(slabs.is_cuda() && anchor.is_cuda(), "pxq4hq: slabs/anchor must be CUDA tensors");
    TORCH_CHECK(slabs.scalar_type() == at::kByte, "pxq4hq: slabs must be uint8, got ", slabs.scalar_type());
    TORCH_CHECK(anchor.scalar_type() == at::kHalf, "pxq4hq: anchor must be float16, got ", anchor.scalar_type());
    TORCH_CHECK(slabs.dim() == 3, "pxq4hq: slabs must be [panels, kslabs, ", PXQ4HQ_SLAB_BYTES,
                "], got dim ", slabs.dim());
    TORCH_CHECK(anchor.dim() == 2, "pxq4hq: anchor must be [panels, 64], got dim ", anchor.dim());
    // THE TIER CHECK THAT MATTERS. A PXQ4 tensor handed to this op has a valid-looking shape in
    // every other respect -- same panel count, same K, 1088 instead of 1152 bytes per slab --
    // and would decode into well-formed garbage. The slab stride is what identifies the tier
    // and it is CHECKED, never inferred.
    TORCH_CHECK(slabs.size(2) == PXQ4HQ_SLAB_BYTES,
                "pxq4hq: this tier wants slab stride ", PXQ4HQ_SLAB_BYTES, ", got ",
                slabs.size(2), " -- this tensor is a different tier (pxq4 is 1088, pxq3 832, "
                "pxq2 576)");
    TORCH_CHECK(anchor.size(1) == 64, "pxq4hq: anchor row must be 64, got ", anchor.size(1));
    TORCH_CHECK(slabs.size(0) == anchor.size(0),
                "pxq4hq: panel count mismatch: slabs ", slabs.size(0), " vs anchor ", anchor.size(0));
    // Contiguity is not cosmetic: the code load is a 16-byte vector load whose alignment is
    // only guaranteed when the slab stride is the natural one and the base comes from a torch
    // allocation. A narrow()ed view would silently misalign it.
    TORCH_CHECK(slabs.is_contiguous(), "pxq4hq: slabs must be contiguous");
    TORCH_CHECK(anchor.is_contiguous(), "pxq4hq: anchor must be contiguous");
    TORCH_CHECK(slabs.size(0) > 0 && slabs.size(1) > 0, "pxq4hq: empty weight");

    Geom g;
    g.panels = (int)slabs.size(0);
    g.kslabs = (int)slabs.size(1);
    g.N = (int64_t)g.panels * 64;
    g.K = (int64_t)g.kslabs * 32;
    return g;
}

// ------------------------------------------------------------------ prefill dequant arena
//
// One persistent fp16 buffer per device, sized by the largest prefill this process has asked
// for and NEVER reassigned while a captured graph could hold its address. The rule and its
// failure mode are the v12b ones: a CUDA graph records raw device addresses, so replacing the
// tensor after a capture hands the old block back to the caching allocator while captured
// graphs still write into it. Prefill shapes are not captured on this stack (capture sizes are
// <= 8 tokens and take the mmv branch), so growth is legal here -- but it is refused DURING
// capture rather than trusted not to happen.
std::array<at::Tensor, 64> g_dequant_arena;

at::Tensor & dequant_arena(const at::Tensor & like, int64_t need) {
    const int dev = (int)like.device().index();
    TORCH_CHECK(dev >= 0 && dev < 64, "pxq4hq: device index out of range: ", dev);
    at::Tensor & a = g_dequant_arena[dev];
    if (!a.defined() || a.numel() < need) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &cap);
        TORCH_CHECK(cap == cudaStreamCaptureStatusNone,
                    "pxq4hq: the prefill dequant arena would have to grow during CUDA graph "
                    "capture (need ", need, " fp16 elems, have ",
                    a.defined() ? a.numel() : 0, "). Every shape a captured graph replays must "
                    "have run once eagerly first.");
        a = at::empty({need}, like.options().dtype(at::kHalf));
    }
    return a;
}

// ------------------------------------------------------------------- mma split-K partials
//
// SHAPE ONLY. pxq4hq_mma_part_floats(panels, kslabs) has no M term, so every CUDA-graph
// capture size asks for the identical allocation for a given module and the request sequence
// is non-growing by construction once the largest module has been seen. That is a stronger
// position than the PXQ4 arena is in (it must cover SIMT split/fused/MoE paths whose need
// scales with M), which is why the freeze contract here is the simple one: grow eagerly,
// refuse to grow under capture, and assert the pointer never moved after the first capture.
struct MmaArena {
    at::Tensor buf;
    bool       frozen = false;
    void *     ptr    = nullptr;
};
std::array<MmaArena, 64> g_mma_arena;

float * mma_partials(const at::Tensor & like, int64_t need) {
    const int dev = (int)like.device().index();
    TORCH_CHECK(dev >= 0 && dev < 64, "pxq4hq: device index out of range: ", dev);
    MmaArena & a = g_mma_arena[dev];
    if (!a.buf.defined() || a.buf.numel() < need) {
        TORCH_CHECK(!a.frozen,
                    "pxq4hq: the tensor-core split-K arena would have to grow after a CUDA "
                    "graph capture has already recorded its address (need ", need,
                    " fp32 words, have ", a.buf.defined() ? a.buf.numel() : 0,
                    "). Call pxq4hq_reserve(panels, kslabs) for every module before the first "
                    "capture.");
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &cap);
        TORCH_CHECK(cap == cudaStreamCaptureStatusNone,
                    "pxq4hq: the tensor-core split-K arena would have to grow DURING CUDA "
                    "graph capture (need ", need, " fp32 words). Every shape a captured graph "
                    "replays must have run once eagerly first.");
        a.buf = at::empty({need}, like.options().dtype(at::kFloat));
        a.ptr = a.buf.data_ptr();
    }
    // Latch on the first touch inside a capture, then assert the block never moved. "Monotone
    // by construction" is an argument about the caller; this is the check that does not depend
    // on one.
    if (!a.frozen) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &cap);
        if (cap != cudaStreamCaptureStatusNone) { a.frozen = true; a.ptr = a.buf.data_ptr(); }
    } else {
        TORCH_CHECK(a.buf.data_ptr() == a.ptr,
                    "pxq4hq: the tensor-core split-K arena moved after a capture recorded its "
                    "address -- every captured graph for this device is now writing into freed "
                    "memory. This is a bug in whatever reallocated it, not a recoverable state.");
    }
    return a.buf.data_ptr<float>();
}

// Route flag for the tensor-core arm, so an A/B does not need two libraries. Default ON where
// the arch admits it; pxq4hq_mma_supported() answers the arch probe once and caches it.
bool & mma_route_flag() {
    static bool v = [] {
        const char * e = getenv("PXQ4HQ_MMA");
        if (e && *e) return atoi(e) != 0;
        e = getenv("PXQ4_MMV_MMA");                 // one knob for both arena paths
        return (e && *e) ? atoi(e) != 0 : true;
    }();
    return v;
}

// --------------------------------------------------------------------------------- the ops
void pxq4hq_dequant_out(at::Tensor & out, const at::Tensor & slabs, const at::Tensor & anchor) {
    const Geom g = check_weight(slabs, anchor);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "pxq4hq: out must be a CUDA float16 tensor");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == g.N && out.size(1) == g.K,
                "pxq4hq: out must be [", g.N, ", ", g.K, "]");
    TORCH_CHECK(out.is_contiguous(), "pxq4hq: out must be contiguous");
    TORCH_CHECK(out.device() == slabs.device(), "pxq4hq: out and slabs on different devices");
    const at::cuda::OptionalCUDAGuard guard(at::device_of(slabs));
    pxq4hq_launch_dequant_f16(PXQ_TIER_PXQ4HQ, slabs.data_ptr<uint8_t>(), anchor.data_ptr(),
                              out.data_ptr(), g.panels, g.kslabs,
                              at::cuda::getCurrentCUDAStream());
}

void pxq4hq_mmv_out(at::Tensor & out, const at::Tensor & x, const at::Tensor & slabs,
                    const at::Tensor & anchor) {
    const Geom g = check_weight(slabs, anchor);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "pxq4hq: x must be a CUDA float16 tensor");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "pxq4hq: out must be a CUDA float16 tensor");
    TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "pxq4hq: x and out must be 2D");
    TORCH_CHECK(x.size(1) == g.K, "pxq4hq: x K=", x.size(1), " does not match weight K=", g.K);
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == g.N,
                "pxq4hq: out must be [", x.size(0), ", ", g.N, "]");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "pxq4hq: x and out must be contiguous");
    TORCH_CHECK(pxq4hq_mmv_supported(g.kslabs),
                "pxq4hq: K=", g.K, " does not fit the mmv shared-memory budget on this device");
    const int64_t M = x.size(0);
    if (M == 0) return;
    const at::cuda::OptionalCUDAGuard guard(at::device_of(slabs));
    pxq4hq_launch_mmv_f16(PXQ_TIER_PXQ4HQ, slabs.data_ptr<uint8_t>(), anchor.data_ptr(),
                          x.data_ptr(), out.data_ptr(), (int)M, g.panels, g.kslabs,
                          /*vecx=*/true, at::cuda::getCurrentCUDAStream());
}

// Tensor-core arm, exposed on its own so an A/B can call it directly and so
// `strings -a lib | grep '^pxq4hq_mma_out$'` identifies a build that carries it.
void pxq4hq_mma_out(at::Tensor & out, const at::Tensor & x, const at::Tensor & slabs,
                    const at::Tensor & anchor) {
    const Geom g = check_weight(slabs, anchor);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "pxq4hq: x must be CUDA float16");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "pxq4hq: out must be CUDA float16");
    TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "pxq4hq: x and out must be 2D");
    TORCH_CHECK(x.size(1) == g.K, "pxq4hq: x K=", x.size(1), " does not match weight K=", g.K);
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == g.N, "pxq4hq: out shape mismatch");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "pxq4hq: x and out must be contiguous");
    const int64_t M = x.size(0);
    if (M == 0) return;
    TORCH_CHECK(pxq4hq_mma_supported(g.panels, g.kslabs, (int)M),
                "pxq4hq: the tensor-core path does not admit M=", M, " panels=", g.panels,
                " kslabs=", g.kslabs, " on this device (needs sm_70+ and 1 <= M <= 16)");
    const at::cuda::OptionalCUDAGuard guard(at::device_of(slabs));
    float * part = mma_partials(x, (int64_t)pxq4hq_mma_part_floats(g.panels, g.kslabs));
    pxq4hq_launch_mma_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(), part,
                          out.data_ptr(), (int)M, g.panels, g.kslabs,
                          at::cuda::getCurrentCUDAStream());
}

void pxq4hq_linear_out(at::Tensor & out, const at::Tensor & x, const at::Tensor & slabs,
                       const at::Tensor & anchor) {
    const Geom g = check_weight(slabs, anchor);
    TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "pxq4hq: x and out must be 2D");
    TORCH_CHECK(x.size(1) == g.K, "pxq4hq: x K=", x.size(1), " does not match weight K=", g.K);
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == g.N, "pxq4hq: out shape mismatch");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "pxq4hq: x and out must be contiguous");
    const int64_t M = x.size(0);
    if (M == 0) return;
    static const int64_t kMaxM     = env_i64("PXQ4_MMV_MAX_M", kPxq4hqMmvMaxM);
    static const int64_t kSliceMax = env_i64("PXQ4_MMV_SLICE_MAX", 16);
    // The M at which the tensor-core arm takes over from the SIMT mmv. Same floor as the PXQ4
    // arena path's default: below it the SIMT kernel is already at the bandwidth roof and the
    // wmma arm only adds staging.
    static const int64_t kMmaMinM  = env_i64("PXQ4_MMA_MIN_M", 5);

    // 1..kMmaMinM-1: the SIMT mmv, which is bandwidth bound and already optimal there.
    if (M < kMmaMinM && M <= kMaxM && pxq4hq_mmv_supported(g.kslabs)) {
        pxq4hq_mmv_out(out, x, slabs, anchor);
        return;
    }
    // kMmaMinM..16: the tensor-core arm when the card and the flag admit it.
    if (M >= kMmaMinM && M <= 16 && mma_route_flag() &&
        pxq4hq_mma_supported(g.panels, g.kslabs, (int)M)) {
        pxq4hq_mma_out(out, x, slabs, anchor);
        return;
    }
    if (M <= kMaxM && pxq4hq_mmv_supported(g.kslabs)) {
        pxq4hq_mmv_out(out, x, slabs, anchor);
        return;
    }
    // Medium batches with no tensor cores: ceil(M/kMaxM) sliced mmv calls. Each slice is one
    // weight pass and each token's fold is the canonical fold, bit-identical to a direct mmv
    // on that row.
    if (M <= kSliceMax && kMaxM > 0 && pxq4hq_mmv_supported(g.kslabs)) {
        for (int64_t r0 = 0; r0 < M; r0 += kMaxM) {
            const int64_t rows = std::min<int64_t>(kMaxM, M - r0);
            at::Tensor xs = x.narrow(0, r0, rows);
            at::Tensor os = out.narrow(0, r0, rows);
            pxq4hq_mmv_out(os, xs, slabs, anchor);
        }
        return;
    }
    // Prefill / large batch: coalesced dequant into the fp16 arena, then cuBLAS.
    const at::cuda::OptionalCUDAGuard guard(at::device_of(slabs));
    at::Tensor & wbuf = dequant_arena(x, g.N * g.K);
    at::Tensor w = wbuf.narrow(0, 0, g.N * g.K).view({g.N, g.K});
    pxq4hq_launch_dequant_f16(PXQ_TIER_PXQ4HQ, slabs.data_ptr<uint8_t>(), anchor.data_ptr(),
                              w.data_ptr(), g.panels, g.kslabs,
                              at::cuda::getCurrentCUDAStream());
    at::mm_out(out, x, w.t());
}

// ---- tables, capability, self-test ----------------------------------------------------------
//
// set_tables uploads BOTH, together, because the two are read from one checkpoint and a
// half-applied pair is worse than neither: the book without its sub decodes every weight
// against the wrong scale ladder.
void pxq4hq_set_tables(const at::Tensor & book, const at::Tensor & sub) {
    at::Tensor b = book.to(at::kFloat).to(at::kCPU).contiguous();
    at::Tensor s = sub.to(at::kFloat).to(at::kCPU).contiguous();
    TORCH_CHECK(b.numel() == PXQ4HQ_BOOK_N, "pxq4hq_set_tables: the book is ", PXQ4HQ_BOOK_N,
                " entries, got ", b.numel());
    TORCH_CHECK(s.numel() == 16, "pxq4hq_set_tables: the SUB8 LUT is 16 entries, got ", s.numel());
    // Order matters only in that each upload fans out to the tensor-core TU with the OTHER
    // table read back from the device, so the sub goes last and the pair ends consistent.
    pxq4hq_upload_book(PXQ_TIER_PXQ4HQ, b.data_ptr<float>(), (int)b.numel());
    pxq4hq_upload_sub(s.data_ptr<float>());
}

at::Tensor pxq4hq_get_book() {
    at::Tensor out = at::empty({PXQ4HQ_BOOK_N}, at::TensorOptions().dtype(at::kFloat));
    pxq4hq_download_book(PXQ_TIER_PXQ4HQ, out.data_ptr<float>(), PXQ4HQ_BOOK_N);
    return out;
}

at::Tensor pxq4hq_get_sub() {
    at::Tensor out = at::empty({16}, at::TensorOptions().dtype(at::kFloat));
    pxq4hq_download_sub(out.data_ptr<float>());
    return out;
}

int64_t pxq4hq_slab_bytes_op() { return (int64_t)PXQ4HQ_SLAB_BYTES; }
int64_t pxq4hq_version_op()    { return kPxq4hqVersion; }
bool    pxq4hq_supported_op(int64_t K) { return pxq4hq_mmv_supported((int)(K / 32)); }
int64_t pxq4hq_selftest_op()   { return (int64_t)pxq4hq_selftest(PXQ_TIER_PXQ4HQ); }

bool pxq4hq_mma_route_set(bool on) {
    const bool prev = mma_route_flag();
    mma_route_flag() = on;
    return prev;
}

// Size the tensor-core arena for a module's shape, EAGERLY and before any capture. The size is
// shape-only, so one call per distinct (panels, kslabs) is enough for every M that module will
// ever be called with.
void pxq4hq_reserve(int64_t panels, int64_t kslabs, const at::Tensor & like) {
    TORCH_CHECK(panels > 0 && kslabs > 0, "pxq4hq_reserve: panels and kslabs must be positive");
    TORCH_CHECK(like.is_cuda(), "pxq4hq_reserve: `like` must be a CUDA tensor naming the device");
    if (!pxq4hq_mma_supported((int)panels, (int)kslabs, 1)) return;   // no tensor cores here
    const at::cuda::OptionalCUDAGuard guard(at::device_of(like));
    (void)mma_partials(like, (int64_t)pxq4hq_mma_part_floats((int)panels, (int)kslabs));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(pxq4, m) {
    m.def("pxq4hq_dequant_out(Tensor(a!) out, Tensor slabs, Tensor anchor) -> ()");
    m.def("pxq4hq_mmv_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("pxq4hq_mma_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("pxq4hq_linear_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("pxq4hq_set_tables(Tensor book, Tensor sub) -> ()");
    m.def("pxq4hq_get_book() -> Tensor");
    m.def("pxq4hq_get_sub() -> Tensor");
    m.def("pxq4hq_slab_bytes() -> int");
    m.def("pxq4hq_supported(int K) -> bool");
    m.def("pxq4hq_selftest() -> int");
    m.def("pxq4hq_version() -> int");
    m.def("pxq4hq_mma_route_set(bool on) -> bool");
    m.def("pxq4hq_reserve(int panels, int kslabs, Tensor like) -> ()");
}

TORCH_LIBRARY_IMPL(pxq4, CUDA, m) {
    m.impl("pxq4hq_dequant_out", &pxq4hq_dequant_out);
    m.impl("pxq4hq_mmv_out",     &pxq4hq_mmv_out);
    m.impl("pxq4hq_mma_out",     &pxq4hq_mma_out);
    m.impl("pxq4hq_linear_out",  &pxq4hq_linear_out);
}

TORCH_LIBRARY_IMPL(pxq4, CompositeExplicitAutograd, m) {
    m.impl("pxq4hq_set_tables",    &pxq4hq_set_tables);
    m.impl("pxq4hq_get_book",      &pxq4hq_get_book);
    m.impl("pxq4hq_get_sub",       &pxq4hq_get_sub);
    m.impl("pxq4hq_slab_bytes",    &pxq4hq_slab_bytes_op);
    m.impl("pxq4hq_supported",     &pxq4hq_supported_op);
    m.impl("pxq4hq_selftest",      &pxq4hq_selftest_op);
    m.impl("pxq4hq_version",       &pxq4hq_version_op);
    m.impl("pxq4hq_mma_route_set", &pxq4hq_mma_route_set);
    m.impl("pxq4hq_reserve",       &pxq4hq_reserve);
}
