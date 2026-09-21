// pxq23_torch.cpp -- torch op bindings for the PXQ2/PXQ3 tiers.
//
// TORCH_LIBRARY_FRAGMENT, not TORCH_LIBRARY. The `pxq4` library is DEFINED in
// pxq4_kernel_torch.cpp, which is frozen; a fragment extends it from a second TU without
// touching that file, and without opening a second namespace (the host fork already owns
// torch.ops._C with 54 registered ops, and a second TORCH_LIBRARY(_C) in one process is a hard
// registration conflict -- the same reason the PXQ4 port chose the `pxq4` namespace).
//
// The library name therefore stays `pxq4` while the tiers it serves are pxq2/pxq3/pxq4. That
// is a WIRE-FORMAT name, exactly as the on-disk key suffix pxq4_slabs/pxq4_anchor is: the tier
// travels in the checkpoint's config.json, never in an op or parameter name. Renaming either
// would break every checkpoint and every launcher already in the field for no functional gain.
//
// OP NAMING: one op per tier rather than one op with a `tier` argument. The frozen PXQ4 ABI
// (plan sec.7.1) is then literally untouched, the fake/meta kernels stay trivial shape checks,
// and a tier can never be passed as a runtime value that a captured graph baked in wrongly.

#include <torch/library.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstdlib>
#include <cstring>

#include "pxq23_kernel_tables.h"
#include "pxq23_kernel_launch.h"

namespace {

// Default token-count ceiling for the mmv path, mirroring the engine's PXA_PXQ4_2D_MAX_NY
// (default 8). Above it the weight-reread cost of the mmv (grid.y is the token axis, so each
// block reads its whole panel once per token) loses to dequant + cuBLAS.
constexpr int64_t kPxqMmvMaxM = 8;

int64_t env_i64(const char * name, int64_t dflt) {
    const char * e = getenv(name);
    return (e && *e) ? (int64_t)atoll(e) : dflt;
}

struct Geom { int panels; int kslabs; int64_t N; int64_t K; };

Geom check_weight(int tier, const at::Tensor & slabs, const at::Tensor & anchor) {
    const int slab_bytes = pxq23_slab_bytes(tier);
    TORCH_CHECK(slab_bytes > 0, "pxq: unknown tier ", tier);
    TORCH_CHECK(slabs.is_cuda() && anchor.is_cuda(), "pxq: slabs/anchor must be CUDA tensors");
    TORCH_CHECK(slabs.scalar_type() == at::kByte, "pxq: slabs must be uint8, got ", slabs.scalar_type());
    TORCH_CHECK(anchor.scalar_type() == at::kHalf, "pxq: anchor must be float16, got ", anchor.scalar_type());
    TORCH_CHECK(slabs.dim() == 3, "pxq: slabs must be [panels, kslabs, ", slab_bytes, "], got dim ", slabs.dim());
    TORCH_CHECK(anchor.dim() == 2, "pxq: anchor must be [panels, 64], got dim ", anchor.dim());
    // THE TIER CHECK THAT MATTERS. A PXQ2 tensor handed to the PXQ3 op has a valid-looking
    // shape in every other respect and would decode into well-formed garbage, so the slab
    // stride is what identifies the tier and it is checked, never inferred.
    TORCH_CHECK(slabs.size(2) == slab_bytes,
                "pxq: tier ", pxq_tier_name(tier), " wants slab stride ", slab_bytes,
                ", got ", slabs.size(2), " -- this tensor is a different tier");
    TORCH_CHECK(anchor.size(1) == 64, "pxq: anchor row must be 64, got ", anchor.size(1));
    TORCH_CHECK(slabs.size(0) == anchor.size(0),
                "pxq: panel count mismatch: slabs ", slabs.size(0), " vs anchor ", anchor.size(0));
    // Contiguity is not cosmetic: the code load is a vector load for PXQ2/PXQ4 whose alignment
    // is only guaranteed when the slab stride is the natural one and the base comes from a
    // torch allocation. A narrow()ed view would silently misalign it.
    TORCH_CHECK(slabs.is_contiguous(), "pxq: slabs must be contiguous");
    TORCH_CHECK(anchor.is_contiguous(), "pxq: anchor must be contiguous");
    TORCH_CHECK(slabs.size(0) > 0 && slabs.size(1) > 0, "pxq: empty weight");

    Geom g;
    g.panels = (int)slabs.size(0);
    g.kslabs = (int)slabs.size(1);
    g.N = (int64_t)g.panels * 64;
    g.K = (int64_t)g.kslabs * 32;
    return g;
}

// Expert-stacked weights: slabs [E, panels, kslabs, SLAB], anchor [E, panels, 64].
Geom check_weight_moe(int tier, const at::Tensor & slabs, const at::Tensor & anchor, int & E) {
    const int slab_bytes = pxq23_slab_bytes(tier);
    TORCH_CHECK(slab_bytes > 0, "pxq: unknown tier ", tier);
    TORCH_CHECK(slabs.dim() == 4 && anchor.dim() == 3, "pxq: moe slabs must be 4-D and anchor 3-D");
    TORCH_CHECK(slabs.size(3) == slab_bytes,
                "pxq: tier ", pxq_tier_name(tier), " wants slab stride ", slab_bytes,
                ", got ", slabs.size(3), " -- this tensor is a different tier");
    TORCH_CHECK(anchor.size(2) == 64, "pxq: moe anchor row must be 64");
    TORCH_CHECK(slabs.size(0) == anchor.size(0) && slabs.size(1) == anchor.size(1),
                "pxq: moe expert/panel mismatch between slabs and anchor");
    TORCH_CHECK(slabs.is_contiguous() && anchor.is_contiguous(), "pxq: moe weights must be contiguous");
    TORCH_CHECK(slabs.scalar_type() == at::kByte && anchor.scalar_type() == at::kHalf,
                "pxq: moe slabs must be uint8 and anchor float16");
    E = (int)slabs.size(0);
    Geom g;
    g.panels = (int)slabs.size(1);
    g.kslabs = (int)slabs.size(2);
    g.N = (int64_t)g.panels * 64;
    g.K = (int64_t)g.kslabs * 32;
    return g;
}

// ------------------------------------------------------------------ prefill dequant arena
//
// One persistent fp16 buffer per device, sized by the largest prefill this process has asked
// for and NEVER reassigned while a captured graph could hold its address. The rule and its
// failure mode are the v12b ones (pxq4_kernel_torch.cpp): a CUDA graph records raw device
// addresses, so replacing the tensor after a capture hands the old block back to the caching
// allocator while captured graphs still write into it. Prefill shapes are not captured on this
// stack (capture sizes are <= 8 tokens and take the mmv branch), so growth is legal here -- but
// it is refused DURING capture rather than trusted not to happen.
std::array<at::Tensor, 64> g_dequant_arena;

at::Tensor & dequant_arena(const at::Tensor & like, int64_t need) {
    const int dev = (int)like.device().index();
    TORCH_CHECK(dev >= 0 && dev < 64, "pxq: device index out of range: ", dev);
    at::Tensor & a = g_dequant_arena[dev];
    if (!a.defined() || a.numel() < need) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &cap);
        TORCH_CHECK(cap == cudaStreamCaptureStatusNone,
                    "pxq: the prefill dequant arena would have to grow during CUDA graph "
                    "capture (need ", need, " fp16 elems, have ",
                    a.defined() ? a.numel() : 0, "). Every shape a captured graph replays must "
                    "have run once eagerly first.");
        a = at::empty({need}, like.options().dtype(at::kHalf));
    }
    return a;
}

// --------------------------------------------------------------------------------- the ops
void tier_dequant_out(int tier, at::Tensor & out, const at::Tensor & slabs,
                      const at::Tensor & anchor) {
    const Geom g = check_weight(tier, slabs, anchor);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "pxq: out must be a CUDA float16 tensor");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == g.N && out.size(1) == g.K,
                "pxq: out must be [", g.N, ", ", g.K, "]");
    TORCH_CHECK(out.is_contiguous(), "pxq: out must be contiguous");
    TORCH_CHECK(out.device() == slabs.device(), "pxq: out and slabs on different devices");
    const at::cuda::OptionalCUDAGuard guard(at::device_of(slabs));
    pxq23_launch_dequant_f16(tier, slabs.data_ptr<uint8_t>(), anchor.data_ptr(), out.data_ptr(),
                             g.panels, g.kslabs, at::cuda::getCurrentCUDAStream());
}

void tier_mmv_out(int tier, at::Tensor & out, const at::Tensor & x, const at::Tensor & slabs,
                  const at::Tensor & anchor) {
    const Geom g = check_weight(tier, slabs, anchor);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "pxq: x must be a CUDA float16 tensor");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "pxq: out must be a CUDA float16 tensor");
    TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "pxq: x and out must be 2D");
    TORCH_CHECK(x.size(1) == g.K, "pxq: x K=", x.size(1), " does not match weight K=", g.K);
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == g.N,
                "pxq: out must be [", x.size(0), ", ", g.N, "]");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "pxq: x and out must be contiguous");
    TORCH_CHECK(pxq23_mmv_supported(g.kslabs),
                "pxq: K=", g.K, " does not fit the mmv shared-memory budget on this device");
    const int64_t M = x.size(0);
    if (M == 0) return;
    const at::cuda::OptionalCUDAGuard guard(at::device_of(slabs));
    pxq23_launch_mmv_f16(tier, slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                         out.data_ptr(), (int)M, g.panels, g.kslabs, /*vecx=*/true,
                         at::cuda::getCurrentCUDAStream());
}

void tier_linear_out(int tier, at::Tensor & out, const at::Tensor & x, const at::Tensor & slabs,
                     const at::Tensor & anchor) {
    const Geom g = check_weight(tier, slabs, anchor);
    TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "pxq: x and out must be 2D");
    TORCH_CHECK(x.size(1) == g.K, "pxq: x K=", x.size(1), " does not match weight K=", g.K);
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == g.N, "pxq: out shape mismatch");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "pxq: x and out must be contiguous");
    const int64_t M = x.size(0);
    if (M == 0) return;
    static const int64_t kMaxM    = env_i64("PXQ4_MMV_MAX_M", kPxqMmvMaxM);
    static const int64_t kSliceMax = env_i64("PXQ4_MMV_SLICE_MAX", 16);

    if (M <= kMaxM && pxq23_mmv_supported(g.kslabs)) {
        tier_mmv_out(tier, out, x, slabs, anchor);
        return;
    }
    // Medium batches: ceil(M/kMaxM) sliced mmv calls. Each slice is one weight pass and each
    // token's fold is the canonical fold, bit-identical to a direct mmv on that row. There is
    // no tensor-core (mma) route for these tiers, so the PXQ4 9..16 special case has no twin
    // here -- slicing is the only medium-batch option.
    if (M <= kSliceMax && kMaxM > 0 && pxq23_mmv_supported(g.kslabs)) {
        for (int64_t r0 = 0; r0 < M; r0 += kMaxM) {
            const int64_t rows = std::min<int64_t>(kMaxM, M - r0);
            at::Tensor xs = x.narrow(0, r0, rows);
            at::Tensor os = out.narrow(0, r0, rows);
            tier_mmv_out(tier, os, xs, slabs, anchor);
        }
        return;
    }
    // Prefill / large batch: coalesced dequant into the fp16 arena, then cuBLAS.
    const at::cuda::OptionalCUDAGuard guard(at::device_of(slabs));
    at::Tensor & wbuf = dequant_arena(x, g.N * g.K);
    at::Tensor w = wbuf.narrow(0, 0, g.N * g.K).view({g.N, g.K});
    pxq23_launch_dequant_f16(tier, slabs.data_ptr<uint8_t>(), anchor.data_ptr(), w.data_ptr(),
                             g.panels, g.kslabs, at::cuda::getCurrentCUDAStream());
    at::mm_out(out, x, w.t());
}

void tier_moe_mmv_out(int tier, at::Tensor & out, const at::Tensor & x, const at::Tensor & ids,
                      const at::Tensor & slabs, const at::Tensor & anchor) {
    int E = 0;
    const Geom g = check_weight_moe(tier, slabs, anchor, E);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "pxq: x must be CUDA float16");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "pxq: out must be CUDA float16");
    TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == at::kInt, "pxq: ids must be CUDA int32");
    TORCH_CHECK(x.dim() == 2 && out.dim() == 2 && ids.dim() == 1, "pxq: moe rank mismatch");
    TORCH_CHECK(x.size(1) == g.K, "pxq: moe x K=", x.size(1), " != weight K=", g.K);
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == g.N, "pxq: moe out shape mismatch");
    TORCH_CHECK(ids.size(0) == x.size(0), "pxq: moe ids length must equal the row count");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && ids.is_contiguous(),
                "pxq: moe x/out/ids must be contiguous");
    TORCH_CHECK(pxq23_mmv_supported(g.kslabs),
                "pxq: K=", g.K, " does not fit the mmv shared-memory budget on this device");
    const int64_t S = x.size(0);
    if (S == 0) return;
    const at::cuda::OptionalCUDAGuard guard(at::device_of(slabs));
    pxq23_launch_moe_mmv_f16(tier, slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                             ids.data_ptr<int32_t>(), out.data_ptr(), (int)S, E, g.panels,
                             g.kslabs, /*vecx=*/true, at::cuda::getCurrentCUDAStream());
}

// ---- per-tier entry points (see OP NAMING above) --------------------------------------------
void pxq2_dequant_out(at::Tensor & o, const at::Tensor & s, const at::Tensor & a) { tier_dequant_out(PXQ_TIER_PXQ2, o, s, a); }
void pxq3_dequant_out(at::Tensor & o, const at::Tensor & s, const at::Tensor & a) { tier_dequant_out(PXQ_TIER_PXQ3, o, s, a); }
void pxq2_mmv_out(at::Tensor & o, const at::Tensor & x, const at::Tensor & s, const at::Tensor & a) { tier_mmv_out(PXQ_TIER_PXQ2, o, x, s, a); }
void pxq3_mmv_out(at::Tensor & o, const at::Tensor & x, const at::Tensor & s, const at::Tensor & a) { tier_mmv_out(PXQ_TIER_PXQ3, o, x, s, a); }
void pxq2_linear_out(at::Tensor & o, const at::Tensor & x, const at::Tensor & s, const at::Tensor & a) { tier_linear_out(PXQ_TIER_PXQ2, o, x, s, a); }
void pxq3_linear_out(at::Tensor & o, const at::Tensor & x, const at::Tensor & s, const at::Tensor & a) { tier_linear_out(PXQ_TIER_PXQ3, o, x, s, a); }
void pxq2_moe_mmv_out(at::Tensor & o, const at::Tensor & x, const at::Tensor & i, const at::Tensor & s, const at::Tensor & a) { tier_moe_mmv_out(PXQ_TIER_PXQ2, o, x, i, s, a); }
void pxq3_moe_mmv_out(at::Tensor & o, const at::Tensor & x, const at::Tensor & i, const at::Tensor & s, const at::Tensor & a) { tier_moe_mmv_out(PXQ_TIER_PXQ3, o, x, i, s, a); }

// ---- tables, capability, self-test ----------------------------------------------------------
void pxq_set_book(int64_t tier, const at::Tensor & book) {
    const int n = pxq23_book_n((int)tier);
    TORCH_CHECK(n > 0, "pxq_set_book: unknown tier ", tier);
    at::Tensor b = book.to(at::kFloat).to(at::kCPU).contiguous();
    TORCH_CHECK(b.numel() == n, "pxq_set_book: tier ", pxq_tier_name((int)tier), " wants ", n,
                " entries, got ", b.numel());
    pxq23_upload_book((int)tier, b.data_ptr<float>(), n);
}

void pxq_set_sub(const at::Tensor & sub) {
    at::Tensor s = sub.to(at::kFloat).to(at::kCPU).contiguous();
    TORCH_CHECK(s.numel() == 16, "pxq_set_sub: the SUB16 LUT is 16 entries, got ", s.numel());
    pxq23_upload_sub(s.data_ptr<float>());
}

at::Tensor pxq_get_book(int64_t tier) {
    const int n = pxq23_book_n((int)tier);
    TORCH_CHECK(n > 0, "pxq_get_book: unknown tier ", tier);
    at::Tensor out = at::empty({n}, at::TensorOptions().dtype(at::kFloat));
    pxq23_download_book((int)tier, out.data_ptr<float>(), n);
    return out;
}

at::Tensor pxq_get_sub() {
    at::Tensor out = at::empty({16}, at::TensorOptions().dtype(at::kFloat));
    pxq23_download_sub(out.data_ptr<float>());
    return out;
}

int64_t pxq_slab_bytes(int64_t tier) { return (int64_t)pxq23_slab_bytes((int)tier); }
bool    pxq_tier_served(int64_t tier) { return pxq23_slab_bytes((int)tier) > 0; }

bool pxq_supported(int64_t tier, int64_t K) {
    if (pxq23_slab_bytes((int)tier) <= 0) return false;
    if (K <= 0 || K % 32 != 0) return false;
    return pxq23_mmv_supported((int)(K / 32));
}

int64_t pxq_selftest(int64_t tier) { return (int64_t)pxq23_selftest((int)tier); }

// Bump on every change to this file's kernels or ABI. 13 == the first build that serves
// PXQ2/PXQ3 alongside the frozen v12b PXQ4 kernels.
int64_t pxq_version() { return 13; }

}  // namespace

TORCH_LIBRARY_FRAGMENT(pxq4, m) {
    m.def("pxq2_dequant_out(Tensor(a!) out, Tensor slabs, Tensor anchor) -> ()");
    m.def("pxq3_dequant_out(Tensor(a!) out, Tensor slabs, Tensor anchor) -> ()");
    m.def("pxq2_mmv_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("pxq3_mmv_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("pxq2_linear_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("pxq3_linear_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("pxq2_moe_mmv_out(Tensor(a!) out, Tensor x, Tensor ids, Tensor slabs, Tensor anchor) -> ()");
    m.def("pxq3_moe_mmv_out(Tensor(a!) out, Tensor x, Tensor ids, Tensor slabs, Tensor anchor) -> ()");
    m.def("pxq_set_book(int tier, Tensor book) -> ()");
    m.def("pxq_set_sub(Tensor sub) -> ()");
    m.def("pxq_get_book(int tier) -> Tensor");
    m.def("pxq_get_sub() -> Tensor");
    m.def("pxq_slab_bytes(int tier) -> int");
    m.def("pxq_tier_served(int tier) -> bool");
    m.def("pxq_supported(int tier, int K) -> bool");
    m.def("pxq_selftest(int tier) -> int");
    m.def("pxq_version() -> int");
}

TORCH_LIBRARY_IMPL(pxq4, CUDA, m) {
    m.impl("pxq2_dequant_out",  &pxq2_dequant_out);
    m.impl("pxq3_dequant_out",  &pxq3_dequant_out);
    m.impl("pxq2_mmv_out",      &pxq2_mmv_out);
    m.impl("pxq3_mmv_out",      &pxq3_mmv_out);
    m.impl("pxq2_linear_out",   &pxq2_linear_out);
    m.impl("pxq3_linear_out",   &pxq3_linear_out);
    m.impl("pxq2_moe_mmv_out",  &pxq2_moe_mmv_out);
    m.impl("pxq3_moe_mmv_out",  &pxq3_moe_mmv_out);
}

TORCH_LIBRARY_IMPL(pxq4, CompositeExplicitAutograd, m) {
    m.impl("pxq_slab_bytes",  &pxq_slab_bytes);
    m.impl("pxq_tier_served", &pxq_tier_served);
    m.impl("pxq_supported",   &pxq_supported);
    m.impl("pxq_selftest",    &pxq_selftest);
    m.impl("pxq_version",     &pxq_version);
    m.impl("pxq_set_book",    &pxq_set_book);
    m.impl("pxq_set_sub",     &pxq_set_sub);
    m.impl("pxq_get_book",    &pxq_get_book);
    m.impl("pxq_get_sub",     &pxq_get_sub);
}
