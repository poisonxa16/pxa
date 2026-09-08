// pxq_moe_fused_torch.cpp -- torch op bindings for the fused MoE decode block.
//
// TORCH_LIBRARY_FRAGMENT(pxq4), not TORCH_LIBRARY, for the reason pxq23_torch.cpp states: the
// `pxq4` library is DEFINED in the frozen pxq4_kernel_torch.cpp and a second TORCH_LIBRARY on
// the same name in one process is a hard registration conflict. A fragment extends it from a
// third TU without touching either file.
//
// OP NAMING follows the convention already in the field: pxq4's own ops are unprefixed, the
// other tiers carry a tier prefix, and the tier is a NAME rather than a runtime argument -- so
// a captured graph can never have baked in the wrong tier, and the fake/meta kernels stay
// trivial shape checks.
//
// EVERY OP IS OUT-OF-PLACE-FREE: the caller supplies `act`, `dn` and `out`. That is not an
// accident of taste, it is the CUDA-graph rule. This TU allocates nothing, so there is no
// arena to grow, no arena to seal, and no raw device address for a captured graph to outlive
// -- the class of bug that produced the v12b use-after-free note in pxq4_kernel_torch.cpp
// simply cannot occur here. The python side owns one frozen per-device scratch pair sized
// before capture; a slice of a persistent tensor has a stable data_ptr by construction.
#include <torch/library.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdlib>

#include "pxq23_kernel_tables.h"    // PXQ_TIER_*, pxq_tier_slab_bytes, pxq_tier_name
#include "pxq_moe_fused_launch.h"

namespace {

// Expert-stacked weight geometry, shared by all three ops. slabs [E, panels, kslabs, SLAB],
// anchor [E, panels, 64].
struct MoeGeom { int E; int panels; int kslabs; int64_t N; int64_t K; };

MoeGeom check_moe_weight(int tier, const at::Tensor & slabs, const at::Tensor & anchor) {
    const int slab_bytes = pxq_tier_slab_bytes(tier);
    TORCH_CHECK(slab_bytes > 0, "pxq moe fused: unknown tier ", tier);
    TORCH_CHECK(slabs.is_cuda() && anchor.is_cuda(),
                "pxq moe fused: slabs/anchor must be CUDA tensors");
    TORCH_CHECK(slabs.dim() == 4 && anchor.dim() == 3,
                "pxq moe fused: slabs must be 4-D [E,panels,kslabs,SLAB] and anchor 3-D");
    TORCH_CHECK(slabs.size(3) == slab_bytes,
                "pxq moe fused: tier ", pxq_tier_name(tier), " wants slab stride ", slab_bytes,
                ", got ", slabs.size(3), " -- this tensor is a different tier");
    TORCH_CHECK(anchor.size(2) == 64, "pxq moe fused: anchor row must be 64");
    TORCH_CHECK(slabs.size(0) == anchor.size(0) && slabs.size(1) == anchor.size(1),
                "pxq moe fused: expert/panel mismatch between slabs and anchor");
    TORCH_CHECK(slabs.is_contiguous() && anchor.is_contiguous(),
                "pxq moe fused: weights must be contiguous -- the code load is a vector load "
                "whose alignment only holds for the natural slab stride");
    TORCH_CHECK(slabs.scalar_type() == at::kByte && anchor.scalar_type() == at::kHalf,
                "pxq moe fused: slabs must be uint8 and anchor float16");
    MoeGeom g;
    g.E      = (int)slabs.size(0);
    g.panels = (int)slabs.size(1);
    g.kslabs = (int)slabs.size(2);
    g.N      = (int64_t)g.panels * 64;
    g.K      = (int64_t)g.kslabs * 32;
    TORCH_CHECK(g.E > 0 && g.panels > 0 && g.kslabs > 0, "pxq moe fused: empty weight");
    return g;
}

// ------------------------------------------------------------------------------- gate+up+GLU
void tier_gateup_glu(int tier, at::Tensor & act, const at::Tensor & x, const at::Tensor & ids,
                     const at::Tensor & slabs, const at::Tensor & anchor, int64_t top_k) {
    const MoeGeom g = check_moe_weight(tier, slabs, anchor);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "pxq moe fused: x must be CUDA fp16");
    TORCH_CHECK(act.is_cuda() && act.scalar_type() == at::kHalf,
                "pxq moe fused: act must be CUDA fp16");
    TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == at::kInt,
                "pxq moe fused: ids must be CUDA int32");
    TORCH_CHECK(x.dim() == 2 && act.dim() == 2 && ids.dim() == 1,
                "pxq moe fused: gateup rank mismatch (want x[M,H], act[S,Ip], ids[S])");
    TORCH_CHECK(x.is_contiguous() && act.is_contiguous() && ids.is_contiguous(),
                "pxq moe fused: x/act/ids must be contiguous");
    TORCH_CHECK(top_k >= 1, "pxq moe fused: top_k must be >= 1");
    TORCH_CHECK(g.panels % 2 == 0,
                "pxq moe fused: w13 must hold an even panel count (gate half then up half), "
                "got ", g.panels);
    const int64_t S  = ids.size(0);
    const int64_t Ip = g.N / 2;
    TORCH_CHECK(S == act.size(0), "pxq moe fused: act rows ", act.size(0), " != ids ", S);
    TORCH_CHECK(S % top_k == 0, "pxq moe fused: S=", S, " is not a multiple of top_k=", top_k);
    TORCH_CHECK(x.size(0) == S / top_k,
                "pxq moe fused: x must hold M=S/top_k=", S / top_k, " rows, got ", x.size(0),
                " -- the kernel reads x per TOKEN, so the caller must NOT pre-expand it");
    TORCH_CHECK(x.size(1) == g.K, "pxq moe fused: x K=", x.size(1), " != w13 K=", g.K);
    TORCH_CHECK(act.size(1) == Ip, "pxq moe fused: act width ", act.size(1), " != Ip=", Ip);
    TORCH_CHECK(pxq_moe_supported(g.kslabs),
                "pxq moe fused: K=", g.K, " does not fit the shared-memory budget on this device");
    if (S == 0) return;
    const at::cuda::OptionalCUDAGuard guard(at::device_of(slabs));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (tier == PXQ_TIER_PXQ4) {
        pxq4_moe_launch_gateup_glu(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                                   ids.data_ptr<int32_t>(), act.data_ptr(), (int)S, (int)Ip,
                                   g.E, g.panels, g.kslabs, (int)top_k, /*vecx=*/true, stream);
    } else {
        pxq23_moe_launch_gateup_glu(tier, slabs.data_ptr<uint8_t>(), anchor.data_ptr(),
                                    x.data_ptr(), ids.data_ptr<int32_t>(), act.data_ptr(),
                                    (int)S, (int)Ip, g.E, g.panels, g.kslabs, (int)top_k,
                                    /*vecx=*/true, stream);
    }
}

// --------------------------------------------------------------- FORM A: down + weighted fold
void tier_down_fold(int tier, at::Tensor & out, const at::Tensor & act, const at::Tensor & ids,
                    const at::Tensor & wts, const at::Tensor & slabs, const at::Tensor & anchor,
                    int64_t top_k) {
    const MoeGeom g = check_moe_weight(tier, slabs, anchor);
    TORCH_CHECK(act.is_cuda() && act.scalar_type() == at::kHalf,
                "pxq moe fused: act must be CUDA fp16");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf,
                "pxq moe fused: out must be CUDA fp16");
    TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == at::kInt,
                "pxq moe fused: ids must be CUDA int32");
    TORCH_CHECK(wts.is_cuda() && wts.scalar_type() == at::kFloat,
                "pxq moe fused: router weights must be CUDA float32 -- the fold is fp32 and the "
                "path it replaces casts them to fp32 too");
    TORCH_CHECK(act.dim() == 2 && out.dim() == 2 && ids.dim() == 1,
                "pxq moe fused: down rank mismatch (want act[S,Ip], out[M,H], ids[S])");
    TORCH_CHECK(act.is_contiguous() && out.is_contiguous() && ids.is_contiguous() &&
                wts.is_contiguous(), "pxq moe fused: act/out/ids/wts must be contiguous");
    TORCH_CHECK(top_k >= 1, "pxq moe fused: top_k must be >= 1");
    const int64_t S = ids.size(0);
    TORCH_CHECK(wts.numel() == S, "pxq moe fused: wts holds ", wts.numel(), " weights for ", S,
                " routed rows");
    TORCH_CHECK(act.size(0) == S, "pxq moe fused: act rows ", act.size(0), " != ids ", S);
    TORCH_CHECK(S % top_k == 0, "pxq moe fused: S=", S, " is not a multiple of top_k=", top_k);
    const int64_t M = S / top_k;
    TORCH_CHECK(out.size(0) == M && out.size(1) == g.N,
                "pxq moe fused: out must be [", M, ", ", g.N, "]");
    TORCH_CHECK(act.size(1) == g.K, "pxq moe fused: act K=", act.size(1), " != w2 K=", g.K);
    TORCH_CHECK(pxq_moe_supported(g.kslabs),
                "pxq moe fused: K=", g.K, " does not fit the shared-memory budget on this device");
    if (S == 0) return;
    const at::cuda::OptionalCUDAGuard guard(at::device_of(slabs));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (tier == PXQ_TIER_PXQ4) {
        pxq4_moe_launch_down_fold(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), act.data_ptr(),
                                  ids.data_ptr<int32_t>(), wts.data_ptr<float>(), out.data_ptr(),
                                  (int)M, g.E, g.panels, g.kslabs, (int)top_k, /*vecx=*/true,
                                  stream);
    } else {
        pxq23_moe_launch_down_fold(tier, slabs.data_ptr<uint8_t>(), anchor.data_ptr(),
                                   act.data_ptr(), ids.data_ptr<int32_t>(), wts.data_ptr<float>(),
                                   out.data_ptr(), (int)M, g.E, g.panels, g.kslabs, (int)top_k,
                                   /*vecx=*/true, stream);
    }
}

// ---------------------------------------------- FORM B: down partials, slot axis kept in grid
void tier_down_part(int tier, at::Tensor & dn, const at::Tensor & act, const at::Tensor & ids,
                    const at::Tensor & slabs, const at::Tensor & anchor) {
    const MoeGeom g = check_moe_weight(tier, slabs, anchor);
    TORCH_CHECK(act.is_cuda() && act.scalar_type() == at::kHalf, "pxq moe fused: act CUDA fp16");
    TORCH_CHECK(dn.is_cuda() && dn.scalar_type() == at::kHalf, "pxq moe fused: dn CUDA fp16");
    TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == at::kInt, "pxq moe fused: ids CUDA int32");
    TORCH_CHECK(act.dim() == 2 && dn.dim() == 2 && ids.dim() == 1,
                "pxq moe fused: down-part rank mismatch");
    TORCH_CHECK(act.is_contiguous() && dn.is_contiguous() && ids.is_contiguous(),
                "pxq moe fused: act/dn/ids must be contiguous");
    const int64_t S = ids.size(0);
    TORCH_CHECK(act.size(0) == S && dn.size(0) == S, "pxq moe fused: down-part row mismatch");
    TORCH_CHECK(act.size(1) == g.K && dn.size(1) == g.N, "pxq moe fused: down-part shape mismatch");
    TORCH_CHECK(pxq_moe_supported(g.kslabs), "pxq moe fused: K=", g.K, " exceeds the smem budget");
    if (S == 0) return;
    const at::cuda::OptionalCUDAGuard guard(at::device_of(slabs));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (tier == PXQ_TIER_PXQ4) {
        pxq4_moe_launch_down_part(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), act.data_ptr(),
                                  ids.data_ptr<int32_t>(), dn.data_ptr(), (int)S, g.E, g.panels,
                                  g.kslabs, /*vecx=*/true, stream);
    } else {
        pxq23_moe_launch_down_part(tier, slabs.data_ptr<uint8_t>(), anchor.data_ptr(),
                                   act.data_ptr(), ids.data_ptr<int32_t>(), dn.data_ptr(),
                                   (int)S, g.E, g.panels, g.kslabs, /*vecx=*/true, stream);
    }
}

void moe_slot_fold_out(at::Tensor & out, const at::Tensor & dn, const at::Tensor & wts,
                       int64_t top_k) {
    TORCH_CHECK(dn.is_cuda() && dn.scalar_type() == at::kHalf, "pxq moe fused: dn CUDA fp16");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "pxq moe fused: out CUDA fp16");
    TORCH_CHECK(wts.is_cuda() && wts.scalar_type() == at::kFloat, "pxq moe fused: wts CUDA fp32");
    TORCH_CHECK(dn.dim() == 2 && out.dim() == 2, "pxq moe fused: slot fold rank mismatch");
    TORCH_CHECK(dn.is_contiguous() && out.is_contiguous() && wts.is_contiguous(),
                "pxq moe fused: slot fold tensors must be contiguous");
    TORCH_CHECK(top_k >= 1, "pxq moe fused: top_k must be >= 1");
    const int64_t S = dn.size(0);
    TORCH_CHECK(S % top_k == 0 && wts.numel() == S, "pxq moe fused: slot fold S/top_k mismatch");
    const int64_t M = S / top_k;
    TORCH_CHECK(out.size(0) == M && out.size(1) == dn.size(1),
                "pxq moe fused: slot fold out must be [", M, ", ", dn.size(1), "]");
    if (S == 0) return;
    const at::cuda::OptionalCUDAGuard guard(at::device_of(dn));
    pxq_moe_launch_slot_fold(dn.data_ptr(), wts.data_ptr<float>(), out.data_ptr(), (int)M,
                             (int)dn.size(1), (int)top_k, at::cuda::getCurrentCUDAStream());
}

// ---- per-tier entry points ------------------------------------------------------------------
void moe_gateup_glu_out(at::Tensor & a, const at::Tensor & x, const at::Tensor & i,
                        const at::Tensor & s, const at::Tensor & an, int64_t k) { tier_gateup_glu(PXQ_TIER_PXQ4, a, x, i, s, an, k); }
void pxq2_moe_gateup_glu_out(at::Tensor & a, const at::Tensor & x, const at::Tensor & i,
                             const at::Tensor & s, const at::Tensor & an, int64_t k) { tier_gateup_glu(PXQ_TIER_PXQ2, a, x, i, s, an, k); }
void pxq3_moe_gateup_glu_out(at::Tensor & a, const at::Tensor & x, const at::Tensor & i,
                             const at::Tensor & s, const at::Tensor & an, int64_t k) { tier_gateup_glu(PXQ_TIER_PXQ3, a, x, i, s, an, k); }

void moe_down_fold_out(at::Tensor & o, const at::Tensor & a, const at::Tensor & i,
                       const at::Tensor & w, const at::Tensor & s, const at::Tensor & an, int64_t k) { tier_down_fold(PXQ_TIER_PXQ4, o, a, i, w, s, an, k); }
void pxq2_moe_down_fold_out(at::Tensor & o, const at::Tensor & a, const at::Tensor & i,
                            const at::Tensor & w, const at::Tensor & s, const at::Tensor & an, int64_t k) { tier_down_fold(PXQ_TIER_PXQ2, o, a, i, w, s, an, k); }
void pxq3_moe_down_fold_out(at::Tensor & o, const at::Tensor & a, const at::Tensor & i,
                            const at::Tensor & w, const at::Tensor & s, const at::Tensor & an, int64_t k) { tier_down_fold(PXQ_TIER_PXQ3, o, a, i, w, s, an, k); }

void moe_down_part_out(at::Tensor & d, const at::Tensor & a, const at::Tensor & i,
                       const at::Tensor & s, const at::Tensor & an) { tier_down_part(PXQ_TIER_PXQ4, d, a, i, s, an); }
void pxq2_moe_down_part_out(at::Tensor & d, const at::Tensor & a, const at::Tensor & i,
                            const at::Tensor & s, const at::Tensor & an) { tier_down_part(PXQ_TIER_PXQ2, d, a, i, s, an); }
void pxq3_moe_down_part_out(at::Tensor & d, const at::Tensor & a, const at::Tensor & i,
                            const at::Tensor & s, const at::Tensor & an) { tier_down_part(PXQ_TIER_PXQ3, d, a, i, s, an); }

int64_t moe_fused_version() { return (int64_t)pxq_moe_fused_version(); }

bool moe_fused_supported(int64_t tier, int64_t K) {
    if (pxq_tier_slab_bytes((int)tier) <= 0) return false;
    if (K <= 0 || K % 32) return false;
    return pxq_moe_supported((int)(K / 32));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(pxq4, m) {
    m.def("moe_gateup_glu_out(Tensor(a!) act, Tensor x, Tensor ids, Tensor slabs, Tensor anchor, int top_k) -> ()");
    m.def("pxq2_moe_gateup_glu_out(Tensor(a!) act, Tensor x, Tensor ids, Tensor slabs, Tensor anchor, int top_k) -> ()");
    m.def("pxq3_moe_gateup_glu_out(Tensor(a!) act, Tensor x, Tensor ids, Tensor slabs, Tensor anchor, int top_k) -> ()");
    m.def("moe_down_fold_out(Tensor(a!) out, Tensor act, Tensor ids, Tensor wts, Tensor slabs, Tensor anchor, int top_k) -> ()");
    m.def("pxq2_moe_down_fold_out(Tensor(a!) out, Tensor act, Tensor ids, Tensor wts, Tensor slabs, Tensor anchor, int top_k) -> ()");
    m.def("pxq3_moe_down_fold_out(Tensor(a!) out, Tensor act, Tensor ids, Tensor wts, Tensor slabs, Tensor anchor, int top_k) -> ()");
    m.def("moe_down_part_out(Tensor(a!) dn, Tensor act, Tensor ids, Tensor slabs, Tensor anchor) -> ()");
    m.def("pxq2_moe_down_part_out(Tensor(a!) dn, Tensor act, Tensor ids, Tensor slabs, Tensor anchor) -> ()");
    m.def("pxq3_moe_down_part_out(Tensor(a!) dn, Tensor act, Tensor ids, Tensor slabs, Tensor anchor) -> ()");
    m.def("moe_slot_fold_out(Tensor(a!) out, Tensor dn, Tensor wts, int top_k) -> ()");
    m.def("moe_fused_version() -> int");
    m.def("moe_fused_supported(int tier, int K) -> bool");
}

TORCH_LIBRARY_IMPL(pxq4, CUDA, m) {
    m.impl("moe_gateup_glu_out",      &moe_gateup_glu_out);
    m.impl("pxq2_moe_gateup_glu_out", &pxq2_moe_gateup_glu_out);
    m.impl("pxq3_moe_gateup_glu_out", &pxq3_moe_gateup_glu_out);
    m.impl("moe_down_fold_out",       &moe_down_fold_out);
    m.impl("pxq2_moe_down_fold_out",  &pxq2_moe_down_fold_out);
    m.impl("pxq3_moe_down_fold_out",  &pxq3_moe_down_fold_out);
    m.impl("moe_down_part_out",       &moe_down_part_out);
    m.impl("pxq2_moe_down_part_out",  &pxq2_moe_down_part_out);
    m.impl("pxq3_moe_down_part_out",  &pxq3_moe_down_part_out);
    m.impl("moe_slot_fold_out",       &moe_slot_fold_out);
}

TORCH_LIBRARY_IMPL(pxq4, CompositeExplicitAutograd, m) {
    m.impl("moe_fused_version",   &moe_fused_version);
    m.impl("moe_fused_supported", &moe_fused_supported);
}
