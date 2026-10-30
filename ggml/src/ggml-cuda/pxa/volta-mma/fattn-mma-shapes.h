#pragma once

// fattn-mma-shapes.h -- the shape table of the vendored MMA flash-attention family (fattn-mma-ml.cuh)
// as it runs on this fork's cards, kept in ONE place that both sides read:
//
//   * the DEVICE side: which compiled variants of flash_attn_ext_f16 carry device code on each arch.
//     The kernel's guards call these functions, so they cannot drift from what is written here.
//   * the HOST side: which (ncols1, ncols2) the sm_70 dispatcher in fattn-volta-mma.cu launches for a
//     node of a given width and GQA ratio, and on which arch that route is taken at all.
//
// Plain constexpr C++ with no CUDA dependency, so the host-only table test
// (tests/test-fa-volta-mma-shapes.cpp) can enumerate every shape the dispatcher can pick and check it
// against the device-code predicate for sm_60, sm_61 and sm_70. A dispatch to a compiled-out stub then
// fails a CPU ctest instead of aborting a server mid-decode with "has no device code compatible with
// CUDA arch 700" (bug #22738: Nemotron-3-Nano-4B, head 128, 40 q-heads / 8 kv-heads = GQA 5, ngram
// self-spec verify widths 9..16 -> ncols1 16 x ncols2 1 = 16 columns, a stub on sm_70).

#if defined(__CUDACC__)
#define PXA_FMS_HD __host__ __device__
#else
#define PXA_FMS_HD
#endif

namespace pxa_fattn_mma_shapes {

// ---- device code ---------------------------------------------------------------------------------

// The sparse (gathered) K/V variant exists for these two shapes only (mainline's list).
static constexpr PXA_FMS_HD bool may_use_sparse(const int DKQ, const int DV, const int ncols1, const int ncols2) {
    return (DKQ == 512 && DV == 512 && ncols1 == 1 && ncols2 == 8) ||
           (DKQ == 576 && DV == 512 && ncols1 == 1 && ncols2 == 16);
}

// The arch-independent variant pruning at the top of flash_attn_ext_f16 ("skip unused kernel variants
// for faster compilation"), one function per guard: logit softcap only at head 128/256/512, head 192
// only with ncols2 8 or 16, the sparse variant only where may_use_sparse() says so.
static constexpr PXA_FMS_HD bool softcap_has_code(const int DKQ, const bool softcap) {
    return !softcap || DKQ == 128 || DKQ == 256 || DKQ == 512;
}
static constexpr PXA_FMS_HD bool d192_has_code(const int DKQ, const int ncols2) {
    return DKQ != 192 || ncols2 == 8 || ncols2 == 16;
}
static constexpr PXA_FMS_HD bool sparse_has_code(const int DKQ, const int DV, const int ncols1, const int ncols2,
                                                 const bool sparse) {
    return !sparse || may_use_sparse(DKQ, DV, ncols1, ncols2);
}
static constexpr PXA_FMS_HD bool variant_has_code(const int DKQ, const int DV, const int ncols1, const int ncols2,
                                                  const bool softcap, const bool sparse) {
    return softcap_has_code(DKQ, softcap) && d192_has_code(DKQ, ncols2) &&
           sparse_has_code(DKQ, DV, ncols1, ncols2, sparse);
}

// sm_70 (VOLTA_MMA_AVAILABLE, i.e. __CUDA_ARCH__ == 700 exactly): the m8n8k4 tiles need at least 32
// columns per block; every narrower variant is compiled out.
static constexpr PXA_FMS_HD bool volta_cols_have_code(const int ncols) {
    return ncols >= 32;
}

// sm_75 (__CUDA_ARCH__ == 750 exactly): at most 32 columns per block.
static constexpr PXA_FMS_HD bool turing_cols_have_code(const int ncols) {
    return ncols <= 32;
}

// Does the variant compiled for `arch` (a __CUDA_ARCH__ value: 600, 610, 700, 750, 800, ...) carry
// device code? Below sm_70 the whole body is compiled out (it needs VOLTA_MMA_AVAILABLE or
// TURING_MMA_AVAILABLE), so a Pascal card must never be routed to this family at all.
static constexpr PXA_FMS_HD bool has_device_code(const int arch, const int DKQ, const int DV,
                                                 const int ncols1, const int ncols2,
                                                 const bool softcap, const bool sparse) {
    return arch >= 700 &&
           variant_has_code(DKQ, DV, ncols1, ncols2, softcap, sparse) &&
           (arch != 700 || volta_cols_have_code(ncols1*ncols2)) &&
           (arch != 750 || turing_cols_have_code(ncols1*ncols2));
}

// ---- host dispatch (fattn-volta-mma.cu, fa-route.cu) ---------------------------------------------

// The route exists on cc 7.0 only; every other arch keeps its own dispatch.
static constexpr PXA_FMS_HD bool volta_route_arch(const int cc) {
    return cc == 700;
}

// Head 128 / 256: the batch gate of pxa_fa_mma_volta_take() (decode widths 1..8 keep the incumbent
// routes; PXA_FA_MMA_VOLTA=2 is accepted and treated as =1).
static constexpr PXA_FMS_HD bool volta_takes_width(const int ne1) {
    return ne1 > 8;
}

// Head 128 / 256, ncols2: mainline's switch_ncols2, Volta branch -- the largest of 8/4/2 that divides
// the GQA ratio when the GQA packing is allowed, else 1.
static constexpr PXA_FMS_HD int volta_pick_ncols2(const int gqa_ratio, const bool use_gqa_opt) {
    return use_gqa_opt && gqa_ratio % 8 == 0 ? 8 :
           use_gqa_opt && gqa_ratio % 4 == 0 ? 4 :
           use_gqa_opt && gqa_ratio % 2 == 0 ? 2 : 1;
}

// Head 128 / 256, ncols1: mainline's switch_ncols1 with turing_mma_available() false, which is the
// sm_70 case. Upstream gates BOTH narrow rungs (8/ncols2 and 16/ncols2) on Turing MMA; the port kept
// the 16/ncols2 rung, and 16 columns is a stub on sm_70 at every ncols2 (bug #22738). The ladder
// therefore starts at 32/ncols2: a node narrower than that runs a 32-column tile with the tail masked
// (ne01 bounds the Q rows), exactly as upstream runs it on Volta.
static constexpr PXA_FMS_HD int volta_pick_ncols1(const int ne1, const int ncols2) {
    return ne1 <= 32/ncols2 ? 32/ncols2 : 64/ncols2;
}

// Head 512 / 512 (PXA_FA_D512_VOLTA=1): GQA packing 8 only, any width.
static constexpr PXA_FMS_HD int volta_d512_ncols2() {
    return 8;
}
static constexpr PXA_FMS_HD int volta_d512_pick_ncols1(const int ne1) {
    return ne1 <= 4 ? 4 : 8;
}

} // namespace pxa_fattn_mma_shapes
