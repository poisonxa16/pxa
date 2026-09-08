// pxq_q8_torch.cpp -- torch bindings for the int8 per-row-scale head path.
//
// TORCH_LIBRARY_FRAGMENT on the existing `pxq4` namespace, additive, touching no existing op
// and no existing file -- the same pattern this tree requires of every other kernel pack.

#include <torch/library.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstdlib>

#include "pxq_q8_launch.h"

namespace {

// Above this many rows the GEMV's one-weight-pass-per-token stops paying and the dequant
// amortises instead. Same shape of policy as pxq linear_out, same env override. Default 8
// mirrors the decode capture ladder: at M <= 8 the head is read once per token either way,
// and the GEMV reads half the bytes.
constexpr int64_t kQ8MmvMaxM = 8;

int64_t env_i64(const char * name, int64_t dflt) {
    const char * e = getenv(name);
    return (e && *e) ? (int64_t)atoll(e) : dflt;
}

struct Q8Geom { int N; int K; };

Q8Geom check_w(const at::Tensor & w, const at::Tensor & scale) {
    TORCH_CHECK(w.is_cuda() && scale.is_cuda(), "pxq_q8: w/scale must be CUDA tensors");
    TORCH_CHECK(w.scalar_type() == at::kChar, "pxq_q8: w must be int8, got ", w.scalar_type());
    TORCH_CHECK(scale.scalar_type() == at::kHalf, "pxq_q8: scale must be float16, got ",
                scale.scalar_type());
    TORCH_CHECK(w.dim() == 2, "pxq_q8: w must be [N, K], got dim ", w.dim());
    // [N] or [N, 1] are both accepted: vLLM's parameter machinery likes a trailing 1 so the
    // scale has an output_dim to shard on, and the kernel only ever indexes it by row.
    TORCH_CHECK(scale.numel() == w.size(0),
                "pxq_q8: scale must have one entry per row: ", scale.numel(), " vs ", w.size(0));
    TORCH_CHECK(w.is_contiguous() && scale.is_contiguous(),
                "pxq_q8: w and scale must be contiguous -- the kernel reads whole rows as "
                "char4 and a narrowed view would misalign them");
    TORCH_CHECK(w.size(0) > 0 && w.size(1) > 0, "pxq_q8: empty weight");
    return Q8Geom{(int)w.size(0), (int)w.size(1)};
}

void q8_dequant_out(at::Tensor & out, const at::Tensor & w, const at::Tensor & scale) {
    const Q8Geom g = check_w(w, scale);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "pxq_q8: out must be CUDA fp16");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == g.N && out.size(1) == g.K,
                "pxq_q8: out must be [", g.N, ", ", g.K, "]");
    TORCH_CHECK(out.is_contiguous(), "pxq_q8: out must be contiguous");
    const at::cuda::OptionalCUDAGuard guard(at::device_of(w));
    pxq_q8_launch_dequant_f16(w.data_ptr<int8_t>(), scale.data_ptr(), out.data_ptr(),
                              g.N, g.K, at::cuda::getCurrentCUDAStream());
}

void q8_mmv_out(at::Tensor & out, const at::Tensor & x, const at::Tensor & w,
                const at::Tensor & scale) {
    const Q8Geom g = check_w(w, scale);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "pxq_q8: x must be CUDA fp16");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "pxq_q8: out must be CUDA fp16");
    TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "pxq_q8: x and out must be 2D");
    TORCH_CHECK(x.size(1) == g.K, "pxq_q8: x K=", x.size(1), " != weight K=", g.K);
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == g.N,
                "pxq_q8: out must be [", x.size(0), ", ", g.N, "]");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "pxq_q8: x and out must be contiguous");
    const int64_t M = x.size(0);
    if (M == 0) return;
    const at::cuda::OptionalCUDAGuard guard(at::device_of(w));
    pxq_q8_launch_mmv_f16(w.data_ptr<int8_t>(), scale.data_ptr(), x.data_ptr(), out.data_ptr(),
                          (int)M, g.N, g.K, at::cuda::getCurrentCUDAStream());
}

// Persistent per-device fp16 arena for the prefill dequant, with the same growth rule and the
// same reason as the pxq one: a CUDA graph records raw addresses, so reassigning the buffer
// after a capture hands a live block back to the allocator.
std::array<at::Tensor, 64> g_q8_arena;

at::Tensor & q8_arena(const at::Tensor & like, int64_t need) {
    const int dev = (int)like.device().index();
    TORCH_CHECK(dev >= 0 && dev < 64, "pxq_q8: device index out of range: ", dev);
    at::Tensor & a = g_q8_arena[dev];
    if (!a.defined() || a.numel() < need) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &cap);
        TORCH_CHECK(cap == cudaStreamCaptureStatusNone,
                    "pxq_q8: the head dequant arena would have to grow during CUDA graph "
                    "capture (need ", need, " fp16 elems). Every shape a captured graph "
                    "replays must have run once eagerly first.");
        a = at::empty({need}, like.options().dtype(at::kHalf));
    }
    return a;
}

void q8_linear_out(at::Tensor & out, const at::Tensor & x, const at::Tensor & w,
                   const at::Tensor & scale) {
    const Q8Geom g = check_w(w, scale);
    const int64_t M = x.size(0);
    if (M == 0) return;
    static const int64_t kMaxM = env_i64("PXQ_Q8_MMV_MAX_M", kQ8MmvMaxM);
    if (M <= kMaxM) {
        q8_mmv_out(out, x, w, scale);
        return;
    }
    // Prefill: dequantise once and let cuBLAS have it. Above the threshold this is the right
    // call and not a fallback -- the dequant cost is paid once per forward rather than once
    // per token, and cuBLAS beats a hand GEMV on a wide batch by more than the extra traffic
    // costs. See the byte account in pxq_q8.cuh for where the crossover comes from.
    const at::cuda::OptionalCUDAGuard guard(at::device_of(w));
    at::Tensor & buf = q8_arena(x, (int64_t)g.N * g.K);
    at::Tensor wd = buf.narrow(0, 0, (int64_t)g.N * g.K).view({g.N, g.K});
    pxq_q8_launch_dequant_f16(w.data_ptr<int8_t>(), scale.data_ptr(), wd.data_ptr(),
                              g.N, g.K, at::cuda::getCurrentCUDAStream());
    at::mm_out(out, x, wd.t());
}

int64_t q8_selftest() { return (int64_t)pxq_q8_selftest(); }
int64_t q8_mmv_max_m() { return env_i64("PXQ_Q8_MMV_MAX_M", kQ8MmvMaxM); }

}  // namespace

TORCH_LIBRARY_FRAGMENT(pxq4, m) {
    m.def("q8_dequant_out(Tensor(a!) out, Tensor w, Tensor scale) -> ()");
    m.def("q8_mmv_out(Tensor(a!) out, Tensor x, Tensor w, Tensor scale) -> ()");
    m.def("q8_linear_out(Tensor(a!) out, Tensor x, Tensor w, Tensor scale) -> ()");
    m.def("q8_selftest() -> int");
    m.def("q8_mmv_max_m() -> int");
}

TORCH_LIBRARY_IMPL(pxq4, CUDA, m) {
    m.impl("q8_dequant_out", &q8_dequant_out);
    m.impl("q8_mmv_out",     &q8_mmv_out);
    m.impl("q8_linear_out",  &q8_linear_out);
}

TORCH_LIBRARY_IMPL(pxq4, CompositeExplicitAutograd, m) {
    m.impl("q8_selftest",  &q8_selftest);
    m.impl("q8_mmv_max_m", &q8_mmv_max_m);
}
