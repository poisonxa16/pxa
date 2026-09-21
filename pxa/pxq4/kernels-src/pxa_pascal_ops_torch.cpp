// pxa_pascal_ops_torch.cpp -- torch op bindings for the hand-fused Pascal/Volta small-op pack.
//
// TORCH_LIBRARY_FRAGMENT(pxq4), never TORCH_LIBRARY and never the `_C` namespace. The `pxq4`
// library is DEFINED once in pxq4_kernel_torch.cpp, which is frozen; a fragment extends it
// from another TU without touching that file. Anything that wants these ops to appear under
// `_C` -- so the fork's own `getattr(torch.ops._C, "silu_and_mul", None)` finds them -- gets
// that from the PYTHON side of the sidecar, where it can be env-gated and reversed without a
// rebuild, rather than from a second C++ registration that would be baked into the image.
//
// Every op is OUT-PARAMETER and this TU allocates nothing. That is not style: these run
// inside the captured decode graph, so an allocation here would be served from the graph's
// private pool and pin per-capture-size memory, and a host read would break capture outright.

#include <torch/library.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "pxa_pascal_ops_launch.h"

namespace {

using namespace pxa_pascal;

// Flattens any contiguous-in-the-last-dim tensor to [rows, H] and returns the row stride in
// ELEMENTS. A 3-D q/k norm input ([tokens, heads, head_dim]) flattens exactly, which is why
// the ops accept dim > 2 instead of forcing a .contiguous() copy at the call site.
struct Rows { int64_t rows; int64_t H; int64_t stride; };

Rows rows_of(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), "pxa_pascal: ", name, " must be a CUDA tensor");
    TORCH_CHECK(t.dim() >= 2, "pxa_pascal: ", name, " must have dim >= 2, got ", t.dim());
    TORCH_CHECK(t.stride(-1) == 1, "pxa_pascal: ", name, " must be contiguous in its last dim");
    const int64_t H = t.size(-1);
    int64_t rows = 1;
    for (int64_t i = 0; i + 1 < t.dim(); ++i) rows *= t.size(i);
    // Leading dims must be packed for the flattening to be exact; a genuinely strided view
    // would silently read the wrong rows, so this is checked rather than assumed.
    TORCH_CHECK(t.is_contiguous(),
                "pxa_pascal: ", name, " must be contiguous (leading dims are flattened)");
    return Rows{rows, H, H};
}

int dtag(const at::Tensor& t, const char* name) {
    if (t.scalar_type() == at::kHalf)  return DT_F16;
    if (t.scalar_type() == at::kFloat) return DT_F32;
    TORCH_CHECK(false, "pxa_pascal: ", name, " must be float16 or float32, got ",
                t.scalar_type());
}

void check_width(int64_t H, const char* what) {
    TORCH_CHECK(H > 0 && H % 2 == 0, "pxa_pascal: ", what, " width must be even, got ", H);
    TORCH_CHECK(H <= 8192, "pxa_pascal: ", what, " width ", H,
                " exceeds the one-block-per-row geometry this pack is gated for");
}

// ------------------------------------------------------------- GemmaRMSNorm + residual ---
void gemma_add_rms_norm_out(at::Tensor& out, at::Tensor& residual_out,
                            const at::Tensor& x, const at::Tensor& residual,
                            const at::Tensor& weight, double eps)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    const Rows rx = rows_of(x, "x");
    const Rows rr = rows_of(residual, "residual");
    const Rows ro = rows_of(out, "out");
    const Rows rq = rows_of(residual_out, "residual_out");
    check_width(rx.H, "x");
    TORCH_CHECK(rr.rows == rx.rows && rr.H == rx.H, "pxa_pascal: residual shape mismatch");
    TORCH_CHECK(ro.rows == rx.rows && ro.H == rx.H, "pxa_pascal: out shape mismatch");
    TORCH_CHECK(rq.rows == rx.rows && rq.H == rx.H, "pxa_pascal: residual_out shape mismatch");
    TORCH_CHECK(out.scalar_type() == at::kHalf, "pxa_pascal: out must be float16");
    TORCH_CHECK(residual_out.scalar_type() == at::kFloat,
                "pxa_pascal: residual_out must be float32 (GemmaRMSNorm carries the residual "
                "in fp32; a float16 residual_out would be a different chain)");
    TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kFloat && weight.dim() == 1
                    && weight.size(0) == rx.H && weight.is_contiguous(),
                "pxa_pascal: weight must be a contiguous fp32 [H] tensor");

    launch_gemma_add_rms_norm(out.data_ptr(), residual_out.data_ptr(), x.data_ptr(),
                              residual.data_ptr(), weight.data_ptr<float>(), (float)eps,
                              (int)rx.rows, (int)rx.H, dtag(x, "x"), dtag(residual, "residual"),
                              rx.stride, rr.stride, rq.stride, ro.stride,
                              at::cuda::getCurrentCUDAStream());
}

// ------------------------------------------------------------- GemmaRMSNorm no residual ---
void gemma_rms_norm_out(at::Tensor& out, const at::Tensor& x,
                        const at::Tensor& weight, double eps)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    const Rows rx = rows_of(x, "x");
    const Rows ro = rows_of(out, "out");
    check_width(rx.H, "x");
    TORCH_CHECK(ro.rows == rx.rows && ro.H == rx.H, "pxa_pascal: out shape mismatch");
    TORCH_CHECK(out.scalar_type() == x.scalar_type(),
                "pxa_pascal: out dtype must equal x dtype (the reference ends in "
                ".to(orig_dtype))");
    TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kFloat && weight.dim() == 1
                    && weight.size(0) == rx.H && weight.is_contiguous(),
                "pxa_pascal: weight must be a contiguous fp32 [H] tensor");

    launch_gemma_rms_norm(out.data_ptr(), x.data_ptr(), weight.data_ptr<float>(), (float)eps,
                          (int)rx.rows, (int)rx.H, dtag(x, "x"), rx.stride, ro.stride,
                          at::cuda::getCurrentCUDAStream());
}

// ------------------------------------------------------------------------- RMSNormGated ---
void rms_norm_gated_out(at::Tensor& out, const at::Tensor& x, const at::Tensor& z,
                        const at::Tensor& weight, double eps, int64_t act)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    const Rows rx = rows_of(x, "x");
    const Rows rz = rows_of(z, "z");
    const Rows ro = rows_of(out, "out");
    check_width(rx.H, "x");
    TORCH_CHECK(rz.rows == rx.rows && rz.H == rx.H, "pxa_pascal: z shape mismatch");
    TORCH_CHECK(ro.rows == rx.rows && ro.H == rx.H, "pxa_pascal: out shape mismatch");
    TORCH_CHECK(z.scalar_type() == x.scalar_type() && out.scalar_type() == x.scalar_type(),
                "pxa_pascal: x, z and out must share a dtype");
    TORCH_CHECK(act == 0 || act == 1, "pxa_pascal: act must be 0 (silu) or 1 (sigmoid)");
    TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kFloat && weight.dim() == 1
                    && weight.size(0) == rx.H && weight.is_contiguous(),
                "pxa_pascal: weight must be a contiguous fp32 [H] tensor");

    launch_rms_norm_gated(out.data_ptr(), x.data_ptr(), z.data_ptr(),
                          weight.data_ptr<float>(), (float)eps, (int)rx.rows, (int)rx.H,
                          dtag(x, "x"), (int)act, rx.stride, rz.stride, ro.stride,
                          at::cuda::getCurrentCUDAStream());
}

// -------------------------------------------------------- plain RMSNorm (the vllm.ir op) ---
void rms_norm_out(at::Tensor& out, const at::Tensor& x, const at::Tensor& weight, double eps)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    const Rows rx = rows_of(x, "x");
    const Rows ro = rows_of(out, "out");
    check_width(rx.H, "x");
    TORCH_CHECK(ro.rows == rx.rows && ro.H == rx.H, "pxa_pascal: out shape mismatch");
    TORCH_CHECK(x.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf
                    && weight.scalar_type() == at::kHalf,
                "pxa_pascal: rms_norm_out is the fp16 x / fp16 weight form; the fp32 weight "
                "case rounds differently and is gemma_rms_norm_out");
    TORCH_CHECK(weight.is_cuda() && weight.dim() == 1 && weight.size(0) == rx.H
                    && weight.is_contiguous(), "pxa_pascal: weight must be contiguous [H]");

    launch_rms_norm_h16(out.data_ptr(), x.data_ptr(), weight.data_ptr(), (float)eps,
                        (int)rx.rows, (int)rx.H, rx.stride, ro.stride,
                        at::cuda::getCurrentCUDAStream());
}

void fused_add_rms_norm_out(at::Tensor& out, at::Tensor& residual_out, const at::Tensor& x,
                            const at::Tensor& residual, const at::Tensor& weight, double eps)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    const Rows rx = rows_of(x, "x");
    const Rows rr = rows_of(residual, "residual");
    const Rows ro = rows_of(out, "out");
    const Rows rq = rows_of(residual_out, "residual_out");
    check_width(rx.H, "x");
    TORCH_CHECK(rr.rows == rx.rows && rr.H == rx.H, "pxa_pascal: residual shape mismatch");
    TORCH_CHECK(ro.rows == rx.rows && ro.H == rx.H, "pxa_pascal: out shape mismatch");
    TORCH_CHECK(rq.rows == rx.rows && rq.H == rx.H, "pxa_pascal: residual_out shape mismatch");
    TORCH_CHECK(x.scalar_type() == at::kHalf && residual.scalar_type() == at::kHalf
                    && out.scalar_type() == at::kHalf
                    && residual_out.scalar_type() == at::kHalf
                    && weight.scalar_type() == at::kHalf,
                "pxa_pascal: fused_add_rms_norm_out is the all-fp16 form");
    TORCH_CHECK(weight.is_cuda() && weight.dim() == 1 && weight.size(0) == rx.H
                    && weight.is_contiguous(), "pxa_pascal: weight must be contiguous [H]");

    launch_fused_add_rms_norm_h16(out.data_ptr(), residual_out.data_ptr(), x.data_ptr(),
                                  residual.data_ptr(), weight.data_ptr(), (float)eps,
                                  (int)rx.rows, (int)rx.H, rx.stride, rq.stride, ro.stride,
                                  at::cuda::getCurrentCUDAStream());
}

// ------------------------------------------------------------------------------ SwiGLU ---
void silu_and_mul_out(at::Tensor& out, const at::Tensor& x)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    const Rows rx = rows_of(x, "x");
    const Rows ro = rows_of(out, "out");
    TORCH_CHECK(x.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf,
                "pxa_pascal: silu_and_mul_out is fp16 only");
    TORCH_CHECK(rx.H % 2 == 0, "pxa_pascal: x last dim must be even, got ", rx.H);
    TORCH_CHECK(ro.rows == rx.rows && ro.H == rx.H / 2,
                "pxa_pascal: out must be [rows, x.size(-1)/2]");

    launch_silu_and_mul_f16(out.data_ptr(), x.data_ptr(), (int)rx.rows, (int)ro.H,
                            rx.stride, ro.stride, at::cuda::getCurrentCUDAStream());
}

int64_t pascal_ops_version_op() { return (int64_t)pascal_ops_version(); }

}  // namespace

TORCH_LIBRARY_FRAGMENT(pxq4, m) {
    m.def("gemma_add_rms_norm_out(Tensor(a!) out, Tensor(b!) residual_out, Tensor x, "
          "Tensor residual, Tensor weight, float eps) -> ()");
    m.def("gemma_rms_norm_out(Tensor(a!) out, Tensor x, Tensor weight, float eps) -> ()");
    m.def("rms_norm_gated_out(Tensor(a!) out, Tensor x, Tensor z, Tensor weight, "
          "float eps, int act) -> ()");
    m.def("rms_norm_out(Tensor(a!) out, Tensor x, Tensor weight, float eps) -> ()");
    m.def("fused_add_rms_norm_out(Tensor(a!) out, Tensor(b!) residual_out, Tensor x, "
          "Tensor residual, Tensor weight, float eps) -> ()");
    m.def("silu_and_mul_out(Tensor(a!) out, Tensor x) -> ()");
    m.def("pascal_ops_version() -> int");
}

TORCH_LIBRARY_IMPL(pxq4, CUDA, m) {
    m.impl("gemma_add_rms_norm_out", &gemma_add_rms_norm_out);
    m.impl("gemma_rms_norm_out",     &gemma_rms_norm_out);
    m.impl("rms_norm_gated_out",     &rms_norm_gated_out);
    m.impl("rms_norm_out",           &rms_norm_out);
    m.impl("fused_add_rms_norm_out", &fused_add_rms_norm_out);
    m.impl("silu_and_mul_out",       &silu_and_mul_out);
}

TORCH_LIBRARY_IMPL(pxq4, CompositeExplicitAutograd, m) {
    m.impl("pascal_ops_version", &pascal_ops_version_op);
}
