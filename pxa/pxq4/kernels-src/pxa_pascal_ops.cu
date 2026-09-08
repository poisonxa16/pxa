// pxa_pascal_ops.cu -- instantiations and launchers for the hand-fused small-op pack.
//
// -use_fast_math IS FORBIDDEN for this TU, for the same reason it is forbidden for the rest
// of the package: it enables contraction and reassociation that silently change the fp32
// order, and the whole correctness argument here is that the order matches the torch chain
// being replaced. The build script does not pass it; do not add it.

#include "pxa_pascal_ops.cuh"
#include "pxa_nat_silu_and_mul.cuh"
#include "pxa_pascal_ops_launch.h"

#include <cstdlib>

namespace pxa_pascal {

// -------------------------------------------------------------------------------------
// LAUNCH GEOMETRY. One block per row; VEC contiguous elements per thread group; the block
// width is H/VEC so every thread owns exactly one group at the widths this model uses
// (2048 -> 256x8, 256 -> 32x8, 128 -> 16x8).
//
// THE GEOMETRY IS PART OF THE NUMERICS. The fold order is a function of (H, VEC, bw), so
// changing this function changes the arithmetic and invalidates every gate run against it.
// The pairing here is the one measured in bench/reduce_probe_c.py; a wider block for the
// 128-wide GDN norm would occupy the card better and is a real tuning opportunity, but it
// must be re-gated rather than tuned in place.
// -------------------------------------------------------------------------------------
struct Geom { int vec; int bw; size_t smem; };

// -------------------------------------------------------------------------------------
// ATen's own reduce configuration, computed rather than searched.
//
// This is at::native::setReduceConfig specialised to the case this pack reduces: an fp32
// `x.pow(2).mean(-1)` over a contiguous [rows, H] tensor, one reduced dimension, ndim 2.
// Reproducing the CONFIG is what makes the kernel reproduce the ANSWER; the geometry is
// the numerics here, not a tuning parameter, so nothing in this function may be adjusted
// for occupancy without re-running the bit-identity table.
//
// PXA_OPS_EXACT=0 disables the exact path and keeps the deterministic fallback, which is
// the escape hatch if a future torch changes setReduceConfig under us -- the kernel would
// then be bit-exact against a reference that no longer exists, and the fallback is at
// least honest about being an approximation.
// -------------------------------------------------------------------------------------
struct AtenCfg {
    bool ok;        // is this shape inside the modelled family?
    int vec, bw, bh;
    size_t smem;
};

static inline int last_pow2_i(int n) {
    int p = 1;
    while ((p << 1) > 0 && (p << 1) <= n) p <<= 1;
    return n > 0 ? p : 0;
}

static inline AtenCfg aten_plan(int rows, int H) {
    AtenCfg c{false, 1, 32, 1, 0};
    static const bool disabled = [] {
        const char* e = getenv("PXA_OPS_EXACT");
        return e && *e && e[0] == '0';
    }();
    if (disabled) return c;

    const int WARP = 32, MNT = 512, VT0 = 4;   // mnt_wrapper<float>::MAX_NUM_THREADS
    int dim0 = H, dim1 = rows, vec = 1;
    // fastest_moving_stride == sizeof(float) and the reduction is on the fastest
    // dimension, so vectorisation engages exactly when the reduced extent exceeds 128.
    if (dim0 > 128) { vec = VT0; dim0 /= vec; }
    if (H % vec != 0) return c;

    const int d0p = dim0 < MNT ? last_pow2_i(dim0) : MNT;
    const int d1p = dim1 < MNT ? last_pow2_i(dim1) : MNT;
    int bw = d0p < WARP ? d0p : WARP;
    int bh = d1p < (MNT / bw) ? d1p : (MNT / bw);
    bw = d0p < (MNT / bh) ? d0p : (MNT / bh);

    // The y-split and the CTA split must NOT engage: both are modelled in
    // bench/aten_reduce_config.py and neither fires for any shape this pack serves, and
    // the CTA split would make the fold depend on the multiprocessor count. If a shape
    // ever reaches them, decline it here rather than guess.
    const int step_input = bw;
    const int vpt = (H + step_input - 1) / step_input;
    const int warp_split_threshold = (bh * 16 < 256) ? bh * 16 : 256;
    if (vpt >= warp_split_threshold) return c;

    // The kernel's x-reduce assumes the shuffle stage sees a full warp.
    if (bw < WARP) return c;
    if (bw * bh > 1024) return c;

    c.ok = true; c.vec = vec; c.bw = bw; c.bh = bh;
    c.smem = (size_t)bw * bh * sizeof(float);
    return c;
}

#define PXA_ATEN_VEC_SWITCH(vec, BODY)                  \
    switch (vec) {                                      \
        case 4: { constexpr int V = 4; BODY; break; }   \
        default: { constexpr int V = 1; BODY; break; }  \
    }


static inline Geom plan(int H) {
    int vec = 1;
    for (int v = 8; v >= 1; v >>= 1) {
        if (H % v == 0 && (H / v) <= 1024) { vec = v; break; }
    }
    int bw = H / vec;
    if (bw > 1024) bw = 1024;
    if (bw < 1) bw = 1;
    return Geom{vec, bw, (size_t)bw * sizeof(float)};
}

#define PXA_VEC_SWITCH(vec, BODY)                       \
    switch (vec) {                                      \
        case 8: { constexpr int V = 8; BODY; break; }   \
        case 4: { constexpr int V = 4; BODY; break; }   \
        case 2: { constexpr int V = 2; BODY; break; }   \
        default: { constexpr int V = 1; BODY; break; }  \
    }

// ------------------------------------------------------------------ gemma + residual ---
template <typename XT, typename RT>
static void gemma_add_dispatch(void* out, void* res_out, const void* x, const void* res_in,
                               const float* w, float eps, int rows, int H, const Geom& g,
                               int64_t xs, int64_t ris, int64_t rs, int64_t os,
                               cudaStream_t stream)
{
    PXA_VEC_SWITCH(g.vec, (k_gemma_add_rms_norm<XT, RT, V><<<rows, g.bw, g.smem, stream>>>(
        (__half*)out, (float*)res_out, (const XT*)x, (const RT*)res_in, w, eps, H, g.bw,
        xs, ris, rs, os)));
}

void launch_gemma_add_rms_norm(void* out, void* res_out, const void* x, const void* res_in,
                               const float* w, float eps, int rows, int H,
                               int x_dtype, int r_dtype,
                               int64_t xs, int64_t ris, int64_t rs, int64_t os,
                               cudaStream_t stream)
{
    const AtenCfg a = aten_plan(rows, H);
    if (a.ok) {
        const dim3 blk(a.bw, a.bh), grd((rows + a.bh - 1) / a.bh);
        if (x_dtype == DT_F16) {
            if (r_dtype == DT_F16) {
                PXA_ATEN_VEC_SWITCH(a.vec, (k_gemma_add_rms_norm_aten<__half, __half, V><<<grd, blk, a.smem, stream>>>(
                    (__half*)out, (float*)res_out, (const __half*)x, (const __half*)res_in, w, eps, H, rows, a.bw, xs, ris, rs, os)));
            } else {
                PXA_ATEN_VEC_SWITCH(a.vec, (k_gemma_add_rms_norm_aten<__half, float, V><<<grd, blk, a.smem, stream>>>(
                    (__half*)out, (float*)res_out, (const __half*)x, (const float*)res_in, w, eps, H, rows, a.bw, xs, ris, rs, os)));
            }
        } else {
            if (r_dtype == DT_F16) {
                PXA_ATEN_VEC_SWITCH(a.vec, (k_gemma_add_rms_norm_aten<float, __half, V><<<grd, blk, a.smem, stream>>>(
                    (__half*)out, (float*)res_out, (const float*)x, (const __half*)res_in, w, eps, H, rows, a.bw, xs, ris, rs, os)));
            } else {
                PXA_ATEN_VEC_SWITCH(a.vec, (k_gemma_add_rms_norm_aten<float, float, V><<<grd, blk, a.smem, stream>>>(
                    (__half*)out, (float*)res_out, (const float*)x, (const float*)res_in, w, eps, H, rows, a.bw, xs, ris, rs, os)));
            }
        }
        return;
    }
    const Geom g = plan(H);
    if (x_dtype == DT_F16) {
        if (r_dtype == DT_F16)
            gemma_add_dispatch<__half, __half>(out, res_out, x, res_in, w, eps, rows, H, g, xs, ris, rs, os, stream);
        else
            gemma_add_dispatch<__half, float>(out, res_out, x, res_in, w, eps, rows, H, g, xs, ris, rs, os, stream);
    } else {
        if (r_dtype == DT_F16)
            gemma_add_dispatch<float, __half>(out, res_out, x, res_in, w, eps, rows, H, g, xs, ris, rs, os, stream);
        else
            gemma_add_dispatch<float, float>(out, res_out, x, res_in, w, eps, rows, H, g, xs, ris, rs, os, stream);
    }
}

// ------------------------------------------------------------------ gemma no residual ---
void launch_gemma_rms_norm(void* out, const void* x, const float* w, float eps,
                           int rows, int H, int dtype, int64_t xs, int64_t os,
                           cudaStream_t stream)
{
    const AtenCfg a = aten_plan(rows, H);
    if (a.ok) {
        const dim3 blk(a.bw, a.bh), grd((rows + a.bh - 1) / a.bh);
        if (dtype == DT_F16) {
            PXA_ATEN_VEC_SWITCH(a.vec, (k_gemma_rms_norm_aten<__half, __half, V><<<grd, blk, a.smem, stream>>>(
                (__half*)out, (const __half*)x, w, eps, H, rows, a.bw, xs, os)));
        } else {
            PXA_ATEN_VEC_SWITCH(a.vec, (k_gemma_rms_norm_aten<float, float, V><<<grd, blk, a.smem, stream>>>(
                (float*)out, (const float*)x, w, eps, H, rows, a.bw, xs, os)));
        }
        return;
    }
    const Geom g = plan(H);
    if (dtype == DT_F16) {
        PXA_VEC_SWITCH(g.vec, (k_gemma_rms_norm<__half, __half, V><<<rows, g.bw, g.smem, stream>>>(
            (__half*)out, (const __half*)x, w, eps, H, g.bw, xs, os)));
    } else {
        PXA_VEC_SWITCH(g.vec, (k_gemma_rms_norm<float, float, V><<<rows, g.bw, g.smem, stream>>>(
            (float*)out, (const float*)x, w, eps, H, g.bw, xs, os)));
    }
}

// ------------------------------------------------------------------------- gated norm ---
void launch_rms_norm_gated(void* out, const void* x, const void* z, const float* w,
                           float eps, int rows, int H, int dtype, int act,
                           int64_t xs, int64_t zs, int64_t os, cudaStream_t stream)
{
    const Geom g = plan(H);
    if (dtype == DT_F16) {
        PXA_VEC_SWITCH(g.vec, (k_rms_norm_gated<__half, V><<<rows, g.bw, g.smem, stream>>>(
            (__half*)out, (const __half*)x, (const __half*)z, w, eps, H, g.bw, act, xs, zs, os)));
    } else {
        PXA_VEC_SWITCH(g.vec, (k_rms_norm_gated<float, V><<<rows, g.bw, g.smem, stream>>>(
            (float*)out, (const float*)x, (const float*)z, w, eps, H, g.bw, act, xs, zs, os)));
    }
}

// ------------------------------------------------------------------- plain RMSNorm f16 ---
void launch_rms_norm_h16(void* out, const void* x, const void* w, float eps, int rows,
                         int H, int64_t xs, int64_t os, cudaStream_t stream)
{
    const Geom g = plan(H);
    PXA_VEC_SWITCH(g.vec, (k_rms_norm_h16<V><<<rows, g.bw, g.smem, stream>>>(
        (__half*)out, (const __half*)x, (const __half*)w, eps, H, g.bw, xs, os)));
}

void launch_fused_add_rms_norm_h16(void* out, void* res_out, const void* x,
                                   const void* res_in, const void* w, float eps,
                                   int rows, int H, int64_t xs, int64_t rs, int64_t os,
                                   cudaStream_t stream)
{
    const Geom g = plan(H);
    PXA_VEC_SWITCH(g.vec, (k_fused_add_rms_norm_h16<V><<<rows, g.bw, g.smem, stream>>>(
        (__half*)out, (__half*)res_out, (const __half*)x, (const __half*)res_in,
        (const __half*)w, eps, H, g.bw, xs, rs, os)));
}

// ------------------------------------------------------------------------------ SwiGLU ---
// Body from the fused-MoE work (pxa_nat_silu_and_mul.cuh); only the launch geometry is ours.
// N is the OUTPUT width, i.e. half the input's last dimension.
void launch_silu_and_mul_f16(void* out, const void* x, int rows, int N,
                             int64_t xs, int64_t os, cudaStream_t stream)
{
    int vec = 1;
    for (int v = 8; v >= 1; v >>= 1) { if (N % v == 0) { vec = v; break; } }
    int threads = N / vec;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    switch (vec) {
        case 8: pxa_nat::k_silu_and_mul_f16<8><<<rows, threads, 0, stream>>>(
                    (__half*)out, (const __half*)x, N, xs, os); break;
        case 4: pxa_nat::k_silu_and_mul_f16<4><<<rows, threads, 0, stream>>>(
                    (__half*)out, (const __half*)x, N, xs, os); break;
        case 2: pxa_nat::k_silu_and_mul_f16<2><<<rows, threads, 0, stream>>>(
                    (__half*)out, (const __half*)x, N, xs, os); break;
        default: pxa_nat::k_silu_and_mul_f16<1><<<rows, threads, 0, stream>>>(
                    (__half*)out, (const __half*)x, N, xs, os); break;
    }
}

// v1: first release of the pack. Bumped whenever the ARITHMETIC changes, so a mismatched
// lib is diagnosable from the sidecar instead of showing up as a failed byte gate.
// v2: the ATen-exact fold for the modelled decode family (bit-identical to the
// native torch reduction), with v1's deterministic fold retained as the fallback.
int pascal_ops_version() { return 2; }

}  // namespace pxa_pascal
