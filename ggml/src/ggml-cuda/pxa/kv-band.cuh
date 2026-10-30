// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// kv-band.cuh -- PXA_KV_BAND_EXACT: the attention chain's answer must not depend on the KV cell a band starts at.
//
// THE DEFECT (todo v100-kv-placement-dependence, test-kv-band-align on sm_70). The unified KV ring lays a
// sequence's band down at the first free run, so one prompt can start at cell 0 in one request and at cell 101
// in the next. Every cell of another sequence is masked to -inf, the softmax sends it to an exact 0, and a
// correct engine must give the same bytes at every start. On a V100 it did not: offsets that are not a multiple
// of 4 moved the logits by up to 0.013 (top-1 unchanged), with -fa on and off alike, while a P100 looked exact at prefill (its decode was not, see below).
//
// THE MECHANISM. The unfused attention chain (-fa off, PXA_MODE=max, and every flash-attention node the card
// declines, which PXA_FA_GPU_FALLBACK keeps on the card as this chain) ends in the P.V product
// dst = mul_mat(V, soft_max(K.Q + mask)), a batched cuBLAS GEMM whose reduction axis k is the KV CELL. On a
// tensor-core card cuBLAS runs it on HMMA (Volta m8n8k4: four k at a time, and with COMPUTE_16F the running sum
// is rounded to fp16 after every step). Which four cells share a step is fixed by the cell index, so a band at
// an offset that is not a multiple of 4 groups its probabilities differently -- {0,0,p0,p1},{p2..p5}.. instead
// of {p0..p3},{p4..p7}.. -- and the fp16 rounding differs. Pascal's cuBLAS has no tensor cores, and a wide call
// (prefill) ignores where the band sits, but a narrow one (n <= 16 query columns: every decode step) does not:
// it reduces the cell axis in a grain of its own, so a P100 moved decode logits by up to 0.011 (test-kv-band-align
// -gpu), and it switches kernels at k = 512, i.e. at an n_kv the cells in front of a band decide.
// The softmax in front of it has the same property in fp32: its row sum is a tree over thread slots that are
// assigned by column, so a moved band is summed in a different order (absorbed by the fp16 conversion in the
// test, not in an fp32 chain).
//
// THE FIX. Anchor both reductions at the band instead of at cell 0.
//   * soft_max (softmax.cu): a row is walked from its first unmasked column, so column `first + i` lands in the
//     slot column `i` used to; identical to the old walk whenever that column is 0.
//   * P.V on a card with fp16 GEMMs (cc >= 600): the first column any row of the mask leaves open (the anchor, found
//     on the device) is moved to k = 0 -- P is converted to fp16 shifted left by it (the conversion the call
//     already made, now with an offset) and V is copied shifted left by it -- and the SAME cuBLAS call runs on
//     the copies. A band at cell a then meets cuBLAS exactly as the same band at cell 0 did. When the anchor
//     is 0 the GQA (pointer-array) route reads the cache view itself and the copy kernel exits at once, so the
//     common placement costs two tiny launches; the strided route (no GQA) always reads the copy.
//   * Pascal also runs the k <= 511 kernel class next to the full-width call when n <= 16 and n_kv >= 512 and keeps
//     the class the same band at cell 0 would have met (kv-band.cu, pxa_kv_band_pick).
//
// WHAT IT DOES NOT COVER. One anchor per call: a ubatch that mixes sequences anchors at the lowest open column,
// so a second sequence's band is anchored only modulo the first one's. Cells of one sequence interleaved with
// another's (several slots decoding in one unified ring) group differently from a contiguous band. Fused
// flash-attention kernels tile the cell axis from cell 0 and are not touched here.
//
// PXA_KV_BAND_EXACT: 1 (default from the DEFAULT config level up) | 0 (and PXA_REFERENCE=1): the previous
// arithmetic byte for byte.
#pragma once

#include "../common.cuh"
#include "ggml-pxa-kqmask.h" // PXA_KV_BAND_ANCHOR0 proof layout

bool ggml_cuda_pxa_kv_band_exact(void);

// the attention P.V product the band anchor applies to: src1 is a masked soft_max of f32 rows, src0 is f16,
// the card is NVIDIA cc >= 6.0
bool pxa_kv_band_pv_applies(const ggml_tensor * src0, const ggml_tensor * src1, int cc);

// anchor_d <- the first column (< ncols) that any of the first nrows rows of the softmax mask leaves open, or a
// value >= ncols when none does (read as 0 by the kernels below)
void pxa_kv_band_anchor(const ggml_tensor * mask, int64_t ncols, int64_t nrows, int * anchor_d, cudaStream_t stream);

// The anchor the kernels below use: anchor_h >= 0 (known on the host, pxa_kv_band_host_known) or else *anchor_d on the
// device (read as 0 unless 0 < a < ncols).

// out[r][c] = half(P[r][c + a]) for c + a < ncols, else 0; P is nrows contiguous f32 rows of ncols
void pxa_kv_band_shift_p(const float * P, half * out, int64_t ncols, int64_t nrows, const int * anchor_d, int anchor_h,
                         cudaStream_t stream);

// out (same strides as V) <- V shifted left along dim 0 by the anchor, zero tail. With ptr_route (the pointer-array
// call) only the matrices pxa_kv_band_batched_ptrs does not read in place are written (none at anchor 0); out then
// only has to hold the bytes from pxa_kv_band_copy_from(.., anchor_h) on (the caller offsets its base by that much)
void pxa_kv_band_shift_v(const char * V, char * out, const int64_t ne[4], const size_t nb[4], const int * anchor_d,
                         int anchor_h, bool ptr_route, int64_t limit_off, cudaStream_t stream);

// The pointer route's copy (review N3): the smallest byte offset (V's strides, from the view)
// of a matrix that is not read in place for anchor a (a < 0: unknown, the worst anchor ne0 - 1), or -1 when none is
// (a == 0, or every read stays inside limit_off). limit_off < 0 (copy route): 0 unless a == 0.
int64_t pxa_kv_band_copy_from(const int64_t ne[4], const size_t nb[4], int64_t limit_off, int a);

// The pointer route's limit_off for this V view (kv_band_v_inplace): the bytes from the view to the end of the V tensor,
// plus llama's zero pad behind it (PXA_KV_BAND_VPAD, ggml-pxa-kqmask.h); -1 (never in place, the copy) on a
// tensor-core card or with PXA_KV_BAND_INPLACE=0 (the escape to the copy route of FINAL-7
// before 0e585b8729, the band itself unchanged).
int64_t pxa_kv_band_limit(const ggml_tensor * src0, int cc);

// k_compute_batched_ptrs with the src0 base picked by the anchor: the V view at anchor 0; the view moved by the anchor
// (src0_view + a*nb00, no copy) while that matrix's last read stays inside the V tensor, limit_off bytes past the view
// (limit_off < 0: never; see kv-band.cu); else src0_copy (pxa_kv_band_shift_v's output)
void pxa_kv_band_batched_ptrs(const void * src0_view, const void * src0_copy, const int * anchor_d, int anchor_h, int64_t ncols,
                              int64_t ne01, size_t nb00, size_t nb01, int64_t limit_off,
                              const void * src1, void * dst, const void ** ptrs_src, void ** ptrs_dst,
                              int64_t ne12, int64_t ne13, int64_t ne23, size_t nb02, size_t nb03, size_t nb12, size_t nb13,
                              size_t nbd2, size_t nbd3, int64_t r2, int64_t r3, cudaStream_t stream);

// Pascal (cc < 7.0): cuBLAS answers the P.V call with one kernel for k < PXA_KV_BAND_PASCAL_K and another from it on,
// when n <= PXA_KV_BAND_PASCAL_NMAX. True when the call must run both classes (k >= the split) -- see kv-band.cu.
#define PXA_KV_BAND_PASCAL_K    512
#define PXA_KV_BAND_PASCAL_NMAX 16
bool pxa_kv_band_pascal_dual(int cc, int64_t k, int64_t n);

// PXA_KV_BAND_ANCHOR0 (default 1, 0 = the band's device work on every call as in
// FINAL-7): true when llama's proof on this mask (GGML_PXA_KQMASK_PROOF_*, ggml-pxa-kqmask.h; written from the host mask
// data of this very compute) shows the band would answer exactly as the plain path: each of the call's nrows rows
// opens at column 0 of its ncols (anchor 0, every row's walk from column 0) and, for the Pascal two-class call
// (dual), the pick keeps the full-width class (an open column >= PXA_KV_BAND_PASCAL_K - pad). Never while the stream
// is being captured (a replayed graph keeps the kernels it was captured with).
bool pxa_kv_band_anchor0_proven(const ggml_tensor * mask, int64_t ncols, int64_t nrows, bool dual, cudaStream_t stream);

// PXA_KV_BAND_HOSTPROOF (default 1, 0 = the device anchor work on every banded call as in
// FINAL-8): true when llama's proof on this mask (written from the host mask data of this very compute) carries the
// first and the last open column of exactly the rows and columns the band kernels read (decode-sized ubatches, column 0
// open or not), outside graph capture. *anchor = what pxa_kv_band_anchor's buffer would be read as, *last = what
// pxa_kv_band_last would write: the band then needs neither kernel, and the Pascal class pick is decided on the host.
bool pxa_kv_band_host_known(const ggml_tensor * mask, int64_t ncols, int64_t nrows, cudaStream_t stream, int * anchor, int * last);

// pxa_kv_band_pick's decision on the host: true when the band [anchor, last] placed at cell 0 would have had
// n_kv < PXA_KV_BAND_PASCAL_K, i.e. the pick keeps the k <= 511 class (then that call alone writes dst)
bool pxa_kv_band_pick_small(int anchor, int last, const ggml_tensor * mask);

// last_d <- the last column (< ncols) any of the first nrows rows of the mask leaves open, or -1
void pxa_kv_band_last(const ggml_tensor * mask, int64_t ncols, int64_t nrows, int * last_d, cudaStream_t stream);

// dst[0..n) <- small[0..n) when the band [anchor, last] placed at cell 0 would have had n_kv < PXA_KV_BAND_PASCAL_K
// (padded like the KV cache: 256 cells for an f16 mask, else 32); otherwise dst keeps the full-width answer
void pxa_kv_band_pick(const half * small, half * dst, int64_t n, const int * anchor_d, const int * last_d, int64_t ncols,
                      const ggml_tensor * mask, cudaStream_t stream);
