// pxq-moemap.cuh — device-side construction of the MoE expert row mapping and tile list.
//
// prepare_row_mappigs() builds the same two buffers on the host: it copies `ids` D2H, does a
// cudaStreamSynchronize inside graph compute, histograms the expert ids, exclusive-scans the
// counts, fills the row mapping and the (expert, row0, nrows) tile list, then uploads them.
// The synchronize is per batched-MoE node, i.e. once per layer per ubatch, and it stops the
// host from enqueuing ubatch k+1 while the devices are still running ubatch k — which is
// exactly the overlap pipeline parallelism exists to create.
//
// The kernels here produce byte-identical buffers with no readback and no synchronize:
//
//   k_pxq_moemap_hist    per-256-row block histogram of the expert ids      -> bcounts[b][e]
//   k_pxq_moemap_scan_b  exclusive scan of bcounts over b, per expert       -> bcounts[b][e], etot[e]
//   k_pxq_moemap_scan_e  exclusive scan of etot over e (rows and tiles)     -> ebase[], tbase[]
//   k_pxq_moemap_fill    row -> mapping slot, stable in (i2,i1) order       -> rmap[]
//   k_pxq_moemap_tiles   per-expert 64-row tile descriptors                 -> tiles[]
//
// The mapping order is the host order exactly: rows are visited flat as r = i2*n_ids + i1 and
// a row's slot inside its expert segment is (block prefix) + (its rank among same-expert rows
// earlier in its own 256-row block). No atomics decide placement, so the buffers do not depend
// on scheduling and the GEMM tiles see the same rows in the same order.
//
// What the host loses is the two scalars it used for launch geometry: the number of routed
// rows and the number of tiles. Both are replaced by their worst case -
//   total  <= n_rows  = ids->ne[1]*n_ids            (equality unless an id is out of range)
//   ntiles <= n_rows/PXQ4_BN + n_as                 (sum of ceils <= ceil of sum + n_as)
// - so the grids are launched for the worst case and the unused blocks exit on a zeroed tile
// (nrows == 0) or a sentinel mapping entry (i1 < 0). The tile and mapping buffers are cleared
// before the fill for exactly that reason.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <vector>
#include <algorithm>

#include "pxq4.cuh"

#define PXQ_MOEMAP_BLK   256    // rows per histogram/fill block
#define PXQ_MOEMAP_MAXAS 1024   // experts the single-block expert scan can cover

// The PXA_MOE_DEVICE_MAP accessors live in ggml-cuda.cu, above prepare_row_mappigs, because
// that function (defined before this header is included) reports the call sites the device map
// has not taken over. 0 (default) = host prepare_row_mappigs; 1 = these kernels; 2 = these
// kernels plus a host cross-check of the first PXQ_MOEMAP_VERIFY_N maps (debug only - the check
// reads the buffers back, so it reinstates the very synchronize this removes).

#define PXQ_MOEMAP_VERIFY_N 64

static __device__ __forceinline__ int pxq_moemap_id(const char * __restrict__ ids,
        const size_t nb0, const size_t nb1, const int n_ids, const int r) {
    const int i2 = r / n_ids;
    const int i1 = r - i2*n_ids;
    return *(const int32_t *)(ids + (size_t)i2*nb1 + (size_t)i1*nb0);
}

// bcounts is [n_blocks][n_as]; every block writes its whole row, so no pre-zeroing.
static __global__ void __launch_bounds__(PXQ_MOEMAP_BLK)
k_pxq_moemap_hist(const char * __restrict__ ids, const size_t nb0, const size_t nb1,
                  const int n_ids, const int n_rows, const int n_as, int * __restrict__ bcounts) {
    extern __shared__ int sh_hist[];
    for (int e = threadIdx.x; e < n_as; e += PXQ_MOEMAP_BLK) sh_hist[e] = 0;
    __syncthreads();

    const int r = blockIdx.x*PXQ_MOEMAP_BLK + threadIdx.x;
    if (r < n_rows) {
        const int e = pxq_moemap_id(ids, nb0, nb1, n_ids, r);
        if (e >= 0 && e < n_as) atomicAdd(&sh_hist[e], 1);
    }
    __syncthreads();

    int * out = bcounts + (size_t)blockIdx.x*n_as;
    for (int e = threadIdx.x; e < n_as; e += PXQ_MOEMAP_BLK) out[e] = sh_hist[e];
}

// per expert: exclusive scan of the per-block counts over blocks, in place; total -> etot[e].
// One thread per expert, consecutive threads on consecutive experts, so each step of the
// block loop is a coalesced read/write of a contiguous n_as row.
static __global__ void k_pxq_moemap_scan_b(int * __restrict__ bcounts, const int n_blocks,
                                           const int n_as, int * __restrict__ etot) {
    const int e = blockIdx.x*blockDim.x + threadIdx.x;
    if (e >= n_as) return;
    int acc = 0;
    for (int b = 0; b < n_blocks; ++b) {
        const size_t idx = (size_t)b*n_as + e;
        const int c = bcounts[idx];
        bcounts[idx] = acc;
        acc += c;
    }
    etot[e] = acc;
}

// single block: exclusive scan of etot over experts, both in rows (ebase) and in 64-row tiles
// (tbase). ebase[n_as] / tbase[n_as] hold the totals. Requires n_as <= PXQ_MOEMAP_MAXAS.
static __global__ void __launch_bounds__(PXQ_MOEMAP_MAXAS)
k_pxq_moemap_scan_e(const int * __restrict__ etot, const int n_as, const int bn,
                    int * __restrict__ ebase, int * __restrict__ tbase) {
    __shared__ int sr[PXQ_MOEMAP_MAXAS];
    __shared__ int st[PXQ_MOEMAP_MAXAS];
    const int t = threadIdx.x;
    const int c  = t < n_as ? etot[t] : 0;
    const int ct = t < n_as ? (c + bn - 1)/bn : 0;
    sr[t] = c;
    st[t] = ct;
    __syncthreads();
    for (int off = 1; off < PXQ_MOEMAP_MAXAS; off <<= 1) {
        const int ar = t >= off ? sr[t-off] : 0;
        const int at = t >= off ? st[t-off] : 0;
        __syncthreads();
        sr[t] += ar;
        st[t] += at;
        __syncthreads();
    }
    if (t < n_as) {
        ebase[t] = sr[t] - c;
        tbase[t] = st[t] - ct;
    }
    if (t == 0) {
        ebase[n_as] = sr[n_as-1];
        tbase[n_as] = st[n_as-1];
    }
}

// rmap must be pre-set to the -1 sentinel: rows past the routed total keep it and are skipped
// by the gather and the scatter.
static __global__ void __launch_bounds__(PXQ_MOEMAP_BLK)
k_pxq_moemap_fill(const char * __restrict__ ids, const size_t nb0, const size_t nb1,
                  const int n_ids, const int n_rows, const int n_as,
                  const int * __restrict__ bcounts, const int * __restrict__ ebase,
                  pxq4_rowmap * __restrict__ rmap) {
    __shared__ int sh_e[PXQ_MOEMAP_BLK];
    const int t = threadIdx.x;
    const int r = blockIdx.x*PXQ_MOEMAP_BLK + t;

    int e = -1, i1 = 0, i2 = 0;
    if (r < n_rows) {
        i2 = r / n_ids;
        i1 = r - i2*n_ids;
        e  = *(const int32_t *)(ids + (size_t)i2*nb1 + (size_t)i1*nb0);
        if (e < 0 || e >= n_as) e = -1;
    }
    sh_e[t] = e;
    __syncthreads();

    if (e >= 0) {
        int rank = 0;                                  // stable: rank among earlier rows of this block
        for (int k = 0; k < t; ++k) rank += (sh_e[k] == e);
        const int slot = ebase[e] + bcounts[(size_t)blockIdx.x*n_as + e] + rank;
        rmap[slot].i1 = i1;
        rmap[slot].i2 = i2;
    }
}

// tiles must be pre-zeroed: entries past the real tile count keep nrows == 0 and their GEMM
// blocks exit immediately.
static __global__ void k_pxq_moemap_tiles(const int * __restrict__ etot, const int * __restrict__ ebase,
                                          const int * __restrict__ tbase, const int n_as, const int bn,
                                          pxq4_tile_info * __restrict__ tiles, const int n_tiles_max) {
    const int e = blockIdx.x;
    if (e >= n_as) return;
    const int c  = etot[e];
    const int nt = (c + bn - 1)/bn;
    for (int t = threadIdx.x; t < nt; t += blockDim.x) {
        const int idx = tbase[e] + t;
        if (idx >= n_tiles_max) return;
        const int rem = c - t*bn;
        pxq4_tile_info ti;
        ti.e     = e;
        ti.row0  = ebase[e] + t*bn;
        ti.nrows = rem < bn ? rem : bn;
        ti._pad  = 0;
        tiles[idx] = ti;
    }
}

// SER equivalent: prepare_row_mappigs reports "some id was out of range" and the caller memsets
// the whole MoE output so unrouted rows read as zero. Without the readback the host cannot know,
// so zero exactly the rows whose id is out of range - the routed rows are all written by the
// scatter, so this matches the host behaviour in both cases and costs one predicated block per
// row when there is nothing to zero.
static __global__ void __launch_bounds__(PXQ_MOEMAP_BLK)
k_pxq_moemap_zero_unrouted(char * __restrict__ dst, const size_t nb1, const size_t nb2,
                           const int ne0, const char * __restrict__ ids,
                           const size_t nb0i, const size_t nb1i,
                           const int n_ids, const int n_rows, const int n_as) {
    const int r = blockIdx.x*PXQ_MOEMAP_BLK + threadIdx.x;   // one thread per row: the common
    if (r >= n_rows) return;                                 // case is "every id is routed",
    const int i2 = r / n_ids;                                // in which case no thread stores
    const int i1 = r - i2*n_ids;
    const int e  = *(const int32_t *)(ids + (size_t)i2*nb1i + (size_t)i1*nb0i);
    if (e >= 0 && e < n_as) return;
    float * row = (float *)(dst + (size_t)i1*nb1 + (size_t)i2*nb2);
    for (int j = 0; j < ne0; ++j) row[j] = 0.0f;
}

// Worst-case tile count for n_rows routed rows over n_as experts:
//   sum_e ceil(c_e/bn) <= sum_e (floor(c_e/bn) + 1) <= floor(n_rows/bn) + n_as
static inline int pxq_moemap_tiles_max(int n_rows, int n_as) {
    return n_rows/PXQ4_BN + n_as;
}

// Builds rmap + tiles on the device. Returns false when the shape is outside what these kernels
// cover, in which case the caller must use the host path.
struct pxq_moemap_bufs {
    ggml_cuda_pool_alloc<int> bcounts;
    ggml_cuda_pool_alloc<int> etot;
    ggml_cuda_pool_alloc<int> ebase;
    ggml_cuda_pool_alloc<int> tbase;
    explicit pxq_moemap_bufs(ggml_cuda_pool & p)
        : bcounts(p), etot(p), ebase(p), tbase(p) {}
};

static bool pxq_moemap_build(ggml_backend_cuda_context & ctx, const ggml_tensor * ids,
                             int64_t n_as, int64_t n_ids, int n_rows, int n_tiles_max,
                             pxq_moemap_bufs & bufs,
                             pxq4_rowmap * rmap, pxq4_tile_info * tiles) {
    if (n_as <= 0 || n_as > PXQ_MOEMAP_MAXAS || n_rows <= 0) return false;

    cudaStream_t stream = ctx.stream();
    const int n_blocks = (n_rows + PXQ_MOEMAP_BLK - 1)/PXQ_MOEMAP_BLK;

    {   // one line per device the first time the device map runs there
        static bool announced[GGML_CUDA_MAX_DEVICES] = {};
        if (ctx.device >= 0 && ctx.device < GGML_CUDA_MAX_DEVICES && !announced[ctx.device]) {
            announced[ctx.device] = true;
            fprintf(stderr, "PXA_MOE_DEVICE_MAP dev%d: FIRING (n_as=%d n_ids=%d rows=%d blocks=%d tiles_max=%d)\n",
                    ctx.device, (int)n_as, (int)n_ids, n_rows, n_blocks, n_tiles_max);
        }
    }

    bufs.bcounts.alloc((size_t)n_blocks*n_as);
    bufs.etot.alloc(n_as);
    bufs.ebase.alloc(n_as + 1);
    bufs.tbase.alloc(n_as + 1);

    CUDA_CHECK(cudaMemsetAsync(rmap,  0xff, (size_t)n_rows*sizeof(pxq4_rowmap), stream));
    CUDA_CHECK(cudaMemsetAsync(tiles, 0,    (size_t)n_tiles_max*sizeof(pxq4_tile_info), stream));

    const char * ids_d = (const char *) ids->data;
    const size_t nb0 = ids->nb[0], nb1 = ids->nb[1];

    k_pxq_moemap_hist<<<n_blocks, PXQ_MOEMAP_BLK, n_as*sizeof(int), stream>>>(
            ids_d, nb0, nb1, (int)n_ids, n_rows, (int)n_as, bufs.bcounts.get());
    CUDA_CHECK(cudaGetLastError());

    k_pxq_moemap_scan_b<<<((int)n_as + 63)/64, 64, 0, stream>>>(
            bufs.bcounts.get(), n_blocks, (int)n_as, bufs.etot.get());
    CUDA_CHECK(cudaGetLastError());

    k_pxq_moemap_scan_e<<<1, PXQ_MOEMAP_MAXAS, 0, stream>>>(
            bufs.etot.get(), (int)n_as, PXQ4_BN, bufs.ebase.get(), bufs.tbase.get());
    CUDA_CHECK(cudaGetLastError());

    k_pxq_moemap_fill<<<n_blocks, PXQ_MOEMAP_BLK, 0, stream>>>(
            ids_d, nb0, nb1, (int)n_ids, n_rows, (int)n_as,
            bufs.bcounts.get(), bufs.ebase.get(), rmap);
    CUDA_CHECK(cudaGetLastError());

    k_pxq_moemap_tiles<<<(unsigned)n_as, 32, 0, stream>>>(
            bufs.etot.get(), bufs.ebase.get(), bufs.tbase.get(), (int)n_as, PXQ4_BN,
            tiles, n_tiles_max);
    CUDA_CHECK(cudaGetLastError());

    return true;
}

static void pxq_moemap_zero_unrouted(ggml_backend_cuda_context & ctx, const ggml_tensor * ids,
                                     ggml_tensor * t, int64_t n_as, int64_t n_ids, int n_rows) {
    const unsigned nb = (unsigned)((n_rows + PXQ_MOEMAP_BLK - 1)/PXQ_MOEMAP_BLK);
    k_pxq_moemap_zero_unrouted<<<nb, PXQ_MOEMAP_BLK, 0, ctx.stream()>>>(
            (char *)t->data, t->nb[1], t->nb[2], (int)t->ne[0],
            (const char *)ids->data, ids->nb[0], ids->nb[1],
            (int)n_ids, n_rows, (int)n_as);
    CUDA_CHECK(cudaGetLastError());
}

// PXA_MOE_DEVICE_MAP=2: rebuild the map on the host exactly as prepare_row_mappigs does and
// compare it with what the kernels produced. Debug only; it synchronizes.
static void pxq_moemap_verify(ggml_backend_cuda_context & ctx, const ggml_tensor * ids,
                              int64_t n_as, int64_t n_ids, int n_rows, int n_tiles_max,
                              const pxq4_rowmap * rmap_dev, const pxq4_tile_info * tiles_dev) {
    static std::atomic<int> budget{PXQ_MOEMAP_VERIFY_N};
    if (budget.fetch_sub(1) <= 0) return;

    cudaStream_t stream = ctx.stream();

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));

    std::vector<pxq4_rowmap>    rmap_got((size_t)n_rows);
    std::vector<pxq4_tile_info> tiles_got((size_t)n_tiles_max);
    CUDA_CHECK(cudaMemcpyAsync(rmap_got.data(),  rmap_dev,  rmap_got.size()*sizeof(pxq4_rowmap),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(tiles_got.data(), tiles_dev, tiles_got.size()*sizeof(pxq4_tile_info),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // the host construction, verbatim from prepare_row_mappigs + the caller's tile loop
    std::vector<int> counts((size_t)n_as, 0), cum((size_t)n_as + 1);
    for (int64_t i2 = 0; i2 < ids->ne[1]; i2++) {
        for (int64_t i1 = 0; i1 < n_ids; i1++) {
            const int32_t e = *(const int32_t *)(ids_host.data() + i2*ids->nb[1] + i1*ids->nb[0]);
            if (e >= 0 && e < n_as) ++counts[e];
        }
    }
    cum[0] = 0;
    for (int e = 0; e < (int)n_as; ++e) cum[e+1] = cum[e] + counts[e];
    std::vector<pxq4_rowmap> rmap_exp((size_t)cum[n_as]);
    std::vector<int> fill(cum.begin(), cum.end());
    for (int64_t i2 = 0; i2 < ids->ne[1]; i2++) {
        for (int64_t i1 = 0; i1 < n_ids; i1++) {
            const int32_t e = *(const int32_t *)(ids_host.data() + i2*ids->nb[1] + i1*ids->nb[0]);
            if (e >= 0 && e < n_as) rmap_exp[fill[e]++] = { (int32_t)i1, (int32_t)i2 };
        }
    }
    std::vector<pxq4_tile_info> tiles_exp;
    for (int e = 0; e < (int)n_as; ++e) {
        for (int t0 = 0; t0 < counts[e]; t0 += PXQ4_BN) {
            tiles_exp.push_back({ e, cum[e] + t0, std::min((int)PXQ4_BN, counts[e] - t0), 0 });
        }
    }

    int bad = 0;
    for (size_t i = 0; i < rmap_exp.size(); ++i) {
        if (rmap_got[i].i1 != rmap_exp[i].i1 || rmap_got[i].i2 != rmap_exp[i].i2) {
            if (bad < 8) fprintf(stderr, "PXA_MOE_DEVICE_MAP VERIFY dev%d: rmap[%zu] got (%d,%d) want (%d,%d)\n",
                    ctx.device, i, rmap_got[i].i1, rmap_got[i].i2, rmap_exp[i].i1, rmap_exp[i].i2);
            ++bad;
        }
    }
    for (size_t i = rmap_exp.size(); i < rmap_got.size(); ++i) {
        if (rmap_got[i].i1 >= 0) { if (bad < 8) fprintf(stderr,
                "PXA_MOE_DEVICE_MAP VERIFY dev%d: rmap[%zu] past total is not the sentinel (%d)\n",
                ctx.device, i, rmap_got[i].i1); ++bad; }
    }
    for (size_t i = 0; i < tiles_got.size(); ++i) {
        const pxq4_tile_info want = i < tiles_exp.size() ? tiles_exp[i] : pxq4_tile_info{0,0,0,0};
        if (tiles_got[i].e != want.e || tiles_got[i].row0 != want.row0 || tiles_got[i].nrows != want.nrows) {
            if (bad < 8) fprintf(stderr, "PXA_MOE_DEVICE_MAP VERIFY dev%d: tile[%zu] got (e=%d,row0=%d,n=%d) want (e=%d,row0=%d,n=%d)\n",
                    ctx.device, i, tiles_got[i].e, tiles_got[i].row0, tiles_got[i].nrows,
                    want.e, want.row0, want.nrows);
            ++bad;
        }
    }
    fprintf(stderr, "PXA_MOE_DEVICE_MAP VERIFY dev%d: rows=%d routed=%d tiles=%zu/%d -> %s (%d mismatches)\n",
            ctx.device, n_rows, cum[n_as], tiles_exp.size(), n_tiles_max, bad ? "MISMATCH" : "identical", bad);
}
