// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// test-topw.cu -- PXA_SAMPLED_TOPW's two CUDA passes (ggml-cuda/argmax.cu, included verbatim) against the CPU
// sampler's raw top-k window rule (common/sampling.cpp pxa_topk_window: values descending, an equal value keeps the lower id
// first, an element equal to the minimum of a full window does not enter), BY BITS of the W values and W ids: random rows,
// rows of a few distinct values (many exact ties), +-0.0, -inf, narrow rows (< W), 1..4 rows, widths 128256 / 248320 / 4096*k.
// Then the timing of the pair at 128256 x 1 (W 64).
// Build (V100): nvcc -O3 -std=c++17 -arch=sm_70 -I ggml/src -I ggml/include -o test-topw tests/test-topw.cu
#include "../ggml/src/ggml-cuda/argmax.cu"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) { fprintf(stderr, "abort %s:%d %s\n", file, line, fmt); abort(); }
std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device) { (void) device; abort(); }
const ggml_cuda_device_info & ggml_cuda_info() { abort(); }
void ggml_cuda_set_device(int device) { (void) device; abort(); }
[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "cuda error %s %s %s:%d %s\n", stmt, func, file, line, msg); abort(); }
extern "C" bool ggml_is_contiguous(const struct ggml_tensor * t) { (void) t; abort(); }
extern "C" int64_t ggml_nrows(const struct ggml_tensor * t) { (void) t; abort(); }

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
static uint64_t g_rs = 0x9E3779B97F4A7C15ull;
static inline uint64_t rnd() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }
static inline float frnd() { return (float) ((rnd() >> 11) * (1.0/9007199254740992.0)); }

static void ref_window(const float * x, int64_t n, int W, float * d) {   // the sampler's pxa_topk_window, verbatim rule
    std::vector<float> wl(W); std::vector<int> wid(W); int cnt = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float v = x[i];
        if (cnt == W && !(v > wl[W-1])) continue;
        int j = cnt < W ? cnt++ : W - 1;
        while (j > 0 && wl[j-1] < v) { wl[j] = wl[j-1]; wid[j] = wid[j-1]; --j; }
        wl[j] = v; wid[j] = (int) i;
    }
    for (int k = 0; k < W; ++k) { d[k] = k < cnt ? wl[k] : -INFINITY; d[W + k] = k < cnt ? (float) wid[k] : -1.0f; }
}

static int run_case(int64_t ncols, int rows, int W, int kind, bool timing = false) {
    std::vector<float> x((size_t) ncols * rows);
    for (auto & v : x) {
        switch (kind) {
            case 0: v = 8.0f * frnd() - 4.0f; break;                                  // continuous
            case 1: v = (float) (int) (rnd() % 7) - 3.0f; break;                       // 7 values: ties everywhere
            case 2: { const int r = (int) (rnd() % 10); v = r == 0 ? -INFINITY : r == 1 ? -0.0f : r == 2 ? 0.0f : (float) (int) (rnd() % 3); } break;
            default: v = (float) (rnd() % 5 == 0 ? 1.0f : -1.0f); break;
        }
    }
    float * dx; unsigned long long * dp, * dq; float * dd;
    const size_t nk = pxa_topw_scratch_keys(ncols, rows, W);
    CK(cudaMalloc(&dx, x.size() * sizeof(float)));
    CK(cudaMalloc(&dp, nk * sizeof(unsigned long long))); CK(cudaMalloc(&dq, nk * sizeof(unsigned long long)));
    CK(cudaMalloc(&dd, (size_t) rows * 2 * W * sizeof(float)));
    CK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    pxa_topw_launch(dx, dd, ncols, rows, ncols, W, dp, dq, 0);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> got((size_t) rows * 2 * W), ref((size_t) rows * 2 * W);
    CK(cudaMemcpy(got.data(), dd, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int r = 0; r < rows; ++r) ref_window(x.data() + (size_t) r * ncols, ncols, W, ref.data() + (size_t) r * 2 * W);
    int bad = 0;
    for (size_t i = 0; i < got.size(); ++i) if (memcmp(&got[i], &ref[i], 4) != 0) { if (bad < 4) printf("  mismatch %zu got %g ref %g\n", i, got[i], ref[i]); ++bad; }
    if (timing) {
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int i = 0; i < 20; ++i) pxa_topw_launch(dx, dd, ncols, rows, ncols, W, dp, dq, 0);
        CK(cudaEventRecord(e0));
        for (int i = 0; i < 200; ++i) pxa_topw_launch(dx, dd, ncols, rows, ncols, W, dp, dq, 0);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("TIMING ncols %lld rows %d W %d: %.2f us per window (both passes, b2b)\n", (long long) ncols, rows, W, 1000.0f * ms / 200);
    }
    cudaFree(dx); cudaFree(dp); cudaFree(dq); cudaFree(dd);
    return bad;
}

int main() {
    int fails = 0, cases = 0;
    const int64_t widths[] = {1, 7, 40, 64, 100, 1023, 1024, 1025, 4097, 50000, 128256, 131072, 248320, 1100000};
    const int Ws[] = {1, 2, 32, 64, 128};
    for (int64_t n : widths) for (int W : Ws) for (int kind = 0; kind < 4; ++kind) for (int rows : {1, 3}) {
        const int b = run_case(n, rows, W, kind);
        ++cases; if (b) { ++fails; printf("FAIL ncols %lld W %d kind %d rows %d: %d mismatches\n", (long long) n, W, kind, rows, b); }
    }
    run_case(128256, 1, 64, 0, true);
    run_case(128256, 1, 128, 1, true);
    run_case(248320, 1, 64, 0, true);
    printf("%s: %d/%d cases bitwise equal to the CPU window\n", fails ? "FAIL" : "PASS", cases - fails, cases);
    return fails ? 1 : 0;
}
