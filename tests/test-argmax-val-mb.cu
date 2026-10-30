// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// test-argmax-val-mb.cu -- PXA_ARGMAX_VAL_MB's three passes against the production one-block
// argmax_val_f32 (ggml-cuda/argmax.cu, included verbatim), BY BITS of the three output floats (index, max, sumexp), over
// random rows with exact ties, -inf, NaN, all-equal and all-(-inf) rows, 1..8 rows, strided rows; then the timing of both on
// the 27B head width (248320) at 1 and 4 rows.
// Build (V100): nvcc -O3 -std=c++17 -use_fast_math -arch=sm_70 -I ggml/src -I ggml/include ... tests/test-argmax-val-mb.cu
#include "../ggml/src/ggml-cuda/argmax.cu"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// link stubs for the host glue argmax.cu carries (never called here)
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

int main(int argc, char ** argv) {
    int reps = 200;
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], "-reps") && i + 1 < argc) reps = atoi(argv[++i]);
    cudaStream_t st; CK(cudaStreamCreate(&st));
    const int64_t maxc = 262144, maxr = 8, maxs = maxc + 64;
    float * dx; float * d0; float * d1; float2 * part; float * ssum;
    CK(cudaMalloc(&dx, maxs*maxr*4)); CK(cudaMalloc(&d0, 3*maxr*4)); CK(cudaMalloc(&d1, 3*maxr*4));
    CK(cudaMalloc(&part, maxr*64*sizeof(float2))); CK(cudaMalloc(&ssum, maxr*64*4));
    float * d2; float * d3; CK(cudaMalloc(&d2, 4*maxr*4)); CK(cudaMalloc(&d3, 4*maxr*4));
    std::vector<float> c4(4*maxr), e4(4*maxr);
    int fails4 = 0;
    std::vector<float> x(maxs*maxr), a(3*maxr), b(3*maxr);
    int cases = 0, fails = 0;
    const int64_t widths[] = {32768, 32769, 65536, 151936, 152064, 248320, 262144};
    for (int64_t nc : widths) for (int rows : {1, 2, 4, 5, 8}) for (int kind = 0; kind < 7; ++kind) {
        const int64_t rs = (kind == 6) ? nc + 64 : nc;                 // kind 6: strided rows
        for (int64_t r = 0; r < rows; ++r) for (int64_t c = 0; c < rs; ++c) {
            float v = (frnd() - 0.5f) * 30.0f;
            if (kind == 1) v = (float) ((int) (frnd() * 8.0f));          // many exact ties
            if (kind == 2 && (rnd() % 7) == 0) v = -INFINITY;
            if (kind == 3 && (rnd() % 1000) == 0) v = NAN;
            if (kind == 4) v = 1.5f;                                     // all equal -> lowest index
            if (kind == 5) v = -INFINITY;                                 // all -inf
            x[r*rs + c] = v;
        }
        if (kind == 0) x[(rows - 1)*rs + nc - 1] = 1000.0f;             // max in the last column of the last row
        CK(cudaMemcpy(dx, x.data(), (size_t) rows*rs*4, cudaMemcpyHostToDevice));
        CK(cudaMemset(d0, 0x7b, 3*maxr*4)); CK(cudaMemset(d1, 0x5c, 3*maxr*4));
        argmax_val_f32<3><<<dim3(rows, 1, 1), dim3(1024, 1, 1), 0, st>>>(dx, d0, nc, rs);
        pxa_argmax_val_mb_launch<3>(dx, d1, nc, rows, rs, part, ssum, st);
        CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
        CK(cudaMemcpy(a.data(), d0, 3*rows*4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), d1, 3*rows*4, cudaMemcpyDeviceToHost));
        // OW = 4 (PXA_DRAFT_ARGMAX): the first three floats bit for bit the OW = 3 ones, the fourth = max over the other columns
        argmax_val_f32<4><<<dim3(rows, 1, 1), dim3(1024, 1, 1), 0, st>>>(dx, d2, nc, rs);
        pxa_argmax_val_mb_launch<4>(dx, d3, nc, rows, rs, part, ssum, st);
        CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
        CK(cudaMemcpy(c4.data(), d2, 4*rows*4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(e4.data(), d3, 4*rows*4, cudaMemcpyDeviceToHost));
        for (int r = 0; r < rows; ++r) {
            const int64_t im = (int64_t) a[3*r];
            float m2 = -INFINITY;
            for (int64_t c = 0; c < nc; ++c) if (c != im) m2 = fmaxf(m2, x[r*rs + c]);
            const bool ok = memcmp(&c4[4*r], &a[3*r], 12) == 0 && memcmp(&e4[4*r], &a[3*r], 12) == 0 &&
                            memcmp(&c4[4*r+3], &m2, 4) == 0 && memcmp(&e4[4*r+3], &m2, 4) == 0;
            if (!ok) { ++fails4; if (fails4 <= 8) printf("  FAIL4 nc=%lld rows=%d kind=%d r%d: m2 host %.9g ref %.9g mb %.9g\n",
                                                      (long long) nc, rows, kind, r, m2, c4[4*r+3], e4[4*r+3]); }
        }
        ++cases;
        if (memcmp(a.data(), b.data(), 3*rows*4) != 0) {
            ++fails;
            printf("  FAIL nc=%lld rows=%d kind=%d\n", (long long) nc, rows, kind);
            for (int r = 0; r < rows; ++r) printf("    r%d ref (%.0f %.9g %.9g) mb (%.0f %.9g %.9g)\n", r, a[3*r], a[3*r+1], a[3*r+2], b[3*r], b[3*r+1], b[3*r+2]);
        }
    }
    printf("BITWISE argmax_val MB vs one-block: %d cases, %d failed\n", cases, fails);
    printf("OW=4 (max2) one-block + MB vs OW=3 + host max2: %d cases, %d rows failed\n", cases, fails4);
    fails += fails4;
    // timing at the 27B head width
    for (int rows : {1, 4}) {
        const int64_t nc = 248320;
        for (int64_t i = 0; i < rows*nc; ++i) x[i] = (frnd() - 0.5f) * 30.0f;
        CK(cudaMemcpy(dx, x.data(), (size_t) rows*nc*4, cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        float t_ref = 0, t_mb = 0;
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<float> ts(reps);
            for (int k = 0; k < reps; ++k) {
                CK(cudaEventRecord(e0, st));
                if (pass == 0) argmax_val_f32<3><<<dim3(rows, 1, 1), dim3(1024, 1, 1), 0, st>>>(dx, d0, nc, nc);
                else           pxa_argmax_val_mb_launch<3>(dx, d1, nc, rows, nc, part, ssum, st);
                CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ts[k], e0, e1));
            }
            std::sort(ts.begin(), ts.end());
            (pass == 0 ? t_ref : t_mb) = ts[reps/2];
        }
        printf("TIME 248320 x %d rows: one-block %.1f us | MB (3 launches) %.1f us\n", rows, 1000*t_ref, 1000*t_mb);
    }
    return fails ? 1 : 0;
}
