// test-pxq-ws-gen.cu -- bug #230: growing the persistent PXQ KSPLIT workspace must bump the
// allocator generation.
//
// pxq6_ksplit_workspace() hands out a per-device buffer that grows on demand with cudaFree +
// cudaMalloc. A CUDA graph captured earlier holds the old pointer as a kernel argument, and the
// graph path's per-node property compare cannot see that, so a replay would read and write freed
// memory. The graph path re-captures when ggml_cuda_alloc_generation_bump() has been called since
// the capture; the fix calls it before every free on growth.
//
// This test defines ggml_cuda_alloc_generation_bump() itself (the executable's own definition wins
// over libggml's for the calls compiled into this file, which is where the header's copy of the
// workspace lives) and counts the calls:
//   first allocation           -> no bump (nothing was captured against it)
//   smaller or equal request   -> same pointer, no bump
//   larger request mid-capture -> declined (nullptr), no bump, old buffer kept
//   larger request             -> one bump, new buffer
// On the unfixed tree the growth makes no call and the last check fails. Needs a CUDA device.
#include "ggml-cuda/pxa/pxq6.cuh"

#include <cstdio>
#include <cstdlib>

static int g_bumps = 0;
extern "C" void ggml_cuda_alloc_generation_bump(void) { ++g_bumps; }

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL " __VA_ARGS__); printf("\n"); ++g_fail; } \
                              else { printf("  ok   " __VA_ARGS__); printf("\n"); } } while (0)

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { printf("SKIP: no CUDA device\n"); return 0; }
    cudaSetDevice(0);
    cudaStream_t st;
    if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess) { printf("SKIP: no stream\n"); return 0; }

    float * a = pxq6_ksplit_workspace(0, st, 1024);
    CHECK(a != nullptr && g_bumps == 0, "first allocation: ptr %p, bumps %d (want non-null, 0)", (void *) a, g_bumps);

    float * b = pxq6_ksplit_workspace(0, st, 512);
    CHECK(b == a && g_bumps == 0, "smaller request reuses the buffer: same %d, bumps %d (want 1, 0)", b == a, g_bumps);

    // mid-capture: growing is not allowed (cudaFree/cudaMalloc cannot be captured) -> decline
    float * c = (float *) 0x1;
    if (cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed) == cudaSuccess) {
        c = pxq6_ksplit_workspace(0, st, 1 << 20);
        cudaGraph_t g = nullptr;
        cudaStreamEndCapture(st, &g);
        if (g) cudaGraphDestroy(g);
        (void) cudaGetLastError();
        CHECK(c == nullptr && g_bumps == 0, "growth mid-capture declines: ptr %p, bumps %d (want null, 0)", (void *) c, g_bumps);
        float * a2 = pxq6_ksplit_workspace(0, st, 1024);
        CHECK(a2 == a && g_bumps == 0, "old buffer kept after the declined growth: same %d, bumps %d (want 1, 0)", a2 == a, g_bumps);
    } else {
        (void) cudaGetLastError();
        printf("  skip capture case (cudaStreamBeginCapture failed)\n");
    }

    float * d = pxq6_ksplit_workspace(0, st, 1 << 20);
    CHECK(d != nullptr && g_bumps == 1,
          "growth bumps the allocator generation before freeing: ptr %p, bumps %d (want non-null, 1)", (void *) d, g_bumps);

    float * e = pxq6_ksplit_workspace(0, st, 1 << 20);
    CHECK(e == d && g_bumps == 1, "same size after growth: same %d, bumps %d (want 1, 1)", e == d, g_bumps);

    cudaStreamDestroy(st);
    printf("%s\n", g_fail ? "FAILED" : "PASSED");
    return g_fail ? 1 : 0;
}
