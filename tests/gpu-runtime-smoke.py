#!/usr/bin/env python3
"""Does the GPU runtime work on a real card? Opens the Pro encoder library through the wrapper's own search (pxqe_runtime.py, the code that ships in the
package), then uses the CUDA libraries it found exactly the way libpxqe does: cuBLAS SGEMM and cuSOLVER Cholesky (potrf + potrs, double). Only the NVIDIA
driver is needed on the machine. Prints one line per check and exits 0 only when every number is right.

    gpu-runtime-smoke.py PACKAGE_DIR            (a Pro package folder whose pxqe_runtime.py and lib/libpxqe.so are used; run with CUDA_VISIBLE_DEVICES=<one card>)"""
import ctypes
import os
import sys

here = os.path.abspath(sys.argv[1])
sys.path.insert(0, here)
import pxqe_runtime as RT  # noqa: E402

fails = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg, flush=True)
    if not ok:
        fails.append(msg)
    return ok


lib, info = RT.load(os.path.join(here, "lib", "libpxqe.so"), here)
check(True, "lib/libpxqe.so opened (CUDA libraries from: %s %s)" % (info["source"], info["dir"]))
maps = open("/proc/self/maps").read()
loaded = sorted({os.path.basename(l.split()[-1]) for l in maps.splitlines() if "/lib" in l and any(n in l for n in ("cublas", "cusolver", "cusparse", "nvJitLink"))})
check(len(loaded) >= 2, "loaded into the process: " + ", ".join(loaded))

cuda = ctypes.CDLL("libcuda.so.1")
assert check(cuda.cuInit(0) == 0, "cuInit (the NVIDIA driver answers)")
dev = ctypes.c_int()
assert check(cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0, "cuDeviceGet(0)")
name = ctypes.create_string_buffer(128)
cuda.cuDeviceGetName(name, 128, dev)
major, minor = ctypes.c_int(), ctypes.c_int()
cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)
cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)
print("  card: %s (sm_%d%d)" % (name.value.decode(), major.value, minor.value), flush=True)
ctx = ctypes.c_void_p()
assert check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) == 0, "cuDevicePrimaryCtxRetain")
assert check(cuda.cuCtxSetCurrent(ctx) == 0, "cuCtxSetCurrent")


def dalloc(nbytes):
    p = ctypes.c_uint64()
    assert cuda.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(nbytes)) == 0
    return p


def h2d(p, arr):
    assert cuda.cuMemcpyHtoD_v2(p, arr, ctypes.c_size_t(ctypes.sizeof(arr))) == 0


def d2h(p, arr):
    assert cuda.cuMemcpyDtoH_v2(arr, p, ctypes.c_size_t(ctypes.sizeof(arr))) == 0


def vp(p):
    return ctypes.c_void_p(p.value)


# ---- cuBLAS: C = A (2x3) * B (3x2), column-major, float32
cublas = ctypes.CDLL("libcublas.so.12")
h = ctypes.c_void_p()
assert check(cublas.cublasCreate_v2(ctypes.byref(h)) == 0, "cublasCreate_v2")
A = (ctypes.c_float * 6)(1, 4, 2, 5, 3, 6)        # [[1,2,3],[4,5,6]] column-major
B = (ctypes.c_float * 6)(7, 9, 11, 8, 10, 12)     # [[7,8],[9,10],[11,12]]
C = (ctypes.c_float * 4)()
dA, dB, dC = dalloc(24), dalloc(24), dalloc(16)
h2d(dA, A)
h2d(dB, B)
alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
r = cublas.cublasSgemm_v2(h, 0, 0, 2, 2, 3, ctypes.byref(alpha), vp(dA), 2, vp(dB), 3, ctypes.byref(beta), vp(dC), 2)
check(r == 0, "cublasSgemm_v2 returned 0")
d2h(dC, C)
want = [58, 139, 64, 154]                          # [[58,64],[139,154]] column-major
check(list(C) == want, "SGEMM result %s == %s" % (list(C), want))
cublas.cublasDestroy_v2(h)

# ---- cuSOLVER: Cholesky solve of a 3x3 SPD system in double precision (what the Hessian chain does)
cs = ctypes.CDLL("libcusolver.so.11")
sh = ctypes.c_void_p()
assert check(cs.cusolverDnCreate(ctypes.byref(sh)) == 0, "cusolverDnCreate")
n = 3
M = [[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]]
b = [1.0, 2.0, 3.0]
Ad = (ctypes.c_double * 9)(*[M[i][j] for j in range(3) for i in range(3)])
bd = (ctypes.c_double * 3)(*b)
dM, db, dinfo = dalloc(72), dalloc(24), dalloc(4)
h2d(dM, Ad)
h2d(db, bd)
lw = ctypes.c_int()
assert check(cs.cusolverDnDpotrf_bufferSize(sh, 0, n, vp(dM), n, ctypes.byref(lw)) == 0, "cusolverDnDpotrf_bufferSize")
dwork = dalloc(max(8, lw.value * 8))
r1 = cs.cusolverDnDpotrf(sh, 0, n, vp(dM), n, vp(dwork), lw.value, vp(dinfo))
r2 = cs.cusolverDnDpotrs(sh, 0, n, 1, vp(dM), n, vp(db), n, vp(dinfo))
check(r1 == 0 and r2 == 0, "cusolverDnDpotrf + cusolverDnDpotrs returned 0")
info_h = (ctypes.c_int * 1)()
d2h(dinfo, info_h)
check(info_h[0] == 0, "devInfo 0 (the factorization succeeded)")
x = (ctypes.c_double * 3)()
d2h(db, x)
res = max(abs(sum(M[i][j] * x[j] for j in range(3)) - b[i]) for i in range(3))
check(res < 1e-12, "Cholesky solve residual %.2e < 1e-12 (x = %s)" % (res, [round(v, 6) for v in x]))
cs.cusolverDnDestroy(sh)
print("RESULT: %s" % ("PASS" if not fails else "FAIL (%d)" % len(fails)))
sys.exit(1 if fails else 0)
