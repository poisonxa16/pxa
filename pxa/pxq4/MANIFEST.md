# PXA PXQ4 Kernel Manifest

Captured 2026-08-27T21:26:12Z; v12b added 2026-09-03; the v13 tier libraries 2026-09-05.
Sizes and checksums for every
prebuilt library in this package, so a rebuild can be verified against
the artifact that produced the recorded numbers.

| library | bytes | md5 |
|---|---|---|
| libpxq4_sm60_v10.so | 2135960 | edfea330185e11237858c74179226400 |
| libpxq4_sm60_v11.so | 2201888 | 5728e86adda1e4c065377b4b0938943a |
| libpxq4_sm60_v8.so | 1314768 | ca13f588342bfb161ffff4a69becf97e |
| libpxq4_sm60_v9.so | 1491856 | ec5827cdbc69bd0712b3bc4d8196b529 |
| libpxq4_sm70_v10.so | 2228200 | dbdb86efb110db760c4180a87abbd89a |
| libpxq4_sm70_v9.so | 1563712 | 9c9b07251a772b9326a2207a9708fc76 |
| libpxq4_sm70_v12b.so | 2715168 | 67f0895de66f1b6a3206fbf43585c5d1 |
| libpxq_sm70_v13.so | 2822408 | 4c9ab114041b10f9760451bda15328a8 |
| libpxq_sm60_v13.so | 5084776 | 4ffc6a203188e490a787d3db77e28efe |

The two `libpxq_*_v13.so` entries are the **tier** libraries: v12b's PXQ4 kernels unmodified,
plus the PXQ2/PXQ3 op family and `pxq_selftest`. They carry no `4` in the name because they
are the first that are not PXQ4-only, and because `libpxq4_sm70_v13.so` already exists as an
unrelated PXQ4-only build. The sm_60 one is larger because it carries **two** cubins (sm_60
and sm_70), matching the sm60 image's own `TORCH_CUDA_ARCH_LIST`; verify with
`cuobjdump --list-elf`.

## Building the v13 tier libraries

`kernels-src/build_pxq_v13.sh`, run once per arch, INSIDE the serving image that will load the
result — the ABI is not portable between the two images (torch 2.7.1+cu126 for sm60, 2.10+cu128
for sm70), and a mismatch dies with `undefined symbol:
_ZNK3c1010TensorImpl15incref_pyobjectEv` part-way through model load rather than at import:

```bash
ARCH=sm70 ./build_pxq_v13.sh     # inside pxa-vllm:sm70-v15
ARCH=sm60 ./build_pxq_v13.sh     # inside pxa-vllm:sm60-v15
```

`build-in-container.sh` in this directory does both, CPU-only and niced. The tensor-core TU
(`pxq4_mma.cu`) is compiled sm_70-only in both builds: it uses wmma m16n16k16, which does not
exist on sm_60 and will not compile for it, and `pxq4_mma_supported()` gates the launch on
compute capability >= 7 so the sm60 library never reaches it on a Pascal card. The symbol has
to be present regardless, because `pxq4_kernel_torch.cpp` references it unconditionally.

## Building `libpxq4_sm70_v12b.so`

This repository ships kernel **source**, not compiled objects, so the v12b library named by the
`PXQ4_LIB=libpxq4_sm70_v12b.so` recipe in `docs/COOKBOOK.md` and `docs/PXA-SM70-SERVING.md` has
to be built once. The script that produces exactly the artifact recorded above is
`kernels-src/build_v12b.sh`; run it inside the environment whose vLLM you will serve with, so
that the torch ABI it links against is the one that will load it:

```bash
cd pxa/pxq4/kernels-src
PXQ4_BUILD_OUT=/where/your/kernels/live ./build_v12b.sh
```

It compiles `pxq4_kernel.cu`, `pxq4_mma.cu` (the v12b tensor-core decode path armed by
`PXQ4_MMV_MMA=1`) and `pxq4_kernel_torch.cpp` for `sm_70`, links them against the local torch,
and prints the resulting size and md5 so you can compare them with the row above. A different
torch version or C++ ABI will legitimately produce a different md5; the size and the exported
symbol list it prints are the useful checks in that case.
