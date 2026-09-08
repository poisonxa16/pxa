# PXA PXQ Sidecar Package

The complete PXQ quantized-inference sidecar for vLLM, as built and shipped by
PXA Network. Everything needed to serve PXQ lives here; nothing in this
directory depends on a path outside the repository.

**Tiers.** As of 2026-09-05 this package serves **PXQ2, PXQ3 and PXQ4**, and a single
checkpoint may mix them across modules. The three are one layout with three code widths —
64-row panels, a 128 B fp16 anchor header, 32-column slabs, one shared SUB16 table, one
reconstruction contract — differing only in bits per code (hence 576 / 832 / 1088 bytes per
slab) and book size (4 / 8 / 16 entries). The directory, the torch namespace and the on-disk
tensor keys all still say `pxq4`; those are wire-format names kept deliberately so that every
checkpoint and launcher already in the field is unaffected. The **tier lives in the
checkpoint's `config.json`**, never in a name. See `docs/VLLM.md` §2.

    kernels/       prebuilt PXQ4 kernel libraries, one per arch + revision.
                   Selected at runtime with the PXQ4_LIB environment variable.
    kernels-src/   the CUDA/C++/Python sources those libraries are built from.
    sidecar/       the pxq4_vllm plugin trees vLLM loads via PYTHONPATH.
                   site-union  - both arches, carries the fp16 mmv hook.
                   site-sm60   - Pascal-targeted variant.

## Why this package exists

Until 2026-08-27 none of this was under version control. The kernel libraries,
the plugin sites and the kernel sources lived only in scratch working
directories on a single machine, referenced by absolute path from launcher
scripts. Losing that machine would have meant losing the ability to rebuild the
kernels at all. This package is the fix: the artifacts, their sources and their
provenance in one branded, self-contained tree.

## Choosing a kernel

| library | arch | status |
|---|---|---|
| `libpxq4_sm70_v12b.so` | sm_70 | **SHIPPED** — what the live Volta seat runs. MMA decode path, armed by `PXQ4_MMV_MMA=1`. |
| `libpxq4_sm70_v10.so` | sm_70 | the scalar decode path it replaced: 51.46 tok/s single-stream, 2x V100. |
| `libpxq4_sm70_v9.so`  | sm_70 | fp16 hook build, superseded by v10. |
| `libpxq4_sm60_v10.so` | sm_60 | survival-gated final; fp16 smem tile kernel. |
| `libpxq4_sm60_v11.so` | sm_60 | adds `gemm2d_out` behind `PXQ4_GEMM2D`, **default off**: ~+34% prefill but it FAILED first-token quality at 87.5%. Do not enable without re-gating quality. |
| `libpxq4_sm60_v9.so`  | sm_60 | adds `f16_mmv_out`, ~+9.4% single-stream on P100. |
| `libpxq4_sm60_v8.so`  | sm_60 | adds expert-indexed MoE (`moe_mmv_out`). |
| `libpxq_sm70_v13.so`  | sm_70 | **required for PXQ2/PXQ3.** v12b's PXQ4 kernels unmodified, plus `pxq2_*`/`pxq3_*` dequant/mmv/linear/moe_mmv and `pxq_selftest`. |
| `libpxq_sm60_v13.so`  | sm_60 | the sm_60 twin (carries sm_60 and sm_70 cubins). |

Note the missing `4`: the `libpxq_*` libraries are the first that are not PXQ4-only, and
`libpxq4_sm70_v13.so` already existed as an unrelated PXQ4-only build. Do not rename either.
A PXQ2/PXQ3 checkpoint pointed at a v12b-or-earlier library fails at layer construction with
a message naming the library it needs — it is not a silent wrong-output path.

`PXQ4_LIB` must always be set explicitly. `sidecar/site-union` bundles an
sm_70-only `.so` built against the torch 2.10 ABI; if `PXQ4_LIB` is unset the
loader can reach that one on a Pascal image and die with
`undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv` part-way through
model load, rather than with a clear message.

## Identifying a kernel revision

The ops are registered through TORCH_LIBRARY string schemas, not exported C
symbols, so `nm` will not find them — `nm -D | grep linear_out` returns nothing
even for a library that has it. Scan printable strings instead:

    strings -a kernels/libpxq4_sm60_v10.so | grep -E '^(moe_mmv_out|f16_mmv_out|gemm2d_out|pxq_selftest)$'

    moe_mmv_out  present -> v8 or later
    f16_mmv_out  present -> v9 or later
    gemm2d_out   present -> v11
    pxq_selftest present -> v13 or later, i.e. it serves PXQ2/PXQ3

`pxq_selftest` is also runnable, and running it is the first thing to do on a new card:
`torch.ops.pxq4.pxq_selftest(254)` decodes a deterministic synthetic panel set with a host
oracle written from the format spec and requires the device kernels to agree BIT-EXACTLY,
then checks the decode GEMV against a host replay of the kernel's own fold. Tier `252` checks
the templated PXQ4 instantiation against the shipped PXQ4 kernel. Wrapped, with real-tensor
and expert-indexing checks, in `kernels-src/tests_pxq23/gpu_selftest.py`.

See `MANIFEST.md` for sizes and md5s of every library in this package.
