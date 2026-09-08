"""gguf_to_vllm — offline converter: mixed-type PXQ4 GGUF -> vLLM-loadable safetensors.

Component A of the PXQ4-in-vLLM port (plan §5 and §9). Pure Python + numpy: no torch, no
CUDA, no vLLM, no GPU, no lease. Every gate this component owns (G1-G4) runs on a laptop.

Modules, and which of them are CROSS-COMPONENT CONTRACT SURFACE:

    layout.py       CONTRACT (plan §6.2). Panel/slab arithmetic and the blob <-> (slabs,
                    anchor) split. Imported by the converter, by the runtime package's
                    parameter shaping, and by the kernel tests. Never imports torch or vllm.
    reference.py    CONTRACT (plan §6.3). The numpy bit-exactness oracle. The CUDA extension
                    is pinned against THIS at gate G6, and THIS is pinned against the engine's
                    own pxa_deq_row_pxq6 at gate G1.
    gguf_raw.py     dependency-free GGUF reader that survives unknown tensor types.
    dequant_ref.py  numpy decoders for q8_0, q6_K, mxfp4, f16, f32 — the other four types a
                    "PXQ4 file" actually contains.
    namemap.py      ggml -> HF names, the per-policy PXQ4 allow-list, and the ignore list.
    safetensors_io.py  minimal streaming writer (needed because BF16 must pass through
                    uninterpreted and numpy has no bfloat16).
    encoder.py      ctypes binding to the native PXQ4 encoder. P2 policies only.
    convert.py      the CLI. ``--dry-run`` plans everything from the header alone.
    verify.py       gates G1/G2/G3 against a real file, read-only, no GPU.

The emitted contract (plan §5.3), which components B, C and D all build against:

    <module>.pxq4_slabs    uint8    [N/64, K/32, 1088]   C-contiguous
    <module>.pxq4_anchor   float16  [N/64, 64]           C-contiguous
    <module>.weight        float16  [N, K]               for everything not served as PXQ4

The PXQ4 pair is a PURE SPLIT of the GGUF panel bytes — no byte reordered, no value
recomputed — which is why the conversion is verifiable by byte comparison rather than by a
numeric tolerance.
"""

from . import layout, reference  # noqa: F401  (the two contract modules, importable directly)

__all__ = ["layout", "reference"]
