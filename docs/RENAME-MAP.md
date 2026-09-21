# PXQ display-name re-ladder (2026-07-19) — RENAME MAP

> **UPDATE 2026-07-21 (supersedes parts of this file):**
> 1. The internal identifiers were ALSO renamed (the "keeps old identifier" row below for
>    `GGML_TYPE_PXQ6*` / `LLAMA_FTYPE_MOSTLY_PXQ6*` is obsolete): id 252/253 are
>    `GGML_TYPE_PXQ4`/`GGML_TYPE_PXQ4HQ` + `LLAMA_FTYPE_MOSTLY_PXQ4`/`PXQ4HQ` (252/253).
> 2. **PXQ6 now names the REAL 5-bit LM32 × E16-row tier**: `GGML_TYPE_PXQ6` = gguf type id 256,
>    `LLAMA_FTYPE_MOSTLY_PXQ6` = ftype 257, display `pxq6`, CLI arg `PXQ6`, ~5.27 bpw
>    (quantizer `src/pxq6r-quantize.inc.cpp`; the `pxq6r_*`/`PXA_PXQ6R` internal names are its
>    historical working name). The transitional `PXQ6`→PXQ4 CLI alias is GONE; `PXQ6HQ` survives
>    only as a deprecated alias for PXQ4-HQ.
> 3. **Type ids 250 (`PXQ4-LEGACY`, the MXFP4-repack legacy) and 251 (`PXQ5`, the learned-book
>    + SE8 legacy) are RETIRED and removed** — enum, traits, kernels, quantize paths, gguf-py,
>    docs. Old id-250/251 files fail with a clean "retired type" error at gguf load; requantize
>    from the source model with `llama-quantize PXQ4` or `PXQ6`. Ids 250/251 (ggml + ftype) are
>    reserved forever. The ladder is now strictly PXQ2/PXQ3/PXQ4/PXQ4-HQ/PXQ6 (+ PXQ_UNIVERSAL).

The PXQ tier names are re-laddered **by bit class**: the name now tells you the bpw class, matching
PXQ2/PXQ3. This is a **display-name change only** — numeric gguf type ids, file formats, kernels,
and every published `.gguf` are byte-identical and keep working.

## The map

| new display name | old display name | gguf type id | ftype id | status |
|---|---|---|---|---|
| **PXQ4** | PXQ6 | 252 | 252 | current 4-bit flagship (PX16 book + E16-row scales) |
| **PXQ4-HQ** | PXQ6HQ | 253 | 253 | 4-bit HQ variant (bs8 sub-scales) |
| **PXQ4V** | — (reserved) | — | — | reserved for the pair-VQ 4-bit type, if it ships |
| **PXQ4-LEGACY** | PXQ4 | 250 | 250 | LEGACY: lossless MXFP4 slab repack (`pxa-bench/pxq4_repack.py`) |
| PXQ5 | PXQ5 | 251 | 251 | LEGACY: superseded numerics (kept for reproducibility) |
| PXQ2 | PXQ2 | 254 | 254 | unchanged (2-bit) |
| PXQ3 | PXQ3 | 255 | 255 | unchanged (3-bit) |
| PXQ_UNIVERSAL | PXQ_UNIVERSAL | — | 256 | unchanged (mixed PXQ2/PXQ3/PXQ4 tier map) |

## What accepts what

- **`llama-quantize` target names:** `PXQ4`, `PXQ4-HQ` are canonical; **old names `PXQ6`, `PXQ6HQ`
  are accepted as aliases forever** (also `PXQ4HQ`). The legacy MXFP4 repack target is `PXQ4-LEGACY`.
  ⚠ The *string* `PXQ4` changed meaning: pre-rename it selected the MXFP4 repack (ftype 250), now it
  selects the 4-bit quality tier (ftype 252). Scripts that relied on the old meaning must switch to
  `PXQ4-LEGACY`.
- **Lowercase per-tensor names** (tier maps, `--custom-q`, `--output-tensor-type`, …): canonical
  `pxq4` / `pxq4hq`; old `pxq6` / `pxq6hq` accepted as aliases. The legacy repack type is
  `pxq4_legacy`.
- **Runtime display** (`llama_model_desc`, quantize logs, gguf dumps): types 252/253 print as
  `pxq4` / `pxq4hq`; type 250 prints as `pxq4_legacy`.
- **gguf-py:** `GGMLQuantizationType.PXQ4` (=252) and `.PXQ4HQ` (=253) are canonical; `.PXQ6` /
  `.PXQ6HQ` remain as value aliases; the legacy type is `.PXQ4_LEGACY` (=250).

## What deliberately KEEPS the old identifier

- **Env vars:** `PXA_PXQ6`, `PXA_PXQ6_{KSPLIT,VECX,GUFUSE,SCATFUSE,RAGTAIL,WMMA,…}` — engine
  contract with existing deployments; unchanged.
- **gguf metadata keys:** `pxa.pxq6.*` (book/tier provenance) — file-format contract; unchanged.
- **Published HF artifact filenames:** `PXA-Fusion2-35B-PXQ6.gguf`, `PXA-Fusion2-35B-PXQ6-MTP.gguf`,
  etc. — already published; those files ARE the PXQ4 tier.
- **Internal identifiers:** `GGML_TYPE_PXQ6*`, `LLAMA_FTYPE_MOSTLY_PXQ6*`, `src/pxq6-quantize.inc.cpp`,
  `ggml/src/ggml-cuda/pxa/pxq6.cuh`, `ggml-pxq6-tables.h` — code-level names, not user-facing.
