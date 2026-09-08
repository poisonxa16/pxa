#!/usr/bin/env python3
"""pxq-anchor-check.py -- audit the fp16 row anchors of every PXQ tensor in a GGUF.

Every PXQ tier (PXQ1/2/3/4/4HQ/6) stores one fp16 ROW ANCHOR per row, packed as a 128 B
header at the head of each 64-row panel; a row's remaining bytes are the panel's scale/code
slabs. The anchor is that row's |absmax|, so it is finite and NON-NEGATIVE by construction.
Anything else means those panel bytes were never written by the codec (the reader is looking
at stale/misaligned data) or a scale overflowed fp16 -- either way the rows in that panel
decode to NaN and any prompt whose router picks them emits "!!!!".

That is exactly what the 2026-07-22 PXA-Fusion2-35B upload shipped: a GGUFReader ->
GGUFWriter metadata round-trip on a gguf-py that did not know about the PXQ per-row anchor
meta (2 B/row; fixed in 1a1b46aa2e) truncated every PXQ tensor by 2*nrows bytes, so each
tensor's last 2*nrows bytes are the head of the NEXT tensor.

Matching guards in the engine: pxa_pxq_check_anchors() in src/llama-quantize.cpp (produce
side, hard error) and pxa_check_pxq_anchors() in ggml/src/pxq-quants.cpp, reached by
--check-tensors / --validate-quants (load side).

    usage: pxq-anchor-check.py FILE.gguf [--tensor-substr=SUBSTR] [--quiet]

Per-expert counts are printed for 3-D MoE tensors. Exit status 1 if any anchor is bad.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
from gguf.gguf_reader import GGUFReader   # noqa: E402

# ggml type id -> (name, blck_size, type_size, row_meta_size); see ggml/src/ggml.c type_traits.
PXQ_TYPES = {
    248: ("pxq1",   32,  5, 2),
    252: ("pxq4",   32, 17, 2),
    253: ("pxq4hq", 32, 18, 2),
    254: ("pxq2",   32,  9, 2),
    255: ("pxq3",   32, 13, 2),
    256: ("pxq6",   32, 21, 2),
}
PANEL_ROWS = 64


def anchors_of(raw, ne, blck, tsz, rms):
    """(n_experts, n_rows) array of fp32 row anchors, read out of the panel headers."""
    k, nrows, nmats = ne[0], ne[1], ne[2] * ne[3]
    kslabs = k // blck
    panel_bytes = rms * PANEL_ROWS + kslabs * tsz * PANEL_ROWS
    mat_bytes = nrows * (rms + kslabs * tsz)
    npanel = nrows // PANEL_ROWS
    offs = (np.arange(nmats)[:, None] * mat_bytes
            + np.arange(npanel)[None, :] * panel_bytes).reshape(-1)
    idx = (offs[:, None] + np.arange(rms * PANEL_ROWS)[None, :]).reshape(-1)
    return raw[idx].view(np.float16).astype(np.float32).reshape(nmats, nrows)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quiet = "--quiet" in sys.argv
    substr = None
    for a in sys.argv[1:]:
        if a.startswith("--tensor-substr="):
            substr = a.split("=", 1)[1]
    if not args:
        print(__doc__)
        return 2
    path = args[0]

    reader = GGUFReader(path, "r")
    n_scanned = 0
    n_bad_total = 0
    for t in reader.tensors:
        tid = int(t.tensor_type)
        if tid not in PXQ_TYPES:
            continue
        if substr and substr not in t.name:
            continue
        tname, blck, tsz, rms = PXQ_TYPES[tid]
        ne = [int(x) for x in t.shape] + [1, 1]
        if ne[1] % PANEL_ROWS or ne[0] % blck:
            print(f"  ! {t.name}: ne={ne[:4]} is not PXQ slab geometry, skipping")
            continue
        n_scanned += 1
        raw = np.asarray(t.data).view(np.uint8).reshape(-1)
        a = anchors_of(raw, ne, blck, tsz, rms)
        bad = ~np.isfinite(a) | (a < 0.0)
        per_expert = bad.sum(axis=1)
        n_bad = int(per_expert.sum())
        n_bad_total += n_bad
        if n_bad:
            n_nf = int((~np.isfinite(a)).sum())
            print(f"BAD {t.name} [{tname}] ne={ne[:4]} -- {n_bad} bad anchors "
                  f"({n_nf} non-finite, {n_bad - n_nf} negative) of {a.size}")
            for e in np.nonzero(per_expert)[0]:
                print(f"      expert {int(e):4d} -> {int(per_expert[e])} of {ne[1]} anchors bad")
        elif not quiet:
            print(f"ok  {t.name} [{tname}] ne={ne[:4]} experts={ne[2]*ne[3]}")

    print(f"\n{os.path.basename(path)}: scanned {n_scanned} PXQ tensors, "
          f"{n_bad_total} bad anchors total")
    return 1 if n_bad_total else 0


if __name__ == "__main__":
    sys.exit(main())
