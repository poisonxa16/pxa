"""QSA block top-k without the fork's CUDA selectors (Pascal port).

Boot 9 of Flash-Next under vLLM on the four P100s (2026-09-06 23:51) compiled every Triton kernel and
died at CUDA-graph capture in ops/qsa.py qsa_select_paged_tokens:

    AttributeError: '_OpNamespace' '_C' object has no attribute 'persistent_topk'

This image's torch.ops._C carries none of the three selectors the fork can call (persistent_topk,
cooperative_topk [sm_90], qsa_lexicographic_topk [the exact Volta one]). Contract, from the callers:

    topk(logits[rows, cols] f32, visible_blocks[rows] i32, blocks[rows, block_topk] i32 (out),
         workspace, block_topk: int, score_columns: int)

Row r ranks columns [0, min(visible_blocks[r], score_columns)) by score descending and writes the
first block_topk column indices into blocks[r]; slots past the row's visible count are -1 (the
expand kernel reads only min(visible, BLOCK_TOPK) ranks and treats block < 0 as no token).

Implementation: torch.topk per row over the visible columns (tie order is torch's, not the Volta
selector's lexicographic order -- documented, not gated) with cached shape-keyed buffers so no
allocation happens after the first call for a shape. CUDA-graph capturable.

PXA_QSA_TOPK=torch (default on sm<70) | off
"""

import logging
import os

log = logging.getLogger(__name__)

_MODE = os.environ.get("PXA_QSA_TOPK", "auto").strip().lower()


def _wanted() -> bool:
    if _MODE in ("off", "0", "no"):
        return False
    if _MODE in ("torch", "on", "1", "force"):
        return True
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability()
        return major < 7
    except Exception:
        return False


def _install() -> None:
    if not _wanted():
        log.info("qsa_topk_torch: not active (mode=%s)", _MODE)
        return
    import torch

    # Boot 15/16 (2026-09-07 02:24): the first version did a stable descending SORT of the whole
    # [rows, cols] score matrix. torch.sort allocates its radix scratch at call time -- 230 MiB on a
    # 16384-column row block -- and with the KV cache sized to the card that allocation failed on
    # rank 1 (OOM, 124 MiB free), which left the other ranks blocked in pp_receive forever.
    # This version: torch.topk (output-sized allocation only, k <= 512) and cached, shape-keyed
    # work buffers, so after the first call for a shape nothing is allocated at all (graph-safe).
    _work = {}

    def _buffers(rows, cols, device):
        key = (rows, cols, str(device))
        b = _work.get(key)
        if b is None:
            # the -inf scalar is a device tensor made ONCE: torch.tensor(...) inside a CUDA-graph
            # capture is a host->device copy and fails the capture
            b = (torch.empty((rows, cols), dtype=torch.float32, device=device),
                 torch.arange(cols, device=device, dtype=torch.int32),
                 torch.full((), float("-inf"), dtype=torch.float32, device=device))
            _work[key] = b
        return b

    def topk_torch(logits, visible_blocks, blocks, workspace=None, block_topk=None, score_columns=None):
        if block_topk is None:
            block_topk = blocks.shape[1]
        rows = blocks.shape[0]
        cols = logits.shape[1] if score_columns is None else min(int(score_columns), logits.shape[1])
        masked, col, neg_inf = _buffers(rows, cols, logits.device)
        limit = torch.clamp(visible_blocks[:rows].to(torch.int32), max=cols)
        live = col[None, :] < limit[:, None]
        torch.where(live, logits[:rows, :cols], neg_inf, out=masked)
        k = min(int(block_topk), cols)
        vals, idx = torch.topk(masked, k, dim=1, largest=True, sorted=True)
        # THE NATIVE SELECTOR RETURNS THE CHOSEN BLOCKS IN ASCENDING BLOCK-INDEX ORDER, not score order
        # (verified against _C.persistent_topk on a V100, 2026-09-07: same set, sorted ids, -1 padding last).
        # Boot 20 served 'ductduct...' with score-ordered ids: the expansion downstream assumes monotone ids.
        idx = torch.where(vals > neg_inf, idx, torch.full_like(idx, cols))   # invalid -> sentinel past the end
        idx, _ = torch.sort(idx, dim=1)                                        # ascending, sentinels last
        idx = torch.where(idx < cols, idx, torch.full_like(idx, -1)).to(torch.int32)
        blocks[:rows, :k].copy_(idx)
        if k < blocks.shape[1]:
            blocks[:rows, k:].fill_(-1)
        return blocks

    ns = torch.ops._C
    installed = []
    for name in ("persistent_topk", "cooperative_topk", "qsa_lexicographic_topk"):
        if name in getattr(ns, "__dict__", {}):
            continue
        try:
            getattr(ns, name)          # present in the library: leave the real kernel alone
            continue
        except (AttributeError, RuntimeError):
            pass
        fn = topk_torch
        if name == "qsa_lexicographic_topk":
            def fn(logits, visible_blocks, blocks, block_topk):   # noqa: E306  (4-arg contract)
                return topk_torch(logits, visible_blocks, blocks, None, block_topk, None)
        try:
            setattr(ns, name, fn)
            installed.append(name)
        except Exception as exc:
            log.warning("qsa_topk_torch: could not install %s (%s)", name, exc)
    log.info("qsa_topk_torch: torch stable-sort top-k installed for %s (sm<70)", ",".join(installed) or "nothing")


_install()
