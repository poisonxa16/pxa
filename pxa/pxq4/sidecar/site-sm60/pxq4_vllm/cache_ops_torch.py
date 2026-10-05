"""KV-cache scatter write without the _C_cache_ops kernels (Pascal port).

Boot 10 of Flash-Next under vLLM on the four P100s (2026-09-07 00:06) died at CUDA-graph capture in
flash_attn.py do_kv_cache_update:

    AttributeError: '_OpNamespace' '_C_cache_ops' object has no attribute 'reshape_and_cache_flash'

Contract (vllm/_custom_ops.py): key/value [num_padded_tokens, H, D]; key_cache/value_cache
[num_blocks, block_size, H, D] (flash layout, may be strided views of a [num_blocks, 2, ...] tensor);
slot_mapping [num_actual_tokens] (slot = block * block_size + offset, < 0 = skip); kv_cache_dtype;
k_scale/v_scale (fp8 only). Only the first slot_mapping.shape[0] tokens are written.

Torch implementation: two-index advanced index_put_ (block, offset) -- works on strided caches,
static shapes, no host sync, so it is CUDA-graph capturable. Tokens with slot < 0 (padding) are
written to block 0, vLLM V1's null block, which never holds a real token. fp8 caches are not
supported here (they never are on Pascal).

PXA_CACHE_OPS=torch (default on sm<70) | off
"""

import logging
import os

log = logging.getLogger(__name__)

_MODE = os.environ.get("PXA_CACHE_OPS", "auto").strip().lower()


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
        log.info("cache_ops_torch: not active (mode=%s)", _MODE)
        return
    import torch

    def reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype="auto",
                                k_scale=None, v_scale=None):
        if isinstance(kv_cache_dtype, str) and "fp8" in kv_cache_dtype:
            raise NotImplementedError("cache_ops_torch: fp8 KV cache is not supported on sm<70")
        n = slot_mapping.shape[0]
        block_size = key_cache.shape[1]
        slots = slot_mapping.to(torch.long)
        safe = torch.where(slots >= 0, slots, torch.zeros_like(slots))
        blk = safe // block_size
        off = safe - blk * block_size
        key_cache[blk, off] = key[:n].to(key_cache.dtype)
        value_cache[blk, off] = value[:n].to(value_cache.dtype)

    def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype="auto",
                          k_scale=None, v_scale=None):
        # paged (non-flash) layout: key_cache [num_blocks, H, D/x, block_size, x], value_cache [num_blocks, H, D, block_size]
        if isinstance(kv_cache_dtype, str) and "fp8" in kv_cache_dtype:
            raise NotImplementedError("cache_ops_torch: fp8 KV cache is not supported on sm<70")
        n = slot_mapping.shape[0]
        nb, H, Dx, block_size, x = key_cache.shape
        slots = slot_mapping.to(torch.long)
        safe = torch.where(slots >= 0, slots, torch.zeros_like(slots))
        blk = safe // block_size
        off = safe - blk * block_size
        k = key[:n].to(key_cache.dtype).reshape(n, H, Dx, x)
        # advanced indexing with a broadcast over H, Dx: build full index tensors
        kc = key_cache.permute(0, 3, 1, 2, 4)   # [nb, block_size, H, Dx, x]
        kc[blk, off] = k
        vc = value_cache.permute(0, 3, 1, 2)    # [nb, block_size, H, D]
        vc[blk, off] = value[:n].to(value_cache.dtype)

    ns = torch.ops._C_cache_ops
    installed = []
    for name, fn in (("reshape_and_cache_flash", reshape_and_cache_flash), ("reshape_and_cache", reshape_and_cache)):
        if name in getattr(ns, "__dict__", {}):
            continue
        try:
            getattr(ns, name)
            continue
        except (AttributeError, RuntimeError):
            pass
        try:
            setattr(ns, name, fn)
            installed.append(name)
        except Exception as exc:
            log.warning("cache_ops_torch: could not install %s (%s)", name, exc)
    log.info("cache_ops_torch: torch KV scatter installed for %s (sm<70)", ",".join(installed) or "nothing")


_install()
