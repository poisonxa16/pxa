"""flashnext_audit.py -- boot-time proof that the checkpoint and the model agree.

Enabled by ``PXA_FLASHNEXT_AUDIT=1`` (log only) or ``=fail`` (log, then refuse the boot).
Written after five silent-garbage defects were found in one checkpoint; the
common shape of all five is that the ENGINE DOES NOT COMPLAIN. A misnamed weight is not a
crash: AutoWeightsLoader is constructed with ``ignore_unexpected_suffixes`` and swallows it,
and a parameter nobody filled keeps whatever ``torch.empty`` left in it.

So this prints three things a boot log would otherwise never carry:

  ORPHANS   every checkpoint key that reached the loader and was IGNORED rather than loaded.
            Hooked at AutoWeightsLoader._can_ignore_unexpected, which is the one place that
            decides to drop a weight silently. A non-empty list means the checkpoint carries
            tensors this model has nowhere to put -- e.g. the 24 indexer tensors when the QSA
            config fields are misspelled, or 513 PLE shards under a module path that does not
            exist.

  UNFILLED  every parameter of the built model that no checkpoint key filled. vLLM has this
            check (DefaultModelLoader.track_weights_loading) but disables it for quantized
            models, which is exactly the case here; it is forced on.

  GEOMETRY  the layer index that actually built a PLE stack, and the indexer's heads / kv
            heads / budget / compress ratio as the model READ them -- not as config.json
            spells them. ``indexer_n_heads`` vs ``index_n_heads`` is the difference between a
            sparse model and a dense one, and both spellings load.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("pxq4_vllm.flashnext_audit")


def _bytes_probe(t, sample: int = 1 << 18) -> str:
    """Describe a parameter's CONTENT without materialising a second copy of it.

    The tensors in question are multi-gigabyte pxq4 slabs on a 16 GB card, so this samples a
    flat prefix and a flat suffix rather than reducing the whole thing: telling "never written"
    (all zeros / denormal noise) from "loaded" needs no more than that.
    """
    import torch as _t
    f = t.detach().reshape(-1)
    n = f.numel()
    take = min(sample, n)
    head = f[:take].to(_t.float32)
    tail = f[max(0, n - take):].to(_t.float32)
    nz = int(head.count_nonzero()) + int(tail.count_nonzero())
    tot = head.numel() + tail.numel()
    return ("dtype=%s shape=%s numel=%d sampled=%d nonzero=%d absmean=%.6g absmax=%.6g"
            % (t.dtype, tuple(t.shape), n, tot, nz,
               float(_t.cat((head, tail)).abs().mean()),
               float(_t.cat((head, tail)).abs().max())))

MODE = os.environ.get("PXA_FLASHNEXT_AUDIT", "").strip().lower()
_ignored: list[str] = []


def _install() -> None:
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    orig_can_ignore = AutoWeightsLoader._can_ignore_unexpected

    def can_ignore(self, qualname: str) -> bool:
        ok = orig_can_ignore(self, qualname)
        if ok:
            _ignored.append(qualname)
        return ok

    AutoWeightsLoader._can_ignore_unexpected = can_ignore

    orig_load = DefaultModelLoader.load_weights

    def load_weights(self, model, model_config):
        _ignored.clear()
        # vLLM turns its own missing-parameter check off for quantized models. This IS a
        # quantized model, and the check is the whole point.
        self.enable_weights_track = False          # we do it ourselves, with both halves
        loaded = None
        try:
            orig_track = DefaultModelLoader.track_weights_loading

            def track(self_, model_, loaded_weights):
                nonlocal loaded
                loaded = loaded_weights
                return None
            DefaultModelLoader.track_weights_loading = track
            self.enable_weights_track = True
            orig_load(self, model, model_config)
        finally:
            DefaultModelLoader.track_weights_loading = orig_track

        params = {n for n, _ in model.named_parameters()}
        unfilled = sorted(params - set(loaded or ())) if loaded is not None else None

        # ---- GEOMETRY: what the MODEL built, not what config.json says ------------------
        ple_layers = sorted({
            n.rsplit(".ple", 1)[0].rsplit(".", 1)[-1]
            for n, m in model.named_modules()
            if n.endswith(".ple") and m is not None
        })
        idx = None
        for n, m in model.named_modules():
            # Found by the parameter it owns, not by the attribute name it happens to have:
            # index_qk_proj is the one thing a QSA indexer always builds.
            if hasattr(m, "index_qk_proj"):
                idx = (getattr(m, "index_n_heads", "?"), getattr(m, "index_kv_heads", "?"),
                       getattr(m, "index_head_dim", "?"), getattr(m, "token_topk", "?"),
                       getattr(m, "compress_ratio", "?"), n)
                break
        cfg = getattr(model_config, "hf_text_config", None)
        logger.info("FLASHNEXT GEOMETRY: PLE stack built on decoder layer(s) %s "
                    "(config ple_layer_ids=%s, one-based)", ple_layers or "NONE",
                    getattr(cfg, "ple_layer_ids", None))
        if idx is None:
            logger.info("FLASHNEXT GEOMETRY: NO QSA indexer was built -- every full-attention "
                        "layer is dense. If this model has indexer tensors, the config's "
                        "indexer_* fields did not reach it.")
        else:
            logger.info("FLASHNEXT GEOMETRY: QSA indexer at %s -- n_heads=%s kv_heads=%s "
                        "head_dim=%s budget=%s compress_ratio=%s",
                        idx[5], idx[0], idx[1], idx[2], idx[3], idx[4])

        # ---- ORPHANS / UNFILLED ----------------------------------------------------------
        logger.info("FLASHNEXT AUDIT: %d checkpoint keys IGNORED, %s model parameters UNFILLED",
                    len(_ignored),
                    "unknown" if unfilled is None else str(len(unfilled)))
        for q in _ignored[:20]:
            logger.info("FLASHNEXT AUDIT: ignored checkpoint key %s", q)
        for q in (unfilled or [])[:20]:
            logger.info("FLASHNEXT AUDIT: unfilled parameter %s", q)
        # ---- WEIGHTS: is an "unfilled" parameter actually empty, or only untracked? ------
        # A fused MoE stack is filled shard-by-shard through the expert weight_loader, which
        # need not register the fused parameter name in loaded_params. Distinguish the benign
        # bookkeeping gap from a genuinely never-written tensor by looking at the bytes.
        if unfilled:
            named = dict(model.named_parameters())
            probe = list(unfilled[:4]) + list(unfilled[-2:])
            for q in probe:
                t = named.get(q)
                if t is None:
                    continue
                try:
                    logger.info("FLASHNEXT AUDIT: unfilled? %s %s", q, _bytes_probe(t))
                except Exception as exc:
                    logger.info("FLASHNEXT AUDIT: unfilled? %s unreadable (%r)", q, exc)
            for q, t in named.items():
                if q not in set(unfilled) and t.numel() > 1024:
                    logger.info("FLASHNEXT AUDIT: control (loaded) %s %s", q, _bytes_probe(t))
                    break

        bad = len(_ignored) + len(unfilled or ())
        if bad and MODE == "fail":
            raise RuntimeError(
                f"FLASHNEXT AUDIT FAILED: {len(_ignored)} ignored checkpoint keys and "
                f"{len(unfilled or ())} unfilled parameters. A weight that is neither loaded "
                f"nor accounted for is the silent-garbage failure this check exists for; "
                f"unset PXA_FLASHNEXT_AUDIT=fail only when every name above is understood.")

    DefaultModelLoader.load_weights = load_weights
    logger.info("flashnext_audit installed (mode=%s)", MODE or "off")


if MODE in ("1", "true", "yes", "on", "fail"):
    try:
        _install()
    except Exception as exc:      # never let the audit be the reason a boot dies
        logger.warning("flashnext_audit could not install: %r", exc)
