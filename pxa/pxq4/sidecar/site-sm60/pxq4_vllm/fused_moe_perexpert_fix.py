"""fused_moe_perexpert_fix.py -- a per-expert QUANTIZED weight is 3-D too, and vLLM assumes it is not.

THE DEFECT (w11 arm 3, the error under the one in #1574).

``FusedMoE.load_weights`` (fused_moe/layer.py:1160-1175) decides whether the checkpoint tensor it
was just handed is the FUSED all-experts stack or ONE expert's weight by looking at its rank::

    # Fused expert weights can be identified by their 3D tensors
    if loaded_weight.dim() == 3:
        if shard_id in {"w1", "w3"}:
            shard_idx = expert_id                      # expert_id repurposed as the w1/w3 half
            experts_shard = loaded_weight.chunk(2, dim=1)[shard_idx]
        else:
            experts_shard = loaded_weight
        start = 0
    else:
        experts_shard = loaded_weight.unsqueeze(0)     # one expert
        start = expert_id

That heuristic holds for an f16 checkpoint, where one expert's gate_proj is [I, H] (2-D) and the
fused stack is [E, 2I, H] (3-D). IT DOES NOT HOLD FOR A PANEL-ADDRESSED QUANTIZED CHECKPOINT: one
pxq4 expert's slab tensor is [panels, rows, bytes] -- e.g. [10, 80, 576] for gate_proj and
[40, 20, 832] for down_proj -- which is 3-D on its own. So a per-expert pxq checkpoint takes the
fused branch, and

  * for gate/up (w1/w3) it evaluates ``chunk(2, dim=1)[expert_id]``, where expert_id is the REAL
    expert index. For expert >= 2 that is IndexError: tuple index out of range -- the hard failure
    that killed every worker of w11 arm 3 at layer 0. For experts 0 and 1 it would have loaded
    HALF THE PANELS of the wrong axis, silently;
  * for down (w2) it would unbind the PANEL axis and load 40 "experts" starting at 0.

THE FIX. Take the fused branch only when the mapping entry that matched is a fused-checkpoint
entry. ``make_expert_params_mapping`` writes the expert index into the checkpoint fragment it
matches ("experts.<id>."); the fused mapping does not. So the presence of f"experts.{expert_id}."
in the fragment is an exact, cheap discriminator, and it changes nothing for a genuinely fused
checkpoint.

HOW. The upstream method is re-compiled from its own source with two textual edits (capture the
matched fragment, then AND it into the rank test), so everything else about the method -- and any
future change the fork makes to it -- is preserved verbatim. If either anchor is not found, the
patch declines and logs; it never half-applies. PXA_MOE_PEREXPERT_FIX=0 disables it.
"""
from __future__ import annotations

import inspect
import logging
import os

logger = logging.getLogger("pxq4_vllm.fused_moe_perexpert_fix")

MODE = os.environ.get("PXA_MOE_PEREXPERT_FIX", "auto").strip().lower()

_ANCHOR_FRAG = "                weight_name = qual_name.replace(weight_name, param_name)"
_ANCHOR_RANK = "                if loaded_weight.dim() == 3:"

_EDIT_FRAG = ("                _pxq_ckpt_frag = weight_name\n"
              "                weight_name = qual_name.replace(weight_name, param_name)")
# A per-expert mapping entry names the expert in the fragment it matched; the fused one does not.
_EDIT_RANK = ('                if loaded_weight.dim() == 3 and '
              'f"experts.{expert_id}." not in _pxq_ckpt_frag:')
# THIRD EDIT (main, 2026-09-07 06:40, the Flash-Next 'ductduct' root cause): the fork's load_weights
# dispatches every expert tensor through the LAYER's ``self.weight_loader``, whose model-weight case is
# ``if "weight" in weight_name`` and otherwise ``return False``. A pxq4 expert tensor is named
# ``*_pxq4_slabs`` / ``*_pxq4_anchor`` -- no "weight" in the name -- so every one of them was
# silently skipped (success False, nothing raised): 192 all-zero MoE params on every rank, every
# MoE layer contributing nothing, one token repeated forever. The pxq4 quant method attaches its own
# ``weight_loader`` to those params (moe.py set_weight_attrs), which the non-MoE loader honours and
# this path ignored. Prefer the parameter's own loader when it has one; the standard params carry
# the layer's bound method there, so nothing changes for them.
_ANCHOR_CALL = "                    success = self.weight_loader("
_EDIT_CALL = ('                    success = (getattr(param, "weight_loader", None) or self.weight_loader)(')


def _install() -> None:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    src = inspect.getsource(FusedMoE.load_weights)
    for anchor in (_ANCHOR_FRAG, _ANCHOR_RANK, _ANCHOR_CALL):
        if anchor not in src:
            logger.warning("fused_moe_perexpert_fix DECLINED: anchor not found in "
                           "FusedMoE.load_weights: %r", anchor.strip())
            return
    src = (src.replace(_ANCHOR_FRAG, _EDIT_FRAG, 1).replace(_ANCHOR_RANK, _EDIT_RANK, 1)
              .replace(_ANCHOR_CALL, _EDIT_CALL, 1))

    # "if True:" keeps the method's own 4-space body indentation legal at module level, so the
    # source is compiled EXACTLY as the fork wrote it apart from the two edits above.
    ns: dict = {}
    exec(compile("if True:\n" + src, "<pxq4 fused_moe_perexpert_fix>", "exec"),
         FusedMoE.load_weights.__globals__, ns)
    patched = ns["load_weights"]
    patched.__pxq4_patched__ = True
    FusedMoE.load_weights = patched
    logger.info("fused_moe_perexpert_fix installed: a 3-D tensor is treated as the FUSED stack "
                "only when the matched checkpoint fragment does not name one expert; expert tensors "
                "load through the parameter's own weight_loader when it has one")


if MODE not in ("0", "off", "false", "no", ""):
    try:
        _install()
    except Exception as exc:          # never be the reason a boot dies
        logger.warning("fused_moe_perexpert_fix could not install: %r", exc)
