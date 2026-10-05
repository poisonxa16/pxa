"""qwen4exp_pp_fix.py -- make Qwen4Exp loadable at pipeline-parallel-size > 1.

THE FORK DEFECT: the hyper-connection mixer is None under pipeline parallelism.
``vllm/models/qwen4_exp/nvidia/model.py:489-505`` builds the final hyper-connection mixer
only on the last pipeline rank and writes the literal ``None`` on every other rank::

    self.hyper_connection_mixer: GatedResidual | None
    if get_pp_group().is_last_rank:
        self.hyper_connection_mixer = GatedResidual(...)
    else:
        self.hyper_connection_mixer = None

``None`` is not an ``nn.Module``, so it appears in neither ``named_children()`` nor
``named_parameters()``.  Every checkpoint of this architecture carries the mixer's three
tensors (``model.language_model.hyper_connection_mixer.{hc_norm,input_mix_weight_down,
input_mix_weight_up}.weight``), so on PP0/PP1/PP2 ``AutoWeightsLoader._load_module`` falls
through to its final ``else`` and raises

    ValueError: There is no module or parameter named 'hyper_connection_mixer' in
    Qwen4ExpModel.

That is the whole boot failure: Qwen4Exp cannot load at ANY pipeline-parallel-size > 1 as
shipped, for any checkpoint.  It is a fork defect, not a checkpoint defect.

THE FIX.  vLLM's own convention for "this rank does not own this module" is
``PPMissingLayer()`` -- ``models/utils.py:288-290`` returns from ``_load_module``
immediately for one, which drops the module's weights on the ranks that do not own it.
So the one-line class of fix in the fork is ``PPMissingLayer()`` instead of ``None``.
Rather than rebuild a 27 GB image for one line, this sidecar performs the same repair at
runtime, after the model is constructed and before its weights are loaded.

IT IS SAFE.  The only read of the attribute is ``model.py:604``
(``final_mixer = self.hyper_connection_mixer; assert final_mixer is not None``), which sits
after ``if not get_pp_group().is_last_rank: return`` at ``:593`` -- i.e. it runs only on the
last rank, where the attribute is a real ``GatedResidual`` and this module never touches it.
Only the literal ``None`` is replaced, and only for the attribute names in the repair list.

ENV.
  PXA_QWEN4EXP_PP_FIX   auto (default, repair and log) | 0/off (do nothing) | fail
                        ("fail" additionally raises if PP>1 and nothing was repaired, i.e.
                        the fork has been fixed upstream or the hook missed).
  PXA_PP_MISSING_ATTRS  comma-separated attribute names to repair
                        (default: hyper_connection_mixer).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("pxq4_vllm.qwen4exp_pp_fix")

MODE = os.environ.get("PXA_QWEN4EXP_PP_FIX", "auto").strip().lower()
ATTRS = tuple(
    a.strip()
    for a in os.environ.get("PXA_PP_MISSING_ATTRS", "hyper_connection_mixer").split(",")
    if a.strip()
)

_MISS = object()


def repair(model) -> list[str]:
    """Replace ``None`` with ``PPMissingLayer()`` for every name in ATTRS. Returns the
    qualified names repaired, in ``named_modules`` order."""
    from vllm.model_executor.models.utils import PPMissingLayer

    done: list[str] = []
    for qual, mod in list(model.named_modules()):
        for attr in ATTRS:
            # A plain ``self.x = None`` lands in the instance __dict__ (nn.Module only
            # routes it to _modules if the name is ALREADY a module slot), so check both.
            val = mod.__dict__.get(attr, _MISS)
            if val is _MISS:
                val = getattr(mod, "_modules", {}).get(attr, _MISS)
            if val is not None:          # a real module, or the attribute does not exist
                continue
            # nn.Module.__setattr__ removes the stale __dict__ entry and registers the
            # PPMissingLayer as a child, which is what makes _load_module skip it.
            setattr(mod, attr, PPMissingLayer())
            name = f"{qual}.{attr}" if qual else attr
            done.append(name)
            logger.info("qwen4exp_pp_fix: repaired %s -> PPMissingLayer "
                        "(was the literal None on a non-last pipeline rank)", name)
    return done


def _pp_size() -> int:
    try:
        from vllm.distributed import get_pp_group
        return get_pp_group().world_size
    except Exception:
        return -1


def _install() -> None:
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    orig_load = DefaultModelLoader.load_weights

    def load_weights(self, model, model_config):
        try:
            done = repair(model)
            pp = _pp_size()
            logger.info("qwen4exp_pp_fix: %d module(s) repaired before weight load "
                        "(pipeline_parallel_size=%s, attrs=%s)", len(done), pp,
                        ",".join(ATTRS))
            if MODE == "fail" and pp > 1 and not done:
                raise RuntimeError(
                    "PXA_QWEN4EXP_PP_FIX=fail: pipeline_parallel_size=%d and NOTHING was "
                    "repaired. Either the fork no longer writes None (drop this sidecar) "
                    "or the attribute was renamed and the load below will raise." % pp)
        except RuntimeError:
            raise
        except Exception:            # the repair must never be the reason a boot dies
            logger.exception("qwen4exp_pp_fix: repair pass failed; loading unrepaired")
        return orig_load(self, model, model_config)

    DefaultModelLoader.load_weights = load_weights
    logger.info("qwen4exp_pp_fix installed (mode=%s, attrs=%s)", MODE, ",".join(ATTRS))


if MODE not in ("0", "off", "false", "no", ""):
    try:
        _install()
    except Exception as exc:         # pragma: no cover
        logger.warning("qwen4exp_pp_fix could not install: %r", exc)
