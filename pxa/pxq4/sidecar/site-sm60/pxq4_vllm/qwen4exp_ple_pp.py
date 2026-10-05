"""qwen4exp_ple_pp.py -- run the Qwen4Exp n-gram PLE at pipeline_parallel_size > 1.

THE FORK WALL (blocker #1595).  ``vllm/models/qwen4_exp/nvidia/model_state.py:34-40``::

    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        raise RuntimeError(
            "N-gram PLE embedding currently requires "
            "pipeline_parallel_size=1 because non-first pipeline ranks do "
            "not receive the raw input_ids required by PLE. Please run "
            "with PP=1.")

Every rank raises it, so Qwen4Exp cannot serve at PP>1 at all.  On the four P100s that is
the whole ballgame: TP=4 cuts the 640-wide expert intermediate (10 pxq panels of 64) mid
panel, and TP=2 is 32 GB against a 46.5 GB checkpoint, so PP is the only legal 4-card shape.

WHY THE WALL IS WRONG.  The premise ("non-first ranks do not receive the raw input_ids") is
false for the V2 runner.  ``vllm/v1/worker/gpu/model_runner.py:prepare_inputs`` fills
``self.input_buffers.input_ids`` from ``req_states.all_token_ids`` on EVERY rank -- the code
is not rank-gated, because the scheduler output reaches every worker.  The runner then
throws that away at :1373-1376 (``if not self.is_first_pp_rank: model_inputs["input_ids"] =
None``).  The n-gram context itself is a pure function of the sequence's token ids
(``Qwen4ExpModelState._prepare_ngram_context`` reads only ``req_states``), so it is
computable, unchanged, on any rank.

WHAT THIS MODULE DOES.

  1. Lifts the assertion WITHOUT re-implementing what follows it.  The raise sits between
     the ``uses_ngram_embedding`` early return and the buffer setup, i.e. everything the
     fork does after the raise is the PLE state this rank needs.  Rather than hand-copy
     those statements (which would drift from the fork), we let the original ``__init__``
     run and raise, catch exactly that RuntimeError, and execute THE FORK'S OWN remaining
     statements, extracted from its source by AST at install time and compiled once.  The
     PLE model state built at PP=4 is therefore identical, statement for statement, to the
     one PP=1 builds -- there is no second implementation to keep in sync.  If the source
     does not have the expected shape (raise, then a tail), nothing is installed and the
     boot fails with the fork's own error instead of a half-initialized state.

  2. Relays the token ids to a PLE layer that lands on a non-first rank.  For a checkpoint
     whose ``ple_layer_ids`` are all in the first rank's layer range this never fires (with
     ple_layer_ids=[2] -> layer_idx 1, the PLE stack is on PP0 = the first rank, which gets
     input_ids from the runner as usual, so the PLE input path is bit-identical to PP=1).
     For any other split, ``Qwen4ExpModelState.prepare_inputs`` publishes the input-ids
     tensor it was given (the runner's own address-stable buffer, before the runner nulls
     the kwarg) and ``Qwen4ExpModel.forward`` puts it back when this rank owns a PLE layer.
     Only the PLE branch of ``Qwen4ExpDecoderLayer.forward`` reads it; the embedding branch
     is guarded by ``get_pp_group().is_first_rank`` and cannot see it.

ENV.
  PXA_QWEN4EXP_PLE_PP   auto (default: install, log what each rank owns)
                        0/off/false/no  do nothing
                        strict          additionally raise at install time if the fork
                                        source no longer matches (rather than logging)
"""
from __future__ import annotations

import ast
import inspect
import logging
import os
import textwrap

logger = logging.getLogger("pxq4_vllm.qwen4exp_ple_pp")

MODE = os.environ.get("PXA_QWEN4EXP_PLE_PP", "auto").strip().lower()

#: the substring that identifies the wall's own RuntimeError, so we never swallow another one
_WALL = "pipeline_parallel_size=1"

#: set by prepare_inputs, read by Qwen4ExpModel.forward on a non-first rank that owns a PLE
#: layer. One runner per process, so a module global is the whole story.
_LAST_INPUT_IDS = None


def _build_tail(cls):
    """Compile the fork's own post-assertion ``__init__`` statements into ``f(self)``.

    Returns ``(fn, n_statements)`` or raises ValueError if the source is not the shape this
    module was written against.
    """
    src = textwrap.dedent(inspect.getsource(cls.__init__))
    tree = ast.parse(src)
    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef) or fn.name != "__init__":
        raise ValueError("Qwen4ExpModelState.__init__ is not a plain function def")

    # the `config = self.model_config.hf_text_config` binding the tail statements use
    config_assign = None
    raise_idx = None
    for i, node in enumerate(fn.body):
        if (
            config_assign is None
            and isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "config"
        ):
            config_assign = node
        if (
            isinstance(node, ast.If)
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Raise)
            and _WALL in ast.dump(node.body[0])
        ):
            raise_idx = i
    if config_assign is None:
        raise ValueError("no `config = ...` binding found in __init__")
    if raise_idx is None:
        raise ValueError(f"no `if pp > 1: raise ...{_WALL}...` found in __init__")
    tail = fn.body[raise_idx + 1 :]
    if not tail:
        raise ValueError("the assertion is the last statement; there is no PLE state to build")

    new = ast.FunctionDef(
        name="_pxa_ple_state_tail",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self")],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[config_assign, *tail],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    mod = ast.Module(body=[new], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict = {}
    # the fork module's globals: torch and everything else the tail statements name
    glb = dict(inspect.getmodule(cls).__dict__)
    exec(compile(mod, "<qwen4exp_ple_pp tail>", "exec"), glb, ns)  # noqa: S102
    return ns["_pxa_ple_state_tail"], len(tail)


def _pp() -> tuple[int, int]:
    """(rank_in_pipeline, pipeline_world_size); (-1, -1) when the group is not up."""
    try:
        from vllm.distributed import get_pp_group

        g = get_pp_group()
        return g.rank_in_group, g.world_size
    except Exception:
        return -1, -1


def _owns_ple(model) -> list[str]:
    """The decoder layers on THIS rank that actually built a PLE stack."""
    return sorted(
        name.rsplit(".ple", 1)[0]
        for name, mod in model.named_modules()
        if name.endswith(".ple") and mod is not None
    )


def _install_model_state() -> None:
    from vllm.models.qwen4_exp.nvidia import model_state as ms

    cls = ms.Qwen4ExpModelState
    if getattr(cls.__init__, "_pxa_ple_pp", False):
        return
    tail, n_tail = _build_tail(cls)
    orig = cls.__init__

    def __init__(self, vllm_config, model, encoder_cache, device):
        try:
            return orig(self, vllm_config, model, encoder_cache, device)
        except RuntimeError as exc:
            if _WALL not in str(exc):
                raise
            # The raise sits after super().__init__ and after uses_ngram_embedding, so the
            # object is fully built except for the PLE buffers -- which is exactly the tail.
            tail(self)
            rank, world = _pp()
            mine = _owns_ple(model)
            logger.info(
                "qwen4exp_ple_pp: PP=%s wall lifted on pp_rank=%s/%s; ran the fork's own %d "
                "post-assertion statements (ngram_context_len=%s eos=%s); PLE stack(s) on "
                "this rank: %s",
                getattr(vllm_config.parallel_config, "pipeline_parallel_size", "?"),
                rank, world, n_tail,
                getattr(self, "ngram_context_len", "?"),
                getattr(self, "ngram_eos_token_id", "?"),
                mine or "none (this rank only forwards the PLE kwargs)",
            )
            return None

    __init__._pxa_ple_pp = True
    cls.__init__ = __init__

    # publish the input-ids tensor for a PLE layer that is NOT on the first rank
    orig_prepare = cls.prepare_inputs

    def prepare_inputs(self, input_batch, req_states):
        global _LAST_INPUT_IDS
        _LAST_INPUT_IDS = getattr(input_batch, "input_ids", None)
        return orig_prepare(self, input_batch, req_states)

    prepare_inputs._pxa_ple_pp = True
    cls.prepare_inputs = prepare_inputs
    logger.info(
        "qwen4exp_ple_pp installed on Qwen4ExpModelState (mode=%s, tail=%d statements)",
        MODE, n_tail,
    )


def _install_model_forward() -> None:
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpModel

    if getattr(Qwen4ExpModel.forward, "_pxa_ple_pp", False):
        return
    orig = Qwen4ExpModel.forward

    def forward(self, input_ids, positions, *args, **kwargs):
        if input_ids is None:
            armed = getattr(self, "_pxa_ple_relay", None)
            if armed is None:
                from vllm.distributed import get_pp_group

                armed = bool(_owns_ple(self)) and not get_pp_group().is_first_rank
                self._pxa_ple_relay = armed
                if armed:
                    logger.info(
                        "qwen4exp_ple_pp: this rank owns a PLE stack and is not the first "
                        "pipeline rank -- relaying the runner's token ids into the PLE."
                    )
            if armed and _LAST_INPUT_IDS is not None:
                ids = _LAST_INPUT_IDS
                want = positions.shape[-1]
                if ids.shape[0] >= want:
                    input_ids = ids[:want]
        return orig(self, input_ids, positions, *args, **kwargs)

    forward._pxa_ple_pp = True
    Qwen4ExpModel.forward = forward


def _install() -> None:
    _install_model_state()
    _install_model_forward()


def _install_deferred() -> None:
    """Patch when the model loader runs, not at interpreter start.

    ``vllm.models.qwen4_exp`` pulls torch and the whole fork model package; importing it
    from sitecustomize would both slow every process down and risk a circular import.  The
    model state is constructed in ``model_runner.load_model`` AFTER the weights are loaded,
    so the loader hook (the same one qwen4exp_pp_fix uses) is always early enough.
    """
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    if getattr(DefaultModelLoader.load_weights, "_pxa_ple_pp", False):
        return
    orig_load = DefaultModelLoader.load_weights

    def load_weights(self, model, model_config):
        try:
            _install()
        except Exception as exc:
            if MODE == "strict":
                raise
            logger.warning(
                "qwen4exp_ple_pp could NOT install (%r) -- Qwen4Exp will refuse PP>1 exactly "
                "as the fork ships it. Nothing was half-patched.", exc,
            )
        return orig_load(self, model, model_config)

    load_weights._pxa_ple_pp = True
    DefaultModelLoader.load_weights = load_weights
    logger.info("qwen4exp_ple_pp armed (mode=%s); patches land at model load.", MODE)


if MODE not in ("0", "off", "false", "no", ""):
    try:
        _install_deferred()
    except Exception as exc:
        if MODE == "strict":
            raise
        logger.warning("qwen4exp_ple_pp could not arm: %r", exc)
