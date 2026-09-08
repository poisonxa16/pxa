"""namemap.py — ggml tensor name -> HF/vLLM tensor name, and the per-policy PXQ4 allow-list.

SINGLE SOURCE OF TRUTH for plan §5.4. Every name below was checked against the AWQ twin's
``model.safetensors.index.json`` and safetensors header on the DGX this session, so the target
side is FACT, not inference: ``linear_attn.in_proj_qkv``/``in_proj_z``/``in_proj_a``/
``in_proj_b``/``out_proj``/``conv1d``/``A_log``/``dt_bias``/``norm``, ``self_attn.{q,k,v,o}_proj``
with ``q_proj`` at [12288, 5120], ``mlp.{gate,up,down}_proj``, ``model.language_model.*``,
``lm_head.weight`` at [248320, 5120] BF16, and 333 ``model.visual.*``.

THE PROJECTIONS ARE SPLIT ON DISK AND FUSED IN THE MODULE. The fork builds
``MergedColumnParallelLinear`` for ``in_proj_qkvz`` (output_sizes [2048,2048,6144,6144],
qwen3_5.py:212-230) and ``gate_up_proj``, but the checkpoint carries the parts separately and
vLLM's ``packed_modules_mapping`` re-fuses them at load. So the converter EMITS THEM
SEPARATELY and never pre-fuses; ``HF_MODULE_OF`` exists to tell you which fused module a
separate on-disk tensor will land in, which is what the §3.1 uniformity invariant is checked
against.

THE ONE HARD REQUIREMENT THAT IS NOT A STYLE CHOICE. ``_uses_split_gdn_input_projections``
(qwen3_5.py:127-157) decides fused-vs-split GDN input projection by scanning the quant
config's ``ignore``/``ignored_layers``/``modules_to_not_convert`` for ``linear_attn.in_proj_a``
or ``linear_attn.in_proj_b``. If it returns False the 48-row ``b``/``a`` fold into
``in_proj_qkvz``, giving 12 rows/rank at TP=4 — not a multiple of 64, and the packed shard
arithmetic TRUNCATES SILENTLY. So the emitted ``quantization_config.ignore`` must always
contain both names. ``BASE_IGNORE`` below enforces that; ``build_quantization_config``
re-asserts it.

THE GDN NAME MAP IS NOT A 1:1 RENAME. THE TWO CHECKPOINTS ORDER THE 48 VALUE-HEADS
DIFFERENTLY, and a converter that only renames produces a model that loads, runs, and emits
fluent garbage. ggml lays the 48-way v-head axis out REPEAT-MAJOR (``i = n_k_heads*r + k``,
which is what a broadcast expansion of the 16 k-heads gives); HF lays it out K-HEAD-MAJOR
(``j = R*k + r`` with ``R = n_v_heads // n_k_heads = 3``, which is what
``repeat_interleave`` gives). So every per-v-head axis needs the gather

    hf[j] = ggml[ n_k_heads * (j % R) + j // R ]                   (``v_head_gather``)

applied in units of one head (the 48-entry vectors) or one 128-wide head block (the weight
matrices). MEASURED on the real artifacts, exactly, not inferred — see ``GDN_PERM_SPEC``.
The 16-way q/k head axes are IDENTICAL in the two files and must NOT be touched.

WHAT IS MEASURED AND WHAT IS STILL ASSUMED. Three mappings used to be flagged here as
inference. Two of them are now measured facts (the ``ssm_beta``/``ssm_alpha`` pairing, and the
conv1d tap order) and the third turned out to be WRONG (``ssm_a`` is not log-space); each
entry below carries its evidence. Anything still unverified is flagged ASSUMPTION.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

HF_LM = "model.language_model"


def arch_kv(kv: dict[str, Any], *suffixes: str, default: Any = None) -> Any:
    """Look up a model KV by its arch-namespaced name without hardcoding the arch.

    ggml namespaces model KVs under ``general.architecture``: the dense 27B writes
    ``qwen35.ssm.group_count`` and its MoE siblings write ``qwen35moe.ssm.group_count``.
    Hardcoding either spelling makes the converter silently arch-specific -- and it fails
    LOUDLY rather than subtly only because ``gdn_geometry`` refuses to guess, which is how
    the qwen35moe artifacts were caught. Every model-namespaced KV read goes through here.
    """
    arch = kv.get("general.architecture")
    for suf in suffixes:
        if arch:
            k = f"{arch}.{suf}"
            if k in kv:
                return kv[k]
        if suf in kv:
            return kv[suf]
    return default

# ---------------------------------------------------------------------------------------------
# vLLM module suffixes (what get_quant_method sees as `prefix`), per policy. Plan §3, §5.5.
# ---------------------------------------------------------------------------------------------
#: Modules served by PXQ4 in P1 — exactly those that are already PXQ4 in the artifact AND are
#: uniformly PXQ4 across every output_partition_size of their fused vLLM module (§3.1).
PXQ4_MODULES_P1: frozenset[str] = frozenset({
    "mlp.gate_up_proj",            # <- ggml ffn_gate + ffn_up, both pxq4
    "mlp.down_proj",               # <- ggml ffn_down, pxq4
    "self_attn.o_proj",            # <- ggml attn_output, pxq4
    "linear_attn.in_proj_qkvz",    # <- ggml attn_qkv + attn_gate, both pxq4
})

#: P2a adds the GDN output projection. It is MXFP4 (ggml id 39) in the artifact, so it must be
#: RE-ENCODED to PXQ4 — this is the single largest decode-bandwidth lever that does not touch a
#: tensor class the backbone table deliberately protects.
PXQ4_MODULES_P2A = PXQ4_MODULES_P1 | {"linear_attn.out_proj"}

#: P2b WOULD add the LM head — and deliberately does NOT.
#:
#: ``lm_head`` IS NOT SERVABLE AS PXQ4 BY THE ENGINE SIDE OF THIS PROJECT, so the converter
#: must not encode it. The rule is owned by the quant-config component
#: (``pxq4_vllm.config.UNSERVABLE_PXQ4_LEAF_MODULES``), which now REJECTS a checkpoint whose
#: ``pxq4_modules`` names it, at ``from_config`` time. Two independent, verified blocks:
#:   1. ``VocabParallelEmbedding.__init__`` forces its own bespoke v1 vocab weight_loader
#:      (vocab_parallel_embedding.py:520-527, :633-682); ``PXQ4LinearMethod.create_weights``
#:      refuses any non-``weight_loader_v2`` loader when tp_size > 1, because v1 cannot slice
#:      the 64-row panel layout. At TP=4 that is a hard error.
#:   2. One vocab weight_loader call cannot fill the two parameters (``pxq4_slabs``,
#:      ``pxq4_anchor``) that one PXQ4 module owns.
#: Serving a 4-bit head needs a dedicated PXQ4 embedding method (plan §3, P3), not a flag.
#:
#: Emitting a PXQ4 ``lm_head`` anyway produced a checkpoint that could not be loaded at all:
#: the engine constructs the head as fp16 and then finds no ``lm_head.weight`` in the file.
#: So P2b is currently EMPTY of content — the same module set as P2a — and is BLOCKED at the
#: CLI rather than silently aliased. See ``BLOCKED_POLICIES``.
PXQ4_MODULES_P2B = frozenset(PXQ4_MODULES_P2A)

#: P2c makes ``self_attn.qkv_proj`` uniformly PXQ4 by re-encoding k/v (q8_0 in the artifact,
#: pinned there for quality by the backbone table). Only then may ``attn_q`` — which IS already
#: pxq4 on disk — finally be served, because the fused QKVParallelLinear admits no per-shard
#: dispatch (§3.1).
#:
#: ECONOMICS CORRECTION, and it matters: plan §0 scored p2c at 3.237 GiB/GPU/token (−8.7% vs
#: the incumbent AWQ's 3.547) ONLY because it counted a 4-bit LM head. With the head fp16,
#: ``bandwidth_report(build_plan(..., "p2c"), tp=4)`` — this package's own accounting, run
#: against the real artifact header — gives 3.616 GiB/GPU, of which the fp16 head is 0.592.
#: A PXQ4 head would be 0.592 × 4.254/16 = 0.157, i.e. the 0.435 GiB the plan booked as its
#: p2a→p2b delta. (The two accountings differ by ~1.7% on the same configuration; use this
#: one for this converter's output, and never mix them in a single comparison.)
#: PROJECTION, same method and assumptions as plan §0 — scale the incumbent's MEASURED 92.8
#: tok/s peak by the weight-bytes ratio, assuming our kernels sustain its effective HBM
#: bandwidth; NO GPU WAS RUN: 3.616 vs 3.547 is +1.9% bytes, so p2c as it now ships projects
#: to ≈91 tok/s. IT DOES NOT BEAT THE INCUMBENT. Nothing in this ladder does until a PXQ4
#: LM head is servable, which is why P2b is blocked rather than quietly dropped.
PXQ4_MODULES_P2C = PXQ4_MODULES_P2B | {"self_attn.qkv_proj"}

#: The MoE policy. Serves exactly what a qwen35moe PXQ4 artifact already carries as type 252,
#: and nothing that would break the §3.1 uniformity invariant.
#:
#: ``self_attn.qkv_proj`` is deliberately ABSENT even though ``attn_q`` is pxq4: its fused
#: partners ``attn_k``/``attn_v`` are q8_0 in the artifact, so the fused module is mixed-type
#: and must be served dense. This is the same call P1 makes on the dense sibling.
#:
#: ``linear_attn.out_proj`` is absent because ``ssm_out`` is MXFP4, not PXQ4 -- re-encoding it
#: is the P2A lever and is orthogonal to MoE support.
PXQ4_MODULES_M1: frozenset[str] = frozenset({
    "mlp.experts",                      # <- ggml ffn_{gate,up,down}_exps, all pxq4
    "mlp.shared_expert.gate_up_proj",   # <- ggml ffn_{gate,up}_shexp, both pxq4
    "mlp.shared_expert.down_proj",      # <- ggml ffn_down_shexp, pxq4
    "self_attn.o_proj",                 # <- ggml attn_output, pxq4
    "linear_attn.in_proj_qkvz",         # <- ggml attn_qkv + attn_gate, both pxq4
})

POLICY_MODULES: dict[str, frozenset[str]] = {
    "m1": PXQ4_MODULES_M1,
    "p1": PXQ4_MODULES_P1,
    "p2a": frozenset(PXQ4_MODULES_P2A),
    "p2b": frozenset(PXQ4_MODULES_P2B),
    "p2c": frozenset(PXQ4_MODULES_P2C),
}

#: Never removable, and never quietly demoted. See ``UNSERVABLE_PXQ4_MODULES``: the only thing
#: p2b added over p2a was the LM head, so with the head withdrawn p2b would emit a checkpoint
#: byte-identical to p2a while its name still promises plan §0's +3.3%. Refuse it by name
#: instead — a silent alias is how a benchmark gets reported against the wrong policy.
BLOCKED_POLICIES: dict[str, str] = {
    "p2b": (
        "policy p2b is p2a PLUS a PXQ4 lm_head, and the LM head is not servable by this "
        "build: pxq4_vllm.config.UNSERVABLE_PXQ4_LEAF_MODULES rejects it at engine start, "
        "because ParallelLMHead forces the v1 vocab weight_loader "
        "(vocab_parallel_embedding.py:520-527, :633-682) which cannot slice the 64-row panel "
        "layout at TP>1 and cannot fill both pxq4_slabs and pxq4_anchor. With the head "
        "withdrawn, p2b would emit a checkpoint IDENTICAL to p2a while its name still claims "
        "plan §0's +3.3%. Use --policy p2a. A real 4-bit head is plan §3 P3 work: it needs a "
        "PXQ4 embedding method with a vocab-sharded panel loader on the engine side FIRST."
    ),
}

#: Leaf module names the engine cannot serve as PXQ4, mirrored from
#: ``pxq4_vllm.config.UNSERVABLE_PXQ4_LEAF_MODULES``. The converter must never put these in
#: ``pxq4_modules`` and must always leave them in ``ignore``; ``build_quantization_config``
#: re-asserts it, and the config component rejects a file that violates it.
UNSERVABLE_PXQ4_MODULES: tuple[str, ...] = ("lm_head", "embed_tokens")


def assert_policy_supported(policy: str) -> None:
    """Refuse a policy that cannot produce a loadable checkpoint. Call before planning."""
    if policy in BLOCKED_POLICIES:
        raise SystemExit(BLOCKED_POLICIES[policy])
    bad = [m for m in POLICY_MODULES[policy] if m.rsplit(".", 1)[-1] in UNSERVABLE_PXQ4_MODULES]
    if bad:
        raise SystemExit(
            f"policy {policy} serves {bad} as PXQ4, which pxq4_vllm.config refuses to load "
            "(UNSERVABLE_PXQ4_LEAF_MODULES). This is a converter bug, not a config choice."
        )

#: Never removable. See the module docstring: dropping either turns the GDN input projection
#: fused and silently unshardable at TP=4.
BASE_IGNORE: tuple[str, ...] = (
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_ba",
)

#: Everything a policy does not serve as PXQ4 has to be declared ignored so
#: ``get_quant_method`` returns ``UnquantizedLinearMethod`` for it. ``model.visual`` covers the
#: 333 BF16 vision tensors copied verbatim from the AWQ twin.
_ALL_LINEAR_MODULES: tuple[str, ...] = (
    "mlp.experts",
    "mlp.shared_expert.gate_up_proj",
    "mlp.shared_expert.down_proj",
    "mlp.gate_up_proj", "mlp.down_proj", "self_attn.o_proj", "self_attn.qkv_proj",
    "linear_attn.in_proj_qkvz", "linear_attn.out_proj", "lm_head",
)


def ignore_list(policy: str) -> list[str]:
    served = POLICY_MODULES[policy]
    ig = list(BASE_IGNORE)
    ig += [m for m in _ALL_LINEAR_MODULES if m not in served]
    ig.append("model.visual")
    return ig


# ---------------------------------------------------------------------------------------------
# ggml -> HF name mapping
# ---------------------------------------------------------------------------------------------
_BLK = re.compile(r"^blk\.(\d+)\.(.+)$")

#: ``...mlp.experts.7.gate_proj`` -> group(1) == ``...mlp.experts``. Anchored on the numeric
#: expert index so a shared_expert (which has no index) is never swallowed by it.
_EXPERT_MOD = re.compile(r"^(.*\.experts)\.\d+\.(?:gate|up|down)_proj$")

#: GDN-block suffixes. ``in_proj_qkv`` and ``in_proj_z`` are separate on disk in BOTH
#: checkpoints (verified in the AWQ index), so the NAME side is a 1:1 rename — but the VALUES
#: are not: every per-v-head axis is reordered by ``GDN_PERM_SPEC`` and ``ssm_a`` additionally
#: takes a log. A bare rename here is the bug this table exists to prevent.
_GDN_MAP: dict[str, str] = {
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    # ne=(128,) — one entry per head_v_dim, NOT per head, so head order does not reach it.
    "ssm_norm.weight": "linear_attn.norm.weight",
    # MEASURED (was an ASSUMPTION): ssm_beta -> in_proj_b and ssm_alpha -> in_proj_a is the
    # right pairing. blk.0 q8_0 rows dequantized against the AWQ twin's BF16 (both are in the
    # AWQ ignore list, so unquantized there): under the v-head gather, alpha->in_proj_a gives
    # per-row rel 0.33 and beta->in_proj_b 0.34, while the two cross-pairings give 1.15-2.77
    # (~sqrt(2) = uncorrelated). The 0.33 residual is this tensor's own q8_0 noise: these rows
    # carry large outliers, so d=absmax/127 is big relative to a typical |w|~0.007.
    # Corroborates qwen3_5.py:265-272 (b = ba[..., :48], a = ba[..., 48:]).
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    # MEASURED, AND THE OLD ASSUMPTION WAS WRONG. ggml does NOT store A in log space: it
    # stores A itself, already negative. blk.0 ssm_a f32[48] vs the AWQ twin's A_log bf16[48]:
    #     log(-ggml[i]) == A_log[gather] to f32 round-trip (max |diff| < 5e-7)
    #     ggml[i]       == -exp(A_log[gather]) exactly
    # while identity ordering gives max |diff| 3.19 and the un-logged value gives 0.28. Emitted
    # as A_log = log(-ssm_a); see VALUE_TRANSFORMS. Shipping ssm_a under the name A_log would
    # have made every GDN decay -exp(A) instead of A — A~-0.04 becoming ~-0.96.
    "ssm_a": "linear_attn.A_log",
    # MEASURED: identical values, permuted head order. blk.0 vs the AWQ twin's dt_bias bf16[48]
    # is max |diff| 0.000000 under the gather and 22.5625 under identity.
    "ssm_dt.bias": "linear_attn.dt_bias",
    # MEASURED (was an ASSUMPTION), and it settles the tap order too. ggml ne=(4, 10240) ->
    # HF [10240, 1, 4] with the ggml row (the 4 conv taps) as the last axis. blk.0 f32 vs the
    # AWQ twin's BF16 conv1d, all 40960 values: EXACT (max |diff| 0.0) under taps-forward plus
    # the channel gather; 0.61 under identity, 0.75 under taps-reversed. So the kernel is NOT
    # reversed, and the 10240 = 2*key_dim + value_dim channel axis needs the v-head gather on
    # its trailing value_dim only.
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
}

#: Full-attention-block suffixes.
_ATTN_MAP: dict[str, str] = {
    # NO PERMUTATION. attn_q is 12288 rows = 2 * (24 heads * 256), gate-fused. ggml lays it out
    # as a per-head interleave [q_h(256) | gate_h(256)] — llama-build-context.cpp:2003-2007 views
    # Q with stride 2*row_size at offset 0 and the gate with the same view at offset row_size —
    # and the fork consumes exactly that: Qwen3NextAttention builds QKVParallelLinear with
    # total_num_heads = 24*(1+attn_output_gate) = 48 (qwen3_next.py:502-513) and its forward does
    # q_gate.view(..., num_heads, -1); torch.chunk(2, dim=-1) (qwen3_next.py:564-571). So a
    # contiguous 3072-row slice at TP=4 is 6 whole (q, gate) head pairs and the bytes transfer
    # as-is. config.json confirms attn_output_gate: true.
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}

#: Suffixes common to every block.
_COMMON_MAP: dict[str, str] = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

#: MoE block suffixes.
#:
#: Target names verified against the module tree the fork actually builds, not guessed:
#: ``Qwen3NextSparseMoeBlock`` (qwen3_next.py:319-418) constructs ``self.gate``
#: (ReplicatedLinear, quant_config=None), ``self.shared_expert_gate`` (ReplicatedLinear
#: hidden->1, quant_config=None), ``self.shared_expert`` (a Qwen3NextMLP with the usual
#: gate/up/down_proj) and ``self.experts`` (FusedMoE). qwen3_5.py:364 installs that block as
#: ``self.mlp``.
#:
#: The two gates are built with ``quant_config=None`` -- so they are ALWAYS unquantized
#: regardless of policy, which matches the artifact (both are f32 in the GGUF).
_MOE_MAP: dict[str, str] = {
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
}

#: Expert STACKS. These ggml tensors are 3-D (ne = (K, N, E)); each maps to E separate HF
#: tensors, one per expert. The ``{e}`` placeholder is filled by ``build_plan``.
#:
#: Per-expert naming is deliberate and is the reason this works with stock vLLM plumbing:
#: ``FusedMoE.make_expert_params_mapping`` (fused_moe/layer.py:1333-1376) builds
#: ``(param="experts.w13_", weight="experts.{e}.gate_proj.", expert_id, shard_id)`` and
#: qwen3_5.py:571 applies it as a pure ``name.replace(weight_name, param_name)``. That rewrite
#: is dtype- and shape-agnostic, so ``experts.7.gate_proj.pxq4_slabs`` becomes
#: ``experts.w13_pxq4_slabs`` with no model-side change at all. Emitting a pre-stacked
#: ``experts.gate_up_proj`` instead would take the ``is_fused_expert`` branch
#: (qwen3_5.py:583-590), which does ``loaded_weight.chunk(2, dim=-2)`` -- valid on a dense
#: [E, 2I, H] tensor and WRONG on a panel-major PXQ4 slab array, where dim -2 is the K-slab
#: axis, not the output-row axis.
_MOE_EXPERT_MAP: dict[str, str] = {
    "ffn_gate_exps.weight": "mlp.experts.{e}.gate_proj.weight",
    "ffn_up_exps.weight": "mlp.experts.{e}.up_proj.weight",
    "ffn_down_exps.weight": "mlp.experts.{e}.down_proj.weight",
}

#: ggml suffixes whose tensor is an expert stack (3-D, slowest axis = expert).
EXPERT_STACK_SUFFIXES: frozenset[str] = frozenset(_MOE_EXPERT_MAP)


def n_experts(kv: dict[str, Any]) -> int:
    """``expert_count`` from the GGUF, or 0 for a dense model."""
    return int(arch_kv(kv, "expert_count", default=0) or 0)


_GLOBAL_MAP: dict[str, str] = {
    "token_embd.weight": f"{HF_LM}.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": f"{HF_LM}.norm.weight",
}


def mtp_block_range(kv: dict[str, Any]) -> range:
    """ggml block indices that belong to the MTP (next-token-prediction) head.

    ``qwen35.block_count`` counts the MTP block (65 here) and ``qwen35.nextn_predict_layers``
    says how many of the trailing blocks are MTP (1 here), so the text stack is blocks
    [0, 64) and block 64 is MTP. Confirmed against the artifact: blk.64 is the only block
    carrying ``nextn.*`` tensors.
    """
    n_block = int(arch_kv(kv, "block_count", default=0))
    n_mtp = int(arch_kv(kv, "nextn_predict_layers", default=0))
    return range(n_block - n_mtp, n_block)


def GGML_TO_HF(name: str, kv: dict[str, Any]) -> str | None:
    """ggml tensor name -> HF tensor name, or None if the tensor is deliberately not emitted.

    Returns None for the MTP block (plan §3: ``mtp.*`` is P3, not emitted in P1/P2) — the fork
    ships ``qwen3_5_mtp.py`` and the AWQ twin carries ``mtp.*`` in a separate file, so adding
    it later is additive and does not invalidate a checkpoint made now.
    """
    if name in _GLOBAL_MAP:
        return _GLOBAL_MAP[name]

    m = _BLK.match(name)
    if m is None:
        return None
    layer, suffix = int(m.group(1)), m.group(2)

    if layer in mtp_block_range(kv):
        return None

    prefix = f"{HF_LM}.layers.{layer}"
    for table in (_COMMON_MAP, _ATTN_MAP, _GDN_MAP, _MOE_MAP, _MOE_EXPERT_MAP):
        if suffix in table:
            return f"{prefix}.{table[suffix]}"
    raise KeyError(f"namemap: no HF name for ggml tensor {name!r} (suffix {suffix!r}). "
                   f"Refusing to guess — an unmapped tensor is a silently incomplete model.")


# ---------------------------------------------------------------------------------------------
# GDN v-head order: the permutation, its geometry, and which axis of which tensor it applies to
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GdnGeometry:
    """The four numbers the v-head permutation needs, read from the GGUF's own KVs.

    Derived, not hardcoded, because the same converter has to survive a re-quantized artifact
    of a differently-shaped sibling; every field is cross-checked against real tensor shapes by
    ``check_against_tensors``.
    """
    n_k_heads: int          # qwen35.ssm.group_count        = 16
    n_v_heads: int          # qwen35.ssm.time_step_rank     = 48
    head_dim: int           # qwen35.ssm.state_size         = 128
    value_dim: int          # qwen35.ssm.inner_size         = 6144 = n_v_heads * head_dim

    @property
    def repeats(self) -> int:
        """How many v-heads share one k-head. 3 here."""
        return self.n_v_heads // self.n_k_heads

    @property
    def key_dim(self) -> int:
        return self.n_k_heads * self.head_dim

    @property
    def qkv_rows(self) -> int:
        """Rows of ``attn_qkv`` / channels of ``conv1d``: q + k + v."""
        return 2 * self.key_dim + self.value_dim

    def check_against_tensors(self, shapes: dict[str, tuple[int, ...]]) -> None:
        """``shapes`` maps a bare ggml suffix to its ne tuple, for one GDN block."""
        want = {
            "attn_gate.weight": self.value_dim,
            "attn_qkv.weight": self.qkv_rows,
            "ssm_conv1d.weight": self.qkv_rows,
        }
        for suf, n in want.items():
            ne = shapes.get(suf)
            if ne is not None and ne[-1] != n:
                raise SystemExit(
                    f"GDN geometry from the GGUF KVs (n_k_heads={self.n_k_heads}, "
                    f"n_v_heads={self.n_v_heads}, head_dim={self.head_dim}) says {suf} should "
                    f"have {n} on its slow axis, but the file says {ne[-1]}. Refusing to "
                    f"permute head blocks on geometry we cannot confirm.")
        ne = shapes.get("ssm_a")
        if ne is not None and ne[0] != self.n_v_heads:
            raise SystemExit(f"ssm_a is ne={ne}, expected ({self.n_v_heads},)")


def gdn_geometry(kv: dict[str, Any]) -> GdnGeometry:
    def need(suffix: str) -> int:
        v = arch_kv(kv, suffix)
        if v is None:
            raise SystemExit(
                f"the GGUF carries no {suffix!r} under arch "
                f"{kv.get('general.architecture')!r} or bare. The GDN v-head permutation "
                f"cannot be derived without it, and emitting an unpermuted GDN checkpoint is "
                f"the defect this refuses to reintroduce.")
        return int(v)
    n_k = need("ssm.group_count")
    n_v = need("ssm.time_step_rank")
    head = need("ssm.state_size")
    inner = need("ssm.inner_size")
    if n_k <= 0 or n_v <= 0 or n_v % n_k:
        raise SystemExit(f"GDN head counts are not a repeat structure: {n_v} v-heads over "
                         f"{n_k} k-heads")
    if n_v * head != inner:
        raise SystemExit(f"GDN geometry inconsistent: n_v_heads*state_size = {n_v * head} but "
                         f"inner_size = {inner}")
    return GdnGeometry(n_k_heads=n_k, n_v_heads=n_v, head_dim=head, value_dim=inner)


def v_head_gather(geom: GdnGeometry) -> list[int]:
    """``out[j] = in[v_head_gather[j]]`` for the 48-way value-head axis.

    ggml index ``i = n_k*r + k`` (repeat-major); HF index ``j = R*k + r`` (k-head-major).
    Inverting, ``i = n_k*(j % R) + j // R``. For 16 k-heads and 3 repeats this is
    ``[0, 16, 32, 1, 17, 33, ...]``, whose inverse is the ``[0,3,6,...,45,1,4,...]`` reported
    from the row-matching measurement.
    """
    R, n_k = geom.repeats, geom.n_k_heads
    return [n_k * (j % R) + j // R for j in range(geom.n_v_heads)]


#: ggml suffix -> (axis of the EMITTED tensor, offset-in-units, "unit" selector).
#: ``unit`` is ``"head_block"`` for a weight axis laid out as one 128-wide block per v-head,
#: and ``"head"`` for an axis with one scalar per v-head. ``offset`` is in elements and is only
#: non-zero for the q|k|v-stacked axes, whose leading 2*key_dim entries are the 16-way q and k
#: head axes — IDENTICAL in both checkpoints and deliberately left alone.
GDN_PERM_SPEC: dict[str, tuple[int, str, str]] = {
    "attn_qkv.weight":   (0, "qk_offset", "head_block"),
    "attn_gate.weight":  (0, "zero", "head_block"),
    "ssm_out.weight":    (1, "zero", "head_block"),
    "ssm_conv1d.weight": (0, "qk_offset", "head_block"),
    "ssm_alpha.weight":  (0, "zero", "head"),
    "ssm_beta.weight":   (0, "zero", "head"),
    "ssm_a":             (0, "zero", "head"),
    "ssm_dt.bias":       (0, "zero", "head"),
}

#: Every GDN suffix that is emitted. Anything in here but NOT in ``GDN_PERM_SPEC`` is asserting
#: "this tensor genuinely has no v-head axis"; ``_check_plan`` uses the difference to make the
#: omission deliberate rather than forgotten.
GDN_NO_PERM: frozenset[str] = frozenset({"ssm_norm.weight"})


def gdn_permutation(ggml_suffix: str, geom: GdnGeometry, axis_len: int
                    ) -> tuple[int, list[int]] | None:
    """(axis, element gather) for one GDN tensor, or None if it has no v-head axis.

    The returned gather is over the FULL axis, identity outside the v-head range, so callers
    can apply it with a single ``take`` and never reason about offsets again.
    """
    spec = GDN_PERM_SPEC.get(ggml_suffix)
    if spec is None:
        return None
    axis, off_kind, unit_kind = spec
    off = geom.key_dim * 2 if off_kind == "qk_offset" else 0
    unit = geom.head_dim if unit_kind == "head_block" else 1
    span = geom.n_v_heads * unit
    if axis_len != off + span:
        raise SystemExit(
            f"{ggml_suffix}: axis {axis} is {axis_len} long, but the GDN geometry says the "
            f"v-head range is [{off}, {off + span}). Refusing to permute a tensor whose shape "
            f"we do not understand.")
    g = v_head_gather(geom)
    out = list(range(off))
    for j in range(geom.n_v_heads):
        base = off + g[j] * unit
        out.extend(range(base, base + unit))
    return axis, out


#: ggml suffix -> (fn(float32 array) -> float32 array, description). Applied AFTER the
#: permutation, on the decoded values, for tensors whose ggml and HF forms differ in more than
#: element order. See the ``ssm_a`` entry in ``_GDN_MAP`` for the measurement.
def _a_to_a_log(w):
    import numpy as _np
    a = _np.asarray(w, dtype=_np.float32)
    if not _np.all(a < 0):
        raise SystemExit(
            "ssm_a holds a non-negative value, so it is not the A = -exp(A_log) this converter "
            "measured. Emitting log(-A) would produce NaN; refusing.")
    return _np.log(-a).astype(_np.float32)


def _minus_one(w):
    import numpy as _np
    return (_np.asarray(w, dtype=_np.float32) - 1.0).astype(_np.float32)


#: GEMMA-NORM CONVENTION (found 2026-08-19 after a day of mojibake): vLLM's
#: qwen3_5 model code binds GemmaRMSNorm (out = normed * (1 + weight)) for the
#: input/post layernorms, the attention q/k norms, and the final norm -- the HF
#: checkpoint stores those weights ZERO-CENTERED. The GGUF, converted for
#: llama.cpp's plain-multiply norms, stores them offset by +1. Copying the GGUF
#: values verbatim makes every one of those norms scale ~2x: activations stay
#: bounded (each later norm renormalizes), all per-layer math is self-
#: consistent, and the model babbles. The old p2a-nf artifact (Aug 18) has
#: zero-centered norms; the tree's converter had lost the subtraction.
#: ssm_norm (RMSNormGated) multiplies by the PLAIN weight and must NOT be
#: offset.
VALUE_TRANSFORMS: dict[str, tuple[Callable[[Any], Any], str]] = {
    "ssm_a": (_a_to_a_log, "A_log = log(-A): ggml stores A, HF stores its log"),
    "attn_norm.weight": (_minus_one, "gemma-norm: HF stores w-1, ggml stores w"),
    "post_attention_norm.weight": (_minus_one, "gemma-norm: HF stores w-1, ggml stores w"),
    "attn_q_norm.weight": (_minus_one, "gemma-norm: HF stores w-1, ggml stores w"),
    "attn_k_norm.weight": (_minus_one, "gemma-norm: HF stores w-1, ggml stores w"),
    "output_norm.weight": (_minus_one, "gemma-norm: HF stores w-1, ggml stores w"),
}


def ggml_suffix(ggml_name: str) -> str:
    """``blk.7.ssm_dt.bias`` -> ``ssm_dt.bias``; a global tensor keeps its own name."""
    m = _BLK.match(ggml_name)
    return m.group(2) if m else ggml_name


# ---------------------------------------------------------------------------------------------
# HF tensor -> owning vLLM module. Needed to enforce the §3.1 uniformity invariant, because a
# fused module's parts arrive as separate tensors and must all be the same type.
# ---------------------------------------------------------------------------------------------
_FUSE: tuple[tuple[str, str], ...] = (
    ("mlp.gate_proj", "mlp.gate_up_proj"),
    ("mlp.up_proj", "mlp.gate_up_proj"),
    # The shared expert is a plain Qwen3NextMLP, so vLLM fuses its gate/up exactly as it does
    # a dense mlp's (qwen3_next.py:380-388 -> Qwen3NextMLP builds a MergedColumnParallelLinear).
    ("mlp.shared_expert.gate_proj", "mlp.shared_expert.gate_up_proj"),
    ("mlp.shared_expert.up_proj", "mlp.shared_expert.gate_up_proj"),
    ("linear_attn.in_proj_qkv", "linear_attn.in_proj_qkvz"),
    ("linear_attn.in_proj_z", "linear_attn.in_proj_qkvz"),
    ("linear_attn.in_proj_b", "linear_attn.in_proj_ba"),
    ("linear_attn.in_proj_a", "linear_attn.in_proj_ba"),
    ("self_attn.q_proj", "self_attn.qkv_proj"),
    ("self_attn.k_proj", "self_attn.qkv_proj"),
    ("self_attn.v_proj", "self_attn.qkv_proj"),
)

#: Trailing components that are parameter names, not module names.
_PARAM_LEAVES = ("weight", "bias", "A_log", "dt_bias", "pxq4_slabs", "pxq4_anchor")


def HF_MODULE_OF(hf_name: str) -> str:
    """HF tensor name -> the vLLM module that will own it (fused where vLLM fuses).

    ``lm_head.weight`` -> ``lm_head``; ``...mlp.gate_proj.weight`` -> ``...mlp.gate_up_proj``;
    ``...linear_attn.A_log`` -> ``...linear_attn`` (a bare parameter, not a Linear).
    """
    mod = hf_name
    for leaf in _PARAM_LEAVES:
        if mod.endswith("." + leaf):
            mod = mod[: -(len(leaf) + 1)]
            break
    # An expert's projection is not its own vLLM module. All E*3 of them live inside ONE
    # FusedMoE built at prefix ``...mlp.experts`` (qwen3_next.py:402), and that prefix is what
    # get_quant_method is handed. Collapsing here is what makes the policy allow-list and the
    # runtime dispatch agree on a single key.
    m_exp = _EXPERT_MOD.match(mod)
    if m_exp:
        return m_exp.group(1)
    for src, dst in _FUSE:
        if mod == src or mod.endswith("." + src):
            return mod[: len(mod) - len(src)] + dst
    return mod


def module_suffix(module: str) -> str:
    """The suffix ``get_quant_method`` matches against ``pxq4_modules`` / ``ignore``.

    vLLM passes a full prefix like ``model.language_model.layers.7.mlp.gate_up_proj``; both our
    allow-list and the fork's own configs are written as trailing suffixes, so the match is
    ``prefix.endswith('.' + pattern)``. This returns the canonical two-component suffix so the
    converter can decide policy membership with the same key the runtime will use.
    """
    parts = module.split(".")
    if len(parts) >= 2 and not parts[-2].isdigit():
        return ".".join(parts[-2:])
    return parts[-1]


def is_pxq4_module(module: str, policy: str) -> bool:
    served = POLICY_MODULES[policy]
    return any(module == p or module.endswith("." + p) for p in served)
