# SPDX-License-Identifier: Apache-2.0
"""PXQ8HeadMethod -- an int8 per-row-scale LM head, served through ParallelLMHead.

WHY THE HEAD AND NOT A BODY MATRIX. On the 35B MoE the LM head is 970 MiB of a 2.78 GiB
per-token decode budget: the single largest item, and the one the PXQ tiers cannot touch. It
is not shaped like a body matrix -- one [V, H] tensor with V = 248,320 against H = 2048, read
in full once per token -- and the 64-row panel layout is exactly wrong for it, because a
panel's vocab axis is panels rather than rows and vLLM's vocab loader shards rows.

WHY THIS LOADS WITH NO CHANGE TO THE FORK, which contradicts an earlier note of ours that said
it could not. Read in vocab_parallel_embedding.py of the v1.5.0 fork:
  * ``VocabParallelEmbedding.__init__`` calls ``quant_config.get_quant_method(self, prefix)``
    (:880) -- a quant method may serve the head.
  * the ``embedding()`` requirement applies only when ``type(self) is VocabParallelEmbedding``
    (:887); ParallelLMHead is a subclass, so it does not apply.
  * ``weight_loader`` runs ONCE PER CHECKPOINT TENSOR, not once per module, so a method may
    register several parameters as long as each has its own key in the checkpoint.
  * a parameter with ``output_dim = 0`` is narrowed on the vocab axis and copied, with the
    padding rows zero-filled (:1080-1083); one with no ``output_dim`` is replicated.
The old note conflated "one checkpoint tensor cannot fill two parameters" (true, and the
reason a PXQ4 head is hard: slabs and anchor are one module's two halves) with "two checkpoint
tensors cannot fill two parameters" (false). int8 + per-row scale is two ordinary tensors.

THE FORMAT, matching pxq_q8.cuh exactly:
    <prefix>.weight        int8    [V, H]     row-major
    <prefix>.weight_scale  float16 [V, 1]     scale[n] = absmax(row n) / 127
    w[n, k] = float(q[n, k]) * float(scale[n])
Per-row and not per-tensor because the vocab axis is the axis the loader shards, so a per-row
scale shards for free -- and because one scale across 248,320 rows is set by the largest row
in the vocabulary and flattens every other row into a handful of levels.

THE KERNEL IS NOT OPTIONAL. At decode this is a GEMV, and the whole point is bytes:
    fp16 head                       1017 MB/token
    int8 + the fused GEMV            509 MB/token
    int8 dequantised then cuBLAS     509 + 1017 + 1017 = 2543 MB/token
So an int8 head served by dequant-then-matmul is a 2.5x REGRESSION. ``apply`` therefore
requires ``torch.ops.pxq4.q8_linear_out`` and refuses to run without it rather than silently
taking the slow path -- a quiet 2.5x is worse than a loud failure.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class PXQ8HeadMethod(QuantizeMethodBase):
    """int8 weights + per-row fp16 scales for a ParallelLMHead."""

    def __init__(self, quant_config=None) -> None:
        self.quant_config = quant_config

    # ------------------------------------------------------------------ weights
    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int, output_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs) -> None:
        # VocabParallelEmbedding calls this as
        #   create_weights(self, embedding_dim, [num_embeddings_per_partition],
        #                  embedding_dim, num_embeddings_padded, ...)
        # so input_size_per_partition is H and output_partition_sizes[0] is this rank's rows.
        H = int(input_size_per_partition)
        rows = int(sum(int(o) for o in output_partition_sizes))
        if rows <= 0 or H <= 0:
            raise ValueError(f"pxq_q8 head: bad geometry rows={rows} H={H}")

        if not (hasattr(torch.ops, "pxq4") and hasattr(torch.ops.pxq4, "q8_linear_out")):
            raise RuntimeError(
                "pxq_q8 head: this checkpoint serves lm_head as int8, but the loaded kernel "
                "library has no torch.ops.pxq4.q8_linear_out. Serving it without that kernel "
                "would dequantise the whole head per forward and read 2.5x the bytes of the "
                "fp16 head it replaces, so this is refused rather than silently slower. Point "
                "PXQ4_LIB at a library that carries the q8 ops.")

        weight = torch.nn.Parameter(
            torch.empty(rows, H, dtype=torch.int8, device=torch.cuda.current_device()),
            requires_grad=False)
        # output_dim=0 is what makes the vocab loader narrow this on the vocab axis. It is the
        # whole sharding contract for this method; without it the loader replicates the head.
        set_weight_attrs(weight, {"output_dim": 0, "input_dim": 1, **extra_weight_attrs})
        layer.register_parameter("weight", weight)

        # [rows, 1] rather than [rows]: the trailing axis gives the loader an output_dim to
        # narrow while keeping one scale per row, and the kernel only ever indexes by row.
        scale = torch.nn.Parameter(
            torch.empty(rows, 1, dtype=torch.float16, device=torch.cuda.current_device()),
            requires_grad=False)
        set_weight_attrs(scale, {"output_dim": 0, **extra_weight_attrs})
        layer.register_parameter("weight_scale", scale)

        layer.pxq_q8_rows = rows
        layer.pxq_q8_H = H

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The kernel reads whole rows as char4 and indexes the scale by row, so both have to be
        # contiguous and the scale has to be flat. Asserted rather than coerced: a silent
        # .contiguous() here would hide a loader that produced a strided view.
        w, s = layer.weight, layer.weight_scale
        if not w.is_contiguous() or not s.is_contiguous():
            raise RuntimeError("pxq_q8 head: weight/scale are not contiguous after loading")
        if s.numel() != w.shape[0]:
            raise RuntimeError(f"pxq_q8 head: {s.numel()} scales for {w.shape[0]} rows")
        layer.weight_scale_flat = s.reshape(-1)

    # ------------------------------------------------------------------ compute
    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        if x2.dtype != torch.float16:
            x2 = x2.to(torch.float16)
        rows = int(layer.weight.shape[0])
        out = torch.empty((x2.shape[0], rows), dtype=torch.float16, device=x2.device)
        scale = getattr(layer, "weight_scale_flat", None)
        if scale is None:
            scale = layer.weight_scale.reshape(-1)
        # Routes on M inside the op: the GEMV for decode-shaped batches, dequant + cuBLAS above
        # the threshold where the dequant amortises.
        torch.ops.pxq4.q8_linear_out(out, x2, layer.weight, scale)
        if bias is not None:
            out = out + bias
        return out.reshape(*x.shape[:-1], rows)
