#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import argparse
import contextlib
import json
import os
import re
import sys
from enum import IntEnum
from pathlib import Path
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Iterable, Iterator, Literal, Sequence, TypeVar, cast
from itertools import chain

import math
import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

logger = logging.getLogger("hf-to-gguf")


###### MODEL DEFINITIONS ######

class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


AnyModel = TypeVar("AnyModel", bound="type[Model]")


class Model:
    _model_classes: dict[str, type[Model]] = {}

    dir_model: Path
    ftype: gguf.LlamaFileType
    fname_out: Path
    is_big_endian: bool
    endianess: gguf.GGUFEndian
    use_temp_file: bool
    lazy: bool
    part_names: list[str]
    is_safetensors: bool
    hparams: dict[str, Any]
    block_count: int
    tensor_map: gguf.TensorNameMap
    tensor_names: set[str] | None
    gguf_writer: gguf.GGUFWriter
    model_name: str | None
    metadata_override: Path | None
    dir_model_card: Path

    # subclasses should define this!
    model_arch: gguf.MODEL_ARCH

    def __init__(self, dir_model: Path, ftype: gguf.LlamaFileType, fname_out: Path, is_big_endian: bool = False,
                 use_temp_file: bool = False, eager: bool = False,
                 metadata_override: Path | None = None, model_name: str | None = None,
                 split_max_tensors: int = 0, split_max_size: int = 0, dry_run: bool = False, small_first_shard: bool = False):
        if type(self) is Model:
            raise TypeError(f"{type(self).__name__!r} should not be directly instantiated")

        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.use_temp_file = use_temp_file
        self.lazy = not eager
        self.part_names = Model.get_model_part_names(self.dir_model, "model", ".safetensors")
        self.is_safetensors = len(self.part_names) > 0
        if not self.is_safetensors:
            self.part_names = Model.get_model_part_names(self.dir_model, "pytorch_model", ".bin")
        self.hparams = Model.load_hparams(self.dir_model)
        self.block_count = self.find_hparam(["n_layers", "num_hidden_layers", "n_layer", "num_layers"])
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        self.tensor_names = None
        self.metadata_override = metadata_override
        self.model_name = model_name
        self.dir_model_card = dir_model  # overridden in convert_lora_to_gguf.py

        # Apply heuristics to figure out typical tensor encoding based on first layer tensor encoding type
        if self.ftype == gguf.LlamaFileType.GUESSED:
            # NOTE: can't use field "torch_dtype" in config.json, because some finetunes lie.
            _, first_tensor = next(self.get_tensors())
            if first_tensor.dtype == torch.float16:
                logger.info(f"choosing --outtype f16 from first tensor type ({first_tensor.dtype})")
                self.ftype = gguf.LlamaFileType.MOSTLY_F16
            else:
                logger.info(f"choosing --outtype bf16 from first tensor type ({first_tensor.dtype})")
                self.ftype = gguf.LlamaFileType.MOSTLY_BF16

        # Configure GGUF Writer
        self.gguf_writer = gguf.GGUFWriter(path=None, arch=gguf.MODEL_ARCH_NAMES[self.model_arch], endianess=self.endianess, use_temp_file=self.use_temp_file,
                                           split_max_tensors=split_max_tensors, split_max_size=split_max_size, dry_run=dry_run, small_first_shard=small_first_shard)

    @classmethod
    def __init_subclass__(cls):
        # can't use an abstract property, because overriding it without type errors
        # would require using decorated functions instead of simply defining the property
        if "model_arch" not in cls.__dict__:
            raise TypeError(f"Missing property 'model_arch' for {cls.__name__!r}")

    def find_hparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def set_vocab(self):
        self._set_vocab_gpt2()

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        tensor_names_from_parts: set[str] = set()

        if len(self.part_names) > 1:
            self.tensor_names = set()
            index_name = "model.safetensors" if self.is_safetensors else "pytorch_model.bin"
            index_name += ".index.json"
            logger.info(f"gguf: loading model weight map from '{index_name}'")
            with open(self.dir_model / index_name, "r", encoding="utf-8") as f:
                index: dict[str, Any] = json.load(f)
                weight_map = index.get("weight_map")
                if weight_map is None or not isinstance(weight_map, dict):
                    raise ValueError(f"Can't load 'weight_map' from {index_name!r}")
                self.tensor_names.update(weight_map.keys())
        else:
            self.tensor_names = tensor_names_from_parts

        for part_name in self.part_names:
            logger.info(f"gguf: loading model part '{part_name}'")
            ctx: ContextManager[Any]
            if self.is_safetensors:
                from safetensors import safe_open
                ctx = cast(ContextManager[Any], safe_open(self.dir_model / part_name, framework="pt", device="cpu"))
            else:
                ctx = contextlib.nullcontext(torch.load(str(self.dir_model / part_name), map_location="cpu", mmap=True, weights_only=True))

            with ctx as model_part:
                tensor_names_from_parts.update(model_part.keys())

                for name in model_part.keys():
                    if self.is_safetensors:
                        if self.lazy:
                            data = model_part.get_slice(name)
                            data = LazyTorchTensor.from_safetensors_slice(data)
                        else:
                            data = model_part.get_tensor(name)
                    else:
                        data = model_part[name]
                        if self.lazy:
                            data = LazyTorchTensor.from_eager(data)
                    yield name, data

        # only verify tensor name presence; it doesn't matter if they are not in the right files
        if len(sym_diff := tensor_names_from_parts.symmetric_difference(self.tensor_names)) > 0:
            raise ValueError(f"Mismatch between weight map and model parts for tensor names: {sym_diff}")

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Hook for subclasses to drop a source tensor while the index is built."""
        return item

    def index_tensors(self) -> dict[str, Callable[[], Tensor]]:
        """Build a `name -> loader` index over ALL shards, up front.

        Adapted from llama.cpp `conversion/base.py::ModelBase.index_tensors`
        @ upstream commit 82dbc4f01 (MIT, Copyright (c) 2023-2026 The ggml authors).

        get_tensors() is a streaming generator, and a generator cannot pair a
        weight with a sidecar scale tensor that lives in a different shard. Any
        converter for a quantized source format (fp8 block-scale, MXFP4
        nibble+scale) needs the index instead. Opt-in: nothing calls this unless
        a model class does.

        The values are FACTORIES, not tensors: LazyBase memoises its result in
        `_data`, so handing out one shared lazy object per name would pin every
        materialised tensor in RAM for the whole run.
        """
        tensors: dict[str, Callable[[], Tensor]] = {}
        tensor_names_from_parts: set[str] = set()
        tensor_names_from_index: set[str] = set()
        self._open_parts: list[Any] = []

        if len(self.part_names) > 1:
            index_name = "model.safetensors" if self.is_safetensors else "pytorch_model.bin"
            index_name += ".index.json"
            logger.info(f"gguf: indexing model weight map from '{index_name}'")
            with open(self.dir_model / index_name, "r", encoding="utf-8") as f:
                index: dict[str, Any] = json.load(f)
                weight_map = index.get("weight_map")
                if weight_map is None or not isinstance(weight_map, dict):
                    raise ValueError(f"Can't load 'weight_map' from {index_name!r}")
                tensor_names_from_index.update(weight_map.keys())

        for part_name in self.part_names:
            logger.info(f"gguf: indexing model part '{part_name}'")
            if self.is_safetensors:
                from safetensors import safe_open
                model_part = safe_open(self.dir_model / part_name, framework="pt", device="cpu")
            else:
                model_part = torch.load(str(self.dir_model / part_name), map_location="cpu", mmap=True, weights_only=True)
            # the loaders below read through this handle long after the loop ends
            self._open_parts.append(model_part)

            for name in model_part.keys():
                tensor_names_from_parts.add(name)
                if self.is_safetensors:
                    data_gen = lambda p=model_part, n=name: LazyTorchTensor.from_safetensors_slice(p.get_slice(n))  # noqa: E731
                else:
                    data_gen = lambda p=model_part, n=name: LazyTorchTensor.from_eager(p[n])  # noqa: E731
                item = self.filter_tensors((name, data_gen))
                if item is None:
                    continue
                tensors[item[0]] = item[1]

        self.tensor_names = tensor_names_from_index or tensor_names_from_parts
        if tensor_names_from_index:
            sym_diff = tensor_names_from_parts.symmetric_difference(tensor_names_from_index)
            if len(sym_diff) > 0:
                raise ValueError(f"Mismatch between weight map and model parts for tensor names: {sym_diff}")

        return tensors

    def format_tensor_name(self, key: gguf.MODEL_TENSOR, bid: int | None = None, suffix: str = ".weight") -> str:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            raise ValueError(f"Missing {key!r} for MODEL_TENSORS of {self.model_arch!r}")
        name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in name:
            assert bid is not None
            name = name.format(bid=bid)
        return name + suffix

    def match_model_tensor_name(self, name: str, key: gguf.MODEL_TENSOR, bid: int | None, suffix: str = ".weight") -> bool:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            return False
        key_name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in key_name:
            if bid is None:
                return False
            key_name = key_name.format(bid=bid)
        else:
            if bid is not None:
                return False
        return name == (key_name + suffix)

    def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
        new_name = self.tensor_map.get_name(key=name, try_suffixes=try_suffixes)
        if new_name is None:
            raise ValueError(f"Can not map tensor {name!r}")
        return new_name

    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.block_count)

        if (n_ctx := self.find_hparam(["max_position_embeddings", "n_ctx"], optional=True)) is not None:
            self.gguf_writer.add_context_length(n_ctx)
            logger.info(f"gguf: context length = {n_ctx}")

        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        self.gguf_writer.add_embedding_length(n_embd)
        logger.info(f"gguf: embedding length = {n_embd}")

        if (n_ff := self.find_hparam(["intermediate_size", "n_inner"], optional=True)) is not None:
            self.gguf_writer.add_feed_forward_length(n_ff)
            logger.info(f"gguf: feed forward length = {n_ff}")

        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        self.gguf_writer.add_head_count(n_head)
        logger.info(f"gguf: head count = {n_head}")

        if (n_head_kv := self.hparams.get("num_key_value_heads")) is not None:
            self.gguf_writer.add_head_count_kv(n_head_kv)
            logger.info(f"gguf: key-value head count = {n_head_kv}")

        if (rope_theta := self.hparams.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(rope_theta)
            logger.info(f"gguf: rope theta = {rope_theta}")
        if (f_rms_eps := self.hparams.get("rms_norm_eps")) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
            logger.info(f"gguf: rms norm epsilon = {f_rms_eps}")
        if (f_norm_eps := self.find_hparam(["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_eps(f_norm_eps)
            logger.info(f"gguf: layer norm epsilon = {f_norm_eps}")
        if (n_experts := self.hparams.get("num_local_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
            logger.info(f"gguf: expert count = {n_experts}")
        if (n_experts_used := self.hparams.get("num_experts_per_tok")) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)
            logger.info(f"gguf: experts used count = {n_experts_used}")

        if (head_dim := self.hparams.get("head_dim")) is not None:
            self.gguf_writer.add_key_length(head_dim)
            self.gguf_writer.add_value_length(head_dim)

        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"gguf: file type = {self.ftype}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        return [(self.map_tensor_name(name), data_torch)]

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        del name, new_name, bid, n_dims  # unused

        return False

    # some models need extra generated tensors (like rope_freqs)
    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        return ()

    def prepare_tensors(self):
        max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(".weight,")

        for name, data_torch in chain(self.generate_extra_tensors(), self.get_tensors()):
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # use the first number-like part of the tensor name as the block id
            bid = None
            for part in name.split("."):
                if part.isdecimal():
                    bid = int(part)
                    break

            for new_name, data in ((n, d.squeeze().numpy()) for n, d in self.modify_tensors(data_torch, name, bid)):
                data: np.ndarray  # type hint
                n_dims = len(data.shape)
                data_qtype: gguf.GGMLQuantizationType | bool = self.tensor_force_quant(name, new_name, bid, n_dims)

                # Most of the codebase that takes in 1D tensors or norms only handles F32 tensors
                if n_dims <= 1 or new_name.endswith("_norm.weight"):
                    data_qtype = gguf.GGMLQuantizationType.F32

                # Conditions should closely match those in llama_model_quantize_internal in llama.cpp
                # Some tensor types are always in float32
                if data_qtype is False and (
                    any(
                        self.match_model_tensor_name(new_name, key, bid)
                        for key in (
                            gguf.MODEL_TENSOR.FFN_GATE_INP,
                            gguf.MODEL_TENSOR.POS_EMBD,
                            gguf.MODEL_TENSOR.TOKEN_TYPES,
                        )
                    )
                    or not name.endswith(".weight")
                ):
                    data_qtype = gguf.GGMLQuantizationType.F32

                if data_qtype is False and any(
                    self.match_model_tensor_name(new_name, key, bid)
                    for key in (
                        gguf.MODEL_TENSOR.TOKEN_EMBD,
                        gguf.MODEL_TENSOR.OUTPUT,
                        gguf.MODEL_TENSOR.ATTN_V,
                        gguf.MODEL_TENSOR.ATTN_K,
                        gguf.MODEL_TENSOR.ATTN_QKV,
                    )
                ):
                    if self.ftype in (
                        gguf.LlamaFileType.MOSTLY_Q4_0,
                        gguf.LlamaFileType.MOSTLY_Q4_1,
                    ):
                        data_qtype = gguf.GGMLQuantizationType.Q5_0
                    elif self.ftype in (
                        gguf.LlamaFileType.MOSTLY_Q5_0,
                        gguf.LlamaFileType.MOSTLY_Q5_1,
                    ):
                        data_qtype = gguf.GGMLQuantizationType.Q6_0

                # No override (data_qtype is False), or wants to be quantized (data_qtype is True)
                if isinstance(data_qtype, bool):
                    if self.ftype == gguf.LlamaFileType.ALL_F32:
                        data_qtype = gguf.GGMLQuantizationType.F32
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_F16:
                        data_qtype = gguf.GGMLQuantizationType.F16
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                        data_qtype = gguf.GGMLQuantizationType.BF16
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_Q4_0:
                        data_qtype = gguf.GGMLQuantizationType.Q4_0
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_Q4_1:
                        data_qtype = gguf.GGMLQuantizationType.Q4_1
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_Q5_0:
                        data_qtype = gguf.GGMLQuantizationType.Q5_0
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_Q5_1:
                        data_qtype = gguf.GGMLQuantizationType.Q5_1
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_Q6_0:
                        data_qtype = gguf.GGMLQuantizationType.Q6_0
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                        data_qtype = gguf.GGMLQuantizationType.Q8_0
                    else:
                        raise ValueError(f"Unknown file type: {self.ftype.name}")

                try:
                    data = gguf.quants.quantize(data, data_qtype)
                except gguf.QuantError as e:
                    logger.warning("%s, %s", e, "falling back to F16")
                    data_qtype = gguf.GGMLQuantizationType.F16
                    data = gguf.quants.quantize(data, data_qtype)

                shape = gguf.quant_shape_from_byte_shape(data.shape, data_qtype) if data.dtype == np.uint8 else data.shape

                # reverse shape to make it similar to the internal ggml dimension order
                shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"

                # n_dims is implicit in the shape
                logger.info(f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

                self.gguf_writer.add_tensor(new_name, data, raw_dtype=data_qtype)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)

    def prepare_metadata(self, vocab_only: bool):

        total_params, shared_params, expert_params, expert_count = self.gguf_writer.get_total_parameter_count()

        self.metadata = gguf.Metadata.load(self.metadata_override, self.dir_model_card, self.model_name, total_params)

        # Fallback to model directory name if metadata name is still missing
        if self.metadata.name is None:
            self.metadata.name = self.dir_model.name

        # Generate parameter weight class (useful for leader boards) if not yet determined
        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(total_params, shared_params, expert_params, expert_count)

        # Extract the encoding scheme from the file type name. e.g. 'gguf.LlamaFileType.MOSTLY_Q8_0' --> 'Q8_0'
        output_type: str = self.ftype.name.partition("_")[2]

        # Filename Output
        if self.fname_out.is_dir():
            # Generate default filename based on model specification and available metadata
            if not vocab_only:
                fname_default: str = gguf.naming_convention(self.metadata.name, self.metadata.basename, self.metadata.finetune, self.metadata.version, self.metadata.size_label, output_type, model_type="LoRA" if total_params < 0 else None)
            else:
                fname_default: str = gguf.naming_convention(self.metadata.name, self.metadata.basename, self.metadata.finetune, self.metadata.version, size_label=None, output_type=None, model_type="vocab")

            # Use the default filename
            self.fname_out = self.fname_out / f"{fname_default}.gguf"
        else:
            # Output path is a custom defined templated filename
            # Note: `not is_dir()` is used because `.is_file()` will not detect
            #       file template strings as it doesn't actually exist as a file

            # Process templated file name with the output ftype, useful with the "auto" ftype
            self.fname_out = self.fname_out.parent / gguf.fill_templated_filename(self.fname_out.name, output_type)

        self.set_type()

        logger.info("Set meta model")
        self.metadata.set_gguf_meta_model(self.gguf_writer)

        logger.info("Set model parameters")
        self.set_gguf_parameters()

        logger.info("Set model tokenizer")
        self.set_vocab()

        logger.info("Set model quantization version")
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

        logger.info("***********************************************************************************************")
        logger.info("** Converting to `q4_0`,`q4_1`,`q5_0`, `q5_1` or `q6_0` is not equiv to using `llama-quantize`!")
        logger.info("** Ftype `q4_0`,`q4_1` are here converting embeddings, output, attn_k and attn_v/qkv in q5_0.")
        logger.info("** Ftype `q5_0`,`q5_1` are here converting embeddings, output, attn_k and attn_v/qkv in q6_0.")
        logger.info("** This, in order to create a small but viable conv. to then for example make an iMatrix file.")
        logger.info("***********************************************************************************************")

    def write(self):
        self.prepare_tensors()
        self.prepare_metadata(vocab_only=False)
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()

    def write_vocab(self):
        if len(self.gguf_writer.tensors) != 1:
            raise ValueError('Splitting the vocabulary is not supported')

        self.prepare_metadata(vocab_only=True)
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    @staticmethod
    def get_model_part_names(dir_model: Path, prefix: str, suffix: str) -> list[str]:
        part_names: list[str] = []
        for filename in os.listdir(dir_model):
            if filename.startswith(prefix) and filename.endswith(suffix):
                part_names.append(filename)

        part_names.sort()

        return part_names

    @staticmethod
    def load_hparams(dir_model: Path):
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def register(cls, *names: str) -> Callable[[AnyModel], AnyModel]:
        assert names

        def func(modelcls: AnyModel) -> AnyModel:
            for name in names:
                cls._model_classes[name] = modelcls
            return modelcls
        return func

    @classmethod
    def from_model_architecture(cls, arch: str) -> type[Model]:
        try:
            return cls._model_classes[arch]
        except KeyError:
            raise NotImplementedError(f'Architecture {arch!r} not supported!') from None

    def does_token_look_special(self, token: str | bytes) -> bool:
        if isinstance(token, (bytes, bytearray)):
            token_text = token.decode(encoding="utf-8")
        elif isinstance(token, memoryview):
            token_text = token.tobytes().decode(encoding="utf-8")
        else:
            token_text = token

        # Some models mark some added tokens which ought to be control tokens as not special.
        # (e.g. command-r, command-r-plus, deepseek-coder, gemma{,-2})
        seems_special = token_text in (
            "<pad>",  # deepseek-coder
            "<mask>", "<2mass>", "[@BOS@]",  # gemma{,-2}
        )

        seems_special = seems_special or (token_text.startswith("<|") and token_text.endswith("|>"))
        seems_special = seems_special or (token_text.startswith("<｜") and token_text.endswith("｜>"))  # deepseek-coder

        # TODO: should these be marked as UNUSED instead? (maybe not)
        seems_special = seems_special or (token_text.startswith("<unused") and token_text.endswith(">"))  # gemma{,-2}

        return seems_special

    # used for GPT-2 BPE and WordPiece vocabs
    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token: str = reverse_vocab[i]
                if token in added_vocab:
                    if tokenizer.added_tokens_decoder[i].special or self.does_token_look_special(token):
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        token = token.replace(b"\xe2\x96\x81".decode("utf-8"), " ")  # pre-normalize user-defined spaces
                        toktypes.append(gguf.TokenType.USER_DEFINED)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)
                tokens.append(token)

        return tokens, toktypes, tokpre

    # NOTE: this function is generated by convert_hf_to_gguf_update.py
    #       do not modify it manually!
    # ref:  https://github.com/ggerganov/llama.cpp/pull/6920
    # Marker: Start get_vocab_base_pre
    def get_vocab_base_pre(self, tokenizer) -> str:
        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that
        # is specific for the BPE pre-tokenizer used by the model
        # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can
        # use in llama.cpp to implement the same pre-tokenizer

        chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

        chktok = tokenizer.encode(chktxt)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        logger.debug(f"chktok: {chktok}")
        logger.debug(f"chkhsh: {chkhsh}")

        res = None

        # NOTE: if you get an error here, you need to update the convert_hf_to_gguf_update.py script
        #       or pull the latest version of the model from Huggingface
        #       don't edit the hashes manually!
        if chkhsh == "66b8d4e19ab16c3bfd89bce5d785fb7e0155e8648708a1f42077cb9fe002c273":
            # ref: https://huggingface.co/alvarobartt/grok-2-tokenizer
            res = "grok-2"
        if chkhsh == "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5":
            # ref: https://huggingface.co/meta-llama/Meta-Llama-3-8B
            res = "llama-bpe"
        if chkhsh == "049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754":
            # ref: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base
            res = "deepseek-llm"
        if chkhsh == "347715f544604f9118bb75ed199f68779f423cabb20db6de6f31b908d04d7821":
            # ref: https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base
            res = "deepseek-coder"
        if chkhsh == "8aeee3860c56296a157a1fe2fad249ec40aa59b1bb5709f4ade11c4e6fe652ed":
            # ref: https://huggingface.co/tiiuae/falcon-7b
            res = "falcon"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/BAAI/bge-small-en-v1.5
            res = "bert-bge"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/mosaicml/mpt-7b
            res = "mpt"
        if chkhsh == "35d91631860c815f952d711435f48d356ebac988362536bed955d43bfa436e34":
            # ref: https://huggingface.co/bigcode/starcoder2-3b
            res = "starcoder"
        if chkhsh == "3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454":
            # ref: https://huggingface.co/openai-community/gpt2
            res = "gpt-2"
        if chkhsh == "32d85c31273f8019248f2559fed492d929ea28b17e51d81d3bb36fff23ca72b3":
            # ref: https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b
            res = "stablelm2"
        if chkhsh == "6221ad2852e85ce96f791f476e0b390cf9b474c9e3d1362f53a24a06dc8220ff":
            # ref: https://huggingface.co/smallcloudai/Refact-1_6-base
            res = "refact"
        if chkhsh == "9c2227e4dd922002fb81bde4fc02b0483ca4f12911410dee2255e4987644e3f8":
            # ref: https://huggingface.co/CohereForAI/c4ai-command-r-v01
            res = "command-r"
        if chkhsh == "52df12b4c8d4176e7481aab4b6e8454d1fd0a210a04a574f6d4e067d10e23c3e":
            # ref: https://huggingface.co/CohereLabs/North-Mini-Code-1.0
            res = "cohere2_moe"
        if chkhsh == "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea":
            # ref: https://huggingface.co/Qwen/Qwen1.5-7B
            res = "qwen2"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/allenai/OLMo-1.7-7B-hf
            res = "olmo"
        if chkhsh == "a8594e3edff7c29c003940395316294b2c623e09894deebbc65f33f1515df79e":
            # ref: https://huggingface.co/databricks/dbrx-base
            res = "dbrx"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-en
            res = "jina-v2-en"
        if chkhsh == "171aeeedd6fb548d418a7461d053f11b6f1f1fc9b387bd66640d28a4b9f5c643":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-es
            res = "jina-v2-es"
        if chkhsh == "27949a2493fc4a9f53f5b9b029c82689cfbe5d3a1929bb25e043089e28466de6":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-de
            res = "jina-v2-de"
        if chkhsh == "c136ed14d01c2745d4f60a9596ae66800e2b61fa45643e72436041855ad4089d":
            # ref: https://huggingface.co/abacusai/Smaug-Llama-3-70B-Instruct
            res = "smaug-bpe"
        if chkhsh == "c7ea5862a53e4272c035c8238367063e2b270d51faa48c0f09e9d5b54746c360":
            # ref: https://huggingface.co/LumiOpen/Poro-34B-chat
            res = "poro-chat"
        if chkhsh == "7967bfa498ade6b757b064f31e964dddbb80f8f9a4d68d4ba7998fcf281c531a":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-code
            res = "jina-v2-code"
        if chkhsh == "b6e8e1518dc4305be2fe39c313ed643381c4da5db34a98f6a04c093f8afbe99b":
            # ref: https://huggingface.co/THUDM/glm-4-9b-chat
            res = "chatglm-bpe"
        if chkhsh == "81d72c7348a9f0ebe86f23298d37debe0a5e71149e29bd283904c02262b27516":
            # ref: https://huggingface.co/THUDM/glm-4-9b-chat
            res = "chatglm-bpe"
        if chkhsh == "a1336059768a55c99a734006ffb02203cd450fed003e9a71886c88acf24fdbc2":
            # ref: https://huggingface.co/THUDM/glm-4-9b-hf
            res = "glm4"
        if chkhsh == "9ca2dd618e8afaf09731a7cf6e2105b373ba6a1821559f258b272fe83e6eb902":
            # ref: https://huggingface.co/zai-org/GLM-4.5-Air, https://huggingface.co/zai-org/GLM-4.5
            res = "glm4"
        if chkhsh == "cdf5f35325780597efd76153d4d1c16778f766173908894c04afc20108536267":
            # ref: https://huggingface.co/zai-org/GLM-4.7-Flash
            res = "glm4"
        if chkhsh == "7fc505bd3104ca1083b150b17d088b59534ede9bde81f0dd2090967d7fe52cee":
            # ref: https://huggingface.co/LumiOpen/Viking-7B
            res = "viking"
        if chkhsh == "b53802fb28e26d645c3a310b34bfe07da813026ec7c7716883404d5e0f8b1901":
            # ref: https://huggingface.co/core42/jais-13b
            res = "jais"
        if chkhsh == "7b3e7548e4308f52a76e8229e4e6cc831195d0d1df43aed21ac6c93da05fec5f":
            # ref: https://huggingface.co/WisdomShell/CodeShell-7B
            res = "codeshell"
        if chkhsh == "63b97e4253352e6f357cc59ea5b583e3a680eaeaf2632188c2b952de2588485e":
            # ref: https://huggingface.co/mistralai/Mistral-Nemo-Base-2407
            res = "tekken"
        if chkhsh == "855059429035d75a914d1eda9f10a876752e281a054a7a3d421ef0533e5b6249":
            # ref: https://huggingface.co/HuggingFaceTB/SmolLM-135M
            res = "smollm"
        if chkhsh == "877081d19cf6996e2c4ff0e1236341e9b7bde288f5311a56a937f0afbbb3aeb5":
            # ref: https://huggingface.co/deepseek-ai/DeepSeek-V3
            res = "deepseek-v3"
        if chkhsh == "d5f1dd6f980fec569fb218a81a7658ac45fc56b38c5a0adeb1c232fbe04ef5ec":
            # ref: https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Base
            res = "seed-coder"
        if chkhsh == "81212dc7cdb7e0c1074ca62c5aeab0d43c9f52b8a737be7b12a777c953027890":
            # ref: https://huggingface.co/moonshotai/Kimi-K2-Base
            res = "kimi-k2"
        if chkhsh == "9b1be57e70d20d9501b2b3186e792d81181ae36ada3903c26f9fea418cf87206":
            # ref: https://huggingface.co/inclusionAI/Ling-mini-base-2.0
            res = "bailingmoe2"
        if chkhsh == "f4f37b6c8eb9ea29b3eac6bb8c8487c5ab7885f8d8022e67edc1c68ce8403e95":
            # ref: https://huggingface.co/MiniMaxAI/MiniMax-M2
            res = "minimax-m2"
        if chkhsh == "9dcf830ee9990cdbf78cc523a5f7bd9ad8f3f9890c2d3581d2785ad10f07049d":
            # ref: https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Base
            res = "mellum2"
        if chkhsh == "972da7b59cec44d1f0a490a86c96df53859e486e481563e5dddac155013d87ac":
            # ref: https://huggingface.co/poolside/Laguna-XS.2
            res = "laguna"
        if chkhsh == "bba3b3366b646dbdded5dbc42d59598b849371afc42f7beafa914afaa5b70aa6":
            # ref: https://huggingface.co/tencent/Hunyuan-4B-Instruct  (shared by tencent/Hy3)
            res = "hunyuan-dense"
        if res is None:
            logger.warning("\n")
            logger.warning("**************************************************************************************")
            logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")
            logger.warning("**          There are 2 possible reasons for this:")
            logger.warning("**          - the model has not been added to convert_hf_to_gguf_update.py yet")
            logger.warning("**          - the pre-tokenization config has changed upstream")
            logger.warning("**          Check your model files and convert_hf_to_gguf_update.py and update them accordingly.")
            logger.warning("** ref:     https://github.com/ggerganov/llama.cpp/pull/6920")
            logger.warning("**")
            logger.warning(f"** chkhsh:  {chkhsh}")
            logger.warning("**************************************************************************************")
            logger.warning("\n")
            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

        logger.debug(f"tokenizer.ggml.pre: {repr(res)}")
        logger.debug(f"chkhsh: {chkhsh}")

        return res
        # Marker: End get_vocab_base_pre

    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_qwen(self):
        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        vocab_size = hparams["vocab_size"]
        assert max(tokenizer.get_vocab().values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks
        for token, rank in mergeable_ranks.items():
            vocab[QwenModel.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
            assert len(merged) == 2
            merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))

        # for this kind of tokenizer, added_vocab is not a subset of vocab, so they need to be combined
        added_vocab = tokenizer.special_tokens
        reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **added_vocab}.items()}

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, load_merges=False)
        special_vocab.merges = merges
        # only add special tokens when they were not already loaded from config.json
        if len(special_vocab.special_token_ids) == 0:
            special_vocab._set_special_token("bos", tokenizer.special_tokens["<|endoftext|>"])
            special_vocab._set_special_token("eos", tokenizer.special_tokens["<|endoftext|>"])
        # this one is usually not in config.json anyway
        special_vocab._set_special_token("unk", tokenizer.special_tokens["<|endoftext|>"])
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_sentencepiece(self, add_to_gguf=True):
        tokens, scores, toktypes = self._create_vocab_sentencepiece()

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _create_vocab_sentencepiece(self):
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / 'tokenizer.model'

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        for token_id in range(tokenizer.vocab_size()):
            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)
                for key in added_tokens_json:
                    token_id = added_tokens_json[key]
                    if token_id >= vocab_size:
                        logger.warning(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                        continue

                    tokens[token_id] = key.encode("utf-8")
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                added_tokens_decoder = tokenizer_config_json.get("added_tokens_decoder", {})
                for token_id, token_data in added_tokens_decoder.items():
                    token_id = int(token_id)
                    token: str = token_data["content"]
                    if toktypes[token_id] != SentencePieceTokenTypes.UNUSED:
                        if tokens[token_id] != token.encode("utf-8"):
                            logger.warning(f'replacing token {token_id}: {tokens[token_id].decode("utf-8")!r} -> {token!r}')
                    if token_data.get("special") or self.does_token_look_special(token):
                        toktypes[token_id] = SentencePieceTokenTypes.CONTROL
                    else:
                        token = token.replace(b"\xe2\x96\x81".decode("utf-8"), " ")  # pre-normalize user-defined spaces
                        toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

                    scores[token_id] = -1000.0
                    tokens[token_id] = token.encode("utf-8")

        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]")
            for i in range(1, pad_count + 1):
                tokens.append(bytes(f"[PAD{i}]", encoding="utf-8"))
                scores.append(-1000.0)
                toktypes.append(SentencePieceTokenTypes.UNUSED)

        return tokens, scores, toktypes

    def _set_vocab_llama_hf(self):
        vocab = gguf.LlamaHfVocab(self.dir_model)
        tokens = []
        scores = []
        toktypes = []

        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_builtin(self, model_name: Literal["gpt-neox", "llama-spm"], vocab_size: int):
        tokenizer_path = Path(sys.path[0]) / "models" / f"ggml-vocab-{model_name}.gguf"
        logger.warning(f"Using tokenizer from '{os.path.relpath(tokenizer_path, os.getcwd())}'")
        vocab_reader = gguf.GGUFReader(tokenizer_path, "r")

        default_pre = "mpt" if model_name == "gpt-neox" else "default"

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.MODEL)
        assert field  # tokenizer model
        self.gguf_writer.add_tokenizer_model(bytes(field.parts[-1]).decode("utf-8"))

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.PRE)
        self.gguf_writer.add_tokenizer_pre(bytes(field.parts[-1]).decode("utf-8") if field else default_pre)

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.LIST)
        assert field  # token list
        self.gguf_writer.add_token_list([bytes(field.parts[i]) for i in field.data][:vocab_size])

        if model_name == "llama-spm":
            field = vocab_reader.get_field(gguf.Keys.Tokenizer.SCORES)
            assert field  # token scores
            self.gguf_writer.add_token_scores([field.parts[i].tolist()[0] for i in field.data][:vocab_size])

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.TOKEN_TYPE)
        assert field  # token types
        self.gguf_writer.add_token_types([field.parts[i].tolist()[0] for i in field.data][:vocab_size])

        if model_name != "llama-spm":
            field = vocab_reader.get_field(gguf.Keys.Tokenizer.MERGES)
            assert field  # token merges
            self.gguf_writer.add_token_merges([bytes(field.parts[i]) for i in field.data])

        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.BOS_ID)) is not None:
            self.gguf_writer.add_bos_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.EOS_ID)) is not None:
            self.gguf_writer.add_eos_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.UNK_ID)) is not None:
            self.gguf_writer.add_unk_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.PAD_ID)) is not None:
            self.gguf_writer.add_pad_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.ADD_BOS)) is not None:
            self.gguf_writer.add_add_bos_token(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.ADD_EOS)) is not None:
            self.gguf_writer.add_add_eos_token(field.parts[-1].tolist()[0])


@Model.register("GPTNeoXForCausalLM")
class GPTNeoXModel(Model):
    model_arch = gguf.MODEL_ARCH.GPTNEOX

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]

        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(
            int(self.hparams["rotary_pct"] * (self.hparams["hidden_size"] // self.hparams["num_attention_heads"])),
        )
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_parallel_residual(self.hparams.get("use_parallel_residual", True))
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_eps"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))

        tensors: list[tuple[str, Tensor]] = []

        if re.match(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", name):
            # Map bloom-style qkv_linear to gpt-style qkv_linear
            # bloom: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L238-L252  # noqa
            # gpt-2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L312  # noqa
            qkv_weights = data_torch.reshape((n_head, 3, n_embed // n_head, n_embed))
            data_torch = torch.cat(
                (
                    qkv_weights[:, 0, :, :].reshape((-1, n_embed)),
                    qkv_weights[:, 1, :, :].reshape((-1, n_embed)),
                    qkv_weights[:, 2, :, :].reshape((-1, n_embed)),
                ),
                dim=0,
            )
            logger.info("re-format attention.linear_qkv.weight")
        elif re.match(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.bias", name):
            qkv_bias = data_torch.reshape((n_head, 3, n_embed // n_head))
            data_torch = torch.cat(
                (
                    qkv_bias[:, 0, :].reshape((n_embed,)),
                    qkv_bias[:, 1, :].reshape((n_embed,)),
                    qkv_bias[:, 2, :].reshape((n_embed,)),
                ),
                dim=0,
            )
            logger.info("re-format attention.linear_qkv.bias")

        tensors.append((self.map_tensor_name(name), data_torch))

        return tensors


@Model.register("BloomForCausalLM")
class BloomModel(Model):
    model_arch = gguf.MODEL_ARCH.BLOOM

    def set_gguf_parameters(self):
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        self.gguf_writer.add_context_length(self.hparams.get("seq_length", n_embed))
        self.gguf_writer.add_embedding_length(n_embed)
        self.gguf_writer.add_feed_forward_length(4 * n_embed)
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))

        name = re.sub(r'transformer\.', '', name)

        tensors: list[tuple[str, Tensor]] = []

        if re.match(r"h\.\d+\.self_attention\.query_key_value\.weight", name):
            # Map bloom-style qkv_linear to gpt-style qkv_linear
            # bloom: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L238-L252  # noqa
            # gpt-2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L312  # noqa
            qkv_weights = data_torch.reshape((n_head, 3, n_embed // n_head, n_embed))
            data_torch = torch.cat(
                (
                    qkv_weights[:, 0, :, :].reshape((-1, n_embed)),
                    qkv_weights[:, 1, :, :].reshape((-1, n_embed)),
                    qkv_weights[:, 2, :, :].reshape((-1, n_embed)),
                ),
                dim=0,
            )
            logger.info("re-format attention.linear_qkv.weight")
        elif re.match(r"h\.\d+\.self_attention\.query_key_value\.bias", name):
            qkv_bias = data_torch.reshape((n_head, 3, n_embed // n_head))
            data_torch = torch.cat(
                (
                    qkv_bias[:, 0, :].reshape((n_embed,)),
                    qkv_bias[:, 1, :].reshape((n_embed,)),
                    qkv_bias[:, 2, :].reshape((n_embed,)),
                ),
                dim=0,
            )
            logger.info("re-format attention.linear_qkv.bias")

        tensors.append((self.map_tensor_name(name), data_torch))

        if name == "word_embeddings.weight":
            assert self.tensor_names is not None

            # TODO: tie them at runtime, don't duplicate in the model file
            if all(s not in self.tensor_names for s in ("lm_head.weight", "output.weight")):
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch))

        return tensors


@Model.register("MPTForCausalLM")
class MPTModel(Model):
    model_arch = gguf.MODEL_ARCH.MPT

    def set_vocab(self):
        try:
            self._set_vocab_gpt2()
        except Exception:
            # Fallback for SEA-LION model
            self._set_vocab_sentencepiece()
            self.gguf_writer.add_add_bos_token(False)
            self.gguf_writer.add_pad_token_id(3)
            self.gguf_writer.add_eos_token_id(1)
            self.gguf_writer.add_unk_token_id(0)

    def set_gguf_parameters(self):
        block_count = self.hparams["n_layers"]
        self.gguf_writer.add_context_length(self.hparams["max_seq_len"])
        self.gguf_writer.add_embedding_length(self.hparams["d_model"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["d_model"])
        self.gguf_writer.add_head_count(self.hparams["n_heads"])
        if kv_n_heads := self.hparams["attn_config"].get("kv_n_heads"):
            self.gguf_writer.add_head_count_kv(kv_n_heads)
        self.gguf_writer.add_layer_norm_eps(1e-5)
        if self.hparams["attn_config"]["clip_qkv"] is not None:
            self.gguf_writer.add_clamp_kqv(self.hparams["attn_config"]["clip_qkv"])
        if self.hparams["attn_config"]["alibi"]:
            self.gguf_writer.add_max_alibi_bias(self.hparams["attn_config"]["alibi_bias_max"])
        else:
            self.gguf_writer.add_max_alibi_bias(0.0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        if "scales" in name:
            new_name = self.map_tensor_name(name, try_suffixes=(".weight", ".bias", ".scales"))
            new_name = new_name.replace("scales", "act.scales")
        else:
            new_name = self.map_tensor_name(name, try_suffixes=(".weight", ".bias"))

        return [(new_name, data_torch)]


@Model.register("OrionForCausalLM")
class OrionModel(Model):
    model_arch = gguf.MODEL_ARCH.ORION

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)

        ctx_length = 0
        if "max_sequence_length" in self.hparams:
            ctx_length = self.hparams["max_sequence_length"]
        elif "max_position_embeddings" in self.hparams:
            ctx_length = self.hparams["max_position_embeddings"]
        elif "model_max_length" in self.hparams:
            ctx_length = self.hparams["model_max_length"]
        else:
            raise ValueError("gguf: can not find ctx length parameter.")

        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_context_length(ctx_length)
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_head_count(head_count)
        self.gguf_writer.add_head_count_kv(head_count_kv)
        # note: config provides rms norm but it is actually layer norm
        # ref:  https://huggingface.co/OrionStarAI/Orion-14B-Chat/blob/276a17221ce42beb45f66fac657a41540e71f4f5/modeling_orion.py#L570-L571
        self.gguf_writer.add_layer_norm_eps(self.hparams["rms_norm_eps"])


@Model.register("BaichuanForCausalLM", "BaiChuanForCausalLM")
class BaichuanModel(Model):
    model_arch = gguf.MODEL_ARCH.BAICHUAN

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)

        ctx_length = 0
        if "max_sequence_length" in self.hparams:
            ctx_length = self.hparams["max_sequence_length"]
        elif "max_position_embeddings" in self.hparams:
            ctx_length = self.hparams["max_position_embeddings"]
        elif "model_max_length" in self.hparams:
            ctx_length = self.hparams["model_max_length"]
        else:
            raise ValueError("gguf: can not find ctx length parameter.")

        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_context_length(ctx_length)
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(head_count)
        self.gguf_writer.add_head_count_kv(head_count_kv)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_file_type(self.ftype)

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "linear":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)

        tensors: list[tuple[str, Tensor]] = []

        if bid is not None and name == f"model.layers.{bid}.self_attn.W_pack.weight":
            logger.info(f"Unpacking and permuting layer {bid}")
            tensors = [
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid),
                    self._reverse_hf_permute_part(data_torch, 0, head_count, head_count)),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid),
                    self._reverse_hf_permute_part(data_torch, 1, head_count, head_count_kv)),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid),
                    self._reverse_hf_part(data_torch, 2)),
            ]
        else:
            tensors = [(self.map_tensor_name(name), data_torch)]

        return tensors

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )

    def _reverse_hf_permute_part(
        self, weights: Tensor, n_part: int, n_head: int, n_head_kv: int | None = None,
    ) -> Tensor:
        r = weights.shape[0] // 3
        return self._reverse_hf_permute(weights[r * n_part:r * n_part + r, ...], n_head, n_head_kv)

    def _reverse_hf_part(self, weights: Tensor, n_part: int) -> Tensor:
        r = weights.shape[0] // 3
        return weights[r * n_part:r * n_part + r, ...]


@Model.register("XverseForCausalLM")
class XverseModel(Model):
    model_arch = gguf.MODEL_ARCH.XVERSE

    def set_vocab(self):
        assert (self.dir_model / "tokenizer.json").is_file()
        dir_model = self.dir_model
        hparams = self.hparams

        tokens: list[bytes] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model)
        vocab_size = hparams.get("vocab_size", len(tokenizer.vocab))
        # Since we are checking the maximum index, we need to ensure it's strictly less than vocab_size,
        # because vocab_size is the count of items, and indexes start at 0.
        max_vocab_index = max(tokenizer.get_vocab().values())
        if max_vocab_index >= vocab_size:
            raise ValueError("Vocabulary size exceeds expected maximum size.")

        reverse_vocab: dict[int, str] = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()

        for token_id in range(vocab_size):
            token_text = reverse_vocab[token_id].encode('utf-8')
            # replace "\x00" to string with length > 0
            if token_text == b"\x00":
                toktype = gguf.TokenType.BYTE  # special
                token_text = f"<{token_text}>".encode('utf-8')
            elif re.fullmatch(br"<0x[0-9A-Fa-f]{2}>", token_text):
                toktype = gguf.TokenType.BYTE  # special
            elif reverse_vocab[token_id] in added_vocab:
                if tokenizer.added_tokens_decoder[token_id].special:
                    toktype = gguf.TokenType.CONTROL
                else:
                    toktype = gguf.TokenType.USER_DEFINED
            else:
                toktype = gguf.TokenType.NORMAL

            tokens.append(token_text)
            toktypes.append(toktype)

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)

        ctx_length = 0
        if "max_sequence_length" in self.hparams:
            ctx_length = self.hparams["max_sequence_length"]
        elif "max_position_embeddings" in self.hparams:
            ctx_length = self.hparams["max_position_embeddings"]
        elif "model_max_length" in self.hparams:
            ctx_length = self.hparams["model_max_length"]
        else:
            raise ValueError("gguf: can not find ctx length parameter.")

        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_context_length(ctx_length)
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(head_count)
        self.gguf_writer.add_head_count_kv(head_count_kv)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_file_type(self.ftype)

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "linear":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)

        # HF models permute some of the tensors, so we need to undo that
        if name.endswith("q_proj.weight"):
            data_torch = self._reverse_hf_permute(data_torch, head_count, head_count)
        if name.endswith("k_proj.weight"):
            data_torch = self._reverse_hf_permute(data_torch, head_count, head_count_kv)

        return [(self.map_tensor_name(name), data_torch)]

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )


@Model.register("FalconForCausalLM", "RWForCausalLM")
class FalconModel(Model):
    model_arch = gguf.MODEL_ARCH.FALCON

    def set_gguf_parameters(self):
        block_count = self.hparams.get("num_hidden_layers")
        if block_count is None:
            block_count = self.hparams["n_layer"]  # old name

        n_head = self.hparams.get("num_attention_heads")
        if n_head is None:
            n_head = self.hparams["n_head"]  # old name

        n_head_kv = self.hparams.get("num_kv_heads")
        if n_head_kv is None:
            n_head_kv = self.hparams.get("n_head_kv", 1)  # old name

        self.gguf_writer.add_context_length(2048)  # not in config.json
        self.gguf_writer.add_tensor_data_layout("jploski")  # qkv tensor transform
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        # QKV tensor transform
        # The original query_key_value tensor contains n_head_kv "kv groups",
        # each consisting of n_head/n_head_kv query weights followed by one key
        # and one value weight (shared by all query heads in the kv group).
        # This layout makes it a big pain to work with in GGML.
        # So we rearrange them here,, so that we have n_head query weights
        # followed by n_head_kv key weights followed by n_head_kv value weights,
        # in contiguous fashion.
        # ref: https://github.com/jploski/ggml/blob/falcon40b/examples/falcon/convert-hf-to-ggml.py

        if "query_key_value" in name:
            n_head = self.find_hparam(["num_attention_heads", "n_head"])
            n_head_kv = self.find_hparam(["num_kv_heads", "n_head_kv"], optional=True) or 1
            head_dim = self.hparams["hidden_size"] // n_head

            qkv = data_torch.view(n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)
            q = qkv[:, :-2].reshape(n_head * head_dim, head_dim * n_head)
            k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
            v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)
            data_torch = torch.cat((q, k, v)).reshape_as(data_torch)

        return [(self.map_tensor_name(name), data_torch)]


@Model.register("GPTBigCodeForCausalLM")
class StarCoderModel(Model):
    model_arch = gguf.MODEL_ARCH.STARCODER

    def set_gguf_parameters(self):
        block_count = self.hparams["n_layer"]

        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)


@Model.register("GPTRefactForCausalLM")
class RefactModel(Model):
    model_arch = gguf.MODEL_ARCH.REFACT

    def set_vocab(self):
        super().set_vocab()

        # TODO: how to determine special FIM tokens automatically?
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False,
                                          special_token_types = ['prefix', 'suffix', 'middle', 'eot'])
        special_vocab._set_special_token("prefix", 1)
        special_vocab._set_special_token("suffix", 3)
        special_vocab._set_special_token("middle", 2)
        special_vocab.chat_template = None  # do not add it twice
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        hidden_dim = self.hparams["n_embd"]
        inner_dim = 4 * hidden_dim
        hidden_dim = int(2 * inner_dim / 3)
        multiple_of = 256
        ff_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        block_count = self.hparams["n_layer"]

        # refact uses Alibi. So this is from config.json which might be used by training.
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])

        self.gguf_writer.add_feed_forward_length(ff_dim)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        hidden_dim = self.hparams["n_embd"]
        inner_dim = 4 * hidden_dim
        hidden_dim = int(2 * inner_dim / 3)
        multiple_of = 256
        ff_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        n_head = self.hparams["n_head"]
        n_head_kv = 1
        head_dim = self.hparams["n_embd"] // n_head

        tensors: list[tuple[str, Tensor]] = []

        if bid is not None:
            if name == f"transformer.h.{bid}.attn.kv.weight":
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), data_torch[:n_head_kv * head_dim]))
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid), data_torch[n_head_kv * head_dim:]))
            elif name == f"transformer.h.{bid}.attn.q.weight":
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), data_torch))
            elif name == f"transformer.h.{bid}.mlp.gate_up_proj.weight":
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid), data_torch[:ff_dim]))
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP, bid), data_torch[ff_dim:]))

        if len(tensors) == 0:
            tensors.append((self.map_tensor_name(name), data_torch))

        return tensors


@Model.register("StableLmForCausalLM", "StableLMEpochForCausalLM", "LlavaStableLMEpochForCausalLM")
class StableLMModel(Model):
    model_arch = gguf.MODEL_ARCH.STABLELM

    def set_vocab(self):
        if (self.dir_model / "tokenizer.json").is_file():
            self._set_vocab_gpt2()
        else:
            # StableLM 2 1.6B used to have a vocab in a similar format to Qwen's vocab
            self._set_vocab_qwen()

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]

        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        rotary_factor = self.find_hparam(["partial_rotary_factor", "rope_pct"])
        self.gguf_writer.add_rope_dimension_count(int(rotary_factor * (hparams["hidden_size"] // hparams["num_attention_heads"])))
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(hparams["num_key_value_heads"])
        self.gguf_writer.add_parallel_residual(hparams["use_parallel_residual"] if "use_parallel_residual" in hparams else True)
        self.gguf_writer.add_layer_norm_eps(self.find_hparam(["layer_norm_eps", "norm_eps"]))
        self.gguf_writer.add_file_type(self.ftype)

    _q_norms: list[dict[str, Tensor]] | None = None
    _k_norms: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams["num_key_value_heads"]

        if name.find("q_layernorm.norms") != -1:
            assert bid is not None

            if self._q_norms is None:
                self._q_norms = [{} for _ in range(self.block_count)]

            self._q_norms[bid][name] = data_torch

            if len(self._q_norms[bid]) >= n_head:
                return self._stack_qk_norm(bid, n_head, self._q_norms[bid], "q_layernorm")
            else:
                return []

        if name.find("k_layernorm.norms") != -1:
            assert bid is not None

            if self._k_norms is None:
                self._k_norms = [{} for _ in range(self.block_count)]

            self._k_norms[bid][name] = data_torch

            if len(self._k_norms[bid]) >= n_kv_head:
                return self._stack_qk_norm(bid, n_kv_head, self._k_norms[bid], "k_layernorm")
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]

    def _stack_qk_norm(self, bid: int, n_head: int, norms: dict[str, Tensor], layer_name: str = "q_layernorm"):
        datas: list[Tensor] = []
        # extract the norms in order
        for xid in range(n_head):
            ename = f"model.layers.{bid}.self_attn.{layer_name}.norms.{xid}.weight"
            datas.append(norms[ename])
            del norms[ename]
        data_torch = torch.stack(datas, dim=0)

        merged_name = f"model.layers.{bid}.self_attn.{layer_name}.weight"
        new_name = self.map_tensor_name(merged_name)

        return [(new_name, data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._q_norms is not None or self._k_norms is not None:
            # flatten two `list[dict[str, Tensor]]` into a single `list[str]`
            norms = (
                [k for d in self._q_norms for k in d.keys()] if self._q_norms is not None else []
            ) + (
                [k for d in self._k_norms for k in d.keys()] if self._k_norms is not None else []
            )
            if len(norms) > 0:
                raise ValueError(f"Unprocessed norms: {norms}")


@Model.register("LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM")
class LlamaModel(Model):
    model_arch = gguf.MODEL_ARCH.LLAMA

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            try:
                self._set_vocab_llama_hf()
            except (FileNotFoundError, TypeError):
                # Llama 3
                self._set_vocab_gpt2()

        # Apply to CodeLlama only (and ignore for Llama 3 with a vocab size of 128256)
        if self.hparams.get("vocab_size", 32000) == 32016:
            special_vocab = gguf.SpecialVocab(
                self.dir_model, load_merges=False,
                special_token_types = ['prefix', 'suffix', 'middle', 'eot']
            )
            special_vocab._set_special_token("prefix", 32007)
            special_vocab._set_special_token("suffix", 32008)
            special_vocab._set_special_token("middle", 32009)
            special_vocab._set_special_token("eot",    32010)
            special_vocab.add_to_gguf(self.gguf_writer)

        # Apply to Seed-Coder only (and ignore otherwise)
        if self.hparams.get("vocab_size", 32000) == 155136:
            special_vocab = gguf.SpecialVocab(
                self.dir_model, load_merges=False,
                special_token_types = ['prefix', 'suffix', 'middle', 'eot']
            )
            special_vocab._set_special_token("prefix", 124)
            special_vocab._set_special_token("suffix", 125)
            special_vocab._set_special_token("middle", 126)
            special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        saved_intermediate_size = self.hparams.get("intermediate_size")
        saved_num_experts_per_tok = self.hparams.pop("num_experts_per_tok")
        self.hparams["intermediate_size"] = self.hparams["prefix_dense_intermediate_size"]
        super().set_gguf_parameters()
        self.hparams["intermediate_size"] = saved_intermediate_size
        self.hparams["num_experts_per_tok"] = saved_num_experts_per_tok
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        if "head_dim" in hparams:
            rope_dim = hparams["head_dim"]
        else:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(rope_dim)

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "linear":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])

        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                if "add_prefix_space" in tokenizer_config_json:
                    self.gguf_writer.add_add_space_prefix(tokenizer_config_json["add_prefix_space"])

        # Apply to granite small models only
        if self.hparams.get("vocab_size", 32000) == 49152:
            self.gguf_writer.add_add_bos_token(False)

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)

        # process the experts separately
        if name.find("block_sparse_moe.experts") != -1:
            n_experts = self.hparams["num_local_experts"]

            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for wid in ["w1", "w2", "w3"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"layers.{bid}.feed_forward.experts.{wid}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if rope_scaling := self.find_hparam(["rope_scaling"], optional=True):
            if rope_scaling.get("rope_type", '').lower() == "llama3":
                base = self.hparams.get("rope_theta", 10000.0)
                dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
                freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

                factor = rope_scaling.get("factor", 8.0)
                low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
                high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
                old_context_len = self.hparams.get("original_max_position_embeddings", 8192)

                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor
                assert low_freq_wavelen != high_freq_wavelen

                rope_factors = []
                for freq in freqs:
                    wavelen = 2 * math.pi / freq
                    if wavelen < high_freq_wavelen:
                        rope_factors.append(1)
                    elif wavelen > low_freq_wavelen:
                        rope_factors.append(factor)
                    else:
                        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                        rope_factors.append(1 / ((1 - smooth) / factor + smooth))

                yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.tensor(rope_factors, dtype=torch.float32))

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@Model.register("DeciLMForCausalLM")
class DeciModel(Model):
    model_arch = gguf.MODEL_ARCH.DECI

    @staticmethod
    def _ffn_mult_to_intermediate_size(ffn_mult: float, n_embd: int) -> int:
        # DeciLM-specific code
        intermediate_size = int(2 * ffn_mult * n_embd / 3)
        return DeciModel._find_multiple(intermediate_size, 256)

    @staticmethod
    def _find_multiple(n: int, k: int) -> int:
        # DeciLM-specific code
        if n % k == 0:
            return n
        return n + k - (n % k)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "block_configs" in self.hparams: # Llama-3_1-Nemotron-51B
            _block_configs: list[dict[str,Any]] = self.hparams["block_configs"]
            assert self.block_count == len(_block_configs)
            self._num_kv_heads = list()
            self._num_heads = list()
            _ffn_multipliers = list()
            # ***linear attention layer***
            # if n_heads_in_group is None and replace_with_linear is True
            # then _num_kv_heads[il] is 0 and _num_heads[il] is num_attention_heads
            # ***attention-free layer***
            # if n_heads_in_group is None and replace_with_linear is False
            # then _num_kv_heads[il] is 0 and _num_heads[il] is 0
            # ***normal attention-layer***
            # if n_heads_in_group is not None, then
            # _num_kv_heads[il] is num_attention_head // n_heads_in_group and
            # _num_heads[il] is num_attention_head
            # ***dummy layer*** for nemotron 253B
            # if n_heads_in_group is None and ffn_mult is None
            # then _num_kv_heads[il] is 0 and _num_heads[il] is 0 and _ffn_dims is 0
            for il in range(len(_block_configs)):
                if _block_configs[il]["attention"]["n_heads_in_group"] is None:
                    if _block_configs[il]["attention"]["replace_with_linear"] is True:
                        self._num_kv_heads.append(0)
                        self._num_heads.append(self.hparams["num_attention_heads"])
                    else:
                        self._num_kv_heads.append(0)
                        self._num_heads.append(0)
                else:
                    self._num_kv_heads.append(self.hparams["num_attention_heads"] // _block_configs[il]["attention"]["n_heads_in_group"])
                    self._num_heads.append(self.hparams["num_attention_heads"])
                if _block_configs[il]["ffn"]["ffn_mult"] is None: # dummy layer
                    _ffn_multipliers.append(0.0)
                else:
                    _ffn_multipliers.append(_block_configs[il]["ffn"]["ffn_mult"])
            assert self.block_count == len(self._num_kv_heads)
            assert self.block_count == len(self._num_heads)
            assert self.block_count == len(_ffn_multipliers)
            assert isinstance(self._num_kv_heads, list) and isinstance(self._num_kv_heads[0], int)
            assert isinstance(self._num_heads, list) and isinstance(self._num_heads[0], int)
            assert isinstance(_ffn_multipliers, list) and isinstance(_ffn_multipliers[0], float)
            self._ffn_dims: list[int] = [
                DeciModel._ffn_mult_to_intermediate_size(multiplier, self.hparams["hidden_size"])
                for multiplier in _ffn_multipliers
            ]

    def set_vocab(self):
        # Please change tokenizer_config.json of Llama-3_1-Nemotron-51B's
        # eos_token from '|eot_id|' to '|end_of_text|'
        if self.hparams.get("vocab_size", 128256) == 128256:
            tokens, toktypes, tokpre = self.get_vocab_base()
            self.gguf_writer.add_tokenizer_model("gpt2")
            self.gguf_writer.add_tokenizer_pre(tokpre)
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_types(toktypes)

            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
            special_vocab.add_to_gguf(self.gguf_writer)
        else:
            # DeciLM-7B
            self._set_vocab_llama_hf()

    def set_gguf_parameters(self):
        if "block_configs" in self.hparams: # Llama-3_1-Nemotron-51B
            assert self.block_count == len(self._num_kv_heads)
            assert self.block_count == len(self._num_heads)
            assert self.block_count == len(self._ffn_dims)
            if (rope_theta := self.hparams.get("rope_theta")) is not None:
                self.gguf_writer.add_rope_freq_base(rope_theta)
            self.gguf_writer.add_head_count_kv(self._num_kv_heads)
            self.gguf_writer.add_head_count(self._num_heads)
            self.gguf_writer.add_feed_forward_length(self._ffn_dims)
            self.gguf_writer.add_block_count(self.block_count)
            self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
            self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
            self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
            self.gguf_writer.add_key_length(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
            self.gguf_writer.add_value_length(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
            self.gguf_writer.add_file_type(self.ftype)
        else: # DeciLM-7B
            super().set_gguf_parameters()
            if "num_key_value_heads_per_layer" in self.hparams: # DeciLM-7B
                self._num_kv_heads: list[int] = self.hparams["num_key_value_heads_per_layer"]
                assert self.block_count == len(self._num_kv_heads)
                self.gguf_writer.add_head_count_kv(self._num_kv_heads)
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        if "head_dim" in hparams:
            rope_dim = hparams["head_dim"]
        else:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(rope_dim)

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "linear":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        if bid is not None:
            if "num_key_value_heads_per_layer" in self.hparams:
                n_kv_head = self.hparams["num_key_value_heads_per_layer"][bid]
            elif "block_configs" in self.hparams:
                n_kv_head = self._num_kv_heads[bid]
                n_head = self._num_heads[bid]
            else:
                n_kv_head = self.hparams.get("num_key_value_heads")
        else:
            n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = DeciModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = DeciModel.permute(data_torch, n_head, n_kv_head)
        return [(self.map_tensor_name(name), data_torch)]

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if rope_scaling := self.find_hparam(["rope_scaling"], optional=True):
            if rope_scaling.get("rope_type", '').lower() == "llama3":
                base = self.hparams.get("rope_theta", 10000.0)
                dim = self.hparams.get("head_dim", self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
                freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

                factor = rope_scaling.get("factor", 8.0)
                low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
                high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
                old_context_len = self.hparams.get("original_max_position_embeddings", 8192)

                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor
                assert low_freq_wavelen != high_freq_wavelen

                rope_factors = []
                for freq in freqs:
                    wavelen = 2 * math.pi / freq
                    if wavelen < high_freq_wavelen:
                        rope_factors.append(1)
                    elif wavelen > low_freq_wavelen:
                        rope_factors.append(factor)
                    else:
                        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                        rope_factors.append(1 / ((1 - smooth) / factor + smooth))

                yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.tensor(rope_factors, dtype=torch.float32))

    def prepare_tensors(self):
        super().prepare_tensors()


@Model.register("BitnetForCausalLM")
@Model.register("BitNetForCausalLM")
class BitnetModel(Model):
    model_arch = gguf.MODEL_ARCH.BITNET

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        self.gguf_writer.add_rope_scaling_factor(1.0)

    def weight_quant(self, weight):
        dtype = weight.dtype
        weight = weight.float()
        s = 1 / weight.abs().mean().clamp(min=1e-5)
        weight = (weight * s).round().clamp(-1, 1) / s
        scale = weight.abs().max().unsqueeze(0)
        weight = torch.where(weight.abs().less(1e-6), 0, weight).type(dtype)
        weight = torch.sign(weight).type(dtype)
        return weight.type(dtype), scale.type(torch.float32)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # transform weight into 1/0/-1 (in fp32)
        if name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight",
                          "down_proj.weight", "up_proj.weight", "gate_proj.weight",
                          "o_proj.weight")):
            weight_torch, scale_torch = self.weight_quant(data_torch)

        tensors: list[tuple[str, Tensor]] = []

        if name.endswith("q_proj.weight"):
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), weight_torch))
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid, suffix=".scale"), scale_torch))
        elif name.endswith("k_proj.weight"):
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), weight_torch))
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid, suffix=".scale"), scale_torch))
        elif name.endswith("v_proj.weight"):
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid), weight_torch))
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid, suffix=".scale"), scale_torch))
        elif name.endswith("o_proj.weight"):
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_OUT, bid), weight_torch))
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_OUT, bid, suffix=".scale"), scale_torch))
        elif name.endswith("up_proj.weight"):
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP, bid), weight_torch))
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP, bid, suffix=".scale"), scale_torch))
        elif name.endswith("down_proj.weight"):
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN, bid), weight_torch))
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN, bid, suffix=".scale"), scale_torch))
        elif name.endswith("gate_proj.weight"):
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid), weight_torch))
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid, suffix=".scale"), scale_torch))

        if len(tensors) == 0:
            tensors.append((self.map_tensor_name(name), data_torch))

        return tensors


@Model.register("GrokForCausalLM", "Grok1ForCausalLM")
class GrokModel(Model):
    model_arch = gguf.MODEL_ARCH.GROK

    def set_vocab(self):
        if (self.dir_model / 'tokenizer.model').is_file():
            self._set_vocab_sentencepiece()
            return

        if not (self.dir_model / 'tokenizer.json').is_file() or not (self.dir_model / 'chat_template.jinja').is_file():
            logger.error('Error: Missing vocab and chat template, download files from https://huggingface.co/alvarobartt/grok-2-tokenizer')
            sys.exit(1)

        self._set_vocab_gpt2()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_attn_logit_softcapping(self.hparams.get("attn_logit_softcapping", 30.0))
        self.gguf_writer.add_router_logit_softcapping(self.hparams.get("router_logit_softcapping", 30.0))
        if (final_logit_softcap := self.hparams.get("final_logit_softcapping")):
            self.gguf_writer.add_final_logit_softcapping(final_logit_softcap)

        if (rope_dim := self.hparams.get("head_dim")) is None:
            rope_dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]

        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)

        # Treat "original" as "yarn", seems to have been a mistake
        if self.hparams.get("rope_type") in ("yarn", "original"):
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(self.hparams["scaling_factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(self.hparams["original_max_position_embeddings"])
            self.gguf_writer.add_rope_scaling_yarn_ext_factor(self.hparams["extrapolation_factor"])
            self.gguf_writer.add_rope_scaling_yarn_attn_factor(self.hparams["attn_factor"])
            self.gguf_writer.add_rope_scaling_yarn_beta_fast(self.hparams["beta_fast"])
            self.gguf_writer.add_rope_scaling_yarn_beta_slow(self.hparams["beta_slow"])

        if temp_len := self.hparams.get("attn_temperature_len"):
            self.gguf_writer.add_attn_temperature_length(temp_len)

        self.gguf_writer.add_attn_output_scale(self.hparams.get("attn_output_multiplier", rope_dim**-0.5))
        self.gguf_writer.add_embedding_scale(self.hparams["embedding_multiplier_scale"])
        self.gguf_writer.add_logit_scale(self.hparams["output_multiplier_scale"])

    _experts: list[dict[str, list[Tensor]]] | None = None
    _cur_expert = ""

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        tensors: list[tuple[str, Tensor]] = []
        is_expert = ".moe." in name or ".block_sparse_moe.experts." in name

        if not is_expert:
            tensors.append((self.map_tensor_name(name), data_torch))

        # process the experts separately
        if is_expert or self._cur_expert:
            n_experts = self.hparams["num_local_experts"]

            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            # concatenate split tensors
            if name in self._experts[bid]:
                self._cur_expert = name
                self._experts[bid][name].append(data_torch)
                return []
            elif is_expert:
                self._cur_expert = name
                self._experts[bid][name] = [data_torch]
                return []
            else:
                self._cur_expert = ""

            for bid in range(self.block_count):
                if len(self._experts[bid]) >= n_experts * 3:
                    # merge the experts into a single 3d tensor
                    for wid in [("linear", "w1", 0), ("linear_1", "w2", 1), ("linear_v", "w3", 0)]:
                        datas: list[Tensor] = []

                        for xid in range(n_experts):
                            ename = f"transformer.decoder_layer.{bid}.moe.{xid}.{wid[0]}.weight"
                            if ename not in self._experts[bid]:
                                ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid[1]}.weight"
                            tensor_list = self._experts[bid][ename]
                            datas.append(torch.cat(tensor_list, dim=wid[2]) if len(tensor_list) > 1 else tensor_list[0])
                            del self._experts[bid][ename]

                        data_torch = torch.stack(datas, dim=0)

                        merged_name = f"transformer.decoder_layer.{bid}.moe.{wid[0]}.weight"

                        new_name = self.map_tensor_name(merged_name)

                        yield (new_name, data_torch)

        yield from tensors


@Model.register("DbrxForCausalLM")
class DbrxModel(Model):
    model_arch = gguf.MODEL_ARCH.DBRX

    def set_gguf_parameters(self):
        ffn_config = self.hparams["ffn_config"]
        attn_config = self.hparams["attn_config"]
        self.gguf_writer.add_block_count(self.hparams["n_layers"])

        self.gguf_writer.add_context_length(self.hparams["max_seq_len"])
        self.gguf_writer.add_embedding_length(self.hparams["d_model"])
        self.gguf_writer.add_feed_forward_length(ffn_config["ffn_hidden_size"])

        self.gguf_writer.add_head_count(self.hparams["n_heads"])
        self.gguf_writer.add_head_count_kv(attn_config["kv_n_heads"])

        self.gguf_writer.add_rope_freq_base(attn_config["rope_theta"])

        self.gguf_writer.add_clamp_kqv(attn_config["clip_qkv"])

        self.gguf_writer.add_expert_count(ffn_config["moe_num_experts"])
        self.gguf_writer.add_expert_used_count(ffn_config["moe_top_k"])

        self.gguf_writer.add_layer_norm_eps(1e-5)

        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"gguf: file type = {self.ftype}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        n_expert = self.hparams["ffn_config"]["moe_num_experts"]
        n_ff = self.hparams["ffn_config"]["ffn_hidden_size"]
        n_embd = self.hparams["d_model"]

        # Specific behavior for experts tensors: suffix .weight, view as 3D and transpose
        # original implementation expects (n_expert, n_ff, n_embd) for all experts weights
        # But llama.cpp moe graph works differently
        # AND the dimensions in ggml are typically in the reverse order of the pytorch dimensions
        # so (n_expert, n_ff, n_embd) in pytorch is {n_embd, n_ff, n_expert} in ggml_tensor
        exp_tensor_names = {"ffn.experts.mlp.w1": None,       # LLM_TENSOR_FFN_GATE_EXPS ggml_tensor->ne{n_embd, n_ff,   n_expert}
                            "ffn.experts.mlp.w2": (0, 2, 1),  # LLM_TENSOR_FFN_DOWN_EXPS ggml_tensor->ne{n_ff,   n_embd, n_expert}
                            "ffn.experts.mlp.v1": None}       # LLM_TENSOR_FFN_UP_EXPS   ggml_tensor->ne{n_embd, n_ff,   n_expert}
        experts = False

        for exp_tensor_name in exp_tensor_names.keys():
            if name.find(exp_tensor_name) != -1 and name.find(".weight") == -1:
                experts = True
                data_torch = data_torch.view(n_expert, n_ff, n_embd)
                if (permute_tensor := exp_tensor_names[exp_tensor_name]) is not None:
                    data_torch = data_torch.permute(*permute_tensor)
                break

        # map tensor names
        # In MoE models the ffn tensors are typically most of the model weights,
        # and need to be quantizable. Quantize expects tensor names to be suffixed by .weight.
        # Every other model has the weight names ending in .weight,
        # let's assume that is the convention which is not the case for dbrx:
        # https://huggingface.co/databricks/dbrx-instruct/blob/main/model.safetensors.index.json#L15
        new_name = self.map_tensor_name(name if not experts else name + ".weight", try_suffixes=(".weight",))

        return [(new_name, data_torch)]

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        del name, new_name, bid  # unused

        return n_dims > 1


@Model.register("MiniCPMForCausalLM")
class MiniCPMModel(Model):
    model_arch = gguf.MODEL_ARCH.MINICPM

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_file_type(self.ftype)

    def set_vocab(self):
        self._set_vocab_llama_hf()

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        # HF models permute some of the tensors, so we need to undo that
        if name.endswith(("q_proj.weight")):
            data_torch = self._reverse_hf_permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight")):
            data_torch = self._reverse_hf_permute(data_torch, n_head, n_kv_head)

        return [(self.map_tensor_name(name), data_torch)]


@Model.register("QWenLMHeadModel")
class QwenModel(Model):
    model_arch = gguf.MODEL_ARCH.QWEN

    @staticmethod
    def token_bytes_to_string(b):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        byte_encoder = bytes_to_unicode()
        return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])

    @staticmethod
    def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int | None = None) -> list[bytes]:
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        return parts

    def set_vocab(self):
        self._set_vocab_qwen()

    def set_gguf_parameters(self):
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_block_count(self.hparams["num_hidden_layers"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_freq_base(self.hparams["rotary_emb_base"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)


@Model.register("Qwen2ForCausalLM")
class Qwen2Model(Model):
    model_arch = gguf.MODEL_ARCH.QWEN2

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()


@Model.register("Qwen2MoeForCausalLM")
class Qwen2MoeModel(Model):
    model_arch = gguf.MODEL_ARCH.QWEN2MOE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (n_experts := self.hparams.get("num_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")
        if (shared_expert_intermediate_size := self.hparams.get('shared_expert_intermediate_size')) is not None:
            self.gguf_writer.add_expert_shared_feed_forward_length(shared_expert_intermediate_size)
            logger.info(f"gguf: expert shared feed forward length = {shared_expert_intermediate_size}")

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        if name.find("experts") != -1:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")

@Model.register("Qwen3ForCausalLM")
class Qwen3Model(Qwen2Model):
    model_arch = gguf.MODEL_ARCH.QWEN3

@Model.register("Qwen3MoeForCausalLM")
class Qwen3MoeModel(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN3MOE


@Model.register("MellumForCausalLM")
class MellumModel(Model):
    model_arch = gguf.MODEL_ARCH.MELLUM

    def set_vocab(self):
        tokenizer_path = self.dir_model / "tokenizer.json"
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_json = json.load(f)

        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        class TokenizerShim:
            def encode(self, text: str) -> list[int]:
                return tokenizer.encode(text).ids

        vocab: dict[str, int] = tokenizer_json["model"]["vocab"]
        vocab_size = self.hparams.get("vocab_size", len(vocab))
        assert max(vocab.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(TokenizerShim())
        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in vocab.items()}
        added_vocab = {
            item["content"]: item
            for item in tokenizer_json.get("added_tokens", [])
            if isinstance(item.get("content"), str)
        }

        tokens: list[str] = []
        toktypes: list[int] = []
        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
                continue

            token = reverse_vocab[i]
            added_token = added_vocab.get(token)
            if added_token is not None:
                if added_token.get("special", False) or self.does_token_look_special(token):
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    token = token.replace("\u2581", " ")
                    toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                toktypes.append(gguf.TokenType.NORMAL)
            tokens.append(token)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if self.hparams.get("num_local_experts") is None and (n_experts := self.hparams.get("num_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)

        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")

        use_sliding_window = self.hparams.get("use_sliding_window")
        sliding_window = self.hparams.get("sliding_window")
        if (use_sliding_window is True or use_sliding_window is None) and sliding_window is not None:
            self.gguf_writer.add_sliding_window(sliding_window)
            logger.info(f"gguf: sliding window = {sliding_window}")
            self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in self.hparams["layer_types"]])
            logger.info(f"gguf: sliding window pattern length = {len(self.hparams['layer_types'])}")

        rope_parameters = self.hparams.get("rope_parameters", {})
        if full_attention_rope := rope_parameters.get("full_attention"):
            if rope_theta := full_attention_rope.get("rope_theta"):
                self.gguf_writer.add_rope_freq_base(rope_theta)
                logger.info(f"gguf: rope freq base = {rope_theta}")

            if full_attention_rope.get("rope_type") == "yarn":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)

                if factor := full_attention_rope.get("factor"):
                    self.gguf_writer.add_rope_scaling_factor(factor)
                if original_context_length := full_attention_rope.get("original_max_position_embeddings"):
                    self.gguf_writer.add_rope_scaling_orig_ctx_len(original_context_length)
                if attention_factor := full_attention_rope.get("attention_factor"):
                    self.gguf_writer.add_rope_scaling_yarn_attn_factor(attention_factor)
                if beta_fast := full_attention_rope.get("beta_fast"):
                    self.gguf_writer.add_rope_scaling_yarn_beta_fast(beta_fast)
                if beta_slow := full_attention_rope.get("beta_slow"):
                    self.gguf_writer.add_rope_scaling_yarn_beta_slow(beta_slow)

        if sliding_attention_rope := rope_parameters.get("sliding_attention"):
            if rope_theta_swa := sliding_attention_rope.get("rope_theta"):
                self.gguf_writer.add_rope_freq_base_swa(rope_theta_swa)
                logger.info(f"gguf: rope freq base swa = {rope_theta_swa}")

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "experts" in name:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    tensors.append((self.map_tensor_name(merged_name), data_torch))
                return tensors
            return []

        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@Model.register("Ernie4_5_ForCausalLM", "Ernie4_5ForCausalLM")
class Ernie4_5Model(Model):
    model_arch = gguf.MODEL_ARCH.ERNIE4_5

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        num_heads = self.hparams["num_attention_heads"]
        num_kv_heads = self.hparams["num_key_value_heads"]
        if (head_dim := self.hparams.get("head_dim")) is None:
            head_dim = self.hparams["hidden_size"] // num_heads

        if "ernie." in name:
            name = name.replace("ernie.", "model.")
        # split the qkv weights
        # qkv_proj shape: [(num_heads + 2 * num_kv_heads) * head_dim, hidden_size]
        if "qkv_proj" in name:
            name_q = name.replace("qkv_proj.weight", "q_proj.weight")
            name_k = name.replace("qkv_proj.weight", "k_proj.weight")
            name_v = name.replace("qkv_proj.weight", "v_proj.weight")
            total_q_dim = num_heads * head_dim
            total_k_dim = num_kv_heads * head_dim
            total_v_dim = num_kv_heads * head_dim
            q_proj_weight, k_proj_weight, v_proj_weight = data_torch.split([total_q_dim, total_k_dim, total_v_dim], dim=0)
            return [
                (self.map_tensor_name(name_q), q_proj_weight),
                (self.map_tensor_name(name_k), k_proj_weight),
                (self.map_tensor_name(name_v), v_proj_weight)
            ]
        # split the up_gate_proj into gate and up
        # up_gate_proj shape: [2 * intermediate_size, hidden_size]
        if "up_gate_proj" in name:
            name_up = name.replace("up_gate_proj.weight", "up_proj.weight")
            name_gate = name.replace("up_gate_proj.weight", "gate_proj.weight")
            dim_half = data_torch.shape[0] // 2
            gate_proj_weight, up_proj_weight = data_torch.split(dim_half, dim=0)
            return [
                (self.map_tensor_name(name_gate), gate_proj_weight),
                (self.map_tensor_name(name_up), up_proj_weight)
            ]
        return [(self.map_tensor_name(name), data_torch)]


@Model.register("Ernie4_5_MoeForCausalLM")
class Ernie4_5MoeModel(Ernie4_5Model):
    model_arch = gguf.MODEL_ARCH.ERNIE4_5_MOE
    _experts: list[dict[str, Tensor]] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._experts = [{} for _ in range(self.block_count)]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_expert_count(self.hparams["moe_num_experts"])
        self.gguf_writer.add_expert_used_count(self.hparams["moe_k"])
        self.gguf_writer.add_interleave_moe_layer_step(self.hparams["moe_layer_interval"])
        self.gguf_writer.add_leading_dense_block_count(self.hparams["moe_layer_start_index"])
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
        if (shared_expert_count := self.hparams.get('moe_num_shared_experts')) is not None:
            self.gguf_writer.add_expert_shared_count(shared_expert_count)
            if shared_expert_count > 0 and (shared_expert_intermediate_size := self.hparams.get('intermediate_size')) is not None and (num_key_value_heads := self.hparams.get('num_key_value_heads')) is not None:
                self.gguf_writer.add_expert_shared_feed_forward_length(shared_expert_intermediate_size // num_key_value_heads)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Modify correction bias name as in DeepseekV2
        if name.endswith("e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")

        # skip Multi-Token Prediction (MTP) layers (again, same as DeepseekV2)
        match = re.match(r"model.mtp_block.(\d+)", name)
        if match:
            return []

        # skip all other MTP tensors for now
        match = re.match(r"model.mtp_emb_norm.(\d+)", name)
        if match:
            return []

        match = re.match(r"model.mtp_hidden_norm.(\d+)", name)
        if match:
            return []

        match = re.match(r"model.mtp_linear_proj.(\d+)", name)
        if match:
            return []

        # process the experts separately
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["moe_num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for w_name in ["gate_proj", "up_proj", "down_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename_to_retrieve = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename_to_retrieve])
                        del self._experts[bid][ename_to_retrieve]

                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    new_name = self.map_tensor_name(merged_name)
                    tensors.append((new_name, data_torch))

                return tensors
            else:
                return []
        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@Model.register("GPT2LMHeadModel")
class GPT2Model(Model):
    model_arch = gguf.MODEL_ARCH.GPT2

    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_context_length(self.hparams["n_ctx"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        tensors: list[tuple[str, Tensor]] = []

        # we don't need these
        if name.endswith((".attn.bias", ".attn.masked_bias")):
            return tensors

        if name.endswith((".c_attn.weight", ".c_proj.weight", ".c_fc.weight", ".c_proj.weight")):
            data_torch = data_torch.transpose(1, 0)

        new_name = self.map_tensor_name(name)

        tensors.append((new_name, data_torch))

        # note: GPT2 output is tied to (same as) wte in original model
        if new_name == self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD):
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch))

        return tensors


@Model.register("PhiForCausalLM")
class Phi2Model(Model):
    model_arch = gguf.MODEL_ARCH.PHI2

    def set_gguf_parameters(self):
        block_count = self.find_hparam(["num_hidden_layers", "n_layer"])

        rot_pct = self.find_hparam(["partial_rotary_factor"])
        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        n_head = self.find_hparam(["num_attention_heads", "n_head"])

        self.gguf_writer.add_context_length(self.find_hparam(["n_positions", "max_position_embeddings"]))

        self.gguf_writer.add_embedding_length(n_embd)
        self.gguf_writer.add_feed_forward_length(4 * n_embd)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head)
        self.gguf_writer.add_layer_norm_eps(self.find_hparam(["layer_norm_epsilon", "layer_norm_eps"]))
        self.gguf_writer.add_rope_dimension_count(int(rot_pct * n_embd) // n_head)
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_add_bos_token(False)


@Model.register("Phi3ForCausalLM")
class Phi3MiniModel(Model):
    model_arch = gguf.MODEL_ARCH.PHI3

    def set_vocab(self):
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / 'tokenizer.model'

        if not tokenizer_path.is_file():
            raise ValueError(f'Error: Missing {tokenizer_path}')

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        for token_id in range(tokenizer.vocab_size()):

            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                for key in added_tokens_json:
                    token_id = added_tokens_json[key]
                    if token_id >= vocab_size:
                        logger.debug(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                        continue

                    tokens[token_id] = key.encode("utf-8")
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                added_tokens_decoder = tokenizer_config_json.get("added_tokens_decoder", {})
                for token_id, foken_data in added_tokens_decoder.items():
                    token_id = int(token_id)
                    token = foken_data["content"].encode("utf-8")
                    if toktypes[token_id] != SentencePieceTokenTypes.UNUSED:
                        if tokens[token_id] != token:
                            logger.warning(f'replacing token {token_id}: {tokens[token_id].decode("utf-8")!r} -> {token.decode("utf-8")!r}')
                    tokens[token_id] = token
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED
                    if foken_data.get("special"):
                        toktypes[token_id] = SentencePieceTokenTypes.CONTROL

        tokenizer_file = self.dir_model / 'tokenizer.json'
        if tokenizer_file.is_file():
            with open(tokenizer_file, "r", encoding="utf-8") as f:
                tokenizer_json = json.load(f)
                added_tokens = tokenizer_json.get("added_tokens", [])
                for foken_data in added_tokens:
                    token_id = int(foken_data["id"])
                    token = foken_data["content"].encode("utf-8")
                    if toktypes[token_id] != SentencePieceTokenTypes.UNUSED:
                        if tokens[token_id] != token:
                            logger.warning(f'replacing token {token_id}: {tokens[token_id].decode("utf-8")!r} -> {token.decode("utf-8")!r}')
                    tokens[token_id] = token
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED
                    if foken_data.get("special"):
                        toktypes[token_id] = SentencePieceTokenTypes.CONTROL

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        block_count = self.find_hparam(["num_hidden_layers", "n_layer"])

        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        n_head_kv = self.find_hparam(["num_key_value_heads", "n_head_kv"])
        rms_eps = self.find_hparam(["rms_norm_eps"])
        max_pos_embds = self.find_hparam(["n_positions", "max_position_embeddings"])
        orig_max_pos_embds = self.find_hparam(["original_max_position_embeddings"])
        rope_dims = n_embd // n_head

        self.gguf_writer.add_context_length(max_pos_embds)
        self.gguf_writer.add_rope_scaling_orig_ctx_len(orig_max_pos_embds)
        self.gguf_writer.add_embedding_length(n_embd)
        self.gguf_writer.add_feed_forward_length(self.find_hparam(["intermediate_size"]))
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_layer_norm_rms_eps(rms_eps)
        self.gguf_writer.add_rope_dimension_count(rope_dims)
        self.gguf_writer.add_rope_freq_base(self.find_hparam(["rope_theta"]))
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_sliding_window(self.find_hparam(["sliding_window"]))

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        max_pos_embds = self.find_hparam(["n_positions", "max_position_embeddings"])
        orig_max_pos_embds = self.find_hparam(["original_max_position_embeddings"])
        rope_dims = n_embd // n_head

        # write rope scaling for long context (128k) model
        rope_scaling = self.find_hparam(['rope_scaling'], True)
        if rope_scaling is None:
            return

        scale = max_pos_embds / orig_max_pos_embds

        rope_scaling_type = rope_scaling.get('type', '').lower()
        if len(rope_scaling_type) == 0:
            raise KeyError('Missing the required key rope_scaling.type')

        if rope_scaling_type == 'su' or rope_scaling_type == 'longrope':
            attn_factor = math.sqrt(1 + math.log(scale) / math.log(orig_max_pos_embds)) if scale > 1.0 else 1.0
        elif rope_scaling_type == 'yarn':
            attn_factor = 0.1 * math.log(scale) + 1.0 if scale > 1.0 else 1.0
        else:
            raise NotImplementedError(f'The rope scaling type {rope_scaling_type} is not supported yet')

        self.gguf_writer.add_rope_scaling_attn_factors(attn_factor)

        long_factors = rope_scaling.get('long_factor', None)
        short_factors = rope_scaling.get('short_factor', None)

        if long_factors is None or short_factors is None:
            raise KeyError('Missing the required key rope_scaling.long_factor or rope_scaling_short_factor')

        if len(long_factors) != len(short_factors) or len(long_factors) != rope_dims / 2:
            raise ValueError(f'The length of rope long and short factors must be {rope_dims / 2}')

        yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FACTORS_LONG), torch.tensor(long_factors, dtype=torch.float32))
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FACTORS_SHORT), torch.tensor(short_factors, dtype=torch.float32))


@Model.register("PlamoForCausalLM")
class PlamoModel(Model):
    model_arch = gguf.MODEL_ARCH.PLAMO

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]

        self.gguf_writer.add_context_length(4096)  # not in config.json
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(5)  # hparams["num_key_value_heads"]) is wrong
        self.gguf_writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])
        self.gguf_writer.add_file_type(self.ftype)

    def shuffle_attn_q_weight(self, data_torch):
        assert data_torch.size() == (5120, 5120)
        data_torch = data_torch.reshape(8, 5, 128, 5120)
        data_torch = torch.permute(data_torch, (1, 0, 2, 3))
        data_torch = torch.reshape(data_torch, (5120, 5120))
        return data_torch

    def shuffle_attn_output_weight(self, data_torch):
        assert data_torch.size() == (5120, 5120)
        data_torch = data_torch.reshape(5120, 8, 5, 128)
        data_torch = torch.permute(data_torch, (0, 2, 1, 3))
        data_torch = torch.reshape(data_torch, (5120, 5120))
        return data_torch

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        new_name = self.map_tensor_name(name)

        # shuffle for broadcasting of gqa in ggml_mul_mat
        if new_name.endswith("attn_q.weight"):
            data_torch = self.shuffle_attn_q_weight(data_torch)
        elif new_name.endswith("attn_output.weight"):
            data_torch = self.shuffle_attn_output_weight(data_torch)

        return [(new_name, data_torch)]


@Model.register("CodeShellForCausalLM")
class CodeShellModel(Model):
    model_arch = gguf.MODEL_ARCH.CODESHELL

    def set_gguf_parameters(self):
        block_count = self.hparams["n_layer"]

        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_query_groups"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_rope_freq_base(10000.0)
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        self.gguf_writer.add_rope_scaling_factor(1.0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        new_name = self.map_tensor_name(name)

        tensors: list[tuple[str, Tensor]] = [(new_name, data_torch)]

        if new_name == self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD):
            assert self.tensor_names is not None

            if all(s not in self.tensor_names for s in ("lm_head.weight", "output.weight")):
                # copy tok_embd.weight to output.weight
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch))

        return tensors


@Model.register("InternLM2ForCausalLM")
class InternLM2Model(Model):
    model_arch = gguf.MODEL_ARCH.INTERNLM2

    def set_vocab(self):
        # (TODO): Is there a better way?
        # Copy from _set_vocab_sentencepiece, The only difference is that we will treat the character
        # \x00 specially and convert it into an emoji character to prevent it from being mistakenly
        # recognized as an empty string in C++.
        from sentencepiece import SentencePieceProcessor
        from sentencepiece import sentencepiece_model_pb2 as model

        tokenizer_path = self.dir_model / 'tokenizer.model'

        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []

        if not tokenizer_path.is_file():
            logger.error(f'Error: Missing {tokenizer_path}')
            sys.exit(1)

        sentencepiece_model = model.ModelProto()  # pyright: ignore[reportAttributeAccessIssue]
        sentencepiece_model.ParseFromString(open(tokenizer_path, "rb").read())
        add_prefix = sentencepiece_model.normalizer_spec.add_dummy_prefix

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        for token_id in range(vocab_size):
            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)
            if text == b"\x00":
                # (TODO): fixme
                # Hack here and replace the \x00 characters.
                logger.warning(f"InternLM2 convert token '{text}' to '🐉'!")
                text = "🐉".encode("utf-8")

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE
            # take care of ununsed raw token
            if piece.startswith('[UNUSED'):
                toktype = SentencePieceTokenTypes.UNUSED

            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                for key in added_tokens_json:
                    tokens.append(key.encode("utf-8"))
                    scores.append(-1000.0)
                    toktypes.append(SentencePieceTokenTypes.USER_DEFINED)

        chat_eos_token = '<|im_end|>'
        chat_eos_token_id = None

        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                added_tokens_decoder = tokenizer_config_json.get("added_tokens_decoder", {})
                for token_id, foken_data in added_tokens_decoder.items():
                    token_id = int(token_id)
                    token = foken_data["content"]
                    if token == chat_eos_token:
                        chat_eos_token_id = token_id
                    token = token.encode("utf-8")
                    if toktypes[token_id] != SentencePieceTokenTypes.UNUSED:
                        if tokens[token_id] != token:
                            logger.warning(f'replacing token {token_id}: {tokens[token_id].decode("utf-8")!r} -> {token.decode("utf-8")!r}')
                    tokens[token_id] = token
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED
                    if foken_data.get("special"):
                        toktypes[token_id] = SentencePieceTokenTypes.CONTROL

        tokenizer_file = self.dir_model / 'tokenizer.json'
        if tokenizer_file.is_file():
            with open(tokenizer_file, "r", encoding="utf-8") as f:
                tokenizer_json = json.load(f)
                added_tokens = tokenizer_json.get("added_tokens", [])
                for foken_data in added_tokens:
                    token_id = int(foken_data["id"])
                    token = foken_data["content"]
                    if token == chat_eos_token:
                        chat_eos_token_id = token_id
                    token = token.encode("utf-8")
                    if toktypes[token_id] != SentencePieceTokenTypes.UNUSED:
                        if tokens[token_id] != token:
                            logger.warning(f'replacing token {token_id}: {tokens[token_id].decode("utf-8")!r} -> {token.decode("utf-8")!r}')
                    tokens[token_id] = token
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED
                    if foken_data.get("special"):
                        toktypes[token_id] = SentencePieceTokenTypes.CONTROL

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_add_space_prefix(add_prefix)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        old_eos = special_vocab.special_token_ids["eos"]
        if chat_eos_token_id is not None:
            # For the chat model, we replace the eos with '<|im_end|>'.
            # TODO: this is a hack, should be fixed
            #       https://github.com/ggerganov/llama.cpp/pull/6745#issuecomment-2067687048
            special_vocab.special_token_ids["eos"] = chat_eos_token_id
            logger.warning(f"Replace eos:{old_eos} with a special token:{chat_eos_token_id}"
                           " in chat mode so that the conversation can end normally.")

        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_block_count(self.hparams["num_hidden_layers"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_freq_base(self.hparams["rope_theta"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"])
        self.gguf_writer.add_file_type(self.ftype)
        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "linear":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        num_heads = self.hparams["num_attention_heads"]
        num_kv_heads = self.hparams["num_key_value_heads"]
        n_embd = self.hparams["hidden_size"]
        q_per_kv = num_heads // num_kv_heads
        head_dim = n_embd // num_heads
        num_groups = num_heads // q_per_kv

        if bid is not None and f"model.layers.{bid}.attention.wqkv" in name:
            qkv = data_torch

            qkv = qkv.reshape((num_groups, q_per_kv + 2, head_dim, n_embd))
            q, k, v = qkv[:, : q_per_kv], qkv[:, -2], qkv[:, -1]

            # The model weights of q and k equire additional reshape.
            q = LlamaModel.permute(q.reshape((-1, q.shape[-1])), num_heads, num_heads)
            k = LlamaModel.permute(k.reshape((-1, k.shape[-1])), num_heads, num_kv_heads)
            v = v.reshape((-1, v.shape[-1]))

            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), q),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), k),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid), v),
            ]
        else:
            return [(self.map_tensor_name(name), data_torch)]


@Model.register("BertModel", "CamembertModel")
class BertModel(Model):
    model_arch = gguf.MODEL_ARCH.BERT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_causal_attention(False)

        # get pooling path
        pooling_path = None
        module_path = self.dir_model / "modules.json"
        if module_path.is_file():
            with open(module_path, encoding="utf-8") as f:
                modules = json.load(f)
            for mod in modules:
                if mod["type"] == "sentence_transformers.models.Pooling":
                    pooling_path = mod["path"]
                    break

        # get pooling type
        if pooling_path is not None:
            with open(self.dir_model / pooling_path / "config.json", encoding="utf-8") as f:
                pooling = json.load(f)
            if pooling["pooling_mode_mean_tokens"]:
                pooling_type = gguf.PoolingType.MEAN
            elif pooling["pooling_mode_cls_token"]:
                pooling_type = gguf.PoolingType.CLS
            else:
                raise NotImplementedError("Only MEAN and CLS pooling types supported")
            self.gguf_writer.add_pooling_type(pooling_type)

    def set_vocab(self):
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.vocab_size = len(tokens)

        # we need this to validate the size of the token_type embeddings
        # though currently we are passing all zeros to the token_type embeddings
        self.gguf_writer.add_token_type_count(2)  # "Sequence A" or "Sequence B"

        # convert to phantom space vocab
        def phantom(tok):
            if tok.startswith("[") and tok.endswith("]"):
                return tok
            if tok.startswith("##"):
                return tok[2:]
            return "\u2581" + tok
        tokens = list(map(phantom, tokens))

        # add vocab to gguf
        self.gguf_writer.add_tokenizer_model("bert")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        # handle special tokens
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        # we are only using BERT for embeddings so we don't need the pooling layer
        if name in ("embeddings.position_ids", "pooler.dense.weight", "pooler.dense.bias"):
            return [] # we don't need these

        return [(self.map_tensor_name(name), data_torch)]


@Model.register("NomicBertModel")
class NomicBertModel(BertModel):
    model_arch = gguf.MODEL_ARCH.NOMIC_BERT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # the HF config claims n_ctx=8192, but it uses RoPE scaling
        self.hparams["n_ctx"] = 2048

        # SwigLU activation
        assert self.hparams["activation_function"] == "swiglu"
        # this doesn't do anything in the HF version
        assert self.hparams["causal"] is False
        # no bias tensors
        assert self.hparams["qkv_proj_bias"] is False
        assert self.hparams["mlp_fc1_bias"] is False
        assert self.hparams["mlp_fc2_bias"] is False
        # norm at end of layer
        assert self.hparams["prenorm"] is False
        # standard RoPE
        assert self.hparams["rotary_emb_fraction"] == 1.0
        assert self.hparams["rotary_emb_interleaved"] is False
        assert self.hparams["rotary_emb_scale_base"] is None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_freq_base(self.hparams["rotary_emb_base"])


@Model.register("XLMRobertaModel")
class XLMRobertaModel(BertModel):
    model_arch = gguf.MODEL_ARCH.BERT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # we need the pad_token_id to know how to chop down position_embd matrix
        if (pad_token_id := self.hparams.get("pad_token_id")) is not None:
            self._position_offset = 1 + pad_token_id
            if "max_position_embeddings" in self.hparams:
                self.hparams["max_position_embeddings"] -= self._position_offset
        else:
            self._position_offset = None

    def set_vocab(self):
        # to avoid TypeError: Descriptors cannot be created directly
        # exception when importing sentencepiece_model_pb2
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        from sentencepiece import SentencePieceProcessor
        from sentencepiece import sentencepiece_model_pb2 as model

        tokenizer_path = self.dir_model / 'sentencepiece.bpe.model'
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")

        sentencepiece_model = model.ModelProto()  # pyright: ignore[reportAttributeAccessIssue]
        sentencepiece_model.ParseFromString(open(tokenizer_path, "rb").read())
        assert sentencepiece_model.trainer_spec.model_type == 1  # UNIGRAM

        add_prefix = sentencepiece_model.normalizer_spec.add_dummy_prefix
        remove_whitespaces = sentencepiece_model.normalizer_spec.remove_extra_whitespaces
        precompiled_charsmap = sentencepiece_model.normalizer_spec.precompiled_charsmap

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        for token_id in range(tokenizer.vocab_size()):
            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]")
            for i in range(1, pad_count + 1):
                tokens.append(bytes(f"[PAD{i}]", encoding="utf-8"))
                scores.append(-1000.0)
                toktypes.append(SentencePieceTokenTypes.UNUSED)

        # realign tokens (see HF tokenizer code)
        tokens = [b'<s>', b'<pad>', b'</s>', b'<unk>'] + tokens[3:-1]
        scores = [0.0, 0.0, 0.0, 0.0] + scores[3:-1]
        toktypes = [
            SentencePieceTokenTypes.CONTROL,
            SentencePieceTokenTypes.CONTROL,
            SentencePieceTokenTypes.CONTROL,
            SentencePieceTokenTypes.UNKNOWN,
        ] + toktypes[3:-1]

        self.gguf_writer.add_tokenizer_model("t5")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_add_space_prefix(add_prefix)
        self.gguf_writer.add_token_type_count(1)
        self.gguf_writer.add_remove_extra_whitespaces(remove_whitespaces)
        if precompiled_charsmap:
            self.gguf_writer.add_precompiled_charsmap(precompiled_charsmap)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

        self.gguf_writer.add_add_bos_token(True)
        self.gguf_writer.add_add_eos_token(True)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # position embeddings start at pad_token_id + 1, so just chop down the weight tensor
        if name == "embeddings.position_embeddings.weight":
            if self._position_offset is not None:
                data_torch = data_torch[self._position_offset:,:]

        return super().modify_tensors(data_torch, name, bid)


@Model.register("GemmaForCausalLM")
class GemmaModel(Model):
    model_arch = gguf.MODEL_ARCH.GEMMA

    def set_vocab(self):
        self._set_vocab_sentencepiece()

        # TODO: these special tokens should be exported only for the CodeGemma family
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False,
                                          special_token_types = ['prefix', 'suffix', 'middle', 'fsep', 'eot'])
        special_vocab._set_special_token("prefix", 67)
        special_vocab._set_special_token("suffix", 69)
        special_vocab._set_special_token("middle", 68)
        special_vocab._set_special_token("fsep",   70)
        special_vocab._set_special_token("eot",    107)
        special_vocab.chat_template = None  # do not add it twice
        special_vocab.add_to_gguf(self.gguf_writer)

        self.gguf_writer.add_add_space_prefix(False)

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]

        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"] if "num_key_value_heads" in hparams else hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_key_length(hparams["head_dim"])
        self.gguf_writer.add_value_length(hparams["head_dim"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        # lm_head is not used in llama.cpp, while autoawq will include this tensor in model
        # To prevent errors, skip loading lm_head.weight.
        if name == "lm_head.weight":
            logger.debug(f"Skipping get tensor {name!r} in safetensors so that convert can end normally.")
            return []

        # ref: https://github.com/huggingface/transformers/blob/fc37f38915372c15992b540dfcbbe00a916d4fc6/src/transformers/models/gemma/modeling_gemma.py#L89
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1

        return [(self.map_tensor_name(name), data_torch)]


@Model.register("Gemma2ForCausalLM")
class Gemma2Model(Model):
    model_arch = gguf.MODEL_ARCH.GEMMA2

    def set_vocab(self):
        self._set_vocab_sentencepiece()

        self.gguf_writer.add_add_space_prefix(False)

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]

        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"] if "num_key_value_heads" in hparams else hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_key_length(hparams["head_dim"])
        self.gguf_writer.add_value_length(hparams["head_dim"])
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_attn_logit_softcapping(
            self.hparams["attn_logit_softcapping"]
        )
        self.gguf_writer.add_final_logit_softcapping(
            self.hparams["final_logit_softcapping"]
        )
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        # lm_head is not used in llama.cpp, while autoawq will include this tensor in model
        # To prevent errors, skip loading lm_head.weight.
        if name == "lm_head.weight":
            logger.debug(f"Skipping get tensor {name!r} in safetensors so that convert can end normally.")
            return []

        # ref: https://github.com/huggingface/transformers/blob/fc37f38915372c15992b540dfcbbe00a916d4fc6/src/transformers/models/gemma/modeling_gemma.py#L89
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1

        return [(self.map_tensor_name(name), data_torch)]


class Gemma4BaseModel(Model):
    model_arch = gguf.MODEL_ARCH.GEMMA4

    def _text_hparams(self) -> dict[str, Any]:
        text_hparams = self.hparams.get("text_config")
        if isinstance(text_hparams, dict):
            return text_hparams
        return self.hparams

    def _arch_name(self) -> str:
        return gguf.MODEL_ARCH_NAMES[self.model_arch]

    def find_hparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        text_hparams = self.hparams.get("text_config")
        if isinstance(text_hparams, dict):
            for key in keys:
                if key in text_hparams:
                    return text_hparams[key]
        return super().find_hparam(keys, optional)

    def set_vocab(self):
        vocab = gguf.LlamaHfVocab(self.dir_model)
        tokens = []
        scores = []
        toktypes = []
        visible_tokens = {
            "<|channel>",
            "<channel|>",
            "<|tool_call>",
            "<tool_call|>",
            "<|tool_response>",
            "<tool_response|>",
            "<|\"|>",
        }

        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            text_str = text.decode()
            if text_str in visible_tokens:
                toktypes.append(gguf.TokenType.USER_DEFINED)
                logger.info(f"Token {text_str!r} is set to USER_DEFINED")
            else:
                toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("gemma4")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)
        self.gguf_writer.add_add_space_prefix(False)
        self.gguf_writer.add_add_bos_token(True)


@Model.register("Gemma4ForConditionalGeneration")
class Gemma4Model(Gemma4BaseModel):
    model_arch = gguf.MODEL_ARCH.GEMMA4

    def set_gguf_parameters(self):
        hparams = self._text_hparams()
        block_count = hparams["num_hidden_layers"]
        arch = self._arch_name()

        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])
        self.gguf_writer.add_file_type(self.ftype)

        swa_layers = [layer_type == "sliding_attention" for layer_type in hparams["layer_types"]]
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(swa_layers)

        num_kv_shared_layers = hparams.get("num_kv_shared_layers", 0)
        self.gguf_writer.add_shared_kv_layers(num_kv_shared_layers)

        n_ff = hparams["intermediate_size"]
        if hparams.get("use_double_wide_mlp", False):
            first_kv_shared_layer_idx = block_count - num_kv_shared_layers
            n_ff_arr = [n_ff if il < first_kv_shared_layer_idx else n_ff * 2 for il in range(block_count)]
            self.gguf_writer.add_feed_forward_length(n_ff_arr)
        else:
            self.gguf_writer.add_feed_forward_length(n_ff)

        expert_intermediate_size = hparams.get("expert_intermediate_size") or hparams.get("moe_intermediate_size")
        if expert_intermediate_size is not None:
            self.gguf_writer.add_expert_feed_forward_length(expert_intermediate_size)

        n_pl_embd = hparams.get("hidden_size_per_layer_input") or 0
        self.gguf_writer.add_embedding_length_per_layer_input(n_pl_embd)

        head_dim_full = int(hparams["global_head_dim"])
        head_dim_swa = int(hparams["head_dim"])
        self.gguf_writer.add_key_length(head_dim_full)
        self.gguf_writer.add_value_length(head_dim_full)
        self.gguf_writer.add_uint32(f"{arch}.attention.key_length_swa", head_dim_swa)
        self.gguf_writer.add_uint32(f"{arch}.attention.value_length_swa", head_dim_swa)

        num_kv_full = hparams.get("num_global_key_value_heads")
        num_kv_swa = hparams.get("num_key_value_heads")
        if num_kv_full is not None and num_kv_swa is not None:
            kv_heads = [num_kv_swa if is_swa else num_kv_full for is_swa in swa_layers]
            self.gguf_writer.add_head_count_kv(kv_heads)
        elif num_kv_swa is not None:
            self.gguf_writer.add_head_count_kv(num_kv_swa)

        rope_parameters = hparams.get("rope_parameters", {})
        rope_full = rope_parameters.get("full_attention", {})
        rope_swa = rope_parameters.get("sliding_attention", {})
        self.gguf_writer.add_rope_dimension_count(head_dim_full)
        partial_rotary_factor_swa = float(rope_swa.get("partial_rotary_factor", hparams.get("partial_rotary_factor", 1.0)))
        self.gguf_writer.add_uint32(f"{arch}.rope.dimension_count_swa", int(head_dim_swa * partial_rotary_factor_swa))
        self.gguf_writer.add_rope_freq_base(float(rope_full.get("rope_theta", 1000000.0)))
        self.gguf_writer.add_float32(f"{arch}.rope.freq_base_swa", float(rope_swa.get("rope_theta", 10000.0)))

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        hparams = self._text_hparams()
        rope_params_full = hparams["rope_parameters"]["full_attention"]
        assert rope_params_full["rope_type"] == "proportional"

        head_dim_full = int(hparams["global_head_dim"])
        partial_rotary_factor_full = rope_params_full["partial_rotary_factor"]
        n_rot_full = int(head_dim_full * partial_rotary_factor_full / 2)
        n_unrot_full = int(head_dim_full / 2) - n_rot_full
        values = [1.0] * n_rot_full + [1e30] * n_unrot_full
        rope_freqs_full = torch.tensor(values, dtype=torch.float32)
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), rope_freqs_full)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith("per_dim_scale") or name.endswith("layer_scalar"):
            name = name + ".weight"

        if "language_model." not in name and "rope_freqs" not in name:
            return []

        name = name.replace("language_model.", "")

        if name == "lm_head.weight":
            logger.debug(f"Skipping get tensor {name!r} in safetensors so that convert can end normally.")
            return []

        if name.endswith("router.scale"):
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_INP, bid, ".scale"), data_torch)]

        if ".per_expert_scale" in name:
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN_EXP, bid, ".scale"), data_torch)]

        if ".experts." in name and not name.endswith(".weight"):
            name += ".weight"

        return [(self.map_tensor_name(name), data_torch)]


@Model.register("Gemma4AssistantForCausalLM")
class Gemma4AssistantModel(Gemma4BaseModel):
    model_arch = gguf.MODEL_ARCH.GEMMA4_MTP

    _root_tensor_map = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "pre_projection.weight": "mtp_pre_proj.weight",
        "post_projection.weight": "mtp_post_proj.weight",
        "masked_embedding.centroids.weight": "mtp_centroids.weight",
        "masked_embedding.token_ordering": "mtp_token_ordering.weight",
        "token_ordering": "mtp_token_ordering.weight",
        "token_ordering.weight": "mtp_token_ordering.weight",
        "model.token_ordering": "mtp_token_ordering.weight",
        "model.token_ordering.weight": "mtp_token_ordering.weight",
        "centroids": "mtp_centroids.weight",
        "centroids.weight": "mtp_centroids.weight",
        "model.centroids": "mtp_centroids.weight",
        "model.centroids.weight": "mtp_centroids.weight",
    }

    _layer_tensor_map = {
        "input_layernorm.weight": "attn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "post_attention_layernorm.weight": "post_attention_norm.weight",
        "pre_feedforward_layernorm.weight": "ffn_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
        "layer_scalar": "layer_output_scale.weight",
        "layer_scalar.weight": "layer_output_scale.weight",
    }

    def set_gguf_parameters(self):
        hparams = self._text_hparams()
        arch = self._arch_name()
        sliding_pattern = [layer_type == "sliding_attention" for layer_type in hparams["layer_types"]]

        head_dim_swa = int(hparams["head_dim"])
        head_dim_full = int(hparams.get("global_head_dim") or head_dim_swa)
        n_kv_swa = int(hparams["num_key_value_heads"])
        n_kv_full = int(hparams.get("num_global_key_value_heads") or n_kv_swa)
        n_kv = [n_kv_swa if is_sliding else n_kv_full for is_sliding in sliding_pattern]

        self.gguf_writer.add_context_length(int(hparams["max_position_embeddings"]))
        self.gguf_writer.add_embedding_length(int(hparams["hidden_size"]))
        self.gguf_writer.add_block_count(int(hparams["num_hidden_layers"]))
        self.gguf_writer.add_feed_forward_length(int(hparams["intermediate_size"]))
        self.gguf_writer.add_head_count(int(hparams["num_attention_heads"]))
        self.gguf_writer.add_head_count_kv(n_kv)
        self.gguf_writer.add_key_length(head_dim_full)
        self.gguf_writer.add_value_length(head_dim_full)
        self.gguf_writer.add_uint32(f"{arch}.attention.key_length_swa", head_dim_swa)
        self.gguf_writer.add_uint32(f"{arch}.attention.value_length_swa", head_dim_swa)
        self.gguf_writer.add_layer_norm_rms_eps(float(hparams["rms_norm_eps"]))
        self.gguf_writer.add_sliding_window(int(hparams["sliding_window"]))
        self.gguf_writer.add_array(f"{arch}.attention.sliding_window_pattern", sliding_pattern)
        self.gguf_writer.add_rope_dimension_count(head_dim_full)
        self.gguf_writer.add_uint32(f"{arch}.rope.dimension_count_swa", head_dim_swa)

        rope_parameters = hparams.get("rope_parameters", {})
        rope_full = rope_parameters.get("full_attention", {})
        rope_swa = rope_parameters.get("sliding_attention", {})
        self.gguf_writer.add_rope_freq_base(float(rope_full.get("rope_theta", 1000000.0)))
        self.gguf_writer.add_float32(f"{arch}.rope.freq_base_swa", float(rope_swa.get("rope_theta", 10000.0)))

        self.gguf_writer.add_uint32(f"{arch}.backbone_embedding_length", int(self.hparams["backbone_hidden_size"]))
        self.gguf_writer.add_bool(f"{arch}.use_ordered_embeddings", bool(self.hparams.get("use_ordered_embeddings", False)))
        self.gguf_writer.add_uint32(f"{arch}.centroid_count", int(self.hparams.get("num_centroids", 0)))
        self.gguf_writer.add_uint32(f"{arch}.centroid_top_k", int(self.hparams.get("centroid_intermediate_top_k", 0)))
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        mapped_name = self._root_tensor_map.get(name)
        if mapped_name is not None:
            if mapped_name == "mtp_token_ordering.weight":
                n_vocab = int(data_torch.shape[0])
                n_centroids = int(self.hparams.get("num_centroids", 2048))
                tokens_per_centroid = n_vocab // n_centroids
                inv_ordering = torch.zeros(n_vocab, dtype=torch.int32)
                tok_ord_i32 = data_torch.to(dtype=torch.int64)
                inv_ordering[tok_ord_i32] = torch.arange(n_vocab, dtype=torch.int32)
                token_to_centroid = (inv_ordering // tokens_per_centroid).to(dtype=torch.int32)
                return [(mapped_name, token_to_centroid)]
            return [(mapped_name, data_torch)]

        prefix = "model.layers."
        if not name.startswith(prefix):
            raise ValueError(f"Unsupported Gemma 4 assistant tensor: {name}")

        layer_id, suffix = name[len(prefix):].split(".", 1)
        mapped_suffix = self._layer_tensor_map.get(suffix)
        if mapped_suffix is None:
            raise ValueError(f"Unsupported Gemma 4 assistant tensor: {name}")

        return [(f"blk.{layer_id}.{mapped_suffix}", data_torch)]


@Model.register("Starcoder2ForCausalLM")
class StarCoder2Model(Model):
    model_arch = gguf.MODEL_ARCH.STARCODER2


@Model.register("MambaForCausalLM", "MambaLMHeadModel")
class MambaModel(Model):
    model_arch = gguf.MODEL_ARCH.MAMBA

    def set_vocab(self):
        vocab_size = self.hparams["vocab_size"]
        # Round vocab size to next multiple of 8
        pad_vocab = self.hparams.get("pad_vocab_size_multiple", 8)
        # pad using ceiling division
        # ref: https://stackoverflow.com/a/17511341/22827863
        vocab_size = -(vocab_size // -pad_vocab) * pad_vocab
        self.hparams["vocab_size"] = vocab_size

        if (self.dir_model / "tokenizer.json").is_file():
            self._set_vocab_gpt2()
        elif (self.dir_model / "tokenizer.model").is_file():
            self._set_vocab_sentencepiece()
        else:
            # Use the GPT-NeoX tokenizer when no tokenizer files are present
            self._set_vocab_builtin("gpt-neox", vocab_size)

    def set_gguf_parameters(self):
        d_model = self.find_hparam(["hidden_size",       "d_model"])
        d_conv  = self.find_hparam(["conv_kernel",       "d_conv"],  optional=True) or 4
        d_inner = self.find_hparam(["intermediate_size", "d_inner"], optional=True) or 2 * d_model
        d_state = self.find_hparam(["state_size",        "d_state"], optional=True) or 16
        # ceiling division
        # ref: https://stackoverflow.com/a/17511341/22827863
        # ref: https://github.com/state-spaces/mamba/blob/ce59daea3a090d011d6476c6e5b97f6d58ddad8b/mamba_ssm/modules/mamba_simple.py#L58
        dt_rank      = self.find_hparam(["time_step_rank",     "dt_rank"],      optional=True) or -(d_model // -16)
        rms_norm_eps = self.find_hparam(["layer_norm_epsilon", "rms_norm_eps"], optional=True) or 1e-5

        # Fail early for models which don't have a block expansion factor of 2
        assert d_inner == 2 * d_model

        self.gguf_writer.add_context_length(2**20) # arbitrary value; for those who use the default
        self.gguf_writer.add_embedding_length(d_model)
        self.gguf_writer.add_feed_forward_length(0) # unused, but seemingly required when loading
        self.gguf_writer.add_head_count(0) # unused, but seemingly required when loading
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_ssm_conv_kernel(d_conv)
        self.gguf_writer.add_ssm_inner_size(d_inner)
        self.gguf_writer.add_ssm_state_size(d_state)
        self.gguf_writer.add_ssm_time_step_rank(dt_rank)
        self.gguf_writer.add_layer_norm_rms_eps(rms_norm_eps)
        self.gguf_writer.add_file_type(self.ftype)

    _tok_embd = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        output_name = self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT)
        tok_embd_name = self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD)

        new_name = self.map_tensor_name(name)

        if name.endswith(".A_log"):
            logger.debug("A_log --> A ==> " + new_name)
            data_torch = -torch.exp(data_torch)

        # assuming token_embd.weight is seen before output.weight
        if self._tok_embd is not None and new_name == output_name:
            if torch.equal(self._tok_embd, data_torch):
                logger.debug(f"{output_name} is equivalent to {tok_embd_name}, omitting")
                return []
        elif new_name == tok_embd_name:
            self._tok_embd = data_torch

        return [(new_name, data_torch)]

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if bid is not None and new_name in (
            self.format_tensor_name(
                n, bid, ".weight" if name.endswith(".weight") else ""
            )
            for n in [
                gguf.MODEL_TENSOR.SSM_CONV1D,
                gguf.MODEL_TENSOR.SSM_X,
                gguf.MODEL_TENSOR.SSM_DT,
                gguf.MODEL_TENSOR.SSM_A,
                gguf.MODEL_TENSOR.SSM_D,
            ]
        ):
            return gguf.GGMLQuantizationType.F32

        return super().tensor_force_quant(name, new_name, bid, n_dims)


@Model.register("CohereForCausalLM")
class CommandR2Model(Model):
    model_arch = gguf.MODEL_ARCH.COMMAND_R

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # max_position_embeddings = 8192 in config.json but model was actually
        # trained on 128k context length
        # aya-23 models don't have model_max_length specified
        self.hparams["max_position_embeddings"] = self.find_hparam(["model_max_length", "max_position_embeddings"])

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)


@Model.register("Cohere2MoeForCausalLM")
class Cohere2MoeModel(Model):
    model_arch = gguf.MODEL_ARCH.COHERE2_MOE

    _experts: list[dict[str, Tensor]] | None = None

    def set_gguf_parameters(self):
        saved_intermediate_size = self.hparams["intermediate_size"]
        saved_num_experts_per_tok = self.hparams.pop("num_experts_per_tok")
        self.hparams["intermediate_size"] = self.hparams["prefix_dense_intermediate_size"]
        super().set_gguf_parameters()
        self.hparams["intermediate_size"] = saved_intermediate_size
        self.hparams["num_experts_per_tok"] = saved_num_experts_per_tok
        hparams = self.hparams

        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_logit_scale(hparams.get("logit_scale", 1.0))
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern([
            layer_type == "sliding_attention"
            for layer_type in hparams["layer_types"]
        ])
        self.gguf_writer.add_rope_dimension_count(hparams.get("head_dim", hparams["hidden_size"] // hparams["num_attention_heads"]))
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

        self.gguf_writer.add_expert_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_expert_count(hparams["num_experts"])
        self.gguf_writer.add_expert_used_count(hparams["num_experts_per_tok"])
        self.gguf_writer.add_expert_weights_norm(bool(hparams.get("norm_topk_prob", False)))

        expert_selection_fn = hparams.get("expert_selection_fn", "softmax")
        if expert_selection_fn != "sigmoid":
            raise ValueError(f"Unsupported Cohere2-MoE expert_selection_fn={expert_selection_fn!r}")
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        if hparams.get("num_shared_experts", 0) != 0:
            raise ValueError("Cohere2-MoE shared experts are not supported in this GGUF converter yet")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Cohere2-MoE HF tensors already use the interleaved RoPE layout expected here.

        if ".mlp.experts." in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) < n_experts * 3:
                return []

            tensors: list[tuple[str, Tensor]] = []
            for src, dst in [
                ("gate_proj", "gate_proj"),
                ("down_proj", "down_proj"),
                ("up_proj", "up_proj"),
            ]:
                datas: list[Tensor] = []
                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.mlp.experts.{xid}.{src}.weight"
                    datas.append(self._experts[bid][ename])
                    del self._experts[bid][ename]

                merged_name = f"model.layers.{bid}.mlp.experts.{dst}.weight"
                tensors.append((self.map_tensor_name(merged_name), torch.stack(datas, dim=0)))
            yield from tensors
            return

        if name == "model.embed_tokens.weight":
            yield self.map_tensor_name(name), data_torch
            if self.tensor_names is None or "lm_head.weight" not in self.tensor_names:
                yield self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT, suffix=".weight"), data_torch
            return

        yield self.map_tensor_name(name), data_torch


@Model.register("OlmoForCausalLM")
@Model.register("OLMoForCausalLM")
class OlmoModel(Model):
    model_arch = gguf.MODEL_ARCH.OLMO

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_layer_norm_eps(1e-5)
        clip_qkv = self.hparams.get("clip_qkv")
        if clip_qkv is not None:
            self.gguf_writer.add_clamp_kqv(clip_qkv)

    # Same as super class, but permuting q_proj, k_proj
    # Copied from: LlamaModel
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith("q_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith("k_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)

        return [(self.map_tensor_name(name), data_torch)]


@Model.register("JinaBertModel", "JinaBertForMaskedLM")
class JinaBertV2Model(BertModel):
    model_arch = gguf.MODEL_ARCH.JINA_BERT_V2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intermediate_size = self.hparams["intermediate_size"]

    def get_tensors(self):
        for name, data in super().get_tensors():
            if 'gated_layer' in name:
                d1 = data[:self.intermediate_size, :]
                name1 = name.replace('gated_layers', 'gated_layers_w')
                name1 = name1.replace('up_gated_layer', 'gated_layers_v')
                d2 = data[self.intermediate_size:, :]
                name2 = name.replace('gated_layers', 'gated_layers_v')
                name2 = name2.replace('up_gated_layer', 'gated_layers_w')
                yield name1, d1
                yield name2, d2
                continue

            yield name, data

    def set_vocab(self):
        tokenizer_class = 'BertTokenizer'
        with open(self.dir_model / "tokenizer_config.json", "r", encoding="utf-8") as f:
            tokenizer_class = json.load(f)['tokenizer_class']

        if tokenizer_class == 'BertTokenizer':
            super().set_vocab()
        elif tokenizer_class == 'RobertaTokenizer':
            self._set_vocab_gpt2()
            self.gguf_writer.add_token_type_count(2)
        else:
            raise NotImplementedError(f'Tokenizer {tokenizer_class} is not supported for JinaBertModel')
        self.gguf_writer.add_add_bos_token(True)
        self.gguf_writer.add_add_eos_token(True)


@Model.register("OpenELMForCausalLM")
class OpenELMModel(Model):
    model_arch = gguf.MODEL_ARCH.OPENELM

    @staticmethod
    def _make_divisible(v: float | int, divisor: int) -> int:
        # ref: https://huggingface.co/apple/OpenELM-270M-Instruct/blob/eb111ff2e6724348e5b905984063d4064d4bc579/configuration_openelm.py#L34-L38
        new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ffn_multipliers: list[float] = self.hparams["ffn_multipliers"]
        ffn_dim_divisor: int = self.hparams["ffn_dim_divisor"]
        self._n_embd: int = self.hparams["model_dim"]
        self._num_kv_heads: list[int] = self.hparams["num_kv_heads"]
        self._num_query_heads: list[int] = self.hparams["num_query_heads"]
        self._ffn_dims: list[int] = [
            OpenELMModel._make_divisible(multiplier * self._n_embd, ffn_dim_divisor)
            for multiplier in ffn_multipliers
        ]
        assert isinstance(self._num_kv_heads, list) and isinstance(self._num_kv_heads[0], int)
        assert isinstance(self._num_query_heads, list) and isinstance(self._num_query_heads[0], int)

    # Uses the tokenizer from meta-llama/Llama-2-7b-hf
    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_builtin("llama-spm", self.hparams["vocab_size"])

    def set_gguf_parameters(self):
        n_embd = self._n_embd
        head_dim = self.hparams["head_dim"]
        rot_pct = 1.0
        assert self.block_count == len(self._num_kv_heads)
        assert self.block_count == len(self._num_query_heads)
        assert self.block_count == len(self._ffn_dims)

        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(self.hparams["max_context_length"])
        self.gguf_writer.add_embedding_length(n_embd)
        self.gguf_writer.add_feed_forward_length(self._ffn_dims)
        self.gguf_writer.add_head_count(self._num_query_heads)
        self.gguf_writer.add_head_count_kv(self._num_kv_heads)
        self.gguf_writer.add_rope_freq_base(self.hparams["rope_freq_constant"])
        # https://huggingface.co/apple/OpenELM-270M-Instruct/blob/c401df2/modeling_openelm.py#L30
        self.gguf_writer.add_layer_norm_rms_eps(1e-6)
        self.gguf_writer.add_rope_dimension_count(int(rot_pct * head_dim))
        self.gguf_writer.add_key_length(head_dim)
        self.gguf_writer.add_value_length(head_dim)
        self.gguf_writer.add_file_type(self.ftype)

    def find_hparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        if "n_layers" in keys:
            return self.hparams["num_transformer_layers"]

        return super().find_hparam(keys, optional)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:

        # split ff
        if bid is not None and name == f"transformer.layers.{bid}.ffn.proj_1.weight":
            ff_dim = self._ffn_dims[bid]
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid), data_torch[:ff_dim])
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP, bid), data_torch[ff_dim:])
            return

        yield (self.map_tensor_name(name), data_torch)


@Model.register("ArcticForCausalLM")
class ArcticModel(Model):
    model_arch = gguf.MODEL_ARCH.ARCTIC

    def set_vocab(self):
        # The reason for using a custom implementation here is that the
        # snowflake-arctic-instruct model redefined tokens 31998 and 31999 from
        # tokenizer.model and used them as BOS and EOS instead of adding new tokens.
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / 'tokenizer.model'

        if not tokenizer_path.is_file():
            logger.error(f'Error: Missing {tokenizer_path}')
            sys.exit(1)

        # Read the whole vocabulary from the tokenizer.model file
        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        for token_id in range(tokenizer.vocab_size()):

            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        # Use the added_tokens_decoder field from tokeniser_config.json as the source
        # of information about added/redefined tokens and modify them accordingly.
        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)

                if "added_tokens_decoder" in tokenizer_config_json:
                    added_tokens_decoder = tokenizer_config_json["added_tokens_decoder"]
                    for token_id, token_json in added_tokens_decoder.items():
                        token_id = int(token_id)
                        if token_id >= vocab_size:
                            logger.debug(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                            continue

                        token_content = token_json["content"]
                        token_type = SentencePieceTokenTypes.USER_DEFINED
                        token_score = -10000.0

                        # Map unk_token to UNKNOWN, other special tokens to CONTROL
                        # Set the score to 0.0 as in the original tokenizer.model
                        if ("special" in token_json) and token_json["special"]:
                            if token_content == tokenizer_config_json["unk_token"]:
                                token_type = SentencePieceTokenTypes.UNKNOWN
                            else:
                                token_type = SentencePieceTokenTypes.CONTROL
                            token_score = 0.0

                        logger.info(f"Setting added token {token_id} to '{token_content}' (type: {token_type}, score: {token_score:.2f})")
                        tokens[token_id] = token_content.encode("utf-8")
                        toktypes[token_id] = token_type
                        scores[token_id] = token_score

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_rope_dimension_count(hparams["hidden_size"] // hparams["num_attention_heads"])

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith("q_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith("k_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)

        # process the experts separately
        if name.find("block_sparse_moe.experts") != -1:
            n_experts = self.hparams["num_local_experts"]

            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for wid in ["w1", "w2", "w3"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"layers.{bid}.feed_forward.experts.{wid}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@Model.register("DeepseekV2ForCausalLM")
@Model.register("DeepseekV3ForCausalLM")
class DeepseekV2Model(Model):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK2

    def set_vocab(self):

        if self.hparams["vocab_size"] == 163840:  # Kimi-K2 model
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.dir_model, trust_remote_code=True
            )
            tokpre = self.get_vocab_base_pre(tokenizer)

            # Build merges list using the approach similar to HunYuanMoE
            merges = []
            vocab = {}
            mergeable_ranks = tokenizer.model._mergeable_ranks
            for token, rank in mergeable_ranks.items():
                vocab[QwenModel.token_bytes_to_string(token)] = rank
                if len(token) == 1:
                    continue
                merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
                if len(merged) == 2:
                    merges.append(
                        " ".join(map(QwenModel.token_bytes_to_string, merged))
                    )

            # Build token list
            vocab_size = self.hparams["vocab_size"]
            special_tokens = tokenizer.special_tokens
            reverse_vocab = {
                id_: encoded_tok
                for encoded_tok, id_ in {**vocab, **special_tokens}.items()
            }
            tokens: list[str] = []
            toktypes: list[int] = []

            for i in range(vocab_size):
                if i not in reverse_vocab:
                    tokens.append(f"[PAD{i}]")
                    toktypes.append(gguf.TokenType.UNUSED)
                else:
                    token = reverse_vocab[i]
                    tokens.append(token)
                    if i in special_tokens.values():
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        toktypes.append(gguf.TokenType.NORMAL)

            self.gguf_writer.add_tokenizer_model("gpt2")
            self.gguf_writer.add_tokenizer_pre(tokpre)
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_types(toktypes)
            self.gguf_writer.add_token_merges(merges)

            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
            special_vocab.add_to_gguf(self.gguf_writer)
        else:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        if "q_lora_rank" in hparams and hparams["q_lora_rank"] is not None:
            self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length(hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"])
        self.gguf_writer.add_value_length(hparams["v_head_dim"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])

        if hparams["scoring_func"] == "sigmoid":
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        elif hparams["scoring_func"] == "softmax":
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
        else:
            raise ValueError(f"Unsupported scoring_func value: {hparams['scoring_func']}")

        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "yarn":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])
                self.gguf_writer.add_rope_scaling_orig_ctx_len(self.hparams["rope_scaling"]["original_max_position_embeddings"])
                self.gguf_writer.add_rope_scaling_yarn_log_mul(0.1 * hparams["rope_scaling"]["mscale_all_dim"])

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # rename e_score_correction_bias tensors
        if name.endswith("e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")

        # skip Multi-Token Prediction (MTP) layers
        block_count = self.hparams["num_hidden_layers"]
        match = re.match(r"model.layers.(\d+)", name)
        if match and int(match.group(1)) >= block_count:
            return []

        # process the experts separately
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []
        if name.endswith("kv_b_proj.weight"):
            name_kb = name.replace("kv_b_proj", "k_b_proj")
            name_vb = name.replace("kv_b_proj", "v_b_proj")

            n_head_kv = self.hparams["num_key_value_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]

            assert data_torch.shape[0] == n_head_kv * (v_head_dim + qk_nope_head_dim)

            kv_b = data_torch.view(n_head_kv, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)
            k_b = k_b.reshape(n_head_kv * data_torch.shape[-1], qk_nope_head_dim)
            v_b = v_b.reshape(n_head_kv * v_head_dim, data_torch.shape[-1])

            return [
                (self.map_tensor_name(name),    data_torch),
                (self.map_tensor_name(name_kb), k_b),
                (self.map_tensor_name(name_vb), v_b)
            ]

        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@Model.register("Glm4MoeLiteForCausalLM")
class Glm4MoeLiteModel(DeepseekV2Model):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # glm4_moe_lite config has no "scoring_func" key (uses topk_method: noaux_tc
        # instead); DeepseekV2Model.set_gguf_parameters hard-indexes hparams["scoring_func"].
        # sigmoid is correct here: the e_score_correction_bias tensor is present and the
        # fork's deepseek2 loader defaults GLM-4.7-Flash's 47-layer shape to sigmoid gating.
        # (NOTE: self.hparams does not exist until Model.__init__ runs, so this must come
        # AFTER super().__init__(), not before — setting it earlier raises AttributeError.)
        self.hparams.setdefault("scoring_func", "sigmoid")

    def set_vocab(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        # Special tokens
        # Note: Using <|endoftext|> (151329) for eot causes endless generation
        special_vocab._set_special_token("bos", tokenizer.get_added_vocab()["[gMASK]"])  # 151331
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|user|>"])  # 151336
        special_vocab._set_special_token("unk", tokenizer.get_added_vocab()["<|endoftext|>"]) # 151329
        special_vocab._set_special_token("eom", tokenizer.get_added_vocab()["<|observation|>"])  # 151338

        # Patch broken chat template
        if isinstance(special_vocab.chat_template, str) and "visible_text(m.content).endswith" in special_vocab.chat_template:
            special_vocab.chat_template = special_vocab.chat_template.replace(
                """{{ visible_text(m.content) }}\n{{- '/nothink' if (enable_thinking is defined and not enable_thinking and not visible_text(m.content).endswith("/nothink")) else '' -}}""",
                """{% set content = visible_text(m.content) %}{{ content }}\n{{- '/nothink' if (enable_thinking is defined and not enable_thinking and not content.endswith("/nothink")) else '' -}}""")

        special_vocab.add_to_gguf(self.gguf_writer)


@Model.register("T5WithLMHeadModel")
@Model.register("T5ForConditionalGeneration")
@Model.register("MT5ForConditionalGeneration")
@Model.register("UMT5ForConditionalGeneration")
class T5Model(Model):
    model_arch = gguf.MODEL_ARCH.T5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_token_embeddings_found = False

    def set_vocab(self):
        # to avoid TypeError: Descriptors cannot be created directly
        # exception when importing sentencepiece_model_pb2
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        from sentencepiece import SentencePieceProcessor
        from sentencepiece import sentencepiece_model_pb2 as model

        tokenizer_path = self.dir_model / 'tokenizer.model'

        # many older models use spiece.model tokenizer model filename
        if not tokenizer_path.is_file():
            tokenizer_path = self.dir_model / 'spiece.model'

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")

        sentencepiece_model = model.ModelProto()  # pyright: ignore[reportAttributeAccessIssue]
        sentencepiece_model.ParseFromString(open(tokenizer_path, "rb").read())

        # some models like Pile-T5 family use BPE tokenizer instead of Unigram
        if sentencepiece_model.trainer_spec.model_type == 2:  # BPE
            # assure the tokenizer model file name is correct
            assert tokenizer_path.name == 'tokenizer.model'
            return self._set_vocab_sentencepiece()
        else:
            assert sentencepiece_model.trainer_spec.model_type == 1  # UNIGRAM

        add_prefix = sentencepiece_model.normalizer_spec.add_dummy_prefix
        remove_whitespaces = sentencepiece_model.normalizer_spec.remove_extra_whitespaces
        precompiled_charsmap = sentencepiece_model.normalizer_spec.precompiled_charsmap

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        for token_id in range(tokenizer.vocab_size()):
            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)
                for key in added_tokens_json:
                    token_id = added_tokens_json[key]
                    if token_id >= vocab_size:
                        logger.warning(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                        continue

                    tokens[token_id] = key.encode("utf-8")
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]")
            for i in range(1, pad_count + 1):
                tokens.append(bytes(f"[PAD{i}]", encoding="utf-8"))
                scores.append(-1000.0)
                toktypes.append(SentencePieceTokenTypes.UNUSED)

        self.gguf_writer.add_tokenizer_model("t5")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_add_space_prefix(add_prefix)
        self.gguf_writer.add_remove_extra_whitespaces(remove_whitespaces)
        if precompiled_charsmap:
            self.gguf_writer.add_precompiled_charsmap(precompiled_charsmap)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(True)

    def set_gguf_parameters(self):
        if (n_ctx := self.find_hparam(["n_positions"], optional=True)) is None:
            logger.warning("Couldn't find context length in config.json, assuming default value of 512")
            n_ctx = 512
        self.gguf_writer.add_context_length(n_ctx)
        self.gguf_writer.add_embedding_length(self.hparams["d_model"])
        self.gguf_writer.add_feed_forward_length(self.hparams["d_ff"])
        self.gguf_writer.add_block_count(self.hparams["num_layers"])
        self.gguf_writer.add_head_count(self.hparams["num_heads"])
        self.gguf_writer.add_key_length(self.hparams["d_kv"])
        self.gguf_writer.add_value_length(self.hparams["d_kv"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_relative_attn_buckets_count(self.hparams["relative_attention_num_buckets"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_decoder_start_token_id(self.hparams["decoder_start_token_id"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        # T5 based models contain shared token embeddings tensors saved randomly as either "encoder.embed_tokens.weight",
        # "decoder.embed_tokens.weight" or "shared.weight" tensor. In some models there are even multiple of them stored
        # in the safetensors files. We use the first tensor from these three as the token embeddings for both encoder
        # and decoder and ignore the remaining ones.
        if name in ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight", "shared.weight"]:
            if not self.shared_token_embeddings_found:
                name = "shared.weight"
                self.shared_token_embeddings_found = True
            else:
                logger.debug(f"Skipping shared tensor {name!r} in safetensors so that convert can end normally.")
                return []

        return [(self.map_tensor_name(name), data_torch)]


@Model.register("T5EncoderModel")
class T5EncoderModel(Model):
    model_arch = gguf.MODEL_ARCH.T5ENCODER

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_token_embeddings_found = False

    def set_vocab(self):
        # to avoid TypeError: Descriptors cannot be created directly
        # exception when importing sentencepiece_model_pb2
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        from sentencepiece import SentencePieceProcessor
        from sentencepiece import sentencepiece_model_pb2 as model

        tokenizer_path = self.dir_model / 'tokenizer.model'

        # many older models use spiece.model tokenizer model filename
        if not tokenizer_path.is_file():
            tokenizer_path = self.dir_model / 'spiece.model'

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")

        sentencepiece_model = model.ModelProto()  # pyright: ignore[reportAttributeAccessIssue]
        sentencepiece_model.ParseFromString(open(tokenizer_path, "rb").read())

        # some models like Pile-T5 family use BPE tokenizer instead of Unigram
        if sentencepiece_model.trainer_spec.model_type == 2:  # BPE
            # assure the tokenizer model file name is correct
            assert tokenizer_path.name == 'tokenizer.model'
            return self._set_vocab_sentencepiece()
        else:
            assert sentencepiece_model.trainer_spec.model_type == 1  # UNIGRAM

        add_prefix = sentencepiece_model.normalizer_spec.add_dummy_prefix
        remove_whitespaces = sentencepiece_model.normalizer_spec.remove_extra_whitespaces
        precompiled_charsmap = sentencepiece_model.normalizer_spec.precompiled_charsmap

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        for token_id in range(tokenizer.vocab_size()):
            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)
                for key in added_tokens_json:
                    token_id = added_tokens_json[key]
                    if token_id >= vocab_size:
                        logger.warning(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                        continue

                    tokens[token_id] = key.encode("utf-8")
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]")
            for i in range(1, pad_count + 1):
                tokens.append(bytes(f"[PAD{i}]", encoding="utf-8"))
                scores.append(-1000.0)
                toktypes.append(SentencePieceTokenTypes.UNUSED)

        self.gguf_writer.add_tokenizer_model("t5")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_add_space_prefix(add_prefix)
        self.gguf_writer.add_remove_extra_whitespaces(remove_whitespaces)
        if precompiled_charsmap:
            self.gguf_writer.add_precompiled_charsmap(precompiled_charsmap)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(True)

    def set_gguf_parameters(self):
        if (n_ctx := self.find_hparam(["n_positions"], optional=True)) is None:
            logger.warning("Couldn't find context length in config.json, assuming default value of 512")
            n_ctx = 512
        self.gguf_writer.add_context_length(n_ctx)
        self.gguf_writer.add_embedding_length(self.hparams["d_model"])
        self.gguf_writer.add_feed_forward_length(self.hparams["d_ff"])
        self.gguf_writer.add_block_count(self.hparams["num_layers"])
        self.gguf_writer.add_head_count(self.hparams["num_heads"])
        self.gguf_writer.add_key_length(self.hparams["d_kv"])
        self.gguf_writer.add_value_length(self.hparams["d_kv"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_relative_attn_buckets_count(self.hparams["relative_attention_num_buckets"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        # T5 based models contain shared token embeddings tensors saved randomly as either "encoder.embed_tokens.weight",
        # "decoder.embed_tokens.weight" or "shared.weight" tensor. In some models there are even multiple of them stored
        # in the safetensors files. We use the first tensor from these three as the token embeddings for both encoder
        # and decoder and ignore the remaining ones.
        if name in ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight", "shared.weight"]:
            if not self.shared_token_embeddings_found:
                name = "shared.weight"
                self.shared_token_embeddings_found = True
            else:
                logger.debug(f"Skipping shared tensor {name!r} in safetensors so that convert can end normally.")
                return []

        return [(self.map_tensor_name(name), data_torch)]


@Model.register("JAISLMHeadModel")
class JaisModel(Model):
    model_arch = gguf.MODEL_ARCH.JAIS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # SwigLU activation
        assert self.hparams["activation_function"] == "swiglu"
        # ALiBi position embedding
        assert self.hparams["position_embedding_type"] == "alibi"

        # Embeddings scale
        self.embeddings_scale = 1.0
        # note: For some JAIS flavors, output is tied to (same as) wte in original model
        self.output_is_wte = False
        if 'mup_embeddings_scale' in self.hparams:
            self.output_is_wte = True   # Hack (?)
            self.embeddings_scale = self.hparams['mup_embeddings_scale']
        elif 'embeddings_scale' in self.hparams:
            self.embeddings_scale = self.hparams['embeddings_scale']
        else:
            assert False

        self.width_scale = 1.0
        if 'mup_output_alpha' in self.hparams:
            assert 'mup_width_scale' in self.hparams
            self.width_scale = self.hparams['mup_output_alpha'] * self.hparams['mup_width_scale']
        elif 'width_scale' in self.hparams:
            self.width_scale = self.hparams['width_scale']
        else:
            assert False

        self.max_alibi_bias = 8.0

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(self.hparams["n_inner"])
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        tensors: list[tuple[str, Tensor]] = []

        # we don't need these
        if name.endswith((".attn.bias")):
            return tensors

        if name.endswith(("relative_pe.slopes")):
            # Calculate max ALiBi bias (this is the inverse of the ALiBi calculation)
            # Some other models has max_alibi_bias spelled out explicitly in the hyperparams,
            # but Jais's PyTorch model simply precalculates the slope values and places them
            # in relative_pes.slopes
            n_head_closest_log2 = 2 ** math.floor(math.log2(self.hparams["n_head"]))
            first_val = float(data_torch[0].item())
            self.max_alibi_bias = -round(math.log2(first_val) * n_head_closest_log2)

            return tensors

        if name.endswith((".c_attn.weight", ".c_proj.weight", ".c_fc.weight", ".c_fc2.weight")):
            data_torch = data_torch.transpose(1, 0)

        new_name = self.map_tensor_name(name)

        if new_name == self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD):
            tensors.append((new_name, data_torch * self.embeddings_scale))
            if self.output_is_wte:
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch * self.width_scale))
        elif new_name == self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT):
            assert not self.output_is_wte
            tensors.append((new_name, data_torch * self.width_scale))
        else:
            tensors.append((new_name, data_torch))

        return tensors

    def prepare_tensors(self):
        super().prepare_tensors()
        self.gguf_writer.add_max_alibi_bias(self.max_alibi_bias)

@Model.register("MiniMaxM2ForCausalLM")
class MiniMaxM2Model(Model):
    model_arch = gguf.MODEL_ARCH.MINIMAXM2
    _experts_cache: dict[int, dict[str, Tensor]] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams["num_experts"] = self.hparams["num_local_experts"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if self.hparams["scoring_func"] == "sigmoid":
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        elif self.hparams["scoring_func"] == "softmax":
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
        else:
            raise ValueError(f"Unsupported scoring_func value: {self.hparams['scoring_func']}")

        self.gguf_writer.add_expert_feed_forward_length(self.find_hparam(["intermediate_size"]))
        self.gguf_writer.add_rope_dimension_count(self.find_hparam(["rotary_dim"]))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        if name.endswith("e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")

        # merge expert weights
        if 'experts' in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            expert_cache = self._experts_cache.setdefault(bid, {})
            expert_cache[name] = data_torch
            expert_weights = ["w1", "w2", "w3"]

            # not enough expert weights to merge
            if len(expert_cache) < n_experts * len(expert_weights):
                return []

            tensors: list[tuple[str, Tensor]] = []
            for w_name in expert_weights:
                datas: list[Tensor] = []

                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{w_name}.weight"
                    datas.append(expert_cache[ename])
                    del expert_cache[ename]

                data_torch = torch.stack(datas, dim=0)
                merged_name = f"model.layers.{bid}.block_sparse_moe.experts.{w_name}.weight"
                new_name = self.map_tensor_name(merged_name)
                tensors.append((new_name, data_torch))

            del self._experts_cache[bid]
            return tensors

        return super().modify_tensors(data_torch, name, bid)


@Model.register("SmolLM3ForCausalLM")
class SmolLM3Model(LlamaModel):
    model_arch = gguf.MODEL_ARCH.SMOLLM3

    def set_vocab(self):
        super().set_vocab()
        # remove unsupported array slicing in chat template
        # ref: https://huggingface.co/ggml-org/SmolLM3-3B-GGUF/discussions/1
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        if tokenizer.chat_template is not None:
            chat_template = tokenizer.chat_template.replace("[:]", "")
            self.gguf_writer.add_chat_template(chat_template)

@Model.register("SeedOssForCausalLM")
class SeedOssModel(Model):
    model_arch = gguf.MODEL_ARCH.SEED_OSS

@Model.register("Dots1ForCausalLM")
class Dots1Model(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.DOTS1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams["num_experts"] = self.hparams["n_routed_experts"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_leading_dense_block_count(self.hparams["first_k_dense_replace"])
        self.gguf_writer.add_expert_shared_count(self.hparams["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(self.hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(self.hparams["norm_topk_prob"])

        if self.hparams["scoring_func"] == "sigmoid":
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        else:
            raise ValueError(f"Unsupported scoring_func value: {self.hparams['scoring_func']}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        if name.endswith("e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")
        if "shared_experts" in name:
            return [(self.map_tensor_name(name), data_torch)]
        return super().modify_tensors(data_torch, name, bid)


@Model.register("Glm4MoeForCausalLM")
class Glm4MoeModel(Model):
    model_arch = gguf.MODEL_ARCH.GLM4_MOE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # GLM4_MOE has num_hidden_layers + 1 actual layers (including NextN layer)
        self.block_count = self.hparams["num_hidden_layers"] + self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        # Special tokens
        # Note: Using <|endoftext|> (151329) for eot causes endless generation
        special_vocab._set_special_token("bos", tokenizer.get_added_vocab()["[gMASK]"])  # 151331
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|user|>"])  # 151336
        special_vocab._set_special_token("unk", tokenizer.get_added_vocab()["<|endoftext|>"]) # 151329
        special_vocab._set_special_token("eom", tokenizer.get_added_vocab()["<|observation|>"])  # 151338

        # Patch broken chat template
        if isinstance(special_vocab.chat_template, str) and "visible_text(m.content).endswith" in special_vocab.chat_template:
            special_vocab.chat_template = special_vocab.chat_template.replace(
                """{{ visible_text(m.content) }}\n{{- '/nothink' if (enable_thinking is defined and not enable_thinking and not visible_text(m.content).endswith("/nothink")) else '' -}}""",
                """{% set content = visible_text(m.content) %}{{ content }}\n{{- '/nothink' if (enable_thinking is defined and not enable_thinking and not content.endswith("/nothink")) else '' -}}""")

        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (rope_dim := self.hparams.get("head_dim")) is None:
            rope_dim = (
                self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
            )
        self.gguf_writer.add_rope_dimension_count(
            int(rope_dim * self.hparams.get("partial_rotary_factor", 0.5))
        )

        # MoE parameters - Use only routed expert count (shared experts handled separately)
        if (n_routed_experts := self.hparams.get("n_routed_experts")) is not None:
            self.gguf_writer.add_expert_count(n_routed_experts)
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
        if (n_shared_experts := self.hparams.get("n_shared_experts")) is not None:
            self.gguf_writer.add_expert_shared_count(n_shared_experts)
        if (first_k_dense_replace := self.hparams.get("first_k_dense_replace")) is not None:
            self.gguf_writer.add_leading_dense_block_count(first_k_dense_replace)

        # Expert gating function (sigmoid for GLM4_MOE)
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        # Routed scaling factor
        if (routed_scaling_factor := self.hparams.get("routed_scaling_factor")) is not None:
            self.gguf_writer.add_expert_weights_scale(routed_scaling_factor)

        # Normalise topk probabilities
        if (norm_topk_prob := self.hparams.get("norm_topk_prob")) is not None:
            self.gguf_writer.add_expert_weights_norm(norm_topk_prob)

        # NextN/MTP prediction layers
        if (num_nextn_predict_layers := self.hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(num_nextn_predict_layers)

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("model.visual."):  # ignore visual part
            return []
        elif name.startswith("model.language_model."):
            name = name.replace("language_model.", "")  # for multimodal variants

        # Handle main token embedding (but not layer-specific NextN embeddings)
        if name == "model.embed_tokens.weight" and ".layers." not in name:
            return [(self.map_tensor_name("token_embd.weight"), data_torch)]

        # Handle routed experts
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    new_name = self.map_tensor_name(merged_name)
                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        if name.endswith("e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")

        new_name = self.map_tensor_name(name)

        return [(new_name, data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")

@Model.register("ChatGLMModel", "ChatGLMForConditionalGeneration")
class ChatGLMModel(Model):
    model_arch = gguf.MODEL_ARCH.CHATGLM

    def set_vocab_chatglm3(self):
        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[bytes] = []
        toktypes: list[int] = []
        scores: list[float] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        vocab_size = hparams.get("padded_vocab_size", len(tokenizer.get_vocab()))
        assert max(tokenizer.get_vocab().values()) < vocab_size
        role_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"]
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"] + role_special_tokens
        for token_id in range(vocab_size):
            piece = tokenizer._convert_id_to_token(token_id)
            if token_id == 0:
                piece = "<unk>"
            elif token_id == 1:
                piece = "<bos>"
            elif token_id == 2:
                piece = "<eos>"

            text = piece.encode("utf-8")
            score = 0.0
            # Referencing the tokenizer Python implementation(https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenization_chatglm.py),
            # it is only valid if it is less than tokenizer.tokenizer.sp_model.vocab_size()
            if len(piece) != 0 and token_id < tokenizer.tokenizer.sp_model.vocab_size():
                score = tokenizer.tokenizer.sp_model.get_score(token_id)

            if token_id >= tokenizer.tokenizer.sp_model.vocab_size():
                if piece in special_tokens:
                    toktype = SentencePieceTokenTypes.CONTROL
                elif len(piece) == 0:
                    text = f"[PAD{token_id}]".encode("utf-8")
                    toktype = SentencePieceTokenTypes.UNUSED
                else:
                    toktype = SentencePieceTokenTypes.USER_DEFINED
                tokens.append(text)
                scores.append(score)
                toktypes.append(toktype)
                continue

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.tokenizer.sp_model.is_unknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.tokenizer.sp_model.is_control(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.tokenizer.sp_model.is_unused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.tokenizer.sp_model.is_byte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        self.gguf_writer.add_tokenizer_model("llama")
        # glm3 needs prefix and suffix formatted as:
        # prompt = "[gMASK]sop<|user|>\n" + prompt + "<|assistant|>"
        self.gguf_writer.add_tokenizer_pre("chatglm-spm")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    @staticmethod
    def token_bytes_to_string(b):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        byte_encoder = bytes_to_unicode()
        return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])

    @staticmethod
    def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int | None = None) -> list[bytes]:
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        return parts

    def set_vocab(self):
        if "THUDM/chatglm3-6b" in self.hparams.get("_name_or_path", ""):
            self.set_vocab_chatglm3()
            return

        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        vocab_size = hparams["padded_vocab_size"]
        assert max(tokenizer.get_vocab().values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks
        for token, rank in mergeable_ranks.items():
            vocab[ChatGLMModel.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = ChatGLMModel.bpe(mergeable_ranks, token, max_rank=rank)
            assert len(merged) >= 2 and len(merged) <= 7
            merges.append(' '.join(map(ChatGLMModel.token_bytes_to_string, merged)))

        # for this kind of tokenizer, added_vocab is not a subset of vocab, so they need to be combined
        added_vocab = tokenizer.get_added_vocab()
        reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **added_vocab}.items()}

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                if tokenizer.added_tokens_decoder[i].special:
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, load_merges=False)
        special_vocab.merges = merges
        # only add special tokens when they were not already loaded from config.json
        special_vocab._set_special_token("eos", tokenizer.get_added_vocab()["<|endoftext|>"])
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|user|>"])
        # this one is usually not in config.json anyway
        special_vocab._set_special_token("unk", tokenizer.get_added_vocab()["<|endoftext|>"])
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        n_head_kv = self.hparams.get("multi_query_group_num", n_head)
        self.gguf_writer.add_context_length(self.hparams.get("seq_length", n_embed))
        self.gguf_writer.add_embedding_length(n_embed)
        self.gguf_writer.add_feed_forward_length(self.hparams.get("ffn_hidden_size", 4 * n_embed))
        self.gguf_writer.add_block_count(self.hparams["num_layers"])
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layernorm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_rope_dimension_count(64)
        self.gguf_writer.add_add_bos_token(False)
        rope_freq = 10000
        if "rope_ratio" in self.hparams:
            rope_freq = rope_freq * self.hparams["rope_ratio"]
        self.gguf_writer.add_rope_freq_base(rope_freq)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        if name.endswith(".rotary_pos_emb.inv_freq"):
            return []

        name = name.removeprefix("transformer.")
        return [(self.map_tensor_name(name), data_torch)]

@Model.register("BailingMoeV2ForCausalLM")
class BailingMoeV2Model(Model):
    model_arch = gguf.MODEL_ARCH.BAILINGMOE2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if nextn_layers := self.hparams.get("num_nextn_predict_layers", 0):
            self.block_count = self.hparams["num_hidden_layers"] + nextn_layers
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]

        self.gguf_writer.add_rope_dimension_count(int(rope_dim * self.hparams.get("partial_rotary_factor", 0.5)))
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "yarn" and "factor" in rope_scaling:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_scaling["original_max_position_embeddings"])
        else:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hparams.get("moe_shared_expert_intermediate_size", hparams["moe_intermediate_size"] * hparams["num_shared_experts"]))
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_count(hparams["num_experts"])
        self.gguf_writer.add_expert_shared_count(hparams["num_shared_experts"])
        self.gguf_writer.add_expert_group_count(hparams["n_group"])
        self.gguf_writer.add_expert_group_used_count(hparams["topk_group"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])

        if hparams["score_function"] == "sigmoid":
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        elif hparams["score_function"] == "softmax":
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
        else:
            raise ValueError(f"Unsupported score_function value: {hparams['score_function']}")

        if (nextn_layers := self.hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(nextn_layers)

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "mlp.experts" in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            tensors: list[tuple[str, Tensor]] = []

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))

            return tensors

        if name.endswith("final_layernorm.weight"):
            # PXA_BAILING_MTP_FINALNORM_FIX: the NextN/MTP block has its OWN post-FFN RMSNorm
            # (BailingMoeV2MTPLayer.final_layernorm in modeling_bailing_moe_v2.py, applied as the
            # LAST op before the shared lm_head). Bailing has NO shared_head.norm tensor, so the
            # kernel pre-head norm slot (mtp_layer.nextn.shared_head_norm) was empty -> the MTP
            # draft head ran lm_head(ffn_out) with NO final normalization -> scrambled draft logits
            # -> ~0.20 MTP acceptance. Route final_layernorm into the shared_head_norm slot so
            # build_bailingmoe2_mtp build_output(..., mtp_layer.nextn.shared_head_norm) applies it.
            # (Dropping it -- the old "reuses main output_norm" claim -- was wrong: build_output
            # gets output_norm=NULL when shared_head_norm is absent, so NO norm was applied at all.)
            assert bid is not None
            return [(self.map_tensor_name(f"model.layers.{bid}.shared_head.norm.weight"), data_torch)]
        if name.endswith(".expert_bias"):
            name = name.replace(".expert_bias", ".expert_bias.bias")

        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@Model.register("LagunaForCausalLM")
class LagunaModel(Model):
    model_arch = gguf.MODEL_ARCH.LAGUNA
    _experts: list[dict] | None = None
    _gate_types: list[str] | None = None

    # --- vocab ---------------------------------------------------------------

    def set_vocab(self) -> None:
        self._set_vocab_gpt2()

        # Some Laguna releases wrap the chat template in tokenizer_config.json as
        # "{% include 'chat_template.jinja' %}", which SpecialVocab embeds verbatim
        # and the jinja engine cannot process. Prefer the resolved template from the
        # chat_template.jinja file so the GGUF is self-contained.
        tmpl_file = self.dir_model / "chat_template.jinja"
        if tmpl_file.is_file():
            self.gguf_writer.add_chat_template(tmpl_file.read_text(encoding="utf-8"))
            logger.info("gguf: embedded resolved chat_template.jinja (overriding include directive)")

        # eos_token_id is a list [2, 24]: token 2 (EOS, also BOS) and token 24
        # (</assistant>, the turn-end). _set_vocab_gpt2 only records the scalar
        # eos, so register the extra id as eot; eot is folded into the EOG set, so
        # the model halts on </assistant> natively.
        eos_ids = self.hparams.get("eos_token_id")
        if isinstance(eos_ids, list):
            bos_id = self.hparams.get("bos_token_id")
            extra = [e for e in eos_ids if e != bos_id]
            if extra:
                self.gguf_writer.add_eot_token_id(extra[0])
                logger.info(f"gguf: registered eot_token_id={extra[0]} from eos list {eos_ids}")

    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        # </assistant> is the assistant turn-end (registered as eot above). The HF
        # tokenizer flags it special=false, so the base classifies it as USER_DEFINED
        # and its text is rendered into generated content, leaking "</assistant>" and
        # breaking response parsing. It is a control marker, so promote it to CONTROL:
        # it is then treated as end-of-generation and its text is suppressed.
        tokens, toktypes, tokpre = super().get_vocab_base()
        for i, tok in enumerate(tokens):
            if tok == "</assistant>":
                toktypes[i] = gguf.TokenType.CONTROL
                logger.info(f"gguf: marked </assistant> (id {i}) as CONTROL token")
        return tokens, toktypes, tokpre

    # --- hparams -------------------------------------------------------------

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        # super() does not emit vocab_size for the gpt2 vocab path.
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        # head_count is overridden with a per-layer array: S-2.1/XS.2 vary heads per
        # layer via num_attention_heads_per_layer (48 on full-attn layers, 72 on SWA
        # layers); M.1 is uniform and omits the key.
        per_layer_heads = hparams.get("num_attention_heads_per_layer")
        if not per_layer_heads:
            per_layer_heads = [hparams["num_attention_heads"]] * hparams["num_hidden_layers"]
        assert len(per_layer_heads) == hparams["num_hidden_layers"], (
            f"num_attention_heads_per_layer length {len(per_layer_heads)} != "
            f"num_hidden_layers {hparams['num_hidden_layers']}"
        )
        self.gguf_writer.add_head_count(per_layer_heads)

        # Resolve + validate the attention gate type now so an inconsistent `gating`
        # field fails at conversion time. See _attn_gate_types.
        self._attn_gate_types()

        # SWA window size (M.1 has none -> key omitted, swa_type stays NONE). The
        # per-layer SWA schedule is deliberately NOT emitted: the loader derives the
        # period-4 dense-first pattern (FULL at il%4==0) itself, matching layer_types.
        sliding_window = hparams.get("sliding_window") or 0
        if sliding_window > 0:
            self.gguf_writer.add_sliding_window(sliding_window)

        # MoE. Laguna spells the expert count "num_experts", which the base
        # set_gguf_parameters does not read (it looks for num_local_experts), so
        # expert_count is emitted here explicitly.
        self.gguf_writer.add_expert_count(self.find_hparam(["num_local_experts", "num_experts"]))
        self.gguf_writer.add_expert_shared_count(1)
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hparams["shared_expert_intermediate_size"])
        self.gguf_writer.add_expert_weights_norm(True)  # HF reference always sum-normalises after top-k
        self.gguf_writer.add_expert_weights_scale(float(hparams["moe_routed_scaling_factor"]))
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        # Leading dense layers (XS.2/S-2.1 have 1, M.1 has 3) before the MoE layers.
        mlp_layer_types: list[str] = hparams["mlp_layer_types"]
        leading_dense = 0
        for t in mlp_layer_types:
            if t == "dense":
                leading_dense += 1
            else:
                break
        self.gguf_writer.add_leading_dense_block_count(leading_dense)

        # Dual RoPE. Laguna carries per-layer-type rope config under rope_parameters:
        # the full-attention layers use YaRN over a partial rotary (head_dim * 0.5),
        # the sliding layers use plain RoPE over the full head_dim.
        head_dim = hparams["head_dim"]
        rope_parameters = hparams.get("rope_parameters", {})
        rope_full = rope_parameters.get("full_attention", {})
        rope_swa = rope_parameters.get("sliding_attention")

        self.gguf_writer.add_rope_dimension_count(
            int(head_dim * float(rope_full.get("partial_rotary_factor", 1.0))))
        if rope_theta := rope_full.get("rope_theta"):
            self.gguf_writer.add_rope_freq_base(rope_theta)
        if rope_full.get("rope_type") == "yarn":
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            if (factor := rope_full.get("factor")) is not None:
                self.gguf_writer.add_rope_scaling_factor(factor)
            if (orig_ctx := rope_full.get("original_max_position_embeddings")) is not None:
                self.gguf_writer.add_rope_scaling_orig_ctx_len(orig_ctx)
            if (attn_factor := rope_full.get("attention_factor")) is not None:
                self.gguf_writer.add_rope_scaling_yarn_attn_factor(attn_factor)
            if (beta_fast := rope_full.get("beta_fast")) is not None:
                self.gguf_writer.add_rope_scaling_yarn_beta_fast(beta_fast)
            if (beta_slow := rope_full.get("beta_slow")) is not None:
                self.gguf_writer.add_rope_scaling_yarn_beta_slow(beta_slow)

        if rope_swa is not None:
            self.gguf_writer.add_rope_dimension_count_swa(
                int(head_dim * float(rope_swa.get("partial_rotary_factor", 1.0))))
            if rope_theta_swa := rope_swa.get("rope_theta"):
                self.gguf_writer.add_rope_freq_base_swa(rope_theta_swa)

    def _attn_gate_types(self) -> list[str]:
        """Per-layer attention output gate type: "per_head" or "per_element".

        `gating_types` (per layer) is authoritative when present; otherwise the
        scalar `gating` field is used (the "per-element"/"per-head" string, or the
        legacy boolean True == per-head, as in Laguna-XS.2).

        Fails loudly when the model is per-element but the `gating` field does not
        declare that as a string: runtimes that key off `gating` (vLLM, transformers)
        ignore gating_types and read a bare boolean True as per-head, silently
        corrupting the model. Surfacing it here keeps a broken checkpoint from being
        packaged as if it were fine.
        """
        if self._gate_types is not None:
            return self._gate_types
        hparams = self.hparams
        n_layer = hparams["num_hidden_layers"]
        gating = hparams.get("gating")
        gating_types = hparams.get("gating_types")

        def _norm(t: object) -> str:
            sval = str(t).replace("-", "_")
            if sval in ("per_element", "per_head"):
                return sval
            raise ValueError(f"Laguna: unrecognised attention gate type {t!r}")

        if gating_types:
            assert len(gating_types) == n_layer, (
                f"gating_types length {len(gating_types)} != num_hidden_layers {n_layer}")
            types = [_norm(t) for t in gating_types]
        elif isinstance(gating, str):
            types = [_norm(gating)] * n_layer
        elif gating is True:
            types = ["per_head"] * n_layer
        else:
            raise ValueError(
                f"Laguna: cannot determine attention gate type "
                f"(gating={gating!r}, gating_types={gating_types!r})")

        if any(t == "per_element" for t in types) and not (
                isinstance(gating, str) and _norm(gating) == "per_element"):
            raise ValueError(
                f"Laguna config declares a per-element attention gate but "
                f"`gating`={gating!r} is not the string \"per-element\". Runtimes that "
                f"read `gating` (vLLM, transformers) will mis-handle this checkpoint as "
                f"per-head. Set gating=\"per-element\" in the source config.")

        self._gate_types = types
        return types

    # --- tensor handling -----------------------------------------------------

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # The router bias is spelled with an underscore; the tensor map keys off the
        # dotted form so the ".bias" suffix rule can find it.
        if name.endswith("e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")

        # Per-expert MoE weights: model.layers.{bid}.mlp.experts.{xid}.{w}.weight.
        # Only the NUMBERED per-expert weights are stacked; the router bias above
        # takes the normal mapping path.
        if re.search(r"mlp\.experts\.\d+\.", name):
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            needed = [f"model.layers.{bid}.mlp.experts.{x}.{w}.weight"
                      for x in range(n_experts) for w in ("gate_proj", "up_proj", "down_proj")]
            if all(e in self._experts[bid] for e in needed):
                for w_name in ["gate_proj", "up_proj", "down_proj"]:
                    datas = [self._experts[bid][f"model.layers.{bid}.mlp.experts.{x}.{w_name}.weight"]
                             for x in range(n_experts)]
                    stacked = torch.stack(datas, dim=0)
                    merged = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    yield from Model.modify_tensors(self, stacked, merged, bid)
                self._experts[bid].clear()
                return
            return

        # Cross-check the gate projection width against the declared gate type; a
        # mismatch means the weights and config disagree -> fail, do not guess.
        if bid is not None and name.endswith("self_attn.g_proj.weight"):
            heads = (self.hparams.get("num_attention_heads_per_layer")
                     or [self.hparams["num_attention_heads"]] * self.hparams["num_hidden_layers"])
            n_head = heads[bid]
            head_dim = self.hparams["head_dim"]
            gate_type = self._attn_gate_types()[bid]
            expected = n_head * head_dim if gate_type == "per_element" else n_head
            out_features = int(data_torch.shape[0])
            if out_features != expected:
                raise ValueError(
                    f"Laguna layer {bid}: g_proj output width {out_features} contradicts the "
                    f"declared {gate_type} gate (expected {expected}); weights and config disagree.")

        yield from Model.modify_tensors(self, data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")



@Model.register("HYV3ForCausalLM")
class HYV3Model(Model):
    model_arch = gguf.MODEL_ARCH.HY_V3
    _experts: list[dict] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # NextN/MTP blocks are appended past num_hidden_layers. Upstream gates this
        # behind a --mtp/--no-mtp filter API this tree does not have; instead follow
        # the idiom the other MTP archs here use (extend block_count so the appended
        # block's tensors resolve to blk.<n>.*), and only when the checkpoint really
        # ships one. The abliterated Hy3 release carries layers 0..num_hidden_layers-1
        # and no MTP block, so this is a no-op there.
        n_nextn = int(self.hparams.get("num_nextn_predict_layers", 0) or 0)
        if n_nextn > 0 and self._has_mtp_block():
            self.block_count += n_nextn
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        else:
            self._declared_nextn = n_nextn
            if n_nextn > 0:
                logger.warning(
                    f"gguf: config declares num_nextn_predict_layers={n_nextn} but the checkpoint "
                    f"has no layer >= {self.hparams['num_hidden_layers']}; converting without an MTP head")

    def _has_mtp_block(self) -> bool:
        """True only if tensors exist at a layer index past the trunk."""
        n_main = int(self.hparams["num_hidden_layers"])
        try:
            names = self.get_tensor_names() if hasattr(self, "get_tensor_names") else None
        except Exception:
            names = None
        if names is None:
            # Fall back to the safetensors index, which is cheap and always present
            # for a sharded HF checkpoint.
            import json as _json
            idx = self.dir_model / "model.safetensors.index.json"
            if not idx.is_file():
                return False
            names = _json.loads(idx.read_text(encoding="utf-8")).get("weight_map", {}).keys()
        for n in names:
            m = re.match(r"model\.layers\.(\d+)\.", n)
            if m and int(m.group(1)) >= n_main:
                return True
        return False

    def set_vocab(self):
        self._set_vocab_gpt2()
        self._fix_str_format_in_chat_template()

    def _fix_str_format_in_chat_template(self):
        """Rewrite Python str.format() calls in Hy3's chat template to jinja concat.

        Hy3 builds its special tokens with a Python string method:

            {%- set eos_token = '<|hy_eos{}|>'.format(HYTK) %}

        minja, the jinja engine llama.cpp embeds, implements no .format method, so a
        stock Hy3 conversion is unusable for chat at BOTH ends: llama-server --jinja
        aborts at init with "Callee is not a function: got Undefined (hint: 'format')",
        and without --jinja every /v1/chat/completions returns 500 because the built-in
        C++ template matcher does not recognise the template either. Net effect on a
        stock conversion: no chat completions and no tool calls, only /completion.

        jinja's concat operator is semantically identical and minja does implement it:

            {%- set eos_token = '<|hy_eos' ~ HYTK ~ '|>' %}
        """
        key = gguf.Keys.Tokenizer.CHAT_TEMPLATE
        kv = self.gguf_writer.kv_data[0].get(key) if self.gguf_writer.kv_data else None
        tmpl = getattr(kv, "value", None)
        if not isinstance(tmpl, str):
            return
        pat = re.compile(r"'([^']*)\{\}([^']*)'\.format\(([A-Za-z_][A-Za-z0-9_]*)\)")
        fixed, n = pat.subn(lambda m: "'%s' ~ %s ~ '%s'" % (m.group(1), m.group(3), m.group(2)), tmpl)
        if n:
            # kv_data is a dict, so re-adding the key replaces the existing entry.
            self.gguf_writer.add_chat_template(fixed)
            logger.info(f"gguf: rewrote {n} str.format() call(s) in the chat template to jinja "
                        f"concat (minja has no .format; --jinja aborts at init without this)")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        # Hy3 spells the routed-expert count "num_experts"; the base only looks for
        # num_local_experts, so expert_count would otherwise be omitted entirely.
        self.gguf_writer.add_expert_count(self.find_hparam(["num_local_experts", "num_experts"]))
        self.gguf_writer.add_expert_shared_count(hparams.get("num_shared_experts", 1))
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(
            hparams["moe_intermediate_size"] * hparams.get("num_shared_experts", 1))

        # Sigmoid router with a per-expert selection bias, sum-normalised after top-k.
        self.gguf_writer.add_expert_weights_norm(hparams.get("route_norm", True))
        self.gguf_writer.add_expert_weights_scale(float(hparams.get("router_scaling_factor", 1.0)))
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        # Leading dense layers before the MoE stack.
        if (dense_lead := hparams.get("first_k_dense_replace")) is not None:
            self.gguf_writer.add_leading_dense_block_count(int(dense_lead))

        # Plain RoPE (rope_type "default"); theta lives under rope_parameters.
        rope = hparams.get("rope_parameters", {}) or {}
        if rope_theta := (rope.get("rope_theta") or hparams.get("rope_theta")):
            self.gguf_writer.add_rope_freq_base(float(rope_theta))

        n_nextn = int(hparams.get("num_nextn_predict_layers", 0) or 0)
        if n_nextn > 0 and self._has_mtp_block():
            self.gguf_writer.add_nextn_predict_layers(n_nextn)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # The MTP block's trailing final_layernorm is the pre-head norm.
        if ".final_layernorm." in name:
            name = name.replace(".final_layernorm.", ".shared_head.norm.")

        # Stack the numbered per-expert weights into the merged 3D expert tensors.
        # mlp.expert_bias (the router bias) is NOT numbered and takes the normal path.
        if name.startswith("model.layers.") and re.search(r"\.mlp\.experts\.\d+\.", name):
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                for w_name in ("down_proj", "gate_proj", "up_proj"):
                    datas = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]
                    merged = torch.stack(datas, dim=0)
                    yield from Model.modify_tensors(
                        self, merged, f"model.layers.{bid}.mlp.experts.{w_name}.weight", bid)
            return

        yield from Model.modify_tensors(self, data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if experts:
                raise ValueError(f"Unprocessed experts: {experts}")



@Model.register("DeepseekV4ForCausalLM")
class DeepseekV4Model(Model):
    """DeepSeek-V4 (`model_type: deepseek_v4`, `DeepseekV4ForCausalLM`).

    Adapted from llama.cpp `conversion/deepseek.py` (class `DeepseekV4Model`)
    @ upstream commit 82dbc4f01 -- MIT, Copyright (c) 2023-2026 The ggml authors.
    Originating upstream PRs: #24162 (base) and follow-ups #25202/#25370/#25521/
    #25588/#25702/#25787.

    The HF checkpoint is NOT in transformers layout: tensors are named
    `layers.N.attn.wq_a.weight` etc., the dense weights ship as fp8 e4m3 with
    128x128 ue8m0 block scales in a sidecar `<name>.scale`, and the routed
    experts ship as MXFP4 nibble pairs with one ue8m0 byte per 32 values.

    Two deliberate deviations from upstream, both reversible by env var:

    * `PXA_DSV4_EXPERTS` (default `dequant`): the routed experts are
      DEQUANTIZED from their MXFP4 nibbles to f32 instead of being repacked raw
      into ggml MXFP4 blocks. Upstream's repack pins every expert at 4.25 bpw,
      and `llama-quantize` then short-circuits on `tensor->type == new_type`,
      so a later `--outtype PXQ2` run silently emits a 4.25-bpw file (137 GiB
      of experts, which does not fit this box). Set `PXA_DSV4_EXPERTS=mxfp4`
      for upstream's byte-for-byte behaviour.
    * `PXA_DSV4_ROUTER_BF16` (default unset): the MoE router `ffn_gate_inp` is
      written F32 rather than inheriting its BF16 source dtype (which is what
      upstream's `tensor_force_quant` ends up doing). Every other MoE arch in
      this tree writes an F32 router and the quantizer never touches it. Set to
      1 to restore upstream's BF16 router.

    NOT converted (conversion v0, matching upstream): the `mtp.*` tensors
    (MTP / DSpark). Upstream PR #25784 is still open.
    """

    model_arch = gguf.MODEL_ARCH.DEEPSEEK4
    _skipped_mtp_tensors = 0

    # e2m1 code points, in ggml's nibble order (see kvalues_mxfp4 in
    # ggml/src/ggml-common.h -- that table is DOUBLED, ggml pairs it with
    # GGML_E8M0_TO_FP32_HALF; the product is identical to what we do here).
    _MXFP4_E2M1 = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

    def __init__(self, *args, **kwargs):
        type(self)._skipped_mtp_tensors = 0
        super().__init__(*args, **kwargs)

        # DS4 keeps every hyperparameter at the top level of config.json; re-read
        # it verbatim so nothing that Model.load_hparams() filtered is lost.
        with open(self.dir_model / "config.json", "r", encoding="utf-8") as f:
            raw_hparams = json.load(f)
        for key, value in raw_hparams.items():
            self.hparams.setdefault(key, value)

        self.block_count = self.hparams["num_hidden_layers"]
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        mode = os.environ.get("PXA_DSV4_EXPERTS", "dequant").strip().lower()
        if mode not in ("dequant", "mxfp4"):
            raise ValueError(f"PXA_DSV4_EXPERTS must be 'dequant' or 'mxfp4', got {mode!r}")
        self._expert_mode = mode
        self._router_bf16 = os.environ.get("PXA_DSV4_ROUTER_BF16", "").strip() not in ("", "0")

        if self._expert_mode == "mxfp4":
            # the experts are written with raw_dtype=MXFP4, so the file really is
            # an MXFP4 file whatever --outtype said. Declare it up front so the
            # general.file_type KV and the default filename are honest.
            logger.warning("DeepSeek-V4: PXA_DSV4_EXPERTS=mxfp4 -- routed experts are repacked, "
                           "NOT dequantized. A later --outtype PXQ*/IQ* run will NOT be able to "
                           "re-quantize them without --allow-requantize.")
            self.ftype = gguf.LlamaFileType.MOSTLY_MXFP4

        self.model_tensors: dict[str, Callable[[], Tensor]] = self.index_tensors()

        self._dsv4_fp8_dequantized: set[str] = set()
        self._dsv4_bf16_tensors: set[str] = set()
        self._dsv4_f32_tensors: set[str] = set()
        self._dsv4_extra_names: set[str] = set()
        self._dsv4_extra_generated = False

        # ORDER IS LOAD-BEARING: the source dtypes must be recorded BEFORE
        # dequant_model() replaces the fp8 loaders, or every tensor looks f32
        # and tensor_force_quant() pins the whole model wrong, silently.
        self._collect_source_dtypes()
        self.dequant_model()

        if type(self)._skipped_mtp_tensors:
            logger.info("Skipping %d DeepSeek-V4 MTP tensor(s) for conversion v0",
                        type(self)._skipped_mtp_tensors)

        # add a default chat template; a built-in template overrides it later
        template_path = Path(__file__).parent / "models" / "templates" / "deepseek-ai-DeepSeek-V4.jinja"
        if template_path.is_file():
            with open(template_path, "r", encoding="utf-8") as f:
                self.gguf_writer.add_chat_template(f.read())

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        if name.startswith("mtp."):
            cls._skipped_mtp_tensors += 1
            return None
        return super().filter_tensors(item)

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        model_tensors = getattr(self, "model_tensors", None)
        if model_tensors is None:
            # Model.__init__ probes the first tensor (--outtype auto) before the
            # index exists; fall back to the streaming reader for that one probe.
            yield from Model.get_tensors(self)
            return
        for name in list(model_tensors.keys()):
            yield name, model_tensors[name]()

    @staticmethod
    def _float8_dtypes() -> tuple[torch.dtype, ...]:
        return tuple(
            dtype for dtype in (
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e5m2", None),
            ) if dtype is not None
        )

    @staticmethod
    def _e8m0_to_float(scale: Tensor) -> Tensor:
        torch_float8_e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if torch_float8_e8m0 is not None and scale.dtype == torch_float8_e8m0:
            return scale.float()

        bits = scale.view(torch.uint8).float()
        return torch.exp2(bits - 127.0)

    def _collect_source_dtypes(self) -> None:
        for name, gen in self.model_tensors.items():
            dtype = gen().dtype
            if dtype == torch.bfloat16:
                self._dsv4_bf16_tensors.add(name)
            elif dtype == torch.float32:
                self._dsv4_f32_tensors.add(name)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])
        self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])

        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])

        score_func = hparams.get("scoring_func")
        if score_func != "sqrtsoftplus":
            raise ValueError(f"DeepSeek-V4 expects scoring_func == 'sqrtsoftplus', got {score_func!r}")
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SQRTSOFTPLUS)

        self.gguf_writer.add_swiglu_clamp_exp([hparams["swiglu_limit"]] * self.block_count)
        self.gguf_writer.add_swiglu_clamp_shexp([hparams["swiglu_limit"]] * self.block_count)

        self.gguf_writer.add_indexer_head_count(hparams["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(hparams["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(hparams["index_topk"])

        self.gguf_writer.add_attention_output_group_count(hparams["o_groups"])
        self.gguf_writer.add_attention_output_lora_rank(hparams["o_lora_rank"])

        compress_ratios = list(hparams["compress_ratios"])
        # The real 0731 checkpoint ships 46 entries for 43 layers: indices
        # 43..45 belong to the (dropped) mtp.0..2 blocks. Written verbatim,
        # exactly as upstream does -- the loader must index by layer, not
        # assume len() == n_layer.
        if len(compress_ratios) != self.block_count:
            logger.info("DeepSeek-V4: compress_ratios has %d entries for %d layers "
                        "(the tail belongs to the MTP blocks); writing it verbatim",
                        len(compress_ratios), self.block_count)
        for ratio in compress_ratios:
            if ratio not in (0, 4, 128):
                raise ValueError(f"Unsupported DeepSeek-V4 compress ratio {ratio!r} "
                                 f"(expected one of 0 / 4 / 128)")
        self.gguf_writer.add_attention_compress_ratios(compress_ratios)
        self.gguf_writer.add_attention_compress_rope_freq_base(hparams["compress_rope_theta"])

        self.gguf_writer.add_hyper_connection_count(hparams["hc_mult"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(hparams["hc_sinkhorn_iters"])
        self.gguf_writer.add_hyper_connection_epsilon(hparams["hc_eps"])
        self.gguf_writer.add_hash_layer_count(hparams["num_hash_layers"])

        # RoPE scaling. Upstream's shared base writes this; ours does not, so the
        # DS4 class has to. Without it a >65536 context ropes wrong, silently.
        rope_scaling = hparams.get("rope_scaling") or {}
        if rope_scaling.get("type") == "yarn" and "factor" in rope_scaling:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_scaling["original_max_position_embeddings"])
            if (beta_fast := rope_scaling.get("beta_fast")) is not None:
                self.gguf_writer.add_rope_scaling_yarn_beta_fast(beta_fast)
            if (beta_slow := rope_scaling.get("beta_slow")) is not None:
                self.gguf_writer.add_rope_scaling_yarn_beta_slow(beta_slow)
            if (mscale_all_dim := rope_scaling.get("mscale_all_dim")) is not None:
                self.gguf_writer.add_rope_scaling_yarn_log_mul(0.1 * mscale_all_dim)

    def dequant_model(self):
        """fp8 e4m3 weight * ue8m0 128x128 block scale -> f32.

        NOTE this is DS4's own convention, NOT DeepSeek-V3's: the sidecar is
        `<weight>.scale` (not `_scale_inv`), it is raw ue8m0, the block size is
        hardcoded 128x128, and the operation is a MULTIPLY.
        """
        fp8_dtypes = self._float8_dtypes()
        tensors_to_remove: list[str] = []

        def dequant_fp8_weight(weight: Tensor, scale: Tensor) -> Tensor:
            out_features, in_features = weight.shape
            scale_f = self._e8m0_to_float(scale)
            scale_f = scale_f.repeat_interleave(128, 0)[:out_features]
            scale_f = scale_f.repeat_interleave(128, 1)[:, :in_features]
            return weight.float() * scale_f

        for name in list(self.model_tensors.keys()):
            if not name.endswith(".scale"):
                continue
            weight_name = name.removesuffix(".scale") + ".weight"
            if weight_name not in self.model_tensors:
                continue

            weight = self.model_tensors[weight_name]
            scale = self.model_tensors[name]
            if weight().dtype not in fp8_dtypes:
                # the routed experts are MXFP4 (int8 nibble pairs), not fp8 --
                # their scale is consumed by the expert path, keep it.
                continue

            self.model_tensors[weight_name] = lambda w=weight, s=scale: dequant_fp8_weight(w(), s())
            self._dsv4_fp8_dequantized.add(weight_name)
            tensors_to_remove.append(name)

        for name in tensors_to_remove:
            del self.model_tensors[name]

        logger.info("DeepSeek-V4: %d fp8 tensor(s) will be dequantized with ue8m0 128x128 block scales",
                    len(self._dsv4_fp8_dequantized))

    # ---------------------------------------------------------------- experts
    @staticmethod
    def _pack_mxfp4_blocks(weight: Tensor, scale: Tensor) -> np.ndarray:
        """Upstream's raw repack into the ggml MXFP4 block layout (17 B / 32 values).

        Only used when PXA_DSV4_EXPERTS=mxfp4. Verbatim from upstream.
        """
        packed = weight.contiguous().view(torch.uint8)
        scale_u8 = scale.contiguous().view(torch.uint8)

        out_features, packed_cols = packed.shape
        logical_cols = packed_cols * 2
        if logical_cols % 32 != 0:
            raise ValueError(f"MXFP4 source row has {logical_cols} values, expected a multiple of 32")

        n_blocks = logical_cols // 32
        if tuple(scale_u8.shape) != (out_features, n_blocks):
            raise ValueError(f"MXFP4 scale shape {tuple(scale_u8.shape)} does not match {(out_features, n_blocks)}")

        src = packed.reshape(out_features, n_blocks, 16)
        low = src & 0x0F
        high = (src >> 4) & 0x0F

        # The safetensors bytes store adjacent values as low/high nibbles.
        # ggml MXFP4 blocks store values 0..15 in low nibbles and 16..31 in high nibbles.
        vals = torch.stack((low, high), dim=-1).reshape(out_features, n_blocks, 32)
        qs = vals[:, :, :16] | (vals[:, :, 16:] << 4)
        raw = torch.cat((scale_u8.unsqueeze(-1), qs.to(torch.uint8)), dim=-1)
        return raw.reshape(out_features, n_blocks * 17).cpu().numpy()

    @classmethod
    def _unpack_mxfp4_blocks(cls, weight: Tensor, scale: Tensor) -> np.ndarray:
        """MXFP4 nibbles + ue8m0 block scales -> f32, bit-for-bit what ggml's own
        dequantize_row_mxfp4() produces: e2m1[nibble] * 2**(scale_byte - 127)."""
        packed = weight.contiguous().view(torch.uint8).cpu().numpy()
        scale_u8 = scale.contiguous().view(torch.uint8).cpu().numpy()

        out_features, packed_cols = packed.shape
        logical_cols = packed_cols * 2
        if logical_cols % 32 != 0:
            raise ValueError(f"MXFP4 source row has {logical_cols} values, expected a multiple of 32")

        n_blocks = logical_cols // 32
        if tuple(scale_u8.shape) != (out_features, n_blocks):
            raise ValueError(f"MXFP4 scale shape {tuple(scale_u8.shape)} does not match {(out_features, n_blocks)}")

        # adjacent logical values are the low/high nibble of one source byte
        vals = np.empty((out_features, logical_cols), dtype=np.uint8)
        vals[:, 0::2] = packed & 0x0F
        vals[:, 1::2] = packed >> 4

        out = cls._MXFP4_E2M1[vals]
        out *= np.repeat(np.exp2(scale_u8.astype(np.float32) - 127.0), 32, axis=1)
        return out

    def _expert_source_names(self, bid: int, proj: str) -> list[tuple[str, str]]:
        n_experts = self.hparams["n_routed_experts"]
        names: list[tuple[str, str]] = []
        for eid in range(n_experts):
            weight_name = f"layers.{bid}.ffn.experts.{eid}.{proj}.weight"
            scale_name = f"layers.{bid}.ffn.experts.{eid}.{proj}.scale"
            if weight_name not in self.model_tensors or scale_name not in self.model_tensors:
                raise KeyError(f"Missing routed expert tensors for {weight_name}")
            names.append((weight_name, scale_name))
        return names

    def _write_mxfp4_expert_tensor(self, bid: int, proj: str, tensor_key: gguf.MODEL_TENSOR) -> list[str]:
        """Upstream path (PXA_DSV4_EXPERTS=mxfp4): repack, write raw MXFP4."""
        data: np.ndarray | None = None
        consumed: list[str] = []

        for eid, (weight_name, scale_name) in enumerate(self._expert_source_names(bid, proj)):
            weight = LazyTorchTensor.to_eager(self.model_tensors[weight_name]())
            scale = LazyTorchTensor.to_eager(self.model_tensors[scale_name]())
            packed = self._pack_mxfp4_blocks(weight, scale)
            if data is None:
                data = np.empty((self.hparams["n_routed_experts"], *packed.shape), dtype=packed.dtype)
            data[eid] = packed
            consumed.extend((weight_name, scale_name))

        assert data is not None
        new_name = self.format_tensor_name(tensor_key, bid)
        shape = gguf.quant_shape_from_byte_shape(data.shape, gguf.GGMLQuantizationType.MXFP4)
        logger.info(f"{new_name}: repacked routed experts to MXFP4, shape = {{{', '.join(str(n) for n in reversed(shape))}}}")
        self.gguf_writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.MXFP4)

        return consumed

    def _dequant_expert_tensor(self, bid: int, proj: str, tensor_key: gguf.MODEL_TENSOR) -> tuple[str, Tensor, list[str]]:
        """PXA path (default): unpack the MXFP4 experts to a stacked f32 tensor and
        let the normal --outtype machinery own it, so `llama-quantize` sees a real
        high-precision source instead of a 4.25-bpw one it will refuse to touch."""
        n_experts = self.hparams["n_routed_experts"]
        names = self._expert_source_names(bid, proj)
        gens = [(self.model_tensors[w], self.model_tensors[s]) for w, s in names]
        consumed = [n for pair in names for n in pair]

        probe = self.model_tensors[names[0][0]]()
        rows, packed_cols = int(probe.shape[0]), int(probe.shape[1])
        shape = (n_experts, rows, packed_cols * 2)

        unpack = self._unpack_mxfp4_blocks

        def build(pairs: list[tuple[Callable[[], Tensor], Callable[[], Tensor]]]) -> Tensor:
            out = np.empty(shape, dtype=np.float32)
            for i, (wgen, sgen) in enumerate(pairs):
                w = LazyTorchTensor.to_eager(wgen())
                s = LazyTorchTensor.to_eager(sgen())
                out[i] = unpack(w, s)
                del w, s
            return torch.from_numpy(out)

        lazy = LazyTorchTensor(
            meta=LazyTorchTensor.meta_with_dtype_and_shape(torch.float32, shape),
            args=(gens,),
            func=build,
        )
        return self.format_tensor_name(tensor_key, bid), cast("Tensor", lazy), consumed

    def _write_hash_routing_tensors(self) -> list[str]:
        consumed: list[str] = []

        for bid in range(self.hparams["num_hash_layers"]):
            name = f"layers.{bid}.ffn.gate.tid2eid"
            if name not in self.model_tensors:
                raise KeyError(f"Missing hash routing tensor {name}")

            data_torch = LazyTorchTensor.to_eager(self.model_tensors[name]())
            data = data_torch.to(torch.int32).cpu().numpy()
            new_name = self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_TID2EID, bid, ".weight")
            logger.info(f"{new_name}: converted hash routing table to I32, shape = {{{', '.join(str(n) for n in reversed(data.shape))}}}")
            self.gguf_writer.add_tensor(new_name, data)
            consumed.append(name)

        return consumed

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if self._dsv4_extra_generated:
            return ()
        self._dsv4_extra_generated = True

        consumed: list[str] = self._write_hash_routing_tensors()
        extra: list[tuple[str, Tensor]] = []

        # w1 -> FFN_GATE_EXP, w2 -> FFN_DOWN_EXP, w3 -> FFN_UP_EXP.
        # (upstream's _map_dsv4_tensor_name() answers FFN_GATE_EXP for all three;
        #  that branch is FILTER-ONLY and copying it here would silently corrupt
        #  every MoE layer.)
        for bid in range(self.block_count):
            for proj, tensor_key in (("w1", gguf.MODEL_TENSOR.FFN_GATE_EXP),
                                     ("w2", gguf.MODEL_TENSOR.FFN_DOWN_EXP),
                                     ("w3", gguf.MODEL_TENSOR.FFN_UP_EXP)):
                if self._expert_mode == "mxfp4":
                    consumed.extend(self._write_mxfp4_expert_tensor(bid, proj, tensor_key))
                else:
                    new_name, tensor, used = self._dequant_expert_tensor(bid, proj, tensor_key)
                    self._dsv4_extra_names.add(new_name)
                    extra.append((new_name, tensor))
                    consumed.extend(used)

        for name in consumed:
            del self.model_tensors[name]

        return extra

    def _format_dsv4_tensor_name(self, key: gguf.MODEL_TENSOR, bid: int | None, suffix: str = ".weight") -> str:
        return self.format_tensor_name(key, bid, suffix)

    def _map_dsv4_tensor_name(self, name: str, bid: int | None) -> tuple[gguf.MODEL_TENSOR, str]:
        root_map: dict[str, tuple[gguf.MODEL_TENSOR, str]] = {
            "embed.weight": (gguf.MODEL_TENSOR.TOKEN_EMBD, ".weight"),
            "norm.weight": (gguf.MODEL_TENSOR.OUTPUT_NORM, ".weight"),
            "head.weight": (gguf.MODEL_TENSOR.OUTPUT, ".weight"),
            "hc_head_fn": (gguf.MODEL_TENSOR.HC_HEAD_FN, ".weight"),
            "hc_head_base": (gguf.MODEL_TENSOR.HC_HEAD_BASE, ".weight"),
            "hc_head_scale": (gguf.MODEL_TENSOR.HC_HEAD_SCALE, ".weight"),
        }
        if name in root_map:
            return root_map[name]

        match = re.match(r"layers\.(\d+)\.(.+)$", name)
        if match is None:
            raise ValueError(f"Unsupported DeepSeek-V4 tensor {name!r}")

        layer = int(match.group(1))
        if bid != layer:
            raise ValueError(f"Tensor {name!r} parsed bid {bid} but layer name has {layer}")

        layer_map: dict[str, tuple[gguf.MODEL_TENSOR, str]] = {
            "hc_attn_fn": (gguf.MODEL_TENSOR.HC_ATTN_FN, ".weight"),
            "hc_attn_base": (gguf.MODEL_TENSOR.HC_ATTN_BASE, ".weight"),
            "hc_attn_scale": (gguf.MODEL_TENSOR.HC_ATTN_SCALE, ".weight"),
            "hc_ffn_fn": (gguf.MODEL_TENSOR.HC_FFN_FN, ".weight"),
            "hc_ffn_base": (gguf.MODEL_TENSOR.HC_FFN_BASE, ".weight"),
            "hc_ffn_scale": (gguf.MODEL_TENSOR.HC_FFN_SCALE, ".weight"),
            "attn.attn_sink": (gguf.MODEL_TENSOR.ATTN_SINKS, ".weight"),
            "attn.wq_a.weight": (gguf.MODEL_TENSOR.ATTN_Q_A, ".weight"),
            "attn.wq_b.weight": (gguf.MODEL_TENSOR.ATTN_Q_B, ".weight"),
            "attn.q_norm.weight": (gguf.MODEL_TENSOR.ATTN_Q_A_NORM, ".weight"),
            "attn.wkv.weight": (gguf.MODEL_TENSOR.ATTN_KV, ".weight"),
            "attn.kv_norm.weight": (gguf.MODEL_TENSOR.ATTN_KV_NORM, ".weight"),
            "attn.wo_a.weight": (gguf.MODEL_TENSOR.ATTN_OUT_A, ".weight"),
            "attn.wo_b.weight": (gguf.MODEL_TENSOR.ATTN_OUT_B, ".weight"),
            "attn.compressor.ape": (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_APE, ".weight"),
            "attn.compressor.wkv.weight": (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_WKV, ".weight"),
            "attn.compressor.wgate.weight": (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_WGATE, ".weight"),
            "attn.compressor.norm.weight": (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_NORM, ".weight"),
            "attn.indexer.wq_b.weight": (gguf.MODEL_TENSOR.INDEXER_ATTN_Q_B, ".weight"),
            "attn.indexer.weights_proj.weight": (gguf.MODEL_TENSOR.INDEXER_PROJ, ".weight"),
            "attn.indexer.compressor.ape": (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_APE, ".weight"),
            "attn.indexer.compressor.wkv.weight": (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_WKV, ".weight"),
            "attn.indexer.compressor.wgate.weight": (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_WGATE, ".weight"),
            "attn.indexer.compressor.norm.weight": (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_NORM, ".weight"),
            "attn_norm.weight": (gguf.MODEL_TENSOR.ATTN_NORM, ".weight"),
            "ffn_norm.weight": (gguf.MODEL_TENSOR.FFN_NORM, ".weight"),
            "ffn.gate.weight": (gguf.MODEL_TENSOR.FFN_GATE_INP, ".weight"),
            "ffn.gate.bias": (gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, ".bias"),
            "ffn.gate.tid2eid": (gguf.MODEL_TENSOR.FFN_GATE_TID2EID, ".weight"),
            "ffn.shared_experts.w1.weight": (gguf.MODEL_TENSOR.FFN_GATE_SHEXP, ".weight"),
            "ffn.shared_experts.w2.weight": (gguf.MODEL_TENSOR.FFN_DOWN_SHEXP, ".weight"),
            "ffn.shared_experts.w3.weight": (gguf.MODEL_TENSOR.FFN_UP_SHEXP, ".weight"),
        }

        tensor_name = match.group(2)
        if tensor_name in layer_map:
            return layer_map[tensor_name]

        raise ValueError(f"Unsupported DeepSeek-V4 tensor {name!r}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # tensors emitted by generate_extra_tensors() are already GGUF-named
        if name in self._dsv4_extra_names:
            return [(name, data_torch)]

        # routed experts are handled wholesale by generate_extra_tensors()
        if re.match(r"layers\.\d+\.ffn\.experts\.\d+\.w[123]\.(weight|scale)$", name):
            return []

        tensor_key, suffix = self._map_dsv4_tensor_name(name, bid)
        if tensor_key == gguf.MODEL_TENSOR.FFN_GATE_TID2EID:
            return []

        new_name = self._format_dsv4_tensor_name(tensor_key, bid, suffix)

        # prepare_tensors() does `d.squeeze().numpy()`, which turns a 1-ELEMENT
        # tensor into a 0-d array and would emit a GGUF tensor with n_dims == 0
        # (ggml requires >= 1). DS4 has exactly one such tensor, `hc_head_scale`
        # with shape [1]. Upstream dropped the squeeze from its own loop; we do
        # not own that shared code, so write these directly with an explicit 1-D
        # shape instead. F32: it is one scalar per model.
        if math.prod(tuple(data_torch.shape)) == 1:
            arr = LazyTorchTensor.to_eager(data_torch).reshape(1).float().cpu().numpy()
            logger.info(f"{new_name}: 1-element tensor written directly as F32 [1]")
            self.gguf_writer.add_tensor(new_name, arr)
            return []

        return [(new_name, data_torch)]

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        del bid  # unused

        # PXA deviation (see the class docstring): keep the MoE router F32.
        if not self._router_bf16 and new_name.endswith("ffn_gate_inp.weight"):
            return gguf.GGMLQuantizationType.F32

        if name in self._dsv4_fp8_dequantized and n_dims >= 2:
            return gguf.GGMLQuantizationType.Q8_0
        if name in self._dsv4_f32_tensors:
            return gguf.GGMLQuantizationType.F32
        if name in self._dsv4_bf16_tensors and n_dims >= 2:
            return gguf.GGMLQuantizationType.BF16

        if self._expert_mode == "mxfp4":
            # self.ftype is MOSTLY_MXFP4 in that mode and the base ftype ladder
            # cannot resolve it; every non-expert tensor must be pinned here.
            return gguf.GGMLQuantizationType.BF16 if n_dims >= 2 else gguf.GGMLQuantizationType.F32

        return False


###### CONVERSION LOGIC ######


# tree of lazy tensors
class LazyTorchTensor(gguf.LazyBase):
    _tensor_type = torch.Tensor
    # to keep the type-checker happy
    dtype: torch.dtype
    shape: torch.Size

    # only used when converting a torch.Tensor to a np.ndarray
    _dtype_map: dict[torch.dtype, type] = {
        torch.float16: np.float16,
        torch.float32: np.float32,
    }

    # used for safetensors slices
    # ref: https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/src/lib.rs#L1046
    # TODO: uncomment U64, U32, and U16, ref: https://github.com/pytorch/pytorch/issues/58734
    _dtype_str_map: dict[str, torch.dtype] = {
        "F64": torch.float64,
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        # "U64": torch.uint64,
        "I64": torch.int64,
        # "U32": torch.uint32,
        "I32": torch.int32,
        # "U16": torch.uint16,
        "I16": torch.int16,
        "U8": torch.uint8,
        "I8": torch.int8,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
        # DeepSeek-V4 ships ue8m0 block scales next to its fp8 e4m3 weights.
        # torch.float8_e8m0fnu exists from torch 2.7; older torch reads the raw
        # bytes instead (DeepseekV4Model._e8m0_to_float handles both).
        "F8_E8M0": getattr(torch, "float8_e8m0fnu", torch.uint8),
    }

    def numpy(self) -> gguf.LazyNumpyTensor:
        dtype = self._dtype_map[self.dtype]
        return gguf.LazyNumpyTensor(
            meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(dtype, self.shape),
            args=(self,),
            func=(lambda s: s.numpy())
        )

    @classmethod
    def meta_with_dtype_and_shape(cls, dtype: torch.dtype, shape: tuple[int, ...]) -> Tensor:
        return torch.empty(size=shape, dtype=dtype, device="meta")

    @classmethod
    def from_safetensors_slice(cls, st_slice: Any) -> Tensor:
        dtype = cls._dtype_str_map[st_slice.get_dtype()]
        shape: tuple[int, ...] = tuple(st_slice.get_shape())
        lazy = cls(meta=cls.meta_with_dtype_and_shape(dtype, shape), args=(st_slice,), func=lambda s: s[:])
        return cast(torch.Tensor, lazy)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.numpy:
            return args[0].numpy()

        return cls._wrap_fn(func)(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a huggingface model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input. {ftype} will be replaced by the outtype.",
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q6_0", "auto"], default="f16",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, q4_0, q4_1, q5_0, q5_1, q6_0 for a smaller conversion to then create an iMatrix file for example, and auto for the highest-fidelity 16-bit float type depending on the first loaded tensor type",
    )
    parser.add_argument(
        "--bigendian", action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "model", type=Path,
        help="directory containing model file",
    )
    parser.add_argument(
        "--use-temp-file", action="store_true",
        help="use the tempfile library while processing (helpful when running out of memory, process killed)",
    )
    parser.add_argument(
        "--no-lazy", action="store_true",
        help="use more RAM by computing all outputs before writing (use in case lazy evaluation is broken)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="name of the model",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--split-max-tensors", type=int, default=0,
        help="max tensors in each split",
    )
    parser.add_argument(
        "--split-max-size", type=str, default="0",
        help="max size per split N(M|G)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="only print out a split plan and exit, without writing any new files",
    )
    parser.add_argument(
        "--no-tensor-first-split", action="store_true",
        help="do not add tensors to the first split (disabled by default)"
    )
    parser.add_argument(
        "--metadata", type=Path,
        help="Specify the path for an authorship metadata override file"
    )

    return parser.parse_args()


def split_str_to_n_bytes(split_str: str) -> int:
    if split_str.endswith("K"):
        n = int(split_str[:-1]) * 1000
    elif split_str.endswith("M"):
        n = int(split_str[:-1]) * 1000 * 1000
    elif split_str.endswith("G"):
        n = int(split_str[:-1]) * 1000 * 1000 * 1000
    elif split_str.isnumeric():
        n = int(split_str)
    else:
        raise ValueError(f"Invalid split size: {split_str}, must be a number, optionally followed by K, M, or G")

    if n < 0:
        raise ValueError(f"Invalid split size: {split_str}, must be positive")

    return n


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    dir_model = args.model

    if not dir_model.is_dir():
        logger.error(f'Error: {args.model} is not a directory')
        sys.exit(1)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
        "q4_1": gguf.LlamaFileType.MOSTLY_Q4_1,
        "q5_0": gguf.LlamaFileType.MOSTLY_Q5_0,
        "q5_1": gguf.LlamaFileType.MOSTLY_Q5_1,
        "q6_0": gguf.LlamaFileType.MOSTLY_Q6_0,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    is_split = args.split_max_tensors > 0 or args.split_max_size != "0"
    if args.use_temp_file and is_split:
        logger.error("Error: Cannot use temp file when splitting")
        sys.exit(1)

    if args.outfile is not None:
        fname_out = args.outfile
    else:
        fname_out = dir_model

    logger.info(f"Loading model: {dir_model.name}")

    hparams = Model.load_hparams(dir_model)

    with torch.inference_mode():
        output_type = ftype_map[args.outtype]
        model_architecture = hparams["architectures"][0]

        try:
            model_class = Model.from_model_architecture(model_architecture)
        except NotImplementedError:
            logger.error(f"Model {model_architecture} is not supported")
            sys.exit(1)

        model_instance = model_class(dir_model=dir_model, ftype=output_type, fname_out=fname_out,
                                     is_big_endian=args.bigendian, use_temp_file=args.use_temp_file,
                                     eager=args.no_lazy,
                                     metadata_override=args.metadata, model_name=args.model_name,
                                     split_max_tensors=args.split_max_tensors,
                                     split_max_size=split_str_to_n_bytes(args.split_max_size), dry_run=args.dry_run,
                                     small_first_shard=args.no_tensor_first_split)

        if args.vocab_only:
            logger.info("Exporting model vocab...")
            model_instance.write_vocab()
            logger.info(f"Model vocab successfully exported to {model_instance.fname_out}")
        else:
            logger.info("Exporting model...")
            model_instance.write()
            out_path = f"{model_instance.fname_out.parent}{os.sep}" if is_split else model_instance.fname_out
            logger.info(f"Model successfully exported to {out_path}")


if __name__ == '__main__':
    main()
