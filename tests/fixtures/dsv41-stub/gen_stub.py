#!/usr/bin/env python3
"""Tiny synthetic deepseek41 (DeepSeek-V4.1-Flash) GGUF for loader tests.

Same key spellings and tensor-name pattern as the real 40-layer file (the ISTA
GGUF header; lanes/NAMES-deepseek41.md), shrunk to 6 layers so it loads in
milliseconds. Values are seeded random numbers: this file checks loading and
binding, not numerics. --variant builds the negative cases the loader must reject.

Usage: gen_stub.py OUT.gguf [--variant NAME]
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'gguf-py'))
try:
    import yaml  # noqa: F401  (gguf.metadata imports it; only model-card parsing uses it)
except ImportError:  # minimal build containers ship numpy but not PyYAML: the writer never needs it
    import types
    sys.modules['yaml'] = types.ModuleType('yaml')
import gguf  # noqa: E402

VARIANTS = ['ok', 'noengram', 'ratio4', 'extra_hchead', 'missing_gate', 'gate_on_ratio1',
            'kvsrc_ratio0', 'hash', 'qnorm', 'tid2eid', 'cand_not_src']

A = 'deepseek41'
N_LAYER, N_EMBD, N_HEAD, HD, ROPE = 6, 64, 4, 32, 16
Q_LORA, O_GROUPS, O_LORA, HC = 16, 2, 8, 4
N_EXP, N_USED, N_FF = 8, 2, 16
IDX_H, IDX_D = 2, 8
N_VOCAB = 256
ENG_NGRAM, ENG_H, ENG_D = 4, 2, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--variant', default='ok', choices=VARIANTS)
    a = ap.parse_args()
    v = a.variant
    rng = np.random.default_rng(41)

    ratios = [0, 0, 2, 2, 1, 1]
    if v == 'ratio4':
        ratios[3] = 4
    kv_src = [2, 4]
    if v == 'kvsrc_ratio0':
        kv_src = [1, 2, 4]
    idx_src = [2, 3, 4, 5]
    eng = [] if v == 'noengram' else [1, 3]
    cand_src = 3 if v == 'cand_not_src' else 4

    w = gguf.GGUFWriter(a.out, A)
    w.add_block_count(N_LAYER)
    w.add_context_length(4096)
    w.add_embedding_length(N_EMBD)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(1)
    w.add_key_length(HD)
    w.add_value_length(HD)
    w.add_rope_dimension_count(ROPE)
    w.add_rope_freq_base(10000.0)
    w.add_string(f'{A}.rope.scaling.type', 'yarn')
    w.add_float32(f'{A}.rope.scaling.factor', 16.0)
    w.add_uint32(f'{A}.rope.scaling.original_context_length', 256)
    w.add_float32(f'{A}.rope.scaling.yarn_beta_fast', 32.0)
    w.add_float32(f'{A}.rope.scaling.yarn_beta_slow', 1.0)
    w.add_layer_norm_rms_eps(1e-20)
    w.add_uint32(f'{A}.vocab_size', N_VOCAB)
    w.add_expert_count(N_EXP)
    w.add_expert_used_count(N_USED)
    w.add_uint32(f'{A}.expert_gating_func', 4)
    w.add_string(f'{A}.expert_scoring_func', 'sqrtsoftplus')
    w.add_uint32(f'{A}.attention.q_lora_rank', Q_LORA)
    w.add_uint32(f'{A}.attention.sliding_window', 8)
    w.add_uint32(f'{A}.expert_feed_forward_length', N_FF)
    w.add_uint32(f'{A}.expert_shared_count', 1)
    w.add_float32(f'{A}.expert_weights_scale', 1.5)
    w.add_bool(f'{A}.expert_weights_norm', True)
    w.add_array(f'{A}.swiglu_clamp_exp', [10.0] * N_LAYER)
    w.add_array(f'{A}.swiglu_clamp_shexp', [10.0] * N_LAYER)
    w.add_uint32(f'{A}.attention.indexer.head_count', IDX_H)
    w.add_uint32(f'{A}.attention.indexer.key_length', IDX_D)
    w.add_uint32(f'{A}.attention.indexer.top_k', 4)
    w.add_uint32(f'{A}.attention.output_group_count', O_GROUPS)
    w.add_uint32(f'{A}.attention.output_lora_rank', O_LORA)
    # 40 layers + 3 DSpark stages in the real file: the trailing entries are padding
    w.add_array(f'{A}.attention.compress_ratios', ratios + [0, 0, 0])
    w.add_float32(f'{A}.attention.compress_rope_freq_base', 160000.0)
    w.add_uint32(f'{A}.hyper_connection.count', HC)
    w.add_uint32(f'{A}.hyper_connection.sinkhorn_iterations', 20)
    w.add_float32(f'{A}.hyper_connection.epsilon', 1e-6)
    w.add_uint32(f'{A}.hash_layer_count', 1 if v == 'hash' else 0)
    w.add_uint32(f'{A}.attention.candidate_block_size', 2)
    w.add_uint32(f'{A}.attention.candidate_topk_blocks', 4)
    w.add_uint32(f'{A}.attention.candidate_source_layer_id', cand_src)
    w.add_array(f'{A}.kv_source_layer_ids', kv_src)
    w.add_array(f'{A}.index_source_layer_ids', idx_src)
    w.add_array(f'{A}.dspark_target_layer_ids', [3, 4, 5])
    w.add_bool(f'{A}.engram.present', bool(eng))
    if eng:
        w.add_array(f'{A}.engram.layer_ids', eng)
        w.add_array(f'{A}.engram.num_embeddings', [1000, 1003])
        w.add_uint32(f'{A}.engram_head_dim', ENG_D)
        w.add_uint32(f'{A}.engram_n_heads', ENG_H)
        w.add_uint32(f'{A}.engram_max_ngram_size', ENG_NGRAM)
        w.add_uint32(f'{A}.engram_vocab_size', 4000)
        w.add_uint32(f'{A}.engram_compressed_vocab_size', 99)
        w.add_uint32(f'{A}.engram_pad_token_id', 2)
    w.add_string('tokenizer.ggml.model', 'none')

    def t(name, ne, dtype=np.float32):
        shape = tuple(reversed(ne))  # ggml ne order -> numpy order
        if dtype == np.float32:
            arr = rng.standard_normal(shape).astype(np.float32) * 0.02
        else:
            arr = rng.integers(0, 97, size=shape).astype(dtype)
        w.add_tensor(name, arr)

    hc_mix = (2 + HC) * HC
    t('token_embd.weight', [N_EMBD, N_VOCAB])
    t('output.weight', [N_EMBD, N_VOCAB])
    t('output_norm.weight', [N_EMBD])
    if v == 'extra_hchead':
        t('output_hc_fn.weight', [HC * N_EMBD, HC])
    if eng:
        n_hash = (ENG_NGRAM - 1) * ENG_H
        t('engram_token_map.weight', [N_VOCAB], np.int32)
        t('engram_primes.weight', [n_hash, len(eng)], np.int64)
        t('engram_offsets.weight', [n_hash, len(eng)], np.int64)
        t('engram_multipliers.weight', [ENG_NGRAM, len(eng)], np.int64)
    for i in range(N_LAYER):
        b = f'blk.{i}.'
        t(b + 'attn_norm.weight', [N_EMBD])
        t(b + 'attn_sinks.weight', [N_HEAD])
        t(b + 'attn_q_a.weight', [N_EMBD, Q_LORA])
        t(b + 'attn_q_a_norm.weight', [Q_LORA])
        t(b + 'attn_q_b.weight', [Q_LORA, N_HEAD * HD])
        t(b + 'attn_kv.weight', [N_EMBD, HD])
        t(b + 'attn_kv_a_norm.weight', [HD])
        t(b + 'attn_output_a.weight', [N_HEAD * HD // O_GROUPS, O_LORA, O_GROUPS])
        t(b + 'attn_output_b.weight', [O_GROUPS * O_LORA, N_EMBD])
        if v == 'qnorm':
            t(b + 'attn_q_norm.weight', [HD])
        for k in ('attn', 'ffn'):
            t(b + f'hc_{k}_fn.weight', [HC * N_EMBD, hc_mix])
            t(b + f'hc_{k}_base.weight', [hc_mix])
            t(b + f'hc_{k}_scale.weight', [3])
        if i in kv_src:
            t(b + 'attn_compressor_kv.weight', [N_EMBD, HD])
            t(b + 'attn_compressor_norm.weight', [HD])
            gate = ratios[i] > 1
            if v == 'missing_gate' and i == 2:
                gate = False
            if v == 'gate_on_ratio1' and i == 4:
                gate = True
            if gate:
                t(b + 'attn_compressor_gate.weight', [N_EMBD, HD])
            t(b + 'indexer.attn_k.weight', [HD, IDX_D])
            t(b + 'indexer.k_norm.weight', [IDX_D])
        if i in idx_src:
            t(b + 'indexer.proj.weight', [N_EMBD, IDX_H])
            t(b + 'indexer.attn_q_b.weight', [Q_LORA, IDX_H * IDX_D])
        if i in eng:
            t(b + 'engram_q.weight', [N_EMBD, HC])
            t(b + 'engram_k.weight', [N_EMBD, HC])
            t(b + 'engram_wkv.weight', [(ENG_NGRAM - 1) * ENG_H * ENG_D, (HC + 1) * N_EMBD])
        if v == 'tid2eid' and i == 0:
            t(b + 'ffn_gate_tid2eid.weight', [N_USED, N_VOCAB], np.int32)
        t(b + 'ffn_gate_inp.weight', [N_EMBD, N_EXP])
        t(b + 'exp_probs_b.bias', [N_EXP])
        t(b + 'ffn_norm.weight', [N_EMBD])
        t(b + 'ffn_gate_exps.weight', [N_EMBD, N_FF, N_EXP])
        t(b + 'ffn_up_exps.weight', [N_EMBD, N_FF, N_EXP])
        t(b + 'ffn_down_exps.weight', [N_FF, N_EMBD, N_EXP])
        t(b + 'ffn_gate_shexp.weight', [N_EMBD, N_FF])
        t(b + 'ffn_up_shexp.weight', [N_EMBD, N_FF])
        t(b + 'ffn_down_shexp.weight', [N_FF, N_EMBD])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == '__main__':
    main()
