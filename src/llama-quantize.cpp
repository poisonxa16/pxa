#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-quantize.h"

#include "ggml.h"
#include "ggml-common.h"

#include "pxq-quants.h"
#include "pxq-cpu.h"

#include <thread>
#include <atomic>
#include <cctype>
#include <regex>
#include <mutex>
#include <fstream>
#include <filesystem>
#include <map>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <set>

//
// quantization
//

// TODO: replace with ggml API call
#define QK_K 256
#define QK_IQ1BN 64

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

static void ensure_output_directory(const std::string & filepath) {
    std::filesystem::path p(filepath);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
        if (ec) {
            fprintf(stderr, "Failed to create directory '%s': %s\n", p.parent_path().string().c_str(), ec.message().c_str());
            exit(EXIT_FAILURE);
        }
    }
}

struct quantize_state_internal {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv    = 0;
    int n_ffn_down        = 0;
    int n_ffn_gate        = 0;
    int n_ffn_up          = 0;
    int i_attention_wv    = 0;
    int i_ffn_down        = 0;
    int i_ffn_gate        = 0;
    int i_ffn_up          = 0;

    int n_k_quantized     = 0;
    int n_fallback        = 0;

    bool has_imatrix      = false;

    // used to figure out if a model shares tok_embd with the output weight
    bool has_output       = false;

    // true for the PXQ ftypes (PXQ1/PXQ2/PXQ3/PXQ4/PXQ4HQ/PXQ6/PXQ_UNIVERSAL): the output head
    // (output.weight, NOT token_embd) defaults to q8_0 — see pxa_pxq_head_type()
    bool is_pxq           = false;

    quantize_state_internal(const llama_model & model, const llama_model_quantize_params * params)
        : model(model)
        , params(params)
        {}
};

std::pair<ggml_type, int> interleaved_properties(ggml_type type) {
    static std::unordered_map<ggml_type, std::pair<ggml_type, int>> k_map = {
        { GGML_TYPE_Q4_0_4_4,    { GGML_TYPE_Q4_0, 4} },
        { GGML_TYPE_Q4_0_4_8,    { GGML_TYPE_Q4_0, 4} },
        { GGML_TYPE_Q4_0_8_8,    { GGML_TYPE_Q4_0, 8} },
    };
    if (auto it = k_map.find(type); it != k_map.end()) return it->second;
    return {type, 1};
}

static void llama_tensor_dequantize_internal(
    struct ggml_tensor * tensor, std::vector<no_init<float>> & output, std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
) {
    if (output.size() < nelements) {
        output.resize(nelements);
    }
    float * f32_output = (float *) output.data();

    ggml_type_traits_t qtype;
    if (ggml_is_quantized(tensor->type)) {
        qtype = ggml_internal_get_type_traits(tensor->type);
        if (qtype.to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16 &&
               tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    if (tensor->type == GGML_TYPE_I2_S) {
        // we need to dequantize the entire tensor for I2_S
        qtype.to_float(tensor->data, f32_output, nelements);
        return;
    }

    if (nthread < 2 || (ggml_is_quantized(tensor->type) && qtype.row_meta_size > 0)) {
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements);
        } else if (tensor->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((ggml_bf16_t *)tensor->data, f32_output, nelements);
        } else if (ggml_is_quantized(tensor->type)) {
            auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
            int nrows = ggml_nrows(tensor);
            auto qsrc = (const char *)tensor->data;
            auto num_rows = interleaved_properties(tensor->type).second;
            for (int row = 0; row < nrows; row += num_rows) {
                qtype.to_float(qsrc, f32_output, num_rows*tensor->ne[0]);
                qsrc += num_rows*row_size;
                f32_output += num_rows*tensor->ne[0];
            }
        } else {
            GGML_ABORT("fatal error"); // unreachable
        }
        return;
    }

    auto num_rows = interleaved_properties(tensor->type).second;
    if (num_rows > 1) {
        int nrows = ggml_nrows(tensor);
        auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
        auto qsrc = (const char *)tensor->data;
        for (int row = 0; row < nrows; row += num_rows) {
            qtype.to_float(qsrc, f32_output, num_rows*tensor->ne[0]);
            qsrc += num_rows*row_size;
            f32_output += num_rows*tensor->ne[0];
        }
        return;
    }

    size_t block_size;
    if (tensor->type == GGML_TYPE_F16 ||
        tensor->type == GGML_TYPE_BF16) {
        block_size = 1;
    } else {
        block_size = (size_t)ggml_blck_size(tensor->type);
    }

    size_t block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    size_t nblocks = nelements / block_size;
    size_t blocks_per_thread = nblocks / nthread;
    size_t spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    size_t in_buff_offs = 0;
    size_t out_buff_offs = 0;

    for (int tnum = 0; tnum < nthread; tnum++) {
        size_t thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        size_t thr_elems = thr_blocks * block_size; // number of elements for this thread
        size_t thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else if (typ == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((ggml_bf16_t *)inbuf, outbuf, nels);
            } else {
                qtype.to_float(inbuf, outbuf, nels);
            }
        };
        workers.emplace_back(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, f32_output + out_buff_offs, thr_elems);
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & w : workers) { w.join(); }
    workers.clear();
}

static ggml_type change_type_if_necessary(ggml_type new_type, int nx, int ny) {
    bool convert_incompatible_tensor = false;
    if (new_type == GGML_TYPE_Q2_K    || new_type == GGML_TYPE_Q3_K    || new_type == GGML_TYPE_Q4_K   ||
        new_type == GGML_TYPE_Q5_K    || new_type == GGML_TYPE_Q6_K    || new_type == GGML_TYPE_IQ4_XS ||
        new_type == GGML_TYPE_IQ2_XS  || new_type == GGML_TYPE_IQ2_XXS || new_type == GGML_TYPE_IQ2_S  ||
        new_type == GGML_TYPE_IQ3_XXS || new_type == GGML_TYPE_IQ1_S   || new_type == GGML_TYPE_IQ3_S  ||
        new_type == GGML_TYPE_IQ1_M) {
        if (nx % QK_K != 0) {
            LLAMA_LOG_WARN("\n\n%s : tensor cols %d x %d are not divisible by %d, required for %s", __func__, nx, ny, QK_K, ggml_type_name(new_type));
            convert_incompatible_tensor = true;
        }
    }
    if (convert_incompatible_tensor) {
        switch (new_type) {
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_IQ4_XS: new_type = GGML_TYPE_IQ4_NL; break;
            case GGML_TYPE_Q4_K:   new_type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_Q5_K:   new_type = GGML_TYPE_Q6_0;   break;
            case GGML_TYPE_Q6_K:   new_type = GGML_TYPE_Q8_0;   break;
            default: throw std::runtime_error("\nUnsupported tensor size encountered\n");
        }
        LLAMA_LOG_WARN(" - using fallback quantization %s\n", ggml_type_name(new_type));
    }
    return new_type;
}

// PXQ ftypes: default output-head type. q8_0 rides Pascal's fast DMMV path where the K-quant
// heads ride the slow scalar path, and the head runs EVERY token over the full (~151k) vocab:
// measured +3.0% P100 decode across all rounds — and q8_0 is also higher precision than the
// K-quant default, so it's speed AND quality. Env override PXA_PXQ_HEAD = q8_0 (default) |
// q6_k | f16; an unknown value warns and falls back to q8_0. Applies to output.weight only
// (NOT token_embd), and only when no explicit --output-tensor-type was given.
static ggml_type pxa_pxq_head_type() {
    static const ggml_type head = [](){
        ggml_type t = GGML_TYPE_Q8_0;
        const char * e = getenv("PXA_PXQ_HEAD");
        if (e && e[0]) {
            std::string s(e);
            for (auto & c : s) c = std::tolower(c);
            if      (s == "q8_0") t = GGML_TYPE_Q8_0;
            else if (s == "q6_k") t = GGML_TYPE_Q6_K;
            else if (s == "f16")  t = GGML_TYPE_F16;
            else {
                LLAMA_LOG_WARN("PXA_PXQ_HEAD: unknown value '%s' (want q8_0|q6_k|f16) — using q8_0\n", e);
                t = GGML_TYPE_Q8_0;
            }
        }
        if (t == GGML_TYPE_Q8_0) {
            LLAMA_LOG_INFO("PXQ head -> q8_0 (P100 +3.0%% decode measured; PXA_PXQ_HEAD overrides)\n");
        } else {
            LLAMA_LOG_INFO("PXQ head -> %s (PXA_PXQ_HEAD override; default q8_0 = P100 +3.0%% decode measured)\n", ggml_type_name(t));
        }
        return t;
    }();
    return head;
}

static ggml_type llama_tensor_get_type_inner(quantize_state_internal & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype) {
    const std::string name = ggml_get_name(tensor);

    // TODO: avoid hardcoded tensor names - use the TN_* constants
    const llm_arch arch = qs.model.arch;
    const auto       tn = LLM_TN(arch);

    auto use_more_bits = [](int i_layer, int n_layers) -> bool {
        return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
    };

    auto custom_type = GGML_TYPE_COUNT;
    if (qs.params->custom_quants) {
        using CustomQ = std::pair<std::string, ggml_type>;
        auto& q_rules = *static_cast<const std::vector<CustomQ>*>(qs.params->custom_quants);
        for (auto& rule : q_rules) {
            std::regex pattern(rule.first);
            if (std::regex_search(name, pattern)) {
                custom_type = rule.second;
                break;
            }
        }
    }

    //auto get_layer = [] (const char * name) {
    //    int il;
    //    if (sscanf(name, "blk.%d.", &il) == 1) return il;
    //    return -1;
    //};
    //int il = get_layer(tensor->name);
    //int nl = qs.model.hparams.n_layer;

    const int n_expert = std::max(1, (int)qs.model.hparams.n_expert);
    auto layer_info = [n_expert] (int i_layer, int n_layer, const char * name) {
        if (n_expert > 1) {
            // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not consecutive, but occasionally randomly
            // sprinkled in the model. Hence, simply dividing i_ffn_down by n_expert does not work
            // for getting the current layer as I initially thought, and we need to resort to parsing the
            // tensor name.
            if (sscanf(name, "blk.%d.", &i_layer) != 1) {
                throw std::runtime_error(format("Failed to determine layer for tensor %s", name));
            }
            if (i_layer < 0 || i_layer >= n_layer) {
                throw std::runtime_error(format("Bad layer %d for tensor %s. Must be in [0, %d)", i_layer, name, n_layer));
            }
        }
        return std::make_pair(i_layer, n_layer);
    };

    // for arches that share the same tensor between the token embeddings and the output, we quantize the token embeddings
    // with the quantization of the output tensor
    if (name == tn(LLM_TENSOR_OUTPUT, "weight") || (!qs.has_output && name == tn(LLM_TENSOR_TOKEN_EMBD, "weight"))) {
        if (qs.params->output_tensor_type < GGML_TYPE_COUNT) {
            new_type = qs.params->output_tensor_type;
        } else if (qs.is_pxq && name == tn(LLM_TENSOR_OUTPUT, "weight")) {
            // PXQ ftypes: q8_0 output head by default (P100 DMMV fast path; PXA_PXQ_HEAD overrides).
            // Deliberately output.weight only — a tied token_embd keeps the stock rules.
            new_type = pxa_pxq_head_type();
        } else {
            int nx = tensor->ne[0];
            if (arch == LLM_ARCH_FALCON || nx % QK_K != 0) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_S  || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M   ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_M   ) {
                new_type = !qs.has_output ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
            }
            else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_S || ftype == LLAMA_FTYPE_MOSTLY_IQ3_M || 
                      ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS ) && !qs.has_output) {
                new_type = GGML_TYPE_Q5_K;
            }
            else if (new_type != GGML_TYPE_Q8_0 && new_type != GGML_TYPE_Q6_K && new_type != GGML_TYPE_Q8_KV) {
                new_type = GGML_TYPE_Q6_K;
            }
        }
    } else if (name == "token_embd.weight") {
        if (qs.params->token_embedding_type < GGML_TYPE_COUNT) {
            new_type = qs.params->token_embedding_type;
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M  ) {
                new_type = GGML_TYPE_Q2_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ) {
                new_type = GGML_TYPE_IQ3_S;
            }
        }
    }
    else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S ||
               ftype == LLAMA_FTYPE_MOSTLY_IQ2_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M ) {
        bool is_iq2_m = ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ;
        if (name.find("attn_v.weight") != std::string::npos) {
            if      (qs.model.hparams.n_gqa() >= 4 || qs.model.hparams.n_expert >= 4) new_type = GGML_TYPE_Q4_K;
            else if (qs.model.hparams.n_gqa() >= 2 || qs.model.hparams.n_expert >= 2) new_type = GGML_TYPE_Q3_K;
            else new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || is_iq2_m ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            ++qs.i_attention_wv;
        }
        else if (qs.model.hparams.n_expert >= 8 && name.find("attn_k") != std::string::npos) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (qs.model.hparams.n_expert >= 8 && name.find("attn_q") != std::string::npos) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (name.find("attn_qkv.weight") != std::string::npos) {
            new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || is_iq2_m ? GGML_TYPE_IQ3_XXS : GGML_TYPE_Q2_K;
        }
        else if (name.find("ffn_down") != std::string::npos) {
            if (qs.i_ffn_down < qs.n_ffn_down/8) {
                new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || is_iq2_m ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            }
            ++qs.i_ffn_down;
        }
        else if (name.find("attn_output.weight") != std::string::npos) {
            if (qs.params->attn_output_type < GGML_TYPE_COUNT) new_type = qs.params->attn_output_type;
            else if (qs.model.hparams.n_expert >= 4) {
                new_type = GGML_TYPE_Q5_K;
            } else {
                if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) new_type = GGML_TYPE_Q2_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || is_iq2_m) new_type = GGML_TYPE_IQ3_S;
            }
        }
    } else if (name.find("attn_v.weight") != std::string::npos) {
        if      (qs.params->attn_v_type < GGML_TYPE_COUNT) new_type = qs.params->attn_v_type;
        else if (qs.model.hparams.n_expert >= 4) {
            // for the 4-8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (qs.model.type == MODEL_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : qs.model.hparams.n_gqa() >= 2 ? GGML_TYPE_Q3_K
                     : !qs.has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S) && qs.model.hparams.n_gqa() >= 2) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = qs.model.hparams.n_gqa() >= 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS ) && qs.model.hparams.n_gqa() >= 2) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_S) {
            if (qs.model.hparams.n_vocab >= 127999 && (qs.model.type == MODEL_8B || qs.model.type == MODEL_70B))
                new_type = GGML_TYPE_Q6_K;
        }
        else if (qs.model.hparams.n_gqa() >= 4) {
            if      (new_type == GGML_TYPE_Q2_K || new_type == GGML_TYPE_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
            else if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_IQ3_S) new_type = GGML_TYPE_Q4_K;
            else if (new_type == GGML_TYPE_Q4_K || new_type == GGML_TYPE_IQ4_XS) new_type = GGML_TYPE_Q5_K;
            else if (new_type == GGML_TYPE_IQ4_NL) new_type = GGML_TYPE_Q5_K;
            else if (new_type == GGML_TYPE_Q5_K) new_type = GGML_TYPE_Q6_K;
        }
        ++qs.i_attention_wv;
    } else if (name.find("attn_k") != std::string::npos) {
        if (qs.params->attn_k_type < GGML_TYPE_COUNT) new_type = qs.params->attn_k_type;
        else if (qs.model.hparams.n_expert >= 4) {
            // for the 4-8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS; // TODO: explore better strategies?
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ) {
            new_type = GGML_TYPE_IQ2_S; // TODO: explore better strategies?
        }
    } else if (name.find("attn_q") != std::string::npos) {
        if (qs.params->attn_q_type < GGML_TYPE_COUNT) new_type = qs.params->attn_q_type;
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ) {
            new_type = GGML_TYPE_IQ2_S;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_S) {
            if (qs.model.hparams.n_vocab >= 127999 && (qs.model.type == MODEL_8B || qs.model.type == MODEL_70B))
                new_type = GGML_TYPE_Q4_K;
        }
    } else if (name.find("ffn_down") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (qs.params->ffn_down_type < GGML_TYPE_COUNT) new_type = qs.params->ffn_down_type;
        else if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs.has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs.model.hparams.n_expert >= 4 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (arch == LLM_ARCH_FALCON) {
                new_type = i_layer < n_layer/16 ? GGML_TYPE_Q6_K :
                           use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else {
                if (use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
            }
        }
        else if (i_layer < n_layer/8 && !qs.has_imatrix &&
                (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS )) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0)
                && qs.has_imatrix && i_layer < n_layer/8) {
            // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
            // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
            // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }
        ++qs.i_ffn_down;
    } else if (name.find("attn_output.weight") != std::string::npos) {
        if (qs.params->attn_output_type < GGML_TYPE_COUNT) new_type = qs.params->attn_output_type;
        else if (arch != LLM_ARCH_FALCON) {
            if (qs.model.hparams.n_expert >= 4) {
                if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL  ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S   ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS ) {
                    new_type = GGML_TYPE_Q5_K; // should the IQ_K quants be applied here as the new type for the IQ_K ftypes ?
                    // also, this condition could be reproduced on attn_q, eventually with Q4_K instead of Q5_K.
                }
            } else {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   ) new_type = GGML_TYPE_Q3_K; // This list could be generalized and streamlined
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M ) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L ) new_type = GGML_TYPE_Q5_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  ) new_type = GGML_TYPE_Q4_K;
            }
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
        }
    }
    else if (name.find("attn_qkv.weight") != std::string::npos) {
        if (qs.params->attn_qkv_type < GGML_TYPE_COUNT) new_type = qs.params->attn_qkv_type;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = GGML_TYPE_Q4_K; // That logic could either be generalized, either be ditched?
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M ) new_type = GGML_TYPE_Q4_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) new_type = GGML_TYPE_Q6_K;
    }
    else if (name.find("ffn_gate") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (qs.params->ffn_gate_type < GGML_TYPE_COUNT) new_type = qs.params->ffn_gate_type;
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_gate;
    }
    else if (name.find("ffn_up") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_up, qs.n_ffn_up, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (qs.params->ffn_up_type < GGML_TYPE_COUNT) new_type = qs.params->ffn_up_type;
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_up;
    }

    if (custom_type < GGML_TYPE_COUNT) {
        new_type = custom_type;
        LLAMA_LOG_INFO("Using custom type %s for tensor %s\n", ggml_type_name(new_type), name.c_str());
    }

    auto working_type = change_type_if_necessary(new_type, tensor->ne[0], tensor->ne[1]);
    if (working_type != new_type) {
        ++qs.n_fallback;
        new_type = working_type;
    }

    if (name == "token_embd.weight") {
        auto working_type = interleaved_properties(new_type).first;
        if (working_type != new_type) {
            printf("\n============ Token embeddings cannot be quantized with row-interleaved quants\n");
            printf("---> Changed %s to %s\n", ggml_type_name(new_type), ggml_type_name(working_type));
            new_type = working_type;
        }
    }

    return new_type;
}

// ------------------------------------------------------------------------------------------
// ROW-GATHER GUARD.
//
// The PXQ codecs and MXFP4 lay a row out across a panel with shared per-panel state. That is
// what makes them fast in a GEMM, and it is exactly what makes a single row unreadable in
// isolation. A tensor that is only ever consumed by GET_ROWS is read ONE ROW AT A TIME, so
// encoding it with a panel codec produces a file that quantizes and loads cleanly and then
// gathers nonsense.
//
// The quantizer cannot see the graph, so the test is structural: a gather table is enormous
// along ne[1] (rows) because it is indexed by a hash or a token id. qwen4exp's
// per_layer_token_embd is 160 x 320,001,536 - 51.2B of the model's 180B parameters - and
// gemma3n carries the same tensor under the same name.
//
// On a match the panel codec is refused and the tensor falls back to a block-quantized type
// whose rows stand alone. This is a CORRECTNESS gate, not a preference: it is not overridable
// by --custom-q, because a file that gathers nonsense is not a tradeoff.
// ------------------------------------------------------------------------------------------
static bool pxa_is_row_gather_tensor(const ggml_tensor * tensor) {
    const std::string name = ggml_get_name(tensor);

    // named gather tables, independent of size
    if (name.find("per_layer_token_embd") != std::string::npos) {
        return true;
    }

    // structural: a row count this large only happens for a hash/id-indexed table
    return tensor->ne[1] >= 1000000;
}

// the six PXQ slab tiers (excludes MXFP4, which is a panel codec here but not a PXQ type)
static bool pxa_is_pxq_slab_type(ggml_type t) {
    switch (t) {
        case GGML_TYPE_PXQ1:
        case GGML_TYPE_PXQ2:
        case GGML_TYPE_PXQ3:
        case GGML_TYPE_PXQ4:
        case GGML_TYPE_PXQ4HQ:
        case GGML_TYPE_PXQ6:
            return true;
        default:
            return false;
    }
}

static bool pxa_is_panel_codec(ggml_type t) {
    switch (t) {
        case GGML_TYPE_PXQ1:
        case GGML_TYPE_PXQ2:
        case GGML_TYPE_PXQ3:
        case GGML_TYPE_PXQ4:
        case GGML_TYPE_PXQ4HQ:
        case GGML_TYPE_PXQ6:
        case GGML_TYPE_MXFP4:
            return true;
        default:
            return false;
    }
}

static ggml_type llama_tensor_get_type(quantize_state_internal & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype) {
    ggml_type chosen = llama_tensor_get_type_inner(qs, new_type, tensor, ftype);

    if (pxa_is_panel_codec(chosen) && pxa_is_row_gather_tensor(tensor)) {
        // Q4_K keeps whole rows independently decodable and is already the fallback the
        // stock rules use for oversized embedding tables.
        const ggml_type safe = GGML_TYPE_Q4_K;
        LLAMA_LOG_WARN("%s: %s is a row-gather table (%lld x %lld); %s is a panel codec and "
                       "cannot be read one row at a time - using %s instead\n",
                       __func__, ggml_get_name(tensor),
                       (long long) tensor->ne[0], (long long) tensor->ne[1],
                       ggml_type_name(chosen), ggml_type_name(safe));
        return safe;
    }

    return chosen;
}



static size_t llama_tensor_quantize_internal(enum ggml_type new_type, const float * f32_data, void * new_data, const int64_t chunk_size, int64_t nrows, int64_t n_per_row,
        const float * imatrix, const quantize_user_data * user_data, std::vector<std::thread> & workers, const int nthread) {
    if (nthread < 2) {
        // single-thread
        size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix, user_data);
        if (!ggml_validate_row_data(new_type, new_data, new_size)) {
            throw std::runtime_error("quantized data validation failed");
        }
        return new_size;
    }

    std::mutex mutex;
    int64_t counter = 0;
    size_t new_size = 0;
    bool valid = true;
    auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data, new_data, chunk_size,
            nrows, n_per_row, imatrix, user_data]() {
        const int64_t nrows_per_chunk = chunk_size / n_per_row;
        size_t local_size = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int64_t first_row = counter; counter += nrows_per_chunk;
            if (first_row >= nrows) {
                if (local_size > 0) {
                    new_size += local_size;
                }
                break;
            }
            lock.unlock();
            const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
            size_t this_size = ggml_quantize_chunk(new_type, f32_data, new_data, first_row * n_per_row, this_nrow, n_per_row, imatrix, user_data);
            local_size += this_size;

            // validate the quantized data
            const size_t row_size  = ggml_row_size(new_type, n_per_row);
            void * this_data = (char *) new_data + first_row * row_size;
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                std::unique_lock<std::mutex> lock(mutex);
                valid = false;
                break;
            }
        }
    };
    for (int it = 0; it < nthread - 1; ++it) {
        workers.emplace_back(compute);
    }
    compute();
    for (auto & w : workers) { w.join(); }
    workers.clear();
    if (!valid) {
        throw std::runtime_error("quantized data validation failed");
    }
    return new_size;
}


static void do_quantize(int nthread, const ggml_tensor * tensor, ggml_type new_type, const float * f32_data, char * new_data,
        const float * imatrix, std::vector<std::thread> & workers, size_t & new_size, int chunk_size_multiplier,
        const llama_model_quantize_params * params) {
    if (nthread > 1 && (tensor->ne[2] % nthread == 0 || tensor->ne[2] >= 2*nthread)) {
        std::mutex mutex;
        int counter = 0;
        bool valid = true;
        auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data, new_data, tensor, imatrix, user_data = params->user_data] () {
            int ne2 = tensor->ne[2];
            auto row_size = ggml_row_size(new_type, tensor->ne[0]);
            auto matrix_size = row_size * tensor->ne[1];
            size_t local_size = 0;
            while (true) {
                std::unique_lock<std::mutex> lock(mutex);
                int i02 = counter++;
                if (i02 >= ne2) {
                    if (local_size > 0) {
                        new_size += local_size;
                    }
                    break;
                }
                lock.unlock();
                auto this_imatrix = imatrix ? imatrix + i02 * tensor->ne[0] : nullptr;
                auto this_data = (char *)new_data + i02*matrix_size;
                auto this_size = ggml_quantize_chunk(new_type, f32_data + i02*tensor->ne[0]*tensor->ne[1], this_data,
                        0, tensor->ne[1], tensor->ne[0], this_imatrix, user_data);
                local_size += this_size;

                // validate the quantized data
                if (!ggml_validate_row_data(new_type, this_data, matrix_size)) {
                    lock.lock();
                    valid = false;
                    break;
                }
            }
        };
        for (int it = 0; it < nthread; ++it) workers.emplace_back(std::thread(compute));
        for (auto & w : workers) w.join();
        workers.clear();
        if (!valid) {
            throw std::runtime_error("quantized data validation failed");
        }
    } else {
        static const int64_t min_chunk_size = 32 * 512;
        const int64_t n_per_row = tensor->ne[0];
        const int64_t nrows     = tensor->ne[1];
        const int64_t chunk_size = (n_per_row >= min_chunk_size
                                 ? n_per_row : n_per_row * ((min_chunk_size + n_per_row - 1)/n_per_row)) * chunk_size_multiplier;

        const int64_t nelements_matrix = tensor->ne[0] * tensor->ne[1];
        const int64_t nchunk = (nelements_matrix + chunk_size - 1)/chunk_size;
        const int64_t nthread_use = nthread > 1 ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk)) : 1;

        // quantize each expert separately since they have different importance matrices
        new_size = 0;
        for (int64_t i03 = 0; i03 < tensor->ne[2]; ++i03) {
            const float * f32_data_03 = f32_data + i03 * nelements_matrix;
            void * new_data_03 = (char *)new_data + ggml_row_size(new_type, n_per_row) * i03 * nrows;
            const float * imatrix_03 = imatrix ? imatrix + i03 * n_per_row : nullptr;

            new_size += llama_tensor_quantize_internal(new_type, f32_data_03, new_data_03, chunk_size,
                    nrows, n_per_row, imatrix_03, params->user_data, workers, nthread_use);
        }
    }
}

// Dead-column imatrix guard, shared by every PXQ codec's `imx_for(e)`.
// Every PXQ codec already CONSUMES the imatrix — `err += (w ? w[i] : 1.0) * e*e` inside the
// anchor pick and the code/sub argmin, i.e. a diagonal-weighted SSE minimisation. The hazard
// is a column of zeros: a routed expert that never fired during imatrix collection makes
// EVERY candidate score exactly 0.0, so the fit stops minimising anything and silently
// degenerates into the deterministic tie-break. On a 256-expert MoE a handful of experts
// never firing over a few thousand chunks is normal and expected, so this is not a corner
// case — it is the default outcome for some fraction of the file. Fall back to an unweighted
// fit for such a column (also catches non-finite / negative weights from a damaged imatrix).
static std::atomic<int64_t> g_pxq_imx_dead_cols{0};

static bool pxq_imatrix_column_usable(const float * w, int64_t K) {
    if (!w) return false;
    double s = 0.0;
    for (int64_t i = 0; i < K; ++i) {
        if (!std::isfinite(w[i]) || w[i] < 0.0f) { g_pxq_imx_dead_cols.fetch_add(1); return false; }
        s += (double) w[i];
    }
    if (s > 0.0) return true;
    g_pxq_imx_dead_cols.fetch_add(1);
    return false;
}

//
#include "pxq6-quantize.inc.cpp"   // PXQ6/PXQ6HQ native quantizer (E16-row scales; 2026-07-17)
#include "pxq2-quantize.inc.cpp"   // PXQ2 native quantizer (LM4 x E16-row; Q-G1/Q-G2 2026-07-17)
#include "pxq3-quantize.inc.cpp"   // PXQ3 native quantizer (LM8 bit-plane x E16-row)
#include "pxq6r-quantize.inc.cpp"  // PXQ6 native quantizer (LM32 5-bit x E16-row; spec PXQ6-REAL v1.0-FINAL 2026-07-21)
#include "pxq1-quantize.inc.cpp"   // PXQ1 native quantizer (1-bit sign x E16-row; the PXQ-UNIVERSAL sub-2-bit tier)

// PXQ slab-tier eligibility (shared by every PXQ tier): expert tensors (_exps.weight),
// rows % 64 == 0 (panel height), K % 32 == 0 (slab width).
// (The id-250 MXFP4-repack legacy production path — pxq4_permute_from_mxfp4 — was removed
//  2026-07-21 with the retirement of GGML_TYPE_PXQ4_LEGACY.)
static bool pxa_name_ends(const std::string & name, const char * suf) {
    const size_t n = strlen(suf);
    return name.size() > n && name.compare(name.size() - n, n, suf) == 0;
}

// Whole-name match: the tensor IS <suf> (a top-level tensor such as token_embd.weight /
// output.weight) or is the "blk.N." -qualified form ".<suf>".
// pxa_name_ends() alone is wrong for both ends of this: it is a STRICT suffix test
// (name.size() > n), so it silently misses the top-level tensors entirely — which is why
// PXA_PXQ_NATIVE=embd has never actually promoted token_embd — and it matches on any
// character boundary, so "output.weight" also matches "blk.0.attn_output.weight".
static bool pxa_name_is(const std::string & name, const char * suf) {
    if (name == suf) {
        return true;
    }
    const std::string dotted = std::string(".") + suf;
    return pxa_name_ends(name, dotted.c_str());
}

// PXA_PXQ_NATIVE: opt-in widening of PXQ eligibility beyond the routed expert stacks,
// for the "100% native PXQ" work (no inherited MXFP4). Comma-separated class list:
//   shexp  shared-expert FFN   (ffn_{up,gate,down}_shexp.weight)
//   attn   attention projections (attn_q/attn_qkv/attn_output/attn_gate.weight)
//   embd   token_embd.weight
//   ssm    Gated-DeltaNet state tensors (ssm_alpha/beta/out.weight)  [RISKY: recurrent]
//   all    every class above
// Unset (default) == historical behaviour: routed experts only, everything else MXFP4.
// The geometry gate below still applies to every class -- a tensor that fails it is
// demoted to MXFP4 by the caller, so a bad list can never produce an unloadable file.
static unsigned pxa_pxq_native_mask() {
    static const unsigned m = [] {
        const char * e = getenv("PXA_PXQ_NATIVE");
        if (!e || !*e) return 0u;
        const std::string s(e);
        unsigned v = 0;
        auto has = [&](const char * k) { return s.find(k) != std::string::npos; };
        if (has("all"))   return 0xFu;
        if (has("shexp")) v |= 1u;
        if (has("attn"))  v |= 2u;
        if (has("embd"))  v |= 4u;
        if (has("ssm"))   v |= 8u;
        return v;
    }();
    return m;
}

static bool pxq4_tensor_geometry_ok(const ggml_tensor * t) {
    // geometry is non-negotiable: the slab codec needs 64-row panels and 32-wide blocks
    return ggml_n_dims(t) >= 2 && t->ne[1] % 64 == 0 && t->ne[0] % 32 == 0;
}

// the pre-BACKBONE_REV2 eligibility classes: routed experts always, plus whatever
// PXA_PXQ_NATIVE opts in. Kept separate because the "an untouched MXFP4 default resolves
// to the whole-file tier" upgrade in the write loop must fire ONLY for these -- the rev-2
// backbone classes carry an explicit resolved type and must never be re-derived from MXFP4
// (that would silently override an explicit --attn-q-type mxfp4 / custom-q rule).
static bool pxq4_legacy_native_class(const std::string & name) {
    if (pxa_name_ends(name, "_exps.weight")) {
        return true;
    }
    const unsigned m = pxa_pxq_native_mask();
    if ((m & 1u) && pxa_name_ends(name, "_shexp.weight")) {
        return true;
    }
    if ((m & 2u) && (pxa_name_is(name, "attn_q.weight")   || pxa_name_is(name, "attn_qkv.weight") ||
                     pxa_name_is(name, "attn_output.weight") || pxa_name_is(name, "attn_gate.weight") ||
                     // DeepSeek-V4 splits the output projection into a LoRA down/up pair
                     // (attn_output_a {n_head*n_embd_head/o_groups, o_lora_rank*o_groups},
                     //  attn_output_b {o_groups*o_lora_rank, n_embd}). Same role as attn_output,
                     // and neither name matches pxa_name_is(..., "attn_output.weight"), so without
                     // this PXA_PXQ_NATIVE=attn was a silent no-op on DS4.
                     pxa_name_is(name, "attn_output_a.weight") || pxa_name_is(name, "attn_output_b.weight"))) {
        return true;
    }
    // NOTE: this used pxa_name_ends(), whose strict-suffix test can never match the top-level
    // "token_embd.weight" — so PXA_PXQ_NATIVE=embd was a silent no-op. Fixed 2026-07-26; a
    // measured "embd is a null" from before that date is null BY CONSTRUCTION, not evidence.
    if ((m & 4u) && pxa_name_is(name, "token_embd.weight")) {
        return true;
    }
    if ((m & 8u) && (pxa_name_is(name, "ssm_alpha.weight") || pxa_name_is(name, "ssm_beta.weight") ||
                     pxa_name_is(name, "ssm_out.weight"))) {
        return true;
    }
    return false;
}

// ============================================================================
// PXQ backbone allocation table — BACKBONE_REV 2 (PXA_PXQ_BACKBONE), 2026-07-26
// ============================================================================
// Until rev 2, every PXQ tier quantized its routed expert stacks natively and then flattened
// EVERYTHING ELSE to MXFP4 ("MXFP4 rules for the rest"). Measured on Laguna-S-2.1 against the
// same bf16 source, relative RMS error vs the stock Q4_K_M recipe on the SAME tensors:
//   attn_output 0.1157 vs 0.0362 (Q5_K)   -> 3.2x
//   attn_gate   0.1609 vs 0.1006          -> 1.6x   (and the file was 8% BIGGER)
//   attn_q / token_embd / ffn_*_shexp     -> 1.6x each
// MXFP4 is a 3.54-effective-bit codec occupying 4.25 bpw with no salient-weight protection, so
// the backbone was the weakest part of every tier we shipped. Laguna made it visible because its
// attn_gate is PER-HEAD (72 softplus scalars/layer, high kurtosis, 0.03% of the file) that
// multiplies whole attention heads: a 16% error there is a 31-38% functional error per head.
//
// Rev 2 replaces the flatten with a per-class table. It is RECIPE-level only: no on-disk slab
// layout changes, so every previously shipped PXQ file still loads unchanged.
//
//   class                                PXQ2      PXQ3      PXQ4/PXQ4HQ   PXQ6
//   attn_q / attn_qkv                    PXQ4      PXQ4HQ    PXQ6          PXQ6
//   attn_output                          PXQ4HQ    PXQ4HQ    PXQ6          PXQ6
//   attn_k / attn_v                      Q8_0      Q8_0      Q8_0          Q8_0   (already, keep)
//   attn_gate  per-HEAD    (ne[1]<=256)  F16       F16       F16           F16    (the Laguna killer)
//   attn_gate  per-CHANNEL (ne[1]> 256)  PXQ4      PXQ4HQ    PXQ6          PXQ6
//   ffn_{up,gate,down}_shexp             PXQ4      PXQ4HQ    PXQ6          PXQ6
//   dense ffn_{up,gate,down}             PXQ4      PXQ4HQ    PXQ6          PXQ6
//   token_embd                           Q6_K      Q6_K      Q6_K          Q6_K   (row gather, not a GEMM)
//   output (head)                        Q8_0 via pxa_pxq_head_type() — unchanged
//   ssm_out                              backbone tier (2026-08-25; was the last flat-MXFP4
//                                        landing in a passing PXQ run — see the class list)
//   ssm_alpha / ssm_beta / nextn.*       Q8_0 (below the 64-row PXQ panel / MTP companions)
//   router / norms                       unchanged (legacy pipeline)
//   anything failing the slab geometry   Q8_0 fallback (never a silent MXFP4 demotion)
//
// Cost on Laguna-S-2.1: +0.45 GiB on a 62.0 GiB PXQ4 file (+0.73%).
//
// PXA_PXQ_BACKBONE (comma-separated tokens, evaluated once):
//   unset / "1" / "v2"   rev-2 table on PXQ2/PXQ3/PXQ4/PXQ4HQ/PXQ6            [DEFAULT]
//   "legacy" / "0"       exact pre-rev-2 behaviour (byte-reproduces old recipes)
//   "hq"                 rev-2 with the PXQ4HQ backbone substituted for PXQ6 on the 4-/5-bit
//                        tiers — the pre-registered fallback: ~82% of the modelled gain at
//                        +0.26 bpw instead of +1.02, and PXQ4HQ (unlike PXQ6) has a CPU
//                        panel-dequant, so the file stays partial-offload capable.
//   "universal"          additionally apply the table to PXQ_UNIVERSAL and PXQ1 (off by
//                        default: a PXQU tier map is user-authored per-tensor, and PXQ1 is a
//                        closed tier — neither has measured backbone evidence).
enum pxa_pxq_bb_mode {
    PXA_BB_LEGACY = 0,
    PXA_BB_REV2   = 1,
};

struct pxa_pxq_bb_cfg {
    int  mode = PXA_BB_REV2;
    bool hq   = false;   // PXQ4HQ backbone instead of PXQ6 on the 4-/5-bit tiers
    bool univ = false;   // also cover PXQ_UNIVERSAL / PXQ1
    bool lite = false;   // only the promotions that cost nothing at decode
    bool core = false;   // GEMM backbone at the byte-parity PXQ4 core tier (MMVQ-eligible)
    bool pxq6 = false;   // restore the pre-2026-07-31 PXQ6 backbone on the 4-/5-bit tiers
};

static const pxa_pxq_bb_cfg & pxa_pxq_backbone_cfg() {
    static const pxa_pxq_bb_cfg cfg = [] {
        pxa_pxq_bb_cfg c;
        const char * e = getenv("PXA_PXQ_BACKBONE");
        std::string s = e ? e : "";
        for (auto & ch : s) ch = std::tolower(ch);
        auto has = [&](const char * k) { return s.find(k) != std::string::npos; };
        if (!s.empty()) {
            if (has("legacy") || has("off") || s == "0") c.mode = PXA_BB_LEGACY;
            if (has("hq"))                               c.hq   = true;
            if (has("lite"))                             c.lite = true;
            if (has("core"))                             c.core = true;
            if (has("pxq6"))                             c.pxq6 = true;
            if (has("universal") || has("all"))          c.univ = true;
        }
        if (c.mode == PXA_BB_LEGACY) {
            LLAMA_LOG_INFO("PXQ backbone rules: LEGACY (flat MXFP4 for every non-expert tensor) "
                           "— PXA_PXQ_BACKBONE=legacy\n");
        } else {
            LLAMA_LOG_INFO("PXQ backbone rules: REV 2 (per-class promotion%s%s) — "
                           "4-/5-bit tiers take a %s GEMM backbone; "
                           "PXA_PXQ_BACKBONE=pxq6 restores the pre-2026-07-31 PXQ6 backbone, "
                           "=legacy the old flat-MXFP4 one\n",
                           c.lite ? ", LITE (decode-free classes only)" : "",
                           c.univ ? ", incl. PXQ_UNIVERSAL/PXQ1" : "",
                           c.lite ? "MXFP4 (lite)"
                                  : (c.hq   ? "PXQ4HQ"
                                  : (c.pxq6 ? "PXQ6"
                                            : "PXQ4 (byte-parity, MMVQ-eligible)")));
        }
        return c;
    }();
    return cfg;
}

// tier context for the table above. PXA_TIER_NONE == "this ftype has no backbone table".
enum pxa_pxq_tier {
    PXA_TIER_NONE = 0,
    PXA_TIER_PXQ1,
    PXA_TIER_PXQ2,
    PXA_TIER_PXQ3,
    PXA_TIER_PXQ4,
    PXA_TIER_PXQ4HQ,
    PXA_TIER_PXQ6,
    PXA_TIER_PXQU,
};

// the classes the table maps onto a NATIVE PXQ slab type (so the write-loop dispatch must
// treat them as eligible). token_embd (Q6_K), attn_k/v (Q8_0) and the per-head gate (F16)
// are deliberately NOT here — they are stock types with full CPU codecs.
static bool pxa_pxq_backbone_native_class(const std::string & name, const ggml_tensor * t) {
    if (pxa_name_ends(name, "_exps.weight")) return false;   // owned by the expert path
    if (name.find(".nextn.") != std::string::npos) return false;   // MTP companion: legacy in v1
    if (pxa_name_is(name, "attn_q.weight") || pxa_name_is(name, "attn_qkv.weight") ||
        pxa_name_is(name, "attn_output.weight")) {
        return true;
    }
    // MLA (DeepSeek / GLM-family) splits the Q projection into a LoRA down/up pair. Same role,
    // same error class as attn_q. The compressed K/V path (attn_kv_a_mqa / attn_kv_b / attn_k_b)
    // is deliberately NOT here: the stock rules already give it q8_0, which is where the table
    // wants K/V anyway. ⚠ Reasoned from the mechanism, not measured on an MLA model.
    if (pxa_name_is(name, "attn_q_a.weight") || pxa_name_is(name, "attn_q_b.weight")) {
        return true;
    }
    // DeepSeek-V4 splits the OUTPUT projection into a LoRA down/up pair
    // (attn_output_a {n_embd, o_groups*o_lora_rank} + attn_output_b). Same role and
    // same error class as attn_output.weight -- which this table's own notes record
    // as the worst-measured class (3.2x) -- and pxa_name_is("attn_output.weight")
    // does NOT match either of them.
    if (pxa_name_is(name, "attn_output_a.weight") || pxa_name_is(name, "attn_output_b.weight")) {
        return true;
    }
    if (pxa_name_is(name, "attn_gate.weight")) {
        return t->ne[1] > 256;                               // per-CHANNEL only; per-head -> F16
    }
    if (pxa_name_is(name, "ffn_up_shexp.weight") || pxa_name_is(name, "ffn_gate_shexp.weight") ||
        pxa_name_is(name, "ffn_down_shexp.weight")) {
        return true;
    }
    // dense (non-MoE, non-shared) FFN — suffix-disjoint from _exps.weight / _shexp.weight
    if (pxa_name_is(name, "ffn_up.weight") || pxa_name_is(name, "ffn_gate.weight") ||
        pxa_name_is(name, "ffn_down.weight")) {
        return true;
    }
    // DeltaNet output projection. Claimed 2026-08-25 — it was the LAST class still landing on
    // flat MXFP4 in a passing PXQ4 run (30 tensors / 127.50 MiB / 0.7% of a 35B build, every
    // one of them blk.N.ssm_out.weight at {4096, 2048}), which is exactly the landing the
    // zero-MXFP4 rule exists to remove. It is not a codec limitation: {4096, 2048} clears the
    // slab geometry with room to spare, the PXQ decode path already covers it (PXA_PXQ4_2D
    // lists ssm_out among the 2D PXQ classes and has defaulted ON since BACKBONE_REV 2), and
    // the SHIPPED 35B recipe already builds it as PXQ4 by hand via
    // --custom-q '(ssm_out\.weight)=pxq4' — so the default recipe was the odd one out, not the
    // ship recipe. At the PXQ4 backbone this is byte-parity with the MXFP4 it replaces
    // (4.2526 vs 4.2500 bpw). ⚠ The direction is reasoned + byte-neutral, NOT measured on a
    // paired perplexity: no ssm_out fidelity arm exists in this tree. PXA_PXQ_BACKBONE=lite
    // (or =legacy) still leaves it on MXFP4, and an explicit --custom-q rule still wins.
    if (pxa_name_is(name, "ssm_out.weight")) {
        return true;
    }
    return false;
}

// Mirrors the --custom-q / --pxq-universal regex scan inside llama_tensor_get_type() so the
// backbone table can stand down for any tensor a user rule already claims. Deliberately a
// second scan rather than a plumbed-through flag: llama_tensor_get_type() is on the public
// path and its signature is not ours to churn. Called only for tensors the table wants to
// move (a few dozen), so the duplicated regex work is negligible.
static ggml_type pxa_custom_rule_type(const llama_model_quantize_params * params, const std::string & name) {
    if (!params || !params->custom_quants) {
        return GGML_TYPE_COUNT;
    }
    using CustomQ = std::pair<std::string, ggml_type>;
    const auto & rules = *static_cast<const std::vector<CustomQ>*>(params->custom_quants);
    for (const auto & rule : rules) {
        std::regex pattern(rule.first);
        if (std::regex_search(name, pattern)) {
            return rule.second;   // first match wins, same as llama_tensor_get_type()
        }
    }
    return GGML_TYPE_COUNT;
}

static bool pxa_custom_rule_matches(const llama_model_quantize_params * params, const std::string & name) {
    return pxa_custom_rule_type(params, name) < GGML_TYPE_COUNT;
}

// LOUD-DEMOTE counter: explicit --custom-q PXQ targets that geometry forced off their
// requested type in this run (printed in the end-of-run summary; see the write loop).
static int g_pxa_customq_demoted = 0;

// The resolver. Returns GGML_TYPE_COUNT for "not mine — leave to the legacy pipeline".
static ggml_type pxa_pxq_backbone_type(const std::string & name, const ggml_tensor * t, pxa_pxq_tier tier,
                                       bool model_has_experts) {
    const pxa_pxq_bb_cfg & cfg = pxa_pxq_backbone_cfg();
    if (cfg.mode == PXA_BB_LEGACY || tier == PXA_TIER_NONE) {
        return GGML_TYPE_COUNT;
    }
    if ((tier == PXA_TIER_PXQU || tier == PXA_TIER_PXQ1) && !cfg.univ) {
        return GGML_TYPE_COUNT;
    }
    if (pxa_name_ends(name, "_exps.weight"))            return GGML_TYPE_COUNT;   // expert path owns it
    if (pxa_name_is(name, "output.weight"))             return GGML_TYPE_COUNT;   // pxa_pxq_head_type()
    // MTP companion (eh_proj etc., non-expert): draft-path only, but a contaminated draft costs
    // acceptance rate, and these are a handful of small tensors. Q8_0 instead of the old flat
    // MXFP4 (zero-MXFP4 rule, 2026-07-28). The nextn _exps stacks are caught above.
    if (name.find(".nextn.") != std::string::npos)      return GGML_TYPE_Q8_0;
    // DeltaNet per-head decay/gate projections: 32 rows < the 64-row PXQ panel — geometrically
    // impossible for the slab codecs, and their error acts multiplicatively on the recurrent
    // state. Q8_0 (+0.011% file size on the 35B) instead of the old flat-MXFP4 landing
    // (zero-MXFP4 rule, 2026-07-28).
    if (pxa_name_is(name, "ssm_alpha.weight") || pxa_name_is(name, "ssm_beta.weight")) {
        return GGML_TYPE_Q8_0;
    }
    // ssm_out (DeltaNet output projection) used to return GGML_TYPE_COUNT here, i.e. "leave it
    // to the legacy pipeline" — and the legacy landing is flat MXFP4. Because the tensor is
    // geometry-ELIGIBLE, the "geometry failure -> q8_0, never a silent MXFP4 demotion" rule
    // below never covered it, so it was the one class that kept violating the zero-MXFP4 rule
    // in an otherwise-passing PXQ run. It is now a backbone class like any other; see the note
    // in pxa_pxq_backbone_native_class().

    // token embeddings: a row GATHER, never a GEMM, so the P100 "k-quants are slow" rule does
    // not apply. Q6_K kills 0.121 relative RMS + the MXFP4 gain bias for ~+0.08 GiB on Laguna.
    if (pxa_name_is(name, "token_embd.weight")) {
        return t->ne[0] % QK_K == 0 ? GGML_TYPE_Q6_K : GGML_TYPE_Q8_0;
    }
    // K and V projections default to q8_0 (parity with our shipped files and Q4_K_M).
    // attn_v_b is MLA's V projection and was inheriting flat MXFP4; same role, same rule.
    // PXA_PXQ_KV (documented in docs/LEVERS.md but never landed in source until 2026-07-28 —
    // the docs row described three silent gates; all three clear here because the resolver
    // returns an EXPLICIT type and the write loop's eligibility is geometry-only now):
    // overrides the pin. Accepts q8_0|pxq4|pxq4hq|pxq6|mxfp4. NOTE the q8_0 "parity" premise is
    // false against a flat-MXFP4 legacy file (its K/V are MXFP4) — `mxfp4` restores true byte
    // parity for A/B work; the pxq tiers are the native option.
    if (pxa_name_is(name, "attn_k.weight") || pxa_name_is(name, "attn_v.weight") ||
        pxa_name_is(name, "attn_v_b.weight")) {
        static const ggml_type kv = [] {
            const char * e = getenv("PXA_PXQ_KV");
            if (!e || !*e) return GGML_TYPE_Q8_0;
            std::string s(e);
            for (auto & ch : s) ch = std::tolower(ch);
            if (s == "q8_0")   return GGML_TYPE_Q8_0;
            if (s == "pxq4")   return GGML_TYPE_PXQ4;
            if (s == "pxq4hq") return GGML_TYPE_PXQ4HQ;
            if (s == "pxq6")   return GGML_TYPE_PXQ6;
            if (s == "mxfp4")  return GGML_TYPE_MXFP4;
            LLAMA_LOG_WARN("PXA_PXQ_KV: unknown type '%s' — keeping q8_0\n", s.c_str());
            return GGML_TYPE_Q8_0;
        }();
        return kv;
    }
    // THE Laguna killer: a per-HEAD attention gate is a handful of softplus scalars per layer
    // (Laguna: 72 rows, 0.03% of the file) whose error multiplies an ENTIRE head. Never
    // quantize it. It also fails the 64-row panel gate, so there is no native option anyway.
    if (pxa_name_is(name, "attn_gate.weight") && t->ne[1] <= 256) {
        return GGML_TYPE_F16;
    }
    // ---------------- DeepSeek-V4 classes the stock rules do not know ----------------
    // Every branch here returns an EXPLICIT type. Without them these tensors fall
    // through to flat MXFP4, which violates this tree's zero-MXFP4 rule and, for the
    // indexer, quantizes the thing that CHOOSES which tokens are attended to.
    //
    // attn_kv is DS4's K and V fused into one projection (1 KV head of 512). The
    // legacy rule only catches it because "attn_kv" happens to contain "attn_k";
    // pin it explicitly so it cannot drift.
    if (pxa_name_is(name, "attn_kv.weight")) {
        return GGML_TYPE_Q8_0;
    }
    // The APE tables are absolute-position embeddings, not GEMM weights, and they
    // are tiny (ratio x coff*head_dim per layer; ~11 MB across all 43 layers).
    // Keep them F32 -- the converter writes F32, so this resolves to a verbatim copy.
    if (pxa_name_ends(name, "attn_compressor_ape.weight") ||
        pxa_name_ends(name, "indexer_compressor_ape.weight")) {
        return GGML_TYPE_F32;
    }
    // Per-layer KV compressors + the lightning indexer. The indexer's output is a
    // top-k SELECTION, so its error is not a small numeric perturbation -- it is the
    // wrong tokens attended to, with no loud signal. q8_0; they are a small share of
    // the file (only compress_ratios[il] != 0 layers carry them).
    if (pxa_name_ends(name, "attn_compressor_kv.weight")   || pxa_name_ends(name, "attn_compressor_gate.weight") ||
        pxa_name_ends(name, "attn_compressor_norm.weight") ||
        name.find(".indexer.") != std::string::npos        || name.find("indexer_compressor_") != std::string::npos) {
        return GGML_TYPE_Q8_0;
    }
    // Hyper-connection mixers: {hc_dim, 24}, {24}, {3} per layer plus the {hc_dim, 4}
    // head. 24 and 4 rows are far below the 64-row PXQ panel, and each one rescales an
    // ENTIRE residual stream -- the same error class as Laguna's per-head attn_gate.
    if (name.find("hc_attn_") != std::string::npos || name.find("hc_ffn_") != std::string::npos ||
        name.find("output_hc_") != std::string::npos) {
        return GGML_TYPE_F16;
    }
    // The hash routing table is not quantized at all (see the skip in the write loop).
    if (pxa_name_ends(name, "ffn_gate_tid2eid.weight")) {
        return GGML_TYPE_COUNT;
    }

    if (!pxa_pxq_backbone_native_class(name, t)) {
        return GGML_TYPE_COUNT;
    }
    // LITE: stop here. Everything above this line (per-head gate -> f16, token_embd -> q6_k,
    // attn_k/v -> q8_0, geometry-fail -> q8_0) is free at decode — the gate is ~0.02% of the
    // file and token_embd is a row GATHER, not a GEMM. Everything below promotes a DENSE GEMM
    // weight onto the PXQ 2D decode path, and that path is measurably slower per byte than
    // MXFP4's mmvq/DMMV on Pascal (Laguna-XS, 2xP100, indicative n=4: legacy 71.6 t/s,
    // PXQ4HQ backbone 51.6, PXQ6 backbone 45.2). In a 256-expert MoE the always-resident
    // backbone is roughly HALF the per-token decode traffic — only ~8/256 of the expert bytes
    // are touched — so that per-byte gap shows up almost undiluted at the token level.
    // LITE keeps the class that actually corrupted Laguna (the per-head attn_gate, 0.1609
    // relative RMS multiplying a whole head) while leaving the GEMM backbone on MXFP4 until
    // the 2D mmv is competitive. Not the default: it is a speed-vs-fidelity offer, and the
    // full table is the fidelity answer.
    if (cfg.lite) {
        return GGML_TYPE_COUNT;
    }
    // GEMM backbone: one notch above the expert tier (the ~1.4-bit attention-over-experts gap
    // implied by the measured per-parameter sensitivity ratio). Geometry failures take q8_0 —
    // NEVER a silent MXFP4 demotion, which is the error class this whole revision removes.
    if (!pxq4_tensor_geometry_ok(t)) {
        return GGML_TYPE_Q8_0;
    }
    // CORE (2026-07-28, the sm_70 dense-decode recipe): the whole GEMM backbone at the
    // byte-parity PXQ4 core tier — 4.2526 bpw vs MXFP4's 4.2500, MMVQ-eligible on every class.
    // Previously only expressible via --custom-q (the PXQ4core arm in the sm_70 campaign).
    //
    // CORE applies to EVERY tier: one uniform PXQ4 GEMM backbone across the whole ladder.
    // Being MMVQ-eligible is not the same as being equally fast on it -- the sm_70 campaign
    // separated the two eligible types on the same path (PXQ4core 33.861 t/s vs PXQ4HQ
    // 30.969), so core is ~9% on PXQ3's all-PXQ4HQ backbone and on PXQ2's attn_output, and
    // it is SMALLER besides (4.2526 bpw vs 4.52). The trade it makes is fidelity on
    // attn_output, the worst-measured class -- deliberate, and the reason core is a token
    // rather than the default for those tiers.
    const bool core_tier = tier == PXA_TIER_PXQ4 || tier == PXA_TIER_PXQ4HQ || tier == PXA_TIER_PXQ6;

    // DENSE MODELS: the named tier governs the backbone.
    //
    // The promotion below is defined as "one notch above the EXPERT tier". A model with no
    // routed experts has no expert tier -- the backbone is the entire model -- so promoting
    // (or, since 2026-07-31, demoting) it relative to nothing produces a file whose contents
    // do not match its name. Measured on Qwen3-0.6B: the PXQ6 tier emitted 140 tensors /
    // 193.92 MiB of PXQ4 and ZERO bytes of pxq6, identical to the PXQ4 arm, and the
    // composition assertion refused to write it -- i.e. PXQ6 could not quantize a dense model
    // at all. Default only: an explicit PXA_PXQ_BACKBONE=core/hq/pxq6 still wins below, and
    // the assertion remains the backstop if that override mislabels the output.
    if (!model_has_experts && core_tier && !cfg.core && !cfg.hq && !cfg.pxq6) {
        return tier == PXA_TIER_PXQ6   ? GGML_TYPE_PXQ6
             : tier == PXA_TIER_PXQ4HQ ? GGML_TYPE_PXQ4HQ
                                       : GGML_TYPE_PXQ4;
    }
    // Same rule for the sub-4-bit tiers on dense models (2026-08-10, first user: muse-glimmer
    // 30B). Without this, tier PXQ3 on a dense model routed the whole GEMM backbone to the
    // MoE table's PXQ4HQ promotion -- a 4.52 bpw file named PXQ3 with ZERO bytes of pxq3,
    // which the composition assertion then refuses to write (same failure the Qwen3-0.6B
    // note above records for PXQ6). The named tier governs; attn_k/v (q8_0), token_embd
    // (q6_k), the per-head gate (f16) and the output head keep their pins above. An explicit
    // PXA_PXQ_BACKBONE=core/hq/pxq6 still wins below.
    if (!model_has_experts && (tier == PXA_TIER_PXQ2 || tier == PXA_TIER_PXQ3)
        && !cfg.core && !cfg.hq && !cfg.pxq6) {
        return tier == PXA_TIER_PXQ2 ? GGML_TYPE_PXQ2 : GGML_TYPE_PXQ3;
    }

    if (cfg.core) {
        return GGML_TYPE_PXQ4;
    }
    // Backbone for the 4-/5-bit tiers. PXQ4 is the DEFAULT as of 2026-07-31 (was PXQ6).
    //
    // PXQ6 is not MMVQ-registered -- pxa_pxq_mmvq_type() admits PXQ4 and PXQ4HQ only -- so a
    // PXQ6 backbone drops the dense and shared-expert FFN off the fused single-kernel decode
    // path onto the per-operand divert. Measured against the PXQ6 backbone it replaces,
    // n=8/arm interleaved, fresh server per arm, one binary:
    //     sm_60 (P100)  +8-10% decode        sm_70 (V100)  +30-35% decode
    // and the file is ~1% smaller: PXQ4 is 4.2526 bpw at K=6144 vs PXQ6's 5.2526, i.e. byte
    // parity with MXFP4's 4.2500 rather than +1.00.
    //
    // Against an MXFP4 backbone it is a wash on Pascal (-2% decode / +2% prefill) and parity
    // on Volta PROVIDED MMVQ IS ARMED. See pxa_pxq_mmvq_auto_default(): arming it at DEFAULT
    // level is what makes the sm_70 number reachable without PXA_ENHANCE.
    //
    // PXA_PXQ_BACKBONE=pxq6 restores the pre-2026-07-31 PXQ6 backbone; =hq gives PXQ4HQ.
    const ggml_type hi = cfg.hq   ? GGML_TYPE_PXQ4HQ
                       : cfg.pxq6 ? GGML_TYPE_PXQ6
                                  : GGML_TYPE_PXQ4;
    switch (tier) {
        case PXA_TIER_PXQ1:                                       // only via PXA_PXQ_BACKBONE=universal
        case PXA_TIER_PXQ2:
            // attn_output is the worst-measured class (3.2x) — give it the HQ 4-bit tier.
            // DeepSeek-V4's o-LoRA pair is the same class (see above).
            return (pxa_name_is(name, "attn_output.weight")   ||
                    pxa_name_is(name, "attn_output_a.weight") ||
                    pxa_name_is(name, "attn_output_b.weight")) ? GGML_TYPE_PXQ4HQ : GGML_TYPE_PXQ4;
        case PXA_TIER_PXQ3:
        case PXA_TIER_PXQU:
            return GGML_TYPE_PXQ4HQ;
        case PXA_TIER_PXQ4:
        case PXA_TIER_PXQ4HQ:
        case PXA_TIER_PXQ6:
            return hi;
        default:
            return GGML_TYPE_COUNT;
    }
}

// PXQ slab-tier eligibility (shared by every PXQ tier): routed expert tensors (_exps.weight)
// always, plus the rev-2 backbone classes and anything opted in via PXA_PXQ_NATIVE; in all
// cases rows % 64 == 0 (panel height) and K % 32 == 0 (slab width).
// (The id-250 MXFP4-repack legacy production path — pxq4_permute_from_mxfp4 — was removed
//  2026-07-21 with the retirement of GGML_TYPE_PXQ4_LEGACY.)
static bool pxq4_tensor_eligible(const std::string & name, const ggml_tensor * t) {
    if (!pxq4_tensor_geometry_ok(t)) {
        return false;
    }
    if (pxq4_legacy_native_class(name)) {
        return true;
    }
    const pxa_pxq_bb_cfg & cfg = pxa_pxq_backbone_cfg();
    return cfg.mode != PXA_BB_LEGACY && !cfg.lite && pxa_pxq_backbone_native_class(name, t);
}

// The type/size the write loop's native-PXQ dispatch will ACTUALLY produce for this tensor,
// given the type the rules pipeline resolved (`rules_type`) and the whole-file tier. Returns
// `rules_type` unchanged wherever no PXQ dispatch applies.
//
// This exists because three places need the same answer and only one of them used to have it:
// the write loop (which had it inline), the --dry-run size projection, and the pre-flight
// guard below. The dry-run branch sits in the `if (params->dry_run)` arm and the native PXQ
// dispatch sits in the `else` arm, so a dry run accounted EVERY routed expert stack as its
// MXFP4 default. On a MoE model that projects a few percent of PXQ-family bytes and trips the
// end-of-run composition assertion — a failure the equivalent real run does not have. Any
// reasoning done from `--dry-run` output about a PXQ target was reasoning about a path the
// real run never takes.
static ggml_type pxa_pxq_landing_type(const std::string & name, const ggml_tensor * t,
                                      ggml_type rules_type, pxa_pxq_tier tier) {
    const bool want_pxq = rules_type == GGML_TYPE_PXQ1 || rules_type == GGML_TYPE_PXQ2 ||
                          rules_type == GGML_TYPE_PXQ3 || rules_type == GGML_TYPE_PXQ4 ||
                          rules_type == GGML_TYPE_PXQ4HQ || rules_type == GGML_TYPE_PXQ6;
    if (!pxq4_tensor_geometry_ok(t)) {
        // mirrors the geometry-demote arm: an explicit PXQ target that geometry cannot honour
        if (want_pxq) {
            return pxa_pxq_backbone_cfg().mode == PXA_BB_LEGACY ? GGML_TYPE_MXFP4 : GGML_TYPE_Q8_0;
        }
        return rules_type;
    }
    if (want_pxq) {
        return rules_type;                       // per-tensor target wins, geometry allows it
    }
    // an untouched MXFP4 default resolves to the whole-file tier. PXA_TIER_PXQU is NOT in this
    // set on purpose: the write loop excludes pxqu_out from the same upgrade, because a
    // UNIVERSAL mix is defined by its map, not by a whole-file tier.
    if (rules_type != GGML_TYPE_MXFP4 || !pxq4_legacy_native_class(name)) {
        return rules_type;
    }
    switch (tier) {
        case PXA_TIER_PXQ1:   return GGML_TYPE_PXQ1;
        case PXA_TIER_PXQ2:   return GGML_TYPE_PXQ2;
        case PXA_TIER_PXQ3:   return GGML_TYPE_PXQ3;
        case PXA_TIER_PXQ6:   return GGML_TYPE_PXQ6;
        case PXA_TIER_PXQ4HQ: return GGML_TYPE_PXQ4HQ;
        case PXA_TIER_PXQ4:   return GGML_TYPE_PXQ4;
        default:              return rules_type;   // PXA_TIER_NONE / PXA_TIER_PXQU
    }
}

// Byte size of a PXQ slab tensor. The slab layout is 64-row panels with an fp16 anchor header,
// so it is NOT ggml_row_size() x nrows; these are the same expressions the write loop uses.
// Returns 0 for a non-PXQ type.
static size_t pxa_pxq_slab_size(ggml_type t, int64_t K, int64_t R, int64_t E) {
    switch (t) {
        case GGML_TYPE_PXQ1:   return (size_t) E*(R/64)*(PXQ1_HDR_BYTES  + (K/32)*(int64_t)PXQ1_SLAB_BYTES);
        case GGML_TYPE_PXQ2:   return (size_t) E*(R/64)*(PXQ2_HDR_BYTES  + (K/32)*(int64_t)PXQ2_SLAB_BYTES);
        case GGML_TYPE_PXQ3:   return (size_t) E*(R/64)*(PXQ3_HDR_BYTES  + (K/32)*(int64_t)PXQ3_SLAB_BYTES);
        case GGML_TYPE_PXQ6:   return (size_t) E*(R/64)*(PXQ6R_HDR_BYTES + (K/32)*(int64_t)PXQ6R_SLAB_BYTES);
        case GGML_TYPE_PXQ4:   return (size_t) E*(R/64)*(PXQ6_HDR_BYTES  + (K/32)*(int64_t)PXQ6_SLAB_BYTES);
        case GGML_TYPE_PXQ4HQ: return (size_t) E*(R/64)*(PXQ6_HDR_BYTES  + (K/32)*(int64_t)PXQ6HQ_SLAB_BYTES);
        default:               return 0;
    }
}

// ---------------------------------------------------------------------------------------------
// PXQ ANCHOR GUARD (2026-09-02) — the check that would have stopped the 2026-07-22 tiers.
//
// Every PXQ tier stores one fp16 ROW ANCHOR per row, packed as a 128 B header at the head of
// each 64-row panel, and that anchor is a row |absmax|: finite and NON-NEGATIVE by construction
// (src/pxq{1,2,3,6,6r}-quantize.inc.cpp all write fp16_rn(absmax)). So a non-finite or negative
// anchor is not a bad quantization — it is not a quantization at all. It means those panel bytes
// were never written by the codec and the reader is looking at whatever the reused `work` buffer
// still held, or at a scale that overflowed fp16.
//
// This is not hypothetical. The published PXA-Fusion2-35B PXQ2/PXQ3/PXQ4/PXQU tiers (uploaded
// 2026-07-22) carry exactly this: in EVERY ffn_*_exps tensor the last 7*R*E elements' worth of
// panels are stale bytes, so the tail experts decode to non-finite weights and any prompt whose
// router picks one emits "!!!!" (non-finite logits). Nothing in the pipeline noticed: PXQ types
// have no CPU codec, ggml_validate_row_data() has no PXQ arm, and pxa_validate_tensor() (the
// --check-tensors scan) covers only the IQ* block types. A quantize run printed a normal
// "size = 512.00 MiB -> 73.00 MiB" line for a file that was broken on disk.
//
// So: scan the anchors of every PXQ tensor as it is produced, and refuse to write a bad one.
// The scan is one fp16 read per row (2 B of every 146 B for PXQ2) — unmeasurable next to the
// full sub-scale search the codec just ran.
static void pxa_pxq_check_anchors(ggml_type t, const void * data, int64_t K, int64_t R, int64_t E,
                                  const std::string & name) {
    const size_t slab = pxa_pxq_slab_size(t, K, R, E);
    if (slab == 0 || R < 64 || E < 1) {
        return;                                  // not a PXQ slab tensor
    }
    const int64_t panels = (R/64)*E;
    const int64_t panel_bytes = (int64_t)(slab / (size_t) panels);
    const uint8_t * base = (const uint8_t *) data;

    int64_t n_bad = 0, first_row = -1, first_expert = -1;
    float   first_val = 0.0f;
    for (int64_t p = 0; p < panels; ++p) {
        const ggml_fp16_t * hdr = (const ggml_fp16_t *)(base + p*panel_bytes);
        for (int r = 0; r < 64; ++r) {
            const float a = ggml_fp16_to_fp32(hdr[r]);
            if (!std::isfinite(a) || a < 0.0f) {
                if (n_bad == 0) {
                    first_expert = p / (R/64);
                    first_row    = (p % (R/64))*64 + r;
                    first_val    = a;
                }
                ++n_bad;
            }
        }
    }
    if (n_bad > 0) {
        throw std::runtime_error(format(
            "PXQ ANCHOR GUARD: %s (%s, K=%" PRId64 " R=%" PRId64 " E=%" PRId64 ") has %" PRId64
            " of %" PRId64 " fp16 row anchors that are non-finite or negative — first at expert %"
            PRId64 ", row %" PRId64 " (value %g). A row anchor is a |absmax| and can be neither, so "
            "those panels were never written by the codec (or a scale overflowed fp16). Refusing to "
            "write a file whose experts decode to NaN. This is the defect that shipped in the "
            "2026-07-22 PXA-Fusion2-35B tiers; see tools/pxq/pxq-anchor-check.py to audit an "
            "existing GGUF.",
            name.c_str(), ggml_type_name(t), K, R, E, n_bad, panels*64,
            first_expert, first_row, (double) first_val));
    }
}

// ============================================================================
// Quantize-time error budget (PXA_PXQ_ERRBUDGET) — the acceptance instrument
// ============================================================================
// Dequantize every written tensor straight back and accumulate relative RMS error vs the f32
// source, grouped by tensor CLASS (the name with the "blk.N." prefix stripped). Prints a table
// at completion and drops a <output>.errbudget.tsv next to the artifact.
//
// Why it exists: the flat-MXFP4 backbone shipped TWO corrupt Laguna artifacts (PXQ4 and PXQ6)
// that looked completely normal on disk and only revealed themselves in generated text. The
// signature was sitting in the numbers the whole time — attn_gate 0.1609 relative RMS, and
// attn_output at 3.2x the Q4_K_M control — i.e. one table print away from being caught before
// a single token was generated.
//
//   PXA_PXQ_ERRBUDGET=1              enable (default off: it costs a full dequant pass)
//   PXA_PXQ_ERRBUDGET_REF=<tsv>      compare against a stored budget; WARN at >1.5x any class
//   PXA_PXQ_ERRBUDGET_MAXELEM=<n>    per-tensor sampling cap, whole rows (default 64M, 0 = all)
//
// PXQ coverage (2026-08-18): the slab types have no per-row to_float (a row's bytes are
// scattered across its 64-row panel), so this instrument used to report "n/a" on every PXQ
// tensor — blind on exactly the types it exists to police. The panel-aware CPU dequant
// (pxa_pxq_dequant_row, pxq-cpu.c) decodes PXQ4/PXQ4HQ/PXQ2/PXQ3 from the tensor base, so
// those measure like any other type now; only the 5-bit PXQ6 slab type (no CPU fallback
// yet) still reports "n/a".
struct pxa_err_acc {
    double sse = 0.0;
    double ssq = 0.0;
    double nel = 0.0;
    int    ntensor = 0;
    int    nskipped = 0;
    std::set<std::string> types;
};

static bool pxa_errbudget_enabled() {
    static const bool v = [] {
        const char * e = getenv("PXA_PXQ_ERRBUDGET");
        return e && atoi(e) != 0;
    }();
    return v;
}

static int64_t pxa_errbudget_max_elem() {
    static const int64_t v = [] {
        const char * e = getenv("PXA_PXQ_ERRBUDGET_MAXELEM");
        return e ? (int64_t) atoll(e) : (int64_t) 64*1024*1024;
    }();
    return v;
}

// "blk.24.attn_output.weight" -> "attn_output.weight"; non-layer names pass through.
static std::string pxa_tensor_class(const std::string & name) {
    if (name.compare(0, 4, "blk.") != 0) {
        return name;
    }
    const size_t p1 = name.find('.', 4);
    return p1 == std::string::npos ? name : name.substr(p1 + 1);
}

static void pxa_errbudget_accumulate(std::map<std::string, pxa_err_acc> & acc,
                                     const std::string & name, ggml_type type,
                                     const void * qdata, const float * src,
                                     int64_t ne0, int64_t nrows) {
    pxa_err_acc & a = acc[pxa_tensor_class(name)];
    a.ntensor++;
    a.types.insert(ggml_type_name(type));
    const ggml_type_traits_t tt = ggml_internal_get_type_traits(type);
    // PXQ slab types keep to_float NULL on purpose — decode them through the panel-aware
    // CPU dequant instead (global row index into the tensor base; expert stacks are
    // contiguous %64-row slices, so the panel arithmetic holds across ne[2])
    const bool pxq = pxa_pxq_is_cpu_supported(type);
    // no CPU codec (the 5-bit PXQ6 slab type), or a row-interleaved layout whose rows are
    // not independently decodable -> honestly report it as unmeasured rather than guess
    if (!pxq && (tt.to_float == nullptr || interleaved_properties(type).second != 1)) {
        a.nskipped++;
        return;
    }
    const int64_t cap = pxa_errbudget_max_elem();
    int64_t rows = nrows;
    if (cap > 0 && ne0 > 0 && ne0*rows > cap) {
        rows = std::max<int64_t>(1, cap/ne0);
    }
    const size_t rs = pxq ? 0 : ggml_row_size(type, ne0);
    std::vector<float> buf(ne0);
    double sse = 0.0, ssq = 0.0;
    for (int64_t r = 0; r < rows; ++r) {
        if (pxq) {
            pxa_pxq_dequant_row(type, qdata, r, ne0, buf.data());
        } else {
            tt.to_float((const char *)qdata + (size_t)r*rs, buf.data(), ne0);
        }
        const float * x = src + (size_t)r*ne0;
        for (int64_t j = 0; j < ne0; ++j) {
            const double d = (double)x[j] - (double)buf[j];
            sse += d*d;
            ssq += (double)x[j]*(double)x[j];
        }
    }
    a.sse += sse;
    a.ssq += ssq;
    a.nel += (double)rows*ne0;
}

static std::map<std::string, double> pxa_errbudget_load_ref(const char * path) {
    std::map<std::string, double> ref;
    std::ifstream in(path);
    if (!in) {
        LLAMA_LOG_WARN("PXA_PXQ_ERRBUDGET_REF: cannot open %s — no comparison\n", path);
        return ref;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        try {
            ref[line.substr(0, tab)] = std::stod(line.substr(tab + 1));
        } catch (...) { /* skip malformed row */ }
    }
    return ref;
}

static void pxa_errbudget_report(const std::map<std::string, pxa_err_acc> & acc, const std::string & fname_out) {
    if (acc.empty()) return;
    std::map<std::string, double> ref;
    const char * refp = getenv("PXA_PXQ_ERRBUDGET_REF");
    if (refp && refp[0]) ref = pxa_errbudget_load_ref(refp);

    LLAMA_LOG_INFO("\n=============== PXQ error budget (relative RMS vs f32 source) ===============\n");
    LLAMA_LOG_INFO("%-34s %-16s %7s %12s %9s %9s %7s\n",
                   "class", "type(s)", "tensors", "elements", "rel_rms", "ref", "ratio");
    int warned = 0;
    std::ofstream tsv(fname_out + ".errbudget.tsv");
    if (tsv) tsv << "# class\trel_rms\ttypes\ttensors\telements\n";
    for (const auto & kv : acc) {
        const pxa_err_acc & a = kv.second;
        std::string types;
        for (const auto & t : a.types) { if (!types.empty()) types += ","; types += t; }
        if (a.nel <= 0.0 || a.ssq <= 0.0) {
            LLAMA_LOG_INFO("%-34s %-16s %7d %12s %9s %9s %7s\n",
                           kv.first.c_str(), types.c_str(), a.ntensor, "-", "n/a", "-", "-");
            continue;
        }
        const double rel = std::sqrt(a.sse/a.ssq);
        double refv = -1.0, ratio = -1.0;
        auto it = ref.find(kv.first);
        if (it != ref.end() && it->second > 0.0) { refv = it->second; ratio = rel/refv; }
        char rbuf[32], qbuf[32];
        if (refv >= 0.0) { snprintf(rbuf, sizeof rbuf, "%.4f", refv); snprintf(qbuf, sizeof qbuf, "%.2f", ratio); }
        else             { snprintf(rbuf, sizeof rbuf, "-");          snprintf(qbuf, sizeof qbuf, "-"); }
        LLAMA_LOG_INFO("%-34s %-16s %7d %12.3e %9.4f %9s %7s%s\n",
                       kv.first.c_str(), types.c_str(), a.ntensor, a.nel, rel, rbuf, qbuf,
                       ratio > 1.5 ? "  <== OVER BUDGET" : "");
        if (ratio > 1.5) ++warned;
        if (tsv) tsv << kv.first << "\t" << rel << "\t" << types << "\t" << a.ntensor << "\t" << (long long)a.nel << "\n";
    }
    if (warned) {
        LLAMA_LOG_WARN("PXQ error budget: %d class(es) OVER 1.5x the reference — inspect before shipping this artifact\n", warned);
    }
    LLAMA_LOG_INFO("budget written to %s.errbudget.tsv\n", fname_out.c_str());
    LLAMA_LOG_INFO("=============================================================================\n");
}

static void llama_model_quantize_internal(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    ggml_type default_type;
    llama_ftype ftype = params->ftype;

    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: default_type = GGML_TYPE_Q4_0; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1: default_type = GGML_TYPE_Q4_1; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0: default_type = GGML_TYPE_Q5_0; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1: default_type = GGML_TYPE_Q5_1; break;
        case LLAMA_FTYPE_MOSTLY_Q6_0: default_type = GGML_TYPE_Q6_0; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0: default_type = GGML_TYPE_Q8_0; break;
        case LLAMA_FTYPE_MOSTLY_Q8_KV:default_type = GGML_TYPE_Q8_KV;break;
        case LLAMA_FTYPE_MOSTLY_F16:  default_type = GGML_TYPE_F16;  break;
        case LLAMA_FTYPE_MOSTLY_BF16: default_type = GGML_TYPE_BF16; break;
        case LLAMA_FTYPE_ALL_F32:     default_type = GGML_TYPE_F32;  break;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    default_type = GGML_TYPE_Q2_K;    break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  default_type = GGML_TYPE_Q3_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  default_type = GGML_TYPE_Q4_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  default_type = GGML_TYPE_Q5_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    default_type = GGML_TYPE_Q6_K;    break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: default_type = GGML_TYPE_IQ2_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  default_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   default_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   default_type = GGML_TYPE_IQ2_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: default_type = GGML_TYPE_IQ3_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   default_type = GGML_TYPE_IQ1_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   default_type = GGML_TYPE_IQ1_M;   break;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  default_type = GGML_TYPE_IQ4_NL;  break;
        case LLAMA_FTYPE_MOSTLY_MXFP4:   default_type = GGML_TYPE_MXFP4;   break;
        case LLAMA_FTYPE_MOSTLY_PXQ4:    default_type = GGML_TYPE_MXFP4;   break; // PXQ4 = native expert quantize (E16-row scales)
        case LLAMA_FTYPE_MOSTLY_PXQ4HQ:  default_type = GGML_TYPE_MXFP4;   break; // PXQ4-HQ = bs8 tier
        case LLAMA_FTYPE_MOSTLY_PXQ6:    default_type = GGML_TYPE_MXFP4;   break; // PXQ6 = native expert quantize (LM32 5-bit x E16-row)
        case LLAMA_FTYPE_MOSTLY_PXQ2:          default_type = GGML_TYPE_MXFP4;   break; // PXQ2 = native expert quantize, MXFP4 rules for the rest
        case LLAMA_FTYPE_MOSTLY_PXQ3:          default_type = GGML_TYPE_MXFP4;   break; // PXQ3 = native expert quantize
        case LLAMA_FTYPE_MOSTLY_PXQ1:          default_type = GGML_TYPE_MXFP4;   break; // PXQ1 = native expert quantize (1-bit sign x E16-row)
        case LLAMA_FTYPE_MOSTLY_PXQ_UNIVERSAL: default_type = GGML_TYPE_MXFP4;   break; // per-tensor tier map via custom_quants
        case LLAMA_FTYPE_MOSTLY_Q1_0_G128: default_type = GGML_TYPE_Q1_0_G128; break;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  default_type = GGML_TYPE_IQ4_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:   default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_Q4_0_4_4: default_type = GGML_TYPE_Q4_0_4_4; break;
        case LLAMA_FTYPE_MOSTLY_Q4_0_4_8: default_type = GGML_TYPE_Q4_0_4_8; break;
        case LLAMA_FTYPE_MOSTLY_Q4_0_8_8: default_type = GGML_TYPE_Q4_0_8_8; break;

        default: throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

    // (The id-250 PXQ4-LEGACY lossless-repack and id-251 PXQ5 native-quantize output paths
    // were removed 2026-07-21 with the retirement of those types.)
    // PXQ4/PXQ4-HQ (the 4-bit quality tiers): eligible expert tensors quantize
    // NATIVELY (E16-row scales); everything else rides the MXFP4 rules pipeline. The gguf
    // keeps pxa.pxq6.* provenance KVs (historical key names — file-format contract).
    const bool pxq6_out   = ftype == LLAMA_FTYPE_MOSTLY_PXQ4;
    const bool pxq6hq_out = ftype == LLAMA_FTYPE_MOSTLY_PXQ4HQ;
    if (pxq6_out || pxq6hq_out) {
        ftype = LLAMA_FTYPE_MOSTLY_MXFP4;
    }
    // PXQ6 (the REAL 5-bit LM32 tier, gguf type id 256): same pattern — native expert
    // quantize, MXFP4 rules for the rest. (Internal pxq6r_* identifiers = the quantizer's
    // historical working name; the user-visible name is PXQ6/pxq6.)
    const bool pxq6r_out = ftype == LLAMA_FTYPE_MOSTLY_PXQ6;
    if (pxq6r_out) {
        ftype = LLAMA_FTYPE_MOSTLY_MXFP4;
    }
    // PXQ2/PXQ3/PXQ-UNIVERSAL: eligible expert tensors quantize NATIVELY; everything else
    // rides the MXFP4 rules pipeline (identical backbone to the shipped PXQ6 file). In
    // UNIVERSAL mode the per-tensor targets arrive via params->custom_quants (the
    // --pxq-universal tier map) and the write-loop dispatches on new_type.
    const bool pxq2_out = ftype == LLAMA_FTYPE_MOSTLY_PXQ2;
    const bool pxq3_out = ftype == LLAMA_FTYPE_MOSTLY_PXQ3;
    const bool pxqu_out = ftype == LLAMA_FTYPE_MOSTLY_PXQ_UNIVERSAL;

    // PXQ_UNIVERSAL IS NOTHING WITHOUT ITS TIER MAP. The whole meaning of this ftype is
    // "a per-tensor mix defined by --pxq-universal". Given no map, every tensor falls to the
    // backbone default and the run emits a UNIFORM PXQ4-HQ file that claims to be
    // PXQ_UNIVERSAL -- with rc=0 and no warning. Nothing downstream can catch it: the
    // end-of-run composition assertion sets spec_type = -1 for this ftype precisely because
    // the mix is map-defined, which disables its named-tier check; and every PXQ file carries
    // the same general.file_type, so the result is byte-indistinguishable from a real
    // PXQ4-HQ build. An operator would ship a mislabelled model and have no way to notice.
    //
    // Refuse HERE, before the write loop, rather than after hours of quantization. A tier map
    // is authored per-tensor and cannot be guessed on the operator's behalf.
    if (pxqu_out) {
        // CustomQ is a function-local alias at every other use site (see the two
        // `using CustomQ = ...` declarations above); it is not a file-scope type.
        using CustomQ = std::pair<std::string, ggml_type>;
        const auto * rules = params->custom_quants
            ? static_cast<const std::vector<CustomQ>*>(params->custom_quants) : nullptr;
        if (!rules || rules->empty()) {
            throw std::runtime_error(
                "PXQ_UNIVERSAL requires a per-tensor tier map and none was supplied. "
                "This ftype has no meaning on its own: without a map every tensor would take "
                "the backbone default and the output would be a UNIFORM PXQ4-HQ file labelled "
                "PXQ_UNIVERSAL -- indistinguishable from a real one, and undetectable by the "
                "composition assertion (which disables its named-tier check for this ftype "
                "because the mix is map-defined). Pass --pxq-universal /path/to/map.tiers "
                "(or a bare <name> resolved under $PXA_PXQU_DIR), or pick a uniform tier such "
                "as PXQ4 / PXQ4-HQ / PXQ3 / PXQ2 if a single tier is what you actually want.");
        }
    }

    if (pxq2_out || pxq3_out || pxqu_out) {
        ftype = LLAMA_FTYPE_MOSTLY_MXFP4;
    }
    // PXQ1 (the sub-2-bit tier): same pattern -- native expert quantize, MXFP4 rules for
    // the rest. Fixed {-1,+1} book + the shared SUB16 -> no provenance KVs needed.
    const bool pxq1_out = ftype == LLAMA_FTYPE_MOSTLY_PXQ1;
    if (pxq1_out) {
        ftype = LLAMA_FTYPE_MOSTLY_MXFP4;
    }

    // BACKBONE_REV 2: the tier context for pxa_pxq_backbone_type(). The MXFP4 flatten above
    // stays -- it remains the carrier for everything the table does not claim (norms, router,
    // ssm_*, MTP companions) and for PXA_PXQ_BACKBONE=legacy.
    const pxa_pxq_tier pxq_tier =
        pxq6_out   ? PXA_TIER_PXQ4   : pxq6hq_out ? PXA_TIER_PXQ4HQ :
        pxq6r_out  ? PXA_TIER_PXQ6   : pxq2_out   ? PXA_TIER_PXQ2   :
        pxq3_out   ? PXA_TIER_PXQ3   : pxq1_out   ? PXA_TIER_PXQ1   :
        pxqu_out   ? PXA_TIER_PXQU   : PXA_TIER_NONE;
    if (pxq_tier != PXA_TIER_NONE) {
        const pxa_pxq_bb_cfg & bbcfg = pxa_pxq_backbone_cfg();   // resolve + log once, before the write loop
        // PXQ1 and PXQ_UNIVERSAL sit behind the `universal` opt-in, so `core` ALONE is a silent
        // no-op on them: the table never runs and the backbone falls all the way back to flat
        // MXFP4 (measured: PXQ1+core on a dense model gave mxfp4 64.7%, vs pxq4 51.7% once
        // `universal` was added). Say so rather than shipping a file the operator believes is a
        // core build. Not auto-enabled: the opt-in is deliberate -- a PXQU tier map is authored
        // per-tensor and neither tier has measured backbone evidence.
        if (bbcfg.core && !bbcfg.univ &&
            (pxq_tier == PXA_TIER_PXQ1 || pxq_tier == PXA_TIER_PXQU)) {
            LLAMA_LOG_WARN("PXQ backbone: PXA_PXQ_BACKBONE=core has NO EFFECT on this tier — "
                           "PXQ1/PXQ_UNIVERSAL need the `universal` opt-in too. "
                           "Use PXA_PXQ_BACKBONE=core,universal for a core backbone here.\n");
        }
    }

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    // mmap consistently increases speed Linux, and also increases speed on Windows with
    // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    llama_model_kv_override * kv_overrides = nullptr;
    if (params->kv_overrides) {
        auto v = (std::vector<llama_model_kv_override>*)params->kv_overrides;
        kv_overrides = v->data();
    }
    llama_model_loader ml(fname_inp, 0, use_mmap, /*check_tensors*/ true, /* repack_tensors */ false,
            /* use_thp */ false, /* merge_qkv */ false, /* merge_up_gate_exps */ false,
            /* defer_experts */ false, kv_overrides, nullptr);
    ml.init_mappings(false); // no prefetching

    llama_model model;
    try {
        llm_load_arch(ml, model);
    } catch(const std::exception & e) {
        LLAMA_LOG_WARN("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX %s\n", e.what());
    }
    try {
        llm_load_hparams(ml, model, true);
    } catch(const std::exception & e) {
        LLAMA_LOG_WARN("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX %s\n", e.what());
    }

    struct quantize_state_internal qs(model, params);
    // PXQ ftypes (incl. UNIVERSAL): flag so the output head defaults to q8_0 (pxa_pxq_head_type)
    qs.is_pxq = pxq6_out || pxq6hq_out || pxq6r_out || pxq1_out || pxq2_out || pxq3_out || pxqu_out;

    // ---------------------------------------------------------------------------------------
    // PXQ MoE PRE-FLIGHT (2026-08-25). A uniform PXQ target takes the overwhelming majority of
    // its bytes from the routed expert stacks. If those cannot land on a PXQ type, the run
    // still does the ENTIRE quantize, then the end-of-run composition assertion deletes every
    // file it wrote and reports a bare percentage. On a large MoE that is hours of work for a
    // number and no output, and nothing in the message says which flag would have helped.
    //
    // Every input to that verdict is in the loader's tensor metadata before the first byte is
    // written, so refuse HERE and name the flag. Deliberately narrow: it fires only when NO
    // routed expert stack can be claimed, which is the case the assertion is certain to catch.
    // A partial shortfall is a warning, not a refusal -- the tier map / --custom-q may well
    // make up the difference and only the real composition can settle it.
    if (pxq_tier != PXA_TIER_NONE && pxq_tier != PXA_TIER_PXQU) {
        // CustomQ is a function-local alias at every use site, not a file-scope type.
        using CustomQ = std::pair<std::string, ggml_type>;
        const auto * rules = params->custom_quants
            ? static_cast<const std::vector<CustomQ>*>(params->custom_quants) : nullptr;
        const bool have_rules = rules && !rules->empty();

        int  n_exps = 0, n_exps_ok = 0;
        std::string bad_example;
        for (const auto & w : ml.weights) {
            const std::string tname(ggml_get_name(w.tensor));
            if (!pxa_name_ends(tname, "_exps.weight")) {
                continue;
            }
            ++n_exps;
            if (pxq4_tensor_geometry_ok(w.tensor)) {
                ++n_exps_ok;
            } else if (bad_example.empty()) {
                bad_example = format("%s [%" PRId64 " x %" PRId64 " x %" PRId64 "]", tname.c_str(),
                                     w.tensor->ne[0], w.tensor->ne[1], w.tensor->ne[2]);
            }
        }

        const int n_expert = (int) model.hparams.n_expert;
        if (n_expert > 1 && n_exps == 0 && !have_rules) {
            // The model declares routed experts but NOTHING matches the name the PXQ expert
            // path keys on. Every expert byte would ride the backbone's MXFP4 default and the
            // composition assertion would refuse the result -- after the full run. This is the
            // one case a --custom-q rule genuinely fixes, so name it.
            throw std::runtime_error(format(
                "PXQ pre-flight: this model declares %d routed experts, but NONE of its tensors "
                "match the '*_exps.weight' name the PXQ native expert path claims. Every expert "
                "stack would fall to the backbone's MXFP4 default, and the end-of-run "
                "composition assertion would then delete the output -- after the whole quantize. "
                "Claim the stacks explicitly, e.g. "
                "--custom-q '(ffn_down_exps|ffn_gate_exps|ffn_up_exps)\\.weight=pxq3' with the "
                "names this model actually uses (llama-gguf / gguf_dump on the source lists "
                "them), or pick a stock ftype instead of a PXQ tier.", n_expert));
        }
        if (n_exps > 0 && n_exps_ok == 0) {
            // Named correctly, but no stack clears the slab geometry, so no PXQ tier can encode
            // them at all and no flag changes that. Say so now rather than after the run.
            throw std::runtime_error(format(
                "PXQ pre-flight: all %d routed expert stacks fail PXQ slab geometry (need "
                "rows %% 64 == 0 and K %% 32 == 0); first offender %s. No PXQ tier can encode "
                "them, so this target would produce a file that is almost entirely non-PXQ and "
                "the end-of-run composition assertion would delete it. Quantize this model to a "
                "stock ftype (e.g. Q4_K_M / MXFP4) instead.",
                n_exps, bad_example.c_str()));
        }
        if (n_exps > 0 && n_exps_ok < n_exps) {
            LLAMA_LOG_WARN("%s: PXQ pre-flight: %d of %d routed expert stacks fail PXQ slab "
                           "geometry and will land on q8_0 (first: %s). The rest still take the "
                           "%s tier.\n", __func__, n_exps - n_exps_ok, n_exps, bad_example.c_str(),
                           llama_model_ftype_name(params->ftype).c_str());
        }
    }

    if (params->only_copy) {
        ftype = model.ftype;
    }
    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (params->imatrix) {
        imatrix_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->imatrix);
        if (imatrix_data) {
            LLAMA_LOG_INFO("================================ Have weights data with %d entries\n",int(imatrix_data->size()));
            qs.has_imatrix = true;
            // check imatrix for nans or infs
            for (const auto & kv : *imatrix_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) {
                        throw std::runtime_error(format("imatrix contains non-finite value %f\n", f));
                    }
                }
            }
        }
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;

    ensure_output_directory(fname_out);

    struct gguf_context * ctx_out = gguf_init_empty();

    // Early exit if partial_requant is enabled and output file already exists
    if (params->partial_requant && !params->keep_split) {
        std::ifstream test_file(fname_out);
        if (test_file) {
            LLAMA_LOG_INFO("%s: output file %s exists, skipping\n", __func__, fname_out.c_str());
            gguf_free(ctx_out);
            return;
        }
    }

    // copy the KV pairs from the input file
    gguf_set_kv     (ctx_out, ml.meta);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION); // TODO: use LLM_KV

    // Remove split metadata
    gguf_remove_key(ctx_out, ml.llm_kv(LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(ctx_out, ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(ctx_out, ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    if (params->kv_overrides) {
        const std::vector<llama_model_kv_override> & overrides = *(const std::vector<llama_model_kv_override> *)params->kv_overrides;
        for (auto & o : overrides) {
            if (o.key[0] == 0) break;
            if (o.tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT) {
                gguf_set_val_f32(ctx_out, o.key, o.val_f64);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_UINT) {
                gguf_set_val_u32(ctx_out, o.key, (uint32_t) o.val_i64);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
                gguf_set_val_i32(ctx_out, o.key, o.val_i64);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) {
                gguf_set_val_bool(ctx_out, o.key, o.val_bool);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_STR) {
                gguf_set_val_str(ctx_out, o.key, o.val_str);
            } else {
                LLAMA_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, o.key);
            }
        }
    }

    // --repack / --repack-pattern are gone with the row-interleaved _R4 / _R8 types they
    // existed to produce (ik separation, phase 3).
    for (int i = 0; i < ml.n_tensors; ++i) {
        const struct ggml_tensor * meta = ml.get_tensor_meta(i);

        const std::string name = ggml_get_name(meta);


        // TODO: avoid hardcoded tensor names - use the TN_* constants
        if (name.find("attn_v.weight")   != std::string::npos ||
            name.find("attn_qkv.weight") != std::string::npos) {
            ++qs.n_attention_wv;
        } else if (name == LLM_TN(model.arch)(LLM_TENSOR_OUTPUT, "weight")) {
            qs.has_output = true;
        }
    }


    gguf_set_val_u32(ctx_out, "general.file_type", ftype); // TODO: use LLM_KV
    if (pxq_tier != PXA_TIER_NONE) {
        // BACKBONE_REV provenance: which allocation table produced this file's NON-expert
        // tensors. 1 = the historical flat-MXFP4 backbone, 2 = the per-class table. The
        // eval harness and the claims registry read this so "which build made this file"
        // is a KV lookup, not archaeology.
        const pxa_pxq_bb_cfg & bb = pxa_pxq_backbone_cfg();
        const bool bb_on = bb.mode != PXA_BB_LEGACY &&
                           !((pxq_tier == PXA_TIER_PXQU || pxq_tier == PXA_TIER_PXQ1) && !bb.univ);
        gguf_set_val_u32(ctx_out, "pxa.pxq.backbone_rev", bb_on ? 2u : 1u);
        // The map is RESOLVED, not recited: run the write loop's own decision chain
        // (pxa_pxq_backbone_type -> the --custom-q stand-down -> the explicit --*-type
        // flags) over this model's actual tensors and record what came out, per role.
        // The hardcoded strings this replaces ignored PXA_PXQ_KV, --custom-q and the
        // model itself — the shipped 27B advertised attn_gate_head=f16 while containing
        // ZERO f16 tensors (qwen35 has no per-head gate), so two arms that differed in
        // backbone policy were indistinguishable by this KV. Roles the table declines
        // ride the legacy MXFP4 rules pipeline and are omitted.
        std::string bb_map = "legacy:mxfp4";
        if (bb_on) {
            std::map<std::string, std::set<std::string>> bb_roles;
            for (int i = 0; i < ml.n_tensors; ++i) {
                const ggml_tensor * meta = ml.get_tensor_meta(i);
                const std::string tname = ggml_get_name(meta);
                // mirror the write loop's eligibility gate: tensors it never quantizes
                // (non-weight, 1D, norms) keep their source type, so a table claim on
                // them (the .nextn. norms) would be one more string the file contradicts
                if (tname.rfind("weight") != tname.size() - 6 ||
                    ggml_n_dims(meta) < 2 ||
                    tname.find("_norm.weight") != std::string::npos) {
                    continue;
                }
                std::string val;
                if (pxa_name_is(tname, "output.weight")) {   // the head: pxa_pxq_head_type() owns it
                    if (!params->quantize_output_tensor) {
                        continue;   // --leave-output-tensor: the head keeps its source type
                    }
                    val = ggml_type_name(params->output_tensor_type < GGML_TYPE_COUNT
                                         ? params->output_tensor_type : pxa_pxq_head_type());
                } else {
                    const ggml_type bt = pxa_pxq_backbone_type(tname, meta, pxq_tier,
                                                               model.hparams.n_expert > 1);
                    if (bt == GGML_TYPE_COUNT) {
                        continue;
                    }
                    val = ggml_type_name(bt);
                }
                // --custom-q claims beat the table (the write loop's stand-down), and the
                // explicit --*-type flags beat both — mirror that order exactly
                if (const ggml_type ct = pxa_custom_rule_type(params, tname); ct < GGML_TYPE_COUNT) {
                    val = std::string("custom:") + ggml_type_name(ct);
                }
                const struct { const char * pat; ggml_type ty; } flag_ovr[] = {
                    { "token_embd.weight",  params->token_embedding_type },
                    { "attn_q.weight",      params->attn_q_type          },
                    { "attn_k.weight",      params->attn_k_type          },
                    { "attn_v.weight",      params->attn_v_type          },
                    { "attn_qkv.weight",    params->attn_qkv_type        },
                    { "attn_output.weight", params->attn_output_type     },
                };
                for (const auto & o : flag_ovr) {
                    if (o.ty < GGML_TYPE_COUNT && tname.find(o.pat) != std::string::npos) {
                        val = ggml_type_name(o.ty);
                    }
                }
                std::string role = pxa_tensor_class(tname);
                if (role.size() > 7 && role.compare(role.size() - 7, 7, ".weight") == 0) {
                    role.resize(role.size() - 7);
                }
                if (role == "attn_gate") {   // keep the head/channel split the harness knows
                    role = meta->ne[1] <= 256 ? "attn_gate_head" : "attn_gate_ch";
                }
                bb_roles[role].insert(val);
            }
            bb_map = bb.lite ? "lite:" : "";
            for (const auto & r : bb_roles) {
                std::string tys;
                for (const auto & v : r.second) { if (!tys.empty()) tys += ","; tys += v; }
                if (!bb_map.empty() && bb_map.back() != ':') bb_map += ";";
                bb_map += r.first + "=" + tys;
            }
        }
        gguf_set_val_str(ctx_out, "pxa.pxq.backbone_map", bb_map.c_str());
        LLAMA_LOG_INFO("PXQ backbone map (resolved): %s\n", bb_map.c_str());
        // and the raw override strings, so "why does this arm differ" is a KV lookup too
        std::string bb_ovr;
        if (const char * e = getenv("PXA_PXQ_BACKBONE"); e && *e) {
            bb_ovr += std::string("PXA_PXQ_BACKBONE=") + e;
        }
        if (const char * e = getenv("PXA_PXQ_KV"); e && *e) {
            if (!bb_ovr.empty()) bb_ovr += ";";
            bb_ovr += std::string("PXA_PXQ_KV=") + e;
        }
        if (const char * e = getenv("PXA_PXQ_HEAD"); e && *e) {
            if (!bb_ovr.empty()) bb_ovr += ";";
            bb_ovr += std::string("PXA_PXQ_HEAD=") + e;
        }
        if (!bb_ovr.empty()) {
            gguf_set_val_str(ctx_out, "pxa.pxq.backbone_overrides", bb_ovr.c_str());
        }
    }
    if (pxq6_out || pxq6hq_out) {   // 4-bit-tier provenance: the frozen tables this file was built with
        gguf_set_val_u32(ctx_out, "pxa.pxq6.version", 1);
        gguf_set_val_str(ctx_out, "pxa.pxq6.tier", pxq6hq_out ? "hq" : "core");
        gguf_set_arr_data(ctx_out, "pxa.pxq6.book", GGUF_TYPE_FLOAT32, pxq6_book_q(), 16);
        gguf_set_arr_data(ctx_out, "pxa.pxq6.sub",  GGUF_TYPE_FLOAT32, pxq6_sub_q(pxq6hq_out ? 1 : 0), 16);
    }
    if (pxq6r_out) {   // PXQ6 (5-bit LM32 tier) provenance: 32-entry book + shared SUB16
        gguf_set_val_u32(ctx_out, "pxa.pxq6.version", 1);
        gguf_set_val_str(ctx_out, "pxa.pxq6.tier", "lm32");
        gguf_set_arr_data(ctx_out, "pxa.pxq6.book", GGUF_TYPE_FLOAT32, pxq6r_book_q(), 32);
        gguf_set_arr_data(ctx_out, "pxa.pxq6.sub",  GGUF_TYPE_FLOAT32, pxq6r_sub_q(), 16);
    }
    if (pxq2_out || pxqu_out) {
        // version 3 == PXA_PXQ2_V3 refit book; version 2 == PXA_PXQ_CEIL_V2 v2 book (the book
        // KV below carries the actual table either way; the version KV is what the loader's
        // mismatch guard reads)
        gguf_set_val_u32(ctx_out, "pxa.pxq2.version", pxq2_v3_enabled() ? 3u : (pxq_ceil_v2_enabled() ? 2u : 1u));
        gguf_set_arr_data(ctx_out, "pxa.pxq2.book", GGUF_TYPE_FLOAT32, pxq2_book_q(), 4);
        gguf_set_arr_data(ctx_out, "pxa.pxq2.sub",  GGUF_TYPE_FLOAT32, pxq2_sub_q(), 16);
    }
    if (pxq3_out || pxqu_out) {
        gguf_set_val_u32(ctx_out, "pxa.pxq3.version", pxq_ceil_v2_enabled() ? 2u : 1u);
        gguf_set_arr_data(ctx_out, "pxa.pxq3.book", GGUF_TYPE_FLOAT32, pxq3_book_q(), 8);
        gguf_set_arr_data(ctx_out, "pxa.pxq3.sub",  GGUF_TYPE_FLOAT32, pxq3_sub_q(), 16);
    }
    if (pxqu_out) {
        // No pxa.pxqu.preset provenance KV: it would need a new `pxqu_preset` field threaded
        // through llama_model_quantize_params (llama.h) and its default-params initializer, i.e.
        // a public C API struct change for a cosmetic string. The tier map is fully recoverable
        // from the per-tensor gguf types (pxa.pxq2/pxq3/pxq6.* + each tensor's own ggml_type),
        // so the KV is not load-bearing.
        gguf_set_val_u32(ctx_out, "pxa.pxqu.version", 1);
        gguf_set_arr_data(ctx_out, "pxa.pxq6.book", GGUF_TYPE_FLOAT32, pxq6_book_q(), 16);   // 4-bit tier rides PXQ6
        gguf_set_arr_data(ctx_out, "pxa.pxq6.sub",  GGUF_TYPE_FLOAT32, pxq6_sub_q(0), 16);
    }

    qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)model.hparams.n_layer;

    // sanity checks
    //
    //  - qs.n_attention_wv == 0                         for Mamba           models
    //  - qs.n_attention_wv == model.hparams.n_layer     for Transformer     models
    //  - qs.n_attention_wv == 3 * model.hparams.n_layer for Encoder-Decoder models
    //  - model.arch == LLM_ARCH_DECI                    for Deci-Nemotron   models
    //
    GGML_ASSERT((qs.n_attention_wv == 0 ||
                 qs.n_attention_wv == (int)model.hparams.n_layer ||
                 qs.n_attention_wv == 3 * (int)model.hparams.n_layer ||
                 model.arch == LLM_ARCH_DECI ||
                 model.arch == LLM_ARCH_GEMMA4 ||
                 model.arch == LLM_ARCH_QWEN35MOE ||   // hybrid: only full-attn layers carry wv (every full_attention_interval)
                 model.arch == LLM_ARCH_QWEN3NEXT ||    // same Gated-DeltaNet hybrid layout
                 model.arch == LLM_ARCH_QWEN35 ||      // dense sibling of QWEN35MOE; identical hybrid layout
                 model.arch == LLM_ARCH_UNKNOWN) && "n_attention_wv is unexpected");

    size_t total_size_org = 0;
    size_t total_size_new = 0;
    // PXQ-P5 composition guard (2026-07-27): per-output-type stats for the end-of-run
    // composition summary + the target/composition assertion. Keyed by ggml tensor type id.
    std::map<int, std::pair<int64_t, size_t>> comp_stats;   // type -> {count, bytes}
    std::vector<std::string> comp_written_files;            // every path this run opened for write
    std::map<std::string, pxa_err_acc> errbudget;   // PXA_PXQ_ERRBUDGET, empty unless enabled

    std::vector<std::thread> workers;
    workers.reserve(nthread);

    int idx = 0;

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;

    uint16_t n_split = 1;
    // Assume split index is continuous
    if (params->keep_split) {
        for (int i = 0; i < ml.n_tensors; ++i) {
            n_split = std::max(uint16_t(ml.get_weight(i)->idx+1), n_split);
        }
    }
    std::vector<gguf_context*> ctx_outs(n_split, NULL);
    ctx_outs[0] = ctx_out;

    ggml_tensor extra;
    ggml_tensor * output_meta = ml.get_tensor_meta("output.weight");
    if (!output_meta) {
        output_meta = ml.get_tensor_meta("token_embd.weight");
    }
    ggml_tensor * output_tensor = nullptr;
    if (params->extra_output_type != GGML_TYPE_COUNT) {
        auto meta = ml.get_tensor_meta("output.weight");
        if (!meta) {
            meta = ml.get_tensor_meta("token_embd.weight");
        }
        if (!meta) {
            LLAMA_LOG_WARN("Extra output tensor requested, but 'output.weight' or 'token_embd.weight' not found\n");
        } else {
            LLAMA_LOG_INFO("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Will duplicate %s as %s\n", meta->name,
                    ggml_type_name(params->extra_output_type));
            auto weights = ml.get_weight(meta->name);
            output_tensor = weights->tensor;
            extra = *output_tensor;
            auto new_type = params->extra_output_type;
            extra.type = new_type;
            auto tt = ggml_internal_get_type_traits(extra.type);
            extra.nb[0] = tt.type_size;
            extra.nb[1] = ggml_row_size(extra.type, extra.ne[0]);
            extra.nb[2] = extra.nb[3] = extra.nb[1]*extra.ne[1];
            extra.data  = nullptr;
            strcpy(extra.name, "output_extra.weight");
            auto orig_size = ggml_nbytes(output_tensor);
            auto new_size  = ggml_nbytes(&extra);
            if (new_size >= orig_size) {
                LLAMA_LOG_INFO("No, duplicating it makes no sense as the new size (%zu) is greater than the original size (%zu)\n",
                        new_size, orig_size);
                output_tensor = nullptr;
            }
        }
    }

    // populate the original tensors so we get an initial meta data
    for (int i = 0; i < ml.n_tensors; ++i) {
        auto weight = ml.get_weight(i);
        uint16_t i_split = params->keep_split ? weight->idx : 0;
        struct ggml_tensor * tensor = weight->tensor;
        if (ctx_outs[i_split] == NULL) {
            ctx_outs[i_split] = gguf_init_empty();
        }
        gguf_add_tensor(ctx_outs[i_split], tensor);
        if (tensor == output_tensor) {
            gguf_add_tensor(ctx_outs[i_split], &extra);
        }
    }

    // Set split info if needed
    if (n_split > 1) {
        for (size_t i = 0; i < ctx_outs.size(); ++i) {
            gguf_set_val_u16(ctx_outs[i], ml.llm_kv(LLM_KV_SPLIT_NO).c_str(), i);
            gguf_set_val_u16(ctx_outs[i], ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str(), n_split);
            gguf_set_val_i32(ctx_outs[i], ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str(), ml.n_tensors);
        }
    }

    int cur_split = -1;
    std::ofstream fout;
    std::vector<bool> split_skipped(n_split, false);
    auto close_ofstream = [&]() {
        // Write metadata and close file handler
        if (fout.is_open()) {
            fout.seekp(0);
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_outs[cur_split]));
            gguf_get_meta_data(ctx_outs[cur_split], data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&](int index) {
        if (params->dry_run) {
            return;
        }
        cur_split = index;
        GGML_ASSERT(ctx_outs[cur_split] && "Find uninitialized gguf_context");
        std::string fname = fname_out;
        if (params->keep_split) {
            char split_path[PATH_MAX] = {0};
            llama_split_path(split_path, sizeof(split_path), fname_out.c_str(), cur_split, n_split);
            fname = std::string(split_path);
        }

        if (params->partial_requant) {
            std::ifstream test_file(fname);
            if (test_file) {
                LLAMA_LOG_INFO("%s: split file %s exists, skipping\n", __func__, fname.c_str());
                split_skipped[cur_split] = true;
                fout = std::ofstream();
                return;
            }
        }

        ensure_output_directory(fname);
        comp_written_files.push_back(fname);
        fout = std::ofstream(fname, std::ios::binary);
        fout.exceptions(std::ofstream::failbit); // fail fast on write errors
        const size_t meta_size = gguf_get_meta_size(ctx_outs[cur_split]);
        // placeholder for the meta data
        ::zeros(fout, meta_size);
    };

    const auto tn = LLM_TN(model.arch);
    new_ofstream(0);
    for (int i = 0; i < ml.n_tensors; ++i) {
        auto weight = ml.get_weight(i);
        struct ggml_tensor * tensor = weight->tensor;
        if (weight->idx != cur_split && params->keep_split) {
            close_ofstream();
            new_ofstream(weight->idx);
        }

        if (params->partial_requant && split_skipped[cur_split]) {
            const std::string name = ggml_get_name(tensor);
            gguf_set_tensor_type(ctx_outs[cur_split], name.c_str(), tensor->type);
            gguf_set_tensor_data(ctx_outs[cur_split], name.c_str(), tensor->data, ggml_nbytes(tensor));
            continue;
        }

        std::string name = ggml_get_name(tensor);

        if (!ml.use_mmap) {
            if (read_data.size() < ggml_nbytes(tensor)) {
                read_data.resize(ggml_nbytes(tensor));
            }
            tensor->data = read_data.data();
        }
        ml.load_data_for(tensor);

        LLAMA_LOG_INFO("[%4d/%4d] %36s - [%s], type = %6s, ",
               ++idx, ml.n_tensors,
               ggml_get_name(tensor),
               llama_format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

        // This used to be a regex, but <regex> has an extreme cost to compile times.
        bool quantize = name.rfind("weight") == name.size() - 6; // ends with 'weight'?

        // quantize only 2D and 3D tensors (experts)
        quantize &= (ggml_n_dims(tensor) >= 2);

        // do not quantize norm tensors
        quantize &= name.find("_norm.weight") == std::string::npos;

        quantize &= params->quantize_output_tensor || name != "output.weight";
        quantize &= !params->only_copy;

        // do not quantize expert gating tensors
        // NOTE: can't use LLM_TN here because the layer number is not known
        if (name.find("ffn_gate_inp.weight") != std::string::npos) {
            if (params->ffn_gate_inp_type == GGML_TYPE_COUNT || params->ffn_gate_inp_type == tensor->type) {
                quantize = false;
            }
        }
        //quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;

        // do not quantize the DeepSeek-V4 hash routing table. It is an I32
        // token-id -> expert-id map: 2-D, and its name ends in "weight", so every
        // generic rule above marks it quantizable. Quantizing it is a textbook
        // "loads and emits fluent garbage" failure. Upstream added the same line
        // in src/llama-quant.cpp (PR #25787).
        quantize &= name.find("ffn_gate_tid2eid.weight") == std::string::npos;

        // do not quantize positional embeddings and token types (BERT)
        quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_POS_EMBD,    "weight");
        quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_TOKEN_TYPES, "weight");

        // do not quantize Mamba's small yet 2D weights
        // NOTE: can't use LLM_TN here because the layer number is not known
        // A row-gather (GET_ROWS) table must not be touched by this quantizer at
        // all. Two independent reasons, both load-bearing:
        //   * its codecs are panel codecs -- a row is spread across a panel with
        //     shared state, so such a table loads cleanly and gathers nonsense.
        //   * the fallback could not be Q4_K anyway: that needs ne0 % 256 == 0 and
        //     per_layer_token_embd has ne0 = 160. Block-32 codecs are the only
        //     legal choice there.
        //   * the dequantize path sizes one f32 buffer to the WHOLE tensor. For
        //     51.2B parameters that is ~205 GB; it OOMs long before it quantizes.
        // The table is quantized where it is created, by the converter, with a
        // block-32 codec. Copy it through untouched.
        quantize &= !pxa_is_row_gather_tensor(tensor);
        // Every codec here has a block of at least 32 and ASSERTS
        // n_per_row % block == 0 -- it aborts the process rather than falling
        // back. Conv kernels are 4 elements wide (ssm_conv1d and ple_conv1d are
        // both [4, 10240]), so a short row must be left alone structurally, not
        // by remembering to name each one. Turns a crash into a copy.
        quantize &= (tensor->ne[0] % 32 == 0);
        quantize &= name.find("ssm_conv1d.weight") == std::string::npos;
        quantize &= name.find("ssm_x.weight")      == std::string::npos;
        quantize &= name.find("ssm_dt.weight")     == std::string::npos;

        // do not quantize relative position bias (T5)
        quantize &= name.find("attn_rel_b.weight") == std::string::npos;

        // quantize the extra output tensor
        quantize = tensor == output_tensor || quantize;

        enum ggml_type new_type;
        void * new_data = nullptr;
        size_t new_size = 0;


        if (quantize) {

            new_type = default_type;

            // get more optimal quantization type based on the tensor shape, layer, etc.
            if (params->pure) {
                auto working_type = change_type_if_necessary(new_type, tensor->ne[0], tensor->ne[1]);
                if (working_type != new_type) {
                    ++qs.n_fallback;
                    new_type = working_type;
                }
            }
            else if (ggml_is_quantized(default_type)) {
                new_type = llama_tensor_get_type(qs, new_type, tensor, ftype);

                // BACKBONE_REV 2: per-class promotion for everything the native expert codecs
                // do not own, replacing the flat-MXFP4 backbone. Priority order is deliberate:
                //   --custom-q / --pxq-universal map  >  this table  >  the MXFP4 rules pipeline
                // and the explicit --attn-q-type/--token-embedding-type/... overrides below
                // still win over all of it. --pure stands down entirely (it means "no rules").
                if (pxq_tier != PXA_TIER_NONE) {
                    const ggml_type bb = pxa_pxq_backbone_type(name, tensor, pxq_tier,
                                                               qs.model.hparams.n_expert > 1);
                    // the regex scan is the expensive half, so only pay it for a tensor the
                    // table actually wants to move
                    if (bb < GGML_TYPE_COUNT && bb != new_type && !pxa_custom_rule_matches(params, name)) {
                        LLAMA_LOG_INFO("\nPXQ backbone rev2: %s -> %s ", ggml_type_name(new_type),
                                       ggml_type_name(bb));
                        new_type = bb;
                    }
                }
            }
            if (params->token_embedding_type < GGML_TYPE_COUNT && strcmp(tensor->name, "token_embd.weight") == 0) {
                new_type = params->token_embedding_type;
            }
            if (params->output_tensor_type < GGML_TYPE_COUNT && strcmp(tensor->name, "output.weight") == 0) {
                new_type = params->output_tensor_type;
            }
            else if (params->only_copy && tensor == output_tensor) {
                new_type = tensor->type;
            }
            if (params->ffn_gate_inp_type < GGML_TYPE_COUNT && name.find("ffn_gate_inp.weight") != std::string::npos) {
                new_type = params->ffn_gate_inp_type;
            }
            if (params->attn_q_type < GGML_TYPE_COUNT && name.find("attn_q.weight") != std::string::npos) {
                new_type = params->attn_q_type;
            }
            if (params->attn_k_type < GGML_TYPE_COUNT && name.find("attn_k.weight") != std::string::npos) {
                new_type = params->attn_k_type;
            }
            if (params->attn_v_type < GGML_TYPE_COUNT && name.find("attn_v.weight") != std::string::npos) {
                new_type = params->attn_v_type;
            }
            if (params->attn_qkv_type < GGML_TYPE_COUNT && name.find("attn_qkv.weight") != std::string::npos) {
                new_type = params->attn_qkv_type;
            }
            if (params->attn_output_type < GGML_TYPE_COUNT && name.find("attn_output.weight") != std::string::npos) {
                new_type = params->attn_output_type;
            }
            if (params->ffn_gate_type < GGML_TYPE_COUNT && strcmp(tensor->name, "ffn_gate") == 0) {
                new_type = params->ffn_gate_type;
            }
            if (params->ffn_down_type < GGML_TYPE_COUNT && strcmp(tensor->name, "ffn_down") == 0) {
                new_type = params->ffn_down_type;
            }
            if (params->ffn_up_type < GGML_TYPE_COUNT && strcmp(tensor->name, "ffn_up") == 0) {
                new_type = params->ffn_up_type;
            }

            if (strcmp(tensor->name, "token_embd.weight") == 0) {
                // token embeddings cannot be quantized with row-interleaved quants
                auto working_type = interleaved_properties(new_type).first;
                if (working_type != new_type) {
                    printf("\n============ Token embeddings cannot be quantized with row-interleaved quants\n");
                    printf("---> Changed %s to %s\n", ggml_type_name(new_type), ggml_type_name(working_type));
                    new_type = working_type;
                }
            }

            // If we've decided to quantize to the same type the tensor is already
            // in then there's nothing to do.
            if (tensor != output_tensor) {
                // ...EXCEPT for a PXQ tier reading an MXFP4 source. The PXQ ftypes are
                // FLATTENED to MOSTLY_MXFP4 above (see pxq2_out/pxq3_out/... at the top
                // of this function), so default_type is MXFP4 and this equality holds for
                // every expert tensor of a GGUF whose converter wrote raw MXFP4 blocks
                // (DeepSeek-V4 upstream-style). quantize would go false, the verbatim-copy
                // branch would run, and the native-PXQ upgrade further down would NEVER
                // fire -- producing a file NAMED PXQ2 that is actually 4.25 bpw. That is a
                // 137 GiB artifact where a 73 GiB one was asked for, printed as a bland
                // "size = ... MB" line with no warning.
                //
                // NOTE this is the FALLBACK, not the plan: MXFP4 -> f32 -> PXQ2 is doubly
                // lossy and makes an imatrix measure the wrong function. Prefer converting
                // from a high-precision source (for DS4 that is the default converter mode,
                // which dequantizes the experts). This exists so the upstream-shaped file
                // cannot silently lie about its own tier. Needs --allow-requantize.
                const bool pxq_upgrade = pxq_tier != PXA_TIER_NONE
                                      && new_type    == GGML_TYPE_MXFP4
                                      && tensor->type == GGML_TYPE_MXFP4
                                      && pxq4_legacy_native_class(name)
                                      && pxq4_tensor_geometry_ok(tensor);
                if (pxq_upgrade) {
                    LLAMA_LOG_INFO("\nPXQ: %s is MXFP4 in the source; requantizing to the "
                                   "%s tier instead of copying it verbatim ", name.c_str(),
                                   ggml_type_name(pxq_tier == PXA_TIER_PXQ1 ? GGML_TYPE_PXQ1 :
                                                  pxq_tier == PXA_TIER_PXQ2 ? GGML_TYPE_PXQ2 :
                                                  pxq_tier == PXA_TIER_PXQ3 ? GGML_TYPE_PXQ3 :
                                                                              GGML_TYPE_PXQ4));
                } else {
                    quantize &= tensor->type != new_type;
                }
            }
        }

        if (!quantize) {
            new_type = tensor->type;
            new_data = tensor->data;
            new_size = ggml_nbytes(tensor);
            LLAMA_LOG_INFO("size = %8.3f MB\n", ggml_nbytes(tensor)/1024.0/1024.0);
        } else {
            const int64_t nelements = ggml_nelements(tensor);

            const float * imatrix = nullptr;
            if (imatrix_data) {
                auto it = imatrix_data->find(tensor->name);
                if (it == imatrix_data->end()) {
                    if (auto pos1 = name.find("ffn_up_exps.weight"), pos2 = name.find("ffn_gate_exps.weight"); pos1 != std::string::npos || pos2 != std::string::npos) {
                        // Merged ffn_up/gate_exps hack
                        auto pos = pos1 != std::string::npos ? pos1 : pos2;
                        auto merged_name = name.substr(0, pos) + "ffn_gate_up_exps.weight";
                        it = imatrix_data->find(merged_name);
                        if (it == imatrix_data->end()) {
                            auto up_name = name.substr(0, pos) + "ffn_up_exps.weight";
                            it = imatrix_data->find(up_name);
                        }
                    } else if (auto pos = name.find("ffn_gate_up_exps.weight"); pos != std::string::npos) {
                        auto not_merged_name = name.substr(0, pos) + "ffn_up_exps.weight";
                        it = imatrix_data->find(not_merged_name);
                    } else if (auto pos2 = name.find("ffn_gate.weight"); pos2 != std::string::npos) {
                        auto up_name = name.substr(0, pos2) + "ffn_up.weight";
                        it = imatrix_data->find(up_name);
                    } else {
                        // MLA hack: most imatrix files floating around the Internet have been computed with standard attention.
                        //           This means that the imatrix file does not contain data for the *.attn_k_b.weight and *.attn_v_b.weight
                        //           required by MLA. But the *.attn_v_b.weight tensors "see" the exact same activations as the
                        //           *.attn_kv_b.weight tensors used in standard attention. Hence, if we find imatrix data for
                        //           *.attn_kv_b.weight we can use it for *.attn_v_b.weight and vice versa.
                        std::string name{tensor->name};
                        static std::array<std::string, 2> alternatives{".attn_v_b.weight", ".attn_kv_b.weight"};
                        for (int j = 0; j < int(alternatives.size()); ++j) {
                            if (auto pos = name.find(alternatives[j]); pos != std::string::npos) {
                                int j1 = (j + 1) % alternatives.size();
                                auto alternative_name = name.substr(0, pos) + alternatives[j1];
                                it = imatrix_data->find(alternative_name);
                                break;
                            }
                        }
                    }
                }
                if (it == imatrix_data->end()) {
                    LLAMA_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, tensor->name);
                } else {
                    if (it->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
                        imatrix = it->second.data();
                    } else {
                        LLAMA_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                                int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name);

                        // this can happen when quantizing an old mixtral model with split tensors with a new incompatible imatrix
                        // this is a significant error and it may be good idea to abort the process if this happens,
                        // since many people will miss the error and not realize that most of the model is being quantized without an imatrix
                        // tok_embd should be ignored in this case, since it always causes this warning
                        if (name != tn(LLM_TENSOR_TOKEN_EMBD, "weight")) {
                            throw std::runtime_error(format("imatrix size %d is different from tensor size %d for %s",
                                    int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name));
                        }
                    }
                }
            }
            if (!params->ignore_imatrix_rules && !imatrix) {
                bool is_very_low_bpw_quant = new_type == GGML_TYPE_IQ2_XXS    ||
                                             
                                             new_type == GGML_TYPE_IQ2_XS     ||
                                             
                                             new_type == GGML_TYPE_IQ2_S      ||
                                             
                                             new_type == GGML_TYPE_IQ1_S      ||
                                             
                                             new_type == GGML_TYPE_IQ1_M      ||

                                            (new_type == GGML_TYPE_Q2_K && ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S);
                if (is_very_low_bpw_quant && strcmp(tensor->name, "token_embd.weight") && strcmp(tensor->name, "output.weight")) {
                    LLAMA_LOG_ERROR("\n\n============================================================\n");
                    LLAMA_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", tensor->name);
                    LLAMA_LOG_ERROR("The result will be garbage, so bailing out\n");
                    LLAMA_LOG_ERROR("============================================================\n\n");
                    throw std::runtime_error(format("Missing importance matrix for tensor %s in a very low-bit quantization", tensor->name));
                }
            }

            int chunk_size_multiplier = 1;
            auto [working_type, num_rows] = interleaved_properties(new_type);
            if (tensor->ne[1] % num_rows != 0) {
                new_type = working_type;
            } else {
                chunk_size_multiplier = num_rows;
            }

            LLAMA_LOG_INFO("converting to %s .. ", ggml_type_name(new_type));
            fflush(stdout);

            if (params->dry_run) {
                // Project the SAME landing the else-arm below would produce. Without this the
                // dry run reported every PXQ-eligible tensor as its MXFP4 default, which on a
                // MoE model made the composition summary (and the assertion that reads it)
                // describe a file the real run never writes.
                const ggml_type landing = pxa_pxq_landing_type(name, tensor, new_type, pxq_tier);
                if (landing != new_type) {
                    LLAMA_LOG_INFO("[dry-run: PXQ dispatch -> %s] ", ggml_type_name(landing));
                    new_type = landing;
                }
                const size_t slab = pxa_pxq_slab_size(new_type, tensor->ne[0], tensor->ne[1],
                                                      tensor->ne[2]*tensor->ne[3]);
                new_size = slab ? slab
                                : tensor->ne[2] * tensor->ne[1] * ggml_row_size(new_type, tensor->ne[0]);
            } else {
                float * f32_data;

                const bool src_is_pxq = pxa_is_pxq_slab_type(tensor->type);

                // Requantizing from an already-lossy source compounds two independent
                // quantization errors and hands the second codec a distribution that is no
                // longer the model's. Q8_0 is near-lossless so the damage is small; a PXQ tier
                // is 1.26-5.27 bpw and the damage is not. Both are legitimate when the original
                // checkpoint is gone -- they are not legitimate by accident, so they need an
                // explicit statement of intent on the command line.
                if (!params->allow_double_lossy &&
                    (src_is_pxq || tensor->type == GGML_TYPE_Q8_0)) {
                    throw std::runtime_error(format(
                        "refusing to requantize from %s: that is a second lossy pass over an "
                        "already-quantized tensor. Quantize from the original F32/BF16 if you "
                        "still have it; if you do not, pass --i-know-this-is-double-lossy.",
                        ggml_type_name(tensor->type)));
                }

                if (src_is_pxq) {
                    // PXQ is a 64-row PANEL format: a row's bytes are scattered across its
                    // panel, so llama_tensor_dequantize_internal() -- which walks row pointers
                    // through .to_float -- cannot read it, and .to_float must stay NULL for
                    // exactly that reason. The panel-aware decoder in ggml/src/pxq-cpu.c takes
                    // the TENSOR BASE plus a global row index instead, which is the only
                    // addressing that is meaningful here. Experts need no loop: a PXQ tensor is
                    // E*(ne1/64) contiguous panels, so nrows = ne1*ne2*ne3 decodes it whole.
                    if (!pxa_pxq_is_cpu_supported(tensor->type)) {
                        throw std::runtime_error(format(
                            "cannot requantize from %s: no CPU panel dequant for this tier "
                            "(ggml/src/pxq-cpu.c)", ggml_type_name(tensor->type)));
                    }
                    if (f32_conv_buf.size() < nelements) {
                        f32_conv_buf.resize(nelements);
                    }
                    pxa_pxq_dequant_2d(tensor->type, tensor->data, (float *) f32_conv_buf.data(),
                                       tensor->ne[1]*tensor->ne[2]*tensor->ne[3], tensor->ne[0]);
                    f32_data = (float *) f32_conv_buf.data();
                } else if (tensor->type == GGML_TYPE_F32) {
                    f32_data = (float *) tensor->data;
                } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                    throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
                } else {
                    llama_tensor_dequantize_internal(tensor, f32_conv_buf, workers, nelements, nthread);
                    f32_data = (float *) f32_conv_buf.data();
                }

                auto expected_size = ggml_row_size(new_type, tensor->ne[0])*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];

                if (work.size() < expected_size) { //(size_t)nelements * 4) {
                    //work.resize(nelements * 4); // upper bound on size
                    work.resize(expected_size); // upper bound on size
                }
                new_data = work.data();

                if (params->extra_output_type != GGML_TYPE_COUNT && tensor == output_tensor) {
                    auto cur_size = ggml_nbytes(tensor);
                    if (new_type != tensor->type) {
                        do_quantize(nthread, tensor, new_type, f32_data, (char *)new_data, imatrix, workers,
                                new_size, chunk_size_multiplier, params);
                        gguf_set_tensor_type(ctx_outs[cur_split], name.c_str(), new_type);
                        gguf_set_tensor_data(ctx_outs[cur_split], name.c_str(), new_data, new_size);
                        fout.write((const char *) new_data, new_size);
                        zeros(fout, GGML_PAD(new_size, align) - new_size);
                        total_size_new += new_size;
                        LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", cur_size/1024.0/1024.0, new_size/1024.0/1024.0);
                    } else {
                        gguf_set_tensor_type(ctx_outs[cur_split], name.c_str(), tensor->type);
                        gguf_set_tensor_data(ctx_outs[cur_split], name.c_str(), tensor->data, cur_size);
                        fout.write((const char *) tensor->data, cur_size);
                        zeros(fout, GGML_PAD(cur_size, align) - cur_size);
                        total_size_new += cur_size;
                        LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", cur_size/1024.0/1024.0, cur_size/1024.0/1024.0);
                    }

                    LLAMA_LOG_INFO("[%4d/%4d] %36s - [%s], type = %6s, ",
                           ++idx, ml.n_tensors,
                           ggml_get_name(tensor),
                           llama_format_tensor_shape(tensor).c_str(),
                           ggml_type_name(tensor->type));

                    new_type = params->extra_output_type;
                    chunk_size_multiplier = 1;
                    auto [working_type, num_rows] = interleaved_properties(new_type);
                    if (tensor->ne[1] % num_rows != 0) {
                        new_type = working_type;
                    } else {
                        chunk_size_multiplier = num_rows;
                    }
                    LLAMA_LOG_INFO("converting to %s .. ", ggml_type_name(new_type));
                    fflush(stdout);

                    do_quantize(nthread, tensor, new_type, f32_data, (char *)new_data, imatrix, workers,
                        new_size, 1, params);

                    name = extra.name;
                } else if (!pxq4_tensor_geometry_ok(tensor) &&
                           (new_type == GGML_TYPE_PXQ1 || new_type == GGML_TYPE_PXQ2 || new_type == GGML_TYPE_PXQ3 ||
                            new_type == GGML_TYPE_PXQ4 || new_type == GGML_TYPE_PXQ4HQ ||
                            new_type == GGML_TYPE_PXQ6)) {
                    // safety: a PXQ target on a geometry-ineligible tensor (bad custom rule) —
                    // the slab codecs need 64-row panels / 32-wide blocks and there is no CPU
                    // codec to fall to. BACKBONE_REV 2 changed the landing type from mxfp4 to
                    // q8_0: this path only ever fires on rare, small tensors, and MXFP4's two
                    // failure channels (a 3.54-effective-bit codec at 4.25 bpw, plus a
                    // systematic gain bias) make it the one type we never want to fall into
                    // silently. Legacy mode keeps the old mxfp4 landing so it byte-reproduces
                    // pre-rev-2 recipes.
                    // NOTE (2026-07-28): this gate used to be the full CLASS-based
                    // pxq4_tensor_eligible(), which silently demoted an EXPLICIT --custom-q PXQ
                    // target under PXA_PXQ_BACKBONE=legacy — the arm came out byte-identical to
                    // its control. An explicit PXQ target is now honoured whenever the GEOMETRY
                    // allows it, in every backbone mode.
                    const bool bb_legacy = pxa_pxq_backbone_cfg().mode == PXA_BB_LEGACY;
                    const ggml_type demoted = bb_legacy ? GGML_TYPE_MXFP4 : GGML_TYPE_Q8_0;
                    // LOUD-DEMOTE (2026-07-28, owner-requested): a silently demoted EXPLICIT
                    // --custom-q target is how an A/B arm ends up byte-identical to its control
                    // and "measures" nothing. Scream, count, and summarize at the end.
                    if (pxa_custom_rule_matches(params, name)) {
                        ++g_pxa_customq_demoted;
                        LLAMA_LOG_ERROR("\n⚠⚠ CUSTOM-Q DEMOTED: %s — your --custom-q rule targeted %s but the tensor "
                                        "fails PXQ slab geometry (ne0=%" PRId64 ", ne1=%" PRId64 "; needs rows%%64==0, K%%32==0). "
                                        "Landing on %s instead. The output will NOT test your rule for this tensor.\n",
                                        name.c_str(), ggml_type_name(new_type), tensor->ne[0], tensor->ne[1],
                                        ggml_type_name(demoted));
                    }
                    LLAMA_LOG_WARN("%s: %s fails PXQ slab geometry (ne0=%" PRId64 ", ne1=%" PRId64 ") — demoting %s -> %s\n",
                                   __func__, name.c_str(), tensor->ne[0], tensor->ne[1],
                                   ggml_type_name(new_type), ggml_type_name(demoted));
                    new_type = demoted;
                    do_quantize(nthread, tensor, new_type, f32_data, (char *)new_data, imatrix, workers,
                            new_size, chunk_size_multiplier, params);
                } else if (pxq4_tensor_geometry_ok(tensor) &&
                           (new_type == GGML_TYPE_PXQ1 || new_type == GGML_TYPE_PXQ2 || new_type == GGML_TYPE_PXQ3 ||
                            new_type == GGML_TYPE_PXQ4 || new_type == GGML_TYPE_PXQ4HQ ||
                            new_type == GGML_TYPE_PXQ6 ||
                            ((pxq1_out || pxq2_out || pxq3_out || pxq6_out || pxq6hq_out || pxq6r_out) &&
                             new_type == GGML_TYPE_MXFP4 && pxq4_legacy_native_class(name)))) {
                    // native PXQ quantize — per-tensor target wins (custom-q / --pxq-universal);
                    // an untouched MXFP4 default resolves to the whole-file ftype's tier.
                    ggml_type tgt = new_type;
                    if (tgt == GGML_TYPE_MXFP4) {
                        tgt = pxq1_out ? GGML_TYPE_PXQ1 :
                              pxq2_out ? GGML_TYPE_PXQ2 : pxq3_out ? GGML_TYPE_PXQ3 :
                              pxq6r_out ? GGML_TYPE_PXQ6 :
                              pxq6hq_out ? GGML_TYPE_PXQ4HQ : GGML_TYPE_PXQ4;
                    }
                    const int64_t K = tensor->ne[0], R = tensor->ne[1], E = tensor->ne[2]*tensor->ne[3];
                    switch (tgt) {
                        case GGML_TYPE_PXQ1:
                            new_size = (size_t)E*(R/64)*(PXQ1_HDR_BYTES + (K/32)*(int64_t)PXQ1_SLAB_BYTES);
                            if (work.size() < new_size) work.resize(new_size);
                            new_data = work.data();
                            pxq1_quantize_tensor(f32_data, (uint8_t *)new_data, R, K, E,
                                                 imatrix, imatrix ? K*E : 0, nthread);
                            LLAMA_LOG_INFO("PXQ1 native quantize (1-bit sign x E16-row) -> pxq1 .. ");
                            break;
                        case GGML_TYPE_PXQ2:
                            new_size = (size_t)E*(R/64)*(PXQ2_HDR_BYTES + (K/32)*(int64_t)PXQ2_SLAB_BYTES);
                            if (work.size() < new_size) work.resize(new_size);
                            new_data = work.data();
                            pxq2_quantize_tensor(f32_data, (uint8_t *)new_data, R, K, E,
                                                 imatrix, imatrix ? K*E : 0, nthread);
                            LLAMA_LOG_INFO("PXQ2 native quantize (LM4 x E16-row) -> pxq2 .. ");
                            break;
                        case GGML_TYPE_PXQ3:
                            new_size = (size_t)E*(R/64)*(PXQ3_HDR_BYTES + (K/32)*(int64_t)PXQ3_SLAB_BYTES);
                            if (work.size() < new_size) work.resize(new_size);
                            new_data = work.data();
                            pxq3_quantize_tensor(f32_data, (uint8_t *)new_data, R, K, E,
                                                 imatrix, imatrix ? K*E : 0, nthread);
                            LLAMA_LOG_INFO("PXQ3 native quantize (LM8 bit-plane x E16-row) -> pxq3 .. ");
                            break;
                        case GGML_TYPE_PXQ6:
                            new_size = (size_t)E*(R/64)*(PXQ6R_HDR_BYTES + (K/32)*(int64_t)PXQ6R_SLAB_BYTES);
                            if (work.size() < new_size) work.resize(new_size);
                            new_data = work.data();
                            pxq6r_quantize_tensor(f32_data, (uint8_t *)new_data, R, K, E,
                                                  imatrix, imatrix ? K*E : 0, nthread);
                            LLAMA_LOG_INFO("PXQ6 native quantize (LM32 5-bit x E16-row) -> pxq6 .. ");
                            break;
                        default: {   // PXQ4 / PXQ4-HQ (the 4-bit tiers)
                            const int tier = tgt == GGML_TYPE_PXQ4HQ ? 1 : 0;
                            new_size = (size_t)E*(R/64)*(PXQ6_HDR_BYTES + (K/32)*(int64_t)(tier ? PXQ6HQ_SLAB_BYTES : PXQ6_SLAB_BYTES));
                            if (work.size() < new_size) work.resize(new_size);
                            new_data = work.data();
                            pxq6_quantize_tensor(f32_data, (uint8_t *)new_data, R, K, E,
                                                 imatrix, imatrix ? K*E : 0, nthread, tier);
                            LLAMA_LOG_INFO("PXQ4 native quantize (E16-row scales, tier %s) .. ", tier ? "HQ/bs8" : "core/bs16");
                        } break;
                    }
                    // Never let a PXQ tensor out of this branch with an unwritten panel in it.
                    pxa_pxq_check_anchors(tgt, new_data, K, R, E, name);
                    new_type = tgt;
                } else {
                    do_quantize(nthread, tensor, new_type, f32_data, (char *)new_data, imatrix, workers,
                            new_size, chunk_size_multiplier, params);
                }

                if (pxa_errbudget_enabled()) {
                    pxa_errbudget_accumulate(errbudget, name, new_type, new_data, f32_data,
                                             tensor->ne[0], ggml_nelements(tensor)/tensor->ne[0]);
                }
            }
            LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", ggml_nbytes(tensor)/1024.0/1024.0, new_size/1024.0/1024.0);
        }

QuantizationDone:;
        total_size_org += ggml_nbytes(tensor);
        total_size_new += new_size;
        comp_stats[(int) new_type].first  += 1;
        comp_stats[(int) new_type].second += new_size;

        if (!params->dry_run && !split_skipped[cur_split]) {
            // update the gguf meta data as we go
            gguf_set_tensor_type(ctx_outs[cur_split], name.c_str(), new_type);
            gguf_set_tensor_data(ctx_outs[cur_split], name.c_str(), new_data, new_size);

            // write tensor data + padding
            fout.write((const char *) new_data, new_size);
            zeros(fout, GGML_PAD(new_size, align) - new_size);
        }
    }
    close_ofstream();
    for (auto & c:ctx_outs) {
        gguf_free(c);
    }

    LLAMA_LOG_INFO("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);
    if (g_pxa_customq_demoted > 0) {
        LLAMA_LOG_ERROR("\n⚠⚠ %d tensor(s) targeted by --custom-q were DEMOTED off their requested PXQ type "
                        "(slab-geometry failures — see the CUSTOM-Q DEMOTED lines above). "
                        "Dump and diff the tier table before benching this artifact.\n", g_pxa_customq_demoted);
    }

    // ------------------------------------------------------------------------------------------
    // PXQ-P5 composition summary + assertion (2026-07-27). Motivation: a dense (no-expert)
    // model quantized with PXA_PXQ_BACKBONE=legacy under a PXQ4 target emitted a file that was
    // 91% MXFP4 / 0% PXQ, exit 0 — a plausible artifact whose NAME misrepresents its contents.
    // The summary prints EVERY run; the assertion fires only for PXQ-family targets:
    //   FAIL if PXQ-family bytes < 50% of the output (majority floor — the smallest legitimate
    //   shipped artifact, the pre-rev-2 uniform PXQ1 35B, is 70.7% PXQ by bytes), or if a
    //   UNIFORM PXQ target contributed ZERO bytes of its named tier (catches a PXQ6 target
    //   emitting PXQ4HQ — the 122B naming-migration class). On failure every file this run
    //   wrote is removed and the run exits non-zero. Explicit override:
    //   --pxq-composition-override (env PXA_PXQ_COMPOSITION_OVERRIDE=1) downgrades to a WARN.
    // ------------------------------------------------------------------------------------------
    {
        std::vector<std::pair<int, std::pair<int64_t, size_t>>> rows(comp_stats.begin(), comp_stats.end());
        std::sort(rows.begin(), rows.end(), [](const auto & x, const auto & y) { return x.second.second > y.second.second; });
        LLAMA_LOG_INFO("%s: ---- output composition (by bytes) ----\n", __func__);
        for (const auto & r : rows) {
            LLAMA_LOG_INFO("%s:   %-10s %5lld tensors  %9.2f MiB  %5.1f%%\n", __func__,
                    ggml_type_name((ggml_type) r.first), (long long) r.second.first,
                    r.second.second/1024.0/1024.0,
                    total_size_new ? 100.0*r.second.second/total_size_new : 0.0);
        }

        static const std::set<int> pxq_family = { 248, 252, 253, 254, 255, 256 };   // PXQ1,4,4HQ,2,3,6
        int spec_type = -1;          // the single tier a UNIFORM PXQ target names (-1 = none/map)
        bool pxq_target = true;
        switch (params->ftype) {
            case LLAMA_FTYPE_MOSTLY_PXQ1:           spec_type = 248; break;
            case LLAMA_FTYPE_MOSTLY_PXQ2:           spec_type = 254; break;
            case LLAMA_FTYPE_MOSTLY_PXQ3:           spec_type = 255; break;
            case LLAMA_FTYPE_MOSTLY_PXQ4:           spec_type = 252; break;
            case LLAMA_FTYPE_MOSTLY_PXQ4HQ:         spec_type = 253; break;
            case LLAMA_FTYPE_MOSTLY_PXQ6:           spec_type = 256; break;
            case LLAMA_FTYPE_MOSTLY_PXQ_UNIVERSAL:  spec_type = -1;  break;   // map-defined mix
            default: pxq_target = false; break;
        }
        if (pxq_target && total_size_new > 0) {
            size_t fam_bytes = 0, spec_bytes = 0;
            for (const auto & r : comp_stats) {
                if (pxq_family.count(r.first))  fam_bytes  += r.second.second;
                if (r.first == spec_type)       spec_bytes += r.second.second;
            }
            const double fam_share = (double) fam_bytes / (double) total_size_new;
            const bool below_floor = fam_share < 0.50;
            const bool tier_absent = (spec_type >= 0) && (spec_bytes == 0);
            if (below_floor || tier_absent) {
                const bool override_on = getenv("PXA_PXQ_COMPOSITION_OVERRIDE")
                                      && atoi(getenv("PXA_PXQ_COMPOSITION_OVERRIDE")) != 0;
                const pxa_pxq_bb_cfg & bb = pxa_pxq_backbone_cfg();
                char msg[512];
                snprintf(msg, sizeof(msg),
                        "PXQ composition assertion: target %s produced %.1f%% PXQ-family bytes"
                        "%s%s (floor 50%%; backbone=%s). The output would misrepresent its"
                        " contents.",
                        llama_model_ftype_name(params->ftype).c_str(), 100.0*fam_share,
                        tier_absent ? " and ZERO bytes of the named tier " : "",
                        tier_absent ? ggml_type_name((ggml_type) spec_type) : "",
                        bb.mode == PXA_BB_LEGACY ? "legacy" : (bb.lite ? "lite" : (bb.hq ? "hq" : "v2")));
                if (override_on) {
                    LLAMA_LOG_WARN("%s: %s OVERRIDDEN by --pxq-composition-override — file kept.\n",
                            __func__, msg);
                } else {
                    for (const auto & f : comp_written_files) {
                        if (std::remove(f.c_str()) == 0) {
                            LLAMA_LOG_ERROR("%s: removed mislabelled output %s\n", __func__, f.c_str());
                        }
                    }
                    throw std::runtime_error(msg);
                }
            }
        }
    }

    if (qs.n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n",
                __func__, qs.n_fallback, qs.n_k_quantized + qs.n_fallback);
    }

    if (const int64_t dead = g_pxq_imx_dead_cols.load(); dead > 0) {
        LLAMA_LOG_WARN("%s: imatrix: %lld expert column(s) had no usable weights (all-zero / "
                       "non-finite) and were fit UNWEIGHTED — normal for routed experts that "
                       "never fired during calibration; a large count means the corpus is too "
                       "thin for this model's expert count\n", __func__, (long long) dead);
    }

    if (pxa_errbudget_enabled()) {
        pxa_errbudget_report(errbudget, fname_out);
    }
}

uint32_t llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params) {
    try {
        llama_model_quantize_internal(fname_inp, fname_out, params);
        return 0;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }
}

