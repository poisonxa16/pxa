#include "llama-model-loader.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "ggml.h"


#include <set>
#include <map>
#include <array>
#include <future>
#include <regex>
#include <unordered_set>

#define LLAMA_API_INTERNAL

struct create_tensors_helper : public create_tensors_helper_interface {

    create_tensors_helper(llama_model_loader & ml, llama_model & model);
    ~create_tensors_helper() = default;

    //virtual std::map<ggml_backend_buffer_type_t, int> & buft_layer_count_map() override {
    //    return buft_layer_count;
    //}

    virtual std::map<ggml_backend_buffer_type_t, ggml_context *> & get_ctx_map() override {
        return ctx_map;
    }

    virtual size_t get_ctx_size() const override { return ctx_size; }

    bool merge_qkv(const LLM_TN & tn, int i, int bias, bool ignore_attn_scale = false);

    bool merge_up_gate_exps(const LLM_TN & tn, int i, int bias);

    bool create_std_ffn_exps(int64_t n_embd, const LLM_TN & tn, int i, int flags = 0, int n_ff_exps_input = 0,
            ggml_context * ffn_ctx = nullptr);

    bool create_tensors() override;

    bool create_llama_tensors(const LLM_TN & tn);

    bool create_deci_tensors(const LLM_TN & tn);

    bool create_llama4_tensors(const LLM_TN & tn);

    bool create_grok_tensors(const LLM_TN & tn);

    bool create_dbrx_tensors(const LLM_TN & tn);

    bool create_baichuan_tensors(const LLM_TN & tn, bool with_ffn_norm = true);

    bool create_falcon_tensors(const LLM_TN & tn);

    bool create_starcoder_tensors(const LLM_TN & tn);

    bool create_bert_tensors(const LLM_TN & tn);

    bool create_jina_bert2_tensors(const LLM_TN & tn);

    bool create_bloom_tensors(const LLM_TN & tn);

    bool create_mpt_tensors(const LLM_TN & tn);

    bool create_stablelm_tensors(const LLM_TN & tn);

    bool create_qwen_tensors(const LLM_TN & tn);

    bool create_qwen2_tensors(const LLM_TN & tn);

    bool create_qwen2_moe_tensors(const LLM_TN & tn);

    bool create_qwen3_tensors(const LLM_TN & tn);

    bool create_qwen3_moe_tensors(const LLM_TN & tn);

    bool create_mellum_tensors(const LLM_TN & tn);

    bool create_qwen3next_tensors(const LLM_TN & tn);

    bool create_qwen35moe_tensors(const LLM_TN & tn);

    bool create_qwen35_tensors(const LLM_TN & tn);

    bool create_qwen4exp_tensors(const LLM_TN & tn);

    bool create_phi2_tensors(const LLM_TN & tn);

    bool create_phi3_tensors(const LLM_TN & tn);

    bool create_gpt2_tensors(const LLM_TN & tn);

    bool create_codeshell_tensors(const LLM_TN & tn);

    bool create_orion_tensors(const LLM_TN & tn);

    bool create_internlm_tensors(const LLM_TN & tn);

    bool create_gemma_tensors(const LLM_TN & tn, int version);

    bool create_gemma4_tensors(const LLM_TN & tn);

    bool create_gemma4_mtp_tensors(const LLM_TN & tn);

    bool create_starcoder2_tensors(const LLM_TN & tn);

    bool create_mamba_tensors(const LLM_TN & tn);

    bool create_xverse_tensors(const LLM_TN & tn);

    bool create_command_r_tensors(const LLM_TN & tn);

    bool create_olmo_tensors(const LLM_TN & tn);

    bool create_openelm_tensors(const LLM_TN & tn);

    bool create_gptneox_tensors(const LLM_TN & tn);

    bool create_arctix_tensors(const LLM_TN & tn);

    bool create_deepseek2_tensors(const LLM_TN & tn);

    bool create_glm_dsa_tensors(const LLM_TN & tn);

    bool create_deepseek4_tensors(const LLM_TN & tn);
    bool create_dspark_tensors(const LLM_TN & tn);

    bool create_glm4_tensors(const LLM_TN & tn);

    bool create_glm4_moe_tensors(const LLM_TN & tn);

    bool create_bitnet_tensors(const LLM_TN & tn);

    bool create_bitnet2_tensors(const LLM_TN & tn);

    bool create_t5_tensors(const LLM_TN & tn);

    bool create_tsencoder_tensors(const LLM_TN & tn);

    bool create_jais_tensors(const LLM_TN & tn);

    bool create_chatglm_tensors(const LLM_TN & tn);

    bool create_cohere2_tensors(const LLM_TN & tn);
    bool create_cohere2_moe_tensors(const LLM_TN & tn);

    bool create_dots1_tensors(const LLM_TN & tn);

    bool create_ernie45_tensors(const LLM_TN & tn);

    bool create_hunyuan_tensors(const LLM_TN & tn);

    bool create_openai_moe_tensors(const LLM_TN & tn);

    bool create_bailingmoe2_tensors(const LLM_TN & tn);

    bool create_minimaxm2_tensors(const LLM_TN & tn);

    bool create_smollm3_tensors(const LLM_TN & tn);

    bool create_mimo2_tensors(const LLM_TN & tn);

    bool create_seedoss_tensors(const LLM_TN & tn);

    bool create_step35_tensors(const LLM_TN & tn);

    bool create_laguna_tensors(const LLM_TN & tn);
    bool create_muse_glimmer_tensors(const LLM_TN & tn);

    bool create_hy3_tensors(const LLM_TN & tn);

    llama_model_loader & ml;
    llama_model        & model;

    ggml_tensor * create_tensor(ggml_context * ctx, const std::string & name, const std::vector<int64_t> & ne, int flags = 0,
            ggml_context ** actual_ctx = nullptr);

    ggml_context * get_context_for_tensor(ggml_context * ctx, const std::string & name);

    void create_default_embd_output(const LLM_TN & tn, int n_embd, int n_vocab, bool norm_bias);
    void create_embd_output(const LLM_TN & tn, int n_embd, int n_vocab, bool has_norm = true, bool use_ctx_split = false);

    void create_std_attn(int i, const LLM_TN & tn, llama_layer & layer, int n_embd, int n_embd_gqa, ggml_context * ctx_split);
    void create_std_ffn(int i, const LLM_TN & tn, llama_layer & layer, int n_ff, int n_embd, ggml_context * ctx_split);

    inline ggml_context * ctx_for_layer(int i) const {
        return ctx_map.at(model.buft_layer[i].buft);
    }
    inline ggml_context * ctx_for_layer_split(int i) const {
        // MTP-SPLIT: the MTP "nextn" tail layer is plain attention + MoE, so split it
        // tensor-parallel across GPUs exactly like a normal transformer layer (use buft_matrix).
        // Previously the tail layer was forced to the non-split buft (single-device island),
        // which OOM'd the last GPU under -sm attn. The split attn/MoE builders fall back to
        // single-device automatically when the tensors lack ->extra, so this is inert for
        // single-GPU / -sm layer / non-MTP models.
        return ctx_map.at(model.buft_layer[i].buft_matrix);
    }

    std::map<ggml_backend_buffer_type_t, int> buft_layer_count;
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    ggml_context * split_ctx = nullptr;
    size_t ctx_size;

    ggml_context * ctx_input;
    ggml_context * ctx_output;
    ggml_context * ctx_output_split;

    ggml_backend_buffer_type_t default_cpu_buft;
    bool has_buft_overrides = false;

    std::unordered_set<ggml_tensor *> split_tensors;

    std::vector<std::pair<std::regex, ggml_backend_buffer_type_t>> overrides;

    inline ggml_context * ctx_for_buft(ggml_backend_buffer_type_t buft) {
        if (auto it = ctx_map.find(buft); it != ctx_map.end()) return it->second;

        ggml_init_params params = { /*.mem_size   =*/ ctx_size, /*.mem_buffer =*/ NULL, /*.no_alloc   =*/ true, };

        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            throw std::runtime_error(format("failed to create ggml context"));
        }

        ctx_map[buft] = ctx;
        model.ctxs.emplace_back(ctx);

        return ctx;

    }
};

create_tensors_helper::create_tensors_helper(llama_model_loader & _ml, llama_model & _model) : ml(_ml), model(_model) {

    const int n_layer = model.hparams.n_layer;
    buft_layer_count[model.buft_input.buft]++;
    buft_layer_count[model.buft_input.buft_matrix]++;
    buft_layer_count[model.buft_output.buft]++;
    buft_layer_count[model.buft_output.buft_matrix]++;
    for (int i = 0; i < n_layer; ++i) {
        buft_layer_count[model.buft_layer[i].buft]++;
        buft_layer_count[model.buft_layer[i].buft_matrix]++;
    }

    default_cpu_buft = llama_default_buffer_type_cpu(true);

    if (ml.tensor_buft_overrides) {
        for (const auto * o = ml.tensor_buft_overrides; o->pattern != nullptr; ++o) {
            auto buft = o->buft;
            if (ggml_backend_buft_is_host(buft)) buft = default_cpu_buft;
            overrides.emplace_back(std::make_pair(std::regex(o->pattern), buft));
        }
    }

    if (ml.ncmoe > 0) {
        auto buft = llama_default_buffer_type_cpu(true);
        if (model.split_mode == LLAMA_SPLIT_MODE_ATTN || model.split_mode == LLAMA_SPLIT_MODE_GRAPH || ml.ncmoe >= n_layer || model.devices.size() < 2) {
            int nmax = std::min(ml.ncmoe, n_layer);
            for (int i = 0; i < nmax; ++i) {
                std::string pattern = "blk\\." + std::to_string(i) + "\\.ffn_(up|down|gate|gate_up)_exps\\.(weight|scale)";
                this->overrides.emplace_back(std::make_pair(std::regex(pattern), buft));
            }
        }
        else if (model.split_mode == LLAMA_SPLIT_MODE_LAYER) {
            std::vector<int> counts(model.devices.size(), 0);
            int nbad = 0;
            for (int i = 0; i < n_layer; ++i) {
                if (model.default_layer_device[i] >= 0 && model.default_layer_device[i] < (int)model.devices.size()) {
                    ++counts[model.default_layer_device[i]];
                } else {
                    LLAMA_LOG_WARN("%s: default device for layer %d is %d?\n", __func__, i, model.default_layer_device[i]);
                    ++nbad;
                }
            }
            if (nbad > 0) {
                throw std::runtime_error("Unexpected device configuration");
            }
            std::vector<int> n_override(counts.size());
            printf("================= %s: split mode layer with ncmoe = %d, %d devices\n", __func__, ml.ncmoe, (int)model.devices.size());
            int ntot = 0;
            for (int i = 0; i < int(counts.size()); ++i) {
                float fraction = 1.f*counts[i]/n_layer;
                n_override[i] = std::roundf(fraction*ml.ncmoe);
                ntot += n_override[i];
            }
            while (ntot > ml.ncmoe) {
                float best_err = -1e30; int ibest = -1;
                for (int i = 0; i < int(counts.size()); ++i) {
                    if (n_override[i] == 0) continue;
                    float n_want = 1.f*counts[i]*ml.ncmoe/n_layer;
                    float err = n_override[i] - 1 - n_want;
                    if (err > best_err) {
                        best_err = err; ibest = i;
                    }
                }
                if (ibest < 0) { // shouldn't happen
                    break;
                }
                --n_override[ibest];
                --ntot;
            }
            while (ntot < ml.ncmoe) {
                float best_err = 1e30; int ibest = -1;
                for (int i = 0; i < int(counts.size()); ++i) {
                    if (n_override[i] >= counts[i]) continue;
                    float n_want = 1.f*counts[i]*ml.ncmoe/n_layer;
                    float err = n_override[i] + 1 - n_want;
                    if (err < best_err) {
                        best_err = err; ibest = i;
                    }
                }
                if (ibest < 0) { // shouldn't happen
                    break;
                }
                ++n_override[ibest];
                ++ntot;
            }
            for (int i = 0; i < int(counts.size()); ++i) {
                printf("    device %d: %d layers -> %d overrides\n", i, counts[i], n_override[i]);
            }
            // it is better to go backwards to avoid (or at least reduce) issues when there are layers without MoE tensors
            for (int i = n_layer-1; i >= 0; --i) {
                int id = model.default_layer_device[i];
                if (n_override[id] > 0) {
                    std::string pattern = "blk\\." + std::to_string(i) + "\\.(ffn_(up|down|gate|gate_up)_exps\\.weight)";
                    printf("Adding override %s=%s\n", pattern.c_str(), ggml_backend_buft_name(buft));
                    this->overrides.emplace_back(std::make_pair(std::regex(pattern), buft));
                    --n_override[id];
                }
            }
        }
    }

    // Split MTP layer's to graph
    if ((model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN) &&
            model.hparams.nextn_predict_layers > 0 && model.splits.size() > 1) {
        [[maybe_unused]] int mtp_first = n_layer - model.hparams.nextn_predict_layers;
        LLAMA_LOG_DEBUG("%s: MTP layer(s) %d-%d: split attention+FFN, nextn on per-device CUDA\n",
                __func__, mtp_first, n_layer - 1);
    }

    auto n_tensors = ml.n_tensors;
    if (ml.merge_qkv) n_tensors += n_layer;
    if (ml.merge_up_gate_exps) n_tensors += n_layer;
    ctx_size = ggml_tensor_overhead()*(n_tensors + 1); // +1 for models where tok_embd is duplicated as output
    ctx_size += ggml_tensor_overhead()*n_layer*3;         // for moe merged tensors

    if (model.splits.size() > 1) {
        ctx_size += ggml_tensor_overhead()*n_layer*4;    // for KV cache
        ctx_size *= (model.splits.size() + 1);
    }

    for (auto & it : buft_layer_count) {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            throw std::runtime_error(format("failed to create context"));
        }
        ctx_map[it.first] = ctx;
        model.ctxs.push_back(ctx);
    }
    if (model.split_buft) {
        if (auto it = ctx_map.find(model.split_buft); it != ctx_map.end()) {
            split_ctx = it->second;
        }
    }
}

static std::vector<int> create_split(int nr, int granularity, const std::vector<float> & splits, const std::vector<size_t> & mem_used,
        bool verbose = false) {
    GGML_ASSERT(nr % granularity == 0);
    GGML_ASSERT(!splits.empty());
    if (granularity < 0) return std::vector<int>(splits.size(), nr);
    GGML_ASSERT(mem_used.size() == splits.size());
    size_t tot_memory_used = 1;
    for (auto & mem : mem_used) tot_memory_used += mem;
    int nchunk = nr / granularity;
    std::vector<int> result(splits.size());
    float last_split = 0;
    int sum = 0;
    if (verbose) LLAMA_LOG_INFO("--- %s: %d chunks\n", __func__, nchunk);
    for (int i = 0; i < (int)splits.size(); ++i) {
        float p = splits[i] - last_split;
        float p0 = p;
        p += (p - 1.f*mem_used[i]/tot_memory_used);
        result[i] = roundf(p*nchunk);
        if (result[i] < 0) result[i] = 0;
        if (verbose) LLAMA_LOG_INFO("i = %d, p0 = %g, p = %g, result = %d\n", i, p0, p, result[i]);
        sum += result[i];
        last_split = splits[i];
    }
    while (sum > nchunk) {
        last_split = 0;
        float best_err = -INFINITY;
        int ibest = -1;
        for (int i = 0; i < (int)splits.size(); ++i) {
            if (result[i] > 0) {
                float p = splits[i] - last_split;
                p += (p - 1.f*mem_used[i]/tot_memory_used);
                float n_want = p*nchunk;
                float err = result[i] - n_want;
                if (err > best_err) {
                    best_err = err; ibest = i;
                }
            }
            last_split = splits[i];
        }
        GGML_ASSERT(ibest >= 0 && result[ibest] > 0);
        --result[ibest];
        --sum;
    }
    while (sum < nchunk) {
        last_split = 0;
        float best_err = -INFINITY;
        int ibest = -1;
        for (int i = 0; i < (int)splits.size(); ++i) {
            float p = splits[i] - last_split;
            p += (p - 1.f*mem_used[i]/tot_memory_used);
            float n_want = p*nchunk;
            float err = n_want - result[i];
            if (err > best_err) {
                best_err = err; ibest = i;
            }
            last_split = splits[i];
        }
        GGML_ASSERT(ibest >= 0);
        ++result[ibest];
        ++sum;
    }
    for (auto & r : result) r *= granularity;
    return result;
}

ggml_context * create_tensors_helper::get_context_for_tensor(ggml_context * ctx, const std::string & name) {
    for (auto & o : overrides) {
        if (std::regex_search(name, o.first)) {
            if (o.second == default_cpu_buft) has_buft_overrides = true;
            const struct ggml_tensor * cur = ml.get_tensor_meta(name.c_str());
            const size_t nbytes = cur ? ggml_nbytes(cur) : 0;
            LLAMA_LOG_INFO("Tensor %s (size = %.2f MiB) buffer type overriden to %s\n", name.c_str(), nbytes/1024./1024., ggml_backend_buft_name(o.second));
            model.override_tensor_names.push_back(name);
            ctx = ctx_for_buft(o.second);
            break;
        }
    }
    //if (ml.tensor_buft_overrides) {
    //    for (const auto * overrides = ml.tensor_buft_overrides; overrides->pattern != nullptr; ++overrides) {
    //        std::regex pattern(overrides->pattern);
    //        if (std::regex_search(name, pattern)) {
    //            const struct ggml_tensor * cur = ml.get_tensor_meta(name.c_str());
    //            const size_t nbytes = cur ? ggml_nbytes(cur) : 0;
    //            LLAMA_LOG_INFO("Tensor %s (size = %.2f MiB) buffer type overriden to %s\n", name.c_str(), nbytes/1024./1024., ggml_backend_buft_name(overrides->buft));
    //            ctx = ctx_for_buft(overrides->buft);
    //            break;
    //        }
    //    }
    //}
    return ctx;
}

ggml_tensor * create_tensors_helper::create_tensor(ggml_context * ctx, const std::string & name, const std::vector<int64_t> & ne,
        int flags, ggml_context ** actual_context) {
    ctx = get_context_for_tensor(ctx, name);
    if (actual_context) *actual_context = ctx;
    auto tensor = ml.create_tensor(ctx, name, ne, flags);
    if (tensor && ctx == split_ctx) {
        split_tensors.insert(tensor);
    }
    return tensor;
}

#define LOADING_PRELUDE \
        [[maybe_unused]] const auto & hparams = model.hparams; \
        [[maybe_unused]] const int64_t n_layer       = hparams.n_layer; \
        [[maybe_unused]] const int64_t n_head        = hparams.n_head(); \
        [[maybe_unused]] const int64_t n_head_kv     = hparams.n_head_kv(); \
        [[maybe_unused]] const int64_t n_embd        = hparams.n_embd / (hparams.n_deepstack_layers + 1); /* For Qwen3-VL we need to divide by the number of deepstack layers + 1, for other models n_deepstack_layers value is 0 by default */ \
        [[maybe_unused]] const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa(); \
        [[maybe_unused]] const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa(); \
        [[maybe_unused]] const int64_t n_embd_head_k = hparams.n_embd_head_k(0); \
        [[maybe_unused]] const int64_t n_embd_head_v = hparams.n_embd_head_v(0); \
        [[maybe_unused]] const int64_t n_ff          = hparams.n_ff(); \
        [[maybe_unused]] const int64_t n_embd_gqa    = n_embd_v_gqa; \
        [[maybe_unused]] const int64_t n_vocab       = hparams.n_vocab; \
        [[maybe_unused]] const int64_t n_vocab_type  = hparams.n_vocab_type; \
        [[maybe_unused]] const int64_t n_rot         = hparams.n_rot; \
        [[maybe_unused]] const int64_t n_expert      = hparams.n_expert; \
        [[maybe_unused]] const int64_t n_expert_used = hparams.n_expert_used; \
        [[maybe_unused]] const int64_t n_ctx_train   = hparams.n_ctx_train; \
        if (n_expert > 0 && hparams.n_expert_used == 0) { \
            throw std::runtime_error("model has expert layers but no expert layers are used"); \
        } \
        ctx_input        = ctx_map.at(model.buft_input.buft); \
        ctx_output       = ctx_map.at(model.buft_output.buft); \
        ctx_output_split = ctx_map.at(model.buft_output.buft_matrix); \
        model.layers.resize(n_layer);\
        bool use_mmap_buffer = true;


void create_tensors_helper::create_embd_output(const LLM_TN & tn, int n_embd, int n_vocab, bool has_norm, bool use_ctx_split) {
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    auto out_ctx = use_ctx_split ? ctx_output_split : ctx_output;
    if (has_norm) {
        model.output_norm = create_tensor(out_ctx, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
    }
    model.output = create_tensor(out_ctx, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (model.output == NULL) {
        model.output = create_tensor(out_ctx, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
    }
}

void create_tensors_helper::create_std_attn(int i, const LLM_TN & tn, llama_layer & layer, int n_embd, int n_embd_gqa, ggml_context * ctx_split) {
    layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
    layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
    layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
    layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
}

void create_tensors_helper::create_std_ffn(int i, const LLM_TN & tn, llama_layer & layer, int n_ff, int n_embd, ggml_context * ctx_split) {
    layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
    layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
    layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
}

bool create_tensors_helper::create_llama_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    create_embd_output(tn, n_embd, n_vocab, true);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        use_mmap_buffer &= !merge_qkv(tn, i, 1);

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});

        // optional bias tensors
        layer.bo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},     llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(model.split_mode == LLAMA_SPLIT_MODE_GRAPH ? ctx_split : ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        layer.rope_freqs = create_tensor(ctx_split, tn(LLM_TENSOR_ROPE_FREQS, "weight"), {n_embd/n_head/2}, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));

        if (n_expert == 0) {
            create_std_ffn(i, tn, layer, n_ff, n_embd, model.split_mode == LLAMA_SPLIT_MODE_GRAPH ? ctx_split : ctx_layer);

            // optional MLP bias
            layer.ffn_gate_b = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ffn_down_b = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ffn_up_b   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        } else {
            layer.ffn_gate_inp = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert});

            layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED);
            if (layer.ffn_gate_exps) {
                layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert});
                layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert});
            } else {
                // merge split expert into a single tensor for compatibility with older models
                // requires disabling mmap
                use_mmap_buffer = false;

                ggml_type type_gate = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_GATE_EXP, "weight", i, 0).c_str())->type;
                ggml_type type_down = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_DOWN_EXP, "weight", i, 0).c_str())->type;
                ggml_type type_up   = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_UP_EXP,   "weight", i, 0).c_str())->type;

                layer.ffn_gate_exps = ggml_new_tensor_3d(ctx_split, type_gate, n_embd,   n_ff, n_expert);
                layer.ffn_down_exps = ggml_new_tensor_3d(ctx_split, type_down,   n_ff, n_embd, n_expert);
                layer.ffn_up_exps   = ggml_new_tensor_3d(ctx_split, type_up,   n_embd,   n_ff, n_expert);

                ggml_set_name(layer.ffn_gate_exps, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i).c_str());
                ggml_set_name(layer.ffn_down_exps, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i).c_str());
                ggml_set_name(layer.ffn_up_exps,   tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i).c_str());

                for (uint32_t x = 0; x < n_expert; ++x) {
                    // the individual experts are loaded into a view of the merged tensor
                    ml.create_tensor_as_view(ctx_split, layer.ffn_gate_exps, tn(LLM_TENSOR_FFN_GATE_EXP, "weight", i, x), { n_embd, n_ff }, layer.ffn_gate_exps->nb[2]*x);
                    ml.create_tensor_as_view(ctx_split, layer.ffn_down_exps, tn(LLM_TENSOR_FFN_DOWN_EXP, "weight", i, x), { n_ff, n_embd }, layer.ffn_down_exps->nb[2]*x);
                    ml.create_tensor_as_view(ctx_split, layer.ffn_up_exps,   tn(LLM_TENSOR_FFN_UP_EXP,   "weight", i, x), { n_embd, n_ff }, layer.ffn_up_exps->nb[2]*x);
                }
            }
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_deci_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];
        const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa(i);
        const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa(i);
        const int64_t n_embd_gqa    = hparams.n_embd_v_gqa(i);
        const int64_t n_ff          = hparams.n_ff(i);
        const int64_t n_head        = hparams.n_head(i);
        const int64_t n_head_kv     = hparams.n_head_kv(i);

        if (n_head_kv == 0 && n_head > 0) {
            // linear attention for DeciLMCausalModel
            layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
            layer.wo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        }
        else if (n_head_kv > 0) {
            layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

            layer.wq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head});
            layer.wk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
            layer.wv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
            layer.wo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});
        }

        // optional bias tensors


        layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},     llama_model_loader::TENSOR_NOT_REQUIRED);
        if (n_ff > 0) {
            layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        }

        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long  = create_tensor(ctx_layer, tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight"), { n_rot/2 }, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));
            layer.rope_short = create_tensor(ctx_layer, tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight"), { n_rot/2 }, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));
        }
        else {
            layer.rope_freqs = create_tensor(ctx_layer, tn(LLM_TENSOR_ROPE_FREQS, "weight"), {n_rot/2}, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));
        }

        if (n_ff > 0) {
            create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
        }

        // optional MLP bias
        layer.ffn_gate_b = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down_b = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_up_b   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_llama4_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    create_embd_output(tn, n_embd, n_vocab);

    GGML_ASSERT(hparams.n_moe_layer_step > 0 && "Llama 4 requires n_moe_layer_step > 0");
    for (int i = 0; i < n_layer; ++i) {
        bool is_moe_layer = (i + 1) % hparams.n_moe_layer_step == 0;
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        use_mmap_buffer &= !merge_qkv(tn, i, 0);

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.rope_freqs = create_tensor(ctx_layer, tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2},
                llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));

        if (is_moe_layer) {
            int n_ff_exp = hparams.n_ff_exp;

            layer.ffn_gate_inp  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
            use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i);

            // Shared expert
            const int64_t n_ff_shexp = n_ff_exp;
            layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {    n_embd, n_ff_shexp}, 0);
            layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd    }, 0);
            layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {    n_embd, n_ff_shexp}, 0);
        } else {
            create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_grok_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    if (n_expert == 0) {
        throw std::runtime_error("Grok model cannot have zero experts");
    }

    create_embd_output(tn, n_embd, n_vocab);

    const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff/* / n_expert_used*/; // grok-1 n_ff_exp == n_ff
    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.attn_out_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff,   n_embd }, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_up = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_gate_inp  = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert});
        layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED);

        if (layer.ffn_gate_exps) {
            layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert});
            layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), { n_embd,   n_ff_exp, n_expert });
        } else {
            // merge split expert into a single tensor for compatibility with older models
            // requires disabling mmap
            use_mmap_buffer = false;

            ggml_type type_gate = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_GATE_EXP, "weight", i, 0).c_str())->type;
            ggml_type type_down = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_DOWN_EXP, "weight", i, 0).c_str())->type;
            ggml_type type_up   = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_UP_EXP,   "weight", i, 0).c_str())->type;

            layer.ffn_gate_exps = ggml_new_tensor_3d(ctx_split, type_gate, n_embd,   n_ff, n_expert);
            layer.ffn_down_exps = ggml_new_tensor_3d(ctx_split, type_down,   n_ff, n_embd, n_expert);
            layer.ffn_up_exps   = ggml_new_tensor_3d(ctx_split, type_up,   n_embd,   n_ff, n_expert);

            ggml_set_name(layer.ffn_gate_exps, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i).c_str());
            ggml_set_name(layer.ffn_down_exps, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i).c_str());
            ggml_set_name(layer.ffn_up_exps,   tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i).c_str());

            for (uint32_t x = 0; x < n_expert; ++x) {
                // the individual experts are loaded into a view of the merged tensor
                ml.create_tensor_as_view(ctx_split, layer.ffn_gate_exps, tn(LLM_TENSOR_FFN_GATE_EXP, "weight", i, x), { n_embd, n_ff }, layer.ffn_gate_exps->nb[2]*x);
                ml.create_tensor_as_view(ctx_split, layer.ffn_down_exps, tn(LLM_TENSOR_FFN_DOWN_EXP, "weight", i, x), { n_ff, n_embd }, layer.ffn_down_exps->nb[2]*x);
                ml.create_tensor_as_view(ctx_split, layer.ffn_up_exps,   tn(LLM_TENSOR_FFN_UP_EXP,   "weight", i, x), { n_embd, n_ff }, layer.ffn_up_exps->nb[2]*x);
            }
        }

        layer.ffn_post_norm = create_tensor(ctx_layer,tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), { n_embd }, llama_model_loader::TENSOR_NOT_REQUIRED);
        if (!layer.ffn_post_norm) {
            layer.ffn_post_norm = create_tensor(ctx_layer,tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), { n_embd }, 0);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_dbrx_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    if (n_expert == 0) {
        throw std::runtime_error("DBRX model cannot have zero experts");
    }

    create_default_embd_output(tn, n_embd, n_vocab, false);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.attn_out_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd});

        layer.ffn_gate_inp  = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert});
        use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_baichuan_tensors(const LLM_TN & tn, bool with_ffn_norm) {
    LOADING_PRELUDE
    create_default_embd_output(tn, n_embd, n_vocab, false);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        if (with_ffn_norm) {
            layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        }

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_falcon_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});

        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        if (!model.output) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED); // needs to be on GPU
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

        layer.attn_norm_2   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.attn_norm_2_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM_2, "bias", i),   {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_starcoder_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});
    model.pos_embd = create_tensor(ctx_input, tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train});

    // output
    {
        model.output_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
        model.output        = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        if (!model.output) {
            // needs to be on GPU
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }

    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa});

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP, "weight", i),   {n_embd, n_ff});
        layer.ffn_up_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP, "bias", i),     {n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_bert_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd     = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab});
    model.type_embd    = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_TYPES, "weight"), {n_embd, n_vocab_type});

    if (model.arch == LLM_ARCH_BERT) {
        model.pos_embd = create_tensor(ctx_input, tn(LLM_TENSOR_POS_EMBD,    "weight"), {n_embd, n_ctx_train});
    }

    model.tok_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd});
    model.tok_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd});

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        if (model.arch == LLM_ARCH_BERT) {
            layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
            layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i),   {n_embd});

            layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
            layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i),   {n_embd_gqa});

            layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
            layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i),   {n_embd_gqa});
        } else {
            layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        }

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {n_embd, n_embd});

        layer.attn_out_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd});
        layer.attn_out_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT_NORM, "bias", i),   {n_embd});

        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,        "weight", i), {n_embd, n_ff});
        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN,      "weight", i), {n_ff, n_embd});

        if (model.arch == LLM_ARCH_BERT) {
            layer.bo         = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd});
            layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff});
            layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd});
        } else {
            layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff});
        }

        layer.layer_out_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd});
        layer.layer_out_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_LAYER_OUT_NORM, "bias", i),   {n_embd});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_jina_bert2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd  = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}); // word_embeddings
    model.type_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_TYPES, "weight"), {n_embd, n_vocab_type}); // token_type_embeddings

    model.tok_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd}); // LayerNorm
    model.tok_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd}); //LayerNorm bias

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i]; // JinaBertLayer

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd});
        layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q, "bias", i),   {n_embd});

        layer.attn_q_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.attn_q_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "bias",   i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa});
        layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K, "bias",   i), {n_embd_gqa});

        layer.attn_k_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.attn_k_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "bias",   i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa});
        layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V, "bias",   i), {n_embd_gqa});

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}); //output_dens
        layer.bo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd}); //output_dens

        layer.attn_out_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}); //output_norm
        layer.attn_out_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT_NORM, "bias",   i), {n_embd});

        layer.attn_norm_2   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.attn_norm_2_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM_2, "bias",   i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
        layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd});

        layer.layer_out_norm   = create_tensor(ctx_split, tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd});
        layer.layer_out_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_LAYER_OUT_NORM, "bias",   i), {n_embd});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_bloom_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd   = create_tensor(ctx_input,  tn(LLM_TENSOR_TOKEN_EMBD,      "weight"), {n_embd, n_vocab});
    model.tok_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd});
    model.tok_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd});

    // output
    {
        model.output_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
        model.output        = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias",   i), {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias",   i), {n_embd + 2*n_embd_gqa});

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias",   i), {n_embd});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd});

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, n_ff});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP, "bias",   i), {n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_mpt_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});
    model.pos_embd = create_tensor(ctx_input, tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train}, llama_model_loader::TENSOR_NOT_REQUIRED);

    // output
    {
        model.output_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        model.output        = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        if (!model.output) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED); // needs to be on GPU
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.attn_q_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.attn_q_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "bias",   i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.attn_k_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.attn_k_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "bias",   i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        // AWQ ScaleActivation layer
        layer.ffn_act = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_ACT, "scales", i), {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_stablelm_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_default_embd_output(tn, n_embd, n_vocab, true);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm =   create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i), {n_embd});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        // optional bias tensors, present in Stable LM 2 1.6B
        layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);

        // optional q and k layernorms, present in StableLM 2 12B
        layer.attn_q_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k, n_head},    llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.attn_k_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k, n_head_kv}, llama_model_loader::TENSOR_NOT_REQUIRED);

        // optional FFN norm, not present in StableLM 2 12B which uses parallel residual
        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_seedoss_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t n_qo_dim              = n_head * n_embd_head_k;
    const int64_t n_kv_dim              = n_head_kv * n_embd_head_k;

     model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_qo_dim});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_kv_dim});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_kv_dim});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_qo_dim, n_embd});

        // optional bias tensors
        layer.bq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_qo_dim}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_kv_dim}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_kv_dim}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.attn_post_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd});
        layer.ffn_norm = layer.attn_post_norm;

        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_step35_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    // output
    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);
    // STEP35 supports per-layer partial RoPE dims; rope factors are stored as a single shared tensor
    // ("rope_freqs.weight") and ggml uses only the first (n_rot_l/2) entries per layer.
    uint32_t n_rot_max = 0;
    for (int i = 0; i < n_layer; ++i) {
        n_rot_max = std::max(n_rot_max, hparams.rope_n_rot(i));
    }
    if (n_rot_max == 0) {
        n_rot_max = n_rot;
    }
    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_split = ctx_for_layer_split(i);
        auto & layer = model.layers[i];
        const uint32_t n_head_l      = hparams.n_head(i);
        layer.attn_norm   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, llama_model_loader::TENSOR_NOT_REQUIRED);
        // optional rope factors (llama3) / longrope tensors
        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long  = create_tensor(ctx_split, tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot_max/2}, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));
            layer.rope_short = create_tensor(ctx_split, tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot_max/2}, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));
        } else {
            layer.rope_freqs = create_tensor(ctx_split, tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot_max/2}, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));
        }
        use_mmap_buffer &= !merge_qkv(tn, i, 0);
        //layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head_l}, 0);
        //layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        //layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_v * n_head_l, n_embd}, 0);
        // head-wise attention gate (Step35 self_attn.g_proj)
        layer.wqkv_gate = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head_l}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        // dense MLP (leading dense blocks)
        layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        // MoE routed experts + selection bias (router_bias)
        const int64_t n_ff_exp = hparams.n_ff_exp;
        if (!layer.ffn_gate) {
            layer.ffn_gate_inp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert},
                    llama_model_loader::TENSOR_NOT_REQUIRED);
            use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i, n_ff_exp);
            //layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert},
            //        llama_model_loader::TENSOR_NOT_REQUIRED);
            //layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert},
            //        llama_model_loader::TENSOR_NOT_REQUIRED);
            //layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert},
            //        llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ffn_exp_probs_b = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert},
                    llama_model_loader::TENSOR_NOT_REQUIRED);
        // shared expert MLP
            layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp},
                    llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp},
                    llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd},
                    llama_model_loader::TENSOR_NOT_REQUIRED);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_laguna_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    model.tok_embd    = create_tensor(ctx_input,  tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab},
                                       llama_model_loader::TENSOR_NOT_REQUIRED);
    if (!model.output) {
        // tied embeddings fallback
        model.output  = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab},
                                       llama_model_loader::TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_split = ctx_for_layer_split(i);
        auto & layer = model.layers[i];
        const int64_t n_head_l    = hparams.n_head(i);        // 48 (full) or 72 (SWA)
        const int64_t n_head_kv_l = hparams.n_head_kv(i);     // 8 (uniform)

        layer.attn_norm   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd}, 0);
        // Separate q/k/v projections (NOT merged) so the multi-GPU tensor-split
        // attention path (build_std_attention) stays available.
        layer.wq          = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,      "weight", i), {n_embd, n_embd_head_k * n_head_l},    0);
        layer.wk          = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,      "weight", i), {n_embd, n_embd_head_k * n_head_kv_l}, 0);
        layer.wv          = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,      "weight", i), {n_embd, n_embd_head_v * n_head_kv_l}, 0);
        layer.wo          = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT,    "weight", i), {n_embd_head_v * n_head_l, n_embd},    0);
        layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        // Per-head softplus attention output gate (self_attn.g_proj -> one scalar per head).
        layer.wqkv_gate   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_GATE,   "weight", i), {n_embd, n_head_l}, 0);
        layer.ffn_norm    = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_NORM,    "weight", i), {n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            // Leading dense layer(s) (Laguna-S: layer 0).
            layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        } else {
            // MoE layers: routed experts + score-correction bias + always-on shared expert.
            layer.ffn_gate_inp    = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
            // creates ffn_{gate,up,down}_exps (3D); fuses up/gate into ffn_up_gate_exps if opted in.
            use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i);
            layer.ffn_gate_shexp  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
            layer.ffn_up_shexp    = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
            layer.ffn_down_shexp  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, 0);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_muse_glimmer_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    model.tok_embd    = create_tensor(ctx_input,  tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab},
                                      llama_model_loader::TENSOR_NOT_REQUIRED);
    if (!model.output) {
        // tied embeddings fallback
        model.output  = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab},
                                      llama_model_loader::TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);
        auto & layer = model.layers[i];

        layer.attn_norm      = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        // Separate q/k/v projections (GQA 32/2, head_dim 128).
        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_head_k * n_head_kv}, 0);
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_head_v * n_head_kv}, 0);
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_v * n_head, n_embd}, 0);

        // Per-head RMS QK-norm; q_norm carries the folded qk_scale_factor from
        // conversion, k_norm is identity on released checkpoints.
        layer.attn_q_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        // FULL-WIDTH sigmoid attention output gate [n_embd, n_head*head_dim],
        // computed from the pre-attention normed hidden state. NOT the per-head
        // scalar gate that build_std_attention assumes for wqkv_gate -- the
        // muse-glimmer graph applies this gate itself and never routes through
        // build_std_attention.
        layer.wqkv_gate = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head_k * n_head}, 0);

        layer.ffn_norm      = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_post_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_hy3_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    GGML_ASSERT(n_expert > 0 && "n_expert must be > 0 for hy_v3");
    GGML_ASSERT(n_expert_used > 0 && "n_expert_used must be > 0 for hy_v3");

    model.tok_embd    = create_tensor(ctx_input,  tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    // tie_word_embeddings=false on every released Hy3, but keep the tied fallback.
    model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab},
                                       llama_model_loader::TENSOR_NOT_REQUIRED);
    if (!model.output) {
        model.output  = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab},
                                       llama_model_loader::TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);
        auto & layer = model.layers[i];

        // NextN/MTP tail. nextn_predict_layers is 0 for the released checkpoint, so
        // is_mtp_layer is never true there; kept so an MTP-bearing GGUF still loads.
        const bool is_mtp_layer = hparams.nextn_predict_layers > 0 &&
                                  static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers;
        const int flags = (!model.mtp && is_mtp_layer) ? llama_model_loader::TENSOR_SKIP : 0;

        layer.attn_norm   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd}, flags);

        // head_dim (128) != n_embd/n_head (64): every attention shape must come from
        // n_embd_head_k / n_embd_{k,v}_gqa, never from n_embd. wq is {4096, 8192}.
        // Separate q/k/v (NOT merged) so the tensor-split attention path in
        // build_std_attention stays available. No q/k/v/o biases (attention_bias=false).
        layer.wq          = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,      "weight", i), {n_embd, n_embd_head_k * n_head}, flags);
        layer.wk          = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,      "weight", i), {n_embd, n_embd_k_gqa},           flags);
        layer.wv          = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,      "weight", i), {n_embd, n_embd_v_gqa},           flags);
        layer.wo          = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT,    "weight", i), {n_embd_head_k * n_head, n_embd}, flags);

        // Per-head RMS QK-norm over head_dim, applied after the reshape and before RoPE.
        layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, flags);
        layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, flags);

        layer.ffn_norm    = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_NORM,    "weight", i), {n_embd}, flags);

        if (i < (int) hparams.n_layer_dense_lead) {
            // Leading dense block(s): first_k_dense_replace = 1 -> layer 0 only.
            layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, flags);
            layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, flags);
            layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, flags);
        } else {
            layer.ffn_gate_inp    = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
            // NO "bias" suffix: the GGUF name is "blk.N.exp_probs_b" (HF mlp.expert_bias
            // is an exact hit in the name map, so nothing is re-appended). REQUIRED on
            // purpose - moe_router_enable_expert_bias=true, and a null bias would silently
            // drop the selection bias and degrade routing with no error.
            layer.ffn_exp_probs_b = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_EXP_PROBS_B, i), {n_expert}, flags);

            // ffn_{gate,up,down}_exps at n_ff_exp=1536; picks up a pre-fused
            // ffn_gate_up_exps or an opt-in runtime merge when present.
            use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i, flags);

            // Always-on shared expert (num_shared_experts=1), no gate, summed unweighted.
            layer.ffn_gate_shexp  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, flags);
            layer.ffn_up_shexp    = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, flags);
            layer.ffn_down_shexp  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, flags);
        }

        if (is_mtp_layer) {
            layer.nextn.eh_proj          = create_tensor(ctx_split, tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2*n_embd, n_embd}, flags);
            layer.nextn.enorm            = create_tensor(ctx_layer, tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, flags);
            layer.nextn.hnorm            = create_tensor(ctx_layer, tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, flags);
            layer.nextn.embed_tokens     = create_tensor(ctx_split, tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab},
                    llama_model_loader::TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_head = create_tensor(ctx_split, tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab},
                    llama_model_loader::TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd},
                    llama_model_loader::TENSOR_NOT_REQUIRED | flags);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_qwen_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd*3});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd*3});
        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        create_std_ffn(i, tn, layer, n_ff/2, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_qwen2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        model.output_b    = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "bias"),   {n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);

        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        // optional bias tensors
        layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd});
        layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa});
        layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_qwen2_moe_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        // optional bias tensors
        layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd});
        layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa});
        layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        layer.ffn_gate_inp = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert});

        if (n_expert == 0) {
            throw std::runtime_error("n_expert must be > 0 for QWEN2MOE");
        }
        if (n_expert_used == 0) {
            throw std::runtime_error("n_expert_used must be > 0 for QWEN2MOE");
        }

        // MoE branch
        use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i);

        // Shared expert branch
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        layer.ffn_gate_inp_shexp = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), {n_embd});
        layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {    n_embd, n_ff_shexp});
        layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp,     n_embd});
        layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {    n_embd, n_ff_shexp});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_qwen3_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        use_mmap_buffer &= !merge_qkv(tn, i, 0);

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});

        layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k});
        layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k});

        layer.ffn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_qwen3_moe_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        use_mmap_buffer &= !merge_qkv(tn, i, 0);

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});

        layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k});
        layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k});

        auto ffn_ctx = model.split_mode == LLAMA_SPLIT_MODE_GRAPH ? ctx_split : ctx_layer;
        layer.ffn_norm = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        layer.ffn_gate_inp = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert});

        if (n_expert == 0) {
            throw std::runtime_error("n_expert must be > 0 for QWEN3MOE");
        }
        if (n_expert_used == 0) {
            throw std::runtime_error("n_expert_used must be > 0 for QWEN3MOE");
        }

        use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i);

    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_mellum_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
    model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        use_mmap_buffer &= !merge_qkv(tn, i, 0);

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});

        layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k});
        layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k});

        auto ffn_ctx = model.split_mode == LLAMA_SPLIT_MODE_GRAPH ? ctx_split : ctx_layer;
        layer.ffn_norm = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        layer.ffn_gate_inp = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert});

        if (n_expert == 0) {
            throw std::runtime_error("n_expert must be > 0 for Mellum");
        }
        if (n_expert_used == 0) {
            throw std::runtime_error("n_expert_used must be > 0 for Mellum");
        }

        use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i, 0, 0, ffn_ctx);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_qwen3next_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
        // PXA qwen3next-MTP: separate output projection for the MTP head (mirror qwen35moe).
        // Gated by model.mtp so a no-MTP GGUF (today's production model) is unaffected: the tensor
        // is SKIPped and output_mtp stays null.
        int mtp_flags = llama_model_loader::TENSOR_NOT_REQUIRED;
        if (!model.mtp) mtp_flags |= llama_model_loader::TENSOR_SKIP;
        auto output_mtp = create_tensor(ctx_output, "output_extra.weight", {n_embd, n_vocab}, mtp_flags);
        if (model.mtp) {
            model.output_mtp = output_mtp;
            if (!model.output_mtp) {
                model.output_mtp = model.output;
            } else {
                LLAMA_LOG_INFO("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Using %s as MTP output\n", model.output_mtp->name);
            }
        }
    }

    const bool has_moe_hparams = n_expert > 0 && n_expert_used > 0;
    const int64_t n_ff_exp   = hparams.n_ff_exp ? hparams.n_ff_exp : (has_moe_hparams ? n_ff / n_expert_used : n_ff);
    const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff_exp;

    const int64_t head_k_dim = hparams.ssm_d_state;
    const int64_t num_k_heads = hparams.ssm_n_group;
    const int64_t num_v_heads = hparams.ssm_dt_rank;
    const int64_t head_v_dim  = hparams.ssm_d_inner / num_v_heads;
    const int64_t key_dim     = head_k_dim * num_k_heads;
    const int64_t value_dim   = head_v_dim * num_v_heads;
    const int64_t conv_dim    = key_dim * 2 + value_dim;
    const int64_t qkvz_dim    = key_dim * 2 + value_dim * 2;
    const int64_t ba_dim      = num_v_heads * 2;

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        // PXA qwen3next-MTP: the trailing nextn_predict_layers are the MTP head (full-attention + MoE).
        // On a no-MTP load (model.mtp==false) skip those tensors so today's production model is unchanged.
        const bool is_mtp_layer = hparams.nextn_predict_layers > 0 &&
                                  static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers;
        int mtp_skip = 0;
        if (!model.mtp && is_mtp_layer) {
            mtp_skip |= llama_model_loader::TENSOR_SKIP;
        }

        layer.attn_norm      = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, mtp_skip);
        layer.attn_post_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, mtp_skip);
        layer.ffn_norm = layer.attn_post_norm;

        if (!hparams.is_recurrent(i)) {
            // Full-attention layer (also the MTP tail layer, which is full-attention + MoE)
            layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head * 2}, mtp_skip);
            layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, mtp_skip);
            layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, mtp_skip);
            layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, mtp_skip);

            layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, mtp_skip);
            layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, mtp_skip);
        } else {
            // Recurrent linear-attention layer
            layer.ssm_in         = create_tensor(ctx_layer, tn(LLM_TENSOR_SSM_IN,         "weight", i), {n_embd, qkvz_dim},
                    llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.wqkv           = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV,       "weight", i), {n_embd, key_dim * 2 + value_dim},
                    llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.wqkv_gate      = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_GATE,      "weight", i), {n_embd, value_dim},
                    llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d     = create_tensor(ctx_layer, tn(LLM_TENSOR_SSM_CONV1D,     "weight", i), {hparams.ssm_d_conv, conv_dim});
            layer.ssm_dt         = create_tensor(ctx_layer, tn(LLM_TENSOR_SSM_DT,         "bias",   i), {hparams.ssm_dt_rank});
            layer.ssm_a          = create_tensor(ctx_layer, tn(LLM_TENSOR_SSM_A_NOSCAN,             i), {hparams.ssm_dt_rank});
            layer.ssm_beta_alpha = create_tensor(ctx_layer, tn(LLM_TENSOR_SSM_BETA_ALPHA, "weight", i), {n_embd, ba_dim});
            layer.ssm_norm       = create_tensor(ctx_layer, tn(LLM_TENSOR_SSM_NORM,       "weight", i), {head_v_dim});
            layer.ssm_out        = create_tensor(ctx_layer, tn(LLM_TENSOR_SSM_OUT,        "weight", i), {value_dim, n_embd});
        }

        auto ffn_ctx = ctx_split; //model.split_mode == LLAMA_SPLIT_MODE_GRAPH ? ctx_split : ctx_layer;

        // Dense FFN path (optional, e.g. mlp_only_layers)
        layer.ffn_gate = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED | mtp_skip);
        layer.ffn_up   = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED | mtp_skip);
        layer.ffn_down = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED | mtp_skip);

        // MoE path (optional per-layer)
        layer.ffn_gate_inp = nullptr;
        if (n_expert > 0) {
            layer.ffn_gate_inp = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED | mtp_skip);
        }

        if (layer.ffn_gate_inp != nullptr) {
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0 when QWEN3NEXT MoE tensors are present");
            }
            use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i, mtp_skip, n_ff_exp);
        }

        // Shared expert path (optional per-layer)
        layer.ffn_gate_inp_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED | mtp_skip);
        if (layer.ffn_gate_inp_shexp != nullptr) {
            layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, llama_model_loader::TENSOR_NOT_REQUIRED | mtp_skip);
            layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, llama_model_loader::TENSOR_NOT_REQUIRED | mtp_skip);
            layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED | mtp_skip);
        }

        // PXA qwen3next-MTP: the nextn fusion tensors live ONLY on the MTP tail layer (mirror qwen35moe).
        if (is_mtp_layer) {
            layer.nextn.eh_proj          = create_tensor(ctx_split,
                    tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i),
                    { 2 * n_embd, n_embd },
                    mtp_skip | llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.nextn.enorm            = create_tensor(ctx_split,
                    tn(LLM_TENSOR_NEXTN_ENORM, "weight", i),
                    { n_embd },
                    mtp_skip);
            layer.nextn.hnorm            = create_tensor(ctx_split,
                    tn(LLM_TENSOR_NEXTN_HNORM, "weight", i),
                    { n_embd },
                    mtp_skip);
            layer.nextn.shared_head_norm = create_tensor(ctx_split,
                    tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i),
                    { n_embd },
                    mtp_skip | llama_model_loader::TENSOR_NOT_REQUIRED);
        }
    }

    return use_mmap_buffer;
}

bool create_tensors_helper::create_qwen35moe_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
        int flags = llama_model_loader::TENSOR_NOT_REQUIRED;
        if (!model.mtp) flags |= llama_model_loader::TENSOR_SKIP;
        auto output_mtp = create_tensor(ctx_output, "output_extra.weight", {n_embd, n_vocab}, flags);
        if (model.mtp) {
            model.output_mtp = output_mtp;
            if (!model.output_mtp) {
                model.output_mtp = model.output;
            } else {
                LLAMA_LOG_INFO("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Using %s as MTP output\n", model.output_mtp->name);
            }
        }
    }

    const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

    const int64_t head_k_dim = hparams.ssm_d_state;
    const int64_t head_v_dim = hparams.ssm_d_state;
    const int64_t n_k_heads  = hparams.ssm_n_group;
    const int64_t n_v_heads  = hparams.ssm_dt_rank;
    const int64_t key_dim    = head_k_dim * n_k_heads;
    const int64_t value_dim  = head_v_dim * n_v_heads;
    const int64_t conv_dim   = key_dim * 2 + value_dim;

    for (int i = 0; i < n_layer; ++i) {
        const bool is_mtp_layer = hparams.nextn_predict_layers > 0 &&
                                  static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers;

        // MTP-SPLIT: load the MTP tail layer into the SPLIT context like every normal layer
        // (was forced to ctx_for_layer(i) = non-split single-device island -> OOM'd last GPU under -sm attn).
        auto ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        int flags = 0;
        if (!model.mtp && is_mtp_layer) {
            flags |= llama_model_loader::TENSOR_SKIP;
        }

        layer.attn_norm      = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM,      "weight", i), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, flags);
        layer.ffn_norm = layer.attn_post_norm;

        if (!hparams.is_recurrent(i)) {
            // Attention layers
            layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head * 2 }, flags);
            layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, flags);
            layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, flags);
            layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            layer.wqkv           = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV,       "weight", i), { n_embd, key_dim * 2 + value_dim }, flags);
            layer.wqkv_gate      = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_GATE,      "weight", i), { n_embd, value_dim }, flags);
            layer.ssm_conv1d     = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_CONV1D,     "weight", i), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_DT,         "bias",   i), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_A_NOSCAN,             i), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_BETA,       "weight", i), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_ALPHA,      "weight", i), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_NORM,       "weight", i), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_OUT,        "weight", i), { value_dim, n_embd }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), { n_embd, n_expert }, flags);
        use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i, flags, n_ff_exp, ctx_split);

        // Shared experts
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        layer.ffn_gate_inp_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", i), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", i), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", i), { n_ff_shexp, n_embd }, flags);

        if (is_mtp_layer) {
            layer.nextn.eh_proj          = create_tensor(ctx_split,
                    tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i),
                    { 2 * n_embd, n_embd },
                    flags | llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.nextn.enorm            = create_tensor(ctx_split,
                    tn(LLM_TENSOR_NEXTN_ENORM, "weight", i),
                    { n_embd },
                    flags);
            layer.nextn.hnorm            = create_tensor(ctx_split,
                    tn(LLM_TENSOR_NEXTN_HNORM, "weight", i),
                    { n_embd },
                    flags);
            layer.nextn.shared_head_norm = create_tensor(ctx_split,
                    tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i),
                    { n_embd },
                    flags | llama_model_loader::TENSOR_NOT_REQUIRED);
        }
    }

    return use_mmap_buffer;
}

// =====================================================================================
// Qwen3.8-Flash-Next (arch "qwen4exp").
//
// Layout, relative to its closest relative here, create_qwen35moe_tensors():
//   * SAME  Gated DeltaNet on the recurrent layers (wqkv + wqkv_gate + ssm_conv1d/dt/a/
//           beta/alpha/norm/out), and the SAME interleaved [q|gate] wq of width
//           n_embd_head_k * n_head * 2 on the full-attention layers.
//   * SAME  MoE: per-layer router + expert stack + a shared expert.
//   * NEW   Low-rank hyper-connections replace attn_norm / attn_post_norm / output_norm.
//           The residual stream is hc-expanded (hc_dim = hc * n_embd) and every layer
//           carries two mixers (pre-token-mixer, pre-MoE); a third sits at the output.
//   * NEW   A QSA "lightning indexer" on the full-attention layers only.
//   * NEW   A PLE n-gram hash-embedding side path on the layers named by <arch>.ple.layers,
//           gathering from one enormous GET_ROWS table (per_layer_token_embd).
//
// Split-mode contract (this fork only - mainline has no tensor-parallel concept, so the
// upstream PR says nothing about any of it). Rules that made the choices below:
//   1. Any tensor created in ctx_for_layer_split() lands in split_tensors when -sm graph/attn
//      is active, i.e. in the CUDA split buffer, whose get_base() is the dummy 0x1000 and
//      whose set_tensor() is a silent no-op for tensors with no ->extra. So a tensor placed
//      there and then NOT handed to prepare_split_tensors() is never uploaded and reads as
//      garbage - silently. Everything below is therefore either split, mirrored, or kept out
//      of the split context on purpose.
//   2. Everything reusing an existing llama_layer field (wq/wk/wv/wo, attn_q_norm/k_norm,
//      wqkv/wqkv_gate/ssm_*, ffn_*_exps, ffn_*_shexp, ffn_gate_inp) is picked up by the
//      generic field-driven loop at the bottom of create_tensors() and needs nothing new
//      here beyond adding QWEN4EXP to the two [q|gate]-interleave conditionals.
//   3. The genuinely new tensors are handled by split_qwen4exp_tensors() below.
// =====================================================================================
bool create_tensors_helper::create_qwen4exp_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // Output. There is deliberately NO output_norm for this arch: the final low-rank
    // hyper-connection mixer (output_hc_norm/down/up) collapses the hc streams and
    // normalises in one step, so model.output_norm stays null and every generic
    // `if (model.output_norm)` / `split_tensors.find(model.output_norm)` site is inert.
    {
        model.hc_head_norm = create_tensor(ctx_output, tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), {hc_dim});
        model.hc_head_down = create_tensor(ctx_output, tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), {hc_dim, hc_lr});
        model.hc_head_up   = create_tensor(ctx_output, tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), {hc_lr, hc_dim});

        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab},
                llama_model_loader::TENSOR_NOT_REQUIRED);
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab},
                    llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    // The PLE n-gram table: [ple_head_dim, n_rows], where n_rows is the SUM of the padded
    // per-head n-gram vocabularies. For Qwen3.8-Flash-Next that is 160 x 320001536 =
    // 51,200,245,760 params, 95.4 GiB at bf16 - far larger than any GPU here, and meant to
    // be pinned to host RAM with `-ot per_layer_token_embd=CPU`. Two consequences:
    //   * the row count is a converter-side padded quantity, so read it back off the file
    //     rather than recomputing it and risking an off-by-a-pad mismatch;
    //   * it goes in ctx_input like GEMMA4's, never the split context. See the NOT-SPLIT
    //     note in split_qwen4exp_tensors() for why splitting it is not merely unnecessary
    //     but wrong.
    if (hparams.ple_n_heads > 0) {
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight");
        const auto * ple_w = ml.get_weight(ple_name.c_str());
        if (ple_w == nullptr) {
            throw std::runtime_error("qwen4exp: hparams declare a PLE n-gram table but " + ple_name + " is missing from the model");
        }
        const int64_t ple_rows = ple_w->tensor->ne[1];
        model.tok_embd_per_layer = create_tensor(ctx_input, ple_name, {hparams.ple_head_dim, ple_rows});
    }

    const int64_t n_ff_exp   = hparams.n_ff_exp   ? hparams.n_ff_exp   : n_ff / n_expert_used;
    const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

    // Gated DeltaNet geometry. head_k_dim == head_v_dim == ssm_d_state, exactly as in
    // create_qwen35moe_tensors; the hparams loader has already checked that ssm_d_inner
    // agrees with that, which is what split_recurrent_tensors() re-derives head_v_dim from.
    const int64_t head_k_dim = hparams.ssm_d_state;
    const int64_t head_v_dim = hparams.ssm_d_state;
    const int64_t n_k_heads  = hparams.ssm_n_group;
    const int64_t n_v_heads  = hparams.ssm_dt_rank;
    const int64_t key_dim    = head_k_dim * n_k_heads;
    const int64_t value_dim  = head_v_dim * n_v_heads;
    const int64_t conv_dim   = key_dim * 2 + value_dim;

    const int64_t idx_dim    = hparams.indexer_head_size;
    const int64_t idx_n_head = hparams.indexer_n_head;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = model.layers[i];

        ggml_context * ctx_split = ctx_for_layer_split(i);

        // NextN/MTP tail (blk.<n_layer-1> of a grafted nextn_predict_layers>0 GGUF): one
        // full-attention layer WITHOUT indexer/PLE tensors, plus the blk.*.nextn.* fusion +
        // head mixer loaded at the bottom of this loop. Skipped entirely when MTP is off,
        // exactly as create_qwen35_tensors does it.
        const bool is_mtp_layer = hparams.nextn_predict_layers > 0 &&
                                  (uint32_t) i >= (uint32_t) n_layer - hparams.nextn_predict_layers;
        int flags = 0;
        if (!model.mtp && is_mtp_layer) {
            flags |= llama_model_loader::TENSOR_SKIP;
        }

        // Two low-rank hyper-connection mixers per layer. No attn_norm / attn_post_norm /
        // ffn_norm exist for this arch - hc_attn_norm and hc_ffn_norm are what carries them,
        // over the hc-expanded stream rather than over n_embd.
        layer.hc_attn_norm   = create_tensor(ctx_split, tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", i), {hc_dim}, flags);
        layer.hc_attn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", i), {hc_dim, hc_lr}, flags);
        layer.hc_attn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_HC_ATTN_UP,     "weight", i), {hc_lr, hc_dim}, flags);
        layer.hc_attn_inject = create_tensor(ctx_split, tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", i), {hc_dim, hc}, flags);
        layer.hc_ffn_norm    = create_tensor(ctx_split, tn(LLM_TENSOR_HC_FFN_NORM,    "weight", i), {hc_dim}, flags);
        layer.hc_ffn_down    = create_tensor(ctx_split, tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", i), {hc_dim, hc_lr}, flags);
        layer.hc_ffn_up      = create_tensor(ctx_split, tn(LLM_TENSOR_HC_FFN_UP,      "weight", i), {hc_lr, hc_dim}, flags);
        layer.hc_ffn_inject  = create_tensor(ctx_split, tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", i), {hc_dim, hc}, flags);

        if (!hparams.is_recurrent(i)) {
            // Full attention. wq holds [q|gate] interleaved per head, hence the *2 - the same
            // packing qwen3next / qwen35 / qwen35moe use, and the reason QWEN4EXP has to join
            // them in the two granularity conditionals in the split loop below.
            layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head * 2}, flags);
            layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, flags);
            layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, flags);
            layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, flags);

            layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, flags);
            layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, flags);

            // QSA lightning indexer. Present only on full-attention MAIN layers; the grafted
            // MTP tail runs its attention DENSE and carries no indexer tensors at all.
            const int idx_flags = is_mtp_layer ? (flags | llama_model_loader::TENSOR_NOT_REQUIRED) : flags;
            layer.index_q_proj = create_tensor(ctx_split, tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", i), {n_embd, idx_n_head * idx_dim}, idx_flags);
            layer.index_k_proj = create_tensor(ctx_split, tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", i), {n_embd, idx_dim}, idx_flags);
            layer.index_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", i), {idx_dim}, idx_flags);
            layer.index_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_INDEXER_K_NORM, "weight", i), {idx_dim}, idx_flags);
        } else {
            // Gated DeltaNet, byte-for-byte the qwen35moe tensor set. src/llama-delta-net.cpp
            // already handles this exact layout (asymmetric k/v head counts with GVA repeat,
            // fused qkvz split), so there is nothing new to port on this branch.
            layer.wqkv       = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV,   "weight", i), {n_embd, key_dim * 2 + value_dim}, flags);
            layer.wqkv_gate  = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_GATE,  "weight", i), {n_embd, value_dim}, flags);
            layer.ssm_conv1d = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {hparams.ssm_d_conv, conv_dim}, flags);
            layer.ssm_dt     = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_DT,     "bias",   i), {hparams.ssm_dt_rank}, flags);
            layer.ssm_a      = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_A_NOSCAN,         i), {hparams.ssm_dt_rank}, flags);
            layer.ssm_beta   = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_BETA,   "weight", i), {n_embd, n_v_heads}, flags);
            layer.ssm_alpha  = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_ALPHA,  "weight", i), {n_embd, n_v_heads}, flags);
            layer.ssm_norm   = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_NORM,   "weight", i), {head_v_dim}, flags);
            layer.ssm_out    = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_OUT,    "weight", i), {value_dim, n_embd}, flags);
        }

        // PLE side path. Only the layers listed in <arch>.ple.layers carry these
        // (layer 2 for Qwen3.8-Flash-Next); every other layer leaves them null.
        if (hparams.is_ple(i)) {
            layer.ple_key        = create_tensor(ctx_split, tn(LLM_TENSOR_PLE_KEY,        "weight", i), {n_embd, hc_dim});
            layer.ple_value      = create_tensor(ctx_split, tn(LLM_TENSOR_PLE_VALUE,      "weight", i), {n_embd, n_embd});
            layer.ple_norm_key   = create_tensor(ctx_split, tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", i), {hc_dim});
            layer.ple_norm_query = create_tensor(ctx_split, tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", i), {hc_dim});
            layer.ple_norm_conv  = create_tensor(ctx_split, tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", i), {hc_dim});
            layer.ple_conv1d     = create_tensor(ctx_split, tn(LLM_TENSOR_PLE_CONV1D,     "weight", i), {hparams.ple_conv_kernel, hc_dim});
        }

        // MoE: every layer, no dense-FFN fallback in this arch.
        layer.ffn_gate_inp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
        use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i, flags, n_ff_exp, ctx_split);

        // Shared expert.
        layer.ffn_gate_inp_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), {n_embd}, flags);
        layer.ffn_gate_shexp     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", i), {n_embd, n_ff_shexp}, flags);
        layer.ffn_up_shexp       = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", i), {n_embd, n_ff_shexp}, flags);
        layer.ffn_down_shexp     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", i), {n_ff_shexp, n_embd}, flags);

        // --- NextN/MTP fusion + the draft's own head mixer, on the tail layer only ---
        // Like qwen35/qwen35moe these live in the plain per-layer ctx (they must REPLICATE
        // under a split mode, not row-split; -sm layer reads them cross-device like any norm).
        //   eh_proj = concat([fc_embedding, fc_hidden], dim=1) -> one [2*n_embd -> n_embd] mm
        //   hnorm   = grouped RMS gamma over the WHOLE hc row (hc_dim wide, +1 baked in)
        //   hc_*    = the MTP block's own low-rank collapse before the shared lm_head
        if (is_mtp_layer) {
            auto nextn_ctx = ctx_for_layer(i);
            layer.nextn.eh_proj = create_tensor(nextn_ctx, tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), {2*n_embd, n_embd}, flags);
            layer.nextn.enorm   = create_tensor(nextn_ctx, tn(LLM_TENSOR_NEXTN_ENORM,   "weight", i), {n_embd}, flags);
            layer.nextn.hnorm   = create_tensor(nextn_ctx, tn(LLM_TENSOR_NEXTN_HNORM,   "weight", i), {hc_dim}, flags);
            layer.nextn.hc_norm = create_tensor(nextn_ctx, tn(LLM_TENSOR_NEXTN_HC_NORM, "weight", i), {hc_dim}, flags);
            layer.nextn.hc_down = create_tensor(nextn_ctx, tn(LLM_TENSOR_NEXTN_HC_DOWN, "weight", i), {hc_dim, hc_lr}, flags);
            layer.nextn.hc_up   = create_tensor(nextn_ctx, tn(LLM_TENSOR_NEXTN_HC_UP,   "weight", i), {hc_lr, hc_dim}, flags);
        }
    }

    return use_mmap_buffer;
}

bool create_tensors_helper::create_qwen35_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab},
                llama_model_loader::TENSOR_NOT_REQUIRED);
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab},
                    llama_model_loader::TENSOR_DUPLICATED);
        }
        int flags = llama_model_loader::TENSOR_NOT_REQUIRED;
        if (!model.mtp) flags |= llama_model_loader::TENSOR_SKIP;
        auto output_mtp = create_tensor(ctx_output, "output_extra.weight", {n_embd, n_vocab}, flags);
        if (model.mtp) {
            model.output_mtp = output_mtp;
            if (!model.output_mtp) {
                model.output_mtp = model.output;
            } else {
                LLAMA_LOG_INFO("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Using %s as MTP output\n", model.output_mtp->name);
            }
        }
    }

    const int64_t head_k_dim = hparams.ssm_d_state;
    const int64_t head_v_dim = hparams.ssm_d_state;
    const int64_t n_k_heads  = hparams.ssm_n_group;
    const int64_t n_v_heads  = hparams.ssm_dt_rank;
    const int64_t key_dim    = head_k_dim * n_k_heads;
    const int64_t value_dim  = head_v_dim * n_v_heads;
    const int64_t conv_dim   = key_dim * 2 + value_dim;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = model.layers[i];

        const bool is_mtp_layer = hparams.nextn_predict_layers > 0 &&
                                  static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers;

        // MTP-SPLIT: load the MTP tail layer into the SPLIT context like every normal layer.
        // NOTE (qwen35 non-MoE / 9B/4B): this variant uses a DENSE FFN (ffn_gate/down/up via
        // llm_build_ffn). The dense-FFN split path may differ from the MoE path; the 122B target
        // is qwen35MOE (handled above), so this is for completeness. The nextn-specific tensors
        // below are still placed in the non-split nextn_ctx (they must REPLICATE, not row-split).
        ggml_context * ctx_split = ctx_for_layer_split(i);

        int flags = 0;
        // Skip loading MTP layers if the feature is disabled
        if (!model.mtp) {
            if (is_mtp_layer) {
                flags |= llama_model_loader::TENSOR_SKIP;
            }
        }
        const int mtp_opt = is_mtp_layer ? llama_model_loader::TENSOR_NOT_REQUIRED : 0;

        layer.attn_norm      = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM,      "weight", i), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, flags);
        layer.ffn_norm = layer.attn_post_norm;

        if (!hparams.is_recurrent(i)) {
            // Attention layers (MTP layer is always standard attention)
            layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head * 2 }, flags | mtp_opt);
            layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, flags);
            layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, flags);
            layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            layer.wqkv           = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV,       "weight", i), { n_embd, key_dim * 2 + value_dim }, llama_model_loader::TENSOR_NOT_REQUIRED | flags);
            layer.wqkv_gate      = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_GATE,      "weight", i), { n_embd, value_dim }, llama_model_loader::TENSOR_NOT_REQUIRED | flags);
            layer.ssm_conv1d     = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_CONV1D,     "weight", i), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_DT,         "bias",   i), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_A_NOSCAN,             i), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_BETA,       "weight", i), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_ALPHA,      "weight", i), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_NORM,       "weight", i), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_OUT,        "weight", i), { value_dim, n_embd }, flags);
        }

        // 9B and 4B don't have MLP in MTP
        layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, flags | mtp_opt);
        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, flags | mtp_opt);
        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, flags | mtp_opt);

        // --- NextN / MTP tensors on the MTP layer ---
        if (is_mtp_layer) {
            const int final_layer = n_layer - 1;
            auto nextn_ctx      = ctx_for_layer(final_layer);
            // 9B doesn't have fc
            layer.nextn.eh_proj          = create_tensor(nextn_ctx,
                    tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", final_layer),
                    { 2*n_embd, n_embd },
                    flags | llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.nextn.enorm            = create_tensor(nextn_ctx,
                    tn(LLM_TENSOR_NEXTN_ENORM, "weight", final_layer),
                    { n_embd },
                    flags);
            layer.nextn.hnorm            = create_tensor(nextn_ctx,
                    tn(LLM_TENSOR_NEXTN_HNORM, "weight", final_layer),
                    { n_embd },
                    flags);
            layer.nextn.shared_head_norm = create_tensor(nextn_ctx,
                    tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", final_layer),
                    { n_embd },
                    flags | llama_model_loader::TENSOR_NOT_REQUIRED);
        }
    }

    // 9B shares q_proj
    if (model.mtp && hparams.nextn_predict_layers > 0) {
        const uint32_t n_main = n_layer - hparams.nextn_predict_layers;
        for (uint32_t i = n_main; i < (uint32_t)n_layer; ++i) {
            auto & mtp_layer = model.layers[i];
            auto & last_main = model.layers[n_main - 1];
            if (mtp_layer.wq == nullptr) {
                mtp_layer.wq = last_main.wq;
            }
        }
    }

    return use_mmap_buffer;
}

bool create_tensors_helper::create_mimo2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab, true);

    for (int i = 0; i < n_layer; ++i) {
        uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(i);
        uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(i);
        uint32_t n_head = hparams.n_head(i);

        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm  = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM,  "weight", i), {n_embd});
        layer.attn_sinks = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_SINKS, "weight", i), {n_head}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, 0);
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_v * n_head, n_embd }, 0);

        auto ffn_ctx = model.split_mode == LLAMA_SPLIT_MODE_GRAPH ? ctx_split : ctx_layer;
        layer.ffn_norm = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        // non-MoE branch
        layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);

        // MoE branch
        layer.ffn_gate_inp  = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert},
                llama_model_loader::TENSOR_NOT_REQUIRED);
        if (layer.ffn_gate_inp) {
            use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i);
            layer.ffn_exp_probs_b = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert},
                    llama_model_loader::TENSOR_NOT_REQUIRED);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_phi2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
        model.output        = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
        model.output_b      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "bias"),   {n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);

        if (layer.wqkv == nullptr) {
            layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd});
            layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q, "bias", i),   {n_embd});

            layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa});
            layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K, "bias", i),   {n_embd_gqa});

            layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa});
            layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V, "bias", i),   {n_embd_gqa});
        }

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_phi3_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    const int64_t n_embd_head = n_embd / n_head;

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output        = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd });

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd });

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd });

        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd });
        layer.ffn_up = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, 2 * n_ff });

        layer.rope_long  = create_tensor(ctx_layer, tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight"), { n_embd_head/2 }, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));
        layer.rope_short = create_tensor(ctx_layer, tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight"), { n_embd_head/2 }, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_gpt2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});
    model.pos_embd = create_tensor(ctx_input, tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train});

    // output
    {
        model.output_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
        model.output        = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,   "bias", i),   {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa});

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_codeshell_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    create_default_embd_output(tn, n_embd, n_vocab, true);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa});

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP, "weight", i),   {n_embd, n_ff});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP, "bias", i),     {n_ff});
    }
    return use_mmap_buffer;
}

void create_tensors_helper::create_default_embd_output(const LLM_TN & tn, int n_embd, int n_vocab, bool norm_bias) {
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    model.output_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
    if (norm_bias) {
    model.output_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
    }
    model.output        = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
}

bool create_tensors_helper::create_orion_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    create_default_embd_output(tn, n_embd, n_vocab, true);
    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_internlm_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    create_default_embd_output(tn, n_embd, n_vocab, false);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        // layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_gemma_tensors(const LLM_TN & tn, int version) {
    LOADING_PRELUDE

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
    model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab},
            llama_model_loader::TENSOR_NOT_REQUIRED);
    if (!model.output) {
        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab},
                llama_model_loader::TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        use_mmap_buffer &= !merge_qkv(tn, i, 0, true);

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});
        if (version > 1) {
            layer.attn_post_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd});
        }
        if (version > 2) {
            layer.attn_k_norm    = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), {n_embd_head_k});
            layer.attn_q_norm    = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head_k});
        }

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
        if (version > 1) {
            layer.ffn_post_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd});
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_gemma4_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const uint32_t n_embd_per_layer = hparams.n_embd_per_layer;
    const int64_t  n_ff_exp         = hparams.n_ff_exp;
    const bool     use_split_ctx    = model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN;

    if (n_embd_head_k != n_embd_head_v) {
        throw std::runtime_error("Gemma 4 requires n_embd_head_k == n_embd_head_v");
    }
    if (hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa) {
        throw std::runtime_error("Gemma 4 requires n_embd_head_k_swa == n_embd_head_v_swa");
    }

    model.output = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (!model.output) {
        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
    }

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    if (n_embd_per_layer > 0) {
        model.tok_embd_per_layer   = create_tensor(ctx_input, tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"), {n_embd_per_layer * n_layer, n_vocab}, 0);
        model.per_layer_model_proj = create_tensor(ctx_output, tn(LLM_TENSOR_PER_LAYER_MODEL_PROJ, "weight"), {n_embd, n_embd_per_layer * n_layer}, 0);
        model.per_layer_proj_norm  = create_tensor(ctx_output, tn(LLM_TENSOR_PER_LAYER_PROJ_NORM,  "weight"), {n_embd_per_layer}, 0);
    }

    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = use_split_ctx ? ctx_for_layer_split(i) : ctx_layer;
        auto & layer = model.layers[i];
        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_embd_k    = hparams.n_embd_k_gqa(i);
        const int64_t n_embd_v    = hparams.n_embd_v_gqa(i);

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // note: use_alternative_attention (v_proj is optional, if it's not present, use k_proj)
        // Gemma 4 shared-KV layers omit fresh K/K-norm tensors.
        const int kv_required = hparams.has_kv(i) ? 0 : llama_model_loader::TENSOR_NOT_REQUIRED;

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head * n_head}, 0);
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k}, kv_required);
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head * n_head, n_embd}, 0);

        layer.attn_q_norm    = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head}, 0);
        layer.attn_k_norm    = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), {n_embd_head}, kv_required);
        layer.attn_post_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.out_scale = create_tensor(ctx_split, tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), {1u}, llama_model_loader::TENSOR_NOT_REQUIRED);

        if (!hparams.swa_layers[i]) {
            // full_attention layers use rope_freqs for proportional rope
            layer.rope_freqs = create_tensor(ctx_split, tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head/2}, rope_freqs_flag);
            rope_freqs_flag = llama_model_loader::TENSOR_DUPLICATED;
        }

        // handle use_double_wide_mlp
        int64_t n_ff_cur = hparams.n_ff(i);

        // for expert layers, we use normal FFN as shared expert (same as python code)
        layer.ffn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_cur, n_embd}, 0);
        layer.ffn_post_norm = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

        // MoE router
        layer.ffn_gate_inp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED);
        bool has_expert = layer.ffn_gate_inp != nullptr;

        // norm
        if (has_expert) {
            layer.ffn_gate_inp_s = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP, "scale", i), {n_embd}, 0);

            layer.ffn_pre_norm_2  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_PRE_NORM_2,  "weight", i), {n_embd}, 0);
            layer.ffn_post_norm_1 = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_POST_NORM_1, "weight", i), {n_embd}, 0);
            layer.ffn_post_norm_2 = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_POST_NORM_2, "weight", i), {n_embd}, 0);

            // MoE FFN
            layer.ffn_up_gate_exps  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_UP_EXPS,  "weight", i), {n_embd, n_ff_exp * 2, n_expert}, 0);
            layer.ffn_down_exps     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS,     "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
            layer.ffn_down_exps_s   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS,     "scale",  i), {n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED);

            // per-expert scale will be loaded as down_exps_s at the end of the current switch case
        }

        // per-layer embeddings
        if (n_embd_per_layer > 0) {
            layer.per_layer_inp_gate   = create_tensor(ctx_split, tn(LLM_TENSOR_PER_LAYER_INP_GATE,  "weight", i), {n_embd, n_embd_per_layer}, 0);
            layer.per_layer_proj       = create_tensor(ctx_split, tn(LLM_TENSOR_PER_LAYER_PROJ,      "weight", i), {n_embd_per_layer, n_embd}, 0);
            layer.per_layer_post_norm  = create_tensor(ctx_split, tn(LLM_TENSOR_PER_LAYER_POST_NORM, "weight", i), {n_embd}, 0);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_gemma4_mtp_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t n_backbone = hparams.mtp_backbone_n_embd;
    const bool    use_split_ctx = model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN;
    if (n_backbone <= 0) {
        throw std::runtime_error("Gemma 4 MTP assistant requires backbone_embedding_length metadata");
    }

    model.tok_embd    = create_tensor(ctx_input,  tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
    auto output_extra = create_tensor(ctx_output, "output_extra.weight", {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
    if (output_extra) {
        model.output = output_extra;
    }
    if (model.output == NULL) {
        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
    }
    model.mtp_pre_proj  = create_tensor(ctx_output, tn(LLM_TENSOR_MTP_PRE_PROJ,  "weight"), {2*n_backbone, n_embd}, 0);
    model.mtp_post_proj = create_tensor(ctx_output, tn(LLM_TENSOR_MTP_POST_PROJ, "weight"), {n_embd, n_backbone}, 0);

    model.mtp_token_ordering = create_tensor(ctx_output, tn(LLM_TENSOR_MTP_TOKEN_ORDERING, "weight"), {n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
    model.mtp_centroids      = create_tensor(ctx_output, tn(LLM_TENSOR_MTP_CENTROIDS,      "weight"), {n_embd, hparams.mtp_num_centroids}, llama_model_loader::TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = use_split_ctx ? ctx_for_layer_split(i) : ctx_layer;
        auto & layer = model.layers[i];
        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff_cur    = hparams.n_ff(i);

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.wq        = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, 0);
        layer.wo        = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);

        layer.attn_q_norm    = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head}, 0);
        layer.attn_post_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);
        layer.out_scale      = create_tensor(ctx_split, tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), {1u}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_norm      = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_gate      = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE,      "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_up        = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,        "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_down      = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN,      "weight", i), {n_ff_cur, n_embd}, 0);
        layer.ffn_post_norm = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
    }

    return use_mmap_buffer;
}

bool create_tensors_helper::create_starcoder2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});

        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }

    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        // optional bias tensors
        layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd});
        layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa});
        layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa});
        layer.bo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});

        // optional bias tensors
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP ,  "bias", i), {  n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_mamba_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t d_conv  = hparams.ssm_d_conv;
    const int64_t d_inner = hparams.ssm_d_inner;
    const int64_t d_state = hparams.ssm_d_state;
    const int64_t dt_rank = hparams.ssm_dt_rank;

    // only an expansion factor of 2 is supported for now
    GGML_ASSERT(2 * n_embd == d_inner);

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});

        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed, duplicated to allow offloading
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        // norm
        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.ssm_in = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_IN, "weight", i), {n_embd, 2*d_inner});

        layer.ssm_conv1d = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner});
        layer.ssm_conv1d_b = create_tensor(ctx_layer, tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {d_inner});

        layer.ssm_x = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_X, "weight", i), {d_inner, dt_rank + 2*d_state});

        layer.ssm_dt = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_DT, "weight", i), {dt_rank, d_inner});
        layer.ssm_dt_b = create_tensor(ctx_layer, tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner});

        // no "weight" suffix for these
        layer.ssm_a = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_A, i), {d_state, d_inner});
        layer.ssm_d = create_tensor(ctx_layer, tn(LLM_TENSOR_SSM_D, i), {d_inner});

        // out_proj
        layer.ssm_out = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_OUT, "weight", i), {d_inner, n_embd});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_xverse_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_command_r_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        // init output from the input tok embed
        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (n_layer >= 64){
            layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k, n_head}, 0);
            layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k, n_head_kv}, 0);
        }

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_olmo_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab, false);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        create_std_attn(i, tn, layer, n_embd, n_embd_gqa, ctx_split);
        create_std_ffn (i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_openelm_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        // init output from the input tok embed
        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        const int64_t n_head      =   hparams.n_head(i);
        const int64_t n_head_qkv  = 2*hparams.n_head_kv(i) + n_head;
        const int64_t n_ff        =   hparams.n_ff(i);

        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_head_qkv*n_embd_head_k});
        layer.attn_q_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k});
        layer.attn_k_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head*n_embd_head_k, n_embd});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_gptneox_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
        model.output        = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa});

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_arctix_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        create_std_attn(i, tn, layer, n_embd, n_embd_gqa, ctx_split);

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        create_std_ffn (i, tn, layer, n_embd, n_embd, ctx_split);

        layer.ffn_gate_inp  = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert});
        layer.ffn_norm_exps = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM_EXPS, "weight", i), {n_embd});
        layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, false);
        layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert});
        layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_deepseek2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const bool is_lite = (hparams.n_layer == 27 || hparams.n_layer == 26);

    const int64_t n_embd_head_qk_rope = hparams.n_rot;
    const int64_t n_embd_head_qk_nope = hparams.n_embd_head_k(0) - hparams.n_rot;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        // Under -sm graph/attn, norms need per-device replicas via prepare_split_tensors(-1, ...).
        auto norm_ctx = (model.split_mode == LLAMA_SPLIT_MODE_GRAPH ||
                         model.split_mode == LLAMA_SPLIT_MODE_ATTN) ? ctx_split : ctx_layer;
        layer.attn_norm = create_tensor(norm_ctx, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        if (!is_lite) {
            layer.attn_q_a_norm = create_tensor(norm_ctx, tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank});
        }

        layer.attn_kv_a_norm = create_tensor(norm_ctx, tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank});

        bool merged = false;
        if (ml.merge_qkv) {
            auto q_name = is_lite ? tn(LLM_TENSOR_ATTN_Q, "weight", i) : tn(LLM_TENSOR_ATTN_Q_A, "weight", i);
            auto k_name = tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i);
            auto wq = ml.require_tensor_meta(q_name.c_str());
            auto wk = ml.require_tensor_meta(k_name.c_str());
            GGML_ASSERT(wq && wk);
            if (wq->type == wk->type) {
                GGML_ASSERT(wq->ne[0] == wk->ne[0]);
                layer.wkq_a_mqa = ggml_new_tensor_2d(ctx_split, wq->type, wq->ne[0], wq->ne[1] + wk->ne[1]);
                snprintf(layer.wkq_a_mqa->name, GGML_MAX_NAME, "blk.%d.attn_qk_a_mqa.weight", i);
                if (is_lite) {
                    layer.wq = ml.create_tensor_as_view(ctx_split, layer.wkq_a_mqa, q_name.c_str(), { wq->ne[0], wq->ne[1] }, 0);
                } else {
                    layer.wq_a = ml.create_tensor_as_view(ctx_split, layer.wkq_a_mqa, q_name.c_str(), { wq->ne[0], wq->ne[1] }, 0);
                }
                layer.wkv_a_mqa = ml.create_tensor_as_view(ctx_split, layer.wkq_a_mqa, k_name.c_str(), { wk->ne[0], wk->ne[1] }, wq->ne[1]*wq->nb[1]);
                merged = true;
                use_mmap_buffer = false;
                LLAMA_LOG_DEBUG("============== Merged %s (%ld x %ld) and %s (%ld x %ld)\n", q_name.c_str(),
                        wq->ne[0], wq->ne[1], k_name.c_str(), wk->ne[0], wk->ne[1]);
            }
        }

        if (!is_lite) {
            if (!merged) {
                layer.wq_a = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank});
            }
            layer.wq_b = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k});
        } else if (!merged) {
            layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_k_gqa});
        }

        if (!merged) {
            layer.wkv_a_mqa = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i),{n_embd, kv_lora_rank + (n_embd_head_qk_rope)});
        }

        // Keep wkv_b on a single-device buffer; llm_prepare_mla reads it back to derive wk_b/wv_b.
        auto wkv_b_ctx = (model.split_mode == LLAMA_SPLIT_MODE_GRAPH ||
                          model.split_mode == LLAMA_SPLIT_MODE_ATTN) ? ctx_layer : ctx_split;
        layer.wkv_b     = create_tensor(wkv_b_ctx, tn(LLM_TENSOR_ATTN_KV_B,     "weight", i),
                {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)}, llama_model_loader::TENSOR_NOT_REQUIRED);
        if (!layer.wkv_b) {
            // Incompatible mainline model. Let's see if we can still load it
            layer.wk_b = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, 0);
            layer.wv_b = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v, n_head}, 0);

        } else {
            layer.wk_b      = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_B,      "weight", i), {n_embd_head_qk_nope, n_head * kv_lora_rank}, 1);
            layer.wv_b      = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V_B,      "weight", i), {kv_lora_rank, n_head * n_embd_head_v}, 1);
        }
        layer.wo        = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {              n_head * (                      n_embd_head_v), n_embd});

        layer.ffn_norm = create_tensor(norm_ctx, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
            layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
            layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
        } else {
            // llm_build_std_moe_ffn needs per-device extras on these under -sm graph/attn.
            const auto moe_ctx = (model.split_mode == LLAMA_SPLIT_MODE_GRAPH ||
                                  model.split_mode == LLAMA_SPLIT_MODE_ATTN) ? ctx_split : ctx_layer;
            layer.ffn_gate_inp = create_tensor(moe_ctx, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert});
            layer.ffn_exp_probs_b = create_tensor(moe_ctx, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 1);

            GGML_ASSERT(n_expert      > 0);
            GGML_ASSERT(n_expert_used > 0);

            // MoE branch
            use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i);

            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared});
            layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd});
            layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared});
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_glm_dsa_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t n_embd_head_qk_rope = hparams.n_rot;
    const int64_t n_embd_head_qk_nope = hparams.n_embd_head_k(0) - hparams.n_rot;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        const bool is_mtp_layer = hparams.nextn_predict_layers > 0 &&
                                  static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers;

        int flags = 0;
        // Skip loading MTP layers if the feature is disabled
        if (!model.mtp) {
            if (is_mtp_layer) {
                flags |= llama_model_loader::TENSOR_SKIP | llama_model_loader::TENSOR_NOT_REQUIRED;
            }
        }
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        // Under -sm graph/attn, norms and MoE-gate tensors need per-device replicas in ctx_split.
        const auto graph_or_attn = (model.split_mode == LLAMA_SPLIT_MODE_GRAPH ||
                                    model.split_mode == LLAMA_SPLIT_MODE_ATTN);
        auto norm_ctx = graph_or_attn ? ctx_split : ctx_layer;
        auto moe_ctx  = graph_or_attn ? ctx_split : ctx_layer;

        layer.attn_norm = create_tensor(norm_ctx, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);
        layer.attn_q_a_norm = create_tensor(norm_ctx, tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);

        layer.attn_kv_a_norm = create_tensor(norm_ctx, tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);

        bool merged = false;
        if (ml.merge_qkv) {
            auto q_name = tn(LLM_TENSOR_ATTN_Q_A, "weight", i);
            auto k_name = tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i);
            auto wq = ml.require_tensor_meta(q_name.c_str());
            auto wk = ml.require_tensor_meta(k_name.c_str());
            GGML_ASSERT(wq && wk);
            if (wq->type == wk->type) {
                GGML_ASSERT(wq->ne[0] == wk->ne[0]);
                layer.wkq_a_mqa = ggml_new_tensor_2d(ctx_split, wq->type, wq->ne[0], wq->ne[1] + wk->ne[1]);
                snprintf(layer.wkq_a_mqa->name, GGML_MAX_NAME, "blk.%d.attn_qk_a_mqa.weight", i);
                layer.wq_a = ml.create_tensor_as_view(ctx_split, layer.wkq_a_mqa, q_name.c_str(), { wq->ne[0], wq->ne[1] }, 0, flags);
                layer.wkv_a_mqa = ml.create_tensor_as_view(ctx_split, layer.wkq_a_mqa, k_name.c_str(), { wk->ne[0], wk->ne[1] }, wq->ne[1]*wq->nb[1], flags);
                merged = true;
                use_mmap_buffer = false;
                LLAMA_LOG_DEBUG("============== Merged %s (%ld x %ld) and %s (%ld x %ld)\n", q_name.c_str(),
                        wq->ne[0], wq->ne[1], k_name.c_str(), wk->ne[0], wk->ne[1]);
            }
        }

        if (!merged) {
            layer.wq_a = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, flags);
        }
        layer.wq_b = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k}, flags);

        if (!merged) {
            layer.wkv_a_mqa = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i),{n_embd, kv_lora_rank + (n_embd_head_qk_rope)}, flags);
        }

            // Incompatible mainline model. Let's see if we can still load it
        layer.wk_b = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, 0);
        layer.wv_b = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v, n_head}, 0);
        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v, n_embd}, flags);

                       // DSA indexer
        layer.indexer_k_norm   = create_tensor(ctx_split, tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {hparams.indexer_head_size}, flags);
        layer.indexer_k_norm_b = create_tensor(ctx_split, tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   i), {hparams.indexer_head_size}, flags);
        layer.indexer_proj     = create_tensor(ctx_split, tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, flags);
        layer.indexer_attn_k   = create_tensor(ctx_split, tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, hparams.indexer_head_size}, flags);
        layer.indexer_attn_q_b = create_tensor(ctx_split, tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * hparams.indexer_head_size}, flags);

        layer.ffn_norm = create_tensor(norm_ctx, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, flags);
            layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, flags);
        } else {
            layer.ffn_gate_inp = create_tensor(moe_ctx, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(moe_ctx, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, flags);

            GGML_ASSERT(n_expert      > 0);
            GGML_ASSERT(n_expert_used > 0);

            // MoE branch
            layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);

            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
            layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
        }

        if (is_mtp_layer) {
            layer.nextn.eh_proj          = create_tensor(ctx_split, tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), { 2 * n_embd, n_embd }, flags);
            layer.nextn.enorm            = create_tensor(ctx_split, tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), { n_embd }, flags);
            layer.nextn.hnorm            = create_tensor(ctx_split, tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), { n_embd }, flags);

            // Optional tensors
            layer.nextn.embed_tokens     = create_tensor(ctx_split, tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", i), { n_embd, n_vocab }, flags | llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(ctx_split, tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab }, flags | llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(ctx_split, tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, flags);
        }
    }
    return use_mmap_buffer;
}

// Adapted from llama.cpp src/models/deepseek4.cpp
// (llama_model_deepseek4::load_arch_tensors, lines 62-148) @ upstream commit
// 82dbc4f01, PR #24162 and follow-ups.
// Copyright (c) 2023-2026 The ggml authors. MIT.
//
// DS4 is NOT an MLA architecture: attn_kv is a plain MQA projection (1 KV head of
// n_embd_head_k) and the output projection is a LoRA pair (attn_output_a/_b). There is
// no attn_k_b / attn_v_b, so nothing here feeds llm_prepare_mla / is_mla_model().
bool create_tensors_helper::create_deepseek4_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t o_groups    = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult     = hparams.dsv4_hc_mult;
    const int64_t hc_dim      = hc_mult * n_embd;
    const int64_t hc_mix_dim  = (2 + hc_mult) * hc_mult;

    if (o_groups <= 0 || o_lora_rank <= 0 || hc_mult <= 0) {
        throw std::runtime_error("DeepSeek-V4: o_groups / o_lora_rank / hc_mult must all be > 0");
    }
    if ((n_head * n_embd_head_k) % o_groups != 0) {
        throw std::runtime_error("DeepSeek-V4: n_head*n_embd_head_k is not divisible by o_groups");
    }

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
    model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});

    model.hc_head_fn    = create_tensor(ctx_output_split, tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc_dim, hc_mult});
    model.hc_head_base  = create_tensor(ctx_output,       tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc_mult});
    model.hc_head_scale = create_tensor(ctx_output,       tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1});

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm     = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd});
        layer.attn_sinks    = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_SINKS,     "weight", i), {n_head});
        layer.wq_a          = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_A,       "weight", i), {n_embd, q_lora_rank});
        layer.attn_q_a_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank});
        layer.wq_b          = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_B,       "weight", i), {q_lora_rank, n_head * n_embd_head_k});
        layer.wkv           = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_KV,        "weight", i), {n_embd, n_embd_head_k});
        // NB: attn_kv_a_norm, NOT a new attn_kv_norm field -- build_deepseek4 reads
        // attn_kv_a_norm, and llm_build_norm() SILENTLY skips the weight multiply when
        // the tensor is null, so a mismatch here loads the weight and then never uses it.
        layer.attn_kv_a_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {n_embd_head_k});
        layer.wo_a          = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT_A,     "weight", i), {n_head * n_embd_head_k / o_groups, o_lora_rank * o_groups});
        layer.wo_b          = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT_B,     "weight", i), {o_groups * o_lora_rank, n_embd});

        layer.hc_attn_fn    = create_tensor(ctx_split, tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim});
        layer.hc_attn_base  = create_tensor(ctx_layer, tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim});
        layer.hc_attn_scale = create_tensor(ctx_layer, tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3});
        layer.hc_ffn_fn     = create_tensor(ctx_split, tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim});
        layer.hc_ffn_base   = create_tensor(ctx_layer, tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim});
        layer.hc_ffn_scale  = create_tensor(ctx_layer, tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3});

        const int64_t ratio = hparams.dsv4_compress_ratios[i];
        if (ratio != 0) {
            // ratio 4 (CSA + lightning indexer) carries a doubled compressor width;
            // ratio 128 (hyper-compressed attention) does not.
            const int64_t coff = ratio == 4 ? 2 : 1;

            layer.attn_comp_wkv   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, coff * n_embd_head_k});
            layer.attn_comp_wgate = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, coff * n_embd_head_k});
            layer.attn_comp_ape   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_COMPRESSOR_APE,   "weight", i), {coff * n_embd_head_k, ratio});
            layer.attn_comp_norm  = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head_k});

            if (ratio == 4) {
                const int64_t n_embd_indexer = hparams.indexer_head_size;

                layer.indexer_proj     = create_tensor(ctx_split, tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, (int64_t) hparams.indexer_n_head});
                layer.indexer_attn_q_b = create_tensor(ctx_split, tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, (int64_t) hparams.indexer_n_head * n_embd_indexer});

                layer.indexer_comp_wkv   = create_tensor(ctx_split, tn(LLM_TENSOR_INDEXER_COMPRESSOR_WKV,   "weight", i), {n_embd, 2 * n_embd_indexer});
                layer.indexer_comp_wgate = create_tensor(ctx_split, tn(LLM_TENSOR_INDEXER_COMPRESSOR_WGATE, "weight", i), {n_embd, 2 * n_embd_indexer});
                layer.indexer_comp_ape   = create_tensor(ctx_split, tn(LLM_TENSOR_INDEXER_COMPRESSOR_APE,   "weight", i), {2 * n_embd_indexer, ratio});
                layer.indexer_comp_norm  = create_tensor(ctx_layer, tn(LLM_TENSOR_INDEXER_COMPRESSOR_NORM,  "weight", i), {n_embd_indexer});
            }
            // ratio values other than 0/4/128 are rejected in llm_load_hparams().
        }

        layer.ffn_gate_inp = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert});
        if ((uint32_t) i < hparams.dsv4_hash_layer_count) {
            // I32 token-id -> expert-id hash routing table. Must never be quantized.
            layer.ffn_gate_tid2eid = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {n_expert_used, n_vocab});
        } else {
            layer.ffn_exp_probs_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert});
        }
        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        GGML_ASSERT(n_expert      > 0);
        GGML_ASSERT(n_expert_used > 0);

        layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert});
        layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert});
        layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert});

        layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared});
        layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    });
        layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared});
    }

    return use_mmap_buffer;
}


// DSpark drafter tensors. Three DS4 raw-SWA blocks under an "mtp.%d." prefix, plus:
//   stage 0    : main_proj {3*n_embd, n_embd}, main_norm {n_embd}
//   last stage : norm, hc_head_{fn,base,scale}, markov_w1/w2 {rank, n_vocab}, conf_proj
// There is NO token_embd, NO output and NO output_norm in the support file - the drafter
// borrows the TARGET's tok_embd and output.weight at graph-build time.
bool create_tensors_helper::create_dspark_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t o_groups    = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult     = hparams.dsv4_hc_mult;
    const int64_t hc_dim      = hc_mult * n_embd;
    const int64_t hc_mix_dim  = (2 + hc_mult) * hc_mult;

    const int64_t markov_rank = hparams.dspark_markov_rank;
    const int64_t n_capture   = hparams.dspark_n_target_layers;

    if (o_groups <= 0 || o_lora_rank <= 0 || hc_mult <= 0 || markov_rank <= 0) {
        throw std::runtime_error("DSpark: o_groups / o_lora_rank / hc_mult / markov_rank must all be > 0");
    }

    const int last = n_layer - 1;

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm      = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd});
        layer.attn_sinks     = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_SINKS,     "weight", i), {n_head});
        layer.wq_a           = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_A,       "weight", i), {n_embd, q_lora_rank});
        layer.attn_q_a_norm  = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank});
        layer.wq_b           = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_B,       "weight", i), {q_lora_rank, n_head * n_embd_head_k});
        layer.wkv            = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_KV,        "weight", i), {n_embd, n_embd_head_k});
        layer.attn_kv_a_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {n_embd_head_k});
        layer.wo_a           = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT_A,     "weight", i), {n_head * n_embd_head_k / o_groups, o_lora_rank * o_groups});
        layer.wo_b           = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT_B,     "weight", i), {o_groups * o_lora_rank, n_embd});

        layer.hc_attn_fn     = create_tensor(ctx_split, tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim});
        layer.hc_attn_base   = create_tensor(ctx_layer, tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim});
        layer.hc_attn_scale  = create_tensor(ctx_layer, tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3});
        layer.hc_ffn_fn      = create_tensor(ctx_split, tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim});
        layer.hc_ffn_base    = create_tensor(ctx_layer, tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim});
        layer.hc_ffn_scale   = create_tensor(ctx_layer, tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3});

        layer.ffn_gate_inp   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP,   "weight", i), {n_embd, n_expert});
        layer.ffn_exp_probs_b= create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",  i), {n_expert});
        layer.ffn_norm       = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd});

        layer.ffn_gate_exps  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert});
        layer.ffn_down_exps  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert});
        layer.ffn_up_exps    = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert});

        layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared});
        layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    });
        layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared});

        if (i == 0) {
            layer.dspark_main_proj = create_tensor(ctx_split, tn(LLM_TENSOR_DSPARK_MAIN_PROJ, "weight", i), {n_capture * n_embd, n_embd});
            layer.dspark_main_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_DSPARK_MAIN_NORM, "weight", i), {n_embd});
        }
        if (i == last) {
            layer.dspark_norm      = create_tensor(ctx_layer, tn(LLM_TENSOR_DSPARK_NORM,      "weight", i), {n_embd});
            // one output head for the whole drafter -> model level, matching DS4
            model.hc_head_fn       = create_tensor(ctx_split, tn(LLM_TENSOR_HC_HEAD_FN,       "weight", i), {hc_dim, hc_mult});
            model.hc_head_base     = create_tensor(ctx_layer, tn(LLM_TENSOR_HC_HEAD_BASE,     "weight", i), {hc_mult});
            model.hc_head_scale    = create_tensor(ctx_layer, tn(LLM_TENSOR_HC_HEAD_SCALE,    "weight", i), {1});
            layer.dspark_markov_w1 = create_tensor(ctx_split, tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight", i), {markov_rank, n_vocab});
            layer.dspark_markov_w2 = create_tensor(ctx_split, tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight", i), {markov_rank, n_vocab});
            layer.dspark_conf_proj = create_tensor(ctx_split, tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight", i), {n_embd + markov_rank, 1});
        }
    }

    return use_mmap_buffer;
}

bool create_tensors_helper::create_glm4_moe_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t n_expert_shared = hparams.n_expert_shared;

    GGML_ASSERT(hparams.n_expert > 0 && "n_expert must be > 0 for GLM4_MOE MoE layers");
    GGML_ASSERT(hparams.n_expert_used > 0 && "n_expert_used must be > 0 for GLM4_MOE MoE layers");

    create_embd_output(tn, n_embd, n_vocab, true);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        const bool is_mtp_layer = hparams.nextn_predict_layers > 0 &&
                                  static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers;

        if (is_mtp_layer) {
            ctx_split = ctx_layer;
        }

        int flags = 0;
        // Skip loading MTP layers if the feature is disabled
        if (!model.mtp) {
            if (is_mtp_layer) {
                // skip all tensors in the NextN layers
                flags |= llama_model_loader::TENSOR_SKIP;
            }
        }

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);

        // GLM-style attention with bias terms
        if (!flags) {
            use_mmap_buffer &= !merge_qkv(tn, i, 2);
        } else {
            layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, flags);
            layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, flags);
            layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, flags);
            layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q, "bias", i), { n_embd_head_k * n_head }, flags);
            layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K, "bias", i), { n_embd_k_gqa }, flags);
            layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V, "bias", i), { n_embd_v_gqa }, flags);
        }

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, flags);

        // K/Q norm tensors (optional for GLM-4.5 355B variant)
        layer.attn_q_norm = create_tensor(ctx_split,
                tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, llama_model_loader::TENSOR_NOT_REQUIRED | flags);
        layer.attn_k_norm = create_tensor(ctx_split,
                tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, llama_model_loader::TENSOR_NOT_REQUIRED | flags);

        auto ffn_ctx = model.split_mode == LLAMA_SPLIT_MODE_GRAPH ? ctx_split : ctx_layer;

        // Why are we adding an additional tensor type?
        // attn_post_norm is the exact same thing as ffn_norm
        //layer.attn_post_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, flags);
        layer.ffn_norm = create_tensor(ffn_ctx, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, flags);

        // Check if this layer uses MoE or dense FFN based on n_layer_dense_lead
        // GLM 4.5 uses hybrid architecture: layer 0 is dense, layers 1+ are MoE
        const bool use_moe = (static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead);

        if (use_moe) {
            // MoE layers
            layer.ffn_gate_inp = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, flags);
            // gate bias
            layer.ffn_exp_probs_b = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), { n_expert }, flags);

            // MoE branch
            use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i, flags, 0, ffn_ctx);

            // Shared expert
            if (n_expert_shared > 0) {
                const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;
                const int64_t n_ff_shexp = n_ff_exp * n_expert_shared;
                layer.ffn_gate_shexp     = create_tensor(ffn_ctx,
                        tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd, n_ff_shexp }, flags);
                layer.ffn_down_shexp = create_tensor(ffn_ctx,
                        tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { n_ff_shexp, n_embd }, flags);
                layer.ffn_up_shexp = create_tensor(ffn_ctx,
                        tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), { n_embd, n_ff_shexp }, flags);
            }
        } else {
            // Dense layers (first k layers) - GLM uses separate gate/up projections
            layer.ffn_gate = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, flags);
            layer.ffn_down = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, flags);
            layer.ffn_up   = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, flags);
        }
        // --- NextN / MTP tensors on the final layer ---
        if (is_mtp_layer) {
            const int final_layer = n_layer - 1;
            auto nextn_ctx      = ctx_for_layer(final_layer);
            auto nextn_host_ctx = ctx_input;
            layer.nextn.eh_proj          = create_tensor(nextn_ctx,
                    tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", final_layer),
                    { 2*n_embd, n_embd },
                    flags);
            layer.nextn.embed_tokens     = create_tensor(nextn_host_ctx,
                    tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", final_layer),
                    { n_embd, n_vocab },
                    flags | llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.nextn.enorm            = create_tensor(nextn_ctx,
                    tn(LLM_TENSOR_NEXTN_ENORM, "weight", final_layer),
                    { n_embd },
                    flags);
            layer.nextn.hnorm            = create_tensor(nextn_ctx,
                    tn(LLM_TENSOR_NEXTN_HNORM, "weight", final_layer),
                    { n_embd },
                    flags);
            layer.nextn.shared_head_head = create_tensor(nextn_ctx,
                    tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", final_layer),
                    { n_embd, n_vocab },
                    flags | llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(nextn_ctx,
                    tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", final_layer),
                    { n_embd },
                    flags | llama_model_loader::TENSOR_NOT_REQUIRED);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_bitnet_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED); // same as tok_embd, duplicated to allow offloading
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_sub_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_SUB_NORM, "weight", i), {n_embd});

        layer.wq       = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd});
        layer.wq_scale = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "scale", i), {1},  llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.wk       = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa});
        layer.wk_scale = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "scale", i), {1},  llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.wv       = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa});
        layer.wv_scale = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "scale", i), {1},  llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.wo       = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.wo_scale = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "scale", i), {1},  llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_sub_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_SUB_NORM, "weight", i), {n_ff});

        layer.ffn_gate       = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff});
        layer.ffn_gate_scale = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "scale", i), {1},  llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down       = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_scale = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "scale", i), {1},  llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_up         = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
        layer.ffn_up_scale   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP, "scale", i), {1},  llama_model_loader::TENSOR_NOT_REQUIRED);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_bitnet2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);

        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.attn_sub_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_SUB_NORM, "weight", i), {n_embd});
        layer.ffn_sub_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_SUB_NORM, "weight", i), {n_ff});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});

        // optional bias tensors
        layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},     llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        layer.rope_freqs = create_tensor(ctx_layer, tn(LLM_TENSOR_ROPE_FREQS, "weight"), {n_rot/2}, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));

        if (n_expert == 0) {
            layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
            layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
            layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});

            // optional MLP bias
            layer.ffn_gate_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        } else {
            layer.ffn_gate_inp = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert});

            layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED);
            if (layer.ffn_gate_exps) {
                layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert});
                layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert});
            } else {
                // merge split expert into a single tensor for compatibility with older models
                // requires disabling mmap
                use_mmap_buffer = false;

                ggml_type type_gate = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_GATE_EXP, "weight", i, 0).c_str())->type;
                ggml_type type_down = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_DOWN_EXP, "weight", i, 0).c_str())->type;
                ggml_type type_up   = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_UP_EXP,   "weight", i, 0).c_str())->type;

                layer.ffn_gate_exps = ggml_new_tensor_3d(ctx_split, type_gate, n_embd,   n_ff, n_expert);
                layer.ffn_down_exps = ggml_new_tensor_3d(ctx_split, type_down,   n_ff, n_embd, n_expert);
                layer.ffn_up_exps   = ggml_new_tensor_3d(ctx_split, type_up,   n_embd,   n_ff, n_expert);

                ggml_set_name(layer.ffn_gate_exps, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i).c_str());
                ggml_set_name(layer.ffn_down_exps, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i).c_str());
                ggml_set_name(layer.ffn_up_exps,   tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i).c_str());

                for (uint32_t x = 0; x < n_expert; ++x) {
                    // the individual experts are loaded into a view of the merged tensor
                    ml.create_tensor_as_view(ctx_split, layer.ffn_gate_exps, tn(LLM_TENSOR_FFN_GATE_EXP, "weight", i, x), { n_embd, n_ff }, layer.ffn_gate_exps->nb[2]*x);
                    ml.create_tensor_as_view(ctx_split, layer.ffn_down_exps, tn(LLM_TENSOR_FFN_DOWN_EXP, "weight", i, x), { n_ff, n_embd }, layer.ffn_down_exps->nb[2]*x);
                    ml.create_tensor_as_view(ctx_split, layer.ffn_up_exps,   tn(LLM_TENSOR_FFN_UP_EXP,   "weight", i, x), { n_embd, n_ff }, layer.ffn_up_exps->nb[2]*x);
                }
            }
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_t5_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    const auto n_rel_attn_bkts = hparams.n_rel_attn_bkts;

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm_enc = create_tensor(ctx_output, tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm     = create_tensor(ctx_output, tn(LLM_TENSOR_DEC_OUTPUT_NORM, "weight"), {n_embd});

        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm_enc  = create_tensor(ctx_layer, tn(LLM_TENSOR_ENC_ATTN_NORM,  "weight", i), {n_embd});
        layer.attn_rel_b_enc = create_tensor(ctx_input, tn(LLM_TENSOR_ENC_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wq_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wk_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wv_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
        layer.wo_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd});

        layer.ffn_norm_enc = create_tensor(ctx_layer, tn(LLM_TENSOR_ENC_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_gate_enc = create_tensor(ctx_layer, tn(LLM_TENSOR_ENC_FFN_GATE, "weight", i), {n_embd,   n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_FFN_DOWN, "weight", i), {  n_ff, n_embd});
        layer.ffn_up_enc   = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_FFN_UP,   "weight", i), {n_embd,   n_ff});

        layer.attn_norm  = create_tensor(ctx_layer, tn(LLM_TENSOR_DEC_ATTN_NORM,  "weight", i), {n_embd});
        layer.attn_rel_b = create_tensor(ctx_input, tn(LLM_TENSOR_DEC_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd});

        layer.attn_norm_cross  = create_tensor(ctx_layer, tn(LLM_TENSOR_DEC_CROSS_ATTN_NORM,  "weight", i), {n_embd});
        // this tensor seems to be unused in HF transformers implementation
        layer.attn_rel_b_cross = create_tensor(ctx_input, tn(LLM_TENSOR_DEC_CROSS_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wq_cross = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_CROSS_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wk_cross = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_CROSS_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wv_cross = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_CROSS_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
        layer.wo_cross = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_CROSS_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_DEC_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_gate = create_tensor(ctx_layer, tn(LLM_TENSOR_DEC_FFN_GATE, "weight", i), {n_embd,   n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_FFN_DOWN, "weight", i), {  n_ff, n_embd});
        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_FFN_UP,   "weight", i), {n_embd,   n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_tsencoder_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    const auto n_rel_attn_bkts = hparams.n_rel_attn_bkts;

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm_enc = create_tensor(ctx_output, tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd});
        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm_enc  = create_tensor(ctx_layer, tn(LLM_TENSOR_ENC_ATTN_NORM,  "weight", i), {n_embd});
        layer.attn_rel_b_enc = create_tensor(ctx_input, tn(LLM_TENSOR_ENC_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wq_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wk_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wv_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
        layer.wo_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd});

        layer.ffn_norm_enc = create_tensor(ctx_layer, tn(LLM_TENSOR_ENC_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_gate_enc = create_tensor(ctx_layer, tn(LLM_TENSOR_ENC_FFN_GATE, "weight", i), {n_embd,   n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_FFN_DOWN, "weight", i), {  n_ff, n_embd});
        layer.ffn_up_enc   = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_FFN_UP,   "weight", i), {n_embd,   n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_jais_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // Output
    {
        model.output_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
        model.output        = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,   "bias", i),   {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa});

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

        layer.ffn_gate   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE,   "weight", i), {n_embd, n_ff});
        layer.ffn_gate_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE,   "bias", i),   {n_ff});

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_chatglm_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd   = create_tensor(ctx_input,  tn(LLM_TENSOR_TOKEN_EMBD,      "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output        = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + (hparams.n_embd_head_k(i) << 2)});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + (hparams.n_embd_head_k(i) << 2)});

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff * 2});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_cohere2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab, true);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = model.layers[i];
        ggml_context * ctx_split = ctx_for_layer_split(i);

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        create_std_attn(i, tn, layer, n_embd, n_embd_gqa, ctx_split);
        create_std_ffn (i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_cohere2_moe_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    model.output = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
    if (model.output == nullptr) {
        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = model.layers[i];
        ggml_context * ctx_split = ctx_for_layer_split(i);

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});

        layer.bq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd_head_k * n_head}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_k_gqa},          llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_v_gqa},          llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},                llama_model_loader::TENSOR_NOT_REQUIRED);

        if (i < (int) hparams.n_layer_dense_lead) {
            create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
        } else {
            layer.ffn_gate_inp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i, 0, hparams.n_ff_exp);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_glm4_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd   = create_tensor(ctx_input,  tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (model.output == NULL) {
        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);

        if (layer.wqkv == nullptr) {
            layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
            layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
            layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
            layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        }

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

        layer.attn_post_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff * 2}, 0);

        layer.ffn_post_norm  = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_dots1_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = model.layers[i];
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            layer.ffn_gate_inp = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED);
            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }
            // MoE branch
            use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i);
            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_bailingmoe2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    GGML_ASSERT(n_expert > 0 && "n_expert must be > 0 for bailingmoe2");
    GGML_ASSERT(n_expert_used > 0 && "n_expert_used must be > 0 for bailingmoe2");

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = model.layers[i];
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        const bool is_mtp_layer = hparams.nextn_predict_layers > 0 &&
                                  static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers;

        int flags = 0;
        // PXA_BAILING_MTP: only SKIP the NextN tail tensors when MTP is DISABLED. With -mtp on
        // (model.mtp), the tail's attention + MoE + nextn head must actually load so the speculative
        // draft graph (build_bailingmoe2_mtp) can run — mirrors create_glm4_moe_tensors /
        // create_qwen35moe_tensors which gate the SKIP on !model.mtp.
        if (!model.mtp && is_mtp_layer) {
            // skip all tensors in the NextN layers
            flags |= llama_model_loader::TENSOR_SKIP;
        }

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, flags);
        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, flags);

        layer.attn_q_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, flags);
        layer.attn_k_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, flags);

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        if (static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead) { // MoE layers
            const int64_t n_ff_shexp = (hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff_exp) * n_expert_shared;

            layer.ffn_gate_inp = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert},
                    llama_model_loader::TENSOR_NOT_REQUIRED | flags);

            use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i, flags);

            layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, flags);
            layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, flags);
        } else { // Dense layers
            layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, flags);
            layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, flags);
        }

        // NextN/MTP tensors (preserved but unused) - conditionally load for last nextn_predict_layers
        if (is_mtp_layer) {
            layer.nextn.eh_proj          = create_tensor(ctx_split, tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), { 2 * n_embd, n_embd }, flags);
            layer.nextn.embed_tokens     = create_tensor(ctx_split, tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", i), { n_embd, n_vocab },
                    llama_model_loader::TENSOR_NOT_REQUIRED | flags);
            layer.nextn.enorm            = create_tensor(ctx_layer, tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), { n_embd }, flags);
            layer.nextn.hnorm            = create_tensor(ctx_layer, tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), { n_embd }, flags);
            layer.nextn.shared_head_head = create_tensor(ctx_split, tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab }, llama_model_loader::TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, llama_model_loader::TENSOR_NOT_REQUIRED | flags);
            // PXA_BAILING_MTP: layer_out_norm is not part of the upstream bailing NextN tensor set and
            // is unused by build_bailingmoe2_mtp; mark NOT_REQUIRED so a Ring GGUF that omits it loads.
            layer.layer_out_norm         = create_tensor(ctx_layer, tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED | flags);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_ernie45_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        auto& layer = model.layers[i];
        ggml_context* ctx_split = ctx_for_layer_split(i);

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        // optional bias tensors
        layer.bq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "bias", i), { n_embd }, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "bias", i), { n_embd_gqa }, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "bias", i), { n_embd_gqa }, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

        if (model.arch == LLM_ARCH_ERNIE4_5_MOE && static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead) { // MoE layers
            int n_ff_exp = hparams.n_ff_exp;

            layer.ffn_gate_inp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, 0);
            layer.ffn_exp_probs_b = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), { n_expert }, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd,   n_ff_exp, n_expert }, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp, n_embd, n_expert }, 0);
            layer.ffn_up_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), { n_embd,   n_ff_exp, n_expert }, 0);

            // Shared expert (if present)
            if (hparams.n_ff_shexp > 0) {
                layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd, hparams.n_ff_shexp }, 0);
                layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { hparams.n_ff_shexp, n_embd }, 0);
                layer.ffn_up_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), { n_embd, hparams.n_ff_shexp }, 0);
            }
        }
        else { // Dense layers
            create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_hunyuan_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_inp  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
        use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i, 0, n_ff);

        layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
        layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
        layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, 0);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_openai_moe_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t n_ff_exp = hparams.n_ff_exp;

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_split = ctx_for_layer_split(i);
        ggml_context * ctx_layer = ctx_for_layer(i);
        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_norm  = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        use_mmap_buffer &= !merge_qkv(tn, i, 2);

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_rot, n_embd}, 0);
        layer.bo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd}, 0);

        layer.attn_sinks = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_SINKS, "weight", i), {n_head}, 0);

        ggml_context *ctx_ffn_gate, *ctx_ffn_up, *ctx_ffn_down;
        layer.ffn_gate_inp  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {  n_embd, n_expert}, 0);
        bool merged = false;
        auto ug_name = tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i);
        auto ug_meta = ml.get_tensor_meta(ug_name.c_str());
        if (ug_meta) {
            auto ug_name_b = tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "bias", i);
            auto ug_meta_b = ml.get_tensor_meta(ug_name_b.c_str());
            GGML_ASSERT(ug_meta_b);
            layer.ffn_up_gate_exps   = create_tensor(ctx_split, ug_name, { ug_meta->ne[0], ug_meta->ne[1], ug_meta->ne[2] }, 0);
            layer.ffn_up_gate_exps_b = create_tensor(ctx_split, ug_name_b, { ug_meta_b->ne[0], ug_meta_b->ne[1], ug_meta_b->ne[2] }, 0);
            ctx_ffn_gate = ctx_ffn_up = ctx_split;
        } else {
            merged = ml.merge_up_gate_exps && merge_up_gate_exps(tn, i, 2);
            use_mmap_buffer &= !merged;
            if (merged) {
                ctx_ffn_gate = ctx_ffn_up = ctx_split;
            } else {
                layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i),
                        {  n_embd, n_ff_exp, n_expert}, 0, &ctx_ffn_up);
                layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i),
                        {  n_embd, n_ff_exp, n_expert}, 0, &ctx_ffn_gate);
            }
        }
        layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i),
                {n_ff_exp, n_embd, n_expert}, 0, &ctx_ffn_down);

        // bias
        layer.ffn_gate_inp_b  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP,  "bias", i), {n_expert}, 0);
        ggml_context *ctx_ffn_gate_b, *ctx_ffn_up_b, *ctx_ffn_down_b;
        auto ctx_gate_b = ctx_ffn_gate == ctx_split ? ctx_split : ctx_layer;
        auto ctx_down_b = ctx_ffn_down == ctx_split ? ctx_split : ctx_layer;
        auto ctx_up_b   = ctx_ffn_up   == ctx_split ? ctx_split : ctx_layer;
        if (!ug_meta && !merged) {
            layer.ffn_up_exps_b   = create_tensor(ctx_up_b,   tn(LLM_TENSOR_FFN_UP_EXPS,   "bias", i), {n_ff_exp, n_expert}, 0, &ctx_ffn_up_b);
            layer.ffn_gate_exps_b = create_tensor(ctx_gate_b, tn(LLM_TENSOR_FFN_GATE_EXPS, "bias", i), {n_ff_exp, n_expert}, 0, &ctx_ffn_gate_b);
        }
        layer.ffn_down_exps_b = create_tensor(ctx_down_b, tn(LLM_TENSOR_FFN_DOWN_EXPS, "bias", i), {  n_embd, n_expert}, 0, &ctx_ffn_down_b);

        if (!merged && ctx_ffn_gate_b != ctx_ffn_gate) {
            layer.ffn_gate_exps_b_dup = create_tensor(ctx_ffn_gate, tn(LLM_TENSOR_FFN_GATE_EXPS, "bias", i), {n_ff_exp, n_expert},
                    llama_model_loader::TENSOR_DUPLICATED);
        }
        if (!merged && ctx_ffn_up_b != ctx_ffn_up) {
            layer.ffn_up_exps_b_dup = create_tensor(ctx_ffn_up, tn(LLM_TENSOR_FFN_UP_EXPS,   "bias", i), {n_ff_exp, n_expert},
                    llama_model_loader::TENSOR_DUPLICATED);
        }
        if (ctx_ffn_down_b != ctx_ffn_down) {
            layer.ffn_down_exps_b_dup = create_tensor(ctx_ffn_down, tn(LLM_TENSOR_FFN_DOWN_EXPS, "bias", i), {  n_embd, n_expert},
                    llama_model_loader::TENSOR_DUPLICATED);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_minimaxm2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context* ctx_split = ctx_for_layer_split(i);
        auto& layer = model.layers[i];

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_norm   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
        layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k * n_head }, 0);
        layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_k_gqa }, 0);

        layer.ffn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

        layer.ffn_gate_inp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, 0);
        use_mmap_buffer &= !create_std_ffn_exps(n_embd, tn, i, 0, n_ff);
        layer.ffn_exp_probs_b = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), { n_expert }, 0);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_smollm3_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context* ctx_layer = ctx_for_layer(i);
        ggml_context* ctx_split = ctx_for_layer_split(i);
        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        use_mmap_buffer &= !merge_qkv(tn, i, 0);

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::merge_up_gate_exps(const LLM_TN & tn, int i, int bias) {
    // Merging ffn_up_exps + ffn_gate_exps into one tensor is only ever useful to the fused
    // GGML_OP_MOE_FUSED_UP_GATE kernel, and it flips llm_build_moe_ffn onto the merged branch
    // (src/llama-build-context.cpp), which builds that node. In a build that has neither the ik
    // CPU kernel nor a CUDA backend the node is unexecutable, so -muge converted a working, if
    // slower, graph into one that used to abort at decode. ggml_moe_up_gate_can_fuse() returns
    // false for EVERY type pair in such a build, so a same-type probe is a sound capability test
    // -- and it costs nothing to decline, since the merge is a pure speed optimisation of a kernel
    // that is not present.
    if (!ggml_moe_up_gate_can_fuse(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)) {
        LLAMA_LOG_INFO("%s: not merging up/gate -- this build has no fused MoE up/gate implementation\n", __func__);
        return false;
    }
    ggml_context * ctx_split = ctx_for_layer_split(i);

    auto & layer = model.layers[i];

    auto u_name = tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i);
    auto g_name = tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i);
    auto u_meta = ml.require_tensor_meta(u_name.c_str());
    auto g_meta = ml.require_tensor_meta(g_name.c_str());

    if (u_meta->type != g_meta->type || u_meta->ne[0] != g_meta->ne[0] || u_meta->ne[2] != g_meta->ne[2]) {
        LLAMA_LOG_INFO("%s: not merging because up/gate meta info is different\n", __func__);
        return false;
    }

    auto u_ctx = get_context_for_tensor(ctx_split, u_name);
    auto g_ctx = get_context_for_tensor(ctx_split, g_name);

    if (u_ctx != g_ctx) {
        LLAMA_LOG_INFO("%s: not merging because of context\n", __func__);
        return false;
    }

    if (bias && (u_ctx != ctx_split || g_ctx != ctx_split)) {
        LLAMA_LOG_INFO("%s: not merging because of context\n", __func__);
        return false;
    }

    LLAMA_LOG_INFO("%s: merging up/gate in layer %d\n", __func__, i);

    layer.ffn_up_gate_exps = ggml_new_tensor_3d(u_ctx, u_meta->type, u_meta->ne[0], u_meta->ne[1] + g_meta->ne[1], u_meta->ne[2]);
    snprintf(layer.ffn_up_gate_exps->name, GGML_MAX_NAME, "blk.%d.ffn_gate_up_exps.weight", i);
    if (u_ctx == ctx_split) {
        split_tensors.insert(layer.ffn_up_gate_exps);
    }
    layer.ffn_gate_exps = ml.create_tensor_as_view(u_ctx, layer.ffn_up_gate_exps, g_name.c_str(),
            { g_meta->ne[0], g_meta->ne[1], g_meta->ne[2] }, 0);
    layer.ffn_up_exps   = ml.create_tensor_as_view(u_ctx, layer.ffn_up_gate_exps, u_name.c_str(),
            { u_meta->ne[0], u_meta->ne[1], u_meta->ne[2] }, ggml_nbytes(layer.ffn_gate_exps));

    if (!bias) return true;

    auto u_name_b = tn(LLM_TENSOR_FFN_UP_EXPS,   "bias", i);
    auto g_name_b = tn(LLM_TENSOR_FFN_GATE_EXPS, "bias", i);
    auto u_meta_b = ml.get_tensor_meta(u_name_b.c_str());
    auto g_meta_b = ml.get_tensor_meta(g_name_b.c_str());
    if (bias == 2) {
        GGML_ASSERT(u_meta_b && g_meta_b);
        GGML_ASSERT(u_meta_b->type == g_meta_b->type);
        GGML_ASSERT(u_meta_b->ne[1] == g_meta_b->ne[1]);
    } else {
        GGML_ASSERT(!u_meta_b && !g_meta_b);
        return true;
    }

    GGML_ASSERT(u_meta->ne[1] == u_meta_b->ne[0]);
    GGML_ASSERT(g_meta->ne[1] == g_meta_b->ne[0]);

    layer.ffn_up_gate_exps_b = ggml_new_tensor_2d(ctx_split, u_meta_b->type, u_meta_b->ne[0] + g_meta_b->ne[0], u_meta->ne[1]);
    snprintf(layer.ffn_up_gate_exps_b->name, GGML_MAX_NAME, "blk.%d.ffn_gate_up_exps.bias", i);
    layer.ffn_gate_exps_b = ml.create_tensor_as_view(ctx_split, layer.ffn_up_gate_exps_b, g_name_b.c_str(),
            { g_meta_b->ne[0], g_meta_b->ne[1] }, 0);
    layer.ffn_up_exps_b   = ml.create_tensor_as_view(ctx_split, layer.ffn_up_gate_exps_b, u_name_b.c_str(),
            { u_meta_b->ne[0], u_meta_b->ne[1] }, ggml_nbytes(layer.ffn_gate_exps_b));

    return true;
}

bool create_tensors_helper::create_std_ffn_exps(int64_t n_embd, const LLM_TN & tn, int i, int flags, int n_ff_exps_input,
        ggml_context * ffn_ctx) {
    const int64_t n_expert      = model.hparams.n_expert;
    const int64_t n_expert_used = model.hparams.n_expert_used;
    const int64_t n_ff     = model.hparams.n_ff();
    const int64_t n_ff_exp = n_ff_exps_input > 0 ? n_ff_exps_input : model.hparams.n_ff_exp ? model.hparams.n_ff_exp : n_ff / n_expert_used;

    auto & layer = model.layers[i];
    if (!ffn_ctx) {
        ffn_ctx = ctx_for_layer_split(i);
    }

    bool merged = false;
    auto ug_name = tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i);
    auto ug_meta = ml.get_tensor_meta(ug_name.c_str());
    //printf("Checking for tensor %s: %s\n", ug_name.c_str(), ug_meta ? "found" : "not found");
    if (ug_meta) {
        layer.ffn_up_gate_exps = create_tensor(ffn_ctx, ug_name, {  n_embd, 2*n_ff_exp, n_expert}, flags);
    } else {
        merged = flags == 0 && ml.merge_up_gate_exps && merge_up_gate_exps(tn, i, 0);
        if (!merged) {
            layer.ffn_up_exps   = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_gate_exps = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);
        }
    }
    layer.ffn_down_exps = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, flags);

    return merged;
}

bool create_tensors_helper::merge_qkv(const LLM_TN & tn, int i, int bias, bool ignore_attn_scale) {
    auto& hparams = model.hparams;
    const int64_t n_head        = hparams.n_head(i);
    const int64_t n_head_kv     = hparams.n_head_kv(i);
    const int64_t n_embd        = hparams.n_embd / (hparams.n_deepstack_layers + 1); // For Qwen3-VL we need to divide by the number of deepstack layers + 1, for other models n_deepstack_layers value is 0 by default
    const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa(i);
    const int64_t n_embd_head_k = hparams.n_embd_head_k(i);
    const int64_t n_embd_gqa    = n_embd_v_gqa;

    ggml_context * ctx_layer = ctx_for_layer(i);
    ggml_context * ctx_split = ctx_for_layer_split(i);

    auto & layer = model.layers[i];

    auto wq_name = tn(LLM_TENSOR_ATTN_Q, "weight", i);
    auto wk_name = tn(LLM_TENSOR_ATTN_K, "weight", i);
    auto wv_name = tn(LLM_TENSOR_ATTN_V, "weight", i);
    auto wq = ml.require_tensor_meta(wq_name.c_str());
    auto wk = ml.require_tensor_meta(wk_name.c_str());
    auto wv = ml.require_tensor_meta(wv_name.c_str());
    GGML_ASSERT(wq && wk && wv);

    bool fused_qkv = false;
    if (ml.merge_qkv && wq->type == wk->type && wq->type == wv->type && (ignore_attn_scale || hparams.f_attention_scale == 0.0f)) {
        GGML_ASSERT(wq->ne[0] == n_embd && wq->ne[1] == n_head * n_embd_head_k);
        GGML_ASSERT(wk->ne[0] == n_embd && wk->ne[1] == n_embd_gqa);
        GGML_ASSERT(wv->ne[0] == n_embd && wv->ne[1] == n_embd_gqa);
        layer.wqkv = ggml_new_tensor_2d(ctx_split, wq->type, n_embd, n_embd_head_k * (n_head + n_head_kv + n_head_kv));
        snprintf(layer.wqkv->name, GGML_MAX_NAME, "blk.%d.attn_qkv.weight", i);
        // This does not work. If we are doing this merge manually, it basically means that the arch does not have
        // an LLM_TENSOR_ATTN_QKV entry, so we will get __missing__ as the tensor name.
        //ggml_set_name(layer.wqkv, tn(LLM_TENSOR_ATTN_QKV, "weight", i).c_str());
        layer.wq = ml.create_tensor_as_view(ctx_split, layer.wqkv, wq_name.c_str(), { wq->ne[0], wq->ne[1] }, 0);
        layer.wk = ml.create_tensor_as_view(ctx_split, layer.wqkv, wk_name.c_str(), { wk->ne[0], wk->ne[1] }, wq->ne[1]*wq->nb[1]);
        layer.wv = ml.create_tensor_as_view(ctx_split, layer.wqkv, wv_name.c_str(), { wv->ne[0], wv->ne[1] }, wq->ne[1]*wq->nb[1] + wk->ne[1]*wk->nb[1] );
        fused_qkv = true;
        if (bias) {
            auto bq_name = tn(LLM_TENSOR_ATTN_Q, "bias", i);
            auto bk_name = tn(LLM_TENSOR_ATTN_K, "bias", i);
            auto bv_name = tn(LLM_TENSOR_ATTN_V, "bias", i);
            auto bq = ml.get_tensor_meta(bq_name.c_str());
            auto bk = ml.get_tensor_meta(bk_name.c_str());
            auto bv = ml.get_tensor_meta(bv_name.c_str());
            if (bias == 2) {
                GGML_ASSERT(bq && bk && bv);
            } else {
                GGML_ASSERT(!bq && !bk && !bv);
            }
            if (bq && bk && bv) {
                GGML_ASSERT(bq->type == GGML_TYPE_F32 && bk->type == GGML_TYPE_F32 && bv->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_nrows(bq) == 1 && bq->ne[0] == wq->ne[1]);
                GGML_ASSERT(ggml_nrows(bk) == 1 && bk->ne[0] == wk->ne[1]);
                GGML_ASSERT(ggml_nrows(bv) == 1 && bv->ne[0] == wv->ne[1]);
                layer.bqkv = ggml_new_tensor_1d(ctx_layer, bq->type, n_embd_head_k * (n_head + n_head_kv + n_head_kv));
                snprintf(layer.bqkv->name, GGML_MAX_NAME, "blk.%d.attn_qkv.bias", i);
                layer.bq = ml.create_tensor_as_view(ctx_layer, layer.bqkv, bq_name.c_str(), { bq->ne[0] }, 0);
                layer.bk = ml.create_tensor_as_view(ctx_layer, layer.bqkv, bk_name.c_str(), { bk->ne[0] }, bq->ne[0]*bq->nb[0]);
                layer.bv = ml.create_tensor_as_view(ctx_layer, layer.bqkv, bv_name.c_str(), { bv->ne[0] }, bq->ne[0]*bq->nb[0] + bk->ne[0]*bk->nb[0] );
            }
        }
    }
    if (!fused_qkv && ml.merge_qkv && wq->type == wk->type && (ignore_attn_scale || hparams.f_attention_scale == 0.0f)) {
        GGML_ASSERT(wq->ne[0] == n_embd && wq->ne[1] == n_head * n_embd_head_k);
        GGML_ASSERT(wk->ne[0] == n_embd && wk->ne[1] == n_embd_gqa);
        layer.wqk = ggml_new_tensor_2d(ctx_split, wq->type, n_embd, n_embd_head_k * (n_head + n_head_kv));
        snprintf(layer.wqk->name, GGML_MAX_NAME, "blk.%d.attn_qk.weight", i);
        layer.wq = ml.create_tensor_as_view(ctx_split, layer.wqk, wq_name.c_str(), { wq->ne[0], wq->ne[1] }, 0);
        layer.wk = ml.create_tensor_as_view(ctx_split, layer.wqk, wk_name.c_str(), { wk->ne[0], wk->ne[1] }, wq->ne[1]*wq->nb[1]);
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        LLAMA_LOG_INFO("====================== Merged only Q and K in layer %d because V is of different type\n", i);
        fused_qkv = true;
        if (bias) {
            auto bq_name = tn(LLM_TENSOR_ATTN_Q, "bias", i);
            auto bk_name = tn(LLM_TENSOR_ATTN_K, "bias", i);
            auto bv_name = tn(LLM_TENSOR_ATTN_V, "bias", i);
            auto bq = ml.get_tensor_meta(bq_name.c_str());
            auto bk = ml.get_tensor_meta(bk_name.c_str());
            auto bv = ml.get_tensor_meta(bv_name.c_str());
            if (bias == 2) {
                GGML_ASSERT(bq && bk && bv);
            } else {
                GGML_ASSERT(!bq && !bk && !bv);
            }
            if (bq && bk && bv) {
                GGML_ASSERT(bq->type == GGML_TYPE_F32 && bk->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_nrows(bq) == 1 && bq->ne[0] == wq->ne[1]);
                GGML_ASSERT(ggml_nrows(bk) == 1 && bk->ne[0] == wk->ne[1]);
                layer.bqk = ggml_new_tensor_1d(ctx_layer, bq->type, n_embd_head_k * (n_head + n_head_kv));
                snprintf(layer.bqk->name, GGML_MAX_NAME, "blk.%d.attn_qk.bias", i);
                layer.bq = ml.create_tensor_as_view(ctx_layer, layer.bqk, bq_name.c_str(), { bq->ne[0] }, 0);
                layer.bk = ml.create_tensor_as_view(ctx_layer, layer.bqk, bk_name.c_str(), { bk->ne[0] }, bq->ne[0]*bq->nb[0]);
                layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {layer.wv->ne[1]});
            }
        }
    }

    if (!fused_qkv) {
        if (ml.merge_qkv) {
            LLAMA_LOG_INFO("%s: did not merge Q, K, V in layer %d because %d, %d, %d\n", __func__, i,
                    wq->type == wk->type, wq->type == wv->type, (ignore_attn_scale || hparams.f_attention_scale == 0.0f));
        }
        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        if (bias) {
            auto flags = bias == 1 ? llama_model_loader::TENSOR_NOT_REQUIRED : 0;
            layer.bq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {layer.wq->ne[1]}, flags);
            layer.bk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "bias", i), {layer.wk->ne[1]}, flags);
            layer.bv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "bias", i), {layer.wv->ne[1]}, flags);
        }
    }

    return fused_qkv;
}

static void prepare_split_tensors(int split_dim, ggml_context * ctx, ggml_tensor * tensor, llama_split_tensor & split_tensor,
        const std::vector<int> & splits, std::vector<size_t> & mem_used) {
    GGML_ASSERT(split_dim <= 2);
    GGML_ASSERT(splits.size() > 1);
    std::string name{tensor->name};
    split_tensor.tensor_splits.resize(splits.size());
    if (split_dim < 0) {
        for (int i = 0; i < int(splits.size()); ++i) {
            if (splits[i] > 0) {
                split_tensor.tensor_splits[i] = ggml_new_tensor_3d(ctx, tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2]);
                auto name_i = name + '.' + std::to_string(i);
                ggml_set_name(split_tensor.tensor_splits[i], name_i.c_str());
            } else {
                split_tensor.tensor_splits[i] = nullptr;
            }
        }
    }
    else if (split_dim == 1) {
        for (int i = 0; i < int(splits.size()); ++i) {
            if (splits[i] > 0) {
                split_tensor.tensor_splits[i] = ggml_new_tensor_3d(ctx, tensor->type, tensor->ne[0], splits[i], tensor->ne[2]);
                auto name_i = name + '.' + std::to_string(i);
                ggml_set_name(split_tensor.tensor_splits[i], name_i.c_str());
            } else {
                split_tensor.tensor_splits[i] = nullptr;
            }
        }
    }
    else if (split_dim == 2) {
        for (int i = 0; i < int(splits.size()); ++i) {
            if (splits[i] > 0) {
                split_tensor.tensor_splits[i] = ggml_new_tensor_3d(ctx, tensor->type, tensor->ne[0], tensor->ne[1], splits[i]);
                auto name_i = name + '.' + std::to_string(i);
                ggml_set_name(split_tensor.tensor_splits[i], name_i.c_str());
            } else {
                split_tensor.tensor_splits[i] = nullptr;
            }
        }
    } else {
        for (int i = 0; i < int(splits.size()); ++i) {
            if (splits[i] > 0) {
                split_tensor.tensor_splits[i] = ggml_new_tensor_3d(ctx, tensor->type, splits[i], tensor->ne[1], tensor->ne[2]);
                auto name_i = name + '.' + std::to_string(i);
                ggml_set_name(split_tensor.tensor_splits[i], name_i.c_str());
            } else {
                split_tensor.tensor_splits[i] = nullptr;
            }
        }
    }
    split_tensor.ggml.n_device  = splits.size();
    split_tensor.ggml.split_dim = split_dim;
    split_tensor.ggml.splits    = split_tensor.tensor_splits.data();
    tensor->extra = (void *)&split_tensor.ggml;
    GGML_ASSERT(mem_used.size() >= splits.size());
    for (int i = 0; i < split_tensor.ggml.n_device; ++i) {
        if (split_tensor.ggml.splits[i]) {
            //auto nbytes = ggml_nbytes(split_tensor.ggml.splits[i]);
            //printf("mem_used(%s): %8.2f, total: %8.2f\n", split_tensor.ggml.splits[i]->name, nbytes/1024./1024., (mem_used[i] + nbytes)/1024./1024.);
            mem_used[i] += ggml_nbytes(split_tensor.ggml.splits[i]);
        }
    }
}

// MLA tensor distribution for -sm graph / -sm attn.
// q_a/wkv_a_mqa/norms replicated; q_b row-split by Q head; wo row-split.
// wk_b/wv_b are per-head split (split_dim=2) — loaded directly when present
// in the GGUF, or produced per-head-split by llm_prepare_mla()'s materialize
// lambda when only wkv_b is in the GGUF.
static void distribute_mla_tensors_for_split_mode_graph(
        llama_layer & layer,
        const llama_hparams & hparams,
        const std::vector<float> & cur_splits,
        std::vector<size_t> & mem_used,
        ggml_context * ctx_split,
        int il) {
    const std::vector<int> mirror(cur_splits.size(), 1);

    const int n_head        = hparams.n_head(il);
    const int n_embd_head_k = hparams.n_embd_head_k(il);
    const int n_embd_head_v = hparams.n_embd_head_v(il);

    // granularity=4: keeps wo row blocks K-quant-aligned (% 256) and gqa_ratio % 4 == 0 for FA-MMA.
    auto split_heads = create_split(n_head, 4, cur_splits, mem_used);

    // Derive per-tensor column/row splits from head splits.
    auto split_wq_b_cols  = split_heads;
    for (auto & s : split_wq_b_cols)  s *= n_embd_head_k;
    auto split_wo_rows    = split_heads;
    for (auto & s : split_wo_rows)    s *= n_embd_head_v;

    LLAMA_LOG_DEBUG("  MLA layer %d split_heads:", il);
    for ([[maybe_unused]] auto s : split_heads) LLAMA_LOG_DEBUG(" %d", s);
    LLAMA_LOG_DEBUG("\n");

    // Replicated norms (Q-LoRA / KV-LoRA)
    if (layer.attn_q_a_norm) {
        prepare_split_tensors(-1, ctx_split, layer.attn_q_a_norm,  layer.split_attn_q_a_norm,  mirror, mem_used);
    }
    if (layer.attn_kv_a_norm) {
        prepare_split_tensors(-1, ctx_split, layer.attn_kv_a_norm, layer.split_attn_kv_a_norm, mirror, mem_used);
    }

    // Q-side: either wq_a + wq_b (Q-LoRA path, DSV3/K2) or wq directly (DSV2-Lite)
    if (layer.wq_a) {
        prepare_split_tensors(-1, ctx_split, layer.wq_a, layer.split_wq_a, mirror, mem_used);
    }
    if (layer.wq_b) {
        prepare_split_tensors(1, ctx_split, layer.wq_b, layer.split_wq_b, split_wq_b_cols, mem_used);
    } else if (layer.wq) {
        // DSV2-Lite / no-Q-LoRA path: column-split wq directly along the head dim.
        auto split_wq_cols = split_heads;
        for (auto & s : split_wq_cols) s *= n_embd_head_k;
        prepare_split_tensors(1, ctx_split, layer.wq, layer.split_wq, split_wq_cols, mem_used);
    }

    // wkv_a_mqa, wk_b, wv_b replicated: the per-head 3D batched mul_mat can't read a split src0.
    if (layer.wkv_a_mqa) {
        prepare_split_tensors(-1, ctx_split, layer.wkv_a_mqa, layer.split_wkv_a_mqa, mirror, mem_used);
    }
    if (layer.wk_b) {
        prepare_split_tensors( 2, ctx_split, layer.wk_b, layer.split_wk_b, split_heads, mem_used);
    }
    if (layer.wv_b) {
        prepare_split_tensors( 2, ctx_split, layer.wv_b, layer.split_wv_b, split_heads, mem_used);
    }

    // Output projection: row-split, partial outputs all-reduced after.
    if (layer.wo) {
        prepare_split_tensors(0, ctx_split, layer.wo, layer.split_wo, split_wo_rows, mem_used);
    }
    if (layer.ffn_norm) {
        prepare_split_tensors(-1, ctx_split, layer.ffn_norm, layer.split_ffn_norm, mirror, mem_used);
    }
}

static void adjust_split(std::vector<float> & split, const std::vector<size_t> & mem_used, int max_gpu) {
    if (max_gpu < 1 || max_gpu >= int(split.size()) || split.size() != mem_used.size()) {
        return;
    }
    size_t tot_mem_used = 1;
    for (auto & mem : mem_used) tot_mem_used += mem;
    for (int i = split.size() - 1; i > 0; --i) split[i] -= split[i-1];
    std::vector<std::pair<float, int>> sorted(split.size());
    for (int i = 0; i < int(split.size()); ++i) {
        float mem_ideal = split[i]*tot_mem_used;
        float err = mem_ideal - mem_used[i];
        sorted[i] = {err, i};
    }
    std::partial_sort(sorted.begin(), sorted.begin() + max_gpu, sorted.end(), std::greater<std::pair<float,int>>{});
    for (auto & p : split) p = 0;
    for (int j = 0; j < max_gpu; ++j) split[sorted[j].second] = 1;
    float sum = 0;
    for (auto & p : split) {
        sum += p/max_gpu;
        p = sum;
    }
}

static void check_delta_split(ggml_tensor * t, llama_split_tensor & l_split) {
    auto extra = (ggml_split_tensor_t *)t->extra;
    GGML_ASSERT(extra);
    if (extra->split_dim < 0) return;
    GGML_ASSERT(extra->n_device == int(l_split.ranges.size()));
    for (int is = 0; is < extra->n_device; ++is) {
        if (!extra->splits[is]) {
            GGML_ASSERT(l_split.ranges[is].empty());
            continue;
        }
        int ntot = 0;
        for (auto & p : l_split.ranges[is]) ntot += p.second;
        GGML_ASSERT(ntot == extra->splits[is]->ne[extra->split_dim]);
        //auto data = &l_split.ranges[is];
        //std::memcpy(extra->splits[is]->op_params, &data, sizeof(data));
    }
    auto data = &l_split.ranges;
    std::memcpy(t->op_params, &data, sizeof(data));
}

static void prepare_up_gate_split(ggml_tensor * t, llama_split_tensor & split) {
    auto extra = (ggml_split_tensor_t *)t->extra;
    GGML_ASSERT(extra);
    split.ranges.resize(extra->n_device);
    int idim = extra->split_dim;
    int nrows = t->ne[idim]/2;
    int ntot = 0;
    for (int is = 0; is < extra->n_device; ++is) {
        if (!extra->splits[is]) continue;
        auto & ranges = split.ranges[is];
        ranges.resize(2);
        int nrows_is = extra->splits[is]->ne[idim]/2;
        ranges[0] = {ntot,         nrows_is};
        ranges[1] = {ntot + nrows, nrows_is};
        ntot += nrows_is;
    }
    check_delta_split(t, split);
}

// ttype = 0 -> q, k, v, always multiplied with head_k_dim/head_v_dim
// ttype = 1 -> q, k, v, v, always multiplied with head_k_dim/head_v_dim
// ttype = 2 -> v
// ttype = 3 -> v, but multiplied with head_v_dim
// ttype = 4 -> v, v, never multiplied with head_v_dim
static void prepare_delta_split(int ttype, int repeat_type, int num_k_heads, int gqa_ratio, int head_k_dim, int head_v_dim, const std::vector<int> & split,
        ggml_tensor * t, llama_split_tensor & l_split) {
    auto extra = (ggml_split_tensor_t *)t->extra;
    GGML_ASSERT(extra && extra->n_device == int(split.size()));
    l_split.ranges.resize(split.size());
    LLAMA_LOG_DEBUG("================= %s(%s, %d, %d)\n", __func__, t->name, ttype, repeat_type);
    int first = 0;
    for (int is = 0; is < int(split.size()); ++is) {
        int s = split[is];
        if (!s) continue;
        auto & ranges = l_split.ranges[is];
        if (ttype == 0 || ttype == 1) {
            LLAMA_LOG_DEBUG("adding type 0/1 entry %d, %d for split %d\n", first*head_k_dim, s*head_k_dim, is);
            ranges.push_back({first*head_k_dim, s*head_k_dim});
        }
        else if (ttype == 2 || ttype == 3 || ttype == 4) {
            int multiplier = ttype == 3 ? head_v_dim : ttype == 4 ? 2 : 1;
            if (repeat_type == 0) {
                LLAMA_LOG_DEBUG("adding type 2/3/4 entry %d, %d for split %d (repeat type is 0)\n", first*gqa_ratio*multiplier, s*gqa_ratio*multiplier, is);
                ranges.push_back({first*gqa_ratio*multiplier, s*gqa_ratio*multiplier});
            } else {
                for (int j = 0; j < gqa_ratio; ++j) {
                    LLAMA_LOG_DEBUG("adding type 2/3/4 entry %d, %d for split %d (repeat type is 1)\n", (first + j*num_k_heads)*multiplier, s*multiplier, is);
                    ranges.push_back({(first + j*num_k_heads)*multiplier, s*multiplier});
                }
            }
        }
        else {
            GGML_ABORT("Unknown tensor type for delta-net split");
        }
        first += s;
    }
    if (ttype == 2 || ttype == 3 || ttype == 4) {
        check_delta_split(t, l_split);
        return;
    }
    //if (ttype == 4) {
    //    first = num_k_heads*gqa_ratio;
    //    for (int is = 0; is < int(split.size()); ++is) {
    //        int s = split[is];
    //        if (!s) continue;
    //        auto & ranges = l_split.ranges[is];
    //        int multiplier = 1;
    //        if (repeat_type == 0) {
    //            ranges.push_back({first*gqa_ratio*multiplier, s*gqa_ratio*multiplier});
    //            LLAMA_LOG_DEBUG("adding type 4 entry %d, %d for split %d (repeat type is 0)\n", first*gqa_ratio*multiplier, s*gqa_ratio*multiplier, is);
    //        } else {
    //            for (int j = 0; j < gqa_ratio; ++j) {
    //                LLAMA_LOG_DEBUG("adding type 4 entry %d, %d for split %d (repeat type is 1)\n", (first + j*num_k_heads)*multiplier, s*multiplier, is);
    //                ranges.push_back({(first + j*num_k_heads)*multiplier, s*multiplier});
    //            }
    //        }
    //        first += s;
    //    }
    //    check_delta_split(t, l_split);
    //    return;
    //}
    // ttype = 0, 1
    // First we need to add the ranges for k
    first = num_k_heads;
    for (int is = 0; is < int(split.size()); ++is) {
        int s = split[is];
        if (!s) continue;
        auto & ranges = l_split.ranges[is];
        LLAMA_LOG_DEBUG("adding type 0/1 entry %d, %d for split %d\n", first*head_k_dim, s*head_k_dim, is);
        ranges.push_back({first*head_k_dim, s*head_k_dim});
        first += s;
    }
    // Then we need to add the ranges for v
    first = 2*num_k_heads;
    for (int is = 0; is < int(split.size()); ++is) {
        int s = split[is];
        if (!s) continue;
        auto & ranges = l_split.ranges[is];
        int multiplier = ttype == 0 ? head_v_dim : 2*head_v_dim;
        if (repeat_type == 0) {
            LLAMA_LOG_DEBUG("adding type 0/1 entry %d, %d for split %d (repeat type is 0)\n", first*gqa_ratio*multiplier, s*gqa_ratio*multiplier, is);
            ranges.push_back({first*multiplier, s*gqa_ratio*multiplier});
            first += gqa_ratio;
        } else {
            for (int j = 0; j < gqa_ratio; ++j) {
                LLAMA_LOG_DEBUG("adding type 0/1 entry %d, %d for split %d (repeat type is 1)\n", (first + j*num_k_heads)*multiplier, s*multiplier, is);
                ranges.push_back({(first + j*num_k_heads)*multiplier, s*multiplier});
            }
            first += s;
        }
    }
    //if (ttype == 0) {
        check_delta_split(t, l_split);
        return;
    //}
    first = (2 + gqa_ratio)*num_k_heads;
    for (int is = 0; is < int(split.size()); ++is) {
        int s = split[is];
        if (!s) continue;
        auto & ranges = l_split.ranges[is];
        int multiplier = head_v_dim;
        if (repeat_type == 0) {
            LLAMA_LOG_DEBUG("adding type 1 entry %d, %d for split %d (repeat type is 0)\n", first*gqa_ratio*multiplier, s*gqa_ratio*multiplier, is);
            ranges.push_back({first*gqa_ratio*multiplier, s*gqa_ratio*multiplier});
        } else {
            for (int j = 0; j < gqa_ratio; ++j) {
                LLAMA_LOG_DEBUG("adding type 1 entry %d, %d for split %d (repeat type is 1)\n", (first + j*num_k_heads)*multiplier, s*multiplier, is);
                ranges.push_back({(first + j*num_k_heads)*multiplier, s*multiplier});
            }
        }
        first += s;
    }
    check_delta_split(t, l_split);
}

// qwen4exp: MIRROR-replicate the tensors that this arch adds and that no existing field-driven
// branch of the split loop knows about. Mirror == prepare_split_tensors(-1, ...) == a FULL copy
// on every participating device, the same treatment attn_norm / ffn_norm / the nextn fusion
// tensors already get.
//
// WHY MIRROR AND NOT SHARD, per group:
//
//  * hc_{attn,ffn}_{norm,down,up,inject}  - the hyper-connection mixers read and rewrite the
//    WHOLE hc-expanded residual stream (hc_dim = hc * n_embd) immediately before the token mixer
//    and immediately before the MoE. They are the exact analogue of attn_norm / ffn_norm /
//    nextn.eh_proj: input-fusion weights that must produce the FULL hidden on each device before
//    the row-split attention slices its heads. Sharding the down/up bottleneck over its rank axis
//    is possible in principle (column-split down, row-split up, all-reduce over the rank), but
//    that needs a graph-side ggml_reduce that does not exist yet, so the door is left open rather
//    than walked through. Cost of mirroring: ~26 MiB bf16 per layer per device.
//
//  * index_{q,k}_{proj,norm}  - the QSA lightning indexer decides WHICH tokens every head
//    attends to. That selection has to be bit-identical on every device, or the devices attend to
//    different token sets and the concatenated head outputs no longer describe one attention.
//    Sharding index_q_proj over its indexer heads yields partial scores that would need an
//    all-reduce BEFORE the top-k. Mirroring is correct by construction and costs ~3.3 MiB bf16
//    per full-attention layer per device.
//
//  * ple_*  - the PLE side path exists on one layer, is tiny, and also writes the whole
//    residual stream. Same argument as the HC mixers.
//
// DELIBERATELY NOT SPLIT, and not merely "not yet":
//
//  * model.tok_embd_per_layer (per_layer_token_embd, 160 x 320001536, 95.4 GiB bf16). It is a
//    GET_ROWS gather table. prepare_split_tensors can only shard along an ne axis; sharding the
//    row axis would put each token's n-gram row on exactly one device while every other device
//    gathers from a hole, and this fork's split machinery has no all-gather to repair that. It is
//    also destined for host RAM via `-ot per_layer_token_embd=CPU`, which reroutes it to the CPU
//    buft inside get_context_for_tensor() before create_tensor() ever runs - so it never enters
//    split_ctx and the split loop could not see it even if asked. Finally, setting ->extra on it
//    would swap its base pointer for the 0x1000 dummy that ggml_backend_cuda_split_buffer_get_base
//    returns, and split_buffer_set_tensor would then upload nothing at all, silently.
//
//  * model.tok_embd / model.output / model.hc_head_*  - the output group is handled (or
//    deliberately left alone) by the existing generic code at the end of create_tensors(), exactly
//    as for qwen3next and qwen35moe. hc_head_* lives in ctx_output, not a split context.
//
//  * attn_norm / attn_post_norm / ffn_norm / output_norm - this arch has none; the HC norms are
//    what stands in for them, so those generic branches are simply inert here.
static void split_qwen4exp_tensors(llama_layer & layer, const std::vector<int> & mirror,
        std::vector<size_t> & mem_used, ggml_context * ctx_split) {
    auto mirror_one = [&](ggml_tensor * t, llama_split_tensor & dst) {
        // A tensor that is absent on this layer (indexer on a recurrent layer, PLE off the PLE
        // layer) has nothing to replicate. A tensor that already carries ->extra was handled by
        // an earlier branch and must not be re-registered.
        if (t && !t->extra) {
            prepare_split_tensors(-1, ctx_split, t, dst, mirror, mem_used);
        }
    };

    mirror_one(layer.hc_attn_norm,   layer.split_hc_attn_norm);
    mirror_one(layer.hc_attn_down,   layer.split_hc_attn_down);
    mirror_one(layer.hc_attn_up,     layer.split_hc_attn_up);
    mirror_one(layer.hc_attn_inject, layer.split_hc_attn_inject);
    mirror_one(layer.hc_ffn_norm,    layer.split_hc_ffn_norm);
    mirror_one(layer.hc_ffn_down,    layer.split_hc_ffn_down);
    mirror_one(layer.hc_ffn_up,      layer.split_hc_ffn_up);
    mirror_one(layer.hc_ffn_inject,  layer.split_hc_ffn_inject);

    mirror_one(layer.index_q_proj,   layer.split_index_q_proj);
    mirror_one(layer.index_k_proj,   layer.split_index_k_proj);
    mirror_one(layer.index_q_norm,   layer.split_index_q_norm);
    mirror_one(layer.index_k_norm,   layer.split_index_k_norm);

    mirror_one(layer.ple_key,        layer.split_ple_key);
    mirror_one(layer.ple_value,      layer.split_ple_value);
    mirror_one(layer.ple_norm_key,   layer.split_ple_norm_key);
    mirror_one(layer.ple_norm_query, layer.split_ple_norm_query);
    mirror_one(layer.ple_norm_conv,  layer.split_ple_norm_conv);
    mirror_one(layer.ple_conv1d,     layer.split_ple_conv1d);
}

static void split_recurrent_tensors(const llama_hparams & hparams, llama_layer & layer, const std::vector<float> & cur_splits, std::vector<size_t> & mem_used,
        ggml_context * ctx_split, [[maybe_unused]] int il) { //, int repeat_type) {
    int head_k_dim  = hparams.ssm_d_state;
    int num_k_heads = hparams.ssm_n_group;
    int num_v_heads = hparams.ssm_dt_rank;
    int head_v_dim  = hparams.ssm_d_inner / num_v_heads;
    int gqa_ratio   = num_v_heads / num_k_heads;

    GGML_ASSERT(layer.ssm_in || (layer.wqkv && layer.wqkv_gate));
    //int repeat_type = layer.ssm_in ? 0 : 1;
    int repeat_type = layer.ssm_beta_alpha ? 0 : 1;

    {
        // We do not support quantized ssm_dt and ssm_a
        auto tt = ggml_internal_get_type_traits(layer.ssm_dt->type);
        GGML_ASSERT(tt.row_meta_size == 0 && tt.blck_size == 1);
        tt = ggml_internal_get_type_traits(layer.ssm_a->type);
        GGML_ASSERT(tt.row_meta_size == 0 && tt.blck_size == 1);
    }

    int k_head_granularity = 1;
    auto tt = ggml_internal_get_type_traits(layer.ssm_out->type);
    auto eff_head_v_dim = repeat_type == 1 ? head_v_dim : head_v_dim * gqa_ratio;
    if (tt.blck_size > eff_head_v_dim) {
        GGML_ASSERT(tt.blck_size % eff_head_v_dim == 0);
        k_head_granularity = tt.blck_size / eff_head_v_dim;
    } else {
        GGML_ASSERT(eff_head_v_dim % tt.blck_size == 0);
    }
    if (tt.row_meta_size > 0) {
        GGML_ABORT("Quantization types with per row meta data are not supported for the ssm_out tensor when using split mode graph");
    }

    auto split = create_split(num_k_heads, k_head_granularity, cur_splits, mem_used);
    LLAMA_LOG_DEBUG("================ %s(%d)", __func__, il);
    int n_on = 0;
    for (auto & s : split) {
        if (s > 0) ++n_on;
        LLAMA_LOG_DEBUG(" %d", s);
    }
    LLAMA_LOG_DEBUG("\n");
    if (n_on < 2) {
        GGML_ABORT("The configuration results in a single GPU participating in the delta-net tensor split. This is not supported");
    }

    // PXA_REPLICATE_RECURRENT (Lever C): when set, MIRROR-replicate every recurrent delta-net weight
    // (a FULL copy on every participating device, split_dim = -1) instead of head-splitting it. Each
    // device then runs the FULL delta-net (all heads) on its own full state, producing the identical
    // full output, returned as a reduce-OFF container by the builder -> the recurrent layers' expensive
    // cross-device ggml_reduce is eliminated. The participating-device set is the SAME as the head-split
    // path (devices where split[d] > 0). Env-unset => the head-split + delta-split path below, unchanged.
    // NOTE: mirror replication uses prepare_split_tensors(-1, ...) and DOES NOT call prepare_delta_split
    //   (check_delta_split() returns early for split_dim<0, storing no ranges; the CUDA split-buffer's
    //   split_dim<0 path (ggml-cuda.cu ~L976) copies the FULL tensor bytes to each device with no ranges).
    const bool pxa_replicate_recurrent = (getenv("PXA_REPLICATE_RECURRENT") != nullptr);
    std::vector<int> pxa_mirror(split.size(), 0);
    for (size_t d = 0; d < split.size(); ++d) pxa_mirror[d] = (split[d] > 0) ? 1 : 0;
    // Mirror-replicate one delta-net weight onto every participating device (FULL copy).
    auto pxa_mirror_tensor = [&](ggml_tensor * t, llama_split_tensor & dst) {
        prepare_split_tensors(-1, ctx_split, t, dst, pxa_mirror, mem_used);
    };

    size_t orig_size = 0, split_size = 0;
    auto add_size = [&orig_size, &split_size] (ggml_tensor * t) {
        orig_size += ggml_nbytes(t);
        auto extra = (ggml_split_tensor_t *)t->extra;
        for (int i = 0; i < extra->n_device; ++i) if (extra->splits[i]) split_size += ggml_nbytes(extra->splits[i]);
    };

    // ttype = 0 -> q, k, v, always multiplied with head_k_dim/head_v_dim
    // ttype = 1 -> q, k, v, v, always multiplied with head_k_dim/head_v_dim
    // ttype = 2 -> v
    // ttype = 3 -> v, but multiplied with head_v_dim
    // ttype = 4 -> v, v, never multiplied with head_v_dim

    prepare_split_tensors(-1, ctx_split, layer.ssm_norm, layer.split_ssm_norm, split, mem_used);
    add_size(layer.ssm_norm);

    auto split_k = split;
    for (auto & k : split_k) k *= (head_k_dim*2 + head_v_dim*gqa_ratio);
    if (pxa_replicate_recurrent) {
        pxa_mirror_tensor(layer.ssm_conv1d, layer.split_ssm_conv1d);
    } else {
        prepare_split_tensors( 1, ctx_split, layer.ssm_conv1d, layer.split_ssm_conv1d, split_k, mem_used);
        prepare_delta_split(0, repeat_type, num_k_heads, gqa_ratio, head_k_dim, head_v_dim, split, layer.ssm_conv1d, layer.split_ssm_conv1d);
    }
    add_size(layer.ssm_conv1d);

    if (layer.wqkv) {
        if (pxa_replicate_recurrent) {
            pxa_mirror_tensor(layer.wqkv, layer.split_wqkv);
        } else {
            prepare_split_tensors( 1, ctx_split, layer.wqkv, layer.split_ssm_wqkv, split_k, mem_used);
            prepare_delta_split(0, repeat_type, num_k_heads, gqa_ratio, head_k_dim, head_v_dim, split, layer.wqkv, layer.split_wqkv);
        }
        add_size(layer.wqkv);
    }
    if (layer.ssm_in) {
        split_k = split;
        for (auto & k : split_k) k *= (head_k_dim*2 + head_v_dim*gqa_ratio*2);
        if (pxa_replicate_recurrent) {
            pxa_mirror_tensor(layer.ssm_in, layer.split_ssm_in);
        } else {
            prepare_split_tensors( 1, ctx_split, layer.ssm_in, layer.split_ssm_in, split_k, mem_used);
            prepare_delta_split(1, repeat_type, num_k_heads, gqa_ratio, head_k_dim, head_v_dim, split, layer.ssm_in, layer.split_ssm_in);
        }
        add_size(layer.ssm_in);
    }

    auto split_v = split;
    for (auto & v : split_v) v *= gqa_ratio;

    if (pxa_replicate_recurrent) {
        pxa_mirror_tensor(layer.ssm_dt, layer.split_ssm_dt);
    } else {
        prepare_split_tensors( 0, ctx_split, layer.ssm_dt, layer.split_ssm_dt, split_v, mem_used);
        prepare_delta_split(2, repeat_type, num_k_heads, gqa_ratio, head_k_dim, head_v_dim, split, layer.ssm_dt, layer.split_ssm_dt);
    }
    add_size(layer.ssm_dt);
    if (pxa_replicate_recurrent) {
        pxa_mirror_tensor(layer.ssm_a, layer.split_ssm_a);
    } else {
        prepare_split_tensors( 0, ctx_split, layer.ssm_a,  layer.split_ssm_a,  split_v, mem_used);
        prepare_delta_split(2, repeat_type, num_k_heads, gqa_ratio, head_k_dim, head_v_dim, split, layer.ssm_a, layer.split_ssm_a);
    }
    add_size(layer.ssm_a);
    if (layer.ssm_beta) {
        if (pxa_replicate_recurrent) {
            pxa_mirror_tensor(layer.ssm_beta, layer.split_ssm_beta);
        } else {
            prepare_split_tensors( 1, ctx_split, layer.ssm_beta, layer.split_ssm_beta, split_v, mem_used);
            prepare_delta_split(2, repeat_type, num_k_heads, gqa_ratio, head_k_dim, head_v_dim, split, layer.ssm_beta, layer.split_ssm_beta);
        }
        add_size(layer.ssm_beta);
    }
    if (layer.ssm_alpha) {
        if (pxa_replicate_recurrent) {
            pxa_mirror_tensor(layer.ssm_alpha, layer.split_ssm_alpha);
        } else {
            prepare_split_tensors( 1, ctx_split, layer.ssm_alpha, layer.split_ssm_alpha, split_v, mem_used);
            prepare_delta_split(2, repeat_type, num_k_heads, gqa_ratio, head_k_dim, head_v_dim, split, layer.ssm_alpha, layer.split_ssm_alpha);
        }
        add_size(layer.ssm_alpha);
    }
    if (layer.ssm_beta_alpha) {
        auto split_v2 = split_v;
        for (auto & v : split_v2) v *= 2;
        if (pxa_replicate_recurrent) {
            pxa_mirror_tensor(layer.ssm_beta_alpha, layer.split_ssm_beta_alpha);
        } else {
            prepare_split_tensors( 1, ctx_split, layer.ssm_beta_alpha, layer.split_ssm_beta_alpha, split_v2, mem_used);
            prepare_delta_split(4, repeat_type, num_k_heads, gqa_ratio, head_k_dim, head_v_dim, split, layer.ssm_beta_alpha, layer.split_ssm_beta_alpha);
        }
        add_size(layer.ssm_beta_alpha);
    }

    for (auto & v : split_v) v *= head_v_dim;
    if (pxa_replicate_recurrent) {
        pxa_mirror_tensor(layer.ssm_out, layer.split_ssm_out);
    } else {
        prepare_split_tensors( 0, ctx_split, layer.ssm_out, layer.split_ssm_out, split_v, mem_used);
        prepare_delta_split(3, repeat_type, num_k_heads, gqa_ratio, head_k_dim, head_v_dim, split, layer.ssm_out, layer.split_ssm_out);
    }
    add_size(layer.ssm_out);
    if (layer.wqkv_gate) {
        if (pxa_replicate_recurrent) {
            pxa_mirror_tensor(layer.wqkv_gate, layer.split_wqkv_gate);
        } else {
            prepare_split_tensors( 1, ctx_split, layer.wqkv_gate, layer.split_ssm_wqkv_gate, split_v, mem_used);
            prepare_delta_split(3, repeat_type, num_k_heads, gqa_ratio, head_k_dim, head_v_dim, split, layer.wqkv_gate, layer.split_wqkv_gate);
        }
        add_size(layer.wqkv_gate);
    }
    LLAMA_LOG_DEBUG("    original size: %g MiB, split size: %g MiB\n", orig_size/1024./1024., split_size/1024./1024.);
}

bool create_tensors_helper::create_tensors() {
    const auto tn = LLM_TN(model.arch);
    bool use_mmap_buffer = true;
    if (ml.merge_qkv && (model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN)) {
        LLAMA_LOG_WARN("\n========================================================\n");
        LLAMA_LOG_WARN("merge_qkv is not compatible with split mode 'graph'\n");
        LLAMA_LOG_WARN("  => turning off merge_qkv\n");
        LLAMA_LOG_WARN("========================================================\n\n");
        ml.merge_qkv = false;
    }
    switch (model.arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
            use_mmap_buffer = create_llama_tensors(tn); break;
        case LLM_ARCH_DECI:
            use_mmap_buffer = create_deci_tensors(tn); break;
        case LLM_ARCH_LLAMA4:
            use_mmap_buffer = create_llama4_tensors(tn); break;
        case LLM_ARCH_GROK:
            use_mmap_buffer = create_grok_tensors(tn); break;
        case LLM_ARCH_DBRX:
            use_mmap_buffer = create_dbrx_tensors(tn); break;
        case LLM_ARCH_BAICHUAN:
            use_mmap_buffer = create_baichuan_tensors(tn); break;
        case LLM_ARCH_FALCON:
            use_mmap_buffer = create_falcon_tensors(tn); break;
        case LLM_ARCH_STARCODER:
            use_mmap_buffer = create_starcoder_tensors(tn); break;
        case LLM_ARCH_BERT:
        case LLM_ARCH_NOMIC_BERT:
            use_mmap_buffer = create_bert_tensors(tn); break;
        case LLM_ARCH_JINA_BERT_V2:
            use_mmap_buffer = create_jina_bert2_tensors(tn); break;
        case LLM_ARCH_BLOOM:
            use_mmap_buffer = create_bloom_tensors(tn); break;
        case LLM_ARCH_MPT:
            use_mmap_buffer = create_mpt_tensors(tn); break;
        case LLM_ARCH_STABLELM:
            use_mmap_buffer = create_stablelm_tensors(tn); break;
        case LLM_ARCH_QWEN:
            use_mmap_buffer = create_qwen_tensors(tn); break;
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_QWEN2VL:
            use_mmap_buffer = create_qwen2_tensors(tn); break;
        case LLM_ARCH_QWEN2MOE:
            use_mmap_buffer = create_qwen2_moe_tensors(tn); break;
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN3VL:
            use_mmap_buffer = create_qwen3_tensors(tn); break;
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3VLMOE:
            use_mmap_buffer = create_qwen3_moe_tensors(tn); break;
        case LLM_ARCH_MELLUM:
            use_mmap_buffer = create_mellum_tensors(tn); break;
        case LLM_ARCH_QWEN3NEXT:
            use_mmap_buffer = create_qwen3next_tensors(tn); break;
        case LLM_ARCH_QWEN35MOE:
            use_mmap_buffer = create_qwen35moe_tensors(tn); break;
        case LLM_ARCH_QWEN35:
            use_mmap_buffer = create_qwen35_tensors(tn); break;
        case LLM_ARCH_QWEN4EXP:
            use_mmap_buffer = create_qwen4exp_tensors(tn); break;
        case LLM_ARCH_PHI2:
            use_mmap_buffer = create_phi2_tensors(tn); break;
        case LLM_ARCH_PHI3:
            use_mmap_buffer = create_phi3_tensors(tn); break;
        case LLM_ARCH_PLAMO:
            use_mmap_buffer = create_baichuan_tensors(tn, false); break;
        case LLM_ARCH_GPT2:
            use_mmap_buffer = create_gpt2_tensors(tn); break;
        case LLM_ARCH_CODESHELL:
            use_mmap_buffer = create_codeshell_tensors(tn); break;
        case LLM_ARCH_ORION:
            use_mmap_buffer = create_orion_tensors(tn); break;
        case LLM_ARCH_INTERNLM2:
            use_mmap_buffer = create_internlm_tensors(tn); break;
        case LLM_ARCH_GEMMA:
            use_mmap_buffer = create_gemma_tensors(tn, 1); break;
        case LLM_ARCH_GEMMA2:
            use_mmap_buffer = create_gemma_tensors(tn, 2); break;
        case LLM_ARCH_GEMMA3:
            use_mmap_buffer = create_gemma_tensors(tn, 3); break;
        case LLM_ARCH_GEMMA4:
            use_mmap_buffer = create_gemma4_tensors(tn); break;
        case LLM_ARCH_GEMMA4_MTP:
            use_mmap_buffer = create_gemma4_mtp_tensors(tn); break;
        case LLM_ARCH_STARCODER2:
            use_mmap_buffer = create_starcoder2_tensors(tn); break;
        case LLM_ARCH_MAMBA:
            use_mmap_buffer = create_mamba_tensors(tn); break;
        case LLM_ARCH_XVERSE:
            use_mmap_buffer = create_xverse_tensors(tn); break;
        case LLM_ARCH_COMMAND_R:
            use_mmap_buffer = create_command_r_tensors(tn); break;
        case LLM_ARCH_OLMO:  // adapted from LLM_ARCH_LLAMA with norm params removed
            use_mmap_buffer = create_olmo_tensors(tn); break;
        case LLM_ARCH_OPENELM:
            use_mmap_buffer = create_openelm_tensors(tn); break;
        case LLM_ARCH_GPTNEOX:
            use_mmap_buffer = create_gptneox_tensors(tn); break;
        case LLM_ARCH_ARCTIC:
            use_mmap_buffer = create_arctix_tensors(tn); break;
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_MISTRAL4:
            use_mmap_buffer = create_deepseek2_tensors(tn); break;
        case LLM_ARCH_GLM_DSA:
            use_mmap_buffer = create_glm_dsa_tensors(tn); break;
        case LLM_ARCH_DEEPSEEK4:
            use_mmap_buffer = create_deepseek4_tensors(tn); break;
        case LLM_ARCH_DEEPSEEK4_DSPARK:
            use_mmap_buffer = create_dspark_tensors(tn); break;
        case LLM_ARCH_GLM4_MOE:
            use_mmap_buffer = create_glm4_moe_tensors(tn); break;
        case LLM_ARCH_BITNET:
            use_mmap_buffer = create_bitnet_tensors(tn); break;
        case LLM_ARCH_BITNET_B158:
        case LLM_ARCH_BITNET_25:
            use_mmap_buffer = create_bitnet2_tensors(tn); break;
        case LLM_ARCH_T5:
            use_mmap_buffer = create_t5_tensors(tn); break;
        case LLM_ARCH_T5ENCODER:
            use_mmap_buffer = create_tsencoder_tensors(tn); break;
        case LLM_ARCH_JAIS:
            use_mmap_buffer = create_jais_tensors(tn); break;
        case LLM_ARCH_CHATGLM:
            use_mmap_buffer = create_chatglm_tensors(tn); break;
        case LLM_ARCH_COHERE2:
            use_mmap_buffer = create_cohere2_tensors(tn); break;
        case LLM_ARCH_COHERE2_MOE:
            use_mmap_buffer = create_cohere2_moe_tensors(tn); break;
        case LLM_ARCH_GLM4:
            use_mmap_buffer = create_glm4_tensors(tn); break;
        case LLM_ARCH_DOTS1:
            use_mmap_buffer = create_dots1_tensors(tn); break;
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
            use_mmap_buffer = create_ernie45_tensors(tn); break;
        case LLM_ARCH_HUNYUAN_MOE:
            use_mmap_buffer = create_hunyuan_tensors(tn); break;
        case LLM_ARCH_OPENAI_MOE:
            use_mmap_buffer = create_openai_moe_tensors(tn); break;
        case LLM_ARCH_BAILINGMOE2:
            use_mmap_buffer = create_bailingmoe2_tensors(tn); break;
        case LLM_ARCH_MINIMAX_M2:
            use_mmap_buffer = create_minimaxm2_tensors(tn); break;
        case LLM_ARCH_SMOLLM3:
            use_mmap_buffer = create_smollm3_tensors(tn); break;
        case LLM_ARCH_MIMO2:
            use_mmap_buffer = create_mimo2_tensors(tn); break;
        case LLM_ARCH_SEED_OSS:
            use_mmap_buffer = create_seedoss_tensors(tn); break;
        case LLM_ARCH_STEP35:
            use_mmap_buffer = create_step35_tensors(tn); break;
        case LLM_ARCH_LAGUNA:
            use_mmap_buffer = create_laguna_tensors(tn); break;
        case LLM_ARCH_HY_V3:
            use_mmap_buffer = create_hy3_tensors(tn); break;
        case LLM_ARCH_MUSE_GLIMMER:
            use_mmap_buffer = create_muse_glimmer_tensors(tn); break;
        default:
            throw std::runtime_error("unknown architecture");
    }

    if (getenv("GGML_CUDA_NO_PINNED") == nullptr) {
        use_mmap_buffer &= !has_buft_overrides;
    }

    {
        const bool unsupported =
            (model.arch == LLM_ARCH_GEMMA4_MTP) ||
            (model.arch == LLM_ARCH_GEMMA4 && model.tok_embd_per_layer);
        if (unsupported && (model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN)) {
            LLAMA_LOG_WARN("\n=========================================================\n");
            LLAMA_LOG_WARN("Split mode 'graph' is not supported for %s\n",
                           model.arch == LLM_ARCH_GEMMA4_MTP ? "Gemma 4 MTP assistant"
                                                              : "this Gemma4 variant");
            LLAMA_LOG_WARN("  => changing split mode to 'layer'\n");
            LLAMA_LOG_WARN("===========================================================\n\n");
            model.split_mode = LLAMA_SPLIT_MODE_LAYER;
        }
    }

    if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN) {
        const int n_layer = model.mtp ? model.layers.size()
                                  : model.layers.size() - model.hparams.nextn_predict_layers;
        LLAMA_LOG_INFO("================================ max_gpu = %d\n", model.max_gpu);
        std::vector<int> mirror(model.splits.size(), 1);
        std::vector<size_t> mem_used(model.splits.size(), 0);
        const auto & hparams = model.hparams;
        auto cur_splits = model.splits;
        int adjust_step = std::max(1, int(n_layer / (2*model.splits.size())));
        if (model.max_gpu > 1 && model.max_gpu < int(cur_splits.size())) {
            bool equal_split = true;
            for (int i = 0; i < int(cur_splits.size()); ++i) {
                float p = i > 0 ? cur_splits[i] - cur_splits[i-1] : cur_splits[i];
                if (std::abs(p*cur_splits.size() - 1.f) > 0.02f) {
                    equal_split = false; break;
                }
            }
            if (equal_split) {
                if (cur_splits.size() % model.max_gpu == 0) {
                    int nadj = cur_splits.size()/model.max_gpu;
                    adjust_step = (n_layer + nadj - 1) / nadj;
                } else {
                    adjust_step = (n_layer + cur_splits.size() - 1)/cur_splits.size();
                }
            }
        }
        std::vector<float> gpu_split_count;
        if (model.max_gpu > 0 && model.max_gpu < int(model.splits.size())) {
            gpu_split_count.resize(model.splits.size(), 0.0f);
        }
        for (int il = 0; il < n_layer; ++il) {
            // MTP-SPLIT: the MTP "nextn" tail layer is plain attention + MoE (no recurrent op),
            // so let it flow through the standard per-layer split path below (split attn wq/wk/wv/wo
            // + MoE expert matmuls) exactly like every normal transformer layer, instead of skipping
            // it (which left it as a non-split single-device island that OOM'd the last GPU under -sm attn).
            // The nextn-specific norms + eh_proj are NOT touched by this loop, so they stay replicated.
            int gqa_ratio = hparams.n_head(il) / hparams.n_head_kv(il);
            if (ggml_backend_buft_is_host(model.buft_layer[il].buft_matrix)) {
                LLAMA_LOG_INFO("%s: not splitting layer %d because buffer type is host\n", __func__, il);
                continue;
            }
            if (model.max_gpu > 0 && model.max_gpu < int(model.splits.size()) && il % adjust_step == 0) {
                cur_splits = model.splits;
                adjust_split(cur_splits, mem_used, model.max_gpu);
                LLAMA_LOG_INFO("Adjusted split at layer %2d:  ", il);
                float last_split = 0;
                for (int i = 0; i < (int)cur_splits.size(); ++i) {
                    if (i > 0) {
                        LLAMA_LOG_INFO(" ; ");
                    }
                    LLAMA_LOG_INFO("GPU%d: %4g", i, cur_splits[i] - last_split);
                    if (i < int(gpu_split_count.size())) {
                        gpu_split_count[i] += cur_splits[i] - last_split;
                    }
                    last_split = cur_splits[i];
                }
                LLAMA_LOG_INFO("\n");
            }
            LLAMA_LOG_DEBUG("=== Layer %2d. Mem used so far:", il);
            for ([[maybe_unused]] auto mem : mem_used) LLAMA_LOG_DEBUG(" %g", mem/1024./1024.);
            LLAMA_LOG_DEBUG("\n");
            auto & layer = model.layers[il];
            auto ctx_split = ctx_for_layer_split(il);
            if (layer.attn_norm) {
                prepare_split_tensors(-1, ctx_split, layer.attn_norm, layer.split_attn_norm, mirror, mem_used);
            }
            if (model.arch == LLM_ARCH_GEMMA4 && layer.attn_post_norm) {
                prepare_split_tensors(-1, ctx_split, layer.attn_post_norm, layer.split_attn_post_norm, mirror, mem_used);
            }
            if (layer.rope_freqs) {
                auto split = create_split(ggml_nrows(layer.rope_freqs), -1, cur_splits, mem_used);
                prepare_split_tensors(-1, ctx_split, layer.rope_freqs, layer.split_rope_freqs, split, mem_used);
            }
            if (hparams.is_recurrent(il)) {
                split_recurrent_tensors(hparams, layer, cur_splits, mem_used, ctx_split, il); //, model.arch == LLM_ARCH_QWEN3NEXT ? 0 : 1);
            }
            else if (layer.wo && layer.wq && layer.wk && (layer.wv || model.arch == LLM_ARCH_GEMMA4)) {
                auto granularity_kq = hparams.n_embd_head_k(il) * gqa_ratio;
                int wq_ne1 = layer.wq->ne[1];
                if (model.arch == LLM_ARCH_QWEN3NEXT || model.arch == LLM_ARCH_QWEN35MOE || model.arch == LLM_ARCH_QWEN35 ||
                    model.arch == LLM_ARCH_QWEN4EXP) {
                    // wq is [q|gate] interleaved per head, so it is twice as wide as the head split implies
                    granularity_kq *= 2; wq_ne1 /= 2;
                }
                auto granularity_vo = hparams.n_embd_head_v(il) * gqa_ratio;
                if (ggml_is_quantized(layer.wo->type)) {
                    auto tt = ggml_internal_get_type_traits(layer.wo->type);
                    if (tt.blck_size > granularity_vo) granularity_vo = tt.blck_size;
                    GGML_ASSERT(granularity_vo % hparams.n_embd_head_v(il) == 0);
                    // Command-R: align KQ split to wo's block size so wq row
                    // counts remain valid after splitting.
                    if (model.arch == LLM_ARCH_COMMAND_R) {
                        if (tt.blck_size > granularity_kq && layer.wq->ne[1] % tt.blck_size == 0) {
                            granularity_kq = tt.blck_size;
                        }
                    }
                }
                auto split_vo = create_split(layer.wo->ne[0], granularity_vo, cur_splits, mem_used); //, true);
                auto split_kq = create_split(layer.wq->ne[1], granularity_kq, cur_splits, mem_used); //, true);
                LLAMA_LOG_DEBUG("  split_vo:"); for ([[maybe_unused]] auto s : split_vo) LLAMA_LOG_DEBUG(" %d", s);
                LLAMA_LOG_DEBUG("\n");
                LLAMA_LOG_DEBUG("  split_kq:"); for ([[maybe_unused]] auto s : split_kq) LLAMA_LOG_DEBUG(" %d", s);
                LLAMA_LOG_DEBUG("\n");

                if (layer.attn_q_norm && layer.attn_q_norm->ne[0] == wq_ne1) {
                    // If RMS norm is not applied per attention head, as it is usually the case, but is applied to the
                    // entire Q tensor (e.g., MiniMax-2), we need to have a copy of the entire wq and attn_q_norm tensors
                    // on each participating GPU.
                    prepare_split_tensors(-1, ctx_split, layer.wq, layer.split_wq, split_vo, mem_used);
                    prepare_split_tensors(-1, ctx_split, layer.attn_q_norm, layer.split_q_norm, split_vo, mem_used);
                    if (layer.bq) {
                        prepare_split_tensors(-1, ctx_split, layer.bq, layer.split_bq, split_vo, mem_used);
                    }
                    LLAMA_LOG_DEBUG("Not splitting wq, attn_q_norm in layer layer %d because of RMS norm\n", il);
                } else {
                    prepare_split_tensors(1, ctx_split, layer.wq, layer.split_wq, split_kq, mem_used);
                    if (layer.attn_q_norm) {
                        if (layer.attn_q_norm->ne[1] > 1) {
                            // 2D per-head norm (e.g., Command-R+): split along the Q-head dimension
                            auto split_q_heads = split_kq;
                            for (auto & s : split_q_heads) s /= hparams.n_embd_head_k(il);
                            prepare_split_tensors(1, ctx_split, layer.attn_q_norm, layer.split_q_norm, split_q_heads, mem_used);
                        } else {
                            prepare_split_tensors(-1, ctx_split, layer.attn_q_norm, layer.split_q_norm, split_kq, mem_used);
                        }
                    }
                    if (layer.bq) {
                        prepare_split_tensors(0, ctx_split, layer.bq, layer.split_bq, split_kq, mem_used);
                    }
                }
                prepare_split_tensors(0, ctx_split, layer.wo, layer.split_wo, split_vo, mem_used);
                if (layer.bo) {
                    prepare_split_tensors(-1, ctx_split, layer.bo, layer.split_bo, split_vo, mem_used);
                }
                if (layer.attn_sinks) {
                    auto split_sinks = split_kq;
                    for (auto & s : split_sinks) {
                        s /= hparams.n_embd_head_k(il);
                    }
                    prepare_split_tensors(0, ctx_split, layer.attn_sinks, layer.split_sinks, split_sinks, mem_used);
                }
                if (layer.wqkv_gate) {
                    auto wqkv_gate_split = split_kq;
                    LLAMA_LOG_DEBUG("=================== wqkv_gate_split:");
                    for (auto & s : wqkv_gate_split) {
                        s /= hparams.n_embd_head_k(il);
                        LLAMA_LOG_DEBUG(" %d", s);
                    }
                    LLAMA_LOG_DEBUG("\n");
                    prepare_split_tensors(1, ctx_split, layer.wqkv_gate, layer.split_wqkv_gate, wqkv_gate_split, mem_used);
                }
                if (model.arch == LLM_ARCH_QWEN3NEXT || model.arch == LLM_ARCH_QWEN35MOE || model.arch == LLM_ARCH_QWEN35 ||
                    model.arch == LLM_ARCH_QWEN4EXP) {
                    for (auto & s : split_kq) s /= 2*gqa_ratio;
                } else {
                    for (auto & s : split_kq) s /= gqa_ratio;
                }
                for (auto & s : split_vo) s /= gqa_ratio;
                if (layer.attn_k_norm && layer.attn_k_norm->ne[0] == layer.wk->ne[1]) {
                    // If RMS norm is not applied per attention head, as it is usually the case, but is applied to the
                    // entire K tensor (e.g., MiniMax-2), we need to have a copy of the entire wk and attn_k_norm tensors
                    // on each participating GPU.
                    prepare_split_tensors(-1, ctx_split, layer.wk, layer.split_wk, split_vo, mem_used);
                    prepare_split_tensors(-1, ctx_split, layer.attn_k_norm, layer.split_k_norm, split_vo, mem_used);
                    if (layer.bk) {
                        prepare_split_tensors(-1, ctx_split, layer.bk, layer.split_bk, split_vo, mem_used);
                    }
                    LLAMA_LOG_DEBUG("Not splitting wk, attn_k_norm in layer layer %d because of RMS norm\n", il);
                } else {
                    prepare_split_tensors(1, ctx_split, layer.wk, layer.split_wk, split_kq, mem_used);
                    if (layer.bk) {
                        prepare_split_tensors(0, ctx_split, layer.bk, layer.split_bk, split_kq, mem_used);
                    }
                    if (layer.attn_k_norm) {
                        if (layer.attn_k_norm->ne[1] > 1) {
                            // 2D per-head norm (e.g., Command-R+): split along the KV-head dimension
                            // split_kq has already been divided by gqa_ratio, so values are in
                            // (n_embd_head_k * n_head_kv) units; divide again to get head units
                            auto split_k_heads = split_kq;
                            for (auto & s : split_k_heads) s /= hparams.n_embd_head_k(il);
                            prepare_split_tensors(1, ctx_split, layer.attn_k_norm, layer.split_k_norm, split_k_heads, mem_used);
                        } else {
                            prepare_split_tensors(-1, ctx_split, layer.attn_k_norm, layer.split_k_norm, split_kq, mem_used);
                        }
                    }
                }
                if (layer.wv) {
                    prepare_split_tensors(1, ctx_split, layer.wv, layer.split_wv, split_vo, mem_used);
                    if (layer.bv) {
                        prepare_split_tensors(0, ctx_split, layer.bv, layer.split_bv, split_vo, mem_used);
                    }
                }
            }

            if (layer.ffn_norm) {
                if (auto it = split_tensors.find(layer.ffn_norm); it != split_tensors.end()) {
                    prepare_split_tensors(-1, ctx_split, layer.ffn_norm, layer.split_ffn_norm, mirror, mem_used);
                }
            }
            if (layer.ffn_post_norm) {
                if (auto it = split_tensors.find(layer.ffn_post_norm); it != split_tensors.end()) {
                    prepare_split_tensors(-1, ctx_split, layer.ffn_post_norm, layer.split_ffn_post_norm, mirror, mem_used);
                }
            }
            if (layer.ffn_post_norm_1) {
                if (auto it = split_tensors.find(layer.ffn_post_norm_1); it != split_tensors.end()) {
                    prepare_split_tensors(-1, ctx_split, layer.ffn_post_norm_1, layer.split_ffn_post_norm_1, mirror, mem_used);
                }
            }
            if (layer.ffn_post_norm_2) {
                if (auto it = split_tensors.find(layer.ffn_post_norm_2); it != split_tensors.end()) {
                    prepare_split_tensors(-1, ctx_split, layer.ffn_post_norm_2, layer.split_ffn_post_norm_2, mirror, mem_used);
                }
            }
            if (layer.ffn_pre_norm_2) {
                if (auto it = split_tensors.find(layer.ffn_pre_norm_2); it != split_tensors.end()) {
                    prepare_split_tensors(-1, ctx_split, layer.ffn_pre_norm_2, layer.split_ffn_pre_norm_2, mirror, mem_used);
                }
            }
            if (layer.ffn_down_exps_s) {
                if (auto it = split_tensors.find(layer.ffn_down_exps_s); it != split_tensors.end()) {
                    prepare_split_tensors(-1, ctx_split, layer.ffn_down_exps_s, layer.split_ffn_down_exps_s, mirror, mem_used);
                }
            }
            if (layer.ffn_gate_inp_s) {
                if (auto it = split_tensors.find(layer.ffn_gate_inp_s); it != split_tensors.end()) {
                    prepare_split_tensors(-1, ctx_split, layer.ffn_gate_inp_s, layer.split_ffn_gate_inp_s, mirror, mem_used);
                }
            }

            // MLA tensor distribution (DEEPSEEK2/GLM_DSA/MISTRAL4). Detect by arch + absence of wk
            // since wkv_b can be null when the model was quantized by mainline llama.cpp.
            if (layer.wo && !layer.wk &&
                (model.arch == LLM_ARCH_DEEPSEEK2 ||
                 model.arch == LLM_ARCH_GLM_DSA ||
                 model.arch == LLM_ARCH_MISTRAL4)) {
                distribute_mla_tensors_for_split_mode_graph(
                    layer, hparams, cur_splits, mem_used, ctx_split, il);
            }

            if (layer.ffn_down && layer.ffn_up && layer.ffn_gate) {
                bool use_split = split_tensors.find(layer.ffn_down) != split_tensors.end() &&
                                 split_tensors.find(layer.ffn_gate) != split_tensors.end() &&
                                 split_tensors.find(layer.ffn_up)   != split_tensors.end();
                if (use_split) {
                    int ffn_granularity = 16;
                    if (ggml_is_quantized(layer.ffn_down->type)) {
                        auto tt = ggml_internal_get_type_traits(layer.ffn_down->type);
                        if (tt.blck_size > ffn_granularity) ffn_granularity = tt.blck_size;
                    }
                    auto split = create_split(layer.ffn_down->ne[0], ffn_granularity, cur_splits, mem_used);
                    LLAMA_LOG_DEBUG("  split_ffn:"); for ([[maybe_unused]] auto s : split) LLAMA_LOG_DEBUG(" %d", s); LLAMA_LOG_DEBUG("\n");
                    prepare_split_tensors(0, ctx_split, layer.ffn_down, layer.split_ffn_down, split, mem_used);
                    prepare_split_tensors(1, ctx_split, layer.ffn_up,   layer.split_ffn_up,   split, mem_used);
                    prepare_split_tensors(1, ctx_split, layer.ffn_gate, layer.split_ffn_gate, split, mem_used);
                }
            }

            std::vector<int> ffn_split;
            if (layer.ffn_down_exps && ((layer.ffn_up_exps && layer.ffn_gate_exps) || layer.ffn_up_gate_exps)) {
                bool has_up_gate = split_tensors.find(layer.ffn_gate_exps) != split_tensors.end() && split_tensors.find(layer.ffn_up_exps) != split_tensors.end();
                has_up_gate |= split_tensors.find(layer.ffn_up_gate_exps) != split_tensors.end();
                bool use_split = split_tensors.find(layer.ffn_down_exps) != split_tensors.end() && has_up_gate;

                if (use_split) {
                    int ffn_granularity = 16;
                    if (ggml_is_quantized(layer.ffn_down_exps->type)) {
                        auto tt = ggml_internal_get_type_traits(layer.ffn_down_exps->type);
                        if (tt.blck_size > ffn_granularity) ffn_granularity = tt.blck_size;
                    }
                    ffn_split = create_split(layer.ffn_down_exps->ne[0], ffn_granularity, cur_splits, mem_used);
                    LLAMA_LOG_DEBUG("  split_ffn_exps:"); for ([[maybe_unused]] auto s : ffn_split) LLAMA_LOG_DEBUG(" %d", s);
                    LLAMA_LOG_DEBUG("\n");
                    prepare_split_tensors(0, ctx_split, layer.ffn_down_exps, layer.split_ffn_down_exps, ffn_split, mem_used);
                    if (layer.ffn_up_gate_exps) {
                        auto up_gate_split = ffn_split;
                        for (auto & v : up_gate_split) v *= 2;
                        prepare_split_tensors(1, ctx_split, layer.ffn_up_gate_exps, layer.split_ffn_up_gate_exps, up_gate_split, mem_used);
                        prepare_up_gate_split(layer.ffn_up_gate_exps, layer.split_ffn_up_gate_exps);
                        if (layer.ffn_up_gate_exps_b) {
                            prepare_split_tensors(0, ctx_split, layer.ffn_up_gate_exps_b, layer.split_ffn_up_gate_exps_b, up_gate_split, mem_used);
                            prepare_up_gate_split(layer.ffn_up_gate_exps_b, layer.split_ffn_up_gate_exps_b);
                        }
                    } else {
                        prepare_split_tensors(1, ctx_split, layer.ffn_up_exps,   layer.split_ffn_up_exps,   ffn_split, mem_used);
                        prepare_split_tensors(1, ctx_split, layer.ffn_gate_exps, layer.split_ffn_gate_exps, ffn_split, mem_used);
                    }
                    if (layer.ffn_down_exps_b) {
                        prepare_split_tensors(-1, ctx_split, layer.ffn_down_exps_b, layer.split_ffn_down_exps_b, ffn_split, mem_used);
                    }
                    if (!layer.ffn_up_gate_exps) {
                        if (layer.ffn_up_exps_b) {
                            prepare_split_tensors( 0, ctx_split, layer.ffn_up_exps_b, layer.split_ffn_up_exps_b, ffn_split, mem_used);
                        }
                        if (layer.ffn_gate_exps_b) {
                            prepare_split_tensors( 0, ctx_split, layer.ffn_gate_exps_b, layer.split_ffn_gate_exps_b, ffn_split, mem_used);
                        }
                    }
                }
            }

            if (layer.ffn_down_shexp && layer.ffn_up_shexp && layer.ffn_gate_shexp) {
                bool use_split = split_tensors.find(layer.ffn_down_shexp) != split_tensors.end() &&
                                 split_tensors.find(layer.ffn_gate_shexp) != split_tensors.end() &&
                                 split_tensors.find(layer.ffn_up_shexp)   != split_tensors.end();
                if (use_split) {
                    int ffn_granularity = 16;
                    if (ggml_is_quantized(layer.ffn_down_shexp->type)) {
                        auto tt = ggml_internal_get_type_traits(layer.ffn_down_shexp->type);
                        if (tt.blck_size > ffn_granularity) ffn_granularity = tt.blck_size;
                    }
                    auto split = create_split(layer.ffn_down_shexp->ne[0], ffn_granularity, cur_splits, mem_used);
                    bool ok = true;
                    if (!ffn_split.empty()) {
                        ok = split.size() == ffn_split.size();
                        if (ok) {
                            for (int j = 0; j < int(ffn_split.size()); ++j) {
                                if ((split[j] == 0 && ffn_split[j] > 0) || (split[j] > 0 && ffn_split[j] == 0)) {
                                    ok = false; break;
                                }
                            }
                        }
                    }
                    if (!ok) {
                        LLAMA_LOG_INFO("=== exp/shexp mismatch in layer %d\n", il);
                        LLAMA_LOG_INFO("    experts:"); for (auto& s : ffn_split) LLAMA_LOG_INFO(" %d", s); LLAMA_LOG_INFO("\n");
                        LLAMA_LOG_INFO(" sh_experts:"); for (auto& s : split    ) LLAMA_LOG_INFO(" %d", s); LLAMA_LOG_INFO("\n");
                        std::vector<float> aux(ffn_split.size());
                        float sum = 0;
                        for (int j = 0; j < int(ffn_split.size()); ++j) {
                            sum += ffn_split[j];
                            aux[j] = sum;
                        }
                        for (auto& s : aux) s /= sum;
                        split = create_split(layer.ffn_down_shexp->ne[0], ffn_granularity, aux, mem_used);
                        LLAMA_LOG_INFO("        new:"); for (auto& s : split    ) LLAMA_LOG_INFO(" %d", s); LLAMA_LOG_INFO("\n");
                    } else {
                        LLAMA_LOG_DEBUG("  split_ffn_shexps:"); for ([[maybe_unused]] auto s : split) LLAMA_LOG_DEBUG(" %d", s);
                        LLAMA_LOG_DEBUG("\n");
                    }
                    prepare_split_tensors(0, ctx_split, layer.ffn_down_shexp, layer.split_ffn_down_shexp, split, mem_used);
                    prepare_split_tensors(1, ctx_split, layer.ffn_up_shexp,   layer.split_ffn_up_shexp,   split, mem_used);
                    prepare_split_tensors(1, ctx_split, layer.ffn_gate_shexp, layer.split_ffn_gate_shexp, split, mem_used);
                    if (layer.ffn_gate_inp_shexp) {
                        prepare_split_tensors(-1, ctx_split, layer.ffn_gate_inp_shexp, layer.split_ffn_gate_inp_shexp, split, mem_used);
                    }
                }
            }

            if (layer.ffn_gate_inp) {
                if (auto it = split_tensors.find(layer.ffn_gate_inp); it != split_tensors.end()) {
                    auto shared_split = create_split(ggml_nrows(layer.ffn_gate_inp), -1, cur_splits, mem_used);
                    prepare_split_tensors(-1, ctx_split, layer.ffn_gate_inp, layer.split_ffn_gate_inp, shared_split, mem_used);
                }
            }
            if (layer.ffn_gate_inp_b) {
                if (auto it = split_tensors.find(layer.ffn_gate_inp_b); it != split_tensors.end()) {
                    auto shared_split = create_split(ggml_nrows(layer.ffn_gate_inp_b), -1, cur_splits, mem_used);
                    prepare_split_tensors(-1, ctx_split, layer.ffn_gate_inp_b, layer.split_ffn_gate_inp_b, shared_split, mem_used);
                }
            }
            if (layer.ffn_exp_probs_b) {
                if (auto it = split_tensors.find(layer.ffn_exp_probs_b); it != split_tensors.end()) {
                    auto shared_split = create_split(ggml_nrows(layer.ffn_exp_probs_b), -1, cur_splits, mem_used);
                    prepare_split_tensors(-1, ctx_split, layer.ffn_exp_probs_b, layer.split_ffn_exp_probs_b, shared_split, mem_used);
                }
            }

            if (layer.out_scale) {
                prepare_split_tensors(-1, ctx_split, layer.out_scale, layer.split_out_scale, std::vector<int>(model.splits.size(), 1), mem_used);
            }

            // Must run AFTER every branch above: mirror_one() skips anything that already
            // carries ->extra, so an earlier, more specific decision always wins.
            if (model.arch == LLM_ARCH_QWEN4EXP) {
                split_qwen4exp_tensors(layer, mirror, mem_used, ctx_split);
            }
        }

        // MTP-UNIFORM: the nextn-head INPUT-FUSION tensors (enorm/hnorm/eh_proj) are NOT
        // touched by the per-layer split loop above (it splits attn_norm/wq/wk/wv/wo + MoE only),
        // so by default they stay single-device -> the MTP input fusion that produces the fused
        // hidden becomes a single-device island feeding the row-split attention/MoE, and the
        // cross-device read crashes under -sm attn (CUDA illegal memory access on device != main).
        // Mirror-replicate the FUSION tensors per-device (split_dim = -1 = a FULL copy on every
        // participating GPU, exactly like attn_norm/ffn_norm above) so each device can compute the
        // FULL fused hidden locally from its own copies. eh_proj is REPLICATED (not row-split):
        // tensor-parallel attention needs the full fused hidden on each device before slicing heads.
        //
        // shared_head_norm is DELIBERATELY *NOT* replicated. It feeds ONLY the final MTP output path
        // build_output(.., model.output_mtp, shared_head_norm, ..) (build_qwen35.cpp). output_mtp is
        // NON-split (it lives in ctx_output / mirrors model.output, which is non-split for qwen35moe),
        // so build_output takes its single-device (else) branch. Replicating shared_head_norm sets
        // ->extra on the base tensor, which converts it into a SPLIT-buffer tensor whose BASE data
        // pointer is the dummy 0x1000 (ggml-cuda.cu split_buffer_get_base; only the per-device
        // ->splits[] get real data). The non-split build_output then reads that base in the
        // ggml_fused_rms_norm weight slot -> CUDA illegal memory access on the OUTPUT device (0).
        // Left un-replicated, shared_head_norm keeps its real single-device buffer (created via
        // ctx_for_layer_split = the buft_matrix single-device offload ctx, NOT split_ctx) on the MTP
        // layer's device, exactly like model.output_norm in the working main path -> coherent.
        if (model.mtp && model.hparams.nextn_predict_layers > 0) {
            const int n_total = (int)model.layers.size();
            for (int il = n_total - (int)model.hparams.nextn_predict_layers; il < n_total; ++il) {
                if (ggml_backend_buft_is_host(model.buft_layer[il].buft_matrix)) {
                    LLAMA_LOG_INFO("%s: not mirroring nextn tensors for layer %d (buffer type is host)\n", __func__, il);
                    continue;
                }
                auto & layer = model.layers[il];
                auto ctx_split = ctx_for_layer_split(il);
                if (layer.nextn.enorm && !layer.nextn.enorm->extra) {
                    prepare_split_tensors(-1, ctx_split, layer.nextn.enorm, layer.split_nextn_enorm, mirror, mem_used);
                }
                if (layer.nextn.hnorm && !layer.nextn.hnorm->extra) {
                    prepare_split_tensors(-1, ctx_split, layer.nextn.hnorm, layer.split_nextn_hnorm, mirror, mem_used);
                }
                if (layer.nextn.eh_proj && !layer.nextn.eh_proj->extra) {
                    prepare_split_tensors(-1, ctx_split, layer.nextn.eh_proj, layer.split_nextn_eh_proj, mirror, mem_used);
                }
                // NOTE: shared_head_norm intentionally NOT replicated (see comment above) — it must
                // stay single-device to match the non-split output_mtp lm_head in build_output.
                LLAMA_LOG_INFO("%s: MTP layer %d nextn fusion tensors (enorm/hnorm/eh_proj) mirror-replicated across %d devices\n",
                        __func__, il, (int)model.splits.size());
            }
        }

        if (!gpu_split_count.empty()) {
            LLAMA_LOG_INFO("Adjusted splits (total)   :  ");
            for (int i = 0; i < (int)gpu_split_count.size(); ++i) {
                if (i > 0) {
                    LLAMA_LOG_INFO(" ; ");
                }
                LLAMA_LOG_INFO("GPU%d: %4g", i, gpu_split_count[i]);
            }
            LLAMA_LOG_INFO("\n");
        }

        if (model.output) {
            if (auto it = split_tensors.find(model.output); it != split_tensors.end()) {
                if (ggml_backend_buft_is_host(model.buft_output.buft_matrix)) {
                    LLAMA_LOG_INFO("%s: not splitting output tensor becausee buffer is host\n", __func__);
                } else {
                    auto ctx_split = ctx_map[model.buft_output.buft_matrix];
                    auto split = create_split(model.output->ne[1], 16, model.splits, mem_used);
                    prepare_split_tensors(1, ctx_split, model.output, model.split_output, split, mem_used);
                    if (auto it = split_tensors.find(model.output_norm); it != split_tensors.end() && !ggml_backend_buft_is_host(model.buft_output.buft_matrix)) {
                        auto ctx_split = ctx_map[model.buft_output.buft_matrix];
                        prepare_split_tensors(-1, ctx_split, model.output_norm, model.split_output_norm, split, mem_used);
                    }
                }
            }
        }

        LLAMA_LOG_INFO("Estimated model buffer size per device:\n");
        for (int i = 0; i < int(mem_used.size()); ++i) {
            LLAMA_LOG_INFO("    Device %d:  %8.2f MiB\n", i, mem_used[i]/1024./1024.);
        }
    }
    return use_mmap_buffer;
}

std::unique_ptr<create_tensors_helper_interface> create_tensors_helper_interface::instance(llama_model_loader & ml, llama_model & model) {
    return std::make_unique<create_tensors_helper>(ml, model);
}
