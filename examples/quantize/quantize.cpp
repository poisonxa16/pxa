//
// Copyright (C) 2023-2025 The llama.cpp authors
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <cmath>

struct quant_option {
    std::string name;
    llama_ftype ftype;
    std::string desc;
};

static const std::vector<struct quant_option> QUANT_OPTIONS = {
    { "Q4_0",     LLAMA_FTYPE_MOSTLY_Q4_0,     " 3.56G, +0.2166 ppl @ LLaMA-v1-7B", },
    { "Q4_1",     LLAMA_FTYPE_MOSTLY_Q4_1,     " 3.90G, +0.1585 ppl @ LLaMA-v1-7B", },
    { "Q5_0",     LLAMA_FTYPE_MOSTLY_Q5_0,     " 4.33G, +0.0683 ppl @ LLaMA-v1-7B", },
    { "Q5_1",     LLAMA_FTYPE_MOSTLY_Q5_1,     " 4.70G, +0.0349 ppl @ LLaMA-v1-7B", },
    { "Q6_0",     LLAMA_FTYPE_MOSTLY_Q6_0,     " 6.5 bpw quantization",             },
    { "MXFP4",    LLAMA_FTYPE_MOSTLY_MXFP4,    " 4.25 bpw 4-bit float quantization",},
    // PXQ names re-laddered by bpw class (2026-07-19 display names; 2026-07-21 honest ladder):
    // the 4-bit quality tier is PXQ4 (formerly PXQ6) with HQ variant PXQ4-HQ (formerly PXQ6HQ),
    // and PXQ6 is the REAL 5-bit LM32 tier (gguf type id 256). Old 4-bit-tier names remain as
    // deprecated aliases below. The legacy types id 250 (MXFP4-repack, old name "PXQ4") and
    // id 251 (PXQ5, learned book + SE8) were RETIRED/removed 2026-07-21.
    { "PXQ4",     LLAMA_FTYPE_MOSTLY_PXQ4,     " 4.27 bpw, PX16 book + E16-row scales, slab layout (formerly PXQ6)",},
    { "PXQ4-HQ",  LLAMA_FTYPE_MOSTLY_PXQ4HQ,   " 4.52 bpw, PXQ4 with bs8 sub-scales, slab layout (formerly PXQ6HQ)",},
    { "PXQ6",     LLAMA_FTYPE_MOSTLY_PXQ6,     " 5.27 bpw, LM32 5-bit x E16-row scales — the quality tier (wrel -56% vs PXQ4)",},
    { "PXQ2",     LLAMA_FTYPE_MOSTLY_PXQ2,     " 2.27 bpw, LM4 x E16-row scales (experts; wrel 4.3x PXQ4)",},
    { "PXQ3",     LLAMA_FTYPE_MOSTLY_PXQ3,     " 3.27 bpw, LM8 bit-plane x E16-row scales (experts; wrel 2.1x PXQ4)",},
    { "PXQ_UNIVERSAL", LLAMA_FTYPE_MOSTLY_PXQ_UNIVERSAL, " mixed PXQ1/PXQ2/PXQ3/PXQ4 per-tensor tier map (--pxq-universal)",},
    { "PXQ1",     LLAMA_FTYPE_MOSTLY_PXQ1,     " 1.26 bpw, 1-bit sign x E16-row scales (experts; the sub-2-bit stretch tier)",},
    { "PXQ6HQ",   LLAMA_FTYPE_MOSTLY_PXQ4HQ,   " deprecated alias for PXQ4-HQ (pre-re-ladder name)",},
    { "PXQ4HQ",   LLAMA_FTYPE_MOSTLY_PXQ4HQ,   " alias for PXQ4-HQ",},
    { "IQ2_XXS",  LLAMA_FTYPE_MOSTLY_IQ2_XXS,  " 2.06 bpw quantization",            },
    { "IQ2_XS",   LLAMA_FTYPE_MOSTLY_IQ2_XS,   " 2.31 bpw quantization",            },
    { "IQ2_S",    LLAMA_FTYPE_MOSTLY_IQ2_S,    " 2.5  bpw quantization",            },
    { "IQ2_M",    LLAMA_FTYPE_MOSTLY_IQ2_M,    " 2.7  bpw quantization",            },
    { "IQ1_S",    LLAMA_FTYPE_MOSTLY_IQ1_S,    " 1.56 bpw quantization",            },
    { "IQ1_M",    LLAMA_FTYPE_MOSTLY_IQ1_M,    " 1.75 bpw quantization",            },
    { "Q2_K",     LLAMA_FTYPE_MOSTLY_Q2_K,     " 2.63G, +0.6717 ppl @ LLaMA-v1-7B", },
    { "Q2_K_S",   LLAMA_FTYPE_MOSTLY_Q2_K_S,   " 2.16G, +9.0634 ppl @ LLaMA-v1-7B", },
    { "IQ3_XXS",  LLAMA_FTYPE_MOSTLY_IQ3_XXS,  " 3.06 bpw quantization",            },
    { "IQ3_S",    LLAMA_FTYPE_MOSTLY_IQ3_S,    " 3.44 bpw quantization",            },
    { "IQ3_M",    LLAMA_FTYPE_MOSTLY_IQ3_M,    " 3.66 bpw quantization mix",        },
    { "Q3_K",     LLAMA_FTYPE_MOSTLY_Q3_K_M,   "alias for Q3_K_M" },
    { "IQ3_XS",   LLAMA_FTYPE_MOSTLY_IQ3_XS,   " 3.3 bpw quantization"   ,          },
    { "Q3_K_S",   LLAMA_FTYPE_MOSTLY_Q3_K_S,   " 2.75G, +0.5551 ppl @ LLaMA-v1-7B", },
    { "Q3_K_M",   LLAMA_FTYPE_MOSTLY_Q3_K_M,   " 3.07G, +0.2496 ppl @ LLaMA-v1-7B", },
    { "Q3_K_L",   LLAMA_FTYPE_MOSTLY_Q3_K_L,   " 3.35G, +0.1764 ppl @ LLaMA-v1-7B", },
    { "IQ4_NL",   LLAMA_FTYPE_MOSTLY_IQ4_NL,   " 4.50 bpw non-linear quantization", },
    { "Q8_KV",    LLAMA_FTYPE_MOSTLY_Q8_KV,    " 8.00 bpw quantization",            },
    { "IQ4_XS",   LLAMA_FTYPE_MOSTLY_IQ4_XS,   " 4.25 bpw non-linear quantization", },
    { "Q4_K",     LLAMA_FTYPE_MOSTLY_Q4_K_M,   "alias for Q4_K_M", },
    { "Q4_K_S",   LLAMA_FTYPE_MOSTLY_Q4_K_S,   " 3.59G, +0.0992 ppl @ LLaMA-v1-7B", },
    { "Q4_K_M",   LLAMA_FTYPE_MOSTLY_Q4_K_M,   " 3.80G, +0.0532 ppl @ LLaMA-v1-7B", },
    { "Q5_K",     LLAMA_FTYPE_MOSTLY_Q5_K_M,   "alias for Q5_K_M", },
    { "Q5_K_S",   LLAMA_FTYPE_MOSTLY_Q5_K_S,   " 4.33G, +0.0400 ppl @ LLaMA-v1-7B", },
    { "Q5_K_M",   LLAMA_FTYPE_MOSTLY_Q5_K_M,   " 4.45G, +0.0122 ppl @ LLaMA-v1-7B", },
    { "Q6_K",     LLAMA_FTYPE_MOSTLY_Q6_K,     " 5.15G, +0.0008 ppl @ LLaMA-v1-7B", },
    { "Q8_0",     LLAMA_FTYPE_MOSTLY_Q8_0,     " 6.70G, +0.0004 ppl @ LLaMA-v1-7B", },
    { "Q4_0_4_4", LLAMA_FTYPE_MOSTLY_Q4_0_4_4, " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "Q4_0_4_8", LLAMA_FTYPE_MOSTLY_Q4_0_4_8, " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "Q4_0_8_8", LLAMA_FTYPE_MOSTLY_Q4_0_8_8, " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "F16",      LLAMA_FTYPE_MOSTLY_F16,      "14.00G, -0.0020 ppl @ Mistral-7B", },
    { "BF16",     LLAMA_FTYPE_MOSTLY_BF16,     "14.00G, -0.0050 ppl @ Mistral-7B", },
    { "F32",      LLAMA_FTYPE_ALL_F32,         "26.00G              @ 7B", },
    // Note: Ensure COPY comes after F32 to avoid ftype 0 from matching.
    { "COPY",     LLAMA_FTYPE_ALL_F32,         "only copy tensors, no quantizing",  },
};

static const char * const LLM_KV_QUANTIZE_IMATRIX_FILE       = "quantize.imatrix.file";
static const char * const LLM_KV_QUANTIZE_IMATRIX_DATASET    = "quantize.imatrix.dataset";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES  = "quantize.imatrix.entries_count";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS   = "quantize.imatrix.chunks_count";

static bool try_parse_ftype(const std::string & ftype_str_in, llama_ftype & ftype, std::string & ftype_str_out) {
    std::string ftype_str;

    for (auto ch : ftype_str_in) {
        ftype_str.push_back(std::toupper(ch));
    }
    for (auto & it : QUANT_OPTIONS) {
        if (it.name == ftype_str) {
            ftype = it.ftype;
            ftype_str_out = it.name;
            return true;
        }
    }
    try {
        int ftype_int = std::stoi(ftype_str);
        for (auto & it : QUANT_OPTIONS) {
            if (it.ftype == ftype_int) {
                ftype = it.ftype;
                ftype_str_out = it.name;
                return true;
            }
        }
    }
    catch (...) {
        // stoi failed
    }
    return false;
}

// usage:
//  ./llama-quantize [--allow-requantize] [--i-know-this-is-double-lossy] [--leave-output-tensor] [--pure] models/llama/ggml-model.gguf [models/llama/ggml-model-quant.gguf] type [nthreads]
//
[[noreturn]]
static void usage(const char * executable) {
    printf("usage: %s [--help] [--allow-requantize] [--i-know-this-is-double-lossy] [--leave-output-tensor] [--pure] [--imatrix] [--hide-imatrix] [--ignore-imatrix-rules] [--dry-run] [--include-weights] [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--extra-output-tensor] [--ffn-gate-inp-type] [--attn-q-type] [--attn-k-type] [--attn-v-type] [--attn-qkv-type] [--attn-output-type] [--ffn-gate-type] [--ffn-down-type] [--ffn-up-type] [--repack] [--repack-pattern] [--keep-split] [--partial-requant] [--override-kv] model-f32.gguf [model-quant.gguf] type [nthreads]\n\n", executable);
    printf("  --allow-requantize: Allows requantizing tensors that have already been quantized. Warning: This can severely reduce quality compared to quantizing from 16bit or 32bit\n");
    printf("  --i-know-this-is-double-lossy: Required to requantize from a Q8_0 or PXQ source. Two lossy passes compound: the second codec fits a grid to weights that were already snapped, so quality is strictly below quantizing the original F32/BF16 once\n");
    printf("  --leave-output-tensor: Will leave output.weight un(re)quantized. Increases model size but may also increase quality, especially when requantizing\n");
    printf("  --pure: Disable k-quant mixtures and quantize all tensors to the same type\n");
    printf("  --imatrix file_name: use data in file_name as importance matrix for quant optimizations\n");
    printf("  --hide-imatrix: do not store imatrix details in the quantized model\n");
    printf("  --ignore-imatrix-rules: ignore importance matrix rules when quantizing\n");
    printf("  --dry-run: show what would be quantized without actually writing the output file\n");
    printf("  --pxq-name-override: write a NON-PXQ type to a PXQ-named output file. Without this the\n");
    printf("        quantizer refuses that combination up front, because it is nearly always a number\n");
    printf("        from the wrong space: stock ftypes are 0-38, the PXQ tiers are 248 and 252-257.\n");
    printf("  --include-weights tensor_name: use importance matrix for this/these tensor(s)\n");
    printf("  --exclude-weights tensor_name: use importance matrix for this/these tensor(s)\n");
    printf("  --output-tensor-type ggml_type: use this ggml_type for the output.weight tensor.\n");
    printf("  --token-embedding-type ggml_type: use this ggml_type for the token_embd.weight tensor.\n\n");
    printf("  --extra-output-tensor ggml_type: requantize and add output tensor of that type.\n");
    printf("  --ffn-gate-inp-type ggml_type: use this ggml_type for the ffn_gate_inp tensors.\n\n");
    printf("  --custom-q regex1=type1,regex2=type2...: use this to specify custom quantization type rules.\n\n");
    printf("  --pxq-universal /path/to/map.tiers: PXQ-UNIVERSAL per-tensor tier map.\n");
    printf("  --pxq-composition-override: keep a PXQ-target output that FAILS the composition assertion\n");
    printf("        (PXQ family < 50%% of bytes, or zero bytes of the named tier). Default: abort + remove.\n");
    printf("        A bare <name> resolves to $PXA_PXQU_DIR/<name>.tiers (default pxa-bench/pxq-universal/ next to CWD).\n");
    printf("        The file is '#'-commented lines of regex=type (pxq1|pxq2|pxq3|pxq4; pxq6 = the 5-bit tier since 2026-07-21), fed through --custom-q.\n\n");
    printf("  --repack Repack all tensors to the corresponding _r4/8 variant if available.\n\n");
    printf("  --repack-pattern Comma separated list of regexs to use for matching tensor names to be repacked.\n\n");
    printf("  --symmetric-q40  Use [-7:7] range for Q4_0 quantization (turns off imatrix)\n\n");
    printf("  --slow-iq2ks Use the original very slow IQ2_KS quantization method.\n\n");
    printf("Additional specific tensor quantization types used in the custom quant scheme 'CQS (default is Q2_K):\n");
    printf("      --attn-q-type ggml_type: use this ggml_type for the attn_q.weight tensor.\n");
    printf("      --attn-k-type ggml_type: use this ggml_type for the attn_k.weight tensor.\n");
    printf("      --attn-v-type ggml_type: use this ggml_type for the attn_v.weight tensor.\n");
    printf("      --attn-qkv-type ggml_type: use this ggml_type for the attn_qkv.weight tensor.\n");
    printf("      --attn-output-type ggml_type: use this ggml_type for the attn_output.weight tensor.\n");
    printf("      --ffn-gate-type ggml_type: use this ggml_type for the ffn_gate tensor.\n");
    printf("      --ffn-down-type ggml_type: use this ggml_type for the ffn_down tensor.\n");
    printf("      --ffn-up-type ggml_type: use this ggml_type for the ffn_up tensor.\n\n");
    printf("  --keep-split: will generate quantized model in the same shards as input\n");
    printf("  --partial-requant: quantize only missing split files in the split quantized .gguf destination directory\n");
    printf("  --override-kv KEY=TYPE:VALUE   (TYPE: int|uint|float|bool|str; uint writes GGUF UINT32)\n");
    printf("      Advanced option to override model metadata by key in the quantized model. May be specified multiple times.\n\n");
    printf("Note: --include-weights and --exclude-weights cannot be used together\n");
    printf("Note: The token embeddings tensor is loaded in system RAM, even in case of full GPU/VRAM offload.\n");
    printf("Note: The recommanded type for the output tensor is q6_K for the ffn types > iq3_xxs and < q8_0.\n\n");
    printf("Note for the Custom Quant Scheme FTYPE:\n");
    printf("    Write the specific tensor legacy quants as qN_N, the K-Quants as qN_K, the IQ-Quants as iqN_xx.\n");
    printf("    Usually, attn-q-type can be one type below the chosen ffn type, and attn-v-type should be one type above.\n");
    printf("    attn-qkv-type replaces the types attn-q, attn-k and attn-v on some models.\n");
    //TODO: - eventually - harmonize the CAPS writing of the FTYPEs, and non CAPS writing of the GGML_TYPEs.
    printf("\nAllowed quantization types:\n");
    for (auto & it : QUANT_OPTIONS) {
        if (it.name != "COPY") {
            printf("  %2d  or  ", it.ftype);
        } else {
            printf("          ");
        }
        printf("%-7s : %s\n", it.name.c_str(), it.desc.c_str());
    }
    exit(1);
}

static int load_imatrix(const std::string & imatrix_file, std::string & imatrix_dataset, std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    std::ifstream in(imatrix_file.c_str(), std::ios::binary);
    if (!in) {
        printf("%s: failed to open %s\n",__func__, imatrix_file.c_str());
        exit(1);
    }
    int n_entries;
    in.read((char *)&n_entries, sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        printf("%s: no data in file %s\n", __func__, imatrix_file.c_str());
        exit(1);
    }
    for (int i = 0; i < n_entries; ++i) {
        int len; in.read((char *)&len, sizeof(len));
        std::vector<char> name_as_vec(len+1);
        in.read((char *)name_as_vec.data(), len);
        if (in.fail()) {
            printf("%s: failed reading name for entry %d from %s\n", __func__, i+1, imatrix_file.c_str());
            exit(1);
        }
        name_as_vec[len] = 0;
        std::string name{name_as_vec.data()};
        auto & e = imatrix_data[name];
        int ncall;
        in.read((char *)&ncall, sizeof(ncall));
        int nval;
        in.read((char *)&nval, sizeof(nval));
        if (in.fail() || nval < 1) {
            printf("%s: failed reading number of values for entry %d\n", __func__, i);
            imatrix_data = {};
            exit(1);
        }
        e.resize(nval);
        in.read((char *)e.data(), nval*sizeof(float));
        if (in.fail()) {
            printf("%s: failed reading data for entry %d\n", __func__, i);
            imatrix_data = {};
            exit(1);
        }
        if (ncall > 0) {
            for (auto& v : e) v /= ncall;
        }

        if (getenv("LLAMA_TRACE")) {
            printf("%s: loaded data (size = %6d, ncall = %6d) for '%s'\n", __func__, int(e.size()), ncall, name.c_str());
        }
    }

    // latest imatrix version contains the dataset filename at the end of the file
    int m_last_call = 0;
    if (in.peek() != EOF) {
        in.read((char *)&m_last_call, sizeof(m_last_call));
        int dataset_len;
        in.read((char *)&dataset_len, sizeof(dataset_len));
        std::vector<char> dataset_as_vec(dataset_len);
        in.read(dataset_as_vec.data(), dataset_len);
        imatrix_dataset.assign(dataset_as_vec.begin(), dataset_as_vec.end());
        printf("%s: imatrix dataset='%s'\n", __func__, imatrix_dataset.c_str());
    }
    printf("%s: loaded %d importance matrix entries from %s computed on %d chunks\n", __func__, int(imatrix_data.size()), imatrix_file.c_str(), m_last_call);
    return m_last_call;
}

static int prepare_imatrix(const std::string & imatrix_file,
        std::string & imatrix_dataset,
        const std::vector<std::string> & included_weights,
        const std::vector<std::string> & excluded_weights,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    int m_last_call = -1;
    if (!imatrix_file.empty()) {
        m_last_call = load_imatrix(imatrix_file, imatrix_dataset, imatrix_data);
    }
    if (imatrix_data.empty()) {
        return m_last_call;
    }
    if (!excluded_weights.empty()) {
        for (auto& name : excluded_weights) {
            for (auto it = imatrix_data.begin(); it != imatrix_data.end(); ) {
                auto pos = it->first.find(name);
                if (pos != std::string::npos) it = imatrix_data.erase(it);
                else ++it;
            }
        }
    }
    if (!included_weights.empty()) {
        std::unordered_map<std::string, std::vector<float>> tmp;
        for (auto& name : included_weights) {
            for (auto& e : imatrix_data) {
                auto pos = e.first.find(name);
                if (pos != std::string::npos) {
                    tmp.emplace(std::move(e));
                }
            }
        }
        imatrix_data = std::move(tmp);
    }
    if (!imatrix_data.empty()) {
        printf("%s: have %d importance matrix entries\n", __func__, int(imatrix_data.size()));
    }
    return m_last_call;
}

// PXA_TYPEARG_STRICT_v1 (2026-07-30): case-insensitive name compare. The canonical ggml
// names are mixed-case ("q6_K", "q4_K") while every human types lowercase — and the old
// strcmp-exact match made `--token-embedding-type q6_k` parse to GGML_TYPE_COUNT, which the
// consumer treats as "flag not given": a silently different model that benches normally.
// (Verified 2026-07-29: token_embd landed on mxfp4, exit 0, no warning.)
static bool pxa_type_name_eq(const char * a, const char * b) {
    for (; *a && *b; ++a, ++b) {
        if (tolower((unsigned char) *a) != tolower((unsigned char) *b)) return false;
    }
    return *a == *b; // both at NUL
}

static ggml_type parse_ggml_type(const char * arg) {
    ggml_type result = GGML_TYPE_COUNT;
    for (int j = 0; j < GGML_TYPE_COUNT; ++j) {
        auto type = ggml_type(j);
        const auto * name = ggml_type_name(type);
        if (name && pxa_type_name_eq(arg, name)) {
            result = type; break;
        }
    }
    if (result == GGML_TYPE_COUNT) {
        // PXQ re-ladder aliases for tier maps / --custom-q / --*-type args. NOTE (2026-07-21):
        // lowercase "pxq6" now resolves via ggml_type_name to the REAL 5-bit tier (id 256) —
        // the transitional 2026-07-19 alias pxq6->pxq4 is gone. "pxq6hq" stays a deprecated
        // alias for the 4-bit HQ tier (id 253).
        if      (pxa_type_name_eq(arg, "pxq6hq")  ) result = GGML_TYPE_PXQ4HQ;  // deprecated: displayed "pxq4hq"
        else if (pxa_type_name_eq(arg, "pxq4-hq") ) result = GGML_TYPE_PXQ4HQ;
        else if (pxa_type_name_eq(arg, "pxq4_hq") ) result = GGML_TYPE_PXQ4HQ;
    }
    return result;
}

// PXA_TYPEARG_STRICT_v1: hard-fail on an unparseable type name. All 12 --*-type flags used
// to assign parse_ggml_type()'s failure value (GGML_TYPE_COUNT) unconditionally, and the
// consumer guard `if (type < GGML_TYPE_COUNT)` then skipped the flag IN SILENCE — exit 0,
// clean logs, wrong artifact. --custom-q always validated; the asymmetry was the bug.
static ggml_type parse_ggml_type_or_die(const char * arg) {
    const ggml_type t = parse_ggml_type(arg);
    if (t == GGML_TYPE_COUNT) {
        fprintf(stderr,
            "\n============================================================\n"
            "ERROR: invalid/unknown ggml type name '%s' for a --*-type flag.\n"
            "Names are matched case-insensitively against ggml_type_name()\n"
            "(e.g. q6_K, q4_K, iq4_ks, mxfp4, pxq4, pxq6). Refusing to run:\n"
            "silently ignoring a tensor-type override would produce a model\n"
            "that differs from build intent while benching normally.\n"
            "============================================================\n\n", arg);
        exit(1);
    }
    return t;
}

using CustomQ = std::pair<std::string, ggml_type>;

static bool parse_custom_quants(const std::string& arg, std::vector<CustomQ>& custom_quants) {
    for (const auto & item : string_split<std::string>(arg, ',')) {
        auto pos = item.find('=');
        if (pos == std::string::npos) {
            fprintf(stderr, "Invalid custom quantization input %s\n", arg.c_str());
            return false;
        }
        auto pattern = item.substr(0, pos);
        auto type_as_string = item.substr(pos + 1);
        auto type = parse_ggml_type(type_as_string.c_str());
        if (type == GGML_TYPE_COUNT) {
            fprintf(stderr, "Invalid quantization type '%s' in custom quantization input %s\n", type_as_string.c_str(), item.c_str());
            return false;
        }
        printf("Adding custom rule %s -> %s\n", pattern.c_str(), ggml_type_name(type));
        custom_quants.emplace_back(std::move(pattern), type);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage(argv[0]);
    }

    llama_model_quantize_params params = llama_model_quantize_default_params();

    int arg_idx = 1;
    std::string imatrix_file;
    std::vector<std::string> included_weights, excluded_weights;
    std::vector<llama_model_kv_override> kv_overrides;
    std::vector<CustomQ> custom_quants;
    std::string pxqu_arg;
    quantize_user_data user_data = { false, false };
    params.user_data = &user_data;


    bool hide_imatrix = false;
    bool pxq_name_guard_override = false;

    for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++) {
        if (strcmp(argv[arg_idx], "--pxq-composition-override") == 0) {
            // explicit opt-out of the PXQ composition assertion (see src/llama-quantize.cpp):
            // keeps an output whose PXQ byte-share is below the 50% floor / missing its named tier.
            setenv("PXA_PXQ_COMPOSITION_OVERRIDE", "1", 1);
        } else if (strcmp(argv[arg_idx], "--leave-output-tensor") == 0) {
            params.quantize_output_tensor = false;
        } else if (strcmp(argv[arg_idx], "--ignore-imatrix-rules") == 0) {
            params.ignore_imatrix_rules = true;
        } else if (strcmp(argv[arg_idx], "--dry-run") == 0) {
            params.dry_run = true;
        } else if (strcmp(argv[arg_idx], "--pxq-name-override") == 0) {
            pxq_name_guard_override = true;
        } else if (strcmp(argv[arg_idx], "--symmetric-q40") == 0) {
            user_data.symmetric_q4_0 = true;
        } else if (strcmp(argv[arg_idx], "--slow-iq2ks") == 0) {
            user_data.slow_iq2_ks = true;
        } else if (strcmp(argv[arg_idx], "--output-tensor-type") == 0) {
            if (arg_idx < argc-1) {
                params.output_tensor_type = parse_ggml_type_or_die(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--extra-output-tensor") == 0) {
            if (arg_idx < argc-1) {
                params.extra_output_type = parse_ggml_type_or_die(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--token-embedding-type") == 0) {
            if (arg_idx < argc-1) {
                params.token_embedding_type = parse_ggml_type_or_die(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--ffn-gate-inp-type") == 0) {
            if (arg_idx < argc-1) {
                params.ffn_gate_inp_type = parse_ggml_type_or_die(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--attn-q-type") == 0) {
            if (arg_idx < argc-1) {
                params.attn_q_type = parse_ggml_type_or_die(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--attn-k-type") == 0) {
            if (arg_idx < argc-1) {
                params.attn_k_type = parse_ggml_type_or_die(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--attn-v-type") == 0) {
            if (arg_idx < argc-1) {
                params.attn_v_type = parse_ggml_type_or_die(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--attn-qkv-type") == 0) {
            if (arg_idx < argc-1) {
                params.attn_qkv_type = parse_ggml_type_or_die(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--attn-output-type") == 0) {
            if (arg_idx < argc-1) {
                params.attn_output_type = parse_ggml_type_or_die(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--ffn-gate-type") == 0) {
            if (arg_idx < argc-1) {
                params.ffn_gate_type = parse_ggml_type_or_die(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--ffn-down-type") == 0) {
            if (arg_idx < argc-1) {
                params.ffn_down_type = parse_ggml_type_or_die(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--ffn-up-type") == 0) {
            if (arg_idx < argc-1) {
                params.ffn_up_type = parse_ggml_type_or_die(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--override-kv") == 0) {
            if (arg_idx == argc-1 || !string_parse_kv_override(argv[++arg_idx], kv_overrides)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--custom-q") == 0) {
            if (arg_idx == argc-1 || !parse_custom_quants(argv[++arg_idx], custom_quants)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--pxq-universal") == 0) {
            if (arg_idx == argc-1) usage(argv[0]);
            pxqu_arg = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "--allow-requantize") == 0) {
            params.allow_requantize = true;
        } else if (strcmp(argv[arg_idx], "--i-know-this-is-double-lossy") == 0) {
            params.allow_double_lossy = true;
        } else if (strcmp(argv[arg_idx], "--pure") == 0) {
            params.pure = true;
        } else if (strcmp(argv[arg_idx], "--imatrix") == 0) {
            if (arg_idx < argc-1) {
                imatrix_file = argv[++arg_idx];
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--hide-imatrix") == 0) {
            hide_imatrix = true;
        } else if (strcmp(argv[arg_idx], "--include-weights") == 0) {
            if (arg_idx < argc-1) {
                included_weights.emplace_back(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--exclude-weights") == 0) {
            if (arg_idx < argc-1) {
                excluded_weights.emplace_back(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--keep-split") == 0) {
            params.keep_split = true;
        } else if (strcmp(argv[arg_idx], "--partial-requant") == 0) {
            params.partial_requant = true;
        } else {
            usage(argv[0]);
        }
    }

    if (argc - arg_idx < 2) {
        printf("%s: bad arguments\n", argv[0]);
        usage(argv[0]);
    }
    if (!included_weights.empty() && !excluded_weights.empty()) {
        usage(argv[0]);
    }

    std::string imatrix_dataset;
    std::unordered_map<std::string, std::vector<float>> imatrix_data;
    int m_last_call = prepare_imatrix(imatrix_file, imatrix_dataset, included_weights, excluded_weights, imatrix_data);
    if (!imatrix_data.empty()) {
        params.imatrix = &imatrix_data;
    }
    if (!imatrix_data.empty()) {
        {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_FILE);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
            if (hide_imatrix) {
                strncpy(kvo.val_str, "top_secret", 127);
            } else {
                strncpy(kvo.val_str, imatrix_file.c_str(), 127);
            }
            kvo.val_str[127] = '\0';
            kv_overrides.emplace_back(std::move(kvo));
        }
        if (!imatrix_dataset.empty()) {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_DATASET);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
            if (hide_imatrix) {
                strncpy(kvo.val_str, "top_secret", 127);
            } else {
                strncpy(kvo.val_str, imatrix_dataset.c_str(), 127);
            }
            kvo.val_str[127] = '\0';
            kv_overrides.emplace_back(std::move(kvo));
        }

        {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
            if (hide_imatrix) {
                kvo.val_i64 = 0;
            } else {
                kvo.val_i64 = imatrix_data.size();
            }
            kv_overrides.emplace_back(std::move(kvo));
        }

        if (m_last_call > 0) {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
            if (hide_imatrix) {
                kvo.val_i64 = 0;
            } else {
                kvo.val_i64 = m_last_call;
            }
            kv_overrides.emplace_back(std::move(kvo));
        }
    }
    if (!kv_overrides.empty()) {
        kv_overrides.emplace_back();
        kv_overrides.back().key[0] = 0;
        params.kv_overrides = &kv_overrides;
    }
    if (!pxqu_arg.empty()) {
        std::string path = pxqu_arg;
        if (path.find('/') == std::string::npos && path.find(".tiers") == std::string::npos) {
            const char * dir = getenv("PXA_PXQU_DIR");
            path = std::string(dir ? dir : "pxa-bench/pxq-universal") + "/" + path + ".tiers";
        }
        std::string line, joined;
        std::ifstream tf(path);
        if (tf) {
            while (std::getline(tf, line)) {
                if (line.empty() || line[0] == '#') continue;
                if (!joined.empty()) joined += ',';
                joined += line;
            }
        } else {
            fprintf(stderr, "--pxq-universal: cannot open tier map %s\n", path.c_str());
            fprintf(stderr, "--pxq-universal takes a path to a .tiers map ('#'-commented lines of "
                            "regex=type, one per expert tensor). See docs/PXQU-CONVERT.md.\n");
            return 1;
        }
        if (joined.empty() || !parse_custom_quants(joined, custom_quants)) {
            fprintf(stderr, "--pxq-universal: bad tier map %s\n", path.c_str());
            return 1;
        }
        printf("--pxq-universal: %zu tier rules from %s\n", custom_quants.size(), path.c_str());
    }
    if (!custom_quants.empty()) {
        params.custom_quants = &custom_quants;
    }

    llama_backend_init();

    // parse command line arguments
    const std::string fname_inp = argv[arg_idx];
    arg_idx++;
    std::string fname_out;

    std::string ftype_str;
    std::string suffix = ".gguf";
    if (try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
        std::string fpath;
        const size_t pos = fname_inp.find_last_of("/\\");
        if (pos != std::string::npos) {
            fpath = fname_inp.substr(0, pos + 1);
        }

        // export as [inp path]/ggml-model-[ftype]. Only add extension if there is no splitting
        fname_out = fpath + "ggml-model-" + ftype_str;
        if (!params.keep_split) {
            fname_out += suffix;
        }
        arg_idx++;
        if (ftype_str == "COPY") {
            params.only_copy = true;
        }
    } else {
        fname_out = argv[arg_idx];
        if (params.keep_split && fname_out.find(suffix) != std::string::npos) {
            fname_out = fname_out.substr(0, fname_out.length() - suffix.length());
        }
        arg_idx++;

        if (argc <= arg_idx) {
            fprintf(stderr, "%s: missing ftype\n", __func__);
            return 1;
        }
        if (!try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
            fprintf(stderr, "%s: invalid ftype '%s'\n", __func__, argv[arg_idx]);
            return 1;
        }
        if (ftype_str == "COPY") {
           params.only_copy = true;
        }
        arg_idx++;
    }

    // parse nthreads
    if (argc > arg_idx) {
        try {
            params.nthread = std::stoi(argv[arg_idx]);
        }
        catch (const std::exception & e) {
            fprintf(stderr, "%s: invalid nthread '%s' (%s)\n", __func__, argv[arg_idx], e.what());
            return 1;
        }
    }

    if (!params.ignore_imatrix_rules && imatrix_data.empty() &&
        (params.ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || params.ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS ||
         params.ftype == LLAMA_FTYPE_MOSTLY_IQ2_S  ||
         params.ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S ||
         params.ftype == LLAMA_FTYPE_MOSTLY_IQ1_S  ||
         params.ftype == LLAMA_FTYPE_MOSTLY_IQ1_M)) {
        fprintf(stderr, "\n==========================================================================================================\n");
        fprintf(stderr, "Please do not use IQ1_S, IQ1_M, IQ2_S, IQ2_XXS, IQ2_XS or Q2_K_S quantization without an importance matrix\n");
        fprintf(stderr, "==========================================================================================================\n\n\n");
        return 1;
    }

    print_build_info();

    // ---------------------------------------------------------------------------------
    // PXQ NAME/TYPE MISMATCH GUARD (2026-08-25).
    // Real user report: `llama-quantize --allow-requantize in.gguf out-PXQ4.gguf 12`
    // ran for twelve minutes and produced a Q3_K. `12` is stock Q3_K_M; PXQ4 is 252.
    // Two numbering spaces overlap in one positional argument - every llama.cpp doc,
    // tutorial and older --help in the world hands people a number from 0..38, and we
    // accepted it silently while the output filename said PXQ4.
    // The filename is the user's stated intent. When it disagrees with the type, stop
    // BEFORE doing the work rather than after.
    {
        const bool ftype_is_pxq =
            params.ftype == (llama_ftype) 248 || params.ftype == (llama_ftype) 252 ||
            params.ftype == (llama_ftype) 253 || params.ftype == (llama_ftype) 254 ||
            params.ftype == (llama_ftype) 255 || params.ftype == (llama_ftype) 256 ||
            params.ftype == (llama_ftype) 257;

        std::string base = fname_out;
        const size_t slash = base.find_last_of("/\\");
        if (slash != std::string::npos) base = base.substr(slash + 1);
        for (auto & ch : base) ch = (char) std::toupper((unsigned char) ch);
        const bool name_says_pxq = base.find("PXQ") != std::string::npos;

        if (name_says_pxq && !ftype_is_pxq && !pxq_name_guard_override) {
            fprintf(stderr,
                "\n"
                "REFUSING: the output filename says PXQ but the requested type is %s (ftype %d),\n"
                "which is NOT a PXQ tier. Nothing has been written.\n"
                "\n"
                "  you asked for : %s   (ftype %d)\n"
                "  you named     : %s\n"
                "\n"
                "This is almost always a number from the WRONG numbering space. Stock llama.cpp\n"
                "ftypes run 0-38; the PXQ tiers are 248 and 252-257. Pass the NAME, not a number:\n"
                "\n"
                "  PXQ4  PXQ4-HQ  PXQ6  PXQ3  PXQ2  PXQ1  PXQ_UNIVERSAL\n"
                "\n"
                "On a MoE, a bare PXQ4 is the whole command - the native expert path claims\n"
                "ffn_{down,gate,up}_exps itself, no --custom-q needed.\n"
                "\n"
                "If you really do want %s written to a PXQ-named file, pass --pxq-name-override.\n"
                "\n",
                ftype_str.c_str(), (int) params.ftype,
                ftype_str.c_str(), (int) params.ftype,
                fname_out.c_str(),
                ftype_str.c_str());
            return 1;
        }
        if (ftype_is_pxq && !name_says_pxq) {
            fprintf(stderr,
                "%s: note: writing %s to '%s' - the filename does not carry the tier. "
                "Naming it *-%s.gguf saves the next person a header dump.\n",
                __func__, ftype_str.c_str(), fname_out.c_str(), ftype_str.c_str());
        }
    }

    fprintf(stderr, "%s: quantizing '%s' to '%s' as %s", __func__, fname_inp.c_str(), fname_out.c_str(), ftype_str.c_str());
    if (params.nthread > 0) {
        fprintf(stderr, " using %d threads", params.nthread);
    }
    fprintf(stderr, "\n");

    const int64_t t_main_start_us = llama_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = llama_time_us();

    // -------------------------------------------------------------------------------------
    // IMATRIX PROVENANCE MUST DESCRIBE WHAT WAS CONSUMED, NOT WHAT WAS SUPPLIED.
    //
    // The imatrix KVs above are assembled from the mere presence of --imatrix, and they are
    // assembled BEFORE params.ftype is parsed from argv - so the target type is not knowable
    // at that point. (A first attempt at this fix put the check up there and it silently
    // never fired, because params.ftype was still its default.) The decision therefore has
    // to happen here, once the target IS known and before llama_model_quantize consumes
    // params.kv_overrides.
    //
    // Why it matters: since 2026-08-25 the PXQ tiers IGNORE an offered imatrix by default
    // (measured net-negative on the PXQ lattice, 8.3076 without vs 8.3542 with). A PXQ4 file
    // built with --imatrix was therefore advertising provenance it did not have - the
    // imatrix touched zero tensors and payloads were byte-identical to a no-imatrix build,
    // yet quantize.imatrix.file was written. This tree's own audit tooling reads that key to
    // decide whether an artifact was imatrix-quantized, so it false-positived on our files.
    //
    // Narrow by construction: only a PXQ TARGET with the gate on. K-quant targets untouched,
    // and params.imatrix is still passed through - the library gate decides per tier, and a
    // Q6_K output head can legitimately consume an imatrix on a PXQ target.
    {
        const bool pxq_target =
            params.ftype == (llama_ftype) 248 || params.ftype == (llama_ftype) 252 ||
            params.ftype == (llama_ftype) 253 || params.ftype == (llama_ftype) 254 ||
            params.ftype == (llama_ftype) 255 || params.ftype == (llama_ftype) 256 ||
            params.ftype == (llama_ftype) 257;
        const char * e = getenv("PXA_PXQ_IMX");
        const bool optin = e && atoi(e) != 0;
        if (pxq_target && !optin && !imatrix_data.empty()) {
            static const char * drop[] = {
                LLM_KV_QUANTIZE_IMATRIX_FILE, LLM_KV_QUANTIZE_IMATRIX_DATASET,
                LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES, LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS,
            };
            // drop the sentinel first; it must stay LAST after we are done editing
            if (!kv_overrides.empty() && kv_overrides.back().key[0] == 0) {
                kv_overrides.pop_back();
            }
            kv_overrides.erase(
                std::remove_if(kv_overrides.begin(), kv_overrides.end(),
                    [&](const llama_model_kv_override & o) {
                        for (const char * d : drop) {
                            if (std::strcmp(o.key, d) == 0) return true;
                        }
                        return false;
                    }),
                kv_overrides.end());
            // Removing our own overrides is NOT enough. A requantize INHERITS the source
            // GGUF's imatrix KVs, so dropping ours just exposes the previous quantizer's
            // claim - measured: a PXQ4 built from an Unsloth Q8_0 came out advertising
            // 'Qwen3-0.6B-GGUF/imatrix_unsloth.dat', an imatrix this run never opened.
            // (That inheritance is a pre-existing bug in its own right and is not limited
            // to PXQ: any K-quant requantize re-asserts its source's provenance too.)
            // So OVERRIDE the keys with truthful values rather than deleting them - an
            // auditor reading entries_count gets 0, which is the honest answer.
            auto push_str = [&](const char * k, const char * v) {
                llama_model_kv_override o;
                std::memset(&o, 0, sizeof(o));
                std::strcpy(o.key, k);
                o.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
                strncpy(o.val_str, v, 127);
                o.val_str[127] = '\0';
                kv_overrides.emplace_back(std::move(o));
            };
            push_str(LLM_KV_QUANTIZE_IMATRIX_FILE,    "");
            push_str(LLM_KV_QUANTIZE_IMATRIX_DATASET, "");
            {
                llama_model_kv_override o;
                std::memset(&o, 0, sizeof(o));
                std::strcpy(o.key, LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES);
                o.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
                o.val_i64 = 0;
                kv_overrides.emplace_back(std::move(o));
            }
            push_str("quantize.imatrix.ignored_by", "pxq-tiers (PXA_PXQ_IMX unset)");
            kv_overrides.emplace_back();
            kv_overrides.back().key[0] = 0;
            params.kv_overrides = &kv_overrides;
            fprintf(stderr,
                "main: NOTE: --imatrix supplied, but the PXQ tiers ignore it by default. "
                "Writing quantize.imatrix.ignored_by instead of the standard provenance keys, "
                "which would claim a consumption that did not happen. PXA_PXQ_IMX=1 to "
                "consume it.\n");
        }
    }

        if (llama_model_quantize(fname_inp.c_str(), fname_out.c_str(), &params)) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = llama_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = llama_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0);
    }

    llama_backend_free();

    return 0;
}
