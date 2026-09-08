// pxq-export.cpp — llama-pxq-export: turn a PXQ slab GGUF back into a stock-loadable GGUF.
//
// WHY THIS EXISTS
// ---------------
// The PXQ types (PXQ1 248, PXQ4 252, PXQ4HQ 253, PXQ2 254, PXQ3 255, PXQ6 256) are 64-row
// PANEL-interleaved CUDA-consumer formats. A single row's bytes are scattered across the slabs
// of its panel, so their ggml type_traits carry no to_float and no vec_dot, and
// src/llama-quantize.cpp refuses them outright:
//
//     cannot requantize from a PXQ slab type (... CUDA-only slab layout, no CPU codec)
//
// That is correct, but it means a PXQ artifact is a terminal format: you cannot move it to
// Q4_K_M, you cannot hand it to stock llama.cpp, and if the original BF16 is gone the weights
// are locked to this engine. This tool is the way out. It reads a PXQ GGUF, decodes every PXQ
// tensor with the SAME CUDA kernels the runtime uses (pxa_pxq_dequant_host ->
// ggml_get_to_fp16_cuda), writes it as F16 (or F32), and copies every other tensor byte for
// byte. The result is an ordinary GGUF: stock `llama-quantize --allow-requantize` will take it
// to Q4_K_M / MXFP4 / anything, and stock llama.cpp and ik will load whatever comes out.
//
// Step 1 adds no quantization error: a PXQ weight is anchor * sub * book[code], the decode
// evaluates that product in fp32, and every stored code maps to exactly one value. --type f32
// keeps that fp32 value exactly; --type f16 rounds it once on store -- the SAME single rounding
// the serving path's dequant -> cuBLAS fallback does before every GEMM, so the file holds the
// numbers this engine multiplies. The SECOND quantization (step 2) is the lossy one. See
// docs/PXQ-EXPORT.md for the full numerical story and when to go back to the original BF16.
//
// STREAMING
// ---------
// Models are 40-200 GB. Nothing is ever fully resident: metadata is read with no_alloc, tensor
// bytes are pulled from the input file one tensor at a time, and a PXQ tensor is decoded in
// chunks of whole 64-row panels (--chunk-mib, default 256). Peak host footprint is one chunk in
// plus one chunk out; peak device footprint is the same. A 200 GB model exports on a card with
// 1 GiB free.
//
// SHAPES
// ------
// A PXQ tensor is E * (ne1/64) contiguous 64-row panels, panels row-major, experts outermost
// (src/llama-quantize.cpp pxa_pxq_slab_size). So a 3-D MoE expert tensor needs no special
// casing at all: nrows = ne1*ne2*ne3 decodes the whole thing, and any 64-row-aligned prefix of
// the panel run decodes independently. That is what makes the chunking legal.
//
// TENSORS THAT ARE NEVER PXQ (verified, not assumed -- the tool refuses if it sees one):
//   * row-gather / GET_ROWS tables ("per_layer_token_embd", or ne1 >= 1e6 -- the qwen4exp PLE
//     n-gram-hash table). pxa_is_row_gather_tensor in the quantizer rules panel codecs out of
//     these as a CORRECTNESS gate, not a preference: a panel row is unreadable in isolation.
//   * ne0 % 32 != 0 (ssm_conv1d / ple_conv1d are [4, 10240]) -- every PXQ codec asserts
//     n_per_row % 32 == 0.
//   * ne1 % 64 != 0 -- below one panel.
//   * norms, biases, 1-D tensors, the routing tables.
// All of those arrive here as F32/F16/MXFP4/Q*_K and are copied verbatim. The output head
// (output.weight) is quantized like any other 2-D weight and needs no special case either: if
// it landed on a PXQ type it is decoded, otherwise it is copied.
//
#include "ggml.h"
#include "llama.h"

#ifdef PXQ_EXPORT_CUDA
#include "ggml-cuda.h"
#endif

#include "pxq-cpu.h"   // panel-aware CPU dequant: the --cpu path and the --verify cross-check

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// ------------------------------------------------------------------------------------------

static bool pxq_is_slab_type(ggml_type t) {
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

// mirrors pxa_is_row_gather_tensor (src/llama-quantize.cpp): a table that is only ever read one
// row at a time by GET_ROWS. The quantizer never gives one a panel codec; if one shows up here
// as PXQ the file is broken and decoding it would silently produce a plausible wrong answer.
static bool pxq_is_row_gather(const char * name, int64_t ne1) {
    return strstr(name, "per_layer_token_embd") != nullptr || ne1 >= 1000000;
}

static void zeros(std::ofstream & f, size_t n) {
    // block-wise: the metadata placeholder can be several MiB (a big vocab array), and a
    // byte-at-a-time loop over that is a measurable share of a small export's runtime.
    static const char block[4096] = {0};
    while (n > 0) {
        const size_t k = n < sizeof(block) ? n : sizeof(block);
        f.write(block, (std::streamsize) k);
        n -= k;
    }
}

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s in.gguf out.gguf [--type f16|f32] [--device N|--cpu] [--verify] [--chunk-mib N]\n"
        "\n"
        "  Dequantizes every PXQ slab tensor (PXQ1/PXQ2/PXQ3/PXQ4/PXQ4HQ/PXQ6) of in.gguf to\n"
        "  F16 (default) or F32 -- on the GPU by default, or on the CPU with --cpu -- copies\n"
        "  every other tensor\n"
        "  byte for byte, and writes out.gguf -- a file stock llama-quantize and stock\n"
        "  llama.cpp accept.\n"
        "\n"
        "  --type f16|f32   landing type for PXQ tensors (default f16)\n"
        "  --device N       CUDA device to decode on (default 0)\n"
        "  --cpu            decode with the panel dequant in ggml/src/pxq-cpu.c instead of\n"
        "                   CUDA: same values (0 ULP against the kernels), much slower, no GPU\n"
        "  --verify         re-decode each PXQ tensor whole and require the streamed, chunked\n"
        "                   result to be bit-identical; also cross-checks the CPU panel\n"
        "                   dequant where one exists. Costs one tensor of host RAM.\n"
        "  --chunk-mib N    decode chunk budget in MiB (default 256)\n",
        argv0);
}

// distance in representable f16 values between two halves stored as raw bits (sign-magnitude
// ordered). 0 == bit-identical.
static uint32_t f16_ulp_diff(uint16_t a, uint16_t b) {
    auto ord = [](uint16_t h) -> int32_t {
        const int32_t m = h & 0x7fff;
        return (h & 0x8000) ? -m : m;
    };
    const int64_t d = (int64_t) ord(a) - (int64_t) ord(b);
    return (uint32_t) (d < 0 ? -d : d);
}

int main(int argc, char ** argv) {
    const char * fname_in  = nullptr;
    const char * fname_out = nullptr;
    ggml_type    dst_type  = GGML_TYPE_F16;
    int          device    = 0;
    bool         verify    = false;
    bool         use_cpu   = false;
    size_t       chunk_mib = 256;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else if (a == "--type" && i + 1 < argc) {
            const std::string t = argv[++i];
            if      (t == "f16" || t == "F16") dst_type = GGML_TYPE_F16;
            else if (t == "f32" || t == "F32") dst_type = GGML_TYPE_F32;
            else { fprintf(stderr, "%s: --type must be f16 or f32 (got '%s')\n", __func__, t.c_str()); return 1; }
        } else if (a == "--device" && i + 1 < argc) {
            device = atoi(argv[++i]);
        } else if (a == "--cpu") {
            use_cpu = true;
        } else if (a == "--verify") {
            verify = true;
        } else if (a == "--chunk-mib" && i + 1 < argc) {
            chunk_mib = (size_t) atoll(argv[++i]);
            if (chunk_mib == 0) chunk_mib = 1;
        } else if (a.size() > 1 && a[0] == '-') {
            fprintf(stderr, "%s: unknown argument '%s'\n", __func__, a.c_str());
            usage(argv[0]);
            return 1;
        } else if (!fname_in) {
            fname_in = argv[i];
        } else if (!fname_out) {
            fname_out = argv[i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (!fname_in || !fname_out) {
        usage(argv[0]);
        return 1;
    }

    // --------------------------------------------------------------------------------------
    // read the input metadata (no_alloc: tensor shapes and offsets only, no data)
    // --------------------------------------------------------------------------------------
    struct ggml_context * ctx_meta = nullptr;
    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx_meta };
    struct gguf_context * ctx_in = gguf_init_from_file(fname_in, gp);
    if (!ctx_in) {
        fprintf(stderr, "%s: failed to read %s\n", __func__, fname_in);
        return 1;
    }

    const int    n_tensors   = gguf_get_n_tensors(ctx_in);
    const size_t data_offset = gguf_get_data_offset(ctx_in);
    const size_t align_in    = gguf_get_alignment(ctx_in);

    // A split file cannot be exported in place: the tensor set is spread over N files and the
    // split KVs would describe a shape this tool did not produce. Merge first (llama-gguf-split
    // --merge), then export.
    if (gguf_find_key(ctx_in, "split.count") >= 0) {
        fprintf(stderr, "%s: %s is a split GGUF -- merge it first (llama-gguf-split --merge)\n", __func__, fname_in);
        return 1;
    }

    // --------------------------------------------------------------------------------------
    // PXQ book provenance guard.
    //
    // The PXQ2/PXQ3 codebooks are ENV-ARMED at decode time (PXA_PXQ2_V3, PXA_PXQ_CEIL_V2 --
    // ggml/src/ggml-cuda/pxa/pxq23.cuh pxq23_maybe_upload_books) and the quantizer stamps the
    // version it used into the file. Decoding a v2/v3 file with v1 tables produces numbers that
    // are wrong and completely plausible. The loader only WARNS about this (a running server
    // can be restarted); an export bakes the mistake into a new artifact, so here it is fatal.
    // --------------------------------------------------------------------------------------
    {
        const char * ce = getenv("PXA_PXQ_CEIL_V2");
        const char * ve = getenv("PXA_PXQ2_V3");
        const bool ceil_v2 = ce && atoi(ce) != 0;
        const bool p2v3    = ve && atoi(ve) != 0;
        const char * vkeys[2] = { "pxa.pxq2.version", "pxa.pxq3.version" };
        const uint32_t rt_v[2] = { p2v3 ? 3u : (ceil_v2 ? 2u : 1u), ceil_v2 ? 2u : 1u };
        const char * fix[2] = {
            "v3 -> PXA_PXQ2_V3=1, v2 -> PXA_PXQ_CEIL_V2=1, v1 -> neither",
            "v2 -> PXA_PXQ_CEIL_V2=1, v1 -> unset PXA_PXQ_CEIL_V2" };
        for (int vi = 0; vi < 2; ++vi) {
            const int ki = gguf_find_key(ctx_in, vkeys[vi]);
            if (ki < 0 || gguf_get_kv_type(ctx_in, ki) != GGUF_TYPE_UINT32) continue;
            const uint32_t fv = gguf_get_val_u32(ctx_in, ki);
            if (fv != rt_v[vi]) {
                fprintf(stderr, "%s: PXQ TABLE MISMATCH: %s = %u but this process's env selects v%u "
                                "tables. The export would bake wrong weights. Set the env to %s, then re-run.\n",
                        __func__, vkeys[vi], fv, rt_v[vi], fix[vi]);
                return 1;
            }
        }
        for (const char * e : { "PXA_PXQ6_BOOK", "PXA_PXQ6_SUB", "PXA_PXQ6_SUB_HQ",
                                "PXA_PXQ6R_BOOK", "PXA_PXQ2_BOOK", "PXA_PXQ3_BOOK" }) {
            if (getenv(e)) {
                fprintf(stderr, "%s: warning: %s overrides a decode codebook -- the export uses it. "
                                "It must match the book this file was quantized with.\n", __func__, e);
            }
        }
    }

    // --------------------------------------------------------------------------------------
    // classify tensors + validate every PXQ one before writing anything
    // --------------------------------------------------------------------------------------
    struct tinfo {
        struct ggml_tensor * t;
        bool     is_pxq;
        int64_t  nrows;      // ne1*ne2*ne3 == total panel rows
        size_t   src_bytes;  // ggml_nbytes of the PXQ slab run
        size_t   panel_stride;
    };
    std::vector<tinfo> tis;
    tis.reserve(n_tensors);

    int n_pxq = 0;
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_in, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
        if (!t) {
            fprintf(stderr, "%s: tensor '%s' is in the gguf index but not in the meta context\n", __func__, name);
            return 1;
        }
        tinfo ti = { t, pxq_is_slab_type(t->type), 0, 0, 0 };
        if (ti.is_pxq) {
            ti.nrows     = t->ne[1] * t->ne[2] * t->ne[3];
            ti.src_bytes = ggml_nbytes(t);
            if (t->ne[0] % 32 != 0 || ti.nrows % 64 != 0) {
                fprintf(stderr, "%s: '%s' is %s but its shape [%" PRId64 ", %" PRId64 ", %" PRId64 "] is not "
                                "slab-aligned (ne0 %% 32, rows %% 64) -- this file cannot have been written by "
                                "this quantizer\n",
                        __func__, name, ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2]);
                return 1;
            }
            if (pxq_is_row_gather(name, t->ne[1])) {
                fprintf(stderr, "%s: '%s' is a row-gather (GET_ROWS) table stored as %s. The quantizer "
                                "refuses panel codecs for these because a panel row is unreadable in "
                                "isolation; this file is already broken and will not be exported.\n",
                        __func__, name, ggml_type_name(t->type));
                return 1;
            }
            ti.panel_stride = ti.src_bytes / (size_t) (ti.nrows / 64);
            ++n_pxq;
        }
        tis.push_back(ti);
    }

    printf("%s: %s -> %s\n", __func__, fname_in, fname_out);
    printf("%s: %d tensors, %d PXQ slab tensors -> %s\n",
           __func__, n_tensors, n_pxq, ggml_type_name(dst_type));

#ifndef PXQ_EXPORT_CUDA
    if (n_pxq > 0 && !use_cpu) {
        fprintf(stderr, "%s: this binary was built without CUDA; decode with --cpu, or rebuild "
                        "with -DGGML_CUDA=ON.\n", __func__);
        return 1;
    }
#endif
    bool cuda_available = false;
#ifdef PXQ_EXPORT_CUDA
    if (n_pxq > 0) {
        const int n_dev = ggml_backend_cuda_get_device_count();
        cuda_available = device >= 0 && device < n_dev;
        if (!use_cpu) {
            if (!cuda_available) {
                fprintf(stderr, "%s: --device %d out of range (%d CUDA device(s) visible); "
                                "use --cpu to decode without a GPU\n", __func__, device, n_dev);
                return 1;
            }
            char desc[256] = {0};
            ggml_backend_cuda_get_device_description(device, desc, sizeof(desc));
            printf("%s: decoding on CUDA device %d (%s)\n", __func__, device, desc);
        }
    }
#endif
    if (n_pxq > 0 && use_cpu) {
        printf("%s: decoding on the CPU (ggml/src/pxq-cpu.c panel dequant)\n", __func__);
        for (const auto & ti2 : tis) {
            if (ti2.is_pxq && !pxa_pxq_is_cpu_supported(ti2.t->type)) {
                fprintf(stderr, "%s: --cpu: '%s' is %s, which has no CPU panel dequant\n",
                        __func__, ggml_get_name(ti2.t), ggml_type_name(ti2.t->type));
                return 1;
            }
        }
    }

    // --------------------------------------------------------------------------------------
    // build the output metadata
    // --------------------------------------------------------------------------------------
    struct gguf_context * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_in);

    // The file-type KV is a one-word summary of the dominant tensor type; the per-tensor types
    // in the tensor index are the authority and stay exact. Declaring what an F16/F32 GGUF
    // declares is what makes stock tooling treat this as a normal source file. (Non-PXQ
    // tensors that were MXFP4 or Q*_K are still MXFP4 / Q*_K -- exactly as in any mixed file
    // llama-quantize itself writes, which also stamps a single ftype.)
    gguf_set_val_u32(ctx_out, "general.file_type",
                     dst_type == GGML_TYPE_F32 ? (uint32_t) LLAMA_FTYPE_ALL_F32
                                               : (uint32_t) LLAMA_FTYPE_MOSTLY_F16);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);

    // Drop the PXQ provenance KVs. They are not inert decoration: pxa.pxq{2,3}.version is read
    // by llama_model_loader (src/llama-model-loader.cpp) and makes a loud PXQ TABLE MISMATCH
    // warning fire on any runtime whose env selects different books -- on a file that has no
    // PXQ tensor left to decode. pxa.pxq{2,3,6}.book / .sub are the codebooks, pxa.pxq.backbone_*
    // records which tensor classes got which tier, pxa.pxqu.version marks a UNIVERSAL mix. All
    // of it describes a layout the exported file no longer contains, so it goes. The exporter
    // stamps its own provenance instead.
    {
        std::vector<std::string> drop;
        for (int i = 0; i < gguf_get_n_kv(ctx_out); ++i) {
            const std::string k = gguf_get_key(ctx_out, i);
            if (k.rfind("pxa.pxq", 0) == 0 && k.rfind("pxa.pxq_export", 0) != 0) {
                drop.push_back(k);
            }
        }
        for (const auto & k : drop) {
            gguf_remove_key(ctx_out, k.c_str());
            printf("%s: dropped KV %s\n", __func__, k.c_str());
        }
    }
    if (n_pxq > 0) {
        gguf_set_val_str(ctx_out, "pxa.pxq_export.source_types", [&] {
            std::string s;
            for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
                if (!pxq_is_slab_type((ggml_type) t)) continue;
                int n = 0;
                for (const auto & ti : tis) if (ti.t->type == (ggml_type) t) ++n;
                if (n) { if (!s.empty()) s += ","; s += ggml_type_name((ggml_type) t); s += ":" + std::to_string(n); }
            }
            return s;
        }().c_str());
    }

    for (const auto & ti : tis) {
        gguf_add_tensor(ctx_out, ti.t);
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    if (align_in != align) {
        // gguf_init_empty() fixes the output alignment at GGUF_DEFAULT_ALIGNMENT and
        // gguf_set_kv copies general.alignment as a KV without changing it, which would make
        // the stamped KV disagree with the real offsets.
        fprintf(stderr, "%s: input alignment %zu != %zu; not supported\n", __func__, align_in, align);
        return 1;
    }

    std::ifstream fin(fname_in, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open %s for reading\n", __func__, fname_in);
        return 1;
    }
    std::ofstream fout(fname_out, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "%s: failed to open %s for writing\n", __func__, fname_out);
        return 1;
    }
    fout.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    zeros(fout, gguf_get_meta_size(ctx_out));   // placeholder, rewritten at the end

    // --------------------------------------------------------------------------------------
    // stream
    // --------------------------------------------------------------------------------------
    std::vector<char>    buf_src;   // raw input bytes (one tensor, or one panel chunk)
    std::vector<char>    buf_dst;   // decoded output bytes
    std::vector<char>    buf_ref;   // --verify: what was actually written, per tensor
    std::vector<char>    buf_alt;   // --verify: the other engine's decode of the same tensor

    std::vector<float>   cpu_f32;   // --cpu f16 staging (the panel dequant always yields f32)

    const size_t dst_elem   = ggml_type_size(dst_type);      // 2 (f16) or 4 (f32)
    const size_t chunk_budget = chunk_mib * 1024 * 1024;

    // One decode step over a run of whole 64-row panels: rows*n_per_row values of dst_type.
    // `cpu` picks the engine. The two agree to 0 ULP on every tier (both evaluate the same
    // fp32 contract eff = anchor*SUB16[s4]; w = eff*book[c]); --verify measures it rather than
    // assuming it.
    auto decode_chunk = [&](ggml_type type, const void * src, size_t src_bytes,
                            int64_t rows, int64_t n_per_row, void * dst, bool cpu) -> bool {
        if (!cpu) {
#ifdef PXQ_EXPORT_CUDA
            return pxa_pxq_dequant_host(device, type, dst_type, src, src_bytes, rows, n_per_row, dst);
#else
            (void) type; (void) src; (void) src_bytes; (void) rows; (void) n_per_row; (void) dst;
            return false;
#endif
        }
        (void) src_bytes;
        if (!pxa_pxq_is_cpu_supported(type)) return false;
        const size_t n = (size_t) rows * (size_t) n_per_row;
        if (dst_type == GGML_TYPE_F32) {
            pxa_pxq_dequant_2d(type, src, (float *) dst, rows, n_per_row);
        } else {
            if (cpu_f32.size() < n) cpu_f32.resize(n);
            pxa_pxq_dequant_2d(type, src, cpu_f32.data(), rows, n_per_row);
            ggml_fp16_t * o = (ggml_fp16_t *) dst;
            for (size_t e = 0; e < n; ++e) o[e] = ggml_fp32_to_fp16(cpu_f32[e]);
        }
        return true;
    };

    size_t total_in = 0, total_out = 0;
    int n_copied = 0, n_decoded = 0;
    uint32_t worst_ulp = 0;
    const char * worst_ulp_name = "";
    std::map<int, uint32_t> worst_by_type;
    std::map<int, int>      n_cross_checked;

    for (int i = 0; i < n_tensors; ++i) {
        const tinfo & ti = tis[i];
        struct ggml_tensor * t = ti.t;
        const char * name = ggml_get_name(t);
        const size_t src_bytes = ggml_nbytes(t);
        const size_t src_off   = data_offset + gguf_get_tensor_offset(ctx_in, i);

        total_in += src_bytes;

        if (!ti.is_pxq) {
            // byte-for-byte copy, streamed in <= chunk_budget pieces so a 95 GiB PLE table
            // does not have to be resident.
            printf("[%4d/%4d] %-40s %6s  copy   %8.2f MiB\n",
                   i + 1, n_tensors, name, ggml_type_name(t->type), src_bytes/1024.0/1024.0);
            fin.seekg((std::streamoff) src_off);
            size_t left = src_bytes;
            if (buf_src.size() < std::min(left, chunk_budget)) buf_src.resize(std::min(left, chunk_budget));
            while (left > 0) {
                const size_t n = std::min(left, buf_src.size());
                fin.read(buf_src.data(), (std::streamsize) n);
                if (!fin) { fprintf(stderr, "%s: short read on '%s'\n", __func__, name); return 1; }
                fout.write(buf_src.data(), (std::streamsize) n);
                left -= n;
            }
            zeros(fout, GGML_PAD(src_bytes, align) - src_bytes);
            total_out += src_bytes;
            ++n_copied;
            continue;
        }

        const int64_t n_per_row = t->ne[0];
        const int64_t nrows     = ti.nrows;
        const size_t  out_bytes = (size_t) nrows * (size_t) n_per_row * dst_elem;

        // chunk in whole 64-row panels: panel p lives at data + p*panel_stride and decodes
        // independently (the anchor header is per panel, the scales are per slab).
        int64_t rows_per_chunk = (int64_t) (chunk_budget / ((size_t) n_per_row * dst_elem));
        rows_per_chunk -= rows_per_chunk % 64;
        if (rows_per_chunk < 64) rows_per_chunk = 64;
        if (rows_per_chunk > nrows) rows_per_chunk = nrows;
        const int n_chunks = (int) ((nrows + rows_per_chunk - 1) / rows_per_chunk);

        printf("[%4d/%4d] %-40s %6s  dequant %8.2f MiB -> %8.2f MiB (%d chunk%s, %s)\n",
               i + 1, n_tensors, name, ggml_type_name(t->type),
               src_bytes/1024.0/1024.0, out_bytes/1024.0/1024.0, n_chunks, n_chunks == 1 ? "" : "s",
               use_cpu ? "cpu" : "cuda");

        const size_t chunk_src_bytes = (size_t) (rows_per_chunk / 64) * ti.panel_stride;
        const size_t chunk_dst_bytes = (size_t) rows_per_chunk * (size_t) n_per_row * dst_elem;
        if (buf_src.size() < chunk_src_bytes) buf_src.resize(chunk_src_bytes);
        if (buf_dst.size() < chunk_dst_bytes) buf_dst.resize(chunk_dst_bytes);
        if (verify && buf_ref.size() < out_bytes) buf_ref.resize(out_bytes);

        for (int64_t r0 = 0; r0 < nrows; r0 += rows_per_chunk) {
            const int64_t rows = std::min(rows_per_chunk, nrows - r0);
            const size_t  cs   = (size_t) (rows / 64) * ti.panel_stride;
            fin.seekg((std::streamoff) (src_off + (size_t) (r0 / 64) * ti.panel_stride));
            fin.read(buf_src.data(), (std::streamsize) cs);
            if (!fin) { fprintf(stderr, "%s: short read on '%s'\n", __func__, name); return 1; }

            if (!decode_chunk(t->type, buf_src.data(), cs, rows, n_per_row, buf_dst.data(), use_cpu)) {
                fprintf(stderr, "%s: dequant failed for '%s' (%s, rows %" PRId64 ", ne0 %" PRId64 ")\n",
                        __func__, name, ggml_type_name(t->type), rows, n_per_row);
                return 1;
            }
            const size_t nb = (size_t) rows * (size_t) n_per_row * dst_elem;
            fout.write(buf_dst.data(), (std::streamsize) nb);
            if (verify) {
                memcpy(buf_ref.data() + (size_t) r0 * (size_t) n_per_row * dst_elem, buf_dst.data(), nb);
            }
        }
        zeros(fout, GGML_PAD(out_bytes, align) - out_bytes);
        total_out += out_bytes;
        ++n_decoded;

        if (verify) {
            // (a) chunked == whole-tensor, on the SAME engine. Proves the panel arithmetic, the
            //     file seeks and the chunk boundaries -- not just the decoder.
            std::vector<char> whole_src(src_bytes);
            std::vector<char> whole_dst(out_bytes);
            fin.seekg((std::streamoff) src_off);
            fin.read(whole_src.data(), (std::streamsize) src_bytes);
            if (!decode_chunk(t->type, whole_src.data(), src_bytes, nrows, n_per_row,
                              whole_dst.data(), use_cpu)) {
                fprintf(stderr, "%s: verify: whole-tensor dequant failed for '%s'\n", __func__, name);
                return 1;
            }
            if (memcmp(whole_dst.data(), buf_ref.data(), out_bytes) != 0) {
                fprintf(stderr, "%s: VERIFY FAILED: chunked dequant of '%s' differs from the "
                                "whole-tensor dequant\n", __func__, name);
                return 1;
            }

            // (b) the OTHER engine. The CUDA kernels (ggml-cuda/pxq6.cuh, pxq23.cuh) and the
            //     CPU panel dequant (ggml/src/pxq-cpu.c) are independent implementations of the
            //     same frozen format; the CPU one is additionally bit-verified against an
            //     element-indexed reference in tests/test-pxq-cpu-dequant.cpp. Running both and
            //     differencing them is the strongest correctness statement available offline.
            const bool other_is_cpu = !use_cpu;
            const bool other_usable = other_is_cpu ? pxa_pxq_is_cpu_supported(t->type)
                                                   : cuda_available;
            if (other_usable) {
                if (buf_alt.size() < out_bytes) buf_alt.resize(out_bytes);
                if (!decode_chunk(t->type, whole_src.data(), src_bytes, nrows, n_per_row,
                                  buf_alt.data(), other_is_cpu)) {
                    fprintf(stderr, "%s: verify: cross-check decode failed for '%s'\n", __func__, name);
                    return 1;
                }
                uint32_t worst = 0;
                if (dst_type == GGML_TYPE_F16) {
                    const uint16_t * a = (const uint16_t *) buf_ref.data();
                    const uint16_t * b = (const uint16_t *) buf_alt.data();
                    for (size_t e = 0; e < out_bytes/2; ++e) {
                        const uint32_t u = f16_ulp_diff(a[e], b[e]);
                        if (u > worst) worst = u;
                    }
                } else {
                    const uint32_t * a = (const uint32_t *) buf_ref.data();
                    const uint32_t * b = (const uint32_t *) buf_alt.data();
                    for (size_t e = 0; e < out_bytes/4; ++e) {
                        if (a[e] != b[e]) { worst = 1; break; }   // fp32: bit-exact or not
                    }
                }
                if (worst > worst_ulp) { worst_ulp = worst; worst_ulp_name = name; }
                if (worst > worst_by_type[t->type]) worst_by_type[t->type] = worst;
                ++n_cross_checked[t->type];
            }
        }

        gguf_set_tensor_type(ctx_out, name, dst_type);
        gguf_set_tensor_data(ctx_out, name, nullptr, (size_t) ti.nrows * (size_t) t->ne[0] * dst_elem);
    }

    // rewrite the real metadata over the placeholder
    {
        std::vector<uint8_t> meta(gguf_get_meta_size(ctx_out));
        gguf_get_meta_data(ctx_out, meta.data());
        fout.seekp(0);
        fout.write((const char *) meta.data(), (std::streamsize) meta.size());
    }
    fout.close();

    printf("%s: copied %d tensor(s) verbatim, dequantized %d PXQ tensor(s)\n", __func__, n_copied, n_decoded);
    printf("%s: %.2f MiB in -> %.2f MiB out (tensor data)\n",
           __func__, total_in/1024.0/1024.0, total_out/1024.0/1024.0);
    if (verify) {
        printf("%s: verify: chunked == whole-tensor dequant for all %d PXQ tensor(s) (0 byte diffs)\n",
               __func__, n_decoded);
        int total_cross = 0;
        for (const auto & kv : n_cross_checked) total_cross += kv.second;
        if (total_cross > 0) {
            printf("%s: verify: %s cross-check on %d tensor(s), worst %u ULP overall%s%s\n",
                   __func__, use_cpu ? "cuda-vs-cpu" : "cpu-vs-cuda", total_cross, worst_ulp,
                   worst_ulp ? " at " : "", worst_ulp ? worst_ulp_name : "");
            for (const auto & kv : n_cross_checked) {
                printf("%s: verify:   %-7s %4d tensor(s), worst %u ULP\n", __func__,
                       ggml_type_name((ggml_type) kv.first), kv.second, worst_by_type[kv.first]);
            }
        }
    }

    gguf_free(ctx_out);
    gguf_free(ctx_in);
    ggml_free(ctx_meta);
    return 0;
}
