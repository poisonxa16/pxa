#pragma once
// PXA_DUMP_MTP_EMBD_v1 (2026-10-14): `--dump-mtp-embd <dir>` on the server's prompt path.
//
// Writes, per request, the MTP head's CONDITIONING rows exactly as the engine hands them to the head: the
// target context's per-token feature rows that the MTP prompt warm-up reads (llama_spec_get_hidden_feature_view_for_seq
// over the decoded prompt batch -- the ctx->embd rows the backward name scan in llama_decode filled; on qwen35 that is
// the node "result_norm", i.e. the post-output_norm row, see ledger mtp-conditioning-pre-vs-post-output-norm), as fp16,
// with the token ids and the backbone's greedy next token (argmax of the target logits row) alongside:
//   <dir>/<tag>.h.npy     float16 [n, n_embd]   row t = conditioning hidden of the token at position pos0+t
//   <dir>/<tag>.tok.npy   int32   [n]          token id at position pos0+t
//   <dir>/<tag>.amax.npy  int32   [n]          argmax of the target logits at that row (-1 if no logits row)
//   <dir>/index.jsonl     one line per request {tag, n, pos0, width, task, slot, t}
// The tag is the request's "dump_tag" string (default t<task id>). Only rows the prompt pass actually decodes are
// written: send cache_prompt=false so the whole prompt is decoded. Needs the MTP head armed (--spec-type mtp:...) and
// eager prompt flags (PXA_MTP_LAZY_WARMUP=0) -- the server refuses to start the dump otherwise (see server-context.cpp).
// Off (empty dir) = no code path changes.

#include "llama.h"
#include "llama-spec-features.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <string>
#include <vector>
#include <sys/stat.h>

struct pxa_mtp_dump_buf {
    std::string              tag;
    int32_t                  width = 0;
    int32_t                  pos0  = -1;
    std::vector<int32_t>     tok;
    std::vector<int32_t>     amax;
    std::vector<ggml_fp16_t> h;

    void clear() {
        tag.clear(); width = 0; pos0 = -1;
        tok.clear(); amax.clear(); h.clear();
    }
};

static inline bool pxa_npy_write(const std::string & path, const char * descr, const std::vector<int64_t> & shape,
                                 const void * data, size_t nbytes) {
    std::string shp = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shp += std::to_string((long long) shape[i]);
        if (shape.size() == 1) shp += ",";
        else if (i + 1 < shape.size()) shp += ", ";
    }
    shp += ")";
    std::string hdr = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + shp + ", }";
    // magic(6) + version(2) + len(2) + header + '\n' padded to a multiple of 64
    size_t total = 10 + hdr.size() + 1;
    size_t pad   = (64 - total % 64) % 64;
    hdr.append(pad, ' ');
    hdr += '\n';
    const std::string tmp = path + ".part";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const unsigned char magic[8] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 };
    const uint16_t hl = (uint16_t) hdr.size();
    bool ok = fwrite(magic, 1, 8, f) == 8 &&
              fwrite(&hl, 1, 2, f) == 2 &&
              fwrite(hdr.data(), 1, hdr.size(), f) == hdr.size() &&
              (nbytes == 0 || fwrite(data, 1, nbytes, f) == nbytes);
    ok = (fclose(f) == 0) && ok;
    if (!ok) { remove(tmp.c_str()); return false; }
    return rename(tmp.c_str(), path.c_str()) == 0;
}

// Append this slot's rows of the batch just decoded. `row_of_seq` = the batch rows (in batch order) whose first seq id is
// `seq_id`; the feature view returns the same rows in the same order.
static inline int32_t pxa_mtp_dump_collect(llama_context * ctx, const llama_batch & batch, llama_seq_id seq_id,
                                           pxa_mtp_dump_buf & buf, int32_t n_vocab) {
    llama_spec_feature_view view;
    if (!llama_spec_get_hidden_feature_view_for_seq(ctx, batch, seq_id, view) || view.width <= 0) {
        return 0;
    }
    std::vector<int32_t> rows;
    rows.reserve(view.rows.size());
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        for (int32_t j = 0; j < batch.n_seq_id[i]; ++j) {
            if (batch.seq_id[i][j] == seq_id) { rows.push_back(i); break; }
        }
    }
    if (rows.size() != view.rows.size()) {
        return -1;
    }
    if (buf.width == 0) buf.width = view.width;
    if (buf.width != view.width) {
        return -1;
    }
    for (size_t k = 0; k < rows.size(); ++k) {
        const int32_t i = rows[k];
        const auto & r = view.rows[k];
        if (buf.pos0 < 0) buf.pos0 = r.pos;
        // positions must be contiguous from pos0 (a prompt-cache hit or a hole would break the row <-> position map)
        if (r.pos != buf.pos0 + (int32_t) buf.tok.size()) {
            return -1;
        }
        buf.tok.push_back(batch.token ? batch.token[i] : -1);
        const size_t off = buf.h.size();
        buf.h.resize(off + (size_t) view.width);
        for (int32_t z = 0; z < view.width; ++z) {
            buf.h[off + z] = ggml_fp32_to_fp16(r.data[z]);
        }
        int32_t best = -1;
        if (batch.logits == nullptr || batch.logits[i]) {
            const float * lg = llama_get_logits_ith(ctx, i);
            if (lg) {
                float bv = lg[0]; best = 0;
                for (int32_t v = 1; v < n_vocab; ++v) {
                    if (lg[v] > bv) { bv = lg[v]; best = v; }
                }
            }
        }
        buf.amax.push_back(best);
    }
    return (int32_t) rows.size();
}

static inline bool pxa_mtp_dump_flush(const std::string & dir, pxa_mtp_dump_buf & buf, int id_task, int id_slot) {
    const int64_t n = (int64_t) buf.tok.size();
    bool ok = n > 0 && buf.width > 0;
    if (ok) {
        const std::string base = dir + "/" + buf.tag;
        ok = pxa_npy_write(base + ".tok.npy",  "<i4", { n }, buf.tok.data(),  buf.tok.size()  * sizeof(int32_t)) &&
             pxa_npy_write(base + ".amax.npy", "<i4", { n }, buf.amax.data(), buf.amax.size() * sizeof(int32_t)) &&
             pxa_npy_write(base + ".h.npy",    "<f2", { n, (int64_t) buf.width }, buf.h.data(), buf.h.size() * sizeof(ggml_fp16_t));
        FILE * f = fopen((dir + "/index.jsonl").c_str(), "a");
        if (f) {
            fprintf(f, "{\"tag\": \"%s\", \"n\": %lld, \"pos0\": %d, \"width\": %d, \"task\": %d, \"slot\": %d, \"ok\": %s, \"t\": %lld}\n",
                    buf.tag.c_str(), (long long) n, buf.pos0, buf.width, id_task, id_slot, ok ? "true" : "false", (long long) time(nullptr));
            fclose(f);
        }
    }
    buf.clear();
    return ok;
}
