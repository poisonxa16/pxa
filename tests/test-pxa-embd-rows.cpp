// test-pxa-embd-rows: the token-embedding GET_ROWS on a K-quant table gives the CPU's dequantize_row bits (bug #376).
//
// The input embedding of every arch is ggml_get_rows(tok_embd, inp_tokens) -> F32 (llm_build_inp_embd). tok_embd sits in
// the input context, which is always a HOST buffer (llama.cpp: buft_input = CPU; with mmap off and PXA_HOST_RESIDENT on
// it is a CUDA_Host copy, with mmap on a CPU buffer over the file mapping). The scheduler places a node on its weight's
// backend, the CUDA backend does not claim a host buffer type (supports_buft) and never offloads a GET_ROWS (offload_op),
// so the gather runs on the CPU backend on every card kind. Bug #376 suspected a P100-only q5_K embedding path behind a
// KLD offset seen only on files with a q5_K token_embd; this harness pins down what that path actually computes:
//
//   (1) every gathered row is BYTE-equal to type_traits[type].to_float of the same source row (the CPU reference), for
//       every row id of the table (0 .. n_vocab-1, ubatches of 256 = the KLD run's -ub), plus ubatches of 1, 7 and 512
//       with duplicates and the highest ids;
//   (2) the table is placed the three ways the loader can place it: a CPU buffer over the mapped bytes (mmap), a plain
//       CPU allocation (--no-mmap), and a CUDA_Host copy (PXA_HOST_RESIDENT) when a CUDA device is present;
//   (3) the scheduler's choice of backend for the GET_ROWS node is printed and must be the CPU for a host-resident table;
//   (4) the reference itself is checked against an independent decoder of the block layout written here (fused and
//       unfused multiply-subtract both computed, so a compiler contraction difference is named rather than hidden).
//
// Without arguments it runs on synthetic Q5_K and Q6_K tables (random valid blocks, 5120 x 4099) so ctest exercises it
// on any box. With arguments it runs on a raw table cut from a model file:
//   test-pxa-embd-rows <table.bin> <q5_K|q6_K> <ne0> <ne1>
// (the bytes of token_embd.weight exactly as stored in the GGUF; see the bug #376 notes for the dd line).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define PXA_HAVE_MMAP 1
#endif

static float h2f(uint16_t h) {
    // IEEE half -> float, exact (independent of ggml's tables)
    const uint32_t s = (uint32_t)(h & 0x8000) << 16;
    const uint32_t e = (h >> 10) & 0x1f;
    const uint32_t m = h & 0x3ff;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) { bits = s; }
        else { float f = std::ldexp((float) m, -24); std::memcpy(&bits, &f, 4); bits |= s; }
    } else if (e == 31) {
        bits = s | 0x7f800000u | (m << 13);
    } else {
        bits = s | ((e + 112) << 23) | (m << 13);
    }
    float f; std::memcpy(&f, &bits, 4); return f;
}

// independent decoders of the GGUF block layouts (QK_K = 256)
static void scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
}

// Q5_K block: d(f16) dmin(f16) scales[12] qh[32] qs[128] = 176 bytes
static void ref_q5_K(const uint8_t * blk, float * y, bool fused) {
    uint16_t hd, hm; std::memcpy(&hd, blk, 2); std::memcpy(&hm, blk + 2, 2);
    const float d = h2f(hd), mn = h2f(hm);
    const uint8_t * sc = blk + 4; const uint8_t * qh = blk + 16; const uint8_t * ql = blk + 48;
    int is = 0; uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        scale_min_k4(is + 0, sc, s, m); const float d1 = d * s; const float m1 = mn * m;
        scale_min_k4(is + 1, sc, s, m); const float d2 = d * s; const float m2 = mn * m;
        for (int l = 0; l < 32; ++l) {
            const float q = (float)((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0));
            if (fused) { *y++ = std::fma(d1, q, -m1); } else { volatile float p = d1 * q; *y++ = p - m1; }
        }
        for (int l = 0; l < 32; ++l) {
            const float q = (float)((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0));
            if (fused) { *y++ = std::fma(d2, q, -m2); } else { volatile float p = d2 * q; *y++ = p - m2; }
        }
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}

// Q6_K block: ql[128] qh[64] scales[16](int8) d(f16) = 210 bytes; y = d * sc * q (no subtract: no contraction question)
static void ref_q6_K(const uint8_t * blk, float * y, bool /*fused*/) {
    const uint8_t * ql = blk; const uint8_t * qh = blk + 128; const int8_t * sc = (const int8_t *)(blk + 192);
    uint16_t hd; std::memcpy(&hd, blk + 208, 2); const float d = h2f(hd);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l +  0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}

struct table {
    ggml_type type;
    int64_t   ne0, ne1;
    size_t    row_bytes;
    const uint8_t * data = nullptr;   // the stored bytes (mapped or owned)
    std::vector<uint8_t> owned_v;            // over-allocated; `owned` points at a 64-byte aligned start inside it
    uint8_t * owned = nullptr;
    size_t    map_len = 0;
    void    * map_ptr = nullptr;
};

static uint16_t rand_half(std::mt19937 & rng, float lo_exp, float hi_exp) {
    std::uniform_real_distribution<float> u(lo_exp, hi_exp);
    const float v = std::ldexp(1.0f + (rng() % 1024) / 1024.0f, (int) u(rng));
    return ggml_fp32_to_fp16(v);
}

static void synth_table(table & t, ggml_type type, int64_t ne0, int64_t ne1, uint32_t seed) {
    t.type = type; t.ne0 = ne0; t.ne1 = ne1; t.row_bytes = ggml_row_size(type, ne0);
    const size_t nbytes = t.row_bytes * ne1;
    t.owned_v.resize(nbytes + 64);
    t.owned = t.owned_v.data() + (64 - ((uintptr_t) t.owned_v.data() & 63)) % 64;
    std::mt19937 rng(seed);
    for (size_t i = 0; i < nbytes; ++i) t.owned[i] = (uint8_t) rng();
    const size_t bs = ggml_type_size(type);
    for (size_t off = 0; off < nbytes; off += bs) {   // valid, finite super-block scales
        uint8_t * blk = t.owned + off;
        if (type == GGML_TYPE_Q5_K) {
            const uint16_t d = rand_half(rng, -14, -6), m = rand_half(rng, -14, -6);
            std::memcpy(blk, &d, 2); std::memcpy(blk + 2, &m, 2);
        } else {
            const uint16_t d = rand_half(rng, -16, -8);
            std::memcpy(blk + 208, &d, 2);
        }
    }
    t.data = t.owned;
}

static bool load_table(table & t, const char * path, ggml_type type, int64_t ne0, int64_t ne1) {
    t.type = type; t.ne0 = ne0; t.ne1 = ne1; t.row_bytes = ggml_row_size(type, ne0);
    const size_t need = t.row_bytes * ne1;
#ifdef PXA_HAVE_MMAP
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return false; }
    struct stat st; fstat(fd, &st);
    if ((size_t) st.st_size != need) { fprintf(stderr, "%s: %zu bytes, expected %zu\n", path, (size_t) st.st_size, need); close(fd); return false; }
    void * p = mmap(nullptr, need, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap"); return false; }
    t.map_ptr = p; t.map_len = need; t.data = (const uint8_t *) p;
    return true;
#else
    FILE * f = fopen(path, "rb"); if (!f) return false;
    t.owned_v.resize(need + 64); t.owned = t.owned_v.data() + (64 - ((uintptr_t) t.owned_v.data() & 63)) % 64;
    const bool ok = fread(t.owned, 1, need, f) == need; fclose(f);
    t.data = t.owned; return ok;
#endif
}

enum placement { PLACE_MAPPED, PLACE_CPU_COPY, PLACE_CUDA_HOST };
static const char * place_name(placement p) {
    return p == PLACE_MAPPED ? "cpu-buffer-over-mapping (mmap)" : p == PLACE_CPU_COPY ? "cpu-copy (--no-mmap)" : "cuda-host copy (PXA_HOST_RESIDENT)";
}

struct run_stats { int64_t rows = 0, bad_rows = 0, first_bad = -1; std::string backend; };

// gather `ids` (one ubatch per call, as llm_build_inp_embd builds it) through a scheduler holding `backends`
static bool gather(ggml_backend_sched_t sched, ggml_tensor * tab_proto, const std::vector<int32_t> & ids,
                   std::vector<float> & out, std::string & where) {
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) ids.size());
    ggml_set_name(inp, "inp_tokens"); ggml_set_input(inp);
    ggml_tensor * rows = ggml_get_rows(ctx, tab_proto, inp);
    ggml_set_name(rows, "inp_embd"); ggml_set_output(rows);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, rows);
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(inp, ids.data(), 0, ids.size() * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) { ggml_free(ctx); return false; }
    ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, rows);
    where = b ? ggml_backend_name(b) : "?";
    out.resize(ids.size() * tab_proto->ne[0]);
    ggml_backend_tensor_get(rows, out.data(), 0, out.size() * sizeof(float));
    ggml_free(ctx);
    return true;
}

static int check_table(const char * label, const table & t, std::vector<ggml_backend_t> & backends, bool has_cuda) {
    int fails = 0;
    const ggml_type_traits_t tr = ggml_internal_get_type_traits(t.type);
    if (!tr.to_float) { fprintf(stderr, "%s: no to_float\n", label); return 1; }
    const int64_t ne0 = t.ne0;

    // (4) the CPU reference against the independent decoder, on a spread of rows incl. the last
    {
        std::vector<float> a(ne0), r0(ne0), r1(ne0);
        int64_t n_eq_unfused = 0, n_eq_fused = 0, n_rows = 0;
        const int64_t nb = ne0 / 256;
        for (int64_t r = 0; r < t.ne1; r += (t.ne1 > 4096 ? 97 : 1)) {
            for (int pass = 0; pass < 2; ++pass) {
                const int64_t row = pass ? t.ne1 - 1 - r : r;
                const uint8_t * src = t.data + row * t.row_bytes;
                tr.to_float(src, a.data(), ne0);
                for (int64_t b = 0; b < nb; ++b) {
                    const uint8_t * blk = src + b * ggml_type_size(t.type);
                    if (t.type == GGML_TYPE_Q5_K) { ref_q5_K(blk, r0.data() + b*256, false); ref_q5_K(blk, r1.data() + b*256, true); }
                    else                          { ref_q6_K(blk, r0.data() + b*256, false); ref_q6_K(blk, r1.data() + b*256, true); }
                }
                if (n_rows == 0 && std::memcmp(a.data(), r0.data(), ne0 * 4) != 0) {
                    for (int64_t i = 0; i < ne0; ++i) if (a[i] != r0[i]) {
                        printf("%s: first decoder difference row %lld elem %lld: to_float %.9g decoder %.9g / fused %.9g\n", label,
                               (long long) row, (long long) i, a[i], r0[i], r1[i]); break; }
                }
                n_eq_unfused += std::memcmp(a.data(), r0.data(), ne0 * 4) == 0;
                n_eq_fused   += std::memcmp(a.data(), r1.data(), ne0 * 4) == 0;
                ++n_rows;
            }
        }
        const bool ok = n_eq_unfused == n_rows || n_eq_fused == n_rows;
        printf("%s: to_float vs independent decoder: %lld/%lld rows equal unfused, %lld/%lld fused -> %s\n", label,
               (long long) n_eq_unfused, (long long) n_rows, (long long) n_eq_fused, (long long) n_rows, ok ? "OK" : "FAIL");
        fails += !ok;
    }

    std::vector<placement> places = { PLACE_MAPPED, PLACE_CPU_COPY };
    if (has_cuda) places.push_back(PLACE_CUDA_HOST);

    for (placement pl : places) {
        // the table as the loader places it
        ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, true };
        ggml_context * wctx = ggml_init(ip);
        ggml_tensor * tab = ggml_new_tensor_2d(wctx, t.type, ne0, t.ne1);
        ggml_set_name(tab, "token_embd.weight");
        ggml_backend_buffer_t wbuf = nullptr;
        if (pl == PLACE_MAPPED) {
            wbuf = ggml_backend_cpu_buffer_from_ptr((void *) t.data, t.row_bytes * t.ne1);
            ggml_backend_tensor_alloc(wbuf, tab, (void *) t.data);
        } else {
            ggml_backend_buffer_type_t bt = ggml_backend_cpu_buffer_type();
#ifdef GGML_USE_CUDA
            if (pl == PLACE_CUDA_HOST) bt = ggml_backend_cuda_host_buffer_type();
#endif
            wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, bt);
            if (!wbuf) { fprintf(stderr, "%s: alloc %s failed\n", label, place_name(pl)); ++fails; ggml_free(wctx); continue; }
            ggml_backend_tensor_set(tab, t.data, 0, t.row_bytes * t.ne1);
        }
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), 1024, false);

        // supports_op of every non-CPU backend for this exact node (informational: a host table never reaches it)
        std::string sup;
        {
            ggml_init_params jp = { ggml_tensor_overhead() * 4, nullptr, true };
            ggml_context * c = ggml_init(jp);
            ggml_tensor * ii = ggml_new_tensor_1d(c, GGML_TYPE_I32, 256);
            ggml_tensor * gr = ggml_get_rows(c, tab, ii);
            for (size_t b = 0; b + 1 < backends.size(); ++b) {
                sup += std::string(" ") + ggml_backend_name(backends[b]) + ".supports_op=" + (ggml_backend_supports_op(backends[b], gr) ? "1" : "0");
                sup += std::string(" supports_buft=") + (ggml_backend_supports_buft(backends[b], wbuf ? ggml_backend_buffer_get_type(wbuf) : nullptr) ? "1" : "0");
            }
            ggml_free(c);
        }

        run_stats st;
        std::vector<float> out, ref(ne0);
        auto run_ubatch = [&](const std::vector<int32_t> & ids) -> bool {
            std::string where;
            if (!gather(sched, tab, ids, out, where)) return false;
            if (st.backend.empty()) st.backend = where;
            else if (st.backend.find(where) == std::string::npos) st.backend += "," + where;
            for (size_t k = 0; k < ids.size(); ++k) {
                tr.to_float(t.data + (int64_t) ids[k] * t.row_bytes, ref.data(), ne0);
                ++st.rows;
                if (std::memcmp(ref.data(), out.data() + k * ne0, ne0 * sizeof(float)) != 0) {
                    if (st.first_bad < 0) st.first_bad = ids[k];
                    ++st.bad_rows;
                }
            }
            return true;
        };

        bool ok = true;
        // every row id, ubatches of 256 (the KLD run's -ub 256)
        for (int64_t r0 = 0; r0 < t.ne1 && ok; r0 += 256) {
            std::vector<int32_t> ids;
            for (int64_t r = r0; r < std::min<int64_t>(r0 + 256, t.ne1); ++r) ids.push_back((int32_t) r);
            ok = run_ubatch(ids);
        }
        // scattered ubatches with duplicates and the highest ids, at the other widths the engine uses
        std::mt19937 rng(1234);
        for (int nt : { 1, 7, 512, 256 }) {
            for (int rep = 0; rep < 8 && ok; ++rep) {
                std::vector<int32_t> ids(nt);
                for (int k = 0; k < nt; ++k) {
                    const uint32_t x = rng();
                    ids[k] = (x % 4 == 0) ? (int32_t)(t.ne1 - 1 - (x >> 8) % 64) : (int32_t)((x >> 2) % t.ne1);
                }
                if (nt > 2) { ids[nt - 1] = ids[0]; ids[1] = (int32_t)(t.ne1 - 1); }
                ok = run_ubatch(ids);
            }
        }
        const bool on_cpu = st.backend == "CPU";
        const bool pass = ok && st.bad_rows == 0 && on_cpu;
        printf("%s [%s]: GET_ROWS on %s;%s | rows %lld, bitwise-unequal %lld%s -> %s\n", label, place_name(pl), st.backend.c_str(),
               sup.c_str(), (long long) st.rows, (long long) st.bad_rows,
               st.first_bad >= 0 ? (" (first id " + std::to_string(st.first_bad) + ")").c_str() : "", pass ? "OK" : "FAIL");
        fails += !pass;
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
    }
    return fails;
}

static ggml_type parse_type(const char * s) {
    if (!strcmp(s, "q5_K") || !strcmp(s, "Q5_K")) return GGML_TYPE_Q5_K;
    if (!strcmp(s, "q6_K") || !strcmp(s, "Q6_K")) return GGML_TYPE_Q6_K;
    return GGML_TYPE_COUNT;
}

int main(int argc, char ** argv) {
    // ggml_init fills the fp16 -> fp32 table the row dequantizers read (GGML_FP16_TO_FP32); without a live context
    // to_float decodes every scale as 0. Keep one context for the whole run.
    ggml_init_params ip0 = { 1024, nullptr, true };
    ggml_context * ctx0 = ggml_init(ip0);
    std::vector<ggml_backend_t> backends;
    bool has_cuda = false;
#ifdef GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() > 0) {
        ggml_backend_t c = ggml_backend_cuda_init(0, nullptr);
        if (c) { backends.push_back(c); has_cuda = true; }
    }
#endif
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
    backends.push_back(cpu);   // the scheduler's last backend is the CPU
    printf("backends:"); for (auto * b : backends) printf(" %s", ggml_backend_name(b)); printf("\n");

    int fails = 0;
    if (argc >= 5) {
        table t;
        const ggml_type ty = parse_type(argv[2]);
        if (ty == GGML_TYPE_COUNT || !load_table(t, argv[1], ty, atoll(argv[3]), atoll(argv[4]))) {
            fprintf(stderr, "usage: %s <table.bin> <q5_K|q6_K> <ne0> <ne1>\n", argv[0]); return 2;
        }
        fails += check_table(argv[1], t, backends, has_cuda);
#ifdef PXA_HAVE_MMAP
        munmap(t.map_ptr, t.map_len);
#endif
    } else {
        for (ggml_type ty : { GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) {
            table t; synth_table(t, ty, 5120, 4099, ty == GGML_TYPE_Q5_K ? 5u : 6u);
            fails += check_table(ty == GGML_TYPE_Q5_K ? "synthetic q5_K 5120x4099" : "synthetic q6_K 5120x4099", t, backends, has_cuda);
        }
    }
    for (auto * b : backends) ggml_backend_free(b);
    ggml_free(ctx0);
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
