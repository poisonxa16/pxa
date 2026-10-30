// PXA_PLE_DIRECT: the O_DIRECT row reader (src/llama-pxa-ple-direct.h) against the bytes a file
// mapping reads - the RAM path's source of truth.
//
//   test-pxa-ple-direct
//       self-test, no model: synthetic tables in PXA_PLE_DIRECT_TEST_DIR, else the first of $TMPDIR,
//       /var/tmp, /tmp whose filesystem takes O_DIRECT, else the working directory (row sizes 170 = Q8_0 x160,
//       110 = Q5_0 x160, 128 = F32 x32; a data offset that makes rows straddle 4 KiB blocks), every
//       cache size (none / tiny = constant eviction / large), ubatches of 1..512 tokens x 48 rows with
//       repeats, Linux AIO and the thread pool at depths 1/4/16. Exit 0 = every gathered row
//       byte-identical to the mapping. SKIPs (exit 0, says so)
//       when the chosen directory's filesystem refuses O_DIRECT (tmpfs).
//
//   test-pxa-ple-direct --gguf FILE [--tensor per_layer_token_embd.weight] [--tokens 10000] [--rows 10000]
//       the same comparison on a real table: --rows uniform random row ids, then --tokens random
//       token ids hashed into rows exactly as llama_set_inputs does (the file's <arch>.ple.* keys)
//       in ubatches of 1..512 tokens, then one ggml GET_ROWS on the table vs on the staging tensor
//       (f32 outputs compared bitwise).
//
//   test-pxa-ple-direct --gguf FILE --bench TOKENS [--rows-per-token 16] [--ubatch 1] [--depth 32]
//                       [--io aio|threads] [--cache-mib 0]
//   test-pxa-ple-direct --raw FILE_OR_BLOCKDEV --bench TOKENS [...]   (rows of --row-bytes 170 from offset 0)
//       latency of uniform random rows (the worst case: no repeats, cache off by default): per-call
//       p50/p90/p99/max, per-row mean, rows/s.

#ifdef _WIN32
// The reader is Linux O_DIRECT + a file mapping (sys/mman.h). On Windows the engine leaves this
// path off; the test reports SKIP so the suite still builds and exits 0.
#include <cstdio>
int main(int, char **) {
    printf("SKIP test-pxa-ple-direct: O_DIRECT row reader is Linux-only\n");
    return 0;
}
#else
#include "llama-pxa-ple-direct.h"
#include "ggml.h"

#include <cinttypes>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using pxa_ple_direct::reader;

namespace {

struct table_src {
    std::string path;
    uint64_t    off = 0;
    size_t      rb = 0;
    int64_t     n_rows = 0;
    int64_t     ne0 = 0;
    ggml_type   type = GGML_TYPE_COUNT;
    // the reference: a read-only mapping of the whole file
    const uint8_t * map = nullptr;
    size_t          map_len = 0;
    // <arch>.ple.* hash constants (gguf mode only)
    int n_gram = 0, per_gram = 0, n_heads = 0;
    std::vector<uint64_t> mult, offs, vocab;
    int64_t eos = 0, n_vocab = 0;
};

bool map_file(table_src & t) {
    const int fd = open(t.path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    fstat(fd, &st);
    void * p = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return false;
    t.map = (const uint8_t *) p;
    t.map_len = (size_t) st.st_size;
    madvise((void *) t.map, t.map_len, MADV_RANDOM);
    return true;
}

void unmap_file(table_src & t) {
    if (t.map) munmap((void *) t.map, t.map_len);
    t.map = nullptr;
}

const uint8_t * ref_row(const table_src & t, int32_t r) { return t.map + t.off + (uint64_t) r * t.rb; }

int g_fail = 0;

// one gather, verified row by row against the mapping
bool check_gather(reader & rd, const table_src & t, const std::vector<int32_t> & ids, const char * what) {
    std::vector<uint8_t> stage(ids.size() * t.rb + 1);
    std::vector<int32_t> loc(ids.size(), -1);
    const size_t nu = rd.gather(ids.data(), ids.size(), stage.data(), loc.data());
    if (nu > ids.size()) {
        fprintf(stderr, "FAIL %s: n_unique %zu > n %zu\n", what, nu, ids.size());
        ++g_fail;
        return false;
    }
    for (size_t i = 0; i < ids.size(); ++i) {
        if (loc[i] < 0 || (size_t) loc[i] >= nu) {
            fprintf(stderr, "FAIL %s: local[%zu] = %d outside [0,%zu)\n", what, i, loc[i], nu);
            ++g_fail;
            return false;
        }
        if (memcmp(stage.data() + (size_t) loc[i] * t.rb, ref_row(t, ids[i]), t.rb) != 0) {
            fprintf(stderr, "FAIL %s: row %d (position %zu) differs from the mapping\n", what, ids[i], i);
            ++g_fail;
            return false;
        }
    }
    return true;
}

// row(head h, position p) exactly as llama_set_inputs computes it (EOS cuts the window)
void hash_rows(const table_src & t, const std::vector<int64_t> & toks, size_t p0, size_t n_tok, std::vector<int32_t> & out) {
    out.resize(n_tok * t.n_heads);
    std::vector<int64_t> ctx(t.n_gram);
    for (size_t i = 0; i < n_tok; ++i) {
        const size_t p = p0 + i;
        ctx[0] = toks[p];
        bool cut = false;
        for (int s = 1; s < t.n_gram; ++s) {
            ctx[s] = cut ? t.eos : (p >= (size_t) s ? toks[p - s] : t.eos);
            if (ctx[s] == t.eos) cut = true;
        }
        for (int n = 2; n <= t.n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * t.mult[0];
            for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * t.mult[j];
            const int base = (n - 2) * t.per_gram;
            for (int g = 0; g < t.per_gram; ++g) {
                const int hi = base + g;
                out[i * t.n_heads + hi] = (int32_t) (mixed % t.vocab[hi] + t.offs[hi]);
            }
        }
    }
}

// a token stream with repeats: 60% of positions copy an earlier phrase (hot n-grams), the rest random
std::vector<int64_t> make_tokens(size_t n, int64_t n_vocab, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> tok(3, n_vocab - 1);
    std::vector<int64_t> v;
    v.reserve(n);
    while (v.size() < n) {
        if (v.size() > 64 && rng() % 10 < 6) {
            const size_t len = 4 + rng() % 12;
            const size_t from = rng() % (v.size() - len);
            for (size_t k = 0; k < len && v.size() < n; ++k) v.push_back(v[from + k]);
        } else {
            v.push_back(tok(rng));
        }
    }
    return v;
}

void print_stats(const char * tag, const reader & rd) {
    const auto s = rd.get_stats();
    printf("  [%s] calls %" PRIu64 " rows %" PRIu64 " unique %" PRIu64 " cache-hits %" PRIu64 " read %" PRIu64
           " extents %" PRIu64 " (%.1f MiB) fallback %" PRIu64 " | %.3f ms/call mean, max %.3f ms\n",
           tag, s.calls, s.rows_req, s.rows_unique, s.cache_hits, s.rows_read, s.extents, s.bytes_read / 1048576.0,
           s.fallback_rows, s.calls ? s.ns_total / 1e6 / s.calls : 0.0, s.ns_max_call / 1e6);
}

// ---- self-test on synthetic tables ------------------------------------------------------------

// true when an aligned O_DIRECT pread works on a file in `dir` (the same proof the reader asks for at open)
bool dir_takes_odirect(const std::string & dir) {
#if defined(__linux__)
    const std::string probe = dir + "/pxa-ple-direct-probe-" + std::to_string((long) getpid()) + ".bin";
    {
        std::vector<uint8_t> blk(8192, 0x5a);
        FILE * f = fopen(probe.c_str(), "wb");
        if (!f) {
            return false;
        }
        const bool wrote = fwrite(blk.data(), 1, blk.size(), f) == blk.size();
        fclose(f);
        if (!wrote) {
            remove(probe.c_str());
            return false;
        }
    }
    bool ok = false;
    const int fd = ::open(probe.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd >= 0) {
        void * buf = nullptr;
        if (posix_memalign(&buf, 4096, 4096) == 0) {
            ok = ::pread(fd, buf, 4096, 4096) == 4096 && ((const uint8_t *) buf)[0] == 0x5a;
            free(buf);
        }
        ::close(fd);
    }
    remove(probe.c_str());
    return ok;
#else
    (void) dir;
    return false;
#endif
}

// Where the synthetic tables go: PXA_PLE_DIRECT_TEST_DIR when set, else the first of $TMPDIR, /var/tmp, /tmp whose
// filesystem takes O_DIRECT, else the working directory (the build tree). What the test proves does not depend on
// the device, but its run time does: the ~50 MB of tables are read back with O_DIRECT in 54 configurations, 6.7 s
// on NVMe and 603 s to over 1500 s (the ctest timeout) in a build tree on a busy array disk (2026-10-18).
std::string self_test_dir() {
    if (const char * e = getenv("PXA_PLE_DIRECT_TEST_DIR")) {
        return e;
    }
    std::vector<std::string> cands;
    if (const char * t = getenv("TMPDIR"); t && *t) {
        cands.push_back(t);
    }
    cands.push_back("/var/tmp");
    cands.push_back("/tmp");
    for (const auto & d : cands) {
        if (dir_takes_odirect(d)) {
            return d;
        }
    }
    return ".";
}

int self_test() {
    const std::string dir = self_test_dir();
    printf("self-test tables in %s\n", dir.c_str());
    struct shape { size_t rb; int64_t n_rows; uint64_t off; const char * name; };
    const shape shapes[] = {
        { 170, 200000, 3*4096 + 160, "q8_0x160" },   // 34 MB: rows straddle blocks every ~24 rows
        { 110, 150000, 4096 + 32,    "q5_0x160" },
        { 128,   4000, 96,           "f32x32"   },   // the tiny fixture's table geometry
    };
    int n_skip = 0;
    for (const shape & sh : shapes) {
        table_src t;
        t.path = dir + "/pxa-ple-direct-selftest-" + sh.name + ".bin";
        t.off = sh.off; t.rb = sh.rb; t.n_rows = sh.n_rows;
        {
            std::mt19937_64 rng(sh.rb);
            std::vector<uint8_t> buf(t.off + sh.rb * sh.n_rows + 777);   // tail: the last block is short
            for (auto & b : buf) b = (uint8_t) rng();
            FILE * f = fopen(t.path.c_str(), "wb");
            if (!f || fwrite(buf.data(), 1, buf.size(), f) != buf.size()) { fprintf(stderr, "cannot write %s\n", t.path.c_str()); return 1; }
            fclose(f);
        }
        if (!map_file(t)) { fprintf(stderr, "cannot map %s\n", t.path.c_str()); return 1; }

        for (size_t cache : { (size_t) 0, (size_t) 64 << 10, (size_t) 64 << 20 }) {
            for (int nt : { 1, 4, 16, -1, -4, -16 }) {   // > 0: Linux AIO at that depth, < 0: a pool of -nt threads
                reader::config cfg;
                cfg.cache_bytes = cache;
                cfg.depth = nt > 0 ? nt : -nt;
                cfg.io = nt > 0 ? 0 : 1;
                std::string err;
                auto rd = reader::open_path(t.path.c_str(), t.off, t.rb, t.n_rows, cfg, err);
                if (!rd) {
                    printf("SKIP %s: %s (filesystem of %s refuses O_DIRECT?)\n", sh.name, err.c_str(), dir.c_str());
                    ++n_skip;
                    goto next_shape;
                }
                std::mt19937_64 rng(1234 + cache + nt);
                // edges: first row, last row, every row that straddles a block among the first 64
                std::vector<int32_t> ids = { 0, (int32_t) (t.n_rows - 1) };
                for (int32_t r = 0; r < 4096 && (int64_t) r < t.n_rows; ++r) {
                    const uint64_t a = t.off + (uint64_t) r * t.rb;
                    if (a / 4096 != (a + t.rb - 1) / 4096) ids.push_back(r);
                }
                check_gather(*rd, t, ids, "edges");
                // ubatches of 1..512 tokens x 48 rows, drawn from a pool so repeats and cache hits happen
                std::uniform_int_distribution<int64_t> any(0, t.n_rows - 1);
                std::vector<int32_t> pool(5000);
                for (auto & p : pool) p = (int32_t) any(rng);
                for (int ub : { 1, 1, 2, 4, 64, 1, 512, 3, 1 }) {
                    ids.resize((size_t) ub * 48);
                    for (auto & id : ids) id = (rng() % 3 == 0) ? (int32_t) any(rng) : pool[rng() % pool.size()];
                    check_gather(*rd, t, ids, "ubatch");
                }
                // read_rows on 10k uniform ids
                ids.resize(10000);
                for (auto & id : ids) id = (int32_t) any(rng);
                std::vector<uint8_t> out(ids.size() * t.rb);
                rd->read_rows(ids.data(), ids.size(), out.data());
                for (size_t i = 0; i < ids.size(); ++i) {
                    if (memcmp(out.data() + i * t.rb, ref_row(t, ids[i]), t.rb) != 0) {
                        fprintf(stderr, "FAIL read_rows %s row %d\n", sh.name, ids[i]);
                        ++g_fail;
                        break;
                    }
                }
                const auto s = rd->get_stats();
                printf("%s cache %zu KiB %s depth %d: calls %" PRIu64 " unique %" PRIu64 " hits %" PRIu64 " read %" PRIu64
                       " extents %" PRIu64 " fallback %" PRIu64 " -> %s\n",
                       sh.name, cache >> 10, rd->io_mode(), rd->depth(), s.calls, s.rows_unique, s.cache_hits, s.rows_read, s.extents, s.fallback_rows,
                       g_fail ? "FAIL" : "ok");
            }
        }
next_shape:
        unmap_file(t);
        remove(t.path.c_str());
    }
    if (g_fail) { printf("RESULT: FAIL (%d)\n", g_fail); return 1; }
    printf("RESULT: PASS%s\n", n_skip ? " (with SKIPs)" : "");
    return 0;
}

// ---- a real GGUF ------------------------------------------------------------------------------

bool open_gguf(const char * path, const char * tname, table_src & t) {
    ggml_context * meta = nullptr;
    gguf_init_params ip = { /*no_alloc*/ true, &meta };
    gguf_context * g = gguf_init_from_file(path, ip);
    if (!g) { fprintf(stderr, "cannot read GGUF %s\n", path); return false; }
    const int ti = gguf_find_tensor(g, tname);
    if (ti < 0) { fprintf(stderr, "no tensor %s in %s\n", tname, path); gguf_free(g); return false; }
    ggml_tensor * tt = ggml_get_tensor(meta, tname);
    t.path = path;
    t.off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, ti);
    t.type = tt->type;
    t.ne0 = tt->ne[0];
    t.rb = ggml_row_size(tt->type, tt->ne[0]);
    t.n_rows = ggml_nrows(tt);
    // the PLE hash constants
    std::string arch = "qwen4exp";
    const int ka = gguf_find_key(g, "general.architecture");
    if (ka >= 0) arch = gguf_get_val_str(g, ka);
    auto key = [&](const char * k) { return gguf_find_key(g, (arch + ".ple." + k).c_str()); };
    auto u32 = [&](int k) -> int64_t {
        return gguf_get_kv_type(g, k) == GGUF_TYPE_UINT32 ? gguf_get_val_u32(g, k) : (int64_t) gguf_get_val_u64(g, k);
    };
    auto arr64 = [&](int k, std::vector<uint64_t> & v) {
        const int n = gguf_get_arr_n(g, k);
        v.resize(n);
        if (gguf_get_arr_type(g, k) == GGUF_TYPE_UINT64) {
            memcpy(v.data(), gguf_get_arr_data(g, k), n * sizeof(uint64_t));
        } else {
            const uint32_t * d = (const uint32_t *) gguf_get_arr_data(g, k);
            for (int i = 0; i < n; ++i) v[i] = d[i];
        }
    };
    const int kn = key("ngram_size"), kh = key("heads_per_ngram"), km = key("layer_multipliers"),
              ko = key("head_offsets"), kv = key("head_vocab_sizes"), ke = key("eos_token_id");
    if (kn >= 0 && kh >= 0 && km >= 0 && ko >= 0 && kv >= 0) {
        t.n_gram = (int) u32(kn);
        t.per_gram = (int) u32(kh);
        t.n_heads = (t.n_gram - 1) * t.per_gram;
        arr64(km, t.mult); arr64(ko, t.offs); arr64(kv, t.vocab);
        t.eos = ke >= 0 ? u32(ke) : 0;
    }
    const int kt = gguf_find_key(g, "tokenizer.ggml.tokens");
    t.n_vocab = kt >= 0 ? gguf_get_arr_n(g, kt) : 248320;
    printf("table %s: %s, %lld rows x %zu B (%s x %lld), file offset %" PRIu64 " (%.2f GiB)%s\n", tname, path,
           (long long) t.n_rows, t.rb, ggml_type_name(t.type), (long long) t.ne0, t.off,
           (double) t.rb * t.n_rows / (1 << 30), t.n_heads ? "" : " - no PLE hash keys");
    if (t.n_heads) printf("  ple: n_gram %d, heads %d (%d per n), eos %lld, vocab %lld\n", t.n_gram, t.n_heads, t.per_gram,
                          (long long) t.eos, (long long) t.n_vocab);
    gguf_free(g);
    ggml_free(meta);
    return true;
}

// one GET_ROWS on the table (the RAM path) and one on the staging tensor (the direct path)
bool check_get_rows_f32(reader & rd, const table_src & t, const std::vector<int32_t> & ids) {
    const size_t n = ids.size();
    std::vector<uint8_t> stage(n * t.rb);
    std::vector<int32_t> loc(n);
    const size_t nu = rd.gather(ids.data(), n, stage.data(), loc.data());

    ggml_init_params mp = { ggml_tensor_overhead() * 8, nullptr, /*no_alloc*/ true };
    ggml_context * cm = ggml_init(mp);
    ggml_tensor * table = ggml_new_tensor_2d(cm, t.type, t.ne0, t.n_rows);
    table->data = (void *) (t.map + t.off);
    ggml_tensor * st = ggml_new_tensor_2d(cm, t.type, t.ne0, (int64_t) nu);
    st->data = stage.data();

    ggml_init_params cp = { (size_t) 64 << 20, nullptr, false };
    ggml_context * cc = ggml_init(cp);
    ggml_tensor * i_ram = ggml_new_tensor_1d(cc, GGML_TYPE_I32, (int64_t) n);
    ggml_tensor * i_dir = ggml_new_tensor_1d(cc, GGML_TYPE_I32, (int64_t) n);
    memcpy(i_ram->data, ids.data(), n * sizeof(int32_t));
    memcpy(i_dir->data, loc.data(), n * sizeof(int32_t));
    ggml_tensor * a = ggml_get_rows(cc, table, i_ram);
    ggml_tensor * b = ggml_get_rows(cc, st, i_dir);
    ggml_cgraph * gf = ggml_new_graph(cc);
    ggml_build_forward_expand(gf, a);
    ggml_build_forward_expand(gf, b);
    ggml_graph_compute_with_ctx(cc, gf, 4);
    const bool same = a->type == GGML_TYPE_F32 && ggml_nbytes(a) == ggml_nbytes(b) &&
                      memcmp(a->data, b->data, ggml_nbytes(a)) == 0;
    printf("GET_ROWS f32: table vs staging, %zu ids (%zu unique), %zu bytes -> %s\n", n, nu, ggml_nbytes(a),
           same ? "bit-identical" : "DIFFERENT");
    if (!same) ++g_fail;
    ggml_free(cc);
    ggml_free(cm);
    return same;
}

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const size_t i = std::min(v.size() - 1, (size_t) (p * (v.size() - 1) + 0.5));
    return v[i];
}

int run_gguf(const char * path, const char * tname, int64_t n_rows_check, int64_t n_tokens, int64_t bench,
             int rows_per_token, int ubatch, int depth, int io, size_t cache_mib, uint64_t seed, size_t raw_rb) {
    table_src t;
    if (raw_rb > 0) {
        // --raw: any file or block device as a table of raw_rb-byte rows from offset 0 (bench only)
        const int fd = open(path, O_RDONLY);
        const off_t sz = fd >= 0 ? lseek(fd, 0, SEEK_END) : -1;
        if (fd >= 0) close(fd);
        if (sz <= 0 || bench <= 0) { fprintf(stderr, "--raw needs a readable path and --bench\n"); return 2; }
        t.path = path; t.off = 0; t.rb = raw_rb; t.n_rows = std::min<int64_t>((int64_t) ((uint64_t) sz / raw_rb) - 1, INT32_MAX);   // row ids are int32
        printf("raw %s: %lld rows x %zu B\n", path, (long long) t.n_rows, t.rb);
    } else if (!open_gguf(path, tname, t)) {
        return 2;
    }
    {
        std::string desc;
        const int fd = open(path, O_RDONLY);
        const int rot = pxa_ple_direct::fd_rotational(fd, desc);
        close(fd);
        printf("device: %s, rotational=%d\n", desc.c_str(), rot);
    }
    reader::config cfg;
    cfg.depth = depth;
    cfg.io = io;
    cfg.cache_bytes = cache_mib << 20;
    std::string err;
    auto rd = reader::open_path(path, t.off, t.rb, t.n_rows, cfg, err);
    if (!rd) { fprintf(stderr, "reader: %s\n", err.c_str()); return 2; }
    printf("reader: %s depth %d, cache %zu MiB (%zu rows)\n", rd->io_mode(), rd->depth(), cache_mib, rd->cache_rows());

    if (bench > 0) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int64_t> any(0, t.n_rows - 1);
        const size_t per_call = (size_t) rows_per_token * ubatch;
        std::vector<int32_t> ids(per_call), loc(per_call);
        std::vector<uint8_t> stage(per_call * t.rb);
        std::vector<double> ms;
        const int64_t calls = std::max<int64_t>(1, bench / ubatch);
        const auto t0 = std::chrono::steady_clock::now();
        for (int64_t c = 0; c < calls; ++c) {
            for (auto & id : ids) id = (int32_t) any(rng);
            const auto a = std::chrono::steady_clock::now();
            rd->gather(ids.data(), ids.size(), stage.data(), loc.data());
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const auto s = rd->get_stats();
        double mean = 0;
        for (double x : ms) mean += x;
        mean /= ms.size();
        printf("BENCH %s rows/token %d ubatch %d %s depth %d cache %zu MiB: %" PRId64 " calls x %zu rows, uniform random rows\n",
               path, rows_per_token, ubatch, rd->io_mode(), rd->depth(), cache_mib, calls, per_call);
        printf("  per call ms: mean %.3f p50 %.3f p90 %.3f p99 %.3f max %.3f | per row %.2f us (mean/rows) | %.0f rows/s"
               " | read %" PRIu64 " rows in %" PRIu64 " extents, %.1f MiB, hits %" PRIu64 ", fallback %" PRIu64 "\n",
               mean, pct(ms, 0.5), pct(ms, 0.9), pct(ms, 0.99), pct(ms, 1.0), mean * 1000.0 / per_call,
               s.rows_req / wall, s.rows_read, s.extents, s.bytes_read / 1048576.0, s.cache_hits, s.fallback_rows);
        return 0;
    }

    if (!map_file(t)) { fprintf(stderr, "cannot map %s\n", path); return 2; }
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> any(0, t.n_rows - 1);

    // (1) uniform random row ids, in 48-row calls (cache cold for these)
    {
        std::vector<int32_t> ids;
        for (int64_t i = 0; i < n_rows_check; i += 48) {
            ids.resize((size_t) std::min<int64_t>(48, n_rows_check - i));
            for (auto & id : ids) id = (int32_t) any(rng);
            if (!check_gather(*rd, t, ids, "uniform")) break;
        }
        printf("uniform: %lld random rows vs the mapping -> %s\n", (long long) n_rows_check, g_fail ? "FAIL" : "identical");
        print_stats("uniform", *rd);
    }
    // (2) random token ids hashed into rows like llama_set_inputs, ubatches of 1..512 tokens
    if (t.n_heads > 0 && n_tokens > 0) {
        rd->reset_stats();
        const auto toks = make_tokens((size_t) n_tokens, t.n_vocab, seed + 1);
        const int ubs[] = { 1, 1, 1, 1, 4, 1, 1, 64, 1, 1, 512, 1, 2, 1 };
        size_t p = 0, k = 0;
        std::vector<int32_t> ids;
        while (p < toks.size()) {
            const size_t ub = std::min<size_t>(ubs[k++ % (sizeof(ubs) / sizeof(ubs[0]))], toks.size() - p);
            hash_rows(t, toks, p, ub, ids);
            if (!check_gather(*rd, t, ids, "tokens")) break;
            p += ub;
        }
        printf("tokens: %lld random token ids (60%% repeated phrases) -> %lld rows vs the mapping -> %s\n",
               (long long) n_tokens, (long long) n_tokens * t.n_heads, g_fail ? "FAIL" : "identical");
        print_stats("tokens", *rd);
        // (3) GET_ROWS through ggml on both sources
        hash_rows(t, toks, 0, std::min<size_t>(512, toks.size()), ids);
        check_get_rows_f32(*rd, t, ids);
    } else if (n_tokens > 0) {
        printf("tokens: skipped (the file has no <arch>.ple.* hash keys)\n");
        std::vector<int32_t> ids(512);
        for (auto & id : ids) id = (int32_t) any(rng);
        check_get_rows_f32(*rd, t, ids);
    }
    unmap_file(t);
    printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

} // namespace

int main(int argc, char ** argv) {
    const char * gguf = nullptr;
    const char * tname = "per_layer_token_embd.weight";
    int64_t n_rows = 10000, n_tokens = 10000, bench = 0;
    int rpt = 16, ub = 1, depth = 32, io = 0;
    size_t cache_mib = 256;
    bool cache_set = false;
    uint64_t seed = 1;
    size_t raw_rb = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> const char * { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if      (a == "--gguf")           gguf = nx();
        else if (a == "--raw")            { gguf = nx(); if (!raw_rb) raw_rb = 170; }
        else if (a == "--row-bytes")      raw_rb = (size_t) atoll(nx());
        else if (a == "--tensor")         tname = nx();
        else if (a == "--rows")           n_rows = atoll(nx());
        else if (a == "--tokens")         n_tokens = atoll(nx());
        else if (a == "--bench")          bench = atoll(nx());
        else if (a == "--rows-per-token") rpt = atoi(nx());
        else if (a == "--ubatch")         ub = atoi(nx());
        else if (a == "--depth")          depth = atoi(nx());
        else if (a == "--io")             io = std::string(nx()) == "threads" ? 1 : 0;
        else if (a == "--cache-mib")      { cache_mib = (size_t) atoll(nx()); cache_set = true; }
        else if (a == "--seed")           seed = (uint64_t) atoll(nx());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (!gguf) return self_test();
    if (bench > 0 && !cache_set) cache_mib = 0;
    return run_gguf(gguf, tname, n_rows, n_tokens, bench, rpt, ub, depth, io, cache_mib, seed, raw_rb);
}
#endif
