#pragma once

// ------------------------------------------------------------------------------------------------
// PXA_PLE_DIRECT (lever, default OFF; 2026-10-13).
//
// The qwen4exp per-layer n-gram table (per_layer_token_embd, 160 x ~320M rows: 50.7 GiB at Q8_0,
// ~33 GiB at Q5_0) is read by exactly one CPU GET_ROWS, ple_n_heads rows per token (16 in Flash-Next:
// n-gram sizes 2 and 3, 8 hash heads each). PXA_PLE_MMAP
// already serves it from a pageable file mapping instead of a pinned copy, but every row a request
// touches then becomes page cache, and on a 48-64 GB host the table's hot pages compete with the
// expert stacks and the rest of the process for the same RAM.
//
// This reader fetches the rows straight from the GGUF file with O_DIRECT preads instead: no page
// cache, no mapping touched. One request per row-containing 4 KiB extent (8 KiB when a row
// straddles a block boundary; adjacent blocks of one batch are merged up to max_extent), `depth`
// reads in flight, and a small RAM row cache sits in front (hot n-grams repeat). The rows of one ubatch are deduplicated first, so the graph gathers from a compact
// staging tensor: stage[local[i]] is the table row ids[i], byte for byte.
//
// The reads are issued with Linux native AIO (io_submit / io_getevents on the O_DIRECT descriptor,
// raw syscalls, no library): one thread keeps `depth` reads in flight and needs no CPU while they
// run, which matters on a host whose cores are busy (a thread pool's workers then queue for a core
// before they can even issue their read). Where io_setup is refused, a pool of `depth` threads doing
// plain preads does the same job.
//
// The cache holds ROWS, not 4 KiB blocks: the rows are hashed n-gram ids, so the 24 other rows of a
// block are almost never wanted next, and a row cache holds ~24x more useful entries per MiB.
// It is 8-way set-associative with LRU inside a set (no hash map to grow, no allocation per row).
//
// Nothing here changes a byte: a row comes from the same file offset the mapping would have read
// it from, so GET_ROWS on the staging tensor produces exactly what GET_ROWS on the table produced.
// A read that fails or comes back short is redone with a plain buffered pread of that row.
//
// Header-only and free of llama/ggml types so tests/test-pxa-ple-direct.cpp can drive it alone.
// Linux only (O_DIRECT, /proc/self/fd, sysfs); elsewhere open() fails and the RAM path stays.
// ------------------------------------------------------------------------------------------------

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include "pxa-win-compat.h"
#endif
#if defined(__linux__)
#include <cerrno>
#include <climits>
#include <fcntl.h>
#include <linux/aio_abi.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#endif

namespace pxa_ple_direct {

static constexpr size_t k_align = 4096;   // O_DIRECT offset / length / buffer alignment

struct stats {
    uint64_t calls         = 0;   // gather() calls
    uint64_t rows_req      = 0;   // row ids asked for (with repeats)
    uint64_t rows_unique   = 0;   // after the per-call dedup
    uint64_t cache_hits    = 0;   // unique rows served from the RAM row cache
    uint64_t rows_read     = 0;   // unique rows read from the file
    uint64_t extents       = 0;   // O_DIRECT preads issued
    uint64_t bytes_read    = 0;   // bytes those preads moved
    uint64_t fallback_rows = 0;   // rows re-read through the buffered fd after a failed/short pread
    uint64_t ns_total      = 0;   // wall time inside gather()
    uint64_t ns_io         = 0;   // wall time of the parallel read phase
    uint64_t ns_max_call   = 0;   // slowest gather()
};

#if defined(__linux__)
// 1 rotational, 0 not, -1 unknown. Reads /sys/dev/block/MAJ:MIN/queue/rotational, or the parent
// disk's queue for a partition.
inline int sysfs_rotational(unsigned maj, unsigned min, std::string & desc) {
    char p[96];
    snprintf(p, sizeof(p), "/sys/dev/block/%u:%u", maj, min);
    char real[PATH_MAX];
    if (realpath(p, real) == nullptr) {
        desc = std::string(p) + " (missing)";
        return -1;
    }
    std::string r(real);
    const size_t sl = r.find_last_of('/');
    desc = (sl == std::string::npos ? r : r.substr(sl + 1)) + " (" + std::to_string(maj) + ":" + std::to_string(min) + ")";
    for (const std::string & q : { r + "/queue/rotational", r + "/../queue/rotational" }) {
        FILE * f = fopen(q.c_str(), "r");
        if (!f) continue;
        int v = -1;
        const int ok = fscanf(f, "%d", &v);
        fclose(f);
        if (ok == 1 && (v == 0 || v == 1)) return v;
    }
    return -1;
}

// The device a file lives on. A filesystem with an anonymous st_dev (btrfs, overlay, zfs, fuse)
// is resolved through /proc/self/mountinfo to its mount source; a source that is not a /dev node
// (fuse, nfs, tmpfs) is "unknown".
inline int fd_rotational(int fd, std::string & desc) {
    struct stat st;
    if (fstat(fd, &st) != 0) { desc = "fstat failed"; return -1; }
    const unsigned maj = major(st.st_dev), min = minor(st.st_dev);
    if (maj != 0) return sysfs_rotational(maj, min, desc);
    FILE * f = fopen("/proc/self/mountinfo", "r");
    if (!f) { desc = "anonymous device, no mountinfo"; return -1; }
    char want[32];
    snprintf(want, sizeof(want), "%u:%u", maj, min);
    // btrfs reports a per-subvolume anonymous st_dev that mountinfo does not list, so fall back to
    // the longest mount point that prefixes the file's own path
    std::string fpath;
    {
        char lk[64], buf[PATH_MAX];
        snprintf(lk, sizeof(lk), "/proc/self/fd/%d", fd);
        const ssize_t n = readlink(lk, buf, sizeof(buf) - 1);
        if (n > 0) { buf[n] = 0; fpath = buf; }
    }
    char line[4096];
    std::string src, fstype;
    size_t best = 0;
    bool exact = false;
    while (fgets(line, sizeof(line), f)) {
        // id parent maj:min root mountpoint opts [optional...] - fstype source superopts
        char id_s[32], par_s[32], mm[32], root[1024], mnt[1024];
        if (sscanf(line, "%31s %31s %31s %1023s %1023s", id_s, par_s, mm, root, mnt) != 5) continue;
        const char * dash = strstr(line, " - ");
        char fs[256], so[1024];
        if (!dash || sscanf(dash + 3, "%255s %1023s", fs, so) != 2) continue;
        if (strcmp(mm, want) == 0) { fstype = fs; src = so; exact = true; break; }
        const size_t ml = strlen(mnt);
        const bool prefix = !fpath.empty() && fpath.compare(0, ml, mnt) == 0 &&
                            (ml == 1 || fpath.size() == ml || fpath[ml] == '/');
        if (prefix && ml > best) { best = ml; fstype = fs; src = so; }
    }
    fclose(f);
    (void) exact;
    if (src.empty()) { desc = std::string("anonymous device ") + want + " not in mountinfo"; return -1; }
    struct stat sd;
    if (src.rfind("/dev/", 0) == 0 && stat(src.c_str(), &sd) == 0 && S_ISBLK(sd.st_mode)) {
        const int r = sysfs_rotational(major(sd.st_rdev), minor(sd.st_rdev), desc);
        desc = fstype + " on " + src + " -> " + desc;
        return r;
    }
    // inside a container the source's /dev node is usually absent, but sysfs still names it
    if (src.rfind("/dev/", 0) == 0) {
        const std::string name = src.substr(src.find_last_of('/') + 1);
        FILE * fd_dev = fopen(("/sys/class/block/" + name + "/dev").c_str(), "r");
        unsigned bmaj = 0, bmin = 0;
        const bool got = fd_dev && fscanf(fd_dev, "%u:%u", &bmaj, &bmin) == 2;
        if (fd_dev) fclose(fd_dev);
        if (got) {
            const int r = sysfs_rotational(bmaj, bmin, desc);
            desc = fstype + " on " + src + " (sysfs) -> " + desc;
            return r;
        }
    }
    desc = fstype + " on " + src + " (not a block device)";
    return -1;
}
#endif

class reader {
public:
    struct config {
        size_t cache_bytes = (size_t) 256 << 20;   // RAM row cache (0 = none)
        int    depth       = 32;                    // reads in flight
        int    io          = 0;                     // 0 = Linux AIO, else a thread pool; 1 = thread pool
        size_t max_extent  = (size_t) 64 << 10;     // largest merged pread
    };

    // `any_fd`: an open descriptor of the model file (the loader's). Two private descriptors are
    // opened through /proc/self/fd, so this one may be closed afterwards. `data_off` is the byte
    // offset of row 0 in the file.
    static std::unique_ptr<reader> open_fd(int any_fd, uint64_t data_off, size_t row_bytes, int64_t n_rows,
                                           const config & cfg, std::string & err) {
#if defined(__linux__)
        char p[64];
        snprintf(p, sizeof(p), "/proc/self/fd/%d", any_fd);
        return open_path(p, data_off, row_bytes, n_rows, cfg, err);
#else
        (void) any_fd; (void) data_off; (void) row_bytes; (void) n_rows; (void) cfg;
        err = "O_DIRECT reader is Linux-only";
        return nullptr;
#endif
    }

    static std::unique_ptr<reader> open_path(const char * path, uint64_t data_off, size_t row_bytes, int64_t n_rows,
                                             const config & cfg, std::string & err) {
#if defined(__linux__)
        if (row_bytes == 0 || n_rows <= 0) { err = "empty table"; return nullptr; }
        if (row_bytes > cfg.max_extent / 2) { err = "row larger than half an extent"; return nullptr; }
        const int fd_dir = ::open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd_dir < 0) {
            err = std::string("open(O_DIRECT) failed: ") + strerror(errno);
            return nullptr;
        }
        const int fd_buf = ::open(path, O_RDONLY | O_CLOEXEC);
        if (fd_buf < 0) {
            err = std::string("open failed: ") + strerror(errno);
            ::close(fd_dir);
            return nullptr;
        }
        const off_t fsize = lseek(fd_buf, 0, SEEK_END);   // also right for a block device
        if (fsize < 0 || (uint64_t) fsize < data_off + (uint64_t) row_bytes * (uint64_t) n_rows) {
            err = "file shorter than the table";
            ::close(fd_dir); ::close(fd_buf);
            return nullptr;
        }
        std::unique_ptr<reader> r(new reader(fd_dir, fd_buf, data_off, row_bytes, n_rows, cfg));
        // one aligned read proves the filesystem really takes O_DIRECT (some accept the flag at open
        // and fail every read with EINVAL)
        void * b = nullptr;
        if (posix_memalign(&b, k_align, k_align) != 0) { err = "posix_memalign failed"; return nullptr; }
        const ssize_t got = pread(fd_dir, b, k_align, (off_t) (data_off & ~(uint64_t) (k_align - 1)));
        const int e = errno;
        free(b);
        if (got <= 0) {
            err = std::string("O_DIRECT pread failed: ") + (got < 0 ? strerror(e) : "EOF");
            return nullptr;
        }
        return r;
#else
        (void) path; (void) data_off; (void) row_bytes; (void) n_rows; (void) cfg;
        err = "O_DIRECT reader is Linux-only";
        return nullptr;
#endif
    }

    ~reader() {
        {
            std::lock_guard<std::mutex> lk(mu_pool_);
            stop_ = true;
        }
        cv_work_.notify_all();
        for (auto & t : workers_) t.join();
#if defined(__linux__)
        if (aio_) syscall(SYS_io_destroy, aio_ctx_);
        if (fd_dir_ >= 0) ::close(fd_dir_);
        if (fd_buf_ >= 0) ::close(fd_buf_);
#endif
        for (void * b : bufs_) free(b);
    }

    size_t  row_bytes() const { return rb_; }
    int64_t n_rows()    const { return n_rows_; }
    size_t  cache_rows() const { return (size_t) n_sets_ * k_ways; }
    int     depth()     const { return aio_ ? (int) iocb_.size() : (int) workers_.size() + 1; }
    const char * io_mode() const { return aio_ ? "aio" : "threads"; }
    int     direct_fd() const { return fd_dir_; }

    stats get_stats() const {
        std::lock_guard<std::mutex> lk(mu_call_);
        stats s = st_;
        s.fallback_rows = fallback_.load();
        return s;
    }
    void reset_stats() { std::lock_guard<std::mutex> lk(mu_call_); st_ = stats(); fallback_.store(0); }

    // Gather ids[0..n) (repeats allowed). Writes the DISTINCT rows, in first-seen order, to
    // stage[0 .. n_unique*row_bytes) and local[i] = the stage row holding ids[i]. Returns n_unique
    // (<= n). Thread-safe: concurrent callers are serialised. Aborts on a row id outside the table
    // (the RAM path would have read out of bounds).
    size_t gather(const int32_t * ids, size_t n, uint8_t * stage, int32_t * local) {
        std::lock_guard<std::mutex> lk(mu_call_);
        const auto t0 = std::chrono::steady_clock::now();

        // 1. dedup (open addressing, power-of-two table >= 2n)
        size_t cap = 64;
        while (cap < 2*n) cap <<= 1;
        if (ht_key_.size() < cap) { ht_key_.resize(cap); ht_val_.resize(cap); }
        std::fill(ht_key_.begin(), ht_key_.begin() + cap, -1);
        const uint32_t mask = (uint32_t) (cap - 1);
        uniq_.clear();
        for (size_t i = 0; i < n; ++i) {
            const int32_t r = ids[i];
            if (r < 0 || (int64_t) r >= n_rows_) {
                fprintf(stderr, "PXA_PLE_DIRECT: row id %d outside the table (0..%lld)\n", r, (long long) n_rows_ - 1);
                abort();
            }
            uint32_t h = ((uint32_t) r * 0x9E3779B1u) & mask;
            while (ht_key_[h] != -1 && ht_key_[h] != r) h = (h + 1) & mask;
            if (ht_key_[h] == -1) {
                ht_key_[h] = r;
                ht_val_[h] = (int32_t) uniq_.size();
                uniq_.push_back(r);
            }
            local[i] = ht_val_[h];
        }
        const size_t nu = uniq_.size();

        // 2. the RAM row cache
        miss_.clear();
        for (size_t u = 0; u < nu; ++u) {
            const int64_t slot = cache_find(uniq_[u]);
            if (slot >= 0) {
                memcpy(stage + u*rb_, cache_data_.get() + (size_t) slot*rb_, rb_);
            } else {
                miss_.push_back({ off_ + (uint64_t) uniq_[u]*rb_, (uint32_t) u });
            }
        }
        st_.cache_hits += nu - miss_.size();

        // 3. the misses, sorted by file offset and merged into aligned extents
        const auto t_io0 = std::chrono::steady_clock::now();
        if (!miss_.empty()) {
            std::sort(miss_.begin(), miss_.end(), [](const miss & a, const miss & b) { return a.off < b.off; });
            ext_.clear();
            const uint64_t A = k_align;
            for (size_t i = 0; i < miss_.size(); ++i) {
                const uint64_t lo = miss_[i].off & ~(A - 1);
                const uint64_t hi = (miss_[i].off + rb_ + A - 1) & ~(A - 1);
                if (!ext_.empty()) {
                    extent & e = ext_.back();
                    const uint64_t nhi = std::max(e.hi, hi);
                    if (lo <= e.hi && nhi - e.lo <= cfg_.max_extent) {
                        e.hi = nhi;
                        e.m1 = (uint32_t) i + 1;
                        continue;
                    }
                }
                ext_.push_back({ lo, hi, (uint32_t) i, (uint32_t) i + 1 });
            }
            run_extents(stage);
            st_.extents += ext_.size();
            for (const extent & e : ext_) st_.bytes_read += e.hi - e.lo;
            // 4. fill the cache from the stage (after the reads: the workers never touch it)
            for (const miss & m : miss_) {
                cache_insert(uniq_[m.u], stage + (size_t) m.u*rb_);
            }
        }
        const auto t1 = std::chrono::steady_clock::now();

        st_.calls       += 1;
        st_.rows_req    += n;
        st_.rows_unique += nu;
        st_.rows_read   += miss_.size();
        const uint64_t ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        st_.ns_total += ns;
        st_.ns_io    += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t_io0).count();
        st_.ns_max_call = std::max(st_.ns_max_call, ns);
        return nu;
    }

    // ids[0..n) -> dst[i*row_bytes ..] in the given order (test helper; same path as gather()).
    void read_rows(const int32_t * ids, size_t n, uint8_t * dst) {
        std::vector<uint8_t> stage(n*rb_);
        std::vector<int32_t> loc(n);
        gather(ids, n, stage.data(), loc.data());
        for (size_t i = 0; i < n; ++i) memcpy(dst + i*rb_, stage.data() + (size_t) loc[i]*rb_, rb_);
    }

    // Uncached random-row read rate (rows/s): `rounds` batches of `n` random rows, one read each,
    // the cache bypassed and left untouched; returns the BEST round, so one scheduling stall on a
    // busy host does not condemn a fast device (the question is what the device can do).
    double probe(size_t n, uint64_t seed, int rounds = 3) {
        std::lock_guard<std::mutex> lk(mu_call_);
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int64_t> d(0, n_rows_ - 1);
        std::vector<uint8_t> stage(n*rb_);
        double best = 0.0;
        for (int r = 0; r < rounds; ++r) {
            miss_.clear();
            for (size_t i = 0; i < n; ++i) miss_.push_back({ off_ + (uint64_t) d(rng)*rb_, (uint32_t) i });
            std::sort(miss_.begin(), miss_.end(), [](const miss & a, const miss & b) { return a.off < b.off; });
            ext_.clear();
            for (size_t i = 0; i < n; ++i) {
                const uint64_t lo = miss_[i].off & ~(uint64_t) (k_align - 1);
                const uint64_t hi = (miss_[i].off + rb_ + k_align - 1) & ~(uint64_t) (k_align - 1);
                ext_.push_back({ lo, hi, (uint32_t) i, (uint32_t) i + 1 });
            }
            const auto t0 = std::chrono::steady_clock::now();
            run_extents(stage.data());
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (s > 0) best = std::max(best, n / s);
        }
        return best;
    }

private:
    static constexpr int k_ways = 8;

    struct miss   { uint64_t off; uint32_t u; };
    struct extent { uint64_t lo, hi; uint32_t m0, m1; };

    reader(int fd_dir, int fd_buf, uint64_t off, size_t rb, int64_t n_rows, const config & cfg)
        : fd_dir_(fd_dir), fd_buf_(fd_buf), off_(off), rb_(rb), n_rows_(n_rows), cfg_(cfg) {
        n_sets_ = (int64_t) (cfg.cache_bytes / (rb * k_ways));
        if (n_sets_ > 0) {
            const size_t slots = (size_t) n_sets_ * k_ways;
            cache_tag_.reset(new int32_t[slots]);
            std::fill(cache_tag_.get(), cache_tag_.get() + slots, -1);
            cache_stamp_.reset(new uint64_t[slots]());
            cache_data_.reset(new uint8_t[slots * rb]);   // not touched until a row lands in it
        }
        const int nd = std::max(1, cfg.depth);
        auto aligned = [&]() -> void * {
            void * b = nullptr;
            return posix_memalign(&b, k_align, cfg.max_extent + 2*k_align) == 0 ? b : nullptr;
        };
#if defined(__linux__)
        if (cfg.io == 0 && syscall(SYS_io_setup, (long) nd, &aio_ctx_) == 0) {
            aio_ = true;
            iocb_.resize(nd);
            aio_job_.resize(nd);
            aio_ev_.resize(nd);
            for (int i = 0; i < nd; ++i) bufs_.push_back(aligned());
            for (void * b : bufs_) if (b == nullptr) { syscall(SYS_io_destroy, aio_ctx_); aio_ = false; break; }
            if (!aio_) { for (void * b : bufs_) free(b); bufs_.clear(); }
        }
#endif
        if (!aio_) {
            for (int i = 0; i < nd; ++i) bufs_.push_back(aligned());
            for (int i = 1; i < nd; ++i) {
                workers_.emplace_back([this, i] { worker_loop(i); });
            }
        }
    }

    int64_t cache_find(int32_t row) {
        if (n_sets_ <= 0) return -1;
        const size_t base = (size_t) set_of(row) * k_ways;
        for (int w = 0; w < k_ways; ++w) {
            if (cache_tag_[base + w] == row) {
                cache_stamp_[base + w] = ++clock_;
                return (int64_t) (base + w);
            }
        }
        return -1;
    }

    void cache_insert(int32_t row, const uint8_t * bytes) {
        if (n_sets_ <= 0) return;
        const size_t base = (size_t) set_of(row) * k_ways;
        size_t v = base;
        for (int w = 0; w < k_ways; ++w) {
            if (cache_tag_[base + w] == row) { v = base + w; break; }    // already there
            if (cache_tag_[base + w] == -1) { v = base + w; break; }
            if (cache_stamp_[base + w] < cache_stamp_[v]) v = base + w;
        }
        cache_tag_[v]   = row;
        cache_stamp_[v] = ++clock_;
        memcpy(cache_data_.get() + v*rb_, bytes, rb_);
    }

    int64_t set_of(int32_t row) const {
        return (int64_t) ((((uint64_t) (uint32_t) row * 0x9E3779B97F4A7C15ull) >> 32) % (uint64_t) n_sets_);
    }

    // Executes ext_ over the pool: every thread (the caller included) claims extents until none are
    // left. The rows of each extent are copied from the thread's aligned buffer into the stage.
    // A worker joins a generation only while it is open (under mu_pool_) and is counted in active_;
    // the caller closes the generation after its own drain and waits for active_ == 0, so no worker
    // can still be reading ext_ / miss_ when the next call rebuilds them.
    void run_extents(uint8_t * stage) {
        job_stage_ = stage;
        job_next_.store(0);
        const size_t nj = ext_.size();
        if (aio_) {
            run_aio(nj);
            return;
        }
        if (!workers_.empty() && nj > 1) {
            {
                std::lock_guard<std::mutex> lk(mu_pool_);
                job_n_ = nj;
                open_  = true;
                ++gen_;
            }
            cv_work_.notify_all();
            drain(0, nj);
            std::unique_lock<std::mutex> lk(mu_pool_);
            open_ = false;
            cv_done_.wait(lk, [&] { return active_ == 0; });
        } else {
            drain(0, nj);
        }
    }

    // Linux AIO: keep up to depth() extents in flight from this thread; a completion frees its
    // buffer slot for the next extent. Extents io_submit will not take are read synchronously.
    void run_aio(size_t nj) {
#if defined(__linux__)
        const int D = (int) iocb_.size();
        aio_free_.clear();
        for (int s = D - 1; s >= 0; --s) aio_free_.push_back(s);
        size_t next = 0, done = 0;
        while (done < nj) {
            aio_sub_.clear();
            while (next < nj && !aio_free_.empty()) {
                const int sl = aio_free_.back();
                aio_free_.pop_back();
                struct iocb & cb = iocb_[sl];
                memset(&cb, 0, sizeof(cb));
                cb.aio_fildes     = (uint32_t) fd_dir_;
                cb.aio_lio_opcode = IOCB_CMD_PREAD;
                cb.aio_buf        = (uint64_t) (uintptr_t) bufs_[sl];
                cb.aio_nbytes     = ext_[next].hi - ext_[next].lo;
                cb.aio_offset     = (int64_t) ext_[next].lo;
                cb.aio_data       = (uint64_t) sl;
                aio_job_[sl] = (uint32_t) next++;
                aio_sub_.push_back(&cb);
            }
            size_t k = 0;
            while (k < aio_sub_.size()) {
                const long r = syscall(SYS_io_submit, aio_ctx_, (long) (aio_sub_.size() - k), aio_sub_.data() + k);
                if (r > 0) { k += (size_t) r; continue; }
                if (r < 0 && errno == EINTR) continue;
                for (; k < aio_sub_.size(); ++k) {   // not accepted: read these synchronously
                    const int sl = (int) aio_sub_[k]->aio_data;
                    do_extent(ext_[aio_job_[sl]], bufs_[sl]);
                    aio_free_.push_back(sl);
                    ++done;
                }
            }
            if ((int) aio_free_.size() == D) continue;   // nothing in flight
            const long n = syscall(SYS_io_getevents, aio_ctx_, 1L, (long) D, aio_ev_.data(), nullptr);
            if (n < 0) {
                if (errno == EINTR) continue;
                fprintf(stderr, "PXA_PLE_DIRECT: io_getevents failed (%s)\n", strerror(errno));
                abort();
            }
            for (long i = 0; i < n; ++i) {
                const int sl = (int) aio_ev_[i].data;
                finish_extent(ext_[aio_job_[sl]], bufs_[sl], (ssize_t) aio_ev_[i].res);
                aio_free_.push_back(sl);
                ++done;
            }
        }
#else
        (void) nj;
#endif
    }

    void worker_loop(int tid) {
        uint64_t seen = 0;
        for (;;) {
            size_t nj = 0;
            {
                std::unique_lock<std::mutex> lk(mu_pool_);
                cv_work_.wait(lk, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
                if (!open_) continue;
                nj = job_n_;
                ++active_;
            }
            drain(tid, nj);
            {
                std::lock_guard<std::mutex> lk(mu_pool_);
                if (--active_ == 0) cv_done_.notify_all();
            }
        }
    }

    void drain(int tid, size_t nj) {
        for (;;) {
            const size_t j = job_next_.fetch_add(1);
            if (j >= nj) return;
            do_extent(ext_[j], bufs_[tid]);
        }
    }

    void do_extent(const extent & e, void * buf) {
#if defined(__linux__)
        const size_t len = (size_t) (e.hi - e.lo);
        ssize_t got = -1;
        if (buf != nullptr) {
            got = pread(fd_dir_, buf, len, (off_t) e.lo);
            if (got < 0 && errno == EINTR) got = pread(fd_dir_, buf, len, (off_t) e.lo);
        }
        finish_extent(e, buf, got);
#else
        (void) e; (void) buf;
#endif
    }

    // copy the extent's rows out of its buffer (`got` bytes landed, < 0 = failed); a row the read did
    // not cover is read again through the buffered descriptor
    void finish_extent(const extent & e, const void * buf, ssize_t got) {
#if defined(__linux__)
        for (uint32_t i = e.m0; i < e.m1; ++i) {
            const miss & m = miss_[i];
            uint8_t * dst = job_stage_ + (size_t) m.u*rb_;
            const uint64_t need = m.off + rb_ - e.lo;
            if (got >= 0 && (uint64_t) got >= need) {
                memcpy(dst, (const uint8_t *) buf + (m.off - e.lo), rb_);
                continue;
            }
            // failed / short O_DIRECT read: the same bytes through the buffered descriptor
            size_t done = 0;
            while (done < rb_) {
                const ssize_t r = pread(fd_buf_, dst + done, rb_ - done, (off_t) (m.off + done));
                if (r < 0 && errno == EINTR) continue;
                if (r <= 0) {
                    fprintf(stderr, "PXA_PLE_DIRECT: read of table row at file offset %llu failed (%s)\n",
                            (unsigned long long) m.off, r < 0 ? strerror(errno) : "EOF");
                    abort();
                }
                done += (size_t) r;
            }
            fallback_.fetch_add(1);
        }
#else
        (void) e; (void) buf; (void) got;
#endif
    }

    int      fd_dir_ = -1;
    int      fd_buf_ = -1;
    uint64_t off_    = 0;
    size_t   rb_     = 0;
    int64_t  n_rows_ = 0;
    config   cfg_;

    // cache
    int64_t                     n_sets_ = 0;
    std::unique_ptr<int32_t[]>  cache_tag_;
    std::unique_ptr<uint64_t[]> cache_stamp_;
    std::unique_ptr<uint8_t[]>  cache_data_;
    uint64_t                    clock_ = 0;

    // per-call scratch (guarded by mu_call_)
    std::vector<int32_t> ht_key_, ht_val_, uniq_;
    std::vector<miss>    miss_;
    std::vector<extent>  ext_;
    mutable std::mutex   mu_call_;
    stats                st_;

    // pool
    std::vector<std::thread> workers_;
    std::vector<void *>      bufs_;
    std::mutex               mu_pool_;
    std::condition_variable  cv_work_, cv_done_;
    uint64_t                 gen_  = 0;
    bool                     stop_ = false;
    bool                     open_ = false;
    int                      active_ = 0;
    size_t                   job_n_ = 0;
    uint8_t *                job_stage_ = nullptr;
    std::atomic<size_t>      job_next_{0};
    std::atomic<uint64_t>    fallback_{0};

    // Linux AIO
    bool                     aio_ = false;
#if defined(__linux__)
    aio_context_t            aio_ctx_ = 0;
    std::vector<struct iocb>     iocb_;
    std::vector<struct io_event> aio_ev_;
    std::vector<struct iocb *>   aio_sub_;
#else
    std::vector<int>         iocb_;
#endif
    std::vector<uint32_t>    aio_job_;
    std::vector<int>         aio_free_;
};

} // namespace pxa_ple_direct
