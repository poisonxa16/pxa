// bench-pxqn-cpu-mmv.c -- CLOSED tree only. Micro-bench for ggml/src/pxqn-cpu-mmv.c.
//
// One routed-expert DECODE STEP of a Flash-Next-class layer (n_embd 2560, expert ffn 640, 10 of 512 experts):
//   up   : 10 x mul_mat [K=2560 -> 640]       (one ggml node: every thread, every expert, then a barrier)
//   gate : 10 x mul_mat [K=2560 -> 640]
//   down : 10 x mul_mat [K=640  -> 2560]
// with the experts drawn at random from a pool far larger than the L3, so every weight byte comes from DRAM.
// Reported: weight bytes per step, time per step, aggregate GB/s of weight bytes, against a stream-like read of the SAME pool
// with the SAME threads (the DRAM ceiling), and against the same kernel on cache-resident weights (the compute ceiling).
//
//   bench-pxqn-cpu-mmv [-t N] [-cpus 0-17,36-53] [-types 1,2,3,4,5,6,7] [-steps 200] [-ny 1] [-pool-mb 1024] [-isa auto|base]
#define _GNU_SOURCE
#include "ggml.h"
#include "ggml-pxqn-codec.h"
#include "ggml-pxqn-api.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

void pxqn_cpu_mmv_isa(int isa, enum ggml_type type, const void * a, int64_t nr0, int64_t k,
                      const char * src1f, size_t nb11, size_t nb12, char * dst, size_t nb1, size_t nb2,
                      const struct ggml_pxqn_rowmap * rows, int ne11, int64_t ny, int ith, int nth);
const char * pxqn_cpu_mmv_isa_name(void);

bool ggml_is_contiguous(const struct ggml_tensor * t) { (void) t; return true; }
bool ggml_are_same_shape(const struct ggml_tensor * a, const struct ggml_tensor * b) { (void) a; (void) b; return true; }
int64_t ggml_nrows(const struct ggml_tensor * t) { (void) t; return 0; }
void ggml_abort(const char * file, int line, const char * fmt, ...) { fprintf(stderr, "abort %s:%d %s\n", file, line, fmt); abort(); }

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9*ts.tv_nsec; }

struct tinfo { enum ggml_type t; const char * name; int qk, slab; };
static const struct tinfo TI[] = {
    { GGML_TYPE_PXQN1,   "PXQN1",   128, PXQN1_SLAB_BYTES },
    { GGML_TYPE_PXQN2,   "PXQN2",   128, PXQN2_SLAB_BYTES },
    { GGML_TYPE_PXQN3,   "PXQN3",   128, PXQN3_SLAB_BYTES },
    { GGML_TYPE_PXQN3S8, "PXQN3S8", 128, PXQN3S8_SLAB_BYTES },
    { GGML_TYPE_PXQN4,   "PXQN4",   32,  PXQN4_SLAB_BYTES },
    { GGML_TYPE_PXQN4S8, "PXQN4S8", 32,  PXQN4S8_SLAB_BYTES },
    { GGML_TYPE_PXQN5,   "PXQN5",   128, PXQN5_SLAB_BYTES },
};

static size_t mat_bytes(const struct tinfo * ti, int64_t nr, int64_t k) { return (size_t) (nr/64)*(PXQN_HDR_BYTES + (size_t) (k/ti->qk)*ti->slab); }

// ---- spin barrier + worker pool ----------------------------------------------------------------
struct bar { atomic_int cnt; atomic_int sense; int n; };
static void bar_wait(struct bar * b, int * local) {
    *local = !*local;
    if (atomic_fetch_add(&b->cnt, 1) == b->n - 1) { atomic_store(&b->cnt, 0); atomic_store(&b->sense, *local); }
    else { while (atomic_load_explicit(&b->sense, memory_order_acquire) != *local) { _mm_pause(); } }
}

enum { J_QUIT = 0, J_STEP, J_STREAM, J_HOT };
struct job {
    int kind; int isa; const struct tinfo * ti; int ny; int nexp;
    const uint8_t * pool[3]; size_t esz[3]; int64_t nr[3], k[3]; int64_t npool[3];
    const int * pick;               // nexp x 3 expert indices for this step
    const float * x[2];             // activations (2560 / 640 floats x ny)
    float * dst;                    // [ny x 2560]
    int steps;
};
struct ctx { int nth; struct bar b; struct job j; int cpus[256]; int ncpu; double sink[256]; };
struct targ { struct ctx * c; int ith; };

static void pin(int cpu) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); pthread_setaffinity_np(pthread_self(), sizeof(s), &s); }

static void do_step(struct ctx * c, int ith, int * sense) {
    struct job * j = &c->j;
    // stage 0 up (K=2560 -> 640), stage 1 gate (same shape), stage 2 down (K=640 -> 2560)
    for (int st = 0; st < 3; ++st) {
        const int sel = st == 2 ? 2 : (st == 0 ? 0 : 1);
        for (int e = 0; e < j->nexp; ++e) {
            const uint8_t * w = j->pool[sel] + (size_t) j->pick[e*3 + sel] * j->esz[sel];
            const float * x = j->x[st == 2 ? 1 : 0];
            pxqn_cpu_mmv_isa(j->isa, j->ti->t, w, j->nr[sel], j->k[sel], (const char *) x, (size_t) j->k[sel]*4, 0,
                             (char *) j->dst + (size_t) e*0, (size_t) j->nr[sel]*4, 0, NULL, 1, j->ny, ith, c->nth);
        }
        bar_wait(&c->b, sense);
    }
}

static void do_stream(struct ctx * c, int ith, int * sense) {
    // read the pick-list experts' bytes with plain wide loads, same partition shape as the kernel (rows split by thread)
    struct job * j = &c->j;
    __m256i acc = _mm256_setzero_si256();
    for (int st = 0; st < 3; ++st) {
        const int sel = st == 2 ? 2 : (st == 0 ? 0 : 1);
        for (int e = 0; e < j->nexp; ++e) {
            const uint8_t * w = j->pool[sel] + (size_t) j->pick[e*3 + sel] * j->esz[sel];
            const size_t n = j->esz[sel]/64;
            const size_t a = n*ith/c->nth, b = n*(ith + 1)/c->nth;
            for (size_t i = a; i < b; ++i) {
                const __m256i * p = (const __m256i *) (w + i*64);
                acc = _mm256_add_epi64(acc, _mm256_add_epi64(_mm256_load_si256(p), _mm256_load_si256(p + 1)));
            }
        }
        bar_wait(&c->b, sense);
    }
    c->sink[ith] += (double) _mm256_extract_epi64(acc, 0);
}

static void * worker(void * arg) {
    struct targ * ta = (struct targ *) arg; struct ctx * c = ta->c; const int ith = ta->ith;
    pin(c->cpus[ith % c->ncpu]);
    int sense = 0;
    for (;;) {
        bar_wait(&c->b, &sense);           // start
        if (c->j.kind == J_QUIT) break;
        for (int s = 0; s < c->j.steps; ++s) {
            if (c->j.kind == J_STREAM) do_stream(c, ith, &sense); else do_step(c, ith, &sense);
        }
        bar_wait(&c->b, &sense);           // done
    }
    return NULL;
}

static unsigned long long g_rng = 88172645463325252ull;
static unsigned long long rnd64(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return g_rng; }

static void fill_pool(uint8_t * p, size_t bytes, const struct tinfo * ti, int64_t nr, int64_t k, size_t esz) {
    uint64_t * q = (uint64_t *) p;
    for (size_t i = 0; i < bytes/8; ++i) q[i] = rnd64();
    const size_t pb = PXQN_HDR_BYTES + (size_t) (k/ti->qk)*ti->slab;
    for (size_t e = 0; e < bytes/esz; ++e)
        for (int64_t pn = 0; pn < nr/64; ++pn) {
            uint16_t * h = (uint16_t *) (p + e*esz + pn*pb);
            for (int r = 0; r < 64; ++r) h[r] = 0x2e66 + (uint16_t) (rnd64() & 0xff);   // ~0.1 .. 0.2 (normal fp16)
        }
}

static int parse_cpus(const char * s, int * out) {
    int n = 0;
    while (*s) {
        int a = (int) strtol(s, (char **) &s, 10), b = a;
        if (*s == '-') { ++s; b = (int) strtol(s, (char **) &s, 10); }
        for (int i = a; i <= b && n < 256; ++i) out[n++] = i;
        if (*s == ',') ++s;
    }
    return n;
}

int main(int argc, char ** argv) {
    int nth = 8, steps = 200, ny = 1, pool_mb = 1024, isa = -1;
    const char * cpus = NULL, * types = "2,1";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) nth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-cpus") && i + 1 < argc) cpus = argv[++i];
        else if (!strcmp(argv[i], "-types") && i + 1 < argc) types = argv[++i];
        else if (!strcmp(argv[i], "-steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ny") && i + 1 < argc) ny = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-pool-mb") && i + 1 < argc) pool_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-isa") && i + 1 < argc) { ++i; isa = !strcmp(argv[i], "base") ? 0 : (!strcmp(argv[i], "ref") ? 2 : 1); }
    }
    static struct ctx C;
    if (cpus) C.ncpu = parse_cpus(cpus, C.cpus);
    else { cpu_set_t s; sched_getaffinity(0, sizeof(s), &s); for (int i = 0; i < CPU_SETSIZE && C.ncpu < 256; ++i) if (CPU_ISSET(i, &s)) C.cpus[C.ncpu++] = i; }
    if (isa < 0) isa = !strcmp(pxqn_cpu_mmv_isa_name(), "avx2");
    C.nth = nth;
    atomic_init(&C.b.cnt, 0); atomic_init(&C.b.sense, 0); C.b.n = nth;
    double la[3] = { 0, 0, 0 }; if (getloadavg(la, 3) < 0) la[0] = -1;
    printf("pxqn-cpu-mmv bench: isa=%s threads=%d cpus=%d..%d (%d allowed) ny=%d steps=%d pool=%d MB/matrix  loadavg %.1f %.1f %.1f\n",
           isa == 1 ? "avx2" : (isa == 0 ? "sse2" : "scalar-ref"), nth, C.cpus[0], C.cpus[C.ncpu > nth ? nth - 1 : C.ncpu - 1], C.ncpu, ny, steps, pool_mb, la[0], la[1], la[2]);

    pthread_t th[256]; struct targ ta[256];
    pin(C.cpus[0]);
    for (int i = 1; i < nth; ++i) { ta[i].c = &C; ta[i].ith = i; pthread_create(&th[i], NULL, worker, &ta[i]); }

    static float xa[8*2560], xb[8*640], dst[8*2560];
    for (int i = 0; i < 8*2560; ++i) xa[i] = (float) ((int) (rnd64() % 2001) - 1000)/500.0f;
    for (int i = 0; i < 8*640;  ++i) xb[i] = (float) ((int) (rnd64() % 2001) - 1000)/500.0f;
    C.j.x[0] = xa; C.j.x[1] = xb; C.j.dst = dst; C.j.ny = ny; C.j.nexp = 10; C.j.isa = isa;
    for (const char * s = types; *s; ) {
        const int ti_i = atoi(s) - 1;
        while (*s && *s != ',') ++s;
        if (*s == ',') ++s;
        if (ti_i < 0 || ti_i > 6) continue;
        const struct tinfo * ti = &TI[ti_i];
        C.j.ti = ti;
        const int64_t nr[3] = { 640, 640, 2560 }, kk[3] = { 2560, 2560, 640 };
        size_t tot = 0;
        for (int m = 0; m < 3; ++m) {
            C.j.nr[m] = nr[m]; C.j.k[m] = kk[m];
            C.j.esz[m] = mat_bytes(ti, nr[m], kk[m]);
            C.j.npool[m] = (int64_t) ((size_t) pool_mb*1048576 / C.j.esz[m]);
            const size_t bytes = (size_t) C.j.npool[m]*C.j.esz[m];
            uint8_t * p = (uint8_t *) aligned_alloc(4096, (bytes + 4095) & ~(size_t) 4095);
            if (!p) { fprintf(stderr, "alloc failed\n"); return 1; }
            fill_pool(p, bytes, ti, nr[m], kk[m], C.j.esz[m]);
            C.j.pool[m] = p;
            tot += (size_t) C.j.nexp*C.j.esz[m];
        }
        static int pick[10*3*512];
        // --- stream-like ceiling (same threads, same bytes, wide loads), then the kernel, interleaved x3 ---
        double best_s = 1e30, best_k = 1e30, best_h = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            for (int pass = 0; pass < 2; ++pass) {
                const int kind = pass == 0 ? J_STREAM : J_STEP;
                g_rng = 0x1234567ull + rep;
                double t_total = 0;
                int sense = 0;
                for (int s = 0; s < steps; ++s) {
                    for (int e = 0; e < C.j.nexp; ++e) for (int m = 0; m < 3; ++m) pick[e*3 + m] = (int) (rnd64() % (unsigned long long) C.j.npool[m]);
                    C.j.pick = pick; C.j.kind = kind; C.j.steps = 1;
                    const double t0 = now_s();
                    bar_wait(&C.b, &sense);                 // start
                    if (kind == J_STREAM) do_stream(&C, 0, &sense); else do_step(&C, 0, &sense);
                    bar_wait(&C.b, &sense);                 // done
                    t_total += now_s() - t0;
                }
                const double per = t_total/steps;
                if (pass == 0) { if (per < best_s) best_s = per; } else { if (per < best_k) best_k = per; }
            }
        }
        // --- compute ceiling: the kernel on the SAME expert every step (L2/L3 resident) ---
        {
            int sense = 0;
            for (int e = 0; e < 10; ++e) for (int m = 0; m < 3; ++m) pick[e*3 + m] = e % 2;   // 2 experts per matrix, hot
            C.j.pick = pick; C.j.kind = J_STEP; C.j.steps = 1;
            double t_total = 0;
            for (int s = 0; s < steps; ++s) {
                const double t0 = now_s();
                bar_wait(&C.b, &sense); do_step(&C, 0, &sense); bar_wait(&C.b, &sense);
                t_total += now_s() - t0;
            }
            best_h = t_total/steps;
        }
        const double gb = tot/1e9;
        printf("%-8s  weight bytes/step %7.2f MB | stream %7.1f us (%6.1f GB/s)  kernel %7.1f us (%6.1f GB/s = %4.0f%% of stream, %.2fx time)  cache-hot kernel %7.1f us (%6.1f GB/s)\n",
               ti->name, tot/1e6, best_s*1e6, gb/best_s, best_k*1e6, gb/best_k, 100.0*best_s/best_k, best_k/best_s, best_h*1e6, gb/best_h);
        fflush(stdout);
        for (int m = 0; m < 3; ++m) free((void *) C.j.pool[m]);
    }
    int sense = 0;
    C.j.kind = J_QUIT;
    bar_wait(&C.b, &sense);
    for (int i = 1; i < nth; ++i) pthread_join(th[i], NULL);
    return 0;
}
