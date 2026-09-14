// PXA_FA_TILE_V2: the re-scheduled tile flash-attention kernel must deliver the SAME BYTES to the
// same arithmetic as the kernel it replaces.
//
// WHAT THIS PINS. ggml/src/ggml-cuda/pxa/fattn-tile-v2.cu is a schedule change and nothing else:
// it moves the staging tile off a padded row stride (half2 KV_tmp[64][D/2 + 1]) onto an unpadded
// one with a phase XOR swizzle (row r keeps its 16-byte chunk c at physical chunk c ^ (r mod R)),
// so that the hot QK loop can read K and Q 128 bits at a time instead of 32. The arithmetic --
// the tile walk, the 64 kqsum slots, the per-tile rescale, the k order, the butterfly fold, the
// fp32 running state -- is the shipping kernel's, unchanged. That claim is only worth as much as
// the addressing behind it, so this test checks the addressing directly, three ways:
//
//   PART 1 -- THE SWIZZLE, COMBINATORIALLY. For every head size the kernel serves it proves the
//   permutation stays inside its row, is its own inverse, and that all three access patterns the
//   kernel actually issues are bank-conflict free: the QK read (32 lanes on 32 DIFFERENT rows,
//   one 16-byte chunk each, four phases of eight lanes), the PV read (32 lanes on ONE row, 32
//   consecutive half2), and the staging store (the PV pattern). It also measures the SHIPPING
//   padded layout under the same rules, so the record shows what the pad bought and what it cost:
//   the pad is conflict free for 32-bit access and cannot express a 16-byte access at all.
//
//   PART 2 -- THE READ-BACK IS THE VALUE THAT WAS WRITTEN. A 128-bit load reads four physically
//   CONSECUTIVE half2. Under the swizzle those four must be exactly the four LOGICAL half2 of the
//   chunk that was staged there. This walks every (row, chunk) of every head size and requires it.
//   A swizzle applied on the store but inverted on the load would pass Part 1 and fail here.
//
//   PART 3 -- THE WHOLE ONLINE SOFTMAX, BOTH SCHEDULES, AT SIX KV PLACEMENTS. A term-for-term
//   mirror of the kernel -- Q staged and scaled to half, K staged through the simulated shared
//   tile and read back through the schedule's own pattern, scores accumulated in half2 and folded,
//   the mask, the fp32 running max/sum/numerator, the probabilities stored back as half, V staged
//   and read back, the 64-slot butterfly -- run once with the shipping layout and once with the
//   swizzled one. It requires them BITWISE EQUAL at every placement, and requires both to stay
//   placement-invariant and to agree with a whole-range double-precision reference, which is the
//   property fp32 accumulation bought (tests/test-fa-band-align.cpp) and that a new schedule must
//   inherit rather than merely not obviously break.
//
// Deterministic inputs from a fixed LCG. CPU only, no GPU, no model.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

#define TILE   64   // FATTN_KQ_STRIDE_TILE_V2
#define SLOTS  64   // one kqsum accumulator per key-slot in a tile (32 lanes x half2)
#define WARP   32

// ---------------------------------------------------------------------------------------------
// fp16 rounding, written out so the test carries no ggml initialisation order of its own.

static float to_half_and_back(float x) {
    if (!std::isfinite(x)) {
        return x;
    }
    const float ax = fabsf(x);
    if (ax >= 65520.0f) {
        return x > 0.0f ? INFINITY : -INFINITY;
    }
    if (ax < 6.103515625e-05f) {
        return rintf(x*16777216.0f)/16777216.0f;
    }
    int e = 0;
    frexpf(ax, &e);
    const float ulp = ldexpf(1.0f, e - 11);
    return rintf(x/ulp)*ulp;
}

// ---------------------------------------------------------------------------------------------
// The swizzle, mirrored from pxa_v2_chunk / pxa_v2_h2 in fattn-tile-v2.cu.

static inline int R_of(int D)                        { return (D/2)/4; }
static inline int sw_chunk(int D, int r, int c)      { return c ^ (r & (R_of(D) - 1)); }
static inline int sw_h2(int D, int r, int i)         { return (sw_chunk(D, r, i >> 2) << 2) | (i & 3); }

// Physical half2 index of a logical (row, half2) under each layout. `stride` is in half2.
//   shipping: row stride D/2 + 1, no permutation  (the pad walks the banks)
//   v2      : row stride D/2,     phase XOR swizzle
static inline int phys_v1(int D, int r, int i) { return r*((D/2) + 1) + i; }
static inline int phys_v2(int D, int r, int i) { return r*(D/2)       + sw_h2(D, r, i); }

// ---------------------------------------------------------------------------------------------
// PART 1: the swizzle, combinatorially.
//
// Banks are 32 x 4 bytes. A half2 is one 4-byte word, so the bank of a physical half2 index p is
// p % 32. A 128-bit access is serviced in four phases of eight lanes; each phase must cover 32
// distinct banks, i.e. its eight 16-byte chunks must land in eight distinct 4-bank groups, so the
// group index (p/4) % 8 must be distinct across the phase.

static int part1(int D) {
    const int R = R_of(D);
    int fail = 0;

    // (a) the permutation stays inside the row and is its own inverse
    for (int r = 0; r < TILE; ++r) {
        std::set<int> seen;
        for (int c = 0; c < R; ++c) {
            const int p = sw_chunk(D, r, c);
            if (p < 0 || p >= R) {
                printf("    FAIL D=%d: row %d chunk %d swizzles out of the row (-> %d, R = %d)\n", D, r, c, p, R);
                fail = 1;
            }
            if (!seen.insert(p).second) {
                printf("    FAIL D=%d: row %d chunk %d collides at physical chunk %d\n", D, r, c, p);
                fail = 1;
            }
            if (sw_chunk(D, r, p) != c) {
                printf("    FAIL D=%d: swizzle is not an involution at row %d chunk %d\n", D, r, c);
                fail = 1;
            }
        }
    }

    // (b) QK read, 128-bit: lanes are ROWS (i_KQ = i_KQ_0 + threadIdx.x), one chunk each.
    int qk_worst = 1;
    for (int i_KQ_0 = 0; i_KQ_0 < TILE; i_KQ_0 += WARP) {
        for (int c = 0; c < R; ++c) {
            for (int phase = 0; phase < 4; ++phase) {
                std::set<int> groups;
                for (int l = 0; l < 8; ++l) {
                    const int r = i_KQ_0 + phase*8 + l;
                    const int p = r*(D/2) + 4*sw_chunk(D, r, c); // physical half2 index of the chunk
                    groups.insert((p/4) % 8);
                }
                const int ways = 8 / (int) groups.size();
                if (ways > qk_worst) qk_worst = ways;
            }
        }
    }
    if (qk_worst != 1) {
        printf("    FAIL D=%d: QK 128-bit read has %d-way bank conflicts\n", D, qk_worst);
        fail = 1;
    }

    // (c) PV read / staging store, 32-bit: lanes are COLUMNS of ONE row.
    int pv_worst = 1;
    for (int r = 0; r < TILE; ++r) {
        for (int i0 = 0; i0 < D/2; i0 += WARP) {
            std::set<int> banks;
            for (int tx = 0; tx < WARP; ++tx) {
                banks.insert(phys_v2(D, r, i0 + tx) % 32);
            }
            const int ways = WARP / (int) banks.size();
            if (ways > pv_worst) pv_worst = ways;
        }
    }
    if (pv_worst != 1) {
        printf("    FAIL D=%d: PV/staging 32-bit access has %d-way bank conflicts\n", D, pv_worst);
        fail = 1;
    }

    // and the shipping padded layout under the same two rules, for the record.
    int qk32_pad = 1, pv32_pad = 1, align_pad = 0;
    for (int i_KQ_0 = 0; i_KQ_0 < TILE; i_KQ_0 += WARP) {
        for (int k = 0; k < D/2; ++k) {
            std::set<int> banks;
            for (int tx = 0; tx < WARP; ++tx) {
                banks.insert(phys_v1(D, i_KQ_0 + tx, k) % 32);
            }
            const int ways = WARP / (int) banks.size();
            if (ways > qk32_pad) qk32_pad = ways;
        }
    }
    for (int r = 0; r < TILE; ++r) {
        for (int i0 = 0; i0 < D/2; i0 += WARP) {
            std::set<int> banks;
            for (int tx = 0; tx < WARP; ++tx) {
                banks.insert(phys_v1(D, r, i0 + tx) % 32);
            }
            const int ways = WARP / (int) banks.size();
            if (ways > pv32_pad) pv32_pad = ways;
        }
    }
    for (int r = 0; r < TILE; ++r) {
        if ((phys_v1(D, r, 0) % 4) != 0) {
            align_pad++; // rows whose start is not 16-byte aligned: a 128-bit load is impossible
        }
    }

    printf("  D = %-3d  R = %-2d chunks/row |  v2 swizzled: QK.128 %d-way, PV.32 %d-way, 16B-misaligned rows %d\n",
           D, R, qk_worst, pv_worst, 0);
    printf("                              |  shipping pad: QK.32 %d-way, PV.32 %d-way, 16B-misaligned rows %d of %d\n",
           qk32_pad, pv32_pad, align_pad, TILE);
    if (align_pad == 0) {
        printf("    FAIL D=%d: the shipping padded layout is claimed to forbid 128-bit rows but every row is aligned\n", D);
        fail = 1; // the comparison has gone stale
    }
    return fail;
}

// ---------------------------------------------------------------------------------------------
// PART 2: a 128-bit load reads four physically consecutive half2 -- they must be the four logical
// half2 of the chunk that was staged there.

static int part2(int D) {
    const int R = R_of(D);
    std::vector<int> smem((size_t) TILE*(D/2), -1);

    for (int r = 0; r < TILE; ++r) {
        for (int i = 0; i < D/2; ++i) {
            smem[(size_t) phys_v2(D, r, i)] = r*(D/2) + i; // stage the logical id
        }
    }
    int fail = 0;
    for (int r = 0; r < TILE; ++r) {
        for (int c = 0; c < R; ++c) {
            const int base = r*(D/2) + 4*sw_chunk(D, r, c); // what the kernel's pxa_h2x4 load reads
            for (int s = 0; s < 4; ++s) {
                const int want = r*(D/2) + 4*c + s;
                if (smem[(size_t) base + s] != want) {
                    printf("    FAIL D=%d: row %d chunk %d element %d read %d, staged %d\n",
                           D, r, c, s, smem[(size_t) base + s], want);
                    fail = 1;
                }
            }
        }
    }
    if (!fail) {
        printf("  D = %-3d  every 16-byte chunk reads back exactly the four logical half2 staged into it\n", D);
    }
    return fail;
}

// ---------------------------------------------------------------------------------------------
// PART 3: the whole online softmax, both schedules.

struct lcg {
    uint64_t s;
    explicit lcg(uint64_t seed) : s(seed) {}
    uint32_t next() { s = s*6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t) (s >> 33); }
    float unit() { return (float) next() / (float) 0x80000000u - 1.0f; }
};

// One simulated shared tile, staged and read back through a schedule's own addressing. Values are
// carried as half-rounded floats -- what a half2 slot holds.
struct tile_smem {
    int D, D2;
    bool v2;
    std::vector<float> lo, hi;
    tile_smem(int D_, bool v2_) : D(D_), D2(D_/2), v2(v2_) {
        const size_t n = (size_t) TILE*(v2_ ? D2 : D2 + 1);
        lo.assign(n, 0.0f);
        hi.assign(n, 0.0f);
    }
    int at(int r, int i) const { return v2 ? phys_v2(D, r, i) : phys_v1(D, r, i); }
    void store(int r, int i, float x, float y) {
        lo[(size_t) at(r, i)] = to_half_and_back(x);
        hi[(size_t) at(r, i)] = to_half_and_back(y);
    }
    // The shipping schedule's read: one half2 per k. The v2 schedule's read: four physically
    // consecutive half2 starting at the swizzled chunk. Both are expressed here as the kernel
    // expresses them, so a wrong inverse shows up as a wrong value rather than as a wrong index.
    void load_k(int r, int k, float & x, float & y) const {
        size_t p;
        if (v2) {
            p = (size_t) (r*D2 + 4*sw_chunk(D, r, k >> 2) + (k & 3));
        } else {
            p = (size_t) phys_v1(D, r, k);
        }
        x = lo[p];
        y = hi[p];
    }
};

// scores are computed, not given: the QK dot product runs through the staged K so the addressing
// is exercised. half2 accumulation with a single rounding per fused multiply-add, then the
// low+high fold in half -- the kernel's own sequence.
static float qk_dot(const tile_smem & K, int row, const std::vector<float> & Qlo,
                    const std::vector<float> & Qhi, int D2) {
    float sx = 0.0f, sy = 0.0f;
    for (int k = 0; k < D2; ++k) {
        float kx, ky;
        K.load_k(row, k, kx, ky);
        sx = to_half_and_back(kx*Qlo[(size_t) k] + sx);
        sy = to_half_and_back(ky*Qhi[(size_t) k] + sy);
    }
    return to_half_and_back(sx + sy);
}

// The kernel's tile walk with fp32 running state, run through one schedule's addressing.
// The band occupies cells [offset, offset + n) of a key range padded up to `extent`.
static std::vector<double> fattn_tile_v(bool v2, int D, const std::vector<float> & Kv,
                                        const std::vector<float> & Vv, int n, int offset,
                                        int extent, const std::vector<float> & Q, float scale) {
    const int D2 = D/2;
    const int Dv = D;

    // Q staged and scaled to half exactly as the kernel does
    std::vector<float> Qlo((size_t) D2), Qhi((size_t) D2);
    for (int i = 0; i < D2; ++i) {
        Qlo[(size_t) i] = to_half_and_back(to_half_and_back(scale)*to_half_and_back(Q[(size_t) 2*i + 0]));
        Qhi[(size_t) i] = to_half_and_back(to_half_and_back(scale)*to_half_and_back(Q[(size_t) 2*i + 1]));
    }

    std::vector<float> acc(SLOTS, 0.0f);
    std::vector<float> VKQ((size_t) Dv, 0.0f);
    float kqmax = -3.402823466e+38f/2.0f;

    tile_smem KV(D, v2);

    for (int t0 = 0; t0 < extent; t0 += TILE) {
        bool any = false;
        for (int k = 0; k < TILE; ++k) {
            const int cell = t0 + k;
            any = any || (cell >= offset && cell < offset + n);
        }
        if (!any) {
            continue; // PXA_FA_MASK_SKIP_TILE: a fully masked tile contributes exactly nothing
        }

        // stage K
        for (int r = 0; r < TILE; ++r) {
            const int cell = t0 + r;
            const bool mine = cell >= offset && cell < offset + n;
            for (int i = 0; i < D2; ++i) {
                const float x = mine ? Kv[(size_t) (cell - offset)*D + 2*i + 0] : 0.0f;
                const float y = mine ? Kv[(size_t) (cell - offset)*D + 2*i + 1] : 0.0f;
                KV.store(r, i, x, y);
            }
        }

        // scores + running max
        float tile_score[TILE];
        float kqmax_new = kqmax;
        for (int k = 0; k < TILE; ++k) {
            const int cell = t0 + k;
            const bool mine = cell >= offset && cell < offset + n;
            tile_score[k] = mine ? qk_dot(KV, k, Qlo, Qhi, D2) : -INFINITY;
            if (tile_score[k] > kqmax_new) kqmax_new = tile_score[k];
        }

        const float KQ_max_scale = expf(kqmax - kqmax_new);
        kqmax = kqmax_new;

        // probabilities: exp in fp32, stored back to the score tile as half
        float val[TILE];
        for (int k = 0; k < TILE; ++k) {
            val[k] = std::isfinite(tile_score[k]) ? to_half_and_back(expf(tile_score[k] - kqmax)) : 0.0f;
        }
        for (int k = 0; k < TILE; ++k) {
            acc[k] = acc[k]*KQ_max_scale + (std::isfinite(tile_score[k]) ? expf(tile_score[k] - kqmax) : 0.0f);
        }
        for (int d = 0; d < Dv; ++d) {
            VKQ[d] = VKQ[d]*KQ_max_scale;
        }

        // stage V into the tile the K rows just vacated, then walk it in key-pairs
        for (int r = 0; r < TILE; ++r) {
            const int cell = t0 + r;
            const bool mine = cell >= offset && cell < offset + n;
            for (int i = 0; i < D2; ++i) {
                const float x = mine ? Vv[(size_t) (cell - offset)*Dv + 2*i + 0] : 0.0f;
                const float y = mine ? Vv[(size_t) (cell - offset)*Dv + 2*i + 1] : 0.0f;
                KV.store(r, i, x, y);
            }
        }
        for (int k0 = 0; k0 < TILE; k0 += 2) {
            for (int i = 0; i < D2; ++i) {
                float v0x, v0y, v1x, v1y;
                KV.load_k(k0 + 0, i, v0x, v0y);
                KV.load_k(k0 + 1, i, v1x, v1y);
                VKQ[(size_t) 2*i + 0] += v0x*val[k0] + v1x*val[k0 + 1];
                VKQ[(size_t) 2*i + 1] += v0y*val[k0] + v1y*val[k0 + 1];
            }
        }
    }

    // fold the two halves of each lane's half2, then the 32-lane butterfly
    std::vector<float> lane(SLOTS/2);
    for (int l = 0; l < SLOTS/2; ++l) {
        lane[l] = acc[2*l] + acc[2*l + 1];
    }
    for (int step = (SLOTS/2)/2; step >= 1; step /= 2) {
        for (int l = 0; l < step; ++l) {
            lane[l] = lane[l] + lane[l + step];
        }
    }
    const float S = lane[0];

    std::vector<double> out((size_t) Dv);
    for (int d = 0; d < Dv; ++d) {
        out[d] = (double) VKQ[d] / (double) S;
    }
    return out;
}

// whole-range softmax in double precision over the same half-rounded scores the arms see
static std::vector<double> reference(int D, const std::vector<float> & Kv, const std::vector<float> & Vv,
                                     int n, const std::vector<float> & Q, float scale) {
    const int D2 = D/2, Dv = D;
    tile_smem KV(D, false);
    std::vector<float> Qlo((size_t) D2), Qhi((size_t) D2);
    for (int i = 0; i < D2; ++i) {
        Qlo[(size_t) i] = to_half_and_back(to_half_and_back(scale)*to_half_and_back(Q[(size_t) 2*i + 0]));
        Qhi[(size_t) i] = to_half_and_back(to_half_and_back(scale)*to_half_and_back(Q[(size_t) 2*i + 1]));
    }
    std::vector<double> s((size_t) n);
    double m = -INFINITY;
    for (int c = 0; c < n; ++c) {
        for (int i = 0; i < D2; ++i) {
            KV.store(0, i, Kv[(size_t) c*D + 2*i + 0], Kv[(size_t) c*D + 2*i + 1]);
        }
        s[(size_t) c] = (double) qk_dot(KV, 0, Qlo, Qhi, D2);
        if (s[(size_t) c] > m) m = s[(size_t) c];
    }
    double sum = 0.0;
    std::vector<double> out((size_t) Dv, 0.0);
    for (int c = 0; c < n; ++c) {
        const double p = exp(s[(size_t) c] - m);
        sum += p;
        for (int d = 0; d < Dv; ++d) {
            out[d] += p*(double) to_half_and_back(Vv[(size_t) c*Dv + d]);
        }
    }
    for (int d = 0; d < Dv; ++d) {
        out[d] /= sum;
    }
    return out;
}

static double reldiff(const std::vector<double> & a, const std::vector<double> & b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = fabs(a[i] - b[i]);
        if (d > num) num = d;
        if (fabs(b[i]) > den) den = fabs(b[i]);
    }
    return den > 0.0 ? num/den : num;
}

static int part3(int D) {
    const int N   = 4096;   // keys in the band
    const int PAD = 256;    // llama_kv_cache_get_padding() under flash attention
    const float scale = 1.0f/sqrtf((float) D);

    lcg r(20260909ull + (uint64_t) D);
    std::vector<float> Kv((size_t) N*D), Vv((size_t) N*D), Q((size_t) D);
    for (size_t i = 0; i < Q.size();  ++i) Q[i]  = r.unit();
    for (size_t i = 0; i < Kv.size(); ++i) Kv[i] = r.unit();
    for (size_t i = 0; i < Vv.size(); ++i) Vv[i] = r.unit();
    for (int i = 0; i < 12; ++i) {                       // a handful of strong keys
        const int c = (int) (r.next() % (uint32_t) N);
        for (int d = 0; d < D; ++d) Kv[(size_t) c*D + d] += 2.0f*Q[(size_t) d];
    }

    // 0 and 4096 start on the attention window's grain, 1/257/4097 sit one cell past it, and
    // 101 / 20833 land deep inside a tile -- the placements the seat answered differently at.
    const int offsets[] = { 0, 1, 101, 257, 4097, 20833 };
    const int n_off     = (int) (sizeof(offsets)/sizeof(offsets[0]));

    const std::vector<double> ref = reference(D, Kv, Vv, N, Q, scale);

    int fail = 0;
    double worst_place = 0.0, worst_ref = 0.0;
    int bitwise_mismatch = 0;
    std::vector<double> base;

    for (int a = 0; a < n_off; ++a) {
        const int off    = offsets[a];
        const int extent = ((off + N + PAD - 1)/PAD)*PAD;

        const std::vector<double> o1 = fattn_tile_v(false, D, Kv, Vv, N, off, extent, Q, scale);
        const std::vector<double> o2 = fattn_tile_v(true,  D, Kv, Vv, N, off, extent, Q, scale);

        int diff = 0;
        for (size_t i = 0; i < o1.size(); ++i) {
            if (memcmp(&o1[i], &o2[i], sizeof(double)) != 0) diff++;
        }
        bitwise_mismatch += diff;

        const double vr = reldiff(o2, ref);
        if (vr > worst_ref) worst_ref = vr;
        if (a == 0) {
            base = o2;
            printf("    offset %-6d schedules bitwise equal: %s   rel to the double reference = %.3e\n",
                   off, diff == 0 ? "yes" : "NO", vr);
            continue;
        }
        const double d = reldiff(o2, base);
        if (d > worst_place) worst_place = d;
        printf("    offset %-6d schedules bitwise equal: %s   rel to offset 0 = %.3e   (vs reference %.3e)\n",
               off, diff == 0 ? "yes" : "NO", d, vr);
    }

    if (bitwise_mismatch != 0) {
        printf("  FAIL D=%d: the swizzled schedule is not bitwise equal to the shipping schedule "
               "(%d of %d output elements differ)\n", D, bitwise_mismatch, n_off*D);
        fail = 1;
    }
    const double BOUND = 1e-5;
    if (!(worst_place < BOUND)) {
        printf("  FAIL D=%d: the swizzled schedule is not placement-invariant (%.3e >= %.3e)\n", D, worst_place, BOUND);
        fail = 1;
    }
    if (!(worst_ref < 1e-2)) {
        printf("  FAIL D=%d: the swizzled schedule disagrees with the double-precision reference (%.3e)\n", D, worst_ref);
        fail = 1;
    }
    printf("  D = %-3d  placement spread %.3e, worst vs double reference %.3e\n", D, worst_place, worst_ref);
    return fail;
}

int main(void) {
    const int heads[] = { 64, 128, 256 };
    int fail = 0;

    printf("test-fa-tile-v2: the re-scheduled tile kernel must deliver the same bytes to the same arithmetic\n");

    printf("  PART 1 -- the phase XOR swizzle, and what the shipping pad bought\n");
    for (int h : heads) fail |= part1(h);

    printf("  PART 2 -- a 128-bit chunk reads back the four logical half2 staged into it\n");
    for (int h : heads) fail |= part2(h);

    printf("  PART 3 -- the whole online softmax, both schedules, six KV placements\n");
    for (int h : heads) {
        printf("   head size %d\n", h);
        fail |= part3(h);
    }

    printf("%s\n", fail ? "FAIL" : "OK");
    return fail;
}
