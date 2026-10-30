// test-fa-volta-mma-shapes -- host-only TABLE TEST of the vendored MMA flash-attention family
// (ggml/src/ggml-cuda/pxa/volta-mma/): every shape the sm_70 dispatcher can launch must be a variant
// that carries device code on the arch it runs on. CPU only, no card needed.
//
// Bug #22738 (2026-10-17): the 4.0-pre server aborted on a V100 with Nemotron-3-Nano-4B (head 128,
// 40 q-heads / 8 kv-heads = GQA 5) during ngram self-speculative decoding:
//   fattn-mma-ml.cuh:1841 ERROR: CUDA kernel flash_attn_ext_f16 has no device code compatible with
//   CUDA arch 700 (compiled for 600,610,700)
// A verify batch of 9..16 tokens passes the route's width gate (> 8); GQA 5 gives ncols2 = 1; the
// ncols1 ladder then picked its 16/ncols2 rung = 16 columns, and sm_70 compiles every variant below
// 32 columns out. Upstream gates that rung on Turing MMA, which Volta does not have.
//
// The dispatcher and the kernel guards both read pxa/volta-mma/fattn-mma-shapes.h, and so does this
// test: it walks every head family the route serves (128/128, 256/256, 512/512), GQA ratio 1..16 with
// and without the GQA packing, batch width 1..64, logit softcap on/off and the sparse variant where it
// exists, on sm_60, sm_61 and sm_70, and asserts
//   (1) routed => the launched (ncols1, ncols2) variant has device code on that arch;
//   (2) Pascal (sm_60 / sm_61) is never routed here (the kernel body is compiled out there);
//   (3) the head-128/256 ladder names a variant with code at EVERY width 1..64 on sm_70, not just the
//       routed widths (PXA_FA_MMA_VOLTA=2 once lifted the width gate and crashed the card at 1-4);
//   (4) the predicate is not vacuous: the 16-column sm_70 stub and the 64-column sm_75 stub are stubs.
// Exit 0 pass, 1 fail.

#include "../ggml/src/ggml-cuda/pxa/volta-mma/fattn-mma-shapes.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <map>
#include <string>

namespace S = pxa_fattn_mma_shapes;

static int g_fail = 0;
static int g_shown = 0;
static std::map<std::string, int> g_by_kind;

static void fail(const char * what, int arch, int D, int gqa, bool opt, int width, bool softcap, bool sparse,
                 int ncols1, int ncols2) {
    ++g_fail;
    ++g_by_kind[what];
    if (g_shown++ < 24 || strcmp(what, "bug-22738") == 0) {
        printf("  FAIL %s: sm_%d D=%d gqa=%d gqa_opt=%d width=%d softcap=%d sparse=%d -> ncols1=%d ncols2=%d (%d cols)\n",
               what, arch/10, D, gqa, (int) opt, width, (int) softcap, (int) sparse, ncols1, ncols2, ncols1*ncols2);
    }
}

struct pick { bool routed; int ncols1; int ncols2; };

// The route as fa-route.cu + fattn-volta-mma.cu take it, from the shared functions only.
static pick route(int cc, int D, int gqa, bool opt, int width) {
    pick p = { false, 0, 0 };
    if (!S::volta_route_arch(cc)) {
        return p;
    }
    if (D == 512) {
        // ggml_cuda_fattn_volta_mma_d512_supported: GQA a multiple of 8 and the GQA packing allowed;
        // any width (the lever decides whether the family is armed, not the width).
        if (gqa % S::volta_d512_ncols2() != 0 || !opt) {
            return p;
        }
        p.routed = true;
        p.ncols2 = S::volta_d512_ncols2();
        p.ncols1 = S::volta_d512_pick_ncols1(width);
        return p;
    }
    if (!S::volta_takes_width(width)) {
        return p;
    }
    p.routed = true;
    p.ncols2 = S::volta_pick_ncols2(gqa, opt);
    p.ncols1 = S::volta_pick_ncols1(width, p.ncols2);
    return p;
}

int main() {
    static const int archs[] = { 600, 610, 700 };
    static const int heads[] = { 128, 256, 512 };

    // (4) the predicate is real
    if (S::has_device_code(700, 128, 128, 16, 1, false, false)) { printf("  FAIL predicate: sm_70 16x1 should be a stub\n"); ++g_fail; }
    if (!S::has_device_code(700, 128, 128, 32, 1, false, false)) { printf("  FAIL predicate: sm_70 32x1 should have code\n"); ++g_fail; }
    if (S::has_device_code(750, 256, 256, 64, 1, false, false))  { printf("  FAIL predicate: sm_75 64x1 should be a stub\n"); ++g_fail; }
    if (S::has_device_code(600, 128, 128, 64, 1, false, false))  { printf("  FAIL predicate: sm_60 has no MMA body\n"); ++g_fail; }
    if (S::has_device_code(700, 512, 512, 1, 8, false, true))    { printf("  FAIL predicate: sm_70 1x8 sparse should be a stub\n"); ++g_fail; }

    long n_routed = 0, n_checked = 0;
    for (const int arch : archs) {
        for (const int D : heads) {
            for (int gqa = 1; gqa <= 16; ++gqa) {
                for (const bool opt : { false, true }) {
                    for (int width = 1; width <= 64; ++width) {
                        const pick p = route(arch, D, gqa, opt, width);
                        // (2)
                        if (p.routed && arch < 700) {
                            fail("pascal-routed", arch, D, gqa, opt, width, false, false, p.ncols1, p.ncols2);
                            continue;
                        }
                        // (3) the head-128/256 ladder at every width on sm_70
                        if (arch == 700 && D != 512) {
                            const int n2 = S::volta_pick_ncols2(gqa, opt);
                            const int n1 = S::volta_pick_ncols1(width, n2);
                            if (!S::has_device_code(arch, D, D, n1, n2, false, false)) {
                                fail("ladder-stub", arch, D, gqa, opt, width, false, false, n1, n2);
                            }
                        }
                        if (!p.routed) {
                            continue;
                        }
                        ++n_routed;
                        // (1)
                        for (const bool softcap : { false, true }) {
                            for (const bool sparse : { false, true }) {
                                if (sparse && !S::may_use_sparse(D, D, p.ncols1, p.ncols2)) {
                                    continue; // the launcher only builds the sparse kernel where it may exist
                                }
                                ++n_checked;
                                if (!S::has_device_code(arch, D, D, p.ncols1, p.ncols2, softcap, sparse)) {
                                    fail("routed-stub", arch, D, gqa, opt, width, softcap, sparse, p.ncols1, p.ncols2);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // The bug's own shape, named: Nemotron-3-Nano-4B head 128, GQA 5, verify widths 9..16 on a V100.
    for (int width = 9; width <= 16; ++width) {
        const pick p = route(700, 128, 5, true, width);
        const bool ok = p.routed && S::has_device_code(700, 128, 128, p.ncols1, p.ncols2, false, false);
        if (!ok) {
            fail("bug-22738", 700, 128, 5, true, width, false, false, p.ncols1, p.ncols2);
        } else if (width == 9 || width == 16) {
            printf("  ok bug-22738 shape: sm_70 D=128 gqa=5 width=%d -> ncols1=%d ncols2=%d\n", width, p.ncols1, p.ncols2);
        }
    }

    printf("test-fa-volta-mma-shapes: %ld routed shapes, %ld variant checks, %d failure(s)%s\n",
           n_routed, n_checked, g_fail, g_shown > 24 ? " (first 24 shown, plus every bug-22738 shape)" : "");
    for (const auto & kv : g_by_kind) {
        printf("  %-14s %d\n", kv.first.c_str(), kv.second);
    }
    printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
