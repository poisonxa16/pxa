// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
// test-pxqn-ext.cpp -- the PXQN library's role tables (ggml-pxqn-ext.h), seen from the engine.
//   (no flag)      every case: an unknown role, a wrong version and a NULL argument answer NULL / -1
//   --want-gemv    the "gemv" table is handed out (library present, CUDA build): version and size check, and an
//                  argument struct shorter than the library's is declined (-1)
//   --want-none    no role table at all (no library, PXA_PXQN_DISABLE=1, or the library hands none out): the getter is NULL and a
//                  launch through the engine wrapper is declined (-1)
//   --want-place   the "place" table is handed out: a short argument struct, an unknown site and an unknown flag bit answer -1,
//                  and a pick is always -1 or a device index below n_device
//   --want-sched   the "sched" table is handed out; answers are masks of known options (PXA_MTP_ASYNC_INPUTS decides)
//   --want-chain   the "chain" table is handed out: a short argument struct and an unknown flag bit answer -1; with its
//                  switches unset the library keeps the engine's rule (-1); every answer is -1, 0 or 1
//   (every case)   role "upload": a NULL or short argument struct and an unknown site answer 0; with PXA_KQ_MASK_TRIM_W unset every
//                  KQ mask site answers 0 (the engine's default upload)
//   --want-fa      the "fa" table is handed out: a short argument struct and an unknown flag bit answer -1; a head-256
//                  node wider than 8 columns on sm_61 is named the tile-f32 kernel, every other shape keeps the engine's
//                  rule (-1); without the table every answer is -1
#include "ggml-pxqn-ext.h"
#include "ggml-pxqn-ext-host.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

int main(int argc, char ** argv) {
    bool want_gemv = false, want_none = false, want_place = false, want_chain = false, want_sched = false, want_fa = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--want-gemv")) want_gemv = true;
        else if (!strcmp(argv[i], "--want-none")) want_none = true;
        else if (!strcmp(argv[i], "--want-place")) want_place = true;
        else if (!strcmp(argv[i], "--want-chain")) want_chain = true;
        else if (!strcmp(argv[i], "--want-sched")) want_sched = true;
        else if (!strcmp(argv[i], "--want-fa")) want_fa = true;
        else { fprintf(stderr, "usage: %s [--want-gemv | --want-none | --want-place | --want-chain | --want-sched | --want-fa]\n", argv[0]); return 2; }
    }
    CHECK(ggml_pxqn_ext_table("no-such-role", 1, (uint32_t) sizeof(struct ggml_pxqn_ext_head)) == nullptr);
    CHECK(ggml_pxqn_ext_table(nullptr, 1, (uint32_t) sizeof(struct ggml_pxqn_ext_head)) == nullptr);
    CHECK(ggml_pxqn_ext_table(GGML_PXQN_GEMV_ROLE, GGML_PXQN_GEMV_VERSION + 1000, (uint32_t) sizeof(struct ggml_pxqn_gemv_api)) == nullptr);
    CHECK(ggml_pxqn_gemv_launch(nullptr, nullptr) == -1);
    CHECK(ggml_pxqn_place_pick(nullptr) == -1);
    CHECK(ggml_pxqn_ext_table(GGML_PXQN_PLACE_ROLE, GGML_PXQN_PLACE_VERSION + 1000, (uint32_t) sizeof(struct ggml_pxqn_place_api)) == nullptr);

    const struct ggml_pxqn_gemv_api * g = ggml_pxqn_gemv_api_get();
    if (want_gemv) {
        CHECK(g != nullptr);
        if (g) {
            CHECK(g->version == GGML_PXQN_GEMV_VERSION);
            CHECK(g->size >= sizeof(struct ggml_pxqn_gemv_api));
            CHECK(g->launch != nullptr);
            struct ggml_pxqn_gemv_args a;
            memset(&a, 0, sizeof(a));
            a.size = 8;   // shorter than any version of the struct: never launched
            CHECK(ggml_pxqn_gemv_launch(&a, nullptr) == -1);
        }
    }
    const struct ggml_pxqn_place_api * pl = ggml_pxqn_place_api_get();
    {   // every answer, table or not: -1, a device index below n_device, or NO_JOIN when the caller offered PARTS_OK
        const char * hd = getenv("PXA_MTP_HEAD_DIRECT");
        const bool hd_on  = hd && *hd && atoi(hd) != 0;   // explicitly on: the answer is pinned below
        const bool hd_may = !(hd && *hd && atoi(hd) == 0); // unset: the library default may be on (sm_60 since 11-02)
        for (int nd = 1; nd <= 4; ++nd) for (int np = 1; np <= 4; ++np) for (uint32_t fl = 0; fl < 8; ++fl) {
            struct ggml_pxqn_place_args a = { (uint32_t) sizeof(struct ggml_pxqn_place_args), GGML_PXQN_PLACE_JOIN, np, nd, fl, 0 };
            const int ix = ggml_pxqn_place_pick(&a);
            CHECK(ix == -1 || (ix >= 0 && ix < nd) || (ix == GGML_PXQN_PLACE_NO_JOIN && (fl & GGML_PXQN_PLACE_PARTS_OK)));
            if (fl & 4u) CHECK(ix == -1);   // a flag bit the library does not know
            if (ix == GGML_PXQN_PLACE_NO_JOIN) CHECK(hd_may && (fl & GGML_PXQN_PLACE_DRAFT) && np == 2 && nd == 2);
            if (pl && hd_on && fl == (uint32_t) (GGML_PXQN_PLACE_DRAFT | GGML_PXQN_PLACE_PARTS_OK) && np == 2 && nd == 2) {
                CHECK(ix == GGML_PXQN_PLACE_NO_JOIN);
            }
        }
    }
    if (want_place) {
        CHECK(pl != nullptr);
        if (pl) {
            CHECK(pl->version == GGML_PXQN_PLACE_VERSION);
            CHECK(pl->size >= sizeof(struct ggml_pxqn_place_api));
            struct ggml_pxqn_place_args a = { 8, GGML_PXQN_PLACE_JOIN, 2, 2, GGML_PXQN_PLACE_DRAFT, 0 };
            CHECK(ggml_pxqn_place_pick(&a) == -1);   // shorter than the struct: never asked
            a.size = (uint32_t) sizeof(a); a.site = 99;
            CHECK(ggml_pxqn_place_pick(&a) == -1);   // unknown site
        }
    }
    const struct ggml_pxqn_chain_api * ch = ggml_pxqn_chain_api_get();
    CHECK(ggml_pxqn_chain_next(nullptr) == -1);
    for (int row = 1; row <= 4; ++row) for (int k = 0; k <= 10; ++k) {
        const float p = 0.1f * (float) k;
        struct ggml_pxqn_chain_args a = { (uint32_t) sizeof(struct ggml_pxqn_chain_args), row, 4, row, 0.8f, p, p * p, 0u };
        const int r = ggml_pxqn_chain_next(&a);
        CHECK(r == -1 || r == 0 || r == 1);
        a.flags = 1u;
        CHECK(ggml_pxqn_chain_next(&a) == -1);   // a flag bit the library does not know
    }
    if (want_chain) {
        CHECK(ch != nullptr);
        if (ch) {
            CHECK(ch->version == GGML_PXQN_CHAIN_VERSION);
            CHECK(ch->size >= sizeof(struct ggml_pxqn_chain_api));
            struct ggml_pxqn_chain_args a = { 8, 1, 4, 1, 0.8f, 0.0f, 0.0f, 0u };
            CHECK(ggml_pxqn_chain_next(&a) == -1);   // shorter than the struct: never asked
            if (!getenv("PXA_MTP_PREGATE") && !getenv("PXA_MTP_PREGATE_LAST")) {
                a.size = (uint32_t) sizeof(a);
                CHECK(ggml_pxqn_chain_next(&a) == -1);   // switches unset: the engine's own rule
            }
        }
    }
    // role "sched": every answer is a mask of known options; an unknown site or flag bit, a short struct or fewer than two
    // devices never arm anything; with PXA_MTP_ASYNC_INPUTS=1 the draft companion of a 2+ device tensor split gets async inputs
    const struct ggml_pxqn_sched_api * sc = ggml_pxqn_sched_api_get();
    CHECK(ggml_pxqn_sched_pick(nullptr) == 0u);
    {
        const char * e = getenv("PXA_MTP_ASYNC_INPUTS");
        const bool env_on = e && *e && atoi(e) != 0;
        const char * g = getenv("PXA_MTP_GRAPH_REUSE");
        const bool keep_on = g && *g && atoi(g) == 1;
        for (int nd = 0; nd <= 4; ++nd) for (uint32_t fl = 0; fl < 4; ++fl) for (int site = 0; site <= 2; ++site) {
            const struct ggml_pxqn_sched_args a = { (uint32_t) sizeof(struct ggml_pxqn_sched_args), site, nd, fl, {0, 0} };
            const uint32_t r = ggml_pxqn_sched_pick(&a);
            CHECK((r & ~GGML_PXQN_SCHED_OPTS_KNOWN) == 0u);
            const bool here = sc && site == GGML_PXQN_SCHED_DRAFT && fl == GGML_PXQN_SCHED_TSPLIT && nd >= 2;
            if (env_on)  CHECK(((r & GGML_PXQN_SCHED_ASYNC_INPUTS) != 0) == here);
            CHECK(((r & GGML_PXQN_SCHED_KEEP_GRAPH) != 0) == (here && keep_on));
        }
        const struct ggml_pxqn_sched_args s8 = { 8, GGML_PXQN_SCHED_DRAFT, 2, GGML_PXQN_SCHED_TSPLIT, {0, 0} };
        CHECK(ggml_pxqn_sched_pick(&s8) == 0u);   // shorter than the struct: never asked
    }
    if (want_sched) {
        CHECK(sc != nullptr);
        if (sc) {
            CHECK(sc->version == GGML_PXQN_SCHED_VERSION);
            CHECK(sc->size >= sizeof(struct ggml_pxqn_sched_api));
        }
    }
    // role "upload": answers are >= 0, and 0 (the engine's default) without a table, for an unknown site or a short struct
    const struct ggml_pxqn_upload_api * up = ggml_pxqn_upload_api_get();
    {
        CHECK(ggml_pxqn_upload_mode(nullptr) == 0);
        const struct ggml_pxqn_upload_args s4 = { 4, GGML_PXQN_UPLOAD_KQ_MASK, {0, 0, 0, 0} };
        CHECK(ggml_pxqn_upload_mode(&s4) == 0);
        const struct ggml_pxqn_upload_args bad = { (uint32_t) sizeof(struct ggml_pxqn_upload_args), 99, {0, 0, 0, 0} };
        CHECK(ggml_pxqn_upload_mode(&bad) == 0);
        const struct ggml_pxqn_upload_args km = { (uint32_t) sizeof(struct ggml_pxqn_upload_args), GGML_PXQN_UPLOAD_KQ_MASK, {0, 0, 0, 0} };
        const int32_t r = ggml_pxqn_upload_mode(&km);
        CHECK(r >= 0);
        const char * tw = getenv("PXA_KQ_MASK_TRIM_W");
        if (!tw || !*tw) CHECK(r == 0);
        if (!tw || !*tw) {   // the KQ mask row / fill sites keep the engine's rule too
            for (int32_t live = 1; live <= 16; ++live) {
                for (int32_t mode = 0; mode <= 2; ++mode) {
                    const struct ggml_pxqn_upload_args kr = { (uint32_t) sizeof(struct ggml_pxqn_upload_args), GGML_PXQN_UPLOAD_KQ_MASK_ROWS, {live, mode, 0, 0} };
                    const struct ggml_pxqn_upload_args kf = { (uint32_t) sizeof(struct ggml_pxqn_upload_args), GGML_PXQN_UPLOAD_KQ_MASK_FILL, {live, mode, 0, 0} };
                    CHECK(ggml_pxqn_upload_mode(&kr) == 0);
                    CHECK(ggml_pxqn_upload_mode(&kf) == 0);
                }
            }
        }
        if (up) { CHECK(up->version == GGML_PXQN_UPLOAD_VERSION); CHECK(up->size >= sizeof(struct ggml_pxqn_upload_api)); }
    }
    // role "fa": every answer is -1 or a kernel of enum ggml_pxqn_fa_route; only a wide head-256 node on sm_61 is named one
    const struct ggml_pxqn_fa_api * fa = ggml_pxqn_fa_api_get();
    CHECK(ggml_pxqn_fa_pick(nullptr) == -1);
    CHECK(ggml_pxqn_ext_table(GGML_PXQN_FA_ROLE, GGML_PXQN_FA_VERSION + 1000, (uint32_t) sizeof(struct ggml_pxqn_fa_api)) == nullptr);
    {
        const int ccs[] = { 520, 600, 610, 620, 700, 750, 860 };
        const int ds[]  = { 64, 128, 256, 512 };
        const int64_t ws[] = { 1, 2, 8, 9, 16, 128, 512 };
        for (int cc : ccs) for (int d : ds) for (int64_t w : ws) for (int prec = 0; prec <= 10; prec += 10) {
            struct ggml_pxqn_fa_args a;
            memset(&a, 0, sizeof(a));
            a.size = (uint32_t) sizeof(a); a.cc = cc; a.d_k = d; a.d_v = d; a.ncols = w; a.n_kv = 4096; a.type_k = 8; a.type_v = 8;
            a.prec = prec;
            const int r = ggml_pxqn_fa_pick(&a);
            const bool here = fa && cc == 610 && d == 256 && w > 8;
            CHECK(r == (here ? GGML_PXQN_FA_TILE_F32 : -1));
            a.flags = 1u;
            CHECK(ggml_pxqn_fa_pick(&a) == -1);   // a flag bit the library does not know
            a.flags = 0u; a.d_v = d == 256 ? 128 : 256;
            CHECK(ggml_pxqn_fa_pick(&a) == -1);   // K and V head sizes differ
        }
        struct ggml_pxqn_fa_args s8;
        memset(&s8, 0, sizeof(s8));
        s8.size = 8; s8.cc = 610; s8.d_k = 256; s8.d_v = 256; s8.ncols = 512;
        CHECK(ggml_pxqn_fa_pick(&s8) == -1);   // shorter than the struct: never asked
    }
    if (want_fa) {
        CHECK(fa != nullptr);
        if (fa) {
            CHECK(fa->version == GGML_PXQN_FA_VERSION);
            CHECK(fa->size >= sizeof(struct ggml_pxqn_fa_api));
        }
    }
    if (want_none) {
        CHECK(up == nullptr);
        CHECK(g == nullptr);
        CHECK(sc == nullptr);
        CHECK(pl == nullptr);
        CHECK(ch == nullptr);
        CHECK(fa == nullptr);
        struct ggml_pxqn_gemv_args a;
        memset(&a, 0, sizeof(a));
        a.size = (uint32_t) sizeof(a);
        CHECK(ggml_pxqn_gemv_launch(&a, nullptr) == -1);
    }
    printf("test-pxqn-ext: %s (gemv table %s, place table %s, chain table %s, sched table %s, upload table %s, fa table %s)\n", g_fail ? "FAIL" : "PASS",
           g ? "present" : "absent", pl ? "present" : "absent", ch ? "present" : "absent", sc ? "present" : "absent",
           up ? "present" : "absent", fa ? "present" : "absent");
    return g_fail ? 1 : 0;
}
