// PXA_XCACHE_OT unit test (glm-mc-2-xcache-multidevice): the per-layer hot/cold split under a user -ot placement and the
// counts reader it plans from (src/llama-pxa-xcache-ot.h). CPU only, no model.
#include "../src/llama-pxa-xcache-ot.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    // 1) split: top-n by count, ties by lower id, both lists ascending, partition of all experts
    {
        std::vector<std::vector<double>> c = { {}, {5, 1, 9, 1, 0, 7, 2, 3}, {1, 1, 1, 1, 1, 1, 1, 1} };
        auto s = pxa_xc_ot_split(c, 3);
        CHECK(s.size() == 3);
        CHECK(s[0].hot.empty() && s[0].cold.empty());                   // no counts: unsplit
        CHECK((s[1].hot  == std::vector<int32_t>{0, 2, 5}));
        CHECK((s[1].cold == std::vector<int32_t>{1, 3, 4, 6, 7}));
        CHECK((s[2].hot  == std::vector<int32_t>{0, 1, 2}));             // flat routing (GLM-like): lowest ids
        auto all = pxa_xc_ot_split(c, 8);
        CHECK(all[1].hot.empty() && all[1].cold.empty());               // n_hot >= n_expert: nothing cold, unsplit
        auto z = pxa_xc_ot_split(c, 0);
        CHECK(z[1].hot.empty() && z[1].cold.size() == 8);               // n_hot 0: every expert cold
        auto r = pxa_xc_ot_split(c, 3, 2, 2);
        CHECK(r[1].cold.empty() && r[2].cold.size() == 5);              // layer range
    }
    // 2) reader: ffn_down rows win over up/gate, header skipped, out-of-range refused
    {
#ifdef _WIN32
        // no /tmp on Windows: fopen would return null and the CRT aborts on fprintf(nullptr)
        const std::string tmp_path = (std::filesystem::temp_directory_path() / "test-pxa-xcache-ot.csv").string();
        const char * p = tmp_path.c_str();
#else
        const char * p = "/tmp/test-pxa-xcache-ot.csv";
#endif
        FILE * f = fopen(p, "w");
        fprintf(f, "tensor,expert,count\n");
        for (int e = 0; e < 4; ++e) fprintf(f, "blk.1.ffn_up_exps.weight,%d,%d\n", e, 100 - e);
        for (int e = 0; e < 4; ++e) fprintf(f, "blk.1.ffn_down_exps.weight,%d,%d\n", e, e*10);
        for (int e = 0; e < 4; ++e) fprintf(f, "blk.2.ffn_gate_exps.weight,%d,%d\n", e, 4 - e);
        fclose(f);
        std::vector<std::vector<double>> c; std::string err;
        CHECK(pxa_xc_ot_read_counts(p, 3, 4, c, err));
        CHECK(c[0].empty());
        CHECK((c[1] == std::vector<double>{0, 10, 20, 30}));
        CHECK((c[2] == std::vector<double>{4, 3, 2, 1}));
        auto s = pxa_xc_ot_split(c, 2);
        CHECK((s[1].hot == std::vector<int32_t>{2, 3}) && (s[2].hot == std::vector<int32_t>{0, 1}));
        f = fopen(p, "w"); fprintf(f, "blk.9.ffn_down_exps.weight,0,1\n"); fclose(f);
        CHECK(!pxa_xc_ot_read_counts(p, 3, 4, c, err));
        f = fopen(p, "w"); fprintf(f, "blk.1.ffn_down_exps.weight,7,1\n"); fclose(f);
        CHECK(!pxa_xc_ot_read_counts(p, 3, 4, c, err));
        CHECK(!pxa_xc_ot_read_counts("/nonexistent/x.csv", 3, 4, c, err));
        remove(p);
    }
    if (fails) { fprintf(stderr, "test-pxa-xcache-ot: %d FAILED\n", fails); return 1; }
    printf("test-pxa-xcache-ot: OK\n");
    return 0;
}
