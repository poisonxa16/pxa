// PXA_FLASHNEXT_V100_SET MTP depth rule (examples/server/pxa-fn-v100-mtp.h). Pure, no model, no GPU.
#include "pxa-fn-v100-mtp.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static pxa_fn_v100_mtp_in board(int n_max = 3, float p_min = 0.75f) {
    pxa_fn_v100_mtp_in in;
    in.qwen4exp = true; in.has_nextn = true; in.volta_only = true; in.config_level = 2;
    in.n_gpu_layers = 99; in.n_parallel = 1; in.has_mtp_stage = true;
    in.mtp_n_max = n_max; in.mtp_p_min = p_min;
    return in;
}

int main() {
    // the measured rows: board n_max 3 and the preset's n_max 4, p_min 0.75 -> ceiling 5 + DEEP 3:0.5:0.8 (= the d1 dp arm)
    for (int n : {3, 4}) {
        const auto o = pxa_fn_v100_mtp_rule(board(n));
        CHECK(o.on); CHECK(o.n_max == 5); CHECK(o.deep && strcmp(o.deep, "3:0.5:0.8") == 0);
    }
    // the set's own off switch; any other value leaves it on
    { auto in = board(); in.set_env = "0"; CHECK(!pxa_fn_v100_mtp_rule(in).on); }
    { auto in = board(); in.set_env = "1"; CHECK(pxa_fn_v100_mtp_rule(in).on); }
    { auto in = board(); in.set_env = "";  CHECK(pxa_fn_v100_mtp_rule(in).on); }
    // an explicit PXA_MTP_DEEP (off or any value) is the user's: no ceiling raise either
    { auto in = board(); in.deep_env = "0";         CHECK(!pxa_fn_v100_mtp_rule(in).on); }
    { auto in = board(); in.deep_env = "2:0.6:0.9"; CHECK(!pxa_fn_v100_mtp_rule(in).on); }
    // other layouts are unchanged
    { auto in = board(); in.qwen4exp = false;    CHECK(!pxa_fn_v100_mtp_rule(in).on); }
    { auto in = board(); in.volta_only = false;  CHECK(!pxa_fn_v100_mtp_rule(in).on); }
    { auto in = board(); in.config_level = 1;    CHECK(!pxa_fn_v100_mtp_rule(in).on); }
    { auto in = board(); in.config_level = 0;    CHECK(!pxa_fn_v100_mtp_rule(in).on); }
    { auto in = board(); in.n_gpu_layers = 0;    CHECK(!pxa_fn_v100_mtp_rule(in).on); }
    { auto in = board(); in.has_nextn = false;   CHECK(!pxa_fn_v100_mtp_rule(in).on); }
    { auto in = board(); in.has_mtp_stage = false; CHECK(!pxa_fn_v100_mtp_rule(in).on); }
    // unmeasured shapes stand down: several slots, a steered depth, other rows
    { auto in = board(); in.n_parallel = 2;      CHECK(!pxa_fn_v100_mtp_rule(in).on); }
    { auto in = board(); in.steered = true;      CHECK(!pxa_fn_v100_mtp_rule(in).on); }
    CHECK(!pxa_fn_v100_mtp_rule(board(1)).on);
    CHECK(!pxa_fn_v100_mtp_rule(board(2)).on);
    CHECK(!pxa_fn_v100_mtp_rule(board(5)).on);
    CHECK(!pxa_fn_v100_mtp_rule(board(16)).on);
    CHECK(!pxa_fn_v100_mtp_rule(board(3, 0.0f)).on);
    CHECK(!pxa_fn_v100_mtp_rule(board(3, 0.5f)).on);
    CHECK(!pxa_fn_v100_mtp_rule(board(3, 0.85f)).on);
    CHECK(pxa_fn_v100_mtp_rule(board(3, 0.751f)).on);
    if (g_fail) {
        fprintf(stderr, "test-pxa-fn-v100-mtp: %d failure(s)\n", g_fail);
        return 1;
    }
    printf("test-pxa-fn-v100-mtp: OK\n");
    return 0;
}
