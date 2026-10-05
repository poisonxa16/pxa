// test-spec-rollback-cost.cpp -- recurrent-state rollback on a rejected draft: per-step CHECKPOINT vs REPLAY.
//
// A hybrid (Gated-DeltaNet) model cannot rewind its recurrent state by dropping KV rows: after a verify of k+1 tokens of which
// `a` drafts were accepted, the state must be the one after token a, not after token k. Two exact ways to get there:
//
//   CHECKPOINT  the verify writes the state after every step into its own slot (the engine's per-step snapshots); a rejection
//               copies slot a back into the live state (the engine batches the 96 segments into one launch per device).
//   REPLAY      the verify runs on a scratch copy; a rejection re-runs the recurrence for the a+1 accepted tokens from the
//               state saved before the verify, from the per-token inputs (k, v, gate, beta) that the verify kept.
//
// Both are exact by construction. What differs is what a rejected round pays, on the host and on the card:
//   * bytes: checkpoint moves S (the restore copy); replay moves S to save the base and S to rebuild, plus the inputs;
//   * work: replay repeats (a+1) steps of the recurrence per layer; checkpoint repeats nothing;
//   * launches: replay needs a kernel per layer (48 on the 27B) after the verify; the checkpoint restore is one multi-segment
//     copy per device. On a host-submit-bound rig (the P100 / V100 seats: ~1340 launches a token) launches are the cost.
//   * memory: checkpoint holds k+1 states (VRAM scales with the draft depth: the reason a deep n-gram chain is budgeted);
//     replay holds one state plus k+1 rows of inputs. This is replay's only win.
// This test measures the host side of the first three on the real per-layer shape of the 27B, checks BOTH are bit-exact against
// a straight-through run, and prints the cost table that the engine's choice (checkpoint, ledger p100mtp-replay-rollback) rests on.
// It is a CPU proxy for the recurrence arithmetic; the GPU-side evidence is the nvprof census recorded in that ledger row.
//
// Usage: test-spec-rollback-cost   (exit 0 = both strategies exact and the table printed)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; printf("  FAIL %s:%d  %s  ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } } while (0)

// Qwen3.8-27B: 48 value heads, head_k = head_v = 128, 48 recurrent layers -> state 48 x 128 x 128 floats = 3.1 MB a layer.
// Scaled for CPU wall time: fewer layers, the real per-layer shape (the per-layer cost is what scales).
static const int H = 48, DK = 128, DV = 128, LAYERS = 8;
static const size_t S = (size_t) H * DK * DV;

struct StepIn { std::vector<float> k, v; float g, beta; };   // per-token inputs of one head (shared across heads for the proxy)

static uint32_t g_rng = 7u;
static float rnd() { g_rng = g_rng * 1664525u + 1013904223u; return ((float) ((g_rng >> 9) & 0xFFFF) / 32768.0f - 1.0f); }

// the gated delta rule for one token, in place: S <- g * S; r = S k; S += beta * (v - r) k^T   (k normalised by the caller)
static void step(float * st, const StepIn & in) {
    for (int h = 0; h < H; ++h) {
        float * s = st + (size_t) h * DK * DV;
        for (int j = 0; j < DV; ++j) {
            float r = 0;
            for (int i = 0; i < DK; ++i) { s[(size_t) i * DV + j] *= in.g; r += s[(size_t) i * DV + j] * in.k[(size_t) i]; }
            const float d = in.beta * (in.v[(size_t) j] - r);
            for (int i = 0; i < DK; ++i) s[(size_t) i * DV + j] += d * in.k[(size_t) i];
        }
    }
}

static double now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    const int K = 3;   // draft depth: verify width K+1
    std::vector<StepIn> ins((size_t) K + 1);
    for (auto & in : ins) {
        in.k.resize(DK); in.v.resize(DV);
        float n = 0; for (auto & x : in.k) { x = rnd(); n += x * x; }
        n = std::sqrt(n); for (auto & x : in.k) x /= n;
        for (auto & x : in.v) x = rnd();
        in.g = 0.9f + 0.05f * rnd(); in.beta = 0.5f + 0.3f * rnd();
    }
    std::vector<std::vector<float>> base((size_t) LAYERS, std::vector<float>(S));
    for (auto & b : base) for (auto & x : b) x = rnd() * 0.1f;

    // ---- the straight-through reference: the state after token a, a = 0..K, for every layer -------------------------------
    std::vector<std::vector<std::vector<float>>> ref((size_t) K + 1, std::vector<std::vector<float>>((size_t) LAYERS));
    {
        std::vector<std::vector<float>> run = base;
        for (int a = 0; a <= K; ++a) {
            for (int l = 0; l < LAYERS; ++l) { step(run[(size_t) l].data(), ins[(size_t) a]); ref[(size_t) a][(size_t) l] = run[(size_t) l]; }
        }
    }

    printf("per layer: state %.2f MB (%d heads x %d x %d floats), %d layers in this proxy, verify width %d\n",
           S * 4 / 1048576.0, H, DK, DV, LAYERS, K + 1);
    printf("  a = accepted drafts | checkpoint restore ms | replay ms | replay / checkpoint | both exact\n");

    const int REPS = 5;
    for (int a = 0; a <= K; ++a) {
        // CHECKPOINT: the per-step snapshots exist already (the verify wrote them); a rejection copies slot a into the live state.
        std::vector<std::vector<std::vector<float>>> snap((size_t) K + 1);
        {
            std::vector<std::vector<float>> run = base;
            for (int t = 0; t <= K; ++t) { for (int l = 0; l < LAYERS; ++l) step(run[(size_t) l].data(), ins[(size_t) t]); snap[(size_t) t] = run; }
        }
        std::vector<std::vector<float>> live((size_t) LAYERS, std::vector<float>(S));
        double t_ck = 1e30;
        for (int r = 0; r < REPS; ++r) {
            const double t0 = now();
            for (int l = 0; l < LAYERS; ++l) memcpy(live[(size_t) l].data(), snap[(size_t) a][(size_t) l].data(), S * sizeof(float));
            t_ck = std::min(t_ck, now() - t0);
        }
        bool ok_ck = true;
        for (int l = 0; l < LAYERS; ++l) ok_ck = ok_ck && memcmp(live[(size_t) l].data(), ref[(size_t) a][(size_t) l].data(), S * sizeof(float)) == 0;

        // REPLAY: restore the saved base, re-run the a+1 accepted tokens from the kept inputs.
        double t_rp = 1e30; bool ok_rp = true;
        for (int r = 0; r < REPS; ++r) {
            const double t0 = now();
            for (int l = 0; l < LAYERS; ++l) {
                memcpy(live[(size_t) l].data(), base[(size_t) l].data(), S * sizeof(float));
                for (int t = 0; t <= a; ++t) step(live[(size_t) l].data(), ins[(size_t) t]);
            }
            t_rp = std::min(t_rp, now() - t0);
        }
        for (int l = 0; l < LAYERS; ++l) ok_rp = ok_rp && memcmp(live[(size_t) l].data(), ref[(size_t) a][(size_t) l].data(), S * sizeof(float)) == 0;

        CHECK(ok_ck, "checkpoint restore at a=%d is not bit-exact", a);
        CHECK(ok_rp, "replay at a=%d is not bit-exact", a);
        printf("  %d                   | %20.3f | %9.3f | %19.1fx | %s\n", a, t_ck, t_rp, t_rp / std::max(t_ck, 1e-9), ok_ck && ok_rp ? "yes" : "NO");
        CHECK(t_rp >= t_ck, "replay cheaper than a checkpoint copy at a=%d (%.3f vs %.3f ms): the premise of the engine's choice moved", a, t_rp, t_ck);
    }

    // ---- the launch and memory ledger, from the model's real dimensions (per rejected round) -------------------------------
    const int layers_real = 48;
    printf("\nper rejected round on the 27B (%d recurrent layers), from the shapes above:\n", layers_real);
    printf("  checkpoint: 1 multi-segment copy launch per device (%d segments), moves %.0f MB, holds %d states = %.2f GB of VRAM at depth %d\n",
           2 * layers_real, 2.0 * S * 4 * layers_real / 1048576.0, K + 1, (double) (K + 1) * S * 4 * layers_real / 1073741824.0, K);
    printf("  replay:     %d kernel launches (one per layer) after the verify, moves %.0f MB twice over (save + rebuild), holds 1 state = %.2f GB + %d rows of inputs\n",
           layers_real, (double) S * 4 * layers_real / 1048576.0, (double) S * 4 * layers_real / 1073741824.0, K + 1);
    printf("  at ~1340 launches a token the host submit is the critical path, so 48 extra launches cost more than the copy they replace;\n");
    printf("  replay buys VRAM only (%.2f GB at depth %d, growing linearly with the depth: %.2f GB at depth 64).\n",
           (double) K * S * 4 * layers_real / 1073741824.0, K, (double) 64 * S * 4 * layers_real / 1073741824.0);

    printf(g_fail ? "FAIL: %d\n" : "PASS test-spec-rollback-cost\n", g_fail);
    return g_fail ? 1 : 0;
}
