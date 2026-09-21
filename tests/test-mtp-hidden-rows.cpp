// PXA_MTP_HIDDEN_BY_BATCH_ROW_v1: which row of the target's embedding buffer a caller's index
// names, on the CPU, with no model, no weights, no context and no GPU.
//
// The original design said "store the target hidden
// rows densely by batch row, so the draft never depends on the output-id pre-pass". Reading the
// decode path settles the first half: we ALREADY store them that way. For an MTP target context at
// MTP_OP_NONE, llama_decode reserves n_outputs_embd = n_tokens_all and the extraction writes one
// row per raw batch token, in batch order -- not one row per output. That is the same shape
// mainline llama.cpp keeps h_nextn in.
//
// What was missing was the CONTRACT, and that is what this covers. The two readers were named and
// documented as taking OUTPUT indices, while every caller passed a BATCH ROW:
//
//   examples/server/server-context.cpp   slot.i_batch_dft entries (verify batch rows)
//   examples/server/server-context.cpp   slot.i_batch - i        (the sampled row)
//   examples/server/server-context.cpp   {0 .. n-1}              (a checkpoint re-decode's rows)
//
// The two coincide only while every row of the batch asks for logits AND the embedding reserve
// covered the whole batch. Neither holds always: PXA_MTP_LAZY_WARMUP_v1 clamps the reserve to
// n_outputs on any batch over 64 tokens, which is any step where one slot prefills while another
// verifies. Where they diverged the old readers did not fail -- they returned a row belonging to a
// different token, silently, which is precisely the "fragility class" P7 names.
//
// llama_spec_hidden_row_for_batch_row() is the rule that replaces the coincidence, and it is a pure
// function of three integers so its whole domain can be enumerated:
//
//   batch_row >= 0   a raw batch row. Answered ONLY while the buffer is batch-dense.
//   batch_row <  0   counts back from the end of the embedding buffer, unchanged -- that is a
//                    genuine output-space sentinel (common_speculative_ensure_sequence_hidden
//                    passes -1 for "the last row produced").
//
// The last case checks the invariant that makes the whole thing safe: whenever the rule answers a
// non-negative request, the row it names is the row that token's own hidden was written to.

#include "llama-spec-features.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <vector>

static int          g_fail = 0;
static const char * g_case = "";

static void begin(const char * name) { g_case = name; printf("  %s\n", name); fflush(stdout); }

static void fail(const char * fmt, ...) {
    fprintf(stderr, "  FAIL [%s]: ", g_case);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    ++g_fail;
}

#define CHECK(cond, ...) do { if (!(cond)) { fail(__VA_ARGS__); } } while (0)

// The two shapes llama_decode leaves the embedding buffer in, as the decode path computes them.
//
//   DENSE   an MTP target context at MTP_OP_NONE with the reserve intact:
//           n_outputs_embd = n_tokens_all, n_embd_rows_batch_dense = n_tokens_all,
//           row i holds batch token i.
//   SPARSE  everything else -- a non-MTP context, the companion at any MTP_OP, or the
//           PXA_MTP_LAZY_WARMUP_v1 clamp: n_outputs_embd = n_outputs,
//           n_embd_rows_batch_dense = 0, and row j holds the j-th row that asked for an output
//           (or, under the clamp, the j-th of the ubatch's trailing rows) -- either way NOT
//           batch token j.
struct buffer_shape {
    int32_t n_rows_batch_dense;
    int32_t n_outputs_embd;
};

static buffer_shape dense(int32_t n_tokens)  { return { n_tokens, n_tokens }; }
static buffer_shape sparse(int32_t n_outputs) { return { 0, n_outputs }; }

static int32_t resolve(const buffer_shape & b, int32_t batch_row) {
    return llama_spec_hidden_row_for_batch_row(batch_row, b.n_rows_batch_dense, b.n_outputs_embd);
}

// ---------------------------------------------------------------------------------------------
static void case_dense() {
    begin("case 1: a batch-dense buffer answers every batch row with its own row");

    for (int32_t n : { 1, 2, 5, 64, 65, 512 }) {
        const buffer_shape b = dense(n);
        for (int32_t i = 0; i < n; ++i) {
            CHECK(resolve(b, i) == i, "n=%d: batch row %d resolved to %d, want %d",
                  n, i, resolve(b, i), i);
        }
        CHECK(resolve(b, n)     == -1, "n=%d: a batch row past the end must be refused", n);
        CHECK(resolve(b, n + 7) == -1, "n=%d: a batch row far past the end must be refused", n);
    }
}

// ---------------------------------------------------------------------------------------------
static void case_sparse_refuses() {
    begin("case 2: an output-indexed buffer refuses batch rows instead of guessing");

    // THE ONE THAT MATTERS. Under the lazy-warmup clamp a 200-token batch with 2 outputs leaves
    // 2 rows, and they are the ubatch's LAST two -- not batch rows 0 and 1. The old reader answered
    // a request for batch row 0 with row 0 of that buffer, i.e. another token's hidden state, and
    // reported success.
    const buffer_shape clamped = sparse(2);
    CHECK(resolve(clamped, 0) == -1, "batch row 0 must be refused on an output-indexed buffer");
    CHECK(resolve(clamped, 1) == -1, "batch row 1 must be refused on an output-indexed buffer");
    CHECK(resolve(clamped, 99) == -1, "batch row 99 must be refused on an output-indexed buffer");

    for (int32_t n_out : { 1, 2, 8 }) {
        const buffer_shape b = sparse(n_out);
        for (int32_t i = 0; i < 16; ++i) {
            CHECK(resolve(b, i) == -1,
                  "n_outputs=%d: batch row %d must be refused, got %d", n_out, i, resolve(b, i));
        }
    }
}

// ---------------------------------------------------------------------------------------------
static void case_negative_sentinel() {
    begin("case 3: a negative index keeps its old meaning on BOTH shapes");

    // common_speculative_ensure_sequence_hidden passes -1 for "the last row produced". That is an
    // output-space request and must keep working when the buffer is output-indexed, which is
    // exactly when the sequence is most likely to have no hidden state yet (a long prompt).
    for (const buffer_shape b : { dense(5), sparse(5) }) {
        CHECK(resolve(b, -1) == 4, "-1 must name the last row (%d)", resolve(b, -1));
        CHECK(resolve(b, -5) == 0, "-5 of 5 rows must name row 0 (%d)", resolve(b, -5));
        CHECK(resolve(b, -6) == -1, "-6 of 5 rows is out of range and must be refused");
        CHECK(resolve(b, -99) == -1, "a far negative index must be refused");
    }
}

// ---------------------------------------------------------------------------------------------
static void case_degenerate() {
    begin("case 4: an empty or unwritten buffer answers nothing");

    const buffer_shape empty = { 0, 0 };
    CHECK(resolve(empty, 0)  == -1, "an empty buffer must refuse row 0");
    CHECK(resolve(empty, -1) == -1, "an empty buffer must refuse the last-row sentinel");

    // A claim of dense rows that the buffer cannot back is a contradiction; it must not be trusted.
    const buffer_shape lying = { 8, 2 };
    for (int32_t i = 0; i < 8; ++i) {
        CHECK(resolve(lying, i) == (i < 2 ? i : -1),
              "rows beyond n_outputs_embd must be refused even when claimed dense (row %d -> %d)",
              i, resolve(lying, i));
    }
}

// ---------------------------------------------------------------------------------------------
static void case_invariant() {
    begin("case 5: whenever the rule answers, the row it names holds that token's hidden");

    // Model the two fills the decode path performs and check the answer against the fill, rather
    // than against the rule restated. `written[j]` is the batch row whose hidden landed in row j.
    for (int32_t n_tokens : { 1, 3, 17, 200 }) {
        for (int32_t stride : { 1, 2, 7 }) {                 // every stride-th row asks for output
            std::vector<int32_t> logits_rows;
            for (int32_t i = 0; i < n_tokens; ++i) {
                if ((i + 1) % stride == 0 || i == n_tokens - 1) {
                    logits_rows.push_back(i);
                }
            }
            const int32_t n_outputs = (int32_t) logits_rows.size();

            // DENSE fill: row i holds batch token i.
            {
                std::vector<int32_t> written(n_tokens);
                for (int32_t i = 0; i < n_tokens; ++i) written[i] = i;
                const buffer_shape b = dense(n_tokens);
                for (int32_t i = 0; i < n_tokens; ++i) {
                    const int32_t row = resolve(b, i);
                    CHECK(row >= 0 && written[row] == i,
                          "dense n=%d: batch row %d resolved to %d which holds token %d",
                          n_tokens, i, row, row >= 0 ? written[row] : -1);
                }
            }

            // SPARSE fill: row j holds the j-th output row's token. Any answer the rule gives for a
            // non-negative request would have to name a row holding that same token -- and it gives
            // none, which is the only correct behaviour when the rows are not batch-addressable.
            {
                std::vector<int32_t> written = logits_rows;
                const buffer_shape b = sparse(n_outputs);
                for (int32_t i = 0; i < n_tokens; ++i) {
                    const int32_t row = resolve(b, i);
                    if (row >= 0) {
                        CHECK(written[row] == i,
                              "sparse n=%d stride=%d: batch row %d resolved to %d which holds token %d",
                              n_tokens, stride, i, row, written[row]);
                    }
                }
                // and the sentinel still names the last row written
                CHECK(resolve(b, -1) == n_outputs - 1,
                      "sparse n=%d stride=%d: -1 must name the last written row", n_tokens, stride);
            }
        }
    }
}

int main() {
    printf("test-mtp-hidden-rows: PXA_MTP_HIDDEN_BY_BATCH_ROW_v1 -- addressing the target's MTP hidden rows\n");

    case_dense();
    case_sparse_refuses();
    case_negative_sentinel();
    case_degenerate();
    case_invariant();

    if (g_fail) {
        printf("FAILED (%d)\n", g_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
