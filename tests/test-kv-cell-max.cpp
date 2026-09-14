// PXA_KV_CELL_MAX_EXACT regression test.
//
// The attention window of a decode is sized by
//
//     n = min(size, max(pad, PAD(cell_max(cache), pad)))
//
// where cell_max is "one past the highest occupied cell". n is the width of the KQ mask AND of
// the K/V view, so a cell_max that is short of the live band does not make a slower graph -- it
// makes a graph in which those cells DO NOT EXIST. Every mask row for that sequence is then all
// -inf, and a fully-masked row is NaN out of both the flash-attention and the softmax path.
//
// The engine used to size the full cache with a SAMPLED cell_max that inspected only every
// pad-th cell. That is sound only while occupancy is a prefix of the cell array. A unified ring
// shared by several sequences breaks the premise: seq_rm(seq, 0, -1) -- the server's "forcing
// full prompt re-processing" branch -- empties a band BELOW the top and pulls `head` back into
// the hole, so the next find_slot lays a live band that can sit entirely between two sampled
// indices. Case A below is the measured one, cell for cell.
//
// This test keeps a verbatim copy of the sampled scan so the defect stays visible: every case
// asserts what the exact scan answers AND records what the sampled one answered. CPU only, no
// model, no backend. Exit 0 when the exact scan covers every live cell in every case.

#include "llama-context.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------------------------
// the two scans

// the shipped one (src/llama.cpp, llama_kv_cache_cell_max)
static uint32_t cell_max_exact(const std::vector<llama_kv_cell> & cells) {
    for (uint32_t i = (uint32_t) cells.size(); i > 0; --i) {
        const llama_kv_cell & cell = cells[i - 1];
        if (cell.pos >= 0 && !cell.is_empty()) {
            return i;
        }
    }
    return 0;
}

// verbatim copy of the scan this test retires (removed from src/llama.cpp on 2026-09-09)
static uint32_t cell_max_sampled(const std::vector<llama_kv_cell> & cells, uint32_t pad) {
    const uint32_t size = (uint32_t) cells.size();
    uint32_t last = size;
    for (uint32_t i = size; i > 0; i -= pad) {
        const llama_kv_cell & cell = cells[i - 1];
        if (cell.pos >= 0 && !cell.is_empty()) {
            for (uint32_t j = last; j > i; --j) {
                const llama_kv_cell & cell_j = cells[j - 1];
                if (cell_j.pos >= 0 && !cell_j.is_empty()) return j;
            }
            return i;
        }
        last = i;
    }
    return 0;
}

static uint32_t n_from_cell_max(uint32_t max_cell, uint32_t pad, uint32_t size) {
    const uint32_t padded = ((max_cell + pad - 1) / pad) * pad;
    const uint32_t n = padded > pad ? padded : pad;
    return n < size ? n : size;
}

// ---------------------------------------------------------------------------------------------
// helpers

static void occupy(std::vector<llama_kv_cell> & cells, uint32_t first, uint32_t n,
                   llama_seq_id seq, llama_pos pos0) {
    for (uint32_t i = 0; i < n; ++i) {
        cells[first + i].pos = pos0 + (llama_pos) i;
        cells[first + i].add_seq(seq);
    }
}

// the mask fill of llama_set_inputs, reduced to the one question that matters: how many keys are
// visible to a query row. A row with zero visible keys is the NaN.
static uint32_t min_row_occupancy(const std::vector<llama_kv_cell> & cells, uint32_t n_kv,
                                  llama_seq_id seq, llama_pos pos_first, uint32_t n_tokens) {
    uint32_t worst = UINT32_MAX;
    for (uint32_t j = 0; j < n_tokens; ++j) {
        const llama_pos pos = pos_first + (llama_pos) j;
        uint32_t occ = 0;
        for (uint32_t i = 0; i < n_kv && i < cells.size(); ++i) {
            if (cells[i].has_seq_id(seq) && cells[i].pos <= pos) {
                ++occ;
            }
        }
        if (occ < worst) worst = occ;
    }
    return worst;
}

static int n_fail = 0;

static void expect_covers(const char * name, const std::vector<llama_kv_cell> & cells, uint32_t pad) {
    const uint32_t size    = (uint32_t) cells.size();
    const uint32_t exact   = cell_max_exact(cells);
    const uint32_t sampled = cell_max_sampled(cells, pad);

    const uint32_t n_exact   = n_from_cell_max(exact,   pad, size);
    const uint32_t n_sampled = n_from_cell_max(sampled, pad, size);

    const bool ok_exact   = n_exact   >= exact;
    const bool ok_sampled = n_sampled >= exact;

    printf("%-28s size=%6u exact=%6u -> n=%6u %s | sampled=%6u -> n=%6u %s\n",
           name, size, exact, n_exact, ok_exact ? "covers" : "SHORT",
           sampled, n_sampled, ok_sampled ? "covers" : "SHORT (the defect)");

    if (!ok_exact) {
        fprintf(stderr, "FAIL %s: the exact scan left live cells outside the window\n", name);
        ++n_fail;
    }
}

int main() {
    const uint32_t pad = 256; // llama_kv_cache_get_padding() with flash attention on

    int n_sampled_short = 0;

    // -----------------------------------------------------------------------------------------
    // Case A -- the measured failure, cell for cell.
    //
    // Server log (hybrid re-entry, -np 2 --kv-unified, forced full re-processing of slot 1):
    //   PXA_KQ_MASK ntok=133 seq=1 pos0=0 n_kv=256 kv_n=256 head=274 used=278 size=8192
    //               min_occupancy=0 empty_rows=133
    //   EMPTY ROW row=0 seq=1 pos=0 : cells_of_seq=0 cells_of_others=144 free=112
    // seq 0 holds 144 cells from 0; seq 1's re-processed prompt was placed at 274..406.
    {
        std::vector<llama_kv_cell> cells(8192);
        occupy(cells,   0, 144, 0, 0);
        occupy(cells, 274, 133, 1, 0);
        expect_covers("A floating band (measured)", cells, pad);

        const uint32_t exact   = cell_max_exact(cells);
        const uint32_t sampled = cell_max_sampled(cells, pad);
        const uint32_t n_exact   = n_from_cell_max(exact,   pad, (uint32_t) cells.size());
        const uint32_t n_sampled = n_from_cell_max(sampled, pad, (uint32_t) cells.size());

        const uint32_t occ_exact   = min_row_occupancy(cells, n_exact,   1, 0, 133);
        const uint32_t occ_sampled = min_row_occupancy(cells, n_sampled, 1, 0, 133);
        printf("%-28s min mask-row occupancy: exact n_kv=%u -> %u | sampled n_kv=%u -> %u\n",
               "A", n_exact, occ_exact, n_sampled, occ_sampled);

        if (exact != 407)      { fprintf(stderr, "FAIL A: exact cell_max %u, expected 407\n", exact); ++n_fail; }
        if (occ_exact == 0)    { fprintf(stderr, "FAIL A: a mask row is fully masked with the exact window\n"); ++n_fail; }
        if (occ_sampled != 0)  { fprintf(stderr, "FAIL A: the sampled window no longer reproduces the defect -- "
                                                 "this test has stopped testing anything\n"); ++n_fail; }
        if (n_sampled < exact) ++n_sampled_short;
    }

    // -----------------------------------------------------------------------------------------
    // Case B -- the second unsoundness of the sampled scan: it lowers its upper bound past cells
    // no probe has looked at, so it can under-report even when a probe DOES hit.
    {
        std::vector<llama_kv_cell> cells(8192);
        occupy(cells, 8000, 1, 0, 0);   // above the lowered bound, never scanned
        occupy(cells, 7679, 1, 0, 1);   // the cell the probe at i=7680 hits
        expect_covers("B under-report above bound", cells, pad);
        if (cell_max_sampled(cells, pad) < cell_max_exact(cells)) ++n_sampled_short;
    }

    // -----------------------------------------------------------------------------------------
    // Case C -- the shape the sampled scan was written for: occupancy is a prefix of the array.
    // Both scans must agree, so the fix costs nothing where the old one was right.
    {
        for (uint32_t used : {1u, 255u, 256u, 257u, 1000u, 4096u, 8192u}) {
            std::vector<llama_kv_cell> cells(8192);
            occupy(cells, 0, used, 0, 0);
            const uint32_t exact   = cell_max_exact(cells);
            const uint32_t sampled = cell_max_sampled(cells, pad);
            const uint32_t n_e = n_from_cell_max(exact,   pad, 8192);
            const uint32_t n_s = n_from_cell_max(sampled, pad, 8192);
            if (n_e != n_s) {
                printf("%-28s used=%u: n differs (exact %u, sampled %u)\n", "C prefix", used, n_e, n_s);
                if (n_s < exact) ++n_sampled_short;
            }
            if (n_e < exact) { fprintf(stderr, "FAIL C: exact window short at used=%u\n", used); ++n_fail; }
        }
        printf("%-28s every prefix shape: exact window covers, and matches the sampled one\n", "C prefix");
    }

    // -----------------------------------------------------------------------------------------
    // Case D -- randomised bands. The exact window must cover the live cells in every draw.
    {
        std::mt19937 rng(20260909);
        int n_short = 0;
        for (int trial = 0; trial < 3000; ++trial) {
            std::vector<llama_kv_cell> cells(2048);
            const int n_bands = 1 + (int) (rng() % 4);
            for (int b = 0; b < n_bands; ++b) {
                const uint32_t len   = 1 + (uint32_t) (rng() % 200);
                const uint32_t first = (uint32_t) (rng() % (cells.size() - len));
                occupy(cells, first, len, (llama_seq_id) (rng() % 4), 0);
            }
            const uint32_t exact   = cell_max_exact(cells);
            const uint32_t sampled = cell_max_sampled(cells, pad);
            if (n_from_cell_max(exact, pad, (uint32_t) cells.size()) < exact) {
                fprintf(stderr, "FAIL D: exact window short on trial %d\n", trial);
                ++n_fail;
            }
            if (n_from_cell_max(sampled, pad, (uint32_t) cells.size()) < exact) ++n_short;
        }
        printf("%-28s 3000 random band layouts: the exact window always covers; "
               "the sampled one was short in %d\n", "D random", n_short);
        n_sampled_short += n_short;
    }

    // -----------------------------------------------------------------------------------------
    // Cost. The exact scan runs once per ubatch and walks down from the top of the cell array,
    // stopping at the first live cell -- so its cost is (size - cell_max), worst at a nearly
    // empty cache. 150016 cells is the largest seat the engine ships (Flash-Next at 150k).
    {
        const uint32_t size = 150016;
        std::vector<llama_kv_cell> cells(size);
        occupy(cells, 0, 1, 0, 0);   // worst case: one live cell at the bottom
        volatile uint32_t sink = 0;
        const int reps = 200;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) sink = cell_max_exact(cells);
        const auto t1 = std::chrono::steady_clock::now();
        (void) sink;
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
        printf("%-28s %u cells, worst case (1 live cell at the bottom): %.1f us per call\n",
               "cost", size, us);
    }

    printf("\nsampled scan was short in %d of the cases above; exact scan short in 0\n", n_sampled_short);
    if (n_sampled_short == 0) {
        fprintf(stderr, "FAIL: the retired sampled scan never under-reported -- the test no longer "
                        "exercises the defect it was written for\n");
        ++n_fail;
    }
    if (n_fail) {
        fprintf(stderr, "\ntest-kv-cell-max: %d FAILURE(S)\n", n_fail);
        return 1;
    }
    printf("test-kv-cell-max: OK\n");
    return 0;
}
