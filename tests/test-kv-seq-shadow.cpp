// PXA_KV_SEQ_SOA parity test: the llama_kv_cell inline seq shadow must answer has_seq_fast()
// exactly like the std::set answers has_seq_id() after any sequence of mutations, including
// whole-cell copies/resets/swaps (the defrag and checkpoint paths copy cells wholesale).
// CPU only, no model. Exit 0 on parity.
#include "llama-context.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int check(const llama_kv_cell & c, const char * what, long step) {
    int bad = 0;
    for (llama_seq_id s = -1; s < 9; ++s) {
        if (c.has_seq_id(s) != c.has_seq_fast(s)) {
            fprintf(stderr, "MISMATCH step=%ld op=%s seq=%d set=%d fast=%d shadow=%d n=%zu\n",
                    step, what, (int) s, (int) c.has_seq_id(s), (int) c.has_seq_fast(s), c.seq_shadow(), c.n_seq());
            bad++;
        }
    }
    if (c.is_empty() != (c.seq_shadow() == llama_kv_cell::SEQ_NONE)) { fprintf(stderr, "MISMATCH empty step=%ld\n", step); bad++; }
    if (c.n_seq() == 1 && c.seq_shadow() != *c.seqs().begin()) { fprintf(stderr, "MISMATCH single step=%ld\n", step); bad++; }
    if (c.n_seq() >= 2 && c.seq_shadow() != llama_kv_cell::SEQ_MULTI) { fprintf(stderr, "MISMATCH multi step=%ld\n", step); bad++; }
    return bad;
}

int main(int argc, char ** argv) {
    const long n_steps = argc > 1 ? atol(argv[1]) : 2000000;
    std::mt19937 rng(12345);
    std::vector<llama_kv_cell> cells(64);
    long bad = 0, n_multi = 0, n_single = 0, n_empty = 0;

    for (long step = 0; step < n_steps; ++step) {
        llama_kv_cell & c = cells[rng() % cells.size()];
        const llama_seq_id s = (llama_seq_id) (rng() % 8);
        const char * what = "?";
        switch (rng() % 10) {
            case 0: case 1: case 2: c.add_seq(s);   what = "add";   break;
            case 3: case 4:         c.erase_seq(s); what = "erase"; break;
            case 5:                 c.clear_seq();  what = "clear"; break;
            case 6:                 c = cells[rng() % cells.size()]; what = "copy";  break;   // defrag-style whole-cell move
            case 7:                 c = llama_kv_cell();            what = "reset"; break;   // `cell1 = llama_kv_cell()`
            case 8:                 std::swap(c, cells[rng() % cells.size()]); what = "swap"; break;
            case 9: {                                                                       // checkpoint snapshot round-trip
                std::vector<llama_kv_cell> snap = cells;
                cells[rng() % cells.size()].add_seq(s);
                cells = snap;
                what = "snapshot";
                break;
            }
        }
        bad += check(c, what, step);
        if (c.n_seq() >= 2) n_multi++; else if (c.n_seq() == 1) n_single++; else n_empty++;
        if (bad > 20) break;
    }
    // exhaustive check at the end over every cell
    for (const auto & c : cells) bad += check(c, "final", -1);

    printf("test-kv-seq-shadow: steps=%ld sizeof(llama_kv_cell)=%zu observed empty=%ld single=%ld multi=%ld mismatches=%ld -> %s\n",
           n_steps, sizeof(llama_kv_cell), n_empty, n_single, n_multi, bad, bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
