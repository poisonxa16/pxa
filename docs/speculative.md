# Speculative Decoding

llama.cpp supports speculative decoding, a technique that can significantly accelerate token generation by predicting multiple tokens ahead of the main model.

[Speculative decoding](https://en.wikipedia.org/wiki/Transformer_(deep_learning)#Speculative_decoding) leverages the fact that computing n tokens in a batch (as in prompt processing) is more efficient than computing n sequentially (as in response generation). By generating draft tokens quickly and then verifying them with the target model in a single batch, this approach can achieve substantial speedups when the draft predictions are frequently correct.

## Implementations

The `llama-server` application supports several implementations of speculative decoding. An implementation with draft model can be mixed with an implementation without draft model.

### Draft Model (`draft`)

A much smaller model (called the _draft model_) generates drafts.
A draft model is the most used approach in speculative decoding.

### n-gram Cache (`ngram-cache`)

An n-gram is a sequence of n tokens. The n-gram cache implementation maintains statistics about short n-gram sequences.
A draft is computed using probabilities derived from these statistics. External statistics can also be loaded from files for improved accuracy.

See:

- #5479, #6828, #6848

### n-gram Map (`ngram-simple`, `ngram-map-*`)

These implementations search the token history for patterns and use matching sequences as draft candidates.
They require no additional model but rely on patterns that have already appeared in the generated text.
An example to use this approach can be the rewriting of source code by a LLM.

#### n-gram Map (`ngram-simple`)

This implementation looks for the last n-gram in history that matches the current n-gram and creates a draft using the m tokens following the matched n-gram. It is the simplest self-speculative approach with minimal overhead.

```
llama-server [...] --spec-type ngram-simple:n_max=64
```

#### n-gram Map Key (`ngram-map-k`)

This implementation looks for the current n-gram of size n (called the _key_) in the token history. If the key n-gram is followed by the same m tokens (called the _mgram_) multiple times, it creates a draft using these m tokens. This approach requires a minimum number of occurrences (stage key `ngram_min_hits`, default is 1) before generating drafts.

The number of accepted tokens is stored for each used n-gram.

**Example:**
```
llama-server [...] --spec-type ngram-map-k:n_max=64,ngram_min_hits=1
```

#### n-gram Map Key-4-Values (`ngram-map-k4v`)

This experimental implementation looks for the current n-gram of size n (called the _key_) in the token history. For each key, up to four _values_ (n-grams of size m, called _mgrams_) are tracked. An internal statistic counts the occurrences of each mgram after the key n-gram. If one mgram is significantly more frequent than the others, it is used as the draft.

The number of accepted tokens is stored for each used n-gram.

**Example:** Server options to be used if there are a lot of longer repetitions.
```
llama-server [...] --spec-type ngram-map-k4v:n_max=64,ngram_size_n=8,ngram_size_m=8,ngram_min_hits=2
```

### n-gram Mod (`ngram-mod`)

Add basic ngram hasher for speculative decoding:

- For each ngram, compute a hash using LCG
- For each computed hash, store the next token
- During speculation, iteratively compute the rolling hash of the last n tokens and pick the next token from the storage

Some characteristics:

- Lightweight (~16 MB)
- Constant memory and complexity
- Can generate variable draft lengths (i.e. m is not fixed)

Currently, a single hash pool is shared across all server slots, so different requests can benefit from each other.

**Sample usage:**

```
# notes:
# - small `n` are not recommended
# - MoEs require long drafts
# - dense models: can reduce `n_min` and `n_max`

llama-server ... --spec-type ngram-mod:n_max=64,n_min=48,ngram_size_n=24
```

Applications:

- Iterating over a block of text/code (e.g. in llama.vim)
- Reasoning models (when they have to repeat their thinking in the final answer)
- Summarization

Example Video:

- See #19164

### Differences between ngram-simple, ngram-map and ngram-mod

- ngram-simple looks for a previous matching n-gram and inserts the following m-gram.
- ngram-map-k looks for a previous matching n-gram and inserts the following m-gram but uses an internal hash-map of n-grams in the current context window.
- ngram-mod uses a hash pool which is shared across all server slots. The hash pool is a map from n-gram hash to the next token (not the next m-gram as in ngram-map).

## Command-Line Options

The canonical startup surface is repeated `--spec-type SPEC[:k=v,...]`. Legacy `--spec-stage`, `--draft-*`, `--spec-ngram-*`, `--suffix-*`, and `-mtp` flags are rejected with replacement guidance.

### `--spec-type SPEC[:k=v,...]`

Each `--spec-type` entry defines one speculative stage. Repeat it to configure the supported two-stage path.

| Type | Description |
|------|-------------|
| `none` | No speculative decoding |
| `draft` | Draft-model speculative decoding; pair with `-md/--model-draft` |
| `mtp` | Embedded or assistant-backed MTP |
| `ngram-cache` | Use n-gram cache lookup |
| `ngram-simple` | Use simple n-gram pattern matching |
| `ngram-map-k` | Use n-gram pattern matching with n-gram keys |
| `ngram-map-k4v` | Use n-gram pattern matching with n-gram keys and up to four m-gram values |
| `ngram-mod` | Use the shared n-gram hasher |
| `suffix` | Use suffix-tree speculative decoding |

Canonical stage keys:

| Key | Meaning |
|-----|---------|
| `n_max` | Maximum drafted tokens for that stage |
| `n_min` | Minimum usable drafted tokens for that stage |
| `p_min` | Minimum speculative probability threshold |
| `ngram_size_n` | Lookup n-gram size |
| `ngram_size_m` | Draft m-gram size |
| `ngram_min_hits` | Minimum matching hits for n-gram map stages |
| `suffix_min_match_len` | Minimum suffix context match length |
| `suffix_max_depth` | Maximum suffix-tree depth |
| `suffix_corpus` | Optional suffix corpus file for pre-warming |

String-valued stage keys such as `suffix_corpus` need shell-safe quoting when the value contains commas. From a normal shell, quote the value inside the stage payload so the parser sees the comma as part of the string value.

Example shell-safe form:

```bash
./llama-server [...] \
    --spec-type "suffix:n_max=16,n_min=2,suffix_min_match_len=5,suffix_max_depth=64,suffix_corpus='/tmp/spec,type-corpus.json'"
```

If you are constructing `argv` directly without shell unescaping, the parser also accepts escaped commas as `\,`.

Examples:

```bash
# Single-stage MTP
./llama-server [...] --spec-type mtp:n_max=1,p_min=0.0

# Single-stage ngram-mod
./llama-server [...] --spec-type ngram-mod:n_max=64,n_min=48,ngram_size_n=24

# Draft-model speculation
./llama-server [...] --model-draft draft.gguf --spec-type draft:n_max=4,p_min=0.0

# Two-stage self-spec -> MTP fallback
./llama-server [...] \
    --spec-type ngram-mod:n_max=64,n_min=2,ngram_size_n=8 \
    --spec-type mtp:n_max=1,p_min=0.0

# Suffix stage with pre-warmed corpus
./llama-server [...] \
    --spec-type suffix:n_max=16,n_min=2,suffix_min_match_len=5,suffix_max_depth=64,suffix_corpus=/path/to/corpus.json

# Suffix stage with a comma-bearing corpus path from a normal shell
./llama-server [...] \
    --spec-type "suffix:n_max=16,n_min=2,suffix_min_match_len=5,suffix_max_depth=64,suffix_corpus='/tmp/spec,type-corpus.json'"
```

### `--spec-autotune`

Autotunes the active stage parameters and reports the best configuration back as a canonical `--spec-type ...` snippet.

## Statistics
Each speculative decoding implementation prints statistics.

```
draft acceptance rate = 0.57576 (  171 accepted /   297 generated)
statistics ngram_simple: #calls = 15, #gen drafts = 5, #acc drafts = 5, #gen tokens = 187, #acc tokens = 73
statistics draft: #calls = 10, #gen drafts = 10, #acc drafts = 10, #gen tokens = 110, #acc tokens = 98
```

```
draft acceptance rate = 0.70312 (   90 accepted /   128 generated)
statistics ngram_mod: #calls = 810, #gen drafts = 15, #acc drafts = 15, #gen tokens = 960, #acc tokens = 730, dur(b,g,a) = 0.149, 0.347, 0.005 ms
```

```
statistics ngram_map_k: #calls(b,g,a) = 6 1690 26, #gen drafts = 26, #acc drafts = 26, #gen tokens = 1248, #acc tokens = 968, dur(b,g,a) = 2.234, 1.427, 0.016 ms
```


- `#calls(b,g,a)`: number of calls of begin (new prompt), generation and accumulation of this implementations
- `#gen drafts`: number of drafts generated by this implementation
- `#acc drafts`: number of drafts accepted (partially) by the main model
- `#gen tokens`: number of tokens generated by this implementation (including rejected tokens)
- `#acc tokens`: number of tokens accepted by the main model
- `dur(b,g,a): durations of begin (new prompt), generation and accumulation (process acceptance).

## MTP rollback, depth cutoff and the 27B preset (2026-10-14)

All numbers: Qwen3.8-27B PXQN4 with its MTP head on **one V100**, greedy, REPS 3, several on
a busy box (lower bounds). "Prose / code" are the two request classes of the bench.

**`PXA_SPEC_CKPT_REPLAY`** (also `--recurrent-ckpt-mode replay`). A rejected draft on a
DeltaNet hybrid must roll the recurrent state back. The default keeps one state row per draft
step. Replay instead keeps the start state plus each token's inputs and, on a rejection,
re-runs the same kernel over the accepted tokens.

| value | meaning |
|---|---|
| unset or `0` | per-step rows, **byte-identical to before** (default) |
| `1` | force replay |
| `2` | replay only when the per-step rows do not fit the requested depth |

Measured: output exact (200 unit rounds, 1509 checks, 0 failures; 8k engine run sha
9c736e088272 identical in both modes, same drafts and acceptance), VRAM 182 MiB against
604 MiB at depth 4, and at 8k replay costs 0.4% prose / 0.3% code. Its point is memory: it
lets a deeper chain fit at 16k-32k. With today's MTP head that deeper chain **does not pay on
prose**: at 32k, depth 4 with replay was 55.02 vs 57.11 t/s (-3.7%) against the current
depth 1, and the shipping ramp unlocked to depth 3 by replay lost 10.5% prose. Code at 16k
gained 7.5%. Hence default off.

**`PXA_SPEC_DEPTH_CUTOFF` and `PXA_SPEC_DEPTH_CUTOFF_CTX`** (opt-in, default off). Stops the
serial MTP draft chain when the cumulative product of draft probabilities falls below a
floor. The floor is `PXA_SPEC_DEPTH_CUTOFF * min(1, n_past / PXA_SPEC_DEPTH_CUTOFF_CTX)`;
`CTX=1` makes it flat. At 8k with `n_max=4,p_min=0.5`: a flat floor of 0.5 gave prose 62.58
vs 57.5 (+9.1%) and code 67.87 vs 65.9 (+3.1%); 0.12 to 0.25 were within noise to +2.8%
prose. Deeper chains (n_max 5 or 6) lost at 8k. **Candidate**: the quiet rerun, the 16k and
32k cells, and the determinism and needle gates on the shipping binary are owed, so it is
not part of any preset default yet.

**The 27B MTP opt-in preset (contexts up to about 8k, one V100).**

```
--spec-type mtp:n_max=4,p_min=0.5
PXA_SPEC_DEPTH_NMAX=0      # lift the depth ramp
PXA_PXQN_QPN=1             # tensor-core verify GEMV, bit-identical to the default kernel
```

Against the shipping ramp (depth 1 below 8k): short context code +29% (84.4 vs 65.5 t/s),
prose +5%; at 8k code +7.8%, prose +0.5%; at 16k it **loses** (prose -7.5%, code -3.8%; the
checkpoint budget caps depth at 3), and at 32k both clamp to depth 1 with identical output.
Determinism 12/12 at `-c 8192`, needle recalled 2 of 2. Opt-in only, not a flat default
. `PXA_PXQN_QPN=2` is faster but **not**
bit-identical and is a separate opt-in.
