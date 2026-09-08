# PXA Network — PXQ4 sm_70 (Tesla V100) serving

Measured on the box, cards 2+4 (2x Tesla V100-PCIE-16GB, PHB, no P2P). Original recipe dated
2026-08-27, refreshed 2026-09-02 for kernel v12b and the NCCL/batched-tokens findings below.
Every arm below was correctness-gated ("The capital of France is" -> Paris) BEFORE its speed
number was recorded. A fast wrong answer is not a result.

## Shipping configuration (2026-09-02)

    scripts/pxa-serve-sm70.sh          # default profile: agg

    PXQ4_LIB=libpxq4_sm70_v12b.so
    PXQ4_MMV_MMA=1
    PXQ4_MMV_SPLIT_MAX_BLOCKS=300
    NCCL_P2P_LEVEL=SYS
    NCCL_BUFFSIZE=1048576
    --max-num-batched-tokens 4096
    --gpu-memory-utilization 0.88
    --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,16]}'

| metric | production (mnbt 4096, GMU 0.92) | this config | change |
|---|---|---|---|
| prefill @3k | 919 t/s | **1,008.6 t/s** | +9.8% |
| prefill @20k | 880 t/s | **983.7 t/s** | +11.8% |
| decode, single stream | 48.5 t/s | **50.35 t/s** | +3.8% |
| decode, aggregate @8 | 129 t/s | **190.3 t/s** | +48% |
| decode, aggregate @16 | 129 t/s | **299.0 t/s** | +132% |

n=7, coherence-gated 3/3. Aggregate is carried almost entirely by v12b below; prefill is carried
almost entirely by the NCCL fix below.

## v12: a Volta tensor-core path for the PXQ4 decode GEMV

Below batch size 5, PXQ4 decode on Volta runs the same per-row GEMV kernel it always has — that
route is untouched. At batch size ≥5, `PXQ4_MMV_MMA=1` routes the same PXQ4 tensors through a
wmma (m16n16k16, fp32-accumulate) tensor-core path instead: flat cost in M up to 16 rows in one
call, where the GEMV route pays per-row. Not bit-exact by construction (a different arithmetic
order) — see the quality gate below before shipping it.

| arm | prefill @3k | @20k | decode, single | agg@8 | agg@16 |
|---|---|---|---|---|---|
| v12, MMA off (bit-identical to v10/v11) | 932.8 | 893.5 | 50.13 | 129.7 | 129.2 |
| v12, `PXQ4_MMV_MMA=1` | 941.0 | 905.0 | 50.62 | **177.5 (+37%)** | **291.0 (+125%)** |

Single-stream decode is unchanged by design (the M=1 route isn't touched); the aggregate lines
are the tensor-core path engaging under load. A known caveat: the split-K partials arena this
path shares with the older kernel family is sized on first use and can be grown mid-run by an
eager small-batch call, which then frees memory a captured larger-batch graph still points at.
The 40-prompt gate below didn't reproduce it in any run tested, but the fix — size the arena
once, at load, for its maximum possible need — ships as `v12b` before this is called closed.

**Quality gate (same-top-token, 40 prompts, ~1k tokens each, greedy 32 tokens):** v12 MMA on
(worst-case `MIN_M=1`) vs MMA off — 40/40 first-token agreement (100%), 38/40 exact 32-token
match (95%). v12 MMA on at the shipping `MIN_M=5` vs off — 40/40 first-token (100%), 39/40 exact
(97.5%). Gate of record is ≥98% first-token agreement to pass; both configurations clear it (a
prior fused-GEMM attempt on this same shape failed this same gate at 87.5% and was never shipped).

## The NCCL P2P finding: prefill was 64% all-reduce

A prefill profile at 20k tokens found the GPUs 99.9% busy with no host bubble, yet the wall clock
was dominated by one collective: per forward chunk, NCCL all-reduce was **130 calls totaling
2,579 ms — 64.4% of the chunk** (cutlass GEMM was 26.6%, at 72% of V100 peak; everything else
under 5%). The all-reduce was moving 40.14 MB per call at 19.84 ms each — 2.02 GB/s, against a
raw peer-copy measurement of 2.82 GB/s on the same link. Both cards are on PCIe gen3 x4 with no
P2P path between them (a PHB topology), and NCCL was not using peer-to-peer at all by default
there.

`NCCL_P2P_LEVEL=SYS` forces it to try anyway: 2.02 → 2.26 GB/s on its own; adding
`NCCL_BUFFSIZE=1048576` (1 MiB, up from NCCL's default) gets to 2.45 GB/s. Combined
(`NCCL_P2P_LEVEL=SYS NCCL_BUFFSIZE=1048576`), measured end to end: **+7.0% prefill @3k, +7.9%
@20k**, aggregate throughput flat, decode −2.9% (inside run-to-run noise), coherence held.
`--gpu-memory-utilization` has to come down from 0.92 to **0.88** with P2P on — at 0.92 the torch
inductor autotune pass OOMs during warmup; it didn't at 0.85, so 0.88 is the highest value
verified not to. A second variant (adding `NCCL_MAX_NCHANNELS=1`) measured slightly higher
prefill but a reproducible decode stall and a 7.6% decode loss — rejected.

The honest hardware conclusion: this pair wants PCIe x16 slots or an NVLink-capable carrier far
more than it wants any kernel tuning — prefill would roughly double at x16. The env fix above is
the software ceiling on the slots this rig actually has.

## Same-top-token gate: also how a lever gets rejected

The same protocol used to pass v12 above rejected a 4-bit PXQ4 output head on a wider run: 70
prompts, ~1k tokens each, greedy 32-token continuations, first-token agreement against the fp16
head. Result: 68/70 (97.1%), below the ≥98% bar — two prompts flip their first token once the
head goes to 4 bits. Kept the fp16 head; the quantized-head idea is not dead (there's ~0.9 ms/step
of GEMV time sitting in it), just not at 4 bits without a smarter per-row scheme.

## The batched-tokens finding

`--max-num-batched-tokens` (the chunked-prefill budget) was swept from 2048 (the prior default)
through 8192, six interleaved boots to separate the effect from boot-to-boot drift:

| value | prefill @3k | @20k | decode | agg@8 | KV pool |
|---|---|---|---|---|---|
| 2048 | 918–920 | 878–880 | flat | flat | 71,680 tok |
| **4096** | **940–945 (+2.5%)** | **908–912 (+3.3%)** | flat | flat | 71,680 tok (unchanged) |
| 8192 | 943 | 903 | flat | flat | 60,757 tok (**−15%**) |

4096 wins outright — a real prefill gain with no decode, aggregate, or KV-pool cost. 8192 buys
nothing further and shrinks the KV pool by cutting into reserved scratch. Shipped as the default.

## The full v10/v11 sweep (2026-08-27, superseded above but kept for the record)

| arm                                   | decode | ms/tok | agg@4  | agg@8  |
|---------------------------------------|--------|--------|--------|--------|
| reproduce final-v10 (recorded 48.40)  | 48.74  | 20.52  | 106.46 | 134.36 |
| + PXQ4_MMV_SPLIT_MAX_BLOCKS=300       | 51.46  | 19.43  | 107.26 | 132.13 |
| + SPLIT=600                           | 51.31  | 19.49  | 106.60 | 134.47 |
| + SPLIT=150                           | 49.51  | 20.20  | 105.07 | 134.66 |
| + SPLIT=300, GMU 0.92                 | 51.32  | 19.49  | 104.85 | 133.76 |
| + SPLIT=300, MNS=16, ladder[1..8,16]  | 50.98  | 19.62  | 107.02 | 135.78 |

Prior recorded best on this pair: 50.76 (median). v11 exceeded it at 51.46; v12 above supersedes
this table for anyone shipping today.

## Why the thin image measured 34.1 before this

Nothing was broken in the build. The gate shipped a conservative recipe while
the later tuning never landed in it. The whole 34.1 -> 51.46 delta is config:

  GMU 0.90 -> 0.85          0.98 + a pinned 12 GiB KV is a 4-card NVLink setting and
                            ABORTS on a 16 GiB card at TP=2 before the model loads
  MNS 4 -> 8/16
  ladder [1,2,3,4] -> [1..8(,16)]
  PXQ4_LIB pinned to the v10 kernel
  PXQ4_MMV_SPLIT_MAX_BLOCKS=300   (gate_up mono -> split; 48.74 -> 51.46)

## Things that will silently cost you, if changed

- **The capture ladder must be passed explicitly.** When `cudagraph_capture_sizes`
  is None the sm70 branch hard-codes `[1,2]`, so every batch above 2 runs eager,
  roughly 4x slower. The launcher reads the value back from the startup log and
  warns if it collapsed to `[1,2]` -- a quoting slip there is otherwise invisible.
- **Bash brace-expands the JSON.** `{"cudagraph_capture_sizes":[1,2,3,4]}` becomes
  `cudagraph_capture_sizes:[1` because of the commas inside the braces. Single-quote
  it for whichever shell finally parses it.
- **PXQ4_LIB must be explicit.** The site tree bundles an sm70-only `.so` built
  against the torch 2.10 ABI; leaving it unset on the wrong image dies with an
  undefined-symbol crash part-way through model load, not a clear message.
- **The f16 mmv hook does not arm on sm_70** (`f16 hook armed: 0` in every arm
  above). It is Pascal-specific. It is worth ~12% on sm_60, so its absence there
  is a real loss; here it is expected and costs nothing.

## The one genuine build bug that had to be fixed first

`vllm/triton_utils/importing.py` looked up its own distribution as only "vllm" or
"1cat-vllm". Ours is named `pxa-vllm`, so both lookups raised, and
`PackageNotFoundError` subclasses `ImportError` -- so it was swallowed by the
enclosing `except ImportError` and misreported as "triton.backends could not be
imported. Disabling Triton." Triton was silently disabled on a machine where it
works, vLLM installed a placeholder module, and a kernel later called
`triton.next_power_of_2` on that placeholder and killed both workers.

Before the fix the image did not serve at all. After it, sm70 gates clean.

The same fix re-enabled Triton on Pascal, where Triton raises fatally at kernel
compile time ("Triton only supports devices of CUDA Capability >= 7.0"). A
capability guard in the same file now disables Triton below sm_70. It reads the
capability via NVML rather than `torch.cuda`, deliberately: NVML does not create
a CUDA context, so the check stays safe at import time in a process that later
forks its workers. Verified in both directions -- V100 keeps Triton, P100 does not.

## Known build-label defect (not yet fixed)

`pxa-vllm:sm60` carries `CUDA_ARCH=sm_70` in its image environment.
`TORCH_CUDA_ARCH_LIST=6.0;7.0` is correct, but `CUDA_ARCH` is inherited wrong, so
it cannot be used to identify the target arch. That is why the capability guard
above reads NVML instead of trusting the image label.

## Engine base

The image is built from our fork of 1Cat-vLLM at tag **v1.5.0** plus the PXA commits
(the Pascal port, the short-prefill guard, the build recipe, the vendored PXQ4
sidecar). The shipped tags are `ghcr.io/poisonxa16/pxa-vllm:sm70` and
`ghcr.io/poisonxa16/pxa-vllm:sm60`; the images built from the previous base are kept
as `:sm70-v1cat-8f5d78e` and `:sm60-v1cat-8f5d78e` so a regression can be bisected
against the exact bytes that produced the earlier numbers.

What that base changes for this seat, and what it does not, is written up once in
[docs/VLLM.md](VLLM.md#the-engine-base) rather than repeated here. The two operational
notes that belong on this page: custom all-reduce stays **on** for the dense Volta seat
and is byte-gated, and `cudagraph_capture_sizes` counts **tokens**, so a speculative
decode arm needs `num_seqs x (k+1)` in the ladder or every step quietly falls out of
the captured graphs.

---

## Small-M route selection on sm_70 (v17 kernels)

The Volta decode/verify fast path is a raw `mma.sync.m8n8k4` kernel, `k_pxq4_mma884`, that
dequantises PXQ4 straight into the A-fragment registers with no shared-memory staging of the
weight. It is **on by default on `sm_70+`**, under the small-M master switch, and it is what the
speculative rows in the release notes were measured on.

It is worth knowing that this route is **`M`-dependent by design**: the same token decoded in a
1-token step and verified inside an 8-token step can land on opposite sides of a near-tie. That is
the documented cause of the occasional near-tie flip in the gate captures, and it is a property of
this fast path, not only of vLLM's batching on `cc 7.0`.

Five environment switches select the route. All are read once, at first use, before any CUDA-graph
capture — set them before the server starts.

| variable | default | effect |
|---|---|---|
| `PXA_PXQ4_SMALLM` | `1` (on) | The small-M master switch. `0` turns the small-M tensor-core dispatch off; in the sidecar it also drops the small-M threshold from 16 back to the library's `pxq4_mmv_max_m`. (`pxq4_mma.cu`, `pxq4_vllm/linear.py`) |
| `PXQ4_MMA884` | unset → follows the master switch | `0` forces the older **wmma** arm back; `1` arms `m8n8k4` even with the master switch off. Note that the wmma arm is also a tensor-core path. (`pxq4_mma.cu`) |
| `PXQ4_MMA884_CTAS` | unset → the v17 rule | Set to `160` to restore the pre-v17 uncapped split-K budget rule **bit for bit** — it reproduces every v15/v16 number exactly. (`pxq4_mma.cu`) |
| `PXQ4_MMA884_FUSE` | `1` (on) | `0` forces the old two-launch `k_pxq4_mma884` + `k_pxq4_mma_reduce` pair instead of the fused single-split epilogue. The fused epilogue is **bit-identical**, not approximately equal, and `FUSE=0` is how that identity was gated. (`pxq4_mma.cu`) |
| `PXQ4_LOG_DISPATCH` | `0` | Prints which route each call took, so the above can be verified rather than assumed. (`pxq4_kernel_torch.cpp`) |

`m8n8k4` is a strict win over wmma at every `M` measured on a V100 — at `M=8`, in microseconds for
`gate_up` / `down` / `qkv_z` / `out_proj`: **117 / 58 / 68 / 30** against wmma's 159 / 87 / 87 / 39
and SIMT's 283 / 139 / 144 / 67 — and it sits 1–2 fp16 ULP from the SIMT path. On `sm_60` the
architecture check fails and neither tensor-core arm is ever reached.
