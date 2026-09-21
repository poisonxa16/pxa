# no-AVX2 CPU inference: degenerate output — investigation state (2026-08-20)

**Status: root cause NOT found.** Reproducible, well-localised, five hypotheses eliminated.
Not on any path we serve (all seats are GPU-resident; full GPU offload is verified correct).

## Reproduce

    cmake -B b -DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_F16C=OFF -DGGML_FMA=OFF \
             -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
    ./b/bin/llama-cli -m <any Q8_0 model> -ngl 0 -c 256 -n 12 --seed 1 --temp 0 \
        -p 'The capital of France is'

    -fa off  ->  PXA_SAMPLE_SOFTFAIL_v1: greedy argmax logit is non-finite
    -fa on   ->  "ux respondters online commonAlice HttpprechInteractionEnabled"

## Established

| fact | evidence |
|---|---|
| AVX2 CPU build is CORRECT | same model/prompt/seed: "Paris, a city renowned for its rich history" |
| defect is no-AVX2-specific | not a general CPU-inference defect |
| corruption is UPSTREAM of attention | with FA OFF the logits are NON-FINITE; FA merely launders NaN into finite-but-wrong values, which is why it first looked like an attention bug |
| the model uses only Q8_0 / F32 / BF16 | GGUF tensor inventory: 443 / 308 / 2 |

## Eliminated

1. **The three scalar fallbacks added in `build: compile without AVX2`** — `test-quantize-fns`
   PASSES `q8_0_x4`, `q8_1_x4`, `q8_2_x4` on a no-AVX2 build. (`q8_0_x4` indexes absolutely via
   `xb = x + i*QK8_0`; `q8_1_x4_T` advances `x += QK8_1`. Different conventions, each correct —
   do not "fix" one to match the other.)
2. **Cross-TU macro split** — all three consumers of `GGML_USE_IQK_MULMAT` (`ggml.c`,
   `ggml-quants.c`, `iqk_quantize.cpp`) include `ggml-impl.h`, and its condition is byte-identical
   to `iqk_config.h`'s `__AVX2__ || __ARM_FEATURE_DOTPROD`.
3. **Include ordering** — `ggml.c` includes `ggml-impl.h` at line 18, first macro use at line 26.
4. **`vec_dot_type` pairing** — every ik-guarded traits block has a correct `#else`
   (`GGML_TYPE_Q8_0` with `ggml_vec_dot_q8_0_q8_0`).
5. **Flash attention** — NaN is present with FA disabled.
6. **`iqk_flash_attn_noalibi` no-ik stub** — returns `false`, correctly falling through to stock.
   (Also moot: `iqk_flash_attn.cpp` is not compiled when ik is off.)

## Not yet checked

- **BF16.** The model carries 2 BF16 tensors and `to_float`/`vec_dot` for BF16 differ by ISA.
  `test-quantize-fns` passes bf16 in isolation but does not exercise its graph use. **Start here.**
- Core ops on the no-AVX2 path: rms_norm, rope, softmax.
- ~~`iqk_cpu_ops.cpp`~~ — **checked and closed, 2026-08-21.** The file is gone; all twelve of
  its exported symbols were reowned or deleted (ik separation phase 2). The answer to the
  question asked here is that most of those call sites had NO stock fallback at all:
  `MUL_MULTI_ADD`, `HADAMARD` and `FUSED_RMS_RMS_ADD` are fork ops whose only body was in that
  file, and `GROUPED_TOPK` had nothing but a type switch. They could never have silently
  no-opped, but they also could never have been dropped — they had to be moved, and now live in
  `ggml/src/pxq-cpu.c` and `ggml/src/pxq-cpu-ops.cpp` as `pxa_*`.

  Nothing in the file was a no-AVX2 defect after the `biased_sigmoid` scalar-tail fix: every
  SIMD block in it except `iqk_ssm_conv4` already had a correct scalar tail underneath. A
  differential test (`<local-path>`) ran the old bodies and the reowned ones
  over the same inputs in the same process at nth=1 and nth=4 and confirmed BIT-IDENTICAL
  output on all three CPU trees, so this reowning moved no numerics in either direction.
  `iqk_ssm_conv4` was deleted rather than ported (230 lines of intrinsics over a
  `#else return false;`); the generic `ggml_compute_forward_ssm_conv` path that every no-AVX2
  build already ran is now the only path, which costs AVX2/NEON speed and nothing else.

## Separate real defects found on the way (no-AVX2, NOT our bug — no type below is in the model)

    mxfp4 dot product        FAILED (inf)
    q6_0  dot product        FAILED (1.090274)
    iq1_m / q1_0_g128 / iq1_bn / iq2_bn / iq2_k   quantization error FAILED

Three types additionally hit `GGML_ABORT("not implemented")` in `iqk_quantize.cpp` without ik —
their `vec_dot` exists only as an `iqk_mul_mat` call with no scalar path. So
`-DGGML_IQK_MUL_MAT=OFF` links and runs, but **aborts if the model uses one of those types**.

## Note for whoever picks this up

`PXA_SAMPLE_SOFTFAIL_v1` is what made this diagnosable: it reports "greedy argmax logit is
non-finite" instead of crashing. Without it this is a bare segfault.

Empty output from a test here is almost always a broken harness, not a result — this
investigation lost several cycles to a wrong mount path, a missing `libcudart`, an unregistered
CMake target and a nonexistent `libggml-base`. Verify the binary exists and runs before reading
anything into silence.
