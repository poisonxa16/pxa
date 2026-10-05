# The native PXQ4 encoder, and policy p2a

## Why P2 was blocked

`gguf_to_vllm/encoder.py` binds a frozen C ABI by ctypes:

    int pxq4_encode(const float *src, uint8_t *dst, int R, int K, const float *imx_or_null)

Nothing built it. P1 does not care — P1 only *moves* bytes that are already PXQ4, which is
why it can be proven by byte comparison. Every P2 policy re-encodes tensors that are not
PXQ4 in the artifact, so without the shared object `convert.py` raised SystemExit at
planning time. `tools/build_encoder.sh` builds it from `src/pxq4_encode_shim.cpp`.

## Two things that silently produce a different codec

**Tier.** `GGML_TYPE_PXQ4` (252) is the CORE tier, `tier = 0`. Read off the engine's own
dispatch — `const int tier = tgt == GGML_TYPE_PXQ4HQ ? 1 : 0;` — and cross-confirmed by the
ABI's panel formula `128 + (K/32)*1088`, since the HQ tier uses 1152-byte slabs. Guessing
here yields a well-formed file in the wrong codec.

**row0.** It seeds the deterministic tie-break `pxq_tie_take_hi`. Always encode a whole
tensor from row 0. Threading is safe because the upstream `pxq6_quantize_tensor` passes each
chunk's ABSOLUTE row offset, so bytes are identical at any thread count — but a caller that
encodes a slice starting mid-tensor gets different bytes.

Build with no `-ffast-math`: the codec is parity-locked to exact fp32 rounding.

## Validation — what was actually proven

| check | result |
|---|---|
| native decode vs numpy `reference.dequant`, 3 real tensors up to 89M elements | **bit-identical fp32** |
| `encode_and_check` tripwire (bound 0.35) | wrel 0.0169 / 0.0172 / 0.0172 |
| synthetic gaussian, published PXQ6-core protocol (lab figure 0.068) | wrel 0.0708 |
| dequant -> re-encode byte identity | 99.89% - 99.95% |

Bit-exact decode parity is the load-bearing one: it pins tables, panel and slab layout,
nibble order and multiply order simultaneously.

**The re-encode is deliberately not 100%, and that is structural, not a bug.** The codec's
largest representable value is `anchor * 0.98779`, strictly below the anchor. Re-encoding a
reconstruction therefore shrinks the fp16 row anchor by about one step-cluster (~44% of
anchor header bytes differ), which drags a small fraction of sub-scale nibbles (~99.4%
match) and codes (~99.97% match). Bounded by the double round-trip: `D(E(D(x)))` vs `D(x)`
gives wrel <= 0.0172, max elementwise diff <= 0.017.

## p2a

Re-encodes the 48 `linear_attn.out_proj` tensors from MXFP4 to PXQ4. This is the only
backbone change in the whole family with a real quality measurement behind it: relative RMS
0.083 (PXQ4) vs 0.199 (MXFP4) against the Q8_0 source, a 2.4x improvement, and it removes
the last flat-MXFP4 landing — the codec class the rev2 backbone was written to eliminate.

| | p1 | p2a |
|---|---|---|
| checkpoint | 22,932,724,192 B | **20,715,477,472 B** (-9.67%) |
| weight bandwidth @ TP=4 | 4.534 GiB/GPU/token | **4.018** (-11.4%) |
| weight bandwidth @ TP=2 | 9.066 | 8.034 |

The plan matched the predicted total to the byte. Gates: G2 byte round-trip 325/325 (all
PXQ4 tensors), G1 fp32 bit-exact 12/12 against the engine's own decoder, G3 shard/dequant
commute 96/96 at TP 2 and 4, G5H GDN v-head order PASS on all 48 layers with `--ref-hf`,
key-set 1184/1184, `--check-output` 96 match / 0 mismatch.

## Known gaps

- Bandwidth figures are the converter's own **projection**, not a throughput measurement.
- The 48 re-encoded tensors were encoded **without an imatrix**. The ABI accepts one
  (`imx_or_null`); the converter supplies none. Re-run p2a once a real imatrix exists,
  otherwise the GGUF is calibrated and the sidecar's re-encoded tensors are not.
- G1 and `--check-output` are sampled (12 and 96, covering all 6 distinct shapes). G2 is
  exhaustive.
- p2c additionally re-encodes `attn_k`/`attn_v` to 4 bits. The encoder now exists, so p2c is
  no longer blocked by tooling — it is blocked by a quality decision. See
  ALL_PXQ4_VERDICT.md.
