"""gate_qwen4exp_vs_producer.py -- close the gap gate_gdn_perm_qwen4exp.py names. CPU ONLY.

That gate proves the qwen4exp v-head permutation is a valid whole-panel bijection and says, in
its own docstring, what it CANNOT prove: that the permutation is WANTED and that its DIRECTION
is right. "No HF-form qwen4exp exists on this box, so there is nothing to compare against" --
and applying a permutation nobody asked for is as wrong as omitting one.

There is something to compare against. The script that BUILT this GGUF from the HF checkpoint
is on the box, and it is the exact inverse of the map the converter needs. Every claim below
is asserted against that file's source text, so the gate fails if the producer is ever changed
under us rather than quietly passing on a stale belief.

WHAT THIS PROVES (all four were "settled only by a live boot" before):

  1. DIRECTION of the v-head permutation. The producer holds
         PERM48 = np.arange(48).reshape(16, 3).T.ravel()      # gguf[i] = hf[PERM48[i]]
     and applies it going HF -> ggml. Our gather g goes the other way, hf[j] = ggml[g[j]].
     Composing: hf[j] = hf[PERM48[g[j]]], so the two are inverse iff PERM48[g[j]] == j for
     every j. That is the assertion, at this model's real geometry read from the GGUF.

  2. WHICH TENSORS carry it, and the q|k offset. The producer permutes attn_gate, ssm_out,
     ssm_a, ssm_dt, ssm_alpha, ssm_beta whole, and attn_qkv / ssm_conv1d only after
     2 * n_kv_in leading rows. GDN_PERM_SPEC must agree tensor for tensor.

  3. ssm_alpha <-> in_proj_a and ssm_beta <-> in_proj_b. The first draft of the name map had
     these crossed, which changes no shape and raises nothing.

  4. THE NORM CONVENTION. "Every RMSNorm weight is stored by HF as (w - 1) ... the converter
     must add 1.0. linear_attn.norm is the sole exception." So every norm this converter
     emits needs the -1 back, except ssm_norm -- and that is exactly the set the name map
     declares.

Run:  python3 gate_qwen4exp_vs_producer.py [gguf] [producer.py]
Exit 0 = every claim holds. Exit 1 = a claim failed, and it is named.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gguf_to_vllm import gguf_raw as G          # noqa: E402
from gguf_to_vllm import namemap as NM          # noqa: E402
from gguf_to_vllm import namemap_qwen4exp as Q4  # noqa: E402

DEFAULT_GGUF = os.environ.get("PXQ_GGUF", "/path/to/model-PXQU.gguf")
DEFAULT_PRODUCER = os.environ.get(
    "PXQ_PRODUCER",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "convert_qwen4exp.py"),
)

#: ggml suffix -> (producer applies the v-head gather?, leading rows it skips first).
#: "qk" means the producer's perm_v(a, 2 * n_kv_in) helper; "" means the whole axis.
PRODUCER_PERM = {
    "attn_qkv.weight":   "qk",
    "ssm_conv1d.weight": "qk",
    "attn_gate.weight":  "",
    "ssm_out.weight":    "",
    "ssm_a":             "",
    "ssm_dt.bias":       "",
    "ssm_alpha.weight":  "",
    "ssm_beta.weight":   "",
}

fails: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    print(("  PASS  " if ok else "  FAIL  ") + what + (("  -- " + detail) if detail else ""))
    if not ok:
        fails.append(what)


def main(argv: list[str]) -> int:
    gguf = argv[1] if len(argv) > 1 else DEFAULT_GGUF
    prod = argv[2] if len(argv) > 2 else DEFAULT_PRODUCER
    if not os.path.exists(prod):
        print(f"SKIP: the producer script is not on this box ({prod}). This gate exists to "
              f"compare against it and has nothing to say without it.")
        return 0
    src = open(prod).read()
    print(f"gguf     {gguf}\nproducer {prod}\n")

    # ---- 1. the permutation, at the geometry the file itself declares ------------------------
    gg = G.GGUFHeaderOnly(gguf, os.path.getsize(gguf)) if os.path.exists(gguf) else None
    if gg is None:
        print(f"SKIP: {gguf} not present")
        return 0
    try:
        geom = NM.gdn_geometry(gg.kv)
        m = re.search(r"PERM48\s*=\s*np\.arange\((\d+)\)\.reshape\((\d+),\s*(\d+)\)\.T\.ravel\(\)",
                      src)
        check(m is not None, "the producer still defines PERM48 as an (n_k, R) transpose",
              "" if m else "its definition changed; re-read it before trusting this gate")
        if m:
            n, n_k, R = (int(x) for x in m.groups())
            check((n, n_k, R) == (geom.n_v_heads, geom.n_k_heads, geom.repeats),
                  "producer PERM48 geometry == the GGUF's own KVs",
                  f"producer ({n}, {n_k}, {R}) vs file "
                  f"({geom.n_v_heads}, {geom.n_k_heads}, {geom.repeats})")
            perm48 = np.arange(n).reshape(n_k, R).T.ravel()
            g = np.asarray(NM.v_head_gather(geom))
            check(g.shape == perm48.shape and np.array_equal(perm48[g], np.arange(n)),
                  "converter v_head_gather is the INVERSE of the producer's PERM48",
                  f"PERM48[g] = {perm48[g][:6].tolist()}... (want 0,1,2,...)")

        # ---- 2. which tensors, and the q|k offset --------------------------------------------
        for suf, kind in PRODUCER_PERM.items():
            spec = NM.GDN_PERM_SPEC.get(suf)
            want_off = "qk_offset" if kind == "qk" else "zero"
            check(spec is not None and spec[1] == want_off,
                  f"GDN_PERM_SPEC[{suf!r}] offset == producer's ({want_off})",
                  f"got {spec!r}")
        extra = sorted(set(NM.GDN_PERM_SPEC) - set(PRODUCER_PERM))
        check(not extra, "no tensor is permuted that the producer leaves alone", f"extra {extra}")
        check("perm_v(store.raw_u16(c), 2 * n_kv_in)" in src,
              "the producer's q|k offset is still 2 * n_kv_in")

        # ---- 3. alpha / beta ------------------------------------------------------------------
        check('(("ssm_alpha", "in_proj_a"), ("ssm_beta", "in_proj_b"))' in src,
              "producer pairs ssm_alpha<->in_proj_a and ssm_beta<->in_proj_b")
        check(Q4.GGML_TO_HF["ssm_alpha.weight"] == "linear_attn.in_proj_a.weight"
              and Q4.GGML_TO_HF["ssm_beta.weight"] == "linear_attn.in_proj_b.weight",
              "the name map pairs them the same way",
              f"alpha->{Q4.GGML_TO_HF['ssm_alpha.weight']}, "
              f"beta->{Q4.GGML_TO_HF['ssm_beta.weight']}")

        # ---- 4. the norm convention -----------------------------------------------------------
        check("linear_attn.norm is the sole exception" in src,
              "producer still says linear_attn.norm is the only norm it does not offset")
        norms = sorted(s_ for s_ in Q4.GGML_TO_HF
                       if s_.endswith("norm.weight") or "_norm" in s_)
        declared = set(Q4.GEMMA_NORM_SUFFIXES)
        missing = [s_ for s_ in norms if s_ != "ssm_norm.weight" and s_ not in declared]
        check(not missing, "every per-layer norm except ssm_norm takes the gemma -1",
              f"undeclared: {missing}")
        check("ssm_norm.weight" not in declared, "ssm_norm does NOT take it")
        check("output_hc_norm.weight" in declared,
              "the model-level output_hc_norm takes it too (producer: norm(output_hc_norm...))")

        # ---- 5. the PLE layer ------------------------------------------------------------------
        check("config's ple_layer_ids is NOT the module index" in src,
              "producer still warns that config ple_layer_ids != the module index")
        blk = Q4.ple_ggml_layer(gg.order)
        kv_ids = gg.kv.get("qwen4exp.ple.layers")
        check(blk is not None and kv_ids is not None and [int(x) for x in kv_ids] == [blk],
              "the GGUF's ple.layers is the ZERO-based block that carries ple_key",
              f"kv {list(kv_ids or [])} vs block {blk}")
    finally:
        gg.close()

    print()
    if fails:
        print(f"FAILED {len(fails)} claim(s): " + "; ".join(fails))
        return 1
    print("ALL CLAIMS HOLD. The qwen4exp v-head permutation, its direction, the alpha/beta "
          "pairing, the norm convention and the PLE layer index are settled against the "
          "script that produced the artifact -- not deferred to a live boot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
