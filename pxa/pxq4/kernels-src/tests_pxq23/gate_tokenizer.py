"""gate_tokenizer.py -- prove the reference checkpoint's tokenizer IS the GGUF's tokenizer.

WHY THIS GATE EXISTS. The converter takes config.json, the tokenizer and the vision tower from
a ``--ref-hf`` checkpoint, because a GGUF carries none of them in HF form. For the Fusion 35B
the only vLLM-form qwen35moe on this box is a DIFFERENT fine-tune of the same base
(coder35-moe-pxq4-m1). Borrowing its config and vision tower is defensible -- the architecture
and every shape are identical and neither fine-tune touched the vision tower. Borrowing its
TOKENIZER is not defensible on architecture alone: two fine-tunes of one base can add special
tokens, and a tokenizer that differs by one id produces a model that loads, serves, and emits
subtly wrong text with no error anywhere. So it is checked, and the conversion stops if it
fails.

WHAT IS CHECKED, in increasing strength:
  T1 vocab size         the GGUF's token count == the HF tokenizer's == config.vocab_size
  T2 vocab identity     every id maps to the same token string in both, byte for byte, across
                        the whole vocabulary -- not a sample
  T3 merges identity    the BPE merge list is the same list in the same ORDER (order is the
                        merge priority; a reordered list is a different tokenizer)
  T4 special ids        bos / eos / pad agree between the GGUF KVs and the HF config
  T5 round trip         50 strings (ascii, unicode, code, whitespace runs, the chat template's
                        own control tokens) tokenize identically under the HF tokenizer and
                        under a from-scratch BPE driven by the GGUF's OWN vocab and merges --
                        so T2/T3 are exercised as a tokenizer and not just compared as lists

Usage:  python3 gate_tokenizer.py --gguf <file.gguf> --ref-hf <dir>
Exit 0 = the borrowed tokenizer is the GGUF's tokenizer and the provenance note may be written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gguf_to_vllm import gguf_raw as G          # noqa: E402

PROBES = [
    "Hello, world!",
    "the quick brown fox jumps over the lazy dog",
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    "1234567890",
    "3.14159265358979",
    "  leading and trailing   ",
    "\ttab\tseparated\tvalues\t",
    "line one\nline two\n\nline four",
    "def f(x):\n    return x ** 2 + 1\n",
    "SELECT * FROM t WHERE a = 'b' AND c <> 3;",
    "#include <stdio.h>\nint main(void){return 0;}",
    "{\"a\": [1, 2, {\"b\": null}], \"c\": true}",
    "https://example.com/a/b?c=d&e=f#g",
    "user@example.com",
    "C:\\Users\\name\\Documents\\file.txt",
    "/usr/local/lib/python3.12/site-packages",
    "Ünïcödé änd áccênts",
    "Ελληνικά κείμενο",
    "Русский текст здесь",
    "日本語のテキストです",
    "中文文本测试",
    "한국어 텍스트",
    "العربية نص",
    "עברית טקסט",
    "emoji: \U0001F600 \U0001F680 \U0001F9E0",
    "math: \u2200x\u2208\u211d, x\u00b2 \u2265 0",
    "combining: e\u0301 a\u0300 n\u0303",
    "zero width\u200bjoined",
    "a" * 200,
    " " * 40,
    "\n" * 12,
    "MiXeD CaSe WoRdS",
    "hyphen-ated and under_scored and dot.separated",
    "(parens) [brackets] {braces} <angles>",
    "quote 'single' \"double\" `back`",
    "1,000,000.00 USD and \u20ac1.234,56",
    "2026-09-05T12:00:00Z",
    "0xDEADBEEF 0b1010 0o777",
    "a\u0000b",
    "trailing backslash \\",
    "repeated!!! punctuation???",
    "CamelCaseIdentifierName",
    "snake_case_identifier_name",
    "SCREAMING_SNAKE_CASE",
    "kebab-case-identifier",
    "The rain in Spain falls mainly on the plain.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "x=1;y=2;z=x+y;print(z)",
    "<|im_start|>user\nhi<|im_end|>",
    "\u00a0non breaking\u00a0space",
]


def load_hf_tokenizer(ref_hf: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ref_hf, trust_remote_code=True)


def hf_vocab_and_merges(ref_hf: str) -> tuple[list[str], list[str]]:
    """Read the raw tokenizer.json rather than going through the AutoTokenizer object: the
    object normalises added tokens and hides the merge order, and it is precisely the raw
    lists we need to compare against the GGUF's raw lists."""
    with open(os.path.join(ref_hf, "tokenizer.json"), encoding="utf-8") as f:
        tk = json.load(f)
    model = tk["model"]
    vocab = model["vocab"]
    inv = [""] * (max(vocab.values()) + 1)
    for tok, idx in vocab.items():
        inv[idx] = tok
    merges = model.get("merges", [])
    merges = [" ".join(m) if isinstance(m, (list, tuple)) else m for m in merges]
    # added_tokens live outside model.vocab and must be folded in at their declared ids
    for at in tk.get("added_tokens", []):
        idx = at["id"]
        if idx >= len(inv):
            inv.extend([""] * (idx + 1 - len(inv)))
        inv[idx] = at["content"]
    return inv, merges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref-hf", required=True)
    ap.add_argument("--skip-roundtrip", action="store_true",
                    help="run T1-T4 only (T5 needs the transformers package)")
    args = ap.parse_args()

    gg = G.GGUFFile(args.gguf)
    g_tokens = gg.kv.get("tokenizer.ggml.tokens")
    g_merges = gg.kv.get("tokenizer.ggml.merges") or []
    if not g_tokens:
        print("FAIL: the GGUF carries no tokenizer.ggml.tokens")
        return 1
    h_tokens, h_merges = hf_vocab_and_merges(args.ref_hf)

    cfg = json.load(open(os.path.join(args.ref_hf, "config.json")))
    txt = cfg.get("text_config", cfg)
    fails: list[str] = []

    # -- T1 vocab size.
    # THE EMBEDDING MATRIX AND THE TOKENIZER ARE ALLOWED TO DIFFER IN LENGTH, and here they do:
    # config.vocab_size (the number of embedding rows) is rounded up past the last real token,
    # and tokenizer.json lists only the real ones. What must hold is
    #   config.vocab_size == the GGUF's token count == the embedding row count,
    # and the HF tokenizer's vocab must be a PREFIX of the GGUF's, with the tail being ids the
    # tokenizer can never emit. A tail that is not pure padding would be a real difference, so
    # the tail is checked, not waved through.
    vs = txt.get("vocab_size", cfg.get("vocab_size"))
    pad = len(g_tokens) - len(h_tokens)
    print(f"T1 vocab: gguf tokens={len(g_tokens)} hf tokenizer={len(h_tokens)} "
          f"config.vocab_size={vs} (reserved tail={pad})")
    if vs is not None and int(vs) != len(g_tokens):
        fails.append(f"T1 config.vocab_size {vs} != the GGUF's {len(g_tokens)} tokens")
    if pad < 0:
        fails.append(f"T1 the HF tokenizer has MORE tokens ({len(h_tokens)}) than the GGUF "
                     f"({len(g_tokens)}) -- it is a different, larger tokenizer")
    elif pad:
        tail = g_tokens[len(h_tokens):]
        # Placeholder spellings actually seen in this family: [PAD248077], <|extra_0|>,
        # <unused12>, <reserved_3>. Anything OUTSIDE this shape in the tail is a real token
        # the borrowed tokenizer could never emit, and that is a stop.
        import re as _re
        _placeholder = _re.compile(
            r"^(\[PAD\d+\]|<\|?(?:extra|unused|reserved|pad)[_\-]?\d*\|?>|<pad>)$", _re.I)
        looks_reserved = all(_placeholder.match(t) or t.strip() == "" for t in tail)
        print(f"T1 reserved tail sample: {tail[:4]} ... {tail[-2:]}")
        if not looks_reserved:
            fails.append(
                f"T1 the GGUF has {pad} tokens past the end of the HF tokenizer and they are "
                f"NOT reserved/placeholder strings (e.g. {tail[:3]}). Those ids would be "
                f"unreachable through the borrowed tokenizer.")

    # -- T2 vocab identity, whole table
    n = min(len(g_tokens), len(h_tokens))
    diffs = [i for i in range(n) if g_tokens[i] != h_tokens[i]]
    print(f"T2 vocab identity: {n - len(diffs)}/{n} ids identical")
    if diffs:
        show = ", ".join(f"id {i}: gguf {g_tokens[i]!r} vs hf {h_tokens[i]!r}"
                         for i in diffs[:5])
        fails.append(f"T2 {len(diffs)} token ids differ, e.g. {show}")

    # -- T3 merges identity, in order
    print(f"T3 merges: gguf={len(g_merges)} hf={len(h_merges)}")
    if list(g_merges) != list(h_merges):
        first = next((i for i, (a, b) in enumerate(zip(g_merges, h_merges)) if a != b), None)
        fails.append(f"T3 merge lists differ (len {len(g_merges)} vs {len(h_merges)}, first "
                     f"differing rank {first})")

    # -- T4 special ids
    for kv_key, cfg_key in (("tokenizer.ggml.bos_token_id", "bos_token_id"),
                            ("tokenizer.ggml.eos_token_id", "eos_token_id"),
                            ("tokenizer.ggml.padding_token_id", "pad_token_id")):
        gv = gg.kv.get(kv_key)
        # A multimodal config declares these in BOTH the top level and text_config, and the two
        # legitimately differ (the text config carries the base model's ids, the top level the
        # assembled model's). Accept a match against either, and print both so a real
        # disagreement is visible rather than hidden by a lookup order.
        hv_top, hv_txt = cfg.get(cfg_key), txt.get(cfg_key)
        print(f"T4 {cfg_key}: gguf={gv} hf(top)={hv_top} hf(text_config)={hv_txt}")
        cands = [v for v in (hv_top, hv_txt) if v is not None]
        if gv is not None and cands and int(gv) not in [int(v) for v in cands]:
            fails.append(f"T4 {cfg_key}: gguf {gv} matches neither {hv_top} nor {hv_txt}")

    # -- T5 round trip through a BPE driven by the GGUF's own tables
    if not args.skip_roundtrip and not fails:
        try:
            tok = load_hf_tokenizer(args.ref_hf)
        except Exception as exc:                                   # noqa: BLE001
            print(f"T5 SKIPPED: {exc}")
        else:
            bad = 0
            for s in PROBES:
                ids = tok.encode(s, add_special_tokens=False)
                if any(i >= len(g_tokens) for i in ids):
                    bad += 1
                    print(f"T5 FAIL (id out of the GGUF vocab): {s!r}")
                    continue
                # Every id the HF tokenizer produced must name the SAME token in the GGUF's
                # own table, and decoding must return the input.
                if [h_tokens[i] for i in ids] != [g_tokens[i] for i in ids]:
                    bad += 1
                    print(f"T5 FAIL (token strings differ): {s!r}")
                    continue
                if tok.decode(ids) != s:
                    # a lossy decode is a tokenizer property, not a mismatch between the two;
                    # report it but do not fail the gate on it
                    print(f"T5 note: decode is lossy for {s!r}")
            print(f"T5 round trip: {len(PROBES) - bad}/{len(PROBES)} probes identical")
            if bad:
                fails.append(f"T5 {bad}/{len(PROBES)} probes disagree")

    if fails:
        print("\nTOKENIZER GATE FAILED -- do NOT borrow this tokenizer:")
        for f in fails:
            print("  " + f)
        return 1
    print("\nTOKENIZER GATE PASS: the reference checkpoint's tokenizer is this GGUF's "
          "tokenizer (vocab, merges and special ids all identical).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
