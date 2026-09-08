"""render_chat_prompts.py -- render the 70 gate prompts through the model's OWN chat template.

WHY. The first correctness run fed the 70 prompts to both engines as BARE COMPLETIONS. The
model is instruct-tuned, so on a raw "...Now begin your answer:" continuation its most likely
next token is the end-of-turn marker: measured top-2 gaps of 0.16-1.35 nats with <|im_end|>
ranked #1 or #2 on every single prompt, 28 of 70 empty from BOTH engines, and a first-token
agreement of 14/70 that was measuring how two engines break a 0.2-nat tie rather than whether
they decode the same weights.

Rendering through the chat template puts the model in the state it was trained for, so the
argmax at token 1 is an answer rather than a stop. It also removes the last variable between
the engines: the rendered strings are written to a file ONCE and both engines are then fed the
identical bytes through /v1/completions, so neither engine's own chat handling, BOS policy or
template version can differ. That is deliberate -- comparing two engines' chat endpoints would
be comparing their template code, not their kernels.

    python3 render_chat_prompts.py --ref-hf <checkpoint> --out prompts-chat.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toptoken70 import gen_prompts        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-hf", required=True,
                    help="checkpoint dir holding tokenizer_config.json / chat_template.jinja")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=70)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ref_hf, trust_remote_code=True)

    raw = gen_prompts(args.n)
    rendered = []
    for p in raw:
        # add_generation_prompt=True is the whole point: it appends the assistant turn header,
        # so the model's next token is the first token of an ANSWER.
        rendered.append(tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True))

    # Prove the rendering actually did something and is uniform, rather than assuming it.
    if any(r == p for r, p in zip(rendered, raw)):
        raise SystemExit("apply_chat_template returned the input unchanged for at least one "
                         "prompt -- this checkpoint has no usable chat template")
    # Uniformity, checked as an actual longest-common-prefix/suffix rather than a fixed
    # window: the user text starts within a few dozen characters of the template header, so a
    # fixed-width comparison measures the prompts, not the template.
    def lcp(strs):
        a, b = min(strs), max(strs)
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return a[:i]

    head = lcp(rendered)
    tail = lcp([r[::-1] for r in rendered])[::-1]
    if not head:
        raise SystemExit("the rendered prompts share no common prefix -- the template is not "
                         "being applied uniformly")
    # The generation prompt is the whole point: without a common trailing assistant header the
    # model is not being asked to answer, and the gate would measure the same thing the bare
    # completions did.
    if not tail.strip():
        raise SystemExit("the rendered prompts share no common trailing text, so "
                         "add_generation_prompt produced nothing -- the model would not be in "
                         "an answering state and this gate would measure tie-breaking again")

    json.dump({"source": "toptoken70.gen_prompts", "ref_hf": args.ref_hf,
               "n": len(rendered), "prompts": rendered}, open(args.out, "w"))
    print(f"wrote {args.out}: {len(rendered)} prompts")
    print(f"  common template header ({len(head)} chars): {head!r}")
    print(f"  common generation prompt ({len(tail)} chars): {tail!r}")
    print(f"  mean chars {sum(len(r) for r in rendered) // len(rendered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
