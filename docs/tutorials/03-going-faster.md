# 03 — Going faster without changing the answers

There is a trick in here that gives you more tokens per second for free. This page explains what
it is, what it costs, and exactly how honest I am being when I call it "lossless".

---

## 1. The idea, in plain words

Writing an answer is slow because the model produces one token at a time, and each token needs
the whole model to run once. Reading is fast, because the model can chew a lot of text in one
pass.

Speculative decoding exploits that gap:

1. Something small and quick **guesses** the next few tokens.
2. The big model **checks all the guesses in one pass** — which costs about as much as producing
   one token normally would.
3. Every guess it agrees with is kept. The first one it disagrees with is replaced with the big
   model's own choice, and the rest are thrown away.

If the guesser is right four times out of five, you got several tokens for the price of one. If
it is wrong every time, you paid a little extra for nothing and the answer is unchanged.

**The answer is always the big model's.** A guess is only kept if the big model would have
produced it anyway. That is the whole point of the check.

---

## 2. What is on by default

By default the engine arms a cheap guesser that costs almost nothing: a **pattern matcher** that
watches what has already been written and proposes a continuation when it spots a repeat. You do
not configure it. You will see it decide at startup:

```
PXA_AUTO: spec VRAM gate OK -- dev0 free 16006 MiB, model share 10175 MiB -> headroom 5831 MiB
          (at or above the 2048 MiB a draft context needs)
PXA_AUTO: spec arch=qwen35moe -> --spec-type ngram-mod:n_max=4,n_min=2 (... drafts only on an
          n-gram match so it costs ~nothing when it cannot predict ...)
```

or, on a card with less room to spare:

```
PXA_AUTO: spec DECLINED -- only dev0 free <N> MiB ... under the 2048 MiB a draft context needs
          (override with --spec-type, or PXA_AUTO_SPEC=1 to force)
```

Both of those are a healthy startup. It needs about 2 GB of spare card memory to arm; if it
declines, it is because your model filled the card, which is not a fault.

**It is very good at repetitive text and nearly useless on fresh prose.** Editing code, filling
in a template, rewriting a document you just pasted — those are where it shines, because the
answer contains chunks of the question.

You can see it working in any reply's `timings`:

```bash
curl -s http://127.0.0.1:8080/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a short paragraph about how rain forms.","n_predict":80,"temperature":0}' \
  | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print({k:v for k,v in t.items() if "draft" in k})'
```

```
{'draft_n': 4, 'draft_n_accepted': 3}
```

`draft_n` is how many tokens were guessed, `draft_n_accepted` how many survived the check. If
both keys are missing, no guesser was armed for that request.

**What just happened.** Nothing you did. The engine looked at your model and your free VRAM,
decided a cheap drafter was affordable, armed it, and told you so.

---

## 3. The opt-in: a trained guesser (MTP)

Some models ship with a small extra piece trained alongside them whose whole job is to guess the
next token. It is called an **MTP head**, and it is a far better guesser than pattern-matching
because it actually understands the sentence.

In this release using it is **opt-in**. You turn it on yourself:

```bash
PXA_SPEC_POLICY=1 ./run-server.sh -m your-model.gguf -ngl 99 -c 8192 -fa on
```

The server prints what it resolved at startup. To turn it off again, drop the variable — there is
nothing to undo.

**Why it is not on by default yet, honestly.** It measured well on my cards and it is not
finished. It changes how much card memory is reserved, it adds a noticeable pause on the very
first request while it warms up, and it has not been through the long multi-user soak test that
anything default here has to pass. A release where nothing you already run gets worse is worth
more to me than a headline. It is first in the queue for the next one.

**Try it if**: your model file actually has an MTP head (the startup log says so — look for a
`mtp` or `nextn` tensor count above zero), you use one conversation at a time, and you are
willing to check the answers.

**Do not bother if**: you serve several people at once. That is the case it has not been soaked
in.

---

## 4. What "lossless" means, precisely

This is the part where other people wave their hands. I will not.

**What I claim:** the acceptance check never emits a token the big model did not itself choose.
Every token in your answer is the big model's own pick. The drafter cannot put a word in its
mouth.

**What I do not claim:** that the text is byte-for-byte identical to the same request run with
the drafter switched off.

Here is why, and it is not weaselling. When the big model checks four guesses at once it runs a
slightly different set of arithmetic kernels than when it produces one token at a time — wider
ones, doing more at once. Floating-point arithmetic is not associative, so those two routes can
disagree in the last bit. When two candidate tokens are almost exactly tied, that last bit is
enough to flip which one wins. So an answer with speculation on can diverge from one with it off,
at a point where the model genuinely had no preference.

That is a property of batched checking itself, not of my implementation — mainline llama.cpp has
it too. It means:

- The answer is a legitimate answer from your model. **Use it.**
- Two runs of the same prompt with the same settings still agree with each other, at temperature
  0 — see the note below for the one case where even that is not quite true.
- If you need byte-identical output against a non-speculative reference — you are comparing
  quantization tiers, say — turn the drafter off for that comparison with `--spec-type none`,
  and turn it back on afterwards.

**One more honest edge, new to this page.** With the pattern-matching table specifically, the
same request sent twice to the same otherwise-untouched server can occasionally come back with
two different valid answers, because which pattern happens to match depends on what the table
still holds from the last request. This is rare and it is the same "legitimate answer, not
necessarily the same one twice" property above, one step further. `--spec-type none` gives you a
server whose answers are reproducible against themselves as well as against an unspeculated run.

---

## 5. Temperature, and the one thing to know

At `temperature: 0` — the repeatable setting — the check is a plain exact match: a guess is kept
only if it is exactly the token the big model picked.

Above temperature 0 the model is sampling rather than picking a winner, and "did it match" is a
harder question. This release keeps the acceptance rule the previous release used. It is a
**fast** rule, not an exact one: it accepts a guess that lands among the candidates the big model
was choosing between and was plausible enough, rather than only the one it actually drew. It
raises the hit rate, and at temperature above 0 the text can legitimately differ from a
drafter-off run in more than just near-ties.

If that matters to you, you have two off-switches:

```bash
# keep the fast drafter but demand an exact match
PXA_SPEC_RELAXED=0 ./run-server.sh -m your-model.gguf -ngl 99 -c 8192 -fa on

# or just run at temperature 0, where the exact rule applies anyway
```

There is also a **new, genuinely lossless** sampling rule in this release, opt-in and labelled
new:

```bash
PXA_SPEC_SAMPLED=1 ./run-server.sh -m your-model.gguf -ngl 99 -c 8192 -fa on
```

It makes the drafter draw its guesses the same way the big model would and accepts them with a
probability that makes the emitted token come out with exactly the big model's own distribution.
It is proven correct on paper and in unit tests; it is opt-in because a review found two bugs in
its plumbing that are still being fixed, and I would rather you met it when it is finished. The
server prints one line at startup when it is on.

---

## 6. Why the first request after a start is slow

Three things happen on the first request that never happen again:

1. The model file is still being read off disk into card memory. A 14 GB file from a spinning
   disk takes as long as a 14 GB copy does.
2. The engine builds and reserves its computation plan for the shapes it is about to see.
3. If a drafter is armed, it warms up too.

So your first answer is slower, sometimes much slower, and then everything settles. This matters
enormously when you go to measure anything — see [guide 09](09-measure-your-card.md), where
throwing away the first run is rule one.

If you want the settling to happen before a person is waiting, send one throwaway request right
after the server reports it is listening.

---

## 7. Things you do not need to do

The engine already picks, per card, which of its GPU tricks are worth using, and prints the list
at startup as a block of `PXA_AUTO:` lines. You do not need to copy any of them onto your command
line. If a page somewhere tells you to export a variable to get a speed I quote, that page is
older than this one.

Two escape hatches exist, and both are for debugging rather than speed:

| | |
|---|---|
| `PXA_ENHANCE=0` | fall back to the plainer, pre-2026-09-03 set of choices |
| `PXA_REFERENCE=1` | every trick off, the slow reference path. Useful when you suspect a bug and want to know if a trick caused it |

---

## Where next

- Measure the difference honestly → [09](09-measure-your-card.md)
- Why the *second* turn of a chat is instant → [08](08-long-chats.md)
- Turning any of this off again → [10](10-when-something-goes-wrong.md)
