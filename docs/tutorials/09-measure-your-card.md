# 09 — Measure your own card honestly

If you are going to post a number — on Discord, on Reddit, in a bug report — this page is how to
get one that survives someone checking it.

I am not being preachy. Every trap below is one **I** fell into on my own hardware, and one of
them put wrong numbers on a public page until I caught it. They are easy to fall into because
each one makes the machine look *better*, and nothing warns you.

---

## The three traps

### Trap 1 — sending the same prompt twice

This is the big one.

The engine arms a drafter that learns from text it has already seen. Send the same prompt three
times and by the third run it is partly replaying its own earlier answer, which is enormously
faster than generating one. Your three runs look like this:

```
-- same prompt, three times --
  run 1   X tokens/s
  run 2   about 1.45 x X
  run 3   about 1.65 x X
```

Numbers that climb run over run are the signature. That is not your card warming up. That is you
measuring the cache.

With a **different** prompt each time, the same server on the same card is much flatter:

```
-- a fresh prompt each time --
  run 1   X tokens/s
  run 2   about 1.1 x X
  run 3   about 1.17 x X
```

(There is still a small rise there. That is a real warm-up ramp — see trap 4 below — and it is
why you throw the first run away.)

**The rule: every repeat gets a prompt it has never seen, and `cache_prompt: false`.** The
simplest way is to salt it with a random number:

```bash
python3 -c 'import json,random,sys; random.seed(int(sys.argv[1])); print(json.dumps({"prompt":"Write a detailed technical summary of how a GPU executes a matrix multiply. Reference case %d." % random.randrange(999999),"n_predict":200,"temperature":0,"cache_prompt":False}))' 1 > /tmp/req.json
```

I got this wrong in my own benchmark harness for weeks. Both engines in the comparison were
being measured on their own replay, so the comparison was not even wrong in a useful direction —
it was just meaningless. I had to correct a set of already-published numbers because of exactly
this trap; see the release notes for what changed.

### Trap 2 — "I did not pass a speculation flag, so speculation is off"

It is not. The engine arms a cheap drafter **by itself** when the card has about 2 GB spare,
because that is the setting I want people to have. Not passing a flag is not the same as turning
it off.

Look at your startup log:

```
PXA_AUTO: spec arch=... -> --spec-type ngram-mod:n_max=4,n_min=2 (...)
```

That line means a drafter is armed. If instead it says `spec DECLINED`, none is.

And check any individual reply:

```bash
curl -s http://127.0.0.1:8080/completion -H 'Content-Type: application/json' --data @/tmp/req.json \
  | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print({k:v for k,v in t.items() if "draft" in k})'
```

```
{'draft_n': 4, 'draft_n_accepted': 3}
```

Keys present means it drafted. Keys absent means it did not.

To actually turn it off for a comparison:

```bash
./run-server.sh -m your-model.gguf -ngl 99 -c 8192 -fa on --spec-type none
```

**Say which one you measured.** "47 tokens/s" means nothing on its own. "47 tokens/s, drafter
off" and "63 tokens/s, default drafter, prose prompt" are both useful and they are different
numbers about the same machine. If you turned on `-sm tensor` or any other opt-in lever, say that
too — see [guide 02](02-pick-settings-for-your-cards.md) for what is opt-in this release.

### Trap 3 — comparing a run with instrumentation against one without

If you turn on timing or profiling to understand *why* something is slow, that switch costs
speed. A number taken with it on is not comparable to one taken with it off, even by a little.

Pick one setting, take **every** arm of the comparison under it, and say which it was. The same
goes for anything else that differs between two runs: a different context length, a different
`-ub`, a background process on the card, a different build. One variable at a time.

### Trap 4 — the warm-up ramp (a bonus, and it fooled me too)

A card that has just loaded a model gets faster over the first minute or so, then settles. If
you take your reading in the first thirty seconds you get a low number; take it a minute later
and it is higher, and nothing you changed caused that.

So: **let it run for a minute before the reading you keep, and throw the first request away.**

A useful sanity check: measure once at the start of your session and once at the end. If those
two disagree, the thing you measured in between is drift, not your change. And note which way the
drift goes — a card that is *contended* by something else gets slower, not faster, so a rising
trend is warm-up and a falling one is a neighbour stealing the card.

---

## The recipe

Start the server and leave it alone for a minute.

```bash
./run-server.sh -m your-model.gguf -ngl 99 -c 8192 -fa on
```

**Decode speed** — how fast it writes. Five fresh prompts, take the median, throw the first away:

```bash
for i in 1 2 3 4 5; do
  python3 -c 'import json,random,sys; random.seed(int(sys.argv[1])); print(json.dumps({"prompt":"Write a detailed technical summary of how a GPU executes a matrix multiply. Reference case %d." % random.randrange(999999),"n_predict":200,"temperature":0,"cache_prompt":False}))' "$i" > /tmp/req.json
  curl -s http://127.0.0.1:8080/completion -H 'Content-Type: application/json' --data @/tmp/req.json \
  | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print("decode %.1f t/s   first token %.0f ms" % (t["predicted_per_second"], t["prompt_ms"]))'
done
```

```
decode  ... t/s   first token  ... ms
decode  ... t/s   first token  ... ms
...
```

**Prefill speed** — how fast it reads. One long, cold prompt. Short prompts cannot produce a
meaningful prefill rate, because the fixed per-request cost swamps everything:

```bash
python3 -c 'import json,random; random.seed(11); print(json.dumps({"prompt":" ".join(f"item{random.randrange(9999)}" for _ in range(1800)),"n_predict":8,"temperature":0,"cache_prompt":False}))' > /tmp/long.json
curl -s http://127.0.0.1:8080/completion -H 'Content-Type: application/json' --data @/tmp/long.json \
  | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print("prompt_n %d   prefill %.1f t/s" % (t["prompt_n"], t["prompt_per_second"]))'
```

Change the seed between repeats so no two prompts are ever the same, and adjust `range(1800)`
until `prompt_n` lands somewhere in the low thousands.

**What just happened.** `timings.predicted_per_second` is the server's own measurement of how
fast it generated, and `timings.prompt_per_second` of how fast it read. They come from the server,
not from your stopwatch, so network and JSON parsing are not in them.

---

## What to post with the number

A number with no context cannot be reproduced, which means it cannot help anyone — including
you, next month, wondering whether you made things better.

Post all of this:

1. **The card**, from `nvidia-smi -L`, and how many of them.
2. **The model file**, by its exact name, and its tier.
3. **The full command** you started the server with, including any opt-in flag like `-sm tensor`.
4. **Whether a drafter was armed** — paste the `PXA_AUTO: spec ...` line.
5. **Prefill or decode**, and the prompt length. They are different numbers.
6. **How many runs, and that each had a fresh prompt.** Median, not best.
7. **The build**, from `./run-server.sh --version`.

That is seven lines and it turns "it feels slow" into something I can actually chase.

---

## Comparing two things fairly

Whether you are comparing two tiers, two split modes, or this engine against another:

- **Same card, same session, back to back.** Not yesterday's number against today's.
- **Same prompts.** Generate the set once, save it, replay the same files at both arms.
- **Same measurement.** Same context, same batch settings, same drafter state.
- **Bracket it.** Measure arm A, then B, then A again. If the two A readings disagree, something
  moved underneath you and the A-vs-B difference is not trustworthy. This one check has saved me
  more wasted hours than anything else on this page.
- **Say what you could not hold constant.** There is always something. Naming it is what makes
  the number honest.

---

## Where next

- Why the first request is slow → [03](03-going-faster.md)
- Why a follow-up is fast → [08](08-long-chats.md)
- Posting a result, or asking why yours is low → [10](10-when-something-goes-wrong.md)
