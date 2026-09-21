# 08 — Long chats and many conversations

Why your second message gets an answer almost instantly, why the fifteenth one still does, and
the one thing that throws it all away.

---

## 1. What the prompt cache does

Every time you send a message, the server does not just read your message — it reads the **whole
conversation**, from the system prompt onwards, because that is what the model needs in front of
it. A twenty-turn chat means re-reading twenty turns.

That would get slower and slower, so it does not do that. After each request it **keeps the
conversation's working state in card memory**. On your next message it compares the new
conversation to the one it already has, finds how much of the beginning is identical, and reads
only the new part.

Turn one reads everything. Turn two reads your new sentence and nothing else.

---

## 2. See it for yourself

Start a server with room for a document:

```bash
./run-server.sh -m your-model.gguf -ngl 99 -c 16384 -fa on
```

Now send a long prompt, then a follow-up that keeps everything and adds one question. The number
to watch is `prompt_n` — how many tokens it actually had to read — and `prompt_ms`, the time
before the first word of the answer appears.

```bash
curl -s http://127.0.0.1:8080/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Here is a document:\n<paste a few pages here>\n\nQuestion: summarise it in one line.\nAnswer:","n_predict":24,"temperature":0,"cache_prompt":true}' \
  | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print("read %d tokens, first token after %.0f ms" % (t["prompt_n"], t["prompt_ms"]))'
```

```
read 3522 tokens, first token after 2860 ms
```

Now the follow-up — the same document, the answer it gave, and a new question on the end:

```bash
curl -s http://127.0.0.1:8080/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Here is a document:\n<the same pages>\n\nQuestion: summarise it in one line.\nAnswer: A list of words.\n\nQuestion: what is the first word?\nAnswer:","n_predict":24,"temperature":0,"cache_prompt":true}' \
  | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print("read %d tokens, first token after %.0f ms" % (t["prompt_n"], t["prompt_ms"]))'
```

```
read 22 tokens, first token after 284 ms
```

It read 22 tokens instead of 3522. That is the cache.

**What just happened.** `cache_prompt: true` — which is on by default in every chat app and in
the `/v1/chat/completions` endpoint — tells the server to reuse what it already has. It matched
the first 3500 tokens of your new prompt against what was in memory, kept them, and read only the
tail. Nothing was recomputed that did not need to be.

---

## 3. The one case that is still slow

**Editing something earlier in the conversation.** If you change your very first message, or the
system prompt, or a document you pasted three turns ago, then everything after the edit is
different too — and the cache can only keep a prefix that is byte-for-byte identical.

Same server, same follow-up question, but five characters inserted at the *start* of the
document:

```
read 3541 tokens, first token after 2580 ms
```

Straight back to a cold read. The cache did not fail; there was simply nothing valid to keep.

This is why "regenerate with a tweak to my first message" feels so much slower than "ask another
question", and it is not something I can optimise away — the model genuinely has to read the
changed text.

### What to do about it

- **Put the stable things first.** System prompt, then the document, then the conversation. Never
  put a timestamp, a random ID or a "turn 7 of 20" counter at the top of a prompt — it changes
  every turn and destroys the cache every turn.
- **Append, don't rewrite.** If you want to correct yourself, add a new message saying so rather
  than editing the old one. It reads a sentence instead of a book.
- **Branching a conversation is cheap; editing its root is not.** Two follow-ups from the same
  point both reuse the same cached prefix.
- **If you must edit, edit as late as possible.** Everything before the edit is still reusable.

---

## 4. Several conversations at once

One server can hold more than one conversation, each in its own **slot**. Ask for slots with
`--np`:

```bash
./run-server.sh -m your-model.gguf -ngl 99 -c 32768 -fa on --np 4
```

Two things to understand about that number:

1. **`-c` is the total, not the amount each conversation gets.** `-c 32768 --np 4` means four
   conversations of 8192 tokens each. If you want four conversations of 32k, you need `-c 131072`
   and the memory to back it.
2. **Each slot keeps its own cache.** Four people chatting get four fast follow-ups. A fifth
   person's conversation displaces someone's cache, and that someone's next message is a cold
   read again.

For one or two users, leave `--np` alone — the default of one slot is the lowest-latency
configuration and it is what I tune for. If you are serving a group, look at
[guide 06](06-run-the-vllm-sidecar.md): the sidecar is built for exactly that and handles it far
better than this server does.

**One thing worth knowing about two or more slots on this release**: a genuinely close call in
the model's own answer can resolve slightly differently between slots — not one slot's content
leaking into another's, just an ordinary near-tie landing on either side depending on the exact
shape of the batch each slot happened to share the card with. It is rare, and if you need
byte-repeatable answers across slots, compare them one at a time rather than side by side. See
[guide 10](10-when-something-goes-wrong.md) if you want the full explanation.

> **If it went wrong**
> - **Every message is a cold read, even simple follow-ups** — your client is probably sending
>   `cache_prompt: false`, or inserting something that changes each turn (a timestamp, a
>   randomised system prompt) at the top. Check the raw request your app sends.
> - **It was fast and then suddenly got slow for everyone** — more conversations than slots. Each
>   new one evicts an old one's cache.
> - **The conversation gets cut off at the start after a while** — you have hit `-c`. The oldest
>   turns are dropped to make room. Raise `-c` if the memory allows, or start a fresh
>   conversation.
> - **You want the cache to survive a restart** — it does not, by design. It lives in card
>   memory. The first message after a restart is always a cold read.

---

## 5. Why the very first request of all is slowest

Separate from the cache: the first request after starting the server also pays for loading
weights off disk, building the computation plan, and warming up the drafter. See
[guide 03](03-going-faster.md) section 6. Send one throwaway request after startup and nobody
ever notices.

---

## Where next

- Measuring any of this properly → [09](09-measure-your-card.md)
- Serving a group → [06](06-run-the-vllm-sidecar.md)
- Running out of memory when you raise `-c` → [02](02-pick-settings-for-your-cards.md)
