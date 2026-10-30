#!/usr/bin/env python3
"""pxa bench -- a shareable, ~2-minute result card for a running (or freshly booted) PXA seat.

2026-09-25. Off the hot path: this tool only ever POSTs to /completion and GETs
/health, /props and /pxa/explain -- it never changes a flag the engine decides with, and its own
requests use greedy decoding (temperature 0) so they cannot perturb anything else on the seat
beyond the tokens they themselves ask for.

Two ways to run it:
  pxa-bench.py --url http://127.0.0.1:8080          # bench an already-running seat (primary path)
  pxa-bench.py --model FILE --gpus 0,1               # boot one via pxa-launch.py, bench it, stop it
  pxa-launch.py --bench --model FILE --gpus 0,1       # the same, as `pxa-launch --bench` (a thin
                                                        # argv pass-through -- see pxa-launch.py)

Method (owner rule, "probe speed before gates" / "trim tests, lever vs board"): 60s warm-up, REPS 3,
three fixed prompt classes (prose / edit / long -- the same vocabulary the release board uses), one
greedy512 identity check (temperature 0, seed 0, sha256 of the completion). Prints a markdown result
card to stdout and to a .md file, and writes an SVG badge (always) plus a PNG badge (if Pillow is
importable; skipped with a clear note otherwise -- there is no hard new dependency here).

`--selftest` exercises the renderer and helpers on canned sample data -- no network, no GPU. Wired
into CTest as test-pxa-bench-render (tests/CMakeLists.txt).
"""

import argparse
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from xml.sax.saxutils import escape

# ---------------------------------------------------------------------------------------------
# Fixed prompts, three classes (the release board's own vocabulary, mirrored here in spirit but not
# by reading any other file: pxa-bench.py ships to end users and must not depend on local paths). "prose" is generation-heavy (the hard case for any drafter); "edit" is mostly-copy (an
# agent/code-edit shape); "long" exercises prefill with a multi-hundred-token document.
# ---------------------------------------------------------------------------------------------
_LONG_PARAGRAPH = (
    "The river below the ridge carries meltwater past a dozen small settlements before it widens "
    "into the delta where the old road finally meets the coast. Each spring the survey crews walk "
    "the embankments, noting where the current cut a new channel and where the old one silted in, "
    "and each autumn the same crews return to see which of their notes held. "
)

CLASS_PROMPTS = {
    "prose": {
        "prompt": (
            "Explain in one paragraph how a speculative decoding drafter and a verifier model "
            "cooperate to make text generation faster without changing the output distribution."
        ),
        "n_predict": 64,
    },
    "edit": {
        "prompt": (
            "Return the function below unchanged except rename the parameter `rows` to `records` "
            "everywhere it appears, including inside the body. Output only the code.\n\n"
            "def summarise(rows):\n"
            "    total = {}\n"
            "    for row in rows:\n"
            "        key = row.get(\"category\", \"other\")\n"
            "        total[key] = total.get(key, 0) + row.get(\"value\", 0)\n"
            "    return total\n"
        ),
        "n_predict": 96,
    },
    "long": {
        "prompt": (_LONG_PARAGRAPH * 26) +
                  "\nQuestion: in one sentence, what does the passage above mostly describe?\nAnswer:",
        "n_predict": 48,
    },
}

GREEDY_PROMPT = (
    "Write a short, factual paragraph about how meltwater rivers shape the deltas they eventually "
    "reach, starting with the ridge line where they begin."
)
GREEDY_N = 512

# The warm-up never uses a measured prompt (FR#12, todo control-bench-honest): a drafter that learns
# from what it generated (an n-gram table, ngram-mod, a suffix tree) would otherwise have seen the
# measured answer already, and on a server that refuses the slot erase run 1 would be replay.
WARMUP_PROMPT = (
    "List five everyday objects made mostly of copper and say in a few words why copper is used "
    "for each one."
)

# A later run this much faster than run 1 is flagged as replay (FR#12 saw >2x); on an honest bench
# every run is a first pass and they differ by noise (a few percent).
REPLAY_SUSPECT_RATIO = 1.15

_BADGE_COLORS = {"prose": "#2f6fed", "edit": "#2fa84f", "long": "#a259e6"}


# ---------------------------------------------------------------------------------------------
# HTTP helpers (stdlib only -- no `requests`, so this tool has zero hard third-party dependency)
# ---------------------------------------------------------------------------------------------
def _get_json(url, path, timeout=15):
    req = urllib.request.Request(url.rstrip("/") + path)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_json(url, path, body, timeout=600):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + path, data=data,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_health(url, timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            _get_json(url, "/health", timeout=5)
            return True
        except Exception:
            time.sleep(2)
    return False


# ---------------------------------------------------------------------------------------------
# Cards: prefer GET /pxa/explain (this release's own page -- see examples/server/pxa-explain.h);
# fall back to `nvidia-smi` for an engine build that predates it, so pxa-bench.py still works
# against an older binary, just without the registry's picks/why.
# ---------------------------------------------------------------------------------------------
def get_cards_and_registry(url):
    """-> (cards, engine{build,commit}, model_tier, model_alias, err|None)."""
    try:
        j = _get_json(url, "/pxa/explain", timeout=10)
    except Exception as e:
        return [], {}, None, "", str(e)
    cards = j.get("cards") or []
    engine = j.get("engine") or {}
    model_tier = (j.get("model") or {}).get("tier") or None
    model_alias = j.get("model_alias") or ""
    return cards, engine, model_tier, model_alias, None


def cards_from_nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except Exception:
        return []
    cards = []
    for line in (out.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx, name, cc, total, used = parts[0], parts[1], parts[2], parts[3], parts[4]
            cc10 = int(round(float(cc) * 10))
            total_m, used_m = int(float(total)), int(float(used))
            cards.append({"index": int(idx), "name": name, "sm": cc10,
                          "vram_total_mib": total_m, "vram_used_mib": used_m,
                          "vram_free_mib": max(0, total_m - used_m)})
        except (ValueError, IndexError):
            continue
    return cards


def card_class_from_name(name, sm=None):
    n = (name or "").upper()
    for tag in ("P100", "V100", "A100", "H100", "1080 TI", "1080TI", "3090", "4090", "P40", "T4"):
        if tag in n:
            return tag.replace(" ", "")
    if sm:
        return "sm_%d" % int(sm)
    return "unknown"


def cards_line(cards):
    if not cards:
        return "(no cards reported -- neither /pxa/explain nor nvidia-smi answered)"
    counts = Counter((c.get("name") or "?", c.get("sm")) for c in cards)
    return ", ".join("%dx %s (sm_%s)" % (n, name, sm) for (name, sm), n in counts.items())


# ---------------------------------------------------------------------------------------------
# Model file identity
# ---------------------------------------------------------------------------------------------
def model_sha256(path, chunk=1 << 24):
    if not path or not os.path.isfile(path):
        return None, "file not found (pass --model, or run this on the box that holds it)"
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
    except OSError as e:
        return None, "unreadable (%s)" % (e.strerror or e)
    return h.hexdigest(), "computed"


# ---------------------------------------------------------------------------------------------
# Measurement: warm-up (owner rule -- 60s before the open bracket), REPS x class, greedy512.
# ---------------------------------------------------------------------------------------------
# A run that ends before it decodes has no decode rate (#361/#362, 2026-10-26): the server times the first token with the
# prompt, so a reply of ONE token reads 1 token / 1 us = 1,000,000 t/s. A chat model given a raw-text bench prompt can end at
# once (Swift-1.5 Flash-Next answered the "edit" and "long" prompts with <|endoftext|> / <|im_end|> as its first token, and
# both classes read 1,000,000 t/s). Such a run is asked again with ignore_eos (one retry), so the class times n_predict
# tokens; a run that still decodes fewer than MIN_DECODE_TOKENS reports 0 (no number) instead of 1e6. A run that does not
# end early sends the same request as before, so every earlier number is unchanged.
MIN_DECODE_TOKENS = 2


def completion_body(prompt, n_predict, ignore_eos=False):
    """the bench's /completion request (greedy, seed 0, no prompt cache); ignore_eos only on the early-end retry"""
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "seed": 0,
            "cache_prompt": False}
    if ignore_eos:
        body["ignore_eos"] = True
    return body


def tokens_predicted(r):
    """generated tokens of a /completion reply (tokens_predicted, else timings.predicted_n); None when the server says neither"""
    n = r.get("tokens_predicted")
    if n is None:
        n = (r.get("timings") or {}).get("predicted_n")
    try:
        return None if n is None else int(n)
    except (TypeError, ValueError):
        return None


def ended_early(r, n_predict):
    """True when a /completion reply stopped (end of generation, a stop word) before a quarter of n_predict tokens, or
    decoded fewer than MIN_DECODE_TOKENS: its decode rate is not a speed. Unknown token count -> False (unchanged)."""
    n = tokens_predicted(r)
    if n is None:
        return False
    return n < max(MIN_DECODE_TOKENS, n_predict // 4)


def decode_rate(r):
    """the server's decode t/s, or 0.0 when the reply decoded fewer than MIN_DECODE_TOKENS tokens (1e6 is not a speed)"""
    n = tokens_predicted(r)
    if n is not None and n < MIN_DECODE_TOKENS:
        return 0.0
    return (r.get("timings") or {}).get("predicted_per_second") or 0.0


# #366: a thinking model can spend every token of a bench reply inside its thought; that rate counts thinking tokens
# and the reply has no answer, so the card says "no answer: all thinking" rather than show it as a plain speed.
THINK_TAGS = (("<think>", "</think>"), ("<seed:think>", "</seed:think>"), ("<|START_THINKING|>", "<|END_THINKING|>"),
              ("[THINK]", "[/THINK]"), ("<|channel|>analysis", "<|channel|>final"),
              ("<|begin_of_thought|>", "<|end_of_thought|>"))
_SPECIAL_RE = re.compile(r"<\|[^|<>]{0,40}\|>")


def answer_kind(r):
    """-> None (the reply has an answer, or too little is known to tell), "all thinking" (a thought that never
    closes, nothing after it, or only reasoning_content) or "ended at once" (under MIN_DECODE_TOKENS tokens, no text).
    Same rule as Control's bench (tools/pxa_control/core.py bench_answer_kind)."""
    r = r or {}
    if "content" not in r and "reasoning_content" not in r:
        return None
    answer = r.get("content") if isinstance(r.get("content"), str) else ""
    rc = r.get("reasoning_content") if isinstance(r.get("reasoning_content"), str) else ""
    thought = bool(rc.strip())
    for op, cl in THINK_TAGS:
        if op in answer:
            thought = True
            after = answer.split(op, 1)[1]
            answer = after.split(cl, 1)[1] if cl in after else ""
        elif cl in answer:                  # the prompt opened the thought
            thought = True
            answer = answer.split(cl, 1)[1]
    if _SPECIAL_RE.sub("", answer).strip():
        return None
    if thought:
        return "all thinking"
    n = tokens_predicted(r)
    if n is not None and n < MIN_DECODE_TOKENS:
        return "ended at once"
    return None


def measure_once(url, prompt, n_predict, timeout=600, ignore_eos=False):
    body = completion_body(prompt, n_predict, ignore_eos)
    t0 = time.time()
    r = _post_json(url, "/completion", body, timeout=timeout)
    wall = time.time() - t0
    timings = r.get("timings") or {}
    decode = timings.get("predicted_per_second")
    prefill = timings.get("prompt_per_second")
    n_pred = r.get("tokens_predicted")
    if decode is None and wall > 0 and n_pred:
        decode = n_pred / wall
    if n_pred is not None and n_pred < MIN_DECODE_TOKENS:
        decode = 0.0                     # one token in ~0 s reads 1e6 t/s: not a speed
    return {
        "decode_tok_s": decode or 0.0,
        "prefill_tok_s": prefill or 0.0,
        "wall_s": wall,
        "tokens_predicted": n_pred,
        "prompt_n": timings.get("prompt_n") or r.get("tokens_evaluated"),
        "draft_n": timings.get("draft_n"),
        "draft_accepted": timings.get("draft_n_accepted"),
        "content": r.get("content", ""),
        "answer_kind": answer_kind(r),
        "stop_type": r.get("stop_type"),
        "ended_early": ended_early(r, n_predict),
        "ignore_eos": bool(ignore_eos),
    }


def warmup(url, seconds, prompt=WARMUP_PROMPT):
    if seconds <= 0:
        return
    print("pxa-bench: warming up for %ds (owner rule: warm before the open bracket) ..." % seconds)
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        try:
            measure_once(url, prompt, 32, timeout=60)
            n += 1
        except Exception as e:
            print("   warm-up request failed (%s); continuing" % e, file=sys.stderr)
            time.sleep(1)
    print("pxa-bench: warmed with %d request(s) over %ds" % (n, seconds))


def forget(url):
    """Erase every slot so the next request is measured FIRST PASS (2026-10-19): the reps ask the same prompt, and a
    server with n-gram speculation otherwise replays its own earlier answer from the table (up to ~2x on a spec run).
    -> True when every slot was erased; False when the server refused (another engine): the result says so."""
    try:
        n = int((_get_json(url, "/props", timeout=10) or {}).get("total_slots") or 1)
    except Exception:
        n = 1
    ok = True
    for sid in range(max(1, min(n, 64))):
        try:
            r = _post_json(url, "/slots/%d?action=erase" % sid, {}, timeout=30)
            if isinstance(r, dict) and r.get("error"):
                ok = False
        except Exception:
            ok = False
    return ok


def run_tag(session, rep):
    """FR#12: when the server refuses the slot erase, the run's prompt starts with this tag instead, so
    no run re-uses another run's prefix (the session is random per bench, so a second bench on the
    same server does not repeat the first one's prompts either)."""
    return "[bench %s run %d]\n" % (session, rep)


def summarize_runs(decodes, accepted=None):
    """FR#12: run 1 (the first pass) vs the mean of runs 2..N for one class.

    -> {"first_pass", "later_mean", "later_vs_first", "replay_suspect", "headline"}. headline is the
    median of all runs, or the first pass when a later run is REPLAY_SUSPECT_RATIO or more faster
    than run 1 (then the later runs were replay and must not lift the number). `accepted` (accepted
    draft tokens per run, same order) is optional: when the server reports it for every run and no
    later run accepted more drafts than run 1, a faster later run is noise (a slow first pass), not
    replay, and is not flagged."""
    if accepted is not None and len(accepted) == len(decodes) and all(a is not None for a in accepted):
        acc = [a for a, x in zip(accepted, decodes) if x]
    else:
        acc = None
    d = [float(x) for x in decodes if x]
    if not d:
        return {"first_pass": 0.0, "later_mean": None, "later_vs_first": None,
                "replay_suspect": False, "headline": 0.0}
    first = d[0]
    later = d[1:]
    later_mean = sum(later) / len(later) if later else None
    ratio = (later_mean / first - 1.0) if (later_mean is not None and first > 0) else None
    suspect = bool(later) and first > 0 and max(later) >= first * REPLAY_SUSPECT_RATIO
    if suspect and acc and len(acc) == len(d) and max(acc[1:]) <= acc[0]:
        suspect = False          # no more drafts accepted than the first pass: the speed-up was noise, not replay
    return {"first_pass": first, "later_mean": later_mean, "later_vs_first": ratio,
            "replay_suspect": suspect, "headline": first if suspect else statistics.median(d)}


def run_class(url, cls, prompt, n_predict, reps, session=None):
    decodes, prefills, accepted = [], [], []
    eos_runs = think_runs = none_runs = 0
    first_pass = True
    session = session or os.urandom(3).hex()
    for i in range(reps):
        erased = forget(url)
        first_pass = erased and first_pass
        # a server that refused the erase gets a fresh prompt per run (FR#12): no prefix re-use at least
        p_i = prompt if erased else run_tag(session, i + 1) + prompt
        m = measure_once(url, p_i, n_predict)
        if m.get("ended_early"):
            # #361: the model ended at once on this raw-text prompt (a chat model may): ask again, EOS ignored, from
            # erased slots, so the class times n_predict tokens instead of reading 1e6 t/s off one token
            print("   [%s] rep %d/%d: the model ended after %s token(s)%s on this raw-text prompt; asked again with ignore_eos" % (
                cls, i + 1, reps, m.get("tokens_predicted"), (" (%s)" % m["stop_type"]) if m.get("stop_type") else ""))
            forget(url)
            m = measure_once(url, p_i, n_predict, ignore_eos=True)
            eos_runs += 1
        kind = m.get("answer_kind")
        if kind == "all thinking":
            think_runs += 1
        elif kind:
            none_runs += 1
        if m["decode_tok_s"]:
            decodes.append(m["decode_tok_s"])
            accepted.append(m.get("draft_accepted"))
        if m["prefill_tok_s"]:
            prefills.append(m["prefill_tok_s"])
        print("   [%s] rep %d/%d: decode %.1f t/s%s%s" % (
            cls, i + 1, reps, m["decode_tok_s"],
            (", prefill %.1f t/s" % m["prefill_tok_s"]) if m["prefill_tok_s"] else "",
            (", drafts %s/%s accepted" % (m["draft_accepted"], m["draft_n"])) if m.get("draft_n") else "")
            + ((", no answer: %s" % kind) if kind else ""))
    no_answer = None
    if reps and think_runs + none_runs >= reps:
        no_answer = "all thinking" if think_runs else "ended at once"
    s = summarize_runs(decodes, accepted)
    return {
        "class": cls,
        "decode_tok_s": s["headline"],
        "prefill_tok_s": statistics.median(prefills) if prefills else 0.0,
        "reps": reps,
        "first_pass": first_pass,
        "first_pass_tok_s": s["first_pass"],
        "later_mean_tok_s": s["later_mean"],
        "replay_suspect": s["replay_suspect"],
        "decode_all": decodes,
        "ignore_eos_runs": eos_runs,
        "all_thinking_runs": think_runs,
        "no_answer_runs": none_runs,
        "no_answer": no_answer,
    }


def greedy512(url):
    m = measure_once(url, GREEDY_PROMPT, GREEDY_N, timeout=900)
    content = m["content"]
    sha = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
    return {"sha256": sha, "chars": len(content), "tokens_predicted": m.get("tokens_predicted"),
            "empty": len(content) < 8}


# ---------------------------------------------------------------------------------------------
# The published board (placeholder for now -- see tools/pxa-bench-board.json). Never fabricates a
# number: a missing/placeholder row says so in plain text instead of a percentage.
# ---------------------------------------------------------------------------------------------
def load_board(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def find_board_row(board, card_class, n_cards, model_tier, cls):
    if not board:
        return None
    rows = board.get("rows") or []
    for want_tier in (model_tier, None):
        for want_n in (n_cards, None):
            for r in rows:
                if r.get("card_class") != card_class or r.get("class") != cls:
                    continue
                if want_n is not None and r.get("n_cards") != want_n:
                    continue
                if want_tier is not None and r.get("model_tier") != want_tier:
                    continue
                return r
    return None


def format_vs_board(decode_tok_s, board_row):
    if not board_row:
        return "no published row yet for this card class/tier"
    if board_row.get("status") != "measured" or board_row.get("decode_tok_s") is None:
        return "board row is a PLACEHOLDER (not yet measured for this release)"
    b = board_row["decode_tok_s"]
    delta = (decode_tok_s - b) / b * 100.0 if b else 0.0
    return "%+.1f%% vs board %.1f t/s (%s)" % (delta, b, board_row.get("source", ""))


# ---------------------------------------------------------------------------------------------
# Rendering: the markdown card (always) and an SVG/PNG badge (SVG always; PNG best-effort).
# ---------------------------------------------------------------------------------------------
def render_markdown(result):
    lines = []
    lines.append("## PXA bench card")
    lines.append("")
    lines.append("- **Cards:** %s" % result["cards_line"])
    model_line = "- **Model:** `%s`" % (result.get("model_alias") or result.get("model_path") or "?")
    if result.get("model_sha"):
        model_line += "  sha256 `%s…`" % result["model_sha"][:16]
    if result.get("model_tier"):
        model_line += "  (%s)" % result["model_tier"]
    lines.append(model_line)
    lines.append("- **Engine:** build %s (`%s`)" % (
        result.get("engine_build", "?"), result.get("engine_commit", "?")))
    g = result.get("greedy") or {}
    lines.append("- **Greedy512 identity:** sha256 `%s…` (%s chars, %s tok, temperature 0, seed 0)%s" % (
        (g.get("sha256") or "?")[:16], g.get("chars", "?"), g.get("tokens_predicted", "?"),
        "  **EMPTY -- FAIL**" if g.get("empty") else ""))
    lines.append("- **Generated:** %s" % result["generated_at"])
    lines.append("")
    lines.append("| class | decode t/s (median x%s) | first pass t/s | later runs mean t/s | prefill t/s (median) | vs published board |"
                 % result.get("reps", 3))
    lines.append("|---|---|---|---|---|---|")
    flagged = []
    for row in result.get("classes", []):
        fp, lm = row.get("first_pass_tok_s"), row.get("later_mean_tok_s")
        lines.append("| %s | %.1f%s | %s | %s | %.1f | %s |" % (
            row["class"], row["decode_tok_s"],
            (" (first pass)" if row.get("replay_suspect") else "")
            + ((" (no answer: %s)" % row["no_answer"]) if row.get("no_answer") else ""),
            "%.1f" % fp if fp else "-", "%.1f" % lm if lm else "-",
            row["prefill_tok_s"], row["vs_board"]))
        if row.get("replay_suspect"):
            flagged.append(row["class"])
    lines.append("")
    lines.append("_Every run starts from a server told to forget its earlier replies (slot erase), so each "
                 "one is a first pass; run 1 is shown next to the mean of the later runs._")
    if flagged:
        lines.append("")
        lines.append("**Later runs were much faster than the first pass for %s: the server replayed "
                     "an earlier answer, so the first pass is the number shown.**" % ", ".join(flagged))
    eos = [row["class"] for row in result.get("classes", []) if row.get("ignore_eos_runs")]
    if eos:
        lines.append("")
        lines.append("_The model ended at once on the raw-text prompt of %s (a chat model may), so those runs were "
                     "asked again with ignore_eos and time the full token count._" % ", ".join(eos))
    silent = [row["class"] for row in result.get("classes", []) if row.get("no_answer")]
    if silent:
        lines.append("")
        lines.append("**No answer for %s: every run was all thinking or ended at once, so that rate is not a speed "
                     "to compare. Try a larger context or thinking off.**" % ", ".join(silent))
    lines.append("")
    lines.append("_%s_" % result.get("board_note", ""))
    lines.append("")
    lines.append("Reproduce: `pxa-launch --bench --model <path> --gpus <ids>` "
                 "(or `pxa-bench.py --url http://host:port` on an already-running seat).")
    return "\n".join(lines)


def _badge_segments(result):
    segs = [("PXA bench", "#333333")]
    for row in result.get("classes", []):
        segs.append(("%s %.0f t/s" % (row["class"], row["decode_tok_s"]),
                     _BADGE_COLORS.get(row["class"], "#555555")))
    return segs


def render_badge_svg(result):
    segs = _badge_segments(result)

    def seg_w(s):
        return max(46, int(len(s) * 6.6) + 20)

    widths = [seg_w(s) for s, _ in segs]
    total_w = sum(widths)
    h = 20
    x = 0
    rects, texts = [], []
    for (label, color), w in zip(segs, widths):
        rects.append('<rect x="%d" width="%d" height="%d" fill="%s"/>' % (x, w, h, color))
        cx = x + w / 2.0
        texts.append(
            '<text x="%.1f" y="14" fill="#ffffff" font-family="Verdana,Geneva,sans-serif" '
            'font-size="11" text-anchor="middle">%s</text>' % (cx, escape(label)))
        x += w
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" role="img" '
        'aria-label="PXA bench result">\n'
        '<rect width="%d" height="%d" fill="#1b1f26"/>\n%s\n%s\n</svg>\n'
        % (total_w, h, total_w, h, "\n".join(rects), "\n".join(texts))
    )


def render_badge_png(result, out_path):
    """Best-effort: draws the same segments as render_badge_svg() with Pillow. Returns
    (out_path, None) on success, or (None, note) if Pillow is not importable -- never raises, and
    the SVG badge above always covers the "writes a badge" requirement on its own."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None, "Pillow not installed (pip install pillow for a PNG badge too; SVG was still written)"

    segs = _badge_segments(result)
    pad = 10
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    scratch = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(scratch)

    def measure(label):
        if hasattr(d, "textbbox"):
            bbox = d.textbbox((0, 0), label, font=font)
            return bbox, (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        w, h_ = d.textsize(label, font=font)  # very old Pillow
        return (0, 0, w, h_), w, h_

    widths = []
    for label, _color in segs:
        _, tw, _th = measure(label)
        widths.append(tw + pad * 2)
    total_w = sum(widths)
    h = 24
    img = Image.new("RGB", (max(total_w, 1), h), "#1b1f26")
    d = ImageDraw.Draw(img)
    x = 0
    for (label, color), w in zip(segs, widths):
        d.rectangle([x, 0, x + w, h], fill=color)
        bbox, tw, th = measure(label)
        d.text((x + (w - tw) / 2.0, (h - th) / 2.0 - bbox[1]), label, fill="white", font=font)
        x += w
    try:
        img.save(out_path)
    except Exception as e:
        return None, "Pillow could not write %s (%s)" % (out_path, e)
    return out_path, None


# ---------------------------------------------------------------------------------------------
# The measurement pass, tied together
# ---------------------------------------------------------------------------------------------
def run_bench(url, a):
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]

    cards, engine, model_tier, model_alias_from_page, explain_err = get_cards_and_registry(url)
    if explain_err:
        print("pxa-bench: GET /pxa/explain not available (%s); falling back to nvidia-smi for "
              "card identity (older engine build, or run against a non-PXA server)." % explain_err,
              file=sys.stderr)
        cards = cards_from_nvidia_smi()

    props = {}
    try:
        props = _get_json(url, "/props", timeout=10)
    except Exception:
        pass
    model_path = props.get("model_path") or a.model or ""
    model_alias = model_alias_from_page or props.get("model_alias") or props.get("model_name") or model_path

    model_sha, _sha_how = (None, "skipped (--no-sha)") if a.no_sha else model_sha256(model_path)

    warmup(url, a.warmup)

    board = load_board(a.board_json)
    names = [c.get("name") for c in cards]
    sm0 = cards[0].get("sm") if cards else None
    card_class = card_class_from_name(names[0] if names else "", sm0)
    n_cards = len(cards) or 1

    class_results = []
    for cls in classes:
        spec = CLASS_PROMPTS.get(cls)
        if not spec:
            print("pxa-bench: unknown class %r, skipping" % cls, file=sys.stderr)
            continue
        row = run_class(url, cls, spec["prompt"], spec["n_predict"], a.reps)
        board_row = find_board_row(board, card_class, n_cards, model_tier, cls)
        row["vs_board"] = format_vs_board(row["decode_tok_s"], board_row)
        class_results.append(row)

    forget(url)
    g = greedy512(url)

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cards": cards, "cards_line": cards_line(cards), "card_class": card_class, "n_cards": n_cards,
        "engine_build": engine.get("build"), "engine_commit": engine.get("commit"),
        "model_alias": model_alias, "model_path": model_path, "model_sha": model_sha,
        "model_tier": model_tier, "greedy": g, "reps": a.reps, "classes": class_results,
        "board_note": (board or {}).get(
            "note", "no board file found at %s (tools/pxa-bench-board.json missing or unreadable)" % a.board_json),
    }


# ---------------------------------------------------------------------------------------------
# Self-boot mode: `pxa-bench.py --model F --gpus G` with no --url. pxa-launch.py, at the end of
# its own normal (non-explain, non-doctor) path, os.execvpe()'s itself into llama-server -- so the
# Popen handle below IS the server process throughout, and a plain SIGTERM stops it cleanly.
# ---------------------------------------------------------------------------------------------
def boot_server(model, gpus, port, workload="chat"):
    here = os.path.dirname(os.path.abspath(__file__))
    launch_py = os.path.join(here, "pxa-launch.py")
    if not os.path.isfile(launch_py):
        print("pxa-bench: tools/pxa-launch.py not found next to this script; cannot self-boot. "
              "Pass --url http://host:port for an already-running seat instead.", file=sys.stderr)
        sys.exit(2)
    argv = [sys.executable, launch_py, "--model", model, "--gpus", gpus, "--port", str(port),
            "--workload", workload, "--yes", "--no-tui", "--no-interactive", "--no-control"]
    print("pxa-bench: booting a seat to measure:  %s" % " ".join(argv))
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    print("pxa-bench: stopping the seat it booted (pid %d) ..." % proc.pid)
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ---------------------------------------------------------------------------------------------
# --selftest: the renderer and helpers, on recorded sample data. No network, no GPU. Wired into
# CTest as test-pxa-bench-render.
# ---------------------------------------------------------------------------------------------
def run_selftest():
    ok = True

    sample = {
        "generated_at": "2026-09-25T00:00:00Z (SAMPLE DATA -- --selftest, not a real measurement)",
        "cards_line": "1x Tesla V100-PCIE-16GB (sm_70)",
        "model_alias": "Qwen3.8-27B-PXQ-mix27", "model_path": "/models/x.gguf",
        "model_sha": "4af1ceed" + "0" * 56, "model_tier": "mix27",
        "engine_build": 4242, "engine_commit": "deadbee",
        "greedy": {"sha256": "a" * 64, "chars": 1800, "tokens_predicted": 512, "empty": False},
        "reps": 3,
        "classes": [
            {"class": "prose", "decode_tok_s": 23.4, "prefill_tok_s": 812.0,
             "vs_board": "board row is a PLACEHOLDER (not yet measured for this release)"},
            {"class": "edit", "decode_tok_s": 25.1, "prefill_tok_s": 790.3,
             "vs_board": "no published row yet for this card class/tier"},
            {"class": "long", "decode_tok_s": 19.8, "prefill_tok_s": 1400.2,
             "vs_board": "+3.2% vs board 19.2 t/s (sample-source)"},
        ],
        "board_note": "sample data for --selftest; not a real measurement",
    }

    md = render_markdown(sample)
    if "PXA bench card" in md and all(c in md for c in ("prose", "edit", "long")) and "23.4" in md:
        print("SELFTEST: markdown card OK (%d chars)" % len(md))
    else:
        print("SELFTEST FAIL: markdown card missing expected content")
        ok = False

    svg = render_badge_svg(sample)
    if svg.startswith("<svg") and "PXA bench" in svg and "</svg>" in svg:
        print("SELFTEST: SVG badge OK (%d bytes)" % len(svg))
    else:
        print("SELFTEST FAIL: SVG badge malformed")
        ok = False

    tmp_png = os.path.join(tempfile.gettempdir(), "pxa-bench-selftest-badge.png")
    png_path, png_note = render_badge_png(sample, tmp_png)
    if png_path:
        if os.path.isfile(png_path) and os.path.getsize(png_path) > 0:
            print("SELFTEST: PNG badge OK (%d bytes)" % os.path.getsize(png_path))
        else:
            print("SELFTEST FAIL: PNG badge file missing/empty")
            ok = False
        try:
            os.remove(png_path)
        except OSError:
            pass
    else:
        print("SELFTEST: PNG badge skipped (%s) -- SVG already covers the badge requirement" % png_note)

    board_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pxa-bench-board.json")
    board = load_board(board_path)
    if board is None:
        print("SELFTEST FAIL: could not load %s" % board_path)
        ok = False
    else:
        row = find_board_row(board, "V100", 1, "mix27", "prose")
        print("SELFTEST: board lookup for V100x1/mix27/prose -> %s"
              % (row.get("status") if row else "no row (graceful)"))
        no_row = find_board_row(board, "NOPE", 99, "???", "prose")
        if no_row is not None:
            print("SELFTEST FAIL: nonsense lookup should return None")
            ok = False
        else:
            print("SELFTEST: nonsense board lookup returns None, no exception")

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"pxa-bench selftest fixture\n")
        fpath = f.name
    try:
        want = hashlib.sha256(b"pxa-bench selftest fixture\n").hexdigest()
        got, _how = model_sha256(fpath)
        if got == want:
            print("SELFTEST: model_sha256 OK")
        else:
            print("SELFTEST FAIL: model_sha256 mismatch (%s != %s)" % (got, want))
            ok = False
    finally:
        try:
            os.remove(fpath)
        except OSError:
            pass

    for name, want in (("Tesla P100-PCIE-16GB", "P100"), ("Tesla V100-PCIE-16GB", "V100"),
                       ("NVIDIA GeForce GTX 1080 Ti", "1080TI")):
        got = card_class_from_name(name)
        if got != want:
            print("SELFTEST FAIL: card_class_from_name(%r) = %r, want %r" % (name, got, want))
            ok = False
    else:
        print("SELFTEST: card_class_from_name OK")

    # #366: a reply that is all thinking has no answer; the card says so instead of showing a bare rate
    for r, want in (({"content": "<think>the passage is about", "tokens_predicted": 512}, "all thinking"),
                    ({"content": "<think>hm</think>\nIt describes a harbour.", "tokens_predicted": 512}, None),
                    ({"content": "", "reasoning_content": "let me think", "tokens_predicted": 512}, "all thinking"),
                    ({"content": "<|im_end|>", "tokens_predicted": 1}, "ended at once"),
                    ({"content": "plain answer", "tokens_predicted": 512}, None)):
        got = answer_kind(r)
        if got != want:
            print("SELFTEST FAIL: answer_kind(%r) = %r, want %r" % (r, got, want))
            ok = False
    think = dict(sample, classes=[dict(sample["classes"][0], all_thinking_runs=3, no_answer="all thinking")])
    if "no answer: all thinking" in render_markdown(think):
        print("SELFTEST: answer_kind + no-answer label OK")
    else:
        print("SELFTEST FAIL: markdown card missing the no-answer label")
        ok = False

    print("SELFTEST: %s" % ("ALL PASS" if ok else "FAILURES ABOVE"))
    return ok


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="pxa bench -- a ~2-minute, shareable result card (markdown + SVG/PNG badge).")
    ap.add_argument("--url", default="", help="bench an already-running seat, e.g. http://127.0.0.1:8080")
    ap.add_argument("--model", default="", help="GGUF path; with --gpus and no --url, boots a seat first")
    ap.add_argument("--gpus", default="", help="e.g. 0,1 -- passed straight to pxa-launch.py")
    ap.add_argument("--port", type=int, default=8099, help="self-boot mode only")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=60, help="seconds, 0 to skip (owner rule default: 60)")
    ap.add_argument("--classes", default="prose,edit,long")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--board-json", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pxa-bench-board.json"))
    ap.add_argument("--no-sha", action="store_true", help="skip hashing the model file")
    ap.add_argument("--boot-timeout", type=int, default=240, help="self-boot mode only")
    ap.add_argument("--selftest", action="store_true", help="renderer/helpers on sample data, no network")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if run_selftest() else 1)

    url = a.url
    proc = None
    owns_server = False
    try:
        if not url:
            if not a.model or not a.gpus:
                print("pxa-bench: need --url of a running seat, or --model + --gpus to boot one.",
                      file=sys.stderr)
                sys.exit(2)
            url = "http://127.0.0.1:%d" % a.port
            proc = boot_server(a.model, a.gpus, a.port)
            owns_server = True

        if not wait_health(url, a.boot_timeout if owns_server else 60):
            print("pxa-bench: %s never became healthy." % url, file=sys.stderr)
            sys.exit(3)

        result = run_bench(url, a)
        md = render_markdown(result)
        print()
        print(md)

        os.makedirs(a.out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        md_path = os.path.join(a.out_dir, "pxa-bench-%s.md" % stamp)
        with open(md_path, "w") as f:
            f.write(md + "\n")
        svg_path = os.path.join(a.out_dir, "pxa-bench-%s.svg" % stamp)
        with open(svg_path, "w") as f:
            f.write(render_badge_svg(result))
        png_path, png_note = render_badge_png(result, os.path.join(a.out_dir, "pxa-bench-%s.png" % stamp))

        print()
        print("pxa-bench: wrote %s" % md_path)
        print("pxa-bench: wrote %s" % svg_path)
        if png_path:
            print("pxa-bench: wrote %s" % png_path)
        else:
            print("pxa-bench: PNG badge skipped (%s)" % png_note)

        if result["greedy"].get("empty"):
            print("pxa-bench: WARNING greedy512 completion was empty -- treat this run as a FAIL, "
                  "not a real result.", file=sys.stderr)
            sys.exit(4)
    finally:
        if owns_server:
            stop_server(proc)


if __name__ == "__main__":
    main()
