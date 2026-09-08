#!/usr/bin/env python3
"""
Minimal single-file SWE-bench agent harness for the PXA flagship 35B model.

- Dataset: SWE-bench_Lite (princeton-nlp/SWE-bench_Lite), 300 real GitHub
  issue->patch instances. Standard, recognized SWE benchmark.
- Model:   OpenAI-compatible endpoint at http://127.0.0.1:8350/v1/chat/completions
           (our ik_llama.cpp fork serving fusion2-35b-PXQ6-MTP-clean.gguf).
- Loop:    ReAct-style tool loop (list_dir / read_file / grep / submit_patch)
           over a real shallow git checkout of the target repo at base_commit.
- Grading: delegates to the OFFICIAL swebench harness
           (python -m swebench.harness.run_evaluation), which pulls prebuilt
           per-instance docker images from Docker Hub (namespace "swebench")
           and runs FAIL_TO_PASS / PASS_TO_PASS against the applied patch.
           This is the standard/authoritative grading path - we do not
           reimplement per-repo env setup.
- Resumable: results.jsonl is append-only, keyed by instance_id; a run
  skips any instance_id already present unless --force is passed.

Usage:
  python3 run_swe.py --limit 10                     # smoke test, first 10 instances
  python3 run_swe.py --instance-ids astropy__astropy-12907 django__django-11099
  python3 run_swe.py --limit 300                     # full SWE-bench Lite run
  python3 run_swe.py --grade-only                    # (re)grade whatever predictions exist

Layout under /root/swe/:
  repos/<org>__<repo>/          cached git checkouts (partial clone, reused across instances)
  predictions/predictions.jsonl swebench-format predictions (one row per attempted instance)
  transcripts/<instance_id>.json   full tool-loop transcript for inspection
  run_logs/                     raw stdout/stderr of the official grading harness
  results.jsonl                 final resumable results: instance_id, resolved, patch_len, turns, error
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/root/swe")
REPO_CACHE = ROOT / "repos"
PRED_DIR = ROOT / "predictions"
TRANSCRIPT_DIR = ROOT / "transcripts"
RUNLOG_DIR = ROOT / "run_logs"
RESULTS_PATH = ROOT / "results.jsonl"
PRED_PATH = PRED_DIR / "predictions.jsonl"

MODEL_URL = "http://127.0.0.1:8350/v1/chat/completions"
MODEL_NAME = "fusion2-35b-pxq6"
MAX_TURNS = 16
MAX_FILE_BYTES = 12000  # cap file reads so we don't blow the 32k ctx window
REQUEST_TIMEOUT = 300

SYSTEM_PROMPT = """You are an expert software engineer fixing a real bug in an open-source \
Python repository. You are given a GitHub issue (problem statement) and read-only access to \
the repository via tools. Explore the relevant file(s), then fix the bug using submit_edits.

Rules:
- Use list_dir / read_file / grep to find and inspect the exact code you need to change.
- Do NOT guess file contents - read them first, and copy old_str EXACTLY (including whitespace)
  from what read_file showed you - it must match the file byte-for-byte and occur exactly once.
- When ready, call submit_edits with one or more {path, old_str, new_str} edits that together
  fix the bug. This ends the task. If an edit fails to apply you'll get an error back and can retry.
- Keep changes minimal and focused on the bug described. Do not reformat unrelated code.
- You have a limited number of turns - do not explore more than necessary before editing.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories at a path relative to the repo root.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative path, e.g. '.' or 'astropy/modeling'"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the repo (optionally a line range). Large files are truncated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "description": "1-indexed, optional"},
                    "end_line": {"type": "integer", "description": "1-indexed inclusive, optional"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a regex pattern across the repo (like grep -rn). Returns matching lines with file:line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Subdirectory to search, default '.'"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_edits",
            "description": (
                "Apply one or more exact search/replace edits to fix the issue, and end the task. "
                "old_str must match the file content EXACTLY (verbatim, including indentation) and "
                "appear exactly once in the file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "File path relative to repo root"},
                                "old_str": {"type": "string", "description": "Exact existing text to replace"},
                                "new_str": {"type": "string", "description": "Replacement text"},
                            },
                            "required": ["path", "old_str", "new_str"],
                        },
                    }
                },
                "required": ["edits"],
            },
        },
    },
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, cwd=None, timeout=600, check=True):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {cmd}\nSTDOUT:{r.stdout[-2000:]}\nSTDERR:{r.stderr[-2000:]}")
    return r


def ensure_repo(repo, base_commit):
    """Return path to a local checkout of `repo` (org/name) at `base_commit`.
    Reuses a cached bare-ish clone per repo across instances (many Lite
    instances share the same repo at different commits)."""
    org, name = repo.split("/")
    cache_dir = REPO_CACHE / f"{org}__{name}"
    if not cache_dir.exists():
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        log(f"cloning {repo} (partial, no checkout) ...")
        run([
            "git", "clone", "--filter=blob:none", "--no-checkout",
            f"https://github.com/{repo}.git", str(cache_dir),
        ], timeout=1800)
    # make sure we have the commit (fetch if missing)
    have = run(["git", "cat-file", "-e", base_commit], cwd=cache_dir, check=False)
    if have.returncode != 0:
        log(f"fetching {base_commit} for {repo} ...")
        run(["git", "fetch", "--depth", "1", "origin", base_commit], cwd=cache_dir, timeout=1200)
    # checkout into an isolated worktree per instance so parallel/successive
    # instances of the same repo never collide
    return cache_dir, base_commit


def checkout_worktree(cache_dir, base_commit, instance_id):
    wt = REPO_CACHE / "worktrees" / instance_id
    if wt.exists():
        run(["git", "checkout", "-f", base_commit], cwd=wt)
        run(["git", "clean", "-fdx"], cwd=wt)
        return wt
    wt.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "worktree", "add", "--detach", "-f", str(wt), base_commit], cwd=cache_dir, timeout=600)
    return wt


def tool_list_dir(repo_dir, path):
    p = (repo_dir / path).resolve()
    if not str(p).startswith(str(repo_dir.resolve())):
        return "ERROR: path escapes repo root"
    if not p.exists():
        return f"ERROR: no such path: {path}"
    if p.is_file():
        return f"ERROR: {path} is a file, not a directory"
    entries = sorted(os.listdir(p))
    entries = [e for e in entries if e != ".git"]
    return "\n".join(entries[:300]) or "(empty)"


def tool_read_file(repo_dir, path, start_line=None, end_line=None):
    p = (repo_dir / path).resolve()
    if not str(p).startswith(str(repo_dir.resolve())):
        return "ERROR: path escapes repo root"
    if not p.exists() or not p.is_file():
        return f"ERROR: no such file: {path}"
    try:
        text = p.read_text(errors="replace")
    except Exception as e:
        return f"ERROR reading file: {e}"
    lines = text.splitlines()
    if start_line or end_line:
        s = max(1, start_line or 1)
        e = min(len(lines), end_line or len(lines))
        chunk = "\n".join(lines[s - 1:e])
        out = f"(lines {s}-{e} of {len(lines)})\n{chunk}"
    else:
        out = text
    if len(out) > MAX_FILE_BYTES:
        out = out[:MAX_FILE_BYTES] + f"\n... [truncated, file continues, {len(text)} bytes total; use start_line/end_line]"
    return out


def tool_grep(repo_dir, pattern, path="."):
    p = (repo_dir / path).resolve()
    if not str(p).startswith(str(repo_dir.resolve())):
        return "ERROR: path escapes repo root"
    r = subprocess.run(
        ["grep", "-rn", "-I", "--include=*.py", "-E", pattern, str(p)],
        capture_output=True, text=True, timeout=60,
    )
    out = r.stdout.strip()
    if not out:
        return "(no matches)"
    lines = out.splitlines()[:80]
    # relativize paths
    lines = [ln.replace(str(repo_dir) + "/", "") for ln in lines]
    return "\n".join(lines)


def call_model(messages, tools):
    import urllib.request
    body = json.dumps({
        "model": MODEL_NAME,
        "messages": messages,
        "tools": tools,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 1500,
    }).encode()
    req = urllib.request.Request(MODEL_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]


def apply_edits(repo_dir, edits):
    """Apply exact search/replace edits directly to the checked-out worktree.
    Returns (ok, error_message). On success the caller should capture the
    diff via `git diff` in repo_dir - this guarantees a syntactically valid
    unified diff (git computes the hunks), sidestepping the classic LLM
    failure mode of hand-writing a diff with wrong hunk-header line counts."""
    for e in edits:
        path = e.get("path", "")
        old_str = e.get("old_str", "")
        new_str = e.get("new_str", "")
        p = (repo_dir / path).resolve()
        if not str(p).startswith(str(repo_dir.resolve())):
            return False, f"path escapes repo root: {path}"
        if not p.exists() or not p.is_file():
            return False, f"no such file: {path}"
        try:
            text = p.read_text()
        except Exception as ex:
            return False, f"could not read {path}: {ex}"
        count = text.count(old_str)
        if count == 0:
            return False, f"old_str not found in {path} (must match exactly, whitespace included)"
        if count > 1:
            return False, f"old_str matches {count} times in {path} - must be unique, add more context"
        p.write_text(text.replace(old_str, new_str, 1))
    return True, None


def run_agent_on_instance(inst, max_turns=MAX_TURNS):
    instance_id = inst["instance_id"]
    repo = inst["repo"]
    base_commit = inst["base_commit"]
    problem = inst["problem_statement"]

    transcript = {"instance_id": instance_id, "repo": repo, "turns": []}

    cache_dir, base_commit = ensure_repo(repo, base_commit)
    repo_dir = checkout_worktree(cache_dir, base_commit, instance_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Repository: {repo}\nBase commit: {base_commit}\n\nIssue:\n{problem}\n\nInvestigate the repo and fix it. Call submit_edits when done. You have {max_turns} turns total."},
    ]

    patch_text = None
    error = None
    for turn in range(max_turns):
        turns_left = max_turns - turn
        if turns_left == 3:
            messages.append({"role": "user", "content": "You have 3 turns left. Stop exploring and call submit_edits now with your best fix."})
        try:
            msg = call_model(messages, TOOLS)
        except Exception as e:
            error = f"model call failed: {e}"
            break
        transcript["turns"].append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg.get("tool_calls")})
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg.get("tool_calls")})

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # nudge it once toward using a tool / submitting
            messages.append({"role": "user", "content": "Please call a tool (list_dir/read_file/grep) to continue investigating, or submit_edits if you have the fix."})
            continue

        submitted = False
        for tc in tool_calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except Exception:
                args = {}
            if fn == "list_dir":
                result = tool_list_dir(repo_dir, args.get("path", "."))
            elif fn == "read_file":
                result = tool_read_file(repo_dir, args.get("path", ""), args.get("start_line"), args.get("end_line"))
            elif fn == "grep":
                result = tool_grep(repo_dir, args.get("pattern", ""), args.get("path", "."))
            elif fn == "submit_edits":
                edits = args.get("edits", [])
                ok, err = apply_edits(repo_dir, edits)
                if ok:
                    diff = run(["git", "diff"], cwd=repo_dir, check=False)
                    patch_text = diff.stdout
                    result = "edits applied successfully, task complete"
                    submitted = bool(patch_text.strip())
                    if not submitted:
                        result = "edits applied but produced an empty diff (old_str == new_str?) - try again"
                else:
                    result = f"ERROR applying edits: {err}"
            else:
                result = f"ERROR: unknown tool {fn}"
            transcript["turns"][-1].setdefault("tool_results", []).append({"tool": fn, "args": args, "result": result[:2000]})
            messages.append({"role": "tool", "tool_call_id": tc.get("id", fn), "content": result})
        if submitted:
            break
    else:
        error = error or "max turns reached without submit_edits"

    if not patch_text:
        error = error or "model never produced a patch"

    transcript["patch"] = patch_text
    transcript["error"] = error
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    (TRANSCRIPT_DIR / f"{instance_id}.json").write_text(json.dumps(transcript, indent=2)[:2_000_000])

    return patch_text, error


def load_done_ids():
    done = set()
    if RESULTS_PATH.exists():
        for line in RESULTS_PATH.read_text().splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["instance_id"])
            except Exception:
                pass
    return done


def append_result(rec):
    ROOT.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def write_predictions(preds):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    with PRED_PATH.open("w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")


def grade(instance_ids, run_id):
    """Invoke the official swebench evaluation harness against predictions.jsonl."""
    RUNLOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNLOG_DIR / f"{run_id}.log"
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "princeton-nlp/SWE-bench_Lite",
        "--predictions_path", str(PRED_PATH),
        "--max_workers", "4",
        "--run_id", run_id,
        "--namespace", "swebench",
        "--cache_level", "instance",
        "--timeout", "1200",
        "--report_dir", str(ROOT),
        "-i", *instance_ids,
    ]
    log(f"grading via official harness: {' '.join(cmd[:6])} ... -i <{len(instance_ids)} ids>")
    with log_path.open("w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600 * 6)
    log(f"grading exit code {r.returncode}, full log at {log_path}")

    report_path = ROOT / f"fusion2-35b-pxq6.{run_id}.json"
    if not report_path.exists():
        # swebench names the report <predictions model_name_or_path>.<run_id>.json
        candidates = list(ROOT.glob(f"*.{run_id}.json"))
        report_path = candidates[0] if candidates else None
    results = {}
    if report_path and report_path.exists():
        rep = json.loads(report_path.read_text())
        resolved = set(rep.get("resolved_ids", []))
        for iid in instance_ids:
            results[iid] = iid in resolved
    else:
        log("WARNING: no report file found; grading may have failed - check the run log")
    return results, log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10, help="number of instances to attempt (from the front of the dataset)")
    ap.add_argument("--instance-ids", nargs="*", default=None, help="explicit instance_ids to run instead of --limit")
    ap.add_argument("--split", default="test")
    ap.add_argument("--force", action="store_true", help="re-run instances already present in results.jsonl")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--grade-only", action="store_true", help="skip generation, just (re)grade existing predictions.jsonl")
    ap.add_argument("--run-id", default=f"smoke-{int(time.time())}")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=args.split)

    if args.instance_ids:
        instances = [r for r in ds if r["instance_id"] in set(args.instance_ids)]
    else:
        instances = list(ds)[: args.limit]

    done = load_done_ids() if not args.force else set()
    todo = [i for i in instances if i["instance_id"] not in done]
    log(f"{len(instances)} requested, {len(done & {i['instance_id'] for i in instances})} already done, {len(todo)} to run")

    preds = []
    attempted_ids = []
    for i, inst in enumerate(todo):
        iid = inst["instance_id"]
        log(f"[{i+1}/{len(todo)}] {iid} ({inst['repo']}) ...")
        t0 = time.time()
        try:
            patch, error = run_agent_on_instance(inst, max_turns=args.max_turns)
        except Exception as e:
            patch, error = None, f"exception: {e}\n{traceback.format_exc()[-1500:]}"
        dt = time.time() - t0
        log(f"  -> patch_len={len(patch) if patch else 0} error={error} ({dt:.1f}s)")
        preds.append({
            "instance_id": iid,
            "model_name_or_path": MODEL_NAME,
            "model_patch": patch or "",
        })
        attempted_ids.append(iid)
        # write predictions incrementally so a crash mid-run doesn't lose earlier work
        write_predictions(preds)

    if not attempted_ids and not args.grade_only:
        log("nothing to do")
        return

    if args.grade_only:
        # grade everything currently in predictions.jsonl
        preds = [json.loads(l) for l in PRED_PATH.read_text().splitlines() if l.strip()]
        attempted_ids = [p["instance_id"] for p in preds]

    if not attempted_ids:
        log("no instances to grade")
        return

    resolved_map, log_path = grade(attempted_ids, args.run_id)

    for p in preds:
        iid = p["instance_id"]
        if iid not in attempted_ids:
            continue
        resolved = resolved_map.get(iid)
        append_result({
            "instance_id": iid,
            "resolved": resolved,
            "patch_len": len(p["model_patch"]),
            "run_id": args.run_id,
            "ts": time.time(),
        })

    n = len(attempted_ids)
    n_resolved = sum(1 for iid in attempted_ids if resolved_map.get(iid))
    log(f"=== DONE: {n_resolved}/{n} resolved ({100*n_resolved/n:.1f}%) | run log: {log_path} | results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
