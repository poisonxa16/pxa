# PXA Control chat harness suite

`harness_suite.py` checks the Chat tab's agent loop (the Assistant view: `tools/pxa_chat`, and the Classic view's
engine proxy) against the edge-case catalogue: streaming, tool calls, context, cancellation, concurrency, errors,
state, parameters, determinism, UX and security. It talks to a real PXA Control over its HTTP API, exactly as the
page does, and needs no GPU and no model: scripted FAKE llama-servers (`harness_fake_server.py`) play every bad
server, bad network and bad model behaviour. With `--endpoint` it adds live cases against a real llama-server.

## Run it

```bash
cd <repo>                                   # your checkout
tools/pxa_control/tests/run-harness-suite.sh --out /tmp/harness-$(date +%H%M)
# with the live cases (only a server on 127.0.0.1; on the PXA box only the judge seat):
curl -s http://127.0.0.1:8263/health && \
  tools/pxa_control/tests/run-harness-suite.sh --endpoint http://127.0.0.1:8263 --out /tmp/harness-live
```

- Needs: Linux, python3 (stdlib only), `pgrep`; `node` for the renderer cases (they SKIP without it); `pytest` for
  the unit cases (or pass `--no-unit`).
- Takes about 12 minutes in FAKE mode (some cases wait on purpose: a 60 s stalled reader, a 20 s silent tool, slow
  streams), about 2 more with `--endpoint`.
- CPU only. The wrapper pins it to cores 36-53 at nice 15 (`PXA_HARNESS_CPUS`, `PXA_HARNESS_NICE`). On the PXA box,
  respect posted QUIET brackets: 36-53 are the cores to use when 54-71 or 0-17 are quiet.
- Flags: `--only S06,T16` / `--skip ...` (case ids), `--no-unit`, `--endpoint URL`, `--live-only` (with `--endpoint`,
  only the live cases), `--port N` (Control test port), `--keep` (leave Control and the fakes running to debug),
  `--list` (print the catalogue).

## What it starts (and what it never touches)

- A scratch PXA Control from this checkout, as a child process: its own config dir under `<out>/scratch-*/cfg`, a
  random token passed in the environment (never printed), a free port, fake GPUs (`PXA_LAUNCH_FAKE_GPUS`), discovery
  off. It never reads or writes `~/.config/pxa`, never attaches to the live Control, never starts an engine.
- Nine FAKE servers on free ports: A (tools, 8k window, 2 slots, attached), B (no native tools: text protocol),
  C (2k window, 2 slots: context cases), D (no --jinja: tools refused), E (no /props), F (1 slot: queueing),
  G (killed and restarted mid-turn), V (vision), N (lists no model). B..N are registered like servers the launcher
  started (`cfg/launched/<pid>.json`).
- A request picks a scripted behaviour with a marker in the user's message: `[[fake:NAME]]` or `[[fake:NAME:ARG]]`
  (`python3 tools/pxa_control/tests/harness_fake_server.py --list` prints all of them). No marker = a plain answer.
- Everything is stopped at the end (unless `--keep`).

## Output

- `<out>/harness-report.md`: one row per case: id, expected, observed, result, log excerpt.
- `<out>/harness-report.json`: the same plus timings, the git head, the counts, the mode.
- `<out>/control.log`, `<out>/fake-*.log`: the processes' own output. A fake also keeps an in-memory log
  (`GET /fake/log`, `GET /fake/requests` on its port while `--keep` holds it up): every request it received and every
  client disconnect it saw (that is how "Stop reached the server" is proven).
- Console: one line per case: `STATUS  ID  seconds  observed`.

| Status | Meaning | Fails the run? |
|---|---|---|
| PASS | the expected behaviour was observed | no |
| FAIL | it was not: a regression or a new bug | **yes** |
| XFAIL | a catalogued gap that is not fixed yet; observed and reported (a todo exists) | no |
| XPASS | a catalogued gap that now passes: tell the coordinator, then drop the xfail mark | no |
| SKIP | not for this mode (a live case in FAKE mode, a unit case with `--no-unit`, no node) | no |
| MANUAL | needs a browser, a human, a second machine or hours; the reason is in the row | no |

Exit code: 0 = no FAIL, 1 = at least one FAIL, 2 = the suite could not start (the report then has `setup_error`).

## When something fails

1. Read the row in `harness-report.md` (observed + log excerpt), then `control.log`.
2. Re-run just that case and keep the processes: `run-harness-suite.sh --only T16 --keep --out /tmp/h1`, then look at
   the fake's view: `curl -s http://127.0.0.1:<fake port>/fake/requests | python3 -m json.tool | less`
   (the ports are in the report header). Stop the leftovers with `pkill -f harness_fake_server.py` and the Control
   child (`pkill -f -- '--_control-child'`).
3. A FAIL in a case that passed before is a regression in `tools/pxa_chat` or `tools/pxa_control`: report the case id,
   the observed line and the commit; do not edit the case to make it pass.

## Coverage

Case ids are the edge-case catalogue ids: S streaming, T tools, C context,
X cancellation, K concurrency, E errors, ST state, P parameters, D determinism, U UX, SEC security, plus L01-L05
(live only) and E10b (renderer). Some ids are checked by the unit files (`tests/test-pxa-chat.py`,
`tools/pxa_control/tests/test_chat_harness.py`) and some are aliases of another case (same check). `--list` shows
which is which. The pytest file `test_chat_harness.py` runs on its own too:
`python3 -m pytest tools/pxa_control/tests/test_chat_harness.py -q`.

## Adding a case

1. If it needs a new server behaviour, add a scenario to `harness_fake_server.py` (`SCENARIOS` + a branch in
   `_stream`; multi-step tool flows read `phase` = tool results since the user message).
2. Add a function in `harness_suite.py` with `@case("ID", "expected behaviour")`; use `env.c.run(...)` (returns a Turn
   with `.events`, `.end_type`, `.text`, `.kinds()`, `.codes()`), `check(cond, observed)`, and return the observed
   line. Use `xfail="why + todo"` for a known gap, `modes=("fake", "live")` if it also makes sense on a real server.
3. Run `--only ID` until it passes, then a full run.
