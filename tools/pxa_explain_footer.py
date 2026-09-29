"""PXA_EXPLAIN_PAGE_v1 (2026-09-25, showcase lane).

A small, self-contained mirror of what `GET /pxa` shows on a running server, printed from
`pxa-launch --doctor` / `--explain` so the same reasoning is available before anything boots and
without a browser. Deliberately dependency-light: it does not resolve the engine binary or spawn a
subprocess (pxa-launch.py's own `doctor()` already prints the registry's picks+why by asking the
engine directly) -- this file only adds the two things that come from ELSEWHERE: per-card free/used
VRAM (arithmetic on the same `gpus` table pxa-launch.py already builds) and which PXA_*/PXQ_* knobs
are currently set in the environment. It never touches CUDA itself and never starts a process.

Kept in its own file, called from one small hook at the end of pxa-launch.py's argument parser, so
that file's much larger, concurrently-edited body stays untouched. See the call sites near
`if a.doctor:` / `if a.explain:`.
"""

import os
import re


def selected_cards(gpus, gpus_arg):
    """The subset of the `gpus` table (pxa-launch.py's gpu_table()) named by a --gpus string, or
    every card when nothing was named -- the same rule pxa-launch.py's own doctor() uses."""
    cards = {int(x) for x in re.split(r"[,\s]+", gpus_arg) if x.strip()} if gpus_arg else set()
    return [g for g in (gpus or []) if g[0] in cards] if cards else (gpus or [])


def active_env_levers():
    """Every PXA_*/PXQ_* variable set in THIS process's environment, name and value, sorted. Not
    the full default-on/user-set classification the engine's registry does (that needs the
    catalog, which lives in the engine binary, not here) -- just what a user would see if they ran
    `env | grep -E 'PXA_|PXQ_'` themselves, gathered in one place."""
    return sorted(k for k in os.environ if k.startswith("PXA_") or k.startswith("PXQ_"))


def print_footer(a, gpus):
    """Print the mirror block. `a` is pxa-launch.py's parsed argparse namespace (only .gpus is
    read); `gpus` is its gpu_table() result. Never raises past this function in normal use -- the
    caller in pxa-launch.py still wraps the call in try/except so a bug here can never break
    --doctor or --explain."""
    sel = selected_cards(gpus, getattr(a, "gpus", "") or "")
    if not sel:
        return
    w = 78
    print("-" * w)
    print("PXA explains itself (mirrors GET /pxa once this seat is serving requests):")
    for g in sel:
        idx, name, cc, total_mib, used_mib = g[0], g[1], g[2], g[3], g[4]
        free_mib = max(0, total_mib - used_mib)
        print("   card %-2s %-24s sm_%-3s %5d/%-5d MiB used  %5d MiB free"
              % (idx, name.replace("NVIDIA ", ""), cc, used_mib, total_mib, free_mib))
    envs = active_env_levers()
    if envs:
        print("   active levers (set in your environment):")
        for k in envs:
            print("      %s=%s" % (k, os.environ[k]))
    else:
        print("   active levers: none set in your environment (every PXA_*/PXQ_* lever is at its "
              "shipped default -- the engine's own -sm/-b/-ub/-fa picks above still apply).")
    print("   live decode/prefill t/s and speculation acceptance are only available once this seat")
    print("   is serving: see GET /pxa in a browser, or /pxa/explain for the raw JSON, once it is up.")
    print("-" * w)
