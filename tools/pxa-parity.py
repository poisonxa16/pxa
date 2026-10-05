#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
"""pxa-parity: do pxa-launch, a bare llama-server and the container pick the same flags?

For each (card set, model), WITHOUT touching a GPU (dry-run), five columns:
  table      pxa-launch's OWN tables (PXA_LAUNCH_TABLES_ONLY=1: the engine is not asked). This is
             the independent answer: it was written before the engine registry and shares no code
             with it.
  launcher   pxa-launch as shipped (it asks the engine, then prints the flags it will pass)
  engine     PXA_EXPLAIN=1 PXA_TOPOLOGY=<set> llama-server -m MODEL  (what a bare boot runs)
  c-args     the container entrypoint with engine arguments: pxa-entrypoint.sh -m MODEL
  c-default  the container entrypoint with NO arguments (the docker default path): it runs
             pxa-launch --explain on PXA_MODEL. With --docker IMAGE (run this harness on the
             docker host) it runs as a real `docker run IMAGE` with no arguments, the build
             mounted at /opt/pxa.
Flags compared: -sm -b -ub -fa -ngl -c.
Verdicts:
  PASS        all five agree on every flag (for -c see below)
  TABLE-DIFF  the four shipped paths agree, the launcher's own table says something else (the
              flags are listed; an INFERRED launcher row the engine does not treat as a cell)
  FAIL        the shipped paths disagree
  n/a         the launcher refuses the seat (nothing to compare)
-c: the launcher passes a recipe row's -c when a row matches; the engine does not know the rows and
picks np*4096. A -c difference that is exactly "row ctx vs 4096" is reported as ROW-CTX and not
counted as a failure; any other -c difference is.

  PXA_ENGINE_DIR=<build> tools/pxa-parity.py MODEL [MODEL...] [--topos 1x600,2x600,2x700,4x600]
                                                              [--docker IMAGE]
"""
import contextlib, importlib.util, io, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGS = ("sm", "b", "ub", "fa", "ngl", "c")


def load_launcher():
    spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(HERE, "pxa-launch.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parse_cmd(out):
    m = re.search(r"^\s*command:\s*(.*)$", out, re.M)
    cmd = m.group(1) if m else ""
    def flag(f):
        mm = re.search(r"(?:^|\s)" + re.escape(f) + r"\s+(\S+)", cmd)
        return mm.group(1) if mm else "-"
    return {k: flag("-" + k) for k in FLAGS}


def launcher_flags(L, model, topo, tables_only):
    argv = ["pxa-launch", "--explain", "--no-interactive", "--no-tui", "--model", model,
            "--gpus", ",".join(str(i) for i in range(int(topo.split("x")[0]))), "--accept-unmeasured"]
    # NOT --engine llama: a forced engine skips the recipe table (decide() returns before step 7),
    # which is the very table this column exists to read
    old, olde = sys.argv, dict(os.environ)
    sys.argv = argv
    os.environ["PXA_LAUNCH_FAKE_GPUS"] = topo
    if tables_only:
        os.environ["PXA_LAUNCH_TABLES_ONLY"] = "1"
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            try:
                L.main()
            except SystemExit:
                pass
    finally:
        sys.argv = old
        os.environ.clear()
        os.environ.update(olde)
    out = buf.getvalue()
    return parse_cmd(out), out


def old_engine_cell(topo, n_expert):
    """ggml_backend_cuda_pxa_suggest_batch() as it stood before the registry (rc-base, unchanged in
    ggml/src/ggml-cuda.cu): what the engine applied at load when the launcher passed no -b/-ub."""
    n, cc = (int(x) for x in topo.split("x"))
    if n == 2 and cc == 700: return 8192, 2048
    if n == 2 and cc == 600: return 8192, 256
    if n == 1 and cc == 610: return 2048, 768
    if n == 4 and cc == 600:
        return 2048, (2048 if (n_expert is None or n_expert != 0) else 256)
    return None


def explain(cmd, topo, env_extra=None):
    env = dict(os.environ)
    env.update({"PXA_EXPLAIN": "1", "PXA_TOPOLOGY": topo})
    env.update(env_extra or {})
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("PXA_EXPLAIN_JSON "):
            j = json.loads(line[len("PXA_EXPLAIN_JSON "):])
            p = j["picks"]
            def v(k):
                x = p.get(k, {}).get("value") or "-"
                return "-" if x == "adaptive" else x
            f = {k: v(k) for k in FLAGS}
            # an INFERRED cell (the gemma4 PXQ3 pair, the V100-pair MoE flagship) still APPLIES its -b/-ub:
            # only OFF / ADAPTIVE / USER mean "no cell, the engine keeps its own"
            if p.get("b", {}).get("status") not in ("MEASURED", "RULE", "INFERRED"):
                f["b"] = "-"
            return f, j
    return None, r.stdout[-400:] + r.stderr[-400:]


def container_default(model, topo, E, image):
    """the no-argument entrypoint path: pxa-launch --explain on PXA_MODEL, cards described by topo"""
    n = int(topo.split("x")[0])
    env = {"PXA_MODEL": model, "PXA_GPUS": ",".join(str(i) for i in range(n)),
           "PXA_LAUNCH_EXTRA": "--explain --accept-unmeasured --no-tui",
           "PXA_LAUNCH_FAKE_GPUS": topo}
    if image:
        mdir = os.path.dirname(os.path.abspath(model))
        cmd = ["docker", "run", "--rm", "--network", "none", "--cpuset-cpus", "8-35,44-71",
               "-v", f"{os.path.abspath(E)}:/opt/pxa:ro",
               "-v", f"{os.path.dirname(HERE)}/tools:/opt/pxa-tools:ro",
               "-v", f"{mdir}:/models:ro", "-e", "PXA_HOME=/opt/pxa", "-e", "PXA_ENGINE_DIR=/opt/pxa",
               "--entrypoint", "/opt/pxa-tools/pxa-entrypoint.sh",
               "--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=none", "-e", "HOME=/tmp", "-e", "PXA_LAUNCH_STATE=/tmp/pl"]
        for k, v in env.items():
            cmd += ["-e", f"{k}={('/models/' + os.path.basename(model)) if k == 'PXA_MODEL' else v}"]
        cmd += [image]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    else:
        e = dict(os.environ)
        e.update(env)
        e.update({"PXA_HOME": E, "PXA_ENGINE_DIR": E})
        r = subprocess.run([os.path.join(HERE, "pxa-entrypoint.sh")], capture_output=True, text=True,
                           timeout=300, env=e)
    return parse_cmd(r.stdout), r.stdout + r.stderr


def main():
    argv = sys.argv[1:]
    topos, image = "1x600,2x600,2x700,4x600", None
    if "--topos" in argv:
        i = argv.index("--topos"); topos = argv[i + 1]; del argv[i:i + 2]
    if "--docker" in argv:
        i = argv.index("--docker"); image = argv[i + 1]; del argv[i:i + 2]
    models = argv
    E = os.environ.get("PXA_ENGINE_DIR")
    if not E:
        sys.exit("set PXA_ENGINE_DIR to the build dir holding bin/llama-server")
    L = load_launcher()
    tally = {"PASS": 0, "TABLE-DIFF": 0, "FAIL": 0, "n/a": 0, "ROW-CTX": 0}
    fmt = lambda f: "?" if not f else " ".join(f"{k}={f[k]}" for k in FLAGS)
    for model in models:
        for topo in topos.split(","):
            tf, tout = launcher_flags(L, model, topo, True)
            lf, lout = launcher_flags(L, model, topo, False)
            ef, ej = explain([f"{E}/bin/llama-server", "-m", model], topo)
            af, aj = explain([os.path.join(HERE, "pxa-entrypoint.sh"), "-m", model], topo, {"PXA_HOME": E})
            df, dout = container_default(model, topo, E, image)
            # the tables-only launcher leaves -b/-ub to the engine where no row covers the seat; the
            # engine it was written against then applied its own card cell at load
            if tf["ub"] == "-" and isinstance(ej, dict):
                ne = (ej.get("model") or {}).get("n_expert")
                oc = old_engine_cell(topo, ne if isinstance(ne, int) and ne >= 0 else None)
                if oc:
                    tf["b"], tf["ub"] = str(oc[0]), str(oc[1])
            name = f"{os.path.basename(model)[:40]:<40} {topo:<6}"
            refused = re.search(r"\[(R-\d+[A-Z]?)\] REFUSING", lout)
            if refused and lf["sm"] == "-":
                tally["n/a"] += 1
                print(f"{name} n/a (launcher refuses {refused.group(1)}); engine: {fmt(ef)}")
                continue
            eng = re.search(r"ENGINE = (\S+)", lout)
            if eng and eng.group(1) != "llama":
                tally["n/a"] += 1
                print(f"{name} n/a (launcher picks engine {eng.group(1)}); engine: {fmt(ef)}")
                continue
            notes = []
            # -c: row ctx vs the engine's np*4096
            ctx_row = False
            if ef and lf["c"] != ef["c"] and ef["c"] == "4096" and lf["c"] != "-":
                ctx_row = True
                notes.append(f"ROW-CTX launcher -c {lf['c']} (recipe row) vs engine -c 4096")
            def same(x, y, skip_c):
                return x is not None and y is not None and all(
                    x[k] == y[k] for k in FLAGS if not (skip_c and k == "c"))
            shipped_ok = (same(lf, ef, ctx_row) and same(ef, af, False) and same(lf, df, False))
            if not shipped_ok:
                v = "FAIL"
            elif not same(tf, lf, False):
                v = "TABLE-DIFF"
                notes.append("table: " + ", ".join(f"-{k} {tf[k]} (shipped {lf[k]})" for k in FLAGS if tf[k] != lf[k]))
            else:
                v = "PASS"
            tally[v] += 1
            tally["ROW-CTX"] += ctx_row
            print(f"{name} {v}")
            print(f"    table     {fmt(tf)}\n    launcher  {fmt(lf)}\n    engine    {fmt(ef)}\n"
                  f"    c-args    {fmt(af)}\n    c-default {fmt(df)}{'  [docker ' + image + ']' if image else ''}")
            for n_ in notes:
                print("    " + n_)
            if v == "FAIL" and os.environ.get("PXA_PARITY_VERBOSE"):
                print(lout[-2000:]); print(dout[-2000:])
    print("TALLY " + "  ".join(f"{k}={v}" for k, v in tally.items()))
    return 1 if tally["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
