#!/usr/bin/env python3
"""Real-browser check of the beta wording against a LOCAL licd with the Valued-only beta on (licd's default) and a beta
release out: the Update dialog's Beta card and the Store > Wallet (PXA Coins) line, before and after making a wallet that is not Valued.

    PXA_LICD_DIR=<store tree>/tools/pxa-licd python3 tests/beta-wording-gui-check.py [--themes dark,light] [--widths 1280,390] [--out DIR]
                                                                                  [--image mcr.microsoft.com/playwright/mcp:latest] [--require]

The licd + Control world is coins-gui-check.py's Cell (real licd, scratch db, throwaway keys, 127.0.0.1 only). No GPU. Exit 0 = every
check passed at every theme and width; skipped (exit 0, says so) when docker, the image or the licd tree is missing, unless --require."""
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("coins_gui_check", os.path.join(HERE, "coins-gui-check.py"))
CG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CG)
RELEASE = "4.0-beta.2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--themes", default="dark,light")
    ap.add_argument("--widths", default="1280,390")
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="mcr.microsoft.com/playwright/mcp:latest")
    ap.add_argument("--require", action="store_true")
    a = ap.parse_args()
    if not CG.have_docker(a.image):
        print("SKIP: docker or the image %s is not available" % a.image)
        return 1 if a.require else 0
    if not CG.LICD_DIR:
        print("SKIP: the licd tree with the coins routes is not found (PXA_LICD_DIR)")
        return 1 if a.require else 0
    sys.path.insert(0, CG.LICD_DIR)
    out = a.out or tempfile.mkdtemp(prefix="pxa-beta-wording-shots-")
    os.makedirs(out, exist_ok=True)
    rc = 0
    for theme in a.themes.split(","):
        for width in a.widths.split(","):
            cell = CG.Cell()
            from pxa_control import routes as R
            R._BETA_VIEW.update(card=None, at=-1e18, until=0.0)      # one Control process, a fresh world per cell: no card from the last one
            try:
                st = cell.lsvc.store
                st.beta_release_env = RELEASE              # a beta is out (as PXA_LIC_BETA_RELEASE would say)
                if not getattr(st, "beta_valued_only", False):
                    print("FAIL: this licd has no Valued-only beta rule (licd/valued-sync or later is needed)")
                    return 1
                cmd = ["docker", "run", "--rm", "--network", "host"] + CG.cpuset() + ["-e", "NODE_PATH=/app/node_modules", "-v", HERE + ":/t:ro",
                       "-v", out + ":/out", "--entrypoint", "node", a.image, "/t/beta-wording-gui-check.js", cell.url, width, "/out", theme]
                print("== %s %s px" % (theme, width), flush=True)
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                sys.stdout.write(r.stdout)
                if r.returncode != 0:
                    sys.stdout.write(r.stderr[-2000:])
                    rc = 1
            finally:
                cell.close()
    print("screenshots in", out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
