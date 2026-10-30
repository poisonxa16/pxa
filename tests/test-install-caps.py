#!/usr/bin/env python3
"""install.sh GPU check: which compute capabilities the one-command installer accepts.

The release builds carry sm_60, sm_61, sm_70 and sm_75 kernels (owner rule 2026-10-20: every release
build ships CMAKE_CUDA_ARCHITECTURES "60;61;70;75"), so a host whose cards are only Turing (T4,
RTX 20xx, GTX 16xx: compute capability 7.5) must get past step 2 instead of dying with "this host
reports only: 7.5". Runs install.sh with a fake nvidia-smi (the caps below) and a fake curl that
fails, so the script stops at step 3 ("Could not read the PXA release") whenever the GPU check
passed; --docker skips the glibc check so the test does not depend on the build host's libc.
"""
import os
import platform
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTALL = os.path.join(ROOT, "install.sh")

# caps reported by the fake nvidia-smi -> (gpu check passes, warning expected)
CASES = [
    (["6.0"], True, False),
    (["6.1"], True, False),
    (["7.0"], True, False),
    (["7.5"], True, False),               # Turing alone: accepted (was refused before sm_75)
    (["7.5", "7.5"], True, False),
    (["6.0", "7.5"], True, False),
    (["7.5", "8.6"], True, True),         # a supported card plus a newer one: warn, install
    (["8.6"], False, True),               # only unsupported cards: refuse
    (["5.2"], False, True),
    (["7.2"], False, True),               # Xavier-class sm_72 is not in the build
]


def run(caps, tmp):
    fake = os.path.join(tmp, "bin")
    os.makedirs(fake, exist_ok=True)
    with open(os.path.join(fake, "nvidia-smi"), "w") as f:
        f.write("#!/bin/sh\n" + "".join(f"echo ' {c}'\n" for c in caps))
    with open(os.path.join(fake, "curl"), "w") as f:
        f.write("#!/bin/sh\nexit 22\n")
    for n in ("nvidia-smi", "curl"):
        os.chmod(os.path.join(fake, n), 0o755)
    env = dict(os.environ, PATH=fake + os.pathsep + os.environ.get("PATH", ""), HOME=tmp)
    p = subprocess.run(["bash", INSTALL, "--docker"], env=env, capture_output=True, text=True, timeout=60)
    return p.returncode, p.stderr


def main():
    if platform.machine() != "x86_64":
        print("SKIP: install.sh is x86_64-only")
        return 0
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for caps, ok, warn in CASES:
            rc, err = run(caps, tmp)
            passed_gpu = "Could not read the PXA release" in err
            refused = "this host reports only" in err
            warned = "install.sh: warning:" in err
            good = rc != 0 and passed_gpu == ok and refused == (not ok) and warned == warn
            print(f"{'ok  ' if good else 'FAIL'} caps={','.join(caps):<10} rc={rc} gpu_check={'pass' if passed_gpu else 'refuse' if refused else '?'} warn={int(warned)}")
            if not good:
                print("     stderr: " + err.strip().replace("\n", "\n             "))
                bad += 1
    print(f"{len(CASES) - bad}/{len(CASES)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
