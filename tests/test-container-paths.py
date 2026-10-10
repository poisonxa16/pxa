#!/usr/bin/env python3
"""A documented container path must be one the container's own user can write.

The image starts as root so the entrypoint can chown a named volume, then drops to
an unprivileged user (`setpriv --reuid=pxq`, home `/work`); `/root` is 0700 root, so
nothing under it is reachable from inside the container.  Every `docker run` recipe in the
shipped docs is written for a user who will copy it verbatim, so a recipe that mounts a cache
volume at a path that user cannot write fails silently at run time and is found only by a user.

Found this way on 2026-10-09: v3.1's learned expert map (a headline feature, with measured
numbers in the release notes) writes under PXA_CACHE_DIR, the entrypoint defaulted that to
/root/.cache/pxa and exported it -- overriding the engine's own writable default of
$HOME/.cache/pxa -- and the docs and compose file repeated the same path.  In the image that
path is not writable, so the map could never persist and the warning told the user to mount a
volume at the unusable path.

Reads the tree only: no docker, no build, no GPU.  Exit 0 pass, 1 failure, 4 refused (the tree
does not look like the image's tree, so a pass would mean nothing).
"""
import os
import re
import sys

# Where a container-side path may be written.  Docs are read as the release ships them; the
# Dockerfile is read for the image's own user and home.
DOCS = ["docker/docker-compose.yml", "docker/README.md", "README.md", "docs/LAUNCHER.md"]
VOLUME_RE = re.compile(r'-v\s+([A-Za-z0-9_.-]+):(/[^\s"\'\\]+)')
CACHE_ENV_RE = re.compile(r'PXA_CACHE_DIR["\']?\s*[:=]\s*["\']?(/[^\s"\',\\]+)')
ENTRYPOINT_CACHE_RE = re.compile(r'^\s*dir=\$\{PXA_CACHE_DIR:-(\S+)\}', re.M)
MIN_VOLUMES = 2          # vacuity: a release that documents no volume cannot be judged


def image_user(root):
    """(user, home) the image runs as, read from docker/Dockerfile."""
    p = os.path.join(root, "docker", "Dockerfile")
    if not os.path.isfile(p):
        return None, None
    text = open(p, encoding="utf-8", errors="replace").read()
    user = None
    for m in re.finditer(r'^\s*USER\s+(\S+)', text, re.M):
        user = m.group(1)
    home = None
    for m in re.finditer(r'useradd\b[^\n]*?\s-d\s+(\S+)', text):
        home = m.group(1)
    # The image starts as root so the entrypoint can chown a named volume,
    # then drops. The user who must be able to write is the drop target.
    ep = os.path.join(root, "tools", "pxa-entrypoint.sh")
    if os.path.isfile(ep):
        etext_user = open(ep, encoding="utf-8", errors="replace").read()
        drop = re.search(r'setpriv\s+--reuid=(\S+)', etext_user)
        if not drop:
            drop = re.search(r'runuser\s+-u\s+(\S+)', etext_user)
        if drop:
            user = drop.group(1)
    return user, home


def main():
    args = [a for a in sys.argv[1:] if a != "--root"]
    root = (args[0] if args else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    user, home = image_user(root)
    if not user or not home:
        print("REFUSED: cannot read the image's USER/useradd home from docker/Dockerfile")
        return 4
    if not os.path.isdir(os.path.join(root, "tools")):
        print("REFUSED: no tools/ here -- this is not the release tree")
        return 4
    root_ok = home.rstrip("/") + "/"

    fails, checked = [], 0
    for rel in DOCS:
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            continue
        for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
            for m in list(VOLUME_RE.finditer(line)) + list(CACHE_ENV_RE.finditer(line)):
                path = m.group(m.lastindex)
                checked += 1
                if not path.startswith(root_ok):
                    fails.append("%s:%d names %s -- the image's user %s (%s) cannot write it"
                                 % (rel, i, path, user, home))
    if checked < MIN_VOLUMES:
        print("REFUSED: found only %d container paths in the docs -- wrong tree?" % checked)
        return 4

    # The entrypoint's own default must sit under the same home (or defer to $HOME).
    ep = os.path.join(root, "tools", "pxa-entrypoint.sh")
    if not os.path.isfile(ep):
        fails.append("tools/pxa-entrypoint.sh is missing from the tree")
    else:
        text = open(ep, encoding="utf-8", errors="replace").read()
        m = ENTRYPOINT_CACHE_RE.search(text)
        if not m:
            fails.append("tools/pxa-entrypoint.sh sets no PXA_CACHE_DIR default (the check's subject "
                         "is gone -- fix or delete this test, do not let it pass)")
        else:
            default = m.group(1).strip().strip('"')
            checked += 1
            if default.startswith("/") and not default.startswith(root_ok):
                fails.append("tools/pxa-entrypoint.sh defaults the cache to %s -- unreachable for %s"
                             % (default, user))

    # A named volume does NOT inherit the image directory's owner. Docker
    # creates it root:root and the mount hides the chown below. The entrypoint
    # has to chown that directory as root and then drop to the runtime user.
    if os.path.isfile(ep):
        etext = open(ep, encoding="utf-8", errors="replace").read()
        checked += 1
        drop_line = "setpriv --reuid=%s --regid=%s --init-groups" % (user, user)
        if drop_line not in etext or "chown %s:%s" % (user, user) not in etext:
            fails.append("tools/pxa-entrypoint.sh does not chown the cache and drop to %s; "
                         "a named volume stays root-owned and unwritable" % user)
        if "could not create" not in etext or "could not chown" not in etext:
            fails.append("tools/pxa-entrypoint.sh aborts when mkdir or chown fails "
                         "(a cache mount must warn and continue)")
        # The process that serves is pxq. The image must still START as root,
        # or this drop never runs and a named volume stays root-owned.
        if "id -u" not in etext or "eq 0" not in etext:
            fails.append("tools/pxa-entrypoint.sh does not take the root branch before dropping to %s" % user)
    dockerfile = os.path.join(root, "docker", "Dockerfile")
    if os.path.isfile(dockerfile):
        dtext = open(dockerfile, encoding="utf-8", errors="replace").read()
        last_user = None
        for m in re.finditer(r'^\s*USER\s+(\S+)', dtext, re.M):
            last_user = m.group(1)
        checked += 1
        if last_user != "root":
            fails.append("docker/Dockerfile final USER is %s; the entrypoint chown runs only as root, "
                         "then the server drops to %s" % (last_user, user))
        mounted = set()
        for rel in DOCS:
            p = os.path.join(root, rel)
            if not os.path.isfile(p):
                continue
            for m in VOLUME_RE.finditer(open(p, encoding="utf-8", errors="replace").read()):
                mounted.add(m.group(2).rstrip("/"))
        for path in sorted(mounted):
            if not path.startswith(root_ok):
                continue                      # already reported above
            made = re.search(r'mkdir\s+-p[^\n]*' + re.escape(path) + r'(?=[\s\\]|$)', dtext)
            owned = re.search(r'chown(\s+-R)?\s+\S*' + re.escape(user) + r'[^\n]*'
                              + re.escape(path) + r'(?=[\s\\]|$)', dtext)
            checked += 1
            if not made:
                fails.append("docker/Dockerfile never creates %s, so a named volume mounted there "
                             "comes out root:root and unwritable" % path)
            elif not owned:
                fails.append("docker/Dockerfile creates %s but never chowns it to %s" % (path, user))

    print("container paths checked: %d (image user %s, home %s)" % (checked, user, home))
    if fails:
        print("FAIL: %d finding(s)" % len(set(fails)))
        for f in sorted(set(fails)):
            print("  " + f)
        return 1
    print("PASS: every documented container path is writable by %s and pre-created by the image" % user)
    return 0


if __name__ == "__main__":
    sys.exit(main())
