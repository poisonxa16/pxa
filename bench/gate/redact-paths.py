#!/usr/bin/env python3
"""Redact build-machine paths from one gate record line on stdin.

The gate runner calls this as it writes each line of LAST-RUN.md. A later
sed is not the fix: the next run would write the raw paths again.

Replacements, longest specific path first:
  PXA_MODEL    -> <model>     (and the file's base name, the internal filename)
  PXA_BIN      -> <package>
  PXA_WORKDIR  -> <logdir>
  PXA_PROMPTS  -> <prompts>
  PXA_ROOT     -> <repo>
  PXA_HOST / PXA_FQDN as a whole word -> <host>
  any /mnt, /root, /home, /boot path that is left -> <mnt> <root> <home> <boot>
"""
import os
import re
import sys

_PATH = re.compile(r"/(?:mnt|home)/\S+|/(?:root|boot)(?:/\S*)?")


def _strip_tail(m):
    s = m.group(0)
    # keep a trailing punctuation mark that is not part of the path
    cut = len(s)
    while cut > 0 and s[cut - 1] in ".,:;)]}\"'":
        cut -= 1
    body, tail = s[:cut], s[cut:]
    if body.startswith("/mnt"):
        tag = "<mnt>"
    elif body.startswith("/home"):
        tag = "<home>"
    elif body.startswith("/root"):
        tag = "<root>"
    else:
        tag = "<boot>"
    return tag + tail


def redact(text):
    pairs = (
        ("PXA_MODEL", "<model>"),
        ("PXA_BIN", "<package>"),
        ("PXA_WORKDIR", "<logdir>"),
        ("PXA_PROMPTS", "<prompts>"),
        ("PXA_ROOT", "<repo>"),
    )
    for key, placeholder in pairs:
        value = os.environ.get(key, "")
        if not value:
            continue
        text = text.replace(value, placeholder)
        if key == "PXA_MODEL":
            base = os.path.basename(value.rstrip("/"))
            if base and base != value and len(base) > 4:
                text = text.replace(base, placeholder)
    for key in ("PXA_FQDN", "PXA_HOST"):
        host = os.environ.get(key, "")
        if len(host) < 4 or host.lower() in ("localhost", "localhost.localdomain"):
            continue
        text = re.sub(r"(?<![A-Za-z0-9._-])" + re.escape(host) + r"(?![A-Za-z0-9._-])", "<host>", text)
    return _PATH.sub(_strip_tail, text)


def main():
    sys.stdout.write(redact(sys.stdin.read()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
