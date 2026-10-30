#!/usr/bin/env python3
"""Clean-package scan of a Nuitka ONEFILE binary (macOS/Linux layout) -- the onefile's payload is zstd-compressed, so a plain
`strings`/gate pass over the outer file sees almost nothing. This unpacks the payload (same bytes the app unpacks to its cache),
makes a scan tree: every file as shipped, but each Mach-O / ELF as the PRINTABLE STRINGS of the file wrapped in an ELF header,
so the gate's binary-scope rules (scripts/release-scan.py, unmodified) run on it; third_party.zip (plain LGPL .py files kept as a
replaceable archive on purpose) is also unzipped so its contents are scanned, and the zip itself is the one known 'nested-archive' hit.

    PYTHONPATH=<dir with zstandard> python3 scan-mac-payload.py (not -I: PYTHONPATH is needed) <pxa-control onefile> <work dir>
  then: PXA_RELEASE_BLOCKLIST=... python3 scripts/release-scan.py <work>/tree-all [<work>/tree-nozip]
Needs the 'zstandard' module (the Nuitka build venv has it; on the box: pip install --target <dir> zstandard)."""
import os
import re
import shutil
import struct
import sys

import zstandard

SR = re.compile(rb"[\x20-\x7e\t]{4,}")
MAGICS = (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\x7fELF")


def payload(path):
    d = open(path, "rb").read()
    i = d.find(b"KAY")
    while i >= 0 and d[i + 3:i + 7] != b"\x28\xb5\x2f\xfd":
        i = d.find(b"KAY", i + 1)
    if i < 0:
        sys.exit("no Nuitka onefile payload found")
    return zstandard.ZstdDecompressor().decompressobj().decompress(d[i + 3:])


def entries(d):
    i = 0
    while i < len(d):
        j = d.index(b"\0", i)
        if j == i:
            return
        name = d[i:j].decode()
        i = j + 1 + 1                      # name, NUL, flag byte
        size = struct.unpack("<Q", d[i:i + 8])[0]
        i += 8 + 4                         # size, 4-byte checksum
        yield name, d[i:i + size]
        i += size


def main(binary, work):
    shutil.rmtree(work, ignore_errors=True)
    n = m = 0
    for name, data in entries(payload(binary)):
        for sub in ("tree-all", "tree-nozip"):
            if sub == "tree-nozip" and name == "third_party.zip":
                continue
            q = os.path.join(work, sub, name)
            os.makedirs(os.path.dirname(q), exist_ok=True)
            if data[:4] in MAGICS:
                open(q, "wb").write(b"\x7fELF" + b"\0" * 12 + b"\n".join(SR.findall(data)))
            else:
                open(q, "wb").write(data)
        n += 1
        m += data[:4] in MAGICS
        if name == "third_party.zip":
            import io
            import zipfile
            zipfile.ZipFile(io.BytesIO(data)).extractall(os.path.join(work, "tree-nozip", "third_party_unzipped"))
    print(f"{n} files, {m} native binaries as strings -> {work}/tree-all (as shipped) and {work}/tree-nozip (zip unpacked)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
