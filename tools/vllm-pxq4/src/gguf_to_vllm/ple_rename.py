"""ple_rename.py -- give the PLE FP8 shards the key names the fork actually looks for.

    python3 -m gguf_to_vllm.ple_rename --src <ple-fp8 dir> --layer <L> [--link-into <ckpt dir>]

WHY THIS EXISTS. gguf_to_vllm.ple_fp8 wrote its 513 tensors as

    model.language_model.ple.ngram_embedding.shard_<k>.weight
    model.language_model.ple.ngram_embedding.weight_scale

and there is no module at that path. The n-gram table is owned by a DECODER LAYER, two levels
down: Qwen4ExpDecoderLayer builds ``self.ple`` (model.py:245-250, prefix ``<layer>.ple``),
Qwen4ExpPLELayer builds ``self.ple_embedding`` (ple_layer.py:884-891, prefix
``<layer>.ple.ple_embedding``), and Qwen4ExpNGramEmbedding builds ``self.ngram_embedding``
(ple_layer.py:522). Only that last module has the ``load_weights`` that understands
``ngram_embedding.shard_<k>.weight`` (ple_layer.py:787-840), and AutoWeightsLoader reaches it
by walking the checkpoint key's own path. So the key has to be

    model.language_model.layers.<L>.ple.ple_embedding.ngram_embedding.shard_<k>.weight

with <L> the ZERO-based layer the GGUF's ple_* tensors sit on. Under the old names the loader
walks into ``model.ple``, which does not exist, and 51 GB of table silently never loads.

WHY THE HEADER IS PATCHED IN PLACE AND 48 GB ARE NOT COPIED. Only the NAMES change: every data
byte is already correct and already gated (PLE-FP8-REPORT: 16/16 heads, rms 2.65 %, 0
saturated), and every tensor's dtype, shape and byte range is unchanged. A safetensors file is
``u64 header_len | JSON header | data``, so the header can be rewritten without touching the
data ONLY IF the new JSON still fits the declared slot -- the data section's start is that
number. The new names are 24 bytes longer each (8 per file), but the writer's ``__metadata__``
block is ~270 bytes of provenance, and dropping it to the bare ``{"format":"pt"}`` frees more
than the names cost. So the file is patched in place, byte-for-byte identical after the
header, and the provenance that block held is preserved in the sidecar written next to it
(ple-fp8-headers.orig.json) rather than lost.

Any file whose new header does NOT fit is copied instead (``--out``), with the data moved by
sendfile so the bytes never enter this process. Both paths verify by re-reading the result:
same tensor count, same (dtype, shape, data_offsets) per tensor, same file size.

IDEMPOTENT. A file already carrying the new names is left alone.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

OLD_PREFIX = "model.language_model.ple.ngram_embedding"
NEW_PREFIX_FMT = "model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding"
MIN_METADATA = {"format": "pt"}

_CHUNK = 1 << 26          # 64 MiB per sendfile call


def read_header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > (1 << 28):
            raise SystemExit(f"{path}: header claims {n} B; refusing to read it")
        return n, json.loads(f.read(n))


def _renamed(hdr: dict, old: str, new: str, path: str) -> dict:
    out: dict = {"__metadata__": dict(MIN_METADATA)}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        if not k.startswith(old):
            raise SystemExit(f"{path}: tensor {k!r} does not start with {old!r}; refusing to "
                             f"rewrite a file whose contents are not what this tool is for")
        out[new + k[len(old):]] = v
    return out


def _encode(hdr: dict, slot: int | None) -> bytes:
    """JSON, 8-aligned, padded out to ``slot`` when one is given."""
    blob = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    if slot is None:
        return blob + b" " * ((-len(blob)) % 8)
    if len(blob) > slot:
        raise ValueError("does not fit")
    return blob + b" " * (slot - len(blob))


def _verify(path: str, want: dict, want_size: int) -> None:
    n, got = read_header(path)
    tw = {k: v for k, v in want.items() if k != "__metadata__"}
    tg = {k: v for k, v in got.items() if k != "__metadata__"}
    if tg != tw:
        raise SystemExit(f"{path}: header round-trip mismatch "
                         f"({len(tg)} tensors read, {len(tw)} expected)")
    if os.path.getsize(path) != want_size:
        raise SystemExit(f"{path}: size is {os.path.getsize(path)}, expected {want_size}")


def patch_in_place(path: str, old: str, new: str) -> int:
    n, hdr = read_header(path)
    out = _renamed(hdr, old, new, path)
    blob = _encode(out, n)                       # raises ValueError if it does not fit
    size = os.path.getsize(path)
    fd = os.open(path, os.O_WRONLY)
    try:
        os.lseek(fd, 8, os.SEEK_SET)
        mv = memoryview(blob)
        while mv:
            mv = mv[os.write(fd, mv):]
        os.fsync(fd)
    finally:
        os.close(fd)
    _verify(path, out, size)
    return len(out) - 1


def copy_renamed(src: str, dst: str, old: str, new: str) -> tuple[int, int]:
    n, hdr = read_header(src)
    out = _renamed(hdr, old, new, src)
    blob = _encode(out, None)
    data_beg, data_len = 8 + n, os.path.getsize(src) - (8 + n)
    if data_len <= 0:
        raise SystemExit(f"{src}: data section is {data_len} B")
    tmp = dst + ".tmp"
    # RAW fds, no buffered writer: os.sendfile() writes at the FILE DESCRIPTOR's offset while a
    # Python buffered writer keeps its own, and mixing them puts the header on top of the data.
    fin = os.open(src, os.O_RDONLY)
    fout = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        head = memoryview(struct.pack("<Q", len(blob)) + blob)
        while head:
            head = head[os.write(fout, head):]
        off, left = data_beg, data_len
        while left:
            sent = os.sendfile(fout, fin, off, min(_CHUNK, left))
            if sent <= 0:
                raise SystemExit(f"{src}: sendfile returned {sent} with {left} B left")
            off, left = off + sent, left - sent
        os.fsync(fout)
    finally:
        os.close(fout)
        os.close(fin)
    _verify(tmp, out, 8 + len(blob) + data_len)
    os.replace(tmp, dst)
    return len(out) - 1, data_len


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="dir holding ple-fp8-*.safetensors")
    ap.add_argument("--layer", type=int, required=True,
                    help="ZERO-based decoder layer that owns the PLE stack (the ggml block "
                         "index of blk.<L>.ple_key.weight)")
    ap.add_argument("--out", default=None,
                    help="dir for files whose header will not fit in place (default: --src)")
    ap.add_argument("--link-into", default=None,
                    help="checkpoint dir to hard-link (or symlink across devices) the shards "
                         "into, so vLLM's *.safetensors glob finds them")
    ap.add_argument("--index-name", default="ple.safetensors.index.json")
    a = ap.parse_args(argv)

    new_prefix = NEW_PREFIX_FMT.format(layer=a.layer)
    out_dir = a.out or a.src
    with open(os.path.join(a.src, a.index_name)) as f:
        idx = json.load(f)
    files = sorted(set(idx["weight_map"].values()))
    print(f"{len(idx['weight_map'])} tensors in {len(files)} files\n"
          f"  {OLD_PREFIX}.*\n  -> {new_prefix}.*", flush=True)

    # Provenance first: the __metadata__ blocks the patch drops, kept beside the files.
    backup = os.path.join(a.src, "ple-fp8-headers.orig.json")
    if not os.path.exists(backup):
        orig = {}
        for fn in files:
            n, h = read_header(os.path.join(a.src, fn))
            orig[fn] = {"header_bytes": n, "metadata": h.get("__metadata__"),
                        "tensors": [k for k in h if k != "__metadata__"]}
        with open(backup, "w") as f:
            json.dump(orig, f, indent=1)
        print(f"  original headers recorded in {backup}", flush=True)

    patched = copied = skipped = 0
    for i, fn in enumerate(files, 1):
        src = os.path.join(a.src, fn)
        n, h = read_header(src)
        if all(k == "__metadata__" or k.startswith(new_prefix) for k in h):
            skipped += 1
            continue
        try:
            k = patch_in_place(src, OLD_PREFIX, new_prefix)
            patched += 1
            where = src
        except ValueError:
            dst = os.path.join(out_dir, fn)
            k, nb = copy_renamed(src, dst, OLD_PREFIX, new_prefix)
            copied += 1
            where = dst
            print(f"  [{i}/{len(files)}] {fn}: header did not fit, copied {nb/1e9:.2f} GB",
                  flush=True)
        if i % 8 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {fn}: {k} tensors -> {os.path.basename(where)}",
                  flush=True)

    weight_map = {new_prefix + k[len(OLD_PREFIX):]: v for k, v in idx["weight_map"].items()}
    meta = dict(idx.get("metadata", {}))
    meta["ple_hf_layer"] = str(a.layer)
    with open(os.path.join(out_dir, a.index_name), "w") as f:
        json.dump({"metadata": meta, "weight_map": weight_map}, f, indent=1)
    print(f"patched {patched} in place, copied {copied}, already-renamed {skipped}", flush=True)

    if a.link_into:
        os.makedirs(a.link_into, exist_ok=True)
        linked = 0
        for fn in files:
            src = os.path.abspath(os.path.join(out_dir if os.path.exists(
                os.path.join(out_dir, fn)) else a.src, fn))
            dst = os.path.join(a.link_into, fn)
            if os.path.lexists(dst):
                os.unlink(dst)
            try:
                os.link(src, dst)
            except OSError:
                os.symlink(src, dst)     # different filesystem
            linked += 1
        print(f"linked {linked} shards into {a.link_into}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
