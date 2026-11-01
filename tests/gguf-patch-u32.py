#!/usr/bin/env python3
"""gguf-patch-u32.py IN.gguf OUT.gguf KEY=VALUE...: copy a GGUF and overwrite existing uint32 metadata values in place
(test-hparams-hostile.sh crafts hostile hparams from a good tiny model this way)."""
import os, shutil, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gguf-py'))
try:
    import yaml  # noqa: F401  (gguf.metadata imports it; the reader never needs it)
except ImportError:
    import types
    sys.modules['yaml'] = types.ModuleType('yaml')
import gguf  # noqa: E402


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    shutil.copyfile(src, dst)
    r = gguf.GGUFReader(dst, 'r+')
    for kv in sys.argv[3:]:
        key, val = kv.split('=', 1)
        f = r.get_field(key)
        if f is None or len(f.types) != 1 or f.types[0] != gguf.GGUFValueType.UINT32:
            sys.exit(f'{key}: not a uint32 scalar in {src}')
        f.parts[f.data[0]][0] = int(val)
    r.data.flush()


if __name__ == '__main__':
    main()
