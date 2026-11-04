#!/usr/bin/env python3
"""convert_hf_to_gguf.py: the no-mmap safetensors reader (_PxaSafetensorsReader, default on Windows, PXA_CONVERT_NO_MMAP=1 elsewhere)
returns the same keys in the same order and the same tensors as safetensors.safe_open, so a GGUF written either way is byte-identical.
2026-11-03: on Windows the memory map of every shard was charged to the commit limit and a 72 GB model failed with os error 1455.
python3 tests/test-convert-nommap.py"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gguf-py"))
sys.path.insert(0, ROOT)
try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    import convert_hf_to_gguf as C
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "needs torch + safetensors")
class Reader(unittest.TestCase):
    def test_same_keys_and_tensors_as_safe_open(self):
        g = torch.Generator().manual_seed(1)
        T = {"b.weight": torch.randn(3, 5, 7, generator=g).to(torch.bfloat16), "a.weight": torch.randn(4, 2, generator=g),
             "z.bias": torch.randn(9, generator=g).half(), "m.idx": torch.arange(6, dtype=torch.int64).reshape(2, 3),
             "e.empty": torch.zeros(0, 4), "s.scalar": torch.tensor([2.5]).to(torch.bfloat16)}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.safetensors")
            save_file(T, p, metadata={"format": "pt"})
            r = C._PxaSafetensorsReader(p)
            with safe_open(p, framework="pt", device="cpu") as f:
                self.assertEqual(list(r.keys()), list(f.keys()))
                for k in f.keys():
                    a, b = r.get_slice(k), f.get_slice(k)
                    self.assertEqual(a.get_dtype(), b.get_dtype(), k)
                    self.assertEqual(list(a.get_shape()), list(b.get_shape()), k)
                    x, y = a[:], b[:]
                    self.assertEqual((x.dtype, tuple(x.shape)), (y.dtype, tuple(y.shape)), k)
                    self.assertTrue(torch.equal(x, y), k)
                    # through the lazy wrapper the converter uses
                    lz = C.LazyTorchTensor.from_safetensors_slice(a)
                    self.assertTrue(torch.equal(C.LazyTorchTensor.to_eager(lz), y), k)

    def test_switch(self):
        for v, want in (("1", True), ("0", False)):
            os.environ["PXA_CONVERT_NO_MMAP"] = v
            self.assertIs(C._pxa_no_mmap(), want)
        os.environ.pop("PXA_CONVERT_NO_MMAP")
        self.assertIs(C._pxa_no_mmap(), os.name == "nt")


if __name__ == "__main__":
    unittest.main()
