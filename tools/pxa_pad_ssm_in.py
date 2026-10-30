#!/usr/bin/env python3
# pxa_pad_ssm_in.py IN OUT -- copy a GGUF, padding every blk.N.ssm_in.weight to a multiple of 64 rows with zero rows (the PXQ /
# PXQN slab codecs need 64-row panels; Nemotron-3-Nano-4B has d_in_proj = 17504). Zero rows are exact in every type
# (bf16/f16/f32 zeros, q8_0 blocks with d = 0). The engine loads the padded projection (the Nemotron-H loader).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
import numpy as np
import gguf
src, dst = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
arch = r.fields["general.architecture"].contents()
w = gguf.GGUFWriter(dst, arch=arch, use_temp_file=False)
for f in r.fields.values():
    if f.name.startswith("GGUF.") or f.name == "general.architecture":
        continue
    vt = f.types[0]
    if vt == gguf.GGUFValueType.ARRAY:
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1])
    else:
        w.add_key_value(f.name, f.contents(), vt)
w.add_key_value("pxa.ssm_in.padded_rows", 64, gguf.GGUFValueType.UINT32)
npad = 0
for t in r.tensors:
    shape = [int(x) for x in t.shape]          # ggml order
    data = t.data
    if t.name.endswith("ssm_in.weight") and len(shape) >= 2 and shape[1] % 64:
        pad = ((shape[1] + 63) // 64) * 64 - shape[1]
        data = np.concatenate([np.asarray(data), np.zeros((pad,) + tuple(data.shape[1:]), dtype=data.dtype)], axis=0)
        shape[1] += pad
        npad += 1
    # the reader returns quantized and bf16/f16 tensors as uint8 BYTE arrays (rows, row_bytes); the writer derives the
    # logical shape from that byte shape when raw_dtype is given, so no raw_shape here (a logical raw_shape would be
    # read as a byte shape and halve K for bf16)
    w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"padded {npad} ssm_in tensors -> {dst}")
