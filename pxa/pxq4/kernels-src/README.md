# PXQ4 kernel sources

## v12b

Opt-in env `PXQ4_MMV_MMA=1` routes M>=5 to the tensor-core (wmma) path;
M<=4 is untouched and bit-identical to v11. The shared mmv arenas are
sized from the shape and frozen on the first captured request. The shipped
library (`libpxq4_sm70_v12b.so`, md5 `2abdb4d8c4fb6017b24121d198ad5951`,
2715168 bytes) was built with `build_v12b.sh`. A rebuild reproduces every
host section byte for byte; only `.nv_fatbin` differs run to run because
`-lineinfo` device debug info is not deterministic in this toolchain, so
compare `readelf -SW` sections and exported symbols rather than the md5.
