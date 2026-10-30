# Per-arch cubin identity between two release builds

1. `xelf-cubins.sh <work_dir> <out.json> <binary>...`: extracts every cubin (`cuobjdump -xelf all`; PE `.nv_fatb` fatbins via
   check-win-package.py) in docker nvidia/cuda:12.8.1-devel and writes sorted sha256 per sm arch.
2. `cubin-compare.py old.json new.json sm_60 sm_61 sm_70 sm_75`: byte identity per arch (SAME/DIFF + new-only archs).
3. `sass-compare.sh <old_work/bN> <new_work/bN> [jobs]` (run inside the same cuda image): for byte-different cubins compares
   `cuobjdump -sass` raw, then normalised for the arch-list tags nvcc injects when an arch is added (CCCL ABI namespace
   `CUB_/THRUST_<ver>_<archs>_NS` incl. its mangled length, `__CUDA_ARCH_LIST__` string). Verdicts: BYTES-SAME,
   SASS-SAME(RAW-SASS-SAME|RAW-SASS-DIFF), SASS-DIFF (with function counts and the first new-only function).

Byte DIFF alone is expected when the arch list changes (RC2.1 sm_86 add, 10-21). SASS-DIFF after normalisation is a real
code difference: check the source refs of both builds first (incremental Windows trees fed by git-archive tars keep commit-time
mtimes, so ninja can skip changed headers: apply deltas with `tar -m` or do a full rebuild for a release).

## ELF (.so) variant
`sass-compare-elf.sh OLD.so NEW.so` pairs the k-th cubin per arch (ELF cubins are numbered globally) and prints the same BYTES-SAME / SASS-SAME / SASS-DIFF lines as sass-compare.sh, with the CCCL arch-list namespace normalised.
