# Windows package acceptance checklist

What `scripts/check-win-package.sh` checks on the exact release files, and what stays manual. Nothing here uploads
or publishes. A package is accepted when the checker exits 0 and every manual item has evidence.

## How to run the review on the exact files

    cd <engine checkout>
    PXA_RELEASE_BLOCKLIST=<private block list> \
      scripts/check-win-package.sh <release-dir or pxa-engine-win64-cuda12.zip> \
        --ctest-log <windows ctest output> --cuobjdump --json <out.json>

Read-only on the inputs; niced on the build cores; work files under $CHECK_WIN_WORK (removed unless --keep);
no GPU (cuobjdump runs in nvidia/cuda:12.8.1-devel-ubuntu22.04 with --network none and no --gpus). Exit 0 = no FAIL.
Each line is PASS / FAIL / WARN / INFO / MANUAL + check id. Hits print file + rule id + count, never the matched text.
A package is ACCEPTED when the checker exits 0 AND every MANUAL line below is ticked with evidence.

## The checklist (A = automated by check-win-package.sh, M = manual, evidence in the PASS post)

### 1. Downloads and hashes
- [A] The release dir holds exactly: pxa-control-win64.exe, pxa-engine-win64-cuda12.zip, ggml-pxqn.dll (unless a PXQ-only
      package, --no-pxqn), RELEASE-NOTES-windows.md, SHA256SUMS.
- [A] SHA256SUMS verifies every listed file and lists every shipped file; sibling *.sha256 files (if any) match.
- [A] The sha256 list is printed first (INFO sha256 lines): paste it into the PASS post and the release notes.

### 2. Zip layout
- [A] Zip opens, CRC ok; member names use '/' (not '\'); no absolute or '..' members.
- [A] One top folder pxa-engine-win64-cuda12/ with README.txt, LICENSE, NOTICE, bin/.
- [A] bin/ has llama-server.exe, ggml.dll, llama.dll, cudart64_12.dll, cublas64_12.dll, cublasLt64_12.dll,
      vcruntime140.dll, vcruntime140_1.dll, msvcp140.dll.
- [A] Refused: sources (.c/.cpp/.cu/.h/.py/.def ...), build/debug files (.pdb/.lib/.exp/.ilk/.obj), scripts (.sh/.bat/.ps1),
      NVIDIA driver files (nvcuda/nvml/...), *.OLD* copies, secret-like names (*.pem, *.key, *.token, id_rsa*, .env ...),
      ggml-pxqn.dll inside the zip unless the package is built with --bundle-pxqn.
- [A] Import closure: every DLL any .exe/.dll imports (normal + delay-load) is in bin/, a Windows system DLL, or the driver
      (nvcuda.dll). A missing CUDA/MSVC runtime DLL is a FAIL here, not a user crash.
- [A] Every .exe/.dll is a PE32+ x64 image.

### 3. CUDA architectures
- [A] ggml.dll (or ggml-cuda.dll) carries device code; it and ggml-pxqn.dll have SASS for sm_60, sm_61, sm_70 (parsed from
      the .nv_fatb section: Linux cuobjdump does not read PE files, it says "does not contain device code").
- [A] The CUDA runtime DLLs also hold SASS for sm_60/61/70 (a CUDA 13 toolkit drops Pascal/Volta: that would FAIL here).
- [A, --cuobjdump] Cross-check: the first fatbin of each device-code DLL is cut out and listed with cuobjdump --list-elf.
- [A] WARN when one of our fatbins lacks a target arch, or when there is no PTX (then cards newer than sm_70 cannot JIT;
      the notes must then name the supported cards).

### 4. The PXQN library
- [A] ggml-pxqn.dll exports exactly the names of ggml/src/win-compat/ggml-pxqn.def (13 since ggml_pxqn_get_overdrive_api;
      the checker and the packager read the list from the .def). The 3 gu_* test hooks are not exported from the DLL on
      Windows (ggml-cuda.h GGML_PXQN_HOOK_API; a test build turns them back on with -DPXA_PXQN_EXPORT_TEST_HOOKS=ON).
      Any other name, a hook, or a missing .def name = FAIL; the packager refuses the same. `--allow-test-hooks`
      accepts a pre-fix or test build (.def + 3).
- [A] Not an *.OLD* all-symbols copy; carries sm_60/61/70 SASS.
- [A] LICENSE-ggml-pxqn.txt (the library's terms) beside the DLL, listed in SHA256SUMS, and in the zip next to LICENSE;
      no store words. [M] Its wording approved before the first upload.

### 5. Licences
- [A] LICENSE (MIT; the copyright notice must be in every copy) and NOTICE in the zip.
- [A] NOTICE names the NVIDIA CUDA runtime (CUDA EULA, Attachment A redistributables) and the Microsoft Visual C++ runtime.
      The packager now appends docs/windows-third-party-notices.txt to NOTICE.
- [M] Control exe: its bundled third-party Python packages (PIL, OpenSSL, sqlite, ...) are covered by the Control
      licence notice.

### 6. Clean-package scan gate (BETA-GATE.md step 0) on the exact files
- [A] scripts/release-scan.py with the private block list on the zip, and on the files beside it (exe, dll, notes, sums):
      public-text block list, secret names-only check (token shapes + the literal secret files the list names), `strings`
      of every ELF/PE file. No block list = FAIL (fails closed).
- [A] Control exe payload: pxa-control-win64.exe is a PyInstaller one-file exe; its Python modules, UI files and nested
      zips are zlib-compressed, so a plain `strings` pass does not see them. The checker unpacks the CArchive and the PYZ,
      scans our modules (pxa*) as text with the block list (wording rules included), scans third-party entries with the block
      list too, and runs the generic shapes on everything.
- [A] Generic shapes on ASCII and UTF-16LE strings of every .exe/.dll and every bundled file: internal lane names, host-private
      paths, a private GitHub repo URL,
      token shapes (ghp_, github_pat_, sk-ant-, xox?-, AKIA, PEM private keys). Competitor names live only in the block list.
- [A] Store wording in the notes and README.txt: no paid / price / sell / buy / purchase; no switch names (PXA_*, PXQ_*),
      lane names or private paths; SmartScreen + "Run anyway" present; WARN if SHA256SUMS is not mentioned.

### 7. Tests and smoke
- [A with --ctest-log, else M] Windows ctest output says "100% tests passed, 0 tests failed out of N" (tests 100%).
- [M] Smoke on a clean Windows box with an NVIDIA GPU: unzip, rename to engine\, copy ggml-pxqn.dll into engine\bin,
      Control finds and boots the engine, one coherent PXQN chat reply, one PXQ chat reply; without ggml-pxqn.dll a PXQN
      model is refused with one message.
- [M] SmartScreen: a downloaded copy (Mark of the Web set) shows "Windows protected your PC" -> More info -> Run anyway,
      exactly as RELEASE-NOTES-windows.md says. The files are not code-signed .
