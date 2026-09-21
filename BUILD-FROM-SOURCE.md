# Build pxa from source

This is the complete, verified path from a fresh `git clone` to a running `llama-server` on a
Pascal or Volta card. Every command here was run start-to-finish on a clean checkout inside a
stock CUDA container; the traps section lists the exact errors you get when a step is skipped.

If you only want the two-line version, it is in the README. This file is the one to follow when
that does not work.

---

## 1. What you need

| | |
|---|---|
| OS | Linux x86-64 |
| GPU | anything from sm_60 (P100 / GP100) upward. sm_60, sm_61 (10-series), sm_70 (V100) are the cards this fork exists for |
| Driver | an NVIDIA driver new enough for CUDA 12.x (>= 525) |
| Toolchain | CUDA 12.x, CMake >= 3.14, a C++17 compiler |
| Disk | ~1 GB for the checkout plus build with a two-arch list, ~2 GB with the wide list. The CUDA container image itself is another ~9 GB |
| RAM | 16 GB is enough; the CUDA compile is the memory-hungry part, throttle `-j` if you have less |

You do **not** need Python, `curl`, `ccache`, or any Python packages to build the engine.
`LLAMA_CURL` defaults to `OFF` in this fork, so there is no libcurl dependency either.

The easiest way to get a matching toolchain is the official CUDA container image. That is the
path documented below.

---

## 2. Build — GPU visible to the build (recommended)

This is the path to use if you are building on the machine that has the cards. It is the
simplest one and it avoids the CUDA-driver link trap entirely.

```bash
# on the host
git clone https://github.com/poisonxa16/pxa
cd pxa

docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -v "$PWD":/src -w /src \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 bash
```

> **Why the GPU has to be visible to the _build_.** It is what puts the real `libcuda.so.1`
> inside the container. Without it the compile succeeds and every executable fails to link —
> see section 3.
>
> `--gpus all` is the modern equivalent and is what most Docker installs want. It depends on how
> your nvidia-container-toolkit is configured: on a host running the toolkit in legacy mode it
> fails with `nvidia-container-cli: ldcache error`, while the `--runtime=nvidia
> -e NVIDIA_VISIBLE_DEVICES=...` form above works in both modes — which is why that is the one
> written here. Either flag takes a device list, `NVIDIA_VISIBLE_DEVICES=0` or
> `--gpus '"device=0"'`, to build against a single card.

Then, **inside the container**:

```bash
# the stock CUDA image ships nvcc, gcc and make but NOT cmake and NOT git
apt-get update && apt-get install -y --no-install-recommends cmake git

cmake -B build -S . -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;61;70" \
  -DGGML_SCHED_MAX_COPIES=2

cmake --build build \
  --target llama-server llama-cli llama-bench llama-quantize llama-pxq-export \
  -j"$(nproc)"
```

`GGML_SCHED_MAX_COPIES` defaults to `1`. Every published multi-GPU number in this repo was
measured with `2`, which is what lets the scheduler keep a second set of input copies in flight
and overlap one micro-batch with the next across devices. Build with the default and the
multi-GPU prefill figures do not reproduce. It costs a little VRAM per device and nothing on a
single card.

Binaries land in `build/bin/`. They carry an rpath to their own shared libraries, so
`./build/bin/llama-server` runs from the repo root with no `LD_LIBRARY_PATH` set.

**Choosing the arch list.** `CMAKE_CUDA_ARCHITECTURES` is the single biggest lever on build
time — each entry is a full recompile of every CUDA translation unit. Trim it to your cards:

| your card | value |
|---|---|
| P100 / GP100 | `"60"` |
| GTX 10-series (1080 Ti etc.) | `"61"` |
| V100 | `"70"` |
| this release's target list | `"60;61;70"` |
| the wide list (adds 3090- and 4090-class) | `"60;61;70;86;89"` |

Expect roughly 25-45 min for `"60;61;70"` on a many-core box, and about twice that for the wide
list. `nvcc` warns that pre-sm_75 offline compilation is deprecated — that is expected and
harmless on CUDA 12.x.

---

## 3. Build — no GPU visible to the build (CI, or a container started without GPU access)

If the build machine has no NVIDIA device node, the compile succeeds and then **the link of every
executable fails**:

```
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `cuMemCreate'
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `cuMemAddressReserve'
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `cuMemSetAccess'
... (cuMemUnmap, cuMemMap, cuMemRelease, cuMemAddressFree,
     cuMemGetAllocationGranularity, cuDeviceGet, cuDeviceGetAttribute, cuGetErrorString)
collect2: error: ld returned 1 exit status
```

`libggml.so` calls into the CUDA **driver** API (the VMM allocator), so it records
`NEEDED libcuda.so.1`. The driver library ships with the driver, not with the toolkit — with no
GPU present the only thing on the box is the toolkit's stub at
`/usr/local/cuda/lib64/stubs/libcuda.so`, and that file is not enough on its own: the stub is
named `libcuda.so`, while what the linker is asked to resolve is the **SONAME** `libcuda.so.1`.
Putting the stubs directory on `LIBRARY_PATH` therefore does not fix it. You need both halves:

```bash
# 1. give the stub the SONAME the linker is actually looking for
ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# 2. point the link at it
cmake -B build -S . -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;61;70" \
  -DGGML_SCHED_MAX_COPIES=2 \
  -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs"

cmake --build build \
  --target llama-server llama-cli llama-bench llama-quantize llama-pxq-export \
  -j"$(nproc)"
```

`-Wl,-rpath-link` is a **link-time** search path only. It is not baked into the binary, so the
resulting executables load the real driver normally on a machine that has one — a binary built
this way was verified to load a PXQ4 model with `offloaded 33/33 layers to GPU`.

> ### ⚠ The other half of the stub trap — at runtime
>
> If you put `/usr/local/cuda/lib64/stubs` on `LD_LIBRARY_PATH` (or created that
> `libcuda.so.1` symlink somewhere the loader searches), **take it back off before you run
> anything**. The stub resolves every driver call to a failure, `cudaGetDeviceCount` reports
> zero devices, and the engine falls back to CPU **with no error message at all**. What you see
> is:
>
> ```
> llm_load_tensors: offloaded 0/33 layers to GPU
> llm_load_tensors:        CPU buffer size =  5375.25 MiB
> ```
>
> instead of `offloaded 33/33`. Output stays correct, so nothing looks broken. It is just
> ~50x slower: the same 9B PXQ4 on the same box measures **49.3 t/s decode on the GPU and
> 0.93 t/s** once the stub shadows the driver. It looks exactly like the GPU tier being broken,
> and it costs people an afternoon. **Link with the stub, run without it.**
>
> `offloaded N/N` is the one line to check every time you start a binary.

There is also `-DGGML_CUDA_NO_VMM=ON`, which the build system documents as removing the direct
link against the driver library. It gives up the VMM pool allocator in exchange, so it is not the
recommended configuration for a shipping build and it is not the path verified here — the stub
recipe above is.

---

## 3b. The tests

Add `-DLLAMA_BUILD_TESTS=ON` to either configure line above and build the targets you want:

```bash
cmake --build build --target test-pxq-cpu-dot test-topk-raw-shadow test-kv-seq-shadow \
  test-kv-mask-parity test-narrow-kernel-parity -j"$(nproc)"
```

| test | what it checks | needs |
|---|---|---|
| `test-pxq-cpu-dot` | the PXQ x Q8 integer dot on CPU agrees with the float reference | x86 with AVX2, no model |
| `test-topk-raw-shadow` | the raw top-k fast path picks the same tokens as the reference sampler | a small GGUF |
| `test-kv-seq-shadow` | the sequence-major KV bookkeeping matches the reference over 2M steps | nothing |
| `test-kv-mask-parity` | dumps per-step state hashes of the padded KQ mask path for diffing | a small GGUF |
| `test-narrow-kernel-parity` | get_rows, cpy, concat, rms_norm and l2_norm narrow kernels vs the generic ones | a CUDA device |

`tests/test-pxq-export.sh` exercises `llama-pxq-export` end to end (CUDA vs CPU export, requantize
gate, generation agreement) and needs a small GGUF plus a CUDA device.

## 4. Verify the build actually works

A build that links is not a build that runs. Run all four of these against a real PXQ4 GGUF.

PXQ has **no CPU codec** — a PXQ model must be fully GPU-resident. Always pass `-ngl 99`, and
never `--n-cpu-moe`. See `docs/KNOWN-ISSUES.md`.

### 4.1 It loads onto the GPU

```bash
./build/bin/llama-cli -m your-model-PXQ4.gguf -ngl 99 -c 2048 -n 8 --temp 0 -p "17*23="
```

Check for, in this order:

```
llm_load_tensors: offloaded 33/33 layers to GPU        <- N/N, never 0/N
PXA_PXQ6 fused kernels: ON (table self-check PASS ...)
17*23=391
```

`offloaded 0/N` means a CUDA stub is shadowing the driver — see the warning above.
A per-token eval time in the tens of milliseconds is GPU; hundreds to thousands is CPU.

### 4.2 Raw, non-chat-templated completions

Start the server:

```bash
./build/bin/llama-server -m your-model-PXQ4.gguf \
  -ngl 99 -c 4096 -b 512 -ub 512 -fa on --host 127.0.0.1 --port 8099
```

Then hit `/v1/completions` — **not** `/v1/chat/completions`. (The stock CUDA image has no `curl`;
either `apt-get install -y curl` inside it, or serve on `--host 0.0.0.0` and query from the host.) Very short raw prompts are the
useful test: a chat template pads the prompt out past the captured graph sizes and can hide
short-prefill corruption that a one-token prompt exposes immediately.

```bash
# 1-token prompt
curl -s localhost:8099/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"The","n_predict":32,"temperature":0}'

# 5-token prompt
curl -s localhost:8099/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","n_predict":24,"temperature":0}'

# arithmetic
curl -s localhost:8099/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"17*23=","n_predict":8,"temperature":0}'
```

All three must return fluent, on-topic text — the second should say Paris, the third `391`.
Repeated tokens, punctuation soup, or an empty string on the 1-token prompt is a real bug; report
it rather than working around it with a longer prompt.

### 4.3 Throughput

```bash
./build/bin/llama-bench -m your-model-PXQ4.gguf -ngl 99 -p 512 -n 128 -r 2
```

You should get a `pp512` and a `tg128` row. For scale, a 9B PXQ4 on a single Tesla P100
(sm_60) built exactly as in section 2 measures pp512 ≈ 663 t/s and tg128 ≈ 50 t/s. A tg128 in
the low single digits means you are on CPU.

`llama-bench` labels the file by its base ftype, so a PXQ4 model can print as `MXFP4 - 4.25 bpw`
in the model column. That is a cosmetic naming artifact, not a sign the wrong codec loaded — the
`type pxq4: N tensors` line at load time is the authoritative one.

### 4.4 Environment

Nothing is required, and since 2026-09-03 there is nothing worth setting either: the tune runs
at config level ENHANCE by default, the auto-detect picks the measured per-card levers and
`-b`/`-ub`, and the PXQ codecs load with no environment at all. `PXA_ENHANCE=0` is the rollback.
Everything else in `docs/LEVERS.md` is a lab knob — see the README's Run section.

### 4.5 What the startup banner means

Every run prints a `pxa:` banner before the first token. `engine | codec=off` means the
file holds no PXQ tensors and the PXQ kernels are idle; `engine | codec=PXQ4 (140 tensors;
pxq2 0, pxq3 0, pxq4 140, pxq6 0)` names the tier holding the most tensors and then the full
per-tier count, so a mixed file cannot hide behind one name. One `dev N <name> cc X.Y -> path:
...` line follows per CUDA device, naming the arch path that card will actually dispatch on —
`sm_60 fp16-hfma2`, `sm_61 dp4a` or `sm_70 fp16-tensor-core-prefill` — which is the fastest way
to confirm a card was detected as the arch you expected. A `lab knobs set: ...` line appears
only when the environment holds a `PXA_`/`PXQ` variable outside `PXA_ENHANCE`, `PXA_MODE` and
`PXA_REFERENCE`: those are lab levers measured on one config and usually a slowdown elsewhere,
so the line names them and changes nothing. Under `llama-cli --verbose`, and in the server's
startup log, a `file:` line adds what the GGUF says about itself — codec, whole-file bits per
weight, and the backbone revision and codec tier the quantizer recorded, with `n/a` for
anything the file does not carry.

---

## 5. Traps, with the exact error

| Symptom | Cause | Fix |
|---|---|---|
| `bash: cmake: command not found`<br>`bash: git: command not found` | the stock `nvidia/cuda:*-devel` image has nvcc, gcc and make but no cmake and no git | `apt-get update && apt-get install -y --no-install-recommends cmake git` |
| `make: *** No targets specified and no makefile found.  Stop.` | there is no Makefile in this tree; CMake is the only build system | use the CMake commands above |
| `undefined reference to 'cuMemCreate'` and friends at link | no GPU visible to the build, so no `libcuda.so.1` | section 3 |
| `cannot find -lcuda` | stubs directory not on the link path | section 3 |
| `offloaded 0/N layers to GPU`, correct but ~50x slow | a CUDA stub is shadowing the real driver at runtime | remove `/usr/local/cuda/lib64/stubs` from `LD_LIBRARY_PATH` |
| `Warning: ccache not found` | informational only, the build proceeds | install ccache, or `-DGGML_CCACHE=OFF` to silence |
| aborts at the first expert op with `-ngl` < 99 or `--n-cpu-moe` | PXQ has no CPU codec | keep the model fully GPU-resident |
| `error: unknown argument: -no-cnv` | mainline llama.cpp flags are not all present in this fork | drop it; `llama-cli` is non-interactive when `-p` is given |
| `Illegal instruction` when the binary is moved to another machine | `GGML_NATIVE` defaults ON, which compiles `-march=native` | build on the target machine, or configure with `-DGGML_NATIVE=OFF` |

---

## 6. Python (conversion tooling only — not needed to build)

The engine build needs no Python. The `convert_*.py` scripts do, and their pins
(`numpy~=1.26.4`, `torch~=2.2.1`) resolve on **Python 3.12** and fail to build on 3.13:

```
error: metadata-generation-failed
╰─> numpy
```

Use a 3.12 virtualenv for conversion work:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`llama-quantize` — the tool you need to make PXQ artifacts from an existing f16/bf16 GGUF — is a
C++ binary from section 2 and needs none of this.
