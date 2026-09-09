# Quickstart

From nothing to a running server on NVIDIA hardware. No prior knowledge of this project
assumed. Everything here is CUDA; if a step fails, [`BUILD-FROM-SOURCE.md`](../BUILD-FROM-SOURCE.md)
has the same path in full, with the exact error text for every trap.

Budget: ~25-45 minutes, nearly all of it CUDA compilation.

---

## 0. What you need

| | |
|---|---|
| OS | Linux x86-64 |
| GPU | anything from sm_60 upward. The cards this fork is tuned for are Tesla P100 (sm_60), GTX 1080 Ti / P40 (sm_61) and Tesla V100 (sm_70) |
| Driver | new enough for CUDA 12.x — r525 or later |
| Toolchain | CUDA 12.x, CMake >= 3.14, a C++17 compiler |
| Disk | ~1 GB for source + a two-architecture build; the CUDA container image is another ~9 GB on top |
| RAM | 16 GB. The CUDA compile is the hungry part — lower `-j` if you have less |

You do **not** need Python, `curl`, or any Python packages to build or run the engine.
`LLAMA_CURL` is `OFF` by default in this fork, so there is no libcurl dependency — and no
`-hf` model downloader either. You fetch model files yourself (step 2).

There is **no Makefile** in this tree. CMake is the only build system.

---

## 1. Get the source

```bash
git clone https://github.com/poisonxa16/pxa
cd pxa
```

---

## 2. Get a model

The engine reads GGUF files. Any GGUF that llama.cpp can load will work; the PXQ tiers this
fork adds are what it is actually for. PXA's own weights are at
<https://huggingface.co/poisonxa> — e.g. `PXA-bench-files-GGUF`, which holds the Fusion2 tiers
(they were first published under `PXA-Fusion2-35B-GGUF`, a repository since renamed to
`PXA-Fusion4-35B-GGUF` that keeps the same bytes at revision
`2d7d9e7e5f445b1ea38a5686b8cf9d6bc9aa29df`). Download the `.gguf` with
your browser, `wget`, or `huggingface-cli`, and note where it landed. The rest of this page
calls it `your-model.gguf`.

> **The one hard rule for PXQ models.** PXQ has **no CPU codec**. A PXQ model must be fully
> GPU-resident: always pass `-ngl 99`, and never `--n-cpu-moe` or a partial `-ngl`. The engine
> aborts at the first expert op otherwise. This does not apply to ordinary Q4_K_M / MXFP4 /
> Q8_0 files, which run partially offloaded as usual.

---

## 3. Build, with the GPU visible to the build

This is the path to use on the machine that has the cards. It is the simplest one, and it
avoids the driver-stub trap in section 4 entirely.

The easiest matching toolchain is the official CUDA container image:

```bash
docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -v "$PWD":/src -w /src \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 bash
```

> `--gpus all` is the modern equivalent and works on most Docker installs. The
> `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=...` form above works in both legacy and
> modern nvidia-container-toolkit configurations, which is why it is the one written here.
> Either flag accepts a device list (`NVIDIA_VISIBLE_DEVICES=0`, `--gpus '"device=0"'`).

Then, **inside the container**:

```bash
# the stock CUDA image ships nvcc, gcc and make — but not cmake and not git
apt-get update && apt-get install -y --no-install-recommends cmake git

cmake -B build -S . -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;70"

cmake --build build \
  --target llama-cli llama-server llama-bench llama-quantize \
  -j"$(nproc)"
```

Binaries land in `build/bin/`. They carry an rpath to their own shared libraries, so
`./build/bin/llama-server` runs from the repo root with no `LD_LIBRARY_PATH` set.

> **Mount the source read-write.** Even with an out-of-source `-B build`, the build
> generates `common/build-info.cpp` *inside the source tree*. Mounting the checkout `:ro`
> fails early with `CMake Error: : System Error: Read-only file system`. The generated file
> is gitignored, so a writable mount leaves nothing behind that `git status` will show.

### Choosing the architecture list

`CMAKE_CUDA_ARCHITECTURES` is the single biggest lever on build time: every entry is a full
recompile of every CUDA translation unit. Trim it to the cards you actually have.

| your card | value |
|---|---|
| Tesla P100 / GP100 | `"60"` |
| GTX 1080 Ti, P40 (10-series) | `"61"` |
| Tesla V100 | `"70"` |
| P100 + V100, this release's target pair | `"60;70"` |
| wide list (adds Ampere and Ada) | `"60;61;70;86;89"` |

`nvcc` warns that offline compilation for architectures before sm_75 is deprecated. That is
expected on CUDA 12.x and harmless — sm_60 and sm_70 still compile.

---

## 4. Build with **no** GPU visible (CI, or a container started without GPU access)

Read this section if you are building in CI, in a container started without `--gpus` /
`--runtime=nvidia`, or on a build host with no NVIDIA device node.

**The failure mode is the reason this section exists: every CUDA source file compiles
successfully, and then the *final link of every executable* fails.**

```
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `cuMemCreate'
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `cuMemAddressReserve'
/usr/bin/ld: ../../ggml/src/libggml.so: undefined reference to `cuMemSetAccess'
... (cuMemUnmap, cuMemMap, cuMemRelease, cuMemAddressFree,
     cuMemGetAllocationGranularity, cuDeviceGet, cuDeviceGetAttribute, cuGetErrorString)
collect2: error: ld returned 1 exit status
```

Because it only bites at the last step, a build can run for forty minutes and *then* fail.

**Why.** `libggml.so` calls the CUDA **driver** API for its VMM allocator, so it records
`NEEDED libcuda.so.1`. The driver library ships with the *driver*, not with the toolkit. With
no GPU present, the only thing on the box is the toolkit's **stub**:
`/usr/local/cuda/lib64/stubs/libcuda.so`. Point the link at it:

```bash
STUBS=/usr/local/cuda/lib64/stubs

cmake -B build -S . -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;70" \
  -DCUDA_CUDA_LIBRARY="$STUBS/libcuda.so" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$STUBS -lcuda"

cmake --build build \
  --target llama-cli llama-server llama-bench llama-quantize \
  -j"$(nproc)"
```

Both flags are here on purpose, and the linker one is the one that does the work.
`-DCMAKE_EXE_LINKER_FLAGS="-L$STUBS -lcuda"` puts `-lcuda` and the stubs directory on the
final **executable** link line, which is the half that fixes the error above.
`-DCUDA_CUDA_LIBRARY` names the driver library for CMake configurations that resolve it by
that variable name.

Do not assume CMake handles this for you. On CMake 3.28.3 with CUDA 12.8,
`find_package(CUDAToolkit)` *does* resolve `CUDA_cuda_driver_LIBRARY` to the toolkit's own
stub by itself — and the build still fails, because that only covers the `libggml.so` link.
`libggml.so` then records `NEEDED libcuda.so.1`, and the executables that link against it
have nothing to satisfy it with. Verified both ways on this tree, no GPU, `sm_70`: without
the flags the CUDA sources all compile and the executable link fails on `cuMemCreate` and
friends; adding the two flags above relinks the same object files and produces `llama-cli`,
`llama-server`, `llama-bench` and `llama-quantize` cleanly.

If the error you get names the **SONAME** rather than the symbols —
`cannot find -lcuda`, or a complaint about `libcuda.so.1` — the stub file is named
`libcuda.so` while the linker is asked for `libcuda.so.1`. Give it the name it wants and
re-run the configure above:

```bash
ln -sf "$STUBS/libcuda.so" "$STUBS/libcuda.so.1"
```

`BUILD-FROM-SOURCE.md` §3 documents the `-Wl,-rpath-link` variant of the same fix. Either
way the stub is a **link-time** convenience only, and neither recipe writes the stubs
directory into the binary's runpath — so binaries built this way load the real driver
normally on a machine that has one.

You can check that it stayed out:

```bash
readelf -d build/bin/llama-cli | grep -E 'RUNPATH|RPATH'
```

Nothing in that line should mention `stubs`. As a side effect, a binary built this way will
*refuse to start* on the driverless build machine —
`error while loading shared libraries: libcuda.so.1` — which is the correct failure, and a
much better one than the silent CPU fallback described next.

> ### The other half of the trap — at runtime
>
> **Never leave the stubs directory on `LD_LIBRARY_PATH` when you run.** The stub resolves
> every driver call to a failure, `cudaGetDeviceCount` reports zero devices, and the engine
> silently falls back to CPU with no error printed. What you see is:
>
> ```
> llm_load_tensors: offloaded 0/33 layers to GPU
> llm_load_tensors:        CPU buffer size =  5375.25 MiB
> ```
>
> instead of `offloaded 33/33`. The output is still *correct*, so nothing looks broken — it
> is simply ~50x slower. **Link with the stub, run without it.**

---

## 5. First run — `llama-cli`

One prompt, one answer, no server. This is the fastest way to prove the build works.

```bash
./build/bin/llama-cli -m your-model.gguf -ngl 99 -c 2048 -n 8 --temp 0 -p "17*23="
```

`llama-cli` is non-interactive when `-p` is given; add `-cnv` if you want a chat loop.

Check the output in this order:

```
llm_load_tensors: offloaded 33/33 layers to GPU        <- N/N, never 0/N
PXA_PXQ6 fused kernels: ON (table self-check PASS; PXA_PXQ6=0 disables)
17*23=391
```

1. **`offloaded N/N`** — the single most useful line in the whole log. `0/N` means a CUDA
   stub is shadowing the real driver (section 4), or you forgot `-ngl`.
2. **`PXA_PXQ* fused kernels: ON`** — printed when a PXQ codec loads and its table
   self-check passes. Only appears for PXQ models.
3. **`- type pxq4: N tensors`** at load time is the authoritative statement of which codec
   the file actually uses.
4. A per-token eval time in the **tens of milliseconds** is GPU. Hundreds to thousands is CPU.

Throughput sanity check:

```bash
./build/bin/llama-bench -m your-model.gguf -ngl 99 -p 512 -n 128 -r 2
```

You get a `pp512` (prefill) and a `tg128` (decode) row. For scale: a 9B PXQ4 on a single
Tesla P100 built exactly as in section 3 measures pp512 ~663 t/s and tg128 ~50 t/s. A
`tg128` in the low single digits means you are running on CPU.

> `llama-bench` labels a model by its base ftype, so a PXQ4 file can print as
> `MXFP4 - 4.25 bpw` in the model column. Cosmetic only — the `type pxq4:` line at load
> time is the one that counts.

---

## 6. First run — `llama-server`

```bash
./build/bin/llama-server -m your-model.gguf \
  -ngl 99 -c 4096 -b 512 -ub 512 -fa on \
  --host 127.0.0.1 --port 8080
```

Wait for the server to report that it is listening, then, from another shell:

```bash
curl -s localhost:8080/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","n_predict":24,"temperature":0}'
```

It should say Paris. Two more worth running, because they catch a class of bug a long prompt
hides — a chat template pads the prompt past the captured graph sizes, so very short **raw**
prompts are the sharper test:

```bash
curl -s localhost:8080/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"The","n_predict":32,"temperature":0}'

curl -s localhost:8080/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"17*23=","n_predict":8,"temperature":0}'
```

All three must return fluent, on-topic text. Repeated tokens, punctuation soup, or an empty
string on the one-token prompt is a real bug — please report it rather than papering over it
with a longer prompt.

For chat clients, add `--jinja` (applies the model's own chat template) and use
`/v1/chat/completions`. To reach the server from another machine, serve on
`--host 0.0.0.0`.

> The stock CUDA image has no `curl`. Either `apt-get install -y curl` inside it, or serve on
> `--host 0.0.0.0` and run the `curl` from the host.

### Configuration, in one line

```bash
./build/bin/llama-server -m your-model.gguf -ngl 99 -c 8192 -fa on \
  --host 127.0.0.1 --port 8080
```

There is no line. The tune is **on by default**: the engine selects the measured-good kernel
levers for each card in the box — a mixed-card machine gets a per-GPU decision — fills `-b`/`-ub`
from the measured cell for that card set when you have not passed them, and prints a ledger of
what it chose and why. `PXA_ENHANCE=0` rolls back to the pre-2026-09-03 shipped set and
`PXA_REFERENCE=1` gives the bit-exact all-levers-off baseline. That is the whole recommended
configuration. Everything else in [`LEVERS.md`](lab/LEVERS.md) is a lab knob, recorded with its
measurement; you do not need any of it to run.

Or let the launcher choose engine, image and parameters for you:

```bash
python3 tools/pxa-launch.py --model your-model.gguf --cards 0,1 --explain
```

`--explain` decides and prints, running nothing. Drop it to actually launch.

---

## 7. Where to go next

| | |
|---|---|
| [`BUILD-FROM-SOURCE.md`](../BUILD-FROM-SOURCE.md) | the complete build path, and the exact error for every trap |
| [`COOKBOOK.md`](COOKBOOK.md) | copy-paste command lines per card, with the measured numbers they produce |
| [`LEVERS.md`](lab/LEVERS.md) | every shipping `PXA_*` lever, its default, and the measurement behind it |
| [`KNOWN-ISSUES.md`](KNOWN-ISSUES.md) | standing traps and their workarounds |
| [`PXQU-CONVERT.md`](PXQU-CONVERT.md) | quantize your own model into a mixed-tier PXQU map |
| [`VLLM.md`](VLLM.md) | the PXQ4 vLLM backend: tier support, conversion, serving, tuning and tool calling |
| [`PXA-SM60-SERVING.md`](PXA-SM60-SERVING.md) · [`PXA-SM70-SERVING.md`](PXA-SM70-SERVING.md) | the vLLM PXQ4 serving recipes, measured, with the reasoning for every value |
| [`parameters.md`](parameters.md) | the full CLI surface |
| [`../bench/README.md`](../bench/README.md) | the reproduction pack for every published number |
| [`../tools/vllm-pxq4/README.md`](../tools/vllm-pxq4/README.md) | the vLLM PXQ4 backend: design, gates and current status |
| [`../README.md`](../README.md) | what PXQ is, what is ours, what we do not claim |

Prebuilt Linux x86-64 CUDA 12 binaries are attached to each
[release](https://github.com/poisonxa16/pxa/releases), and container images are
described in [`docker.md`](docker.md). `./install.sh` reads your card with `nvidia-smi` and
names the supported path rather than guessing.
