# 00 — Start here

## What this is, in three sentences

This is a language-model server for old NVIDIA cards. If you have a Tesla P100, a GTX 1080 Ti or
a Tesla V100 sitting in a box, the usual tools either refuse to run on them or run badly, because
everyone else stopped tuning for that silicon years ago — so I wrote the quantization formats and
the GPU code that those cards are actually good at. You download one file, point it at a model,
and you have a private chat server on hardware that costs less than a night out.

Nothing here talks to the internet once it is running, nothing phones home, and there is no
account to make.

---

## 1. Is your card one of these?

```bash
nvidia-smi -L
```

You should see something like:

```
GPU 0: Tesla P100-PCIE-16GB (UUID: GPU-aad5ef40-...)
GPU 1: Tesla V100-PCIE-16GB (UUID: GPU-9af9aaf0-...)
```

| card | works | notes |
|---|---|---|
| Tesla P100 (16 GB) | yes | the card this project is named for |
| GTX 1080 Ti, GTX 10-series, Tesla P40 | yes | 11 GB on a 1080 Ti, so smaller models |
| Tesla V100 (16 GB / 32 GB) | yes | the fastest of the three |
| Titan Xp, Quadro P-series | yes | same generation as the 1080 Ti |
| RTX 20-series and anything newer | **no** | the released build is compiled for the three
  architectures above only. Your card is not broken — this build simply has no code for it, and
  mainline llama.cpp serves you better anyway. |

**What just happened.** `nvidia-smi` is the tool NVIDIA's driver installs. If that command printed
nothing or said "command not found", you do not have a working NVIDIA driver yet, and nothing in
these guides will work until you do.

---

## 2. What you need installed

Almost nothing. The download brings its own copy of everything except the driver.

```bash
nvidia-smi | head -3
ldd --version | head -1
```

You should see a driver version in the top-right of the `nvidia-smi` banner, and a glibc version:

```
| NVIDIA-SMI 580.142                Driver Version: 580.142        CUDA Version: 13.0     |
ldd (GNU libc) 2.42
```

| thing | what you need | how to check |
|---|---|---|
| Linux, 64-bit Intel/AMD | any modern distro | — |
| NVIDIA **driver** | **570.00 or newer** | the number in the `nvidia-smi` banner |
| glibc | **2.34 or newer** | `ldd --version`. Ubuntu 22.04+, Debian 12+, Rocky/Alma 9+ are all fine |
| `python3` | any version on your PATH | only the launcher and a couple of one-liners use it |
| free disk | ~5 GB for the package, plus your model (10–20 GB) | `df -h .` |

You do **not** need the CUDA toolkit, a compiler, `pip`, conda, or a Python environment. The
package carries its own CUDA runtime. It deliberately does not carry a driver, because the driver
has to match your kernel.

> **If it went wrong**
> - **`nvidia-smi: command not found`** — no driver. Install your distro's NVIDIA driver package
>   first, reboot, and try again.
> - **Driver version below 570** — upgrade the *driver*, not CUDA. Installing a CUDA toolkit will
>   not help and is not needed.
> - **glibc below 2.34** — your distro is too old for the binaries. Either upgrade the distro or
>   use the container path (guide 01), which brings its own userspace.

---

## 3. Tarball or container?

Two ways to get the same binaries. Pick one; you can switch later.

| | **tarball** | **container image** |
|---|---|---|
| what it is | a `.tar.gz` you unpack into a folder | a Docker image you pull |
| you need | just the driver | the driver, Docker, and the NVIDIA Container Toolkit |
| glibc too old? | will not start | works anyway |
| where models live | anywhere you like | a folder you mount |
| upgrading | download the new tarball | `docker pull` the new tag |
| my advice | **start here if you have one box and root on it** | use this if you already run Docker, or your distro is old |

Both paths are written out side by side in guide 01, so you do not have to decide yet. A third
option, if you want both servers (the engine and the vLLM sidecar) running at once with one
command, is [guide 07 — Docker Compose](07-docker-compose.md).

---

## 4. The rest of the guides

Read 01 first. After that, jump to whichever one you need.

| | |
|---|---|
| [01 — Run your first model](01-run-your-first-model.md) | download, start the server, talk to it. Start here. |
| [02 — Pick settings for your cards](02-pick-settings-for-your-cards.md) | one card, two cards, four cards, mixed cards, context length, and how to tell a model fits before you wait for it to load |
| [03 — Going faster without changing the answers](03-going-faster.md) | what speculative decoding is, in plain words, and what "lossless" honestly means |
| [04 — Quantize your own model](04-quantize-your-own-model.md) | from a Hugging Face folder to a PXQ file you can run |
| [05 — Quantize for the vLLM sidecar](05-quantize-for-vllm.md) | the same job for the other server |
| [06 — Run the vLLM sidecar](06-run-the-vllm-sidecar.md) | when you have several people using one box |
| [07 — The whole stack with Docker Compose](07-docker-compose.md) | both servers, one command |
| [08 — Long chats and many conversations](08-long-chats.md) | why a follow-up reply is instant, and the one case that is still slow |
| [09 — Measure your own card honestly](09-measure-your-card.md) | the three traps I fell into myself |
| [10 — When something goes wrong](10-when-something-goes-wrong.md) | the failures people actually hit, and where to ask |
| [11 — Switching models from your app](11-switching-models-from-your-app.md) | letting an app pick the model by name, instead of one model per server |
| [Glossary](GLOSSARY.md) | every unfamiliar word, in two sentences each |

---

## 5. A promise about numbers

I do not put speed numbers in these guides. They go stale the moment I make the engine faster,
and a stale number in a tutorial is worse than none — you end up debugging a perfectly healthy
machine against a figure from three releases ago. The current measured numbers live in the
project README, with the exact command and file they were taken on. Guide 09 shows you how to
measure your own card in a way you can defend.
