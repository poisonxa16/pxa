# pxa server image — Pascal/Volta (sm 60;61;70): P100, GTX 1080 Ti/P40, V100.
#
# Multi-stage build. The build stage matches the CUDA 12.8 devel base this engine is
# developed and gated against; the runtime stage ships only the binaries, the shared
# libraries they need, and the CUDA 12.8 runtime libs (cudart/cublas/cublasLt) — no
# compiler, no headers, no build tooling.
#
# The image carries one engine: this one, as /usr/local/bin/llama-server (run by the
# ENTRYPOINT, pxa-entrypoint). To compare against another engine on the same card, driver,
# CUDA runtime and weight file, supply that binary yourself — see docker/README.md,
# "Comparing against another engine", and bench/fair/run.sh (UPSTREAM_BIN).
#
# Build:
#   docker build -f docker/Dockerfile \
#     --build-arg CUDA_ARCHS="60;61;70" \
#     -t ghcr.io/poisonxa16/pxa:latest .
#
# Models are NOT baked into this image — mount a model directory at /models (see
# docker/README.md).

ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.8.1
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# ---------------------------------------------------------------------------
# Build stage
# ---------------------------------------------------------------------------
FROM ${BASE_CUDA_DEV_CONTAINER} AS build

# sm 60 = P100, sm 61 = GTX 1080 Ti / P40, sm 70 = V100. Override for a narrower image
# (e.g. "70" only) or to add newer archs; the shipped defaults gate every card this
# engine's benchmark table covers.
ARG CUDA_ARCHS="60;61;70"
# Cap parallel compile jobs on shared build hosts; empty/0 means use nproc.
ARG BUILD_JOBS="0"

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        build-essential cmake git ca-certificates python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Same core flags as the gate build (build-spd/CMakeCache.txt): shared libs on,
# curl off (the server does not need outbound HTTP fetch for this image), Release.
# The explicit linker flags point ggml's CUDA driver symbols (cuMemCreate etc, used
# by the VMM pool allocator) at the devel image's driver stub -- needed for a clean
# container build where no real libcuda.so.1 is present until run time.
#
# THE CPU FLAGS ARE PINNED, not left to ggml's default: ggml's default is -march=native of the
# machine that compiles, and an image that runs on other people's CPUs must not carry its build
# machine's instruction set (a build host with AVX-512 makes an image that stops with "Illegal
# instruction" on a Haswell or a Zen 2 box). Fast = AVX, AVX2, FMA, F16C (any CPU since 2013).
# llama-imatrix is the open activation-dump tool PXA Control's Encode tab runs.
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_CURL=OFF \
        -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    && cmake --build build --target llama-server llama-cli llama-bench llama-quantize llama-perplexity llama-imatrix \
        -j "$([ "${BUILD_JOBS}" != "0" ] && [ -n "${BUILD_JOBS}" ] && echo "${BUILD_JOBS}" || nproc)"

# lib-compat: the same libggml.so for plain x86-64 (no AVX / AVX2 / FMA / F16C), the one the entrypoint puts first on the
# library path on a CPU without them (tools/pxa-cpu-check.sh reads /proc/cpuinfo). The fast library stops with "Illegal
# instruction" while it loads on such a CPU. Same recipe, -DPXA_X86_COMPAT=ON, target ggml only (the GPU code is the same).
RUN cmake -S . -B build-compat \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_CURL=OFF \
        -DPXA_X86_COMPAT=ON \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    && cmake --build build-compat --target ggml \
        -j "$([ "${BUILD_JOBS}" != "0" ] && [ -n "${BUILD_JOBS}" ] && echo "${BUILD_JOBS}" || nproc)"

# The Encode tab's python requirements (the same renderer the release tarball uses): the converter's own list minus "gguf".
RUN sh scripts/make-convert-requirements.sh requirements > /src/requirements-convert.txt

# The closed PXQN library, libggml-pxqn.so: the PXQN kernels AND the PXA Overdrive levers (the Flash-Next preset turns them
# on). libggml loads it from its own directory, so the image MUST carry it next to libggml.so or every PXQN file is refused
# and the Overdrive hooks are no-ops. A private-tree build makes it (build/ggml/src, a dependency of libllama); a build from
# the public source takes the prebuilt library the release ships, dropped into the context as docker/prebuilt/libggml-pxqn.so.
# Neither = the open fallback image, allowed only when PXQN_REQUIRED=0 (release builds pass PXQN_REQUIRED=1).
ARG PXQN_REQUIRED=0
RUN mkdir -p /src/pxqn-dist && \
    if [ -f build/ggml/src/libggml-pxqn.so ]; then cp -a build/ggml/src/libggml-pxqn.so* /src/pxqn-dist/; \
    elif [ -f docker/prebuilt/libggml-pxqn.so ]; then cp docker/prebuilt/libggml-pxqn.so /src/pxqn-dist/; fi; \
    if [ -f /src/pxqn-dist/libggml-pxqn.so ]; then echo "image ships libggml-pxqn.so (closed PXQN library)"; \
    elif [ "${PXQN_REQUIRED}" = 1 ] || grep -q 'PXA_PXQN_CLOSED_SRC:BOOL=ON' build/CMakeCache.txt; then \
        echo "libggml-pxqn.so required but not built / not supplied (docker/prebuilt/libggml-pxqn.so)" >&2; exit 1; \
    else echo "WARNING: no libggml-pxqn.so -- open fallback image (no PXQN, Overdrive hooks off)"; fi

# The lever catalog the image ships is the one the engine embeds: build/pxa-catalog, the PUBLIC table (owner 2026-10-08:
# no closed rows in what ships). A private-tree build checks the catalog and the shipped binaries for closed lever names.
RUN cp build/pxa-catalog/pxa-lever-catalog-build.inc /src/pxa-lever-catalog.inc && \
    if [ -f scripts/pxa-closed-levers.txt ]; then \
        python3 scripts/pxa-lever-catalog.py --assert-clean build/pxa-catalog/pxqn-closed-levers.inc /src/pxa-lever-catalog.inc \
            build/bin/llama-server build/bin/llama-cli build/src/libllama.so build/ggml/src/libggml.so build/examples/mtmd/libmtmd.so; \
    fi

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM ${BASE_CUDA_RUN_CONTAINER} AS runtime

# Provenance for the OCI labels below. TAG/GIT_SHA are passed by
# .github/workflows/ghcr.yml (github.ref_name / github.sha) on a tagged release
# build; a manual `docker build` with neither set gets "dev"/"unknown" instead of
# a wrong-but-plausible-looking value.
ARG TAG=dev
ARG GIT_SHA=unknown

LABEL org.opencontainers.image.source="https://github.com/poisonxa16/pxa" \
      org.opencontainers.image.title="PXA" \
      org.opencontainers.image.description="Pascal/Volta 4-bit codec + kernel pack for llama.cpp" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${TAG}" \
      org.opencontainers.image.revision="${GIT_SHA}"

# python3 is for pxa-launch (standard library only, no pip): the entrypoint runs it when the image is
# started with no arguments, and for `docker run IMAGE doctor`.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends curl ca-certificates libgomp1 python3 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r pxq && useradd -r -g pxq -d /work -m pxq

WORKDIR /work

COPY --from=build /src/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /src/build/bin/llama-cli    /usr/local/bin/llama-cli
COPY --from=build /src/build/bin/llama-bench  /usr/local/bin/llama-bench
COPY --from=build /src/build/bin/llama-quantize   /usr/local/bin/llama-quantize
COPY --from=build /src/build/bin/llama-perplexity /usr/local/bin/llama-perplexity
# the Encode tab's activation dump (the open tool; the PXQN encoder itself is closed and comes from "Get the encoder")
COPY --from=build /src/build/bin/llama-imatrix    /usr/local/bin/llama-imatrix
COPY tools/pxa-launch.py     /usr/local/bin/pxa-launch
COPY tools/pxa-entrypoint.sh /usr/local/bin/pxa-entrypoint
# the CPU preflight the entrypoint sources (lib/ or lib-compat/); pxa-launch carries the Python twin
COPY tools/pxa-cpu-check.sh  /usr/local/bin/pxa-cpu-check.sh
# PXA Control (`docker run -p 7777:7777 IMAGE gui`): the launcher's web front end, stdlib only.
COPY tools/pxa_control.py tools/pxa_expert_map.py tools/pxa-bench.py tools/pxa_explain_footer.py tools/pxa_thinking.py tools/pxa_thinking_profiles.json tools/pxa_telemetry.py tools/pxa_mqtt.py /usr/local/bin/
# the Encode tab: the wizard backend, the one adapter that knows the quantizer command line, the package installer, the arithmetic, the measured cells
COPY tools/pxa_encode.py tools/pxa_encode_adapter.py tools/pxa_encode_pkg.py tools/pxa_encode_plan.py tools/pxa_encode_cells.json /usr/local/bin/
COPY tools/pxa_control_ui /usr/local/bin/pxa_control_ui
# the Profiles tab's backend (per-card power, clocks, auto-starts; stdlib only)
COPY tools/pxa_ctl /usr/local/bin/pxa_ctl
# the Chat tab's in-house agent (stdlib package; Control runs without it, minus the Assistant view)
COPY tools/pxa_chat /usr/local/bin/pxa_chat
COPY --from=build /src/pxa-lever-catalog.inc /usr/local/share/pxa/pxa-lever-catalog.inc
# The PXA names next to the llama-* ones: one binary behind both.
RUN set -e; for t in server cli bench quantize perplexity; do \
        ln -sf "llama-$t" "/usr/local/bin/pxa-$t"; \
    done; ln -sf pxa-launch /usr/local/bin/pxa; chmod 0755 /usr/local/bin/pxa-launch /usr/local/bin/pxa-entrypoint /usr/local/bin/pxa-cpu-check.sh
COPY --from=build /src/build/src/libllama.so           /usr/local/lib/
COPY --from=build /src/build/ggml/src/libggml.so       /usr/local/lib/
COPY --from=build /src/build/examples/mtmd/libmtmd.so  /usr/local/lib/
RUN ldconfig
# NOT in ldconfig's path on purpose: only the entrypoint and pxa-launch put it on LD_LIBRARY_PATH, and only on a CPU without AVX2.
COPY --from=build /src/build-compat/ggml/src/libggml.so /usr/local/lib-compat/libggml.so
# the closed PXQN library next to BOTH libggml.so copies (it is plain x86-64 with a run-time AVX2 choice: one file serves both)
COPY --from=build /src/pxqn-dist/ /usr/local/lib/
RUN if [ -f /usr/local/lib/libggml-pxqn.so ]; then ln -sf /usr/local/lib/libggml-pxqn.so /usr/local/lib-compat/libggml-pxqn.so; fi

# The Encode tab's converter (Hugging Face model -> GGUF) and the gguf python package of THIS engine commit it imports first
# (tools/pxa_encode.py find_convert reads PXA_CONVERT_SCRIPT). Its python packages (torch, transformers, ...) are NOT in the
# image: `pip install -r /usr/local/share/pxa/convert/requirements-convert.txt` into a virtual environment and point
# PXA_CONVERT_PYTHON at it; the Encode tab's tool check says which package is missing.
COPY convert_hf_to_gguf.py /usr/local/share/pxa/convert/convert_hf_to_gguf.py
COPY gguf-py/gguf          /usr/local/share/pxa/convert/gguf-py/gguf
COPY --from=build /src/requirements-convert.txt /usr/local/share/pxa/convert/requirements-convert.txt

# One file, both documented names: the `docker run` lines and the release docs
# say /usr/local/bin/llama-server, while the comparison harness (bench/fair) looks in
# /opt/pxa/bin. A symlink, so there is exactly one binary and no chance of the two
# paths drifting to different builds. (The directory used to be created by the
# comparison-binary COPY; it is made explicitly now.)
RUN mkdir -p /opt/pxa/bin && ln -sf /usr/local/bin/llama-server /opt/pxa/bin/llama-server

ENV PXA_CONVERT_SCRIPT=/usr/local/share/pxa/convert/convert_hf_to_gguf.py

# Model directory mount point (empty in the image — models are never baked in).
RUN mkdir -p /models /work/.cache/pxa \
    && chown pxq:pxq /models /work /work/.cache /work/.cache/pxa

# NOTE ON LLAMA_ARG_* ENV VARS: this engine applies them *after* parsing the CLI
# args (gpt_params_parse_from_env runs after gpt_params_parse), so a LLAMA_ARG_*
# env var always wins over the equivalent CLI flag, not the other way round. That
# is why the only one baked in here is LLAMA_ARG_HOST -- baking in defaults for
# -ngl/-c/-b/-ub/--port would silently ignore the flags docker/README.md tells
# you to pass on the command line. Configure those either as CLI flags (as the
# README examples do) or as -e LLAMA_ARG_* env vars, not a mix of both for the
# same setting.
ENV LLAMA_ARG_HOST=0.0.0.0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # PXA levers — these are the shipped defaults of the unified build of 2026-09-03 (see
    # docker/README.md for what each does); left unset here so the binary's own
    # compiled-in defaults apply. Uncomment / pass -e to override:
    # PXA_PIPELINE_PP=1 \
    # PXA_CKPT_PROMPT_POLICY=1 \
    # PXA_CKPT_LEAN_SYNC=1 \
    # PXA_DN_SHARE_INPUTS=-1 \
    # PXA_FUSE_DELTANET=55 \
    LD_LIBRARY_PATH=/usr/local/lib

# root only for the entrypoint. pxa-entrypoint chowns a fresh empty cache
# mount and then setpriv --reuid=pxq, so the server is pid 1 and does not
# stay root. A named volume mounted at /work/.cache/pxa is root:root even
# after the chown above.
USER root

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -f "http://127.0.0.1:${LLAMA_ARG_PORT:-8080}/health" || exit 1

# pxa-entrypoint (2026-09-25): with NO arguments the image runs pxa-launch, which picks the cards,
# the model under /models and the flags. With arguments that start with '-' they are ENGINE
# arguments and go straight to llama-server, unchanged - every existing `docker run ... -m ...`
# line means what it meant; the engine fills the flags you did not give (-sm/-b/-ub/-fa) from the
# same registry pxa-launch reads. `docker run IMAGE doctor` prints the one-screen check.
ENTRYPOINT ["/usr/local/bin/pxa-entrypoint"]
