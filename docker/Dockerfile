# pxa server image — Pascal/Volta (sm 60;61;70): P100, GTX 1080 Ti/P40, V100.
#
# Multi-stage build. The build stage matches the CUDA 12.8 devel base this engine is
# developed and gated against; the runtime stage ships only the binaries, the shared
# libraries they need, and the CUDA 12.8 runtime libs (cudart/cublas/cublasLt) — no
# compiler, no headers, no build tooling.
#
# The image carries TWO engines: this one, as /usr/local/bin/llama-server (the
# ENTRYPOINT), and the upstream engine every published comparison was measured
# against, as /opt/pxa/bin/upstream-ik-server, pinned to one commit and labelled
# with it. An engine-only comparison is then one `docker run --entrypoint` away,
# on the same card, the same driver, the same CUDA runtime and the same weight
# file — see docker/README.md, "The second binary", and bench/fair/run.sh.
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

# The upstream engine to ship alongside this one, pinned to ONE commit.
# 3c58ae373a0081c884099f435fb16ca720852bf7 (2026-08-31) is the revision named in
# every published upstream row: docs/ENGINE.md's three comparison tables,
# bench/fair-battle.md's cross-engine sections, and the `upstream ik_llama.cpp
# 3c58ae37` rows of docs/data/chart-data-2026-09-02.csv, which is the CSV behind
# the README's own chart. The fork point (`1520eda98056`, docs/ENGINE.md
# "Do they rebase?") is deliberately NOT used here: it is where this codebase
# started, not what anything was measured against.
ARG IK_SHA=3c58ae373a0081c884099f435fb16ca720852bf7
ARG IK_URL=https://github.com/ikawrakow/ik_llama.cpp

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
        build-essential cmake git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Same core flags as the gate build (build-spd/CMakeCache.txt): shared libs on,
# curl off (the server does not need outbound HTTP fetch for this image), Release.
# The explicit linker flags point ggml's CUDA driver symbols (cuMemCreate etc, used
# by the VMM pool allocator) at the devel image's driver stub -- needed for a clean
# container build where no real libcuda.so.1 is present until run time.
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_CURL=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    && cmake --build build --target llama-server llama-cli llama-bench \
        -j "$([ "${BUILD_JOBS}" != "0" ] && [ -n "${BUILD_JOBS}" ] && echo "${BUILD_JOBS}" || nproc)"

# ---------------------------------------------------------------------------
# Upstream engine stage — the second binary
# ---------------------------------------------------------------------------
# Same CUDA toolkit, same three architectures, same container as the engine
# above, so an engine-only comparison differs in exactly one thing: the binary.
# Nothing from this stage reaches the runtime image except the one executable.
FROM ${BASE_CUDA_DEV_CONTAINER} AS build-upstream

ARG CUDA_ARCHS="60;61;70"
ARG BUILD_JOBS="0"
ARG IK_SHA
ARG IK_URL

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        build-essential cmake git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /upstream

# Shallow fetch of exactly the pinned commit — no clone, no history, no tags,
# no branch that could move. This layer's cache key is IK_SHA and nothing else:
# editing this repo's own source never re-fetches it, and changing IK_SHA
# re-fetches only this. The recorded revision line ships next to the binary so
# an image can be asked what it is carrying without trusting its labels.
RUN git init -q . && \
    git remote add origin "${IK_URL}" && \
    git fetch --depth 1 --no-tags origin "${IK_SHA}" && \
    git checkout -q FETCH_HEAD && \
    git --no-pager log -1 --format="ik_llama.cpp %H %ad %s" > /upstream/UPSTREAM-REV

# BUILD_SHARED_LIBS=OFF is not a preference. The runtime image already carries
# THIS engine's libllama.so/libggml.so in /usr/local/lib, and a dynamically
# linked upstream server would load those at run time — benchmarking this
# engine's kernels while wearing upstream's name. Static links it against its
# own ggml/llama; the only shared libraries it then needs are the CUDA runtime
# ones both engines share.
# GGML_CUDA_F16=ON is the configuration the published upstream rows were
# measured on (bench/fair-battle.md, "2026-09-03 cross-engine, same file, three
# engines").
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_F16=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_CURL=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    && cmake --build build --target llama-server \
        -j "$([ "${BUILD_JOBS}" != "0" ] && [ -n "${BUILD_JOBS}" ] && echo "${BUILD_JOBS}" || nproc)"

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
ARG IK_SHA
ARG IK_URL

LABEL org.opencontainers.image.source="https://github.com/poisonxa16/pxa" \
      org.opencontainers.image.title="PXA" \
      org.opencontainers.image.description="Pascal/Volta 4-bit codec + kernel pack for llama.cpp" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${TAG}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.pxa.upstream.ik.sha="${IK_SHA}" \
      org.pxa.upstream.ik.url="${IK_URL}"

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends curl ca-certificates libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r pxq && useradd -r -g pxq -d /work -m pxq

WORKDIR /work

COPY --from=build /src/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /src/build/bin/llama-cli    /usr/local/bin/llama-cli
COPY --from=build /src/build/bin/llama-bench  /usr/local/bin/llama-bench
COPY --from=build /src/build/src/libllama.so           /usr/local/lib/
COPY --from=build /src/build/ggml/src/libggml.so       /usr/local/lib/
COPY --from=build /src/build/examples/mtmd/libmtmd.so  /usr/local/lib/
RUN ldconfig

# The second engine. Deliberately NOT on PATH and NOT in /usr/local/lib's
# neighbourhood: it is a comparison instrument, not a way to serve. Run it with
#   docker run --entrypoint /opt/pxa/bin/upstream-ik-server <image> -m ... -ngl 99 ...
# Its revision is in the org.pxa.upstream.ik.sha label and, verbatim from its own
# git log, in the .rev file next to it.
COPY --from=build-upstream /upstream/build/bin/llama-server /opt/pxa/bin/upstream-ik-server
COPY --from=build-upstream /upstream/UPSTREAM-REV           /opt/pxa/bin/upstream-ik-server.rev

# One file, both documented names: the `docker run` lines and the release docs
# say /usr/local/bin/llama-server, while everything that reasons about the two
# engines together looks in /opt/pxa/bin. A symlink, so there is exactly one
# binary and no chance of the two paths drifting to different builds.
RUN ln -sf /usr/local/bin/llama-server /opt/pxa/bin/llama-server

# Model directory mount point (empty in the image — models are never baked in).
RUN mkdir -p /models && chown pxq:pxq /models /work

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
    # PXA levers — these are the shipped rc/unified-final-20260903 defaults (see
    # docker/README.md for what each does); left unset here so the binary's own
    # compiled-in defaults apply. Uncomment / pass -e to override:
    # PXA_PIPELINE_PP=1 \
    # PXA_CKPT_PROMPT_POLICY=1 \
    # PXA_CKPT_LEAN_SYNC=1 \
    # PXA_DN_SHARE_INPUTS=-1 \
    # PXA_FUSE_DELTANET=55 \
    LD_LIBRARY_PATH=/usr/local/lib

USER pxq

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -f "http://127.0.0.1:${LLAMA_ARG_PORT:-8080}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/llama-server"]
