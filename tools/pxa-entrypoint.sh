#!/bin/sh
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
#
# pxa-entrypoint - the container's ENTRYPOINT (2026-09-25).
#
# One rule, so an existing `docker run` line never changes meaning:
#
#   docker run IMAGE                          -> pxa-launch picks the cards, the model and the flags
#                                                (the model: $PXA_MODEL, else the only .gguf under
#                                                /models; the cards: $PXA_GPUS, else all visible)
#   docker run IMAGE -m /models/x.gguf ...    -> ENGINE ARGUMENTS pass straight through to
#                                                llama-server, unchanged. The engine fills every
#                                                flag you did not give (-sm, -b, -ub, -fa) from the
#                                                same registry pxa-launch reads, so this and the line
#                                                above pick the same settings on the same cards.
#   docker run IMAGE launch [pxa-launch args] -> pxa-launch with your arguments
#   docker run IMAGE doctor                   -> pxa-launch --doctor (cards, driver, P2P, the model,
#                                                and what defaults would be picked and why)
#   docker run IMAGE <program> [args]         -> that program (llama-bench, pxa-server, bash, ...)
#
# Anything that starts with '-' is an engine argument. Nothing is added to or removed from it.

set -e

# A named volume is created root:root and hides the chown in the image.
# Start as root, and chown the cache only when it is empty and owned by uid 0
# (a fresh named volume). A host bind that already has files, or a directory
# someone else owns, is left alone. A failed mkdir or chown warns and continues.
# setpriv execs this script again, so the server ends up as pid 1 and receives
# SIGTERM. runuser would stay pid 1 and the engine would not.
# `docker run --user pxq` skips this and the writable check below warns.
if [ "$(id -u)" -eq 0 ]; then
    cache=${PXA_CACHE_DIR:-/work/.cache/pxa}
    if mkdir -p "$cache" 2>/dev/null; then
        owner=$(stat -c '%u' "$cache" 2>/dev/null || true)
        # One entry is enough. -quit stops at the first name, so a full cache
        # directory is not listed. A failure means "do not chown".
        nonempty=$(find "$cache" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || echo FAIL)
        if [ "$owner" = "0" ] && [ -z "$nonempty" ]; then
            if ! chown pxq:pxq "$cache"; then
                echo "pxa-entrypoint: could not chown $cache; continuing" >&2
            fi
        elif [ "$nonempty" = "FAIL" ]; then
            echo "pxa-entrypoint: could not read $cache; not changing its owner" >&2
        fi
    else
        echo "pxa-entrypoint: could not create $cache; continuing" >&2
    fi
    export PXA_CACHE_DIR="$cache"
    export HOME=/work
    exec setpriv --reuid=pxq --regid=pxq --init-groups -- "$0" "$@"
fi

HERE=$(dirname "$(readlink -f "$0")")
# PXA_HOME: the release tarball puts this script in the dist ROOT (next to run-server.sh, bin/,
# pxa-launch); the source tree keeps it in tools/. Pick whichever layout is actually here.
if [ -z "${PXA_HOME:-}" ]; then
    if [ -x "$HERE/run-server.sh" ] || [ -x "$HERE/bin/llama-server" ] || [ -x "$HERE/pxa-launch" ]; then
        PXA_HOME=$HERE
    else
        PXA_HOME=$(dirname "$HERE")
    fi
fi

find_bin() {
    for d in "$PXA_HOME/bin" /usr/local/bin "$HERE"; do
        if [ -x "$d/$1" ]; then echo "$d/$1"; return 0; fi
    done
    command -v "$1" 2>/dev/null || return 1
}

# /dev/shm preflight (2026-09-27, bug #206 follow-up; pxa-launch prints the same check on its own
# path). NCCL's shared-memory transport carries the multi-card reduce and lives in /dev/shm, which
# docker and podman size at 64 MiB unless told otherwise. Warn, never stop: one card, or a big
# enough /dev/shm, says nothing.
# Learned expert counts live under PXA_CACHE_DIR. A container that does not mount
# a volume there loses them on remove. Warn, never stop.
cache_check() {
    dir=${PXA_CACHE_DIR:-${HOME:-/work}/.cache/pxa}
    export PXA_CACHE_DIR="$dir"
    if ! mkdir -p "$dir" 2>/dev/null || [ ! -w "$dir" ]; then
        echo "pxa-entrypoint: the expert-count cache $dir is not writable. Learned counts will not be kept." >&2
        echo "  Mount a volume: -v pxa-cache:$dir -e PXA_CACHE_DIR=$dir" >&2
        return 0
    fi
    if [ -e /.dockerenv ]; then
        root_dev=$(stat -c '%d' / 2>/dev/null || true)
        dir_dev=$(stat -c '%d' "$dir" 2>/dev/null || true)
        if [ -n "$root_dev" ] && [ "$root_dev" = "$dir_dev" ]; then
            echo "pxa-entrypoint: the expert-count cache $dir is on the container's own disk." >&2
            echo "  Learned counts are deleted when this container is removed." >&2
            echo "  Mount a volume: -v pxa-cache:$dir -e PXA_CACHE_DIR=$dir" >&2
        fi
    fi
}

shm_check() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        n=$(printf '%s\n' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c . || true)
    else
        n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
    fi
    [ "${n:-0}" -ge 2 ] || return 0
    kb=$(df -Pk /dev/shm 2>/dev/null | awk 'NR == 2 { print $2 }')
    [ -n "$kb" ] || kb=0
    [ "$kb" -lt 1048576 ] || return 0
    echo "pxa-entrypoint: WARNING: /dev/shm is $((kb / 1024)) MiB and $n cards are visible. The multi-card" >&2
    echo "  reduce (NCCL's shared-memory transport) needs more than that; a 4-card tensor split falls back" >&2
    echo "  to a slower route (older builds served wrong tokens, bug #206). Fix: docker run --shm-size=1g" >&2
    echo "  (or --ipc=host; compose: shm_size: 1gb; podman: --shm-size=1g; LXC: a 1G tmpfs on /dev/shm)." >&2
}

# CPU preflight for the source-built image (docker/Dockerfile): the fast libggml.so in /usr/local/lib is built with AVX, AVX2, FMA and
# F16C and stops with "Illegal instruction" on a CPU without them; /usr/local/lib-compat holds the plain x86-64 build, and
# tools/pxa-cpu-check.sh reads /proc/cpuinfo and puts it first on LD_LIBRARY_PATH. The release tarball's run-server.sh and
# pxa-launch do the same for the tarball layout, so this is only for the path that runs llama-server directly.
cpu_pick() {
    for c in "$PXA_HOME/tools/pxa-cpu-check.sh" "$HERE/pxa-cpu-check.sh" /usr/local/bin/pxa-cpu-check.sh; do
        if [ -r "$c" ]; then
            . "$c"
            pxa_cpu_pick "${PXA_COMPAT_LIB_DIR:-/usr/local/lib-compat}" || exit 1
            return 0
        fi
    done
}

# PXA Control in a container is OPT-IN: PXA_CONTROL=1 (and publish its port, -p 7777:7777). A
# 127.0.0.1 bind inside a container is unreachable from the host, so it listens on every interface
# with the access token (--lan), printed once at start. Without PXA_CONTROL=1 nothing changes: no
# web port opens unless you asked for one.
control_wanted() {
    case "${PXA_CONTROL:-}" in 1|on|yes|true|always) return 0 ;; esac
    return 1
}

launcher_py() {
    for p in "$PXA_HOME/tools/pxa-launch.py" /usr/local/bin/pxa-launch "$HERE/pxa-launch.py"; do
        if [ -f "$p" ]; then echo "$p"; return 0; fi
    done
    return 1
}

engine() {
    cache_check
    shm_check
    if control_wanted && command -v python3 >/dev/null 2>&1; then
        LP=$(launcher_py) && python3 "$LP" --control-for-pid $$ --lan || true
    fi
    if [ -x "$PXA_HOME/run-server.sh" ]; then
        exec "$PXA_HOME/run-server.sh" "$@"
    fi
    cpu_pick
    # card numbers mean what nvidia-smi means by them (run-server.sh does the same)
    export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
    exec "$(find_bin llama-server)" "$@"
}

launcher() {
    cache_check
    if [ -x "$PXA_HOME/pxa-launch" ]; then
        exec "$PXA_HOME/pxa-launch" "$@"
    fi
    L=$(find_bin pxa-launch) || L=""
    if [ -n "$L" ]; then exec "$L" "$@"; fi
    for p in "$PXA_HOME/tools/pxa-launch.py" "$HERE/pxa-launch.py"; do
        if [ -f "$p" ]; then exec python3 "$p" "$@"; fi
    done
    echo "pxa-entrypoint: pxa-launch is not in this image; passing to the engine instead" >&2
    engine "$@"
}

if [ $# -eq 0 ]; then
    MODEL=${PXA_MODEL:-}
    if [ -z "$MODEL" ]; then
        # the only .gguf under /models (first shard of a split file counts once)
        LIST=$(find /models -maxdepth 3 -name '*.gguf' ! -name '*-0000[2-9]-of-*' ! -name '*mmproj*' 2>/dev/null | sort)
        n=$(printf '%s\n' "$LIST" | grep -c . || true)
        if [ "$n" -ne 1 ]; then
            echo "pxa-entrypoint: $n model files under /models - say which one:" >&2
            printf '%s\n' "$LIST" | sed '/^$/d; s/^/    /' >&2
            echo "  docker run ... -e PXA_MODEL=/models/<file>.gguf IMAGE" >&2
            echo "  or pass engine arguments:  docker run ... IMAGE -m /models/<file>.gguf" >&2
            exit 2
        fi
        MODEL=$LIST
    fi
    GPUS=${PXA_GPUS:-}
    if [ -z "$GPUS" ] && command -v nvidia-smi >/dev/null 2>&1; then
        GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, -)
    fi
    set -- --model "$MODEL" --no-interactive --yes
    [ -n "$GPUS" ] && set -- "$@" --gpus "$GPUS"
    [ -n "${LLAMA_ARG_PORT:-}" ] && set -- "$@" --port "$LLAMA_ARG_PORT"
    [ "${PXA_ALLOW_BUSY:-0}" = "1" ] && set -- "$@" --allow-busy
    # PXA_CONTROL=1: PXA Control next to the server, reachable through the published port (see control_wanted)
    control_wanted && set -- "$@" --lan
    # PXA_LAUNCH_EXTRA: more pxa-launch arguments for the no-argument path (e.g. "--explain" for a
    # dry run, "--np 2"); split on spaces
    # shellcheck disable=SC2086
    [ -n "${PXA_LAUNCH_EXTRA:-}" ] && set -- "$@" $PXA_LAUNCH_EXTRA
    launcher "$@"
fi

case "$1" in
    -*)                     engine "$@" ;;
    launch|pxa-launch)      shift; launcher "$@" ;;
    doctor|--doctor)        shift; launcher --doctor "$@" ;;
    # PXA Control in a container: a 127.0.0.1 bind is unreachable from the host, so it listens on
    # every interface and prints the access token; publish it with -p 7777:7777.
    gui|--gui)              shift; launcher --gui --lan --no-browser --models-dir /models "$@" ;;
    *)                      exec "$@" ;;
esac
