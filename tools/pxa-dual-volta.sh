#!/bin/sh
# SPDX-License-Identifier: MIT
# Opt-in launch profile qualified on resident Q6_K / two V100-SXM2-32GB cards.
set -eu
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 /path/to/llama-server [server arguments...]" >&2
    exit 64
fi
# A value the operator set, including an empty value, always wins.
if [ "${CUDA_SCALE_LAUNCH_QUEUES+x}" != x ]; then
    CUDA_SCALE_LAUNCH_QUEUES=4x
    export CUDA_SCALE_LAUNCH_QUEUES
fi
exec "$@"
