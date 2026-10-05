#!/bin/bash
# run-xcache-async-test.sh -- build + run tests/test-pxa-xcache-async.cpp against a build tree of this worktree, in the pxa-sm60-dev image
# usage: tests/run-xcache-async-test.sh [test args]          host driver (no GPU): the worker is bit-identical to the plain CPU graph
#        CARD=0 tests/run-xcache-async-test.sh --gpu 0       the real submit / wait kernels on that card (take the card's lock first)
# env:   BD (build dir, default build-ca; needs ggml/src/libggml.so), OUT (default /tmp), CPUS
W=${W:-$(cd "$(dirname "$0")/.." && pwd)}
BD=${BD:-build-ca}
OUT=${OUT:-/tmp}
GPUARGS="-e NVIDIA_VISIBLE_DEVICES=none"
STUBS="ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs && "
[ -n "$CARD" ] && STUBS="" && GPUARGS="--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$CARD -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NVIDIA_DRIVER_CAPABILITIES=compute,utility"
docker run --rm $GPUARGS --cpuset-cpus ${CPUS:-18-35,54-71} -e LD_LIBRARY_PATH=/w/$BD/ggml/src:/usr/local/cuda/lib64 -e LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
  -v $W:/w -v $OUT:/o -v ${MODELDIR:-$PWD/models}:${MODELDIR:-$PWD/models}:ro --entrypoint /bin/bash pxa-sm60-dev:ccache -c "cd /w && $STUBS \
  nice -n 19 g++ -O1 -g -std=c++17 -DGGML_USE_CUDA -Wall -Wno-unused-parameter -Iinclude -Iggml/include -Iggml/src -Isrc \
    tests/test-pxa-xcache-async.cpp src/llama-pxa-xcache-async-core.cpp -L$BD/ggml/src -lggml -L/usr/local/cuda/lib64/stubs -lcuda -pthread -o /o/test-xc-async 2>&1 | head -60 && /o/test-xc-async $*"
