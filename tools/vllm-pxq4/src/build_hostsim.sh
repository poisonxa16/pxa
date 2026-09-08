#!/usr/bin/env bash
# build_hostsim.sh — build the CPU simulator and run the GPU-free gates.
# Needs nothing but g++ and numpy. No CUDA, no GPU, no container, no lease.
set -euo pipefail
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -shared -fPIC -Ihostsim -I. -pthread -Wno-unknown-pragmas \
    pxq4_kernel_hostsim.cpp -o libpxq4_hostsim.so
python3 test_pxq4_kernel_ref.py
python3 test_pxq4_mmv_split.py
python3 test_pxq4_mmv_mt.py
