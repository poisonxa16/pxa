#!/bin/bash
# Runs INSIDE the bare ubuntu:22.04 container (see encode-runtime-bare.sh): installs only python3 (apt), starts PXA Control, drives the GPU-runtime flow, runs the
# card smoke test. Mounts: /w = the engine worktree (read-only), /proof = pro-old, pro-new (read-only), /out = work + logs. Env: PXA_PACKAGE_PUBKEY, PXA_LICENCE_URL.
set -eu
export DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1
echo "=== the machine: $(. /etc/os-release; echo $PRETTY_NAME), driver only"
nvidia-smi -L
echo "--- CUDA libraries visible to the loader (expect none):"; (ldconfig -p | grep -E 'cublas|cusolver|cusparse|nvJitLink|cudart' || echo "(none)")
echo "--- CUDA toolkit folders (expect none):"; (ls -d /usr/local/cuda* /opt/cuda* 2>/dev/null || echo "(none)")
apt-get update -qq >/dev/null && apt-get install -y -qq python3 >/dev/null
echo "--- $(python3 --version) from apt (the only thing installed)"
export HOME=/out/home PXA_CONTROL_CONFIG_DIR=/out/cfg PXA_LAUNCH_STATE=/out/state PXA_ENCODE_HOME=/out/encode PXA_ENCODER_HOME=/out/encoder PXA_CONTROL_DISCOVER=0 PXA_MODELS_DIR=/out/models
mkdir -p $HOME /out/cfg /out/state /out/encode /out/encoder /out/models
python3 /w/tools/pxa-launch.py --gui --port 7814 --no-browser > /out/control.log 2>&1 &
CP=$!
for i in $(seq 1 40); do python3 -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:7814/', timeout=2)" 2>/dev/null && break; sleep 1; done
python3 /w/tests/encode-runtime-bare.py --control http://127.0.0.1:7814 --pro-old /proof/pro-old --pro-new /proof/pro-new --enchome /out/encoder --summary /out/bare-summary.json
RC=$?
echo "=== the card: the GPU runtime in use on a real GPU (cuBLAS SGEMM, cuSOLVER Cholesky), libpxqe.so opened through the shipped wrapper"
PACK=$(ls -d /out/encoder/runtime/*/ | head -1)
CUDA_DEVICE_ORDER=PCI_BUS_ID PXQE_RUNTIME_DIR=$PACK python3 /w/tests/gpu-runtime-smoke.py /proof/pro-new
RC2=$?
kill $CP 2>/dev/null || true
echo "driver rc=$RC smoke rc=$RC2"
[ "$RC" = 0 ] && [ "$RC2" = 0 ]
