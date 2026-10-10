#!/bin/bash
# run-pxa-smalln-float.sh CARD -- build tests/test-pxa-smalln-float.cpp against build4 and run it on one GPU, lever on and off
WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}; BD=${BD:?set BD to a CUDA build dir}; CARD=${1:?card}
docker run --rm --runtime=nvidia -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NVIDIA_VISIBLE_DEVICES=$CARD -v $WT:/w -v $BD:/b -v /tmp:/o --entrypoint bash pxa-sm60-dev:pkg -c "
  cd /w && g++ -O2 -std=c++17 -Iggml/include tests/test-pxa-smalln-float.cpp -L/b/ggml/src -lggml -Wl,-rpath,/b/ggml/src -o /o/test-pxa-smalln-float 2>&1 | grep -v -i warn | head -20
  echo '=== lever on';  PXA_SMALLN_FLOAT=1 /o/test-pxa-smalln-float /o/snf-on.bin
  echo '=== lever off'; /o/test-pxa-smalln-float /o/snf-off.bin
  python3 -c \"
import struct
a=open('/o/snf-on.bin','rb').read(); b=open('/o/snf-off.bin','rb').read(); n=len(a)//4
fa=struct.unpack('%df'%n,a); fb=struct.unpack('%df'%n,b)
import math; r=math.sqrt(sum(y*y for y in fb)/n); d=max(abs(x-y) for x,y in zip(fa,fb))/r; print('kernel vs cuBLAS: %d outputs, max |diff| / rms(cublas) = %.3g' % (n,d)); assert d < 1e-2\""
