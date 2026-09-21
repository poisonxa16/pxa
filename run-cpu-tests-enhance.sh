#!/bin/bash
set -e
cd /w
g++ -O2 -std=c++17 -mf16c -Iggml/include -Iggml/src tests/test-pxq-cpu-dequant.cpp -Lbuild-cpu/ggml/src -lggml -o /tmp/t1 -lm
LD_LIBRARY_PATH=build-cpu/ggml/src /tmp/t1
g++ -O2 -std=c++17 -mf16c -Iggml/include -Iggml/src tests/test-pxq-cpu-moe.cpp -Lbuild-cpu/ggml/src -lggml -o /tmp/t2 -lm
LD_LIBRARY_PATH=build-cpu/ggml/src /tmp/t2
cd /w/.tie-harness
g++ -O2 -std=c++17 -mf16c -c tu_old.cpp -o /tmp/o.o
g++ -O2 -std=c++17 -mf16c -c tu_new.cpp -o /tmp/n.o
g++ -O2 -std=c++17 main.cpp /tmp/o.o /tmp/n.o -o /tmp/tieab -lm
/tmp/tieab
