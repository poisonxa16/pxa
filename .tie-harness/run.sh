#!/bin/bash
set -e
cd /w/.tie-harness
g++ -O2 -std=c++17 -mf16c -c tu_old.cpp -o /tmp/o.o
g++ -O2 -std=c++17 -mf16c -c tu_new.cpp -o /tmp/n.o
g++ -O2 -std=c++17 main.cpp /tmp/o.o /tmp/n.o -o /tmp/tieab -lm
/tmp/tieab
