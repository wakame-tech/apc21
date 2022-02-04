#!/bin/bash
echo "n,time"
for n in 128 256 512 1024 2048; do
  for _ in {1..3}; do
    make -s run p=16 n=$n
  done
done