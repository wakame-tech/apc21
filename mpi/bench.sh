#!/bin/bash
echo "p,n,time"
for p in 1 2 4 8; do
  for n in 128 256 512 1024 2048; do
    for _ in {1..3}; do
      make -s run p=$p n=$n
    done
  done
done
