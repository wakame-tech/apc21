#!/bin/bash
echo "n,time"
for n in 128 256 512 1024 2048; do
  for _ in {1..5}; do
    ./a.out $n
  done
done
