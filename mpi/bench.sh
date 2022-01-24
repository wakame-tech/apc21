#!/bin/bash
echo "p,time"
for p in 1 2 4 8 16 24; do
  for n in 128 256 512 1024 2048; do
    for _ in {1..5}; do
      ./a.out $p $n
    done
  done
done
