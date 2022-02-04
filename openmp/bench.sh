#!/bin/bash
# echo "t,n,time"
# for t in 1 2 4 8 16 24; do
#   for n in 128 256 512 1024 2048; do
#     for _ in {1..3}; do
#       ./a.out $t $n
#     done
#   done
# done

echo "n,time"
for n in 128 256 512 1024 2048; do
  for _ in {1..3}; do
    ./a.out 24 $n
  done
done
