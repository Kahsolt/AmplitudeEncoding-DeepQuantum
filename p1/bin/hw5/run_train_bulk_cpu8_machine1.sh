#!/usr/bin/env bash

# 8 核机器上跑这个
# Intel(R) Xeon(R) Platinum 8378C CPU @ 2.80GHz

export MY_LABORATORY=1

mkdir log

nohup python train_bulk.py -A 1028 -B 1156 >> ./log/slice_08.log 2>&1 &
nohup python train_bulk.py -A 1156 -B 1285 >> ./log/slice_09.log 2>&1 &
nohup python train_bulk.py -A 1285 -B 1413 >> ./log/slice_10.log 2>&1 &
nohup python train_bulk.py -A 1413 -B 1542 >> ./log/slice_11.log 2>&1 &
nohup python train_bulk.py -A 1542 -B 1670 >> ./log/slice_12.log 2>&1 &
nohup python train_bulk.py -A 1670 -B 1799 >> ./log/slice_13.log 2>&1 &
nohup python train_bulk.py -A 1799 -B 1927 >> ./log/slice_14.log 2>&1 &
nohup python train_bulk.py -A 1927 -B 2056 >> ./log/slice_15.log 2>&1 &