#!/usr/bin/env bash

export MY_LABORATORY=1

mkdir log

nohup python train_bulk.py -A 1712 -B 1819 >> ./log/slice_16.log 2>&1 &
nohup python train_bulk.py -A 1819 -B 1926 >> ./log/slice_17.log 2>&1 &
nohup python train_bulk.py -A 1926 -B 2033 >> ./log/slice_18.log 2>&1 &
nohup python train_bulk.py -A 2033 -B 2140 >> ./log/slice_19.log 2>&1 &
nohup python train_bulk.py -A 2140 -B 2247 >> ./log/slice_20.log 2>&1 &
nohup python train_bulk.py -A 2247 -B 2354 >> ./log/slice_21.log 2>&1 &
nohup python train_bulk.py -A 2354 -B 2461 >> ./log/slice_22.log 2>&1 &
nohup python train_bulk.py -A 2461 -B 2568 >> ./log/slice_23.log 2>&1 &
