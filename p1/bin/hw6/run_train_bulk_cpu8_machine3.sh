#!/usr/bin/env bash

export MY_LABORATORY=1

mkdir log

nohup python train_bulk.py -A 2568 -B 2675 >> ./log/slice_24.log 2>&1 &
nohup python train_bulk.py -A 2675 -B 2782 >> ./log/slice_25.log 2>&1 &
nohup python train_bulk.py -A 2782 -B 2889 >> ./log/slice_26.log 2>&1 &
nohup python train_bulk.py -A 2889 -B 2996 >> ./log/slice_27.log 2>&1 &
nohup python train_bulk.py -A 2996 -B 3103 >> ./log/slice_28.log 2>&1 &
nohup python train_bulk.py -A 3103 -B 3210 >> ./log/slice_29.log 2>&1 &
nohup python train_bulk.py -A 3210 -B 3317 >> ./log/slice_30.log 2>&1 &
nohup python train_bulk.py -A 3317 -B 3424 >> ./log/slice_31.log 2>&1 &
