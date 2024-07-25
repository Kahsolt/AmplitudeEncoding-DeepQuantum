#!/usr/bin/env bash

export MY_LABORATORY=1

mkdir log

nohup python train_bulk.py -A  856 -B  963 >> ./log/slice_08.log 2>&1 &
nohup python train_bulk.py -A  963 -B 1070 >> ./log/slice_09.log 2>&1 &
nohup python train_bulk.py -A 1070 -B 1177 >> ./log/slice_10.log 2>&1 &
nohup python train_bulk.py -A 1177 -B 1284 >> ./log/slice_11.log 2>&1 &
nohup python train_bulk.py -A 1284 -B 1391 >> ./log/slice_12.log 2>&1 &
nohup python train_bulk.py -A 1391 -B 1498 >> ./log/slice_13.log 2>&1 &
nohup python train_bulk.py -A 1498 -B 1605 >> ./log/slice_14.log 2>&1 &
nohup python train_bulk.py -A 1605 -B 1712 >> ./log/slice_15.log 2>&1 &
