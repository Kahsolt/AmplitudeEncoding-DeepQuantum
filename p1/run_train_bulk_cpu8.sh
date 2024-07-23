#!/usr/bin/env bash

# 8 核机器上跑这个

export MY_LABORATORY=1

mkdir log

nohup python train_bulk.py -A 3440 -B 3655 >> ./log/slice_16.log 2>&1 &
nohup python train_bulk.py -A 3655 -B 3870 >> ./log/slice_17.log 2>&1 &
nohup python train_bulk.py -A 3870 -B 4085 >> ./log/slice_18.log 2>&1 &
nohup python train_bulk.py -A 4085 -B 4300 >> ./log/slice_19.log 2>&1 &
nohup python train_bulk.py -A 4300 -B 4515 >> ./log/slice_20.log 2>&1 &
nohup python train_bulk.py -A 4515 -B 4730 >> ./log/slice_21.log 2>&1 &
nohup python train_bulk.py -A 4730 -B 4935 >> ./log/slice_22.log 2>&1 &
nohup python train_bulk.py -A 4935 -B 5139 >> ./log/slice_23.log 2>&1 &
