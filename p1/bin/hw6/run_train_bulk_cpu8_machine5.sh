#!/usr/bin/env bash

export MY_LABORATORY=1

mkdir log

nohup python train_bulk.py -A 4280 -B 4387 >> ./log/slice_40.log 2>&1 &
nohup python train_bulk.py -A 4387 -B 4494 >> ./log/slice_41.log 2>&1 &
nohup python train_bulk.py -A 4494 -B 4601 >> ./log/slice_42.log 2>&1 &
nohup python train_bulk.py -A 4601 -B 4708 >> ./log/slice_43.log 2>&1 &
nohup python train_bulk.py -A 4708 -B 4815 >> ./log/slice_44.log 2>&1 &
nohup python train_bulk.py -A 4815 -B 4922 >> ./log/slice_45.log 2>&1 &
nohup python train_bulk.py -A 4922 -B 5029 >> ./log/slice_46.log 2>&1 &
nohup python train_bulk.py -A 5029 -B 5139 >> ./log/slice_47.log 2>&1 &
