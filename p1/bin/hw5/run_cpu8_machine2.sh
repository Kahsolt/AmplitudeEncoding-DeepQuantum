#!/usr/bin/env bash

# 8 核机器上跑这个
# Intel(R) Xeon(R) Platinum 8378C CPU @ 2.80GHz

export MY_LABORATORY=1

mkdir log

nohup python amp_enc_vqc.py -A 2056 -B 2184 >> ./log/slice_16.log 2>&1 &
nohup python amp_enc_vqc.py -A 2184 -B 2313 >> ./log/slice_17.log 2>&1 &
nohup python amp_enc_vqc.py -A 2313 -B 2441 >> ./log/slice_18.log 2>&1 &
nohup python amp_enc_vqc.py -A 2441 -B 2570 >> ./log/slice_19.log 2>&1 &
nohup python amp_enc_vqc.py -A 2570 -B 2698 >> ./log/slice_20.log 2>&1 &
nohup python amp_enc_vqc.py -A 2698 -B 2827 >> ./log/slice_21.log 2>&1 &
nohup python amp_enc_vqc.py -A 2827 -B 2955 >> ./log/slice_22.log 2>&1 &
nohup python amp_enc_vqc.py -A 2955 -B 3084 >> ./log/slice_23.log 2>&1 &
