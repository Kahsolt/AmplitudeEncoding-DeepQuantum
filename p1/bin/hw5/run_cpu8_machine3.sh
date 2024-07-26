#!/usr/bin/env bash

# 8 核机器上跑这个
# Intel(R) Xeon(R) Platinum 8378C CPU @ 2.80GHz

export MY_LABORATORY=1

mkdir log

nohup python amp_enc_vqc.py -A 3084 -B 3212 >> ./log/slice_24.log 2>&1 &
nohup python amp_enc_vqc.py -A 3212 -B 3341 >> ./log/slice_25.log 2>&1 &
nohup python amp_enc_vqc.py -A 3341 -B 3469 >> ./log/slice_26.log 2>&1 &
nohup python amp_enc_vqc.py -A 3469 -B 3598 >> ./log/slice_27.log 2>&1 &
nohup python amp_enc_vqc.py -A 3598 -B 3726 >> ./log/slice_28.log 2>&1 &
nohup python amp_enc_vqc.py -A 3726 -B 3855 >> ./log/slice_29.log 2>&1 &
nohup python amp_enc_vqc.py -A 3855 -B 3983 >> ./log/slice_30.log 2>&1 &
nohup python amp_enc_vqc.py -A 3983 -B 4112 >> ./log/slice_31.log 2>&1 &
