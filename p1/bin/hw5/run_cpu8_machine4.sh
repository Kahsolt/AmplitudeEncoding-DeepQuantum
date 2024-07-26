#!/usr/bin/env bash

# 8 核机器上跑这个
# Intel(R) Xeon(R) Platinum 8378C CPU @ 2.80GHz

export MY_LABORATORY=1

mkdir log

nohup python amp_enc_vqc.py -A 4112 -B 4240 >> ./log/slice_32.log 2>&1 &
nohup python amp_enc_vqc.py -A 4240 -B 4369 >> ./log/slice_33.log 2>&1 &
nohup python amp_enc_vqc.py -A 4369 -B 4497 >> ./log/slice_34.log 2>&1 &
nohup python amp_enc_vqc.py -A 4497 -B 4626 >> ./log/slice_35.log 2>&1 &
nohup python amp_enc_vqc.py -A 4626 -B 4754 >> ./log/slice_36.log 2>&1 &
nohup python amp_enc_vqc.py -A 4754 -B 4883 >> ./log/slice_37.log 2>&1 &
nohup python amp_enc_vqc.py -A 4883 -B 5011 >> ./log/slice_38.log 2>&1 &
nohup python amp_enc_vqc.py -A 5011 -B 5139 >> ./log/slice_39.log 2>&1 &
