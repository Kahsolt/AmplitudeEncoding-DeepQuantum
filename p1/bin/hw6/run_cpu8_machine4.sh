#!/usr/bin/env bash

export MY_LABORATORY=1

mkdir log

nohup python amp_enc_vqc.py -A 3424 -B 3531 >> ./log/slice_32.log 2>&1 &
nohup python amp_enc_vqc.py -A 3531 -B 3638 >> ./log/slice_33.log 2>&1 &
nohup python amp_enc_vqc.py -A 3638 -B 3745 >> ./log/slice_34.log 2>&1 &
nohup python amp_enc_vqc.py -A 3745 -B 3852 >> ./log/slice_35.log 2>&1 &
nohup python amp_enc_vqc.py -A 3852 -B 3959 >> ./log/slice_36.log 2>&1 &
nohup python amp_enc_vqc.py -A 3959 -B 4066 >> ./log/slice_37.log 2>&1 &
nohup python amp_enc_vqc.py -A 4066 -B 4173 >> ./log/slice_38.log 2>&1 &
nohup python amp_enc_vqc.py -A 4173 -B 4280 >> ./log/slice_39.log 2>&1 &
