#!/usr/bin/env bash

# 8 核机器上跑这个
# Intel(R) Xeon(R) Platinum 8378C CPU @ 2.80GHz

export MY_LABORATORY=1

mkdir log

nohup python amp_enc_vqc.py -A    0 -B  128 >> ./log/slice_00.log 2>&1 &
nohup python amp_enc_vqc.py -A  128 -B  257 >> ./log/slice_01.log 2>&1 &
nohup python amp_enc_vqc.py -A  257 -B  385 >> ./log/slice_02.log 2>&1 &
nohup python amp_enc_vqc.py -A  385 -B  514 >> ./log/slice_03.log 2>&1 &
nohup python amp_enc_vqc.py -A  514 -B  642 >> ./log/slice_04.log 2>&1 &
nohup python amp_enc_vqc.py -A  642 -B  771 >> ./log/slice_05.log 2>&1 &
nohup python amp_enc_vqc.py -A  771 -B  899 >> ./log/slice_06.log 2>&1 &
nohup python amp_enc_vqc.py -A  899 -B 1028 >> ./log/slice_07.log 2>&1 &
