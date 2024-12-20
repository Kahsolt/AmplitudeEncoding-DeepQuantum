#!/usr/bin/env bash

# 16 核机器上跑这个
# Intel Xeon Gold 6226R@2.9Ghz
# https://versus.com/en/intel-xeon-gold-6226r-vs-intel-xeon-silver-4215r

export MY_LABORATORY=1

mkdir log

nohup python amp_enc_vqc.py -A    0 -B  215 >> ./log/slice_00.log 2>&1 &
nohup python amp_enc_vqc.py -A  215 -B  430 >> ./log/slice_01.log 2>&1 &
nohup python amp_enc_vqc.py -A  430 -B  645 >> ./log/slice_02.log 2>&1 &
nohup python amp_enc_vqc.py -A  645 -B  860 >> ./log/slice_03.log 2>&1 &
nohup python amp_enc_vqc.py -A  860 -B 1075 >> ./log/slice_04.log 2>&1 &
nohup python amp_enc_vqc.py -A 1075 -B 1290 >> ./log/slice_05.log 2>&1 &
nohup python amp_enc_vqc.py -A 1290 -B 1505 >> ./log/slice_06.log 2>&1 &
nohup python amp_enc_vqc.py -A 1505 -B 1720 >> ./log/slice_07.log 2>&1 &
nohup python amp_enc_vqc.py -A 1720 -B 1935 >> ./log/slice_08.log 2>&1 &
nohup python amp_enc_vqc.py -A 1935 -B 2150 >> ./log/slice_09.log 2>&1 &
nohup python amp_enc_vqc.py -A 2150 -B 2365 >> ./log/slice_10.log 2>&1 &
nohup python amp_enc_vqc.py -A 2365 -B 2580 >> ./log/slice_11.log 2>&1 &
nohup python amp_enc_vqc.py -A 2580 -B 2795 >> ./log/slice_12.log 2>&1 &
nohup python amp_enc_vqc.py -A 2795 -B 3010 >> ./log/slice_13.log 2>&1 &
nohup python amp_enc_vqc.py -A 3010 -B 3225 >> ./log/slice_14.log 2>&1 &
nohup python amp_enc_vqc.py -A 3225 -B 3440 >> ./log/slice_15.log 2>&1 &
