#!/usr/bin/env bash

export MY_LABORATORY=1

mkdir log

nohup python amp_enc_vqc.py -A    0 -B  107 >> ./log/slice_00.log 2>&1 &
nohup python amp_enc_vqc.py -A  107 -B  214 >> ./log/slice_01.log 2>&1 &
nohup python amp_enc_vqc.py -A  214 -B  321 >> ./log/slice_02.log 2>&1 &
nohup python amp_enc_vqc.py -A  321 -B  428 >> ./log/slice_03.log 2>&1 &
nohup python amp_enc_vqc.py -A  428 -B  535 >> ./log/slice_04.log 2>&1 &
nohup python amp_enc_vqc.py -A  535 -B  642 >> ./log/slice_05.log 2>&1 &
nohup python amp_enc_vqc.py -A  642 -B  749 >> ./log/slice_06.log 2>&1 &
nohup python amp_enc_vqc.py -A  749 -B  856 >> ./log/slice_07.log 2>&1 &
