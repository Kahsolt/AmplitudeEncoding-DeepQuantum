#!/usr/bin/env bash

# 8 核机器上跑这个

export MY_LABORATORY=1

mkdir log

nohup python amp_enc_vqc.py -A ??? -B ??? >> ./log/slice_00.log 2>&1 &
nohup python amp_enc_vqc.py -A ??? -B ??? >> ./log/slice_01.log 2>&1 &
nohup python amp_enc_vqc.py -A ??? -B ??? >> ./log/slice_02.log 2>&1 &
nohup python amp_enc_vqc.py -A ??? -B ??? >> ./log/slice_03.log 2>&1 &
nohup python amp_enc_vqc.py -A ??? -B ??? >> ./log/slice_04.log 2>&1 &
nohup python amp_enc_vqc.py -A ??? -B ??? >> ./log/slice_05.log 2>&1 &
nohup python amp_enc_vqc.py -A ??? -B ??? >> ./log/slice_06.log 2>&1 &
nohup python amp_enc_vqc.py -A ??? -B ??? >> ./log/slice_07.log 2>&1 &
