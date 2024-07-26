#!/usr/bin/env bash

export MY_LABORATORY=1

alias py=python
alias cls=clear

pip install deepquantum

# just download the dataset :)
if [ ! -d ../data ]; then
  python amp_enc_vqc.py -A 0 -B 0
fi
