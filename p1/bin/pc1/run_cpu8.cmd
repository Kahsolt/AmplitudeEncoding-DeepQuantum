@ECHO OFF

REM run on 8 core machine

SET MY_LABORATORY=1

START python amp_enc_vqc.py -A    0 -B  642
START python amp_enc_vqc.py -A  642 -B 1285
START python amp_enc_vqc.py -A 1927 -B 2570
START python amp_enc_vqc.py -A 1285 -B 1927
START python amp_enc_vqc.py -A 3212 -B 3854
START python amp_enc_vqc.py -A 2570 -B 3212
START python amp_enc_vqc.py -A 3854 -B 4497
START python amp_enc_vqc.py -A 4497 -B 5139
