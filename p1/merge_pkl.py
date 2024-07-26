#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/24 

# 合并 amp_enc_vqc.py 产生的多个 test_dataset_A=a_B=b.pkl 为单个 test_dataset.pkl

import os
os.environ['MY_LABORATORY'] = '1'

from re import compile as Regex
import pickle as pkl
from time import time
from glob import glob
from tqdm import tqdm
from typing import List, Tuple

from amp_enc_vqc import N_SAMPLES
from utils import QMNISTDataset, mean

OUTPUT_DIR = './output'
assert os.path.exists(OUTPUT_DIR)

R_FILENAME = Regex('A=(\d+)_B=(\d+)')
R_FIDELITY = Regex('>> fid: ([\d\.]+)')

def parse_filename(fn:str) -> Tuple[int, int]:
  return [int(e) for e in R_FILENAME.findall(fn)[0]]

def logs_to_fids(logs:List[str]) -> List[float]:
  fids = []
  for log in logs:
    m = R_FIDELITY.findall(log)
    if not m: continue
    fids.append(float(m[0]))
  return fids


''' Read & Merge '''
s = time()

fns = glob(f'{OUTPUT_DIR}/test_dataset_A=*_B=*.pkl')
data_list = [None] * N_SAMPLES
fid_list  = [None] * N_SAMPLES
cnt = 0

for fn in tqdm(fns):
  print(f'>> processing {fn}...')
  with open(fn, 'rb') as fh:
    samples = pkl.load(fh)

  with open(fn.replace('.pkl', '.log'), 'r', encoding='utf-8') as fh:
    logs = fh.read().strip().split('\n')
    fids = logs_to_fids(logs)

  A, B = parse_filename(fn)
  for i, j in zip(range(A, B), range(len(samples))):
    data_list[i] = samples[j]
    fid_list[i] = fids[j]
    cnt += 1

print(f'>> total need: {N_SAMPLES}, found processed: {cnt}')
if N_SAMPLES != cnt:
  print('[Missing samples]')
  for i, sample in  enumerate(data_list):
    if sample is None:
      print(f'  {i}')
fids_valid = [e for e in fids if e is not None]
print(f'mean(fid): {mean(fids_valid)}')

t = time()
print(f'>> merge files (time: {t - s:.3f}s)')


''' Write '''
s = time()
test_dataset = QMNISTDataset(label_list=[0,1,2,3,4], train=False, skip_generate_data=True)
test_dataset.data_list = data_list
fp = f'{OUTPUT_DIR}/test_dataset.pkl'
with open(fp, 'wb') as fh:
  pkl.dump(test_dataset, fh)
t = time()
fsize = os.path.getsize(fp)
print(f'>> save to {fp} {fsize / 2**20:.3f} MB (time: {t - s:.3f}s)')
