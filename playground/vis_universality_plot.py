#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/10 

# 从注释里抽数据来画图

from re import Match
from re import compile as Regex
import numpy as np
import matplotlib.pyplot as plt

PY_FILE = './vis_universality.py'

R_RECORD = Regex('gcnt\=(\d+), fid\=([\d\.]+), ts\=([\d\.]+)s')


with open(PY_FILE, 'r', encoding='utf-8') as fh:
  lines = fh.read().strip().split('\n')
  lines = [ln.strip() for ln in lines if ln.strip()]
  lines = [ln for ln in lines if ln.startswith('#') and not ln.startswith('##(noplot)')]

gcnt_list = []
fid_list = []
sc_list = []
for line in lines:
  m: Match = R_RECORD.search(line)
  if not m: continue
  gcnt, fid, ts = m.groups()
  gcnt = int(gcnt)
  fid = float(fid)
  sc = 2 * fid + (1 - gcnt / 1000)
  gcnt_list.append(gcnt)
  fid_list.append(fid)
  sc_list.append(sc)

print('gcnt:', gcnt_list)
print('fid:', fid_list)
print('sc:', sc_list)

point_size = 5**np.asarray(sc_list)

plt.scatter(gcnt_list, fid_list, alpha=0.7, s=point_size)
plt.xlabel('Gate Count')
plt.ylabel('Fidelity')
plt.tight_layout()
plt.show()
