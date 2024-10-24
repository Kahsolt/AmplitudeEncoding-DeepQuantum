#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/22 

# 查看 QAM 格点图 (大端序！)

import sys
from copy import deepcopy
import numpy as np
from numpy import ndarray

def add_suffix(x:ndarray, suffix:str) -> ndarray:
  x = deepcopy(x)
  W, H = x.shape
  for i in range(W):
    for j in range(H):
      x[i, j] = x[i, j] + suffix
  return x

def get_qam_array(n:int) -> ndarray:
  if n == 2:
    return np.asarray([
      ['10', '11'],
      ['00', '01'],
    ], dtype=object)
  else:
    x = get_qam_array(n - 2)
    W, H = x.shape
    x_ex = np.empty([2*W, 2*H], dtype=x.dtype)
    x_ex[:W, :H] = add_suffix(x[:, ::-1],    '01')
    x_ex[:W, H:] = add_suffix(x,             '00')
    x_ex[W:, :H] = add_suffix(x[::-1, ::-1], '11')
    x_ex[W:, H:] = add_suffix(x[::-1, :],    '10')
    return x_ex

nq = int(sys.argv[1]) if len(sys.argv) >= 2 else 8
assert nq % 2 == 0
array = get_qam_array(nq)
for row in array:
  for cell in row:
    print(cell, end=' ')
  print()
