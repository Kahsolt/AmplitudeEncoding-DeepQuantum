#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/09 

# 后处理 pkl: 微小的性能优化...

from typing import List
import pickle
import deepquantum as dq
from utils import QMNISTDataset

OUTPUT_DIR = './output'
EPS = 1e-1


# 数据集
fp = f'{OUTPUT_DIR}/test_dataset.tmp.pkl'   # debug use
#fp = f'{OUTPUT_DIR}/test_dataset.pkl'
print(f'>> loading from {fp}')
with open(fp, 'rb') as file:
  test_dataset: QMNISTDataset = pickle.load(file)

for i, (x, y, z) in enumerate(test_dataset):
  z: dq.QubitCircuit

  # no grad
  z.zero_grad()
  for op in z.operators:
    if hasattr(op, 'theta'):
      op.theta.requires_grad = False

  # prune rot gates
  n_ops_old = len(z.operators)
  ops: List[dq.operation.Operation] = []
  for op in z.operators:
    if not hasattr(op, 'theta'):
      ops.append(op)
    else:
      v = abs(op.theta.item())
      if v > EPS:
        ops.append(op)
      else:
        print(f'    trim small v: {v}')
  n_ops_new = len(ops)
  print(f'>> [trim] n_ops: {n_ops_old} => {n_ops_new}')
  if n_ops_new != n_ops_old:
    qc_new = dq.QubitCircuit(nqubit=z.nqubit)
    for op in ops:
      op.requires_grad = False
      qc_new.add(op)
    test_dataset.data_list[i] = (x, y, qc_new)
    z = qc_new


fp = f'{OUTPUT_DIR}/test_dataset-prune.pkl'
print(f'>> saving to {fp}')
with open(fp, 'wb') as file:
  pickle.dump(test_dataset, file)
