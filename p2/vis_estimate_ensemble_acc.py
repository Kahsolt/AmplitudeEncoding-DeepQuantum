#!/usr/bin/env python3
# Author: Armit
# Create Time: 周六 2024/11/23 

# 模拟计算：每个二分类模型都达到准确率 p 时，集成后的 5 分类模型约可达到准确率 Acc ~= p**2

'''
NOTE: acc ~ p**2
p=0.70, acc=46.314%
p=0.75, acc=54.294%
p=0.80, acc=62.911%
p=0.85, acc=71.928%
p=0.90, acc=81.173%
'''

import numpy as np
from numpy import ndarray


class BinaryModel:

  def __init__(self, i:int, j:int, acc:float):
    self.dom = (i, j)
    self.acc = acc

  def __call__(self, ys:ndarray):
    preds = []
    for y in ys:
      if y in self.dom:   # 这个模型能判断这个样本
        if np.random.rand() < self.acc:  # 恰好正确判断这个样本
          pred = self.dom.index(y)
        else:
          pred = list({0, 1} - {self.dom.index(y)})[0]
      else:
        pred = np.random.randint(0, 2)
      preds.append(pred)
    return np.asarray(preds, dtype=np.int32)


def run_simulate(p:float, N:int=100000):
  Y = np.random.randint(0, 5, size=[N])
  preds = []
  for i in range(4):
    for j in range(i+1, 5):
      model = BinaryModel(i, j, p)
      pred = model(Y)
      pred = np.asarray([[i, j][e] for e in pred])
      preds.append(pred)
  preds = np.stack(preds, axis=-1)    # [N, V=10]
  voted = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=preds)

  acc = (Y == voted).sum() / len(Y)
  return acc


for p in [0.70, 0.75, 0.80, 0.85, 0.90]:
  acc_list = []
  for i in range(10):
    acc_list.append(run_simulate(p))
  print(f'>> p={p}, acc={np.mean(acc_list):.3%}')
