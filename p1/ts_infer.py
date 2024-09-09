#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/09 

# 计时: 运行训练完的 VQC 耗时

import time

from utils import *


def run():
  OUTPUT_DIR = './output'

  # 数据集
  t0 = time.time()
  with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'rb') as file:
    test_dataset = pickle.load(file)
  t1 = time.time()
  print(f'>> load pickle done ({t1 - t0:.3f}s)')    # 185.275s

  t0 = time.time()
  for x, y, z in test_dataset:
    z_v = z()
  t1 = time.time()
  print(f'>> run circuit done ({t1 - t0:.3f}s)')    # 174.562s


if __name__ == '__main__':
  run()
