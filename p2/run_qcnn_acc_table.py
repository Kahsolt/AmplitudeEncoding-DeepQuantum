#!/usr/bin/env python3
# Author: Armit
# Create Time: 周四 2024/11/07 

import json
import numpy as np
from pathlib import Path

'''
[qcnn]
| * |     0     |     1     |     2     |     3     |     4     |
| 0 |           | 84.158416 | 78.896104 | 72.697368 | 82.333333 |
| 1 | 81.848185 |           | 71.854305 | 78.187919 | 69.387755 |
| 2 | 78.246753 | 72.18543  |           | 68.316832 | 82.214765 |
| 3 | 76.644737 | 73.825503 | 66.006601 |           | 71.768707 |
| 4 | 82.333333 | 71.088435 | 82.214765 | 73.129252 |           |

[mera]
| * |     0     |     1     |     2     |     3     |     4     |
| 0 |           | 83.49835  | 78.571429 | 72.697368 | 82.666667 |
| 1 | 83.828383 |           | 72.516556 | 76.174497 | 69.047619 |
| 2 | 80.519481 | 71.192053 |           | 66.9967   | 84.228188 |
| 3 | 69.078947 | 77.516779 | 67.656766 |           | 79.591837 |
| 4 | 82.333333 | 64.285714 | 81.208054 | 79.251701 |           |
'''

for model in ['qcnn', 'mera']:
  acc = np.zeros((5, 5))
  for i in range(5):
    for j in range(5):
      fp = Path(f'output/{model}_{i}-{j}/loss_history.json')
      if not fp.exists(): continue
      with open(fp, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
      acc[i, j] = data.get('max_valid_acc', 0.0)
  print(f'[{model}]')
  print(acc)
  print()
