#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/13 

# 查看分数等高线

from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

BASE_PATH = Path(__file__).parent.parent
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)


def get_score(fid:float, gcnt:int) -> float:
  return 2 * fid + (1 - (gcnt / 1000))


x = np.linspace(0.5, 1.0, 101)[:-1]   # fid
y = np.linspace(0, 300, 101) [:-1]   # gcnt

sc_map = np.zeros([len(x), len(y)], dtype=np.float32)
print('sc_map.shape:', sc_map.shape)

for i in range(len(x)):
  for j in range(len(y)):
    sc_map[i, j] = get_score(x[i], y[j])

plt.clf()
plt.figure(figsize=(8, 8))
sns.heatmap(sc_map, cbar=True, fmt='.2f')
plt.gca().invert_yaxis()
plt.ylabel('Fidelity')
plt.yticks([i * 10 for i in range(10)], [str(round(x[i * 10], 2)) for i in range(10)])
plt.xlabel('Gate Count')
plt.xticks([i * 10 for i in range(10)], [str(int(y[i * 10])) for i in range(10)])
plt.tight_layout()
fp = IMG_PATH / 'score_map.png'
plt.savefig(fp, dpi=400)
print(f'savefig: {fp}')
