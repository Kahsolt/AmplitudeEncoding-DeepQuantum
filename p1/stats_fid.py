#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/27 

# 查看训练完的 vqc 的保真度分布直方图

import time
import pickle
import matplotlib.pyplot as plt

from utils import reshape_norm_padding, get_fidelity
from utils import QMNISTDataset, QMNISTDatasetIdea, DataHolder      # keep for unpickle

t0 = time.time()
with open('./output/test_dataset.pkl', 'rb') as file:
  dataset = pickle.load(file)
t1 = time.time()
print(f'>> load pickle done ({t1 - t0:.3f}s)')

t0 = time.time()
fid_list = []
for x, _, z_func in dataset:
  z_hat = z_func()
  z = reshape_norm_padding(x.unsqueeze(0))
  fid = get_fidelity(z_hat, z).item()
  print('fid:', fid)
  fid_list.append(fid)
t1 = time.time()
print(f'>> calc fidelity done ({t1 - t0:.3f}s)')

plt.hist(fid_list, bins=50)
plt.savefig('./img/stats_fid.png', dpi=400)
