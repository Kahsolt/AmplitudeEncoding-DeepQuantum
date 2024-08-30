#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/30 

# 查看所有样本各像素的值的和，用以确定最优的像素读取顺序

import seaborn as sns
import matplotlib.pyplot as plt

from utils import *

datatset = QMNISTDatasetIdea(label_list=[0,1,2,3,4], train=False)
cnt = torch.zeros([1, 28, 28], dtype=torch.int32)
for x, _, _ in tqdm(datatset):
  cnt += (denormalize(x) > 1/255).to(torch.int32)

Z = cnt[0].numpy()
sns.heatmap(Z, cmap='Reds', fmt='d', annot=True)
plt.suptitle('vis_sample_cntsum')
#plt.show()

if 'save stats':
  cnt_loc = []
  for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
      cnt_loc.append((Z[i, j], (i, j)))

  def cmp(it):
    # cnt 越大, (i,j) 越靠近中心
    cnt, (i, j) = it
    return (-cnt, np.sqrt((i-14)**2 + (j-14)**2))
  cnt_loc.sort(key=cmp)
  print(cnt_loc)

  loc_list = [loc for cnt, loc in cnt_loc]   # [(x, y)]
  np.save('./output/loc.npy', np.asarray(loc_list, dtype=np.uint8))
