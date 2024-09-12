#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/30 

# 查看所有样本各像素的值的和，用以确定最优的像素读取顺序

import seaborn as sns
import matplotlib.pyplot as plt

from utils import *

datatset = QMNISTDatasetIdea(label_list=[0,1,2,3,4], train=False)
cnt = torch.zeros([1, 28, 28], dtype=torch.int32)
avg = torch.zeros([1, 28, 28], dtype=torch.float32)
nonzero_pixel_cnt = []
for x, y, z_func in tqdm(datatset):
  x = denormalize(x).clamp(0.0, 1.0)
  nonzero = (x > 1/255).to(torch.int32)
  nonzero_pixel_cnt.append(nonzero.sum())
  cnt += nonzero
  avg += x.to(torch.float32)
avg = (avg / len(datatset) * 255).ceil().to(torch.uint8)
avg[avg==1] = 0   # ignore small value

map_cnt = cnt[0].numpy()
map_avg = avg[0].numpy()

plt.clf()
plt.figure(figsize=(14, 10))
sns.heatmap(map_cnt, cmap='Reds', fmt='d', annot=True)
plt.suptitle('vis_sample_cnt')
plt.tight_layout()
plt.savefig('./img/vis_sample_cnt.png', dpi=400)

plt.clf()
plt.figure(figsize=(14, 10))
sns.heatmap(map_avg, cmap='Reds', fmt='d', annot=True)
plt.suptitle('vis_sample_avg')
plt.tight_layout()
plt.savefig('./img/vis_sample_avg.png', dpi=400)

avg_nzp = mean(nonzero_pixel_cnt)
# >> non-zero pixel: 148.46974182128906 (ratio: 0.1893746703863144)
print(f'>> non-zero pixel: {avg_nzp} (ratio: {avg_nzp / 28**2})')
avg_empty = (avg == 0).sum()
# >> empty: 313 (ratio: 0.39923468232154846)
print(f'>> empty: {avg_empty} (ratio: {avg_empty / avg.numel()})')

Z = map_avg    # 以值的avg为重要性
cnt_loc = []
for i in range(Z.shape[0]):
  for j in range(Z.shape[1]):
    cnt_loc.append((Z[i, j], (i, j)))

def cmp(it):  # 优先: cnt 越大, (i,j) 越靠近中心
  cnt, (i, j) = it
  return (-cnt, np.sqrt((i-14)**2 + (j-14)**2))
cnt_loc.sort(key=cmp)

loc_list = [loc for cnt, loc in cnt_loc if cnt]   # [(x, y)]
# len(loc_list): 471
print('len(loc_list):', len(loc_list))
np.save('./img/loc.npy', np.asarray(loc_list, dtype=np.uint8))
