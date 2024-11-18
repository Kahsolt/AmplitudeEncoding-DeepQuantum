#!/usr/bin/env python3
# Author: Armit
# Create Time: 周一 2024/11/18 

# 查看数据的纠缠程度 (即是否能被 SVD 分解成 MPS)
# 或者说，我们的 vqc 大概等价于拟合了 SVD 的前多少个分量
# | target fid | need svd k |
# |   0.99     |   10.400   |
# |   0.98     |    7.984   |
# |   0.97     |    6.716   |
# |   0.96     |    5.872   |
# |    vqc     |    5.330   |
# |   0.95     |    5.224   |

'''
按照比赛指定的 reshape_flatten 方式，向量对应的二维图像阵列 layout 为:
  |00...>  | R | R' |
  |01...>  | G | G' |
  |10...>  | B | B' |
  |11...>  | 0 | 0' |
即每个 channel 占据若干行，并且宽度被重复、高度被砍半了
'''

import torch.nn.functional as F
import matplotlib.pyplot as plt
from test import *

mean = lambda x: sum(x) / len(x)


OUTPUT_DIR = './output'
with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'rb') as file:
  test_dataset = pkl.load(file)

def minmax_norm(x):
  vmin, vmax = x.min(), x.max()
  return (x - vmin) / (vmax - vmin)


k_list = []
for x, _, z in test_dataset:
  # vqc 所面对的 64x64 的图 (reshape_flatten 所得)
  z = z.real
  im_vec = z.reshape(64, 64)

  # vqc 拟合的保真度
  vqc_fid = get_fidelity(z, reshape_norm_padding(x)).item()

  # 在 reshape_flatten 的图上计算 svd
  U, S, V = torch.svd(im_vec, some=False, compute_uv=True)
  for k in range(1, 32):
    S_copy = S.clone()
    S_copy[k:] = 0.0
    im_vec_svd = U @ torch.diag(S_copy) @ V.T
    z_svd = F.normalize(im_vec_svd.flatten(), dim=-1)
    svd_fid = get_fidelity(z, z_svd).item()
    if svd_fid > vqc_fid: break
  k_list.append(k)

  if not 'plot':
    print('k:', k, 'vqc_fid:', vqc_fid, 'svd_fid:', svd_fid)

    plt.subplot(221) ; plt.title('orig')     ; plt.imshow(minmax_norm(x.permute(1,2,0)))
    plt.subplot(222) ; plt.title('vqc')      ; plt.imshow(minmax_norm(z.reshape(-1, 32, 32)[:3, ...].permute(1,2,0)))
    plt.subplot(223) ; plt.title('flat')     ; plt.imshow(minmax_norm(im_vec))
    plt.subplot(224) ; plt.title('flat-svd') ; plt.imshow(minmax_norm(im_vec_svd))
    plt.tight_layout()
    plt.show()

print('mean(k):', mean(k_list))
