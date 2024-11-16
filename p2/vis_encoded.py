#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/22 

# 查看编码后的图

import pickle as pkl
from argparse import ArgumentParser

from torch import Tensor
import matplotlib.pyplot as plt

from utils import reshape_norm_padding, get_fidelity
from utils import QCIFAR10Dataset       # keep for unpickle


def img_to_01(x:Tensor) -> Tensor:
  vmin, vmax = x.min(), x.max()
  x = (x - vmin) / (vmax - vmin)
  return x


def run(args):
  with open(args.fp, 'rb') as file:
    test_dataset = pkl.load(file)
    
  for x, y, z_ in test_dataset:
    vec_z = z_.flatten().real
    if args.F == 'std':
      vec_x = reshape_norm_padding(x).squeeze_(0)
      z = vec_z.reshape(-1, 32, 32)[:3, ...]

    im_x = img_to_01(x).permute([1, 2, 0]).numpy()
    im_z = img_to_01(z).permute([1, 2, 0]).numpy()
    fid = get_fidelity(vec_x, z_)

    plt.clf()
    plt.subplot(221) ; plt.imshow(im_x) ; plt.title('x')
    plt.subplot(222) ; plt.imshow(im_z) ; plt.title('z')
    plt.subplot(212)
    plt.plot(vec_x, 'b', label='vec_x')
    plt.plot(vec_z, 'r', label='vec_z')
    plt.legend()
    plt.suptitle(f'Fid: {fid}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-fp', default='./output/test_dataset.pkl', help='path to encode test_dataset.pkl')
  parser.add_argument('-F', default='std', choices=['std'], help='flatten method')
  args = parser.parse_args()

  #print('WARN: 如果可视化图像看起来不对，请确认 -F 参数设置正确')
  #print('大端序/hwc格式目前没实现合适的 inv 函数，所以可视化就是不对的 :(')

  run(args)
