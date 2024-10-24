#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/22 

# 查看编码后的图

import pickle as pkl
from argparse import ArgumentParser

import torch
from torch import Tensor
import matplotlib.pyplot as plt

from utils import reshape_norm_padding, qam_reshape_norm_padding, qam_index_generator
from utils import QCIFAR10Dataset       # keep for unpickle


def img_to_01(x:Tensor) -> Tensor:
  vmin, vmax = x.min(), x.max()
  x = (x - vmin) / (vmax - vmin)
  return x

def get_fidelity(state_pred:Tensor, state_true:Tensor) -> Tensor:
  state_pred = state_pred.view(-1, 4096).real
  state_true = state_true.view(-1, 4096).real
  fidelity = (state_pred * state_true).sum(-1)**2
  return fidelity.mean()

def inv_qam_reshape_norm_padding(z:Tensor) -> Tensor:
  assert len(z) == 2**12
  cvs = torch.empty([64, 64], dtype=z.dtype, device=z.device)
  for i, (x, y) in enumerate(qam_index_generator()):
    cvs[x, y] = z[i]
  B = cvs[:32, :32]
  O = cvs[:32, 32:]
  R = cvs[32:, :32]
  G = cvs[32:, 32:]
  print('>> Err(O):', O.mean().item())
  return torch.stack([R, G, B], dim=0)   # [C, H, W]


def run(args):
  with open(args.fp, 'rb') as file:
    test_dataset = pkl.load(file)
    
  for x, y, z_ in test_dataset:
    vec_z = z_.flatten().real
    if args.F == 'std':
      vec_x = reshape_norm_padding(x, use_hijack=False)
      z = vec_z.reshape(-1, 32, 32)[:3, ...]
    elif args.F == 'qam':
      vec_x = qam_reshape_norm_padding(x)
      z = inv_qam_reshape_norm_padding(vec_z)

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
  parser.add_argument('-F', default='std', choices=['std', 'qam'], help='flatten method')
  args = parser.parse_args()

  run(args)
