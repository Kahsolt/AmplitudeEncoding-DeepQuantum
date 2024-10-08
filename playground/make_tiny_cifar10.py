#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/08 

import random
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

from utils import *

Y_LABELS = [0, 1, 2, 3, 4]
N_SAMPLE_PER_CLS = 100

set_seed()

transform = T.Compose([
  T.ToTensor(),
])

dataset = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
y_x = {}
for x, y in dataset:
  if y not in Y_LABELS: continue
  if y not in y_x: y_x[y] = []
  y_x[y].append(x)
for y, x in y_x.items():
  random.shuffle(x)
  y_x[y] = x[:N_SAMPLE_PER_CLS]
X = torch.cat([torch.stack(y_x[y]) for y in Y_LABELS])
print('[X]')
print(' shape:', X.shape)
print(' max:',   X.max())
print(' min:',   X.min())
print(' mean:',  X.mean())
print(' std:',   X.std())
torch.save(X, DATA_PATH / 'tiny_cifar10.pt')


transform_n = T.Compose([
  T.ToTensor(),
  T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

dataset_n = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_n)
y_x_n = {}
for x, y in dataset_n:
  if y not in Y_LABELS: continue
  if y not in y_x_n: y_x_n[y] = []
  y_x_n[y].append(x)
for y, x in y_x_n.items():
  random.shuffle(x)
  y_x_n[y] = x[:N_SAMPLE_PER_CLS]
X_n = torch.cat([torch.stack(y_x_n[y]) for y in Y_LABELS])
print('[X_n]')
print(' shape:', X_n.shape)
print(' max:',   X_n.max())
print(' min:',   X_n.min())
print(' mean:',  X_n.mean())
print(' std:',   X_n.std())
torch.save(X_n, DATA_PATH / 'tiny_cifar10_n.pt')
