#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/08 

# 查看 VQC 方法中不同的线路深度拟合出来的结果差异

from time import time
from typing import *

import numpy as np
import deepquantum as dq
import torch
import torch.nn.functional as F
from torch import Tensor, optim
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

device = 'cpu'
N_ITER = 500


def img_to_01(x:Tensor) -> Tensor:
  vmin, vmax = x.min(), x.max()
  x = (x - vmin) / (vmax - vmin)
  return x

def get_fidelity(state_pred:Tensor, state_true:Tensor) -> Tensor:
  state_pred = state_pred.view(-1, 4096).real
  state_true = state_true.view(-1, 4096).real
  fidelity = (state_pred * state_true).sum(-1)**2
  return fidelity.mean()

def get_gate_count(qc:dq.QubitCircuit) -> int:
  return len([op for op in qc.operators.modules() if isinstance(op, dq.operation.Operation)])

def vec_to_state(x:Tensor, nq:int) -> Tensor:
  if 'legacy':
    x = F.pad(x, (0, 2**nq - len(x)), mode='constant', value=0.0)
    x_n = F.normalize(x, p=2, dim=-1)
    x_o = x_n
  else:
    x_n = F.normalize(x, p=2, dim=-1)
    x = F.pad(x_n, (0, 2**nq - len(x_n)), mode='constant', value=0.0)
    x_o = x
  if not 'DEBUG_INPUT':
    plt.clf()
    plt.plot(x  .flatten().numpy())
    plt.plot(x_n.flatten().numpy())
    plt.show()
  return x_o

def reshape_norm_padding(x:Tensor, trim:int=0) -> Tensor:
  if trim: x[x <= trim / 255] = 0.0
  x = x.flatten()
  return vec_to_state(x, 12)


def get_model(n_layer:int, nq:int=12) -> dq.QubitCircuit:
  vqc = dq.QubitCircuit(nqubit=nq)
  g = dq.Ry(nqubit=nq, wires=0, requires_grad=True)   # only init wire 0
  g.init_para([np.pi/2])   # MAGIC: 2*arccos(sqrt(2/3)) = 1.2309594173407747
  vqc.add(g)
  for _ in range(n_layer):
    for i in range(nq-1):   # qubit order
      for j in range(i+1, nq):
        g = dq.Ry(nqubit=nq, wires=j, controls=i, requires_grad=True)
        g.init_para([0.0])
        vqc.add(g)
    for i in range(nq):
      g = dq.Ry(nqubit=nq, wires=i, requires_grad=True)
      g.init_para([0.0])
      vqc.add(g)
  return vqc


''' Data '''
TINY_CIFAR10 = torch.load('../playground/data/tiny_cifar10.pt')
x = TINY_CIFAR10[1].to(device)
z = reshape_norm_padding(x)
print('x.shape:', x.shape)
print('z.shape:', z.shape)


''' Plots '''
fig = plt.figure(figsize=[8, 8])
axs: List[List[Axes]] = fig.subplots(2, 3)
axs[0][0].imshow(img_to_01(x).permute([1, 2, 0]).numpy())
axs[0][0].set_title('x')
axs[0][1].imshow(img_to_01(z.real.reshape(-1, 32, 32))[:3, ...].permute([1, 2, 0]).numpy())
axs[0][1].set_title('z')


''' Encode '''
for i_fig, n_layer in enumerate(range(1, 3+1)):
  circ = get_model(n_layer).to(device)
  gcnt = get_gate_count(circ)
  print(f'[n_layer={n_layer} gcnt={gcnt}]')
  optimizer = optim.Adam(circ.parameters(), lr=0.2)
  s = time()
  for i in range(N_ITER):
    optimizer.zero_grad()
    state = circ().swapaxes(0, 1).real     # [B=1, D=1024]
    loss = -get_fidelity(state, z)
    loss.backward()
    optimizer.step()
    if i % 50 == 0:
      print(f'[{i}] fid:', -loss.item())
  t = time()
  z_vqc = circ().detach().real.swapaxes(0, 1)
  fid = get_fidelity(z_vqc, z).item()

  ax = axs[i_fig // 4 + 1][i_fig % 4]
  ax.imshow(img_to_01(z_vqc.reshape(-1, 32, 32))[:3, ...].permute([1, 2, 0]).numpy())
  ax.set_title(f'[L={n_layer}] fid: {fid:.4f}, ts: {t - s:.2f}')


for ax_row in axs:
  for ax in ax_row:
    ax.axis('off')
plt.suptitle('Abaltion on n_layer')
plt.tight_layout()
plt.savefig('./img/vis_nlayer.png', dpi=600)
plt.show()
