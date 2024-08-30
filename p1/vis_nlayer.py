#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/30 

# 查看 VQC 方法中不同的线路深度拟合出来的结果差异

from time import time

import deepquantum as dq
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from amp_enc_vqc import QMNISTDatasetDummy, device
from utils import *

N_ITER = 500


def get_model(n_layer:int, nq:int=10) -> dq.QubitCircuit:
  vqc = dq.QubitCircuit(nq)
  vqc.x(0)
  for _ in range(n_layer):
    for q in range(nq-1):
      g = dq.Ry(nqubit=nq, wires=(q+1)%nq, controls=q, condition=True, requires_grad=True)
      g.init_para([0])
      vqc.add(g)
      g = dq.Ry(nqubit=nq, wires=q, controls=(q+1)%nq, condition=True, requires_grad=True)
      g.init_para([0])
      vqc.add(g)
  return vqc


''' Data '''
dataset = QMNISTDatasetDummy(label_list=[0,1,2,3,4], train=False, per_cls_size=1)
for idx, (x, y, _) in enumerate(dataset):
  z = reshape_norm_padding(x.unsqueeze(0))
  x, y, z = x.to(device), y.to(device), z.to(device)
  break


''' Plots '''
fig = plt.figure(figsize=[11, 11])
axs: List[List[Axes]] = fig.subplots(5, 4)
axs[0][0].imshow(img_to_01(x).permute([1, 2, 0]).numpy())
axs[0][0].set_title('x')
axs[0][1].imshow(img_to_01(z.real.reshape(-1, 32, 32)).permute([1, 2, 0]).numpy())
axs[0][1].set_title('z')


''' Encode '''
for i_fig, n_layer in enumerate(range(1, 16+1)):
  print(f'[n_layer={n_layer}]')
  circ = get_model(n_layer).to(device)
  optimizer = optim.Adam(circ.parameters(), lr=0.02)
  s = time()
  for i in range(N_ITER):
    optimizer.zero_grad()
    state = circ().swapaxes(0, 1)     # [B=1, D=1024]
    loss = -get_fidelity(state, z)
    loss.backward()
    optimizer.step()
    if i % 50 == 0:
      print(f'[{i}] fid:', -loss.item())
  t = time()
  z_vqc = circ().detach().real.swapaxes(0, 1)
  fid = get_fidelity(z_vqc, z).item()

  ax = axs[i_fig // 4 + 1][i_fig % 4]
  ax.imshow(img_to_01(z_vqc.reshape(-1, 32, 32)).permute([1, 2, 0]).numpy())
  ax.set_title(f'[L={n_layer}] fid: {fid:.4f}, ts: {t - s:.2f}')


for ax_row in axs:
  for ax in ax_row:
    ax.axis('off')
plt.suptitle('Abaltion on n_layer')
plt.tight_layout()
plt.savefig('./output/vis_nlayer.png', dpi=800)
plt.show()
