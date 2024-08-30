#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/30 

# 跟踪查看训练时的梯度和参数变化

import deepquantum as dq
import matplotlib.pyplot as plt

from amp_enc_vqc import QMNISTDatasetDummy, device
from utils import *

N_ITER = 500
N_LAYER = 14


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


''' Encode '''
circ = get_model(N_LAYER).to(device)
optimizer = optim.Adam(circ.parameters(), lr=0.02)

params_list = []
grads_list = []
grad_mean_list = []
for i in range(N_ITER):
  optimizer.zero_grad()
  state = circ().swapaxes(0, 1)     # [B=1, D=1024]
  loss = -get_fidelity(state, z)
  loss.backward()

  if not 'fix grads':   # suppress small perturbation
    for op in circ.operators:
      if isinstance(op, dq.PauliX): continue
      grad = op.theta.grad
      if grad.abs().item() < 1e-5:
        grad.zero_()

  optimizer.step()

  if 'chk grads & params':
    grads = [op.theta.grad.item() for op in circ.operators if not isinstance(op, dq.PauliX)]
    for j, g in enumerate(grads):
      if len(grads_list) == 0:
        grads_list = [[] for _ in range(len(grads))]
      grads_list[j].append(g)
    grad_mean_list.append(mean(grads))

    params = [op.theta.item() for op in circ.operators if not isinstance(op, dq.PauliX)]
    for j, p in enumerate(params):
      if len(params_list) == 0:
        params_list = [[] for _ in range(len(params))]
      params_list[j].append(p)

  if i % 20 == 0:
    print(f'[{i}] fid:', -loss.item())
z_vqc = circ().detach().real.swapaxes(0, 1)
fid = get_fidelity(z_vqc, z).item()
print('Fidelity:', fid)


''' Plot '''
plt.clf()
plt.plot(grad_mean_list)
plt.suptitle('grad-mean')
plt.tight_layout()
plt.savefig('./output/grad-mean.png', dpi=400)
plt.show()

plt.clf()
for grads in grads_list:
  plt.plot(grads)
plt.suptitle('grad')
plt.tight_layout()
plt.savefig('./output/grad.png', dpi=600)
plt.show()

plt.clf()
for params in params_list:
  plt.plot(params)
plt.suptitle('param')
plt.tight_layout()
plt.savefig('./output/param.png', dpi=600)
plt.show()

ops = circ.operators
params = [op.theta.item() for op in ops if not isinstance(op, dq.PauliX)]

breakpoint()
