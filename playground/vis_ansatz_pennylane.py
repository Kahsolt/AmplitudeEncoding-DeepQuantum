#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/12 

# 用 pennylane 实现一遍试试，我感觉 deepquantum 效率性能可能有问题 :(
# - 结论是 pennylane 比 deepquantum 还慢，并且 fid 没有提升

from typing import Union
from vis_ansatz import *

import os
os.environ['OMP_NUM_THREADS'] = '4'
import pennylane as qml
from pennylane import numpy as np   # 使用 hijack 过的 numpy
from pennylane.numpy import ndarray
from pennylane.optimize import AdamOptimizer

from utils import *

DEVICE = 'default.qubit'    # 'lightning.qubit'
INTERFACE = 'autograd'      # ['autograd', 'torch']
DIFF_METHOD = 'best'        # ['best', 'backprop', 'adjoint', 'parameter-shift', 'finite-diff']

Data = Union[ndarray, Tensor]


def get_fidelity(x:Data, y:Data, keep_grad:bool=False) -> Data:
  fid = (x * y).sum()**2
  return fid if keep_grad else fid.item()

def run_test(get_vqc:Callable[[], Tuple[qml.QNode, int]], lr:float=0.1, n_repeat:int=N_REPEAT, n_iter:int=N_ITER, data_gen:Callable=rand_state) -> Tuple[float, float]:
  set_seed()

  vqc, pcnt = get_vqc()
  nq = len(vqc.device.wires)

  ts_start = time()
  fids: list[float] = []
  pbar = tqdm(total=n_repeat)
  for _ in range(n_repeat):
    vqc, pcnt = get_vqc()
    target: Tensor = data_gen(nq)

    if INTERFACE == 'autograd':
      target = np.asarray(target.numpy())
      param = np.random.randn(pcnt, requires_grad=True)

      optim = AdamOptimizer(lr)

      def loss_fn(vqc, param:Data, target:Data) -> Data:
        loss = -get_fidelity(np.real(vqc(param)), target, keep_grad=True)
        if DEBUG_LOSS: print('loss:', loss._value.item())
        return loss
      for _ in range(n_iter):
        vqc, param, state = optim.step(loss_fn, vqc, param, target)

    elif INTERFACE == 'torch':
      param = torch.randn([pcnt], requires_grad=True)

      optim = Adam([param], lr=lr)

      for _ in range(n_iter):
        def closure():
          optim.zero_grad()
          loss = -get_fidelity(torch.real(vqc(param)), target, keep_grad=True)
          if loss.requires_grad:
            loss.backward()
          return loss.item()
        loss = optim.step(closure)
        if DEBUG_LOSS: print('loss:', loss)

    fid = get_fidelity(vqc(param).real, target)
    fids.append(fid)

    pbar.update()
    pbar.set_postfix({'pcnt': pcnt, 'fid': mean(fids)})
  pbar.close()
  ts_end = time()

  fid = mean(fids)
  ts = ts_end - ts_start
  print(f'[nq = {nq}] pcnt={pcnt}, fid={fid:.5f}, ts={ts:.3f}s')
  return pcnt, fid, ts


def vqc_submit_p1_CRY_RY(nq:int, n_rep:int=8):
  dev = qml.device(DEVICE, wires=nq)
  @qml.qnode(dev, interface=INTERFACE, diff_method=DIFF_METHOD)
  def get_vqc(param):
    '''' X - [cyclic(CRY) - RY] '''
    qml.X(wires=0)
    idx = 0
    for _ in range(n_rep):
      for q in range(nq):
        qml.CRY(param[idx], wires=[q, (q+1)%nq]) ; idx += 1
      for q in range(nq):
        qml.RY(param[idx], wires=q) ; idx += 1
    return qml.state()
  pcnt = 1 + n_rep * (nq * 2)
  return get_vqc, pcnt

def vqc_F2_all_wise_init(nq:int, n_rep:int=1):
  dev = qml.device(DEVICE, wires=nq)
  @qml.qnode(dev, interface=INTERFACE, diff_method=DIFF_METHOD)
  def get_vqc(param):
    ''' RY(init) - [pairwise(F2) - RY] '''
    idx = 0
    qml.RY(param[idx], wires=0) ; idx += 1
    for _ in range(n_rep):
      for i in range(nq-1):
        for j in range(i+1, nq):
          qml.CRY(param[idx], wires=[i, j]) ; idx += 1
          qml.CRY(param[idx], wires=[j, i]) ; idx += 1
      for q in range(nq):
        qml.RY(param[idx], wires=q) ; idx += 1
    return qml.state()
  pcnt = 1 + n_rep * (nq*(nq-1) + nq)
  return get_vqc, pcnt

def vqc_F1_all_wise_init(nq:int, n_rep:int=1):
  dev = qml.device(DEVICE, wires=nq)
  @qml.qnode(dev, interface=INTERFACE, diff_method=DIFF_METHOD)
  def get_vqc(param):
    ''' RY(init) - [pairwise(F2) - RY] '''
    idx = 0
    qml.RY(param[idx], wires=0) ; idx += 1
    for _ in range(n_rep):
      for i in range(nq-1):
        for j in range(i+1, nq):
          qml.CRY(param[idx], wires=[i, j]) ; idx += 1
      for q in range(nq):
        qml.RY(param[idx], wires=q) ; idx += 1
    return qml.state()
  pcnt = 1 + n_rep * (nq*(nq-1)//2 + nq)
  return get_vqc, pcnt

if not 'nq=6':
  # pcnt=73, fid=0.99894, ts=210.217s (autograd-best)
  run_test(partial(vqc_F2_all_wise_init, 6, 2), n_repeat=5)

if not 'nq=8':
  # pcnt=193, fid=0.98637, ts=590.125s (autograd-best)
  run_test(partial(vqc_F2_all_wise_init, 8, 3), lr=0.02, n_repeat=5)

if not 'nq=10':
  # pcnt=166, fid=0.80495, ts=535.165s (autograd-best)
  # pcnt=166, fid=0.77427, ts=355.004s (torch-backprop)
  run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=5, data_gen=rand_mnist_freq)
