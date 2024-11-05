#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/10 

# 探究 VQC 线路拟合任意分布的普适性

'''
- 完全随机数据下 nqubits 扩展瓶颈: 比特数+1 => 门数量翻倍
  - nq= 6, gcnt=  73, fid=0.99918
  - nq= 7, gcnt= 141, fid=0.99970
  - nq= 8, gcnt= 257, fid=0.99950
  - nq= 9, gcnt= 487, fid=0.99862 (预计需要511门)
  - nq=10, gcnt=1001, fid=0.99970 (预计需要1023门)
- 稀疏向量对拟合更友好
  - normalize 会导致 MNIST 不再稀疏，更难拟合
  - 稀疏值聚集而连续比随机而零散更容易拟合
  - CIFAR10比MNSIT更容易拟合 (因为它含有大色块??!)
- 关于拟合能力: 线路结构决定下限，门数量决定上限
'''

import os
from time import time
from functools import partial
from typing import Tuple, Callable, Generator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
import deepquantum as dq
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import *

DEBUG_INPUT = os.getenv('DEBUG_INPUT')
DEBUG_LOSS = os.getenv('DEBUG_LOSS')

N_ITER = 500
N_REPEAT = 30

try:
  TINY_MNIST = torch.load(DATA_PATH / 'tiny_mnist.pt')
  TINY_CIFAR10 = torch.load(DATA_PATH / 'tiny_cifar10.pt')
  loc = np.load('../p1/img/loc.npy')
except:
  pass


def rand_state(nq:int) -> Tensor:
  v = torch.rand(2**nq) - 0.5
  x = v / torch.linalg.norm(v)
  if DEBUG_INPUT: plt.plot(x.numpy()) ; plt.show()
  return x

def rand_first_n(nlen:int, nq:int) -> Tensor:
  assert nlen <= 2**nq
  v = torch.rand(2**nq) - 0.5
  v[nlen:] = 0.0
  x = v / torch.linalg.norm(v)
  if DEBUG_INPUT: plt.plot(x.numpy()) ; plt.show()
  return x

def vec_to_state(x:Tensor, nq:int) -> Tensor:
  # FIXME: 对于恒正数据，下述两种操作顺序结果上等价；对于预先已均值-方差规范化的数据，建议此处先 norm 再 pad 以避免浮点误差 :(
  if 'legacy':
    x = F.pad(x, (0, 2**nq - len(x)), mode='constant', value=0.0)
    x_n = F.normalize(x, p=2, dim=-1)
    x_o = x_n
  else:
    x_n = F.normalize(x, p=2, dim=-1)
    x = F.pad(x_n, (0, 2**nq - len(x_n)), mode='constant', value=0.0)
    x_o = x
  if DEBUG_INPUT:
    plt.clf()
    plt.plot(x  .flatten().numpy())
    plt.plot(x_n.flatten().numpy())
    plt.show()
  return x_o

def rand_mnist(nq:int, trim:int=0) -> Tensor:
  assert nq >= 10
  idx = torch.randint(low=0, high=len(TINY_MNIST), size=[1]).item()
  x = TINY_MNIST[idx].flatten()
  if trim: x[x <= trim / 255] = 0.0
  return vec_to_state(x, nq)

def snake_index_generator(N:int=28) -> Generator[Tuple[int, int], int, None]:
  dir = 0     # 0: →, 1: ↓, 2: ←, 3: ↑
  i, j = 0, -1
  # the first stage only repeats once
  steps_stage = N
  steps_stage_repeat = 0
  steps = steps_stage
  # other stages will repeat twice
  while steps_stage > 0:
    if   dir == 0: j += 1
    elif dir == 1: i += 1
    elif dir == 2: j -= 1
    elif dir == 3: i -= 1
    yield i, j
    steps -= 1
    # next repeat or stage?
    if steps == 0:
      if steps_stage_repeat == 1:
        steps_stage_repeat -= 1
      else:
        steps_stage_repeat = 1
        steps_stage -= 1
      steps = steps_stage
      dir = (dir + 1) % 4

def rand_mnist_snake(nq:int, rev:bool=False, trim:int=0) -> Tensor:
  assert nq >= 10
  idx = torch.randint(low=0, high=len(TINY_MNIST), size=[1]).item()
  x = TINY_MNIST[idx].squeeze(dim=0)  # [H, W]
  if trim: x[x <= trim / 255] = 0.0
  x = torch.tensor([x[i, j] for i, j in snake_index_generator()])
  if rev: x = x.flip(-1)    # re-roder center to border
  return vec_to_state(x, nq)

def rand_mnist_freq(nq:int) -> Tensor:
  assert nq >= 10
  idx = torch.randint(low=0, high=len(TINY_MNIST), size=[1]).item()
  x = TINY_MNIST[idx].squeeze(dim=0)  # [H, W]
  x = torch.tensor([x[i, j] for i, j in loc])
  return vec_to_state(x, nq)

def rand_mock_sparse_mnist(nq:int) -> Tensor:
  assert nq >= 10
  v = torch.rand(2**nq) - 0.5
  m = torch.zeros_like(v, dtype=torch.float32)
  # pickle 150 random elem from the first 784 to set non-zero
  idx = torch.randperm(784)[:150]
  for i in idx: m[i] = 1.0
  v *= m
  x = v / torch.linalg.norm(v)
  if DEBUG_INPUT: plt.plot(x.numpy()) ; plt.show()
  return x

rand_mnist_snake_rev      = lambda nq:       rand_mnist_snake(nq, rev=True)
rand_mnist_snake_rev_trim = lambda trim, nq: rand_mnist_snake(nq, rev=True, trim=trim)

def rand_cifar10(nq:int, order:str='CHW', trim:int=0) -> Tensor:
  assert order in ['CHW', 'HWC']
  assert nq >= 12
  idx = torch.randint(low=0, high=len(TINY_CIFAR10), size=[1]).item()
  x = TINY_CIFAR10[idx]
  if trim: x[x <= trim / 255] = 0.0
  if order == 'HWC':
    x = x.permute((1, 2, 0))
  x = x.flatten()
  return vec_to_state(x, nq)

def rand_cifar10_snake(nq:int, rev:bool=False, trim:int=0) -> Tensor:
  assert nq >= 12
  idx = torch.randint(low=0, high=len(TINY_CIFAR10), size=[1]).item()
  x = TINY_CIFAR10[idx]  # [C, H, W]
  if trim: x[x <= trim / 255] = 0.0
  x = torch.stack([x[:, i, j] for i, j in snake_index_generator(x.shape[1])], dim=-1)
  if rev: x = x.flip(-1)    # re-roder center to border
  x = x.flatten()
  return vec_to_state(x, nq)

def rand_mock_cifar10(nq:int) -> Tensor:
  assert nq >= 12
  # dense vec of first 3*32*32=3072 elems
  return rand_first_n(3072, nq)

rand_cifar10_HWC       = lambda nq, trim=0: rand_cifar10      (nq, order='HWC', trim=trim)
rand_cifar10_snake_rev = lambda nq, trim=0: rand_cifar10_snake(nq, rev=True,    trim=trim)

def get_fidelity(x:Tensor, y:Tensor) -> Tensor:
  return (x * y).sum()**2

def run_test(get_vqc:Callable[[], dq.QubitCircuit], lr:float=0.01, n_repeat:int=N_REPEAT, n_iter:int=N_ITER, data_gen:Callable=rand_state) -> Tuple[float, float]:
  set_seed()

  gcnt = get_gate_count(get_vqc())

  ts_start = time()
  fids: list[float] = []
  pbar = tqdm(total=n_repeat)
  for _ in range(n_repeat):
    vqc = get_vqc()
    tgt = data_gen(vqc.nqubit)

    optim = Adam(vqc.parameters(), lr=lr)
    for _ in range(n_iter):
      optim.zero_grad()
      state = vqc().squeeze().real
      loss = -get_fidelity(state, tgt)
      loss.backward()
      optim.step()

      if DEBUG_LOSS:
        print('loss:', loss.item())

    fid = get_fidelity(vqc().squeeze().real, tgt).item()
    fids.append(fid)

    pbar.update()
    pbar.set_postfix({'gcnt': gcnt, 'fid': mean(fids)})
  pbar.close()
  ts_end = time()

  fid = mean(fids)
  ts = ts_end - ts_start
  print(f'[nq = {vqc.nqubit}] gcnt={gcnt}, fid={fid:.5f}, ts={ts:.3f}s')
  return gcnt, fid, ts


''' The final submit ansatz for p1(MNIST) '''

def vqc_submit_p1(nq:int, n_rep:int=14, init='zeros'):
  vqc = dq.QubitCircuit(nqubit=nq)
  vqc.x(0)
  for _ in range(n_rep):
    for q in range(nq-1):
      if init == 'zeros':
        g = dq.Ry(nqubit=nq, wires=(q+1)%nq, controls=q, requires_grad=True)
        g.init_para([0])
        vqc.add(g)
        g = dq.Ry(nqubit=nq, wires=q, controls=(q+1)%nq, requires_grad=True)
        g.init_para([0])
        vqc.add(g)
      elif init == 'rand':   # 随机初始化打破对称性，不然loss可能不降
        vqc.ry(wires=(q+1)%nq, controls=q)
        vqc.ry(wires=q, controls=(q+1)%nq)
      else: raise ValueError
  return vqc

def vqc_submit_p1_CRY(nq:int, n_rep:int=14):
  '''' X - [cyclic(CRY)] '''
  vqc = dq.QubitCircuit(nqubit=nq)
  vqc.x(0)
  for _ in range(n_rep):
    for q in range(nq):
      vqc.ry(wires=(q+1)%nq, controls=q)
  return vqc

def vqc_submit_p1_CRY_RY(nq:int, n_rep:int=8, entgl_rule:str='linear', ladder:str='seq'):
  '''' X - [entgl_rule(CRY) - RY] '''
  if   entgl_rule == 'linear': offset = 1
  elif entgl_rule == 'cyclic': offset = 0
  else: raise ValueError
  vqc = dq.QubitCircuit(nqubit=nq)
  vqc.x(0)
  for i in range(n_rep):
    is_even = i % 2 == 0
    if ladder == 'seq':
      for q in range(nq-offset):
        vqc.ry(wires=(q+1)%nq, controls=q)
    elif ladder == 'mirror':
      for q in (list if is_even else reversed)(range(nq-offset)):
        vqc.ry(wires=(q+1)%nq, controls=q)
    else: raise ValueError
    for q in range(nq):
      vqc.ry(wires=q)
  return vqc

if not 'nq=10':
  # gcnt=253, fid=0.61323, ts=525.098s
  run_test(partial(vqc_submit_p1, 10, 14), lr=0.02, n_repeat=10)
  # gcnt=289, fid=0.65839, ts=69.527s
  run_test(partial(vqc_submit_p1, 10, 16), lr=0.02, n_repeat=1)

  # gcnt=253, fid=0.86467, ts=169.015s
  run_test(partial(vqc_submit_p1, 10, 14, 'rand'), lr=0.02, n_repeat=3, data_gen=rand_mnist)
  # gcnt=253, fid=0.91658, ts=166.272s
  run_test(partial(vqc_submit_p1, 10, 14, 'rand'), lr=0.02, n_repeat=3, data_gen=rand_mnist_snake_rev)
  # gcnt=253, fid=0.93580, ts=166.227s
  run_test(partial(vqc_submit_p1, 10, 14, 'rand'), lr=0.02, n_repeat=3, data_gen=rand_mnist_freq)

  # gcnt=161, fid=0.66969, ts=113.731s
  run_test(partial(vqc_submit_p1_CRY, 10, 16), lr=0.02, n_repeat=3, data_gen=rand_mnist_freq)
  # gcnt=161, fid=0.70001, ts=116.396s
  run_test(partial(vqc_submit_p1_CRY, 10, 16), lr=0.1, n_repeat=3, data_gen=rand_mnist_freq)
  # gcnt=153, fid=0.80350, ts=85.992s
  run_test(partial(vqc_submit_p1_CRY_RY, 10, 8, 'linear'), lr=0.1, n_repeat=3, data_gen=rand_mnist_freq)
  # gcnt=153, fid=0.81822, ts=85.038s
  run_test(partial(vqc_submit_p1_CRY_RY, 10, 8, 'linear'), lr=0.2, n_repeat=3, data_gen=rand_mnist_freq)
  # gcnt=153, fid=0.78644, ts=91.076s
  run_test(partial(vqc_submit_p1_CRY_RY, 10, 8, 'linear', 'mirror'), lr=0.1, n_repeat=3, data_gen=rand_mnist_freq)
  # gcnt=161, fid=0.82088, ts=93.563s
  run_test(partial(vqc_submit_p1_CRY_RY, 10, 8, 'cyclic'), lr=0.1, n_repeat=3, data_gen=rand_mnist_freq)
  # gcnt=161, fid=0.82067, ts=91.592s
  run_test(partial(vqc_submit_p1_CRY_RY, 10, 8, 'cyclic'), lr=0.2, n_repeat=3, data_gen=rand_mnist_freq)
  # gcnt=161, fid=0.80582, ts=92.425s
  run_test(partial(vqc_submit_p1_CRY_RY, 10, 8, 'cyclic', 'mirror'), lr=0.1, n_repeat=3, data_gen=rand_mnist_freq)


''' The evolutional story for vqc_F2 ansatz series '''

def vqc_F2_all(nq:int, n_rep:int=1, wise_init:bool=False):
  ''' RY - [pairwise(F2) - RY] '''
  vqc = dq.QubitCircuit(nqubit=nq)
  if wise_init:
    vqc.ry(wires=0)   # only init wire 0
  else:
    for i in range(nq):
      vqc.ry(wires=i)
  for _ in range(n_rep):
    for i in range(nq-1):   # qubit order
      for j in range(i+1, nq):
        vqc.ry(wires=j, controls=i)
        vqc.ry(wires=i, controls=j)
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

vqc_F2_all_wise_init = lambda nq, n_rep: vqc_F2_all(nq, n_rep, wise_init=True)

def vqc_F2_all_gap_order_wise_init(nq:int, n_rep:int=1):
  ''' RY(single init) - [pairwise(F2, gap_order) - RY] '''
  vqc = dq.QubitCircuit(nqubit=nq)
  vqc.ry(wires=0)   # only init wire 0
  for _ in range(n_rep):
    for gap in range(1, nq-1):   # gap order
      for i in range(nq-gap):
        vqc.ry(wires=i+gap, controls=i)
        vqc.ry(wires=i, controls=i+gap)
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

def vqc_F2_all_no_inter_RY_wise_init(nq:int, n_rep:int=1):
  ''' RY(single init) - [pairwise(F2)] '''
  vqc = dq.QubitCircuit(nqubit=nq)
  vqc.ry(wires=0)   # only init wire 0
  for _ in range(n_rep):
    for i in range(nq-1):   # qubit order
      for j in range(i+1, nq):
        vqc.ry(wires=j, controls=i)
        vqc.ry(wires=i, controls=j)
  return vqc

def vqc_exF2(nq:int, n_rep:int=1, entgl_rule:str='linear', end_RY:bool=True):
  ''' entgl_rule(ry-cry-rcry) - (RY) '''
  vqc = dq.QubitCircuit(nqubit=nq)
  if   entgl_rule == 'linear': offset = 1
  elif entgl_rule == 'cyclic': offset = 0
  for _ in range(n_rep):
    if entgl_rule == 'all':
      for i in range(nq-1):
        for j in range(i+1, nq):
          vqc.ry(wires=i)
          vqc.ry(wires=j, controls=i)
          vqc.ry(wires=i, controls=j)
    else:
      for i in range(nq-offset):
        j = (i+1)%nq
        vqc.ry(wires=i)
        vqc.ry(wires=j, controls=i)
        vqc.ry(wires=i, controls=j)
  if end_RY:
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

def vqc_F1_all_wise_init(nq:int, n_rep:int=1):
  ''' RY(single init) - [pairwise(F1) - RY] '''
  vqc = dq.QubitCircuit(nqubit=nq)
  vqc.ry(wires=0)   # only init wire 0
  for _ in range(n_rep):
    for i in range(nq-1):   # qubit order
      for j in range(i+1, nq):
        vqc.ry(wires=j, controls=i)
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

def vqc_F1_all_wise_init_0(nq:int, n_rep:int=1):
  ''' RY(single init) - [pairwise(F1) - RY], param zero init '''
  vqc = dq.QubitCircuit(nqubit=nq)
  g = dq.Ry(nqubit=nq, wires=0, requires_grad=True)   # only init wire 0
  g.init_para([np.pi/2])   # MAGIC: 2*arccos(sqrt(2/3)) = 1.2309594173407747
  vqc.add(g)
  for _ in range(n_rep):
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

def vqc_F1_all_RX_0(nq:int, n_rep:int=1):
  ''' RY - [pairwise(F1) - RX], param zero init; aka. RealAmplitudes, see https://github.com/Qiskit/qiskit/blob/main/qiskit/circuit/library/n_local/real_amplitudes.py '''
  vqc = dq.QubitCircuit(nqubit=nq)
  for i in range(nq):
    g = dq.Ry(nqubit=nq, wires=i, requires_grad=True)
    g.init_para([0.0])
    vqc.add(g)
  for _ in range(n_rep):
    for i in range(nq-1):   # qubit order
      for j in range(i+1, nq):
        g = dq.Rx(nqubit=nq, wires=j, controls=i, requires_grad=True)
        g.init_para([0.0])
        vqc.add(g)
    for i in range(nq):
      g = dq.Ry(nqubit=nq, wires=i, requires_grad=True)
      g.init_para([0.0])
      vqc.add(g)
  return vqc

def vqc_F1_gap_order_wise_init(nq:int, n_rep:int=1):
  ''' RY(single init) - [pairwise(F1, gap_order) - RY] '''
  vqc = dq.QubitCircuit(nqubit=nq)
  vqc.ry(wires=0)   # only init wire 0
  for _ in range(n_rep):
    for gap in range(1, nq-1):   # gap order
      for i in range(nq-gap):
        vqc.ry(wires=i+gap, controls=i)
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

def vqc_F1_gap_order_wise_init_0(nq:int, n_rep:int=1):
  ''' RY(single init) - [pairwise(F1, gap_order) - RY], param zero init '''
  vqc = dq.QubitCircuit(nqubit=nq)
  g = dq.Ry(wires=0, nqubit=nq, requires_grad=True)   # only init wire 0
  g.init_para([np.pi/2])   # MAGIC: 2*arccos(sqrt(2/3)) = 1.2309594173407747
  vqc.add(g)
  for _ in range(n_rep):
    for gap in range(1, nq-1):   # gap order
      for i in range(nq-gap):
        g = dq.Ry(wires=i+gap, controls=i, nqubit=nq, requires_grad=True)
        g.init_para([0.0])
        vqc.add(g)
    for i in range(nq):
      g = dq.Ry(wires=i, nqubit=nq, requires_grad=True)
      g.init_para([0.0])
      vqc.add(g)
  return vqc

def vqc_F1_s_all_wise_init(nq:int, n_rep:int=1):
  ''' RY(single init) - [directed-pairwise(F2) - RY] '''
  vqc = dq.QubitCircuit(nqubit=nq)
  vqc.ry(wires=0)   # only init wire 0
  for _ in range(n_rep):
    for i in range(nq):
      for j in range(nq):
        if i == j: continue
        vqc.ry(wires=j, controls=i)
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

def vqc_F1_HEA_wise_init(nq:int, n_rep:int=1, entgl:str='CRY', entgl_rule:str='linear'):
  ''' RY(single init) - [linear(F1) - RY] '''
  if   entgl_rule == 'linear': offset = 1
  elif entgl_rule == 'cyclic': offset = 0
  else: raise ValueError
  vqc = dq.QubitCircuit(nqubit=nq)
  vqc.ry(wires=0)   # only init wire 0
  for _ in range(n_rep):
    for i in range(nq-offset):
      if   entgl == 'CRY':  vqc.ry(wires=(i+1)%nq, controls=i)
      elif entgl == 'CRX':  vqc.rx(wires=(i+1)%nq, controls=i)
      elif entgl == 'CNOT': vqc.x (wires=(i+1)%nq, controls=i)
      else: raise ValueError(entgl)
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

def vqc_exF1(nq:int, n_rep:int=1, entgl_rule:str='linear', end_RY:bool=True):
  ''' entgl_rule(ry-cry) - (RY) '''
  vqc = dq.QubitCircuit(nqubit=nq)
  if   entgl_rule == 'linear': offset = 1
  elif entgl_rule == 'cyclic': offset = 0
  for _ in range(n_rep):
    if entgl_rule == 'all':
      for i in range(nq-1):
        for j in range(i+1, nq):
          vqc.ry(wires=i)
          vqc.ry(wires=j, controls=i)
    else:
      for i in range(nq-offset):
        j = (i+1)%nq
        vqc.ry(wires=i)
        vqc.ry(wires=j, controls=i)
  if end_RY:
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

def vqc_HEA(nq:int, n_rep:int=1, entgl_rule:str='linear'):
  ''' HEA(RY, CNOT) '''
  if   entgl_rule == 'linear': offset = 1
  elif entgl_rule == 'cyclic': offset = 0
  else: raise ValueError
  vqc = dq.QubitCircuit(nqubit=nq)
  for _ in range(n_rep):
    for i in range(nq):
      vqc.ry(wires=i)
    for q in range(nq-offset):
      vqc.x(wires=(q+1)%nq, controls=q)
  for i in range(nq):
    vqc.ry(wires=i)
  return vqc

def vqc_F2_flat(nq:int, n_rep:int=1):
  ''' RY(init) - flat(F2) '''
  vqc = dq.QubitCircuit(nqubit=nq)
  vqc.ry(wires=0)
  for _ in range(n_rep):
    for q in range(nq):
      p = (q + 1) % nq
      vqc.ry(wires=p, controls=q)
      vqc.ry(wires=q, controls=p)
  return vqc

def vqc_F2_brick(nq:int, n_rep:int=1):
  ''' brick(F2) '''
  vqc = dq.QubitCircuit(nqubit=nq)
  for i in range(nq):
    vqc.ry(wires=i)
  for i in range(n_rep):
    is_even = i % 2 == 0
    for q in range(nq) if is_even else range(1, nq-1):
      p = (q + 1) % nq
      vqc.ry(wires=p, controls=q)
      vqc.ry(wires=q, controls=p)
  return vqc

def vqc_F2_mera(nq:int, n_rep:int=1):
  ''' RY - [enc-RY-dec-RY] '''
  mid_wires = [nq//2] if nq % 2 == 1 else [nq//2, nq//2+1]
  vqc = dq.QubitCircuit(nqubit=nq)
  for i in range(nq):
    vqc.ry(wires=i)
  for _ in range(n_rep):
    # down (->10-8-6-4-2)
    for offset in range(nq//2):
      for q in range(offset, nq-1-offset, 2):
        vqc.ry(wires=q+1, controls=q)
        vqc.ry(wires=q, controls=q+1)
    for i in mid_wires:
      vqc.ry(wires=i)
    # up (->4-6-8-10)
    for offset in range(nq//2-1, 0, -1):
      for q in range(offset, nq-1-offset, 2):
        vqc.ry(wires=q+1, controls=q)
        vqc.ry(wires=q, controls=q+1)
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

def vqc_F2_distro(nq:int, n_rep:int=1):
  ''' RY - [dec-RY-enc-RY] - dec '''
  vqc = dq.QubitCircuit(nqubit=nq)
  mid_wires = [nq//2] if nq % 2 == 1 else [nq//2, nq//2+1]
  pivot = nq//2-1
  vqc.ry(wires=pivot)
  for i in range(n_rep):
    # up (->2-4-6-8-10)
    for offset in range(nq//2-1, -1, -1):
      for q in range(offset, nq-1-offset, 2):
        vqc.ry(q+1, controls=q)
        vqc.ry(q, controls=q+1)
    for i in range(nq):
      vqc.ry(wires=i)
    # down (->8-6-4-2)
    for offset in range(1, nq//2):
      for q in range(offset, nq-1-offset, 2):
        vqc.ry(q+1, controls=q)
        vqc.ry(q, controls=q+1)
    for i in mid_wires:
      vqc.rylayer(wires=i)
  # dec
  for offset in range(nq//2-1, -1, -1):
    for q in range(offset, nq-1-offset, 2):
      vqc.ry(q+1, controls=q)
      vqc.ry(q, controls=q+1)
  return vqc

def vqc_BS(nq:int, n_rep:int=1, wise_init:bool=False):
  ''' RY - [pairwise(BS) - RY] '''
  vqc = dq.QubitCircuit(nqubit=nq)
  if wise_init:
    vqc.ry(wires=0)   # only init wire 0
  else:
    for i in range(nq):
      vqc.ry(wires=i)
  for _ in range(n_rep):
    for i in range(nq-1):
      for j in range(i+1, nq):
        vqc.rbs(wires=[i, j])
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

vqc_BS_wise_init = lambda nq, n_rep: vqc_BS(nq, n_rep, wise_init=True)

if not 'nq=1':
  def vqc():      # <- optimal in theory/pracitice
    '''
    |0>--RY--
    '''
    vqc = dq.QubitCircuit(nqubit=1)
    vqc.ry(wires=0)
    return vqc

  # fid=1.00000, ts=10.372s
  run_test(vqc)

if not 'nq=2':
  def vqc():      # <- optimal in theory
    '''
    |0>--RY--o---RY--
             |   |
    |0>------RY--o---
    '''
    vqc = dq.QubitCircuit(nqubit=2)
    vqc.ry(wires=0)
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=0, controls=1)
    return vqc

  def vqc_ex():   # <- optimal in pracitice
    '''
    |0>--RY--o---RY--o--
             |   |   |  
    |0>------RY--o---RY-
    '''
    vqc = dq.QubitCircuit(nqubit=2)
    vqc.ry(wires=0)
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=0, controls=1)
    vqc.ry(wires=1, controls=0)
    return vqc
  
  # fid=0.98684, ts=23.090s
  run_test(vqc)
  # fid=1.00000, ts=27.421s
  run_test(vqc_ex)

if not 'nq=3':
  def vqc():
    '''
    |0>--RY--o---RY---------o---RY--
             |   |          |   |
    |0>------RY--o--o---RY--|---|---
                    |   |   |   |
    |0>-------------RY--o---RY--o---
    '''
    vqc = dq.QubitCircuit(nqubit=3)
    vqc.ry(wires=0)
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=0, controls=1)
    vqc.ry(wires=2, controls=1)
    vqc.ry(wires=1, controls=2)
    vqc.ry(wires=2, controls=0)
    vqc.ry(wires=0, controls=2)
    return vqc

  def vqc_ex():     # <- optimal in pracitice
    '''
    |0>--RY--o---RY--o---------------o---RY--o--
             |   |   |               |   |   |  
    |0>------RY--o---RY--o---RY--o---|---|---|--
                         |   |   |   |   |   |  
    |0>------------------RY--o---RY--RY--o---RY-
    '''
    vqc = dq.QubitCircuit(nqubit=3)
    vqc.ry(wires=0)
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=0, controls=1)
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=2, controls=1)
    vqc.ry(wires=1, controls=2)
    vqc.ry(wires=2, controls=1)
    vqc.ry(wires=2, controls=0)
    vqc.ry(wires=0, controls=2)
    vqc.ry(wires=2, controls=0)
    return vqc

  def vqc_ex_red():
    '''
    |0>--RY--o---RY--o---------------
             |   |   |               
    |0>------RY--o---RY--o---RY--o---
                         |   |   |   
    |0>------------------RY--o---RY--
    '''
    vqc = dq.QubitCircuit(nqubit=3)
    vqc.ry(wires=0)
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=0, controls=1)
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=2, controls=1)
    vqc.ry(wires=1, controls=2)
    vqc.ry(wires=2, controls=1)
    return vqc

  def vqc_ex_red_ex():
    '''
    |0>--RY--o---RY--o---------------RY--
             |   |   |                   
    |0>------RY--o---RY--o---RY--o---RY--
                         |   |   |       
    |0>------------------RY--o---RY--RY--
    '''
    vqc = dq.QubitCircuit(nqubit=3)
    vqc.ry(wires=0)
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=0, controls=1)
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=2, controls=1)
    vqc.ry(wires=1, controls=2)
    vqc.ry(wires=2, controls=1)
    vqc.rylayer()
    return vqc

  def vqc_ex_red_ex_red():
    '''
    |0>--RY--o---RY--o---------------
             |   |   |               
    |0>--RY--RY--o---RY--o---RY--o---
                         |   |   |   
    |0>--RY--------------RY--o---RY--
    '''
    vqc = dq.QubitCircuit(nqubit=3)
    vqc.rylayer()
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=0, controls=1)
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=2, controls=1)
    vqc.ry(wires=1, controls=2)
    vqc.ry(wires=2, controls=1)
    return vqc

  def vqc_ex_red_ex_red_ex():
    '''
    |0>--RY--o---RY--o---------------RY--
             |   |   |                   
    |0>--RY--RY--o---RY--o---RY--o---RY--
                         |   |   |       
    |0>--RY--------------RY--o---RY--RY--
    '''
    vqc = dq.QubitCircuit(nqubit=3)
    vqc.rylayer()
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=0, controls=1)
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=2, controls=1)
    vqc.ry(wires=1, controls=2)
    vqc.ry(wires=2, controls=1)
    vqc.rylayer()
    return vqc

  def vqc_ex_red_ex_red_ex_red():
    '''
    |0>--RY--o---RY---------RY--
             |   |              
    |0>--RY--RY--o--o---RY--RY--
                    |   |       
    |0>--RY---------RY--o---RY--
    '''
    vqc = dq.QubitCircuit(nqubit=3)
    vqc.rylayer()
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=0, controls=1)
    vqc.ry(wires=2, controls=1)
    vqc.ry(wires=1, controls=2)
    vqc.rylayer()
    return vqc

  # fid=0.97787, ts=44.313s
  run_test(vqc)
  # fid=0.99529, ts=59.925s
  run_test(vqc_ex)
  ##(noplot) fid=0.84881, ts=44.702s
  run_test(vqc_ex_red)
  # fid=0.98703, ts=57.821s
  run_test(vqc_ex_red_ex)
  # fid=0.95806, ts=55.165s
  run_test(vqc_ex_red_ex_red)
  # fid=0.99812, ts=66.295s
  run_test(vqc_ex_red_ex_red_ex)
  # fid=0.99301, ts=54.474s
  run_test(vqc_ex_red_ex_red_ex_red)

if not 'nq=4':
  def vqc():
    '''
    |0>--RY--o---RY--o---------------------------RY--
             |   |   |                               
    |0>--RY--RY--o---RY--o---RY--o---------------RY--
                         |   |   |                   
    |0>--RY--------------RY--o---RY--o---RY--o---RY--
                                     |   |   |       
    |0>--RY--------------------------RY--o---RY--RY--
    '''
    vqc = dq.QubitCircuit(nqubit=4)
    vqc.rylayer()
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=0, controls=1)
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=2, controls=1)
    vqc.ry(wires=1, controls=2)
    vqc.ry(wires=2, controls=1)
    vqc.ry(wires=3, controls=2)
    vqc.ry(wires=2, controls=3)
    vqc.ry(wires=3, controls=2)
    vqc.rylayer()
    return vqc

  def vqc_red():
    '''
    |0>--RY--o---RY----------------RY--
             |   |                     
    |0>--RY--RY--o--o---RY---------RY--
                    |   |              
    |0>--RY---------RY--o--o---RY--RY--
                           |   |       
    |0>--RY----------------RY--o---RY--
    '''
    vqc = dq.QubitCircuit(nqubit=4)
    vqc.rylayer()
    vqc.ry(wires=1, controls=0)
    vqc.ry(wires=0, controls=1)
    vqc.ry(wires=2, controls=1)
    vqc.ry(wires=1, controls=2)
    vqc.ry(wires=3, controls=2)
    vqc.ry(wires=2, controls=3)
    vqc.rylayer()
    return vqc

  def vqc_F2_all_red(nq:int):
    ''' RY - pairwise(F2) '''
    vqc = dq.QubitCircuit(nqubit=nq)
    for i in range(nq):
      vqc.ry(wires=i)
    for i in range(nq-1):
      for j in range(i+1, nq):
        vqc.ry(wires=j, controls=i)
        vqc.ry(wires=i, controls=j)
    return vqc

  # fid=0.85385, ts=92.072s
  run_test(vqc)
  # fid=0.86965, ts=77.083s
  run_test(vqc_red)
  # gcnt=14, fid=0.99587, ts=110.074s
  run_test(partial(vqc_F2_all, 4))
  # gcnt=10, fid=0.97027, ts=94.975s
  run_test(partial(vqc_F2_all_red, 4))
  # gcnt=11, fid=0.99347, ts=99.389s
  run_test(partial(vqc_F2_all_wise_init, 4))

if not 'nq=5':
  # gcnt=30, fid=0.96353, ts=172.511s
  run_test(partial(vqc_F2_all, 5))
  # gcnt=26, fid=0.95208, ts=152.437s
  run_test(partial(vqc_F2_all_wise_init, 5))
  # gcnt=51, fid=1.00000, ts=307.370s
  run_test(partial(vqc_F2_all_wise_init, 5, 2))

if not 'nq=6':
  # gcnt=42, fid=0.87989, ts=233.897s
  run_test(partial(vqc_F2_all, 6))
  # gcnt=78, fid=0.99942, ts=441.007s
  run_test(partial(vqc_F2_all, 6, 2))
  # gcnt=73, fid=0.99918, ts=405.029s
  run_test(partial(vqc_F2_all_wise_init, 6, 2))
  # gcnt=73, fid=0.98284, ts=162.440s
  run_test(partial(vqc_F2_all_wise_init, 6, 2), n_iter=200)
  # gcnt=73, fid=0.99668, ts=167.478s
  run_test(partial(vqc_F2_all_wise_init, 6, 2), lr=0.02, n_iter=200)

if not 'nq=7':
  if 'F2/F1 标准结构，不同层数':
    # gcnt=99, fid=0.98078, ts=185.501s
    run_test(partial(vqc_F2_all_wise_init, 7, 2), lr=0.02, n_repeat=10)
    # gcnt=148, fid=0.99999, ts=284.290s
    run_test(partial(vqc_F2_all_wise_init, 7, 3), lr=0.02, n_repeat=10)

    # gcnt=85, fid=0.94860, ts=50.450s
    run_test(partial(vqc_F1_all_wise_init, 7, 3), lr=0.02, n_repeat=3)
    # gcnt=113, fid=0.99493, ts=65.707s
    run_test(partial(vqc_F1_all_wise_init, 7, 4), lr=0.02, n_repeat=3)
    # gcnt=141, fid=0.99970, ts=87.547s
    run_test(partial(vqc_F1_all_wise_init, 7, 5), lr=0.02, n_repeat=3)
    # gcnt=169, fid=0.99997, ts=99.258s
    run_test(partial(vqc_F1_all_wise_init, 7, 6), lr=0.02, n_repeat=3)

  if 'F2 变体结构':
    # gcnt=57, fid=0.83657, ts=36.646s
    run_test(partial(vqc_F2_flat, 7, 4), lr=0.02, n_repeat=3)
    # gcnt=71, fid=0.89794, ts=45.388s
    run_test(partial(vqc_F2_flat, 7, 5), lr=0.02, n_repeat=3)
    # gcnt=85, fid=0.96389, ts=54.286s
    run_test(partial(vqc_F2_flat, 7, 6), lr=0.02, n_repeat=3)

    # gcnt=111, fid=0.56015, ts=64.973s
    run_test(partial(vqc_F2_mera, 7, 4), lr=0.02, n_repeat=3)
    # gcnt=137, fid=0.56474, ts=80.533s
    run_test(partial(vqc_F2_mera, 7, 5), lr=0.02, n_repeat=3)

if not 'nq=8':
  # gcnt=129, fid=0.88666, ts=268.715s
  run_test(partial(vqc_F2_all_wise_init, 8, 2), lr=0.02, n_repeat=10)
  # gcnt=193, fid=0.98283, ts=375.533s
  run_test(partial(vqc_F2_all_wise_init, 8, 3), lr=0.02, n_repeat=10)
  # gcnt=257, fid=0.99950, ts=507.713s
  run_test(partial(vqc_F2_all_wise_init, 8, 4), lr=0.02, n_repeat=10)

if not 'nq=9':
  # gcnt=244, fid=0.88509, ts=481.398s
  run_test(partial(vqc_F2_all_wise_init, 9, 3), lr=0.02, n_repeat=10)
  # gcnt=325, fid=0.96041, ts=644.703s
  run_test(partial(vqc_F2_all_wise_init, 9, 4), lr=0.02, n_repeat=10)
  # gcnt=406, fid=0.99051, ts=880.120s
  run_test(partial(vqc_F2_all_wise_init, 9, 5), lr=0.02, n_repeat=10)
  # gcnt=487, fid=0.99862, ts=101.916s
  run_test(partial(vqc_F2_all_wise_init, 9, 6), lr=0.02, n_repeat=1)

  # gcnt=181, fid=0.76182, ts=439.282s
  run_test(partial(vqc_F1_all_wise_init, 9, 4), lr=0.02, n_repeat=10)

if not 'nq=10':
  if 'F2/F1 标准结构，不同层数':
    # gcnt=301, fid=0.70754, ts=609.749s
    run_test(partial(vqc_F2_all_wise_init, 10, 3), lr=0.02, n_repeat=10)
    # gcnt=401, fid=0.82405, ts=283.174s
    run_test(partial(vqc_F2_all_wise_init, 10, 4), lr=0.02, n_repeat=3)
    # gcnt=501, fid=0.90123, ts=353.806s
    run_test(partial(vqc_F2_all_wise_init, 10, 5), lr=0.02, n_repeat=3)
    # gcnt=601, fid=0.94936, ts=426.945s
    run_test(partial(vqc_F2_all_wise_init, 10, 6), lr=0.02, n_repeat=3)
    # gcnt=701, fid=0.97634, ts=491.314s
    run_test(partial(vqc_F2_all_wise_init, 10, 7), lr=0.02, n_repeat=3)
    # gcnt=801, fid=0.99098, ts=173.252s
    run_test(partial(vqc_F2_all_wise_init, 10, 8), lr=0.02, n_repeat=1)
    # gcnt=901, fid=0.99756, ts=196.935s
    run_test(partial(vqc_F2_all_wise_init, 10, 9), lr=0.02, n_repeat=1)
    # gcnt=1001, fid=0.99970, ts=217.438s
    run_test(partial(vqc_F2_all_wise_init, 10, 10), lr=0.02, n_repeat=1)

    # gcnt=166, fid=0.45164, ts=105.297s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=3)
    # gcnt=221, fid=0.56197, ts=139.564s
    run_test(partial(vqc_F1_all_wise_init, 10, 4), lr=0.02, n_repeat=3)
    # gcnt=276, fid=0.66346, ts=174.074s
    run_test(partial(vqc_F1_all_wise_init, 10, 5), lr=0.02, n_repeat=3)
    # gcnt=331, fid=0.72233, ts=209.322s
    run_test(partial(vqc_F1_all_wise_init, 10, 6), lr=0.02, n_repeat=3)
    # gcnt=386, fid=0.79541, ts=252.923s
    run_test(partial(vqc_F1_all_wise_init, 10, 7), lr=0.02, n_repeat=3)
    # gcnt=441, fid=0.84820, ts=283.122s
    run_test(partial(vqc_F1_all_wise_init, 10, 8), lr=0.02, n_repeat=3)
    # gcnt=496, fid=0.88548, ts=318.240s
    run_test(partial(vqc_F1_all_wise_init, 10, 9), lr=0.02, n_repeat=3)

  if 'F2/F1 变体结构':
    # gcnt=310, fid=0.71880, ts=197.384s
    run_test(partial(vqc_F2_all, 10, 3), lr=0.02, n_repeat=3)
    # gcnt=295, fid=0.70110, ts=193.905s
    run_test(partial(vqc_F2_all_gap_order_wise_init, 10, 3), lr=0.02, n_repeat=3)
    # gcnt=271, fid=0.66854, ts=183.463s
    run_test(partial(vqc_F2_all_no_inter_RY_wise_init, 10, 3), lr=0.02, n_repeat=3)

    # gcnt=201, fid=0.55018, ts=134.062s
    run_test(partial(vqc_F2_flat, 10, 10), lr=0.02, n_repeat=3)
    # gcnt=190, fid=0.51738, ts=125.604s
    run_test(partial(vqc_F2_brick, 10, 10), lr=0.02, n_repeat=3)
    # gcnt=258, fid=0.61370, ts=161.577s
    run_test(partial(vqc_F2_mera, 10, 4), lr=0.02, n_repeat=3)
    # gcnt=287, fid=0.66776, ts=175.550s
    run_test(partial(vqc_F2_distro, 10, 4), lr=0.02, n_repeat=3)
    # gcnt=200, fid=0.31347, ts=80.861s
    run_test(partial(vqc_HEA, 10, 10, 'linear'), lr=0.02, n_repeat=3)
    # gcnt=210, fid=0.31798, ts=85.247s
    run_test(partial(vqc_HEA, 10, 10, 'cyclic'), lr=0.02, n_repeat=3)

    # gcnt=175, fid=0.48017, ts=83.229s
    run_test(partial(vqc_BS, 10, 3), lr=0.02, n_repeat=3)
    # gcnt=230, fid=0.60165, ts=110.115s
    run_test(partial(vqc_BS, 10, 4), lr=0.02, n_repeat=3)
    # gcnt=285, fid=0.68850, ts=135.206s
    run_test(partial(vqc_BS, 10, 5), lr=0.02, n_repeat=3)
    # gcnt=166, fid=0.37754, ts=80.340s
    run_test(partial(vqc_BS_wise_init, 10, 3), lr=0.02, n_repeat=3)
    # gcnt=221, fid=0.51621, ts=105.545s
    run_test(partial(vqc_BS_wise_init, 10, 4), lr=0.02, n_repeat=3)
    # gcnt=276, fid=0.61899, ts=135.262s
    run_test(partial(vqc_BS_wise_init, 10, 5), lr=0.02, n_repeat=3)

    # gcnt=172, fid=0.41739, ts=98.280s
    run_test(partial(vqc_exF2, 10, 6, 'linear'), lr=0.02, n_repeat=3)
    # gcnt=190, fid=0.49950, ts=107.652s
    run_test(partial(vqc_exF2, 10, 6, 'cyclic'), lr=0.02, n_repeat=3)
    # gcnt=280, fid=0.66356, ts=163.317s
    run_test(partial(vqc_exF2, 10, 2, 'all'),    lr=0.02, n_repeat=3)
    # gcnt=154, fid=0.40053, ts=92.381s
    run_test(partial(vqc_exF1, 10, 8, 'linear'), lr=0.02, n_repeat=3)
    # gcnt=170, fid=0.45917, ts=102.027s
    run_test(partial(vqc_exF1, 10, 8, 'cyclic'), lr=0.02, n_repeat=3)
    # gcnt=190, fid=0.49421, ts=117.244s
    run_test(partial(vqc_exF1, 10, 2, 'all'),    lr=0.02, n_repeat=3)

  if '稀疏输入':
    # gcnt=301, fid=0.70633, ts=200.103s
    run_test(partial(vqc_F2_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=rand_mock_sparse_mnist)
    # gcnt=166, fid=0.52533, ts=107.391s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=rand_mock_sparse_mnist)

    # gcnt=301, fid=0.70559, ts=193.343s
    run_test(partial(vqc_F2_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 784))
    # gcnt=301, fid=0.72193, ts=193.390s
    run_test(partial(vqc_F2_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 512))
    # gcnt=301, fid=0.72861, ts=192.723s
    run_test(partial(vqc_F2_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 256))
    # gcnt=301, fid=0.91777, ts=197.471s
    run_test(partial(vqc_F2_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 150))
    # gcnt=301, fid=0.90637, ts=193.184s
    run_test(partial(vqc_F2_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 128))
    # gcnt=301, fid=0.99996, ts=193.212s
    run_test(partial(vqc_F2_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 64))
    # gcnt=301, fid=0.82617, ts=193.291s
    run_test(partial(vqc_F2_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 32))
    # gcnt=301, fid=0.99984, ts=193.202s
    run_test(partial(vqc_F2_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 16))

    # gcnt=166, fid=0.49191, ts=109.344s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 784))
    # gcnt=166, fid=0.66571, ts=105.723s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 512))
    # gcnt=166, fid=0.78985, ts=105.153s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 256))
    # gcnt=166, fid=0.86335, ts=108.861s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 150))
    # gcnt=166, fid=0.91255, ts=733.685s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_iter=1000, n_repeat=10, data_gen=partial(rand_first_n, 150))
    # gcnt=166, fid=0.93118, ts=104.933s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 128))
    # gcnt=166, fid=0.97598, ts=105.044s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 64))
    # gcnt=166, fid=0.99978, ts=105.467s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 32))
    # gcnt=166, fid=0.99999, ts=105.050s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=3, data_gen=partial(rand_first_n, 16))

  if '真MNSIT输入':
    # gcnt=166, fid=0.75795, ts=179.494s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=5, data_gen=rand_mnist)
    # gcnt=166, fid=0.77113, ts=178.002s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=5, data_gen=rand_mnist_snake)
    # gcnt=166, fid=0.78747, ts=175.810s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=5, data_gen=rand_mnist_snake_rev)
    # gcnt=166, fid=0.85730, ts=180.815s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=5, data_gen=rand_mnist_freq)
    # gcnt=166, fid=0.87208, ts=351.334s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=5, n_iter=1000, data_gen=rand_mnist_freq)
    # gcnt=201, fid=0.83103, ts=247.784s
    run_test(partial(vqc_F1_s_all_wise_init, 10, 2), lr=0.02, n_repeat=5, data_gen=rand_mnist_freq)
    # gcnt=166, fid=0.45923, ts=156.974s
    run_test(partial(vqc_F1_all_wise_init_0, 10, 3), lr=0.02, n_repeat=5, data_gen=rand_mnist_freq)

    # gcnt=172, fid=0.84517, ts=109.191s
    run_test(partial(vqc_exF2, 10, 6, 'linear'), lr=0.02, n_repeat=3, data_gen=rand_mnist_freq)
    # gcnt=190, fid=0.85821, ts=122.588s
    run_test(partial(vqc_exF2, 10, 6, 'cyclic'), lr=0.02, n_repeat=3, data_gen=rand_mnist_freq)
    # gcnt=280, fid=0.91147, ts=176.436s
    run_test(partial(vqc_exF2, 10, 2, 'all'),    lr=0.02, n_repeat=3, data_gen=rand_mnist_freq)
    # gcnt=154, fid=0.79679, ts=93.245s
    run_test(partial(vqc_exF1, 10, 8, 'linear'), lr=0.02, n_repeat=3, data_gen=rand_mnist_freq)
    # gcnt=170, fid=0.79948, ts=101.545s
    run_test(partial(vqc_exF1, 10, 8, 'cyclic'), lr=0.02, n_repeat=3, data_gen=rand_mnist_freq)
    # gcnt=190, fid=0.81678, ts=113.719s
    run_test(partial(vqc_exF1, 10, 2, 'all'),    lr=0.02, n_repeat=3, data_gen=rand_mnist_freq)

    # gcnt=172, fid=0.75403, ts=172.880s
    run_test(partial(vqc_F1_HEA_wise_init, 10, 9, 'CRY', 'linear'), lr=0.02, n_repeat=5, data_gen=rand_mnist)
    # gcnt=172, fid=0.61785, ts=169.108s
    run_test(partial(vqc_F1_HEA_wise_init, 10, 9, 'CRX', 'linear'), lr=0.02, n_repeat=5, data_gen=rand_mnist)
    # gcnt=172, fid=0.68059, ts=119.570s
    run_test(partial(vqc_F1_HEA_wise_init, 10, 9, 'CNOT', 'linear'), lr=0.02, n_repeat=5, data_gen=rand_mnist)

    # gcnt=166, fid=0.79077, ts=345.262s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.5, n_repeat=5, n_iter=1000, data_gen=rand_mnist_freq)
    # gcnt=166, fid=0.86040, ts=353.107s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.3, n_repeat=5, n_iter=1000, data_gen=rand_mnist_freq)
    # gcnt=166, fid=0.83410, ts=356.523s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.2, n_repeat=5, n_iter=1000, data_gen=rand_mnist_freq)
    # gcnt=166, fid=0.88042, ts=348.083s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.1, n_repeat=5, n_iter=1000, data_gen=rand_mnist_freq)
    # gcnt=166, fid=0.87448, ts=344.784s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.075, n_repeat=5, n_iter=1000, data_gen=rand_mnist_freq)
    # gcnt=166, fid=0.88032, ts=346.654s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.05, n_repeat=5, n_iter=1000, data_gen=rand_mnist_freq)
    # gcnt=166, fid=0.87544, ts=345.272s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.03, n_repeat=5, n_iter=1000, data_gen=rand_mnist_freq)
    # gcnt=166, fid=0.86568, ts=355.811s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.01, n_repeat=5, n_iter=1000, data_gen=rand_mnist_freq)
    # gcnt=166, fid=0.85603, ts=341.982s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.005, n_repeat=5, n_iter=1000, data_gen=rand_mnist_freq)

    # gcnt=166, fid=0.78495, ts=176.477s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=5, data_gen=partial(rand_mnist_snake_rev_trim, 8))
    # gcnt=166, fid=0.78998, ts=178.903s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=5, data_gen=partial(rand_mnist_snake_rev_trim, 32))
    # gcnt=166, fid=0.79544, ts=175.104s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=5, data_gen=partial(rand_mnist_snake_rev_trim, 64))
    # gcnt=166, fid=0.77591, ts=174.973s
    run_test(partial(vqc_F1_all_wise_init, 10, 3), lr=0.02, n_repeat=5, data_gen=partial(rand_mnist_snake_rev_trim, 128))

if not 'nq=12':
  if '真CIFAR10输入':
    # gcnt=79, fid=0.83038, ts=87.078s; score~=2.58176
    run_test(partial(vqc_F1_all_wise_init, 12, 1), lr=0.02, n_repeat=5, data_gen=rand_cifar10)
    # gcnt=157, fid=0.92497, ts=169.481s; score~=2.69294
    run_test(partial(vqc_F1_all_wise_init, 12, 2), lr=0.02, n_repeat=5, data_gen=rand_cifar10)
    # gcnt=235, fid=0.93357, ts=256.822s; score~=2.63214
    run_test(partial(vqc_F1_all_wise_init, 12, 3), lr=0.02, n_repeat=5, data_gen=rand_cifar10)

    # gcnt=157, fid=0.96159, ts=169.883s; score~=2.76618 (⭐)
    run_test(partial(vqc_F1_all_wise_init_0,       12, 2), lr=0.02, n_repeat=5, data_gen=rand_cifar10)
    # gcnt=155, fid=0.94067, ts=172.713s; score~=2.72634
    run_test(partial(vqc_F1_gap_order_wise_init,   12, 2), lr=0.02, n_repeat=5, data_gen=rand_cifar10)
    # gcnt=155, fid=0.95962, ts=175.082s; score~=2.76424
    run_test(partial(vqc_F1_gap_order_wise_init_0, 12, 2), lr=0.02, n_repeat=5, data_gen=rand_cifar10)
    # gcnt=168, fid=0.77040, ts=202.099s
    run_test(partial(vqc_F1_all_RX_0,              12, 2), lr=0.02, n_repeat=5, data_gen=rand_cifar10)

    # gcnt=79, fid=0.76987, ts=86.685s
    run_test(partial(vqc_F1_all_wise_init, 12, 1), lr=0.02, n_repeat=5, data_gen=rand_cifar10_HWC)
    # gcnt=157, fid=0.88757, ts=175.101s
    run_test(partial(vqc_F1_all_wise_init, 12, 2), lr=0.02, n_repeat=5, data_gen=rand_cifar10_HWC)
    # gcnt=235, fid=0.89124, ts=259.387s
    run_test(partial(vqc_F1_all_wise_init, 12, 3), lr=0.02, n_repeat=5, data_gen=rand_cifar10_HWC)

    # gcnt=79, fid=0.80797, ts=88.516s
    run_test(partial(vqc_F1_all_wise_init, 12, 1), lr=0.02, n_repeat=5, data_gen=rand_cifar10_snake_rev)
    # gcnt=157, fid=0.87726, ts=172.158s
    run_test(partial(vqc_F1_all_wise_init, 12, 2), lr=0.02, n_repeat=5, data_gen=rand_cifar10_snake_rev)
    # gcnt=235, fid=0.88875, ts=255.159s
    run_test(partial(vqc_F1_all_wise_init, 12, 3), lr=0.02, n_repeat=5, data_gen=rand_cifar10_snake_rev)

  if '随机稠密输入':
    # gcnt=79, fid=0.07442, ts=84.774s
    run_test(partial(vqc_F1_all_wise_init, 12, 1), lr=0.02, n_repeat=5, data_gen=rand_mock_cifar10)
    # gcnt=157, fid=0.15322, ts=173.066s
    run_test(partial(vqc_F1_all_wise_init, 12, 2), lr=0.02, n_repeat=5, data_gen=rand_mock_cifar10)
    # gcnt=235, fid=0.22628, ts=257.779s
    run_test(partial(vqc_F1_all_wise_init, 12, 3), lr=0.02, n_repeat=5, data_gen=rand_mock_cifar10)


''' The Mindspore Ansatz Zoo: https://www.mindspore.cn/mindquantum/docs/zh-CN/r0.9/algorithm/mindquantum.algorithm.nisq.html '''

def vqc_RY_linear(nq:int, n_rep:int=1):
  vqc = dq.QubitCircuit(nqubit=nq)
  for i in range(nq):
    vqc.ry(wires=i)
  for _ in range(n_rep):
    for i in range(nq-1):
      vqc.x(wires=i+1, controls=i)
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

def vqc_RY_full(nq:int, n_rep:int=1):
  vqc = dq.QubitCircuit(nqubit=nq)
  for i in range(nq):
    vqc.ry(wires=i)
  for _ in range(n_rep):
    for i in range(nq-1):
      for j in range(i+1, nq):
        vqc.x(wires=j, controls=i)
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

def vqc_RY_cascade(nq:int, n_rep:int=1):
  vqc = dq.QubitCircuit(nqubit=nq)
  for i in range(nq):
    vqc.ry(wires=i)
  for _ in range(n_rep):
    for i in range(nq-1):
      vqc.x(wires=i+1, controls=i)
    for i in range(nq):
      vqc.ry(wires=i)
    for i in reversed(range(nq-1)):
      vqc.x(wires=i+1, controls=i)
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

def vqc_RY_RZ_full(nq:int, n_rep:int=1):
  vqc = dq.QubitCircuit(nqubit=nq)
  for i in range(nq):
    vqc.ry(wires=i)
    vqc.rz(wires=i)
  for _ in range(n_rep):
    for i in range(nq-1):
      for j in range(i+1, nq):
        vqc.x(wires=j, controls=i)
    for i in range(nq):
      vqc.ry(wires=i)
      vqc.rz(wires=i)
  return vqc

if not 'nq=10':
  # gcnt=162, fid=0.26004, ts=76.253s
  run_test(partial(vqc_RY_linear,  10, 8), lr=0.02, n_repeat=3)
  # gcnt=175, fid=0.10327, ts=74.291s
  run_test(partial(vqc_RY_full,    10, 3), lr=0.02, n_repeat=3)
  # gcnt=162, fid=0.26393, ts=74.255s
  run_test(partial(vqc_RY_cascade, 10, 4), lr=0.02, n_repeat=3)
  # gcnt=215, fid=0.12827, ts=95.191s
  run_test(partial(vqc_RY_RZ_full, 10, 3), lr=0.02, n_repeat=3)

  # gcnt=162, fid=0.64805, ts=75.997s
  run_test(partial(vqc_RY_linear,  10, 8), lr=0.02, n_repeat=3, data_gen=rand_mnist_freq)
  # gcnt=175, fid=0.60505, ts=74.177s
  run_test(partial(vqc_RY_full,    10, 3), lr=0.02, n_repeat=3, data_gen=rand_mnist_freq)
  # gcnt=162, fid=0.65853, ts=75.625s
  run_test(partial(vqc_RY_cascade, 10, 4), lr=0.02, n_repeat=3, data_gen=rand_mnist_freq)
  # gcnt=215, fid=0.51854, ts=95.574s
  run_test(partial(vqc_RY_RZ_full, 10, 3), lr=0.02, n_repeat=3, data_gen=rand_mnist_freq)


''' The QAOA ansatz '''

def vqc_MA_QAOA(nq:int, n_rep:int=1):
  ''' https://arxiv.org/pdf/2302.04479 '''
  vqc = dq.QubitCircuit(nqubit=nq)
  for i in range(nq):
    vqc.h(wires=i)
  for _ in range(n_rep):
    for i in range(nq-1):
      for j in range(i+1, nq):
        vqc.rzz(wires=[i, j])
    for i in range(nq):
      vqc.rx(wires=i)
  return vqc

def vqc_XQAOA(nq:int, n_rep:int=1):
  ''' https://arxiv.org/pdf/2302.04479 '''
  vqc = dq.QubitCircuit(nqubit=nq)
  for i in range(nq):
    vqc.h(wires=i)
  for _ in range(n_rep):
    for i in range(nq-1):
      for j in range(i+1, nq):
        vqc.rzz(wires=[i, j])
    for i in range(nq):
      vqc.rx(wires=i)
      vqc.ry(wires=i)
  return vqc

if not 'nq=10':
  # gcnt=175, fid=0.23441, ts=161.724s
  run_test(partial(vqc_MA_QAOA, 10, 3), lr=0.02, n_repeat=5)
  # gcnt=175, fid=0.23279, ts=163.927s
  run_test(partial(vqc_MA_QAOA, 10, 3), lr=0.02, n_repeat=5, data_gen=rand_mnist_freq)

  # gcnt=205, fid=0.30375, ts=184.506s
  run_test(partial(vqc_XQAOA, 10, 3), lr=0.02, n_repeat=5)
  # gcnt=205, fid=0.30028, ts=187.019s
  run_test(partial(vqc_XQAOA, 10, 3), lr=0.02, n_repeat=5, data_gen=rand_mnist_freq)


''' Expressivity Efficient Ansatz: https://arxiv.org/pdf/1905.10876 '''

def vqc_EEA_2(nq:int, n_rep:int=1):
  vqc = dq.QubitCircuit(nqubit=nq)
  for _ in range(n_rep):
    for i in range(nq):
      vqc.rx(wires=i)
      vqc.rz(wires=i)
    for i in range(nq-1):
      vqc.x(wires=i+1, controls=i)
  return vqc

def vqc_EEA_5(nq:int, n_rep:int=1):
  vqc = dq.QubitCircuit(nqubit=nq)
  for i in range(nq):
    vqc.rx(wires=i)
    vqc.rz(wires=i)
  for _ in range(n_rep):
    for i in range(nq):
      for j in range(nq):
        if i == j: continue
        vqc.rz(wires=j, controls=i)
    for i in range(nq):
      vqc.rx(wires=i)
      vqc.rz(wires=i)
  return vqc

def vqc_EEA_6(nq:int, n_rep:int=1):
  vqc = dq.QubitCircuit(nqubit=nq)
  for i in range(nq):
    vqc.rx(wires=i)
    vqc.rz(wires=i)
  for _ in range(n_rep):
    for i in range(nq):
      for j in range(nq):
        if i == j: continue
        vqc.rx(wires=j, controls=i)
    for i in range(nq):
      vqc.rx(wires=i)
      vqc.rz(wires=i)
  return vqc

def vqc_EEA_9(nq:int, n_rep:int=1):
  vqc = dq.QubitCircuit(nqubit=nq)
  for i in range(nq):
    vqc.h(wires=i)
  for _ in range(n_rep):
    for i in range(nq-1):
      vqc.cz(i, i+1)
    for i in range(nq):
      vqc.rx(wires=i)
  return vqc

def vqc_EEA_13(nq:int, n_rep:int=1):
  vqc = dq.QubitCircuit(nqubit=nq)
  for _ in range(n_rep):
    for i in range(nq):
      vqc.ry(wires=i)
    for i in range(nq):
      vqc.rz(wires=(i-1+nq)%nq, controls=i)
    for i in range(nq):
      vqc.ry(wires=i)
    for i in [0] + list(range(nq-1, 0, -1)):
      vqc.rz(wires=(i+1)%nq, controls=i)
  return vqc

def vqc_EEA_14(nq:int, n_rep:int=1):
  vqc = dq.QubitCircuit(nqubit=nq)
  for _ in range(n_rep):
    for i in range(nq):
      vqc.ry(wires=i)
    for i in range(nq):
      vqc.rx(wires=(i-1+nq)%nq, controls=i)
    for i in range(nq):
      vqc.ry(wires=i)
    for i in [0] + list(range(nq-1, 0, -1)):
      vqc.rx(wires=(i+1)%nq, controls=i)
  return vqc

def vqc_EEA_19(nq:int, n_rep:int=1):
  vqc = dq.QubitCircuit(nqubit=nq)
  for _ in range(n_rep):
    for i in range(nq):
      vqc.rx(wires=i)
      vqc.rz(wires=i)
    for i in range(nq):
      vqc.rx(wires=i, controls=(i+1)%nq)
  return vqc

if not 'nq=10':
  '''
  Expressibility: 6 > 14 > 13 > 5 > 19 > 8 > 7 > 11 > 4 > 17 > 15
  Fidelity: 6 > 14 > 13 > 5 > 2 > 9 > 19
  '''
  # gcnt=174, fid=0.18395, ts=88.235s
  run_test(partial(vqc_EEA_2,  10, 6), lr=0.02, n_repeat=3)
  # gcnt=240, fid=0.23821, ts=179.804s
  run_test(partial(vqc_EEA_5,  10, 2), lr=0.02, n_repeat=3)
  # gcnt=240, fid=0.36405, ts=179.811s
  run_test(partial(vqc_EEA_6,  10, 2), lr=0.02, n_repeat=3)
  # gcnt=162, fid=0.06655, ts=72.050s
  run_test(partial(vqc_EEA_9,  10, 8), lr=0.02, n_repeat=3)
  # gcnt=160, fid=0.24862, ts=106.884s
  run_test(partial(vqc_EEA_13, 10, 4), lr=0.02, n_repeat=3)
  # gcnt=160, fid=0.24903, ts=107.673s
  run_test(partial(vqc_EEA_14, 10, 4), lr=0.02, n_repeat=3)
  # gcnt=180, fid=0.02550, ts=82.862s
  run_test(partial(vqc_EEA_19, 10, 6), lr=0.02, n_repeat=3)


''' RY-cyclic(CZ) from QGAN: https://arxiv.org/pdf/1904.00043 '''

def vqc_RY_cyclic_CZ(nq:int, n_rep:int=10):
  vqc = dq.QubitCircuit(nqubit=nq)
  for _ in range(n_rep):
    for i in range(nq):
      vqc.ry(i)
    for i in range(nq):
      vqc.cz((i+1)%nq, i)
  for i in range(nq):
    vqc.ry(i)
  return vqc

if not 'nq=6':
  # gcnt=78, fid=0.62437, ts=44.150s
  run_test(partial(vqc_RY_cyclic_CZ, 6, 6), lr=0.02, n_repeat=5)

if not 'nq=10':
  # gcnt=170, fid=0.17342, ts=60.000s
  run_test(partial(vqc_RY_cyclic_CZ, 10,  8), lr=0.02, n_repeat=3)
  # gcnt=210, fid=0.16999, ts=73.361s
  run_test(partial(vqc_RY_cyclic_CZ, 10, 10), lr=0.02, n_repeat=3)
  # gcnt=250, fid=0.16384, ts=86.240s
  run_test(partial(vqc_RY_cyclic_CZ, 10, 12), lr=0.02, n_repeat=3)
  # gcnt=330, fid=0.16255, ts=114.043s
  run_test(partial(vqc_RY_cyclic_CZ, 10, 16), lr=0.02, n_repeat=3)


''' U3-cyclic(CZ) from arXiv:2107.03630 '''

def vqc_U3_cyclic_CZ(nq:int, n_rep:int=10):
  ''' The idea is straigh-forward: U3 for abs(amp) value, CZ for sign of the phase (pos/neg) '''
  vqc = dq.QubitCircuit(nqubit=nq)
  for _ in range(n_rep):
    for i in range(nq):
      vqc.u3(i)
    for i in range(0, nq-1, 2):
      vqc.cz((i+1)%nq, i)
  return vqc

if not 'nq=10':
  # gcnt=150, fid=0.03829, ts=108.025s
  run_test(partial(vqc_U3_cyclic_CZ, 10, 10), lr=0.02, n_repeat=3)

if not 'nq=12':
  # gcnt=180, fid=0.83543, ts=134.693s
  run_test(partial(vqc_U3_cyclic_CZ, 12, 10), lr=0.02, n_repeat=3, data_gen=rand_cifar10)


''' RY-adjacent(CNOT) from https://arxiv.org/pdf/2103.13211 '''

def vqc_RY_adjacent_CNOT(nq:int, n_rep:int=10):
  vqc = dq.QubitCircuit(nqubit=nq)
  for _ in range(n_rep):
    for i in range(nq):
      vqc.ry(i)
    for i in range(0, nq-1, 2):
      vqc.cnot(i+1, i)
    for i in range(1, nq, 2):
      vqc.cnot((i+1)%nq, i)
  for i in range(nq):
    vqc.ry(i)
  return vqc

if not 'nq=10':
  # gcnt=170, fid=0.27346, ts=48.144s
  run_test(partial(vqc_RY_adjacent_CNOT, 10,  8), lr=0.02, n_repeat=3)
  # gcnt=210, fid=0.33862, ts=57.630s
  run_test(partial(vqc_RY_adjacent_CNOT, 10, 10), lr=0.02, n_repeat=3)
  # gcnt=250, fid=0.37947, ts=66.279s
  run_test(partial(vqc_RY_adjacent_CNOT, 10, 12), lr=0.02, n_repeat=3)
  # gcnt=330, fid=0.47512, ts=87.080s
  run_test(partial(vqc_RY_adjacent_CNOT, 10, 16), lr=0.02, n_repeat=3)


''' RX/RY/RZ-adjacent(CNOT) from https://arxiv.org/pdf/2103.13211 '''

def vqc_Rrnd_adjacent_CNOT(nq:int, n_rep:int=10):
  import random
  rand = lambda: random.randrange(3)
  vqc = dq.QubitCircuit(nqubit=nq)
  for _ in range(n_rep):
    for i in range(nq):
      r = rand()
      if   r == 0: vqc.rx(i)
      elif r == 1: vqc.ry(i)
      elif r == 2: vqc.rz(i)
    for i in range(0, nq-1, 2):
      vqc.cnot(i+1, i)
    for i in range(1, nq, 2):
      vqc.cnot((i+1)%nq, i)
  for i in range(nq):
    r = rand()
    if   r == 0: vqc.rx(i)
    elif r == 1: vqc.ry(i)
    elif r == 2: vqc.rz(i)
  return vqc

if not 'nq=10':
  # gcnt=170, fid=0.13739, ts=47.712s
  run_test(partial(vqc_Rrnd_adjacent_CNOT, 10,  8), lr=0.02, n_repeat=3)
  # gcnt=210, fid=0.17433, ts=59.170s
  run_test(partial(vqc_Rrnd_adjacent_CNOT, 10, 10), lr=0.02, n_repeat=3)
  # gcnt=250, fid=0.19957, ts=69.347s
  run_test(partial(vqc_Rrnd_adjacent_CNOT, 10, 12), lr=0.02, n_repeat=3)
  # gcnt=330, fid=0.26883, ts=92.022s
  run_test(partial(vqc_Rrnd_adjacent_CNOT, 10, 16), lr=0.02, n_repeat=3)
