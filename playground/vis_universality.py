#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/10 

# 探究 VQC 线路拟合任意分布的普适性

from time import time
from functools import partial
from typing import Tuple, Callable

import torch
from torch import Tensor
from torch.optim import Adam
import deepquantum as dq
from tqdm import tqdm

N_ITER = 500
N_REPEAT = 30
SEED = 114514
mean = lambda x: sum(x) / len(x)

def set_seed():
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)

def rand_state(nq:int) -> Tensor:
  v = torch.rand(2**nq) - 0.5
  return v / torch.linalg.norm(v)

def rand_mnist(nq:int) -> Tensor:
  assert nq >= 10
  v = torch.rand(2**nq)   # assume all elem positive
  v[784:] = 0.0   # 28*28 non-pad pixles
  return v / torch.linalg.norm(v)

def rand_mnist9(nq:int) -> Tensor:
  assert nq >= 9
  v = torch.rand(2**nq)   # assume all elem positive
  v[512:] = 0.0   # only pick 512 dim
  return v / torch.linalg.norm(v)

def get_fidelity(x:Tensor, y:Tensor) -> Tensor:
  return (x * y).sum()**2

def get_gate_count(qc:dq.QubitCircuit) -> int:
  return len([op for op in qc.operators.modules() if isinstance(op, dq.operation.Operation)])

def run_test(get_vqc:Callable[[], dq.QubitCircuit], lr:float=0.01, n_repeat:int=N_REPEAT, n_iter:int=N_ITER, 
             mock_mnist:bool=False, mock_mnist9:bool=False) -> Tuple[float, float]:
  set_seed()

  gcnt = get_gate_count(get_vqc())

  ts_start = time()
  fids: list[float] = []
  pbar = tqdm(total=n_repeat)
  for _ in range(n_repeat):
    vqc = get_vqc()
    if   mock_mnist:  tgt = rand_mnist (vqc.nqubit)
    elif mock_mnist9: tgt = rand_mnist9(vqc.nqubit)
    else:             tgt = rand_state (vqc.nqubit)

    optim = Adam(vqc.parameters(), lr=lr)
    for _ in range(n_iter):
      optim.zero_grad()
      state = vqc().squeeze().real
      loss = -get_fidelity(state, tgt)
      loss.backward()
      optim.step()

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

def vqc_submit_p1(nq:int, n_rep:int=14):
  vqc = dq.QubitCircuit(nqubit=nq)
  vqc.x(0)
  for _ in range(n_rep):
    for q in range(nq-1):
      g = dq.Ry(nqubit=nq, wires=(q+1)%nq, controls=q, requires_grad=True)
      g.init_para([0])
      vqc.add(g)
      g = dq.Ry(nqubit=nq, wires=q, controls=(q+1)%nq, requires_grad=True)
      g.init_para([0])
      vqc.add(g)
  return vqc

if not 'nq=10':
  # gcnt=253, fid=0.61323, ts=525.098s
  run_test(partial(vqc_submit_p1, 10, 14), lr=0.02, n_repeat=10)
  # gcnt=289, fid=0.65839, ts=69.527s
  run_test(partial(vqc_submit_p1, 10, 16), lr=0.02, n_repeat=1)


''' The evolutional story for vqc_F2 ansatz series '''

def vqc_F2_all(nq:int, n_rep:int=1):
  ''' RY - [pairwise(F2) - RY] '''
  vqc = dq.QubitCircuit(nqubit=nq)
  for i in range(nq):
    vqc.ry(wires=i)
  for _ in range(n_rep):
    for i in range(nq-1):
      for j in range(i+1, nq):
        vqc.ry(wires=j, controls=i)
        vqc.ry(wires=i, controls=j)
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

def vqc_F2_all_wise_init(nq:int, n_rep:int=1):
  ''' RY(single init) - [pairwise(F2) - RY] '''
  vqc = dq.QubitCircuit(nqubit=nq)
  vqc.ry(wires=0)   # only init wire 0
  for _ in range(n_rep):
    for i in range(nq-1):   # qubit order
      for j in range(i+1, nq):
        vqc.ry(wires=j, controls=i)
        vqc.ry(wires=i, controls=j)
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

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
  # gcnt=99, fid=0.98078, ts=185.501s
  run_test(partial(vqc_F2_all_wise_init, 7, 2), lr=0.02, n_repeat=10)
  # gcnt=148, fid=0.99999, ts=284.290s
  run_test(partial(vqc_F2_all_wise_init, 7, 3), lr=0.02, n_repeat=10)

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

if not 'nq=10':
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

  # gcnt=310, fid=0.71880, ts=197.384s
  run_test(partial(vqc_F2_all, 10, 3), lr=0.02, n_repeat=3)
  # gcnt=295, fid=0.70110, ts=193.905s
  run_test(partial(vqc_F2_all_gap_order_wise_init, 10, 3), lr=0.02, n_repeat=3)
  ##(noplot) gcnt=301, fid=0.78257, ts=653.304s
  run_test(partial(vqc_F2_all_wise_init, 10, 3), lr=0.02, n_repeat=10, mock_mnist=True)
  ##(noplot) gcnt=301, fid=0.78741, ts=648.813s
  run_test(partial(vqc_F2_all_wise_init, 10, 3), lr=0.02, n_repeat=10, mock_mnist9=True)
  # gcnt=271, fid=0.66854, ts=183.463s
  run_test(partial(vqc_F2_all_no_inter_RY_wise_init, 10, 3), lr=0.02, n_repeat=3)


''' vqc_F2_mera ansatz series '''

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
        print(q, q+1)
        vqc.ry(wires=q+1, controls=q)
        vqc.ry(wires=q, controls=q+1)
    for i in mid_wires:
      vqc.ry(wires=i)
    # up (->4-6-8-10)
    for offset in range(nq//2-1, 0, -1):
      for q in range(offset, nq-1-offset, 2):
        print(q, q+1)
        vqc.ry(wires=q+1, controls=q)
        vqc.ry(wires=q, controls=q+1)
    for i in range(nq):
      vqc.ry(wires=i)
  return vqc

if not 'nq=7':
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

if not 'nq=10':
  # gcnt=201, fid=0.55018, ts=134.062s
  run_test(partial(vqc_F2_flat, 10, 10), lr=0.02, n_repeat=3)

  # gcnt=258, fid=0.61370, ts=161.577s
  run_test(partial(vqc_F2_mera, 10, 4), lr=0.02, n_repeat=3)
