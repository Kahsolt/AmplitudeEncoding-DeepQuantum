#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/22 

# 尝试纯粹训练线路 (实验性小规模训练来筛选Ansatz)
# normal: gate ~300, fid ~0.85
# snake: gate ~280, fid ~0.94 (wtf?!)

import os
os.environ['MY_LABORATORY'] = '1'

from time import time
from utils import *

N_ITER = 1000
device = 'cpu'


def get_model(n_layer:int=5, nq:int=10) -> dq.QubitCircuit:
  vqc = dq.QubitCircuit(nq)

  if not 'u3 bridge-cu':  # n_layer=15, n_gate=228;  fid=0.7716873288154602
    for i in range(n_layer):
      vqc.u3layer()
      offset = int(i % 2 == 1)
      for q in range(offset, nq - offset, 2):
        if offset: vqc.cu(q + 1, q)
        else:      vqc.cu(q, q + 1)
    vqc.u3layer()

  if not 'mottonen-like':   # n_layer=5, n_gate=280; fid=0.7938
    for i in range(n_layer):
      for q in range(1, nq):
        # cond
        vqc.rylayer(list(range(q)))
        # mctrl-rot
        vqc.ry(q, controls=list(range(q)), condition=True)
    vqc.rylayer()   # 最后一层需要初始化

  if not 'mottonen-like zero-init':   # n_layer=5, n_gate=280; fid=0.8511381149291992
    for i in range(n_layer):
      for q in range(1, nq):
        # cond
        lyr = dq.RyLayer(nq, wires=list(range(q)), requires_grad=True)
        lyr.init_para([0] * q)
        vqc.add(lyr)
        # mctrl-rot
        g = dq.Ry(nqubit=nq, wires=q, controls=list(range(q)), condition=True, requires_grad=True)
        g.init_para([0])
        vqc.add(g)
    vqc.rylayer()   # 最后一层需要初始化

  # [normal_reshape]
  # n_layer=5, n_gate=280; fid=0.8616576194763184
  # [snake_reshape]
  # nlayer=6 gate=334 time=110.43422794342041 fid=0.9528707265853882
  # nlayer=5 gate=280 time= 92.87968921661377 fid=0.9477213025093079 (可能最优!!)
  # nlayer=4 gate=226 time= 73.86409831047058 fid=0.9128199219703674 (可能最优!!)
  # nlayer=3 gate=172 time= 56.92887687683106 fid=0.8413708209991455
  if not 'mottonen-like zero-init ↓↑':
    for i in range(n_layer):
      flag = i % 2 == 1
      if flag == 0:
        for q in range(1, nq):
          # cond
          lyr = dq.RyLayer(nq, wires=list(range(q)), requires_grad=True)
          lyr.init_para([0] * q)
          vqc.add(lyr)
          # mctrl-rot
          g = dq.Ry(nqubit=nq, wires=q, controls=list(range(q)), condition=True, requires_grad=True)
          g.init_para([0])
          vqc.add(g)
      else:
        for q in reversed(range(1, nq)):
          # cond
          lyr = dq.RyLayer(nq, wires=list(range(q, nq)), requires_grad=True)
          lyr.init_para([0] * (nq - q + 1))
          vqc.add(lyr)
          # mctrl-rot
          g = dq.Ry(nqubit=nq, wires=q-1, controls=list(range(q, nq)), condition=True, requires_grad=True)
          g.init_para([0])
          vqc.add(g)
    vqc.rylayer()   # 最后一层需要初始化

  if not 'mottonen-like zero-init ↓↑ u3-gate':   # n_layer=5, n_gate=280, fid=0.8324395418167114
    for i in range(n_layer):
      flag = i % 2 == 1
      if flag == 0:
        for q in range(1, nq):
          # cond
          lyr = dq.U3Layer(nq, wires=list(range(q)), requires_grad=True)
          lyr.init_para([0] * (q*3))
          vqc.add(lyr)
          # mctrl-rot
          g = dq.U3Gate(nqubit=nq, wires=q, controls=list(range(q)), condition=True, requires_grad=True)
          g.init_para([0, 0, 0])
          vqc.add(g)
      else:
        for q in reversed(range(1, nq)):
          # cond
          lyr = dq.U3Layer(nq, wires=list(range(q, nq)), requires_grad=True)
          lyr.init_para([0] * (3*(nq - q + 1)))
          vqc.add(lyr)
          # mctrl-rot
          g = dq.U3Gate(nqubit=nq, wires=q-1, controls=list(range(q, nq)), condition=True, requires_grad=True)
          g.init_para([0, 0, 0])
          vqc.add(g)
    vqc.u3layer()   # 最后一层需要初始化

  if not 'mottonen-like zero-init shift↓':   # n_layer=5, n_gate=280, fid=0.8263607025146484
    for i in range(n_layer):
      for q in range(1, nq):
        ctrl_wires = [(e + i) % nq for e in range(q)]
        tgt_wire = (q + i) % nq
        # cond
        lyr = dq.RyLayer(nq, wires=ctrl_wires, requires_grad=True)
        lyr.init_para([0] * len(ctrl_wires))
        vqc.add(lyr)
        # mctrl-rot
        g = dq.Ry(nqubit=nq, wires=tgt_wire, controls=ctrl_wires, condition=True, requires_grad=True)
        g.init_para([0])
        vqc.add(g)
    vqc.rylayer()   # 最后一层需要初始化

  if not 'mera':    # n_layer=6, n_gate=280, fid=0.816463828086853
    for i in range(n_layer):
      for s in range(nq//2):
        vqc.rylayer(list(range(s, nq-s)))
        for k in range(s, nq-s, 2):
          vqc.cnot(k, k+1)
    vqc.rylayer()   # 最后一层需要初始化

  if not 'mera zero-init':    # n_layer=6, n_gate=280, fid=0.7790230512619019
    for i in range(n_layer):
      for s in range(nq//2):
        lyr = dq.RyLayer(nq, list(range(s, nq-s)))
        lyr.init_para([0] * (nq-2*s))
        vqc.add(lyr)
        for k in range(s, nq-s, 2):
          vqc.cnot(k, k+1)
    vqc.rylayer()   # 最后一层需要初始化

  if not 'mera ><':    # n_layer=6, n_gate=280, fid=0.8184420466423035
    for i in range(n_layer):
      is_odd = i % 2 == 1
      for s in (reversed if is_odd else list)(range(nq//2)):
        vqc.rylayer(list(range(s, nq-s)))
        for k in range(s, nq-s, 2):
          vqc.cnot(k, k+1)
    vqc.rylayer()   # 最后一层需要初始化

  if not 'mera ↓><↑':    # n_layer=6, n_gate=280, fid=0.8278782367706299 (lr=0.05)
    for i in range(n_layer):
      is_even = i % 2 == 0
      if is_even:
        for s in range(nq//2):
          vqc.rylayer(list(range(s, nq-s)))
          for k in range(s, nq-s, 2):
            vqc.cnot(k, k+1)
      else:
        for s in reversed(range(nq//2)):
          vqc.rylayer(list(range(s, nq-s)))
          for k in range(s, nq-s, 2):
            vqc.cnot(k+1, k)
    vqc.rylayer()   # 最后一层需要初始化

  if not 'swap-like':    # n_layer=15, n_gate=271, fid=0.9302663803100586
    vqc.x(0)
    for i in range(n_layer):
      for q in range(nq):
        vqc.cry(q, q+1)
        vqc.cry(q+1, q)

  # [normal_reshape]
  # n_layer=15, n_gate=271, fid=0.8838197588920593
  # n_layer=12, n_gate=241, fid=0.8564713001251221
  # [snake_reshape]
  # n_layer=15, n_gate=271, fid=0.9552421569824219
  # n_layer=12, n_gate=241, fid=0.9351020455360413
  if 'swap-like zero init':
    vqc.x(0)
    for i in range(n_layer):
      for q in range(nq):
        g = dq.Ry(nqubit=nq, wires=(q+1)%nq, controls=q, condition=True, requires_grad=True)
        g.init_para([0])
        vqc.add(g)
        g = dq.Ry(nqubit=nq, wires=q, controls=(q+1)%nq, condition=True, requires_grad=True)
        g.init_para([0])
        vqc.add(g)

  if not 'swap-like cyclic zero init':  # n_layer=15, n_gate=301, fid=0.9552421569824219
    vqc.x(0)
    for i in range(n_layer):
      for q in range(nq):
        g = dq.Ry(nqubit=nq, wires=(q+1)%nq, controls=q, condition=True, requires_grad=True)
        g.init_para([0])
        vqc.add(g)
        g = dq.Ry(nqubit=nq, wires=q, controls=(q+1)%nq, condition=True, requires_grad=True)
        g.init_para([0])
        vqc.add(g)

  if not 'swap-like cyclic zero init \/':   # n_layer=15, n_gate=273, fid=0.9490880966186523
    vqc.x(0)
    for i in range(n_layer):
      is_even = i % 2 == 0
      if is_even:
        for q in range(nq):   # outer, 10 qubits
          g = dq.Ry(nqubit=nq, wires=(q+1)%nq, controls=q, condition=True, requires_grad=True)
          g.init_para([0])
          vqc.add(g)
          g = dq.Ry(nqubit=nq, wires=q, controls=(q+1)%nq, condition=True, requires_grad=True)
          g.init_para([0])
          vqc.add(g)
      else:                   # inner, 8 qubits
        nnqq = nq - 2
        offset = 1
        for q in range(1, nq-1):
          g = dq.Ry(nqubit=nq, wires=(q-offset+1)%nnqq+offset, controls=q, condition=True, requires_grad=True)
          g.init_para([0])
          vqc.add(g)
          g = dq.Ry(nqubit=nq, wires=q, controls=(q-offset+1)%nnqq+offset, condition=True, requires_grad=True)
          g.init_para([0])
          vqc.add(g)

  return vqc


def run():
  circ = get_model(n_layer=12).to(device)
  print('gate count:', count_gates(circ))
  print('param count:', sum([p.numel() for p in circ.parameters()]))

  dataset = QMNISTDatasetIdea(label_list=[0], train=False, per_cls_size=10)

  for idx, (x, y, z_func) in enumerate(dataset):

    z = snake_reshape_norm_padding(x.unsqueeze(0), rev=True)
    #z = reshape_norm_padding(x.unsqueeze(0), use_hijack=False)

    x, y, z = x.to(device), y.to(device), z.to(device)
    optimizer = optim.Adam(circ.parameters(), lr=0.02)
    s = time()
    for i in range(N_ITER):
      state = circ().swapaxes(0, 1)     # [B=1, D=1024]
      loss = -get_fidelity(state, z)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if i % 10 == 0:
        print('fid:', -loss.item())
    t = time()
    state = circ().swapaxes(0, 1)
    print(f'[{idx}] >> Fidelity:', get_fidelity(state, z).item(), f'(time: {t - s})')

    break


if __name__ == '__main__':
  run()
