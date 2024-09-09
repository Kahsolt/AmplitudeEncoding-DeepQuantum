#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/09 

# 后处理 pkl: 微小的性能优化...

from typing import List
import pickle
import deepquantum as dq
from utils import QMNISTDataset

OUTPUT_DIR = './output'
EPS = 1e-1

'''
from fractions import Fraction
from numpy import pi
import pyzx as zx
from pyzx.graph.graph_s import GraphS

def dq_to_xz(qc:dq.QubitCircuit) -> zx.Circuit:
  c = zx.Circuit(qc.nqubit)
  for op in qc.operators:
    if op.name == 'PauliX':
      c.add_gate('NOT', op.wires[0])
    elif op.name == 'Ry':
      c.add_gate('CRY', op.controls[0], op.wires[0], Fraction(round(op.theta.item() / pi, 4)))
    else:
      print('unknown op:', op)
  return c

def xz_to_dq(c:zx.Circuit) -> dq.QubitCircuit:
  qc = dq.QubitCircuit(c.qubits)
  for g in c:
    if   g.name == 'NOT':    qc.x(g.target)
    elif g.name == 'HAD':    qc.h(g.target)
    elif g.name == 'XPhase': qc.rx(g.target, float(g.phase)*pi)
    elif g.name == 'YPhase': qc.ry(g.target, float(g.phase)*pi)
    elif g.name == 'ZPhase': qc.rz(g.target, float(g.phase)*pi)
    elif g.name == 'CRY':    qc.ry(g.target, float(g.phase)*pi, controls=g.control)
    elif g.name == 'SWAP':   qc.swap([g.target, g.control])
    elif g.name == 'CZ':     qc.cz  (g.target, g.control)
    elif g.name == 'CNOT':   qc.cnot(g.target, g.control)
    else:
      print('unknown g:', g)
      breakpoint()
  return qc

def xz_simplify(c:zx.Circuit) -> zx.Circuit:
  g: GraphS = c.to_graph()
  zx.full_reduce(g, quiet=True)
  return zx.extract_circuit(g.copy())
'''

# 数据集
fp = f'{OUTPUT_DIR}/test_dataset.tmp.pkl'   # debug use
#fp = f'{OUTPUT_DIR}/test_dataset.pkl'
print(f'>> loading from {fp}')
with open(fp, 'rb') as file:
  test_dataset: QMNISTDataset = pickle.load(file)

for i, (x, y, z) in enumerate(test_dataset):
  z: dq.QubitCircuit

  # no grad
  z.zero_grad()
  for op in z.operators:
    if hasattr(op, 'theta'):
      op.theta.requires_grad = False

  # prune rot gates
  n_ops_old = len(z.operators)
  ops: List[dq.operation.Operation] = []
  for op in z.operators:
    if not hasattr(op, 'theta'):
      ops.append(op)
    else:
      v = abs(op.theta.item())
      if v > EPS:
        ops.append(op)
      else:
        print(f'    trim small v: {v}')
  n_ops_new = len(ops)
  print(f'>> [trim] n_ops: {n_ops_old} => {n_ops_new}')
  if n_ops_new != n_ops_old:
    qc_new = dq.QubitCircuit(nqubit=z.nqubit)
    for op in ops:
      op.requires_grad = False
      qc_new.add(op)
    test_dataset.data_list[i] = (x, y, qc_new)
    z = qc_new

  # pyxz simplify
  if not 'pyxz simplify':
    n_ops_old = len(z.operators)
    qc_new = xz_to_dq(xz_simplify(dq_to_xz(z)))
    n_ops_new = len(qc_new.operators)
    print(f'>> [pyxz] n_ops: {n_ops_old} => {n_ops_new}')
    if n_ops_new < n_ops_old:
      test_dataset.data_list[i] = (x, y, qc_new)
      z = qc_new

fp = f'{OUTPUT_DIR}/test_dataset-prune.pkl'
print(f'>> saving to {fp}')
with open(fp, 'wb') as file:
  pickle.dump(test_dataset, file)
