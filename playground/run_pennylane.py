#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/21 

import torch
import numpy as np
import pennylane as qml
import pennylane.templates as T
from pennylane.tape import QuantumTape

from utils import QMNISTDatasetIdea

T.state_preparations.MottonenStatePreparation
T.state_preparations.ArbitraryStatePreparation


#nq = 2
#state = np.asarray([-np.sqrt(0.2), np.sqrt(0.5), -np.sqrt(0.1), np.sqrt(0.2)])
nq = 10
state = np.random.uniform(size=2**nq)
state = state / np.linalg.norm(state)

dataset = QMNISTDatasetIdea(label_list=[0,1,2,3,4], train=False, size=10, per_cls_size=2)
x, y, z = dataset[0]
state = z()[0].to(dtype=torch.complex128)

dev = qml.device('default.qubit', wires=nq)
#@qml.qnode(dev)
#def circuit(state):
#  qml.MottonenStatePreparation(state, wires=range(nq))
#  return qml.state()
#if nq <= 5:
#  print(circuit(state))
#  print(qml.draw(circuit, expansion_strategy="device", max_length=80)(state))
#  print()
ops = qml.MottonenStatePreparation.compute_decomposition(state, wires=range(nq))
print('ops:', len(ops))
print(ops[:30])

ops_new = []
for op in ops:
  skip = False
  if isinstance(op, qml.RY):
    if abs(op.data[0].item()) <= 4e-2:    # 0.8641
      skip = True
  if not skip:
    ops_new.append(op)

print('ops_new:', len(ops_new))
tape = QuantumTape(ops_new, measurements=[qml.state()])
state_approx = qml.execute([tape], dev)[0]
breakpoint()
fid = torch.abs(state.conj() @ state_approx) ** 2
print('fid:', fid.item())
