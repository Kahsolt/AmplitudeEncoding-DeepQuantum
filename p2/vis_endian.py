#!/usr/bin/env python3
# Author: Armit
# Create Time: 周六 2024/11/23 

# 测定端序
# - 大端序 |q0,q1,q2,...>，高位控制 = 小 qid 控制大 qid
# - 系统扩张：新增 qubits 应在右侧追加，即 |q0,q1,q2,...,a0,a1,a2,...>

import torch
import deepquantum as dq

# |0>
v0 = torch.tensor([1, 0], dtype=torch.float32)

qc = dq.QubitCircuit(3)
qc.ry(0, 0.233)
qc.cnot(0, 1)
psi = qc().real.T
print(psi)
qc = dq.QubitCircuit(2)
qc.ry(0, 0.233)
qc.cnot(0, 1)
psi = qc().real.T
print(psi)
print(torch.kron(psi, v0))    # <- match!
print(torch.kron(v0, psi))

print('=' * 42)

qc = dq.QubitCircuit(3)
qc.ry(1, 0.233)
qc.cnot(1, 0)
psi = qc().real.T
print(psi)
qc = dq.QubitCircuit(2)
qc.ry(1, 0.233)
qc.cnot(1, 0)
psi = qc().real.T
print(psi)
print(torch.kron(psi, v0))    # <- match!
print(torch.kron(v0, psi))
