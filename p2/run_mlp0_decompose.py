#!/usr/bin/env python3
# Author: Armit
# Create Time: 周日 2024/11/10 

# 查看 MLP0 投影矩阵 (笑死，太太太太慢了分解不了一点)
# https://quantumcomputing.stackexchange.com/questions/13821/generate-a-3-qubit-swap-unitary-in-terms-of-elementary-gates/13826#13826

import torch
import numpy as np
from qiskit import transpile
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.generalized_gates import UnitaryGate

if not 'qiskit stuff':
  from qiskit.circuit.library.standard_gates import CU3Gate, U1Gate, U2Gate, U3Gate
  from qiskit.circuit.library.arithmetic.linear_amplitude_function import LinearAmplitudeFunction

if not 'qiskit-ML stuff':
  # qiskit_machine_learning 仅研究 VQC/QSVM 上的**二分类**问题，默认线路结构为 ZZFeatureMap + RealAmplitudes
  # VQC  的后端为 SamplerQNN，测量比特串的奇偶性解释为模型输出
  # QSVM 的后端为 sklearn.svm，使用量子核函数而已
  from qiskit.circuit.library.data_preparation import ZFeatureMap, ZZFeatureMap, StatePreparation   # encoder
  from qiskit.circuit.library.n_local import TwoLocal, EfficientSU2, RealAmplitudes                 # ansatz
  from qiskit_machine_learning.circuit.library import RawFeatureVector, QNNCircuit                  # algorithm / whole circuit
  from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN                      # backend/simulator/executor
  from qiskit_machine_learning.algorithms.classifiers import VQC, QSVC, PegasosQSVC                 # out-of-box app
  from qiskit_machine_learning.algorithms.regressors import VQR, QSVR


d = torch.load('output/mlp0-best.pth', map_location='cpu')
U, _ = torch.linalg.qr(d['U_holder'])
print('U.shape:', U.shape)
diff = (U @ U.T - np.eye(4096))
assert np.abs(diff).max() < 1e-5, 'not unitary'
U: np.ndarray = U.cpu().numpy()


if 'eig':
  # U 是正交阵，因此解是圆周上的单位虚根。。。
  E, V = np.linalg.eig(U[:3072, :3072])


if not 'decompose':
  nq = 12
  circuit = QuantumCircuit(nq)
  circuit.append(UnitaryGate(U, check_input=False), list(range(nq)))

  print('>> start transpile()')
  new_circuit = transpile(circuit, basis_gates=['cu3', 'u3'])
  print('>> finish transpile()')

  new_circuit.draw('mpl')
  breakpoint()
