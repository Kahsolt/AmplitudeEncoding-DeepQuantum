#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/21 

import numpy as np
from pyqpanda import CPUQVM, amplitude_encode, Encode

#nq = 2
#state = np.asarray([-np.sqrt(0.2), np.sqrt(0.5), -np.sqrt(0.1), np.sqrt(0.2)])

nq = 4
state = np.random.uniform(size=2**nq)
state = state / np.linalg.norm(state)

qvm = CPUQVM()
qvm.init_qvm()
qv = qvm.qAlloc_many(nq)

# 等价于 Encode.amplitude_encode_recursive
qcir = amplitude_encode(qv, state)
print(qcir)

'''
https://pyqpanda-toturial.readthedocs.io/zh/latest/13.%E9%87%8F%E5%AD%90%E7%AE%97%E6%B3%95%E7%BB%84%E4%BB%B6/index.html#dc_amplitude_encode

[1] Schuld, Maria. "Quantum machine learning models are kernel methods."[J] arXiv:2101.11020 (2021).
[2] Araujo I F, Park D K, Ludermir T B, et al. "Configurable sublinear circuits for quantum state preparation."[J]. arXiv preprint arXiv:2108.10182, 2021.
[3] Ghosh K. "Encoding classical data into a quantum computer"[J]. arXiv preprint arXiv:2107.09155, 2021.
[4] Rudolph M S, Chen J, Miller J, et al. Decomposition of matrix product states into shallow quantum circuits[J]. arXiv preprint arXiv:2209.00595, 2022.
[5] de Veras T M L, da Silva L D, da Silva A J. "Double sparse quantum state preparation"[J]. arXiv preprint arXiv:2108.13527, 2021.
[6] Malvetti E, Iten R, Colbeck R. "Quantum circuits for sparse isometries"[J]. Quantum, 2021, 5: 412.
[7] N. Gleinig and T. Hoefler, "An Efficient Algorithm for Sparse Quantum State Preparation," 2021 58th ACM/IEEE Design Automation Conference (DAC), 2021, pp. 433-438, doi: 10.1109/DAC18074.2021.9586240.
[8] Havlíček, Vojtěch, et al. "Supervised learning with quantum-enhanced feature spaces." Nature 567.7747 (2019): 209-212.
'''

# 朴素做法，用了多比特门控，但没有用格雷码来尽可能对消控制位上的 X 门
encoder = Encode()
encoder.amplitude_encode(qv, state)
print('[amplitude_encode]')
print(encoder.get_circuit())

# 改良做法，控制位上的一组 X 门合并为一个多比特受控 X 门
encoder = Encode()
encoder.amplitude_encode_recursive(qv, state)
print('[amplitude_encode_recursive]')
print(encoder.get_circuit())

#encoder = Encode()
#encoder.bid_amplitude_encode(qv, state, split=2)
#print('[bid_amplitude_encode]')
#print(encoder.get_circuit())

#encoder = Encode()
#encoder.dc_amplitude_encode(qv, state)
#print('[dc_amplitude_encode]')
#print(encoder.get_circuit())

# 基本门很多，优点是仅使用基本门 RX/RY/RZ/CNOT/U1/U3
encoder = Encode()
encoder.schmidt_encode(qv, state, 0)
print('[schmidt_encode]')
print(encoder.get_circuit())

# 线路很比 schmidt_encode 还长，仅使用基本门 RX/RY/RZ/CNOT/U1/U3
encoder = Encode()
encoder.approx_mps(qv, state)
print('[approx_mps]')
print(encoder.get_circuit())

#encoder = Encode()
#encoder.ds_quantum_state_preparation(qv, state)
#print('[ds_quantum_state_preparation]')
#print(encoder.get_circuit())

# 在向量稠密的情况下基本等价于 amplitude_encode，稀疏时结构也差不多
encoder = Encode()
encoder.sparse_isometry(qv, state)
print('[sparse_isometry]')
print(encoder.get_circuit())

# 使用 X/CNOT/mctrl-U3
encoder = Encode()
encoder.efficient_sparse(qv, state)
print('[efficient_sparse]')
print(encoder.get_circuit())
