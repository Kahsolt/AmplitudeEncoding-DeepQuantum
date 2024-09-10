#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/10 

# 查看线路的数学结构

import sympy as sp
from sympy import Symbol as s
from sympy import sin, cos, simplify

import numpy as np
from numpy import ndarray

if 'syntax hijack':
  class float(float):
    def __xor__(self, other):
      # NOTE: the operator priority of __xor__ (^) is lower than __pow__ (**) in Python
      # so just mind your expression order :(
      return other.__rpow__(self)

  class array(ndarray):
    @property
    def dagger(self):
      return self.conj().T

  π = np.pi
  e = float(np.e)
  i = np.complex64(0 + 1j)    # imaginary unit
  sqrt2 = np.sqrt(2)          # √2

Is = { }    # identity gate caching
def get_I(n: int) -> ndarray:
  if n not in Is: Is[n] = I @ n
  return Is[n]

def Control(u: ndarray) -> ndarray:   # 高控制低
  h, w = u.shape
  assert h == w
  nq = int(np.log2(h)) + 1
  v = np.eye(2**nq, dtype=u.dtype)
  v[-h:, -w:] = u
  return v


# 仅列出 DeepQuantum 中 QubitCircuit 所直接支持的门 (不含 phontic 包)
# https://dqapi.turingq.com/deepquantum.html#module-deepquantum.gate

U3 = lambda θ, φ, λ: np.asarray([
  [        cos(θ/2), -(e^(i*   λ) *sin(θ/2))],
  [e^(i*φ)*sin(θ/2),   e^(i*(φ+λ))*cos(θ/2)],
])
P = lambda θ: np.asarray([
  [1, 0],
  [0, e^(i*θ)],
])
I = np.asarray([
  [1, 0],
  [0, 1],
])
X = np.asarray([
  [0, 1],
  [1, 0],
])
Y = np.asarray([
  [0, -i],
  [i,  0],
])
Z = np.asarray([
  [1,  0],
  [0, -1],
])
H = np.asarray([
  [1,  1],
  [1, -1],
]) / sqrt2
S = np.asarray([
  [1, 0],
  [0, i],
])
T = np.asarray([
  [1, 0],
  [0, e^(i*π/4)],
])
RX = lambda θ: np.asarray([
  [cos(θ/2), -i*sin(θ/2)],
  [-i*sin(θ/2), cos(θ/2)],
])
RY = lambda θ: np.asarray([
  [cos(θ/2), -sin(θ/2)],
  [sin(θ/2),  cos(θ/2)],
])
RZ = lambda θ: np.asarray([
  [e^(-i*θ/2), 0],
  [0, e^(i*θ/2)],
])
CNOT = CX = np.asarray([       # make entanglement
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 0, 1],
  [0, 0, 1, 0],
])
rCNOT = np.asarray([           # reversed CNOT
  [1, 0, 0, 0],
  [0, 0, 0, 1],
  [0, 0, 1, 0],
  [0, 1, 0, 0],
])
SWAP = np.asarray([
  [1, 0, 0, 0],
  [0, 0, 1, 0],
  [0, 1, 0, 0],
  [0, 0, 0, 1],
])
RXX = lambda φ: np.asarray([   # exp(-i*(φ/2)*(X@X))
  [cos(φ/2), 0, 0, -i*sin(φ/2)],
  [0, cos(φ/2), -i*sin(φ/2), 0],
  [0, -i*sin(φ/2), cos(φ/2), 0],
  [-i*sin(φ/2), 0, 0, cos(φ/2)],
])
RYY = lambda φ: np.asarray([   # exp(-i*(φ/2)*(Y@Y))
  [cos(φ/2), 0, 0, i*sin(φ/2)],
  [0, cos(φ/2), -i*sin(φ/2), 0],
  [0, -i*sin(φ/2), cos(φ/2), 0],
  [i*sin(φ/2), 0, 0, cos(φ/2)],
])
RZZ = lambda φ: np.asarray([   # exp(-i*(φ/2)*(Z@Z))
  [e^(-i*φ/2), 0, 0, 0],
  [0, e^(i*φ/2), 0, 0],
  [0, 0, e^(i*φ/2), 0],
  [0, 0, 0, e^(-i*φ/2)],
])
RXY = lambda φ: np.asarray([   # exp(-i*(φ/2)*(Y@Y))
  [1, 0, 0, 0],
  [0, cos(φ/2), -i*sin(φ/2), 0],
  [0, -i*sin(φ/2), cos(φ/2), 0],
  [0, 0, 0, 1],
])
RBS = lambda φ: np.asarray([   # aka. SWAP(θ)?
  [1, 0, 0, 0],
  [0,  cos(φ), sin(φ), 0],
  [0, -sin(φ), cos(φ), 0],
  [0, 0, 0, 1],
])
CCNOT = CCX = Toffoli = np.asarray([
  [1, 0, 0, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 1, 0],
])
CSWAP = Fredkin = np.asarray([
  [1, 0, 0, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 1],
])


''' Playground '''
if 'xCRY':
  # ↓CRY - ↑CRY (低控制高)
  xCRY = Control(RY(s('a1'))) @ (SWAP @ Control(RY(s('a0'))) @ SWAP)
  print('[xCRY]')
  print(xCRY)

  # ↑CRY - ↓CRY (高控制低)
  rxCRY = (SWAP @ Control(RY(s('a1'))) @ SWAP) @ Control(RY(s('a0')))
  print('[rxCRY]')
  print(rxCRY)

  # ↓CRY - ↑CRY - ↓CRY (低控制高)
  xCRY3 = (SWAP @ Control(RY(s('a2'))) @ SWAP) @ Control(RY(s('a1'))) @ (SWAP @ Control(RY(s('a0'))) @ SWAP)
  print('[xCRY3]')
  print(xCRY3)

  # ↑CRY - ↓CRY - ↑CRY (高控制低)
  rxCRY3 = Control(RY(s('a2'))) @ (SWAP @ Control(RY(s('a1'))) @ SWAP) @ Control(RY(s('a0')))
  print('[rxCRY3]')
  print(rxCRY3)

if not '↓CU3':
  # ↓CU3 - ↑CU3 (低控制高)
  xCU3 = Control(U3(s('a10'), s('a11'), s('a12'))) @ (SWAP @ Control(U3(s('a00'), s('a01'), s('a02'))) @ SWAP)
  print('[xCU3]')
  print(xCU3)

  # ↑CU3 - ↓CU3 (高控制低)
  rxCU3 = (SWAP @ Control(U3(s('a00'), s('a01'), s('a02'))) @ SWAP) @ Control(U3(s('a10'), s('a11'), s('a12')))
  print('[rxCU3]')
  print(rxCU3)

if not 'xRBS':
  xRBS = (SWAP @ RBS(s('a1')) @ SWAP) @ RBS(s('a0'))
  print('[xRBS]')
  print(xRBS)
