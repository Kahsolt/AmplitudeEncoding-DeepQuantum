#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/21 

from typing import List, Union, Tuple

import torch
import numpy as np
import deepquantum as dq
from utils import count_gates, QMNISTDatasetIdea, normalize, denormalize, reshape_norm_padding


class AmpTree:

  '''
  概率幅分配二叉树：
    1                     |  ctrl_bit
    |     \               |
    0.7        0.3        |     - 
    |  \       |  \       |
    0.2  0.5   0.1  0.2   |  highest(1)
  -------------------------------------
    |    |     |    |
  -√0.2 √0.5 -√0.1 √0.2   <- 各项振幅 (带符号)
  '''

  def __init__(self, amp:List[float]):
    nlen = len(amp)
    assert nlen & (nlen - 1) == 0, 'coeff length is not power of 2'
    assert np.isclose(sum([e**2 for e in amp]), 1.0)

    self.amp = amp
    self.nq = int(np.floor(np.log2(nlen)))
    self.tree: List[List[float]] = None
    self.sign: List[List[float]] = None
    self._build_tree()

  def _build_tree(self):
    # prob
    tree = []
    sign = []
    tree.append([e**2 for e in self.amp])
    sign.append([(1 if e >= 0 else -1) for e in self.amp])
    for _ in range(self.nq):
      base_layer = tree[0]
      up_layer = [base_layer[i] + base_layer[i+1] for i in range(0, len(base_layer), 2)]
      up_layer = to_amp(up_layer)   # renorm to fix accumulating floating error :(
      tree.insert(0, up_layer)
      sign_layer = [1 for _ in range(0, len(base_layer), 2)]
      sign.insert(0, sign_layer)
    # amp
    for l in range(len(tree)):
      layer = tree[l]
      for i in range(len(layer)):
        layer[i] = np.sqrt(layer[i])
    # fix sign of middle layers
    for q in range(1, self.nq):
      for c in range(2**q):
        ok, sig = self.is_sub_all_eqv(q, c)
        if ok and sig < 0:
          sign[q][c] = -1
    self.tree = tree
    self.sign = sign

  def get_split_RY_angle(self, idx:int, cut:int, eps:float=1e-5) -> float:
    assert idx >= 0 and (cut >= 0 and cut < 2 ** idx), f'idx={idx} cut={cut}'
    tree = self.tree[idx + 1]    # shift by 1
    sign = self.sign[idx + 1]    # shift by 1
    sub_prob = [e**2 for e in tree[2*cut:2*cut+2]]

    # case 0: 零概率 no-op
    # case 1: 等概率 |0> + |1> -> tht=1.5707963267948966 -> H
    # case 2: 不等概率，两个极端 |0> -> tht=0 -> I, |1> -> tht=pi -> X
    sub_prob_sum = sum(sub_prob)
    if abs(sub_prob_sum) < eps: return 0
    sub_amp = [np.sqrt(e / sub_prob_sum) for e in sub_prob]
    # arccos(θ/2) = amp, where amp related to |0>
    tht = 2 * np.arccos(sub_amp[0])     # vrng [0, pi]
    # 确定符号 (BUG: ...)
    sx, sy = sign[2*cut: 2*cut+2]
    # (+, +), [0, pi/2]
    if   sx > 0 and sy > 0: return tht
    # (+, -), [pi/2, pi]
    elif sx > 0 and sy < 0: return -tht
    # (-, -), [pi, 3*pi/2]
    elif sx < 0 and sy < 0: return -2 * np.pi + tht
    # (-, +), [3*pi/2, 2*pi]
    elif sx < 0 and sy > 0: return 2 * np.pi - tht

  def is_sub_all_eqv(self, idx:int, cut:int, std_thresh:float=1e-3) -> Tuple[bool, int]:
    assert idx >= 0 and (cut >= 0 and cut < 2 ** idx), f'idx={idx} cut={cut}'
    nlen = 1              # 覆盖区间长度
    while idx < self.nq:  # 下沉
      cut *= 2
      nlen *= 2
      idx += 1
    ok = np.std(self.amp[cut : cut + nlen]) <= std_thresh
    #if ok: print(self.amp[cut : cut + nlen])
    return ok, np.sign(self.amp[cut])

  def __getitem__(self, idx:Union[int, tuple]) -> Union[List[float], float]:
    if isinstance(idx, int):
      return self.tree[idx]
    if isinstance(idx, tuple):
      assert len(idx) == 2
      layer, colum = idx
      return self.tree[layer][colum]


# ~pennylane.templates.state_preparations.mottonen.py
def gray_code(rank):
  """Generates the Gray code of given rank.
  Args:
      rank (int): rank of the Gray code (i.e. number of bits)
  """
  def gray_code_recurse(g, rank):
    k = len(g)
    if rank <= 0: return
    for i in range(k - 1, -1, -1):
      char = "1" + g[i]
      g.append(char)
    for i in range(k - 1, -1, -1):
      g[i] = "0" + g[i]
    gray_code_recurse(g, rank - 1)
  g = ["0", "1"]
  gray_code_recurse(g, rank - 1)
  return g


def amplitude_encode(amp:List[float], encode:bool=False, eps:float=1e-3, gamma:float=2e-2) -> dq.QubitCircuit:
  at = AmpTree(amp)
  qc = dq.QubitCircuit(nqubit=at.nq)
  isclose = lambda x, y: np.isclose(x, y, atol=eps)

  # tree visit mark
  flag: List = []

  # divide-and-conquer
  tht = at.get_split_RY_angle(0, 0)
  flag.append((0, 0)) # 标记
  if abs(tht) < eps:
    pass
  elif isclose(abs(tht), np.pi):
    qc.x(0)
  elif isclose(tht, np.pi/2):
    qc.h(0)
  else:
    #qc.ry(0, tht, encode=encode)
    g = dq.gate.Ry(nqubit=at.nq, wires=0, requires_grad=True)
    g.init_para([tht])
    qc.add(g, encode=encode)

  for q in range(1, at.nq):
    mctrl = list(range(q))
    is_leaf_layer = q == at.nq - 1
    for cond in gray_code(q):
      c = int(cond, base=2)
      if (q, c) in flag: continue

      # 非叶子层且其下叶子全部均权，可剪枝
      can_trim, sig = at.is_sub_all_eqv(q, c, std_thresh=gamma)
      if not is_leaf_layer and can_trim:
        nlen = 1
        cut = c
        idx = q
        while idx < at.nq:  # 子代全部标记
          for cc in range(cut, cut + nlen):
            flag.append((idx, cc))
          cut *= 2
          nlen *= 2
          idx += 1

        for t, v in enumerate(cond):
          if v == '0': qc.x(t)
        for idx in range(q, at.nq):
          qc.h(idx, controls=mctrl, condition=True)
        for t, v in enumerate(cond):
          if v == '0': qc.x(t)

      # 其下叶子非均权
      else:
        tht = at.get_split_RY_angle(q, c)
        flag.append((q, c)) # 标记
        if abs(tht) < eps: continue

        for t, v in enumerate(cond):
          if v == '0': qc.x(t)
        if isclose(abs(tht), np.pi):
          qc.x(q, controls=mctrl, condition=True)
        elif isclose(tht, np.pi/2):
          qc.h(q, controls=mctrl, condition=True)
        else:
          #qc.ry(q, tht, controls=mctrl, condition=True, encode=encode)
          g = dq.gate.Ry(nqubit=at.nq, wires=q, controls=mctrl, condition=True, requires_grad=True)
          g.init_para([tht])
          qc.add(g, encode=encode)
        for t, v in enumerate(cond):
          if v == '0': qc.x(t)

  # cancel inverses X gate
  ops = qc.operators
  ops_new: List[dq.operation.Operation] = []
  for op in ops:
    cancelled = False
    if isinstance(op, dq.PauliX) and len(ops_new) > 0:
      j = len(ops_new) - 1
      while j >= 0:
        op_X = ops_new[j]
        if not isinstance(op_X, dq.PauliX): break
        if len(op_X.controls): break   # DO NOT cross mctrl-CNOT
        if op_X.wires == op.wires and op_X.controls == op.controls:
          ops_new.pop(j)
          cancelled = True
        j -= 1
    if not cancelled:
      ops_new.append(op)
  #print(f'prune circuit: {len(ops)} -> {len(ops_new)}')

  # reconstruct
  qc_new = dq.QubitCircuit(nqubit=at.nq)
  for op in ops_new:
    op.requires_grad = True
    qc_new.add(op)
  return qc_new


def to_amp(coeffs:List[float]) -> List[float]:
  amp = np.asarray(coeffs)
  amp = amp / np.linalg.norm(amp)
  return amp.tolist()


def test_amp_tree():
  amp = [-np.sqrt(0.2), np.sqrt(0.5), -np.sqrt(0.1), np.sqrt(0.2)]
  print('amp:', amp)
  at = AmpTree(amp)
  print('tree:')
  for layer in at.tree:
    print(layer)
  print('sign(amp):', at.amp_sign)

  print('at[2]:', at[2])
  print('at[2,1]:', at[2, 1])
  print()

  tht00 = at.get_split_RY_angle(0, 0)
  tht10 = at.get_split_RY_angle(1, 0)
  tht11 = at.get_split_RY_angle(1, 1)
  print('tht00:', tht00)
  print('tht10:', tht10)
  print('tht11:', tht11)
  print()

  circ = dq.QubitCircuit(nqubit=2)
  circ.ry(0, tht00)
  circ.x(0)
  circ.ry(1, tht10, controls=0, condition=True)
  circ.x(0)
  circ.ry(1, tht11, controls=0, condition=True)
  print('state:', circ().real.flatten())

  qc = amplitude_encode(amp)
  print('state:', qc().real.flatten())
  assert torch.allclose(circ(), qc())


def test_amplitude_encode(amp:List[float], eps:float=1e-3, gamma:float=1e-2) -> dq.QubitCircuit:
  state = amp / np.linalg.norm(amp)
  qc = amplitude_encode(state, eps=eps, gamma=gamma)
  print('gate count:', count_gates(qc))
  state_hat = qc().real.flatten().numpy()
  fidelity = np.abs(state_hat @ state)**2
  print('fidelity:', fidelity)
  #if fidelity < 0.75: breakpoint()
  return qc

def test_amplitude_encode_mnist(qt:int=None, eps:float=1e-3, gamma:float=1e-2):
  dataset = QMNISTDatasetIdea(label_list=[0,1,2,3,4], train=False, size=5, per_cls_size=1)
  for x, y, z in dataset:
    if qt:
      x = denormalize(x)
      x = (x - x.min()) / (x.max() - x.min())
      x = ((x * 255 / qt).round() * qt) / 255
      x = normalize(x)
      state = reshape_norm_padding(x.unsqueeze(0)).real.flatten().numpy().tolist()
    else:
      state = z().real.flatten().numpy()
    test_amplitude_encode(state, eps=eps, gamma=gamma)


if __name__ == '__main__':
  if not 'test sanity':
    test_amp_tree()

  print()

  if 'test specific':
    print('test_amplitude_encode_eqv(4)')
    test_amplitude_encode(np.ones(2**4))
    test_amplitude_encode([1,-2,3,3,0,0,0,0])
    test_amplitude_encode([1,2,3,3,0,0,5,6])
    test_amplitude_encode([1,1,1,1,-3,-3,-3,-3])
    test_amplitude_encode([1,1,2,-2,-3,-3,-4,4])
    test_amplitude_encode([1,-1,-2,-2,3,3,-4,4])

  print()

  if 'test rand':
    print('test_amplitude_encode_rand(3)')
    for _ in range(30):
      test_amplitude_encode(np.random.uniform(low=-1, high=1, size=2**3))
    # 9197 -> 2059
    print('test_amplitude_encode_rand(10)')
    test_amplitude_encode(np.random.uniform(low=-1, high=1, size=2**10))

  print()

  if 'test dataset':
    print('test_amplitude_encode_mnist()')
    test_amplitude_encode_mnist(qt=None)
    test_amplitude_encode_mnist(qt=4)
    test_amplitude_encode_mnist(qt=8)
    test_amplitude_encode_mnist(qt=16)
    test_amplitude_encode_mnist(qt=32)
    test_amplitude_encode_mnist(qt=64)
    test_amplitude_encode_mnist(qt=128)
