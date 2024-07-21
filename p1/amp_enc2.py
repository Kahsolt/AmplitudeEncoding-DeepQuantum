#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/22 

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import deepquantum as dq
from utils import count_gates, QMNISTDatasetIdea, normalize, denormalize, reshape_norm_padding


def to_amp(coeffs:List[float]) -> List[float]:
  amp = np.asarray(coeffs)
  amp = amp / np.linalg.norm(amp)
  return amp.tolist()

def sign(x:int):
  if x == 0: return  0
  if x  > 0: return  1
  if x  < 0: return -1


@dataclass
class Node:
  type: str   # ['RY', 'I', 'X', 'H', 'H*', '-', None]
  val: float
  sig: int
  flag: bool = False

  @property
  def is_leaf(self):
    return self.type is None
  
  def __repr__(self):
    return f'<{self.type} v={self.val:.4f} s={self.sig} f={int(self.flag)}>'


class AmpTreeEx:

  def __init__(self, coeff:List[float], eps:float=1e-3, gamma:float=0.01):

    nlen = len(coeff)
    assert nlen & (nlen - 1) == 0, 'coeff length is not power of 2'
    amp = to_amp(coeff)
    assert np.isclose(np.linalg.norm(amp), 1.0)

    self.amp = amp
    self.eps = eps
    self.gamma = gamma
    self.nq = int(np.floor(np.log2(nlen)))
    self.tree: List[List[Node]] = None
    self._build_tree()
    self._override_H_star_recursive(0, len(self.amp))

  def _build_tree(self):
    isclose = lambda x, y: np.isclose(x, y, atol=self.eps)

    tree: List[List[Node]] = []
    tree.append([Node(None, abs(e), sign(e)) for e in self.amp])
    for _ in range(self.nq):
      # renorm
      base_layer = tree[0]
      vals = to_amp([n.val for n in base_layer])
      for i, n in enumerate(base_layer):
        n.val = vals[i]
      # re-distro
      up_layer: List[Node] = []
      for i in range(0, len(base_layer), 2):
        x = base_layer[i]
        y = base_layer[i+1]
        x_is_0 = isclose(x.val, 0)
        y_is_0 = isclose(y.val, 0)
        val_sum = np.sqrt(x.val**2 + y.val**2)
        if x_is_0 and y_is_0:
          node = Node('-', 0, 1)
        elif y_is_0:
          node = Node('I', x.val, x.sig)
        elif x_is_0:
          node = Node('X', y.val, y.sig)
        elif isclose(x.val, y.val) and x.sig == y.sig:
          node = Node('H', val_sum, x.sig)
        else:
          node = Node('RY', val_sum, 1)
        up_layer.append(node)
      tree.insert(0, up_layer)
    self.tree = tree

  def _override_H_star_recursive(self, L:int, R:int):
    ''' 尝试把一棵子树的根替换为 H* 节点，并重置子节点类型、直接标记为已完成 '''
    if np.std(self.amp[L:R]) <= self.gamma:
      layer = self.nq
      nlen = R - L + 1
      rank = L
      while nlen > 1:
        layer -= 1
        nlen //= 2
        rank //= 2
      node = self.tree[layer][rank]
      node.type = 'H*'
      node.sig = sign(np.mean(self.amp[L:R]))
      self.reset_H_star_children(layer, rank)
    else:
      M = (L + R) // 2
      if L + 1 < M: self._override_H_star_recursive(L, M)
      if M < R - 1: self._override_H_star_recursive(M, R)

  def reset_H_star_children(self, layer:int, rank:int):
    try:
      l, r = layer + 1, rank * 2
      lchild = self.tree[l][r]
      lchild.type = '-'
      lchild.flag = True
      self.reset_H_star_children(l, r)
    except IndexError: pass
    try:
      l, r = layer + 1, rank * 2 + 1
      rchild = self.tree[l][r]
      rchild.type = '-'
      rchild.flag = True
      self.reset_H_star_children(l, r)
    except IndexError: pass

  def mark_H_star_childs(self):
    ''' 若父节点被标记为 H*，把所有子节点重置为 - 节点 '''
    pass

  def process_node(self, layer:int, rank:int) -> Tuple[str, float]:
    assert (0 <= layer < self.nq) and (0 <= rank < 2 ** layer), f'invalid layer={layer} rank={rank}'
    root   = self.tree[layer]    [rank]
    lchild = self.tree[layer + 1][rank * 2]
    rchild = self.tree[layer + 1][rank * 2 + 1]
    if root.flag: return None, None   # have done!
    root.flag = True

    if root.type == '-':   # case 0: 零概率 no-op
      return None, None
    if root.type == 'I':   # case 1: 默认基态 no-op
      return None, None
    if root.type == 'X':   # case 2: 基态翻转
      return 'X', None
    if root.type == 'H':   # case 3: 等概率
      return 'H', None
    if root.type == 'RY':  # case 4: 不等概率
      x, y = to_amp([lchild.val, rchild.val])
      sx, sy = lchild.sig, rchild.sig
      # x|0>+y|1>, x=arccos(θ/2)
      tht = 2 * np.arccos(x)  # vrng [0, pi]
      '''
        (+, +)  [0, pi/2]
        (+, -)  [pi/2, pi]
        (-, -)  [pi, 3*pi/2]
        (-, +)  [3*pi/2, 2*pi]
      '''
      if   sx > 0 and sy > 0: return 'RY', tht
      elif sx > 0 and sy < 0: return 'RY', -tht
      elif sx < 0 and sy < 0: return 'RY', tht - 2 * np.pi
      elif sx < 0 and sy > 0: return 'RY', 2 * np.pi - tht
    if root.type == 'H*':  # case 3: 近似等概率递归
      return 'H*', list(range(layer, self.nq))

    raise TypeError(f'>> bad root type: {root.type}')

  def print_tree(self, include_leaf:bool=False):
    for q in range(self.nq + int(include_leaf)):
      print(self.tree[q])
    print()


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


def amplitude_encode(amp:List[float], encode:bool=False, eps:float=1e-3, gamma:float=0.01, limit_depth:int=-1) -> dq.QubitCircuit:
  at = AmpTreeEx(amp, eps, gamma)
  qc = dq.QubitCircuit(nqubit=at.nq)

  # divide-and-conquer
  name, args = at.process_node(0, 0)
  if name is None: pass
  elif name == 'X': qc.x(0)
  elif name == 'H': qc.h(0)
  elif name == 'RY':
    g = dq.gate.Ry(nqubit=at.nq, wires=0, requires_grad=True)
    g.init_para([args])
    qc.add(g, encode=encode)
  elif name == 'H*':
    for q in args:
      qc.h(q)
  else: raise TypeError(name, args)

  for q in range(1, at.nq):
    if limit_depth > 0 and q >= limit_depth: break
    mctrl = list(range(q))
    for cond in gray_code(q):
      c = int(cond, base=2)
      name, args = at.process_node(q, c)
      if name is None: continue

      # cond
      for t, v in enumerate(cond):
        if v == '0': qc.x(t)
      # mctrl-rot
      if name == 'X':
        qc.x(q, controls=mctrl, condition=True)
      elif name == 'H':
        qc.h(q, controls=mctrl, condition=True)
      elif name == 'RY':
        g = dq.gate.Ry(nqubit=at.nq, wires=q, controls=mctrl, condition=True, requires_grad=True)
        g.init_para([args])
        qc.add(g, encode=encode)
      elif name == 'H*':
        for qq in args:
          qc.h(qq, controls=mctrl, condition=True)
      else:
        raise TypeError(name, args)
      # uncond
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


def test_amplitude_encode(amp:List[float], eps:float=1e-3, gamma:float=1e-2) -> dq.QubitCircuit:
  state = amp / np.linalg.norm(amp)
  qc = amplitude_encode(state, eps=eps, gamma=gamma)
  print('gate count:', count_gates(qc))
  state_hat = qc().detach().real.flatten().numpy()
  fidelity = np.abs(state_hat @ state)**2
  print('fidelity:', fidelity)
  if fidelity < 0.75: breakpoint()
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
      state = z().detach().real.flatten().numpy()
    test_amplitude_encode(state, eps=eps, gamma=gamma)


if __name__ == '__main__':
  if not 'test AmpTreeEx':
    AmpTreeEx(np.ones(2**4)).print_tree()
    AmpTreeEx([1,-2,3,3,0,0,0,0]).print_tree()
    AmpTreeEx([1,2,3,3,0,0,5,6]).print_tree()
    AmpTreeEx([1,1,1,1,-3,-3,-3,-3]).print_tree()
    AmpTreeEx([1,1,2,-2,-3,-3,-4,4]).print_tree()
    AmpTreeEx([1,-1,-2,-2,3,3,-4,4]).print_tree()

  if not 'test amplitude encode':
    test_amplitude_encode(np.ones(2**4))
    test_amplitude_encode([1,-2,3,3,0,0,0,0])
    test_amplitude_encode([1,2,3,3,0,0,5,6])
    test_amplitude_encode([1,1,1,1,-3,-3,-3,-3])
    test_amplitude_encode([1,1,2,-2,-3,-3,-4,4])
    test_amplitude_encode([1,-1,-2,-2,3,3,-4,4])

  if 'test rand':
    print('test_amplitude_encode_rand(4)')
    for _ in range(30):
      test_amplitude_encode(np.random.uniform(low=-1, high=1, size=2**4))
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
