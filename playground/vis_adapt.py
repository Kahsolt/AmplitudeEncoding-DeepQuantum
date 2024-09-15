#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/16 

# 探究单门级别的 ADAPT 线路构造在真实稀疏MNIST上能省多少个门
# 以 vqc_F1_all_wise_init (gcnt=166) + rand_mnist_freq + n_iter=500 为基准

import pickle as pkl
from argparse import ArgumentParser
from typing import List

from vis_universality import *

nq = 10   # FIXME: magic


def run_optim_baseline(args, tgt:Tensor) -> dq.QubitCircuit:
  ts_start = time()
  vqc = vqc_F1_all_wise_init(nq, n_rep=3)

  optim = Adam(vqc.parameters(), lr=args.lr)
  for i in range(args.n_iter):
    optim.zero_grad()
    state = vqc().squeeze().real
    loss = -get_fidelity(state, tgt)
    loss.backward()
    optim.step()
    if i % 10 == 0:
      print('fid:', -loss.item())
  ts_end = time()

  gcnt = get_gate_count(vqc)
  fid = get_fidelity(vqc().squeeze().real, tgt).item()
  ts = ts_end - ts_start
  print(f'[baseline] gcnt={gcnt}, fid={fid:.5f}, ts={ts:.3f}s')


def clone_circ(circ:dq.QubitCircuit) -> dq.QubitCircuit:
  # 通过序列号-反序列化实现安全的深拷贝 :(
  return pkl.loads(pkl.dumps(circ))

def expand_vqc(args, vqc:dq.QubitCircuit, tgt:Tensor, optim:Adam, n_add:int=1, tau:float=0.01) -> Adam:
  operator_pool = [(i, j) for i in range(nq) for j in range(nq) if i <= j]

  # try each candidate op
  g_op = []
  for i, j in operator_pool:
    # tmp circ
    vqc_tmp = clone_circ(vqc)
    if args.freeze:
      for op in vqc_tmp.operators:
        op.requires_grad_(False)
        if hasattr(op, 'theta'):
          op.theta.requires_grad = False
    if i == j:
      g = dq.Ry(nqubit=nq, wires=i, requires_grad=True)
      g.init_para([tau])
      vqc_tmp.add(g)
    else:
      g = dq.Ry(nqubit=nq, wires=j, controls=i, requires_grad=True)
      g.init_para([tau])
      vqc_tmp.add(g)
      g = dq.Ry(nqubit=nq, wires=i, controls=j, requires_grad=True)
      g.init_para([tau])
      vqc_tmp.add(g)
    # probe grad
    vqc_tmp.zero_grad()
    state = vqc_tmp().squeeze().real
    loss = -get_fidelity(state, tgt)
    loss.backward()
    g = vqc_tmp.operators[-1].theta.grad.abs().item()
    g_op.append((g, (i, j)))
  g_op.sort(reverse=True)

  # add the best op
  print(f'>> add op: {[op for _, op in g_op[:n_add]]}')
  if args.freeze:
    for op in vqc.operators:
      op.requires_grad_(False)
      if hasattr(op, 'theta'):
        op.theta.requires_grad = False
  for _, (i, j) in g_op[:n_add]:
    if i == j:
      g = dq.Ry(nqubit=nq, wires=i, requires_grad=True)
      g.init_para([tau])
      vqc.add(g)
    else:
      g = dq.Ry(nqubit=nq, wires=j, controls=i, requires_grad=True)
      g.init_para([tau])
      vqc.add(g)
      g = dq.Ry(nqubit=nq, wires=i, controls=j, requires_grad=True)
      g.init_para([tau])
      vqc.add(g)

  if optim is None:
    optim = Adam(vqc.parameters(), lr=args.lr)
  else:
    pg0 = optim.param_groups[0]
    optim.add_param_group({
      'params': vqc.operators[-1].parameters(),
      'lr': pg0['lr'],
      'betas': pg0['betas'],
      'eps': pg0['eps'],
      'weight_decay': pg0['weight_decay'],
      'amsgrad': pg0['amsgrad'],
      'maximize': pg0['maximize'],
      'foreach': pg0['foreach'],
      'capturable': pg0['capturable'],
      'differentiable': pg0['differentiable'],
      'fused': pg0['fused'],
    })
  return optim

def run_optim_adapt(args, tgt:Tensor) -> dq.QubitCircuit:
  losses: List[float] = []
  def get_loss_fluct(win_len:int=10) -> float:
    maxx = max(losses[-win_len:]) if losses else 0.0
    minn = min(losses[-win_len:]) if losses else 0.0
    return abs(maxx - minn)

  ts_start = time()
  vqc = dq.QubitCircuit(nq)
  #for i in range(vqc.nqubit):
  #  vqc.ry(wires=i)
  #optim = Adam(vqc.parameters(), lr=args.lr)
  optim = None

  win_len = 10
  i_iter = 0
  x_iter = -win_len
  while i_iter < args.n_iter:
    if i_iter - x_iter >= win_len and get_loss_fluct(win_len) < args.eps:
      optim = expand_vqc(args, vqc, tgt, optim)
      x_iter = i_iter

    optim.zero_grad()
    state = vqc().squeeze().real
    loss = -get_fidelity(state, tgt)
    loss.backward()
    optim.step()

    loss_val = loss.item()
    losses.append(loss_val)

    i_iter += 1
    if i_iter % 10 == 0:
      print('fid:', -loss_val)

  ts_end = time()

  gcnt = get_gate_count(vqc)
  fid = get_fidelity(vqc().squeeze().real, tgt).item()
  ts = ts_end - ts_start
  print(f'[adapt] gcnt={gcnt}, fid={fid:.5f}, ts={ts:.3f}s')


def run(args):
  set_seed()
  tgt = rand_mnist_freq(nq)

  if args.baseline:
    # fid ~= 
    run_optim_baseline(args, tgt)
  else:
    # fid ~= 0.33
    run_optim_adapt(args, tgt)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-S',  '--n_iter', default=500, type=int, help='optim steps')
  parser.add_argument('-lr', '--lr',     default=0.02, type=float)
  parser.add_argument('-eps', '--eps',   default=1e-3, type=float)
  parser.add_argument('--baseline', action='store_true')
  parser.add_argument('--freeze', action='store_true')
  args = parser.parse_args()

  run(args)
