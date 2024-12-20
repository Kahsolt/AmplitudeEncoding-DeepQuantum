#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/23 

# VQC方法: 纯粹训练线路 (批量训练并保存，之后可以合并)

import os
os.environ['MY_LABORATORY'] = '1'

import logging
from time import time
from argparse import ArgumentParser

from utils import *

# 数据集样本个数
N_SAMPLES = 5139
# 训练迭代次数
N_ITER = 1000
# 保存路径
OUTPUT_DIR = './output'
# 设备 (不要用 cuda，cuda 比 cpu 慢!!)
device = 'cpu'


class QMNISTDatasetDummy(QMNISTDatasetIdea):   # 无预处理操作的数据集

  def generate_data(self) -> List[Tuple[Tensor, int, None]]:
    data_list = []
    for image, label in tqdm(self.sub_dataset):
      data_list.append((image, label, None))    # 不产生 z，我们在外面去产生 z
    return data_list


def get_model(n_layer:int, nq:int=10) -> dq.QubitCircuit:
  vqc = dq.QubitCircuit(nq)

  if not 'mottonen-like zero-init':   # n_layer=5, n_gate=280; fid=0.8511381149291992
    for i in range(n_layer):
      for q in range(1, nq):
        # cond
        lyr = dq.RyLayer(nq, wires=list(range(q)), requires_grad=True)
        lyr.init_para([0] * q)
        vqc.add(lyr)
        # mctrl-rot
        g = dq.Ry(nqubit=nq, wires=q, controls=list(range(q)), requires_grad=True)
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
          g = dq.Ry(nqubit=nq, wires=q, controls=list(range(q)), requires_grad=True)
          g.init_para([0])
          vqc.add(g)
      else:
        for q in reversed(range(1, nq)):
          # cond
          lyr = dq.RyLayer(nq, wires=list(range(q, nq)), requires_grad=True)
          lyr.init_para([0] * (nq - q + 1))
          vqc.add(lyr)
          # mctrl-rot
          g = dq.Ry(nqubit=nq, wires=q-1, controls=list(range(q, nq)), requires_grad=True)
          g.init_para([0])
          vqc.add(g)
    vqc.rylayer()   # 最后一层需要初始化

  if not 'swap-like':    # n_layer=15, n_gate=271, fid=0.9302663803100586
    vqc.x(0)
    for i in range(n_layer):
      for q in range(nq):
        vqc.cry(q, q+1)
        vqc.cry(q+1, q)

  # [normal_reshape]
  # n_layer=15, n_gate=271, fid=0.8838197588920593
  # n_layer=12, n_gate=217, fid=0.8564713001251221
  # [snake_reshape] (5 samples estimate)
  # n_layer=15, n_gate=271, fid=0.9141733288764954; score=0.9141733288764954*2+(1-271/1000)=2.5573466577530
  # n_layer=14, n_gate=253, fid=0.9165129184722900; score=0.9165129184722900*2+(1-253/1000)=2.5800258369446
  # n_layer=13, n_gate=235, fid=0.9017681717872620; score=0.9017681717872620*2+(1-235/1000)=2.5685363435745
  # n_layer=12, n_gate=217, fid=0.8845824599266052; score=0.8845824599266052*2+(1-217/1000)=2.5521649198532
  # n_layer=11, n_gate=199, fid=0.8601076364517212; score=0.8601076364517212*2+(1-199/1000)=2.5212152729034
  # n_layer=10, n_gate=181, fid=0.8576362848281860; score=0.8576362848281860*2+(1-181/1000)=2.5342725696564
  # [freq_sorted_reshape] (5 samples estimate)
  # n_layer=14, n_gate=253, fid=0.9230663657188416; score=0.9230663657188416*2+(1-253/1000)=2.5931327314377
  # n_layer=13, n_gate=235, fid=0.9084532022476196; score=0.9084532022476196*2+(1-235/1000)=2.5819064044952
  # n_layer=12, n_gate=217, fid=0.8967987418174743; score=0.8967987418174743*2+(1-217/1000)=2.5765974836349
  if not 'swap-like zero init':
    vqc.x(0)
    for i in range(n_layer):
      for q in range(nq-1):
        g = dq.Ry(nqubit=nq, wires=(q+1)%nq, controls=q, requires_grad=True)
        g.init_para([0])
        vqc.add(g)
        g = dq.Ry(nqubit=nq, wires=q, controls=(q+1)%nq, requires_grad=True)
        g.init_para([0])
        vqc.add(g)

  # [snake_reshape] (1 sample estimate)
  # n_layer=14, n_gate=253, fid=~0.8817921876907349
  if not 'swap-like zero init, dilated':
    GAPS = [1, 2, 3, 4, 5]
    vqc.x(0)
    for i in range(n_layer):
      for q in range(nq-1):
        gap = GAPS[i % len(GAPS)]
        g = dq.Ry(nqubit=nq, wires=(q+gap)%nq, controls=q, requires_grad=True)
        g.init_para([0])
        vqc.add(g)
        g = dq.Ry(nqubit=nq, wires=q, controls=(q+gap)%nq, requires_grad=True)
        g.init_para([0])
        vqc.add(g)

  # [snake_reshape] (5 samples estimate)
  # n_layer=14, n_gate=262, fid=0.9085301160812378; score=0.9085301160812378*2+(1-262/1000)=2.5550602321625
  if not 'RY + swap-like zero init':
    for q in range(nq):
      g = dq.Ry(nqubit=nq, wires=q, requires_grad=True)
      g.init_para([0])
      vqc.add(g)
    for i in range(n_layer):
      for q in range(nq-1):
        g = dq.Ry(nqubit=nq, wires=(q+1)%nq, controls=q, requires_grad=True)
        g.init_para([0])
        vqc.add(g)
        g = dq.Ry(nqubit=nq, wires=q, controls=(q+1)%nq, requires_grad=True)
        g.init_para([0])
        vqc.add(g)

  if not 'swap-like zero init \/':
    vqc.x(0)
    for i in range(n_layer):
      for q in range(nq-1):
        g = dq.Ry(nqubit=nq, wires=(q+1)%nq, controls=q, requires_grad=True)
        g.init_para([0])
        vqc.add(g)
      for q in range(nq-1):
        g = dq.Ry(nqubit=nq, wires=q, controls=(q+1)%nq, requires_grad=True)
        g.init_para([0])
        vqc.add(g)

  if not 'swap-like cyclic zero init':  # n_layer=15, n_gate=301, fid=0.9552421569824219
    vqc.x(0)
    for i in range(n_layer):
      for q in range(nq):
        g = dq.Ry(nqubit=nq, wires=(q+1)%nq, controls=q, requires_grad=True)
        g.init_para([0])
        vqc.add(g)
        g = dq.Ry(nqubit=nq, wires=q, controls=(q+1)%nq, requires_grad=True)
        g.init_para([0])
        vqc.add(g)

  # n_layer=14, n_gate=266, fid=0.6950063467025757
  if not 'RY + zigzag(CNOT), arXiv:2103.13211':
    for i in range(n_layer):
      vqc.rylayer()
      for q in range(0, nq, 2):
        vqc.cnot(q, q+1)
      for q in range(1, nq-1, 2):
        vqc.cnot(q, q+1)

  # [snake_reshape] (1 sample estimate)
  # n_layer=14, n_gate=267, fid=~0.9390624761581421
  # n_layer=10, n_gate=191, fid=~0.8758159279823303
  if not 'RY + [↓↑CRY - RY]*n, zero init':
    g = dq.Ry(nqubit=nq, wires=0, requires_grad=True)
    g.init_para([0])
    vqc.add(g)
    for i in range(n_layer):
      # ↓↑CRY
      for q in range(1, nq):
        g = dq.Ry(nqubit=nq, wires=q, controls=0, requires_grad=True)
        g.init_para([0])
        vqc.add(g)
        g = dq.Ry(nqubit=nq, wires=0, controls=q, requires_grad=True)
        g.init_para([0])
        vqc.add(g)
      # RY
      g = dq.Ry(nqubit=nq, wires=0, requires_grad=True)
      g.init_para([0])
      vqc.add(g)

  # n_layer=4, n_gate=236, fid=~0.9358370900154114
  if not 'mera-like, skel (un-modified)':
    for i in range(n_layer):
      vqc.rylayer()
      # down (->10-8-6-4-2)
      for offset in range(nq // 2):
        for q in range(offset, nq - 1 - offset, 2):
          vqc.cry(q, q + 1)
          vqc.cry(q + 1, q)
      if i < n_layer-1:
        vqc.rylayer(wires=[4, 5])
        # up (->2-4-6-8)
        for offset in range(nq // 2 - 1, 0, -1):
          for q in range(offset, nq - 1 - offset, 2):
            vqc.cry(q + 1, q)
            vqc.cry(q, q + 1)
    vqc.rylayer()

  # n_layer=4, n_gate=258, fid=~0.9281691908836365
  if not 'mera-like, init-[enc-ry-dec-ry]':
    # init
    vqc.rylayer()
    # [enc-ry-dec-ry]
    for i in range(n_layer):
      # down (->10-8-6-4-2)
      for offset in range(nq // 2):
        for q in range(offset, nq - 1 - offset, 2):
          vqc.cry(q, q + 1)
          vqc.cry(q + 1, q)
      vqc.rylayer(wires=[4, 5])
      # up (->4-6-8-10)
      for offset in range(nq // 2 - 1, 0, -1):
        for q in range(offset, nq - 1 - offset, 2):
          vqc.cry(q + 1, q)
          vqc.cry(q, q + 1)
      vqc.rylayer()

  # n_layer=4, n_gate=210, fid=~0.9031393527984619
  if not 'mera-like, init-[enc-dec]':
    # init
    vqc.rylayer()
    # [enc-dec]
    for i in range(n_layer):
      # down (->10-8-6-4-2)
      for offset in range(nq // 2):
        for q in range(offset, nq - 1 - offset, 2):
          vqc.cry(q, q + 1)
          vqc.cry(q + 1, q)
      # up (->4-6-8-10)
      for offset in range(nq // 2 - 1, 0, -1):
        for q in range(offset, nq - 1 - offset, 2):
          vqc.cry(q + 1, q)
          vqc.cry(q, q + 1)

  # n_layer=4, n_gate=279, fid=~0.9503393769264221
  if not 'mera-like, init-[dec-ry-enc-ry]-dec':
    # init
    pivot = nq // 2 - 1
    vqc.ry(wires=[pivot])
    # [dec-ry-enc-ry]
    for i in range(n_layer):
      # up (->2-4-6-8-10)
      for offset in range(nq // 2 - 1, -1, -1):
        for q in range(offset, nq - 1 - offset, 2):
          vqc.cry(q + 1, q)
          vqc.cry(q, q + 1)
      vqc.rylayer()
      # down (->8-6-4-2)
      for offset in range(1, nq // 2):
        for q in range(offset, nq - 1 - offset, 2):
          vqc.cry(q, q + 1)
          vqc.cry(q + 1, q)
      vqc.rylayer(wires=[pivot, pivot+1])
    # dec
    for offset in range(nq // 2 - 1, -1, -1):
      for q in range(offset, nq - 1 - offset, 2):
        vqc.cry(q + 1, q)
        vqc.cry(q, q + 1)

  # n_layer=4, n_gate=231, fid=~0.9109880328178406
  if not 'mera-like, init-[dec-enc]-dec':
    # init
    pivot = nq // 2 - 1
    vqc.ry(wires=[pivot])
    # [dec-enc]
    for i in range(n_layer):
      # up (->2-4-6-8-10)
      for offset in range(nq // 2 - 1, -1, -1):
        for q in range(offset, nq - 1 - offset, 2):
          vqc.cry(q + 1, q)
          vqc.cry(q, q + 1)
      # down (->8-6-4-2)
      for offset in range(1, nq // 2):
        for q in range(offset, nq - 1 - offset, 2):
          vqc.cry(q, q + 1)
          vqc.cry(q + 1, q)
    # dec
    for offset in range(nq // 2 - 1, -1, -1):
      for q in range(offset, nq - 1 - offset, 2):
        vqc.cry(q + 1, q)
        vqc.cry(q, q + 1)

  # n_layer=4, n_gate=173, fid=~0.8660510778427124
  # n_layer=5, n_gate=211, fid=~0.8980904817581177
  # n_layer=6, n_gate=249, fid=~0.9232343435287476
  if not 'distro-like':
    # init
    pivot = nq // 2 - 1
    vqc.ry(wires=[pivot])
    # up ([4,5] -> [[0,1], [8,9]])
    for offset in range(1, pivot+1):
      vqc.cry(pivot - offset, pivot - offset + 1)
      vqc.cry(pivot - offset + 1, pivot - offset)
      vqc.cry(pivot + offset, pivot + offset + 1)
      vqc.cry(pivot + offset + 1, pivot + offset)
    vqc.rylayer(wires=[0, 1, nq-2, nq-1])
    # enc-dec reps
    for i in range(n_layer):
      # down ([[1,2], [7,8]] -> [4,5])
      for offset in range(pivot - 1, -1, -1):
        vqc.cry(pivot - offset, pivot - offset + 1)
        vqc.cry(pivot - offset + 1, pivot - offset)
        vqc.cry(pivot + offset, pivot + offset + 1)
        vqc.cry(pivot + offset + 1, pivot + offset)
      vqc.rylayer(wires=[pivot, pivot+1])
      # up ([4,5] -> [[0,1], [8,9]])
      for offset in range(1, pivot+1):
        vqc.cry(pivot - offset, pivot - offset + 1)
        vqc.cry(pivot - offset + 1, pivot - offset)
        vqc.cry(pivot + offset, pivot + offset + 1)
        vqc.cry(pivot + offset + 1, pivot + offset)
      vqc.rylayer(wires=[0, 1, nq-2, nq-1])

  # n_layer=3, n_gate=301, fid=0.729882538318634
  if 'vqc_F2_all_wise_init':
    vqc.ry(wires=0)   # only init wire 0
    for _ in range(n_layer):
      for i in range(nq-1):   # qubit order
        for j in range(i+1, nq):
          vqc.ry(wires=j, controls=i)
          vqc.ry(wires=i, controls=j)
      for i in range(nq):
        vqc.ry(wires=i)

  # n_layer=3, n_gate=166, fid=0.7076160311698914
  if not 'vqc_F1_all_wise_init':
    vqc.ry(wires=0)   # only init wire 0
    for _ in range(n_layer):
      for i in range(nq-1):   # qubit order
        for j in range(i+1, nq):
          vqc.ry(wires=j, controls=i)
      for i in range(nq):
        vqc.ry(wires=i)

  return vqc


CHECK_GCNT = True

def amplitude_encode_vqc(tgt:Tensor) -> dq.QubitCircuit:
  # NOTE: hard encode the config that we're to run distributedly :)
  global CHECK_GCNT
  circ = get_model(n_layer=14).to(device)
  if CHECK_GCNT:
    assert count_gates(circ) == 253
    CHECK_GCNT = False

  optimizer = optim.Adam(circ.parameters(), lr=0.2)
  for _ in range(N_ITER):
    optimizer.zero_grad()
    state = circ().swapaxes(0, 1)     # [B=1, D=1024]
    loss = -get_fidelity(state, tgt)
    loss.backward()
    optimizer.step()
  return circ


def run_all(args):
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  log_fp = f'{OUTPUT_DIR}/test_dataset_A={args.start}_B={args.stop}.log'
  save_fp = f'{OUTPUT_DIR}/test_dataset_A={args.start}_B={args.stop}.pkl'

  logger = logging.getLogger(__file__)
  logger.setLevel(logging.INFO)
  h = logging.FileHandler(log_fp, encoding='utf-8')
  h.setLevel(logging.INFO)
  logger.addHandler(h)
  h = logging.StreamHandler()
  h.setLevel(logging.INFO)
  logger.addHandler(h)
  del h

  logger.info(f'>> training samples in range [{args.start}, {args.stop}), collect {args.stop - args.start} samples')

  dataset = QMNISTDatasetDummy(label_list=[0,1,2,3,4], train=False)
  assert len(dataset) == N_SAMPLES
  if args.start == args.stop: return

  fid_list = []
  save_data = []
  process_cnt = 0
  for idx, (x, y, _) in enumerate(tqdm(dataset)):
    if not (args.start <= idx < args.stop): continue

    s = time()
    z = reshape_norm_padding(x.unsqueeze(0))
    x, y, z = x.to(device), y.to(device), z.to(device)
    circ = amplitude_encode_vqc(z)
    t = time()
    save_data.append((x, y, circ))

    fid = get_fidelity(circ().swapaxes(0, 1), z).item()
    fid_list.append(fid)
    logger.info(f'[{idx}] >> fid: {fid} (time: {t - s}); mean(fid): {mean(fid_list)}')

    process_cnt += 1

    if process_cnt % 40 == 0:
      logger.info(f'>> save to {save_fp}')
      with open(save_fp, 'wb') as fh:
        pickle.dump(save_data, fh)

  logger.info(f'>> save to {save_fp}')
  with open(save_fp, 'wb') as fh:
    pickle.dump(save_data, fh)

  logger.info('>> Done!!')


def run_few(args):
  circ = get_model(args.n_layer)
  print('gate count:', count_gates(circ))
  print('param count:', sum([p.numel() for p in circ.parameters()]))

  dataset = QMNISTDatasetDummy(label_list=[0,1,2,3,4], train=False, per_cls_size=max(1, args.n_samples//5))

  fid_list = []
  for idx, (x, y, _) in enumerate(dataset):
    if idx > args.n_samples: break

    z = reshape_norm_padding(x.unsqueeze(0))
    #z = reshape_norm_padding(x.unsqueeze(0), use_hijack=False)
    x, y, z = x.to(device), y.to(device), z.to(device)

    circ = get_model(args.n_layer).to(device)
    optimizer = optim.Adam(circ.parameters(), lr=0.02)
    s = time()
    for i in range(N_ITER):
      state = circ().swapaxes(0, 1)     # [B=1, D=1024]
      loss = -get_fidelity(state, z)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if i % 20 == 0:
        print('fid:', -loss.item())
    t = time()

    fid = get_fidelity(circ().swapaxes(0, 1), z).item()
    fid_list.append(fid)
    print(f'[{idx}] >> Fidelity:', fid, f'(time: {t - s})')

  print('mean(fid):', mean(fid_list))


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-D', '--debug', action='store_true')
  # run_few
  parser.add_argument('-N', '--n_samples', default=5, type=int)
  parser.add_argument('-L', '--n_layer', default=14, type=int)
  # run_all
  parser.add_argument('-A', '--start', type=int, help='sample index range start')
  parser.add_argument('-B', '--stop',  type=int, help='sample index range stop (not including)')
  args = parser.parse_args()

  if args.debug:
    run_few(args)
  else:
    assert 0 <= args.start <= args.stop <= N_SAMPLES
    run_all(args)
