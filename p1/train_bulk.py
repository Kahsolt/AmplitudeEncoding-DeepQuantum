#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/23 

# 尝试纯粹训练线路 (批量训练并保存，之后可以合并)
# NOTE: 测试集一共有 5139 个样本，注意 --start/--stop 指定的是左闭右开区间 [a,b)
# 每个样本训练时间约 1min30s，期间没有log信息输出，每个样本训练完之后会有一条log

import logging
from argparse import ArgumentParser
from train_single import *

# 数据集样本个数
N_SAMPLES = 5139
# 训练迭代次数
N_ITER = 1000
# 保存路径
OUTPUT_DIR = './output'
# 设备 (不要用 cuda，cuda 比 cpu 慢!!)
device = 'cpu'

mean = lambda x: sum(x) / len(x) if len(x) else -1


class QMNISTDatasetDummy(QMNISTDatasetIdea):   # 无预处理操作的数据集

  def generate_data(self) -> List[Tuple[Tensor, int, None]]:
    data_list = []
    for image, label in tqdm(self.sub_dataset):
      data_list.append((image, label, None))    # 不产生 z，我们在外面去产生 z
    return data_list


def get_model(n_layer:int=5, nq:int=10) -> dq.QubitCircuit:
  ''' mottonen-like zero-init ↓↑ '''
  vqc = dq.QubitCircuit(nq)
  for i in range(n_layer):
    flag = i % 2 == 1
    if flag == 0:
      for q in range(1, nq):
        # cond
        lyr = dq.RyLayer(nq, wires=list(range(q)), requires_grad=True)
        lyr.init_para([0] * q)
        vqc.add(lyr)
        # mctrl-rot
        g = dq.Ry(nqubit=nq, wires=q, controls=list(range(q)), condition=True, requires_grad=True)
        g.init_para([0])
        vqc.add(g)
    else:
      for q in reversed(range(1, nq)):
        # cond
        lyr = dq.RyLayer(nq, wires=list(range(q, nq)), requires_grad=True)
        lyr.init_para([0] * (nq - q + 1))
        vqc.add(lyr)
        # mctrl-rot
        g = dq.Ry(nqubit=nq, wires=q-1, controls=list(range(q, nq)), condition=True, requires_grad=True)
        g.init_para([0])
        vqc.add(g)
  vqc.rylayer()   # 最后一层需要初始化
  return vqc


def train_amp_enc_vqc(tgt:Tensor) -> dq.QubitCircuit:
  circ = get_model().to(device)
  optimizer = optim.Adam(circ.parameters(), lr=0.02)
  for _ in range(N_ITER):
    optimizer.zero_grad()
    state = circ().swapaxes(0, 1)     # [B=1, D=1024]
    loss = -get_fidelity(state, tgt)
    loss.backward()
    optimizer.step()
  return circ


def run(args):
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

  dataset = QMNISTDatasetDummy(label_list=[0,1,2,3,4], train=False)
  assert len(dataset) == N_SAMPLES

  fid_list = []
  save_data = []
  process_cnt = 0
  for idx, (x, y, _) in enumerate(tqdm(dataset)):
    if not (args.start <= idx < args.stop): continue

    s = time()
    z = snake_reshape_norm_padding(x.unsqueeze(0))
    x, y, z = x.to(device), y.to(device), z.to(device)
    circ = train_amp_enc_vqc(z)
    t = time()
    save_data.append((x, y, circ))

    fid = get_fidelity(circ().swapaxes(0, 1), z).item()
    fid_list.append(fid)
    logger.info(f'[{idx}] >> fid: {fid} (time: {t - s}); mean(fid): {mean(fid_list)}')

    process_cnt += 1

    if process_cnt % 20 == 0:
      logger.info(f'>> save to {save_fp}')
      with open(save_fp, 'wb') as fh:
        pickle.dump(save_data, fh)

  logger.info(f'>> save to {save_fp}')
  with open(save_fp, 'wb') as fh:
    pickle.dump(save_data, fh)

  logger.info('>> Done!!')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-A', '--start', required=True, type=int, help='sample index range start')
  parser.add_argument('-B', '--stop',  required=True, type=int, help='sample index range stop (not including)')
  args = parser.parse_args()

  assert 0 <= args.start < args.stop <= N_SAMPLES
  print(f'>> training samples in range [{args.start}, {args.stop}), collect {args.stop - args.start} samples')

  run(args)
