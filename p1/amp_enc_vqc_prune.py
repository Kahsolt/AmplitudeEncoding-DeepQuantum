#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/30 

# VQC方法(实验性的): 训练时剪枝

from amp_enc_vqc_adapt import *


if 'logger' and __name__ == '__main__':
  log_fp = OUTPUT_DIR + '/amp_enc_vqc_prune.log'
  print('>> log_fp:', log_fp)
  logger = logging.getLogger(__file__)
  logger.setLevel(logging.DEBUG)
  h = logging.FileHandler(log_fp, encoding='utf-8')
  h.setLevel(logging.INFO)
  logger.addHandler(h)
  h = logging.StreamHandler()
  h.setLevel(logging.DEBUG)
  logger.addHandler(h)
  del h


def prune_circ(circ:dq.QubitCircuit) -> dq.QubitCircuit:
  vqc = dq.QubitCircuit(circ.nqubit)
  for op in circ.operators:
    if isinstance(op, dq.PauliX) or op.theta.abs().item() > 0.1:
      vqc.add(op, encode=False)
  logger.info(f'[prune_circ] gate count: {count_gates(vqc)}')
  return vqc


def amplitude_encoding_vqc_prune(tgt:Tensor, n_layer:int=14, n_iter:int=1000, lr:float=0.02) -> dq.QubitCircuit:
  # 初始线路结构
  circ = init_circ(n_layer)
  optimizer = optim.Adam(circ.parameters(), lr=lr)
  # 保存最好的结果
  for i in range(n_iter):
    optimizer.zero_grad()
    state = circ().swapaxes(0, 1)     # [B=1, D=1024]
    loss = -get_fidelity(state, tgt)
    loss.backward()
    optimizer.step()

    if (i + 1) % 10 == 0:
      logger.debug(f'>> [{i+1}/{n_iter}] fid: {-loss.item()}')

    if (i + 1) % 100 == 0:
      circ = prune_circ(circ)
  return circ


def run(args):
  dataset = QMNISTDatasetDummy(label_list=[0,1,2,3,4], train=False, per_cls_size=args.n_samples//5)

  logger.info(f'[Start Time] {datetime.now()}')

  sc_list = []
  for idx, (x, y, _) in enumerate(dataset):
    # 数据
    z = reshape_norm_padding(x.unsqueeze(0))
    #z = reshape_norm_padding(x.unsqueeze(0), use_hijack=False)
    z = z.to(device)
    # 训练
    s = time()
    circ = amplitude_encoding_vqc_prune(z, args.n_layer, args.n_iter, args.lr)
    t = time()
    # 评估
    sc = get_score(circ, z)
    sc_list.append(sc)
    logger.info(f'[{idx}] >> Score: {sc} (time: {t - s})\n')

  logger.info(f'-' * 42)
  logger.info(f'mean(score): {mean(sc_list)}')

  logger.info(f'[Finish Time] {datetime.now()}')
  logger.info('\n\n')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-N', '--n_samples', default=10, type=int)
  parser.add_argument('-L', '--n_layer',   default=14, type=int)
  parser.add_argument('-S',  '--n_iter',   default=1000, type=int, help='optim steps')
  parser.add_argument('-lr', '--lr',       default=0.02, type=float)
  args = parser.parse_args()

  run(args)
