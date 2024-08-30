#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/27 

# VQC方法(实验性的): 渐进式纯粹训练线路，对每个样本优化到评分函数极大值
# 基线配置: python amp_enc_vqc_adapt.py -Lmin 14 -Lmax 14
# => mean(score): 2.576, runtime 20min
# 默认配置: python amp_enc_vqc_adapt.py
# => mean(score): 2.608, runtime ~2h
# 默认配置 (冻结): python amp_enc_vqc_adapt.py --freeze
# => mean(score): 2.608, runtime ~1.7h = 1h40min

from datetime import datetime
from amp_enc_vqc import *

if 'logger':
  log_fp = OUTPUT_DIR + '/amp_enc_vqc_adapt.log'
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


def get_score(circ:dq.QubitCircuit, state_gt:Tensor) -> float:
  # runtime_score 测不准，就先不考虑了
  fid_score = get_fidelity(circ(), state_gt).item()
  gcnt_score = 1 - count_gates(circ) / 1000
  logger.info(f'  fid_score: {fid_score}')
  logger.info(f'  gcnt_score: {gcnt_score}')
  return 2 * fid_score + gcnt_score


def init_circ(n_layer:int, nq:int=10) -> dq.QubitCircuit:
  vqc = dq.QubitCircuit(nq)
  vqc.x(0)
  for _ in range(n_layer):
    for q in range(nq-1):
      g = dq.Ry(nqubit=nq, wires=(q+1)%nq, controls=q, condition=True, requires_grad=True)
      g.init_para([0])
      vqc.add(g)
      g = dq.Ry(nqubit=nq, wires=q, controls=(q+1)%nq, condition=True, requires_grad=True)
      g.init_para([0])
      vqc.add(g)
  
  logger.info(f'[init_circ] gate count: {count_gates(vqc)}')
  return vqc


def upgrade_circ(circ:dq.QubitCircuit, freeze:bool=False) -> dq.QubitCircuit:
  vqc = dq.QubitCircuit(circ.nqubit)
  for op in circ.operators:
    if freeze and isinstance(op, dq.Ry):    # 冻结含参门
      op.requires_grad = False
      vqc.add(op, encode=True)
    else:
      vqc.add(op, encode=False)

  nq = vqc.nqubit
  for q in range(nq-1):
    g = dq.Ry(nqubit=nq, wires=(q+1)%nq, controls=q, condition=True, requires_grad=True)
    g.init_para([0])
    vqc.add(g)
    g = dq.Ry(nqubit=nq, wires=q, controls=(q+1)%nq, condition=True, requires_grad=True)
    g.init_para([0])
    vqc.add(g)

  logger.info(f'[upgrade_circ] gate count: {count_gates(vqc)}')
  return vqc


def clone_circ(circ:dq.QubitCircuit) -> dq.QubitCircuit:
  # 通过序列号-反序列化实现安全的深拷贝 :(
  return pickle.loads(pickle.dumps(circ))


def amplitude_encode_vqc(circ:dq.QubitCircuit, tgt:Tensor, n_iter:int=1000, lr:float=0.02):
  optimizer = optim.Adam(circ.parameters(), lr=lr)
  for i in range(n_iter):
    optimizer.zero_grad()
    state = circ().swapaxes(0, 1)     # [B=1, D=1024]
    loss = -get_fidelity(state, tgt)
    loss.backward()
    optimizer.step()

    if (i + 1) % 10 == 0:
      logger.debug(f'>> [{i+1}/{n_iter}] fid: {-loss.item()}')

    # TODO: 若 fid 的变化已经趋于平稳，10步迭代积累的总变化不超过0.001，则提前返回
    # 你需要定义一些辅助变量来跟踪 fid 的变化情况


def amplitude_encoding_vqc_adapt(tgt:Tensor, n_layer_min:int=4, n_layer_max:int=16, n_iter:int=1000, lr:float=0.02, freeze:bool=False) -> dq.QubitCircuit:
  # 初始线路结构
  circ = init_circ(n_layer_min)
  if n_layer_max == n_layer_min:
    amplitude_encode_vqc(circ, tgt, n_iter, lr)
    return circ
  # 保存最好的结果
  best_socre = -1
  best_circ = None
  for _ in range(n_layer_max - n_layer_min):
    # 训练
    amplitude_encode_vqc(circ, tgt, n_iter, lr)
    # 评估
    score = get_score(circ, tgt)
    if score  <= best_socre + 0.001: break   # 分数不涨了，提前结束
    logger.info(f'>> new best_socre: {score}')
    best_socre = score
    best_circ = clone_circ(circ)
    # 下一阶段
    circ = upgrade_circ(circ, freeze)
  return best_circ


def run(args):
  dataset = QMNISTDatasetDummy(label_list=[0,1,2,3,4], train=False, per_cls_size=args.n_samples//5)

  logger.info(f'[Start Time] {datetime.now()}')

  sc_list = []
  for idx, (x, y, _) in enumerate(dataset):
    # 数据
    z = snake_reshape_norm_padding(x.unsqueeze(0), rev=True)
    #z = reshape_norm_padding(x.unsqueeze(0), use_hijack=False)
    z = z.to(device)
    # 训练
    s = time()
    circ = amplitude_encoding_vqc_adapt(z, args.n_layer_min, args.n_layer_max, args.n_iter, args.lr, args.freeze)
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
  parser.add_argument('-N',    '--n_samples',   default=10,   type=int)
  parser.add_argument('-Lmin', '--n_layer_min', default=10,   type=int)
  parser.add_argument('-Lmax', '--n_layer_max', default=20,   type=int)
  parser.add_argument('-S',  '--n_iter',        default=1000, type=int, help='optim steps')
  parser.add_argument('-lr', '--lr',            default=0.02, type=float)
  parser.add_argument('--freeze', action='store_true')
  args = parser.parse_args()

  run(args)
