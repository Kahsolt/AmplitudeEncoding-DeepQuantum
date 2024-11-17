#!/usr/bin/env python3
# Author: Armit
# Create Time: 周日 2024/11/17 

# 用二分类模型进行集成-投票推理

'''
[测试集精度] 59.6% / 59.4%
注: 该脚本的集成投票实现没有batchify，速度太慢了((
'''

from typing import List
from collections import Counter

from torch import Tensor
from test_binary import *


def get_pred_voted_recursive(probs_grid:List[List[Tensor]], candidates:List[int]) -> int:
  votes = []
  weightsum = {k: 0.0 for k in range(5)}
  for i in candidates:
    for j in candidates:
      probs = probs_grid[i][j]
      if probs is None: continue
      pred = probs.argmax(-1).item()
      vote = [i, j][pred]
      votes.append(vote)
      weightsum[vote] += probs[pred].item()
  lbl_cnt = Counter(votes).most_common()
  maxcnt = lbl_cnt[0][1]
  new_candidates = [k for k, v in lbl_cnt if v == maxcnt]
  if len(new_candidates) == 1:
    return new_candidates[0]
  if len(new_candidates) < len(candidates):
    return get_pred_voted_recursive(probs_grid, new_candidates)
  ranklist = sorted([(p, k) for k, p in weightsum.items()], reverse=True)
  return ranklist[0][1]


def get_pred_voted_non_recursive(probs_grid:List[List[Tensor]], candidates:List[int]) -> int:
  weightsum = {k: 0.0 for k in range(5)}
  for i in candidates:
    for j in candidates:
      probs = probs_grid[i][j]
      if probs is None: continue
      pred = probs.argmax(-1).item()
      vote = [i, j][pred]
      weightsum[vote] += probs[pred].item()
  ranklist = sorted([(p, k) for k, p in weightsum.items()], reverse=True)
  return ranklist[0][1]


@torch.inference_mode
def infer_model(model_grid:List[List[QuantumNeuralNetwork]], test_loader:DataLoader, device:str='cpu'):
  ok, tot = 0, 0
  pbar = tqdm(test_loader)
  for _, y, z in pbar:
    z = z.to(device)
    y = y.item()

    probs_grid = [[None for _ in range(5)] for _ in range(5)]
    for i in range(5):
      for j in range(5):
        model = model_grid[i][j]
        if model is None: continue
        probs_grid[i][j] = model.inference(z).squeeze(0)   # [NC=2]
    pred_voted = get_pred_voted_non_recursive(probs_grid, range(5))

    ok += pred_voted == y
    tot += 1
    pbar.set_description(f'Acc: {ok / tot:.3%}')

  print(f'Acc: {ok / tot:.3%}')


if __name__ == '__main__':
  DEVICE = "cpu"

  with open('./output/test_dataset.pkl', 'rb') as file:
    test_dataset: QCIFAR10Dataset = pkl.load(file)
  print('dataset labels:', Counter(sample[1].item() for sample in test_dataset))
  print('dataset len:', len(test_dataset))
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

  n_models = 0
  model_grid = [[None] * 5 for _ in range(5)]
  for i in range(5):
    for j in range(5):
      if i >= j: continue
      fp = os.path.join('output', f'bin_{i}-{j}', 'best_model.pt')
      if not os.path.exists(fp): continue
      n_models += 1
      with open(f'./output/bin_{i}-{j}/model_config.pkl', 'rb') as file:
        model_config = pkl.load(file)
      model = QuantumNeuralNetwork(**model_config)
      model.ref_qstate = nn.Parameter(torch.zeros([2, 36], requires_grad=False))  # fix shape
      model.load_state_dict(torch.load(fp))
      model_grid[i][j] = model.eval().to(DEVICE)
  print(f'>> loaded n_models: {n_models}')

  ts_start = time()
  infer_model(
    model_grid,
    test_loader,
    device=DEVICE,
  )
  ts_end = time()
  print('>> infer model_grid time cost:', ts_end - ts_start)
