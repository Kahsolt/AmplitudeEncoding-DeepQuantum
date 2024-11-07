#!/usr/bin/env python3
# Author: Armit
# Create Time: 周四 2024/11/07 

# 用标准 QCNN **二分类模型** 进行集成-投票推理

from run_qcnn import *

'''
qcnn: 43%
mera: 44%
'''

@torch.inference_mode
def infer_model(model_grid:List[List[QNN_bin_clf]], test_loader:DataLoader, device:str='cpu'):
  ok, tot = 0, 0
  pbar = tqdm(test_loader)
  for _, y, z in pbar:
    z = z.to(device)
    y = y.item()

    votes = []
    for i in range(5):
      for j in range(5):
        model = model_grid[i][j]
        if model is None: continue
        logits = model.inference(z)
        pred = logits.argmax(-1).item()
        votes.append([i, j][pred])
    pred_voted = Counter(votes).most_common()[0][0]
    if pred_voted != y:
      print(pred_voted, y)

    ok += pred_voted == y
    tot += 1
    pbar.set_description(f'Acc: {ok / tot}')

  print('>> Acc:', ok / tot)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model',    default='qcnn', choices=['qcnn', 'mera'], help='ansatz model')
  parser.add_argument('-L', '--n_layers', default=1, type=int, help='ansatz layers')
  args = parser.parse_args()

  dataset = PerfectAmplitudeEncodingDataset(train=False, size=2500)
  print('dataset labels:', Counter(sample[1].item() for sample in dataset))
  print('dataset len:', len(dataset))
  test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

  n_models = 0
  model_grid = [[None] * 5 for _ in range(5)]
  for i in range(5):
    for j in range(5):
      if i == j: continue
      fp = os.path.join('output', f'{args.model}_{i}-{j}', 'best_model.acc.pt')
      if not os.path.exists(fp): continue
      n_models += 1
      model = QNN_bin_clf(args.model, args.n_layers)
      model.load_state_dict(torch.load(fp))
      model_grid[i][j] = model.eval()
  print(f'>> loaded n_models: {n_models}')

  ts_start = time()
  infer_model(
    model_grid,
    test_loader, 
    device='cpu',
  )
  ts_end = time()
  print('>> infer model_grid time cost:', ts_end - ts_start)
