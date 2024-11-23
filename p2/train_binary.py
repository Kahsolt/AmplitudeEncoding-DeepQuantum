#!/usr/bin/env python3
# Author: Armit
# Create Time: 周日 2024/11/17 

# 训练二分类模型

import os
from argparse import ArgumentParser

if __name__ == '__main__':    # for fast check & ignore
  parser = ArgumentParser()
  parser.add_argument('-T', '--targets', default='0,1', type=(lambda s: [int(e) for e in s.split(',')]), help='class labels for binary clf')
  args = parser.parse_args()

  assert len(args.targets) == 2
  A, B = args.targets
  assert 0 <= A <= 4 and 0 <= B <= 4

  BASE_OUTPUT_DIR = './output'
  OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f'bin_{A}-{B}')
  if os.path.isdir(OUTPUT_DIR):
    print(f'>> ignored, folder exists: {OUTPUT_DIR}')
    exit(0)
  os.makedirs(OUTPUT_DIR, exist_ok=True)

from typing import List

from model import QuantumNeuralNetworkCL
from train import *
from utils import reshape_norm_padding


class PerfectAmplitudeEncodingDatasetTwoLabel(PerfectAmplitudeEncodingDataset):

  def __init__(self, targets:List[int], train=True, size=100000000):
    self.targets = targets
    super().__init__(train, size)

  def encode_data(self):
    quantum_dataset = []
    for image, label in tqdm(self.dataset):
      label: int = label.item()
      if label not in self.targets: continue
      vec = reshape_norm_padding(image).squeeze(0)
      lbl = self.targets.index(label)
      quantum_dataset.append((-1, lbl, vec))
    return quantum_dataset


def train_model_cl(model:QuantumNeuralNetworkCL, optimizer:optim.Optimizer, train_loader:DataLoader, valid_loader:DataLoader, num_epochs:int, output_dir:str, device='cpu'):
  model.to(device)
  print(f"参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

  os.makedirs(output_dir, exist_ok=True)
  save_path = os.path.join(output_dir, "best_model.pt")
  history_path = os.path.join(output_dir, "loss_history.json")
  fig_path = os.path.join(output_dir, "loss.png")
  pbar = tqdm(total=num_epochs * len(train_loader), position=0, leave=True)
  best_valid_loss = float('inf')
  history = {
    'train_loss': [], 
    'valid_loss': [],
    'valid_fid_eq': [],
    'valid_fid_ne': [],
  }

  # mark for correct .postprocess()
  model.is_training = True
  for epoch in range(num_epochs):
    ''' Train '''
    model.train()
    train_loss = 0.0
    inner_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, position=1)
    for _, y, z in inner_pbar:
      y, z = y.to(device), z.to(device)
      optimizer.zero_grad()
      loss, fid_eq, fid_ne = model(z, y)
      loss.backward()
      optimizer.step()
      train_loss += loss.item() * y.size(0)
      inner_pbar.set_description(f'Batch Loss: {loss.item():.4f}')
      pbar.update(1)
    train_loss /= len(train_loader.dataset)
    history['train_loss'].append(train_loss)
    print(f'Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.7f}')
    with open(history_path, 'w', encoding='utf-8') as fh:
      json.dump(history, fh, indent=2, ensure_ascii=False)

    ''' Eval '''
    model.eval()
    valid_loss = 0.0
    fid_eq_sum, fid_ne_sum = 0.0, 0.0
    with torch.no_grad():
      for _, y, z in valid_loader:
        y, z = y.to(device), z.to(device)
        loss, fid_eq, fid_ne = model(z, y)
        n_sample_batch = y.shape[0]
        fid_eq_sum += fid_eq.item() * n_sample_batch
        fid_ne_sum += fid_ne.item() * n_sample_batch
        valid_loss += loss  .item() * n_sample_batch
    n_samples = len(valid_loader.dataset)
    valid_loss /= n_samples
    fid_eq_sum /= n_samples
    fid_ne_sum /= n_samples
    history['valid_loss'  ].append(valid_loss)
    history['valid_fid_eq'].append(fid_eq_sum)
    history['valid_fid_ne'].append(fid_ne_sum)
    print(f'Epoch {epoch+1}/{num_epochs} - Valid loss: {valid_loss:.7f}, fid_eq: {fid_eq_sum}, fid_ne: {fid_ne_sum}')
    with open(history_path, 'w', encoding='utf-8') as fh:
      json.dump(history, fh, indent=2, ensure_ascii=False)

    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      print('>> save new best_acc ckpt')
      model.mk_ref_qstate(train_loader.dataset, DEVICE)   # backfill refdata before ckpt save
      torch.save(model.state_dict(), save_path)

  pbar.close()

  plt.clf()
  plt.plot(history['train_loss'], 'b', label='train_loss')
  plt.plot(history['valid_loss'], 'r', label='valid_loss')
  plt.savefig(fig_path, dpi=400)
  plt.close()


if __name__ == '__main__':
  # Settings
  DEVICE   = "cuda:0"
  NUM_LAYER  = 3     # todo: 修改为合适的配置
  BATCH_SIZE = 64    # todo: 修改为合适的配置
  NUM_EPOCHS = 30    # [30, 50]
  RESUME   = True    # 基于底模！

  dataset = PerfectAmplitudeEncodingDatasetTwoLabel(args.targets, train=True)   # 用全部数据训练，防止过拟合
  print('dataset labels:', Counter(sample[1] for sample in dataset))
  train_size = int(0.7 * len(dataset))
  valid_size = len(dataset) - train_size
  train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
  # 构建数据加载器，用于加载训练和验证数据
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
  valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

  # 创建一个量子神经网络模型
  if RESUME:
    with open(f'{BASE_OUTPUT_DIR}/model_config.pkl', 'rb') as file:
      model_config = pkl.load(file)
    model = QuantumNeuralNetworkCL(**model_config)
    save_fp = os.path.join(BASE_OUTPUT_DIR, "best_model.pt")
    print(f'>> resume from {save_fp}')
    state_dict = torch.load(save_fp)
    model.load_state_dict(state_dict)
  else:
    model_config = {'num_qubits': 12, 'num_layers': NUM_LAYER}
    model = QuantumNeuralNetworkCL(**model_config)
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # 将字典保存到文件中
  with open(f'{OUTPUT_DIR}/model_config.pkl', 'wb') as file:
    pkl.dump(model_config, file)

  # 训练模型
  ts_start = time()
  train_model_cl(
    model,
    optimizer,
    train_loader, 
    valid_loader,
    num_epochs=NUM_EPOCHS, 
    output_dir=OUTPUT_DIR,
    device=DEVICE,
  )
  ts_end = time()
  print('>> train clf_model time cost:', ts_end - ts_start)   # 5531s
