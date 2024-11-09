#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/11/10 

# 专题研究 MLP0 方法，看实际得到的 AmpEnc 编码对于精确编码上训练的线性模型而言是否仍然是可识别的输入
# 结论: 常规训-验-测精度 52.8%，过拟合到测试集可达 67.6%

import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import PerfectAmplitudeEncodingDataset
from utils import QCIFAR10Dataset   # keep for unpickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
epochs = 15
img_path = './img'
overfit = False


I = torch.tensor([
  [1, 0],
  [0, 1],
], device=device, dtype=torch.float32)
X = torch.tensor([
  [0, 1],
  [1, 0],
], device=device, dtype=torch.float32)
Z = torch.tensor([
  [1, 0],
  [0, -1],
], device=device, dtype=torch.float32)
term_to_gate = {
  'I': I,
  'X': X,
  'Z': Z,
}
def get_pauli_unitary(s:str):
  u = term_to_gate[s[0]]
  for c in s[1:]:
    u = torch.kron(term_to_gate[c], u)
  return u

class MLP0(nn.Module):     # 模拟任意 ansatz 方法

  def __init__(self):
    super().__init__()

    # 假设该矩阵可以被 QR 分解，其中矩阵 Q 即某 ansatz 所对应酉阵 U
    self.U_holder = nn.Parameter(torch.eye(4096, device=device, requires_grad=True))
    self.meas = [
      get_pauli_unitary('ZIIIIIIIIIII'),    # Z0
      get_pauli_unitary('IZIIIIIIIIII'),    # Z1
      get_pauli_unitary('IIZIIIIIIIII'),    # Z2
      get_pauli_unitary('IIIZIIIIIIII'),    # Z3
      get_pauli_unitary('IIIIZIIIIIII'),    # Z4
    ]

  def forward(self, x):
    x = x.real.squeeze(-1)   # in case of std_flat
    U, _ = torch.linalg.qr(self.U_holder, mode='complete')
    out = x @ U         # [B, D=4096], qstate
    res = []
    for H in self.meas:     # simulate expval(), i.e. <ψ|H|ψ>
      r = out @ H @ out.T
      res.append(r.diag())
    return torch.stack(res, dim=-1)


@torch.inference_mode
def test(model:nn.Module, testloader:DataLoader) -> float:
  ok, tot = 0, 0
  model.eval()
  for _, Y, X in testloader:
    X = X.real.to(device)
    Y = Y.to(device)
    logits = model(X)
    pred = logits.argmax(-1)
    tot += len(Y)
    ok  += (pred == Y).sum().item()
  return ok / tot


@torch.enable_grad
def train(model:nn.Module, trainloader:DataLoader, testloader:DataLoader) -> float:
  optim = torch.optim.Adam(model.parameters(), lr=0.001)
  loss_list = []
  test_acc_list, train_acc_list = [], []
  best_test_acc = -1
  for epoch in range(epochs):
    model.train()
    loss_sum = 0.0
    for _, Y, X in trainloader:
      X = X.to(device)
      Y = Y.to(device)
      logits = model(X)
      loss = F.cross_entropy(logits, Y)
      optim.zero_grad()
      loss.backward()
      optim.step()
      loss_sum += loss.item()

    acc_train = test(model, trainloader)
    acc_test  = test(model, testloader)
    train_acc_list.append(acc_train)
    test_acc_list .append(acc_test)
    loss_list.append(loss_sum / len(trainloader.dataset))
    print(f'  {epoch} Train Acc: {acc_train:.3%}, Test Acc: {acc_test:.3%}')

    if acc_test > best_test_acc:
      best_test_acc = acc_test
      save_fp = './output/mlp0-best.pth'
      print(f'>> save ckpt to {save_fp}')
      torch.save(model.state_dict(), save_fp)

  print(f'>> best acc: {max(test_acc_list):.3f}')

  plt.clf()
  plt.plot(loss_list, 'b')
  ax = plt.twinx()
  ax.plot(test_acc_list,       'r')
  ax.plot(train_acc_list, 'orange')
  plt.suptitle(f'train best: {max(train_acc_list):.3%} / test best: {max(test_acc_list):.3%}')
  plt.tight_layout()
  plt.savefig(f'{img_path}/mlp0.png', dpi=400)
  plt.close()


if __name__ == '__main__':
  with open('output/test_dataset.pkl', 'rb') as file:
    testset = pkl.load(file)
  trainset = testset if overfit else PerfectAmplitudeEncodingDataset(train=True)   # 使用全部数据集训练，防止过拟合！
  trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  drop_last=True)
  testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, drop_last=False)
  print('len(dataset):', len(trainset), 'len(trainloader):', len(trainloader))
  print('len(dataset):', len(testset),  'len(testloader):',  len(testloader))

  model = MLP0().to(device)
  print('device:', device)
  print('param_cnt:', sum(p.numel() for p in model.parameters() if p.requires_grad))
  train(model, trainloader, testloader)

  save_fp = './output/mlp0.pth'
  print(f'>> save ckpt to {save_fp}')
  torch.save(model.state_dict(), save_fp)
