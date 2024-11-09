#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/11/09 

# 跑经典神经网络的 MLP 基线

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import PerfectAmplitudeEncodingDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
epochs = 100
img_path = './img'


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
  acc_list = []
  iter = 0
  for epoch in range(epochs):
    model.train()
    for _, Y, X in trainloader:
      X = X.real.to(device)
      Y = Y.to(device)
      logits = model(X)
      loss = F.cross_entropy(logits, Y)
      optim.zero_grad()
      loss.backward()
      optim.step()
      iter += 1
      if iter % 20 == 0:
        loss_list.append(loss.item())

    acc = test(model, testloader)
    acc_list.append(acc)
    print(f'  {epoch} Acc: {acc:.3%}')

  print(f'>> best acc: {max(acc_list):.3f}')

  plt.clf()
  plt.subplot(211) ; plt.plot(loss_list, 'b') ; plt.title('loss')
  plt.subplot(212) ; plt.plot(acc_list,  'r') ; plt.title(f'acc (best: {max(acc_list):.3%})')
  plt.tight_layout()
  plt.savefig(f'{img_path}/{model.__class__.__name__}.png', dpi=400)
  plt.close()


def get_model_mlp0():     # 模拟任意 ansatz 方法
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

  class MLP0(nn.Module):
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
      x = x.squeeze(-1)   # in case of std_flat
      U, _ = torch.linalg.qr(self.U_holder, mode='complete')
      out = x @ U         # [B, D=4096], qstate
      res = []
      for H in self.meas:     # simulate expval(), i.e. <ψ|H|ψ>
        r = out @ H @ out.T
        res.append(r.diag())
      return torch.stack(res, dim=-1)
  return MLP0()

def get_model_mlp1():
  class MLP1(nn.Module):
    def __init__(self):
      super().__init__()
      self.mlp = nn.Sequential(
        nn.Linear(3072, 5),
      )
    def forward(self, x):
      return self.mlp(x[:, :3072])
  return MLP1()

def get_model_mlp1_nb():  # 纯 ansatz 方法的理论上限
  class MLP1_nb(nn.Module):
    def __init__(self):
      super().__init__()
      self.mlp = nn.Sequential(
        nn.Linear(3072, 5, bias=False),
      )
    def forward(self, x):
      return self.mlp(x[:, :3072])
  return MLP1_nb()

def get_model_mlp2():
  class MLP2(nn.Module):
    def __init__(self):
      super().__init__()
      self.mlp = nn.Sequential(
        nn.Linear(3072, 256),
        nn.ReLU(),
        nn.Linear(256, 5),
      )
    def forward(self, x):
      return self.mlp(x[:, :3072])
  return MLP2()

def get_model_mlp2_drop():
  class MLP2_drop(nn.Module):
    def __init__(self):
      super().__init__()
      self.mlp = nn.Sequential(
        nn.Linear(3072, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 5),
      )
    def forward(self, x):
      return self.mlp(x[:, :3072])
  return MLP2_drop()

def get_model_mlp3():
  class MLP3(nn.Module):
    def __init__(self):
      super().__init__()
      self.mlp = nn.Sequential(
        nn.Linear(3072, 1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 5),
      )
    def forward(self, x):
      return self.mlp(x[:, :3072])
  return MLP3()


#model_getters = [v for k, v in globals().items() if k.startswith('get_model')]
model_getters = [get_model_mlp0]


trainset = PerfectAmplitudeEncodingDataset(train=True, size=2500)
testset  = PerfectAmplitudeEncodingDataset(train=False)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  drop_last=True)
testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, drop_last=False)
print('len(trainset):', len(trainset), 'len(trainloader):', len(trainloader))
print('len(testset):',  len(testset),  'len(testloader):',  len(testloader))


for get_model in model_getters:
  model = get_model().to(device)
  print('param_cnt:', sum(p.numel() for p in model.parameters() if p.requires_grad))
  train(model, trainloader, testloader)
