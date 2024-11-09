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

def get_model_mlp1_nb():
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
model_getters = [get_model_mlp1_nb]  # 纯 ansatz 方法的理论上限


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
