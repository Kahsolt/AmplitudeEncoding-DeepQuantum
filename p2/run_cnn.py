#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/21 

# 跑经典神经网络的 CNN 基线 (with/without pretrained weights)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as M
import matplotlib.pyplot as plt

from utils import CIFAR10Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
epochs = 100
img_path = './img'


@torch.inference_mode
def test(model:nn.Module, testloader:DataLoader) -> float:
  ok, tot = 0, 0
  model.eval()
  for X, Y in testloader:
    X = X.to(device)
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
    for X, Y in trainloader:
      X = X.to(device)
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
  plt.subplot(212) ; plt.plot(acc_list,  'r') ; plt.title('acc')
  plt.savefig(f'{img_path}/{model.__class__.__name__}.png', dpi=400)
  plt.close()


def get_model_vgg11():
  model = M.vgg11(pretrained=True)
  fc_old: nn.Linear = model.classifier[-1]
  fc_new = nn.Linear(fc_old.in_features, 5, fc_old.bias is not None)
  model.classifier[-1] = fc_new
  return model

def get_model_resnet18():
  model = M.resnet18(pretrained=True)
  conv_old = model.conv1
  conv_new = nn.Conv2d(conv_old.in_channels, conv_old.out_channels, 3, conv_old.stride, conv_old.padding, conv_old.dilation, conv_old.groups, conv_old.bias is not None)
  model.conv1 = conv_new
  fc_old = model.fc
  fc_new = nn.Linear(fc_old.in_features, 5, fc_old.bias is not None)
  model.fc = fc_new
  return model

def get_model_mobilenet_v3_small():
  model = M.mobilenet_v3_small(pretrained=True)
  fc_old: nn.Linear = model.classifier[-1]
  fc_new = nn.Linear(fc_old.in_features, 5, fc_old.bias is not None)
  model.classifier[-1] = fc_new
  return model

def get_model_cnn():
  class CNN(nn.Module):
    def __init__(self):
      super().__init__()
      self.cnn = nn.Sequential(
        nn.Conv2d(3, 6, kernel_size=3, padding=1),
        nn.AvgPool2d(2, 2),
        nn.ReLU(),
        nn.Conv2d(6, 12, kernel_size=3, padding=1),
        nn.AvgPool2d(2, 2),
        nn.ReLU(),
        nn.Conv2d(12, 6, kernel_size=3, padding=1),
        nn.AvgPool2d(2, 2),
        nn.ReLU(),
        nn.Conv2d(6, 3, kernel_size=3, padding=1),
        nn.AvgPool2d(2, 2),
        nn.ReLU(),
        nn.Flatten(start_dim=1),
        nn.Linear(12, 5),
      )
    def forward(self, x):
      return self.cnn(x)
  return CNN()

def get_model_cnn_d3():
  class CNN_d3(nn.Module):
    def __init__(self):
      super().__init__()
      self.cnn = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=3, padding=1),
        nn.AvgPool2d(2, 2),
        nn.ReLU(),
        nn.Conv2d(3, 3, kernel_size=3, padding=1),
        nn.AvgPool2d(2, 2),
        nn.ReLU(),
        nn.Conv2d(3, 3, kernel_size=3, padding=1),
        nn.AvgPool2d(2, 2),
        nn.ReLU(),
        nn.Flatten(start_dim=1),
        nn.Linear(48, 5),
      )
    def forward(self, x):
      return self.cnn(x)
  return CNN_d3()

def get_model_cnn_d1():
  class CNN_d1(nn.Module):
    def __init__(self):
      super().__init__()
      self.cnn = nn.Sequential(
        nn.Conv2d(3, 1, kernel_size=3, padding=1),
        nn.AvgPool2d(2, 2),
        nn.ReLU(),
        nn.Conv2d(1, 1, kernel_size=3, padding=1),
        nn.AvgPool2d(2, 2),
        nn.ReLU(),
        nn.Conv2d(1, 1, kernel_size=3, padding=1),
        nn.AvgPool2d(2, 2),
        nn.ReLU(),
        nn.Flatten(start_dim=1),
        nn.Linear(16, 5),
      )
    def forward(self, x):
      return self.cnn(x)
  return CNN_d1()

def get_model_cnn_d1_L():
  class CNN_d1_L(nn.Module):
    def __init__(self):
      super().__init__()
      self.cnn = nn.Sequential(
        nn.Conv2d(3, 1, kernel_size=3, padding=1),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(1, 1, kernel_size=3, padding=1),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(1, 1, kernel_size=3, padding=1),
        nn.AvgPool2d(2, 2),
        nn.Flatten(start_dim=1),
        nn.Linear(16, 5),
      )
    def forward(self, x):
      return self.cnn(x)
  return CNN_d1_L()

def get_model_cnn_d1_s2():
  class CNN_d1_s2(nn.Module):
    def __init__(self):
      super().__init__()
      self.cnn = nn.Sequential(
        nn.Conv2d(3, 1, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1),
        nn.Linear(16, 5),
      )
    def forward(self, x):
      return self.cnn(x)
  return CNN_d1_s2()

def get_model_cnn_d1_s2_nb():
  class CNN_d1_s2_nb(nn.Module):
    def __init__(self):
      super().__init__()
      self.cnn = nn.Sequential(
        nn.Conv2d(3, 1, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(),
        nn.Flatten(start_dim=1),
        nn.Linear(16, 5, bias=False),
      )
    def forward(self, x):
      return self.cnn(x)
  return CNN_d1_s2_nb()

def get_model_cnn_d1_s2_x16():
  class CNN_d1_s2_x16(nn.Module):
    def __init__(self):
      super().__init__()
      self.cnn = nn.Sequential(
        nn.Conv2d(3, 1, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1),
        nn.Linear(4, 5),
      )
    def forward(self, x):
      return self.cnn(x)
  return CNN_d1_s2_x16()

def get_model_cnn_d1_s2_x16_nb():
  class CNN_d1_s2_x16_nb(nn.Module):
    def __init__(self):
      super().__init__()
      self.cnn = nn.Sequential(
        nn.Conv2d(3, 1, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(),
        nn.Flatten(start_dim=1),
        nn.Linear(4, 5, bias=False),
      )
    def forward(self, x):
      return self.cnn(x)
  return CNN_d1_s2_x16_nb()

def get_model_cnn_d1_s2_x16_L():
  class CNN_d1_s2_x16_L(nn.Module):
    def __init__(self):
      super().__init__()
      self.cnn = nn.Sequential(
        nn.Conv2d(3, 1, kernel_size=3, stride=2, padding=1, bias=False),
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
        nn.Flatten(start_dim=1),
        nn.Linear(4, 5, bias=False),
      )
    def forward(self, x):
      return self.cnn(x)
  return CNN_d1_s2_x16_L()

def get_model_cnn_nano():
  class CNN_nano(nn.Module):    # 最接近 QAMCNN 的工作方式
    def __init__(self):
      super().__init__()
      self.cnn = nn.Sequential(
        nn.Conv2d(3, 1, kernel_size=2, stride=2, bias=False),
        nn.Conv2d(1, 1, kernel_size=2, stride=2, bias=False),
        nn.Conv2d(1, 1, kernel_size=2, stride=2, bias=False),
        nn.Conv2d(1, 1, kernel_size=2, stride=2, bias=False),
        nn.Flatten(start_dim=1),
        nn.Linear(4, 5, bias=False),
      )
    def forward(self, x):
      return self.cnn(x)
  return CNN_nano()


model_getters = [v for k, v in globals().items() if k.startswith('get_model')]
#model_getters = [v for k, v in globals().items() if k.startswith('get_model_cnn')]
#model_getters = [get_model_cnn_d1_L]


trainset = CIFAR10Dataset(train=True)
testset  = CIFAR10Dataset(train=False)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  drop_last=True)
testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, drop_last=False)
print('len(trainset):', len(trainset), 'len(trainloader):', len(trainloader))
print('len(testset):',  len(testset),  'len(testloader):',  len(testloader))


for get_model in model_getters:
  model = get_model().to(device)
  print('param_cnt:', sum(p.numel() for p in model.parameters() if p.requires_grad))
  train(model, trainloader, testloader)
