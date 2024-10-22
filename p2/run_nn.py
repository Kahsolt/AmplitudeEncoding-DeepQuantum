#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/21 

# 跑神经网络的经典基线 (with pretrained)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as M
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
img_path = './img'

try:
  from utils import CIFAR10Dataset
except ImportError:   # in case we run this single script on cloud machines :) 
  import random
  from torch.utils.data import Dataset
  from torchvision.datasets import CIFAR10
  import torchvision.transforms as T

  DATA_PATH = './data'
  cifar10_transforms = T.ToTensor()
  batch_size = 16

  class CIFAR10Dataset(Dataset):

      def __init__(self, train:bool = True, size:int = 100000000, transform=cifar10_transforms):
          """
          随机选择5个类构造CIFAR10数据集；测试集每个类仅随机抽选100个测试样本。
          Args:
              train (bool): 是否加载训练数据集，如果为 False，则加载测试数据集。默认为 True。
              size (int): 数据集的大小。
          """
          self.dataset = CIFAR10(root=DATA_PATH, train=train, download=True, transform=transform)
          self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
          
          # 随机选择5个类别
          label_list= [8, 7, 9, 1, 2]
          self.label_map = {l: i for i, l in enumerate(label_list)} # [8, 7, 9, 1, 2] -> [0, 1, 2, 3, 4]
          self.inverse_label_map = {i: l for i, l in enumerate(label_list)} # [0, 1, 2, 3, 4] -> [8, 7, 9, 1, 2]
          
          # 从数据集中筛选出我们感兴趣的标签/类别，并且映射标签 
          self.sub_dataset = []
          for image, label in self.dataset:
              if label in label_list:
                  self.sub_dataset.append((image, self.label_map[label]))

          # 如果是测试集，每个类别随机抽选100个样本
          if not train:
              selected_samples = []
              for label in range(5):
                  samples = [sample for sample in self.sub_dataset if sample[1] == label]
                  selected_samples.extend(random.sample(samples, min(100, len(samples))))
              self.sub_dataset = selected_samples
          
          # shuffle
          random.seed(42)
          random.shuffle(self.sub_dataset)
          self.sub_dataset = self.sub_dataset[:size]
          del self.dataset
          
      def get_label_name(self, label):
          return self.class_names[self.inverse_label_map[label]]

      def __len__(self):
          return len(self.sub_dataset)

      def __getitem__(self, idx):
          x = self.sub_dataset[idx][0]
          y = torch.tensor(self.sub_dataset[idx][1], dtype=torch.long)
          return x, y


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
  for epoch in range(30):
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

model_getters = [v for k, v in globals().items() if k.startswith('get_model')]


trainset = CIFAR10Dataset(train=True)
testset  = CIFAR10Dataset(train=False)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  drop_last=True)
testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, drop_last=False)
print('len(trainset):', len(trainset), 'len(trainloader):', len(trainloader))
print('len(testset):',  len(testset),  'len(testloader):',  len(testloader))


'''
[resnet18]
  0 Acc: 73.400%
  1 Acc: 82.200%
  2 Acc: 80.800%
  3 Acc: 81.400%
  4 Acc: 87.600%
  5 Acc: 87.600%
  6 Acc: 78.000%
  7 Acc: 89.200%
  8 Acc: 89.600%
  9 Acc: 87.200%
  10 Acc: 88.600%
  11 Acc: 87.200%
  12 Acc: 88.200%
  13 Acc: 88.000%
  14 Acc: 89.800%
  15 Acc: 88.400%
  16 Acc: 88.200%
  17 Acc: 89.000%
  18 Acc: 90.000%
  19 Acc: 91.000%
  20 Acc: 88.600%
  21 Acc: 87.400%
  22 Acc: 88.600%
  23 Acc: 89.200%
  24 Acc: 89.000%
  25 Acc: 88.400%
  26 Acc: 87.400%
  27 Acc: 88.800%
  28 Acc: 89.400%
  29 Acc: 88.200%
>> best acc: 0.91

[mobilenet_v3_small]
  0 Acc: 72.000%
  1 Acc: 79.400%
  2 Acc: 78.800%
  3 Acc: 82.200%
  4 Acc: 81.000%
  5 Acc: 81.800%
  6 Acc: 82.200%
  7 Acc: 81.400%
  8 Acc: 83.600%
  9 Acc: 83.600%
  10 Acc: 84.400%
  11 Acc: 83.800%
  12 Acc: 82.600%
  13 Acc: 84.600%
  14 Acc: 82.400%
  15 Acc: 83.600%
  16 Acc: 83.600%
  17 Acc: 83.800%
  18 Acc: 84.400%
  19 Acc: 85.200%
  20 Acc: 84.200%
  21 Acc: 85.000%
  22 Acc: 85.200%
  23 Acc: 85.800%
  24 Acc: 85.800%
  25 Acc: 84.800%
  26 Acc: 85.200%
  27 Acc: 84.000%
  28 Acc: 86.600%
  29 Acc: 85.600%
>> best acc: 0.866
'''
for get_model in model_getters:
  model = get_model().to(device)
  train(model, trainloader, testloader)
