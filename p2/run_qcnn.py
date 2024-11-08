#!/usr/bin/env python3
# Author: Armit
# Create Time: 周四 2024/11/07 

# 用标准 QCNN 训练 **二分类模型**
# FIXME: deepquantum 做不到 CNN 参数共享，无论如何先试试大概效果吧 :(

import os
from argparse import ArgumentParser

if __name__ == '__main__':    # for fast check & ignore
  parser = ArgumentParser()
  parser.add_argument('-M', '--model',      default='qcnn', choices=['qcnn', 'mera'], help='ansatz model')
  parser.add_argument('-L', '--n_layers',   default=1, type=int, help='ansatz layers')
  parser.add_argument('-T', '--targets',    default='0,1', type=(lambda s: [int(e) for e in s.split(',')]), help='class labels for binary clf')
  parser.add_argument('-E', '--epochs',     default=50, type=int)
  parser.add_argument('-B', '--batch_size', default=32, type=int)
  args = parser.parse_args()

  assert len(args.targets) == 2
  A, B = args.targets
  assert 0 <= A <= 4 and 0 <= B <= 4

  OUTPUT_DIR = os.path.join('output', f'{args.model}_{A}-{B}')
  if os.path.isdir(OUTPUT_DIR):
    print(f'>> ignored, folder exists: {OUTPUT_DIR}')
    exit(0)
  os.makedirs(OUTPUT_DIR, exist_ok=True)

import json
from time import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
import deepquantum as dq

from utils import *

DEVICE = "cuda:0"


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
      quantum_dataset.append((vec, lbl))
    return quantum_dataset


def vqc_qcnn(n_layers:int=1) -> dq.QubitCircuit:
  ''' [arXiv:2107.03630] A Quantum Convolutional Neural Network for Image Classification; 做了一些结构改变来适应 12 比特 '''

  # n_layer=1, gcnt=232, pcnt=312
  nq = 12
  vqc = dq.QubitCircuit(nq) # fig. 3

  def add_U(i:int, j:int):  # conv
    vqc.u3(i) ; vqc.u3(j)
    vqc.cnot(j, i) ; vqc.rz(i) ; vqc.ry(j)
    vqc.cnot(i, j) ;             vqc.ry(j)
    vqc.cnot(j, i)
    vqc.u3(i) ; vqc.u3(j)

  def add_V(i:int, j:int):  # pool
    vqc.u3(i)
    g = dq.U3Gate(nqubit=nq)
    vqc.add(g, wires=j)
    vqc.cnot(i, j)
    vqc.add(g.inverse(), wires=j)

  def add_F(wires:List[int]): # fc, 沿用 CCQC (arXiv:1804.00633)
    wire_p1 = wires[1:] + wires[:1]
    # stride=1
    for i in wires: vqc.u3(i)
    for i, j in zip(wires, wire_p1):
      vqc.cnot(i, j)
      vqc.cnot(j, i)
    # stride=2
    for i in wires: vqc.u3(i)
    for i, j in zip(wire_p1, wires):
      vqc.cnot(i, j)
      vqc.cnot(j, i)

  for _ in range(n_layers):
    # layer1
    add_U(1, 2) ; add_U(3, 4) ; add_U(5, 6) ; add_U(7, 8) ; add_U(9, 10)
    add_U(0, 1) ; add_U(2, 3) ; add_U(4, 5) ; add_U(6, 7) ; add_U(8, 9) ; add_U(10, 11)
    add_V(0, 1) ; add_V(2, 3) ; add_V(4, 5) ; add_V(6, 7) ; add_V(8, 9) ; add_V(10, 11)
    # layer2
    add_U(1, 3) ; add_U(5, 7) ; add_U(9, 11)
    add_U(3, 5) ; add_U(7, 9)
    add_V(3, 5) ; add_V(7, 9)
    # layer3
    add_U(3, 7) ; add_V(3, 7)
    add_U(7, 11) ; add_V(7, 11)
  # fc
  add_F([7, 11])
  # meas
  vqc.observable(7,  basis='z')
  vqc.observable(11, basis='z')
  return vqc

def vqc_mera(n_layers:int=1, with_fc:bool=True) -> dq.QubitCircuit:
  ''' [arXiv:2108.00661] Quantum convolutional neural network for classical data classification; 追加 FC 层，用 CE 代替 MSE '''

  # n_layer=1, gcnt=222, pcnt=327
  nq = 12
  vqc = dq.QubitCircuit(nq) # fig. 5

  def add_U(i:int, j:int):  # conv, 沿用 arXiv:2107.03630
    vqc.u3(i) ; vqc.u3(j)
    vqc.cnot(i, j) ; vqc.ry(i) ; vqc.rz(j)
    vqc.cnot(j, i) ;             vqc.ry(i)
    vqc.cnot(i, j)
    vqc.u3(i) ; vqc.u3(j)

  def add_F(wires:List[int]): # 原论文做二分类，无fc层，此处我们继续沿用 arXiv:2107.03630
    wire_p1 = wires[1:] + wires[:1]
    # stride=1
    for i in wires: vqc.u3(i)
    for i, j in zip(wires, wire_p1):
      vqc.cnot(i, j)
      vqc.cnot(j, i)
    # stride=2
    for i in wires: vqc.u3(i)
    for i, j in zip(wire_p1, wires):
      vqc.cnot(i, j)
      vqc.cnot(j, i)

  # conv
  for _ in range(n_layers):
    add_U(0, 1) ; add_U(2, 3) ; add_U(4, 5) ; add_U(6, 7) ; add_U(8, 9) ; add_U(10, 11)
    add_U(1, 2) ; add_U(3, 4) ; add_U(5, 6) ; add_U(7, 8) ; add_U(9, 10)
    add_U(2, 3) ; add_U(4, 5) ; add_U(6, 7) ; add_U(8, 9)
    add_U(3, 4) ; add_U(5, 6) ; add_U(7, 8)
    add_U(4, 5) ; add_U(6, 7)
    add_U(5, 6)
  # fc
  if with_fc:
    add_F([5, 6])
  # meas
  vqc.observable(5, basis='z')
  vqc.observable(6, basis='z')
  return vqc


class QNN_bin_clf(nn.Module):

  def __init__(self, ansatz:str, num_layers:int):
    super().__init__()

    self.num_layers = num_layers
    self.loss_fn = F.cross_entropy
    self.var_circuit = globals()[f'vqc_{ansatz}'](self.num_layers)

  def forward(self, z, y):
    self.var_circuit(state=z)   
    output = self.var_circuit.expectation()      
    return self.loss_fn(output, y), output

  @torch.inference_mode()
  def inference(self, z):
    self.var_circuit(state=z)
    output = self.var_circuit.expectation()   
    return output


def train_model(model:nn.Module, optimizer:optim.Optimizer, train_loader:DataLoader, valid_loader:DataLoader, num_epochs:int, output_dir:str, device:str='cpu'):
  model.to(device)
  print(f"参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

  os.makedirs(output_dir, exist_ok=True)
  save_path = os.path.join(output_dir, "best_model.pt")
  history_path = os.path.join(output_dir, "loss_history.json")
  fig_path = os.path.join(output_dir, "loss_acc.png")

  best_valid_acc = 0.0
  history = {
    'train_loss': [], 
    'valid_loss': [],
    'train_acc': [], 
    'valid_acc': [],
  }

  pbar = tqdm(total=num_epochs * len(train_loader), position=0, leave=True)
  for epoch in range(num_epochs):
    ''' Train '''
    model.train()
    train_loss = 0.0
    train_acc = 0
    inner_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, position=1)
    for z, y in inner_pbar:
      z, y = z.to(device), y.to(device)
      optimizer.zero_grad()
      loss, output = model(z, y)
      loss.backward()
      optimizer.step()
      train_loss += loss.item() * y.size(0)
      train_acc += (output.argmax(-1) == y).sum().item()
      inner_pbar.set_description(f'Batch Loss: {loss.item():.4f}')
      pbar.update(1)
    train_loss /= len(train_loader.dataset)
    train_acc  /= len(train_loader.dataset)
    history['train_loss'].append(train_loss)
    history['train_acc' ].append(train_acc)
    print(f'Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.7f}, Train acc: {train_acc:.3%}')
    with open(history_path, 'w', encoding='utf-8') as fh:
      history['max_train_acc'] = max(history['train_acc'])
      json.dump(history, fh, indent=2, ensure_ascii=False)

    ''' Eval '''
    model.eval()
    valid_loss = 0.0
    valid_acc = 0
    with torch.no_grad():
      for z, y in valid_loader:
        z, y = z.to(device), y.to(device)
        loss, output = model(z, y)
        valid_loss += loss.item() * y.size(0)
        valid_acc += (output.argmax(-1) == y).sum().item()
    valid_loss /= len(valid_loader.dataset)
    valid_acc  /= len(valid_loader.dataset)
    history['valid_loss'].append(valid_loss)
    history['valid_acc' ].append(valid_acc)
    print(f'Epoch {epoch+1}/{num_epochs} - Valid loss: {valid_loss:.7f}, Valid acc: {valid_acc:.3%}')
    with open(history_path, 'w', encoding='utf-8') as fh:
      history['max_valid_acc'] = max(history['valid_acc'])
      json.dump(history, fh, indent=2, ensure_ascii=False)

    if valid_acc > best_valid_acc:
      best_valid_acc = valid_acc
      torch.save(model.state_dict(), save_path)

  plt.clf()
  plt.plot(history['train_loss'], 'dodgerblue', label='train_loss')
  plt.plot(history['valid_loss'], 'orange',     label='valid_loss')
  ax = plt.twinx()
  ax.plot(history['train_acc'], 'b', label='train_acc')
  ax.plot(history['valid_acc'], 'r', label='valid_acc')
  plt.savefig(fig_path, dpi=400)
  plt.close()


if __name__ == '__main__':
  # 筛选目标俩分类的数据
  dataset = PerfectAmplitudeEncodingDatasetTwoLabel(args.targets, train=True, size=2500)
  print('dataset labels:', Counter(sample[1] for sample in dataset))
  print('dataset len:', len(dataset))
  train_size = int(0.7 * len(dataset))
  valid_size = len(dataset) - train_size
  train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
  # 构建数据加载器，用于加载训练和验证数据
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

  # 创建一个量子神经网络模型
  model = QNN_bin_clf(args.model, args.n_layers)
  print('classifier gcnt:', count_gates(model.var_circuit))
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # 训练模型
  ts_start = time()
  train_model(
    model,
    optimizer,
    train_loader, 
    valid_loader,
    num_epochs=args.epochs, 
    output_dir=OUTPUT_DIR,
    device=DEVICE,
  )
  ts_end = time()
  print('>> train clf_model time cost:', ts_end - ts_start)
