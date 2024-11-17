#!/usr/bin/env python3
# Author: Armit
# Create Time: 周日 2024/11/17 

# 组装一个 CLCascade 模型出来，保存为 *.pt

import pickle as pkl

import torch
from model import QuantumNeuralNetworkCLCascade

OUTPUT_DIR = "output"

# 复用底模的 model_config.pkl
with open(f'{OUTPUT_DIR}/model_config.pkl', 'rb') as file:
  model_config = pkl.load(file)
model = QuantumNeuralNetworkCLCascade(**model_config)

# 加载 model5
fp = f'{OUTPUT_DIR}/best_model.pt'
print(f'>> load from {fp}')
model.model5.load_state_dict(torch.load(fp))

# 加载 model2
for i in range(4):
  for j in range(i+1, 5):
    fp = f'{OUTPUT_DIR}/bin_{i}-{j}/best_model.pt'
    print(f'>> load from {fp}')
    model.model2_grid[f'bin_{i}-{j}'].load_state_dict(torch.load(fp))

fp = f'{OUTPUT_DIR}/best_model.cascade.pt'
torch.save(model.state_dict(), fp)
print(f'>> save to {fp}')
