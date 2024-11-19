# 训练二分类模型 (针对性地重训精度差的子模型)

import os
import pickle as pkl
from collections import Counter
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader, random_split

from model import QuantumNeuralNetwork, QuantumNeuralNetworkCL
from utils import PerfectAmplitudeEncodingDataset
from train import train_model


parser = ArgumentParser()
parser.add_argument('-T', '--targets', default='0,1', type=(lambda s: [int(e) for e in s.split(',')]), help='class labels for binary clf')
args = parser.parse_args()

assert len(args.targets) == 2
A, B = args.targets
assert 0 <= A <= 4 and 0 <= B <= 4
key = f'bin_{A}-{B}'

DEVICE = "cuda:0"
OUTPUT_DIR = './output'
SUB_OUTPUT_DIR = os.path.join(OUTPUT_DIR, key)
os.makedirs(SUB_OUTPUT_DIR, exist_ok=True)
assert os.path.isdir(OUTPUT_DIR)
model_fp = f'{OUTPUT_DIR}/best_model.pt'
submodel_fp = f'{SUB_OUTPUT_DIR}/best_model.pt'
submodel_optim_fp = f'{SUB_OUTPUT_DIR}/best_model.optim.pt'


''' HParams '''
NUM_EPOCHS = 72         # [30, 50, 75]
BATCH_SIZE = 128        # [32, 64, 128]
LR         = 0.001      # [0.01, 0.004, 0.001, 2e-4]
RESUME     = True       # [False, True]

''' Data '''
dataset = PerfectAmplitudeEncodingDataset(train=True)   # 用全部数据训练，防止过拟合
print('dataset labels:', Counter(it[1].item() for it in dataset))
dataset.quantum_dataset = [(it[0], torch.tensor(args.targets.index(it[1].item()), dtype=torch.long), it[2]) for it in dataset.quantum_dataset if it[1].item() in args.targets]
print('dataset labels:', Counter(it[1].item() for it in dataset))
train_size = int(0.7 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

''' Ckpt '''
if RESUME and os.path.exists(submodel_fp):
    print(f'>> load best submodel from: {submodel_fp}')
    model_state = torch.load(submodel_fp)
    print(f'>> load best submodel optim from: {submodel_optim_fp}')
    optim_state = torch.load(submodel_optim_fp)
else:
    model_state = None
    optim_state = None

''' SubModel '''
with open(f'{OUTPUT_DIR}/model_config.pkl', 'rb') as file:
    model_config = pkl.load(file)
submodel = QuantumNeuralNetworkCL(**model_config)
if model_state:
    submodel.load_state_dict(model_state)

''' Train '''
submodel = submodel.to(DEVICE)
train_model(submodel, train_loader, valid_loader, num_epochs=NUM_EPOCHS, output_dir=SUB_OUTPUT_DIR, device=DEVICE, optim_state=optim_state)
submodel = submodel.cpu()

''' Save to Ensemble Model '''
model = QuantumNeuralNetwork(**model_config)
model.load_state_dict(torch.load(model_fp))
submodel: QuantumNeuralNetworkCL = model.model2_grid[key]
submodel.load_state_dict(torch.load(submodel_fp))       # overwrite!
torch.save(model.state_dict(), model_fp)
