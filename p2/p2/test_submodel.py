# 测试二分类模型 (从 EnsembleModel/sub_folder 里读)

'''
[测试集精度] (bs=32 epoch=30)
| * |   0   |   1   |   2   |   3   |   4   |
| 0 |       | 0.830 | 0.780 | 0.785 | 0.845 |
| 1 |       |       | 0.850 | 0.865 | 0.750 |
| 2 |       |       |       | 0.730 | 0.860 |
| 3 |       |       |       |       | 0.875 |
| 4 |       |       |       |       |       |
'''

import os
from time import time
from argparse import ArgumentParser
import pickle as pkl

import torch
from torch.utils.data import DataLoader

from utils import QCIFAR10Dataset
from model import QuantumNeuralNetwork, QuantumNeuralNetworkCL
from test import test_model


parser = ArgumentParser()
parser.add_argument('-T', '--targets', default='0,1', type=(lambda s: [int(e) for e in s.split(',')]), help='class labels for binary clf')
parser.add_argument('--tmp', action='store_true', help='load from subfolder instead')
args = parser.parse_args()

assert len(args.targets) == 2
A, B = args.targets
assert 0 <= A <= 4 and 0 <= B <= 4
key = f'bin_{A}-{B}'

OUTPUT_DIR = './output'
assert os.path.isdir(OUTPUT_DIR)
DEVICE = "cuda:0"
BATCH_SIZE = 128


t0 = time()
with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'rb') as file:
    test_dataset: QCIFAR10Dataset = pkl.load(file)
t1 = time()
print(f'>> load pickle done ({t1 - t0:.3f}s)')      # 0.121s

with open(f'{OUTPUT_DIR}/model_config.pkl', 'rb') as file:
    model_config = pkl.load(file)

if args.tmp:
    submodel = QuantumNeuralNetworkCL(**model_config)
    submodel.load_state_dict(torch.load(f'{OUTPUT_DIR}/{key}/best_model.pt'))
else:
    model = QuantumNeuralNetwork(**model_config)
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/best_model.pt'))
    submodel: QuantumNeuralNetworkCL = model.model2_grid[key]
submodel = submodel.to(DEVICE)

test_dataset.quantum_dataset = [(it[0], args.targets.index(it[1].item()), it[2]) for it in test_dataset.quantum_dataset if it[1].item() in args.targets]
print('len(testset):', len(test_dataset))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
_, y, z = next(iter(test_loader))
y, z = y.to(DEVICE), z.to(DEVICE)
output = submodel.inference(z)

# 测试模型
t0 = time()
acc, fid, gates = test_model(submodel, test_loader, DEVICE)
torch.cuda.current_stream().synchronize()
t1 = time()
runtime = t1 - t0

print('=' * 42)
print(f'test acc: {acc:.3%}')
