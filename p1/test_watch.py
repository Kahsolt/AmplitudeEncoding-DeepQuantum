#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/26

# 交互式测试检查，可以一次 data_load 多次 test_model

import os
import time
import pickle
import torch
from torch.utils.data import DataLoader

from model import QuantumNeuralNetwork
from utils import MNISTDataset, cir_collate_fn, reshape_norm_padding, get_fidelity, get_acc
from utils import QMNISTDataset, QMNISTDatasetIdea, DataHolder      # keep for unpickle

import sys
sys.path.insert(0, '.')
from test import test_model
# make linter happy :(
try: from .test import test_model
except ImportError: pass


DEVICE = "cuda:0"
OUTPUT_DIR = "output"
BATCH_SIZE = 514    # test samples: 5139


# 数据集
with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'rb') as file:
    test_dataset = pickle.load(file)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=cir_collate_fn)
print('>> load pickle done')

try:
    while True:
        cmd = input('>> input you ckpt (best/<epoch>/*.pt): ').strip()
        if cmd == 'q': break

        # 检查点
        if cmd in ['b', 'best']:
            ckpt_fp = f'{OUTPUT_DIR}/best_model.pt'
        elif cmd.isdigit():
            ckpt_fp = f'{OUTPUT_DIR}/model_epoch={cmd}.pt'
        elif cmd.endswith('.pt'):
            ckpt_fp = cmd if os.path.isfile(cmd) else f'{OUTPUT_DIR}/{cmd}'
        if not os.path.isfile(ckpt_fp):
            print(f'>> Error: invalid ckpt: {ckpt_fp}')
            continue

        try:
            # 模型
            with open(f'{OUTPUT_DIR}/model_config.pkl', 'rb') as file:
                model_config = pickle.load(file)
            model = QuantumNeuralNetwork(**model_config)
            model.load_state_dict(torch.load(ckpt_fp, map_location=torch.device('cpu')))
            print('>> load model done')

            # 测试
            t0 = time.time()
            acc, fid, gates = test_model(model, test_loader, DEVICE)
            torch.cuda.current_stream().synchronize()
            t1 = time.time()
            runtime = t1 - t0

            print(f'test fid: {fid:.3f}')
            print(f'test acc: {acc:.3f}')
            print(f'test gates: {gates:.3f}')
            print(f'runtime: {runtime:.3f}')

            # 计算客观得分
            gates_score = 1 - gates / 1000.0
            runtime_score = 1 - runtime / 360.0
            score = (2 * fid + acc + gates_score + 0.1 * runtime_score) * 100
            print(f'客观得分: {score:.3f}')
        except KeyboardInterrupt:
            print('>> Abort by Ctrl+C')
except KeyboardInterrupt:
    pass
