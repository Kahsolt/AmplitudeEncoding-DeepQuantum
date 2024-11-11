#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/11/10 

import os
import json
import pickle
import random
from time import time
from collections import Counter
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, random_split
import deepquantum as dq
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from test import test_model
from utils import PerfectAmplitudeEncodingDataset, reshape_norm_padding, get_fidelity, get_acc, count_gates
from utils import QCIFAR10Dataset       # keep for unpickle

# 对比学习会有用吗？ 有的！

if 'env':
    # 设置随机种子
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # 确定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = "cuda:0"
OUTPUT_DIR = "output"


def get_fidelity_NtoN(state_pred:Tensor, state_true:Tensor) -> Tensor:
    # state_pred, state_true: (batch_size, 4096, 1)
    state_pred = state_pred.flatten(1).real
    state_true = state_true.flatten(1).real
    fid_mat = torch.abs(torch.matmul(state_true, state_pred.T)) ** 2
    return fid_mat


class QuantumNeuralNetwork(nn.Module):

    def __init__(self, num_qubits, num_layers):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.var_circuit = dq.QubitCircuit(num_qubits)
        self.create_var_circuit()

        # acc: 0.546 (no_data_norm) / 0.562 (data_norm)
        self.ref_qstate = None

    def create_var_circuit(self):
        vqc = self.var_circuit

        # n_layer=8,  gcnt=1772, pcnt=2412
        if not 'qcnn':
            def add_U(i:int, j:int):  # conv
                vqc.u3(i) ; vqc.u3(j)
                vqc.cnot(j, i) ; vqc.rz(i) ; vqc.ry(j)
                vqc.cnot(i, j) ;             vqc.ry(j)
                vqc.cnot(j, i)
                vqc.u3(i) ; vqc.u3(j)

            def add_V(i:int, j:int):  # pool
                vqc.u3(i)
                g = dq.U3Gate(nqubit=self.num_qubits)
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

            for _ in range(self.num_layers):
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

        # n_layer=8,  gcnt=1164, pcnt=1164
        if not 'F2_all_0':
            ''' RY - [pairwise(F2) - RY], param zero init '''
            nq = self.num_qubits
            for i in range(nq):
                g = dq.Ry(nqubit=nq, wires=0, requires_grad=True)
                g.init_para([0.0])
                vqc.add(g)
            for _ in range(self.num_layers):
                for i in range(nq-1):   # qubit order
                    for j in range(i+1, nq):
                        g = dq.Ry(nqubit=nq, wires=j, controls=i, requires_grad=True)
                        g.init_para([0.0])
                        vqc.add(g)
                        g = dq.Ry(nqubit=nq, wires=i, controls=j, requires_grad=True)
                        g.init_para([0.0])
                        vqc.add(g)
                for i in range(nq):
                    g = dq.Ry(nqubit=nq, wires=i, requires_grad=True)
                    g.init_para([0.0])
                    vqc.add(g)

        # n_layer=8,  gcnt=1224, pcnt=1656
        # n_layer=10, gcnt=1512, pcnt=2052
        if 'U-V brick':
            def add_U(i:int, j:int):  # conv
                vqc.u3(i) ; vqc.u3(j)
                vqc.cnot(j, i) ; vqc.rz(i) ; vqc.ry(j)
                vqc.cnot(i, j) ;             vqc.ry(j)
                vqc.cnot(j, i)
                vqc.u3(i) ; vqc.u3(j)

            def add_V(i:int, j:int):  # pool
                vqc.u3(i)
                g = dq.U3Gate(nqubit=self.num_qubits)
                vqc.add(g, wires=j)
                vqc.cnot(i, j)
                vqc.add(g.inverse(), wires=j)

            def add_F(wires:List[int]): # fc, 沿用 CCQC (arXiv:1804.00633)
                wire_p1 = wires[1:] + wires[:1]
                wire_p3 = wires[3:] + wires[:3]
                # stride=1
                for i in wires: vqc.u3(i)
                for i, j in zip(wires, wire_p1):
                    vqc.cnot(i, j)
                    vqc.cnot(j, i)
                # stride=3
                for i in wires: vqc.u3(i)
                for i, j in zip(wires, wire_p3):
                    vqc.cnot(i, j)
                    vqc.cnot(j, i)

            for _ in range(self.num_layers):
                add_U(1, 2) ; add_U(3, 4) ; add_U(5, 6) ; add_U(7, 8) ; add_U(9, 10) ; add_U(11, 0)
                add_U(0, 1) ; add_U(2, 3) ; add_U(4, 5) ; add_U(6, 7) ; add_U(8, 9)  ; add_U(10, 11)
                add_V(0, 1) ; add_V(2, 3) ; add_V(4, 5) ; add_V(6, 7) ; add_V(8, 9)  ; add_V(10, 11)
            # fc
            add_F(list(range(12)))  # 后面不接 u3 更好

        for i in range(self.num_qubits):
            vqc.observable(i, 'x')
            vqc.observable(i, 'y')
            vqc.observable(i, 'z')

        print('classifier gate count:', count_gates(vqc))

    def mk_ref_qstate(self, ref_data:Dataset):
        # 类中心的测量结果视作参考
        y_x = {}
        for _, y, x in ref_data:
            x = x.real.flatten()
            y = y.item()
            if y not in y_x: y_x[y] = []
            y_x[y].append(x)
        y_x = sorted([(y, np.stack(xs, axis=-1).mean(axis=-1)) for y, xs in y_x.items()])
        z = torch.from_numpy(np.stack([x for y, x in y_x], axis=0)).to(device=DEVICE, dtype=torch.complex64)
        self.var_circuit(state=z)
        outputs = self.var_circuit.expectation()
        fake_qstate = F.normalize(outputs, dim=-1)
        self.ref_qstate = fake_qstate

    def postprocess(self, outputs:Tensor):
        fake_qstate = F.normalize(outputs, dim=-1)
        ref_states = self.ref_qstate if self.ref_qstate is not None else fake_qstate
        fid_mat = get_fidelity_NtoN(ref_states, fake_qstate)    # [B, NC=5]
        return fid_mat  

    def forward(self, z:Tensor, y:Tensor):
        '''
          A bite of quantum contrastive learning
          - 不能直接对态的保真度进行优化，因为酉变换保正交性 =_=||
          - 那就把一组投影测量指当作某种概率分布来求余弦相似度吧
        '''
        self.var_circuit(state=z)
        outputs = self.var_circuit.expectation()     # [B, M=36]
        fid_mat = self.postprocess(outputs)
        lbl_mat = y.unsqueeze(0) == y.unsqueeze(1)     # 同类 mask
        sum_eq = lbl_mat.sum()              # 同类样本对数
        sum_ne = lbl_mat.numel() - sum_eq   # 不同类样本对数
        fid_eq = (fid_mat *  lbl_mat).sum() / sum_eq
        fid_ne = (fid_mat * ~lbl_mat).sum() / sum_ne
        loss = fid_ne - fid_eq
        return loss, fid_eq, fid_ne

    @torch.inference_mode()
    def inference(self, z):
        self.var_circuit(state=z)
        outputs = self.var_circuit.expectation()    # [B, M=36]
        fid_mat = self.postprocess(outputs)
        return F.softmax(fid_mat, dim=-1)           # [B, NC=5]


def train_model(model:QuantumNeuralNetwork, optimizer:optim.Optimizer, train_loader:DataLoader, valid_loader:DataLoader, num_epochs:int, output_dir:str, device='cpu'):
    model.to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "best_model.pt")
    history_path = os.path.join(output_dir, "loss_history.json")
    fig_path = os.path.join(output_dir, "loss.png")
    pbar = tqdm(total=num_epochs * len(train_loader), position=0, leave=True)
    best_valid_loss = float('inf')
    history = {
        'train_loss': [], 
        'valid_loss': [],
        'valid_fid_eq': [],
        'valid_fid_ne': [],
    }

    for epoch in range(num_epochs):
        ''' Train '''
        model.train()
        train_loss = 0.0
        inner_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, position=1)
        for x, y, z in inner_pbar:
            x, y, z = x.to(device), y.to(device), z.to(device)
            optimizer.zero_grad()
            loss, fid_eq, fid_ne = model(z, y)
            loss.backward()
            #total_norm = clip_grad_norm_(model.parameters(), float('inf'))  # Calculates the total norm value and clips gradients
            total_norm = 0
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            inner_pbar.set_description(f'Batch Loss: {loss.item():.4f} | Grad Norm: {total_norm}')
            pbar.update(1)
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        print(f'Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.7f}')
        with open(history_path, 'w', encoding='utf-8') as fh:
            json.dump(history, fh, indent=2, ensure_ascii=False)

        ''' Eval '''
        model.eval()
        valid_loss = 0.0
        fid_eq_sum, fid_ne_sum = 0.0, 0.0
        with torch.no_grad():
            for x, y, z in valid_loader:
                x, y, z = x.to(device), y.to(device), z.to(device)
                loss, fid_eq, fid_ne = model(z, y)
                n_sample_batch = y.shape[0]
                fid_eq_sum += fid_eq.item() * n_sample_batch
                fid_ne_sum += fid_ne.item() * n_sample_batch
                valid_loss += loss  .item() * n_sample_batch
        n_samples = len(valid_loader.dataset)
        valid_loss /= n_samples
        fid_eq_sum /= n_samples
        fid_ne_sum /= n_samples
        history['valid_loss'  ].append(valid_loss)
        history['valid_fid_eq'].append(fid_eq_sum)
        history['valid_fid_ne'].append(fid_ne_sum)
        print(f'Epoch {epoch+1}/{num_epochs} - Valid loss: {valid_loss:.7f}, fid_eq: {fid_eq_sum}, fid_ne: {fid_ne_sum}')
        with open(history_path, 'w', encoding='utf-8') as fh:
            json.dump(history, fh, indent=2, ensure_ascii=False)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('>> save new best_acc ckpt')
            torch.save(model.state_dict(), save_path)

    pbar.close()

    plt.clf()
    plt.plot(history['train_loss'], 'b', label='train_loss')
    plt.plot(history['valid_loss'], 'r', label='valid_loss')
    plt.savefig(fig_path, dpi=400)
    plt.close()


def run_train():
    BATCH_SIZE = 32 # todo: 修改为合适的配置
    NUM_EPOCHS = 100

    if 'test overfit':      # 实验性地过拟合测试集，使用编码数据
        with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'rb') as fh:
            dataset = pickle.load(fh)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
        valid_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    else:
        dataset = PerfectAmplitudeEncodingDataset(train=True)   # 用全部数据训练，防止过拟合
        print('dataset labels:', Counter(sample[1].item() for sample in dataset))
        train_size = int(0.7 * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        # 构建数据加载器，用于加载训练和验证数据
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 创建一个量子神经网络模型
    model_config = {'num_qubits': 12, 'num_layers': 10} # todo: 修改为合适的配置
    model = QuantumNeuralNetwork(**model_config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 将字典保存到文件中
    with open(f'{OUTPUT_DIR}/model_config.pkl', 'wb') as file:
        pickle.dump(model_config, file)

    # 训练模型
    ts_start = time()
    train_model(
        model,
        optimizer,
        train_loader, 
        valid_loader,
        num_epochs=NUM_EPOCHS, 
        output_dir=OUTPUT_DIR,
        device=DEVICE,
    )
    ts_end = time()
    print('>> train clf_model time cost:', ts_end - ts_start)   # 5531s


def run_test():
    BATCH_SIZE = 64    # todo: 修改为合适的配置

    t0 = time()
    with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'rb') as file:
        test_dataset = pickle.load(file)
    t1 = time()
    print(f'>> load pickle done ({t1 - t0:.3f}s)')      # 0.121s

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    x, y, z = next(iter(test_loader))
    x, y, z = x.to(DEVICE), y.to(DEVICE), z.to(DEVICE)

    with open(f'{OUTPUT_DIR}/model_config.pkl', 'rb') as file:
        model_config = pickle.load(file)
    model = QuantumNeuralNetwork(**model_config).to(DEVICE)
    
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/best_model.pt', map_location=torch.device('cpu')))
    model.mk_ref_qstate(test_dataset)
    output = model.inference(z)

    # 测试模型
    t0 = time()
    acc, fid, gates = test_model(model, test_loader, DEVICE)
    torch.cuda.current_stream().synchronize()
    t1 = time()
    runtime = t1 - t0

    print(f'test fid: {fid:.3f}')
    print(f'test acc: {acc:.3f}')
    print(f'test gates: {gates:.3f}')
    print(f'runtime: {runtime:.3f}')

    # 计算客观得分
    gates_score = 1 - gates / 2000.0 
    runtime_score = 1 - runtime / 360.0
    score = (2 * fid + acc + gates_score + 0.1 * runtime_score) * 100
    print(f'客观得分: {score:.3f}')


if __name__ == '__main__':
    #run_train()
    run_test()
