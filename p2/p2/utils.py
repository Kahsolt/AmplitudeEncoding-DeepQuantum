import os
import sys
import pickle
import random
from time import time
import multiprocessing
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
import numpy as np
import deepquantum as dq
from tqdm import tqdm

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

if os.getenv('MY_LABORATORY') or sys.platform == 'win32':
    DATA_PATH = '../../data'
else:
    DATA_PATH = '/data'

get_score = lambda fid, gcnt: 2 * fid - gcnt / 2000


def count_gates(cir):
    # cir is dq.QubitCircuit
    count = 0
    for m in cir.operators.modules():
        if isinstance(m, dq.operation.Gate) and (not isinstance(m, dq.gate.Barrier)):
            count += 1
    return count

def reshape_norm_padding(x):
    # x: CIFAR10 image with shape (3, 32, 32) or (batch_size, 3, 32, 32)
    if x.dim() == 3:
        assert x.shape == (3, 32, 32), f"Expected input shape (3, 32, 32), got {x.shape}"
    elif x.dim() == 4:
        assert x.shape[1:] == (3, 32, 32), f"Expected input shape (batch_size, 3, 32, 32), got {x.shape}"
    else:
        raise ValueError(f"Unexpected input shape: {x.shape}")

    PADDING_SIZE = 4096 - 3 * 32 * 32
    original_shape = x.shape
    x = x.reshape(original_shape[0] if x.dim() == 4 else 1, -1)
    x = F.normalize(x, p=2, dim=1)
    x = F.pad(x, (0, PADDING_SIZE), mode='constant', value=0)
    x = x.to(torch.complex64)
    return x.unsqueeze(-1)  # (batch_size, 4096, 1)

def get_fidelity(state_pred, state_true):
    # state_pred, state_true: (batch_size, 4096, 1)
    state_pred = state_pred.view(-1, 4096)
    state_true = state_true.view(-1, 4096)
    fidelity = torch.abs(torch.matmul(state_true.conj(), state_pred.T)) ** 2
    return fidelity.diag().mean()

def get_fidelity_fast(state_pred, state_true):
    # state_pred, state_true: (batch_size, 4096, 1)
    state_pred = state_pred.view(-1, 4096).real
    state_true = state_true.view(-1, 4096).real
    fidelity = (state_pred * state_true).sum(-1)**2
    return fidelity.mean()

def get_fidelity_NxN(state_pred:Tensor, state_true:Tensor) -> Tensor:
    # state_pred, state_true: (batch_size, 4096, 1)
    state_pred = state_pred.flatten(1).real
    state_true = state_true.flatten(1).real
    fid_mat = torch.abs(torch.matmul(state_true, state_pred.T)) ** 2
    return fid_mat

def get_acc(y_pred, y_true):
    # 计算准确率
    correct = (y_pred == y_true).float()
    accuracy = correct.sum() / len(correct)
    return accuracy.item()


# 将数据的平均值调整为接近0 大部分像素值会落在[-1, 1]范围内
cifar10_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

class CIFAR10Dataset(Dataset):
    def __init__(self, train:bool = True, size:int = 100000000):
        """
        随机选择5个类构造CIFAR10数据集；测试集每个类仅随机抽选100个测试样本。
        Args:
            train (bool): 是否加载训练数据集，如果为 False，则加载测试数据集。默认为 True。
            size (int): 数据集的大小。
        """
        self.dataset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=train, 
                                                    download=True, transform=cifar10_transforms)

        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                            'dog', 'frog', 'horse', 'ship', 'truck']
        
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


class PerfectAmplitudeEncodingCircuit:
    def __init__(self, image):
        """image: (3, 32, 32)"""
        self.state_vector = reshape_norm_padding(image).squeeze(0)
        self.operators = nn.Sequential()
    def __call__(self):
        return self.state_vector # (4096, 1)

class PerfectAmplitudeEncodingDataset(Dataset):
    def __init__(self, train:bool = True, size:int = 100000000):
        """
        完美振幅编码的CIFAR10Dataset类，跳过了编码线路的构建，可用于快速迭代量子神经网络的训练。
        Args:
            train (bool): 是否加载训练数据集，如果为 False，则加载测试数据集。默认为 True。
            size (int): 数据集的大小。
        """
        
        self.dataset = CIFAR10Dataset(train=train, size=size)
        self.num_total_gates = 0
        self.quantum_dataset = self.encode_data()
        del self.dataset

    def encode_data(self):
        """
        返回: a list of tuples (原始经典数据, 标签, 振幅编码量子线路的输出)=(image, label, state_vector)
        """
        quantum_dataset = []
        for image, label in tqdm(self.dataset):
            encoding_circuit = PerfectAmplitudeEncodingCircuit(image)
            self.num_total_gates += count_gates(encoding_circuit)
            quantum_dataset.append((image, label, encoding_circuit()))
        return quantum_dataset
    
    def get_gates_count(self):
        """计算在这个数据集上的编码线路门的平均个数"""
        return self.num_total_gates / len(self.quantum_dataset)

    def __len__(self):
        return len(self.quantum_dataset)

    def __getitem__(self, idx):
        x = self.quantum_dataset[idx][0]
        y = self.quantum_dataset[idx][1]
        z = self.quantum_dataset[idx][2]
        return x, y, z


# 注意: 决赛的数据集名字必须固定为 QCIFAR10Dataset
# todo: 构建振幅编码线路
def encode_single_data(data, num_qubits=12, num_layers=10):
    image, label = data     # [3, 32, 32], []
    target = reshape_norm_padding(image)
    
    def vqc_F2_all_wise_init_0(nq:int=12, n_rep:int=1):
        ''' RY(single init) - [pairwise(F2) - RY], param zero init '''
        vqc = dq.QubitCircuit(nqubit=nq)
        # MAGIC: 2*arccos(sqrt(2/3)) = 1.2309594173407747, only init wire 0
        g = dq.Ry(nqubit=nq, wires=0, requires_grad=True) ; g.init_para([2.4619188346815495]) ; vqc.add(g)
        for _ in range(n_rep):
            for i in range(nq-1):   # qubit order
                for j in range(i+1, nq):
                    g = dq.Ry(nqubit=nq, wires=j, controls=i, requires_grad=True) ; g.init_para([0.0]) ; vqc.add(g)
                    g = dq.Ry(nqubit=nq, wires=i, controls=j, requires_grad=True) ; g.init_para([0.0]) ; vqc.add(g)
            for i in range(nq):
                g = dq.Ry(nqubit=nq, wires=i, requires_grad=True) ; g.init_para([0.0]) ; vqc.add(g)
        return vqc

    n_iter = 800
    train_iter    = n_iter // 4 * 3
    finetune_iter = n_iter // 4

    best_score = -1
    best_vqc = None
    best_gate_count = None
    best_nlayer = -1
    score_list = []
    for nlayer in [2, 3]:
        vqc = vqc_F2_all_wise_init_0(num_qubits, nlayer)
        gate_count = count_gates(vqc)
        optimizer = torch.optim.Adam(vqc.parameters(), lr=0.1)

        for _ in range(train_iter):
            loss = -get_fidelity_fast(vqc(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        vqc = prune_circuit(vqc, target, optimizer)
        gate_count = count_gates(vqc)

        for _ in range(finetune_iter):
            loss = -get_fidelity_fast(vqc(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fid = get_fidelity_fast(vqc(), target).item()
        score = get_score(fid, gate_count)
        score_list.append(score)
        print('nlayer:', nlayer, 'score:', score, 'fid:', fid, 'gcnt:', gate_count)

        if score > best_score:   # pick the best!
            best_nlayer = nlayer
            best_score = score
            best_vqc = vqc
            best_gate_count = gate_count

    print('>> best_nlayer:', best_nlayer, 'score_list', score_list)
    return (image, label, best_vqc().detach(), best_gate_count)

@torch.inference_mode
def prune_circuit(qc:dq.QubitCircuit, tgt:torch.Tensor, opt:torch.optim.Adam) -> dq.QubitCircuit:
    ''' Trim small rotations '''
    PI = np.pi
    PI2 = np.pi * 2
    def phi_norm(agl:float) -> float:
        agl %= PI2
        if agl > +PI: agl -= PI2
        if agl < -PI: agl += PI2
        return agl

    params = opt.param_groups[0]['params']
    ops: nn.Sequential = qc.operators
    fid = get_fidelity_fast(qc(), tgt).item()
    gcnt = count_gates(qc)
    sc = get_score(fid, gcnt)
    sc_new = sc
    while sc_new >= sc:
        idx_sel = -1
        min_agl = 3.14
        for idx, op in enumerate(ops):
            agl = abs(phi_norm(op.theta.item()))
            if agl < min_agl:
                min_agl = agl
                idx_sel = idx
        if idx_sel < 0: break
        # remove from circuit
        del ops[idx_sel]
        # remove from optimizer
        tensor = params[idx_sel]
        del params[idx_sel]
        del opt.state[tensor]
        # new score
        fid = get_fidelity_fast(qc(), tgt).item()
        gcnt -= 1
        sc_new = get_score(fid, gcnt)

    ''' Remove |0>-ctrl CRY '''
    wires = [False] * qc.nqubit     # 当前 qubit 上是否至少有一个旋转作用
    ops_new = []
    for op in qc.operators:
        if op.controls:
            c = op.controls[0]
            if not wires[c]: continue
        for w in op.wires:
            wires[w] = True
        ops_new.append(op)
    qc.operators = nn.Sequential(*ops_new)

    return qc


class QCIFAR10Dataset(Dataset):
    def __init__(self, train:bool = True, size:int = 100000000):
        """
        初始化 QCIFAR10Dataset 类。
        Args:
            train (bool): 是否加载训练数据集，如果为 False，则加载测试数据集。默认为 True。
            size (int): 数据集的大小。
        """
        self.dataset = CIFAR10Dataset(train=train, size=size)
        self.num_total_gates = 0
        self.quantum_dataset = self.encode_data()
        del self.dataset

    def encode_data(self):
        """
        返回: a list of tuples (原始经典数据, 标签, 振幅编码量子线路的输出)=(image, label, state_vector)
        """
        num_cores = 16  # todo: 修改为合适的配置
        pool = multiprocessing.Pool(num_cores)
        
        # Create a partial function with fixed parameters
        encode_func = partial(encode_single_data, num_qubits=12, num_layers=100)
        
        # Use tqdm to show progress
        results = list(tqdm(pool.imap(encode_func, self.dataset), total=len(self.dataset)))
        
        pool.close()
        pool.join()
        
        # Separate the results and count total gates
        quantum_dataset = [(image, label, state_vector) for image, label, state_vector, _ in results]
        self.num_total_gates = sum(gate_count for _, _, _, gate_count in results)
        
        return quantum_dataset
    
    def get_gates_count(self):
        """计算在这个数据集上的编码线路门的平均个数"""
        return self.num_total_gates / len(self.quantum_dataset)

    def __len__(self):
        return len(self.quantum_dataset)

    def __getitem__(self, idx):
        x = self.quantum_dataset[idx][0]
        y = self.quantum_dataset[idx][1]
        z = self.quantum_dataset[idx][2]
        return x, y, z


if __name__ == '__main__':
    ts_start = time()

    # 实例化测试集 QCIFAR10Dataset 并保存为pickle文件
    OUTPUT_DIR = 'output'
    # 确保目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_dataset = QCIFAR10Dataset(train=False) 
    print('dataset labels:', [sample[1].item() for sample in test_dataset])
    with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'wb') as file:
        pickle.dump(test_dataset, file)

    ts_end = time()
    print(f'>> Done! (timecost: {ts_end - ts_start:5f}s)')      # 15569.065662s
