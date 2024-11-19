import os
import pickle
import random
import multiprocessing
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
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
        self.dataset = torchvision.datasets.CIFAR10(root='/data', train=train, 
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
    image, label = data
    
    # 构建振幅编码线路
    encoding_circuit = dq.QubitCircuit(num_qubits)
    encoding_circuit.rylayer(encode=True)
    for _ in range(num_layers):
        encoding_circuit.rylayer()
   
    # Count gates before encoding
    gate_count = count_gates(encoding_circuit)
    
    encoding_circuit.encode([image.mean()]*num_qubits)
    
    # 优化参数，使得线路能够制备出|x>
    optimizer = torch.optim.Adam(encoding_circuit.parameters(), lr=0.01)
    for _ in range(200):
        state = encoding_circuit()
        loss = 1 - get_fidelity(state.unsqueeze(0), reshape_norm_padding(image))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Detach the state tensor before returning
    return (image, label, encoding_circuit().detach(), gate_count)


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
        num_cores = 20  # todo: 修改为合适的配置
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

    # 实例化测试集 QCIFAR10Dataset 并保存为pickle文件
    OUTPUT_DIR = 'output'
    # 确保目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_dataset = QCIFAR10Dataset(train=False) 
    print('dataset labels', [sample[1].item() for sample in test_dataset])
    with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'wb') as file:
        pickle.dump(test_dataset, file)












