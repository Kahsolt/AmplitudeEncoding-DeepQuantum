import os
import random
import pickle
from typing import List, Tuple, Dict

import torch
from torch import Tensor
from torch import optim
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as T
import deepquantum as dq
import numpy as np
from tqdm import tqdm

if 'fix seed':
    random.seed(42)
    # 设置np随机种子为固定值，来控制 fake data 的随机性
    np.random.seed(42)
    # 设置torch随机种子为固定值，来控制 PQC 的随机性
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


# MNIST preprocess
# https://gist.github.com/kdubovikov/eb2a4c3ecadd5295f68c126542e59f0a
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,)),
])


def count_gates(cir:dq.QubitCircuit) -> int:
    count = 0
    for m in cir.operators.modules():
        if isinstance(m, dq.operation.Gate) and (not isinstance(m, dq.gate.Barrier)):
            count += 1
    return count


def reshape_norm_padding(x:Tensor) -> Tensor:
    # x: [B, C=1, H=28, W=28]
    x = x.reshape(x.size(0), -1)
    x = F.normalize(x, p=2, dim=-1)
    x = F.pad(x, (0, 1024 - x.size(1)), mode='constant', value=0)
    return x.unsqueeze(-1).to(torch.complex64)  # [B, D=1024, C=1]


def get_fidelity(state_pred:Tensor, state_true:Tensor) -> Tensor:
    state_pred = state_pred.view(-1, 1024)
    state_true = state_true.view(-1, 1024)
    fidelity = torch.abs(torch.matmul(state_true.conj(), state_pred.T)) ** 2
    return fidelity.diag().mean()


def get_acc(y_pred:Tensor, y_true:Tensor) -> Tensor:
    correct = (y_pred == y_true)
    accuracy = correct.sum() / len(correct)
    return accuracy.item()


def cir_collate_fn(batch:List[Tuple[Tensor, int, dq.QubitCircuit]]) -> Tuple[Tensor, Tensor, Tensor]:
    """
    返回：(原始经典数据batch, 标签batch, 振幅编码的量子态矢量batch)
    """
    xs, ys, zs = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    zs = torch.stack([cir() for cir in zs])
    return xs, ys, zs


class FakeDataset(Dataset):     # X-随机生成10个旋转角作AngleEncode；Y-随机二分类标签

    def __init__(self, size=10000, noise_strength=0.0):
        # a list of tuples (x, y, encoding_cir)
        self.data_list = self.generate_fake_data(size, noise_strength)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        x = sample[0]
        y = torch.tensor(sample[1], dtype=torch.long)
        z = sample[2]
        return x, y, z

    def get_gates_count(self):
        """计算在这个数据集上的编码线路门的平均个数"""
        gates_count = 0
        for x, y, encoding_cir in self.data_list:
            gates_count += count_gates(encoding_cir)
        return gates_count / len(self.data_list)

    def generate_fake_data(self, size, noise_strength):
        """Generate fake data, a list of tuples (原始经典数据, 标签, 振幅编码量子线路)=(x1024, label, circuit_x1024) """
        data_list = []
        for i in tqdm(range(size)):
            angles10 = np.random.rand(10) 
            label = np.random.randint(2, size=None)

            # 简单振幅编码量子线路
            circuit_x1024 = dq.QubitCircuit(10)
            circuit_x1024.rylayer(inputs=angles10)
            state = circuit_x1024().squeeze() # (1024,)

            x1024 = state + noise_strength * torch.randn(state.shape) 
            data_list.append((x1024, label, circuit_x1024))
        return data_list


class FakeDatasetApprox(FakeDataset):   # X-随机生成10个旋转角作AngleEncode，用另一个线路来拟合这个AglEnc线路；Y-随机二分类标签

    def generate_fake_data(self, size, noise_strength):
        data_list = []
        for i in tqdm(range(size)):
            angles10 = np.random.rand(10) 
            label = np.random.randint(2, size=None)

            # 生成可以被简单振幅编码量子线路编码的矢量
            circuit = dq.QubitCircuit(10)
            circuit.rylayer(inputs=angles10)
            x1024 = circuit().squeeze() # (1024,)

            # 随机初始化一个参数化的量子线路来自动学习angles10
            circuit_x1024 = dq.QubitCircuit(10)
            circuit_x1024.rylayer()

            # 优化rylayer的参数，使得线路能够制备出x1024
            optimizer = optim.Adam(circuit_x1024.parameters(), lr=0.1)  # NOTE: lr 足够大，loss可降至 1e-7
            for _ in range(200):
                state = circuit_x1024().squeeze()
                loss = 1 - get_fidelity(state, x1024)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            data_list.append((x1024, label, circuit_x1024))
        return data_list


class MNISTDataset(Dataset):

    def __init__(self, label_list:list = [0, 1], train:bool = True, size:int = 100000000):
        """
        初始化 QMNISTDataset 类。
        Args:
            label_list (list): 要使用数据的类别列表，默认为 [0, 1]。
            train (bool): 是否加载训练数据集，如果为 False，则加载测试数据集。默认为 True。
            size (int): 数据集的大小
        """

        self.dataset = MNIST(root='./data', train=train, download=True, transform=transform)

        # 构造原始标签到顺序标签的映射字典
        self.label_map = {}
        self.inverse_label_map = {}
        for i, l in enumerate(label_list):
            self.label_map[l] = i
            self.inverse_label_map[i] = l

        # 从数据集中筛选出我们感兴趣的标签, 并且映射标签
        self.sub_dataset = []
        for image, label in self.dataset:
            if label in label_list:
                self.sub_dataset.append((image, self.label_map[label]))
        self.sub_dataset = self.sub_dataset[:size]

    def __len__(self):
        return len(self.sub_dataset)

    def __getitem__(self, idx):
        sample = self.sub_dataset[idx]
        x = sample[0]
        y = torch.tensor(sample[1], dtype=torch.long)
        return x, y


# 注意: 初赛的数据集名字必须固定为 QMNISTDataset
# TODO: 构建振幅编码线路
class QMNISTDataset(Dataset):

    def __init__(self, label_list:list = [0, 1], train:bool = True, size:int = 100000000, per_cls_size:int = 100000000):
        """
        初始化 QMNISTDataset 类。
        Args:
            label_list (list): 要使用数据的类别列表，默认为 [0, 1]。
            train (bool): 是否加载训练数据集，如果为 False，则加载测试数据集。默认为 True。
            size (int): 数据集的大小
        """

        self.dataset = MNIST(root='./data', train=train, download=True, transform=transform)

        # 构造原始标签到顺序标签的映射字典
        self.label_map = {}
        self.inverse_label_map = {}
        for i, l in enumerate(label_list):
            self.label_map[l] = i
            self.inverse_label_map[i] = l

        # 从数据集中筛选出我们感兴趣的标签, 并且映射标签
        lbl_cnt = {}
        sub_dataset = []
        for image, label in self.dataset:
            if label in label_list:
                lbl = self.label_map[label]
                if lbl not in lbl_cnt: lbl_cnt[lbl] = 0
                lbl_cnt[lbl] += 1
                sub_dataset.append((image, lbl))
        print('original per-label sample count:')
        print(lbl_cnt)

        if any([cnt > per_cls_size for cnt in lbl_cnt.values()]):
            lbl_img: Dict[int, List[Tensor]] = {}
            for img, lbl in sub_dataset:
                if lbl not in lbl_img: lbl_img[lbl] = []
                lbl_img[lbl].append(img)
            sub_dataset = []
            for k, v in lbl_img.items():
                random.shuffle(v)
                v_subset = v[:per_cls_size]
                for v in v_subset:
                    sub_dataset.append((v, k))
            random.shuffle(sub_dataset)
            print(f'>> shrink per-label sample count to {per_cls_size}')

        self.sub_dataset = sub_dataset[:size]
        self.data_list = self.generate_data()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        x = sample[0]
        y = torch.tensor(sample[1], dtype=torch.long)
        z = sample[2]
        return x, y, z

    def get_gates_count(self):
        """计算在这个数据集上的编码线路门的平均个数"""
        gates_count = 0
        for x, y, encoding_cir in self.data_list:
            gates_count += count_gates(encoding_cir)
        return gates_count / len(self.data_list)

    def generate_data(self) -> List[Tuple[Tensor, int, dq.QubitCircuit]]:
        """ a list of tuples (原始经典数据, 标签, 振幅编码量子线路)=(image, label, encoding_circuit) """
        data_list = []
        for image, label in tqdm(self.sub_dataset):
            # 构建振幅编码线路
            # 随机初始化一个参数化的量子线路来自动学习U_w s.t. U_w|0>=|x>
            encoding_circuit = dq.QubitCircuit(10)
            encoding_circuit.rylayer()

            # 优化rylayer的参数，使得线路能够制备出|x>
            optimizer = optim.Adam(encoding_circuit.parameters(), lr=0.1)
            for _ in range(200):
                state = encoding_circuit()
                loss = 1 - get_fidelity(state, reshape_norm_padding(image.unsqueeze(0)))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            data_list.append((image, label, encoding_circuit))
        return data_list


class DataHolder:   # 模仿 QubitCircuit 的函数桩
    def __init__(self, data):
        self.data = data
    def __call__(self):
        return self.data

class QMNISTDatasetIdea(QMNISTDataset):   # 理想的振幅编码数据集，用于直接打桩调试 Ansatz 部分

    def __init__(self, label_list:list = [0, 1], train:bool = True, size:int = 100000000, per_cls_size:int = 100000000):
        super().__init__(label_list, train, size, per_cls_size)

    def get_gates_count(self):
        return 0

    def generate_data(self) -> List[Tuple[Tensor, int, DataHolder]]:
        data_list = []
        for image, label in tqdm(self.sub_dataset):
            data_list.append((image, label, DataHolder(reshape_norm_padding(image.unsqueeze(0)))))
        return data_list


if __name__ == '__main__':
    OUTPUT_DIR = './output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 实例化测试集 QMNISTDataset 并保存为pickle文件
    # 这里取0, 1, 2, 3, 4五个数字，初赛中只需要完成五分类任务
    #test_dataset = QMNISTDataset(label_list=[0,1,2,3,4], train=False)
    test_dataset = QMNISTDatasetIdea(label_list=[0,1,2,3,4], train=False)
    with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'wb') as file:
        pickle.dump(test_dataset, file)
