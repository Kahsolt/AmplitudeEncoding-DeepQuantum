import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
import deepquantum as dq
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import pickle
import random



# random.seed(42)

# # 设置np随机种子为固定值，来控制 fake data 的随机性
# np.random.seed(42)

# # 设置torch随机种子为固定值，来控制 PQC 的随机性
# torch.manual_seed(4)
# torch.cuda.manual_seed_all(4)



def count_gates(cir):
    # cir is dq.QubitCircuit
    count = 0
    for m in cir.operators.modules():
        if isinstance(m, dq.operation.Gate) and (not isinstance(m, dq.gate.Barrier)):
            count += 1
    return count


def reshape_norm_padding(x):
    # x: (batch_size, ...)
    x = x.reshape(x.size(0), -1)
    x = F.normalize(x, p=2, dim=-1)
    x = F.pad(x, (0, 1024 - x.size(1)), mode='constant', value=0)
    return x.unsqueeze(-1).to(torch.complex64) # (batch_size, feaures1024, 1)


def get_fidelity(state_pred, state_true):
    state_pred = state_pred.view(-1, 1024)
    state_true = state_true.view(-1, 1024)
    fidelity = torch.abs(torch.matmul(state_true.conj(), state_pred.T)) ** 2
    return fidelity.diag().mean()



def get_acc(y_pred, y_true):
    # 计算准确率
    correct = (y_pred == y_true).float()
    accuracy = correct.sum() / len(correct)
    return accuracy.item()


class FakeDataset(Dataset):
    def __init__(self, size=10000, noise_strength=0.0):
        # a list of tuples (x, y, encoding_cir)
        self.data_list = self.generate_fake_data(size, noise_strength)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x = self.data_list[idx][0]
        y = torch.tensor(self.data_list[idx][1], dtype=torch.long)
        z = self.data_list[idx][2]
        return x, y, z
    
    def get_gates_count(self):
        """计算在这个数据集上的编码线路门的平均个数"""
        gates_count = 0
        for x, y, encoding_cir in self.data_list:
            gates_count += count_gates(encoding_cir)
        return gates_count / len(self.data_list)
    
    def generate_fake_data(self, size, noise_strength):
        """Generate fake data, a list of tuples 
        (原始经典数据, 标签, 振幅编码量子线路)=(x1024, label, circuit_x1024)
        """
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







class FakeDatasetApprox(Dataset):
    def __init__(self, size=10000, noise_strength=0.0):
        # a list of tuples (x, y, encoding_cir)
        self.data_list = self.generate_fake_data(size, noise_strength)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x = self.data_list[idx][0]
        y = torch.tensor(self.data_list[idx][1], dtype=torch.long)
        z = self.data_list[idx][2]
        return x, y, z
    
    def get_gates_count(self):
        """计算在这个数据集上的编码线路门的平均个数"""
        gates_count = 0
        for x, y, encoding_cir in self.data_list:
            gates_count += count_gates(encoding_cir)
        return gates_count / len(self.data_list)
    
    def generate_fake_data(self, size, noise_strength):
        """Generate fake data, a list of tuples 
        (原始经典数据, 标签, 振幅编码量子线路)=(x1024, label, circuit_x1024)
        """
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
            optimizer = optim.Adam(circuit_x1024.parameters(), lr=0.01)
            for _ in range(200):
                state = circuit_x1024().squeeze()
                loss = 1 - get_fidelity(state, x1024)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            data_list.append((x1024, label, circuit_x1024))

        return data_list





        


def cir_collate_fn(batch):
    """
    返回：(原始经典数据batch, 标签batch, 振幅编码的量子态矢量batch)
    """
    xs, ys, zs = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    zs = torch.stack([cir() for cir in zs])
    return xs, ys, zs








Transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])



class MNISTDataset(Dataset):
    def __init__(self, label_list:list = [0, 1], train:bool = True, size:int = 100000000):
        """
        初始化 QMNISTDataset 类。
        Args:
            label_list (list): 要使用数据的类别列表，默认为 [0, 1]。
            train (bool): 是否加载训练数据集，如果为 False，则加载测试数据集。默认为 True。
            size (int): 数据集的大小
        """
        self.dataset = torchvision.datasets.MNIST(root='/data', train=train, download=True, transform=Transforms)
        
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
        x = self.sub_dataset[idx][0]
        y = torch.tensor(self.sub_dataset[idx][1], dtype=torch.long)
        return x, y

    






# 注意: 初赛的数据集名字必须固定为 QMNISTDataset
# todo: 构建振幅编码线路
class QMNISTDataset(Dataset):
    def __init__(self, label_list:list = [0, 1], train:bool = True, size:int = 100000000):
        """
        初始化 QMNISTDataset 类。
        Args:
            label_list (list): 要使用数据的类别列表，默认为 [0, 1]。
            train (bool): 是否加载训练数据集，如果为 False，则加载测试数据集。默认为 True。
            size (int): 数据集的大小
        """
        self.dataset = torchvision.datasets.MNIST(root='/data', train=train, download=True, transform=Transforms)
        
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

        self.data_list = self.generate_data()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x = self.data_list[idx][0]
        y = torch.tensor(self.data_list[idx][1], dtype=torch.long)
        z = self.data_list[idx][2]
        return x, y, z

    def get_gates_count(self):
        """计算在这个数据集上的编码线路门的平均个数"""
        gates_count = 0
        for x, y, encoding_cir in self.data_list:
            gates_count += count_gates(encoding_cir)
        return gates_count / len(self.data_list)
    
    
    def generate_data(self):
        """
        返回： a list of tuples (原始经典数据, 标签, 振幅编码量子线路)=(image, label, encoding_circuit)
        """
        data_list = []
        for image, label in tqdm(self.sub_dataset):
            # 构建振幅编码线路
            # 随机初始化一个参数化的量子线路来自动学习U_w s.t. U_w|0>=|x>
            encoding_circuit = dq.QubitCircuit(10)
            encoding_circuit.rylayer()
            
            # # 优化rylayer的参数，使得线路能够制备出|x>
            # optimizer = optim.Adam(encoding_circuit.parameters(), lr=0.01)
            # for _ in range(200):
            #     state = encoding_circuit()
            #     loss = 1 - get_fidelity(state, reshape_norm_padding(image.unsqueeze(0)))
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()

            data_list.append((image, label, encoding_circuit))

        return data_list






if __name__ == '__main__':

    OUTPUT_DIR = 'output'

    # dataset = FakeDatasetApprox(size=20, noise_strength=0.0)
    # print(dataset[0])
    # print(dataset.get_gates_count())

    # 实例化测试集 QMNISTDataset 并保存为pickle文件
    # 这里取0, 1, 2, 3, 4五个数字，初赛中只需要完成五分类任务
    test_dataset = QMNISTDataset(label_list=[0,1,2,3,4], train=False)

    with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'wb') as file:
        pickle.dump(test_dataset, file)

    












