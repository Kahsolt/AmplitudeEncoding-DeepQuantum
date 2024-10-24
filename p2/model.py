import os
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import deepquantum as dq

from utils import QCIFAR10Dataset, PerfectAmplitudeEncodingDataset, reshape_norm_padding, get_fidelity, count_gates

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


# 注意: 选手的模型名字必须固定为 QuantumNeuralNetwork
# todo: 构建表达能力更强的变分量子线路以提高分类准确率
class QuantumNeuralNetwork(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.loss_fn = F.cross_entropy
        self.var_circuit = dq.QubitCircuit(num_qubits)
        self.create_var_circuit()

    def create_var_circuit(self):
        """构建变分量子线路"""

        # n_layer=600, gcnt=14400, pcnt=7200
        if 'baseline':
            for i in range(self.num_layers):
                self.var_circuit.rylayer()
                self.var_circuit.cnot_ring()
                self.var_circuit.barrier()
            # num of observable == num of classes
            self.var_circuit.observable(wires=0, basis='z')
            self.var_circuit.observable(wires=1, basis='z')
            self.var_circuit.observable(wires=2, basis='z')
            self.var_circuit.observable(wires=3, basis='z')
            self.var_circuit.observable(wires=4, basis='z')

        # n_layer=80, gcnt=6850, pcnt=20550; acc=0.340
        if not 'real mera-like, [u3-enc-u3-dec]-enc-u3':
            for i in range(self.num_layers):
                self.var_circuit.u3layer()
                # down (->12-10-8-6-4-2)
                for offset in range(self.num_qubits // 2):
                    for q in range(offset, self.num_qubits - 1 - offset, 2):
                        self.var_circuit.cu(q, q + 1)
                        self.var_circuit.cu(q + 1, q)
                if i < self.num_layers-1:
                    self.var_circuit.u3layer(wires=[5, 6])
                    # up (->2-4-6-8-10)
                    for offset in range(self.num_qubits // 2 - 1, 0, -1):
                        for q in range(offset, self.num_qubits - 1 - offset, 2):
                            self.var_circuit.cu(q + 1, q)
                            self.var_circuit.cu(q, q + 1)
            self.var_circuit.u3layer(wires=[5, 6])

            self.var_circuit.observable(wires=5, basis='z')
            self.var_circuit.observable(wires=6, basis='z')
            self.var_circuit.observable(wires=5, basis='x')
            self.var_circuit.observable(wires=6, basis='x')
            self.var_circuit.observable(wires=5, basis='y')

        # n_layer=80, gcnt=8812, pcnt=7852; acc=
        if not 'real mera-like, [ry-enc-ry-dec]-ry':
            for i in range(self.num_layers):
                self.var_circuit.rylayer()
                # down (->12-10-8-6-4-2)
                for offset in range(self.num_qubits // 2):
                    for q in range(offset, self.num_qubits - 1 - offset, 2):
                        self.var_circuit.cry(q, q + 1)
                        self.var_circuit.cry(q + 1, q)
                self.var_circuit.rylayer(wires=[5, 6])
                # up (->2-4-6-8-10)
                for offset in range(self.num_qubits // 2 - 1, 0, -1):
                    for q in range(offset, self.num_qubits - 1 - offset, 2):
                        self.var_circuit.cry(q + 1, q)
                        self.var_circuit.cry(q, q + 1)
            self.var_circuit.rylayer()

            self.var_circuit.observable(wires=0, basis='z')
            self.var_circuit.observable(wires=1, basis='z')
            self.var_circuit.observable(wires=2, basis='z')
            self.var_circuit.observable(wires=3, basis='z')
            self.var_circuit.observable(wires=4, basis='z')

        print('classifier gate count:', count_gates(self.var_circuit))

    def forward(self, z, y):
        self.var_circuit(state=z)   
        output = self.var_circuit.expectation()          
        return self.loss_fn(output, y), output

    @torch.inference_mode()
    def inference(self, z):
        """
        推理接口。
        输入：
            数据z是一个batch的图像数据对应的量子态，z的形状是(batch_size, 2**num_qubits, 1)，数据类型是torch.complex64。
        返回：
            output的形状是(batch_size, num_class)，数据类型是torch.float32。
        """
        self.var_circuit(state=z)
        output = self.var_circuit.expectation()     
        return output


if __name__ == '__main__':
    # dataset = QCIFAR10Dataset(train=True, size=20) 
    # dataset = QCIFAR10Dataset(train=False, size=20) 
    dataset = PerfectAmplitudeEncodingDataset(train=True)
    print('dataset labels:', Counter(sample[1].item() for sample in dataset))
    print('gates_count:', dataset.get_gates_count())

    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    x, y, z = next(iter(data_loader))
    x, y, z = x.to('cuda:0'), y.to('cuda:0'), z.to('cuda:0')
    print('x:', x.shape)
    print('y:', y.shape)
    print('z:', z.shape)
    print('fid:', get_fidelity(z, reshape_norm_padding(x)))
    
    # 创建一个量子神经网络模型
    model = QuantumNeuralNetwork(num_qubits=12, num_layers=600).to('cuda:0')
    loss, output = model(z, y)
    output = model.inference(z)
    print('output', output)
