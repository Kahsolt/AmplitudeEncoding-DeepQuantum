import os
import random
from collections import Counter
from typing import List

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
        if not 'baseline':
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

        # n_layer=10, gcnt=2230, pcnt=3030; acc=43%
        # n_layer=30, gcnt=6612, pcnt=9012; acc=43% (wtf?)
        if 'qcnn':
            vqc = self.var_circuit

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
            # meas
            vqc.observable(7,  basis='z')
            vqc.observable(7,  basis='x')
            vqc.observable(7,  basis='y')
            vqc.observable(11, basis='z')
            vqc.observable(11, basis='x')

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
