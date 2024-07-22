from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import deepquantum as dq

from utils import QMNISTDataset, cir_collate_fn, get_fidelity, reshape_norm_padding, count_gates


# 注意: 选手的模型名字必须固定为 QuantumNeuralNetwork
# TODO: 构建表达能力更强的变分量子线路以提高分类准确率
class QuantumNeuralNetwork(nn.Module):

    def __init__(self, n_qubit:int, n_layer:int):
        super().__init__()
        self.n_qubit = n_qubit
        self.n_layer = n_layer
        self.loss_fn = F.cross_entropy
        self.create_var_circuit()

    def create_var_circuit(self):
        """构建变分量子线路"""
        self.var_circuit = dq.QubitCircuit(self.n_qubit)

        if not 'original':          # 89.333% 有点过拟合
            for i in range(self.n_layer):
                self.var_circuit.rzlayer()      # 换成 u3layer 区别不大
                self.var_circuit.rylayer()
                self.var_circuit.rzlayer()
                self.var_circuit.cnot_ring()    # 换成 cxlayer 将很差 ~35%
                self.var_circuit.barrier()

        if not 'original-seq':      # 90.133%
            for i in range(self.n_layer):
                for q in range(self.n_qubit):
                    self.var_circuit.rz(wires=q)
                    self.var_circuit.ry(wires=q)
                    self.var_circuit.rz(wires=q)
                    self.var_circuit.cnot(q, (q+1)%self.n_qubit)

        if not 'HAE-cyclic':        # 88.133%
            for i in range(self.n_layer):
                self.var_circuit.rzlayer()
                self.var_circuit.rylayer()
                self.var_circuit.rzlayer()
                self.var_circuit.cnot_ring()
            self.var_circuit.rzlayer()
            self.var_circuit.rylayer()
            self.var_circuit.rzlayer()

        if not 'HAE-bicyclic':      # 89.067%
            for i in range(self.n_layer):
                self.var_circuit.rzlayer()
                self.var_circuit.rylayer()
                self.var_circuit.rzlayer()
                self.var_circuit.cnot_ring(reverse=(i % 2 == 1))
            self.var_circuit.rzlayer()
            self.var_circuit.rylayer()
            self.var_circuit.rzlayer()

        if not 'CCQC':              # 90.333% (depth=30), 87.867% (depth=10)
            steps = [1, 3, 2] * 10
            for i in range(self.n_layer):
                self.var_circuit.u3layer()
                for q in range(self.n_qubit):
                    c = (q + steps[i]) % self.n_qubit
                    self.var_circuit.cu(c, q)
            self.var_circuit.u3layer()

        if not 'mera-cnot':         # 87.000%
            for i in range(self.n_layer):
                self.var_circuit.u3layer()
                offset = int(i % 2 == 1)
                for q in range(offset, self.n_qubit - offset, 2):
                    self.var_circuit.cnot(q, q + 1)
            self.var_circuit.u3layer()

        if not 'mera-updown-cnot':  # 89.733%
            for i in range(self.n_layer):
                self.var_circuit.u3layer()
                offset = int(i % 2 == 1)
                for q in range(offset, self.n_qubit - offset, 2):
                    if offset: self.var_circuit.cnot(q + 1, q)
                    else:      self.var_circuit.cnot(q, q + 1)
            self.var_circuit.u3layer()

        if not 'mera-updown-crx':   # 88.067%
            for i in range(self.n_layer):
                self.var_circuit.u3layer()
                offset = int(i % 2 == 1)
                for q in range(offset, self.n_qubit - offset, 2):
                    if offset: self.var_circuit.crx(q + 1, q)
                    else:      self.var_circuit.crx(q, q + 1)
            self.var_circuit.u3layer()

        if not 'mera-updown-cry':   # 88.600%/91.467%，有希望
            for i in range(self.n_layer):
                self.var_circuit.u3layer()
                offset = int(i % 2 == 1)
                for q in range(offset, self.n_qubit - offset, 2):
                    if offset: self.var_circuit.cry(q + 1, q)
                    else:      self.var_circuit.cry(q, q + 1)
            self.var_circuit.u3layer()

        if 'mera-updown-cu':        # 88.333%/92.667%
            for i in range(self.n_layer):
                self.var_circuit.u3layer()
                offset = int(i % 2 == 1)
                for q in range(offset, self.n_qubit - offset, 2):
                    if offset: self.var_circuit.cu(q + 1, q)
                    else:      self.var_circuit.cu(q, q + 1)
            self.var_circuit.u3layer()

        if not 'mera-updown CNOT control(z-y-z) + target(x-y-x)':   # 87.133%/91.133%
            for i in range(self.n_layer):
                offset = int(i % 2 == 1)
                for q in range(offset, self.n_qubit - offset, 2):
                    if offset:
                        self.var_circuit.rx(q)
                        self.var_circuit.ry(q)
                        self.var_circuit.rx(q)
                        self.var_circuit.rz(q + 1)
                        self.var_circuit.ry(q + 1)
                        self.var_circuit.rz(q + 1)
                        self.var_circuit.cnot(q + 1, q)
                    else:
                        self.var_circuit.rz(q)
                        self.var_circuit.ry(q)
                        self.var_circuit.rz(q)
                        self.var_circuit.rx(q + 1)
                        self.var_circuit.ry(q + 1)
                        self.var_circuit.rx(q + 1)
                        self.var_circuit.cnot(q, q + 1)
            self.var_circuit.u3layer()

        if not 'swap-like':      # (参数量 4080 =_=||)
            for i in range(self.n_layer):
                steps = [1, 3, 2] * 10
                # up
                self.var_circuit.u3layer()
                for q in range(0, self.n_qubit, 2):
                    self.var_circuit.cu(q, (q+steps[i])%self.n_qubit)
                # down
                self.var_circuit.u3layer()
                for q in range(0, self.n_qubit, 2):
                    self.var_circuit.cu((q+steps[i])%self.n_qubit, q)
                # up
                self.var_circuit.u3layer()
                for q in range(0, self.n_qubit, 2):
                    self.var_circuit.cu(q, (q+steps[i])%self.n_qubit)
            self.var_circuit.u3layer()

        if not 'uccsd-like':        # 86.200%/87.533%
            for i in range(self.n_layer):
                self.var_circuit.u3layer()
                for q in range(self.n_qubit-1):
                    self.var_circuit.cnot(q, q + 1)
                    if q != 0: self.var_circuit.u3(q + 1)
                self.var_circuit.u3(self.n_qubit - 1)
                for q in reversed(range(self.n_qubit-1)):
                    if q != 0: self.var_circuit.u3(q + 1)
                    self.var_circuit.cnot(q, q + 1)
            self.var_circuit.u3layer()

        # num of observable == num of classes
        self.var_circuit.observable(wires=0, basis='z')
        self.var_circuit.observable(wires=1, basis='z')
        self.var_circuit.observable(wires=0, basis='x')
        self.var_circuit.observable(wires=1, basis='x')
        self.var_circuit.observable(wires=0, basis='y')

        print('gate count:', count_gates(self.var_circuit))

    def forward(self, z:Tensor, y:Tensor) -> Tuple[Tensor, Tensor]:
        self.var_circuit(state=z)   
        output = self.var_circuit.expectation()          
        return self.loss_fn(output, y), output

    def inference(self, z:Tensor) -> Tensor:
        """
        推理接口。
        输入：
            数据z是一个batch的图像数据对应的量子态，z的形状是(batch_size, 2**n_qubit, 1)，数据类型是torch.complex64。
        返回：
            output的形状是(batch_size, num_class)，数据类型是torch.float32。
        """
        self.var_circuit(state=z)
        output = self.var_circuit.expectation()     
        return output


if __name__ == '__main__':
    dataset = QMNISTDataset(label_list=[0,1], train=False, size=10)
    x, y, z = dataset[0]
    print('x.shape:', x.shape)
    print('y.shape:', y.shape)
    print('z:', z)

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=cir_collate_fn)
    x, y, z = next(iter(data_loader))
    x, y, z = x.to('cuda:0'), y.to('cuda:0'), z.to('cuda:0')

    model = QuantumNeuralNetwork(n_qubit=10, n_layer=30).to('cuda:0')
    loss, output = model(z, y)
    output = model.inference(z)
    print('loss:', loss)
    print('output.shape:', output.shape)
    print('fidelity:', get_fidelity(z, reshape_norm_padding(x)))
    breakpoint()
