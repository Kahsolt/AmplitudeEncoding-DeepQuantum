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

Gate = dq.operation.Operation


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

        vqc = self.var_circuit

        # n_layer=600, gcnt=14400, pcnt=7200
        if not 'baseline':
            for i in range(self.num_layers):
                vqc.rylayer()
                vqc.cnot_ring()
                vqc.barrier()
            # num of observable == num of classes
            vqc.observable(wires=0, basis='z')
            vqc.observable(wires=1, basis='z')
            vqc.observable(wires=2, basis='z')
            vqc.observable(wires=3, basis='z')
            vqc.observable(wires=4, basis='z')

        # n_layer=80, gcnt=6850, pcnt=20550; acc=0.340
        if not 'real mera-like, [u3-enc-u3-dec]-enc-u3':
            for i in range(self.num_layers):
                vqc.u3layer()
                # down (->12-10-8-6-4-2)
                for offset in range(self.num_qubits // 2):
                    for q in range(offset, self.num_qubits - 1 - offset, 2):
                        vqc.cu(q, q + 1)
                        vqc.cu(q + 1, q)
                if i < self.num_layers-1:
                    vqc.u3layer(wires=[5, 6])
                    # up (->2-4-6-8-10)
                    for offset in range(self.num_qubits // 2 - 1, 0, -1):
                        for q in range(offset, self.num_qubits - 1 - offset, 2):
                            vqc.cu(q + 1, q)
                            vqc.cu(q, q + 1)
            vqc.u3layer(wires=[5, 6])

            vqc.observable(wires=5, basis='z')
            vqc.observable(wires=6, basis='z')
            vqc.observable(wires=5, basis='x')
            vqc.observable(wires=6, basis='x')
            vqc.observable(wires=5, basis='y')

        # n_layer=80, gcnt=8812, pcnt=7852; acc=
        if not 'real mera-like, [ry-enc-ry-dec]-ry':
            for i in range(self.num_layers):
                vqc.rylayer()
                # down (->12-10-8-6-4-2)
                for offset in range(self.num_qubits // 2):
                    for q in range(offset, self.num_qubits - 1 - offset, 2):
                        vqc.cry(q, q + 1)
                        vqc.cry(q + 1, q)
                vqc.rylayer(wires=[5, 6])
                # up (->2-4-6-8-10)
                for offset in range(self.num_qubits // 2 - 1, 0, -1):
                    for q in range(offset, self.num_qubits - 1 - offset, 2):
                        vqc.cry(q + 1, q)
                        vqc.cry(q, q + 1)
            vqc.rylayer()

            vqc.observable(wires=0, basis='z')
            vqc.observable(wires=1, basis='z')
            vqc.observable(wires=2, basis='z')
            vqc.observable(wires=3, basis='z')
            vqc.observable(wires=4, basis='z')

        # n_layer=8,  gcnt=1772, pcnt=2412; acc=43%
        # n_layer=10, gcnt=2230, pcnt=3030; acc=43%
        # n_layer=30, gcnt=6612, pcnt=9012; acc=43% (wtf?)
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
            # meas
            vqc.observable(7,  basis='z')
            vqc.observable(7,  basis='x')
            vqc.observable(7,  basis='y')
            vqc.observable(11, basis='z')
            vqc.observable(11, basis='x')

        # [std_flatten]
        # n_layer=24, gcnt=5292, pcnt=1524; best overfit acc=42.8%
        # [qam_flatten]
        # n_layer=8,  gcnt=1772, pcnt=516;  best overfit acc=38.6%
        # n_layer=12, gcnt=2652, pcnt=768;  best overfit acc=41.0%
        # n_layer=24, gcnt=5292, pcnt=1524; best overfit acc=44.4%
        if not 'real qcnn (3 blocks)':     # parameter shared!
            def mk_U_gates() -> List[Gate]:
                return [
                    dq.U3Gate(nqubit=self.num_qubits, requires_grad=True),
                    dq.U3Gate(nqubit=self.num_qubits, requires_grad=True),
                    dq.Rz    (nqubit=self.num_qubits, requires_grad=True),
                    dq.Ry    (nqubit=self.num_qubits, requires_grad=True),
                    dq.Ry    (nqubit=self.num_qubits, requires_grad=True),
                    dq.U3Gate(nqubit=self.num_qubits, requires_grad=True),
                    dq.U3Gate(nqubit=self.num_qubits, requires_grad=True),
                ]

            def add_U(i:int, j:int, gates:List[Gate]):  # conv
                vqc.add(gates[0], wires=i) ; vqc.add(gates[1], wires=j)
                vqc.cnot(j, i) ; vqc.add(gates[2], wires=i) ; vqc.add(gates[3], wires=j)
                vqc.cnot(i, j) ;                              vqc.add(gates[4], wires=j)
                vqc.cnot(j, i)
                vqc.add(gates[5], wires=i) ; vqc.add(gates[6], wires=j)

            def mk_V_gates() -> List[Gate]:
                g = dq.U3Gate(nqubit=self.num_qubits, requires_grad=True)
                return [
                    dq.U3Gate(nqubit=self.num_qubits, requires_grad=True),
                    g, 
                    g.inverse(),
                ]

            def add_V(i:int, j:int, gates:List[Gate]):  # pool
                vqc.add(gates[0], wires=i)
                vqc.add(gates[1], wires=j)
                vqc.cnot(i, j)
                vqc.add(gates[2], wires=j)

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
                gs_U = mk_U_gates()
                add_U(1, 2, gs_U) ; add_U(3, 4, gs_U) ; add_U(5, 6, gs_U) ; add_U(7, 8, gs_U) ; add_U(9, 10, gs_U)
                add_U(0, 1, gs_U) ; add_U(2, 3, gs_U) ; add_U(4, 5, gs_U) ; add_U(6, 7, gs_U) ; add_U(8,  9, gs_U) ; add_U(10, 11, gs_U)
                gs_V = mk_V_gates()
                add_V(0, 1, gs_V) ; add_V(2, 3, gs_V) ; add_V(4, 5, gs_V) ; add_V(6, 7, gs_V) ; add_V(8,  9, gs_V) ; add_V(10, 11, gs_V)
                # layer2
                gs_U = mk_U_gates()
                add_U(1, 3, gs_U) ; add_U(5, 7, gs_U) ; add_U(9, 11, gs_U)
                add_U(3, 5, gs_U) ; add_U(7, 9, gs_U)
                gs_V = mk_V_gates()
                add_V(3, 5, gs_V) ; add_V(7, 9, gs_V)
                # layer3
                gs_U = mk_U_gates()
                gs_V = mk_V_gates()
                add_U(3,  7, gs_U) ; add_V(3,  7, gs_V)
                add_U(7, 11, gs_U) ; add_V(7, 11, gs_V)
            # fc
            add_F([7, 11])
            # meas
            vqc.observable(7,  basis='z')
            vqc.observable(7,  basis='x')
            vqc.observable(7,  basis='y')
            vqc.observable(11, basis='z')
            vqc.observable(11, basis='x')

        # n_layer=12, gcnt=2940, pcnt=768;  best overfit acc=41.4%
        if not 'real qcnn (3 blocks), cyclic':     # parameter shared!
            def mk_U_gates() -> List[Gate]:
                return [
                    dq.U3Gate(nqubit=self.num_qubits, requires_grad=True),
                    dq.U3Gate(nqubit=self.num_qubits, requires_grad=True),
                    dq.Rz    (nqubit=self.num_qubits, requires_grad=True),
                    dq.Ry    (nqubit=self.num_qubits, requires_grad=True),
                    dq.Ry    (nqubit=self.num_qubits, requires_grad=True),
                    dq.U3Gate(nqubit=self.num_qubits, requires_grad=True),
                    dq.U3Gate(nqubit=self.num_qubits, requires_grad=True),
                ]

            def add_U(i:int, j:int, gates:List[Gate]):  # conv
                vqc.add(gates[0], wires=i) ; vqc.add(gates[1], wires=j)
                vqc.cnot(j, i) ; vqc.add(gates[2], wires=i) ; vqc.add(gates[3], wires=j)
                vqc.cnot(i, j) ;                              vqc.add(gates[4], wires=j)
                vqc.cnot(j, i)
                vqc.add(gates[5], wires=i) ; vqc.add(gates[6], wires=j)

            def mk_V_gates() -> List[Gate]:
                g = dq.U3Gate(nqubit=self.num_qubits, requires_grad=True)
                return [
                    dq.U3Gate(nqubit=self.num_qubits, requires_grad=True),
                    g, 
                    g.inverse(),
                ]

            def add_V(i:int, j:int, gates:List[Gate]):  # pool
                vqc.add(gates[0], wires=i)
                vqc.add(gates[1], wires=j)
                vqc.cnot(i, j)
                vqc.add(gates[2], wires=j)

            def add_F(wires:List[int]): # fc, 沿用 CCQC (arXiv:1804.00633)
                wire_p1 = wires[1:] + wires[:1]
                # stride=1
                for i in wires: vqc.u3(i)
                for i, j in zip(wires, wire_p1):
                    vqc.cnot(i, j)
                    vqc.cnot(j, i)
                # stride=1
                for i in wires: vqc.u3(i)
                for i, j in zip(wire_p1, wires):
                    vqc.cnot(i, j)
                    vqc.cnot(j, i)

            for _ in range(self.num_layers):
                # layer1
                gs_U = mk_U_gates()
                add_U(1, 2, gs_U) ; add_U(3, 4, gs_U) ; add_U(5, 6, gs_U) ; add_U(7, 8, gs_U) ; add_U(9, 10, gs_U) ; add_U(11,  0, gs_U)
                add_U(0, 1, gs_U) ; add_U(2, 3, gs_U) ; add_U(4, 5, gs_U) ; add_U(6, 7, gs_U) ; add_U(8,  9, gs_U) ; add_U(10, 11, gs_U)
                gs_V = mk_V_gates()
                add_V(0, 1, gs_V) ; add_V(2, 3, gs_V) ; add_V(4, 5, gs_V) ; add_V(6, 7, gs_V) ; add_V(8,  9, gs_V) ; add_V(10, 11, gs_V)
                # layer2
                gs_U = mk_U_gates()
                add_U(3, 5, gs_U) ; add_U(7, 9, gs_U) ; add_U(11, 1, gs_U)
                add_U(1, 3, gs_U) ; add_U(5, 7, gs_U) ; add_U(9, 11, gs_U)
                gs_V = mk_V_gates()
                add_V(1, 3, gs_V) ; add_V(5, 7, gs_V) ; add_V(9, 11, gs_V)
                # layer3
                gs_U = mk_U_gates()
                add_U(3,  7, gs_U) ; add_U(7, 11, gs_U) 
                gs_V = mk_V_gates()
                add_V(3,  7, gs_V) ; add_V(7, 11, gs_V)
            # fc
            add_F([7, 11])
            # meas
            vqc.observable(7,  basis='z')
            vqc.observable(7,  basis='x')
            vqc.observable(7,  basis='y')
            vqc.observable(11, basis='z')
            vqc.observable(11, basis='x')

        # n_layer=16, gcnt=3096, pcnt=696;  best overfit acc=38.8%
        # n_layer=24, gcnt=4632, pcnt=1032; best overfit acc=42.6%
        if not 'real qcnn (2 blocks)':     # parameter shared!
            def mk_U_gates() -> List[Gate]:
                return [
                    dq.U3Gate(nqubit=self.num_qubits, requires_grad=True),
                    dq.U3Gate(nqubit=self.num_qubits, requires_grad=True),
                    dq.Rz    (nqubit=self.num_qubits, requires_grad=True),
                    dq.Ry    (nqubit=self.num_qubits, requires_grad=True),
                    dq.Ry    (nqubit=self.num_qubits, requires_grad=True),
                    dq.U3Gate(nqubit=self.num_qubits, requires_grad=True),
                    dq.U3Gate(nqubit=self.num_qubits, requires_grad=True),
                ]

            def add_U(i:int, j:int, gates:List[Gate]):  # conv
                vqc.add(gates[0], wires=i) ; vqc.add(gates[1], wires=j)
                vqc.cnot(j, i) ; vqc.add(gates[2], wires=i) ; vqc.add(gates[3], wires=j)
                vqc.cnot(i, j) ;                              vqc.add(gates[4], wires=j)
                vqc.cnot(j, i)
                vqc.add(gates[5], wires=i) ; vqc.add(gates[6], wires=j)

            def mk_V_gates() -> List[Gate]:
                g = dq.U3Gate(nqubit=self.num_qubits, requires_grad=True)
                return [
                    dq.U3Gate(nqubit=self.num_qubits, requires_grad=True),
                    g, 
                    g.inverse(),
                ]

            def add_V(i:int, j:int, gates:List[Gate]):  # pool
                vqc.add(gates[0], wires=i)
                vqc.add(gates[1], wires=j)
                vqc.cnot(i, j)
                vqc.add(gates[2], wires=j)

            def add_F(wires:List[int]): # fc, 沿用 CCQC (arXiv:1804.00633)
                wire_p1 = wires[1:] + wires[:1]
                wire_p2 = wires[2:] + wires[:2]
                # stride=1
                for i in wires: vqc.u3(i)
                for i, j in zip(wires, wire_p1):
                    vqc.cnot(i, j)
                    vqc.cnot(j, i)
                # stride=2
                for i in wires: vqc.u3(i)
                for i, j in zip(wires, wire_p2):
                    vqc.cnot(i, j)
                    vqc.cnot(j, i)

            for _ in range(self.num_layers):
                # layer1
                gs_U = mk_U_gates()
                add_U(1, 2, gs_U) ; add_U(3, 4, gs_U) ; add_U(5, 6, gs_U) ; add_U(7, 8, gs_U) ; add_U(9, 10, gs_U)
                add_U(0, 1, gs_U) ; add_U(2, 3, gs_U) ; add_U(4, 5, gs_U) ; add_U(6, 7, gs_U) ; add_U(8,  9, gs_U) ; add_U(10, 11, gs_U)
                gs_V = mk_V_gates()
                add_V(0, 1, gs_V) ; add_V(2, 3, gs_V) ; add_V(4, 5, gs_V) ; add_V(6, 7, gs_V) ; add_V(8,  9, gs_V) ; add_V(10, 11, gs_V)
                # layer2
                gs_U = mk_U_gates()
                add_U(1, 3, gs_U) ; add_U(5, 7, gs_U) ; add_U(9, 11, gs_U)
                add_U(3, 5, gs_U) ; add_U(7, 9, gs_U)
                gs_V = mk_V_gates()
                add_V(3, 5, gs_V) ; add_V(7, 9, gs_V)
            # fc
            add_F([3, 5, 7, 9])
            # meas
            vqc.observable(3,  basis='z')
            vqc.observable(5,  basis='z')
            vqc.observable(7,  basis='z')
            vqc.observable(9,  basis='z')
            vqc.observable(3,  basis='x')

        # [std_flatten]
        # n_layer=10, gcnt=1452, pcnt=1452; best overfit acc=34.0%
        # [qam_flatten(?)]
        # n_layer=10, gcnt=1452, pcnt=1452; acc=39.486%/45.867% (test: 42%)
        if 'F2_all_0':
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
            # meas
            vqc.observable(0, basis='z')
            vqc.observable(1, basis='z')
            vqc.observable(2, basis='z')
            vqc.observable(3, basis='z')
            vqc.observable(4, basis='z')

        print('classifier gate count:', count_gates(vqc))

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
