import os
import random
from collections import Counter
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import numpy as np
import deepquantum as dq

from utils import QCIFAR10Dataset, PerfectAmplitudeEncodingDataset, reshape_norm_padding, get_fidelity, get_fidelity_NtoN, count_gates

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
class QuantumNeuralNetworkAnsatz(nn.Module):

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

        # n_layer=8,  gcnt=1772, pcnt=2412; best overfit acc=42.8%
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

        # n_layer=24, gcnt=5292, pcnt=1524; best overfit acc=42.8%
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

        # n_layer=10, gcnt=1452, pcnt=1452; best overfit acc=34.0%
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
            # meas
            vqc.observable(0, basis='z')
            vqc.observable(1, basis='z')
            vqc.observable(2, basis='z')
            vqc.observable(3, basis='z')
            vqc.observable(4, basis='z')

        # n_layer=6,  gcnt=936,  pcnt=1260; best overfit acc=39.8%
        # n_layer=8,  gcnt=1224, pcnt=1656; best overfit acc=44.4% (meas xyz-01)
        #                                   best overfit acc=41.0% (meas xyz-67)
        #                                   best overfit acc=41.8% (meas z-all)
        # n_layer=12, gcnt=1800, pcnt=2448; best overfit acc=41.6%
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
            # meas
            sel = 'xyz-01'
            if sel == 'xyz-01':
                vqc.observable(0, basis='z')
                vqc.observable(0, basis='x')
                vqc.observable(0, basis='y')
                vqc.observable(1, basis='z')
                vqc.observable(1, basis='x')
            elif sel == 'xyz-67':
                vqc.observable(6, basis='z')
                vqc.observable(6, basis='x')
                vqc.observable(6, basis='y')
                vqc.observable(7, basis='z')
                vqc.observable(7, basis='x')
            elif sel == 'z-all':
                vqc.observable(0, basis='z')
                vqc.observable(1, basis='z')
                vqc.observable(2, basis='z')
                vqc.observable(3, basis='z')
                vqc.observable(4, basis='z')

        # n_layer=8,  gcnt=1034, pcnt=1518; best overfit acc=35.8%
        if not 'U cyclic':
            def add_U(i:int, j:int):  # conv
                vqc.u3(i) ; vqc.u3(j)
                vqc.cnot(j, i) ; vqc.rz(i) ; vqc.ry(j)
                vqc.cnot(i, j) ;             vqc.ry(j)
                vqc.cnot(j, i)
                vqc.u3(i) ; vqc.u3(j)

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
                for i in range(self.num_qubits):
                    add_U(i, (i + 1) % self.num_qubits)
            # fc
            add_F(list(range(12)))
            vqc.u3(wires=0)
            vqc.u3(wires=1)
            # meas
            vqc.observable(0, basis='z')
            vqc.observable(0, basis='x')
            vqc.observable(0, basis='y')
            vqc.observable(1, basis='z')
            vqc.observable(1, basis='x')

        # n_layer=8,  gcnt=5282, pcnt=7926; best overfit acc=35.8%
        if not 'U all':
            def add_U(i:int, j:int):  # conv
                vqc.u3(i) ; vqc.u3(j)
                vqc.cnot(j, i) ; vqc.rz(i) ; vqc.ry(j)
                vqc.cnot(i, j) ;             vqc.ry(j)
                vqc.cnot(j, i)
                vqc.u3(i) ; vqc.u3(j)

            for _ in range(self.num_layers):
                for i in range(self.num_qubits-1):
                    for j in range(i + 1, self.num_qubits):
                        add_U(i, j)
            # fc
            vqc.u3(wires=0)
            vqc.u3(wires=1)
            # meas
            vqc.observable(0, basis='z')
            vqc.observable(0, basis='x')
            vqc.observable(0, basis='y')
            vqc.observable(1, basis='z')
            vqc.observable(1, basis='x')

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


# 实验：能用 ansatz-encoder-ansatz 结构充当 QMLP 吗？
# 结论是不行，哎。。。
class QuantumNeuralNetworkAnsatzMLP(nn.Module):

    def __init__(self, num_qubits, num_layers):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.loss_fn = F.cross_entropy
        self.var_circuit  = self.create_var_circuit1()
        self.var_circuit2 = self.create_var_circuit2()

    def get_ansatz(self, vqc:dq.QubitCircuit=None) -> dq.QubitCircuit:
        vqc = vqc or dq.QubitCircuit(self.num_qubits)

        # n_layer=6,  gcnt=936,  pcnt=1260; best overfit acc=39.8%
        # n_layer=8,  gcnt=1224, pcnt=1656; best overfit acc=44.4% (meas xyz-01)
        #                                   best overfit acc=41.0% (meas xyz-67)
        #                                   best overfit acc=41.8% (meas z-all)
        # n_layer=12, gcnt=1800, pcnt=2448; best overfit acc=41.6%
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

        return vqc

    def get_encoder(self, vqc:dq.QubitCircuit=None) -> dq.QubitCircuit:
        vqc = vqc or dq.QubitCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            vqc.rz(i, encode=True)      # accept Z-axis measure
            vqc.ry(i, encode=True)      # accept X-axis measure
        return vqc

    def create_var_circuit1(self):
        # ansatz
        vqc = self.get_ansatz()
        # measure (hidden layer)
        for i in range(self.num_qubits):
            vqc.observable(i, basis='z')
            vqc.observable(i, basis='x')
        print('clf-ansatz1 gate count:', count_gates(vqc))
        return vqc

    def create_var_circuit2(self):
        # encoder (reupload!)
        vqc = self.get_encoder()
        # ansatz
        vqc = self.get_ansatz(vqc)
        # measure (output layer)
        vqc.observable(0, basis='z')
        vqc.observable(0, basis='x')
        vqc.observable(1, basis='z')
        vqc.observable(1, basis='x')
        vqc.observable(2, basis='z')
        print('clf-ansatz2 gate count:', count_gates(vqc))
        return vqc

    def forward(self, z:Tensor, y:Tensor):
        assert z.shape[0] == 1, 'QubitCircuit.encode() only support bs=1'
        self.var_circuit(state=z)
        hidden = self.var_circuit.expectation().flatten()       # [M=24]
        self.var_circuit2.encode(hidden)
        self.var_circuit2()
        output = self.var_circuit2.expectation().unsqueeze(0)   # [B=1, NC=5]
        return self.loss_fn(output, y), output

    @torch.inference_mode
    def inference(self, z:Tensor):
        assert z.shape[0] == 1, 'QubitCircuit.encode() only support bs=1'
        self.var_circuit(state=z)
        hidden = self.var_circuit.expectation().flatten()       # [M=24]
        self.var_circuit2.encode(hidden)
        self.var_circuit2()
        output = self.var_circuit2.expectation().unsqueeze(0)   # [B=1, NC=5]
        return output


# 对比学习!!
class QuantumNeuralNetworkCL(nn.Module):

    def __init__(self, num_qubits, num_layers):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.var_circuit = dq.QubitCircuit(num_qubits)
        self.create_var_circuit()

        # [NC=5, M=36], remember to backfill refdata after training finished
        self.ref_qstate: nn.Parameter = nn.Parameter(torch.zeros([5, 36], requires_grad=False))
        self.is_training = False    # when True, do not use `self.ref_qstate`

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

    @torch.no_grad
    def mk_ref_qstate(self, ref_data:Dataset, device:str):
        # 类中心的测量结果视作参考
        y_x = {}
        for _, y, x in ref_data:
            x = x.real.flatten()
            y = y.item()
            if y not in y_x: y_x[y] = []
            y_x[y].append(x)
        y_x = sorted([(y, np.stack(xs, axis=-1).mean(axis=-1)) for y, xs in y_x.items()])
        z = torch.from_numpy(np.stack([x for y, x in y_x], axis=0)).to(device=device, dtype=torch.complex64)
        self.var_circuit(state=z)
        outputs = self.var_circuit.expectation()
        fake_qstate = F.normalize(outputs, dim=-1)
        fake_qstate.requires_grad = False
        self.ref_qstate = nn.Parameter(fake_qstate)

    def postprocess(self, outputs:Tensor):
        fake_qstate = F.normalize(outputs, dim=-1)
        ref_states = fake_qstate if self.is_training else self.ref_qstate
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


# 量子-经典混合模型!!
class QuantumNeuralNetworkCLMLP(nn.Module):

    def __init__(self, num_qubits, num_layers):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.var_circuit = dq.QubitCircuit(num_qubits)
        self.create_var_circuit()
        self.mlp = nn.Sequential(
            nn.Linear(36, 12),
            nn.ReLU(),
            nn.Linear(12, 5),
        )

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

    def forward(self, z:Tensor, y:Tensor):
        self.var_circuit(state=z)
        outputs = self.var_circuit.expectation()     # [B, M=36]
        outputs = F.normalize(outputs, dim=-1)
        logits = self.mlp(outputs)
        loss = F.cross_entropy(logits, y)
        return loss, logits

    @torch.inference_mode()
    def inference(self, z):
        self.var_circuit(state=z)
        outputs = self.var_circuit.expectation()    # [B, M=36]
        outputs = F.normalize(outputs, dim=-1)
        logits = self.mlp(outputs)
        return logits                               # [B, NC=5]


QuantumNeuralNetwork = QuantumNeuralNetworkCL


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
