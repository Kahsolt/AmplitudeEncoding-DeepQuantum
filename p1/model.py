from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import deepquantum as dq
import numpy as np

from utils import QMNISTDataset, cir_collate_fn, get_fidelity, reshape_norm_padding, count_gates, reshape_norm_padding


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

        # gcnt=445, pcnt=1335; (88.333%/92.667%)
        if not 'mera-updown-cu':
            for i in range(self.n_layer):
                self.var_circuit.u3layer()
                offset = int(i % 2 == 1)
                for q in range(offset, self.n_qubit - offset, 2):
                    if offset: self.var_circuit.cu(q + 1, q)
                    else:      self.var_circuit.cu(q, q + 1)
            self.var_circuit.u3layer()

        # n_layer=8,  gcnt=404, pcnt=1212
        # n_layer=10, gcnt=500, pcnt=1500
        if not 'real mera-like, [enc-dec]-u3':
            for i in range(self.n_layer):
                # down (->10-8-6-4-2)
                for offset in range(self.n_qubit // 2):
                    for q in range(offset, self.n_qubit - 1 - offset, 2):
                        self.var_circuit.cu(q, q + 1)
                        self.var_circuit.cu(q + 1, q)
                # up (->4-6-8)
                for offset in range(self.n_qubit // 2 - 2, 0, -1):
                    for q in range(offset, self.n_qubit - 1 - offset, 2):
                        self.var_circuit.cu(q, q + 1)
                        self.var_circuit.cu(q + 1, q)
            if 'last up (->10)':
                for q in range(0, self.n_qubit - 1, 2):
                    self.var_circuit.cu(q, q + 1)
                    self.var_circuit.cu(q + 1, q)
            self.var_circuit.u3layer()

        # n_layer=10, gcnt=255, pcnt=765  (70.286%/64.667%)
        # n_layer=20, gcnt=495, pcnt=1485 (76.000%/67.333%)
        if not 'real mera-like, [enc↓-dec↑]-u3':
            for i in range(self.n_layer):
                # down (->10-8-6-4-2)
                for offset in range(self.n_qubit // 2):
                    for q in range(offset, self.n_qubit - 1 - offset, 2):
                        self.var_circuit.cu(q, q + 1)
                # up (->4-6-8)
                for offset in range(self.n_qubit // 2 - 2, 0, -1):
                    for q in range(offset, self.n_qubit - 1 - offset, 2):
                        self.var_circuit.cu(q + 1, q)
            if 'last up (->10)':
                for q in range(0, self.n_qubit - 1, 2):
                    self.var_circuit.cu(q + 1, q)
            self.var_circuit.u3layer()

        # n_layer=10, gcnt=241, pcnt=723 (92.286%/91.200%)  # NOTE: nice!!
        if not 'real mera-like, [enc↓-dec↑]-enc↓-u3':
            for i in range(self.n_layer):
                # down (->10-8-6-4-2)
                for offset in range(self.n_qubit // 2):
                    for q in range(offset, self.n_qubit - 1 - offset, 2):
                        self.var_circuit.cu(q, q + 1)
                if i < self.n_layer-1:
                    # up (->4-6-8)
                    for offset in range(self.n_qubit // 2 - 2, 0, -1):
                        for q in range(offset, self.n_qubit - 1 - offset, 2):
                            self.var_circuit.cu(q + 1, q)
            self.var_circuit.u3layer()

        # n_layer=8, gcnt=273, pcnt=819 (92.914%/91.933%)   # NOTE: nice!!
        if not 'real mera-like, [u3-enc↓-dec↑]-enc↓-u3':
            for i in range(self.n_layer):
                self.var_circuit.u3layer()
                # down (->10-8-6-4-2)
                for offset in range(self.n_qubit // 2):
                    for q in range(offset, self.n_qubit - 1 - offset, 2):
                        self.var_circuit.cu(q, q + 1)
                if i < self.n_layer-1:
                    # up (->4-6-8)
                    for offset in range(self.n_qubit // 2 - 2, 0, -1):
                        for q in range(offset, self.n_qubit - 1 - offset, 2):
                            self.var_circuit.cu(q + 1, q)
            self.var_circuit.u3layer()

        # n_layer=6, gcnt=352, pcnt=1056 (92.600%/92.600%)  # NOTE: very nice!!
        if 'real mera-like, [u3-enc-u3-dec]-enc-u3':
            for i in range(self.n_layer):
                self.var_circuit.u3layer()
                # down (->10-8-6-4-2)
                for offset in range(self.n_qubit // 2):
                    for q in range(offset, self.n_qubit - 1 - offset, 2):
                        self.var_circuit.cu(q, q + 1)
                        self.var_circuit.cu(q + 1, q)
                if i < self.n_layer-1:
                    self.var_circuit.u3layer(wires=[4, 5])
                    # up (->2-4-6-8)
                    for offset in range(self.n_qubit // 2 - 1, 0, -1):
                        for q in range(offset, self.n_qubit - 1 - offset, 2):
                            self.var_circuit.cu(q + 1, q)
                            self.var_circuit.cu(q, q + 1)
            self.var_circuit.u3layer(wires=[4, 5])

        # n_layer=10, gcnt=160, pcnt=480 (88.057%/88.733%)
        # n_layer=15, gcnt=235, pcnt=705 (89.029%/89.067%)
        if not 'real mera-like, down↓ only':
            for i in range(self.n_layer):
                # down (->10-8-6-4-2)
                for offset in range(self.n_qubit // 2):
                    for q in range(offset, self.n_qubit - 1 - offset, 2):
                        self.var_circuit.cu(q, q + 1)
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

        if not 'swap-like':         # pcnt=4080 =_=||
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

        if not 'swap-like':          # (snake) 89.400%/90.000%; n_gate=540, n_param=541
            self.var_circuit.x(0)
            for i in range(self.n_layer):
                for q in range(self.n_qubit-1):
                    self.var_circuit.cry(q, (q+1)%self.n_qubit)
                    self.var_circuit.cry((q+1)%self.n_qubit, q)

        # num of observable == num of classes
        if not 'original q0,q1':
            self.var_circuit.observable(wires=0, basis='z')
            self.var_circuit.observable(wires=1, basis='z')
            self.var_circuit.observable(wires=0, basis='x')
            self.var_circuit.observable(wires=1, basis='x')
            self.var_circuit.observable(wires=0, basis='y')

        if 'mera q4,q5':
            self.var_circuit.observable(wires=4, basis='z')
            self.var_circuit.observable(wires=5, basis='z')
            self.var_circuit.observable(wires=4, basis='x')
            self.var_circuit.observable(wires=5, basis='x')
            self.var_circuit.observable(wires=4, basis='y')

        print('classifier gate count:', count_gates(self.var_circuit))

    def forward(self, z:Tensor, y:Tensor) -> Tuple[Tensor, Tensor]:
        self.var_circuit(state=z)   
        output = self.var_circuit.expectation()          
        return self.loss_fn(output, y), output

    @torch.inference_mode()
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


class HadamardTest(nn.Module):

    ''' This is the non-parametrical quantum kNN :( '''

    def __init__(self, n_qubit:int, n_layer:int):
        super().__init__()
        self.nq = n_qubit
        self.n_qubit = 2 * n_qubit + 1
        self.anc = self.n_qubit - 1     # idx of ancilla
        print('>> self.anc:', self.anc)
        self.canon = torch.from_numpy(np.load('./img/canon.npy')).unsqueeze_(dim=1)    # [NC=5, C=1, H, W], already normalized
        self.n_cls = len(self.canon)

    def load_state_dict(*args, **kwargs):
        pass

    @torch.inference_mode()
    def inference(self, z:Tensor) -> Tensor:
        n_samples = z.shape[0]
        output = torch.zeros([n_samples, self.n_cls], dtype=torch.float32)
        for i in range(n_samples):     # sample
            val = z[i].flatten()
            for c in range(self.n_cls): # class
                ref = reshape_norm_padding(self.canon[c].unsqueeze(0)).squeeze(0).to(val.device)
                # |psi,phi,anc>
                init = torch.kron(torch.kron(val, ref), torch.tensor([1, 0], dtype=torch.cfloat, device=val.device))

                # Hadamard test circuit
                qc = dq.QubitCircuit(self.n_qubit)
                qc(state=init)      # init data encoding
                qc.h(self.anc)      # ancilla qubit
                for i in range(self.nq):
                    qc.cswap(self.anc, i, i + self.nq)
                qc.h(self.anc)
                qc.observable(wires=self.anc, basis='z')

                exp = qc.expectation().item()
                print('>> exp:', exp)
                output[i][c] = exp
        breakpoint()
        return output

# NOTE: force naming replace!
QuantumNeuralNetwork = HadamardTest


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
