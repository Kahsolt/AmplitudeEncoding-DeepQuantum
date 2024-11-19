import os
import random
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import deepquantum as dq

from utils import reshape_norm_padding, get_fidelity, PerfectAmplitudeEncodingDataset, QCIFAR10Dataset
from utils import count_gates, get_fidelity_NxN


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

        self.model2_grid = nn.ModuleDict()
        for i in range(4):
            for j in range(i+1, 5):
                self.model2_grid[f'bin_{i}-{j}'] = QuantumNeuralNetworkCL(num_qubits, num_layers)

    @torch.inference_mode
    def inference(self, z:Tensor):
        votes2_list: List[Tensor] = []
        for i in range(4):
            for j in range(i+1, 5):
                model: QuantumNeuralNetworkCL = self.model2_grid[f'bin_{i}-{j}']
                raw_preds: Tensor = model.inference(z).argmax(-1).cpu()     # [B]
                key = (i, j)
                # https://discuss.pytorch.org/t/mapping-values-in-a-tensor/117731
                preds = raw_preds.apply_(lambda p: key[p])
                votes2_list.append(preds)

        # https://stackoverflow.com/questions/16330831/most-efficient-way-to-find-mode-in-numpy-array
        votes = np.stack(votes2_list, axis=-1)   # [B, V=10]
        preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=votes)
        preds = torch.from_numpy(preds).to(z.device)

        # 输出必须是 [B, NC=5]
        return F.one_hot(preds, num_classes=5).to(torch.float32)


class QuantumNeuralNetworkCL(nn.Module):

    def __init__(self, num_qubits, num_layers, n_class:int=2):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.var_circuit = dq.QubitCircuit(num_qubits)
        self.create_var_circuit()

        # 注: 这不是可学习的经典参数，是训练完成后缓存的参考值 :(
        # [NC=2, M=36], remember to backfill refdata after training finished
        self.ref_qstate = nn.Parameter(torch.zeros([n_class, 36], requires_grad=False))
        self.is_training = False    # when True, do not use `self.ref_qstate`

    # gcnt=, pcnt=
    def create_var_circuit(self):
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

        # U-V brick
        for _ in range(self.num_layers):
            add_U(1, 2) ; add_U(3, 4) ; add_U(5, 6) ; add_U(7, 8) ; add_U(9, 10) ; add_U(11, 0)
            add_U(0, 1) ; add_U(2, 3) ; add_U(4, 5) ; add_U(6, 7) ; add_U(8, 9)  ; add_U(10, 11)
            add_V(0, 1) ; add_V(2, 3) ; add_V(4, 5) ; add_V(6, 7) ; add_V(8, 9)  ; add_V(10, 11)
        # fc
        add_F(list(range(12)))
        # meas
        for i in range(self.num_qubits):
            vqc.observable(i, 'x')
            vqc.observable(i, 'y')
            vqc.observable(i, 'z')

        #print('classifier gate count:', count_gates(vqc))

    @torch.no_grad
    def mk_ref_qstate(self, ref_data:Dataset, device:str):
        # 测量结果的类中心视作参考
        ref_loader = DataLoader(ref_data, batch_size=512, shuffle=False, drop_last=False, pin_memory=False)
        y_list, v_list = [], []
        for _, y, x in ref_loader:
            x = x.to(device)
            self.var_circuit(state=x)
            E = self.var_circuit.expectation()  # [B, M=36]
            v = F.normalize(E, dim=-1)
            v_list.append(v)
            y_list.extend(y.numpy().tolist())
        v_list = torch.cat(v_list, dim=0)   # [N=25000, M=36]
        # 分组-聚合
        y_v = {}
        for y, v in zip(y_list, v_list):
            if y not in y_v: y_v[y] = []
            y_v[y].append(v)
        y_v = sorted([(y, F.normalize(torch.stack(vs, dim=0).mean(dim=0), dim=-1)) for y, vs in y_v.items()])
        fake_qstate = torch.stack([v for _, v in y_v], dim=0)   # [NC=2, M=36]
        fake_qstate.requires_grad = False
        self.ref_qstate.data = fake_qstate

    def postprocess(self, outputs:Tensor):
        fake_qstate = F.normalize(outputs, dim=-1)
        ref_states = fake_qstate if self.is_training else self.ref_qstate
        fid_mat = get_fidelity_NxN(ref_states, fake_qstate)    # [B, NC=2]
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

    @torch.inference_mode
    def inference(self, z):
        self.var_circuit(state=z)
        outputs = self.var_circuit.expectation()    # [B, M=36]
        fid_mat = self.postprocess(outputs)         # [B, NC=5]
        return fid_mat


if __name__ == '__main__':
    # dataset = QCIFAR10Dataset(train=True, size=20) 
    # dataset = QCIFAR10Dataset(train=False, size=20) 
    dataset = PerfectAmplitudeEncodingDataset(train=True)
    print('dataset labels', [sample[1].item() for sample in dataset])
    print('gates_count', dataset.get_gates_count())

    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    x, y, z = next(iter(data_loader))
    x, y, z = x.to('cuda:0'), y.to('cuda:0'), z.to('cuda:0')
    print('x', x.shape)
    print('y', y.shape)
    print('z', z.shape)
    print('fid', get_fidelity(z, reshape_norm_padding(x)))
    
    # 创建一个量子神经网络模型
    model = QuantumNeuralNetwork(num_qubits=12, num_layers=3).to('cuda:0')
    loss, output = model(z, y)
    output = model.inference(z)
    print('output', output)
