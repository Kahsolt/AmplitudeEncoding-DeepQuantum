import torch.nn as nn
from torch.utils.data import DataLoader
import deepquantum as dq

from utils import reshape_norm_padding, count_gates, cir_collate_fn, FakeDataset, FakeDatasetApprox, get_fidelity, QMNISTDataset


# 注意: 选手的模型名字必须固定为 QuantumNeuralNetwork
# todo: 构建表达能力更强的变分量子线路以提高分类准确率
class QuantumNeuralNetwork(nn.Module):

    def __init__(self, n_qubit, n_layer):
        super().__init__()
        self.n_qubit = n_qubit
        self.n_layer = n_layer
        self.loss_fn =  nn.CrossEntropyLoss()
        self.var_circuit = dq.QubitCircuit(n_qubit)
        self.create_var_circuit()

    def create_var_circuit(self):
        """构建变分量子线路"""
        for i in range(self.n_layer):
            self.var_circuit.rzlayer()
            self.var_circuit.rylayer()
            self.var_circuit.rzlayer()
            self.var_circuit.cnot_ring()
            self.var_circuit.barrier()
        # num of observable == num of classes
        self.var_circuit.observable(wires=0, basis='z') 
        self.var_circuit.observable(wires=1, basis='z') 
        self.var_circuit.observable(wires=0, basis='x') 
        self.var_circuit.observable(wires=1, basis='x')   
        self.var_circuit.observable(wires=0, basis='y') 

    def forward(self, z, y):
        # 将输入态经过设定的线路进行演化
        self.var_circuit(state=z)   
        # 测量期望值                 
        output = self.var_circuit.expectation()          
        return self.loss_fn(output, y), output
    
    def inference(self, z):
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
    #dataset = FakeDataset(size=10, noise_strength=0.0)
    #dataset = FakeDatasetApprox(size=20, noise_strength=0.0)
    dataset = QMNISTDataset(label_list=[0,1], train=True, size=20)
    
    print('debug', dataset[0])
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=cir_collate_fn)
    x, y, z = next(iter(data_loader))
    x, y, z = x.to('cuda:0'), y.to('cuda:0'), z.to('cuda:0')
   
    # 创建一个量子神经网络模型
    model = QuantumNeuralNetwork(n_qubit=10, n_layer=30).to('cuda:0')
    loss, output = model(z, y)
    output = model.inference(z)
    print('fid', get_fidelity(z, reshape_norm_padding(x)))
