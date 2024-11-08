import os
import sys
import random
import pickle
from copy import deepcopy
from time import time
import multiprocessing
from collections import Counter
from typing import Tuple, Generator

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import deepquantum as dq
import numpy as np
from numpy import ndarray
from tqdm import tqdm
import matplotlib.pyplot as plt

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

if os.getenv('MY_LABORATORY') or sys.platform == 'win32':
    DATA_PATH = '../data'
else:
    DATA_PATH = '/data'


def count_gates(cir:dq.QubitCircuit):
    count = 0
    for m in cir.operators.modules():
        if isinstance(m, dq.operation.Gate) and (not isinstance(m, dq.gate.Barrier)):
            count += 1
    return count

def get_fidelity(state_pred, state_true):
    # state_pred, state_true: (batch_size, 4096, 1)
    state_pred = state_pred.view(-1, 4096).real
    state_true = state_true.view(-1, 4096).real
    fidelity = (state_pred * state_true).sum(-1)**2
    return fidelity.mean()

@torch.inference_mode
def get_acc(y_pred, y_true):
    # 计算准确率
    correct = (y_pred == y_true).float()
    accuracy = correct.sum() / len(correct)
    return accuracy.item()


def add_suffix(x:ndarray, suffix:str) -> ndarray:
  x = deepcopy(x)
  W, H = x.shape
  for i in range(W):
    for j in range(H):
      x[i, j] = x[i, j] + suffix
  return x

def get_qam_array(n:int) -> ndarray:
  if n == 2:
    ''' 大端序读取顺序：
      10 ← 00
         ↘
      11 ← 01
    '''
    return np.asarray([
      ['10', '00'],
      ['11', '01'],
    ], dtype=object)
  else:
    x = get_qam_array(n - 2)
    W, H = x.shape
    x_ex = np.empty([2*W, 2*H], dtype=x.dtype)
    x_ex[:W, :H] = add_suffix(x[:, ::-1],    '10')
    x_ex[:W, H:] = add_suffix(x,             '00')
    x_ex[W:, :H] = add_suffix(x[::-1, ::-1], '11')
    x_ex[W:, H:] = add_suffix(x[::-1, :],    '01')
    return x_ex

bits_to_coord_cache = None
bits_to_coord_flip_cache = None
def qam_index_generator(nbps:int, flip:bool=False) -> Generator[Tuple[int, int], int, None]:
  global bits_to_coord_cache, bits_to_coord_flip_cache
  if bits_to_coord_cache is None:
    def bstr_as_le_int(s:str) -> int:
      return int(s, base=2)
    def bstr_as_be_int(s:str) -> int:
      return int(s[::-1], base=2)

    array = get_qam_array(nbps)
    H, W = array.shape
    le_bstr_to_loc = []
    be_bstr_to_loc = []
    for x in range(H):
      for y in range(W):
        bstr = array[x, y]
        le_bstr_to_loc.append((bstr_as_le_int(bstr), (x, y)))
        be_bstr_to_loc.append((bstr_as_be_int(bstr), (x, y)))
    le_bstr_to_loc.sort()
    be_bstr_to_loc.sort()
    bits_to_coord_cache      = [xy for _, xy in be_bstr_to_loc]   # deepquantum is big endian
    bits_to_coord_flip_cache = [xy for _, xy in le_bstr_to_loc]   # if we flip it, then it's little endian

  for xy in (bits_to_coord_flip_cache if flip else bits_to_coord_cache):
    yield xy

def qam_reshape_norm_padding(x:Tensor, nbps:int=12, hwc_order:bool=False) -> Tensor:
  has_batch = True
  if x.dim() == 3:
    x = x.unsqueeze(0)
    has_batch = False
  assert len(x.shape) == 4        # [B, C, H, W]
  ''' NOTE: must arrange in PLANNAR fmt instead of PACKED!
  三通道RGB拼贴为单通道大图，布局为：(原点在左下角，图在第一象限)
    | R | O |  =>  ['10','00']
    | B | G |  <=  ['11','01']
  以满足后缀索引 (大端序)：
    |...00> - 全0
    |...10> - R
    |...01> - G
    |...11> - B
  '''
  null_channel = torch.zeros_like(x[:, 0, ...])   # [B, H=32, W=32]
  ch_order = 'RGB0'           # 通道序列化顺序
  if ch_order == '0RGB':      # 开头补0
    x_ex = torch.cat([        # [B, H_ex=64, W_ex=64]
      torch.cat([x[:, 0, ...], null_channel], dim=-1),
      torch.cat([x[:, 2, ...], x[:, 1, ...]], dim=-1),
    ], dim=1)
  elif ch_order == 'RGB0':    # 末尾补0 (better!)
    x_ex = torch.cat([
      torch.cat([x[:, 1, ...], x[:, 0, ...]], dim=-1),
      torch.cat([null_channel, x[:, 2, ...]], dim=-1),
    ], dim=1)
  assert len(x_ex.shape) == 3     # [B, H_ex, W_ex]
  pixels = []
  for i, j in qam_index_generator(nbps, flip=hwc_order):
    pixels.append(x_ex[:, i, j])
  x = torch.stack(pixels, -1)     # [B, H_ex*W_ex]
  x = F.normalize(x, p=2, dim=-1)
  #x = F.pad(x, (0, 2**nbps - x.size(-1)), mode='constant', value=0.0)
  if not has_batch:
    x = x.squeeze(0)
  return x.to(torch.complex64)  # [B, D=256]

def reshape_norm_padding(x:Tensor, use_hijack:bool=False) -> Tensor:
    # NOTE: 因为test脚本不能修改，所以需要在云评测时直接替换具体实现
    if use_hijack:
        return qam_reshape_norm_padding(x)

    # x: CIFAR10 image with shape (3, 32, 32) or (batch_size, 3, 32, 32)
    if x.dim() == 3:
        assert x.shape == (3, 32, 32), f"Expected input shape (3, 32, 32), got {x.shape}"
    elif x.dim() == 4:
        assert x.shape[1:] == (3, 32, 32), f"Expected input shape (batch_size, 3, 32, 32), got {x.shape}"
    else:
        raise ValueError(f"Unexpected input shape: {x.shape}")

    PADDING_SIZE = 4096 - 3 * 32 * 32
    original_shape = x.shape
    x = x.reshape(original_shape[0] if x.dim() == 4 else 1, -1)
    x = F.normalize(x, p=2, dim=1)
    x = F.pad(x, (0, PADDING_SIZE), mode='constant', value=0)
    x = x.to(torch.complex64)
    return x.unsqueeze(-1)  # (batch_size, 4096, 1)


# 将数据的平均值调整为接近0 大部分像素值会落在[-1, 1]范围内
# https://blog.csdn.net/weixin_44579633/article/details/123128976
# https://github.com/kuangliu/pytorch-cifar/issues/8
# https://github.com/kuangliu/pytorch-cifar/issues/16
# https://github.com/kuangliu/pytorch-cifar/issues/19
# https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
cifar10_transforms = T.Compose([
    T.ToTensor(),
    # this is wrong!
    #T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # use this instead :)
    #T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    # This is 5-class stats on trainset
    T.Normalize((0.4903, 0.4873, 0.4642), (0.2519, 0.2498, 0.2657)),
])


class CIFAR10Dataset(Dataset):

    def __init__(self, train:bool = True, size:int = 100000000, transform=cifar10_transforms):
        """
        随机选择5个类构造CIFAR10数据集；测试集每个类仅随机抽选100个测试样本。
        Args:
            train (bool): 是否加载训练数据集，如果为 False，则加载测试数据集。默认为 True。
            size (int): 数据集的大小。
        """
        self.dataset = CIFAR10(root=DATA_PATH, train=train, download=True, transform=transform)
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # 随机选择5个类别
        label_list= [8, 7, 9, 1, 2]
        self.label_map = {l: i for i, l in enumerate(label_list)} # [8, 7, 9, 1, 2] -> [0, 1, 2, 3, 4]
        self.inverse_label_map = {i: l for i, l in enumerate(label_list)} # [0, 1, 2, 3, 4] -> [8, 7, 9, 1, 2]
        
        # 从数据集中筛选出我们感兴趣的标签/类别，并且映射标签 
        self.sub_dataset = []
        for image, label in self.dataset:
            if label in label_list:
                self.sub_dataset.append((image, self.label_map[label]))

        # 如果是测试集，每个类别随机抽选100个样本
        if not train:
            selected_samples = []
            for label in range(5):
                samples = [sample for sample in self.sub_dataset if sample[1] == label]
                selected_samples.extend(random.sample(samples, min(100, len(samples))))
            self.sub_dataset = selected_samples
        
        # shuffle
        random.seed(42)
        random.shuffle(self.sub_dataset)
        self.sub_dataset = self.sub_dataset[:size]
        del self.dataset
        
    def get_label_name(self, label):
        return self.class_names[self.inverse_label_map[label]]

    def __len__(self):
        return len(self.sub_dataset)

    def __getitem__(self, idx):
        x = self.sub_dataset[idx][0]
        y = torch.tensor(self.sub_dataset[idx][1], dtype=torch.long)
        return x, y


class PerfectAmplitudeEncodingCircuit:
    def __init__(self, image):
        """image: (3, 32, 32)"""
        self.state_vector = reshape_norm_padding(image).squeeze(0)
        self.operators = nn.Sequential()
    def __call__(self):
        return self.state_vector # (4096, 1)

class PerfectAmplitudeEncodingDataset(Dataset):

    def __init__(self, train:bool = True, size:int = 100000000):
        """
        完美振幅编码的CIFAR10Dataset类，跳过了编码线路的构建，可用于快速迭代量子神经网络的训练。
        Args:
            train (bool): 是否加载训练数据集，如果为 False，则加载测试数据集。默认为 True。
            size (int): 数据集的大小。
        """
        
        self.dataset = CIFAR10Dataset(train=train, size=size)
        self.num_total_gates = 0
        self.quantum_dataset = self.encode_data()
        del self.dataset

    def encode_data(self):
        """
        返回: a list of tuples (原始经典数据, 标签, 振幅编码量子线路的输出)=(image, label, state_vector)
        """
        quantum_dataset = []
        for image, label in tqdm(self.dataset):
            encoding_circuit = PerfectAmplitudeEncodingCircuit(image)
            self.num_total_gates += count_gates(encoding_circuit)
            quantum_dataset.append((image, label, encoding_circuit()))
        return quantum_dataset
    
    def get_gates_count(self):
        """计算在这个数据集上的编码线路门的平均个数"""
        return self.num_total_gates / len(self.quantum_dataset)

    def __len__(self):
        return len(self.quantum_dataset)

    def __getitem__(self, idx):
        return self.quantum_dataset[idx]


# 注意: 决赛的数据集名字必须固定为 QCIFAR10Dataset
# todo: 构建振幅编码线路
is_show_gate_count = True
def encode_single_data(data, debug:bool=False):
    global is_show_gate_count
    image, label = data     # [3, 32, 32], []

    # qam flatten:
    # [n_rep=1] fid=0.910, gcnt=79,  timecost=447s; n_iter=200, n_worker=16
    # [n_rep=2] fid=0.949, gcnt=157, timecost=853s; n_iter=200, n_worker=16
    def vqc_F1_all_wise_init_0(nq:int=12, n_rep:int=1):
        ''' RY(single init) - [pairwise(F1) - RY], param zero init '''
        vqc = dq.QubitCircuit(nqubit=nq)
        g = dq.Ry(nqubit=nq, wires=0, requires_grad=True)   # only init wire 0
        #g.init_para([np.pi/2])
        g.init_para([2.4619188346815495])   # MAGIC: 2*arccos(sqrt(2/3)) = 1.2309594173407747
        vqc.add(g)
        for _ in range(n_rep):
            for i in range(nq-1):   # qubit order
                for j in range(i+1, nq):
                    g = dq.Ry(nqubit=nq, wires=j, controls=i, requires_grad=True)
                    g.init_para([0.0])
                    vqc.add(g)
            for i in range(nq):
                g = dq.Ry(nqubit=nq, wires=i, requires_grad=True)
                g.init_para([0.0])
                vqc.add(g)
        return vqc

    # std flatten:
    # [n_rep=1] fid=0.846, gcnt=145, timecost=869s; n_iter=200, n_worker=16
    # [n_rep=2] fid=0.919, gcnt=289, timecost=1699s; n_iter=200, n_worker=16
    # qam flatten:
    # [n_rep=1] fid=0.956, gcnt=145, timecost=795s; n_iter=200, n_worker=16
    # [n_rep=1] fid=0.959, gcnt=145, timecost=1947s; n_iter=500, n_worker=16
    # [n_rep=2] fid=0.973, gcnt=289, timecost=1573s; n_iter=200, n_worker=16
    # qam flatten (hwc order):
    # [n_rep=1] fid=0.951, gcnt=145, timecost=795s; n_iter=200, n_worker=16
    def vqc_F2_all_wise_init_0(nq:int=12, n_rep:int=1):
        ''' RY(single init) - [pairwise(F2) - RY], param zero init '''
        vqc = dq.QubitCircuit(nqubit=nq)
        g = dq.Ry(nqubit=nq, wires=0, requires_grad=True)   # only init wire 0
        #g.init_para([np.pi/2])
        g.init_para([2.4619188346815495])   # MAGIC: 2*arccos(sqrt(2/3)) = 1.2309594173407747
        vqc.add(g)
        for _ in range(n_rep):
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
        return vqc

    n_iter = 200
    encoding_circuit = vqc_F2_all_wise_init_0(12, 1)
    gate_count = count_gates(encoding_circuit)
    if is_show_gate_count:
        is_show_gate_count = False
        print('gate_count:', gate_count)

    # 优化参数，使得线路能够制备出|x>
    target = reshape_norm_padding(image)
    optimizer = torch.optim.Adam(encoding_circuit.parameters(), lr=0.1)
    loss_list = []
    for _ in range(n_iter):
        state = encoding_circuit().unsqueeze(0)
        loss = -get_fidelity(state, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if debug: loss_list.append(loss.item())
    if debug:
        tht0 = encoding_circuit.operators[0].theta
        print('tht0:', tht0.item())
        print('fid:', -loss.item())
        import matplotlib.pyplot as plt
        plt.subplot(211) ; plt.plot(loss_list) ; plt.title('loss')
        plt.subplot(212)
        plt.plot(target        .real.flatten(), 'b', label='target')
        plt.plot(state.detach().real.flatten(), 'r', label='state')
        plt.legend()
        plt.tight_layout()
        plt.show()
        breakpoint()
    print('fid:', -loss.item())

    # Detach the state tensor before returning
    return (image, label, encoding_circuit().detach(), gate_count)


class QCIFAR10Dataset(Dataset):

    def __init__(self, train:bool = True, size:int = 100000000):
        """
        初始化 QCIFAR10Dataset 类。
        Args:
            train (bool): 是否加载训练数据集，如果为 False，则加载测试数据集。默认为 True。
            size (int): 数据集的大小。
        """
        self.dataset = CIFAR10Dataset(train=train, size=size)
        self.num_total_gates = 0
        ts_start = time()
        if os.getenv('DEBUG'):
            self.quantum_dataset = self.encode_data_single_thread()
        else:
            self.quantum_dataset = self.encode_data()
        ts_end = time()
        print('>> amp_enc_vqc time cost:', ts_end - ts_start)
        del self.dataset

    def encode_data_single_thread(self):
        results = [encode_single_data(it, debug=True) for it in self.dataset]
        quantum_dataset = [(image, label, state_vector) for image, label, state_vector, _ in results]
        self.num_total_gates = sum(gate_count for _, _, _, gate_count in results)
        return quantum_dataset

    def encode_data(self):
        """
        返回: a list of tuples (原始经典数据, 标签, 振幅编码量子线路的输出)=(image, label, state_vector)
        """
        num_cores = 16  # todo: 修改为合适的配置
        pool = multiprocessing.Pool(num_cores)
        results = list(tqdm(pool.imap(encode_single_data, self.dataset), total=len(self.dataset)))
        pool.close()
        pool.join()

        # Separate the results and count total gates
        quantum_dataset = [(image, label, state_vector) for image, label, state_vector, _ in results]
        self.num_total_gates = sum(gate_count for _, _, _, gate_count in results)
        return quantum_dataset

    def get_gates_count(self):
        """计算在这个数据集上的编码线路门的平均个数"""
        return self.num_total_gates / len(self.quantum_dataset)

    def __len__(self):
        return len(self.quantum_dataset)

    def __getitem__(self, idx):
        return self.quantum_dataset[idx]


if __name__ == '__main__':
    # 实例化测试集 QCIFAR10Dataset 并保存为pickle文件
    OUTPUT_DIR = 'output'
    # 确保目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_dataset = QCIFAR10Dataset(train=False) 
    print('dataset labels:', Counter(sample[1].item() for sample in test_dataset))
    with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'wb') as file:
        pickle.dump(test_dataset, file)
