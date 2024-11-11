import os
import json
import shutil
import glob
import random
import pickle as pkl
from time import time
from collections import Counter

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import QuantumNeuralNetwork, QuantumNeuralNetworkAnsatz, QuantumNeuralNetworkCL
from utils import QCIFAR10Dataset, PerfectAmplitudeEncodingDataset

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


def train_model(model:QuantumNeuralNetworkAnsatz, optimizer:optim.Optimizer, train_loader:DataLoader, valid_loader:DataLoader, num_epochs:int, output_dir:str, device='cpu'):
    model.to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    os.makedirs(output_dir, exist_ok=True)
    output_src_dir = os.path.join(output_dir, "src")
    os.makedirs(output_src_dir, exist_ok=True)
    # Copy .py files to output_dir when train.py is executed
    source_code_files = glob.glob('*.py')  # Adjust the path if your code files are in a different directory
    for file in source_code_files: shutil.copy(file, output_src_dir)
    save_path = os.path.join(output_dir, "best_model.pt")
    history_path = os.path.join(output_dir, "loss_history.json")
    fig_path = os.path.join(output_dir, "loss_acc.png")

    pbar = tqdm(total=num_epochs * len(train_loader), position=0, leave=True)
    best_valid_acc = 0.0
    history = {
        'train_loss': [], 
        'valid_loss': [],
        'train_acc': [], 
        'valid_acc': [],
    }

    for epoch in range(num_epochs):
        ''' Train '''
        model.train()
        train_loss = 0.0
        train_acc = 0
        inner_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, position=1)
        for x, y, z in inner_pbar:
            x, y, z = x.to(device), y.to(device), z.to(device)
            optimizer.zero_grad()
            loss, output = model(z, y)
            loss.backward()
            #total_norm = clip_grad_norm_(model.parameters(), float('inf'))  # Calculates the total norm value and clips gradients
            total_norm = 0
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.argmax(-1) == y).sum().item()
            inner_pbar.set_description(f'Batch Loss: {loss.item():.4f} | Grad Norm: {total_norm:.4f}')
            pbar.update(1)
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        print(f'Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.7f}, Train acc: {train_acc:.3%}')
        with open(history_path, 'w', encoding='utf-8') as fh:
            json.dump(history, fh, indent=2, ensure_ascii=False)

        ''' Eval '''
        model.eval()
        valid_loss = 0.0
        valid_acc = 0
        with torch.no_grad():
            for x, y, z in valid_loader:
                x, y, z = x.to(device), y.to(device), z.to(device)
                loss, output = model(z, y)
                valid_loss += loss.item() * y.size(0)
                valid_acc += (output.argmax(-1) == y).sum().item()
        valid_loss /= len(valid_loader.dataset)
        valid_acc /= len(valid_loader.dataset)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        print(f'Epoch {epoch+1}/{num_epochs} - Valid loss: {valid_loss:.7f}, Valid acc: {valid_acc:.3%}')
        with open(history_path, 'w', encoding='utf-8') as fh:
            json.dump(history, fh, indent=2, ensure_ascii=False)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            print('>> save new best_acc ckpt')
            torch.save(model.state_dict(), save_path)

    pbar.close()

    plt.clf()
    plt.plot(history['train_loss'], 'dodgerblue', label='train_loss')
    plt.plot(history['valid_loss'], 'orange',     label='valid_loss')
    ax = plt.twinx()
    ax.plot(history['train_acc'], 'b', label='train_acc')
    ax.plot(history['valid_acc'], 'r', label='valid_acc')
    plt.savefig(fig_path, dpi=400)
    plt.close()


def train_model_cl(model:QuantumNeuralNetworkCL, optimizer:optim.Optimizer, train_loader:DataLoader, valid_loader:DataLoader, num_epochs:int, output_dir:str, device='cpu'):
    model.to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "best_model.pt")
    history_path = os.path.join(output_dir, "loss_history.json")
    fig_path = os.path.join(output_dir, "loss.png")
    pbar = tqdm(total=num_epochs * len(train_loader), position=0, leave=True)
    best_valid_loss = float('inf')
    history = {
        'train_loss': [], 
        'valid_loss': [],
        'valid_fid_eq': [],
        'valid_fid_ne': [],
    }

    # mark for correct .postprocess()
    model.is_training = True
    for epoch in range(num_epochs):
        ''' Train '''
        model.train()
        train_loss = 0.0
        inner_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, position=1)
        for x, y, z in inner_pbar:
            x, y, z = x.to(device), y.to(device), z.to(device)
            optimizer.zero_grad()
            loss, fid_eq, fid_ne = model(z, y)
            loss.backward()
            #total_norm = clip_grad_norm_(model.parameters(), float('inf'))  # Calculates the total norm value and clips gradients
            total_norm = 0
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            inner_pbar.set_description(f'Batch Loss: {loss.item():.4f} | Grad Norm: {total_norm}')
            pbar.update(1)
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        print(f'Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.7f}')
        with open(history_path, 'w', encoding='utf-8') as fh:
            json.dump(history, fh, indent=2, ensure_ascii=False)

        ''' Eval '''
        model.eval()
        valid_loss = 0.0
        fid_eq_sum, fid_ne_sum = 0.0, 0.0
        with torch.no_grad():
            for x, y, z in valid_loader:
                x, y, z = x.to(device), y.to(device), z.to(device)
                loss, fid_eq, fid_ne = model(z, y)
                n_sample_batch = y.shape[0]
                fid_eq_sum += fid_eq.item() * n_sample_batch
                fid_ne_sum += fid_ne.item() * n_sample_batch
                valid_loss += loss  .item() * n_sample_batch
        n_samples = len(valid_loader.dataset)
        valid_loss /= n_samples
        fid_eq_sum /= n_samples
        fid_ne_sum /= n_samples
        history['valid_loss'  ].append(valid_loss)
        history['valid_fid_eq'].append(fid_eq_sum)
        history['valid_fid_ne'].append(fid_ne_sum)
        print(f'Epoch {epoch+1}/{num_epochs} - Valid loss: {valid_loss:.7f}, fid_eq: {fid_eq_sum}, fid_ne: {fid_ne_sum}')
        with open(history_path, 'w', encoding='utf-8') as fh:
            json.dump(history, fh, indent=2, ensure_ascii=False)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('>> save new best_acc ckpt')
            model.mk_ref_qstate(train_loader.dataset, DEVICE)   # backfill refdata before ckpt save
            torch.save(model.state_dict(), save_path)

    pbar.close()

    plt.clf()
    plt.plot(history['train_loss'], 'b', label='train_loss')
    plt.plot(history['valid_loss'], 'r', label='valid_loss')
    plt.savefig(fig_path, dpi=400)
    plt.close()


if __name__ == '__main__':
    # Settings
    DEVICE     = "cuda:0"
    OUTPUT_DIR = 'output'
    NUM_LAYER  = 10      # todo: 修改为合适的配置
    BATCH_SIZE = 32      # todo: 修改为合适的配置
    NUM_EPOCHS = 100     # [30, 100]
    OVERFIT    = True

    if OVERFIT:      # 实验性地过拟合测试集，使用编码数据
        with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'rb') as fh:
            dataset = pkl.load(fh)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
        valid_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    else:
        dataset = PerfectAmplitudeEncodingDataset(train=True)   # 用全部数据训练，防止过拟合
        print('dataset labels:', Counter(sample[1].item() for sample in dataset))
        train_size = int(0.7 * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        # 构建数据加载器，用于加载训练和验证数据
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 创建一个量子神经网络模型
    model_config = {'num_qubits': 12, 'num_layers': NUM_LAYER}
    model = QuantumNeuralNetwork(**model_config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 将字典保存到文件中
    with open(f'{OUTPUT_DIR}/model_config.pkl', 'wb') as file:
        pkl.dump(model_config, file)

    # 训练模型
    train_model_fn = train_model_cl if isinstance(model, QuantumNeuralNetworkCL) else train_model
    ts_start = time()
    train_model_fn(
        model,
        optimizer,
        train_loader, 
        valid_loader,
        num_epochs=NUM_EPOCHS, 
        output_dir=OUTPUT_DIR,
        device=DEVICE,
    )
    ts_end = time()
    print('>> train clf_model time cost:', ts_end - ts_start)   # 5531s
