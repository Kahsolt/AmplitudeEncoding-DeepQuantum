import os
import json
import shutil
import glob
import pickle
import random
import numpy as np
from copy import deepcopy
from time import time
from collections import Counter
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import QuantumNeuralNetwork, QuantumNeuralNetworkCL
from utils import PerfectAmplitudeEncodingDataset


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


def train_model(model:QuantumNeuralNetworkCL, train_loader:DataLoader, valid_loader:DataLoader, num_epochs:int, output_dir:str, device='cpu', optim_state=None):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if optim_state is not None:
        print('>> load optim state!!')
        optimizer.load_state_dict(optim_state)
    print(f"参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    save_path       = os.path.join(output_dir, "best_model.pt")
    save_optim_path = os.path.join(output_dir, "best_model.optim.pt")
    history_path    = os.path.join(output_dir, "loss_history.json")
    fig_path        = os.path.join(output_dir, "loss.png")
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
        for _, y, z in inner_pbar:
            y, z = y.to(device), z.to(device)
            optimizer.zero_grad()
            loss, fid_eq, fid_ne = model(z, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.shape[0]
            inner_pbar.set_description(f'Batch Loss: {loss.item():.4f}')
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
            for _, y, z in valid_loader:
                y, z = y.to(device), z.to(device)
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

        if 'plot':
            plt.clf()
            plt.plot(history['train_loss'], 'b', label='train_loss')
            plt.plot(history['valid_loss'], 'r', label='valid_loss')
            plt.savefig(fig_path, dpi=400)
            plt.close()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('>> save new best_acc ckpt')
            model.mk_ref_qstate(train_loader.dataset, device)   # backfill refdata before ckpt save
            torch.save(model.state_dict(), save_path)
            torch.save(optimizer.state_dict(), save_optim_path)

    # mark for correct .postprocess()
    model.is_training = False
    pbar.close()


if __name__ == '__main__':
    ts_start = time()

    # Settings 
    DEVICE = "cuda:0"
    OUTPUT_DIR = 'output'

    # Copy .py files to output_dir when train.py is executed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_src_dir = os.path.join(OUTPUT_DIR, "src")
    os.makedirs(output_src_dir, exist_ok=True)
    source_code_files = glob.glob('*.py')  # Adjust the path if your code files are in a different directory
    for file in source_code_files:
        shutil.copy(file, output_src_dir)

    # 总数据集
    dataset = PerfectAmplitudeEncodingDataset(train=True)   # 用全部数据训练，防止过拟合
    print('dataset labels:', Counter(it[1].item() for it in dataset))

    # 总模型
    model_fp = f'{OUTPUT_DIR}/best_model.pt'
    model_config = {'num_qubits': 12, 'num_layers': 3}  # todo: 修改为合适的配置
    model = QuantumNeuralNetwork(**model_config)
    with open(f'{OUTPUT_DIR}/model_config.pkl', 'wb') as file:
        pickle.dump(model_config, file)

    # 训练各子模型
    BATCH_SIZE = 128
    NUM_EPOCHS = 75
    REFINE_FROM_BASE = False
    for i in range(4):
        for j in range(i+1, 5):
            ts_start_sub = time()

            ''' Model or Ckpt '''
            key = f'bin_{i}-{j}'
            SUB_OUTPUT_DIR = f'./{OUTPUT_DIR}/{key}'
            submodel_fp = f'{SUB_OUTPUT_DIR}/best_model.pt'
            submodel_optim_fp = f'{SUB_OUTPUT_DIR}/best_model.optim.pt'
            submodel = model.model2_grid[key]
            if os.path.exists(submodel_fp):
                submodel.load_state_dict(torch.load(submodel_fp))
                print(f'>> load pretrained submodel from: {submodel_fp}')
                sub_optim_state = torch.load(submodel_optim_fp)
                print(f'>> load pretrained submodel optim from: {submodel_optim_fp}')
                continue
            else:
                os.makedirs(SUB_OUTPUT_DIR, exist_ok=True)
                print(f'>> start train submodel: {key}')

            ''' Data '''
            targets = [i, j]
            subdataset: PerfectAmplitudeEncodingDataset = deepcopy(dataset)
            subdataset.quantum_dataset = [(it[0], torch.tensor(targets.index(it[1].item()), dtype=torch.long), it[2]) for it in subdataset.quantum_dataset if it[1].item() in targets]
            print('dataset labels:', Counter(it[1].item() for it in subdataset))
            train_size = int(0.7 * len(subdataset))
            valid_size = len(subdataset) - train_size
            train_dataset, valid_dataset = random_split(subdataset, [train_size, valid_size])
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
            valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

            ''' Train '''
            submodel = submodel.to(DEVICE)
            train_model(submodel, train_loader, valid_loader, num_epochs=NUM_EPOCHS, output_dir=SUB_OUTPUT_DIR, device=DEVICE, optim_state=sub_optim_state)
            submodel = submodel.cpu()

            ''' Load & save best weight'''
            print(f'>> load best submodel from: {submodel_fp}')
            submodel.load_state_dict(torch.load(submodel_fp))
            print(f'>> save model to: {model_fp}')
            torch.save(model.state_dict(), model_fp)

            ts_end_sub = time()
            print(f'>> train submodel done: (timecost: {ts_end_sub - ts_start_sub:5f}s)')

    ts_end = time()
    print(f'>> Done! (timecost: {ts_end - ts_start:5f}s)')
