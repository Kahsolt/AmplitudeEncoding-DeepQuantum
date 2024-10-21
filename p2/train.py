import os
import json
import shutil
import glob
import pickle
import random
from time import time
from collections import Counter

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm

from model import QuantumNeuralNetwork
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


def train_model(model, optimizer, train_loader, valid_loader, num_epochs,  output_dir, patience=10, device='cpu'):
    """
    Train and validate the model, implementing early stopping based on validation loss.
    The best model is saved, along with the loss history after each epoch and gradient norms.
    Additionally, copies all Python source code files to the output directory.

    Args:
        model (torch.nn.Module): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs to train.

    Returns:
        model (torch.nn.Module): Trained model.
        history (dict): Dictionary containing training and validation loss history.
    """
    
    model.to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    os.makedirs(output_dir, exist_ok=True)
    output_src_dir = os.path.join(output_dir, "src")
    os.makedirs(output_src_dir, exist_ok=True)
    
    # Copy .py files to output_dir when train.py is executed
    source_code_files = glob.glob('*.py')  # Adjust the path if your code files are in a different directory
    for file in source_code_files:
        shutil.copy(file, output_src_dir)

    save_path = os.path.join(output_dir, "best_model.pt")
    save_acc_path = os.path.join(output_dir, "best_model.acc.pt")
    history_path = os.path.join(output_dir, "loss_history.json")
    pbar = tqdm(total=num_epochs * len(train_loader), position=0, leave=True)
    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    history = {
        'train_loss': [], 
        'valid_loss': [],
        'train_acc': [], 
        'valid_acc': [],
    }
    epochs_no_improve = 0  # Counter for epochs with no improvement in validation loss

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
            torch.save(model.state_dict(), save_acc_path)
            epochs_no_improve = 0

        if valid_loss < best_valid_loss - 0.003: 
            best_valid_loss = valid_loss
            print('')
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0  # Reset the counter since we have improvement
        else:
            epochs_no_improve += 1  # Increment the counter when no improvement

        if epochs_no_improve >= patience:
            tqdm.write("Early stopping triggered.")
            break  # Break out of the loop if no improvement for 'patience' number of epochs

    pbar.close()


if __name__ == '__main__':
    # Settings 
    DEVICE = "cuda:0"
    BATCH_SIZE = 32 # todo: 修改为合适的配置
    NUM_EPOCHS = 30 
    PATIENCE = 5 # if PATIENCE与NUM_EPOCHS相等，则不会触发early stopping
    OUTPUT_DIR  = 'output'

    dataset = PerfectAmplitudeEncodingDataset(train=True, size=2500) # todo: 修改为合适的配置
    print('dataset labels:', Counter(sample[1].item() for sample in dataset))
    train_size = int(0.7 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    # 构建数据加载器，用于加载训练和验证数据
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    '''
    赛方给定的基线表现：
    dataset labels: Counter({0: 515, 2: 510, 3: 497, 1: 495, 4: 483})
    参数量: 7200
    Epoch  1/30 - Train loss: 1.5869934, Train acc: 35.257%, Valid loss: 1.5594593, Valid acc: 40.533%
    Epoch  2/30 - Train loss: 1.5459839, Train acc: 42.400%, Valid loss: 1.5383592, Valid acc: 42.933%
    Epoch  3/30 - Train loss: 1.5273991, Train acc: 43.943%, Valid loss: 1.5279218, Valid acc: 43.333%
    Epoch  4/30 - Train loss: 1.5165007, Train acc: 45.943%, Valid loss: 1.5214744, Valid acc: 44.000%
    Epoch  5/30 - Train loss: 1.5088304, Train acc: 46.914%, Valid loss: 1.5160467, Valid acc: 44.800%
    Epoch  6/30 - Train loss: 1.5037423, Train acc: 47.829%, Valid loss: 1.5135677, Valid acc: 44.800%
    Epoch  7/30 - Train loss: 1.4994826, Train acc: 48.629%, Valid loss: 1.5113233, Valid acc: 44.800%
    Epoch  8/30 - Train loss: 1.4961005, Train acc: 48.800%, Valid loss: 1.5089825, Valid acc: 46.000%
    Epoch  9/30 - Train loss: 1.4937836, Train acc: 49.829%, Valid loss: 1.5073154, Valid acc: 45.600%
    Epoch 10/30 - Train loss: 1.4913128, Train acc: 49.943%, Valid loss: 1.5060765, Valid acc: 45.600%
    Epoch 11/30 - Train loss: 1.4896842, Train acc: 49.886%, Valid loss: 1.5047691, Valid acc: 45.200%
    Epoch 12/30 - Train loss: 1.4878187, Train acc: 50.057%, Valid loss: 1.5038306, Valid acc: 46.000%
    Epoch 13/30 - Train loss: 1.4863519, Train acc: 51.029%, Valid loss: 1.5032385, Valid acc: 46.667%  <-best
    Epoch 14/30 - Train loss: 1.4849304, Train acc: 50.971%, Valid loss: 1.5022921, Valid acc: 46.667%
    Epoch 15/30 - Train loss: 1.4837737, Train acc: 50.743%, Valid loss: 1.5020756, Valid acc: 46.000%
    Epoch 16/30 - Train loss: 1.4827636, Train acc: 50.629%, Valid loss: 1.5008316, Valid acc: 46.667%
    Epoch 17/30 - Train loss: 1.4813888, Train acc: 51.086%, Valid loss: 1.5009940, Valid acc: 45.867%
    Early stopping triggered.
    [修正mean_std之后，好像区别不太大]
    Epoch  1/30 - Train loss: 1.5874051, Train acc: 35.371%, Valid loss: 1.5602612, Valid acc: 40.800%
    Epoch  2/30 - Train loss: 1.5466901, Train acc: 42.457%, Valid loss: 1.5392548, Valid acc: 42.133%
    Epoch  3/30 - Train loss: 1.5281732, Train acc: 43.829%, Valid loss: 1.5288768, Valid acc: 43.600%
    Epoch  4/30 - Train loss: 1.5173102, Train acc: 45.657%, Valid loss: 1.5224680, Valid acc: 44.267%   <-best
    [使用 5-class mean_std, 收敛略快了一点点]
    Epoch  1/30 - Train loss: 1.5875892, Train acc: 35.371%, Valid loss: 1.5607864, Valid acc: 40.933%
    Epoch  2/30 - Train loss: 1.5469346, Train acc: 42.514%, Valid loss: 1.5397606, Valid acc: 42.933%
    Epoch  3/30 - Train loss: 1.5284595, Train acc: 44.286%, Valid loss: 1.5293732, Valid acc: 43.600%
    Epoch  4/30 - Train loss: 1.5176255, Train acc: 45.829%, Valid loss: 1.5231557, Valid acc: 44.267%
    Epoch  5/30 - Train loss: 1.5100189, Train acc: 46.857%, Valid loss: 1.5177592, Valid acc: 45.600%
    Epoch  6/30 - Train loss: 1.5048596, Train acc: 47.714%, Valid loss: 1.5153277, Valid acc: 46.000%
    Epoch  7/30 - Train loss: 1.5005771, Train acc: 48.686%, Valid loss: 1.5130805, Valid acc: 46.133%  <-best
    Epoch  8/30 - Train loss: 1.4971259, Train acc: 48.571%, Valid loss: 1.5106573, Valid acc: 46.400%
    Epoch  9/30 - Train loss: 1.4947398, Train acc: 49.086%, Valid loss: 1.5090124, Valid acc: 45.867%
    Epoch 10/30 - Train loss: 1.4922973, Train acc: 49.829%, Valid loss: 1.5078132, Valid acc: 45.600%
    Epoch 11/30 - Train loss: 1.4906985, Train acc: 50.057%, Valid loss: 1.5063520, Valid acc: 46.133%
    Epoch 12/30 - Train loss: 1.4888429, Train acc: 49.771%, Valid loss: 1.5053731, Valid acc: 46.000%
    [不使用数据规范化, 精度略次]
    Epoch  1/30 - Train loss: 1.5854637, Train acc: 29.829%, Valid loss: 1.5572556, Valid acc: 39.200%
    Epoch  2/30 - Train loss: 1.5456625, Train acc: 37.886%, Valid loss: 1.5323892, Valid acc: 41.733%
    Epoch  3/30 - Train loss: 1.5274053, Train acc: 42.343%, Valid loss: 1.5203095, Valid acc: 43.333%
    Epoch  4/30 - Train loss: 1.5179104, Train acc: 40.057%, Valid loss: 1.5149054, Valid acc: 38.267%
    Epoch  5/30 - Train loss: 1.5128518, Train acc: 42.400%, Valid loss: 1.5080526, Valid acc: 42.133%
    Epoch  6/30 - Train loss: 1.5071378, Train acc: 42.343%, Valid loss: 1.5039527, Valid acc: 44.000%
    Epoch  7/30 - Train loss: 1.5027398, Train acc: 43.829%, Valid loss: 1.5022688, Valid acc: 43.333%
    Epoch  8/30 - Train loss: 1.4988990, Train acc: 44.229%, Valid loss: 1.4997599, Valid acc: 43.733%
    Epoch  9/30 - Train loss: 1.4982745, Train acc: 42.400%, Valid loss: 1.4976709, Valid acc: 43.467%
    Epoch 10/30 - Train loss: 1.4956283, Train acc: 44.571%, Valid loss: 1.4969729, Valid acc: 44.933%  <-best
    Epoch 11/30 - Train loss: 1.4954090, Train acc: 45.029%, Valid loss: 1.4965294, Valid acc: 41.867%
    Epoch 12/30 - Train loss: 1.4933487, Train acc: 45.257%, Valid loss: 1.4942989, Valid acc: 44.933%
    Epoch 13/30 - Train loss: 1.4916334, Train acc: 42.171%, Valid loss: 1.4929709, Valid acc: 44.667%
    Epoch 14/30 - Train loss: 1.4907677, Train acc: 44.343%, Valid loss: 1.4925091, Valid acc: 44.400%
    Epoch 15/30 - Train loss: 1.4904052, Train acc: 42.514%, Valid loss: 1.4926861, Valid acc: 43.200%
    Epoch 16/30 - Train loss: 1.4891938, Train acc: 44.686%, Valid loss: 1.4935781, Valid acc: 41.200%
    Epoch 17/30 - Train loss: 1.4880423, Train acc: 44.629%, Valid loss: 1.4902040, Valid acc: 42.400%
    '''
    # 创建一个量子神经网络模型
    model_config = {'num_qubits': 12, 'num_layers': 600} # todo: 修改为合适的配置
    model = QuantumNeuralNetwork(**model_config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 将字典保存到文件中
    with open(f'{OUTPUT_DIR}/model_config.pkl', 'wb') as file:
        pickle.dump(model_config, file)

    # 训练模型
    ts_start = time()
    train_model(
        model,
        optimizer,
        train_loader, 
        valid_loader,
        num_epochs=NUM_EPOCHS, 
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        patience=PATIENCE,
    )
    ts_end = time()
    print('>> train clf_model time cost:', ts_end - ts_start)
