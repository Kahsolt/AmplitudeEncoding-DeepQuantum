import os
import json
import shutil
import glob
import pickle

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import QuantumNeuralNetwork
from utils import reshape_norm_padding, count_gates, cir_collate_fn, QMNISTDataset, FakeDataset, FakeDatasetApprox


def train_model(model:QuantumNeuralNetwork, optimizer:optim.Optimizer, train_loader:DataLoader, valid_loader:DataLoader, num_epochs:int, output_dir, patience=10, device='cpu'):
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

    save_path = os.path.join(output_dir, f"best_model.pt")
    history_path = os.path.join(output_dir, f"loss_history.json")
    pbar = tqdm(total=num_epochs * len(train_loader), position=0, leave=True)
    best_valid_loss = float('inf')
    history = {'train_loss': [], 'valid_loss': []}
    epochs_no_improve = 0  # Counter for epochs with no improvement in validation loss

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        inner_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, position=1)

        for x, y, z in inner_pbar:
            x, y, z = x.to(device), y.to(device), z.to(device)
            optimizer.zero_grad()
            loss, output = model(z, y)
            loss.backward()
            # Calculate gradient norms
            total_norm = clip_grad_norm_(model.parameters(), float('inf'))  # Calculates the total norm value and clips gradients
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            inner_pbar.set_description(f'Batch Loss: {loss.item():.4f} | Grad Norm: {total_norm:.2f}')
            pbar.update(1)

        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x, y, z in valid_loader:
                x, y, z = x.to(device), y.to(device), z.to(device)
                loss, output = model(z, y)
                valid_loss += loss.item() * y.size(0)

        valid_loss /= len(valid_loader.dataset)
        history['valid_loss'].append(valid_loss)
        pbar.set_description(f'Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.3f}, Valid loss: {valid_loss:.3f}')

        with open(history_path, 'w') as f:
            json.dump(history, f)

        if valid_loss < best_valid_loss-0.003: 
            best_valid_loss = valid_loss
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
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    PATIENCE = 3 # if PATIENCE与NUM_EPOCHS相等，则不会触发early stopping
    OUTPUT_DIR  = 'output'

    # dataset = FakeDataset(size=10, noise_strength=0.0)
    # dataset = FakeDatasetApprox(size=20, noise_strength=0.0)
    dataset = QMNISTDataset(label_list=[0,1,2,3,4], train=True)

    # 构建数据加载器，用于加载训练和验证数据
    train_size = int(0.7 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=cir_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=cir_collate_fn)
 
    # 创建一个量子神经网络模型
    model_config = {'n_qubit': 10, 'n_layer': 30}
    model = QuantumNeuralNetwork(**model_config)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
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

    # 将字典保存到文件中
    with open(f'{OUTPUT_DIR}/model_config.pkl', 'wb') as file:
        pickle.dump(model_config, file)
