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
import matplotlib.pyplot as plt

from model import QuantumNeuralNetwork
from utils import QMNISTDataset, QMNISTDatasetIdea, cir_collate_fn


def train_model(model:QuantumNeuralNetwork, optimizer:optim.Optimizer, train_loader:DataLoader, valid_loader:DataLoader, num_epochs:int, output_dir:str, patience=10, device='cpu'):
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
    history_path = os.path.join(output_dir, "loss_history.json")
    pbar = tqdm(total=num_epochs * len(train_loader), position=0, leave=True)
    best_valid_loss = float('inf')
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
            #total_norm = clip_grad_norm_(model.parameters(), float('inf'))  # Calculate gradient norms
            total_norm = 0
            optimizer.step()
            train_loss += loss.item() * len(x)
            train_acc += (output.argmax(-1) == y).sum().item()
            inner_pbar.set_description(f'Batch Loss: {loss.item():.4f} | Grad Norm: {total_norm:.2f}')
            pbar.update()
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
                valid_loss += loss.item() * len(x)
                valid_acc += (output.argmax(-1) == y).sum().item()
        valid_loss /= len(valid_loader.dataset)
        valid_acc /= len(valid_loader.dataset)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        print(f'Epoch {epoch+1}/{num_epochs} - Valid loss: {valid_loss:.7f}, Valid acc: {valid_acc:.3%}')
        with open(history_path, 'w', encoding='utf-8') as fh:
            json.dump(history, fh, indent=2, ensure_ascii=False)

        ''' Ckpt '''
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch={epoch}.pt"))

        if valid_loss < best_valid_loss:
            print(f'>> New best found {best_valid_loss:.7f} => {valid_loss:.7f}')
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0   # Reset the counter since we have improvement
        else:
            epochs_no_improve += 1  # Increment the counter when no improvement

        if epochs_no_improve >= patience:
            tqdm.write(">> Early stopping triggered.")
            break  # Break out of the loop if no improvement for 'patience' number of epochs

    pbar.close()

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history['train_acc'], 'orange', label='train acc')
    ax.plot(history['valid_acc'], 'red',    label='valid acc')
    ax.set_ylabel('accuracy')
    ax2 = ax.twinx()
    ax2.plot(history['train_loss'], 'dodgerblue', label='train loss')
    ax2.plot(history['valid_loss'], 'blue',       label='valid loss')
    ax2.set_ylabel('loss')
    fig.legend(loc=7, bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)    # 7 = center right
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/loss_acc.png', dpi=300)


if __name__ == '__main__':
    # Settings 
    DEVICE = "cuda:0"
    BATCH_SIZE = 64
    NUM_EPOCHS = 35
    PATIENCE = NUM_EPOCHS   # if PATIENCE与NUM_EPOCHS相等，则不会触发early stopping
    OUTPUT_DIR = 'output'

    # Data
    #dataset = QMNISTDataset(label_list=[0,1,2,3,4], train=True, per_cls_size=1000)
    dataset = QMNISTDatasetIdea(label_list=[0,1,2,3,4], train=True, per_cls_size=1000)
    #dataset = QMNISTDatasetIdea(label_list=[0,1,2,3,4], train=False)
    #with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'rb') as file:
    #    dataset = pickle.load(file)
    train_size = int(0.7 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=cir_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=cir_collate_fn)
 
    # Model
    model_config = {'n_qubit': 10, 'n_layer': 30}
    model = QuantumNeuralNetwork(**model_config)
    model = model.train().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # Hparam
    with open(f'{OUTPUT_DIR}/model_config.pkl', 'wb') as file:
        pickle.dump(model_config, file)

    # Train
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
