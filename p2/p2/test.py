import os
import time
import pickle
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from model import QuantumNeuralNetwork
from utils import reshape_norm_padding, get_fidelity, get_acc, QCIFAR10Dataset, CIFAR10Dataset

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


def test_model(model, test_loader, device):
    """
    测试模型。

    Args:
        model (torch.nn.Module): 要测试的模型。
        test_loader (torch.utils.data.DataLoader): 测试数据加载器。

    Returns:
        acc (float): 模型在测试集上的准确率。
        fid (float): 在测试集上的平均保真度。
        gates (float): 在测试集上的编码线路门的平均个数。
    """
    model.to(device)
    model.eval()  # 切换到评估模式

    state_pred = []
    state_true = []

    y_pred = []
    y_true = []

    with torch.no_grad():  # 不需要计算梯度
        for x, y, z in tqdm(test_loader, desc="Testing"):
            x, y, z = x.to(device), y.to(device), z.to(device)
            output = model.inference(z)

            state_pred.append(z)
            state_true.append(reshape_norm_padding(x))
            y_pred.append(output)
            y_true.append(y)

    state_pred = torch.cat(state_pred, dim=0)
    state_true = torch.cat(state_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    
    y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    acc = get_acc(y_pred, y_true)
    fid = get_fidelity(state_pred, state_true)

    cmat = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
    print(cmat)

    return acc, fid.item(), test_loader.dataset.get_gates_count()

def validate_test_dataset(test_dataset):
    """
    验证选手上传的测试集是否有效。
    test_dataset: 选手上传的测试集，包括(x, y, z)=(image, label, state_vector)
    """
    test_mnist_tensor = torch.stack([x for x, y, z in test_dataset])
    original_mnist_tensor = torch.stack([x for x, y in CIFAR10Dataset(train=False)])
    return torch.allclose(test_mnist_tensor, original_mnist_tensor)


if __name__ == '__main__':

    DEVICE = "cuda:0"
    OUTPUT_DIR = "output"
    BATCH_SIZE = 128    # todo: 修改为合适的配置

    with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'rb') as file:
        test_dataset = pickle.load(file)

    is_valid = validate_test_dataset(test_dataset)

    if is_valid:
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        x, y, z = next(iter(test_loader))
        x, y, z = x.to(DEVICE), y.to(DEVICE), z.to(DEVICE)

        with open(f'{OUTPUT_DIR}/model_config.pkl', 'rb') as file:
            model_config = pickle.load(file)
        model = QuantumNeuralNetwork(**model_config).to(DEVICE)
        
        output = model.inference(z)
        model.load_state_dict(torch.load(f'{OUTPUT_DIR}/best_model.pt', map_location=torch.device('cpu')))
                    
        # 测试模型
        t0 = time.time()
        
        acc, fid, gates = test_model(model, test_loader, DEVICE)
        
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        runtime = t1 - t0

        print(f'test fid: {fid:.3f}')
        print(f'test acc: {acc:.3f}')
        print(f'test gates: {gates:.3f}')
        print(f'runtime: {runtime:.3f}')


        # 计算客观得分
        gates_score = 1 - gates/2000.0 
        runtime_score = 1 - runtime/360.0
        score = (2 * fid + acc + gates_score + 0.1 * runtime_score) * 100

        print(f'客观得分: {score:.3f}')

    else:
        print(f'test_dataset is not valid')