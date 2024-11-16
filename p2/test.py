import os
import random
from time import time
import pickle as pkl

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from model import QuantumNeuralNetwork
from utils import CIFAR10Dataset, reshape_norm_padding, get_fidelity, get_acc
from utils import QCIFAR10Dataset       # keep for unpickle

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


@torch.inference_mode
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

    state_pred = []
    state_true = []
    y_pred = []
    y_true = []

    model.to(device).eval()  # 切换到评估模式
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
    
    y_pred = torch.argmax(y_pred, dim=1)
    acc = get_acc(y_pred, y_true)
    fid = get_fidelity(state_pred, state_true)

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    cmat = confusion_matrix(y_true, y_pred)
    print('acc:', acc)
    print('prec:', prec)
    print('recall:', recall)
    print('f1:', f1)
    print(cmat)
    print('-' * 32)

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
    BATCH_SIZE = 512    # todo: 修改为合适的配置

    t0 = time()
    with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'rb') as file:
        test_dataset = pkl.load(file)
    t1 = time()
    print(f'>> load pickle done ({t1 - t0:.3f}s)')      # 0.121s
    t0 = time()
    is_valid = validate_test_dataset(test_dataset)
    t1 = time()
    print(f'>> check dataset done ({t1 - t0:.3f}s)')    # 1.583s
    if not is_valid:
        raise RuntimeError('test_dataset is not valid')

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    x, y, z = next(iter(test_loader))
    x, y, z = x.to(DEVICE), y.to(DEVICE), z.to(DEVICE)

    with open(f'{OUTPUT_DIR}/model_config.pkl', 'rb') as file:
        model_config = pkl.load(file)
    model = QuantumNeuralNetwork(**model_config).to(DEVICE)
    
    output = model.inference(z)
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/best_model.pt', map_location=torch.device('cpu')))

    # 测试模型
    t0 = time()
    acc, fid, gates = test_model(model, test_loader, DEVICE)
    torch.cuda.current_stream().synchronize()
    t1 = time()
    runtime = t1 - t0

    print(f'test fid: {fid:.3f}')
    print(f'test acc: {acc:.3f}')
    print(f'test gates: {gates:.3f}')
    print(f'runtime: {runtime:.3f}')

    # 计算客观得分
    gates_score = 1 - gates / 2000.0 
    runtime_score = 1 - runtime / 360.0
    score = (2 * fid + acc + gates_score + 0.1 * runtime_score) * 100
    print(f'客观得分: {score:.3f}')

    print('-' * 32)
    enc_sore = 2 * fid + 1 - gates / 2000
    print(f'enc_sore: {enc_sore:.6f}')
    print(f'total_sore: {(enc_sore + acc + 0.1) * 100:.6f}')
