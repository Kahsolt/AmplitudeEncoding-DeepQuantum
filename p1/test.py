import time
import pickle

import torch
from torch.utils.data import DataLoader
#import torch.profiler as profiler
#from sklearn.metrics import confusion_matrix

from model import QuantumNeuralNetwork
from utils import MNISTDataset, cir_collate_fn, reshape_norm_padding, get_fidelity, get_acc
from utils import QMNISTDataset, QMNISTDatasetIdea, DataHolder      # keep for unpickle


@torch.inference_mode()
def test_model(model:QuantumNeuralNetwork, test_loader:DataLoader, device:torch.device):
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

    model = model.eval().to(device)
    for x, y, z in test_loader:
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

    return acc, fid.item(), test_loader.dataset.get_gates_count()


def validate_test_dataset(test_dataset):
    """
    验证选手上传的测试集是否有效。
    test_dataset: 选手上传的测试集，包括(x, y, z)=(image, label, encoding_circuit)
    """
    test_mnist_tensor = torch.stack([x for x, y, z in test_dataset])
    original_mnist_tensor = torch.stack([x for x, y in MNISTDataset(label_list=[0,1,2,3,4], train=False)])
    return torch.allclose(test_mnist_tensor, original_mnist_tensor)


if __name__ == '__main__':
    DEVICE = "cuda:0"
    OUTPUT_DIR = "output"
    BATCH_SIZE = 514    # test samples: 5139

    # 数据集
    with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'rb') as file:
        test_dataset = pickle.load(file)
    print('>> load pickle done')
    if not validate_test_dataset(test_dataset):
        raise RuntimeError('>> Error: test_dataset is not valid')
    print('>> check dataset done')

    # 模型
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=cir_collate_fn)
    with open(f'{OUTPUT_DIR}/model_config.pkl', 'rb') as file:
        model_config = pickle.load(file)
    model = QuantumNeuralNetwork(**model_config)
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/best_model.pt', map_location=torch.device('cpu')))
    print('>> load model done')

    # 模型理智检查
    if not 'sanity check':
        x, y, z = next(iter(test_loader))
        output = model.inference(z)

    # 模型测试
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
    gates_score = 1 - gates / 1000.0
    runtime_score = 1 - runtime / 360.0
    score = (2 * fid + acc + gates_score + 0.1 * runtime_score) * 100
    print(f'客观得分: {score:.3f}')
