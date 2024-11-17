
# 测试二分类模型

'''
[测试集精度]
| * |   0   |   1   |   2   |   3   |   4   |
| 0 |       | 0.870 | 0.750 | 0.790 | 0.850 |
| 1 | 0.845 |       | 0.845 | 0.860 | 0.750 |
| 2 | 0.730 | 0.845 |       | 0.735 | 0.860 |
| 3 | 0.745 | 0.855 |       |       | 0.880 |
| 4 |       |       |       |       |       |
'''

import os
import torch.nn as nn
from argparse import ArgumentParser

from test import *


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-T', '--targets', default='0,1', type=(lambda s: [int(e) for e in s.split(',')]), help='class labels for binary clf')
    args = parser.parse_args()

    assert len(args.targets) == 2
    A, B = args.targets
    assert 0 <= A <= 4 and 0 <= B <= 4

    BASE_OUTPUT_DIR = './output'
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f'bin_{A}-{B}')
    assert os.path.isdir(OUTPUT_DIR)

    DEVICE = "cuda:0"
    BATCH_SIZE = 512    # todo: 修改为合适的配置

    t0 = time()
    with open(f'{BASE_OUTPUT_DIR}/test_dataset.pkl', 'rb') as file:
        test_dataset: QCIFAR10Dataset = pkl.load(file)
    t1 = time()
    print(f'>> load pickle done ({t1 - t0:.3f}s)')      # 0.121s

    test_dataset.quantum_dataset = [(it[0], args.targets.index(it[1].item()), it[2]) for it in test_dataset.quantum_dataset if it[1].item() in args.targets]
    print('len(testset):', len(test_dataset))

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    x, y, z = next(iter(test_loader))
    x, y, z = x.to(DEVICE), y.to(DEVICE), z.to(DEVICE)

    with open(f'{OUTPUT_DIR}/model_config.pkl', 'rb') as file:
        model_config = pkl.load(file)
    model = QuantumNeuralNetwork(**model_config)
    model.ref_qstate = nn.Parameter(torch.zeros([2, 36], requires_grad=False))      # fix shape
    model = model.to(DEVICE)

    output = model.inference(z)
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/best_model.pt', map_location=DEVICE))

    # 测试模型
    t0 = time()
    acc, fid, gates = test_model(model, test_loader, DEVICE)
    torch.cuda.current_stream().synchronize()
    t1 = time()
    runtime = t1 - t0

    print(f'targets: {args.targets}')
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
