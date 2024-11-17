from sklearn.metrics import top_k_accuracy_score
from test import *

# 测试模型的 topk 准确率
'''
top-k: 0.528
top-k: 0.734
top-k: 0.868
top-k: 0.942
'''

@torch.inference_mode
def test_model(model:QuantumNeuralNetwork, test_loader, device):
    state_pred = []
    state_true = []
    y_pred = []
    y_true = []

    model.to(device).eval()
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

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    for k in [1, 2, 3, 4]:
        acc = top_k_accuracy_score(y_true, y_pred, k=k)
        print('top-k:', acc)


if __name__ == '__main__':
    DEVICE = "cuda:0"
    OUTPUT_DIR = "output"
    BATCH_SIZE = 512    # todo: 修改为合适的配置

    t0 = time()
    with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'rb') as file:
        test_dataset = pkl.load(file)
    t1 = time()
    print(f'>> load pickle done ({t1 - t0:.3f}s)')      # 0.121s

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
    test_model(model, test_loader, DEVICE)
    torch.cuda.current_stream().synchronize()
    t1 = time()
    runtime = t1 - t0

    print(f'runtime: {runtime:.3f}')
