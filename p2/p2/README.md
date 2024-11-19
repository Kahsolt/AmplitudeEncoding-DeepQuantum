# 信通院大赛赛题


## p2 决赛

【赛题要求】

在CIFAR-10数据集上完成5分类任务。基于初赛的方案，实现振幅编码线路，能将 3 * 32 * 32 像素振幅编码到12个量子比特上。然后训练量子神经网络对编码后的输入数据 x 进行分类。针对决赛工作制作PPT，对决赛工作进行线下答辩。

决赛评判标准请参考初赛、复赛评判标准。

【评分标准】
1. 振幅编码后的量子态矢量与原数据的相似度（Fidelity），振幅编码线路的复杂度（门的个数）。
2. 量子神经网络的Top-1分类准确率（Accuracy）。
3. `客观得分 =（ 2*Fidelity + Accuracy + 线路复杂度得分 + 0.1*运行时间得分 ) * 100` 。
4. `线路复杂度得分 = 1-振幅编码线路门的个数/2000`。要求振幅编码线路门的个数小于等于2000。
5. `运行时间得分 = 1-运行时间/360秒`。要求线路调用test_model()函数的运行时间小于等于360秒。test_model()函数定义请参考决赛赛题初始代码`p2/`。


【项目结构】

```
.       
|-- README.md 
|-- utils.py 
|-- model.py
|-- train.py
|-- test.py
|-- output
| |-- src 
| |-- best_model.pt
| |-- loss_history.json
| |-- model_config.pkl
| |-- test_dataset.pkl
```

1. 请选手在 `utils.py` 中实现 `QCIFAR10Dataset`, 主要修改 `encode_single_data` 方法，该方法返回一个列表，列表中的元素是 (原始经典数据, 标签, 振幅编码后的量子态, 编码线路的门数)=(image, label, state_vector, gate_count)。可以采用启发式的量子线路编码每一张图片，也可以采用量子机器学习算法编码每一张图片。请选手实例化测试集` QCIFAR10Dataset(train=False) `并保存为 `test_dataset.pkl` 文件。测试集随机抽5个类，每个类随机抽选100个测试样本。
2. 请选手在 `model.py` 中实现 `QuantumNeuralNetwork`, 主要修改`create_var_circuit` 方法， 该方法构建量子变分线路并添加可观测量。
3. 选手可以使用 `train.py` 的训练代码，也可以自己编写训练代码，请保证 `OUTPUT_DIR='output'`, 在输出文件夹下存放训练好的`QuantumNeuralNetwork`的权重为`best_model.pt`，训练过程中的损失历史为`loss_history.json`，以及实例化`QuantumNeuralNetwork`所用到的参数`model_config.pkl`。默认使用PerfectAmplitudeEncodingDataset进行训练，选手也可以使用`QCIFAR10Dataset(train=True)`进行训练。
4. 请选手认真查看 `test.py`中的自动打分函数 `test_model` 的实现，打分函数要求 `QuantumNeuralNetwork.inference` 输入一个batch的图像数据对应的量子态，输出分类结果。



【提交说明】

项目目录p2提交（登录赛题提交系统：项目管理->创建新项目->上传项目，项目类型选择：执行训练文件）。



【如何安装】

- PyTorch >= 2.0.0  https://pytorch.org/get-started/locally/
- DeepQuantum >=3.0.0 https://deepquantum.turingq.com/

【如何使用】


用量子线路编码每一张图片，实例化测试集 QCIFAR10Dataset 并保存为pickle文件:
`python utils.py`

调试模型：
`python model.py`

训练模型：
`python train.py`

测试模型：
`python test.py`



   






