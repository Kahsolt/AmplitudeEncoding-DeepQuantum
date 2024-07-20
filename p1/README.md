# 信通院大赛赛题

## p1 初赛

【赛题说明】

在量子机器学习中，经典数据需要被编码到量子态上以供量子算法处理。振幅编码是一种常见的编码方法，它利用量子态的振幅来表示数据。这种编码方式可以高效地利用量子态的多个维度，使得量子算法能够在更小的量子系统上处理大规模的数据。

振幅编码是很多量子机器学习算法的性能瓶颈。例如，在著名的HHL算法中，如果不能实现高效的振幅编码，该算法的指数加速效果将被复杂的编码过程抵消。

尽管振幅编码在理论上是高效的，但在实际中，将经典数据编码到量子态上是一个非常具有挑战性的任务。研究者们已经提出了多种振幅编码策略，例如， Mottonen 线路可以精确地将 2^n 个复数编码到 n 个量子比特的振幅上，尽管线路的深度是指数增长的。 Zoufal 等人使用量子生成对抗网络，能够近似地将 2^n 个实数编码到很浅的线路上。

参赛团队被要求使用DeepQuantum实现振幅编码线路，能够将任意实数向量 x 编码到量子态 psi 中，其中 c 是归一化常数。然后使用量子神经网络对输入数据 x 进行分类。

【赛题要求】

在MNIST数据集上完成分类任务。实现振幅编码线路，能将 28 * 28像素振幅编码到10个量子比特上。然后训练量子神经网络对编码后的输入数据 x进行分类。初赛仅要求使用数字0、1、2、3、4对应的数据，完成5分类任务。

【评分标准】
1. 振幅编码后的量子态矢量与原数据的相似度（Fidelity），振幅编码线路的复杂度（门的个数）。
2. 量子神经网络的Top-1分类准确率（Accuracy）。
3. `客观得分 = (2*Fidelity + Accuracy + 线路复杂度得分 + 0.1*运行时间得分) * 100` 。
4. `线路复杂度得分 = 1-振幅编码线路门的个数/1000`。要求振幅编码线路门的个数小于等于1000。
5. `运行时间得分 = 1-运行时间/360秒`。要求线路调用test_model()函数的运行时间小于等于360秒。test_model()函数定义请参考初赛赛题初始代码`p1/`。

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

1. 请选手在 `utils.py` 中实现 `QMNISTDataset`, 主要修改 `generate_data` 方法，该方法返回一个列表，列表中的元素是 (原始经典数据, 标签, 振幅编码量子线路)=(image, label, encoding_circuit)。
可以采用启发式的量子线路编码每一张图片，也可以采用量子机器学习算法编码每一张图片。请选手实例化测试集`QMNISTDataset(label_list=[0,1,2,3,4], train=False)`并保存为 `test_dataset.pkl` 文件。
2. 请选手在 `model.py` 中实现 `QuantumNeuralNetwork`, 主要修改`create_var_circuit` 方法， 该方法构建量子变分线路并添加可观测量。
3. 选手可以使用 `train.py` 的训练代码，也可以自己编写训练代码，请保证 `OUTPUT_DIR='output'`, 在输出文件夹下存放训练好的`QuantumNeuralNetwork`的权重为`best_model.pt`，训练过程中的损失历史为`loss_history.json`，以及实例化`QuantumNeuralNetwork`所用到的参数`model_config.pkl`。
4. 请选手认真查看 `test.py`中的自动打分函数 `test_model` 的实现，打分函数要求 `QuantumNeuralNetwork.inference` 输入一个batch的图像数据对应的量子态，输出分类结果。

【提交说明】

项目目录p1提交（登录赛题提交系统：项目管理->创建新项目->上传项目，项目类型选择：执行训练文件）。

【如何安装】

- PyTorch >= 2.0.0  https://pytorch.org/get-started/locally/
- DeepQuantum 

【如何使用】

- `python utils.py`: 用量子线路编码每一张图片，实例化测试集 QMNISTDataset 并保存为 pickle 文件
- `python model.py`: 调试模型
- `python train.py`: 训练模型
- `python test.py`: 测试模型
