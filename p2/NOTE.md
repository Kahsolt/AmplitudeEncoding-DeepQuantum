### Score Metrics

决赛评判标准 (满分为 410 分)

1. 振幅编码后的量子态矢量与原数据的相似度 Fidelity，振幅编码线路的复杂度 (门的个数)
2. 量子神经网络的 Top-1 分类准确率 Accuracy
3. 客观得分 = (2 * Fidelity + Accuracy + 线路复杂度得分 + 0.1 * 运行时间得分) * 100
4. 线路复杂度得分 = 1 - 振幅编码线路门的个数 / 2000；要求振幅编码线路门的个数 <= 1000
5. 运行时间得分 = 1 - 运行时间 / 360秒；要求线路调用 test_model() 函数的运行时间 <= 360s


### Trails

> Q: 训练集和验证集的损失都能同步下降，但精度并非稳步同步上升，Why? (p1也有此现象)
> A: https://blog.51cto.com/u_12891/7785560

```
[Baseline]
test fid: 0.348
test acc: 0.306
test gates: 1212.000
runtime: 111.535
客观得分: 146.514

[Trail 0]
- enc: vqc_F1_all_wise_init_0 (d=1/2/3), no_data_norm
- clf: baseline[RY-cyclic(CNOT)|z(0~4)]
test fid:   0.930  /  0.954   /  0.965
test acc:   0.360  /  0.352   /  0.360
test gates:   79   /   157    /   235
runtime:    15.163 /  15.341  /  15.272
客观得分:   327.702 / 327.634  / 326.741
ℹ 需达到 fid=0.95 左右，可视化为图像才看起来有人类可识别的同一性

[Trail 1] qam_flatten
| encoder | n_layer | gate count | fidelity | score | comment |
| vqc_F1_all_wise_init   | 3 |  79 | 0.903 | 2.7665 | no_data_norm |
| vqc_F1_all_wise_init_0 | 1 |  79 | 0.906 | 2.7725 | no_data_norm |
| vqc_F2_all_wise_init_0 | 1 | 145 | 0.946 | 2.8195 | no_data_norm |
| vqc_F2_all_wise_init_0 | 2 | 289 | 0.961 | 2.7775 | no_data_norm |
| vqc_F1_all_wise_init_0 | 1 |  79 | 0.935 | 2.8305 | no_data_norm, hwc order |
| vqc_F2_all_wise_init_0 | 1 | 145 | 0.952 | 2.8315 | no_data_norm, hwc order (⭐) |
[Local]
  classifier gate count: 14400
  test fid: 0.952
  test acc: 0.420
  test gates: 145.000
  runtime: 15.296
  客观得分: 334.770
[Submit]
  Fidelity: 0.952
  Accuracy: 0.420
  振幅编码线路门的个数: 145.0
  运行时间: 5.347104549407959
  客观得分: 335.0461788574855

[Trail 2] qam_flatten (optimize & bugfix)
| encoder | n_layer | gate count | fidelity | score | comment |
| vqc_F1_all_wise_init_0 | 1 |  79 | 0.910 | 2.7805 | no_data_norm |
| vqc_F1_all_wise_init_0 | 2 | 157 | 0.949 | 2.8195 | no_data_norm |
| vqc_F2_all_wise_init_0 | 1 | 145 | 0.956 | 2.8385 | no_data_norm |
| vqc_F2_all_wise_init_0 | 1 | 145 | 0.959 | 2.8455 | no_data_norm, n_iter=500 (⭐) |
| vqc_F2_all_wise_init_0 | 2 | 289 | 0.973 | 2.8015 | no_data_norm |
| vqc_F2_all_wise_init_0 | 1 | 145 | 0.951 | 2.8295 | no_data_norm, hwc order |
[Local] (暂用基线clf，qcnn过拟合了更烂)
  classifier gate count: 14400
  test fid: 0.959
  test acc: 0.420
  test gates: 145.000
  runtime: 15.562
  客观得分: 336.131

[Trail 3] std_flatten + data_norm (we'are fucking back!)
| encoder | n_layer | gate count | fidelity | score | comment |
| vqc_F2_all_wise_init_0 | 1 | 145 | 0.846 | 2.6195 | data_norm |
| vqc_F2_all_wise_init_0 | 2 | 289 | 0.919 | 2.6935 | data_norm |
😈 分类模型使用 qcnn，离奇的是训练时验证集精度仍然在 42% 左右，测试精度 39.4%
难道任何 ansatz 结构无论在 std 还是 qam 展开方式下，最高精度都突破不了这个神秘数字 42%??

[Trail 4] qam_flatten
使用 F2_all 作分类器，使用 best_acc 检查点；相比基线减少了门数量，精度依然在瓶颈 42% 处
[Local]
  classifier gate count: 1452
  test fid: 0.959 (qam_flatten + F2 layer=2)
  test acc: 0.420 (F2 + best acc ckpt)
  test gates: 145.000
  runtime: 3.161
  客观得分: 336.476
[Submit]
  Fidelity: 0.959
  Accuracy: 0.420
  振幅编码线路门的个数: 145.0
  运行时间: 1.098531723022461
  客观得分: 336.5330392784543

[Trail 4] qam_flatten (overfit!)
[Local]
  classifier gate count: 1452 (F2_all nlayer=10)
  test fid: 0.959
  test acc: 0.424
  test gates: 145.000
  runtime: 3.118
  客观得分: 336.877
-----------------------
  classifier gate count: 1772  (qcnn nlayer=8)
  test fid: 0.959
  test acc: 0.430
  test gates: 145.000
  runtime: 4.231
  客观得分: 337.446
-----------------------
  classifier gate count: 5292 (real qcnn nlayer=24)
  test fid: 0.959
  test acc: 0.444
  test gates: 145.000
  runtime: 9.898
  客观得分: 338.689
[Submit]
  classifier gate count: 1452 (F2_all nlayer=10)
  Fidelity: 0.959
  Accuracy: 0.424
  振幅编码线路门的个数: 145.0
  运行时间: 1.0803706645965576
  客观得分: 336.9335445629226
-----------------------
  xxx
-----------------------
  classifier gate count: 5292
  Fidelity: 0.959
  Accuracy: 0.444
  振幅编码线路门的个数: 145.0
  运行时间: 3.3697571754455566
  客观得分: 338.86995156606037

[Trail 5] no_data_norm + std_flatten (overfit!)
enc:
  | encoder | n_layer | gate count | fidelity | score | comment |
  | vqc_F2_all_wise_init_0 | 1 | 145     | 0.959 | 2.8455   | no_data_norm, std_flatten, n_iter=500 |
  | vqc_F2_all_wise_init_0 | 1 | 145     | 0.966 | 2.8595   | no_data_norm, std_flatten, n_iter=500 |
  | vqc_F2_all_wise_init_0 | 1 | 101.446 | 0.961 | 2.871277 | no_data_norm, std_flatten, n_iter=400(use_finetune=3:1) |
clf:
  | vqc | acc |
  | qcnn     (nlayer=8)  | 42.8% |
  | F2_all_0 (nlayer=10) | 34.0% |
  | U-V brick (nlayer=8) | 43.4% |
[Local]
  classifier gate count: 1772
  test fid: 0.966
  test acc: 0.428
  test gates: 145.000
  runtime: 3.512
  客观得分: 338.730
-----------------------
  classifier gate count: 1772
  test fid: 0.961
  test acc: 0.428
  test gates: 101.446
  runtime: 3.433
  客观得分: 339.793
-----------------------
  classifier gate count: 1224
  test fid: 0.961
  test acc: 0.434
  test gates: 101.446
  runtime: 2.380
  客观得分: 340.422
[Submit]
  xxx
-----------------------
  classifier gate count: 1772
  Fidelity: 0.961
  Accuracy: 0.428
  振幅编码线路门的个数: 101.446
  运行时间: 1.1226468086242676
  客观得分: 339.8572680920283
-----------------------
  classifier gate count: 1224
  Fidelity: 0.961
  Accuracy: 0.434
  振幅编码线路门的个数: 101.446
  运行时间: 0.7958643436431885
  客观得分: 340.46634361842473
```


### 关于数据规范化の分析

$$ \text{Use} \ \left| x \right> = \frac{x}{\lvert| x \rvert|} \ \text{or} \ \left| x \right> = \frac{x - \mu}{\lvert| x - \mu \rvert|} \ \text{?} $$

|     | 非规范化数据 | 规范化数据 | comment |
| :-: | :-: | :-: | :-: |
| 分布 | 不对称            | 比较对称，均值0 | |
| 符号 | 恒正，无需学习相位 | 有正有负，需要学习相位 | |
| enc  | fid=0.954         | fid=0.70    | vqc_F1_all_wise_init_0(d=2) |
| clf  | acc=~42%          | acc=46.667% | baseline |

规范化数据の内积，`sqrt(保真度/余弦相似度)`:

$$
\left< x_2 | x_1 \right> \
= \frac{x_2 - \mu}{\lvert| x_2 - \mu \rvert|} \cdot \frac{x_1 - \mu}{\lvert| x_1 - \mu \rvert|}
$$


### Appendix

⚪ Classical classifiers baselines

| model | param_cnt | accuracy | comment |
| :-: | :-: | :-: | :-: |
| vgg11            | 128786821 | 94.0% | 有预训练权重 |
| resnet18         |  11171397 | 91.0% | 有预训练权重 |
| mbnetv3_s        |   1522981 | 86.6% | 有预训练权重 |
| mbnetv3_s        |   1522981 | 80.6% | 无预训练权重，天花板 |
| cnn              |      1712 | 71.0% | 复杂结构可以涨分，但可见天花板 |
| cnn_d3           |       497 | 60.2% | 更厚的特征图，收益不大 |
| cnn_d1           |       133 | 58.6% | 卷积模型参考标准 ⭐ |
| cnn_d1_L         |       133 | 50.2% | 激活函数重要 |
| cnn_d1_s2        |       133 | 58.4% | AvgPool可忽略 |
| cnn_d1_s2_nb     |       125 | 52.4% | bias 重要 |
| cnn_d1_s2_x16    |        83 | 48.4% | 更小的特征图确实更差 |
| cnn_d1_s2_x16_L  |        74 | 46.2% | 图太小时，激活函数可忽略 |
| cnn_d1_s2_x16_nb |        74 | 45.6% | 极端压缩，仍然高于基线 QNN |
| cnn_nano         |        44 | 35.4% | 底线，不应该比这个差 |
| mlp0             |         - | 53.2%/67.6% | 模拟纯 ansatz 方法! 😈 |
| mlp1             |     15365 | 57.2% | 线性模型参考标准 ⭐ |
| mlp1_nb          |     15365 | 56.6% | wtm 直接线路合成??! 参考 mlp0 方法 |
| mlp2             |    787973 | 64.2% | 激活函数重要 |
| mlp2_drop        |    787973 | 63.2% | p=0.5 |
| mlp3             |   3410437 | 62.8% | 过拟合了 |
| knn              |           | 64.9% | k=5, p=1 |
| knn1             |           | 43.6%/45.2% | 暗示数据分布呈多中心化 |
| rf               |           | 65.5% | |

结论:

- 无偏置无激活的极简卷积模型 `cnn_d1_s2_nb` 精度为 `52.4%`，精度稍欠但参数量确实少
- 无偏置无激活的极简线性模型 `mlp1_nb` 精度为 `56.6%`
  - 模拟纯 ansatz 方法 `mlp0` 精度为 `53.2%`，任何单纯 ansatz 方法不应突破这个数字...
  - 在配置 `data_norm + std_flatten` 下可以达到精度 `57.8%`!! 我们真的还需要 qam_flatten 吗??
- **卷积模型不如线性模型那样容易在 circuit 上实现**

⚪ 理想模拟结果

| settings | amp_enc fid/gcnt actual | `mlp0` acc expected | maximun score expected |
| :-: | :-: | :-: | :-: |
|    data_norm + std_flatten |           | 57.8% |  |
|    data_norm + qam_flatten |           | 57.6% |  |
| no_data_norm + std_flatten | 0.966/145 | 53.2% | 3.3915 |
| no_data_norm + qam_flatten | 0.959/145 | 52.8% | 3.3735 |

讨论:

- flatten 方式不太改变线性模型的精度，甚至我们根本不需要 qam_flatten 呜呜呜。。。 :(
- 但 data_norm 确实很影响编码保真度


#### reference

- (Quanvolution) https://github.com/anthonysmaldone/qcnn-multi-channel-supervised-learning
- (Hybrid) https://github.com/DRA-chaos/Quantum-Classical-Hyrid-Neural-Network-for-binary-image-classification-using-PyTorch-Qiskit-pipeline
- Qiskit
  - qiskit-machine-learning: https://github.com/qiskit-community/qiskit-machine-learning
    - doc: https://qiskit-community.github.io/qiskit-machine-learning/index.html
  - circuit synthesis: https://quantumcomputing.stackexchange.com/questions/13821/generate-a-3-qubit-swap-unitary-in-terms-of-elementary-gates/13826#13826
