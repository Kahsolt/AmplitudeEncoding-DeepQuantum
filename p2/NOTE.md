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
```

⚪ AmpEnc results

ℹ 一般而言，需达到 `fid=0.95` 左右，可视化为图像才看起来有人类可识别的同一性；另可参考 [wtf-quantum-fidelity](https://github.com/Kahsolt/wtf-quantum-fidelity)  
⚠ 须注意：编码保真度与分类精度的相关性很弱，即 enc 和 clf 两者并不解耦 (因为振幅编码仅保留了数据的余弦相似度信息，可以说局域性信息已破坏殆尽)；因此必须先确定好编码方案，才能着手研究分类器！  

> enc_score = 2 * fid + (1 - gcnt / 2000)

| enc(n_layer) | gcnt | fid | enc_score | comment |
| :-: | :-: | :-: | :-: | :-: |
| F1(1)   |  79     | 0.930 | 2.8205   | no_norm, n_iter=200 |
| F1(2)   | 157     | 0.954 | 2.8295   | no_norm, n_iter=200 |
| F1(3)   | 235     | 0.965 | 2.8125   | no_norm, n_iter=200 |
| F2(1)   | 145     | 0.966 | 2.8595   | no_norm, n_iter=500 |
| F2(3)   | 236.164 | 0.985 | 2.851918 | no_norm, n_iter=800(use_finetune=3:1) |
| F2(1)   | 101.446 | 0.961 | 2.871277 | no_norm, n_iter=400(use_finetune=3:1) (⭐) |
| F2(1)   | 145     | 0.846 | 2.6195   |    norm, n_iter=500 |
| F2(2)   | 289     | 0.919 | 2.6935   |    norm, n_iter=500 |
| F2(1)   | 116.982 | 0.849 | 2.639509 |    norm, n_iter=400(use_finetune=3:1) |
| F2(2)   | 200.908 | 0.921 | 2.741546 |    norm, n_iter=400(use_finetune=3:1) |
| F2(3)   | 286.830 | 0.947 | 2.750585 |    norm, n_iter=400(use_finetune=3:1) |
| F2(3)   | 287.730 | 0.951 | 2.758135 |    norm, n_iter=800(use_finetune=3:1) (⭐) |
| F2(2/3) | 248.210 | 0.944 | 2.763895 |    norm, n_iter=800(use_finetune=3:1); score:0.576↑, fid:0.007↓ |
| F2(2/3) | 241.652 | 0.944 | 2.767174 |    norm, n_iter=800(use_finetune=3:1); remove |0>-ctrl |
| F2(3)   | 292.868 | 0.949 | 2.750686 |    norm, n_iter=800(use_finetune=3:1), +mse_loss |

⚪ clf results

> total_score = (2 * fid + (1 - gcnt / 2000) + acc + 0.1) * 100

**with data_norm:**

| enc ckpt | clf(n_layer) | gcnt/pcnt | acc | ~total_score | comment |
| :-: | :-: | :-: | :-: | :-: | :-: |
| F2(3) [score:2.758135] | U-V brick(3)  [Ansatz]     |  504/ 846 | 45.2% | 331.0135 | |
| F2(3) [score:2.758135] | U-V all(1)    [AnsatzExt]  | 1285/1671 | 48.4% | 334.2135 | |
| F2(3) [score:2.758135] | U-V brick(1)  [CL]         |  216/ 450 | 40.8% | 326.6135 | |
| F2(3) [score:2.758135] | U-V brick(2)  [CL]         |  360/ 648 | 48.2% | 334.0135 | |
| F2(3) [score:2.758135] | U-V brick(3)  [CL]         |  504/ 846 | 52.8% | 338.6135 | 修复 ref_data 算法，仍有7.4%的样本无法从理想训练集泛化来 (⭐) |
| F2(3) [score:2.758135] | U-V brick(4)  [CL]         |  648/1044 | 50.0% | 335.8135 | |
| F2(3) [score:2.758135] | U-V brick(6)  [CL]         |  936/1440 | 49.2% | 335.0135 | |
| F2(3) [score:2.758135] | U-V brick(3)  [CLCascade]  |     -     | 56.2% | 342.0135 | 提升没有想象中的大。。。 |
| F2(3) [score:2.758135] | U-V brick(3)  [CLEnsemble] |     -     | 60.2% | 346.0135 | 万物皆可集成 |
| F2(3) [score:2.758135] | U-V brick(3)  [CLMLP]      |  504/1175 | 63.4% | 349.2135 | 逼近经典模型 mlp2 |
| F2(3) [score:2.750686] | U-V brick(4)  [CL]         |  648/1044 | 51.4% | 336.4686 | +mse_loss 对分类器友好一点 |
| F2(3) [score:2.750686] | U-V brick(3)  [CLMLP]      |  504/1175 | 65.2% | 350.2686 | +mse_loss 对分类器友好一点 |

**without data_norm:**

| enc ckpt | clf(n_layer) | gcnt/pcnt | acc | ~total_score | comment |
| :-: | :-: | :-: | :-: | :-: | :-: |
| F2(1) [score:2.871277] | U-V brick(8)  [Ansatz] | 1224/1656 | 44.4% | 341.5277 | overfit; 纯 Ansatz 方法很差，无法逼近理论 mlp0 |
| F2(1) [score:2.871277] | U-V brick(3)  [CL]     |  504/ 846 | 44.0% | 341.1277 | 我们认为最合理而并不作弊的解决方案 |
| F2(1) [score:2.871277] | U-V brick(10) [CL]     | 1512/2232 | 56.2% | 353.2100 | overfit |
| F2(1) [score:2.871277] | U-V brick(10) [CLMLP]  | 1512/2561 | 60.6% | 357.7277 | |
| F2(1) [score:2.871277] | U-V brick(10) [CLMLP]  | 1512/2561 | 85.0% | 382.1277 | overfit; loss/acc 并未完全收敛 (most cheaty! 😈) |


### 关于数据规范化の分析

如果遵守这个游戏规则，编码问题将非常困难... 😈

$$ \text{Use} \ \left| x \right> = \frac{x}{\lvert| x \rvert|} \ \text{or} \ \left| x \right> = \frac{x - \mu}{\lvert| x - \mu \rvert|} \ \text{?} $$

|     | 非规范化数据 | 规范化数据 |
| :-: | :-: | :-: |
| 分布 | 不对称，对分类器不利 | 比较对称，均值0；利好分类器 |
| 符号 | 恒正，无需学习相位   | 有正有负，需要学习相位 |
| enc(best) | score=2.871277 | score=2.750585 |

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
  - 在配置 `data_norm` 下可以达到精度 `57.8%`!! 我们真的还需要 qam_flatten 吗??
- **卷积模型不如线性模型那样容易在 circuit 上实现**


#### reference

- (Quanvolution) https://github.com/anthonysmaldone/qcnn-multi-channel-supervised-learning
- (Hybrid) https://github.com/DRA-chaos/Quantum-Classical-Hyrid-Neural-Network-for-binary-image-classification-using-PyTorch-Qiskit-pipeline
- Qiskit
  - qiskit-machine-learning: https://github.com/qiskit-community/qiskit-machine-learning
    - doc: https://qiskit-community.github.io/qiskit-machine-learning/index.html
  - circuit synthesis: https://quantumcomputing.stackexchange.com/questions/13821/generate-a-3-qubit-swap-unitary-in-terms-of-elementary-gates/13826#13826
