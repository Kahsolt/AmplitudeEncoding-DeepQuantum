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
[Local 1]
classifier gate count: 14400
test fid: 0.952
test acc: 0.420
test gates: 145.000
runtime: 15.296
客观得分: 334.770
[Submit 1]
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
[Local 2]
classifier gate count: 14400
test fid: 0.959
test acc: 0.420
test gates: 145.000
runtime: 15.562
客观得分: 336.131
```


### 关于数据规范化の分析

$$ \text{Use} \ \left| x \right> = \frac{x}{\lvert| x \rvert|} \ \text{or} \ \left| x \right> = \frac{x - \mu}{\lvert| x - \mu \rvert|} \ \text{?} $$

|     | 非规范化数据 | 规范化数据 | comment |
| :-: | :-: | :-: | :-: |
| 分布 | 不对称            | 比较对称，均值0 | |
| 符号 | 恒正，无需学习相位 | 有正有负，需要学习相位 | |
| enc  | fid=0.954         | fid=0.70    | vqc_F1_all_wise_init_0(d=2) |
| clf  | acc=?             | acc=46.667% | baseline |

规范化数据の内积，`sqrt(保真度/余弦相似度)`:

$$
\left< x_2 | x_1 \right> \
= \frac{x_2 - \mu}{\lvert| x_2 - \mu \rvert|} \cdot \frac{x_1 - \mu}{\lvert| x_1 - \mu \rvert|}
$$


### Appendix

⚪ Classical classifiers baselines

| model | accuracy |
| :-: | :-: |
| vgg11     | 94.0% |
| resnet18  | 91.0% |
| mbnetv3_s | 86.6% |
