### Score Metrics

初赛评判标准 (满分为 410 分)

1. 振幅编码后的量子态矢量与原数据的相似度 Fidelity，振幅编码线路的复杂度 (门的个数)
2. 量子神经网络的 Top-1 分类准确率 Accuracy
3. 客观得分 = (2 * Fidelity + Accuracy + 线路复杂度得分 + 0.1 * 运行时间得分) * 100
4. 线路复杂度得分 = 1 - 振幅编码线路门的个数 / 1000；要求振幅编码线路门的个数 <= 1000
5. 运行时间得分 = 1 - 运行时间 / 360秒；要求线路调用 test_model() 函数的运行时间 <= 360s


### Tasks

- 对每个样本，实现其 AmplitudeEncode 线路

ℹ 裸构造法的保真度就已经足够高了 (~0.984)，进一步训练优化的时间成本原小于收益；之后大概只考虑近似处理，降低门的数量

| eps | gamma | qt | fidelity | gate count | comment |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 0.001 | 0.01  | 4 | 0.984 | 839.850 | no training (100 samples stats) |
| 0.001 | 0.01  | 4 | 0.992 | 839.240 | train 10 steps (20 samples stats) |
| 0.001 | 0.016 | None | 0.972 | 789.35 | no training (20 samples stats) |
| 0.001 | 0.016 | 2 | 0.956 | 797.65 | no training (20 samples stats) |
| 0.001 | 0.018 | 2 | 0.939 | 766.85 | no training (20 samples stats) |
| 0.001 | 0.020 | 2 | 0.893 | 719.75 | no training (20 samples stats) |
| 0.001 | 0.022 | 2 | 0.866 | 694.2 | no training (20 samples stats) |

- 构建 Ansantz 进行分类


⚪ 分数排行榜

- 393.919 (假的)
- 354.017
- 344.470
- 343.471 (铅笔芯奇)

我的

```
test fid: 0.971
test acc: 0.929
test gates: 774.610
runtime: 293.514
客观得分: 311.429
```
