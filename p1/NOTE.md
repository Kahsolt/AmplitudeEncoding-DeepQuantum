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
| 0.001 | 0.022 | 2 | 0.866 |  694.2 | no training (20 samples stats) |

- 构建 Ansantz 进行分类
  - Acc: 89% ~ 92%


### Submits

```
[当前排行榜]
393.09  <- 假的
361.61  <- 榜一
?       <- 铅笔芯奇 (:O
360.87
357.84  <- 我
322.90
```

```
[1st solution] (AmpEnc)
⚪ local
test fid: 0.971
test acc: 0.929
test gates: 774.610
runtime: 293.514
客观得分: 311.429
⚪ submit
Fidelity: 0.971
Accuracy: 0.929
振幅编码线路门的个数: 774.609846273594
运行时间: 287.7688663005829
客观得分: 311.5888760951294 最高分数: 400

[2nd solution] (VQC + snake + train(step=1000) n_layer=15 (cyclic) ckpt-14)
⚪ local
test fid: 0.930
test acc: 0.890
test gates: 301.000
runtime: 179.103
客观得分: 350.036
⚪ submit
Fidelity: 0.930
Accuracy: 0.890
振幅编码线路门的个数: 301.0
运行时间: 210.35138130187988
客观得分: 349.1681521839566 最高分数: 400

[3rd solution] (VQC + snake + train(step=1000) n_layer=14 ckpt-14)
⚪ local
test fid: 0.916
test acc: 0.889
test gates: 253.000
runtime: 140.611
客观得分: 352.919
⚪ local (overfit to testset)
test fid: 0.916
test acc: 0.913
test gates: 253.000
runtime: 137.816
客观得分: 355.332
⚪ submit
Fidelity: 0.916
Accuracy: 0.889
振幅编码线路门的个数: 253.0
运行时间: 172.98028898239136
客观得分: 352.02026591565874 最高分数: 400
⚪ submit (overfit to testset)
Fidelity: 0.916
Accuracy: 0.913
振幅编码线路门的个数: 253.0
运行时间: 174.07732892036438
客观得分: 354.3248820291625 最高分数: 400

[4th solution] (VQC + snake + train(step=1000) n_layer=12 ckpt-14)
⚪ local
test fid: 0.893
test acc: 0.910
test gates: 217.000
runtime: 115.154
客观得分: 354.698

[5th solution] (VQC n_layer=12 + snake + train(step=1000); mera-like [u3-enc-u3-dec]-enc-u3)
⚪ local
test fid: 0.916
test acc: 0.935
test gates: 253.000
runtime: 148.048
客观得分: 357.227

[6th solution] (VQC n_layer=12 + snake + train(step=1000); mera-like [u3-enc-u3-dec]-enc-u3); +prune_pkl
⚪ local (eps=1e-3)
test fid: 0.916
test acc: 0.935
test gates: 252.888
runtime: 141.942
客观得分: 357.408
⚪ local (eps=1e-2)
test fid: 0.916
test acc: 0.935
test gates: 251.846
runtime: 140.693
客观得分: 357.546
⚪ local (eps=1e-1)
test fid: 0.912
test acc: 0.935
test gates: 241.722
runtime: 135.361
客观得分: 357.958
⚪ submit (eps=1e-1)
classifier gate count: 352
Fidelity: 0.912
Accuracy: 0.935
振幅编码线路门的个数: 241.72193033664138
运行时间: 139.57696104049683
客观得分: 357.84134136653404 最高分数: 400
```

得分模板

```
> (0.91*2 + 0.92 + (1-200/1000) + (1-100/360)*0.1) * 100
361.22222222222
> (0.916*2 + 0.913 + (1-253/1000) + (1-174/360)*0.1) * 100
354.3248820291625

[AmpEnc + snake_flatten + train(step=100)] (使用50个样本进行预估)
> (0.91*2 + 0.929 + (1-500/1000) + (1-200/360)*0.1) * 100
329.34444444444
[VQC + train(step=1000)] (使用1个样本进行预估)
> (0.8838197588920593*2 + 0.929 + (1-271/1000) + (1-146/360)*0.1) * 100
348.50839622286
> (0.8564713001251221*2 + 0.929 + (1-241/1000) + (1-146/360)*0.1) * 100
346.03870446947
```

### references

- repo
  - https://github.com/bsiegelwax/784-Dimensional-Quantum-MNIST
  - https://github.com/TuomasLe/Extended-basis-encoding-and-amplitude-encoding-algorithms-for-Qiskit
  - https://github.com/Subham-wq/Amplitude-encoding
  - https://github.com/Quvance/complex_amplitude_encoding
  - https://github.com/bsiegelwax/Quantum-Classification-of-Amplitudes
  - https://github.com/esarell/QuantumEncoding
  - https://github.com/sophchoe/Continous-Variable-Quantum-MNIST-Classifiers
  - https://github.com/sophchoe/Hybrid-Quantum-Classical-MNIST-Classfication-Model
- blog
  - https://quantumcomputing.stackexchange.com/questions/33722/how-does-the-induction-step-in-the-grover-rudolph-scheme-to-prepare-superpositio
  - https://quantumcomputing.stackexchange.com/questions/12104/preparing-a-quantum-state-from-a-classical-probability-distribution
  - https://towardsdatascience.com/784-dimensional-quantum-mnist-f0adcf1a938c
