### Score Metrics

决赛评判标准 (满分为 410 分)

1. 振幅编码后的量子态矢量与原数据的相似度 Fidelity，振幅编码线路的复杂度 (门的个数)
2. 量子神经网络的 Top-1 分类准确率 Accuracy
3. 客观得分 = (2 * Fidelity + Accuracy + 线路复杂度得分 + 0.1 * 运行时间得分) * 100
4. 线路复杂度得分 = 1 - 振幅编码线路门的个数 / 2000；要求振幅编码线路门的个数 <= 1000
5. 运行时间得分 = 1 - 运行时间 / 360秒；要求线路调用 test_model() 函数的运行时间 <= 360s


### Trails

```
[Baseline]
test fid: 0.348
test acc: 0.306
test gates: 1212.000
runtime: 111.535
客观得分: 146.514
```


### Appendix

⚪ Classical classifiers baselines

| model | accuracy |
| :-: | :-: |
| vgg11     | 94.0% |
| resnet18  | 91.0% |
| mbnetv3_s | 86.6% |
