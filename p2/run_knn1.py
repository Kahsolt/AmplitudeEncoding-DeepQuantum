#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/11/10 

# 专题研究 KNN(1) 方法，能做到什么程度？
# 结论: 常规训-测精度 43.6%，过拟合到测试集可达 45.2%；这应该就是我目前为止设计的 ansatz 瓶颈所在

import pickle as pkl

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils import PerfectAmplitudeEncodingDataset
from utils import QCIFAR10Dataset   # keep for unpickle

overfit = True


with open('output/test_dataset.pkl', 'rb') as file:
  testset = pkl.load(file)
trainset = testset if overfit else PerfectAmplitudeEncodingDataset(train=True)   # 使用全部数据集训练，防止过拟合！
print('len(dataset):', len(trainset))
print('len(dataset):', len(testset))

# average trainset to 5 centroids
train_data: dict = {}
for _, y, x in trainset:
  x = x.real.flatten().numpy()
  y = y.item()
  if y not in train_data:
    train_data[y] = []
  train_data[y].append(x)
train_data = sorted([(y, np.stack(xs, axis=-1).mean(axis=-1)) for y, xs in train_data.items()])
X_train = np.stack([x for y, x in train_data], axis=0)
y_train = np.stack([y for y, x in train_data], axis=0)
print('X_train.shape:', X_train.shape)
print('Y_train.shape:', y_train.shape)
# leave testset as it is
X_test = np.stack([x.real.flatten().numpy() for _, _, x in testset], axis=0)
y_test = np.stack([y.item()                 for _, y, _ in testset], axis=0)
print('X_test.shape:', X_test.shape)
print('Y_test.shape:', y_test.shape)


model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('  acc:', acc)
prec, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
print('  prec:', prec)
print('  recall:', recall)
print('  f1:', f1)
cmat = confusion_matrix(y_test, y_pred)
print('conf matrix:')
print(cmat)
