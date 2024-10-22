#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/21 

# 跑统计学习的经典基线

import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC, LinearSVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from utils import CIFAR10Dataset, reshape_norm_padding

# 结果和 Euclidean 距离完全一致，真的假的？
def cosdist(x, y):
  return 1 - np.square(np.sum(x * y))

model_getters = [
  LogisticRegression,                                 # prec: 0.584
  lambda: SVC(max_iter=500),                          # prec: 0.586
  lambda: LinearSVC(max_iter=500),                    # prec: 0.594
  lambda: KNeighborsClassifier(n_neighbors=1),        # prec: 0.630   <- try Hadamard Test!
  lambda: KNeighborsClassifier(n_neighbors=3),        # prec: 0.587
  lambda: KNeighborsClassifier(n_neighbors=5),        # prec: 0.604
  lambda: KNeighborsClassifier(n_neighbors=7),        # prec: 0.626
  lambda: KNeighborsClassifier(n_neighbors=1, p=1),   # prec: 0.611
  lambda: KNeighborsClassifier(n_neighbors=3, p=1),   # prec: 0.617
  lambda: KNeighborsClassifier(n_neighbors=5, p=1),   # prec: 0.649   <- best
  lambda: KNeighborsClassifier(n_neighbors=7, p=1),   # prec: 0.647
  RandomForestClassifier,                             # prec: 0.655   <- best
]

trainset = CIFAR10Dataset(train=True)
testset = CIFAR10Dataset(train=False)
X_train = np.stack([reshape_norm_padding(e).real.flatten().numpy() for e, _ in trainset.sub_dataset], axis=0)    # [C, H, W] => [CHW]
y_train = np.stack([e for _, e in trainset.sub_dataset], axis=0)
X_test  = np.stack([reshape_norm_padding(e).real.flatten().numpy() for e, _ in testset.sub_dataset],  axis=0)
y_test  = np.stack([e for _, e in testset.sub_dataset],  axis=0)
print('X_train.shape:', X_train.shape)
print('Y_train.shape:', y_train.shape)
print('X_test.shape:',  X_test.shape)
print('Y_test.shape:',  y_test.shape)

'''
[LogisticRegression]
  prec: 0.5840284564170702
  recall: 0.5820000000000001
  f1: 0.5811000797928716
conf matrix:
  [[67  4  9 12  8]
  [13 51 12  8 16]
  [12  9 57 17  5]
  [14  7 17 54  8]
  [14 13  1 10 62]]
[SVC]
  prec: 0.5863198234236328
  recall: 0.538
  f1: 0.5283536581433116
conf matrix:
  [[48  1 10  9 32]
  [ 3 30 11  2 54]
  [ 4  2 61 17 16]
  [10  5 24 44 17]
  [ 3 10  1  0 86]]
[LinearSVC]
  prec: 0.5946235246235246
  recall: 0.592
  f1: 0.5912388548348868
conf matrix:
  [[70  4 10 10  6]
  [13 54 10  9 14]
  [10  8 54 21  7]
  [12 11 15 58  4]
  [15 14  1 10 60]]

[KNeighborsClassifier (k=1)]
  prec: 0.6304464334754977
  recall: 0.5799999999999998
  f1: 0.5665172135624622
conf matrix:
[[78  6  2  5  9]
 [13 60  2  0 25]
 [35  6 38  9 12]
 [38  5  9 32 16]
 [10  6  1  1 82]]
[KNeighborsClassifier (k=3)]
  prec: 0.5869494008422653
  recall: 0.514
  f1: 0.48948102454491166
conf matrix:
  [[82  5  1  4  8]
  [23 49  1  0 27]
  [47  7 25 10 11]
  [50  8  9 23 10]
  [17  5  0  0 78]]
[KNeighborsClassifier (k=5)]
  prec: 0.6046130494574216
  recall: 0.532
  f1: 0.5065692625834564
conf matrix:
  [[83  5  0  3  9]
  [24 53  1  0 22]
  [48  4 26 10 12]
  [44  9 10 23 14]
  [15  4  0  0 81]]
[KNeighborsClassifier (k=7)]
  prec: 0.6204602358897666
  recall: 0.526
  f1: 0.5059710368266046
conf matrix:
  [[84  3  0  2 11]
  [23 45  2  1 29]
  [45  5 31  8 11]
  [48  5  6 25 16]
  [15  6  0  1 78]]

[KNeighborsClassifier (k=1, p=1)]
  prec: 0.6106769374416434
  recall: 0.584
  f1: 0.5728933884963252
conf matrix:
  [[76  5  3  6 10]
  [11 58  2  2 27]
  [27  7 39 13 14]
  [32  7  9 39 13]
  [ 6  8  3  3 80]]
[KNeighborsClassifier (k=3, p=1)]
  prec: 0.6167734104839179
  recall: 0.556
  f1: 0.5417535820822856
conf matrix:
  [[81  7  1  3  8]
  [23 49  3  1 24]
  [37  8 34 13  8]
  [44 10  5 33  8]
  [11  7  0  1 81]]
[KNeighborsClassifier (k=5, p=1)]
  prec: 0.6489338578432055
  recall: 0.594
  f1: 0.5836738822724336
conf matrix:
  [[82  6  1  4  7]
  [20 58  1  0 21]
  [36  4 45  9  6]
  [39  8 10 31 12]
  [11  6  0  2 81]]
[KNeighborsClassifier (k=7, p=1)]
  prec: 0.6466483951209426
  recall: 0.5740000000000001
  f1: 0.5656000457168912
conf matrix:
  [[81  7  1  4  7]
  [22 55  1  0 22]
  [38  3 42  9  8]
  [43  8  6 31 12]
  [13  8  0  1 78]]

[RandomForestClassifier]
  prec: 0.6553472483812945
  recall: 0.6499999999999999
  f1: 0.6514997685820155
conf matrix:
  [[69  7  9 10  5]
  [ 8 65 11  7  9]
  [ 9 10 63 17  1]
  [ 5  6 20 63  6]
  [ 5 17  3 10 65]]
'''
for get_model in model_getters:
  model = get_model()
  name = model.__class__.__name__
  print(f'[{name}]')

  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  prec, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
  print('  prec:', prec)
  print('  recall:', recall)
  print('  f1:', f1)
  cmat = confusion_matrix(y_test, y_pred)
  print('conf matrix:')
  print(cmat)
