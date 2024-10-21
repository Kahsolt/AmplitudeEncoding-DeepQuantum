#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/21 

import torch
import torchvision.transforms as T
from utils import CIFAR10Dataset

trainset = CIFAR10Dataset(train=True, transform=T.ToTensor())
testset = CIFAR10Dataset(train=False, transform=T.ToTensor())

# len(trainset): 25000
# len(testset): 500
print('len(trainset):', len(trainset))
print('len(testset):', len(testset))

traindata = torch.stack([e for e, _ in trainset.sub_dataset], axis=0)
testdata  = torch.stack([e for e, _ in testset.sub_dataset],  axis=0)

# NOTE: train 和 test 统计量只有千分位的差异，约 1/255 即一个像素值单位
# 但和总体均值偏差 4.5(mean) 个和 1.6(std) 个像素值单位
# traindata.mean: tensor([0.4903, 0.4873, 0.4642])
# traindata.std:  tensor([0.2519, 0.2498, 0.2657])
# testdata.mean:  tensor([0.4928, 0.4919, 0.4674])
# testdata.std:   tensor([0.2515, 0.2528, 0.2695])
print('traindata.mean:', traindata.mean(axis=(0,2,3)))
print('traindata.std:',  traindata.std (axis=(0,2,3)))
print('testdata.mean:',  testdata.mean (axis=(0,2,3)))
print('testdata.std:',   testdata.std  (axis=(0,2,3)))
