#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/30 

# 查看reshape之后的样本

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from utils import *

datatset = QMNISTDatasetIdea(label_list=[0,1,2,3,4], train=False, per_cls_size=51)
z_imgs = []
for x, _, _ in tqdm(datatset):
  z = reshape_norm_padding(x.unsqueeze(0))
  z_im = img_to_01(z.real.reshape(-1, 32, 32)).permute([1, 2, 0])
  z_imgs.append(z_im)
z_imgs = torch.stack(z_imgs).permute([0, 3, 1, 2])
print('z_imgs.shape:', z_imgs.shape)
Z = make_grid(z_imgs, nrow=16)
print('Z.shape:', Z.shape)

plt.imshow(Z.permute(1, 2, 0).numpy())
plt.suptitle('vis_snake_bulk')
plt.show()
