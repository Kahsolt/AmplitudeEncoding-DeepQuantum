#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/09 

# 制作每个数字的平均范例

import matplotlib.pyplot as plt

from utils import *

#datatset = QMNISTDatasetIdea(label_list=[0,1,2,3,4], train=True, per_cls_size=1000)   # Acc: 0.9164
datatset = QMNISTDatasetIdea(label_list=[0,1,2,3,4], train=False, per_cls_size=1000)

y_xsum: Dict[int, Tensor] = {}
y_xcnt: Dict[int, int] = {}
for x, y, z_func in tqdm(datatset):
  y = y.item()
  if y not in y_xsum:
    y_xsum[y] = x
    y_xcnt[y] = 1
  else:
    y_xsum[y] += x
    y_xcnt[y] += 1

for y, xsum in y_xsum.items():
  y_xsum[y] = xsum / y_xcnt[y]
  y_xsum[y] = y_xsum[y][0]
  #y_xsum[y] = np.where(y_xsum[y] > 0, 1.0, -1.0)  # binarize!


if 'save stats':
  plt.clf()
  plt.subplot(231) ; plt.imshow(y_xsum[0]) ; plt.title('0')
  plt.subplot(232) ; plt.imshow(y_xsum[1]) ; plt.title('1')
  plt.subplot(233) ; plt.imshow(y_xsum[2]) ; plt.title('2')
  plt.subplot(234) ; plt.imshow(y_xsum[3]) ; plt.title('3')
  plt.subplot(235) ; plt.imshow(y_xsum[4]) ; plt.title('4')
  plt.tight_layout()
  plt.savefig('./img/vis_canon.png')

  data = np.stack([y_xsum[i] for i in range(len(y_xsum))], axis=0)
  print('data.shape', data.shape)
  np.save('./img/canon.npy', data)


# esitimate Hadamard test accuracy
data = torch.from_numpy(data).unsqueeze(dim=1)  # [NC=5, C=1, H, W]
cmat = np.zeros([5, 5], dtype=np.int32)
ok, tot = 0, 0
for x, y, z_func in tqdm(datatset):
  y = y.item()
  val = reshape_norm_padding(x.unsqueeze(0))
  logits = []
  for r in data:
    ref = reshape_norm_padding(r.unsqueeze(0))
    logits.append(get_fidelity(val, ref).item())
  pred = np.argmax(logits)
  
  cmat[y, pred] += 1
  ok += pred == y
  tot += 1

  if not 'debug plot' and pred != y:
    plt.clf()
    plt.subplot(211) ; plt.imshow(x[0]) ; plt.title(f'pred={pred}, y={y}')
    plt.subplot(212) ; plt.plot(logits) ; plt.title('logits') ; plt.xticks(list(range(5)), [str(e) for e in range(5)])
    plt.show()

'''
>> Acc: 0.9239151585911656
>> ConfMat:
[[ 941    0    8   28    3]
 [   0 1095   30    9    1]
 [  31   57  850   51   43]
 [   4   26   43  923   14]
 [   8   29    4    2  939]]
'''
print('>> Acc:', ok / tot)
print('>> ConfMat:')
print(cmat)
