#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/07/23 

# 探究 向量量化(qt) 和 位序重排(reshape) 对保真度的影响
# - qt=1, fid=0.9274693276891914
# (gate count) gamma=0.01: 819.3545436855419/782.9462930531232
# (gate count) gamma=0.016: 774.609846273594/767.3010313290523
# - qt=2, fid=0.9774327386003172
# - qt=3, fid=0.9890995125497772
# - qt=4, fid=0.9937793612364179
# 结论:
# - 量化只能降低 ~10 这个量级数量的门 (效果不好，代码实现已删除)
# - 位序重排可以把门数量降到 500~600，但保真度会降到 0.9

'''
stats of 50 samples:

⚪ rev=True

[GAMMA=0.02 QT=4]
qt approx fid: 0.994049254655838
vqc fidelity: 0.936056500673294
vqc fidelity(qt): 0.9342165088653565
vqc gate_count: 528.96
vqc gate_count(qt): 532.74

[GAMMA=0.022 QT=4]
qt approx fid: 0.994049254655838
vqc fidelity: 0.9102801764011383
vqc fidelity(qt): 0.9046344113349915
vqc gate_count: 503.14
vqc gate_count(qt): 501.36

[GAMMA=0.022 QT=2]
qt approx fid: 0.9789317691326141
vqc fidelity: 0.9102801764011383
vqc fidelity(qt): 0.8975578606128692
vqc gate_count: 503.14
vqc gate_count(qt): 505.1

[GAMMA=0.022 QT=16]
qt approx fid: 0.9995727849006653
vqc fidelity: 0.9102801764011383
vqc fidelity(qt): 0.9091472232341766
vqc gate_count: 503.14
vqc gate_count(qt): 501.22

[GAMMA=0.022 QT=None]                 <- best
qt approx fid: 1.0000008344650269
vqc fidelity: 0.9102801764011383
vqc fidelity(qt): 0.9102801764011383
vqc gate_count: 503.14
vqc gate_count(qt): 503.14

[GAMMA=0.022 QT=None train_step=100]
vqc fidelity: 0.9137811291217804 (+0.03)
vqc gate_count: 503.14

[GAMMA=0.024 QT=None]
qt approx fid: 1.0000008344650269
vqc fidelity: 0.8631777346134186
vqc fidelity(qt): 0.8631777346134186
vqc gate_count: 463.5
vqc gate_count(qt): 463.5

[GAMMA=0.024 QT=4]
qt approx fid: 0.994049254655838
vqc fidelity: 0.8631777346134186
vqc fidelity(qt): 0.8664456617832184
vqc gate_count: 463.5
vqc gate_count(qt): 469.14

⚪ rev=False (fid is low!!)

[GAMMA=0.022 QT=4]
qt approx fid: 0.9940487551689148
vqc fidelity: 0.791480575799942
vqc fidelity(qt): 0.7896462714672089
vqc gate_count: 450.78
vqc gate_count(qt): 451.14

[GAMMA=0.024 QT=4]
qt approx fid: 0.9940487551689148
vqc fidelity: 0.7279720222949981
vqc fidelity(qt): 0.7184738558530808
vqc gate_count: 408.24
vqc gate_count(qt): 406.46
'''

from time import time

from utils import *
from amp_enc import *
import matplotlib.pyplot as plt

datatset = QMNISTDatasetIdea(label_list=[0,1,2,3,4], train=False, per_cls_size=10)

fid_list = []
z_enc_gc = []
z_enc_fid = []
for x, y, z_get in tqdm(datatset):
  # x: [1, 28, 28]
  # z = z_get()
  # z: [1, 1024] := reshape_norm_padding(x.unsqueeze(0))

  EPS = 0.001
  GAMMA = 0.022

  z = reshape_norm_padding(x.unsqueeze(0))   # truth
  z_vqc = amplitude_encode(z.flatten().numpy(), eps=EPS, gamma=GAMMA)

  if not 'train':
    N_ITER = 100

    s = time()
    optimizer = optim.Adam(z_vqc.parameters(), lr=0.02)
    for i in range(N_ITER):
      state = z_vqc().swapaxes(0, 1)     # [B=1, D=1024]
      loss = -get_fidelity(state, z)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if i % 10 == 0:
        print('fid:', -loss.item())
    t = time()
    state = z_vqc().swapaxes(0, 1)
    print(f'>> Fidelity:', get_fidelity(state, z).item(), f'(time: {t - s})')

  if 'vqc fid & gate cnt':
    z_vqc_fid = get_fidelity(z_vqc(), z).item()
    z_vqc_gc  = count_gates(z_vqc)
    print('z_vqc_fid:', z_vqc_fid)
    print('z_vqc_gc:',  z_vqc_gc)
    z_enc_fid.append(z_vqc_fid)
    z_enc_gc .append(z_vqc_gc)

  if 'plot':
    plt.subplot(131) ; plt.title('x')           ; plt.imshow(img_to_01(x)                                        .permute([1, 2, 0]).numpy())
    plt.subplot(132) ; plt.title('z_snake')     ; plt.imshow(img_to_01(z               .real.reshape(-1, 32, 32)).permute([1, 2, 0]).numpy())
    plt.subplot(133) ; plt.title('z_snake_vqc') ; plt.imshow(img_to_01(z_vqc().detach().real.reshape(-1, 32, 32)).permute([1, 2, 0]).numpy())
    plt.suptitle(f'({z_vqc_gc}; {z_vqc_fid})')
    plt.tight_layout()
    plt.show()

print('qt approx fid:', mean(fid_list))

print('vqc fidelity:',   mean(z_enc_fid))
print('vqc gate_count:', mean(z_enc_gc))
