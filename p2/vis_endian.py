#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/24 

# 测定位序: deepquantum 使用小端序还是大端序？
# wtf: 大端序！反人类了。。。

import deepquantum as dq

# tensor([0.9888, 0.0000, 0.0000, 0.0000, 0.1494, 0.0000, 0.0000, 0.0000])
qc = dq.QubitCircuit(nqubit=3)
qc.ry(wires=0, inputs=0.3)
state = qc()
print(state.real.flatten())

# tensor([0.9888, 0.1494, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
qc = dq.QubitCircuit(nqubit=3)
qc.ry(wires=2, inputs=0.3)
state = qc()
print(state.real.flatten())
