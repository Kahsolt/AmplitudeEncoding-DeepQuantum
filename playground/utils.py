#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/08 

from pathlib import Path

import torch
import deepquantum as dq
import numpy as np

SEED = 114514
DATA_ROOT = '../data'

BASE_PATH = Path(__file__).parent
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)
DATA_PATH = BASE_PATH / 'data' ; DATA_PATH.mkdir(exist_ok=True)

mean = lambda x: sum(x) / len(x)


def set_seed():
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)


def get_gate_count(qc:dq.QubitCircuit) -> int:
  return len([op for op in qc.operators.modules() if isinstance(op, dq.operation.Operation)])
