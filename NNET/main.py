# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import argparse
from flyai.dataset import Dataset
from NNET.model import Model
from NNET.net import Net
from NNET.path import MODEL_PATH

# flyai库中的提供的数据处理方法
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)

# dataset.get_step() 获取数据的总迭代次数
for step in range(dataset.get_step()):
    x_train, y_train = dataset.next_train_batch()
    x_val, y_val = dataset.next_validation_batch()
    """
    实现自己的模型保存逻辑
    """
