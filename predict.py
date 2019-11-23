# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset
from model import Model
import pandas as pd


data = Dataset()
model = Model(data)

p = model.predict(TARGET='IphoneSE', TEXT='已经等不及想要去看看这款手机了')
print(p)