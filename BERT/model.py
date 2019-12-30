# -*- coding: utf-8 -*
import os
import torch
import jieba
from flyai.model.base import Base

from BERT.data_utils import ABSADataset
import args

__import__('net', fromlist=["Net"])


class Model(Base):
    def __init__(self, data):
        self.net = None
        self.data = data
        self.model_dir = os.path.join(os.getcwd(), args.best_model_path)

    def predict(self, **data):
        if self.net is None:
            self.net = torch.load(self.model_dir)
        TARGET, TEXT = self.data.predict_data(**data)
        indexes = " ".join(jieba.cut(TARGET[0], cut_all=False))
        questions = " ".join(jieba.cut(TEXT[0], cut_all=False))

        return None

    def predict_all(self, datas):
        """
        预测所有的数据
        :param datas:
        :return:
        """
        labels = []
        for data in datas:
            predicts = self.predict(TARGET=data['TARGET'], TEXT=data['TEXT'])

            labels.append(predicts)

        return labels
