# -*- coding: utf-8 -*
import os
import torch
import args

from flyai.model.base import Base

__import__('net', fromlist=["Net"])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(Base):
    def __init__(self, data):
        self.net = None
        self.data = data
        self.model_dir = os.path.join(os.getcwd(), args.model_dir)

    def predict(self, **data):
        if self.net is None:
            self.net = torch.load(self.model_dir)
        TARGET, TEXT = self.data.predict_data(**data)


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
