# -*- coding: utf-8 -*
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, input):
        return