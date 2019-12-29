# -*- coding: utf-8 -*
from flyai.processor.base import Base
import args


class Processor(Base):
    def __init__(self):
        self.label_map = dict((i, args.labels[i]) for i in range(len(args.labels)))

    def input_x(self, TARGET, TEXT):
        """
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        """
        return TARGET, TEXT

    def input_y(self, STANCE):
        """
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        """
        # if STANCE == 'NONE':
        #     return 0
        # elif STANCE == 'FAVOR':
        #     return 1
        # elif STANCE == 'AGAINST':
        #     return 2
        return STANCE

    def output_y(self, data):
        """
        验证时使用，把模型输出的y转为对应的结果
        """
        label = []
        for i in data:
            label.append(self.label_map[i])

        return label
