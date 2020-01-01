import numpy as np
import pandas as pd


def replacement_order(predict_dir, predict_order_dir):
    target = []
    text = []
    topics = ['IphoneSE', '春节放鞭炮', '深圳禁摩限电', '俄罗斯在叙利亚的反恐行动', '开放二胎']
    with open(predict_dir, 'r') as file_1, open(predict_order_dir, 'w') as file_2:
        for line in file_1.readlines():
            target.append(line.split(',     ')[1])
            text.append(line.split(',     ')[2])

        target = np.asarray(target)
        text = np.asarray(text)

        for topic in topics:
            index = np.where(target == topic)
            for (target_one, text_one) in zip(target[index].tolist(), text[index].tolist()):
                file_2.write('\t' + target_one + '\t' + text_one)


if __name__ == '__main__':
    predict_dir = 'predict_1.txt'
    predict_order_dir = 'predict_order_1.txt'

    replacement_order(predict_dir, predict_order_dir)
