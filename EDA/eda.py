import sys
import os
import jieba
from collections import Counter
import copy
import re
import numpy as np

os.chdir(sys.path[0])
import pandas as pd


def replacement_order(predict_dir, predict_order_dir, taskaa_dir, predict_order_different_dir):
    """
    :param predict_dir:
    :param predict_order_dir:
    :return:
    """
    topics = ['IphoneSE', '春节放鞭炮', '深圳禁摩限电', '俄罗斯在叙利亚的反恐行动', '开放二胎']
    data = pd.read_csv(taskaa_dir, sep='\t')
    taskaa = None
    for topic in ['IphoneSE', '春节放鞭炮', '深圳禁摩限电', '俄罗斯在叙利亚的反恐行动', '开放二胎']:
        _data = data[data['TARGET'] == topic]
        _data['LENGTH'] = _data['TEXT'].apply(lambda x: len(x))
        _data = _data.sort_values(by='LENGTH', ascending=True)
        stance = list(_data["STANCE"])
        target = list(_data['TARGET'])
        text = _data['TEXT'].values.tolist()
        if taskaa is None:
            taskaa = pd.DataFrame(data={"stance": stance, "target": target, "text": text})
        else:
            taskaa = pd.concat([taskaa, pd.DataFrame(data={"stance": stance, "target": target, "text": text})])
    stance = []
    target = []
    text = []
    text_1 = []
    taskaa = dict(zip(list(taskaa['text']), list(taskaa['stance'])))
    with open(predict_dir, 'r') as file_1:
        for line in file_1.readlines():
            stance.append(line.split(',        ')[0].strip().lstrip('[').rstrip(']'))
            target.append(line.split(',        ')[1].strip())
            text.append(line.split(',        ')[2].strip())
            text_1.append(line.split(',        ')[3].strip())

        data = pd.DataFrame(data={'stance': stance, 'target': target, 'text': text, 'text_1': text_1})

        result = None
        for topic in topics:
            current_data = data[data['target'] == topic]
            current_data['length'] = current_data['text_1'].apply(lambda x: len(x))
            current_data.sort_values(by='length', inplace=True)
            if result is not None:
                print(result.shape)
            current_data['true'] = current_data['text'].apply(lambda x: taskaa[x])
            if result is None:
                result = current_data[['true', 'stance', 'target', "text"]]
            else:
                result = pd.concat([result, current_data[['true', 'stance', 'target', "text"]]])

        result.to_csv(path_or_buf=predict_order_dir, sep='\t', index=None)

        result_different = result[result['true'] != result['stance']]
        result_different.to_csv(path_or_buf=predict_order_different_dir, sep='\t', index=None)
        print('different_length: {}'.format(result_different.shape[0]))


def calculate_word_count(train_dir):
    topics = ['IphoneSE', '春节放鞭炮', '深圳禁摩限电', '俄罗斯在叙利亚的反恐行动', '开放二胎']
    with open(train_dir, 'r') as file_1:
        for topic in topics:
            print('topic: {}'.format(topic))
            none_label = None
            favor_label = None
            against_label = None

            for line in file_1.readlines():
                if line.split(',        ')[1].strip().lower() != topic.lower():
                    continue
                label = line.split(',        ')[0].strip().lstrip('[').rstrip(']')
                count = Counter(list(
                    jieba.cut(re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5]", '', line.split(',        ')[2].strip()),
                              cut_all=False)))
                if label == 'AGAINST':
                    if against_label is None:
                        against_label = copy.deepcopy(count)
                    else:
                        against_label.update(count)

                elif label == 'FAVOR':
                    if favor_label is None:
                        favor_label = copy.deepcopy(count)
                    else:
                        favor_label.update(count)

                else:
                    if none_label is None:
                        none_label = copy.deepcopy(count)
                    else:
                        none_label.update(count)

            all_label = copy.deepcopy(none_label)
            all_label.update(favor_label)
            all_label.update(against_label)
            all_label_most = all_label.most_common(1000)
            result = pd.DataFrame(data={"word": [i for (i, j) in all_label_most]})
            result["none"] = [none_label.get(i) for (i, j) in all_label_most]
            result["favor"] = [favor_label.get(i) for (i, j) in all_label_most]
            result["against"] = [against_label.get(i) for (i, j) in all_label_most]
            result.fillna(value=0, inplace=True)

            result['freq'] = [j for (i, j) in all_label_most]
            result['support'] = np.nan_to_num(
                [max(none, favor, against) / (1.0 * (none + favor + against)) for (none, favor, against) in
                 zip(result['none'], result['favor'], result['against'])])

            result.sort_values(by='support', inplace=True, ascending=False)
            result.to_csv(path_or_buf='word_count' + topic + '.csv', index=None)


if __name__ == '__main__':
    taskaa_dir = 'evasampledata4-TaskAA.txt'
    predict_dir = 'predict.txt'
    predict_order_dir = 'predict_order.txt'
    predict_order_different_dir = 'predict_order_different.txt'

    train_dir = 'train.txt'
    word_count_dir = 'word_count.csv'


    # replacement_order(predict_dir=predict_dir, predict_order_dir=predict_order_dir, taskaa_dir=taskaa_dir,
    #                   predict_order_different_dir=predict_order_different_dir)

    calculate_word_count(train_dir=train_dir)
