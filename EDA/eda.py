import sys
import os
import jieba
from collections import Counter
import copy
import re
import numpy as np
import pandas as pd

os.chdir(sys.path[0])


def replacement_order(predict_dir, taskaa_dir, predict_order_different_dir):
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

    stance, none, favor, against, target, text, text_1 = [], [], [], [], [], [], []
    taskaa = dict(zip(list(taskaa['text']), list(taskaa['stance'])))
    with open(predict_dir, 'r') as file_1:
        for line in file_1.readlines():
            stance.append(line.split(',        ')[0].strip())
            none.append(line.split(',        ')[1].strip())
            favor.append(line.split(',        ')[2].strip())
            against.append(line.split(',        ')[3].strip())
            target.append(line.split(',        ')[4].strip())
            text.append(line.split(',        ')[5].strip())
            text_1.append(line.split(',        ')[6].strip())

        data = pd.DataFrame(
            data={'stance': stance, 'none': none, 'favor': favor, 'against': against, 'target': target, 'text': text,
                  'text_1': text_1})

        result = None
        for topic in topics:
            current_data = data[data['target'] == topic]
            current_data['length'] = current_data['text_1'].apply(lambda x: len(x))
            current_data.sort_values(by='length', inplace=True)
            if result is not None:
                print(result.shape)
            current_data['true'] = current_data['text'].apply(lambda x: taskaa[x])
            if result is None:
                result = current_data[['true', 'stance', 'none', 'favor', 'against', 'target', "text"]]
            else:
                result = pd.concat(
                    [result, current_data[['true', 'stance', 'none', 'favor', 'against', 'target', "text"]]])

        result_different = result[result['true'] != result['stance']]
        result_different.to_csv(path_or_buf=predict_order_different_dir, sep='\t', index=None)
        print('different_length: {}'.format(result_different.shape[0]))


def calculate_word_count(train_dir, word_count_dir):
    topics = ['IphoneSE', '春节放鞭炮', '深圳禁摩限电', '俄罗斯在叙利亚的反恐行动', '开放二胎']

    results = None
    for topic in topics:
        print('topic: {}'.format(topic))
        none_label = None
        favor_label = None
        against_label = None
        with open(train_dir, 'r') as file_1:
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
        result = result[result['freq'] >= 6]
        result['support'] = np.nan_to_num(
            [max(none, favor, against) / (1.0 * (none + favor + against)) for (none, favor, against) in
             zip(result['none'], result['favor'], result['against'])])

        result = result[result['support'] >= 0.6]
        result.sort_values(by='support', inplace=True, ascending=False)
        result['target'] = topic

        if results is None:
            results = result[['target', 'word', 'none', 'favor', 'against', 'freq', 'support']]
        else:
            results = pd.concat([results, result[['target', 'word', 'none', 'favor', 'against', 'freq', 'support']]])

    results.to_csv(path_or_buf=word_count_dir, index=None)


def predict_rate(predict_order_different_dir, word_count_dir, predict_order_different_ngram_dir):
    topics = ['IphoneSE', '春节放鞭炮', '深圳禁摩限电', '俄罗斯在叙利亚的反恐行动', '开放二胎']

    word_count = pd.read_csv(filepath_or_buffer=word_count_dir)
    predict_order_different = pd.read_csv(predict_order_different_dir, sep='\t')

    predict_order_different_ngram = None
    for topic in topics:
        current_predict = predict_order_different[predict_order_different['target'] == topic]
        current_predict['text'] = current_predict['text'].apply(lambda x: list(jieba.cut(x, cut_all=False)))
        current_word_count = word_count[word_count['target'] == topic]
        current_word_count_word = list(current_word_count['word'])
        current_word_count_none = dict(
            zip(current_word_count_word, list(current_word_count['none'] / current_word_count['freq'])))
        current_word_count_favor = dict(
            zip(current_word_count_word, list(current_word_count['favor'] / current_word_count['freq'])))
        current_word_count_against = dict(
            zip(current_word_count_word, list(current_word_count['against'] / current_word_count['freq'])))

        nones, favors, againsts = [], [], []
        for text in current_predict['text']:
            none, favor, against = 0, 0, 0
            for word in text:
                if word in current_word_count_none:
                    none += current_word_count_none[word]
                if word in current_word_count_favor:
                    favor += current_word_count_favor[word]
                if word in current_word_count_against:
                    against += current_word_count_against[word]
            nones.append(none)
            favors.append(favor)
            againsts.append(against)

        current_predict['none'] = nones
        current_predict['favor'] = favors
        current_predict['against'] = againsts

        if predict_order_different_ngram is None:
            predict_order_different_ngram = current_predict
        else:
            predict_order_different_ngram = pd.concat([predict_order_different_ngram, current_predict])

    true = 0
    false = 0
    predict_order_different_ngram['predict'] = predict_order_different_ngram[['none', 'favor', 'against']].idxmax(
        axis=1)

    predict_order_different_ngram.to_csv(path_or_buf=predict_order_different_ngram_dir, index=None)


if __name__ == '__main__':
    taskaa_dir = 'evasampledata4-TaskAA.txt'
    predict_dir = 'predict.txt'
    predict_order_different_dir = 'predict_order_different.txt'

    train_dir = 'train.txt'
    word_count_dir = 'word_count.csv'
    predict_order_different_ngram_dir = 'predict_order_different_ngram.csv'

    replacement_order(predict_dir=predict_dir, taskaa_dir=taskaa_dir,
                      predict_order_different_dir=predict_order_different_dir)

    calculate_word_count(train_dir=train_dir, word_count_dir=word_count_dir)

    predict_rate(predict_order_different_dir=predict_order_different_dir, word_count_dir=word_count_dir,
                 predict_order_different_ngram_dir=predict_order_different_ngram_dir)
