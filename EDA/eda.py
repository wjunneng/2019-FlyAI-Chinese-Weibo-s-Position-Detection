import sys
import os

os.chdir(sys.path[0])
import pandas as pd


def taskaa_order(taskaa_dir, taskaa_order_dir):
    data = pd.read_csv(taskaa_dir, sep='\t')
    result = None
    for topic in ['IphoneSE', '春节放鞭炮', '深圳禁摩限电', '俄罗斯在叙利亚的反恐行动', '开放二胎']:
        _data = data[data['TARGET'] == topic]
        _data['LENGTH'] = _data['TEXT'].apply(lambda x: len(x))
        _data = _data.sort_values(by='LENGTH', ascending=True)
        stance = list(_data["STANCE"])
        target = list(_data['TARGET'])
        text = _data['TEXT'].values.tolist()

        # text = PreProcessing(text).get_file_text().tolist()

        if result is None:
            result = pd.DataFrame(data={"stance": stance, "target": target, "text": text})
        else:
            result = pd.concat([result, pd.DataFrame(data={"stance": stance, "target": target, "text": text})])

    result.to_csv(path_or_buf=taskaa_order_dir, index=None, sep='\t')


def replacement_order(predict_dir, predict_order_dir, taskaa_order_dir):
    """
    :param predict_dir:
    :param predict_order_dir:
    :return:
    """
    stance = []
    target = []
    text = []
    text_1 = []
    topics = ['IphoneSE', '春节放鞭炮', '深圳禁摩限电', '俄罗斯在叙利亚的反恐行动', '开放二胎']
    taskaa = pd.read_csv(taskaa_order_dir, sep='\t')
    taskaa = dict(zip(list(taskaa['text']), list(taskaa['stance'])))
    with open(predict_dir, 'r') as file_1:
        for line in file_1.readlines():
            stance.append(line.split(',        ')[0].strip())
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


if __name__ == '__main__':
    taskaa_dir = 'evasampledata4-TaskAA.txt'
    taskaa_order_dir = 'evasampledata4-TaskAA_order.txt'
    taskaa_order(taskaa_dir=taskaa_dir, taskaa_order_dir=taskaa_order_dir)

    predict_dir = 'predict.txt'
    predict_order_dir = 'predict_order.txt'
    taskaa_order_dir = 'evasampledata4-TaskAA_order.txt'

    replacement_order(predict_dir=predict_dir, predict_order_dir=predict_order_dir, taskaa_order_dir=taskaa_order_dir)
