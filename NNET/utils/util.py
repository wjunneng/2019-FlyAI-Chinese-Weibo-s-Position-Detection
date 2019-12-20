# -*- coding:utf-8 -*-
import jieba
import pandas as pd

from NNET import args


def generate_answers_questions_labels():
    """
    生成三个文件
    :return:
    """
    data = pd.read_csv(args.SOURCES_FILE, encoding='utf-8-sig')
    ID = [' '.join(jieba.cut(i, cut_all=False)) for i in data['ID']]
    TARGET = [' '.join(jieba.cut(i, cut_all=False)) for i in data['TARGET']]
    TEXT = [' '.join(jieba.cut(i, cut_all=False)) for i in data['TEXT']]
    STANCE = list(data['STANCE'])

    with open(args.ids_file, 'w') as file0, open(args.answers_file, 'w') as file1, open(args.questions_file,
                                                                                        'w') as file2, open(
        args.labels_file, 'w') as file3:

        for data_type in [ID, TARGET, TEXT, STANCE]:
            for file_type in [file0, file1, file2, file3]:
                for line in data_type:
                    file_type.write(line + '\n')


if __name__ == '__main__':
    generate_answers_questions_labels()
