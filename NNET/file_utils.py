# -*- coding:utf-8 -*-
import jieba
import pandas as pd
import pickle

import args
from str_utils import str_to_list, list_to_str


#############################################################
###################
#   Read and write files
###################
# contents is a list of Strings
def write_list2file(contents, filename):
    s = ''
    for i in contents:
        s += (str(i) + "\n")
    with open(filename, 'wb') as f:
        f.write(s.encode())
    print("********** Write to %s Successfully" % filename)


def write_lol2file(contents, filename):
    s = ''
    for i in contents:
        s += (str(list_to_str(i)) + "\n")
    with open(filename, 'wb') as f:
        f.write(s.encode())
    print("********** Write to %s Successfully" % filename)


# read raw text into list (sentence in strings)
def read_file2list(filename):
    with open(filename, 'rb') as f:
        contents = [line.strip().decode() for line in f]
    print("The %s has lines: %d" % (filename, len(contents)))
    return contents


# read segmented corpus into list (sentence in list of words)
def read_file2lol(filename):
    with open(filename, 'rb') as f:
        contents = [str_to_list(line.strip().decode()) for line in f]
    print("The %s has lines: %d" % (filename, len(contents)))
    return contents


#############################################################

###################
#   Serialization to pickle
###################
def data_to_pickle(your_dict, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(your_dict, f)


def pickle_to_data(in_file):
    with open(in_file, 'rb') as f:
        your_dict = pickle.load(f)

    return your_dict


###################
#   Generate source file
###################
def generate_answers_questions_labels():
    """
    生成三个文件
    :return:
    """
    data = pd.read_csv(args.SOURCES_FILE, encoding='utf-8-sig')
    ID = list(data['ID'])
    TARGET = [' '.join(jieba.cut(i, cut_all=False)) for i in data['TARGET']]
    TEXT = [' '.join(jieba.cut(i, cut_all=False)) for i in data['TEXT']]
    STANCE = list(data['STANCE'])

    with open(args.ids_file, 'w') as file0, open(args.answers_file, 'w') as file1, open(args.questions_file,
                                                                                        'w') as file2, open(
        args.labels_file, 'w') as file3:

        for (data, file) in zip([ID, TARGET, TEXT, STANCE], [file0, file1, file2, file3]):
            for line in data:
                file.write(str(line) + '\n')

# if __name__ == '__main__':
#     generate_answers_questions_labels()
