import numpy as np
import os
import time
import sys
import random

sys.path.append('../')
from yutils.file_utils import read_file2lol, read_file2list, data_to_pickle, pickle_to_data, write_list2file, \
    write_lol2file
from yutils.vec_utils import read_predefined_vocab
from yutils.vec_text import make_datasets, preload_tvt

np.random.seed(1234567)


def shuffle(lol, seed=123456):
    """
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    """
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


################################
# Raw data --> json or pickle
# output file style looks like this:
#     {"training":{
#         "xIndexes":[]
#         "yLabels":[]
#         "xQuestions":[]
#             }
#      "validation": (looks like training)
#      "test": (looks like training)
#      "word2idx":{"_padding":0,"_unk":1, "1st":2, "hello":3, ...}
#      "label2idx"{"-1":0, "0":1, "1":2}
#      "embedding":[ [word0], [word1], [word2], ...] ********************* Can be moved to separate file
#     }
################################
def read_dataset(args):
    """
    Make it easy to generate training and test sets
    1) decide to_read directories
    2) split the data set into training/test/validation
    3) ??? what about cross-validation
    :param args:
    :return:
    """
    data = []  # answer, label, question, remark
    fns = ["10k", "3k"]  # "../data/10k/"

    for fn in fns:
        data.append([read_file2lol("%s/%s/segment_answers.txt" % (args.in_dir, fn)),
                     read_file2lol("%s/%s/segment_questions.txt" % (args.in_dir, fn)),
                     read_file2list("%s/%s/labels.txt" % (args.in_dir, fn))]
                    )

    assert len(data[0][0]) == len(data[0][1]) == len(data[0][2])
    assert len(data[1][0]) == len(data[1][1]) == len(data[1][2])

    # Data follows this order: train, test
    shuffle(data[0], seed=123456)
    test = data[1]
    if args.has_valid:
        train_num = int(len(data[0][0]) * args.portion)
        train = [d[:train_num] for d in data[0]]
        valid = [d[train_num:] for d in data[0]]
    else:
        train = data[0]
        valid = test

    assert len(train[0]) == len(train[1]) == len(train[2])
    assert len(valid[0]) == len(valid[1]) == len(valid[2])
    assert len(test[0]) == len(test[1]) == len(test[2])

    raw_data = {"training": train,
                "validation": valid,
                "test": test}
    return raw_data


def processing(args):
    out_dir = args.out_dir + "/" + str(args.task) + "/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # 1. Split the data, read into defined format
    raw_data = read_dataset(args)

    # 2. Read the vocab text file and get VOCAB dictionary
    vocab = read_predefined_vocab(emb=args.emb, emb_type=args.emb, stat_lines=0)

    # 3. Transform text into indexes
    feat_names = ["xIndexes", "xQuestions", "yLabels"]
    label2idx = dict((i - 1, i) for i in range(args.num_class))
    datasets, word2idx, embeddings = make_datasets(vocab,
                                                   raw_data,
                                                   label2idx=label2idx,  # !!! Critical for label2idx
                                                   big_voc=args.big_voc,
                                                   feat_names=feat_names)

    # 4. Write training materials into pickles
    data_to_pickle(datasets, out_dir + "/features_" + args.emb + ".pkl")
    data_to_pickle(word2idx, out_dir + "/word2idx_" + args.emb + ".pkl")
    data_to_pickle(embeddings, out_dir + "/embeddings_" + args.emb + ".pkl")

    print("hello------------------------")
    tit = time.time()
    # 5. Pad the training data
    preload_tvt(datasets, max_lens=[args.sen_max_len, args.ask_max_len],
                out_dir=out_dir, emb=args.emb, feat_names=feat_names)
    print("hello------------------------", (time.time() - tit))
    # test correctness
    datasets = pickle_to_data(out_dir + "/features_" + args.emb + ".pkl")
    word2idx = pickle_to_data(out_dir + "/word2idx_" + args.emb + ".pkl")

    print(datasets["label2idx"])
    print(word2idx["_padding"], word2idx["_unk"])
    print(len(datasets["test"]["yLabels"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vectorize Sogou AnswerStance data")
    ''' May not changed '''
    parser.add_argument("--num_class", type=int, default=3,
                        help="number of classes")
    parser.add_argument("--in_dir", type=str, default="../data/processed/",
                        help="directory for input data")
    parser.add_argument("--out_dir", type=str, default="../data/vec",
                        help="directory for output pickles")
    parser.add_argument("--task", type=str, default="3k10k",
                        help="use which part of data for training and test")

    ''' May change '''
    parser.add_argument("--emb", type=str, default="baike",
                        help="type of word embeddings baike")

    parser.add_argument("--has_valid", action="store_true",
                        help="whether have 'real' validation data for tuning the model")
    parser.add_argument("--big_voc", action="store_true",
                        help="if set, use the all vocabulary")

    parser.add_argument("--portion", type=float, default=0.9,
                        help="decide portion to spare for training and validation")
    parser.add_argument("--sen_max_len", type=int, default=45,
                        help="max time step of sentence sequence")
    parser.add_argument("--ask_max_len", type=int, default=25,
                        help="max time step of sentence sequence")

    my_args = parser.parse_args()

    processing(my_args)
