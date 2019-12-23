import numpy as np
import os
import time
import sys
import random

sys.path.append('../')
from file_utils import read_file2lol, read_file2list, data_to_pickle, pickle_to_data, write_list2file, write_lol2file
from vec_utils import read_emb
from vec_text import make_datasets, preload_tvt
import args

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
    # answer, label, question, remark
    data = [read_file2lol("%s/input/answers.txt" % (args.in_dir)),
            read_file2lol("%s/input/questions.txt" % (args.in_dir)),
            read_file2list("%s/input/labels.txt" % (args.in_dir))]

    assert len(data[0]) == len(data[1]) == len(data[2])

    # Data follows this order: train, test
    shuffle(data, seed=123456)
    train_num = int(len(data[0]) * args.portion)
    train = [d[:train_num] for d in data]
    valid = [d[train_num:] for d in data]

    assert len(train[0]) == len(train[1]) == len(train[2])
    assert len(valid[0]) == len(valid[1]) == len(valid[2])

    raw_data = {"training": train, "validation": valid}

    return raw_data


def processing():
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # 1. Split the data, read into defined format
    raw_data = read_dataset(args)

    # 2. Read the vocab text file and get VOCAB dictionary
    vocab = read_emb(filename=args.baike_dir, emb_type=2, stat_lines=0)

    # 3. Transform text into indexes
    feat_names = ["xIndexes", "xQuestions", "yLabels"]
    label2idx = dict((args.labels[i], i) for i in range(len(args.labels)))
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


if __name__ == "__main__":
    processing()
