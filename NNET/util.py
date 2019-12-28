import random
import jieba
import argparse
from flyai.dataset import Dataset

import args as arguments
from NNET.utils.vectorize import shuffle
from NNET.utils.vec_utils import read_emb
from NNET.utils.vec_text import make_datasets, load_tvt

# 项目的超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=8, type=int, help="batch size")
args = parser.parse_args()
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)

# 1. Split the data, read into defined format
label2idx = dict((arguments.labels[i], i) for i in range(len(arguments.labels)))
target_text, stance, _, _ = dataset.get_all_data()

indexes = [" ".join(jieba.cut(i['TARGET'], cut_all=False)) for i in target_text]
questions = [" ".join(jieba.cut(i['TEXT'], cut_all=False)) for i in target_text]
labels = [label2idx[i['STANCE']] for i in stance]
data = [indexes, questions, labels]
assert len(data[0]) == len(data[1]) == len(data[2])

# Data follows this order: train, test
shuffle(data, seed=123456)
train_num = int(len(data[0]) * arguments.portion)
train_data = [d[:train_num] for d in data]
dev_data = [d[train_num:] for d in data]

# 2. Read the vocab text file and get VOCAB dictionary
vocab = read_emb(filename=arguments.baike_dir, emb_type=2, stat_lines=0)

# 3. Transform text into indexes
datasets, word2idx, embeddings = make_datasets(vocab=vocab,
                                               raw_data={'training': train_data, 'validation': dev_data},
                                               label2idx=label2idx,
                                               big_voc=arguments.big_voc, feat_names=arguments.feat_names)
features_train = load_tvt(tvt_set=datasets['training'], max_lens=[args.sen_max_len, args.ask_max_len],
                          feat_names=arguments.feat_names)
features_dev = load_tvt(tvt_set=datasets['validation'], max_lens=[args.sen_max_len, args.ask_max_len],
                        feat_names=arguments.feat_names)

# # 1. Split the data, read into defined format
# (train_indexes, train_questions), train_labels = dataset.next_train_batch()
#
# train_indexes = [" ".join(jieba.cut(i, cut_all=False)) for i in train_indexes]
# train_questions = [" ".join(jieba.cut(i, cut_all=False)) for i in train_questions]
# train_data = [train_indexes, train_questions, train_labels]
# assert len(train_data[0]) == len(train_data[1]) == len(train_data[2])
#
# # Data follows this order: train, test
# shuffle(train_data, seed=123456)
#
# (dev_indexes, dev_questions), dev_labels = dataset.next_validation_batch()
#
# dev_indexes = [" ".join(jieba.cut(i, cut_all=False)) for i in dev_indexes]
# dev_questions = [" ".join(jieba.cut(i, cut_all=False)) for i in dev_questions]
# dev_data = [dev_indexes, dev_questions, dev_labels]
# assert len(dev_data[0]) == len(dev_data[1]) == len(dev_data[2])
