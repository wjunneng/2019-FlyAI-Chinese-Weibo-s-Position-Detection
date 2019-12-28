# -*- coding:utf-8 -*-
import sys

sys.path.append('../')

import numpy as np
from NNET.utils.file_utils import data_to_pickle
from NNET.utils.vec_utils import YDataset, sentences_to_idx, sentences_to_idx_small_vocab, label_to_idx

################################
# Raw data --> json or pickle
# output file style looks like this:
#     {"training":{
#         "xIndexes":[]
#         "yLabels":[]
#         "xFeatures":[]
#             }
#      "validation": (looks like training)
#      "test": (looks like training)
#      "word2idx":{"_padding":0,"_unk":1, "1st":2, "hello":3, ...}
#      "label2idx"{"-1":0, "0":1, "1":2}
#      "embedding":[ [word0], [word1], [word2], ...] ********************* Can be moved to separate file
#     }
################################


def make_data(raw_data_i, vocab, word2idx, embeddings, label2idx, big_voc=False, feat_names=None):
    """
    make a dictionary for training or validation or test
    "training":{
        "xIndexes":[]
        "yLabels":[]
        "xFeatures":[]
    }
    [ if there are other features, just modify this file and the read_dataset() function ]
    :param raw_data_i: features, labels
    :param vocab: embeddings in format {word: emb, ...}
        vocab = {"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}
    :param word2idx:
    :param embeddings:
    :param label2idx:
    :param big_voc: whether to use the full vocabulary or the small one
    :param feat_names: looks like ["xIndexes", "yLabels"]
    :return:
    """
    assert len(feat_names) == len(raw_data_i)

    """ raw data --> key in the data"""
    feats = raw_data_i[:-1]
    labels = raw_data_i[-1]

    data = dict()
    if big_voc:
        for idx in range(len(feat_names) - 1):
            feat_name = feat_names[idx]
            data[feat_name] = sentences_to_idx(feats[idx], word2idx)
    else:
        for idx in range(len(feat_names) - 1):
            feat_name = feat_names[idx]
            data[feat_name] = sentences_to_idx_small_vocab(feats[idx], vocab, word2idx,
                                                           embeddings, prompt=feat_name)

    label_name = feat_names[-1]
    data[label_name] = label_to_idx(labels, label2idx)
    return data


def make_datasets(vocab, raw_data, label2idx, big_voc=False, feat_names=None):
    """
    :param vocab: embeddings in format {word: emb, ...}
        vocab = {"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}
    :param raw_data: dictionary  of {"training": (inputs, labels), "validation", "test"}
    :param label2idx:
    :param big_voc: whether to use the full vocabulary or the small one
    :param feat_names: looks like ["xIndexes", "yLabels"]
    :return:
    """
    datasets = dict()

    word2idx = dict()
    embeddings = []

    if big_voc:  # padding and unk available
        word2idx = vocab["word2idx"]
        embeddings = vocab["embeddings"]
    else:  # no padding or unk available
        # Add PADDING, UNK to word2idx and embeddings in the small setting
        word2idx["_padding"] = vocab["word2idx"]["_padding"]
        embeddings.append(vocab["embeddings"][0])
        word2idx["_unk"] = vocab["word2idx"]["_unk"]
        embeddings.append(vocab["embeddings"][1])

    datasets["label2idx"] = label2idx

    if not feat_names:
        feat_names = ["xIndexes", "yLabels"]
    names = ["training", "validation"]
    for name in names:
        # sentences, labels = raw_data[i]
        datasets[name] = make_data(raw_data[name], vocab, word2idx, embeddings, datasets["label2idx"], big_voc,
                                   feat_names)
        print("%s has %d words, %d labels" % (name, len(word2idx), len(datasets["label2idx"])))
        assert len(word2idx) == len(embeddings)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = embeddings.reshape(len(embeddings), -1)

    return datasets, word2idx, embeddings


def preload_tvt(datasets, max_lens, out_dir, emb="glove", feat_names=None):
    """
    Pre-load the training, validation, test data, then do the padding and mask matrix
    And write to pickle file
    :param datasets: {training, validation, test}
    :param feat_names: ["xIndexes", "yLabels"]
    :param max_lens:
    :param out_dir:
    :param emb:
    :return:
        vocab = {"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}
    """

    for part in ["training", "validation"]:
        tvt_set = datasets[part]
        tvt = load_tvt(tvt_set, max_lens, feat_names)
        data_to_pickle(tvt, "%s/features_%s_%s.pkl" % (out_dir, emb, part))


def load_tvt(tvt_set, max_lens, feat_names=None):
    """
    Load the training, validation, test data, then do the padding and mask matrix
    :param tvt_set: x_features, y_labels
    :param feat_names: ["xIndexes", "yLabels"]
    :param max_lens:
    :return:
        tvt: training, validation or test set that support batch
    """

    if not feat_names:
        feat_names = ["xIndexes", "yLabels"]
    assert len(feat_names) == len(max_lens) + 1

    # decide how many inputs!!!
    feats = [tvt_set[key] for key in feat_names]

    tvt = YDataset(feats[:-1], feats[-1],
                   to_pad=True,
                   max_lens=max_lens)

    return tvt
