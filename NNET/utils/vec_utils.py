import numpy as np
from file_utils import read_file2list, data_to_pickle, pickle_to_data
from str_utils import decide_run_place


# np.random.seed(123456)

#############################################################
###################
# Word embedding utilities
# 1. read different types of embedding files (senna; glove, baike; weibo; Google-News)
#      and get corresponding embeddings lists and words list
# 2. create vocab dict with the given embeddings and words list (!!! separate embedding matrix and word2idx)
# 3. pickle the above file
# 4. read the pickles to python data
#  Whether to add one unified Vocab class ???????????????


###################

def read_emb_idx(filename, stat_lines=0):
    """
    read glove, baike, weibo or other word embs trained with gensim
    :param filename: single emb and word file, w/o stat info
    :param stat_lines: number of lines showing the statistic information of th word embeddings
    """
    with open(filename, 'rb') as f:
        for i in range(stat_lines):  # ignore irrelevant lines
            _ = f.readline()

        embeddings = []
        words = []
        for line in f:
            # print(line)
            line = line.strip().decode()
            one = line.split(' ')
            word = one[0].lower()
            emb = [float(i) for i in one[1:]]
            embeddings.append(emb)
            words.append(word)
        return embeddings, words


def create_vocab(embeddings, words):
    """
    Given the [embeddings, words list]
    1) add _padding and _unk to embedding/word2idx/idx2word dictionaries
    2) transform embeddings list to numpy nd-arrays

    :param embeddings:
        list of vectors each line
    :param words:
        list of word
    :return:
        vocab = {"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}
    """

    word2idx = dict((word, idx + 2) for idx, word in enumerate(words))
    word2idx["_padding"] = 0
    word2idx["_unk"] = 1

    # Add padding and unknown word to embeddings and word2idx
    emb_dim = len(embeddings[0])
    embeddings.insert(0, np.zeros(emb_dim))  # _padding
    embeddings.insert(1, np.random.random(emb_dim))  # _unk

    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = embeddings.reshape(len(embeddings), emb_dim)

    # idx2word = dict((word2idx[word], word) for word in word2idx)
    idx2word = dict((v, k) for k, v in word2idx.items())
    vocab = {"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}

    return vocab


def read_emb(filename, emb_type=2, stat_lines=0):
    """

    :param filename:
    :param emb_type:
        1. glove, baike, weibo, single emb and word file, no stat info
        2. we trained with gensim, single emb and word file, with stat info (size and dim)
        3. senna, separate embs and words file, no stat info (size, dim)
        4. google-news file
    :param stat_lines: number of statistic lines on top of the file

    :return:
    """
    embeddings = None
    words = None
    type1 = [1, 2, "glove", "baike", "weibo", "zhwiki"]
    if emb_type in type1:
        embeddings, words = read_emb_idx(filename, stat_lines)

    print("Finish loading embedding %s * * * * * * * * * * * *" % filename)

    vocab = create_vocab(embeddings, words)

    return vocab


def vocab_to_pickle(vocab, emb_dir=""):
    """
    Transform vocab to pickle file
    :param vocab: {"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}
    :param emb_dir: place to output pickles
    :return:
    """
    for key in vocab:
        data_to_pickle(vocab[key], out_file=emb_dir + "/" + key + ".pkl")


def pickle_to_vocab(pkl_dir):
    """
    Transform pickle file to vocab dictionary
    :param pkl_dir: place to output pickles
    :return: vocab: {"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}
    """
    keys = ["word2idx", "idx2word", "embeddings"]
    vocab = dict()
    for key in keys:
        vocab[key] = pickle_to_data(in_file=pkl_dir + "/" + key + ".pkl")
    # vocab = dict((key, pickle_to_data(out_file=pkl_dir + "/" + key + ".pkl"))
    #              for key in keys)
    return vocab


def sentence_to_idx_small_vocab(sentence, vocab, word2idx, embeddings):
    s_index = []
    for word in sentence:
        if word == "\n":
            continue
        # gradually add new words into word2idx
        if word in vocab["word2idx"]:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                idx = vocab["word2idx"][word]
                embeddings.append(vocab["embeddings"][idx])
            s_index.append(word2idx[word])
        else:
            print("  1--%s--  " % word)
            word = word.strip("`'._-*")
            if word in vocab["word2idx"]:
                if word not in word2idx:
                    word2idx[word] = len(word2idx)
                    idx = vocab["word2idx"][word]
                    embeddings.append(vocab["embeddings"][idx])
                s_index.append(word2idx[word])
            else:
                s_index.append(word2idx["_unk"])
                print("  2--%s--  " % word)

    if len(s_index) == 0:
        print(len(sentence), "+++++++++++++++++++++++++++++++++, empty sentence")
        s_index.append(word2idx["_unk"])

    return s_index


def sentences_to_idx_small_vocab(sentences, vocab, word2idx, embeddings, prompt="sentence"):
    """
    transform word to index in word2idx and store the embeddings

    :param sentences:
        list of sentences which are list of word
    :param vocab: embeddings in format {word: emb, ...}
        vocab = {"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}
    :param word2idx:
        word2idx = {word:idx, ...}
    :param embeddings:
    :param prompt:
    :return:
    """
    print("-------------begin making " + prompt + " xIndexes-------------")
    sentences_indexes = []
    for sentence in sentences:
        s_index = sentence_to_idx_small_vocab(sentence, vocab, word2idx, embeddings)
        sentences_indexes.append(s_index)

    assert len(word2idx) == len(embeddings)
    assert len(sentences_indexes) == len(sentences)

    print("-------------finish making " + prompt + " xIndexes-------------")
    return sentences_indexes


def sentences_to_idx(sentences, word2idx):
    """
    :param sentences:
        list of sentences which are list of word
    :param word2idx:
        word2idx = {word:idx, ...}
    :return:
    """
    sentences_indexes = []
    for sentence in sentences:
        s_index = sentence_to_idx(sentence, word2idx)
        sentences_indexes.append(s_index)

    assert len(sentences_indexes) == len(sentences)

    return sentences_indexes


def sentence_to_idx(sentence, word2idx):
    s_index = []
    for word in sentence:
        word = word.strip()
        if word == "\n":
            continue
        if word in word2idx:
            s_index.append(word2idx[word])
        else:
            s_index.append(word2idx["_unk"])
            # print("  -- %s --  " % word)

    if len(s_index) == 0:
        print(len(sentence), "+++++++++++++++++++++++++++++++++empty sentence")
        s_index.append(word2idx["_unk"])

    return s_index


def label_to_idx(labels, label2idx):
    """
    Transform label to index in label2idx
    :param labels: label like [-1, 0, 1], [FAVOR, NONE, AGAINST]
    :param label2idx: to positive numbers [0, 1, 2, 3] ...
    :return:
    """
    print("-------------begin to make yLabels-------------")
    label_indexes = []  # to avoid ne
    for label in labels:
        label_indexes.append(label2idx[label])

    label_indexes = np.asarray(label_indexes, dtype=np.int64).reshape(len(labels))
    print("-------------finish making yLabels-------------")
    return label_indexes


def make_one_hot(num_classes, value=1):
    """
    Make one-hot matrix for labels (if needed)
    :param num_classes:
    :param value: value for one-hot intensification
    :return:
        [
          [1,0,0,0...,num_classes],
          [0,1,0,0...,num_classes],
                   ...
          [0,0,0,0,...,         1]
        ]
    """
    # Numpy matrix is very easy to implement
    one_hot = np.identity(num_classes) * value
    return one_hot


def word2idx_to_idx2word(word2idx):
    """

    :param word2idx:
    :return:
    """
    idx2word = dict((v, k) for k, v in word2idx.items())
    return idx2word


def idx2word_to_word2idx(idx2word):
    """

    :param idx2word:
    :return:
    """
    word2idx = dict((v, k) for k, v in idx2word.items())
    return word2idx


#############################################################
def get_padding(sentences, max_len):
    """
    :param sentences: raw sentence --> index_padded sentence
                    [2, 3, 4], 5 --> [2, 3, 4, 0, 0]
    :param max_len: number of steps to unroll for a LSTM
    :return: sentence of max_len size with zero paddings

    If it need to pad very large data set, then set a max length for one loop


    """
    num_sen = len(sentences)
    seq_len = np.zeros((num_sen,), dtype=np.int64)  # num_sen
    padded = np.zeros((num_sen, max_len), dtype=np.int64)  # num_sen * max_len

    for i in range(num_sen):
        sentence = sentences[i]
        num_words = len(sentence)  # before truncation

        if max_len == 60 and num_words > 60:
            sentence = sentence[:45] + sentence[num_words - 15:]
        sentence = sentence[:max_len]

        num_words = len(sentence)  # after truncation

        padded[i][:num_words] = sentence
        seq_len[i] = num_words

    return padded, seq_len


def get_mask_matrix(seq_lengths, max_len):
    """
    [5, 2, 4,... 7], 10 -->
            [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
             ...,
             [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
            ]
    :param seq_lengths:
    :param max_len:
    :return:
    """
    num_sen = len(seq_lengths)
    mask_matrix = np.zeros((num_sen, max_len), dtype=np.int64)

    for i in range(num_sen):
        seq_len = seq_lengths[i]
        # add ones to zeros according to sequence length
        mask_matrix[i][:seq_len] = np.ones(seq_len, dtype=np.int64)

    return mask_matrix


class YDataset(object):
    """
    A user-defined dataset class for deep learning in NLP
    Features:
        1) support padding, sequence length, mask matrix
        2) support batch training,
        3) support batch indexing for limited memory
        4) support multiple text features
    DRAWBACKS:
        1) don't support discret features like pos tag at present

    """

    def __init__(self, list_of_features, labels, to_pad=True, max_lens=[6, 25]):
        """
        All sentences are indexes of words!
        :param list_of_features: list containing sequences to be padded and batched
        :param labels:
        """

        self.features = list_of_features
        self.labels = labels
        self.pad_max_lens = max_lens
        self.seq_lens = None
        self.mask_matrix = None
        self.num_feature = len(self.features)

        for one_kind_of_feature in self.features:
            assert len(one_kind_of_feature) == len(self.labels)

        self._num_examples = len(self.labels)
        self._epochs_completed = 0
        self._index_in_epoch = 0

        if to_pad:
            if len(self.pad_max_lens) == self.num_feature:
                self._padding()
                self._mask()
            else:
                print("Need more information about padding max_length")

    def __len__(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def _padding(self):
        """
        :return: tuple(list_of_padded_features, list_of_seqlen)
        [answers, questions]
        [answers_len, questions_len]
        """
        all_features = []
        all_seq_lens = []
        for i in range(len(self.features)):
            padded_feature, seq_len = get_padding(self.features[i], max_len=self.pad_max_lens[i])
            # print(padded_feature)
            # print(seq_len)
            all_features.append(padded_feature)
            all_seq_lens.append(seq_len)
        self.seq_lens = all_seq_lens
        self.features = all_features

    def _mask(self):
        all_mask_matrix = []
        for i in range(len(self.seq_lens)):
            mask_matrix = get_mask_matrix(self.seq_lens[i], max_len=self.pad_max_lens[i])
            all_mask_matrix.append(mask_matrix)
            # print(mask_matrix)
        self.mask_matrix = all_mask_matrix

    def _shuffle(self, seed):
        """
        After each epoch, the data need to be shuffled
        :return:
        """

        # shuffle(self.features, seed=seed)
        # shuffle(self.mask_matrix, seed=seed)
        # shuffle(self.seq_lens, seed=seed)
        # shuffle([self.labels], seed=seed)

        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)  # non-reproducible
        for i in range(self.num_feature):
            self.features[i] = self.features[i][perm]
            self.seq_lens[i] = self.seq_lens[i][perm]
            self.mask_matrix[i] = self.mask_matrix[i][perm]
        self.labels = self.labels[perm]

    def next_batch(self, batch_size, seed=123456):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            """  shuffle feature  and labels"""
            self._shuffle(seed)

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        _features = [feat[start:end] for feat in self.features]
        _seq_lens = [seqlen[start:end] for seqlen in self.seq_lens]
        _mask_matrix = [matrix[start:end] for matrix in self.mask_matrix]
        _labels = self.labels[start:end]

        return _features, _seq_lens, _mask_matrix, _labels

    def get_batch(self, batch_size, total_num):
        num_batch = int(total_num / batch_size)
        left = len(total_num) - batch_size * num_batch
        for idx in range(num_batch):
            feature_batch = [feat[idx * batch_size: (idx + 1) * batch_size]
                             for feat in self.features]
            yield feature_batch
        if left > 0:
            feature_batch = [feat[num_batch * batch_size:]
                             for feat in self.features]
            yield feature_batch


def get_batch(batch_size, total_num, features):
    """

    :param batch_size:
    :param total_num:
    :param features: list of input features need to be generate
    :return:
    """
    """
    pred_batch, max_index_batch = test_batch(args, model,
                                                     questions[batch * 1000:(batch + 1) * 1000],
                                                     questions_seqlen[batch * 1000:(batch + 1) * 1000],
                                                     questions_mask[batch * 1000:(batch + 1) * 1000],
                                                     answers[batch * 1000:(batch + 1) * 1000],
                                                     answers_seqlen[batch * 1000:(batch + 1) * 1000],
                                                     answers_mask[batch * 1000:(batch + 1) * 1000]
    """
    num_batch = int(total_num / batch_size)
    left = total_num - batch_size * num_batch
    for idx in range(num_batch):
        feature_batch = [feature[idx * batch_size: (idx + 1) * batch_size]
                         for feature in features]
        yield feature_batch
    if left > 0:
        feature_batch = [feature[num_batch * batch_size:]
                         for feature in features]
        yield feature_batch


# if __name__ == "__main__":
#     print("------------This is for utility test--------------")
#     print(get_padding([[1, 2, 3, 1, 2, 3], [6, 7, 8], [2, 3, 4]], max_len=5))
#     print(get_mask_matrix([9, 2, 3], max_len=10))
